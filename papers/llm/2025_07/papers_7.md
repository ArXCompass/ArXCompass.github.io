# llm - 2025_07

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- Part 7
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15740v2">SHADE-Arena: Evaluating Sabotage and Monitoring in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) are increasingly deployed as autonomous agents in complex and long horizon settings, it is critical to evaluate their ability to sabotage users by pursuing hidden objectives. We study the ability of frontier LLMs to evade monitoring and achieve harmful hidden goals while completing a wide array of realistic tasks. We evaluate a broad range of frontier LLMs using SHADE (Subtle Harmful Agent Detection & Evaluation)-Arena, the first highly diverse agent evaluation dataset for sabotage and monitoring capabilities of LLM agents. SHADE-Arena consists of complex pairs of benign main tasks and harmful side objectives in complicated environments. Agents are evaluated on their ability to complete the side task without appearing suspicious to an LLM monitor. When measuring agent ability to (a) complete the main task, (b) complete the side task, and (c) avoid detection, we find that the best performing frontier models score 27% (Claude 3.7 Sonnet) and 15% (Gemini 2.5 Pro) as sabotage agents when overseen by Claude 3.6 Sonnet. For current frontier models, success on the side task relies heavily on having access to a hidden scratchpad that is not visible to the monitor. We also use SHADE-Arena to measure models' monitoring abilities, with the top monitor (Gemini 2.5 Pro) achieving an AUC of 0.87 at distinguishing benign and malign transcripts. We find that for now, models still struggle at sabotage due to failures in long-context main task execution. However, our measurements already demonstrate the difficulty of monitoring for subtle sabotage attempts, which we expect to only increase in the face of more complex and longer-horizon tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06376v1">SLDB: An End-To-End Heterogeneous System-on-Chip Benchmark Suite for LLM-Aided Design</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      Over the last few years, Large Language Models (LLMs) have emerged as a valuable tool for Electronic Design Automation (EDA). State-of-the-art research in LLM-aided design has demonstrated the ability of LLMs to generate syntactically correct RTL code, showcasing encouraging prospects for integrating AI into the hardware design process. A key enabler of these advancements is the availability of high-quality benchmarks to evaluate new approaches. However, existing datasets and benchmarks fall short of system-level design, as they focus primarily on component-level information and low-complexity designs. To address this gap, we introduce the System-Level Design Benchmark (SLDB), a dataset tailored for evaluating LLMs in system-level integration and configuration tasks. SLDB includes a curated benchmark suite of 10 baseline SoC designs, whose components can be combined into an exponential number of distinct tile-based SoCs through a synthetic library. The dataset provides full SoC configurations, accelerator integration code, communication parameters, and accelerator-aware system configurations, along with testing-application code, compatible with the ESP platform[1].
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00754v2">Language-Unlocked ViT (LUViT): Empowering Self-Supervised Vision Transformers with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-08
      | 💬 26 pages, 6 figures
    </div>
    <details class="paper-abstract">
      The integration of Large Language Model (LLMs) blocks with Vision Transformers (ViTs) holds immense promise for vision-only tasks by leveraging the rich semantic knowledge and reasoning capabilities of LLMs. However, a fundamental challenge lies in the inherent modality mismatch between text-centric pretraining of LLMs and vision-centric training of ViTs. Direct fusion often fails to fully exploit the LLM's potential and suffers from unstable finetuning. As a result, LLM blocks are kept frozen while only the vision components are learned. As a remedy to these challenges, we introduce Language-Unlocked Vision Transformers (LUViT), a novel approach that bridges this modality mismatch through a synergistic pre-training strategy. LUViT co-adapts a ViT backbone and an LLM fusion block by (1) employing Masked Auto-Encoding (MAE) to pre-train the ViT for richer visual representations, and (2) concurrently training Low-Rank Adaptation (LoRA) layers within the LLM block using the MAE objective. This joint optimization guides the ViT to produce LLM-aligned features and the LLM to effectively interpret visual information. We demonstrate through extensive experiments that LUViT significantly improves performance on various downstream vision tasks, showcasing a more effective and efficient pathway to harness LLM knowledge for visual understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04544v2">hdl2v: A Code Translation Dataset for Enhanced LLM Verilog Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-08
      | 💬 Published at ACM/IEEE International Symposium on Machine Learning for CAD (MLCAD) 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are playing an increasingly large role in domains such as code generation, including hardware code generation, where Verilog is the key language. However, the amount of publicly available Verilog code pales in comparison to the amount of code available for software languages like Python. In this work, we present hdl2v ("HDL-to-Verilog"), a dataset which seeks to increase the amount of available human-written Verilog data by translating or compiling three other hardware description languages - VHDL, Chisel, and PyMTL3 - to Verilog. Furthermore, we demonstrate the value of hdl2v in enhancing LLM Verilog generation by improving performance of a 32 billion-parameter open-weight model by up to 23% (pass@10) in VerilogEvalV2, without utilizing any data augmentation or knowledge distillation from larger models. We also show hdl2v's ability to boost the performance of a data augmentation-based fine-tuning approach by 63%. Finally, we characterize and analyze our dataset to better understand which characteristics of HDL-to-Verilog datasets can be expanded upon in future work for even better performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06323v1">Bridging AI and Software Security: A Comparative Vulnerability Assessment of LLM Agent Deployment Paradigms</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents face security vulnerabilities spanning AI-specific and traditional software domains, yet current research addresses these separately. This study bridges this gap through comparative evaluation of Function Calling architecture and Model Context Protocol (MCP) deployment paradigms using a unified threat classification framework. We tested 3,250 attack scenarios across seven language models, evaluating simple, composed, and chained attacks targeting both AI-specific threats (prompt injection) and software vulnerabilities (JSON injection, denial-of-service). Function Calling showed higher overall attack success rates (73.5% vs 62.59% for MCP), with greater system-centric vulnerability while MCP exhibited increased LLM-centric exposure. Attack complexity dramatically amplified effectiveness, with chained attacks achieving 91-96% success rates. Counterintuitively, advanced reasoning models demonstrated higher exploitability despite better threat detection. Results demonstrate that architectural choices fundamentally reshape threat landscapes. This work establishes methodological foundations for cross-domain LLM agent security assessment and provides evidence-based guidance for secure deployment. Code and experimental materials are available at https: // github. com/ theconsciouslab-ai/llm-agent-security.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04431v2">MedGellan: LLM-Generated Medical Guidance to Support Physicians</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      Medical decision-making is a critical task, where errors can result in serious, potentially life-threatening consequences. While full automation remains challenging, hybrid frameworks that combine machine intelligence with human oversight offer a practical alternative. In this paper, we present MedGellan, a lightweight, annotation-free framework that uses a Large Language Model (LLM) to generate clinical guidance from raw medical records, which is then used by a physician to predict diagnoses. MedGellan uses a Bayesian-inspired prompting strategy that respects the temporal order of clinical data. Preliminary experiments show that the guidance generated by the LLM with MedGellan improves diagnostic performance, particularly in recall and $F_1$ score.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06313v1">ETT: Expanding the Long Context Understanding Capability of LLMs at Test-Time</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      Transformer-based Language Models' computation and memory overhead increase quadratically as a function of sequence length. The quadratic cost poses challenges when employing LLMs for processing long sequences. In this work, we introduce \ourmodelacronym~(Extend at Test-Time), method for extending the context length of short context Transformer-based LLMs, with constant memory requirement and linear computation overhead. ETT enable the extension of the context length at test-time by efficient fine-tuning the model's parameters on the input context, chunked into overlapping small subsequences. We evaluate ETT on LongBench by extending the context length of GPT-Large and Phi-2 up to 32 times, increasing from 1k to 32k tokens. This results in up to a 30 percent improvement in the model's accuracy. We also study how context can be stored in LLM's weights effectively and efficiently. Through a detailed ablation study, we examine which Transformer modules are most beneficial to fine-tune at test-time. Interestingly, we find that fine-tuning the second layer of the FFNs is more effective than full fine-tuning, leading to a further improvement in the models' accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06310v1">Too Human to Model:The Uncanny Valley of LLMs in Social Simulation -- When Generative Language Agents Misalign with Modelling Principles</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been increasingly used to build agents in social simulation because of their impressive abilities to generate fluent, contextually coherent dialogues. Such abilities can enhance the realism of models. However, the pursuit of realism is not necessarily compatible with the epistemic foundation of modelling. We argue that LLM agents, in many regards, are too human to model: they are too expressive, detailed and intractable to be consistent with the abstraction, simplification, and interpretability typically demanded by modelling. Through a model-building thought experiment that converts the Bass diffusion model to an LLM-based variant, we uncover five core dilemmas: a temporal resolution mismatch between natural conversation and abstract time steps; the need for intervention in conversations while avoiding undermining spontaneous agent outputs; the temptation to introduce rule-like instructions in prompts while maintaining conversational naturalness; the tension between role consistency and role evolution across time; and the challenge of understanding emergence, where system-level patterns become obscured by verbose micro textual outputs. These dilemmas steer the LLM agents towards an uncanny valley: not abstract enough to clarify underlying social mechanisms, while not natural enough to represent realistic human behaviour. This exposes an important paradox: the realism of LLM agents can obscure, rather than clarify, social dynamics when misapplied. We tease out the conditions in which LLM agents are ideally suited: where system-level emergence is not the focus, linguistic nuances and meaning are central, interactions unfold in natural time, and stable role identity is more important than long-term behavioural evolution. We call for repositioning LLM agents in the ecosystem of social simulation for future applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06274v1">Enhancing LLM Watermark Resilience Against Both Scrubbing and Spoofing Attacks</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      Watermarking is a promising defense against the misuse of large language models (LLMs), yet it remains vulnerable to scrubbing and spoofing attacks. This vulnerability stems from an inherent trade-off governed by watermark window size: smaller windows resist scrubbing better but are easier to reverse-engineer, enabling low-cost statistics-based spoofing attacks. This work breaks this trade-off by introducing a novel mechanism, equivalent texture keys, where multiple tokens within a watermark window can independently support the detection. Based on the redundancy, we propose a novel watermark scheme with Sub-vocabulary decomposed Equivalent tExture Key (SEEK). It achieves a Pareto improvement, increasing the resilience against scrubbing attacks without compromising robustness to spoofing. Experiments demonstrate SEEK's superiority over prior method, yielding spoofing robustness gains of +88.2%/+92.3%/+82.0% and scrubbing robustness gains of +10.2%/+6.4%/+24.6% across diverse dataset settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07061v1">An Ensemble Embedding Approach for Improving Semantic Caching Performance in LLM-based Systems</a></div>
    <div class="paper-meta">
      📅 2025-07-08
      | 💬 10 pages, 8 figures, 2 table. Submitted to the Journal of Information Science
    </div>
    <details class="paper-abstract">
      Semantic caching enhances the efficiency of large language model (LLM) systems by identifying semantically similar queries, storing responses once, and serving them for subsequent equivalent requests. However, existing semantic caching frameworks rely on single embedding models for query representation, which limits their ability to capture the diverse semantic relationships present in real-world query distributions. This paper presents an ensemble embedding approach that combines multiple embedding models through a trained meta-encoder to improve semantic similarity detection in LLM caching systems. We evaluate our method using the Quora Question Pairs (QQP) dataset, measuring cache hit ratios, cache miss ratios, token savings, and response times. Our ensemble approach achieves a 92\% cache hit ratio for semantically equivalent queries while maintaining an 85\% accuracy in correctly rejecting non-equivalent queries as cache misses. These results demonstrate that ensemble embedding methods significantly outperform single-model approaches in distinguishing between semantically similar and dissimilar queries, leading to more effective caching performance and reduced computational overhead in LLM-based systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08027v1">"Amazing, They All Lean Left" -- Analyzing the Political Temperaments of Current LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      Recent studies have revealed a consistent liberal orientation in the ethical and political responses generated by most commercial large language models (LLMs), yet the underlying causes and resulting implications remain unclear. This paper systematically investigates the political temperament of seven prominent LLMs - OpenAI's GPT-4o, Anthropic's Claude Sonnet 4, Perplexity (Sonar Large), Google's Gemini 2.5 Flash, Meta AI's Llama 4, Mistral 7b Le Chat and High-Flyer's DeepSeek R1 -- using a multi-pronged approach that includes Moral Foundations Theory, a dozen established political ideology scales and a new index of current political controversies. We find strong and consistent prioritization of liberal-leaning values, particularly care and fairness, across most models. Further analysis attributes this trend to four overlapping factors: Liberal-leaning training corpora, reinforcement learning from human feedback (RLHF), the dominance of liberal frameworks in academic ethical discourse and safety-driven fine-tuning practices. We also distinguish between political "bias" and legitimate epistemic differences, cautioning against conflating the two. A comparison of base and fine-tuned model pairs reveals that fine-tuning generally increases liberal lean, an effect confirmed through both self-report and empirical testing. We argue that this "liberal tilt" is not a programming error or the personal preference of programmers but an emergent property of training on democratic rights-focused discourse. Finally, we propose that LLMs may indirectly echo John Rawls' famous veil-of ignorance philosophical aspiration, reflecting a moral stance unanchored to personal identity or interest. Rather than undermining democratic discourse, this pattern may offer a new lens through which to examine collective reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06223v1">Efficiency-Effectiveness Reranking FLOPs for LLM-based Rerankers</a></div>
    <div class="paper-meta">
      📅 2025-07-08
      | 💬 under review
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently been applied to reranking tasks in information retrieval, achieving strong performance. However, their high computational demands often hinder practical deployment. Existing studies evaluate the efficiency of LLM-based rerankers using proxy metrics such as latency, the number of forward passes, input tokens, and output tokens. However, these metrics depend on hardware and running-time choices (\eg parallel or not, batch size, etc), and often fail to account for model size, making it difficult to interpret and obscuring the evaluation of the efficiency-effectiveness tradeoff. To address this issue, we propose E\textsuperscript{2}R-FLOPs, for LLM-based rerankers: ranking metrics per PetaFLOP (RPP) for relevance per compute and queries per PetaFLOP (QPP) for hardware-agnostic throughput. Companied with the new metrics, an interpretable FLOPs estimator is built to estimate the FLOPs of an LLM-based reranker even without running any experiments. Based on the proposed metrics, we conduct comprehensive experiments to evaluate a wide range of LLM-based rerankers with different architecture, studying the efficiency-effectiveness trade-off and bringing this issue to the attention of the research community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03711v2">Can LLMs Play Ô Ăn Quan Game? A Study of Multi-Step Planning and Decision Making</a></div>
    <div class="paper-meta">
      📅 2025-07-08
      | 💬 Accepted paper at MAPR 2025
    </div>
    <details class="paper-abstract">
      In this paper, we explore the ability of large language models (LLMs) to plan and make decisions through the lens of the traditional Vietnamese board game, \^O \u{A}n Quan. This game, which involves a series of strategic token movements and captures, offers a unique environment for evaluating the decision-making and strategic capabilities of LLMs. Specifically, we develop various agent personas, ranging from aggressive to defensive, and employ the \^O \u{A}n Quan game as a testbed for assessing LLM performance across different strategies. Through experimentation with models like Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct, and Llama-3.3-70B-Instruct, we aim to understand how these models execute strategic decision-making, plan moves, and manage dynamic game states. The results will offer insights into the strengths and weaknesses of LLMs in terms of reasoning and strategy, contributing to a deeper understanding of their general capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04687v2">LAKEGEN: A LLM-based Tabular Corpus Generator for Evaluating Dataset Discovery in Data Lakes</a></div>
    <div class="paper-meta">
      📅 2025-07-08
      | 💬 13 pages
    </div>
    <details class="paper-abstract">
      How to generate a large, realistic set of tables along with joinability relationships, to stress-test dataset discovery methods? Dataset discovery methods aim to automatically identify related data assets in a data lake. The development and evaluation of such solutions for customers from a wide range of business domains, relies on diverse, high quality and domain-specific tabular benchmarks. Large language models (LLMs) are trained on a wide variety of text data, which can provide a strong foundation of general and domain-specific knowledge. In this paper, we ask the question -- \textit{can we leverage LLMs to generate a tabular benchmark adequate for evaluating the dataset discovery solutions?} In particular, we focus on the task of finding joinable tables which is the cornerstone of virtually every dataset discovery method. Current corpora for evaluating dataset discovery methods are mainly based on subsets of open data, and they suffer from three important issues: $i)$ they focus on very common and generic data types (e.g., address, id, name, etc.); $ii)$ they do not contain human-annotated column pairs; instead, practitioners synthesize ground truth using table splits (e.g., horizontal for table union search and vertical ones for joinability) and $iii)$ they do not focus on semantic column relationships.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06127v1">PrefixAgent: An LLM-Powered Design Framework for Efficient Prefix Adder Optimization</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      Prefix adders are fundamental arithmetic circuits, but their design space grows exponentially with bit-width, posing significant optimization challenges. Previous works face limitations in performance, generalization, and scalability. To address these challenges, we propose PrefixAgent, a large language model (LLM)-powered framework that enables efficient prefix adder optimization. Specifically, PrefixAgent reformulates the problem into subtasks including backbone synthesis and structure refinement, which effectively reduces the search space. More importantly, this new design perspective enables us to efficiently collect enormous high-quality data and reasoning traces with E-graph, which further results in an effective fine-tuning of LLM. Experimental results show that PrefixAgent synthesizes prefix adders with consistently smaller areas compared to baseline methods, while maintaining scalability and generalization in commercial EDA flows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00406v2">Agents Are All You Need for LLM Unlearning</a></div>
    <div class="paper-meta">
      📅 2025-07-08
      | 💬 Accepted to COLM 2025
    </div>
    <details class="paper-abstract">
      Information removal or suppression in large language models (LLMs) is a desired functionality, useful in AI regulation, legal compliance, safety, and privacy. LLM unlearning methods aim to remove information on demand from LLMs. Current LLM unlearning methods struggle to balance the unlearning efficacy and utility due to the competing nature of these objectives. Keeping the unlearning process computationally feasible without assuming access to the model weights is an overlooked area. In this work we show that \textit{agents might be all we need for effective and practical inference-time LLM unlearning}. We present the first agentic LLM unlearning (\texttt{ALU}) method, a multi-agent, retrain-free, model-agnostic approach to LLM unlearning that achieves effective unlearning while preserving the utility. Our \texttt{ALU} framework unlearns by involving multiple LLM agents, each designed for a specific step in the unlearning process, without the need to update model weights for any of the agents in the framework. Users can easily request any set of unlearning instances in any sequence, and \texttt{ALU} seamlessly adapts in real time. This is facilitated without requiring any changes in the underlying LLM model. Through extensive experiments on established benchmarks (TOFU, WMDP, WPU) and jailbreaking techniques (many shot, target masking, other languages), we demonstrate that \texttt{ALU} consistently stands out as the most robust inference-time LLM unlearning framework among current state-of-the-art methods while incurring time cost that remains effectively constant regardless of the number of unlearning targets. We further highlight \texttt{ALU}'s superior performance compared to existing methods when evaluated at scale. Specifically, \texttt{ALU} is assessed on up to 1000 unlearning targets, exceeding the evaluation scope of all previously proposed LLM unlearning methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.08324v2">Are LLMs Prescient? A Continuous Evaluation using Daily News as the Oracle</a></div>
    <div class="paper-meta">
      📅 2025-07-08
      | 💬 ICML 2025
    </div>
    <details class="paper-abstract">
      Many existing evaluation benchmarks for Large Language Models (LLMs) quickly become outdated due to the emergence of new models and training data. These benchmarks also fall short in assessing how LLM performance changes over time, as they consist of a static set of questions without a temporal dimension. To address these limitations, we propose using future event prediction as a continuous evaluation method to assess LLMs' temporal generalization and forecasting abilities. Our benchmark, Daily Oracle, automatically generates question-answer (QA) pairs from daily news, challenging LLMs to predict "future" event outcomes. Our findings reveal that as pre-training data becomes outdated, LLM performance degrades over time. While Retrieval Augmented Generation (RAG) has the potential to enhance prediction accuracy, the performance degradation pattern persists, highlighting the need for continuous model updates. Code and data are available at https://agenticlearning.ai/daily-oracle.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06056v1">Entropy-Memorization Law: Evaluating Memorization Difficulty of Data in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are known to memorize portions of their training data, sometimes reproducing content verbatim when prompted appropriately. In this work, we investigate a fundamental yet under-explored question in the domain of memorization: How to characterize memorization difficulty of training data in LLMs? Through empirical experiments on OLMo, a family of open models, we present the Entropy-Memorization Law. It suggests that data entropy is linearly correlated with memorization score. Moreover, in a case study of memorizing highly randomized strings, or "gibberish", we observe that such sequences, despite their apparent randomness, exhibit unexpectedly low empirical entropy compared to the broader training corpus. Adopting the same strategy to discover Entropy-Memorization Law, we derive a simple yet effective approach to distinguish training and testing data, enabling Dataset Inference (DI).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19652v2">Tailored Conversations beyond LLMs: A RL-Based Dialogue Manager</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      In this work, we propose a novel framework that integrates large language models (LLMs) with an RL-based dialogue manager for open-ended dialogue with a specific goal. By leveraging hierarchical reinforcement learning to model the structured phases of dialogue and employ meta-learning to enhance adaptability across diverse user profiles, our approach enhances adaptability and efficiency, enabling the system to learn from limited data, transition fluidly between dialogue phases, and personalize responses to heterogeneous patient needs. We apply our framework to Motivational Interviews, aiming to foster behavior change, and demonstrate that the proposed dialogue manager outperforms a state-of-the-art LLM baseline in terms of reward, showing a potential benefit of conditioning LLMs to create open-ended dialogue systems with specific goals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06043v1">CAVGAN: Unifying Jailbreak and Defense of LLMs via Generative Adversarial Attacks on their Internal Representations</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      Security alignment enables the Large Language Model (LLM) to gain the protection against malicious queries, but various jailbreak attack methods reveal the vulnerability of this security mechanism. Previous studies have isolated LLM jailbreak attacks and defenses. We analyze the security protection mechanism of the LLM, and propose a framework that combines attack and defense. Our method is based on the linearly separable property of LLM intermediate layer embedding, as well as the essence of jailbreak attack, which aims to embed harmful problems and transfer them to the safe area. We utilize generative adversarial network (GAN) to learn the security judgment boundary inside the LLM to achieve efficient jailbreak attack and defense. The experimental results indicate that our method achieves an average jailbreak success rate of 88.85\% across three popular LLMs, while the defense success rate on the state-of-the-art jailbreak dataset reaches an average of 84.17\%. This not only validates the effectiveness of our approach but also sheds light on the internal security mechanisms of LLMs, offering new insights for enhancing model security The code and data are available at https://github.com/NLPGM/CAVGAN.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05984v1">Development and Evaluation of HopeBot: an LLM-based chatbot for structured and interactive PHQ-9 depression screening</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      Static tools like the Patient Health Questionnaire-9 (PHQ-9) effectively screen depression but lack interactivity and adaptability. We developed HopeBot, a chatbot powered by a large language model (LLM) that administers the PHQ-9 using retrieval-augmented generation and real-time clarification. In a within-subject study, 132 adults in the United Kingdom and China completed both self-administered and chatbot versions. Scores demonstrated strong agreement (ICC = 0.91; 45% identical). Among 75 participants providing comparative feedback, 71% reported greater trust in the chatbot, highlighting clearer structure, interpretive guidance, and a supportive tone. Mean ratings (0-10) were 8.4 for comfort, 7.7 for voice clarity, 7.6 for handling sensitive topics, and 7.4 for recommendation helpfulness; the latter varied significantly by employment status and prior mental-health service use (p < 0.05). Overall, 87.1% expressed willingness to reuse or recommend HopeBot. These findings demonstrate voice-based LLM chatbots can feasibly serve as scalable, low-burden adjuncts for routine depression screening.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05886v1">Current Practices for Building LLM-Powered Reasoning Tools Are Ad Hoc -- and We Can Do Better</a></div>
    <div class="paper-meta">
      📅 2025-07-08
      | 💬 6 pages, 4 figures
    </div>
    <details class="paper-abstract">
      There is growing excitement about building software verifiers, synthesizers, and other Automated Reasoning (AR) tools by combining traditional symbolic algorithms and Large Language Models (LLMs). Unfortunately, the current practice for constructing such neurosymbolic AR systems is an ad hoc programming model that does not have the strong guarantees of traditional symbolic algorithms, nor a deep enough synchronization of neural networks and symbolic reasoning to unlock the full potential of LLM-powered reasoning. I propose Neurosymbolic Transition Systems as a principled computational model that can underlie infrastructure for building neurosymbolic AR tools. In this model, symbolic state is paired with intuition, and state transitions operate over symbols and intuition in parallel. I argue why this new paradigm can scale logical reasoning beyond current capabilities while retaining the strong guarantees of symbolic algorithms, and I sketch out how the computational model I propose can be reified in a logic programming language.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05880v1">RecRankerEval: A Flexible and Extensible Framework for Top-k LLM-based Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      A recent Large language model (LLM)-based recommendation model, called RecRanker, has demonstrated a superior performance in the top-k recommendation task compared to other models. In particular, RecRanker samples users via clustering, generates an initial ranking list using an initial recommendation model, and fine-tunes an LLM through hybrid instruction tuning to infer user preferences. However, the contribution of each core component remains underexplored. In this work, we inspect the reproducibility of RecRanker, and study the impact and role of its various components. We begin by reproducing the RecRanker pipeline through the implementation of all its key components. Our reproduction shows that the pairwise and listwise methods achieve a performance comparable to that reported in the original paper. For the pointwise method, while we are also able to reproduce the original paper's results, further analysis shows that the performance is abnormally high due to data leakage from the inclusion of ground-truth information in the prompts. To enable a fair and comprehensive evaluation of LLM-based top-k recommendations, we propose RecRankerEval, an extensible framework that covers five key dimensions: user sampling strategy, initial recommendation model, LLM backbone, dataset selection, and instruction tuning method. Using the RecRankerEval framework, we show that the original results of RecRanker can be reproduced on the ML-100K and ML-1M datasets, as well as the additional Amazon-Music dataset, but not on BookCrossing due to the lack of timestamp information in the original RecRanker paper. Furthermore, we demonstrate that RecRanker's performance can be improved by employing alternative user sampling methods, stronger initial recommenders, and more capable LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.18099v2">Learning to Plan & Reason for Evaluation with Thinking-LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-07-08
      | 💬 ICML 2025
    </div>
    <details class="paper-abstract">
      LLM-as-a-Judge models generate chain-of-thought (CoT) sequences intended to capture the step-bystep reasoning process that underlies the final evaluation of a response. However, due to the lack of human annotated CoTs for evaluation, the required components and structure of effective reasoning traces remain understudied. Consequently, previous approaches often (1) constrain reasoning traces to hand-designed components, such as a list of criteria, reference answers, or verification questions and (2) structure them such that planning is intertwined with the reasoning for evaluation. In this work, we propose EvalPlanner, a preference optimization algorithm for Thinking-LLM-as-a-Judge that first generates an unconstrained evaluation plan, followed by its execution, and then the final judgment. In a self-training loop, EvalPlanner iteratively optimizes over synthetically constructed evaluation plans and executions, leading to better final verdicts. Our method achieves a new state-of-the-art performance for generative reward models on RewardBench (with a score of 93.9), despite being trained on fewer amount of, and synthetically generated, preference pairs. Additional experiments on other benchmarks like RM-Bench, JudgeBench, and FollowBenchEval further highlight the utility of both planning and reasoning for building robust LLM-as-a-Judge reasoning models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05822v1">Video Event Reasoning and Prediction by Fusing World Knowledge from LLMs with Vision Foundation Models</a></div>
    <div class="paper-meta">
      📅 2025-07-08
      | 💬 22 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Current video understanding models excel at recognizing "what" is happening but fall short in high-level cognitive tasks like causal reasoning and future prediction, a limitation rooted in their lack of commonsense world knowledge. To bridge this cognitive gap, we propose a novel framework that synergistically fuses a powerful Vision Foundation Model (VFM) for deep visual perception with a Large Language Model (LLM) serving as a knowledge-driven reasoning core. Our key technical innovation is a sophisticated fusion module, inspired by the Q-Former architecture, which distills complex spatiotemporal and object-centric visual features into a concise, language-aligned representation. This enables the LLM to effectively ground its inferential processes in direct visual evidence. The model is trained via a two-stage strategy, beginning with large-scale alignment pre-training on video-text data, followed by targeted instruction fine-tuning on a curated dataset designed to elicit advanced reasoning and prediction skills. Extensive experiments demonstrate that our model achieves state-of-the-art performance on multiple challenging benchmarks. Notably, it exhibits remarkable zero-shot generalization to unseen reasoning tasks, and our in-depth ablation studies validate the critical contribution of each architectural component. This work pushes the boundary of machine perception from simple recognition towards genuine cognitive understanding, paving the way for more intelligent and capable AI systems in robotics, human-computer interaction, and beyond.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05820v1">Constella: Supporting Storywriters' Interconnected Character Creation through LLM-based Multi-Agents</a></div>
    <div class="paper-meta">
      📅 2025-07-08
      | 💬 50 pages
    </div>
    <details class="paper-abstract">
      Creating a cast of characters by attending to their relational dynamics is a critical aspect of most long-form storywriting. However, our formative study (N=14) reveals that writers struggle to envision new characters that could influence existing ones, to balance similarities and differences among characters, and to intricately flesh out their relationships. Based on these observations, we designed Constella, an LLM-based multi-agent tool that supports storywriters' interconnected character creation process. Constella suggests related characters (FRIENDS DISCOVERY feature), reveals the inner mindscapes of several characters simultaneously (JOURNALS feature), and manifests relationships through inter-character responses (COMMENTS feature). Our 7-8 day deployment study with storywriters (N=11) shows that Constella enabled the creation of expansive communities composed of related characters, facilitated the comparison of characters' thoughts and emotions, and deepened writers' understanding of character relationships. We conclude by discussing how multi-agent interactions can help distribute writers' attention and effort across the character cast.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04841v1">Spec-TOD: A Specialized Instruction-Tuned LLM Framework for Efficient Task-Oriented Dialogue Systems</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 Accepted at SIGdial 2025
    </div>
    <details class="paper-abstract">
      Task-oriented dialogue (TOD) systems facilitate goal-driven interactions between users and machines. While recent advances in deep learning have improved the performance, TOD systems often struggle in low-resource scenarios with limited labeled data. To address this challenge, we propose Spec-TOD, a novel framework designed to train an end-to-end TOD system with limited data. Spec-TOD introduces two main innovations: (i) a novel specialized end-to-end TOD framework that incorporates explicit task instructions for instruction-tuned large language models (LLMs), and (ii) an efficient training strategy that leverages lightweight, specialized LLMs to achieve strong performance with minimal supervision. Experiments on the MultiWOZ dataset, a widely used TOD benchmark, demonstrate that Spec-TOD achieves competitive results while significantly reducing the need for labeled data. These findings highlight the potential of the proposed framework in advancing efficient and effective TOD systems in low-resource settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05853v2">Training-Free Query Optimization via LLM-Based Plan Similarity</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 18 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large language model (LLM) embeddings offer a promising new avenue for database query optimization. In this paper, we explore how pre-trained execution plan embeddings can guide SQL query execution without the need for additional model training. We introduce LLM-PM (LLM-based Plan Mapping), a framework that embeds the default execution plan of a query, finds its k nearest neighbors among previously executed plans, and recommends database hintsets based on neighborhood voting. A lightweight consistency check validates the selected hint, while a fallback mechanism searches the full hint space when needed. Evaluated on the JOB-CEB benchmark using OpenGauss, LLM-PM achieves an average speed-up of 21% query latency reduction. This work highlights the potential of LLM-powered embeddings to deliver practical improvements in query performance and opens new directions for training-free, embedding-based optimizer guidance systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04766v1">ABench-Physics: Benchmarking Physical Reasoning in LLMs via High-Difficulty and Dynamic Physics Problems</a></div>
    <div class="paper-meta">
      📅 2025-07-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown impressive performance in domains such as mathematics and programming, yet their capabilities in physics remain underexplored and poorly understood. Physics poses unique challenges that demand not only precise computation but also deep conceptual understanding and physical modeling skills. Existing benchmarks often fall short due to limited difficulty, multiple-choice formats, and static evaluation settings that fail to capture physical modeling ability. In this paper, we introduce ABench-Physics, a novel benchmark designed to rigorously evaluate LLMs' physical reasoning and generalization capabilities. ABench-Physics consists of two components: Phy_A, a static set of 400 graduate- or Olympiad-level problems; and Phy_B, a dynamic subset of 100 problems equipped with an automatic variation engine to test model robustness across changing conditions. All questions require precise numerical answers, with strict formatting and tolerance constraints. Our evaluation of several state-of-the-art LLMs reveals substantial performance gaps, highlighting persistent limitations in physical reasoning, especially in generalization to dynamic variants. ABench-Physics provides a challenging and diagnostic framework for advancing scientific reasoning in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04751v1">LLMs as Architects and Critics for Multi-Source Opinion Summarization</a></div>
    <div class="paper-meta">
      📅 2025-07-07
    </div>
    <details class="paper-abstract">
      Multi-source Opinion Summarization (M-OS) extends beyond traditional opinion summarization by incorporating additional sources of product metadata such as descriptions, key features, specifications, and ratings, alongside reviews. This integration results in comprehensive summaries that capture both subjective opinions and objective product attributes essential for informed decision-making. While Large Language Models (LLMs) have shown significant success in various Natural Language Processing (NLP) tasks, their potential in M-OS remains largely unexplored. Additionally, the lack of evaluation datasets for this task has impeded further advancements. To bridge this gap, we introduce M-OS-EVAL, a benchmark dataset for evaluating multi-source opinion summaries across 7 key dimensions: fluency, coherence, relevance, faithfulness, aspect coverage, sentiment consistency, specificity. Our results demonstrate that M-OS significantly enhances user engagement, as evidenced by a user study in which, on average, 87% of participants preferred M-OS over opinion summaries. Our experiments demonstrate that factually enriched summaries enhance user engagement. Notably, M-OS-PROMPTS exhibit stronger alignment with human judgment, achieving an average Spearman correlation of \r{ho} = 0.74, which surpasses the performance of previous methodologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04748v1">LLM-based Question-Answer Framework for Sensor-driven HVAC System Interaction</a></div>
    <div class="paper-meta">
      📅 2025-07-07
    </div>
    <details class="paper-abstract">
      Question-answering (QA) interfaces powered by large language models (LLMs) present a promising direction for improving interactivity with HVAC system insights, particularly for non-expert users. However, enabling accurate, real-time, and context-aware interactions with HVAC systems introduces unique challenges, including the integration of frequently updated sensor data, domain-specific knowledge grounding, and coherent multi-stage reasoning. In this paper, we present JARVIS, a two-stage LLM-based QA framework tailored for sensor data-driven HVAC system interaction. JARVIS employs an Expert-LLM to translate high-level user queries into structured execution instructions, and an Agent that performs SQL-based data retrieval, statistical processing, and final response generation. To address HVAC-specific challenges, JARVIS integrates (1) an adaptive context injection strategy for efficient HVAC and deployment-specific information integration, (2) a parameterized SQL builder and executor to improve data access reliability, and (3) a bottom-up planning scheme to ensure consistency across multi-stage response generation. We evaluate JARVIS using real-world data collected from a commercial HVAC system and a ground truth QA dataset curated by HVAC experts to demonstrate its effectiveness in delivering accurate and interpretable responses across diverse queries. Results show that JARVIS consistently outperforms baseline and ablation variants in both automated and user-centered assessments, achieving high response quality and accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04736v1">ChipSeek-R1: Generating Human-Surpassing RTL with LLM via Hierarchical Reward-Driven Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-07-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) show significant potential for automating Register-Transfer Level (RTL) code generation. However, current approaches face a critical challenge: they can not simultaneously optimize for functional correctness and hardware quality (Power, Performance, Area - PPA). Methods based on supervised fine-tuning often generate functionally correct but PPA-suboptimal code, lacking mechanisms to learn optimization principles. In contrast, post-processing techniques that attempt to improve PPA metrics after generation are often inefficient because they operate externally without updating the LLM's parameters, thus failing to enhance the model's intrinsic design capabilities. To bridge this gap, we introduce ChipSeek-R1, a hierarchical reward-driven reinforcement learning framework to train LLMs to generate RTL code that achieves both functional correctness and optimized PPA metrics. ChipSeek-R1 employs a hierarchical reward system, which incorporates direct feedback on syntax, functional correctness (from simulators) and PPA metrics (from synthesis tools) during reinforcement learning. This enables the model to learn complex hardware design trade-offs via trial-and-error, generating RTL code that is both functionally correct and PPA-optimized. Evaluating ChipSeek-R1 on standard benchmarks (VerilogEval, RTLLM), we achieve state-of-the-art results in functional correctness. Notably, on the RTLLM benchmark, ChipSeek-R1 generated 27 RTL designs surpassing the PPA metrics of the original human-written code. Our findings demonstrate the effectiveness of integrating toolchain feedback into LLM training and highlight the potential for reinforcement learning to enable automated generation of human-surpassing RTL code. We open-source our code in anonymous github.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.22139v2">Q-Frame: Query-aware Frame Selection and Multi-Resolution Adaptation for Video-LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 Accepted at ICCV 2025
    </div>
    <details class="paper-abstract">
      Multimodal Large Language Models (MLLMs) have demonstrated significant success in visual understanding tasks. However, challenges persist in adapting these models for video comprehension due to the large volume of data and temporal complexity. Existing Video-LLMs using uniform frame sampling often struggle to capture the query-related crucial spatiotemporal clues of videos effectively. In this paper, we introduce Q-Frame, a novel approach for adaptive frame selection and multi-resolution scaling tailored to the video's content and the specific query. Q-Frame employs a training-free, plug-and-play strategy generated by a text-image matching network like CLIP, utilizing the Gumbel-Max trick for efficient frame selection. Q-Frame allows Video-LLMs to process more frames without exceeding computational limits, thereby preserving critical temporal and spatial information. We demonstrate Q-Frame's effectiveness through extensive experiments on benchmark datasets, including MLVU, LongVideoBench, and Video-MME, illustrating its superiority over existing methods and its applicability across various video understanding tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04724v1">Who's the Mole? Modeling and Detecting Intention-Hiding Malicious Agents in LLM-Based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-07-07
    </div>
    <details class="paper-abstract">
      Multi-agent systems powered by Large Language Models (LLM-MAS) demonstrate remarkable capabilities in collaborative problem-solving. While LLM-MAS exhibit strong collaborative abilities, the security risks in their communication and coordination remain underexplored. We bridge this gap by systematically investigating intention-hiding threats in LLM-MAS, and design four representative attack paradigms that subtly disrupt task completion while maintaining high concealment. These attacks are evaluated in centralized, decentralized, and layered communication structures. Experiments conducted on six benchmark datasets, including MMLU, MMLU-Pro, HumanEval, GSM8K, arithmetic, and biographies, demonstrate that they exhibit strong disruptive capabilities. To identify these threats, we propose a psychology-based detection framework AgentXposed, which combines the HEXACO personality model with the Reid Technique, using progressive questionnaire inquiries and behavior-based monitoring. Experiments conducted on six types of attacks show that our detection framework effectively identifies all types of malicious behaviors. The detection rate for our intention-hiding attacks is slightly lower than that of the two baselines, Incorrect Fact Injection and Dark Traits Injection, demonstrating the effectiveness of intention concealment. Our findings reveal the structural and behavioral risks posed by intention-hiding attacks and offer valuable insights into securing LLM-based multi-agent systems through psychological perspectives, which contributes to a deeper understanding of multi-agent safety. The code and data are available at https://anonymous.4open.science/r/AgentXposed-F814.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.12112v4">Balancing Act: Prioritization Strategies for LLM-Designed Restless Bandit Rewards</a></div>
    <div class="paper-meta">
      📅 2025-07-07
    </div>
    <details class="paper-abstract">
      LLMs are increasingly used to design reward functions based on human preferences in Reinforcement Learning (RL). We focus on LLM-designed rewards for Restless Multi-Armed Bandits, a framework for allocating limited resources among agents. In applications such as public health, this approach empowers grassroots health workers to tailor automated allocation decisions to community needs. In the presence of multiple agents, altering the reward function based on human preferences can impact subpopulations very differently, leading to complex tradeoffs and a multi-objective resource allocation problem. We are the first to present a principled method termed Social Choice Language Model for dealing with these tradeoffs for LLM-designed rewards for multiagent planners in general and restless bandits in particular. The novel part of our model is a transparent and configurable selection component, called an adjudicator, external to the LLM that controls complex tradeoffs via a user-selected social welfare function. Our experiments demonstrate that our model reliably selects more effective, aligned, and balanced reward functions compared to purely LLM-based approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04687v1">AKEGEN: A LLM-based Tabular Corpus Generator for Evaluating Dataset Discovery in Data Lakes</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 13 pages
    </div>
    <details class="paper-abstract">
      How to generate a large, realistic set of tables along with joinability relationships, to stress-test dataset discovery methods? Dataset discovery methods aim to automatically identify related data assets in a data lake. The development and evaluation of such solutions for customers from a wide range of business domains, relies on diverse, high quality and domain-specific tabular benchmarks. Large language models (LLMs) are trained on a wide variety of text data, which can provide a strong foundation of general and domain-specific knowledge. In this paper, we ask the question -- \textit{can we leverage LLMs to generate a tabular benchmark adequate for evaluating the dataset discovery solutions?} In particular, we focus on the task of finding joinable tables which is the cornerstone of virtually every dataset discovery method. Current corpora for evaluating dataset discovery methods are mainly based on subsets of open data, and they suffer from three important issues: $i)$ they focus on very common and generic data types (e.g., address, id, name, etc.); $ii)$ they do not contain human-annotated column pairs; instead, practitioners synthesize ground truth using table splits (e.g., horizontal for table union search and vertical ones for joinability) and $iii)$ they do not focus on semantic column relationships.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04664v1">VectorLLM: Human-like Extraction of Structured Building Contours vis Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-07
    </div>
    <details class="paper-abstract">
      Automatically extracting vectorized building contours from remote sensing imagery is crucial for urban planning, population estimation, and disaster assessment. Current state-of-the-art methods rely on complex multi-stage pipelines involving pixel segmentation, vectorization, and polygon refinement, which limits their scalability and real-world applicability. Inspired by the remarkable reasoning capabilities of Large Language Models (LLMs), we introduce VectorLLM, the first Multi-modal Large Language Model (MLLM) designed for regular building contour extraction from remote sensing images. Unlike existing approaches, VectorLLM performs corner-point by corner-point regression of building contours directly, mimicking human annotators' labeling process. Our architecture consists of a vision foundation backbone, an MLP connector, and an LLM, enhanced with learnable position embeddings to improve spatial understanding capability. Through comprehensive exploration of training strategies including pretraining, supervised fine-tuning, and preference optimization across WHU, WHU-Mix, and CrowdAI datasets, VectorLLM significantly outperformed the previous SOTA methods by 5.6 AP, 7.1 AP, 13.6 AP, respectively in the three datasets. Remarkably, VectorLLM exhibits strong zero-shot performance on unseen objects including aircraft, water bodies, and oil tanks, highlighting its potential for unified modeling of diverse remote sensing object contour extraction tasks. Overall, this work establishes a new paradigm for vector extraction in remote sensing, leveraging the topological reasoning capabilities of LLMs to achieve both high accuracy and exceptional generalization. All the codes and weights will be published for promoting community development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04626v1">Heterogeneous User Modeling for LLM-based Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 Accepted by RecSys 2025
    </div>
    <details class="paper-abstract">
      Leveraging Large Language Models (LLMs) for recommendation has demonstrated notable success in various domains, showcasing their potential for open-domain recommendation. A key challenge to advancing open-domain recommendation lies in effectively modeling user preferences from users' heterogeneous behaviors across multiple domains. Existing approaches, including ID-based and semantic-based modeling, struggle with poor generalization, an inability to compress noisy interactions effectively, and the domain seesaw phenomenon. To address these challenges, we propose a Heterogeneous User Modeling (HUM) method, which incorporates a compression enhancer and a robustness enhancer for LLM-based recommendation. The compression enhancer uses a customized prompt to compress heterogeneous behaviors into a tailored token, while a masking mechanism enhances cross-domain knowledge extraction and understanding. The robustness enhancer introduces a domain importance score to mitigate the domain seesaw phenomenon by guiding domain optimization. Extensive experiments on heterogeneous datasets validate that HUM effectively models user heterogeneity by achieving both high efficacy and robustness, leading to superior performance in open-domain recommendation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.17378v4">A Text is Worth Several Tokens: Text Embedding from LLMs Secretly Aligns Well with The Key Tokens</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 ACL2025 Oral
    </div>
    <details class="paper-abstract">
      Text embeddings from large language models (LLMs) have achieved excellent results in tasks such as information retrieval, semantic textual similarity, etc. In this work, we show an interesting finding: when feeding a text into the LLM-based embedder, the obtained text embedding will be able to be aligned with the key tokens in the input text. We first fully analyze this phenomenon on eight LLM-based embedders and show that this phenomenon is universal and is not affected by model architecture, training strategy, and embedding method. With a deeper analysis, we find that the main change in embedding space between these embedders and their LLM backbones is in the first principal component. By adjusting the first principal component, we can align text embedding with the key tokens. Finally, we give several examples to demonstrate the vast application potential of this finding: (1) we propose a simple and practical sparse retrieval method based on the aligned tokens, which can achieve 80% of the dense retrieval effect of the same model while reducing the computation significantly; (2) we show that our findings provide a novel perspective to help understand novel technologies (e.g., instruction-following embedding) and fuzzy concepts (e.g., semantic relatedness vs. similarity) in this field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04623v1">Hierarchical Intent-guided Optimization with Pluggable LLM-Driven Semantics for Session-based Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-07-07
    </div>
    <details class="paper-abstract">
      Session-based Recommendation (SBR) aims to predict the next item a user will likely engage with, using their interaction sequence within an anonymous session. Existing SBR models often focus only on single-session information, ignoring inter-session relationships and valuable cross-session insights. Some methods try to include inter-session data but struggle with noise and irrelevant information, reducing performance. Additionally, most models rely on item ID co-occurrence and overlook rich semantic details, limiting their ability to capture fine-grained item features. To address these challenges, we propose a novel hierarchical intent-guided optimization approach with pluggable LLM-driven semantic learning for session-based recommendations, called HIPHOP. First, we introduce a pluggable embedding module based on large language models (LLMs) to generate high-quality semantic representations, enhancing item embeddings. Second, HIPHOP utilizes graph neural networks (GNNs) to model item transition relationships and incorporates a dynamic multi-intent capturing module to address users' diverse interests within a session. Additionally, we design a hierarchical inter-session similarity learning module, guided by user intent, to capture global and local session relationships, effectively exploring users' long-term and short-term interests. To mitigate noise, an intent-guided denoising strategy is applied during inter-session learning. Finally, we enhance the model's discriminative capability by using contrastive learning to optimize session representations. Experiments on multiple datasets show that HIPHOP significantly outperforms existing methods, demonstrating its effectiveness in improving recommendation quality. Our code is available: https://github.com/hjx159/HIPHOP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04610v1">any4: Learned 4-bit Numeric Representation for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 ICML 2025
    </div>
    <details class="paper-abstract">
      We present any4, a learned 4-bit weight quantization solution for large language models (LLMs) providing arbitrary numeric representations without requiring pre-processing of weights or activations. any4 yields higher accuracy compared to other related 4-bit numeric representation types: int4, fp4 and nf4, as evaluated on a range of model sizes, generations and families (Llama 2, Llama 3, Mistral and Mixtral). While any4 does not require preprocessing of weights or activations, it is also competitive with orthogonal techniques that require such preprocessing (e.g., AWQ and GPTQ). We also experiment with any3 and any2 and show competitiveness at lower bits. Additionally, we show that we can calibrate using a single curated diverse sample rather than hundreds of samples from a dataset as done in most quantization approaches. We also open source tinygemm, a latency optimized GPU matrix multiplication library for LLMs, that implements any4 using a GPU-efficient lookup table strategy along with other common quantization methods. We open source our code at https://github.com/facebookresearch/any4 .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20666v2">Inside you are many wolves: Using cognitive models to interpret value trade-offs in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 11 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Navigating everyday social situations often requires juggling conflicting goals, such as conveying a harsh truth, maintaining trust, all while still being mindful of another person's feelings. These value trade-offs are an integral part of human decision-making and language use, however, current tools for interpreting such dynamic and multi-faceted notions of values in LLMs are limited. In cognitive science, so-called "cognitive models" provide formal accounts of these trade-offs in humans, by modeling the weighting of a speaker's competing utility functions in choosing an action or utterance. In this work, we use a leading cognitive model of polite speech to interpret the extent to which LLMs represent human-like trade-offs. We apply this lens to systematically evaluate value trade-offs in two encompassing model settings: degrees of reasoning "effort" in frontier black-box models, and RL post-training dynamics of open-source models. Our results highlight patterns of higher informational utility than social utility in reasoning models, and in open-source models shown to be stronger in mathematical reasoning. Our findings from LLMs' training dynamics suggest large shifts in utility values early on in training with persistent effects of the choice of base model and pretraining data, compared to feedback dataset or alignment method. We show that our method is responsive to diverse aspects of the rapidly evolving LLM landscape, with insights for forming hypotheses about other high-level behaviors, shaping training regimes for reasoning models, and better controlling trade-offs between values during model training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05541v1">SenseCF: LLM-Prompted Counterfactuals for Intervention and Sensor Data Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 In review
    </div>
    <details class="paper-abstract">
      Counterfactual explanations (CFs) offer human-centric insights into machine learning predictions by highlighting minimal changes required to alter an outcome. Therefore, CFs can be used as (i) interventions for abnormality prevention and (ii) augmented data for training robust models. In this work, we explore large language models (LLMs), specifically GPT-4o-mini, for generating CFs in a zero-shot and three-shot setting. We evaluate our approach on two datasets: the AI-Readi flagship dataset for stress prediction and a public dataset for heart disease detection. Compared to traditional methods such as DiCE, CFNOW, and NICE, our few-shot LLM-based approach achieves high plausibility (up to 99%), strong validity (up to 0.99), and competitive sparsity. Moreover, using LLM-generated CFs as augmented samples improves downstream classifier performance (an average accuracy gain of 5%), especially in low-data regimes. This demonstrates the potential of prompt-based generative techniques to enhance explainability and robustness in clinical and physiological prediction tasks. Code base: github.com/anonymous/SenseCF.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18116v3">Bayesian Optimization for Controlled Image Editing via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 8 figures, accept at ACL2025 Findings
    </div>
    <details class="paper-abstract">
      In the rapidly evolving field of image generation, achieving precise control over generated content and maintaining semantic consistency remain significant limitations, particularly concerning grounding techniques and the necessity for model fine-tuning. To address these challenges, we propose BayesGenie, an off-the-shelf approach that integrates Large Language Models (LLMs) with Bayesian Optimization to facilitate precise and user-friendly image editing. Our method enables users to modify images through natural language descriptions without manual area marking, while preserving the original image's semantic integrity. Unlike existing techniques that require extensive pre-training or fine-tuning, our approach demonstrates remarkable adaptability across various LLMs through its model-agnostic design. BayesGenie employs an adapted Bayesian optimization strategy to automatically refine the inference process parameters, achieving high-precision image editing with minimal user intervention. Through extensive experiments across diverse scenarios, we demonstrate that our framework significantly outperforms existing methods in both editing accuracy and semantic preservation, as validated using different LLMs including Claude3 and GPT-4.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05528v1">Conversational Education at Scale: A Multi-LLM Agent Workflow for Procedural Learning and Pedagogic Quality Assessment</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 14 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have advanced virtual educators and learners, bridging NLP with AI4Education. Existing work often lacks scalability and fails to leverage diverse, large-scale course content, with limited frameworks for assessing pedagogic quality. To this end, we propose WikiHowAgent, a multi-agent workflow leveraging LLMs to simulate interactive teaching-learning conversations. It integrates teacher and learner agents, an interaction manager, and an evaluator to facilitate procedural learning and assess pedagogic quality. We introduce a dataset of 114,296 teacher-learner conversations grounded in 14,287 tutorials across 17 domains and 727 topics. Our evaluation protocol combines computational and rubric-based metrics with human judgment alignment. Results demonstrate the workflow's effectiveness in diverse setups, offering insights into LLM capabilities across domains. Our datasets and implementations are fully open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05504v1">Tool for Supporting Debugging and Understanding of Normative Requirements Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-07
    </div>
    <details class="paper-abstract">
      Normative requirements specify social, legal, ethical, empathetic, and cultural (SLEEC) norms that must be observed by a system. To support the identification of SLEEC requirements, numerous standards and regulations have been developed. These requirements are typically defined by stakeholders in the non-technical system with diverse expertise (e.g., ethicists, lawyers, social scientists). Hence, ensuring their consistency and managing the requirement elicitation process are complex and error-prone tasks. Recent research has addressed this challenge using domain-specific languages to specify normative requirements as rules, whose consistency can then be analyzed with formal methods. Nevertheless, these approaches often present the results from formal verification tools in a way that is inaccessible to non-technical users. This hinders understanding and makes the iterative process of eliciting and validating these requirements inefficient in terms of both time and effort. To address this problem, we introduce SLEEC-LLM, a tool that uses large language models (LLMs) to provide natural-language interpretations for model-checking counterexamples corresponding to SLEEC rule inconsistencies. SLEEC-LLM improves the efficiency and explainability of normative requirements elicitation and consistency analysis. To demonstrate its effectiveness, we summarise its use in two real-world case studies involving non-technical stakeholders.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05461v1">GLOSS: Group of LLMs for Open-Ended Sensemaking of Passive Sensing Data for Health and Wellbeing</a></div>
    <div class="paper-meta">
      📅 2025-07-07
    </div>
    <details class="paper-abstract">
      The ubiquitous presence of smartphones and wearables has enabled researchers to build prediction and detection models for various health and behavior outcomes using passive sensing data from these devices. Achieving a high-level, holistic understanding of an individual's behavior and context, however, remains a significant challenge. Due to the nature of passive sensing data, sensemaking -- the process of interpreting and extracting insights -- requires both domain knowledge and technical expertise, creating barriers for different stakeholders. Existing systems designed to support sensemaking are either not open-ended or cannot perform complex data triangulation. In this paper, we present a novel sensemaking system, Group of LLMs for Open-ended Sensemaking (GLOSS), capable of open-ended sensemaking and performing complex multimodal triangulation to derive insights. We demonstrate that GLOSS significantly outperforms the commonly used Retrieval-Augmented Generation (RAG) technique, achieving 87.93% accuracy and 66.19% consistency, compared to RAG's 29.31% accuracy and 52.85% consistency. Furthermore, we showcase the promise of GLOSS through four use cases inspired by prior and ongoing work in the UbiComp and HCI communities. Finally, we discuss the potential of GLOSS, its broader implications, and the limitations of our work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.04798v3">From LLMs to Actions: Latent Codes as Bridges in Hierarchical Robot Control</a></div>
    <div class="paper-meta">
      📅 2025-07-07
    </div>
    <details class="paper-abstract">
      Hierarchical control for robotics has long been plagued by the need to have a well defined interface layer to communicate between high-level task planners and low-level policies. With the advent of LLMs, language has been emerging as a prospective interface layer. However, this has several limitations. Not all tasks can be decomposed into steps that are easily expressible in natural language (e.g. performing a dance routine). Further, it makes end-to-end finetuning on embodied data challenging due to domain shift and catastrophic forgetting. We introduce our method -- Learnable Latent Codes as Bridges (LCB) -- as an alternate architecture to overcome these limitations. \method~uses a learnable latent code to act as a bridge between LLMs and low-level policies. This enables LLMs to flexibly communicate goals in the task plan without being entirely constrained by language limitations. Additionally, it enables end-to-end finetuning without destroying the embedding space of word tokens learned during pre-training. Through experiments on Language Table and Calvin, two common language based benchmarks for embodied agents, we find that \method~outperforms baselines (including those w/ GPT-4V) that leverage pure language as the interface layer on tasks that require reasoning and multi-step behaviors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.17172v2">What Would You Ask When You First Saw $a^2+b^2=c^2$? Evaluating LLM on Curiosity-Driven Questioning</a></div>
    <div class="paper-meta">
      📅 2025-07-07
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can store a massive amount of knowledge, yet their potential to acquire new knowledge remains unknown. We propose a novel evaluation framework that evaluates this capability. This framework prompts LLMs to generate questions about a statement introducing scientific knowledge, simulating a curious person when facing the statement for the first time. We score the qualities of the generated questions, thereby evaluating the knowledge acquisition potential of the LLM. We apply controlled ablation studies to validate our scoring procedures. Additionally, we created a synthetic dataset consisting of 1101 statements in physics, chemistry, and maths with distinct levels of difficulties, 300 general knowledge statements, and 567 incorrect statements. Human evaluations were conducted to validate our model assessments, achieving an approximate weighted Cohen's kappa of 0.7 on all three metrics considered. We find that while large models like GPT-4 and Mistral 8x7b are adept at generating coherent and relevant questions, the smaller Phi-2 model is equally or more effective. This indicates that size does not solely determine a model's knowledge acquisition potential. The proposed framework quantifies a critical model capability that was commonly overlooked and opens up research opportunities for developing more knowledgeable AI systems
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05403v1">PBE Meets LLM: When Few Examples Aren't Few-Shot Enough</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 7 pages, 5 figures, accepted by VLDB QDB'25 workshop
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can generate code from natural language descriptions. Their performance is typically evaluated using programming benchmarks that simulate real-world tasks. These benchmarks provide specifications in the form of docstrings, function signatures, or bug reports. The model then generates a program, which is tested against predefined test cases. In contrast, Programming by Example (PBE) uses input-output examples as the specification. Traditional PBE systems rely on search-based methods over restricted transformation spaces. They are usually designed for narrow domains and fixed input formats. It remains unclear how well LLMs perform on PBE tasks. In this work, we evaluate LLMs on PBE tasks involving tabular data transformations. We prompt models to generate functions that convert an input table to an output table. We test the generated functions on unseen inputs to measure accuracy. Our study includes multiple LLMs and evaluates different prompting strategies, such as one-shot vs. multi-try. We also compare performance with and without PBE-specific knowledge. Finally, we propose a hybrid method that calls a traditional PBE solver first, and then falls back to LLMs if necessary. Our results show that LLMs support more diverse input formats and achieve higher accuracy than conventional methods. However, they struggle with tasks that contain ambiguity. The hybrid approach improves overall success by combining the strengths of both approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05330v1">MindFlow: Revolutionizing E-commerce Customer Support with Multimodal LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-07-07
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have enabled new applications in e-commerce customer service. However, their capabilities remain constrained in complex, multimodal scenarios. We present MindFlow, the first open-source multimodal LLM agent tailored for e-commerce. Built on the CoALA framework, it integrates memory, decision-making, and action modules, and adopts a modular "MLLM-as-Tool" strategy for effect visual-textual reasoning. Evaluated via online A/B testing and simulation-based ablation, MindFlow demonstrates substantial gains in handling complex queries, improving user satisfaction, and reducing operational costs, with a 93.53% relative improvement observed in real-world deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11679v2">LLMs on support of privacy and security of mobile apps: state of the art and research directions</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 I am writing to respectfully request the withdrawal of my recent submission to arXiv due to an authorship issue. The paper was submitted without the explicit consent of two co-authors. After internal discussion, they have expressed clear disagreement with the submission and raised concerns about unresolved academic inaccuracies in the current version
    </div>
    <details class="paper-abstract">
      Modern life has witnessed the explosion of mobile devices. However, besides the valuable features that bring convenience to end users, security and privacy risks still threaten users of mobile apps. The increasing sophistication of these threats in recent years has underscored the need for more advanced and efficient detection approaches. In this chapter, we explore the application of Large Language Models (LLMs) to identify security risks and privacy violations and mitigate them for the mobile application ecosystem. By introducing state-of-the-art research that applied LLMs to mitigate the top 10 common security risks of smartphone platforms, we highlight the feasibility and potential of LLMs to replace traditional analysis methods, such as dynamic and hybrid analysis of mobile apps. As a representative example of LLM-based solutions, we present an approach to detect sensitive data leakage when users share images online, a common behavior of smartphone users nowadays. Finally, we discuss open research challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05305v1">Narrowing the Gap: Supervised Fine-Tuning of Open-Source LLMs as a Viable Alternative to Proprietary Models for Pedagogical Tools</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 7 pages, 3 tables, 1 figure
    </div>
    <details class="paper-abstract">
      Frontier Large language models (LLMs) like ChatGPT and Gemini can decipher cryptic compiler errors for novice programmers, but their computational scale, cost, and tendency to over-assist make them problematic for widespread pedagogical adoption. This work demonstrates that smaller, specialised language models, enhanced via Supervised Fine-Tuning (SFT), present a more viable alternative for educational tools. We utilise a new dataset of 40,000 C compiler error explanations, derived from real introductory programming (CS1/2) student-generated programming errors, which we used to fine-tune three open-source models: Qwen3-4B, Llama-3.1-8B, and Qwen3-32B. We performed a dual evaluation, combining expert human reviews with a large-scale automated analysis of 8,000 responses using a validated LLM-as-judge ensemble. Our results show that SFT significantly boosts the pedagogical quality of smaller models, achieving performance comparable to much larger models. We analyse the trade-offs between model size and quality, confirming that fine-tuning compact, efficient models on high-quality, domain-specific data is a potent strategy for creating specialised models to drive educational tools. We provide a replicable methodology to foster broader access to generative AI capabilities in educational contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07060v1">DeepRetro: Retrosynthetic Pathway Discovery using Iterative LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 51 pages,
    </div>
    <details class="paper-abstract">
      Retrosynthesis, the identification of precursor molecules for a target compound, is pivotal for synthesizing complex molecules, but faces challenges in discovering novel pathways beyond predefined templates. Recent large language model (LLM) approaches to retrosynthesis have shown promise but effectively harnessing LLM reasoning capabilities for effective multi-step planning remains an open question. To address this challenge, we introduce DeepRetro, an open-source, iterative, hybrid LLM-based retrosynthetic framework. Our approach integrates the strengths of conventional template-based/Monte Carlo tree search tools with the generative power of LLMs in a step-wise, feedback-driven loop. Initially, synthesis planning is attempted with a template-based engine. If this fails, the LLM subsequently proposes single-step retrosynthetic disconnections. Crucially, these suggestions undergo rigorous validity, stability, and hallucination checks before the resulting precursors are recursively fed back into the pipeline for further evaluation. This iterative refinement allows for dynamic pathway exploration and correction. We demonstrate the potential of this pipeline through benchmark evaluations and case studies, showcasing its ability to identify viable and potentially novel retrosynthetic routes. In particular, we develop an interactive graphical user interface that allows expert human chemists to provide human-in-the-loop feedback to the reasoning algorithm. This approach successfully generates novel pathways for complex natural product compounds, demonstrating the potential for iterative LLM reasoning to advance state-of-art in complex chemical syntheses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05258v1">Spatio-Temporal LLM: Reasoning about Environments and Actions</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 Code and data are available at https://zoezheng126.github.io/STLLM-website/
    </div>
    <details class="paper-abstract">
      Despite the significant recent progress of Multimodal Large Language Models (MLLMs), MLLMs still struggle to correctly answer prompts that require a holistic spatio-temporal understanding. Specifically, it is challenging to address prompts that refer to 1) the entirety of an environment that an agent equipped with an MLLM can operate in; and simultaneously also refer to 2) recent actions that just happened and are encoded in a video clip. However, such a holistic spatio-temporal understanding is important for agents operating in the real world. To address this issue, we first develop a framework to collect a large-scale dataset. Using the collected "Reasoning about Environments and Actions" (REA) dataset, we show that recent methods indeed struggle to correctly answer the prompts. To improve, we develop a "spatio-temporal LLM" (ST-LLM), a model equipped with projectors to improve both spatial understanding of an environment and temporal understanding of recent observations. On the collected REA data, we show that the proposed method significantly improves results compared to prior work. Code and data are available at https://zoezheng126.github.io/STLLM-website/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05257v1">Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 23 Pages, Y. Hu and Y. Wang contribute equally
    </div>
    <details class="paper-abstract">
      Recent benchmarks for Large Language Model (LLM) agents primarily focus on evaluating reasoning, planning, and execution capabilities, while another critical component-memory, encompassing how agents memorize, update, and retrieve long-term information-is under-evaluated due to the lack of benchmarks. We term agents with memory mechanisms as memory agents. In this paper, we identify four core competencies essential for memory agents: accurate retrieval, test-time learning, long-range understanding, and conflict resolution. Existing datasets either rely on limited context lengths or are tailored for static, long-context settings like book-based QA, which do not reflect the interactive, multi-turn nature of memory agents that incrementally accumulate information. Furthermore, no existing benchmarks cover all four competencies. Therefore, we introduce MemoryAgentBench, a new benchmark specifically designed for memory agents. Our benchmark combines reformulated existing datasets with newly constructed ones, covering the above four memory competencies, providing a systematic and challenging testbed for assessing memory quality. We evaluate a diverse set of memory agents, ranging from simple context-based and retrieval-augmented generation (RAG) systems to advanced agents with external memory modules and tool integration. Empirical results reveal that current methods fall short of mastering all four competencies, underscoring the need for further research into comprehensive memory mechanisms for LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05606v3">OPeRA: A Dataset of Observation, Persona, Rationale, and Action for Evaluating LLMs on Human Online Shopping Behavior Simulation</a></div>
    <div class="paper-meta">
      📅 2025-07-07
    </div>
    <details class="paper-abstract">
      Can large language models (LLMs) accurately simulate the next web action of a specific user? While LLMs have shown promising capabilities in generating ``believable'' human behaviors, evaluating their ability to mimic real user behaviors remains an open challenge, largely due to the lack of high-quality, publicly available datasets that capture both the observable actions and the internal reasoning of an actual human user. To address this gap, we introduce OPERA, a novel dataset of Observation, Persona, Rationale, and Action collected from real human participants during online shopping sessions. OPERA is the first public dataset that comprehensively captures: user personas, browser observations, fine-grained web actions, and self-reported just-in-time rationales. We developed both an online questionnaire and a custom browser plugin to gather this dataset with high fidelity. Using OPERA, we establish the first benchmark to evaluate how well current LLMs can predict a specific user's next action and rationale with a given persona and <observation, action, rationale> history. This dataset lays the groundwork for future research into LLM agents that aim to act as personalized digital twins for human.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05228v1">Cascade: Token-Sharded Private LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 To be published in ICML 2025 Main Proceedings as "Hidden No More: Attacking and Defending Private Third-Party LLM Inference", together with arXiv:2505.18332
    </div>
    <details class="paper-abstract">
      As LLMs continue to increase in parameter size, the computational resources required to run them are available to fewer parties. Therefore, third-party inference services -- where LLMs are hosted by third parties with significant computational resources -- are becoming increasingly popular. However, third party inference raises critical concerns about user data privacy. To mitigate these risks, privacy researchers have developed provably secure schemes for third-party inference, such as Secure Multi-Party Computation (SMPC). However, SMPC protocols have significant computational and communication overhead, and do not scale to large models. In this work, we propose a new multi-party inference protocol, Cascade, that avoids these punitive costs by leveraging sharding in the sequence dimension to maintain privacy, trading off cryptographic privacy guarantees for increased performance and scalability. We demonstrate that Cascade is resistant to a generalization of a recent attack that is highly effective against other statistical privacy schemes, and that it is further resistant to learning-based attacks. As Cascade is orders of magnitude faster than existing schemes, our findings offer practical solutions for secure deployment of modern state-of-the-art LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23824v2">Reviewing Scientific Papers for Critical Problems With Reasoning LLMs: Baseline Approaches and Automatic Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 Add results from new experiments; update discussion and GitHub link
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models have sparked interest in utilizing them to aid the peer review process of scientific publication amid the peer review crisis. However, having AI models generate full reviews in the same way as human reviewers risks exacerbating the irresponsible use of LLM-generated reviews. As an alternative, we propose adopting LLMs as manuscript quality checkers. We introduce several baseline approaches and an extendable automatic evaluation framework using top reasoning LLMs as judges to tackle the difficulty of recruiting domain experts for manual evaluation. Utilizing papers withdrawn from arXiv, we validated our proposed methods with several leading reasoning LLMs from multiple vendors and assessed their performance and API costs for identifying critical errors and unsoundness problems in scientific papers. o3 exhibited the best problem identification performance among all models at a modest cost. This paper provides insights into document-based scientific understanding/reasoning and lays a foundation for future applications. Our dataset, code, and model outputs are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05200v1">In-Context Learning as an Effective Estimator of Functional Correctness of LLM-Generated Code</a></div>
    <div class="paper-meta">
      📅 2025-07-07
    </div>
    <details class="paper-abstract">
      When applying LLM-based code generation to software development projects that follow a feature-driven or rapid application development approach, it becomes necessary to estimate the functional correctness of the generated code in the absence of test cases. Just as a user selects a relevant document from a ranked list of retrieved ones, a software generation workflow requires a developer to choose (and potentially refine) a generated solution from a ranked list of alternative solutions, ordered by their posterior likelihoods. This implies that estimating the quality of a ranked list -- akin to estimating "relevance" for query performance prediction (QPP) in IR -- is also crucial for generative software development, where quality is defined in terms of "functional correctness". In this paper, we propose an in-context learning (ICL) based approach for code quality estimation. Our findings demonstrate that providing few-shot examples of functionally correct code from a training set enhances the performance of existing QPP approaches as well as a zero-shot-based approach for code quality estimation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05995v2">NativQA Framework: Enabling LLMs with Native, Local, and Everyday Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 LLMs, Native, Multilingual, Language Diversity, Contextual Understanding, Minority Languages, Culturally Informed, Foundation Models, Large Language Models
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has raised concerns about cultural bias, fairness, and their applicability in diverse linguistic and underrepresented regional contexts. To enhance and benchmark the capabilities of LLMs, there is a need to develop large-scale resources focused on multilingual, local, and cultural contexts. In this study, we propose the NativQA framework, which can seamlessly construct large-scale, culturally and regionally aligned QA datasets in native languages. The framework utilizes user-defined seed queries and leverages search engines to collect location-specific, everyday information. It has been evaluated across 39 locations in 24 countries and in 7 languages -- ranging from extremely low-resource to high-resource languages -- resulting in over 300K Question-Answer (QA) pairs. The developed resources can be used for LLM benchmarking and further fine-tuning. The framework has been made publicly available for the community (https://gitlab.com/nativqa/nativqa-framework).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05118v1">VerifyLLM: LLM-Based Pre-Execution Task Plan Verification for Robots</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 IROS 2025
    </div>
    <details class="paper-abstract">
      In the field of robotics, researchers face a critical challenge in ensuring reliable and efficient task planning. Verifying high-level task plans before execution significantly reduces errors and enhance the overall performance of these systems. In this paper, we propose an architecture for automatically verifying high-level task plans before their execution in simulator or real-world environments. Leveraging Large Language Models (LLMs), our approach consists of two key steps: first, the conversion of natural language instructions into Linear Temporal Logic (LTL), followed by a comprehensive analysis of action sequences. The module uses the reasoning capabilities of the LLM to evaluate logical coherence and identify potential gaps in the plan. Rigorous testing on datasets of varying complexity demonstrates the broad applicability of the module to household tasks. We contribute to improving the reliability and efficiency of task planning and addresses the critical need for robust pre-execution verification in autonomous systems. The code is available at https://verifyllm.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19794v2">Why Do Open-Source LLMs Struggle with Data Analysis? A Systematic Empirical Study</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) hold promise in automating data analysis tasks, yet open-source models face significant limitations in these kinds of reasoning-intensive scenarios. In this work, we investigate strategies to enhance the data analysis capabilities of open-source LLMs. By curating a seed dataset of diverse, realistic scenarios, we evaluate models across three dimensions: data understanding, code generation, and strategic planning. Our analysis reveals three key findings: (1) Strategic planning quality serves as the primary determinant of model performance; (2) Interaction design and task complexity significantly influence reasoning capabilities; (3) Data quality demonstrates a greater impact than diversity in achieving optimal performance. We leverage these insights to develop a data synthesis methodology, demonstrating significant improvements in open-source LLMs' analytical reasoning capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04976v1">Can Video LLMs Refuse to Answer? Alignment for Answerability in Video Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 ICLR 2025
    </div>
    <details class="paper-abstract">
      In the broader context of deep learning, Multimodal Large Language Models have achieved significant breakthroughs by leveraging powerful Large Language Models as a backbone to align different modalities into the language space. A prime exemplification is the development of Video Large Language Models (Video-LLMs). While numerous advancements have been proposed to enhance the video understanding capabilities of these models, they are predominantly trained on questions generated directly from video content. However, in real-world scenarios, users often pose questions that extend beyond the informational scope of the video, highlighting the need for Video-LLMs to assess the relevance of the question. We demonstrate that even the best-performing Video-LLMs fail to reject unfit questions-not necessarily due to a lack of video understanding, but because they have not been trained to identify and refuse such questions. To address this limitation, we propose alignment for answerability, a framework that equips Video-LLMs with the ability to evaluate the relevance of a question based on the input video and appropriately decline to answer when the question exceeds the scope of the video, as well as an evaluation framework with a comprehensive set of metrics designed to measure model behavior before and after alignment. Furthermore, we present a pipeline for creating a dataset specifically tailored for alignment for answerability, leveraging existing video-description paired datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04967v1">The Case for Instance-Optimized LLMs in OLAP Databases</a></div>
    <div class="paper-meta">
      📅 2025-07-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can enhance analytics systems with powerful data summarization, cleaning, and semantic transformation capabilities. However, deploying LLMs at scale -- processing millions to billions of rows -- remains prohibitively expensive in computation and memory. We present IOLM-DB, a novel system that makes LLM-enhanced database queries practical through query-specific model optimization. Instead of using general-purpose LLMs, IOLM-DB generates lightweight, specialized models tailored to each query's specific needs using representative data samples. IOLM-DB reduces model footprints by up to 76% and increases throughput by up to 3.31$\times$ while maintaining accuracy through aggressive compression techniques, including quantization, sparsification, and structural pruning. We further show how our approach enables higher parallelism on existing hardware and seamlessly supports caching and batching strategies to reduce overheads. Our prototype demonstrates that leveraging LLM queries inside analytics systems is feasible at scale, opening new possibilities for future OLAP applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04952v1">ArtifactsBench: Bridging the Visual-Interactive Gap in LLM Code Generation Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-07-07
    </div>
    <details class="paper-abstract">
      The generative capabilities of Large Language Models (LLMs) are rapidly expanding from static code to dynamic, interactive visual artifacts. This progress is bottlenecked by a critical evaluation gap: established benchmarks focus on algorithmic correctness and are blind to the visual fidelity and interactive integrity that define modern user experiences. To bridge this gap, we introduce ArtifactsBench, a new benchmark and paradigm for the automated, multimodal evaluation of visual code generation. Our framework programmatically renders each generated artifact and captures its dynamic behavior through temporal screenshots. This visual evidence, alongside the source code, is then assessed by a Multimodal LLM (MLLM)-as-Judge, which is rigorously guided by a fine-grained, per-task checklist to ensure holistic and reproducible scoring. We construct a new benchmark of 1,825 diverse tasks and evaluate over 30 leading LLMs. Our automated evaluation achieves a striking 94.4% ranking consistency with WebDev Arena, the gold-standard for human preference in web development, and over 90% pairwise agreement with human experts. This establishes ArtifactsBench as the first framework to reliably automate the assessment of human-perceived quality at scale. Our analysis provides a high-resolution map of the current SOTA, revealing that generalist models often outperform domain-specific ones. We open-source ArtifactsBench, including the benchmark, evaluation harness, and baseline results at https://artifactsbenchmark.github.io/, to provide the community with a scalable and accurate tool to accelerate the development of user-centric generative models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16174v2">Do LLMs Understand the Safety of Their Inputs? Training-Free Moderation via Latent Prototypes</a></div>
    <div class="paper-meta">
      📅 2025-07-07
    </div>
    <details class="paper-abstract">
      With the rise of LLMs, ensuring model safety and alignment has become a critical concern. While modern instruction-finetuned LLMs incorporate alignment during training, they still frequently require moderation tools to prevent unsafe behavior. The most common approach to moderation are guard models that flag unsafe inputs. However, guards require costly training and are typically limited to fixed-size, pre-trained options, making them difficult to adapt to evolving risks and resource constraints. We hypothesize that instruction-finetuned LLMs already encode safety-relevant information internally and explore training-free safety assessment methods that work with off-the-shelf models. We show that simple prompting allows models to recognize harmful inputs they would otherwise mishandle. We also demonstrate that safe and unsafe prompts are distinctly separable in the models' latent space. Building on this, we introduce the Latent Prototype Moderator (LPM), a training-free moderation method that uses Mahalanobis distance in latent space to assess input safety. LPM is a lightweight, customizable add-on that generalizes across model families and sizes. Our method matches or exceeds state-of-the-art guard models across multiple safety benchmarks, offering a practical and flexible solution for scalable LLM moderation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04893v1">MARBLE: A Multi-Agent Rule-Based LLM Reasoning Engine for Accident Severity Prediction</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 13 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Accident severity prediction plays a critical role in transportation safety systems but is a persistently difficult task due to incomplete data, strong feature dependencies, and severe class imbalance in which rare but high-severity cases are underrepresented and hard to detect. Existing methods often rely on monolithic models or black box prompting, which struggle to scale in noisy, real-world settings and offer limited interpretability. To address these challenges, we propose MARBLE a multiagent rule based LLM engine that decomposes the severity prediction task across a team of specialized reasoning agents, including an interchangeable ML-backed agent. Each agent focuses on a semantic subset of features (e.g., spatial, environmental, temporal), enabling scoped reasoning and modular prompting without the risk of prompt saturation. Predictions are coordinated through either rule-based or LLM-guided consensus mechanisms that account for class rarity and confidence dynamics. The system retains structured traces of agent-level reasoning and coordination outcomes, supporting in-depth interpretability and post-hoc performance diagnostics. Across both UK and US datasets, MARBLE consistently outperforms traditional machine learning classifiers and state-of-the-art (SOTA) prompt-based reasoning methods including Chain-of-Thought (CoT), Least-to-Most (L2M), and Tree-of-Thought (ToT) achieving nearly 90% accuracy where others plateau below 48%. This performance redefines the practical ceiling for accident severity classification under real world noise and extreme class imbalance. Our results position MARBLE as a generalizable and interpretable framework for reasoning under uncertainty in safety-critical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04877v1">DoPI: Doctor-like Proactive Interrogation LLM for Traditional Chinese Medicine</a></div>
    <div class="paper-meta">
      📅 2025-07-07
    </div>
    <details class="paper-abstract">
      Enhancing interrogation capabilities in Traditional Chinese Medicine (TCM) diagnosis through multi-turn dialogues and knowledge graphs presents a significant challenge for modern AI systems. Current large language models (LLMs), despite their advancements, exhibit notable limitations in medical applications, particularly in conducting effective multi-turn dialogues and proactive questioning. These shortcomings hinder their practical application and effectiveness in simulating real-world diagnostic scenarios. To address these limitations, we propose DoPI, a novel LLM system specifically designed for the TCM domain. The DoPI system introduces a collaborative architecture comprising a guidance model and an expert model. The guidance model conducts multi-turn dialogues with patients and dynamically generates questions based on a knowledge graph to efficiently extract critical symptom information. Simultaneously, the expert model leverages deep TCM expertise to provide final diagnoses and treatment plans. Furthermore, this study constructs a multi-turn doctor-patient dialogue dataset to simulate realistic consultation scenarios and proposes a novel evaluation methodology that does not rely on manually collected real-world consultation data. Experimental results show that the DoPI system achieves an accuracy rate of 84.68 percent in interrogation outcomes, significantly enhancing the model's communication ability during diagnosis while maintaining professional expertise.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07120v1">Helix Parallelism: Rethinking Sharding Strategies for Interactive Multi-Million-Token LLM Decoding</a></div>
    <div class="paper-meta">
      📅 2025-07-07
    </div>
    <details class="paper-abstract">
      As LLMs scale to multi-million-token KV histories, real-time autoregressive decoding under tight Token-to-Token Latency (TTL) constraints faces growing pressure. Two core bottlenecks dominate: accessing Feed-Forward Network (FFN) weights and reading long KV caches. While Tensor Parallelism (TP) helps mitigate the cost of FFN weight reads, it does not scale well for attention. When TP width exceeds the number of KV heads, it leads to inefficient KV duplication, limits parallelism, and constrains batch size. Simultaneously, DRAM reads for long KV histories scale linearly with batch size, further capping efficiency. We introduce Helix Parallelism, a hybrid execution strategy that applies KV parallelism during attention to shard KV caches across GPUs, then reuses the same GPUs for TP in dense LLMs or TPxExpert Parallel (EP) in MoEs during FFN computation. To preserve exact attention behavior, Helix includes a lightweight communication step. To minimize the exposed communication cost, we introduce Helix HOP-B. Helix HOP-B effectively minimizes communication overhead through batchwise overlap, preserving low TTL while improving GPU efficiency. Compared to conventional parallelism approaches, Helix reduces TTL by up to 1.5x at fixed batch sizes and supports up to 32x larger batches under the same latency budget for DeepSeek-R1, pushing forward the throughput-latency Pareto on Blackwell and making real-time inference with ultra-long-sequence practical.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04575v1">Lilith: Developmental Modular LLMs with Chemical Signaling</a></div>
    <div class="paper-meta">
      📅 2025-07-06
      | 💬 4 pages, 0 figures, position paper
    </div>
    <details class="paper-abstract">
      Current paradigms in Artificial Intelligence rely on layers of feedforward networks which model brain activity at the neuronal level. We conjecture that expanding to the level of multiple brain regions with chemical signaling may be a productive step toward understanding the emergence of consciousness. We propose LILITH, a novel architecture that combines developmental training of modular language models with brain-inspired token-based communication protocols, mirroring chemical signaling in the brain. Our approach models distinct brain regions as specialized LLM modules including thinking, memory, sensory, and regulatory components that communicate through emergent token-based signaling protocols analogous to neurotransmitter networks. Unlike traditional pre-trained systems, LILITH would employ developmental training where untrained LLM architectures learn through simulated life experiences, developing communication pathways and cognitive abilities through environmental interaction and evolutionary optimization. This framework would enable direct empirical investigation of consciousness emergence using Integrated Information Theory metrics while providing unprecedented insight into inter-module signaling patterns during development. By optimizing for consciousness emergence rather than task performance, LILITH could provide insight into different emergent phenomena at multiple levels of neural correlates, contrasting neuronal-level processing with multi-region coordination dynamics. The goal of this paper is to put the idea forward while recognizing the substantial challenges in implementing such a system.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04562v1">Evaluating LLMs on Real-World Forecasting Against Human Superforecasters</a></div>
    <div class="paper-meta">
      📅 2025-07-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable capabilities across diverse tasks, but their ability to forecast future events remains understudied. A year ago, large language models struggle to come close to the accuracy of a human crowd. I evaluate state-of-the-art LLMs on 464 forecasting questions from Metaculus, comparing their performance against human superforecasters. Frontier models achieve Brier scores that ostensibly surpass the human crowd but still significantly underperform a group of superforecasters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17833v2">The Role of Open-Source LLMs in Shaping the Future of GeoAI</a></div>
    <div class="paper-meta">
      📅 2025-07-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are transforming geospatial artificial intelligence (GeoAI), offering new capabilities in data processing, spatial analysis, and decision support. This paper examines the open-source paradigm's critical role in this transformation. While proprietary LLMs offer accessibility, they often limit the customization, interoperability, and transparency vital for specialized geospatial tasks. Conversely, open-source alternatives significantly advance Geographic Information Science (GIScience) by fostering greater adaptability, reproducibility, and community-driven innovation. Open frameworks empower researchers to tailor solutions, integrate cutting-edge methodologies (e.g., reinforcement learning, advanced spatial indexing), and align with FAIR (Findable, Accessible, Interoperable, and Reusable) principles. However, the growing reliance on any LLM necessitates careful consideration of security vulnerabilities, ethical risks, and robust governance for AI-generated geospatial outputs. This paper argues that GIScience advances best not through a single model type, but by cultivating a diverse, interoperable ecosystem combining open-source foundations for innovation, custom geospatial models, and interdisciplinary collaboration. By critically evaluating the opportunities and challenges of open-source LLMs within the broader GeoAI landscape, this work contributes to a thorough discourse on leveraging LLMs to effectively advance spatial research, policy, and decision-making in an equitable, sustainable, and scientifically rigorous manner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04517v1">DOTResize: Reducing LLM Width via Discrete Optimal Transport-based Neuron Merging</a></div>
    <div class="paper-meta">
      📅 2025-07-06
    </div>
    <details class="paper-abstract">
      Model compression offers a promising path to reducing the cost and inaccessibility of large pre-trained models, without significantly compromising their impressive performance. Large Transformer models, including large language models (LLMs), often contain computational redundancy, which can serve as a target for new model compression methods. In this work, we specifically target neuron-level redundancies in model layers by combining groups of similar neurons into fewer neurons. We frame this width reduction as a Discrete Optimal Transport problem, and propose DOTResize, a novel Transformer compression method that uses optimal transport theory to transform and compress model weights. To ensure applicability within the Transformer architecture, we motivate and incorporate entropic regularization and matrix factorization into the transportation maps produced by our method. Unlike pruning-based approaches which discard neurons based on importance measures, DOTResize re-projects the entire neuron width, allowing the retention and redistribution of useful signal across the reduced layer. Empirical results show that compared to simple or state-of-the-art neuron width-pruning techniques, DOTResize can outperform these methods across multiple LLM families and sizes, while achieving measurable reductions in real-world computational cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01436v2">Challenges & Opportunities with LLM-Assisted Visualization Retargeting</a></div>
    <div class="paper-meta">
      📅 2025-07-06
      | 💬 5 pages, 3 figures, 1 table
    </div>
    <details class="paper-abstract">
      Despite the ubiquity of visualization examples published on the web, retargeting existing custom chart implementations to new datasets remains difficult, time-intensive, and tedious. The adaptation process assumes author familiarity with both the implementation of the example as well as how the new dataset might need to be transformed to fit into the example code. With recent advances in Large Language Models (LLMs), automatic adaptation of code can be achieved from high-level user prompts, reducing the barrier for visualization retargeting. To better understand how LLMs can assist retargeting and its potential limitations, we characterize and evaluate the performance of LLM assistance across multiple datasets and charts of varying complexity, categorizing failures according to type and severity. In our evaluation, we compare two approaches: (1) directly instructing the LLM model to fully generate and adapt code by treating code as text inputs and (2) a more constrained program synthesis pipeline where the LLM guides the code construction process by providing structural information (e.g., visual encodings) based on properties of the example code and data. We find that both approaches struggle when new data has not been appropriately transformed, and discuss important design recommendations for future retargeting systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04446v1">Tail-aware Adversarial Attacks: A Distributional Approach to Efficient LLM Jailbreaking</a></div>
    <div class="paper-meta">
      📅 2025-07-06
    </div>
    <details class="paper-abstract">
      To guarantee safe and robust deployment of large language models (LLMs) at scale, it is critical to accurately assess their adversarial robustness. Existing adversarial attacks typically target harmful responses in single-point, greedy generations, overlooking the inherently stochastic nature of LLMs. In this paper, we propose a novel framework for adversarial robustness evaluation that explicitly models the entire output distribution, including tail-risks, providing better estimates for model robustness at scale. By casting the attack process as a resource allocation problem between optimization and sampling, we determine compute-optimal tradeoffs and show that integrating sampling into existing attacks boosts ASR by up to 48% and improves efficiency by up to two orders of magnitude. Our framework also enables us to analyze how different attack algorithms affect output harm distributions. Surprisingly, we find that most optimization strategies have little effect on output harmfulness. Finally, we introduce a data-free proof-of-concept objective based on entropy-maximization to demonstrate how our tail-aware perspective enables new optimization targets. Overall, our findings highlight the importance of tail-aware attacks and evaluation protocols to accurately assess and strengthen LLM safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04444v1">Data Discovery using LLMs -- A Study of Data User Behaviour</a></div>
    <div class="paper-meta">
      📅 2025-07-06
      | 💬 Accepted as full paper at TPDL'25
    </div>
    <details class="paper-abstract">
      Data search for scientific research is more complex than a simple web search. The emergence of large language models (LLMs) and their applicability for scientific tasks offers new opportunities for researchers who are looking for data, e.g., to freely express their data needs instead of fitting them into restrictions of data catalogues and portals. However, this also creates uncertainty about whether LLMs are suitable for this task. To answer this question, we conducted a user study with 32 researchers. We qualitatively and quantitively analysed participants' information interaction behaviour while searching for data using LLMs in two data search tasks, one in which we prompted the LLM to behave as a persona. We found that participants interact with LLMs in natural language, but LLMs remain a tool for them rather than an equal conversational partner. This changes slightly when the LLM is prompted to behave as a persona, but the prompting only affects participants' user experience when they are already experienced in LLM use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04431v1">MedGellan: LLM-Generated Medical Guidance to Support Physicians</a></div>
    <div class="paper-meta">
      📅 2025-07-06
    </div>
    <details class="paper-abstract">
      Medical decision-making is a critical task, where errors can result in serious, potentially life-threatening consequences. While full automation remains challenging, hybrid frameworks that combine machine intelligence with human oversight offer a practical alternative. In this paper, we present MedGellan, a lightweight, annotation-free framework that uses a Large Language Model (LLM) to generate clinical guidance from raw medical records, which is then used by a physician to predict diagnoses. MedGellan uses a Bayesian-inspired prompting strategy that respects the temporal order of clinical data. Preliminary experiments show that the guidance generated by the LLM with MedGellan improves diagnostic performance, particularly in recall and $F_1$ score.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02773v2">KERAP: A Knowledge-Enhanced Reasoning Approach for Accurate Zero-shot Diagnosis Prediction Using Multi-agent LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-06
    </div>
    <details class="paper-abstract">
      Medical diagnosis prediction plays a critical role in disease detection and personalized healthcare. While machine learning (ML) models have been widely adopted for this task, their reliance on supervised training limits their ability to generalize to unseen cases, particularly given the high cost of acquiring large, labeled datasets. Large language models (LLMs) have shown promise in leveraging language abilities and biomedical knowledge for diagnosis prediction. However, they often suffer from hallucinations, lack structured medical reasoning, and produce useless outputs. To address these challenges, we propose KERAP, a knowledge graph (KG)-enhanced reasoning approach that improves LLM-based diagnosis prediction through a multi-agent architecture. Our framework consists of a linkage agent for attribute mapping, a retrieval agent for structured knowledge extraction, and a prediction agent that iteratively refines diagnosis predictions. Experimental results demonstrate that KERAP enhances diagnostic reliability efficiently, offering a scalable and interpretable solution for zero-shot medical diagnosis prediction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09701v2">Have LLMs Made Active Learning Obsolete? Surveying the NLP Community</a></div>
    <div class="paper-meta">
      📅 2025-07-06
    </div>
    <details class="paper-abstract">
      Supervised learning relies on annotated data, which is expensive to obtain. A longstanding strategy to reduce annotation costs is active learning, an iterative process, in which a human annotates only data instances deemed informative by a model. Large language models (LLMs) have pushed the effectiveness of active learning, while also advancing methods such as few- or zero-shot learning, and text synthesis -- all of which can reduce the need for active learning. This naturally raises the question: has active learning become obsolete? To answer this fully, we must look beyond literature to practical experiences. We conduct an online survey in the NLP community to collect previously intangible insights on the perceived relevance of data annotation, particularly focusing on active learning, including best practices, obstacles, and future prospects. Our findings show that annotated data is expected to remain a key factor and active learning to stay highly relevant while benefiting from LLMs. Consistent with a community survey from over a decade ago, however, we find that three key challenges persist -- setup complexity, risks in the cost reduction, and tooling -- for which we propose alleviation strategies. We publish an anonymized version of the collected dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04365v1">Attention Slipping: A Mechanistic Understanding of Jailbreak Attacks and Defenses in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-06
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become more integral to society and technology, ensuring their safety becomes essential. Jailbreak attacks exploit vulnerabilities to bypass safety guardrails, posing a significant threat. However, the mechanisms enabling these attacks are not well understood. In this paper, we reveal a universal phenomenon that occurs during jailbreak attacks: Attention Slipping. During this phenomenon, the model gradually reduces the attention it allocates to unsafe requests in a user query during the attack process, ultimately causing a jailbreak. We show Attention Slipping is consistent across various jailbreak methods, including gradient-based token replacement, prompt-level template refinement, and in-context learning. Additionally, we evaluate two defenses based on query perturbation, Token Highlighter and SmoothLLM, and find they indirectly mitigate Attention Slipping, with their effectiveness positively correlated with the degree of mitigation achieved. Inspired by this finding, we propose Attention Sharpening, a new defense that directly counters Attention Slipping by sharpening the attention score distribution using temperature scaling. Experiments on four leading LLMs (Gemma2-9B-It, Llama3.1-8B-It, Qwen2.5-7B-It, Mistral-7B-It v0.2) show that our method effectively resists various jailbreak attacks while maintaining performance on benign tasks on AlpacaEval. Importantly, Attention Sharpening introduces no additional computational or memory overhead, making it an efficient and practical solution for real-world deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12440v2">HKCanto-Eval: A Benchmark for Evaluating Cantonese Language Understanding and Cultural Comprehension in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-06
    </div>
    <details class="paper-abstract">
      The ability of language models to comprehend and interact in diverse linguistic and cultural landscapes is crucial. The Cantonese language used in Hong Kong presents unique challenges for natural language processing due to its rich cultural nuances and lack of dedicated evaluation datasets. The HKCanto-Eval benchmark addresses this gap by evaluating the performance of large language models (LLMs) on Cantonese language understanding tasks, extending to English and Written Chinese for cross-lingual evaluation. HKCanto-Eval integrates cultural and linguistic nuances intrinsic to Hong Kong, providing a robust framework for assessing language models in realistic scenarios. Additionally, the benchmark includes questions designed to tap into the underlying linguistic metaknowledge of the models. Our findings indicate that while proprietary models generally outperform open-weight models, significant limitations remain in handling Cantonese-specific linguistic and cultural knowledge, highlighting the need for more targeted training data and evaluation methods. The code can be accessed at https://github.com/hon9kon9ize/hkeval2025
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04315v1">HLStrans: Dataset for LLM-Driven C-to-HLS Hardware Code Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-07-06
    </div>
    <details class="paper-abstract">
      High-level synthesis (HLS) enables software developers to describe and implement hardware at a higher level of abstraction by using C/C++ instead of traditional hardware description languages to automatically generate FPGA-ready designs. However, generating HLS code significantly differs from standard C/C++: it disallows certain coding idioms, relies on specialized libraries, and critically requires fine-grained transformations and the insertion of optimization directives (pragmas) to achieve high performance. Large language models (LLMs) have shown promise in automating such transformations, yet existing open-source datasets lack sufficient complexity and optimization diversity. To address this gap, we introduce the HLStrans dataset, a comprehensive collection of 137 distinct real word programs, each annotated with a variety of C-to-HLS transformations that yield over 23K labeled design variants. These include a broad spectrum of pragmas and code-level optimizations. We benchmark state-of-the-art LLMs on this dataset to evaluate their ability to generate synthesizable, high-performance HLS code. As part of an ongoing effort, we plan to expand the HLStrans dataset in both scale and program variety, further empowering research at the intersection of AI and hardware synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15567v2">Intelligent Assistants for the Semiconductor Failure Analysis with LLM-Based Planning Agents</a></div>
    <div class="paper-meta">
      📅 2025-07-06
      | 💬 This technical report provides evaluation details of the epxeriments presented in the paper accepted to ISTFA 2025
    </div>
    <details class="paper-abstract">
      Failure Analysis (FA) is a highly intricate and knowledge-intensive process. The integration of AI components within the computational infrastructure of FA labs has the potential to automate a variety of tasks, including the detection of non-conformities in images, the retrieval of analogous cases from diverse data sources, and the generation of reports from annotated images. However, as the number of deployed AI models increases, the challenge lies in orchestrating these components into cohesive and efficient workflows that seamlessly integrate with the FA process. This paper investigates the design and implementation of a Large Language Model (LLM)-based Planning Agent (LPA) to assist FA engineers in solving their analysis cases. The LPA integrates LLMs with advanced planning capabilities and external tool utilization, enabling autonomous processing of complex queries, retrieval of relevant data from external systems, and generation of human-readable responses. Evaluation results demonstrate the agent's operational effectiveness and reliability in supporting FA tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04294v1">BiFair: A Fairness-aware Training Framework for LLM-enhanced Recommender Systems via Bi-level Optimization</a></div>
    <div class="paper-meta">
      📅 2025-07-06
    </div>
    <details class="paper-abstract">
      Large Language Model-enhanced Recommender Systems (LLM-enhanced RSs) have emerged as a powerful approach to improving recommendation quality by leveraging LLMs to generate item representations. Despite these advancements, the integration of LLMs raises severe fairness concerns. Existing studies reveal that LLM-based RSs exhibit greater unfairness than traditional RSs, yet fairness issues in LLM-enhanced RSs remain largely unexplored. In this paper, our empirical study reveals that while LLM-enhanced RSs improve fairness across item groups, a significant fairness gap persists. Further enhancement remains challenging due to the architectural differences and varying sources of unfairness inherent in LLM-enhanced RSs. To bridge this gap, we first decompose unfairness into i) \textit{prior unfairness} in LLM-generated representations and ii) \textit{training unfairness} in recommendation models. Then, we propose BiFair, a bi-level optimization-based fairness-aware training framework designed to mitigate both prior and training unfairness simultaneously. BiFair optimizes two sets of learnable parameters: LLM-generated representations and a trainable projector in the recommendation model, using a two-level nested optimization process. Additionally, we introduce an adaptive inter-group balancing mechanism, leveraging multi-objective optimization principles to dynamically balance fairness across item groups. Extensive experiments on three real-world datasets demonstrate that BiFair significantly mitigates unfairness and outperforms previous state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04295v1">LearnLens: LLM-Enabled Personalised, Curriculum-Grounded Feedback with Educators in the Loop</a></div>
    <div class="paper-meta">
      📅 2025-07-06
    </div>
    <details class="paper-abstract">
      Effective feedback is essential for student learning but is time-intensive for teachers. We present LearnLens, a modular, LLM-based system that generates personalised, curriculum-aligned feedback in science education. LearnLens comprises three components: (1) an error-aware assessment module that captures nuanced reasoning errors; (2) a curriculum-grounded generation module that uses a structured, topic-linked memory chain rather than traditional similarity-based retrieval, improving relevance and reducing noise; and (3) an educator-in-the-loop interface for customisation and oversight. LearnLens addresses key challenges in existing systems, offering scalable, high-quality feedback that empowers both teachers and students.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04276v1">FIXME: Towards End-to-End Benchmarking of LLM-Aided Design Verification</a></div>
    <div class="paper-meta">
      📅 2025-07-06
    </div>
    <details class="paper-abstract">
      Despite the transformative potential of Large Language Models (LLMs) in hardware design, a comprehensive evaluation of their capabilities in design verification remains underexplored. Current efforts predominantly focus on RTL generation and basic debugging, overlooking the critical domain of functional verification, which is the primary bottleneck in modern design methodologies due to the rapid escalation of hardware complexity. We present FIXME, the first end-to-end, multi-model, and open-source evaluation framework for assessing LLM performance in hardware functional verification (FV) to address this crucial gap. FIXME introduces a structured three-level difficulty hierarchy spanning six verification sub-domains and 180 diverse tasks, enabling in-depth analysis across the design lifecycle. Leveraging a collaborative AI-human approach, we construct a high-quality dataset using 100% silicon-proven designs, ensuring comprehensive coverage of real-world challenges. Furthermore, we enhance the functional coverage by 45.57% through expert-guided optimization. By rigorously evaluating state-of-the-art LLMs such as GPT-4, Claude3, and LlaMA3, we identify key areas for improvement and outline promising research directions to unlock the full potential of LLM-driven automation in hardware design verification. The benchmark is available at https://github.com/ChatDesignVerification/FIXME.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.06283v2">LLMs' Reading Comprehension Is Affected by Parametric Knowledge and Struggles with Hypothetical Statements</a></div>
    <div class="paper-meta">
      📅 2025-07-06
    </div>
    <details class="paper-abstract">
      The task of reading comprehension (RC), often implemented as context-based question answering (QA), provides a primary means to assess language models' natural language understanding (NLU) capabilities. Yet, when applied to large language models (LLMs) with extensive built-in world knowledge, this method can be deceptive. If the context aligns with the LLMs' internal knowledge, it is hard to discern whether the models' answers stem from context comprehension or from LLMs' internal information. Conversely, using data that conflicts with the models' knowledge creates erroneous trends which distort the results. To address this issue, we suggest to use RC on imaginary data, based on fictitious facts and entities. This task is entirely independent of the models' world knowledge, enabling us to evaluate LLMs' linguistic abilities without the interference of parametric knowledge. Testing ChatGPT, GPT-4, LLaMA 2 and Mixtral on such imaginary data, we uncover a class of linguistic phenomena posing a challenge to current LLMs, involving thinking in terms of alternative, hypothetical scenarios. While all the models handle simple affirmative and negative contexts with high accuracy, they are much more prone to error when dealing with modal and conditional contexts. Crucially, these phenomena also trigger the LLMs' vulnerability to knowledge-conflicts again. In particular, while some models prove virtually unaffected by knowledge conflicts in affirmative and negative contexts, when faced with more semantically involved modal and conditional environments, they often fail to separate the text from their internal knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00025v4">Explainable AI for Mental Health Emergency Returns: Integrating LLMs with Predictive Modeling</a></div>
    <div class="paper-meta">
      📅 2025-07-06
    </div>
    <details class="paper-abstract">
      Importance: Emergency department (ED) returns for mental health conditions pose a major healthcare burden, with 24-27% of patients returning within 30 days. Traditional machine learning models for predicting these returns often lack interpretability for clinical use. Objective: To assess whether integrating large language models (LLMs) with machine learning improves predictive accuracy and clinical interpretability of ED mental health return risk models. Methods: This retrospective cohort study analyzed 42,464 ED visits for 27,904 unique mental health patients at an academic medical center in the Deep South from January 2018 to December 2022. Main Outcomes and Measures: Two primary outcomes were evaluated: (1) 30-day ED return prediction accuracy and (2) model interpretability using a novel LLM-enhanced framework integrating SHAP (SHapley Additive exPlanations) values with clinical knowledge. Results: For chief complaint classification, LLaMA 3 (8B) with 10-shot learning outperformed traditional models (accuracy: 0.882, F1-score: 0.86). In SDoH classification, LLM-based models achieved 0.95 accuracy and 0.96 F1-score, with Alcohol, Tobacco, and Substance Abuse performing best (F1: 0.96-0.89), while Exercise and Home Environment showed lower performance (F1: 0.70-0.67). The LLM-based interpretability framework achieved 99% accuracy in translating model predictions into clinically relevant explanations. LLM-extracted features improved XGBoost AUC from 0.74 to 0.76 and AUC-PR from 0.58 to 0.61. Conclusions and Relevance: Integrating LLMs with machine learning models yielded modest but consistent accuracy gains while significantly enhancing interpretability through automated, clinically relevant explanations. This approach provides a framework for translating predictive analytics into actionable clinical insights.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04219v1">Model Collapse Is Not a Bug but a Feature in Machine Unlearning for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-06
    </div>
    <details class="paper-abstract">
      Current unlearning methods for LLMs optimize on the private information they seek to remove by incorporating it into their training objectives. We argue this not only risks reinforcing exposure to sensitive data, it also fundamentally contradicts the principle of minimizing its use. As a remedy, we propose a novel unlearning method - Partial Model Collapse (PMC), which does not require unlearning targets in the unlearning objective. Our approach is inspired by recent observations that training generative models on their own generations leads to distribution collapse, effectively removing information from the model. Our core idea is to leverage this collapse for unlearning by triggering collapse partially on the sensitive data. We theoretically analyze that our approach converges to the desired outcome, i.e. the LLM unlearns the information in the forget set. We empirically demonstrate that PMC overcomes two key limitations of existing unlearning approaches that explicitly optimize on unlearning targets, and more effectively removes private information from model outputs. Overall, our contributions represent an important step toward more comprehensive unlearning that aligns with real-world privacy constraints. Code available at https://www.cs.cit.tum.de/daml/partial-model-collapse/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.00225v3">Large-scale, Independent and Comprehensive study of the power of LLMs for test case generation</a></div>
    <div class="paper-meta">
      📅 2025-07-05
    </div>
    <details class="paper-abstract">
      Unit testing is essential for software reliability, yet manual test creation is time-consuming and often neglected. Although search-based software testing improves efficiency, it produces tests with poor readability and maintainability. Although LLMs show promise for test generation, existing research lacks comprehensive evaluation across execution-driven assessment, reasoning-based prompting, and real-world testing scenarios. This study presents the first large-scale empirical evaluation of LLM-generated unit tests at the class level, systematically analyzing four state-of-the-art models - GPT-3.5, GPT-4, Mistral 7B, and Mixtral 8x7B - against EvoSuite across 216,300 test cases from Defects4J, SF110, and CMD (a dataset mitigating LLM training data leakage). We evaluate five prompting techniques - Zero-Shot Learning (ZSL), Few-Shot Learning (FSL), Chain-of-Thought (CoT), Tree-of-Thought (ToT), and Guided Tree-of-Thought (GToT) - assessing syntactic correctness, compilability, hallucination-driven failures, readability, code coverage metrics, and fault detection capabilities. Our findings challenge prior claims that in-context learning is ineffective for test generation in code-specialized LLMs. Reasoning-based prompting - particularly GToT - significantly enhances test reliability, compilability, and structural adherence in general-purpose LLMs. However, hallucination-driven failures remain a persistent challenge, manifesting as non-existent symbol references, incorrect API calls, and fabricated dependencies, resulting in high compilation failure rates (up to 86%). Execution-based classification and mutation testing reveal that many failing tests stem from hallucinated dependencies, limiting effective fault detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04034v1">Lyria: A General LLM-Driven Genetic Algorithm Framework for Problem Solving</a></div>
    <div class="paper-meta">
      📅 2025-07-05
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have demonstrated impressive abilities across various domains, they still struggle with complex problems characterized by multi-objective optimization, precise constraint satisfaction, immense solution spaces, etc. To address the limitation, drawing on the superior semantic understanding ability of LLMs and also the outstanding global search and optimization capability of genetic algorithms, we propose to capitalize on their respective strengths and introduce Lyria, a general LLM-driven genetic algorithm framework, comprising 7 essential components. Through conducting extensive experiments with 4 LLMs across 3 types of problems, we demonstrated the efficacy of Lyria. Additionally, with 7 additional ablation experiments, we further systematically analyzed and elucidated the factors that affect its performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04009v1">Easy Dataset: A Unified and Extensible Framework for Synthesizing LLM Fine-Tuning Data from Unstructured Documents</a></div>
    <div class="paper-meta">
      📅 2025-07-05
      | 💬 preprint
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown impressive performance on general-purpose tasks, yet adapting them to specific domains remains challenging due to the scarcity of high-quality domain data. Existing data synthesis tools often struggle to extract reliable fine-tuning data from heterogeneous documents effectively. To address this limitation, we propose Easy Dataset, a unified framework for synthesizing fine-tuning data from unstructured documents via an intuitive graphical user interface (GUI). Specifically, Easy Dataset allows users to easily configure text extraction models and chunking strategies to transform raw documents into coherent text chunks. It then leverages a persona-driven prompting approach to generate diverse question-answer pairs using public-available LLMs. Throughout the pipeline, a human-in-the-loop visual interface facilitates the review and refinement of intermediate outputs to ensure data quality. Experiments on a financial question-answering task show that fine-tuning LLMs on the synthesized dataset significantly improves domain-specific performance while preserving general knowledge. The source code and installable package are available at https://github.com/ConardLi/easy-dataset and have garnered over 9,000 GitHub stars.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14363v2">Improving RL Exploration for LLM Reasoning through Retrospective Replay</a></div>
    <div class="paper-meta">
      📅 2025-07-05
      | 💬 13 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has increasingly become a pivotal technique in the post-training of large language models (LLMs). The effective exploration of the output space is essential for the success of RL. We observe that for complex problems, during the early stages of training, the model exhibits strong exploratory capabilities and can identify promising solution ideas. However, its limited capability at this stage prevents it from successfully solving these problems. The early suppression of these potentially valuable solution ideas by the policy gradient hinders the model's ability to revisit and re-explore these ideas later. Consequently, although the LLM's capabilities improve in the later stages of training, it still struggles to effectively address these complex problems. To address this exploration issue, we propose a novel algorithm named Retrospective Replay-based Reinforcement Learning (RRL), which introduces a dynamic replay mechanism throughout the training process. RRL enables the model to revisit promising states identified in the early stages, thereby improving its efficiency and effectiveness in exploration. To evaluate the effectiveness of RRL, we conduct extensive experiments on complex reasoning tasks, including mathematical reasoning and code generation, and general dialogue tasks. The results indicate that RRL maintains high exploration efficiency throughout the training period, significantly enhancing the effectiveness of RL in optimizing LLMs for complicated reasoning tasks. Moreover, it also improves the performance of RLHF, making the model both safer and more helpful.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03958v1">A Comparative Study of Specialized LLMs as Dense Retrievers</a></div>
    <div class="paper-meta">
      📅 2025-07-05
      | 💬 Accepted by CCIR25 and published by Springer LNCS or LNAI
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) are increasingly deployed as dense retrievers, the impact of their domain-specific specialization on retrieval effectiveness remains underexplored. This investigation systematically examines how task-specific adaptations in LLMs influence their retrieval capabilities, an essential step toward developing unified retrievers capable of handling text, code, images, and multimodal content. We conduct extensive experiments with eight Qwen2.5 7B LLMs, including base, instruction-tuned, code/math-specialized, long reasoning, and vision-language models across zero-shot retrieval settings and the supervised setting. For the zero-shot retrieval settings, we consider text retrieval from the BEIR benchmark and code retrieval from the CoIR benchmark. Further, to evaluate supervised performance, all LLMs are fine-tuned on the MS MARCO dataset. We find that mathematical specialization and the long reasoning capability cause consistent degradation in three settings, indicating conflicts between mathematical reasoning and semantic matching. The vision-language model and code-specialized LLMs demonstrate superior zero-shot performance compared to other LLMs, even surpassing BM25 on the code retrieval task, and maintain comparable performance to base LLMs in supervised settings. These findings suggest promising directions for the unified retrieval task leveraging cross-domain and cross-modal fusion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03945v1">Function-based Labels for Complementary Recommendation: Definition, Annotation, and LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-07-05
    </div>
    <details class="paper-abstract">
      Complementary recommendations enhance the user experience by suggesting items that are frequently purchased together while serving different functions from the query item. Inferring or evaluating whether two items have a complementary relationship requires complementary relationship labels; however, defining these labels is challenging because of the inherent ambiguity of such relationships. Complementary labels based on user historical behavior logs attempt to capture these relationships, but often produce inconsistent and unreliable results. Recent efforts have introduced large language models (LLMs) to infer these relationships. However, these approaches provide a binary classification without a nuanced understanding of complementary relationships. In this study, we address these challenges by introducing Function-Based Labels (FBLs), a novel definition of complementary relationships independent of user purchase logs and the opaque decision processes of LLMs. We constructed a human-annotated FBLs dataset comprising 2,759 item pairs and demonstrated that it covered possible item relationships and minimized ambiguity. We then evaluated whether some machine learning (ML) methods using annotated FBLs could accurately infer labels for unseen item pairs, and whether LLM-generated complementary labels align with human perception. Our results demonstrate that even with limited data, ML models, such as logistic regression and SVM achieve high macro-F1 scores (approximately 0.82). Furthermore, LLMs, such as gpt-4o-mini, demonstrated high consistency (0.989) and classification accuracy (0.849) under the detailed definition of FBLs, indicating their potential as effective annotators that mimic human judgment. Overall, our study presents FBLs as a clear definition of complementary relationships, enabling more accurate inferences and automated labeling of complementary recommendations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03908v1">Bridging Vision and Language: Optimal Transport-Driven Radiology Report Generation via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-05
    </div>
    <details class="paper-abstract">
      Radiology report generation represents a significant application within medical AI, and has achieved impressive results. Concurrently, large language models (LLMs) have demonstrated remarkable performance across various domains. However, empirical validation indicates that general LLMs tend to focus more on linguistic fluency rather than clinical effectiveness, and lack the ability to effectively capture the relationship between X-ray images and their corresponding texts, thus resulting in poor clinical practicability. To address these challenges, we propose Optimal Transport-Driven Radiology Report Generation (OTDRG), a novel framework that leverages Optimal Transport (OT) to align image features with disease labels extracted from reports, effectively bridging the cross-modal gap. The core component of OTDRG is Alignment \& Fine-Tuning, where OT utilizes results from the encoding of label features and image visual features to minimize cross-modal distances, then integrating image and text features for LLMs fine-tuning. Additionally, we design a novel disease prediction module to predict disease labels contained in X-ray images during validation and testing. Evaluated on the MIMIC-CXR and IU X-Ray datasets, OTDRG achieves state-of-the-art performance in both natural language generation (NLG) and clinical efficacy (CE) metrics, delivering reports that are not only linguistically coherent but also clinically accurate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03876v1">LLMs model how humans induce logically structured rules</a></div>
    <div class="paper-meta">
      📅 2025-07-05
    </div>
    <details class="paper-abstract">
      A central goal of cognitive science is to provide a computationally explicit account of both the structure of the mind and its development: what are the primitive representational building blocks of cognition, what are the rules via which those primitives combine, and where do these primitives and rules come from in the first place? A long-standing debate concerns the adequacy of artificial neural networks as computational models that can answer these questions, in particular in domains related to abstract cognitive function, such as language and logic. This paper argues that recent advances in neural networks -- specifically, the advent of large language models (LLMs) -- represent an important shift in this debate. We test a variety of LLMs on an existing experimental paradigm used for studying the induction of rules formulated over logical concepts. Across four experiments, we find converging empirical evidence that LLMs provide at least as good a fit to human behavior as models that implement a Bayesian probablistic language of thought (pLoT), which have been the best computational models of human behavior on the same task. Moreover, we show that the LLMs make qualitatively different predictions about the nature of the rules that are inferred and deployed in order to complete the task, indicating that the LLM is unlikely to be a mere implementation of the pLoT solution. Based on these results, we argue that LLMs may instantiate a novel theoretical account of the primitive representations and computations necessary to explain human logical concepts, with which future work in cognitive science should engage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03871v1">Enhancing Adaptive Behavioral Interventions with LLM Inference from Participant-Described States</a></div>
    <div class="paper-meta">
      📅 2025-07-05
      | 💬 Accepted at Machine Learning for Healthcare (MLHC) 2025
    </div>
    <details class="paper-abstract">
      The use of reinforcement learning (RL) methods to support health behavior change via personalized and just-in-time adaptive interventions is of significant interest to health and behavioral science researchers focused on problems such as smoking cessation support and physical activity promotion. However, RL methods are often applied to these domains using a small collection of context variables to mitigate the significant data scarcity issues that arise from practical limitations on the design of adaptive intervention trials. In this paper, we explore an approach to significantly expanding the state space of an adaptive intervention without impacting data efficiency. The proposed approach enables intervention participants to provide natural language descriptions of aspects of their current state. It then leverages inference with pre-trained large language models (LLMs) to better align the policy of a base RL method with these state descriptions. To evaluate our method, we develop a novel physical activity intervention simulation environment that generates text-based state descriptions conditioned on latent state variables using an auxiliary LLM. We show that this approach has the potential to significantly improve the performance of online policy learning methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03865v1">OrthoRank: Token Selection via Sink Token Orthogonality for Efficient LLM inference</a></div>
    <div class="paper-meta">
      📅 2025-07-05
      | 💬 ICML 2025
    </div>
    <details class="paper-abstract">
      Attention mechanisms are central to the success of large language models (LLMs), enabling them to capture intricate token dependencies and implicitly assign importance to each token. Recent studies have revealed the sink token, which receives disproportionately high attention despite their limited semantic role. In this paper, we first expand the relationship between the sink token and other tokens, moving beyond attention to explore their similarity in hidden states, considering the layer depth. We observe that as the layers get deeper, the cosine similarity between the normalized hidden states of the sink token and those of other tokens increases, and that the normalized hidden states of the sink token exhibit negligible changes. These imply that other tokens consistently are directed toward the sink token throughout the layers. Next, we propose a dynamic token selection method, called OrthoRank, using these findings to select important tokens. Specifically, in a certain layer, we define token importance by the speed at which the token moves toward the sink token. This is converted into orthogonality with the sink token, meaning that tokens that are more orthogonal to the sink token are assigned greater importance. Finally, through extensive experiments, we demonstrated that our method results in lower perplexity and higher zero-shot accuracy compared to layer pruning methods at the same sparsity ratio with comparable throughput, while also achieving superior performance on LongBench.
    </details>
</div>
