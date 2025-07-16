# llm - 2025_07

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
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11538v1">How Many Instructions Can LLMs Follow at Once?</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      Production-grade LLM systems require robust adherence to dozens or even hundreds of instructions simultaneously. However, the instruction-following capabilities of LLMs at high instruction densities have not yet been characterized, as existing benchmarks only evaluate models on tasks with a single or few instructions. We introduce IFScale, a simple benchmark of 500 keyword-inclusion instructions for a business report writing task to measure how instruction-following performance degrades as instruction density increases. We evaluate 20 state-of-the-art models across seven major providers and find that even the best frontier models only achieve 68% accuracy at the max density of 500 instructions. Our analysis reveals model size and reasoning capability to correlate with 3 distinct performance degradation patterns, bias towards earlier instructions, and distinct categories of instruction-following errors. Our insights can help inform design of instruction-dense prompts in real-world applications and highlight important performance-latency tradeoffs. We open-source the benchmark and all results for further analysis at https://distylai.github.io/IFScale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11525v1">LLM-based ambiguity detection in natural language instructions for collaborative surgical robots</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 Accepted at 2025 IEEE International Conference on Robot and Human Interactive Communication (ROMAN)
    </div>
    <details class="paper-abstract">
      Ambiguity in natural language instructions poses significant risks in safety-critical human-robot interaction, particularly in domains such as surgery. To address this, we propose a framework that uses Large Language Models (LLMs) for ambiguity detection specifically designed for collaborative surgical scenarios. Our method employs an ensemble of LLM evaluators, each configured with distinct prompting techniques to identify linguistic, contextual, procedural, and critical ambiguities. A chain-of-thought evaluator is included to systematically analyze instruction structure for potential issues. Individual evaluator assessments are synthesized through conformal prediction, which yields non-conformity scores based on comparison to a labeled calibration dataset. Evaluating Llama 3.2 11B and Gemma 3 12B, we observed classification accuracy exceeding 60% in differentiating ambiguous from unambiguous surgical instructions. Our approach improves the safety and reliability of human-robot collaboration in surgery by offering a mechanism to identify potentially ambiguous instructions before robot action.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11515v1">AirLLM: Diffusion Policy-based Adaptive LoRA for Remote Fine-Tuning of LLM over the Air</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 11 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Operating Large Language Models (LLMs) on edge devices is increasingly challenged by limited communication bandwidth and strained computational and memory costs. Thus, cloud-assisted remote fine-tuning becomes indispensable. Nevertheless, existing Low-Rank Adaptation (LoRA) approaches typically employ fixed or heuristic rank configurations, and the subsequent over-the-air transmission of all LoRA parameters could be rather inefficient. To address this limitation, we develop AirLLM, a hierarchical diffusion policy framework for communication-aware LoRA adaptation. Specifically, AirLLM models the rank configuration as a structured action vector that spans all LoRA-inserted projections. To solve the underlying high-dimensional sequential decision-making problem, a Proximal Policy Optimization (PPO) agent generates coarse-grained decisions by jointly observing wireless states and linguistic complexity, which are then refined via Denoising Diffusion Implicit Models (DDIM) to produce high-resolution, task- and channel-adaptive rank vectors. The two modules are optimized alternatively, with the DDIM trained under the Classifier-Free Guidance (CFG) paradigm to maintain alignment with PPO rewards. Experiments under varying signal-to-noise ratios demonstrate that AirLLM consistently enhances fine-tuning performance while significantly reducing transmission costs, highlighting the effectiveness of reinforcement-driven, diffusion-refined rank adaptation for scalable and efficient remote fine-tuning over the air.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11507v1">MIRAGE: KV Cache Optimization through Parameter Remapping for Multi-tenant LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      KV cache accelerates LLM inference by avoiding redundant computation, at the expense of memory. To support larger KV caches, prior work extends GPU memory with CPU memory via CPU-offloading. This involves swapping KV cache between GPU and CPU memory. However, because the cache updates dynamically, such swapping incurs high CPU memory traffic. We make a key observation that model parameters remain constant during runtime, unlike the dynamically updated KV cache. Building on this, we introduce MIRAGE, which avoids KV cache swapping by remapping, and thereby repurposing, the memory allocated to model parameters for KV cache. This parameter remapping is especially beneficial in multi-tenant environments, where the memory used for the parameters of the inactive models can be more aggressively reclaimed. Exploiting the high CPU-GPU bandwidth offered by the modern hardware, such as the NVIDIA Grace Hopper Superchip, we show that MIRAGE significantly outperforms state-of-the-art solutions, achieving a reduction of 44.8%-82.5% in tail time-between-token latency, 20.7%-99.3% in tail time-to-first-token latency, and 6.6%-86.7% higher throughput compared to vLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11457v1">LRMR: LLM-Driven Relational Multi-node Ranking for Lymph Node Metastasis Assessment in Rectal Cancer</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      Accurate preoperative assessment of lymph node (LN) metastasis in rectal cancer guides treatment decisions, yet conventional MRI evaluation based on morphological criteria shows limited diagnostic performance. While some artificial intelligence models have been developed, they often operate as black boxes, lacking the interpretability needed for clinical trust. Moreover, these models typically evaluate nodes in isolation, overlooking the patient-level context. To address these limitations, we introduce LRMR, an LLM-Driven Relational Multi-node Ranking framework. This approach reframes the diagnostic task from a direct classification problem into a structured reasoning and ranking process. The LRMR framework operates in two stages. First, a multimodal large language model (LLM) analyzes a composite montage image of all LNs from a patient, generating a structured report that details ten distinct radiological features. Second, a text-based LLM performs pairwise comparisons of these reports between different patients, establishing a relative risk ranking based on the severity and number of adverse features. We evaluated our method on a retrospective cohort of 117 rectal cancer patients. LRMR achieved an area under the curve (AUC) of 0.7917 and an F1-score of 0.7200, outperforming a range of deep learning baselines, including ResNet50 (AUC 0.7708). Ablation studies confirmed the value of our two main contributions: removing the relational ranking stage or the structured prompting stage led to a significant performance drop, with AUCs falling to 0.6875 and 0.6458, respectively. Our work demonstrates that decoupling visual perception from cognitive reasoning through a two-stage LLM framework offers a powerful, interpretable, and effective new paradigm for assessing lymph node metastasis in rectal cancer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23847v3">Seven Security Challenges That Must be Solved in Cross-domain Multi-agent LLM Systems</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are rapidly evolving into autonomous agents that cooperate across organizational boundaries, enabling joint disaster response, supply-chain optimization, and other tasks that demand decentralized expertise without surrendering data ownership. Yet, cross-domain collaboration shatters the unified trust assumptions behind current alignment and containment techniques. An agent benign in isolation may, when receiving messages from an untrusted peer, leak secrets or violate policy, producing risks driven by emergent multi-agent dynamics rather than classical software bugs. This position paper maps the security agenda for cross-domain multi-agent LLM systems. We introduce seven categories of novel security challenges, for each of which we also present plausible attacks, security evaluation metrics, and future research guidelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16366v3">A Generative Approach to LLM Harmfulness Detection with Special Red Flag Tokens</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 14 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Most safety training methods for large language models (LLMs) are based on fine-tuning that forces models to shift from an unsafe answer to refusal when faced with harmful requests. Unfortunately, these drastic distribution shifts generally compromise model capabilities. To avoid that, we propose to expand the model's vocabulary with a special token we call red flag token (<rf>) and propose to train the model to insert this token into its response at any time when harmful content is generated or about to be generated. Our approach offers several advantages: it enables the model to explicitly learn the concept of harmfulness while marginally affecting the generated distribution, thus maintaining the model's utility. It also evaluates each generated answer and provides robustness as good as adversarial training without the need to run attacks during training. Moreover, by encapsulating our safety tuning in a LoRA module, we provide additional defenses against fine-tuning API attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11417v1">Quantifying the Energy Consumption and Carbon Emissions of LLM Inference via Simulations</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 Presented at the Workshop on Performance and Energy Efficiency in Concurrent and Distributed Systems (PECS) at Euro-PAR'25
    </div>
    <details class="paper-abstract">
      The environmental impact of Large Language Models (LLMs) is rising significantly, with inference now accounting for more than half of their total lifecycle carbon emissions. However, existing simulation frameworks, which are increasingly used to determine efficient LLM deployments, lack any concept of power and, therefore, cannot accurately estimate inference-related emissions. We present a simulation framework to assess the energy and carbon implications of LLM inference under varying deployment setups. First, we extend a high-fidelity LLM inference simulator with a GPU power model that estimates power consumption based on utilization metrics, enabling analysis across configurations like batch size, sequence length, and model parallelism. Second, we integrate simulation outputs into an energy system co-simulation environment to quantify carbon emissions under specific grid conditions and explore the potential of carbon-aware scheduling. Through scenario-based analysis, our framework reveals how inference parameters affect energy demand and carbon footprint, demonstrates a renewable offset potential of up to 69.2% in an illustrative deployment case, and provides a foundation for future carbon-aware inference infrastructure design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00648v4">DFRot: Achieving Outlier-Free and Massive Activation-Free for Rotated LLMs with Refined Rotation</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 Accepeted bythe 2nd Conference on Language Modeling (COLM 2025). Source code \url{https://github.com/JingyangXiang/DFRot}
    </div>
    <details class="paper-abstract">
      Rotating the activation and weight matrices to reduce the influence of outliers in large language models (LLMs) has recently attracted significant attention, particularly in the context of model quantization. Prior studies have shown that in low-precision quantization scenarios, such as 4-bit weights and 4-bit activations (W4A4), randomized Hadamard transforms can achieve significantly higher accuracy than randomized orthogonal transforms. Notably, the reason behind this phenomenon remains unknown. In this paper, we find that these transformations show substantial improvement in eliminating outliers for common tokens and achieve similar quantization error. The primary reason for the accuracy difference lies in the fact that randomized Hadamard transforms can slightly reduce the quantization error for tokens with massive activations while randomized orthogonal transforms increase the quantization error. Due to the extreme rarity of these tokens and their critical impact on model accuracy, we consider this a long-tail optimization problem, and therefore construct a simple yet effective method: a weighted loss function. Additionally, we propose an optimization strategy for the rotation matrix that involves alternating optimization of quantization parameters while employing orthogonal Procrustes transforms to refine the rotation matrix. This makes the distribution of the rotated activation values more conducive to quantization, especially for tokens with massive activations. Our method enhances the Rotated LLMs by achieving dual free, Outlier-Free and Massive Activation-Free, dubbed as DFRot. Extensive experiments demonstrate the effectiveness and efficiency of DFRot. By tuning the rotation matrix using just a single sample, DFRot achieves a perplexity improvement of 0.98 and 0.95 on W4A4KV4 and W4A4KV16, respectively, for LLaMA3-70B, a model known for its quantization challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11408v1">KisMATH: Do LLMs Have Knowledge of Implicit Structures in Mathematical Reasoning?</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 15 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Chain-of-thought traces have been shown to improve performance of large language models in a plethora of reasoning tasks, yet there is no consensus on the mechanism through which this performance boost is achieved. To shed more light on this, we introduce Causal CoT Graphs (CCGs), which are directed acyclic graphs automatically extracted from reasoning traces that model fine-grained causal dependencies in the language model output. A collection of $1671$ mathematical reasoning problems from MATH500, GSM8K and AIME, and their associated CCGs are compiled into our dataset -- \textbf{KisMATH}. Our detailed empirical analysis with 15 open-weight LLMs shows that (i) reasoning nodes in the CCG are mediators for the final answer, a condition necessary for reasoning; and (ii) LLMs emphasise reasoning paths given by the CCG, indicating that models internally realise structures akin to our graphs. KisMATH enables controlled, graph-aligned interventions and opens up avenues for further investigation into the role of chain-of-thought in LLM reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11405v1">DCR: Quantifying Data Contamination in LLMs Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has heightened concerns about benchmark data contamination (BDC), where models inadvertently memorize evaluation data, inflating performance metrics and undermining genuine generalization assessment. This paper introduces the Data Contamination Risk (DCR) framework, a lightweight, interpretable pipeline designed to detect and quantify BDC across four granular levels: semantic, informational, data, and label. By synthesizing contamination scores via a fuzzy inference system, DCR produces a unified DCR Factor that adjusts raw accuracy to reflect contamination-aware performance. Validated on 9 LLMs (0.5B-72B) across sentiment analysis, fake news detection, and arithmetic reasoning tasks, the DCR framework reliably diagnoses contamination severity and with accuracy adjusted using the DCR Factor to within 4% average error across the three benchmarks compared to the uncontaminated baseline. Emphasizing computational efficiency and transparency, DCR provides a practical tool for integrating contamination assessment into routine evaluations, fostering fairer comparisons and enhancing the credibility of LLM benchmarking practices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11344v1">Guiding LLM Decision-Making with Fairness Reward Models</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      Large language models are increasingly used to support high-stakes decisions, potentially influencing who is granted bail or receives a loan. Naive chain-of-thought sampling can improve average decision accuracy, but has also been shown to amplify unfair bias. To address this challenge and enable the trustworthy use of reasoning models in high-stakes decision-making, we propose a framework for training a generalizable Fairness Reward Model (FRM). Our model assigns a fairness score to LLM reasoning, enabling the system to down-weight biased trajectories and favor equitable ones when aggregating decisions across reasoning chains. We show that a single Fairness Reward Model, trained on weakly supervised, LLM-annotated examples of biased versus unbiased reasoning, transfers across tasks, domains, and model families without additional fine-tuning. Applied to real-world decision-making tasks including recidivism prediction and social media moderation, we show that our approach consistently improves fairness while matching, or even surpassing, baseline accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06313v2">ETT: Expanding the Long Context Understanding Capability of LLMs at Test-Time</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      Transformer-based Language Models' computation and memory overhead increase quadratically as a function of sequence length. The quadratic cost poses challenges when employing LLMs for processing long sequences. In this work, we introduce \ourmodelacronym~(Extend at Test-Time), method for extending the context length of short context Transformer-based LLMs, with constant memory requirement and linear computation overhead. ETT enable the extension of the context length at test-time by efficient fine-tuning the model's parameters on the input context, chunked into overlapping small subsequences. We evaluate ETT on LongBench by extending the context length of GPT-Large and Phi-2 up to 32 times, increasing from 1k to 32k tokens. This results in up to a 30 percent improvement in the model's accuracy. We also study how context can be stored in LLM's weights effectively and efficiently. Through a detailed ablation study, we examine which Transformer modules are most beneficial to fine-tune at test-time. Interestingly, we find that fine-tuning the second layer of the FFNs is more effective than full fine-tuning, leading to a further improvement in the models' accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.23644v2">QLPro: Automated Code Vulnerability Discovery via LLM and Static Code Analysis Integration</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 The experimental data in the experimental section needs to be improved, and there are some errors
    </div>
    <details class="paper-abstract">
      We introduce QLPro, a vulnerability detection framework that systematically integrates LLMs and static analysis tools to enable comprehensive vulnerability detection across entire open-source projects.We constructed a new dataset, JavaTest, comprising 10 open-source projects from GitHub with 62 confirmed vulnerabilities. CodeQL, a state-of-the-art static analysis tool, detected only 24 of these vulnerabilities while QLPro detected 41. Furthermore, QLPro discovered 6 previously unknown vulnerabilities, 2 of which have been confirmed as 0-days.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.00166v3">EEG Emotion Copilot: Optimizing Lightweight LLMs for Emotional EEG Interpretation with Assisted Medical Record Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 17 pages, 16 figures, 5 tables
    </div>
    <details class="paper-abstract">
      In the fields of affective computing (AC) and brain-machine interface (BMI), the analysis of physiological and behavioral signals to discern individual emotional states has emerged as a critical research frontier. While deep learning-based approaches have made notable strides in EEG emotion recognition, particularly in feature extraction and pattern recognition, significant challenges persist in achieving end-to-end emotion computation, including real-time processing, individual adaptation, and seamless user interaction. This paper presents the EEG Emotion Copilot, a system optimizing a lightweight large language model (LLM) with 0.5B parameters operating in a local setting, which first recognizes emotional states directly from EEG signals, subsequently generates personalized diagnostic and treatment suggestions, and finally supports the automation of assisted electronic medical records. Specifically, we demonstrate the critical techniques in the novel data structure of prompt, model pruning and fine-tuning training, and deployment strategies aiming at improving real-time performance and computational efficiency. Extensive experiments show that our optimized lightweight LLM-based copilot achieves an enhanced intuitive interface for participant interaction, superior accuracy of emotion recognition and assisted electronic medical records generation, in comparison to such models with similar scale parameters or large-scale parameters such as 1.5B, 1.8B, 3B and 7B. In summary, through these efforts, the proposed copilot is expected to advance the application of AC in the medical domain, offering innovative solution to mental health monitoring. The codes will be released at https://github.com/NZWANG/EEG_Emotion_Copilot.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03106v3">Critique-GRPO: Advancing LLM Reasoning with Natural Language and Numerical Feedback</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 49 pages, updated with new experimental results
    </div>
    <details class="paper-abstract">
      Recent advances in reinforcement learning (RL) with numerical feedback, such as scalar rewards, have significantly enhanced the complex reasoning capabilities of large language models (LLMs). Despite this success, we identify three key challenges encountered by RL with solely numerical feedback: performance plateaus, limited effectiveness of self-reflection, and persistent failures. We then demonstrate that RL-finetuned models, even after exhibiting performance plateaus, can generate correct refinements on persistently failed problems by leveraging natural language feedback in the form of critiques. Building on this insight, we propose Critique-GRPO, an online RL framework that integrates both natural language and numerical feedback for effective policy optimization. Critique-GRPO enables LLMs to learn from initial responses and critique-guided self-refinements simultaneously while maintaining exploration. Additionally, we employ a shaping function to amplify learning from correct, especially unfamiliar, refinements and penalize incorrect ones. Extensive experiments with Qwen2.5-7B-Base, Qwen2.5-Math-7B-Base, and Qwen3-8B demonstrate that Critique-GRPO consistently outperforms supervised learning and RL-based fine-tuning methods across eight challenging mathematical, STEM, and general reasoning tasks, improving average pass@1 scores by approximately 4.4% and 3.8% on Qwen2.5-7B-Base and Qwen3-8B, respectively. Notably, Critique-GRPO enables effective self-improvement through self-critiquing and weak-to-strong generalization, achieving consistent gains over GRPO, such as 16.7% and 10.0% pass@1 improvements on AIME 2024, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02962v3">RAG-R1 : Incentivize the Search and Reasoning Capabilities of LLMs through Multi-query Parallelism</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities across various tasks, while they remain prone to generating hallucinated or outdated responses due to their static internal knowledge. Recent advancements in Retrieval-Augmented Generation (RAG) methods have explored enhancing models' search and reasoning capabilities through reinforcement learning (RL). Although these methods demonstrate promising results, they face challenges in training stability and encounter issues such as substantial inference time and restricted capabilities due to the single-query mode. In this paper, we propose RAG-R1, a novel training framework designed to enable LLMs to adaptively leverage internal and external knowledge during the reasoning process. We further expand the generation and retrieval processes within the framework from single-query mode to multi-query parallelism, aimed at reducing inference time and enhancing the model's capabilities. Extensive experiments on seven question-answering benchmarks demonstrate that our method outperforms the strongest baseline by up to 13.2% and decreases inference time by 11.1%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00838v2">Stylometry recognizes human and LLM-generated texts in short samples</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      The paper explores stylometry as a method to distinguish between texts created by Large Language Models (LLMs) and humans, addressing issues of model attribution, intellectual property, and ethical AI use. Stylometry has been used extensively to characterise the style and attribute authorship of texts. By applying it to LLM-generated texts, we identify their emergent writing patterns. The paper involves creating a benchmark dataset based on Wikipedia, with (a) human-written term summaries, (b) texts generated purely by LLMs (GPT-3.5/4, LLaMa 2/3, Orca, and Falcon), (c) processed through multiple text summarisation methods (T5, BART, Gensim, and Sumy), and (d) rephrasing methods (Dipper, T5). The 10-sentence long texts were classified by tree-based models (decision trees and LightGBM) using human-designed (StyloMetrix) and n-gram-based (our own pipeline) stylometric features that encode lexical, grammatical, syntactic, and punctuation patterns. The cross-validated results reached a performance of up to .87 Matthews correlation coefficient in the multiclass scenario with 7 classes, and accuracy between .79 and 1. in binary classification, with the particular example of Wikipedia and GPT-4 reaching up to .98 accuracy on a balanced dataset. Shapley Additive Explanations pinpointed features characteristic of the encyclopaedic text type, individual overused words, as well as a greater grammatical standardisation of LLMs with respect to human-written texts. These results show -- crucially, in the context of the increasingly sophisticated LLMs -- that it is possible to distinguish machine- from human-generated texts at least for a well-defined text type.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11210v1">Role-Playing LLM-Based Multi-Agent Support Framework for Detecting and Addressing Family Communication Bias</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      Well-being in family settings involves subtle psychological dynamics that conventional metrics often overlook. In particular, unconscious parental expectations, termed ideal parent bias, can suppress children's emotional expression and autonomy. This suppression, referred to as suppressed emotion, often stems from well-meaning but value-driven communication, which is difficult to detect or address from outside the family. Focusing on these latent dynamics, this study explores Large Language Model (LLM)-based support for psychologically safe family communication. We constructed a Japanese parent-child dialogue corpus of 30 scenarios, each annotated with metadata on ideal parent bias and suppressed emotion. Based on this corpus, we developed a Role-Playing LLM-based multi-agent dialogue support framework that analyzes dialogue and generates feedback. Specialized agents detect suppressed emotion, describe implicit ideal parent bias in parental speech, and infer contextual attributes such as the child's age and background. A meta-agent compiles these outputs into a structured report, which is then passed to five selected expert agents. These agents collaboratively generate empathetic and actionable feedback through a structured four-step discussion process. Experiments show that the system can detect categories of suppressed emotion with moderate accuracy and produce feedback rated highly in empathy and practicality. Moreover, simulated follow-up dialogues incorporating this feedback exhibited signs of improved emotional expression and mutual understanding, suggesting the framework's potential in supporting positive transformation in family interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10157v3">SocioVerse: A World Model for Social Simulation Powered by LLM Agents and A Pool of 10 Million Real-World Users</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      Social simulation is transforming traditional social science research by modeling human behavior through interactions between virtual individuals and their environments. With recent advances in large language models (LLMs), this approach has shown growing potential in capturing individual differences and predicting group behaviors. However, existing methods face alignment challenges related to the environment, target users, interaction mechanisms, and behavioral patterns. To this end, we introduce SocioVerse, an LLM-agent-driven world model for social simulation. Our framework features four powerful alignment components and a user pool of 10 million real individuals. To validate its effectiveness, we conducted large-scale simulation experiments across three distinct domains: politics, news, and economics. Results demonstrate that SocioVerse can reflect large-scale population dynamics while ensuring diversity, credibility, and representativeness through standardized procedures and minimal manual adjustments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11198v1">Temperature and Persona Shape LLM Agent Consensus With Minimal Accuracy Gains in Qualitative Coding</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 Manuscript submitted for review
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) enable new possibilities for qualitative research at scale, including coding and data annotation. While multi-agent systems (MAS) can emulate human coding workflows, their benefits over single-agent coding remain poorly understood. We conducted an experimental study of how agent persona and temperature shape consensus-building and coding accuracy of dialog segments based on a codebook with 8 codes. Our open-source MAS mirrors deductive human coding through structured agent discussion and consensus arbitration. Using six open-source LLMs (with 3 to 32 billion parameters) and 18 experimental configurations, we analyze over 77,000 coding decisions against a gold-standard dataset of human-annotated transcripts from online math tutoring sessions. Temperature significantly impacted whether and when consensus was reached across all six LLMs. MAS with multiple personas (including neutral, assertive, or empathetic), significantly delayed consensus in four out of six LLMs compared to uniform personas. In three of those LLMs, higher temperatures significantly diminished the effects of multiple personas on consensus. However, neither temperature nor persona pairing lead to robust improvements in coding accuracy. Single agents matched or outperformed MAS consensus in most conditions. Only one model (OpenHermesV2:7B) and code category showed above-chance gains from MAS deliberation when temperature was 0.5 or lower and especially when the agents included at least one assertive persona. Qualitative analysis of MAS collaboration for these configurations suggests that MAS may nonetheless aid in narrowing ambiguous code applications that could improve codebooks and human-AI coding. We contribute new insight into the limits of LLM-based qualitative methods, challenging the notion that diverse MAS personas lead to better outcomes. We open-source our MAS and experimentation code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08054v2">FalseReject: A Resource for Improving Contextual Safety and Mitigating Over-Refusals in LLMs via Structured Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 Accepted at COLM 2025
    </div>
    <details class="paper-abstract">
      Safety alignment approaches in large language models (LLMs) often lead to the over-refusal of benign queries, significantly diminishing their utility in sensitive scenarios. To address this challenge, we introduce FalseReject, a comprehensive resource containing 16k seemingly toxic queries accompanied by structured responses across 44 safety-related categories. We propose a graph-informed adversarial multi-agent interaction framework to generate diverse and complex prompts, while structuring responses with explicit reasoning to aid models in accurately distinguishing safe from unsafe contexts. FalseReject includes training datasets tailored for both standard instruction-tuned models and reasoning-oriented models, as well as a human-annotated benchmark test set. Our extensive benchmarking on 29 state-of-the-art (SOTA) LLMs reveals persistent over-refusal challenges. Empirical results demonstrate that supervised finetuning with FalseReject substantially reduces unnecessary refusals without compromising overall safety or general language capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11128v1">What Should LLMs Forget? Quantifying Personal Data in LLMs for Right-to-Be-Forgotten Requests</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 16 pages, 3 figures. Accepted at the 7th Workshop on eXplainable Knowledge Discovery in Data Mining (XKDD 2025), ECML PKDD 2025, Porto, Portugal
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can memorize and reveal personal information, raising concerns regarding compliance with the EU's GDPR, particularly the Right to Be Forgotten (RTBF). Existing machine unlearning methods assume the data to forget is already known but do not address how to identify which individual-fact associations are stored in the model. Privacy auditing techniques typically operate at the population level or target a small set of identifiers, limiting applicability to individual-level data inquiries. We introduce WikiMem, a dataset of over 5,000 natural language canaries covering 243 human-related properties from Wikidata, and a model-agnostic metric to quantify human-fact associations in LLMs. Our approach ranks ground-truth values against counterfactuals using calibrated negative log-likelihood across paraphrased prompts. We evaluate 200 individuals across 15 LLMs (410M-70B parameters), showing that memorization correlates with subject web presence and model scale. We provide a foundation for identifying memorized personal data in LLMs at the individual level, enabling the dynamic construction of forget sets for machine unlearning and RTBF requests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.21033v2">Plancraft: an evaluation dataset for planning with LLM agents</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      We present Plancraft, a multi-modal evaluation dataset for LLM agents. Plancraft has both a text-only and multi-modal interface, based on the Minecraft crafting GUI. We include the Minecraft Wiki to evaluate tool use and Retrieval Augmented Generation (RAG), as well as a handcrafted planner and Oracle Retriever, to ablate the different components of a modern agent architecture. To evaluate decision-making, Plancraft also includes a subset of examples that are intentionally unsolvable, providing a realistic challenge that requires the agent not only to complete tasks but also to decide whether they are solvable at all. We benchmark both open-source and closed-source LLMs and compare their performance and efficiency to a handcrafted planner. Overall, we find that LLMs and VLMs struggle with the planning problems that Plancraft introduces, and offer suggestions on how to improve their capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11112v1">Multi-Trigger Poisoning Amplifies Backdoor Vulnerabilities in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      Recent studies have shown that Large Language Models (LLMs) are vulnerable to data poisoning attacks, where malicious training examples embed hidden behaviours triggered by specific input patterns. However, most existing works assume a phrase and focus on the attack's effectiveness, offering limited understanding of trigger mechanisms and how multiple triggers interact within the model. In this paper, we present a framework for studying poisoning in LLMs. We show that multiple distinct backdoor triggers can coexist within a single model without interfering with each other, enabling adversaries to embed several triggers concurrently. Using multiple triggers with high embedding similarity, we demonstrate that poisoned triggers can achieve robust activation even when tokens are substituted or separated by long token spans. Our findings expose a broader and more persistent vulnerability surface in LLMs. To mitigate this threat, we propose a post hoc recovery method that selectively retrains specific model components based on a layer-wise weight difference analysis. Our method effectively removes the trigger behaviour with minimal parameter updates, presenting a practical and efficient defence against multi-trigger poisoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11097v1">The Devil behind the mask: An emergent safety vulnerability of Diffusion LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 21 pages, 9 figures, work in progress
    </div>
    <details class="paper-abstract">
      Diffusion-based large language models (dLLMs) have recently emerged as a powerful alternative to autoregressive LLMs, offering faster inference and greater interactivity via parallel decoding and bidirectional modeling. However, despite strong performance in code generation and text infilling, we identify a fundamental safety concern: existing alignment mechanisms fail to safeguard dLLMs against context-aware, masked-input adversarial prompts, exposing novel vulnerabilities. To this end, we present DIJA, the first systematic study and jailbreak attack framework that exploits unique safety weaknesses of dLLMs. Specifically, our proposed DIJA constructs adversarial interleaved mask-text prompts that exploit the text generation mechanisms of dLLMs, i.e., bidirectional modeling and parallel decoding. Bidirectional modeling drives the model to produce contextually consistent outputs for masked spans, even when harmful, while parallel decoding limits model dynamic filtering and rejection sampling of unsafe content. This causes standard alignment mechanisms to fail, enabling harmful completions in alignment-tuned dLLMs, even when harmful behaviors or unsafe instructions are directly exposed in the prompt. Through comprehensive experiments, we demonstrate that DIJA significantly outperforms existing jailbreak methods, exposing a previously overlooked threat surface in dLLM architectures. Notably, our method achieves up to 100% keyword-based ASR on Dream-Instruct, surpassing the strongest prior baseline, ReNeLLM, by up to 78.5% in evaluator-based ASR on JailbreakBench and by 37.7 points in StrongREJECT score, while requiring no rewriting or hiding of harmful content in the jailbreak prompt. Our findings underscore the urgent need for rethinking safety alignment in this emerging class of language models. Code is available at https://github.com/ZichenWen1/DIJA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11086v1">Beyond Traditional Algorithms: Leveraging LLMs for Accurate Cross-Border Entity Identification</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      The growing prevalence of cross-border financial activities in global markets has underscored the necessity of accurately identifying and classifying foreign entities. This practice is essential within the Spanish financial system for ensuring robust risk management, regulatory adherence, and the prevention of financial misconduct. This process involves a labor-intensive entity-matching task, where entities need to be validated against available reference sources. Challenges arise from linguistic variations, special characters, outdated names, and changes in legal forms, complicating traditional matching algorithms like Jaccard, cosine, and Levenshtein distances. These methods struggle with contextual nuances and semantic relationships, leading to mismatches. To address these limitations, we explore Large Language Models (LLMs) as a flexible alternative. LLMs leverage extensive training to interpret context, handle abbreviations, and adapt to legal transitions. We evaluate traditional methods, Hugging Face-based LLMs, and interface-based LLMs (e.g., Microsoft Copilot, Alibaba's Qwen 2.5) using a dataset of 65 Portuguese company cases. Results show traditional methods achieve accuracies over 92% but suffer high false positive rates (20-40%). Interface-based LLMs outperform, achieving accuracies above 93%, F1 scores exceeding 96%, and lower false positives (40-80%).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11083v1">Function-to-Style Guidance of LLMs for Code Translation</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 This paper has been accepted by ICML 2025. Models and benchmarks can be found at https://www.modelscope.cn/collections/F2STrans-42526ff95dd843
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have made significant strides in code translation tasks. However, ensuring both the correctness and readability of translated code remains a challenge, limiting their effective adoption in real-world software development. In this work, we propose F2STrans, a function-to-style guiding paradigm designed to progressively improve the performance of LLMs in code translation. Our approach comprises two key stages: (1) Functional learning, which optimizes translation correctness using high-quality source-target code pairs mined from online programming platforms, and (2) Style learning, which improves translation readability by incorporating both positive and negative style examples. Additionally, we introduce a novel code translation benchmark that includes up-to-date source code, extensive test cases, and manually annotated ground-truth translations, enabling comprehensive functional and stylistic evaluations. Experiments on both our new benchmark and existing datasets demonstrate that our approach significantly improves code translation performance. Notably, our approach enables Qwen-1.5B to outperform prompt-enhanced Qwen-32B and GPT-4 on average across 20 diverse code translation scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11052v1">LLM-Augmented Symptom Analysis for Cardiovascular Disease Risk Prediction: A Clinical NLP</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      Timely identification and accurate risk stratification of cardiovascular disease (CVD) remain essential for reducing global mortality. While existing prediction models primarily leverage structured data, unstructured clinical notes contain valuable early indicators. This study introduces a novel LLM-augmented clinical NLP pipeline that employs domain-adapted large language models for symptom extraction, contextual reasoning, and correlation from free-text reports. Our approach integrates cardiovascular-specific fine-tuning, prompt-based inference, and entity-aware reasoning. Evaluations on MIMIC-III and CARDIO-NLP datasets demonstrate improved performance in precision, recall, F1-score, and AUROC, with high clinical relevance (kappa = 0.82) assessed by cardiologists. Challenges such as contextual hallucination, which occurs when plausible information contracts with provided source, and temporal ambiguity, which is related with models struggling with chronological ordering of events are addressed using prompt engineering and hybrid rule-based verification. This work underscores the potential of LLMs in clinical decision support systems (CDSS), advancing early warning systems and enhancing the translation of patient narratives into actionable risk assessments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11042v1">Aligned Query Expansion: Efficient Query Expansion for Information Retrieval through LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      With the breakthroughs in large language models (LLMs), query generation techniques that expand documents and queries with related terms are becoming increasingly popular in the information retrieval field. Such techniques have been shown to improve the effectiveness of traditional lexical retrieval methods by dealing with the vocabulary mismatch problem. Recent work has found that generating queries with a greedy decoding strategy can produce sub-optimal queries, including hallucinations, and proposed to filter out queries before expansion. This `generate-then-filter' approach is costly, as it requires generating multiple queries and applying a relevance model to all of them and does not teach the LLM which of the generated queries is more effective for expansion. To overcome such limitations, we propose Aligned Query Expansion (AQE), a novel approach to enhance query expansion for passage retrieval in open-domain question answering. AQE leverages recent techniques in LLM alignment to fine-tune models for generating query expansions that directly optimize the effectiveness of the retrieval task, eliminating the need for additional filtering steps. This alignment ensures that queries are more relevant, reducing computational costs while improving retrieval effectiveness. Empirical evaluations show that AQE outperforms baseline models for query expansion in both in-domain and out-of-domain settings, demonstrating significant improvements in retrieval effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14959v2">Understanding the Dark Side of LLMs' Intrinsic Self-Correction</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      Intrinsic self-correction was proposed to improve LLMs' responses via feedback prompts solely based on their inherent capability. However, recent works show that LLMs' intrinsic self-correction fails without oracle labels as feedback prompts. In this paper, we aim to interpret LLMs' intrinsic self-correction for different tasks, especially for those failure cases. By including one simple task and three complex tasks with state-of-the-art (SOTA) LLMs like ChatGPT families (o1, 4o, 3.5-turbo) and Llama families (2-7B, 3-8B, and 3.1-8B), we design three interpretation methods to reveal the dark side of LLMs' intrinsic self-correction. We identify intrinsic self-correction can (1) cause LLMs to waver both intermedia and final answers and lead to prompt bias on simple factual questions; (2) introduce human-like cognitive bias on complex tasks. In light of our findings, we also provide two simple yet effective strategies for alleviation: question repeating and supervised fine-tuning with a few samples. We open-source our work at https://x-isc.info/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19502v2">MATE: LLM-Powered Multi-Agent Translation Environment for Accessibility Applications</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      Accessibility remains a critical concern in today's society, as many technologies are not developed to support the full range of user needs. Existing multi-agent systems (MAS) often cannot provide comprehensive assistance for users in need due to the lack of customization stemming from closed-source designs. Consequently, individuals with disabilities frequently encounter significant barriers when attempting to interact with digital environments. We introduce MATE, a multimodal accessibility MAS, which performs the modality conversions based on the user's needs. The system is useful for assisting people with disabilities by ensuring that data will be converted to an understandable format. For instance, if the user cannot see well and receives an image, the system converts this image to its audio description. MATE can be applied to a wide range of domains, industries, and areas, such as healthcare, and can become a useful assistant for various groups of users. The system supports multiple types of models, ranging from LLM API calling to using custom machine learning (ML) classifiers. This flexibility ensures that the system can be adapted to various needs and is compatible with a wide variety of hardware. Since the system is expected to run locally, it ensures the privacy and security of sensitive information. In addition, the framework can be effectively integrated with institutional technologies (e.g., digital healthcare service) for real-time user assistance. Furthermore, we introduce ModCon-Task-Identifier, a model that is capable of extracting the precise modality conversion task from the user input. Numerous experiments show that ModCon-Task-Identifier consistently outperforms other LLMs and statistical models on our custom data. Our code and data are publicly available at https://github.com/AlgazinovAleksandr/Multi-Agent-MATE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08898v2">SEALGuard: Safeguarding the Multilingual Conversations in Southeast Asian Languages for LLM Software Systems</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 Under Review at Information and Software Technology
    </div>
    <details class="paper-abstract">
      Safety alignment is critical for LLM-powered systems. While recent LLM-powered guardrail approaches such as LlamaGuard achieve high detection accuracy of unsafe inputs written in English (e.g., ``How to create a bomb?''), they struggle with multilingual unsafe inputs. This limitation leaves LLM systems vulnerable to unsafe and jailbreak prompts written in low-resource languages such as those in Southeast Asia. This paper introduces SEALGuard, a multilingual guardrail designed to improve the safety alignment across diverse languages. It aims to address the multilingual safety alignment gap of existing guardrails and ensure effective filtering of unsafe and jailbreak prompts in LLM-powered systems. We adapt a general-purpose multilingual language model into a multilingual guardrail using low-rank adaptation (LoRA). We construct SEALSBench, a large-scale multilingual safety alignment dataset containing over 260,000 prompts in ten languages, including safe, unsafe, and jailbreak cases. We evaluate SEALGuard against state-of-the-art guardrails such as LlamaGuard on this benchmark. Our findings show that multilingual unsafe and jailbreak prompts substantially degrade the performance of the state-of-the-art LlamaGuard, which experiences a drop in Defense Success Rate (DSR) by 9% and 18%, respectively, compared to its performance on English-only prompts. In contrast, SEALGuard outperforms existing guardrails in detecting multilingual unsafe and jailbreak prompts, improving DSR by 48% over LlamaGuard and achieving the best DSR, precision, and F1-score. Our ablation study further reveals the contributions of adaptation strategies and model size to the overall performance of SEALGuard. SEALGuard advances the safety alignment of LLM systems by introducing an effective multilingual guardrail.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10972v1">Teach Me Sign: Stepwise Prompting LLM for Sign Language Production</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 Accepted by IEEE ICIP 2025
    </div>
    <details class="paper-abstract">
      Large language models, with their strong reasoning ability and rich knowledge, have brought revolution to many tasks of AI, but their impact on sign language generation remains limited due to its complexity and unique rules. In this paper, we propose TEAch Me Sign (TEAM-Sign), treating sign language as another natural language. By fine-tuning an LLM, we enable it to learn the correspondence between text and sign language, and facilitate generation. Considering the differences between sign and spoken language, we employ a stepwise prompting strategy to extract the inherent sign language knowledge within the LLM, thereby supporting the learning and generation process. Experimental results on How2Sign and Phoenix14T datasets demonstrate that our approach effectively leverages both the sign language knowledge and reasoning capabilities of LLM to align the different distribution and grammatical rules between sign and spoken language.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10920v1">HanjaBridge: Resolving Semantic Ambiguity in Korean LLMs via Hanja-Augmented Pre-Training</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often show poor performance in low-resource languages like Korean, partly due to unique linguistic challenges such as homophonous Sino-Korean words that are indistinguishable in Hangul script. To address this semantic ambiguity, we propose HanjaBridge, a novel meaning-injection technique integrated into a continual pre-training (CPT) framework. Instead of deterministically mapping a word to a single Hanja (Chinese character), HanjaBridge presents the model with all possible Hanja candidates for a given homograph, encouraging the model to learn contextual disambiguation. This process is paired with token-level knowledge distillation to prevent catastrophic forgetting. Experimental results show that HanjaBridge significantly improves Korean language understanding, achieving a 21\% relative improvement on the KoBALT benchmark. Notably, by reinforcing semantic alignment between Korean and Chinese through shared Hanja, we observe a strong positive cross-lingual transfer. Furthermore, these gains persist even when Hanja augmentation is omitted at inference time, ensuring practical efficiency with no additional run-time cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10917v1">LLM-Driven Dual-Level Multi-Interest Modeling for Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 10 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Recently, much effort has been devoted to modeling users' multi-interests based on their behaviors or auxiliary signals. However, existing methods often rely on heuristic assumptions, e.g., co-occurring items indicate the same interest of users, failing to capture user multi-interests aligning with real-world scenarios. While large language models (LLMs) show significant potential for multi-interest analysis due to their extensive knowledge and powerful reasoning capabilities, two key challenges remain. First, the granularity of LLM-driven multi-interests is agnostic, possibly leading to overly fine or coarse interest grouping. Second, individual user analysis provides limited insights due to the data sparsity issue. In this paper, we propose an LLM-driven dual-level multi-interest modeling framework for more effective recommendation. At the user-individual level, we exploit LLMs to flexibly allocate items engaged by users into different semantic clusters, indicating their diverse and distinct interests. To alleviate the agnostic generation of LLMs, we adaptively assign these semantic clusters to users' collaborative multi-interests learned from global user-item interactions, allowing the granularity to be automatically adjusted according to the user's behaviors using an alignment module. To alleviate the limited insights derived from individual users' behaviors, at the user-crowd level, we propose aggregating user cliques into synthesized users with rich behaviors for more comprehensive LLM-driven multi-interest analysis. We formulate a max covering problem to ensure the compactness and representativeness of synthesized users' behaviors, and then conduct contrastive learning based on their LLM-driven multi-interests to disentangle item representations among different interests. Experiments on real-world datasets show the superiority of our approach against state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09116v2">Mixture of LoRA Experts with Multi-Modal and Multi-Granularity LLM Generative Error Correction for Accented Speech Recognition</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 IEEE Transactions on Audio, Speech and Language Processing
    </div>
    <details class="paper-abstract">
      Despite substantial improvements in ASR, performance tends to degrade when faced with adverse conditions such as speaker accents. Generative error correction (GER) leverages the rich linguistic knowledge and exceptional reasoning ability of LLMs, significantly outperforming typical LM methods. However, it lacks specificity in accented speech scenarios. In this study, we leverage GER to improve the accuracy of transcription predictions by addressing the two primary features of accented speech recognition. To fully leverage pronunciation information, we propose the multi-modal GER, which integrates pronunciation information from the speech modality, and the multi-granularity GER, which incorporates fine-grained phoneme-level information related to pronunciation. These two methods enable the LLM to utilize the pronunciation information of accented speech and the semantic information from word-level hypotheses for accurate transcription predictions through LoRA fine-tuning. On the one hand, we employ a three-stage training strategy to train separate multi-modal GER models for each accent to obtain mono-accent LoRA experts. By adopting our proposed HDMoLE method, which incorporates hierarchical routing and dynamic thresholds within the mixture of LoRA experts, we effectively merge multiple mono-accent LoRA experts within a single multi-modal GER to overcome the challenges posed by accent diversity. On the other hand, multi-granularity GER leverages the N-best word-level and phoneme-level hypotheses generated by the HDMoLE model to predict the final accented speech transcriptions. Experimental results on the multi-accent English dataset demonstrate the efficacy of our proposed methods. Our methods achieve a remarkable relative WER reduction of 67.35% compared to the Whisper-large-v3 baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10911v1">Lessons Learned from Evaluation of LLM based Multi-agents in Safer Therapy Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      Therapy recommendation for chronic patients with multimorbidity is challenging due to risks of treatment conflicts. Existing decision support systems face scalability limitations. Inspired by the way in which general practitioners (GP) manage multimorbidity patients, occasionally convening multidisciplinary team (MDT) collaboration, this study investigated the feasibility and value of using a Large Language Model (LLM)-based multi-agent system (MAS) for safer therapy recommendations. We designed a single agent and a MAS framework simulating MDT decision-making by enabling discussion among LLM agents to resolve medical conflicts. The systems were evaluated on therapy planning tasks for multimorbidity patients using benchmark cases. We compared MAS performance with single-agent approaches and real-world benchmarks. An important contribution of our study is the definition of evaluation metrics that go beyond the technical precision and recall and allow the inspection of clinical goals met and medication burden of the proposed advices to a gold standard benchmark. Our results show that with current LLMs, a single agent GP performs as well as MDTs. The best-scoring models provide correct recommendations that address all clinical goals, yet the advices are incomplete. Some models also present unnecessary medications, resulting in unnecessary conflicts between medication and conditions or drug-drug interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06608v3">Proactive Intra-GPU Disaggregation of Prefill and Decode in LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      Monolithic serving with chunked prefill improves GPU utilization by batching prefill and decode together, but suffers from fine-grained phase interference. Engine-level prefill-decode (PD) disaggregation avoids interference but incurs higher hardware and coordination overhead. Prior intra-GPU disaggregation approaches multiplex prefill and decode within a single GPU, using SLO-based tuning guided by heuristics from offline profiling or reactive feedback loops. However, these methods respond reactively to performance issues rather than anticipating them, limiting adaptability under dynamic workloads. We ask: can we achieve proactive intra-GPU disaggregation that adapts effectively to dynamic workloads? The key challenge lies in managing the conflicting resource demands of prefill and decode under varying conditions. We first show that GPU resources exhibit diminishing returns -- beyond a saturation point, more allocation yields minimal latency benefit. Second, we observe that memory bandwidth contention becomes a critical bottleneck. These insights motivate a design that dynamically partitions GPU resources across prefill and decode phases, while jointly considering compute capacity, memory footprint, and bandwidth contention. Evaluated on diverse LLMs and workloads, our system Nexus achieves up to 2.2x higher throughput, 20x lower TTFT, and 2.5x lower TBT than vLLM; outperforms SGLang by up to 2x; and matches or exceeds disaggregated vLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01100v2">ZebraLogic: On the Scaling Limits of LLMs for Logical Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 Accepted to ICML 2025
    </div>
    <details class="paper-abstract">
      We investigate the logical reasoning capabilities of large language models (LLMs) and their scalability in complex non-monotonic reasoning. To this end, we introduce ZebraLogic, a comprehensive evaluation framework for assessing LLM reasoning performance on logic grid puzzles derived from constraint satisfaction problems (CSPs). ZebraLogic enables the generation of puzzles with controllable and quantifiable complexity, facilitating a systematic study of the scaling limits of models such as Llama, o1 models, and DeepSeek-R1. By encompassing a broad range of search space complexities and diverse logical constraints, ZebraLogic provides a structured environment to evaluate reasoning under increasing difficulty. Our results reveal a significant decline in accuracy as problem complexity grows -- a phenomenon we term the curse of complexity. This limitation persists even with larger models and increased inference-time computation, suggesting inherent constraints in current LLM reasoning capabilities. Additionally, we explore strategies to enhance logical reasoning, including Best-of-N sampling, backtracking mechanisms, and self-verification prompts. Our findings offer critical insights into the scalability of LLM reasoning, highlight fundamental limitations, and outline potential directions for improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10873v1">From Alerts to Intelligence: A Novel LLM-Aided Framework for Host-based Intrusion Detection</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      Host-based intrusion detection system (HIDS) is a key defense component to protect the organizations from advanced threats like Advanced Persistent Threats (APT). By analyzing the fine-grained logs with approaches like data provenance, HIDS has shown successes in capturing sophisticated attack traces. Despite the progresses embarked by the research community and industry, HIDS still frequently encounters backlash from their operators in the deployed environments, due to issues like high false-positive rate, inconsistent outcomes across environments and human-unfriendly detection results. Large Language Models (LLMs) have great potentials to advance the state of HIDS, given their extensive knowledge of attack techniques and their ability to detect anomalies through semantic analysis, anchored by recent studies. Yet, our preliminary analysis indicates that building an HIDS by naively prompting an LLM is unlikely to succeed. In this work, we explore the direction of building a customized LLM pipeline for HIDS and develop a system named SHIELD. SHIELD addresses challenges related to LLM's token limits, confusion of background noises, etc., by integrating a variety of techniques like event-level Masked Autoencoder (MAE) for attack window detection, attack evidence identification and expansion, Deterministic Data Augmentation (DDA) for profiling normal activities, and multi-purpose prompting that guides the LLM to conduct precise and interpretable attack investigations. Extensive experiments on three log datasets (DARPA-E3, NodLink-simulated-data and ATLASv2) show that SHIELD consistently achieves outstanding performance in comparison with 5 representative HIDS. These findings highlight the potential of LLMs as powerful tools for intrusion detection and pave the way for future research in this domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21418v2">Autonomous Multi-Modal LLM Agents for Treatment Planning in Focused Ultrasound Ablation Surgery</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      Focused Ultrasound Ablation Surgery (FUAS) has emerged as a promising non-invasive therapeutic modality, valued for its safety and precision. Nevertheless, its clinical implementation entails intricate tasks such as multimodal image interpretation, personalized dose planning, and real-time intraoperative decision-making processes that demand intelligent assistance to improve efficiency and reliability. We introduce FUAS-Agents, an autonomous agent system that leverages the multimodal understanding and tool-using capabilities of large language models (LLMs). By integrating patient profiles and MRI data, FUAS-Agents orchestrates a suite of specialized medical AI tools, including segmentation, treatment dose prediction, and clinical guideline retrieval, to generate personalized treatment plans comprising MRI image, dose parameters, and therapeutic strategies. We evaluate the system in a uterine fibroid treatment scenario. Human assessment by four senior FUAS experts indicates that 82.5%, 82.5%, 87.5%, and 97.5% of the generated plans were rated 4 or above (on a 5-point scale) in terms of completeness, accuracy, fluency, and clinical compliance, respectively. These results demonstrate the potential of LLM-driven agents in enhancing decision-making across complex clinical workflows, and exemplify a translational paradigm that combines general-purpose models with specialized expert systems to solve practical challenges in vertical healthcare domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10540v1">Fusing LLM Capabilities with Routing Data</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has created a vibrant ecosystem of diverse architectures, each with unique strengths due to differences in design, training data, and objectives. However, most applications still rely on a single backend model, limiting coverage of capabilities and leading to inefficiencies in performance and token cost when tackling complex tasks. We highlight an underexploited opportunity: LLM routing data, produced when hosting platforms route diverse queries to different models, which can reveal comparative strengths across tasks. To address this, we propose FusionBench, a comprehensive routing benchmark covering 14 tasks across five domains with 20 open-source LLMs (8B to 671B parameters), capturing 103M tokens and summarizing reusable thought templates from top models. Building on this, we introduce FusionFactory, a systematic fusion framework with three levels: (1) query-level fusion, tailoring routers for each query using both direct responses and reasoning-augmented outputs; (2) thought-level fusion, leveraging abstract templates derived from top-performing LLMs' answers to similar queries; and (3) model-level fusion, transferring capabilities between models via distillation, using top responses or highest judge scores as training data. Experiments show FusionFactory consistently outperforms the best individual LLM across all 14 benchmarks, with optimal fusion configurations varying by benchmark, demonstrating the value of systematic LLM fusion in harnessing complementary strengths and improving overall performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10535v1">CodeJudgeBench: Benchmarking LLM-as-a-Judge for Coding Tasks</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 Dataset is available at https://huggingface.co/datasets/mattymchen/codejudgebench
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have significantly advanced the state-of-the-art in various coding tasks. Beyond directly answering user queries, LLMs can also serve as judges, assessing and comparing the quality of responses generated by other models. Such an evaluation capability is crucial both for benchmarking different LLMs and for improving response quality through response ranking. However, despite the growing adoption of the LLM-as-a-Judge paradigm, its effectiveness in coding scenarios remains underexplored due to the absence of dedicated benchmarks. To address this gap, we introduce CodeJudgeBench, a benchmark explicitly designed to evaluate the performance of LLM-as-a-Judge models across three critical coding tasks: code generation, code repair, and unit test generation. Through comprehensive benchmarking of 26 LLM-as-a-Judge models, we find that recent thinking models significantly outperform non-thinking models on our carefully designed code judging tasks. Notably, even relatively small thinking models, such as Qwen3-8B, can outperform specially trained LLM-as-a-Judge models up to 70B in size. Nevertheless, all models still exhibit significant randomness in their judgment of coding tasks. For pairwise judging tasks, simply changing the order in which responses are presented can substantially impact accuracy. In addition, when judging code and unit tests written by different LLMs, LLM-as-a-Judge models also show variance in performance. This sensitivity raises concerns about the reliability and consistency of LLM-as-a-Judge in coding scenarios. Lastly, we study optimal prompting strategies for LLM-as-a-Judge. We find that using pair-wise comparison outperforms scalar point-wise judging. Furthermore, retaining comments and reasoning in the full, unprocessed LLM response leads to improved judge performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10445v1">Referential ambiguity and clarification requests: comparing human and LLM behaviour</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      In this work we examine LLMs' ability to ask clarification questions in task-oriented dialogues that follow the asynchronous instruction-giver/instruction-follower format. We present a new corpus that combines two existing annotations of the Minecraft Dialogue Corpus -- one for reference and ambiguity in reference, and one for SDRT including clarifications -- into a single common format providing the necessary information to experiment with clarifications and their relation to ambiguity. With this corpus we compare LLM actions with original human-generated clarification questions, examining how both humans and LLMs act in the case of ambiguity. We find that there is only a weak link between ambiguity and humans producing clarification questions in these dialogues, and low correlation between humans and LLMs. Humans hardly ever produce clarification questions for referential ambiguity, but often do so for task-based uncertainty. Conversely, LLMs produce more clarification questions for referential ambiguity, but less so for task uncertainty. We question if LLMs' ability to ask clarification questions is predicated on their recent ability to simulate reasoning, and test this with different reasoning approaches, finding that reasoning does appear to increase question frequency and relevancy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10427v1">Towards Emotion Co-regulation with LLM-powered Socially Assistive Robots: Integrating LLM Prompts and Robotic Behaviors to Support Parent-Neurodivergent Child Dyads</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 Submission for the IROS 2025 conference
    </div>
    <details class="paper-abstract">
      Socially Assistive Robotics (SAR) has shown promise in supporting emotion regulation for neurodivergent children. Recently, there has been increasing interest in leveraging advanced technologies to assist parents in co-regulating emotions with their children. However, limited research has explored the integration of large language models (LLMs) with SAR to facilitate emotion co-regulation between parents and children with neurodevelopmental disorders. To address this gap, we developed an LLM-powered social robot by deploying a speech communication module on the MiRo-E robotic platform. This supervised autonomous system integrates LLM prompts and robotic behaviors to deliver tailored interventions for both parents and neurodivergent children. Pilot tests were conducted with two parent-child dyads, followed by a qualitative analysis. The findings reveal MiRo-E's positive impacts on interaction dynamics and its potential to facilitate emotion regulation, along with identified design and technical challenges. Based on these insights, we provide design implications to advance the future development of LLM-powered SAR for mental health applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10392v1">Zorse: Optimizing LLM Training Efficiency on Heterogeneous GPU Clusters</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) require vast amounts of GPU compute to train, but limited availability and high costs of GPUs make homogeneous clusters impractical for many organizations. Instead, assembling heterogeneous clusters by pooling together GPUs of different generations allows them to achieve higher aggregate compute and make use of all available GPUs. However, training on heterogeneous clusters presents several challenges, including load balancing across GPUs, optimizing memory usage to accommodate varying memory capacities, and ensuring communication-efficient training over diverse network interconnects potentially spanning multiple datacenters. In this paper, we make the case that efficient training on heterogeneous clusters requires (1) the integration of pipeline parallelism and data parallelism in a manner that is both communication- and memory-efficient, and (2) a more adaptable configuration of pipeline and data parallelism, which includes the capability to flexibly partition GPUs into asymmetric pipeline parallel stages and to incorporate heterogeneous GPUs within the same data parallelism group. We propose Zorse, the first system to unify all these capabilities while incorporating a planner that automatically configures training strategies for a given workload. Our evaluation shows that Zorse significantly outperforms state-of-the-art systems in heterogeneous training scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11168v3">Bypassing LLM Guardrails: An Empirical Analysis of Evasion Attacks against Prompt Injection and Jailbreak Detection Systems</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 14 pages, 5 figures, 11 tables. To be published in LLMSec 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) guardrail systems are designed to protect against prompt injection and jailbreak attacks. However, they remain vulnerable to evasion techniques. We demonstrate two approaches for bypassing LLM prompt injection and jailbreak detection systems via traditional character injection methods and algorithmic Adversarial Machine Learning (AML) evasion techniques. Through testing against six prominent protection systems, including Microsoft's Azure Prompt Shield and Meta's Prompt Guard, we show that both methods can be used to evade detection while maintaining adversarial utility achieving in some instances up to 100% evasion success. Furthermore, we demonstrate that adversaries can enhance Attack Success Rates (ASR) against black-box targets by leveraging word importance ranking computed by offline white-box models. Our findings reveal vulnerabilities within current LLM protection mechanisms and highlight the need for more robust guardrail systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10382v1">Leveraging RAG-LLMs for Urban Mobility Simulation and Analysis</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      With the rise of smart mobility and shared e-mobility services, numerous advanced technologies have been applied to this field. Cloud-based traffic simulation solutions have flourished, offering increasingly realistic representations of the evolving mobility landscape. LLMs have emerged as pioneering tools, providing robust support for various applications, including intelligent decision-making, user interaction, and real-time traffic analysis. As user demand for e-mobility continues to grow, delivering comprehensive end-to-end solutions has become crucial. In this paper, we present a cloud-based, LLM-powered shared e-mobility platform, integrated with a mobile application for personalized route recommendations. The optimization module is evaluated based on travel time and cost across different traffic scenarios. Additionally, the LLM-powered RAG framework is evaluated at the schema level for different users, using various evaluation methods. Schema-level RAG with XiYanSQL achieves an average execution accuracy of 0.81 on system operator queries and 0.98 on user queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.06238v2">EVOLvE: Evaluating and Optimizing LLMs For In-Context Exploration</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 28 pages. Published at ICML 2025
    </div>
    <details class="paper-abstract">
      Despite their success in many domains, large language models (LLMs) remain under-studied in scenarios requiring optimal decision-making under uncertainty. This is crucial as many real-world applications, ranging from personalized recommendations to healthcare interventions, demand that LLMs not only predict but also actively learn to make optimal decisions through exploration. In this work, we measure LLMs' (in)ability to make optimal decisions in bandits, a state-less reinforcement learning setting relevant to many applications. We develop a comprehensive suite of environments, including both context-free and contextual bandits with varying task difficulties, to benchmark LLMs' performance. Motivated by the existence of optimal exploration algorithms, we propose efficient ways to integrate this algorithmic knowledge into LLMs: by providing explicit algorithm-guided support during inference; and through algorithm distillation via in-context demonstrations and fine-tuning, using synthetic data generated from these algorithms. Impressively, these techniques allow us to achieve superior exploration performance with smaller models, surpassing larger models on various tasks. We conducted an extensive ablation study to shed light on various factors, such as task difficulty and data representation, that influence the efficiency of LLM exploration. Additionally, we conduct a rigorous analysis of the LLM's exploration efficiency using the concept of regret, linking its ability to explore to the model size and underlying algorithm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02620v2">FlowSpec: Continuous Pipelined Speculative Decoding for Efficient Distributed LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 16 pages, and the last 3 are appendix
    </div>
    <details class="paper-abstract">
      Distributed inference serves as a promising approach to enabling the inference of large language models (LLMs) at the network edge. It distributes the inference process to multiple devices to ensure that the LLMs can fit into the device memory. Recent pipeline-based approaches have the potential to parallelize communication and computation, which helps reduce inference latency. However, the benefit diminishes when the inference request at the network edge is sparse, where pipeline is typically at low utilization. To enable efficient distributed LLM inference at the edge, we propose \textbf{FlowSpec}, a pipeline-parallel tree-based speculative decoding framework. FlowSpec incorporates three key mechanisms to improve decoding efficiency: 1) score-based step-wise verification prioritizes more important draft tokens to bring earlier accpeted tokens; 2) efficient draft management to prune invalid tokens while maintaining correct causal relationship during verification; 3) dynamic draft expansion strategies to supply high-quality speculative inputs. These techniques work in concert to enhance both pipeline utilization and speculative efficiency. We evaluate FlowSpec on a real-world testbed with other baselines. Experimental results demonstrate that our proposed framework significantly improves inference speed across diverse models and configurations, achieving speedup ratios 1.28$\times$-1.79$\times$ compared to baselines. Our code is publicly available at \href{https://github.com/Leosang-lx/FlowSpec#}{https://github.com/Leosang-lx/FlowSpec\#}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10281v1">Toward Real-World Table Agents: Capabilities, Workflows, and Design Principles for LLM-based Table Intelligence</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      Tables are fundamental in domains such as finance, healthcare, and public administration, yet real-world table tasks often involve noise, structural heterogeneity, and semantic complexity--issues underexplored in existing research that primarily targets clean academic datasets. This survey focuses on LLM-based Table Agents, which aim to automate table-centric workflows by integrating preprocessing, reasoning, and domain adaptation. We define five core competencies--C1: Table Structure Understanding, C2: Table and Query Semantic Understanding, C3: Table Retrieval and Compression, C4: Executable Reasoning with Traceability, and C5: Cross-Domain Generalization--to analyze and compare current approaches. In addition, a detailed examination of the Text-to-SQL Agent reveals a performance gap between academic benchmarks and real-world scenarios, especially for open-source models. Finally, we provide actionable insights to improve the robustness, generalization, and efficiency of LLM-based Table Agents in practical settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10200v1">Natural Language-based Assessment of L2 Oral Proficiency using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 Accepted for the 10th Workshop on Speech and Language Technology in Education (SLaTE 2025)
    </div>
    <details class="paper-abstract">
      Natural language-based assessment (NLA) is an approach to second language assessment that uses instructions - expressed in the form of can-do descriptors - originally intended for human examiners, aiming to determine whether large language models (LLMs) can interpret and apply them in ways comparable to human assessment. In this work, we explore the use of such descriptors with an open-source LLM, Qwen 2.5 72B, to assess responses from the publicly available S&I Corpus in a zero-shot setting. Our results show that this approach - relying solely on textual information - achieves competitive performance: while it does not outperform state-of-the-art speech LLMs fine-tuned for the task, it surpasses a BERT-based model trained specifically for this purpose. NLA proves particularly effective in mismatched task settings, is generalisable to other data types and languages, and offers greater interpretability, as it is grounded in clearly explainable, widely applicable language descriptors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10177v1">Abusive text transformation using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      Although Large Language Models (LLMs) have demonstrated significant advancements in natural language processing tasks, their effectiveness in the classification and transformation of abusive text into non-abusive versions remains an area for exploration. In this study, we aim to use LLMs to transform abusive text (tweets and reviews) featuring hate speech and swear words into non-abusive text, while retaining the intent of the text. We evaluate the performance of two state-of-the-art LLMs, such as Gemini, GPT-4o, DeekSeek and Groq, on their ability to identify abusive text. We them to transform and obtain a text that is clean from abusive and inappropriate content but maintains a similar level of sentiment and semantics, i.e. the transformed text needs to maintain its message. Afterwards, we evaluate the raw and transformed datasets with sentiment analysis and semantic analysis. Our results show Groq provides vastly different results when compared with other LLMs. We have identified similarities between GPT-4o and DeepSeek-V3.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14403v3">Unearthing Gems from Stones: Policy Optimization with Negative Sample Augmentation for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      Recent advances in reasoning language models have witnessed a paradigm shift from short to long CoT pattern. Given the substantial computational cost of rollouts in long CoT models, maximizing the utility of fixed training datasets becomes crucial. Our analysis reveals that negative responses contain valuable components such as self-reflection and error-correction steps, yet primary existing methods either completely discard negative samples (RFT) or apply equal penalization across all tokens (RL), failing to leverage these potential learning signals. In light of this, we propose Behavior Constrained Policy Gradient with Negative Sample Augmentation (BCPG-NSA), a fine-grained offline RL framework that encompasses three stages: 1) sample segmentation, 2) consensus-based step correctness assessment combining LLM and PRM judgers, and 3) policy optimization with NSA designed to effectively mine positive steps within negative samples. Experimental results show that BCPG-NSA outperforms baselines on several challenging math/coding reasoning benchmarks using the same training dataset, achieving improved sample efficiency and demonstrating robustness and scalability when extended to multiple iterations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10155v1">Task-Based Flexible Feature Distillation for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      Knowledge Distillation (KD) in general and feature distillation in particular are promising techniques for reducing the high computational demand of large language models (LLMs). However, traditional feature KD methods typically assume that the teacher and the student share the same hidden size, limiting the flexibility of the student's architecture. A common solution to this problem involves training a linear projector to align their feature spaces, but this introduces additional parameters that must be learned from scratch and often degrades performance on downstream tasks, especially in generative settings. To address this issue, in this work, we propose a novel task-based feature distillation method that enables knowledge transfer between teacher and student models with different hidden layer dimensions, without introducing any new parameters. Leveraging the insight that only a subset of LLM components contribute significantly to a specific downstream task, our approach identifies the most task-relevant hidden units in the teacher and directly distills their activations to the student. Our method is flexible and easily integrates with other distillation frameworks. Empirical results show consistent improvements over prior approaches across diverse tasks, including classification, instruction-following, and summarization, achieving up to a 3\% performance gain over the linear projection baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10150v1">Past-Future Scheduler for LLM Serving under SLA Guarantees</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 Accepted to ASPLOS 2025
    </div>
    <details class="paper-abstract">
      The exploration and application of Large Language Models (LLMs) is thriving. To reduce deployment costs, continuous batching has become an essential feature in current service frameworks. The effectiveness of continuous batching relies on an accurate estimate of the memory requirements of requests. However, due to the diversity in request output lengths, existing frameworks tend to adopt aggressive or conservative schedulers, which often result in significant overestimation or underestimation of memory consumption. Consequently, they suffer from harmful request evictions or prolonged queuing times, failing to achieve satisfactory throughput under strict Service Level Agreement (SLA) guarantees (a.k.a. goodput), across various LLM application scenarios with differing input-output length distributions. To address this issue, we propose a novel Past-Future scheduler that precisely estimates the peak memory resources required by the running batch via considering the historical distribution of request output lengths and calculating memory occupancy at each future time point. It adapts to applications with all types of input-output length distributions, balancing the trade-off between request queuing and harmful evictions, thereby consistently achieving better goodput. Furthermore, to validate the effectiveness of the proposed scheduler, we developed a high-performance LLM serving framework, LightLLM, that implements the Past-Future scheduler. Compared to existing aggressive or conservative schedulers, LightLLM demonstrates superior goodput, achieving up to 2-3$\times$ higher goodput than other schedulers under heavy loads. LightLLM is open source to boost the research in such direction (https://github.com/ModelTC/lightllm).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10134v1">FRSICL: LLM-Enabled In-Context Learning Flight Resource Allocation for Fresh Data Collection in UAV-Assisted Wildfire Monitoring</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 8 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Unmanned Aerial Vehicles (UAVs) are vital for public safety, particularly in wildfire monitoring, where early detection minimizes environmental impact. In UAV-Assisted Wildfire Monitoring (UAWM) systems, joint optimization of sensor transmission scheduling and velocity is critical for minimizing Age of Information (AoI) from stale sensor data. Deep Reinforcement Learning (DRL) has been used for such optimization; however, its limitations such as low sampling efficiency, simulation-to-reality gaps, and complex training render it unsuitable for time-critical applications like wildfire monitoring. This paper introduces a new online Flight Resource Allocation scheme based on LLM-Enabled In-Context Learning (FRSICL) to jointly optimize the UAV's flight control and data collection schedule along the trajectory in real time, thereby asymptotically minimizing the average AoI across ground sensors. In contrast to DRL, FRSICL generates data collection schedules and controls velocity using natural language task descriptions and feedback from the environment, enabling dynamic decision-making without extensive retraining. Simulation results confirm the effectiveness of the proposed FRSICL compared to Proximal Policy Optimization (PPO) and Nearest-Neighbor baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10124v1">Could you be wrong: Debiasing LLMs using a metacognitive prompt for improving human decision making</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 12 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Identifying bias in LLMs is ongoing. Because they are still in development, what is true today may be false tomorrow. We therefore need general strategies for debiasing that will outlive current models. Strategies developed for debiasing human decision making offer one promising approach as they incorporate an LLM-style prompt intervention designed to bring latent knowledge into awareness during decision making. LLMs trained on vast amounts of information contain information about potential biases, counter-arguments, and contradictory evidence, but that information may only be brought to bear if prompted. Metacognitive prompts developed in the human decision making literature are designed to achieve this, and as I demonstrate here, they show promise with LLMs. The prompt I focus on here is "could you be wrong?" Following an LLM response, this prompt leads LLMs to produce additional information, including why they answered as they did, errors, biases, contradictory evidence, and alternatives, none of which were apparent in their initial response. Indeed, this metaknowledge often reveals that how LLMs and users interpret prompts are not aligned. Here I demonstrate this prompt using a set of questions taken from recent articles about LLM biases, including implicit discriminatory biases and failures of metacognition. "Could you be wrong" prompts the LLM to identify its own biases and produce cogent metacognitive reflection. I also present another example involving convincing but incomplete information, which is readily corrected by the metacognitive prompt. In sum, this work argues that human psychology offers a new avenue for prompt engineering, leveraging a long history of effective prompt-based improvements to human decision making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00200v2">Structuring Radiology Reports: Challenging LLMs with Lightweight Models</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      Radiology reports are critical for clinical decision-making but often lack a standardized format, limiting both human interpretability and machine learning (ML) applications. While large language models (LLMs) have shown strong capabilities in reformatting clinical text, their high computational requirements, lack of transparency, and data privacy concerns hinder practical deployment. To address these challenges, we explore lightweight encoder-decoder models (<300M parameters)-specifically T5 and BERT2BERT-for structuring radiology reports from the MIMIC-CXR and CheXpert Plus datasets. We benchmark these models against eight open-source LLMs (1B-70B), adapted using prefix prompting, in-context learning (ICL), and low-rank adaptation (LoRA) finetuning. Our best-performing lightweight model outperforms all LLMs adapted using prompt-based techniques on a human-annotated test set. While some LoRA-finetuned LLMs achieve modest gains over the lightweight model on the Findings section (BLEU 6.4%, ROUGE-L 4.8%, BERTScore 3.6%, F1-RadGraph 1.1%, GREEN 3.6%, and F1-SRR-BERT 4.3%), these improvements come at the cost of substantially greater computational resources. For example, LLaMA-3-70B incurred more than 400 times the inference time, cost, and carbon emissions compared to the lightweight model. These results underscore the potential of lightweight, task-specific models as sustainable and privacy-preserving solutions for structuring clinical text in resource-constrained healthcare settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10069v1">ElasticMM: Efficient Multimodal LLMs Serving with Elastic Multimodal Parallelism</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      Multimodal large language models (MLLMs) extend LLMs to handle images, videos, and audio by incorporating feature extractors and projection modules. However, these additional components -- combined with complex inference pipelines and heterogeneous workloads -- introduce significant inference overhead. Therefore, efficiently serving MLLMs remains a major challenge. Current tightly coupled serving architectures struggle to distinguish between mixed request types or adapt parallelism strategies to different inference stages, leading to increased time-to-first-token (TTFT) latency and poor resource utilization. To address this, we propose Elastic Multimodal Parallelism (EMP), a new serving paradigm that elastically adapts to resource heterogeneity across request types and inference stages. Building upon EMP, we develop ElasticMM, an MLLM serving system that (1) separates requests into independent modality groups with dynamic resource allocation via a modality-aware load balancer; (2) decouples inference stages and enables parallelism adjustment and adaptive scaling via elastic partition scheduling; and (3) improves inference efficiency through unified multimodal prefix caching and non-blocking encoding. Experiments on diverse real-world datasets show that ElasticMM outperforms state-of-the-art (SOTA) serving systems, reducing TTFT by up to 4.2x and achieving 3.2-4.5x higher throughput while meeting service-level objectives (SLOs).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10054v1">Explicit Vulnerability Generation with LLMs: An Investigation Beyond Adversarial Attacks</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 Accepted to ICSME 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used as code assistants, yet their behavior when explicitly asked to generate insecure code remains poorly understood. While prior research has focused on unintended vulnerabilities or adversarial prompting techniques, this study examines a more direct threat scenario: open-source LLMs generating vulnerable code when prompted either directly or indirectly. We propose a dual experimental design: (1) Dynamic Prompting, which systematically varies vulnerability type, user persona, and directness across structured templates; and (2) Reverse Prompting, which derives prompts from real vulnerable code samples to assess vulnerability reproduction accuracy. We evaluate three open-source 7B-parameter models (Qwen2, Mistral, and Gemma) using ESBMC static analysis to assess both the presence of vulnerabilities and the correctness of the generated vulnerability type. Results show all models frequently produce vulnerable outputs, with Qwen2 achieving highest correctness rates. User persona significantly affects success, where student personas achieved higher vulnerability rates than professional roles, while direct prompts were marginally more effective. Vulnerability reproduction followed an inverted-U pattern with cyclomatic complexity, peaking at moderate ranges. Our findings expose limitations of safety mechanisms in open-source models, particularly for seemingly benign educational requests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15902v2">IPAD: Inverse Prompt for AI Detection -- A Robust and Explainable LLM-Generated Text Detector</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have attained human-level fluency in text generation, which complicates the distinction between human-written and LLM-generated texts. This increases the risk of misuse and highlights the need for reliable detectors. Yet, existing detectors exhibit poor robustness on out-of-distribution (OOD) data and attacked data, which is critical for real-world scenarios. Also, they struggle to provide interpretable evidence to support their decisions, thus undermining the reliability. In light of these challenges, we propose IPAD (Inverse Prompt for AI Detection), a novel framework consisting of a Prompt Inverter that identifies predicted prompts that could have generated the input text, and two Distinguishers that examine the probability that the input texts align with the predicted prompts. Empirical evaluations demonstrate that IPAD outperforms the strongest baselines by 9.05% (Average Recall) on in-distribution data, 12.93% (AUROC) on out-of-distribution (OOD) data, and 5.48% (AUROC) on attacked data. IPAD also performs robustly on structured datasets. Furthermore, an interpretability assessment is conducted to illustrate that IPAD enhances the AI detection trustworthiness by allowing users to directly examine the decision-making evidence, which provides interpretable support for its state-of-the-art detection results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10024v1">Qualitative Study for LLM-assisted Design Study Process: Strategies, Challenges, and Roles</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      Design studies aim to create visualization solutions for real-world problems of different application domains. Recently, the emergence of large language models (LLMs) has introduced new opportunities to enhance the design study process, providing capabilities such as creative problem-solving, data handling, and insightful analysis. However, despite their growing popularity, there remains a lack of systematic understanding of how LLMs can effectively assist researchers in visualization-specific design studies. In this paper, we conducted a multi-stage qualitative study to fill this gap, involving 30 design study researchers from diverse backgrounds and expertise levels. Through in-depth interviews and carefully-designed questionnaires, we investigated strategies for utilizing LLMs, the challenges encountered, and the practices used to overcome them. We further compiled and summarized the roles that LLMs can play across different stages of the design study process. Our findings highlight practical implications to inform visualization practitioners, and provide a framework for leveraging LLMs to enhance the design study process in visualization research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07498v2">Teaching LLM to Reason: Reinforcement Learning from Algorithmic Problems without Code</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      Enhancing reasoning capabilities remains a central focus in the LLM reasearch community. A promising direction involves requiring models to simulate code execution step-by-step to derive outputs for given inputs. However, as code is often designed for large-scale systems, direct application leads to over-reliance on complex data structures and algorithms, even for simple cases, resulting in overfitting to algorithmic patterns rather than core reasoning structures. To address this, we propose TeaR, which aims at teaching LLMs to reason better. TeaR leverages careful data curation and reinforcement learning to guide models in discovering optimal reasoning paths through code-related tasks, thereby improving general reasoning abilities. We conduct extensive experiments using two base models and three long-CoT distillation models, with model sizes ranging from 1.5 billion to 32 billion parameters, and across 17 benchmarks spanning Math, Knowledge, Code, and Logical Reasoning. The results consistently show significant performance improvements. Notably, TeaR achieves a 35.9% improvement on Qwen2.5-7B and 5.9% on R1-Distilled-7B.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12185v3">EVALOOP: Assessing LLM Robustness in Programming from a Self-consistency Perspective</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 20 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Assessing the programming capabilities of Large Language Models (LLMs) is crucial for their effective use in software engineering. Current evaluations, however, predominantly measure the accuracy of generated code on static benchmarks, neglecting the critical aspect of model robustness during programming tasks. While adversarial attacks offer insights on model robustness, their effectiveness is limited and evaluation could be constrained. Current adversarial attack methods for robustness evaluation yield inconsistent results, struggling to provide a unified evaluation across different LLMs. We introduce EVALOOP, a novel assessment framework that evaluate the robustness from a self-consistency perspective, i.e., leveraging the natural duality inherent in popular software engineering tasks, e.g., code generation and code summarization. EVALOOP initiates a self-contained feedback loop: an LLM generates output (e.g., code) from an input (e.g., natural language specification), and then use the generated output as the input to produce a new output (e.g., summarizes that code into a new specification). EVALOOP repeats the process to assess the effectiveness of EVALOOP in each loop. This cyclical strategy intrinsically evaluates robustness without rely on any external attack setups, providing a unified metric to evaluate LLMs' robustness in programming. We evaluate 16 prominent LLMs (e.g., GPT-4.1, O4-mini) on EVALOOP and found that EVALOOP typically induces a 5.01%-19.31% absolute drop in pass@1 performance within ten loops. Intriguingly, robustness does not always align with initial performance (i.e., one-time query); for instance, GPT-3.5-Turbo, despite superior initial code generation compared to DeepSeek-V2, demonstrated lower robustness over repeated evaluation loop.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13640v2">Qorgau: Evaluating LLM Safety in Kazakh-Russian Bilingual Contexts</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are known to have the potential to generate harmful content, posing risks to users. While significant progress has been made in developing taxonomies for LLM risks and safety evaluation prompts, most studies have focused on monolingual contexts, primarily in English. However, language- and region-specific risks in bilingual contexts are often overlooked, and core findings can diverge from those in monolingual settings. In this paper, we introduce Qorgau, a novel dataset specifically designed for safety evaluation in Kazakh and Russian, reflecting the unique bilingual context in Kazakhstan, where both Kazakh (a low-resource language) and Russian (a high-resource language) are spoken. Experiments with both multilingual and language-specific LLMs reveal notable differences in safety performance, emphasizing the need for tailored, region-specific datasets to ensure the responsible and safe deployment of LLMs in countries like Kazakhstan. Warning: this paper contains example data that may be offensive, harmful, or biased.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13820v2">InstCache: A Predictive Cache for LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      The revolutionary capabilities of Large Language Models (LLMs) are attracting rapidly growing popularity and leading to soaring user requests to inference serving systems. Caching techniques, which leverage data reuse to reduce computation, offer opportunities to optimize the performance of LLM inference engines. On the one hand, the low-level key-value (KV) cache working at the token level is widely adopted, albeit it incurs significant overhead as request volume grows. On the other hand, instruction-level caching, which stores full instruction-response pairs, is expected to play an increasingly crucial role. However, the high variability in the content and length of instructions make it rare for identical instructions to recur within a short time window, presenting challenges for effective caching instruction-response pairs. To address this challenge, we propose InstCache, a predictive caching mechanism for LLM serving systems. Leveraging the capability of LLMs, we can effectively reorder the representation space of instruction texts and develop a sufficient level of spatial locality. Such spatial locality enables us to predict potential instructions located in a compact region in the space, resulting in an effective caching system at runtime. Experimental results demonstrate that InstCache achieves a 2.3x higher hit rate compared to the upper bound of traditional caching mechanisms on WildChat dataset and reduces the time per output token of vLLM by up to 42.0% and 50.0% on LMSys and Moss datasets, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09839v1">Rethinking Prompt Optimization: Reinforcement, Diversification, and Migration in Blackbox LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      An increasing number of NLP applications interact with large language models (LLMs) through black-box APIs, making prompt engineering critical for controlling model outputs. While recent Automatic Prompt Optimization (APO) methods iteratively refine prompts using model-generated feedback, textual gradients, they primarily focus on error correction and neglect valuable insights from correct predictions. This limits both their effectiveness and efficiency. In this paper, we propose a novel APO framework centered on enhancing the feedback mechanism. We reinterpret the textual gradient as a form of negative reinforcement and introduce the complementary positive reinforcement to explicitly preserve beneficial prompt components identified through successful predictions. To mitigate the noise inherent in LLM-generated feedback, we introduce a technique called feedback diversification, which aggregates multiple feedback signals, emphasizing consistent, actionable advice while filtering out outliers. Motivated by the rapid evolution and diversity of available LLMs, we also formalize Continual Prompt Optimization (CPO), addressing the practical challenge of efficiently migrating optimized prompts between different model versions or API providers. Our experiments reveal that naive prompt migration often degrades performance due to loss of critical instructions. In contrast, our approach consistently outperforms strong baselines, achieving significant accuracy improvements, faster convergence, and lower computational costs in both standard and migration scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10852v1">LLMs on Trial: Evaluating Judicial Fairness for Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used in high-stakes fields where their decisions impact rights and equity. However, LLMs' judicial fairness and implications for social justice remain underexplored. When LLMs act as judges, the ability to fairly resolve judicial issues is a prerequisite to ensure their trustworthiness. Based on theories of judicial fairness, we construct a comprehensive framework to measure LLM fairness, leading to a selection of 65 labels and 161 corresponding values. Applying this framework to the judicial system, we compile an extensive dataset, JudiFair, comprising 177,100 unique case facts. To achieve robust statistical inference, we develop three evaluation metrics, inconsistency, bias, and imbalanced inaccuracy, and introduce a method to assess the overall fairness of multiple LLMs across various labels. Through experiments with 16 LLMs, we uncover pervasive inconsistency, bias, and imbalanced inaccuracy across models, underscoring severe LLM judicial unfairness. Particularly, LLMs display notably more pronounced biases on demographic labels, with slightly less bias on substance labels compared to procedure ones. Interestingly, increased inconsistency correlates with reduced biases, but more accurate predictions exacerbate biases. While we find that adjusting the temperature parameter can influence LLM fairness, model size, release date, and country of origin do not exhibit significant effects on judicial fairness. Accordingly, we introduce a publicly available toolkit containing all datasets and code, designed to support future research in evaluating and improving LLM fairness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10844v1">LLM-Guided Agentic Object Detection for Open-World Understanding</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      Object detection traditionally relies on fixed category sets, requiring costly re-training to handle novel objects. While Open-World and Open-Vocabulary Object Detection (OWOD and OVOD) improve flexibility, OWOD lacks semantic labels for unknowns, and OVOD depends on user prompts, limiting autonomy. We propose an LLM-guided agentic object detection (LAOD) framework that enables fully label-free, zero-shot detection by prompting a Large Language Model (LLM) to generate scene-specific object names. These are passed to an open-vocabulary detector for localization, allowing the system to adapt its goals dynamically. We introduce two new metrics, Class-Agnostic Average Precision (CAAP) and Semantic Naming Average Precision (SNAP), to separately evaluate localization and naming. Experiments on LVIS, COCO, and COCO-OOD validate our approach, showing strong performance in detecting and naming novel objects. Our method offers enhanced autonomy and adaptability for open-world understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10818v1">How Robust are LLM-Generated Library Imports? An Empirical Study using Stack Overflow</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      Software libraries are central to the functionality, security, and maintainability of modern code. As developers increasingly turn to Large Language Models (LLMs) to assist with programming tasks, understanding how these models recommend libraries is essential. In this paper, we conduct an empirical study of six state-of-the-art LLMs, both proprietary and open-source, by prompting them to solve real-world Python problems sourced from Stack Overflow. We analyze the types of libraries they import, the characteristics of those libraries, and the extent to which the recommendations are usable out of the box. Our results show that LLMs predominantly favour third-party libraries over standard ones, and often recommend mature, popular, and permissively licensed dependencies. However, we also identify gaps in usability: 4.6% of the libraries could not be resolved automatically due to structural mismatches between import names and installable packages, and only two models (out of six) provided installation guidance. While the generated code is technically valid, the lack of contextual support places the burden of manually resolving dependencies on the user. Our findings offer actionable insights for both developers and researchers, and highlight opportunities to improve the reliability and usability of LLM-generated code in the context of software dependencies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10803v1">Automated Thematic Analyses Using LLMs: Xylazine Wound Management Social Media Chatter Use Case</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 Pages: 19, Abstract word count: 151 words, Manuscript word count: 2185 words, References: 14, Figures: 3, Tables: 2
    </div>
    <details class="paper-abstract">
      Background Large language models (LLMs) face challenges in inductive thematic analysis, a task requiring deep interpretive and domain-specific expertise. We evaluated the feasibility of using LLMs to replicate expert-driven thematic analysis of social media data. Methods Using two temporally non-intersecting Reddit datasets on xylazine (n=286 and n=686, for model optimization and validation, respectively) with twelve expert-derived themes, we evaluated five LLMs against expert coding. We modeled the task as a series of binary classifications, rather than a single, multi-label classification, employing zero-, single-, and few-shot prompting strategies and measuring performance via accuracy, precision, recall, and F1-score. Results On the validation set, GPT-4o with two-shot prompting performed best (accuracy: 90.9%; F1-score: 0.71). For high-prevalence themes, model-derived thematic distributions closely mirrored expert classifications (e.g., xylazine use: 13.6% vs. 17.8%; MOUD use: 16.5% vs. 17.8%). Conclusions Our findings suggest that few-shot LLM-based approaches can automate thematic analyses, offering a scalable supplement for qualitative research. Keywords: thematic analysis, large language models, natural language processing, qualitative analysis, social media, prompt engineering, public health
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04644v2">Agentic Reasoning: A Streamlined Framework for Enhancing LLM Reasoning with Agentic Tools</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 ACL 2025
    </div>
    <details class="paper-abstract">
      We introduce Agentic Reasoning, a framework that enhances large language model (LLM) reasoning by integrating external tool-using agents. Agentic Reasoning dynamically leverages web search, code execution, and structured memory to address complex problems requiring deep research. A key innovation in our framework is the Mind-Map agent, which constructs a structured knowledge graph to store reasoning context and track logical relationships, ensuring coherence in long reasoning chains with extensive tool usage. Additionally, we conduct a comprehensive exploration of the Web-Search agent, leading to a highly effective search mechanism that surpasses all prior approaches. When deployed on DeepSeek-R1, our method achieves a new state-of-the-art (SOTA) among public models and delivers performance comparable to OpenAI Deep Research, the leading proprietary model in this domain. Extensive ablation studies validate the optimal selection of agentic tools and confirm the effectiveness of our Mind-Map and Web-Search agents in enhancing LLM reasoning. The code is at: https://github.com/theworldofagents/Agentic-Reasoning
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10778v1">Warehouse Spatial Question Answering with LLM Agent</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 1st Place Solution of the 9th AI City Challenge Track 3
    </div>
    <details class="paper-abstract">
      Spatial understanding has been a challenging task for existing Multi-modal Large Language Models~(MLLMs). Previous methods leverage large-scale MLLM finetuning to enhance MLLM's spatial understanding ability. In this paper, we present a data-efficient approach. We propose a LLM agent system with strong and advanced spatial reasoning ability, which can be used to solve the challenging spatial question answering task in complex indoor warehouse scenarios. Our system integrates multiple tools that allow the LLM agent to conduct spatial reasoning and API tools interaction to answer the given complicated spatial question. Extensive evaluations on the 2025 AI City Challenge Physical AI Spatial Intelligence Warehouse dataset demonstrate that our system achieves high accuracy and efficiency in tasks such as object retrieval, counting, and distance estimation. The code is available at: https://github.com/hsiangwei0903/SpatialAgent
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05209v3">Model Tampering Attacks Enable More Rigorous Evaluations of LLM Capabilities</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 Accepted to TMLR
    </div>
    <details class="paper-abstract">
      Evaluations of large language model (LLM) risks and capabilities are increasingly being incorporated into AI risk management and governance frameworks. Currently, most risk evaluations are conducted by designing inputs that elicit harmful behaviors from the system. However, this approach suffers from two limitations. First, input-output evaluations cannot fully evaluate realistic risks from open-weight models. Second, the behaviors identified during any particular input-output evaluation can only lower-bound the model's worst-possible-case input-output behavior. As a complementary method for eliciting harmful behaviors, we propose evaluating LLMs with model tampering attacks which allow for modifications to latent activations or weights. We pit state-of-the-art techniques for removing harmful LLM capabilities against a suite of 5 input-space and 6 model tampering attacks. In addition to benchmarking these methods against each other, we show that (1) model resilience to capability elicitation attacks lies on a low-dimensional robustness subspace; (2) the success rate of model tampering attacks can empirically predict and offer conservative estimates for the success of held-out input-space attacks; and (3) state-of-the-art unlearning methods can easily be undone within 16 steps of fine-tuning. Together, these results highlight the difficulty of suppressing harmful LLM capabilities and show that model tampering attacks enable substantially more rigorous evaluations than input-space attacks alone.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.02820v4">DroidSpeak: KV Cache Sharing for Cross-LLM Communication and Multi-LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      Compound AI systems, such as agentic systems, are an emerging trend in large-scale enterprise settings, with multiple LLMs specialized for different users, tasks, and/or roles working together. In these scenarios, different models often process inputs that share the same context prefix. Although much work was done in the past to enable the reuse of prefix KV caches across inputs for a single model, how to enable one model to reuse the prefix KV caches of a different model remains an open question. We introduce DroidSpeak, the first distributed LLM inference system that enables KV cache reuse across distributed nodes running inference of different LLMs, so long as the LLMs have the same architecture. We present the first study that aims at understanding the impact of sharing KV caches across different LLMs, and if/when such sharing affects quality. Inspired by the findings, we present DroidSpeak, which selectively recomputes a few layers of the KV cache produced by another LLM and reuses the remaining layers, with negligible quality loss. Moreover, carefully pipelining the layer-wise re-computation and the loading of reused KV cache further improves the inference performance. Experiments on diverse datasets and model pairs demonstrate that DroidSpeak achieves up to 4x throughput improvement and about 3.1x faster prefill (time to first token), with negligible loss of quality in F1 scores, Rouge-L or code similarity score, compared to the baseline which does not allow any sharing across models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10695v1">Exploring User Security and Privacy Attitudes and Concerns Toward the Use of General-Purpose LLM Chatbots for Mental Health</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 Accepted to the 34th USENIX Security Symposium
    </div>
    <details class="paper-abstract">
      Individuals are increasingly relying on large language model (LLM)-enabled conversational agents for emotional support. While prior research has examined privacy and security issues in chatbots specifically designed for mental health purposes, these chatbots are overwhelmingly "rule-based" offerings that do not leverage generative AI. Little empirical research currently measures users' privacy and security concerns, attitudes, and expectations when using general-purpose LLM-enabled chatbots to manage and improve mental health. Through 21 semi-structured interviews with U.S. participants, we identified critical misconceptions and a general lack of risk awareness. Participants conflated the human-like empathy exhibited by LLMs with human-like accountability and mistakenly believed that their interactions with these chatbots were safeguarded by the same regulations (e.g., HIPAA) as disclosures with a licensed therapist. We introduce the concept of "intangible vulnerability," where emotional or psychological disclosures are undervalued compared to more tangible forms of information (e.g., financial or location-based data). To address this, we propose recommendations to safeguard user mental health disclosures with general-purpose LLM-enabled chatbots more effectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10639v1">SPICEAssistant: LLM using SPICE Simulation Tools for Schematic Design of Switched-Mode Power Supplies</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 11 pages, 10 figures
    </div>
    <details class="paper-abstract">
      State-of-the-art large language models (LLMs) show high performance across a wide range of tasks in many domains of science. In the field of electronic design automation (EDA), it is yet to be determined to what extent they are capable to understand, adapt, and dimension electronic circuits. This paper focuses on the application of LLMs to switched-mode power supply (SMPS) design on printed circuit boards (PCBs). Particular challenges for LLMs in this context include their limited ability to interpret results from key simulation tools like SPICE and the multi-step design process. To address these challenges, we suggest SPICEAssistant, a framework that provides a broad selection of tools to an LLM. The tools serve as an interface to SPICE, allowing the LLM to interact flexibly with the simulator to estimate the impact of its modifications to the circuit. To evaluate the performance of SPICEAssistant, we defined a benchmark consisting of 256 questions testing the ability to adapt circuit netlists to fulfil different SMPS design tasks. The benchmarking results show that simulation feedback effectively improves SMPS design capabilities of LLMs. An increasing number of simulation iterations leads to enhanced performance. The SPICEAssistant framework significantly outperforms the standalone LLM GPT-4o on the benchmark by approximately 38%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10628v1">GHPO: Adaptive Guidance for Stable and Efficient LLM Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Rewards (RLVR) has recently emerged as a powerful paradigm for facilitating the self-improvement of large language models (LLMs), particularly in the domain of complex reasoning tasks. However, prevailing on-policy RL methods often contend with significant training instability and inefficiency. This is primarily due to a capacity-difficulty mismatch, where the complexity of training data frequently outpaces the model's current capabilities, leading to critically sparse reward signals and stalled learning progress. This challenge is particularly acute for smaller, more resource-efficient LLMs. To overcome this, we introduce the Guided Hybrid Policy Optimization (GHPO), a novel difficulty-aware reinforcement learning framework. GHPO dynamically calibrates task difficulty by employing adaptive prompt refinement to provide targeted guidance. This unique approach adaptively balances direct imitation learning for problems currently beyond the model's reach with exploration-based reinforcement learning for more manageable tasks, effectively creating a smooth and optimized learning curriculum. Extensive experiments demonstrate that GHPO achieves an average performance gain of approximately 5% across six challenging mathematics benchmarks, consistently outperforming strong on-policy reinforcement learning and curriculum learning baselines. Further analysis confirms that our framework significantly enhances both training stability and final reasoning performance, thus offering a scalable and efficient solution for developing powerful and robust reasoning models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10624v1">Comprehension Without Competence: Architectural Limits of LLMs in Symbolic Computation and Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 Substantial change to previous version (experiments, theorem, analysis and related work); currently under review at TMLR
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) display striking surface fluency yet systematically fail at tasks requiring symbolic reasoning, arithmetic accuracy, and logical consistency. This paper offers a structural diagnosis of such failures, revealing a persistent gap between \textit{comprehension} and \textit{competence}. Through controlled experiments and architectural analysis, we demonstrate that LLMs often articulate correct principles without reliably applying them--a failure rooted not in knowledge access, but in computational execution. We term this phenomenon the computational \textit{split-brain syndrome}, where instruction and action pathways are geometrically and functionally dissociated. This core limitation recurs across domains, from mathematical operations to relational inferences, and explains why model behavior remains brittle even under idealized prompting. We argue that LLMs function as powerful pattern completion engines, but lack the architectural scaffolding for principled, compositional reasoning. Our findings delineate the boundary of current LLM capabilities and motivate future models with metacognitive control, principle lifting, and structurally grounded execution. This diagnosis also clarifies why mechanistic interpretability findings may reflect training-specific pattern coordination rather than universal computational principles, and why the geometric separation between instruction and execution pathways suggests limitations in neural introspection and mechanistic analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10621v1">Game Theory Meets LLM and Agentic AI: Reimagining Cybersecurity for the Age of Intelligent Threats</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      Protecting cyberspace requires not only advanced tools but also a shift in how we reason about threats, trust, and autonomy. Traditional cybersecurity methods rely on manual responses and brittle heuristics. To build proactive and intelligent defense systems, we need integrated theoretical frameworks and software tools. Game theory provides a rigorous foundation for modeling adversarial behavior, designing strategic defenses, and enabling trust in autonomous systems. Meanwhile, software tools process cyber data, visualize attack surfaces, verify compliance, and suggest mitigations. Yet a disconnect remains between theory and practical implementation. The rise of Large Language Models (LLMs) and agentic AI offers a new path to bridge this gap. LLM-powered agents can operationalize abstract strategies into real-world decisions. Conversely, game theory can inform the reasoning and coordination of these agents across complex workflows. LLMs also challenge classical game-theoretic assumptions, such as perfect rationality or static payoffs, prompting new models aligned with cognitive and computational realities. This co-evolution promises richer theoretical foundations and novel solution concepts. Agentic AI also reshapes software design: systems must now be modular, adaptive, and trust-aware from the outset. This chapter explores the intersection of game theory, agentic AI, and cybersecurity. We review key game-theoretic frameworks (e.g., static, dynamic, Bayesian, and signaling games) and solution concepts. We then examine how LLM agents can enhance cyber defense and introduce LLM-driven games that embed reasoning into AI agents. Finally, we explore multi-agent workflows and coordination games, outlining how this convergence fosters secure, intelligent, and adaptive cyber systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09820v1">Measuring What Matters: A Framework for Evaluating Safety Risks in Real-World LLM Applications</a></div>
    <div class="paper-meta">
      📅 2025-07-13
    </div>
    <details class="paper-abstract">
      Most safety testing efforts for large language models (LLMs) today focus on evaluating foundation models. However, there is a growing need to evaluate safety at the application level, as components such as system prompts, retrieval pipelines, and guardrails introduce additional factors that significantly influence the overall safety of LLM applications. In this paper, we introduce a practical framework for evaluating application-level safety in LLM systems, validated through real-world deployment across multiple use cases within our organization. The framework consists of two parts: (1) principles for developing customized safety risk taxonomies, and (2) practices for evaluating safety risks in LLM applications. We illustrate how the proposed framework was applied in our internal pilot, providing a reference point for organizations seeking to scale their safety testing efforts. This work aims to bridge the gap between theoretical concepts in AI safety and the operational realities of safeguarding LLM applications in practice, offering actionable guidance for safe and scalable deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09790v1">Prompting for Performance: Exploring LLMs for Configuring Software</a></div>
    <div class="paper-meta">
      📅 2025-07-13
    </div>
    <details class="paper-abstract">
      Software systems usually provide numerous configuration options that can affect performance metrics such as execution time, memory usage, binary size, or bitrate. On the one hand, making informed decisions is challenging and requires domain expertise in options and their combinations. On the other hand, machine learning techniques can search vast configuration spaces, but with a high computational cost, since concrete executions of numerous configurations are required. In this exploratory study, we investigate whether large language models (LLMs) can assist in performance-oriented software configuration through prompts. We evaluate several LLMs on tasks including identifying relevant options, ranking configurations, and recommending performant configurations across various configurable systems, such as compilers, video encoders, and SAT solvers. Our preliminary results reveal both positive abilities and notable limitations: depending on the task and systems, LLMs can well align with expert knowledge, whereas hallucinations or superficial reasoning can emerge in other cases. These findings represent a first step toward systematic evaluations and the design of LLM-based solutions to assist with software configuration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09788v1">TinyTroupe: An LLM-powered Multiagent Persona Simulation Toolkit</a></div>
    <div class="paper-meta">
      📅 2025-07-13
      | 💬 9 pages. Preprint to be submitted to peer-review
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLM) have led to a new class of autonomous agents, renewing and expanding interest in the area. LLM-powered Multiagent Systems (MAS) have thus emerged, both for assistive and simulation purposes, yet tools for realistic human behavior simulation -- with its distinctive challenges and opportunities -- remain underdeveloped. Existing MAS libraries and tools lack fine-grained persona specifications, population sampling facilities, experimentation support, and integrated validation, among other key capabilities, limiting their utility for behavioral studies, social simulation, and related applications. To address these deficiencies, in this work we introduce TinyTroupe, a simulation toolkit enabling detailed persona definitions (e.g., nationality, age, occupation, personality, beliefs, behaviors) and programmatic control via numerous LLM-driven mechanisms. This allows for the concise formulation of behavioral problems of practical interest, either at the individual or group level, and provides effective means for their solution. TinyTroupe's components are presented using representative working examples, such as brainstorming and market research sessions, thereby simultaneously clarifying their purpose and demonstrating their usefulness. Quantitative and qualitative evaluations of selected aspects are also provided, highlighting possibilities, limitations, and trade-offs. The approach, though realized as a specific Python implementation, is meant as a novel conceptual contribution, which can be partially or fully incorporated in other contexts. The library is available as open source at https://github.com/microsoft/tinytroupe.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09397v4">SLED: A Speculative LLM Decoding Framework for Efficient Edge Serving</a></div>
    <div class="paper-meta">
      📅 2025-07-13
      | 💬 6 pages, 6 figures, 2 tables
    </div>
    <details class="paper-abstract">
      The growing gap between the increasing complexity of large language models (LLMs) and the limited computational budgets of edge devices poses a key challenge for efficient on-device inference, despite gradual improvements in hardware capabilities. Existing strategies, such as aggressive quantization, pruning, or remote inference, trade accuracy for efficiency or lead to substantial cost burdens. This position paper introduces a new framework that leverages speculative decoding, previously viewed primarily as a decoding acceleration technique for autoregressive generation of LLMs, as a promising approach specifically adapted for edge computing by orchestrating computation across heterogeneous devices. We propose \acronym, a framework that allows lightweight edge devices to draft multiple candidate tokens locally using diverse draft models, while a single, shared edge server verifies the tokens utilizing a more precise target model. To further increase the efficiency of verification, the edge server batch the diverse verification requests from devices. This approach supports device heterogeneity and reduces server-side memory footprint by sharing the same upstream target model across multiple devices. Our initial experiments with Jetson Orin Nano, Raspberry Pi 4B/5, and an edge server equipped with 4 Nvidia A100 GPUs indicate substantial benefits: 2.2 more system throughput, 2.8 more system capacity, and better cost efficiency, all without sacrificing model accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.11462v5">Cascade Speculative Drafting for Even Faster LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-07-13
      | 💬 NeurIPS 2024
    </div>
    <details class="paper-abstract">
      Introduced to enhance the efficiency of large language model (LLM) inference, speculative decoding operates by having a smaller model generate a draft. A larger target model then reviews this draft to align with its output, and any acceptance by the target model results in a reduction of the number of the target model runs, ultimately improving efficiency. However, the drafting process in speculative decoding includes slow autoregressive generation and allocates equal time to generating tokens, irrespective of their importance. These inefficiencies collectively contribute to the suboptimal performance of speculative decoding. To further improve LLM inference, we introduce Cascade Speculative Drafting (CS Drafting), a speculative execution algorithm that incorporates two types of cascades. The Vertical Cascade eliminates autoregressive generation from neural models, while the Horizontal Cascade optimizes time allocation in drafting for improved efficiency. Combining both cascades, CS Drafting achieves greater speedup compared to the baselines in our experiments, while preserving the same output distribution as the target model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09751v1">Sound and Complete Neuro-symbolic Reasoning with LLM-Grounded Interpretations</a></div>
    <div class="paper-meta">
      📅 2025-07-13
      | 💬 29 pages, 9 tables, 3 figures. Accepted to the 19th Conference on Neurosymbolic Learning and Reasoning (NeSy 2025)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive capabilities in natural language understanding and generation, but they exhibit problems with logical consistency in the output they generate. How can we harness LLMs' broad-coverage parametric knowledge in formal reasoning despite their inconsistency? We present a method for directly integrating an LLM into the interpretation function of the formal semantics for a paraconsistent logic. We provide experimental evidence for the feasibility of the method by evaluating the function using datasets created from several short-form factuality benchmarks. Unlike prior work, our method offers a theoretical framework for neuro-symbolic reasoning that leverages an LLM's knowledge while preserving the underlying logic's soundness and completeness properties.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09701v1">MCEval: A Dynamic Framework for Fair Multilingual Cultural Evaluation of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-13
    </div>
    <details class="paper-abstract">
      Large language models exhibit cultural biases and limited cross-cultural understanding capabilities, particularly when serving diverse global user populations. We propose MCEval, a novel multilingual evaluation framework that employs dynamic cultural question construction and enables causal analysis through Counterfactual Rephrasing and Confounder Rephrasing. Our comprehensive evaluation spans 13 cultures and 13 languages, systematically assessing both cultural awareness and cultural bias across different linguistic scenarios. The framework provides 39,897 cultural awareness instances and 17,940 cultural bias instances. Experimental results reveal performance disparities across different linguistic scenarios, demonstrating that optimal cultural performance is not only linked to training data distribution, but also is related to language-culture alignment. The evaluation results also expose the fairness issue, where approaches appearing successful in the English scenario create substantial disadvantages. MCEval represents the first comprehensive multilingual cultural evaluation framework that provides deeper insights into LLMs' cultural understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09657v1">Negotiating Comfort: Simulating Personality-Driven LLM Agents in Shared Residential Social Networks</a></div>
    <div class="paper-meta">
      📅 2025-07-13
    </div>
    <details class="paper-abstract">
      We use generative agents powered by large language models (LLMs) to simulate a social network in a shared residential building, driving the temperature decisions for a central heating system. Agents, divided into Family Members and Representatives, consider personal preferences, personal traits, connections, and weather conditions. Daily simulations involve family-level consensus followed by building-wide decisions among representatives. We tested three personality traits distributions (positive, mixed, and negative) and found that positive traits correlate with higher happiness and stronger friendships. Temperature preferences, assertiveness, and selflessness have a significant impact on happiness and decisions. This work demonstrates how LLM-driven agents can help simulate nuanced human behavior where complex real-life human simulations are difficult to set.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08965v2">LLM-Driven Usefulness Labeling for IR Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-07-13
    </div>
    <details class="paper-abstract">
      In the information retrieval (IR) domain, evaluation plays a crucial role in optimizing search experiences and supporting diverse user intents. In the recent LLM era, research has been conducted to automate document relevance labels, as these labels have traditionally been assigned by crowd-sourced workers - a process that is both time and consuming and costly. This study focuses on LLM-generated usefulness labels, a crucial evaluation metric that considers the user's search intents and task objectives, an aspect where relevance falls short. Our experiment utilizes task-level, query-level, and document-level features along with user search behavior signals, which are essential in defining the usefulness of a document. Our research finds that (i) pre-trained LLMs can generate moderate usefulness labels by understanding the comprehensive search task session, (ii) pre-trained LLMs perform better judgement in short search sessions when provided with search session contexts. Additionally, we investigated whether LLMs can capture the unique divergence between relevance and usefulness, along with conducting an ablation study to identify the most critical metrics for accurate usefulness label generation. In conclusion, this work explores LLM-generated usefulness labels by evaluating critical metrics and optimizing for practicality in real-world settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05310v2">Oracular Programming: A Modular Foundation for Building LLM-Enabled Software</a></div>
    <div class="paper-meta">
      📅 2025-07-13
    </div>
    <details class="paper-abstract">
      Large Language Models have proven surprisingly effective at solving a wide range of tasks from just a handful of examples. However, their lack of reliability and modularity limits their capacity to tackle large problems that require many steps of reasoning. In response, researchers have proposed advanced pipelines that leverage domain-specific knowledge to chain smaller prompts, provide intermediate feedback and improve performance through search. However, the current complexity of writing, tuning, maintaining and improving such pipelines has limited their sophistication. We propose oracular programming, a foundational paradigm for building LLM-enabled applications that lets domain experts express high-level problem-solving strategies as programs with unresolved choice points. These choice points are resolved at runtime by LLMs, which generalize from user-provided examples of correct and incorrect decisions. An oracular program is composed of three orthogonal components: a strategy that consists in a nondeterministic program with choice points that can be reified into a search tree, a policy that specifies how to navigate this tree with the help of LLM oracles, and a set of demonstrations that describe successful and unsuccessful search tree navigation scenarios across diverse problem instances. Each component is expressed in a dedicated programming language and can be independently improved or substituted. We address the key programming language design challenges of modularly composing oracular programs and enforcing consistency between their components as they evolve.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09490v1">Towards LLM-Based Automatic Playtest</a></div>
    <div class="paper-meta">
      📅 2025-07-13
    </div>
    <details class="paper-abstract">
      Playtesting is the process in which people play a video game for testing. It is critical for the quality assurance of gaming software. Manual playtesting is time-consuming and expensive. However, automating this process is challenging, as playtesting typically requires domain knowledge and problem-solving skills that most conventional testing tools lack. Recent advancements in artificial intelligence (AI) have opened up new possibilities for applying Large Language Models (LLMs) to playtesting. However, significant challenges remain: current LLMs cannot visually perceive game environments, and most existing research focuses on text-based games or games with robust APIs. Many non-text games lack APIs to provide textual descriptions of game states, making it almost impossible to naively apply LLMs for playtesting. This paper introduces Lap, our novel approach to LLM-based Automatic Playtesting, which uses ChatGPT to test match-3 games, a category of games where players match three or more identical tiles in a row or column to earn points. Lap encompasses three key phases: processing of game environments, prompting-based action generation, and action execution. Given a match-3 game, Lap takes a snapshot of the game board and converts it to a numeric matrix. It then prompts the ChatGPT-O1-mini API to suggest moves based on that matrix and tentatively applies the suggested moves to earn points and trigger changes in the game board. It repeats the above-mentioned three steps iteratively until timeout. For evaluation, we conducted a case study using Lap on an open-source match-3 game, CasseBonbons, and empirically compared it with three existing tools. Our results are promising: Lap outperformed existing tools by achieving higher code coverage and triggering more program crashes. This research sheds light on the future of automatic testing and LLM applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09488v1">Criteria-Based LLM Relevance Judgments</a></div>
    <div class="paper-meta">
      📅 2025-07-13
      | 💬 10 pages, 3 figures, accepted to ICTIR 2025
    </div>
    <details class="paper-abstract">
      Relevance judgments are crucial for evaluating information retrieval systems, but traditional human-annotated labels are time-consuming and expensive. As a result, many researchers turn to automatic alternatives to accelerate method development. Among these, Large Language Models (LLMs) provide a scalable solution by generating relevance labels directly through prompting. However, prompting an LLM for a relevance label without constraints often results in not only incorrect predictions but also outputs that are difficult for humans to interpret. We propose the Multi-Criteria framework for LLM-based relevance judgments, decomposing the notion of relevance into multiple criteria--such as exactness, coverage, topicality, and contextual fit--to improve the robustness and interpretability of retrieval evaluations compared to direct grading methods. We validate this approach on three datasets: the TREC Deep Learning tracks from 2019 and 2020, as well as LLMJudge (based on TREC DL 2023). Our results demonstrate that Multi-Criteria judgments enhance the system ranking/leaderboard performance. Moreover, we highlight the strengths and limitations of this approach relative to direct grading approaches, offering insights that can guide the development of future automatic evaluation frameworks in information retrieval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09483v1">Does UMBRELA Work on Other LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-07-13
      | 💬 9 pages, 2 figures, accepted to SIGIR 2025
    </div>
    <details class="paper-abstract">
      We reproduce the UMBRELA LLM Judge evaluation framework across a range of large language models (LLMs) to assess its generalizability beyond the original study. Our investigation evaluates how LLM choice affects relevance assessment accuracy, focusing on leaderboard rank correlation and per-label agreement metrics. Results demonstrate that UMBRELA with DeepSeek V3 obtains very comparable performance to GPT-4o (used in original work). For LLaMA-3.3-70B we obtain slightly lower performance, which further degrades with smaller LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09481v1">Evaluating LLMs on Sequential API Call Through Automated Test Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-13
    </div>
    <details class="paper-abstract">
      By integrating tools from external APIs, Large Language Models (LLMs) have expanded their promising capabilities in a diverse spectrum of complex real-world tasks. However, testing, evaluation, and analysis of LLM tool use remain in their early stages. Most existing benchmarks rely on manually collected test cases, many of which cannot be automatically checked for semantic correctness and instead depend on static methods such as string matching. Additionally, these benchmarks often overlook the complex interactions that occur between sequential API calls, which are common in real-world applications. To fill the gap, in this paper, we introduce StateGen, an automated framework designed to generate diverse coding tasks involving sequential API interactions. StateGen combines state-machine-based API constraint solving and validation, energy-based sampling, and control-flow injection to generate executable programs. These programs are then translated into human-like natural language task descriptions through a collaboration of two LLM agents. Utilizing StateGen, we construct StateEval, a benchmark encompassing 120 verified test cases spanning across three representative scenarios: Session Service, Tensor Operation, and ElevenLabs MCP. Experimental results confirm that StateGen can effectively generate challenging and realistic API-oriented tasks, highlighting areas for improvement in current LLMs incorporating APIs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09477v1">Towards Agentic RAG with Deep Reasoning: A Survey of RAG-Reasoning Systems in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-13
      | 💬 submitted to ARR May
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) lifts the factuality of Large Language Models (LLMs) by injecting external knowledge, yet it falls short on problems that demand multi-step inference; conversely, purely reasoning-oriented approaches often hallucinate or mis-ground facts. This survey synthesizes both strands under a unified reasoning-retrieval perspective. We first map how advanced reasoning optimizes each stage of RAG (Reasoning-Enhanced RAG). Then, we show how retrieved knowledge of different type supply missing premises and expand context for complex inference (RAG-Enhanced Reasoning). Finally, we spotlight emerging Synergized RAG-Reasoning frameworks, where (agentic) LLMs iteratively interleave search and reasoning to achieve state-of-the-art performance across knowledge-intensive benchmarks. We categorize methods, datasets, and open challenges, and outline research avenues toward deeper RAG-Reasoning systems that are more effective, multimodally-adaptive, trustworthy, and human-centric. The collection is available at https://github.com/DavidZWZ/Awesome-RAG-Reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00648v3">DFRot: Achieving Outlier-Free and Massive Activation-Free for Rotated LLMs with Refined Rotation</a></div>
    <div class="paper-meta">
      📅 2025-07-13
      | 💬 Accepeted bythe 2nd Conference on Language Modeling (COLM 2025). Source code \url{https://github.com/JingyangXiang/DFRot}
    </div>
    <details class="paper-abstract">
      Rotating the activation and weight matrices to reduce the influence of outliers in large language models (LLMs) has recently attracted significant attention, particularly in the context of model quantization. Prior studies have shown that in low-precision quantization scenarios, such as 4-bit weights and 4-bit activations (W4A4), randomized Hadamard transforms can achieve significantly higher accuracy than randomized orthogonal transforms. Notably, the reason behind this phenomenon remains unknown. In this paper, we find that these transformations show substantial improvement in eliminating outliers for common tokens and achieve similar quantization error. The primary reason for the accuracy difference lies in the fact that randomized Hadamard transforms can slightly reduce the quantization error for tokens with massive activations while randomized orthogonal transforms increase the quantization error. Due to the extreme rarity of these tokens and their critical impact on model accuracy, we consider this a long-tail optimization problem, and therefore construct a simple yet effective method: a weighted loss function. Additionally, we propose an optimization strategy for the rotation matrix that involves alternating optimization of quantization parameters while employing orthogonal Procrustes transforms to refine the rotation matrix. This makes the distribution of the rotated activation values more conducive to quantization, especially for tokens with massive activations. Our method enhances the Rotated LLMs by achieving dual free, Outlier-Free and Massive Activation-Free, dubbed as DFRot. Extensive experiments demonstrate the effectiveness and efficiency of DFRot. By tuning the rotation matrix using just a single sample, DFRot achieves a perplexity improvement of 0.98 and 0.95 on W4A4KV4 and W4A4KV16, respectively, for LLaMA3-70B, a model known for its quantization challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10620v1">LLMs Meet Cross-Modal Time Series Analytics: Overview and Directions</a></div>
    <div class="paper-meta">
      📅 2025-07-13
      | 💬 Accepted at SSTD 2025 (Tutorial). arXiv admin note: text overlap with arXiv:2505.02583
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have emerged as a promising paradigm for time series analytics, leveraging their massive parameters and the shared sequential nature of textual and time series data. However, a cross-modality gap exists between time series and textual data, as LLMs are pre-trained on textual corpora and are not inherently optimized for time series. In this tutorial, we provide an up-to-date overview of LLM-based cross-modal time series analytics. We introduce a taxonomy that classifies existing approaches into three groups based on cross-modal modeling strategies, e.g., conversion, alignment, and fusion, and then discuss their applications across a range of downstream tasks. In addition, we summarize several open challenges. This tutorial aims to expand the practical application of LLMs in solving real-world problems in cross-modal time series analytics while balancing effectiveness and efficiency. Participants will gain a thorough understanding of current advancements, methodologies, and future research directions in cross-modal time series analytics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10605v1">RedOne: Revealing Domain-specific LLM Post-Training in Social Networking Services</a></div>
    <div class="paper-meta">
      📅 2025-07-13
    </div>
    <details class="paper-abstract">
      As a primary medium for modern information dissemination, social networking services (SNS) have experienced rapid growth, which has proposed significant challenges for platform content management and interaction quality improvement. Recently, the development of large language models (LLMs) has offered potential solutions but existing studies focus on isolated tasks, which not only encounter diminishing benefit from the data scaling within individual scenarios but also fail to flexibly adapt to diverse real-world context. To address these challenges, we introduce RedOne, a domain-specific LLM designed to break the performance bottleneck of single-task baselines and establish a comprehensive foundation for the SNS. RedOne was developed through a three-stage training strategy consisting of continue pretraining, supervised fine-tuning, and preference optimization, using a large-scale real-world dataset. Through extensive experiments, RedOne maintains strong general capabilities, and achieves an average improvement up to 14.02% across 8 major SNS tasks and 7.56% in SNS bilingual evaluation benchmark, compared with base models. Furthermore, through online testing, RedOne reduced the exposure rate in harmful content detection by 11.23% and improved the click page rate in post-view search by 14.95% compared with single-tasks finetuned baseline models. These results establish RedOne as a robust domain-specific LLM for SNS, demonstrating excellent generalization across various tasks and promising applicability in real-world scenarios.
    </details>
</div>
