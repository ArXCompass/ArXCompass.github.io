# llm - 2025_11

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- Part 7

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03304v4">Harmony in Divergence: Towards Fast, Accurate, and Memory-efficient Zeroth-order LLM Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2025-11-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel across various tasks, but standard first-order (FO) fine-tuning demands considerable memory, significantly limiting real-world deployment. Recently, zeroth-order (ZO) optimization stood out as a promising memory-efficient training paradigm, avoiding backward passes and relying solely on forward passes for gradient estimation, making it attractive for resource-constrained scenarios. However, ZO method lags far behind FO method in both convergence speed and accuracy. To bridge the gap, we introduce a novel layer-wise divergence analysis that uncovers the distinct update pattern of FO and ZO optimization. Aiming to resemble the learning capacity of FO method from the findings, we propose Divergence-driven Zeroth-Order (DiZO) optimization. DiZO conducts divergence-driven layer adaptation by incorporating projections to ZO updates, generating diverse-magnitude updates precisely scaled to layer-wise individual optimization needs. Our results demonstrate that DiZO significantly reduces the needed iterations for convergence without sacrificing throughput, cutting training GPU hours by up to 48\% on various datasets. Moreover, DiZO consistently outperforms the representative ZO baselines in fine-tuning RoBERTa-large, OPT-series, and Llama-series on downstream tasks and, in some cases, even surpasses memory-intensive FO fine-tuning. Our code is released at https://github.com/Skilteee/DiZO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10142v3">Debiasing LLMs by Masking Unfairness-Driving Attention Heads</a></div>
    <div class="paper-meta">
      📅 2025-11-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly mediate decisions in domains where unfair treatment of demographic groups is unacceptable. Existing work probes when biased outputs appear, but gives little insight into the mechanisms that generate them, leaving existing mitigations largely fragile. In this paper, we conduct a systematic investigation LLM unfairness and propose DiffHeads, a lightweight debiasing framework for LLMs. We first compare Direct-Answer (DA) prompting to Chain-of-Thought (CoT) prompting across eight representative open- and closed-source LLMs. DA will trigger the nature bias part of LLM and improve measured unfairness by 534.5%-391.9% in both one-turn and two-turn dialogues. Next, we define a token-to-head contribution score that traces each token's influence back to individual attention heads. This reveals a small cluster of bias heads that activate under DA but stay largely dormant with CoT, providing the first causal link between prompting strategy and bias emergence. Finally, building on this insight, we propose DiffHeads that identifies bias heads through differential activation analysis between DA and CoT, and selectively masks only those heads. DiffHeads reduces unfairness by 49.4%, and 40.3% under DA and CoT, respectively, without harming model utility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18253v2">From BERT to LLMs: Comparing and Understanding Chinese Classifier Prediction in Language Models</a></div>
    <div class="paper-meta">
      📅 2025-11-02
    </div>
    <details class="paper-abstract">
      Classifiers are an important and defining feature of the Chinese language, and their correct prediction is key to numerous educational applications. Yet, whether the most popular Large Language Models (LLMs) possess proper knowledge the Chinese classifiers is an issue that has largely remain unexplored in the Natural Language Processing (NLP) literature. To address such a question, we employ various masking strategies to evaluate the LLMs' intrinsic ability, the contribution of different sentence elements, and the working of the attention mechanisms during prediction. Besides, we explore fine-tuning for LLMs to enhance the classifier performance. Our findings reveal that LLMs perform worse than BERT, even with fine-tuning. The prediction, as expected, greatly benefits from the information about the following noun, which also explains the advantage of models with a bidirectional attention mechanism such as BERT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22603v2">Mitigating Attention Sinks and Massive Activations in Audio-Visual Speech Recognition with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-02
      | 💬 The code is available at https://github.com/umbertocappellazzo/Llama-AVSR
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently advanced auditory speech recognition (ASR), visual speech recognition (VSR), and audio-visual speech recognition (AVSR). However, understanding of their internal dynamics under fine-tuning remains limited. In natural language processing, recent work has revealed attention sinks, tokens that attract disproportionately high attention, and associated massive activations in which some features of sink tokens exhibit huge activation in LLMs. In this work, we are the first to study these phenomena in multimodal speech recognition. Through a detailed analysis of audio-visual LLMs, we identify attention sinks and massive activations not only at the BOS token but also at intermediate low-semantic tokens across ASR, VSR, and AVSR. We show that massive activations originate in the MLP layers and correspond to fixed feature indices across all sink tokens. We further show that intermediate sink tokens exhibit high cosine similarity to the BOS token, thereby amplifying attention and activation. Building on these insights, we introduce a simple decorrelation loss that reduces cosine similarity between BOS and other tokens, effectively mitigating intermediate sinks and massive activations. Furthermore, our method improves word error rate (WER) under high audio-visual feature downsampling while remaining stable at lower downsampling rates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05286v2">HEXGEN-FLOW: Optimizing LLM Inference Request Scheduling for Agentic Text-to-SQL</a></div>
    <div class="paper-meta">
      📅 2025-11-02
    </div>
    <details class="paper-abstract">
      Recent advancements in leveraging the agentic paradigm of large language models (LLMs) have substantially improved Text-to-SQL capabilities, empowering users without specialized database knowledge to intuitively query databases. However, deploying agentic LLM-based Text-to-SQL systems in production presents significant challenges, stemming from their inherently multi-stage computational dependencies, strict latency requirements, and the complexity of deployment across heterogeneous GPUs widely existing in enterprise clusters. Meanwhile, existing LLM serving frameworks are primarily designed for independent inference tasks, resulting in suboptimal performance and frequent service-level objective (SLO) violations in Text-to-SQL workloads. In this paper, we introduce HEXGEN-FLOW, a novel framework designed explicitly to schedule and execute agentic multi-stage LLM-based Text-to-SQL workflows on heterogeneous GPU clusters serving multi-tenant Text-to-SQL requests. HEXGEN-FLOW introduces a hierarchical scheduling approach that combines global workload-balanced task dispatching with an adaptive local priority queue, guided by a systematic analysis of agentic Text-to-SQL workflows. Additionally, we propose a lightweight simulation-based method for tuning critical scheduling hyperparameters, further enhancing robustness and adaptability. Our evaluation on realistic Text-to-SQL benchmarks demonstrates that HEXGEN-FLOW significantly outperforms state-of-the-art LLM serving frameworks. Across all traces, HEXGEN-FLOW reduces P95 tail latency by $1.42{\sim}1.56\times$ and increases throughput by $1.49{\sim}1.81\times$, demonstrating robust improvements under diverse workloads. Our code is available at https://github.com/Relaxed-System-Lab/Hexgen-Flow.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.19354v2">The Emerged Security and Privacy of LLM Agent: A Survey with Case Studies</a></div>
    <div class="paper-meta">
      📅 2025-11-02
      | 💬 35 pages, 19 figures. Accepted to ACM Computing Surveys (CSUR), 2025
    </div>
    <details class="paper-abstract">
      Inspired by the rapid development of Large Language Models (LLMs), LLM agents have evolved to perform complex tasks. LLM agents are now extensively applied across various domains, handling vast amounts of data to interact with humans and execute tasks. The widespread applications of LLM agents demonstrate their significant commercial value; however, they also expose security and privacy vulnerabilities. At the current stage, comprehensive research on the security and privacy of LLM agents is highly needed. This survey aims to provide a comprehensive overview of the newly emerged privacy and security issues faced by LLM agents. We begin by introducing the fundamental knowledge of LLM agents, followed by a categorization and analysis of the threats. We then discuss the impacts of these threats on humans, environment, and other agents. Subsequently, we review existing defensive strategies, and finally explore future trends. Additionally, the survey incorporates diverse case studies to facilitate a more accessible understanding. By highlighting these critical security and privacy issues, the survey seeks to stimulate future research towards enhancing the security and privacy of LLM agents, thereby increasing their reliability and trustworthiness in future applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16648v3">FESTA: Functionally Equivalent Sampling for Trust Assessment of Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-02
      | 💬 Accepted in the Findings of EMNLP, 2025
    </div>
    <details class="paper-abstract">
      The accurate trust assessment of multimodal large language models (MLLMs) generated predictions, which can enable selective prediction and improve user confidence, is challenging due to the diverse multi-modal input paradigms. We propose Functionally Equivalent Sampling for Trust Assessment (FESTA), a multimodal input sampling technique for MLLMs, that generates an uncertainty measure based on the equivalent and complementary input samplings. The proposed task-preserving sampling approach for uncertainty quantification expands the input space to probe the consistency (through equivalent samples) and sensitivity (through complementary samples) of the model. FESTA uses only input-output access of the model (black-box), and does not require ground truth (unsupervised). The experiments are conducted with various off-the-shelf multi-modal LLMs, on both visual and audio reasoning tasks. The proposed FESTA uncertainty estimate achieves significant improvement (33.3% relative improvement for vision-LLMs and 29.6% relative improvement for audio-LLMs) in selective prediction performance, based on area-under-receiver-operating-characteristic curve (AUROC) metric in detecting mispredictions. The code implementation is open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06632v2">Curriculum Reinforcement Learning from Easy to Hard Tasks Improves LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-02
    </div>
    <details class="paper-abstract">
      We aim to improve the reasoning capabilities of language models via reinforcement learning (RL). Recent RL post-trained models like DeepSeek-R1 have demonstrated reasoning abilities on mathematical and coding tasks. However, prior studies suggest that using RL alone to improve reasoning on inherently difficult tasks is less effective. Here, we draw inspiration from curriculum learning and propose to schedule tasks from easy to hard (E2H), allowing LLMs to build reasoning skills gradually. Our method is termed E2H Reasoner. Empirically, we observe that, although easy tasks are important initially, fading them out through appropriate scheduling is essential in preventing overfitting. Theoretically, we establish convergence guarantees for E2H Reasoner within an approximate policy iteration framework. We derive finite-sample complexity bounds and show that when tasks are appropriately decomposed and conditioned, learning through curriculum stages requires fewer total samples than direct learning. Experiments across multiple domains show that E2H Reasoner significantly improves the reasoning ability of small LLMs (1.5B to 3B), which otherwise struggle when trained with vanilla RL alone, highlighting the effectiveness of our method. Our code can be found on https://github.com/divelab/E2H-Reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.01090v1">Improving Romanian LLM Pretraining Data using Diversity and Quality Filtering</a></div>
    <div class="paper-meta">
      📅 2025-11-02
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently exploded in popularity, often matching or outperforming human abilities on many tasks. One of the key factors in training LLMs is the availability and curation of high-quality data. Data quality is especially crucial for under-represented languages, where high-quality corpora are scarce. In this work we study the characteristics and coverage of Romanian pretraining corpora and we examine how they differ from English data. By training a lightweight multitask model on carefully LLM-annotated Romanian texts, we are able to analyze and perform multi-level filtering (e.g., educational value, topic, format) to generate high-quality pretraining datasets. Our experiments show noteworthy trends in the topics present in Romanian and English data, while also proving the effectiveness of filtering data through improved LLM pretraining performance across multiple benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.01077v1">AI Progress Should Be Measured by Capability-Per-Resource, Not Scale Alone: A Framework for Gradient-Guided Resource Allocation in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-02
      | 💬 9 pages (main) + appendix, 3 figures. Accepted at NeurIPS 2025 (Position Paper Track), submission #491. OpenReview: https://openreview.net/forum?id=6plSmhBI33&noteId=KP5ZqY7JLg
    </div>
    <details class="paper-abstract">
      This position paper challenges the "scaling fundamentalism" dominating AI research, where unbounded growth in model size and computation has led to unsustainable environmental impacts and widening resource inequality. We argue that LLM development should be fundamentally reoriented toward capability-per-resource rather than capability alone. We present a theoretical framework demonstrating that resource-allocation decisions guided by gradient influence patterns can dramatically improve efficiency throughout the AI lifecycle. Our analysis shows that in transformer-based models, where a small fraction of parameters exert outsized influence (following heavy-tailed distributions), three critical insights emerge: (1) updating only high-influence parameters strictly outperforms full-parameter tuning on a performance-per-resource basis; (2) simple gradient norms provide computationally efficient proxies for identifying these high-influence components; and (3) coordinated parameter and data selection yields multiplicative efficiency gains, potentially reducing resource requirements by orders of magnitude. Building on these theoretical foundations, we propose a two stage paradigm marginal-return pretraining for foundation developers and influence guided adaptation for downstream users bridged by gradient blueprints, metadata describing which parameters matter most for various tasks. This capability-per-resource perspective transforms what were once considered pragmatic hardware workarounds into theoretically optimal strategies, democratizing access to cutting-edge AI capabilities while significantly reducing environmental impact. By embedding resource consciousness into how we develop, adapt, and evaluate models, we can reshape AI progress toward a more sustainable and equitable future.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.01066v1">HPLT~3.0: Very Large-Scale Multilingual Resources for LLM and MT. Mono- and Bi-lingual Data, Multilingual Evaluation, and Pre-Trained Models</a></div>
    <div class="paper-meta">
      📅 2025-11-02
    </div>
    <details class="paper-abstract">
      We present an ongoing initiative to provide open, very large, high-quality, and richly annotated textual datasets for almost 200 languages. At 30 trillion tokens, this is likely the largest generally available multilingual collection of LLM pre-training data. At 30 trillion tokens, this is likely the largest generally available multilingual collection of LLM pre-training data. These datasets are derived from web crawls from different sources and accompanied with a complete, open-source pipeline for document selection from web archives, text extraction from HTML, language identification for noisy texts, exact and near-deduplication, annotation with, among others, register labels, text quality estimates, and personally identifiable information; and final selection and filtering. We report on data quality probes through contrastive and analytical statistics, through manual inspection of samples for 24 languages, and through end-to-end evaluation of various language model architectures trained on this data. For multilingual LLM evaluation, we provide a comprehensive collection of benchmarks for nine European languages, with special emphasis on natively created tasks, mechanisms to mitigate prompt sensitivity, and refined normalization and aggregation of scores. Additionally, we train and evaluate a family of 57 monolingual encoder-decoder models, as well as a handful of monolingual GPT-like reference models. Besides the monolingual data and models, we also present a very large collection of parallel texts automatically mined from this data, together with a novel parallel corpus synthesized via machine translation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.01053v1">Building a Silver-Standard Dataset from NICE Guidelines for Clinical LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-02
      | 💬 Submitted to EFMI Medical Informatics Europe 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used in healthcare, yet standardised benchmarks for evaluating guideline-based clinical reasoning are missing. This study introduces a validated dataset derived from publicly available guidelines across multiple diagnoses. The dataset was created with the help of GPT and contains realistic patient scenarios, as well as clinical questions. We benchmark a range of recent popular LLMs to showcase the validity of our dataset. The framework supports systematic evaluation of LLMs' clinical utility and guideline adherence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.01014v1">IF-CRITIC: Towards a Fine-Grained LLM Critic for Instruction-Following Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-11-02
      | 💬 21 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Instruction following is a fundamental ability of Large Language Models (LLMs), requiring their generated outputs to follow multiple constraints imposed in input instructions. Numerous studies have attempted to enhance this ability through preference optimization or reinforcement learning based on reward signals from LLM-as-a-Judge. However, existing evaluation models for instruction following still possess many deficiencies, such as substantial costs and unreliable assessments. To this end, we propose IF-CRITIC, an LLM critic that can provide efficient and reliable assessments of constraint following in the instructions. We first develop a checklist generator to decompose instructions and generate constraint checklists. With the assistance of the checklists, we collect high-quality critique training data through a multi-stage critique filtering mechanism and employ a constraint-level preference optimization method to train IF-CRITIC. Extensive experiments demonstrate that the evaluation performance of IF-CRITIC can beat strong LLM-as-a-Judge baselines, including Deepseek-R1 and o4-mini. With the scalable reward signals provided by IF-CRITIC, LLMs can achieve substantial performance gains in instruction-following optimization under lower computational overhead compared to strong LLM critic baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.01934v1">Tool Zero: Training Tool-Augmented LLMs via Pure RL from Scratch</a></div>
    <div class="paper-meta">
      📅 2025-11-02
      | 💬 EMNLP 2025 finding
    </div>
    <details class="paper-abstract">
      Training tool-augmented LLMs has emerged as a promising approach to enhancing language models' capabilities for complex tasks. The current supervised fine-tuning paradigm relies on constructing extensive domain-specific datasets to train models. However, this approach often struggles to generalize effectively to unfamiliar or intricate tool-use scenarios. Recently, reinforcement learning (RL) paradigm can endow LLMs with superior reasoning and generalization abilities. In this work, we address a key question: Can the pure RL be used to effectively elicit a model's intrinsic reasoning capabilities and enhance the tool-agnostic generalization? We propose a dynamic generalization-guided reward design for rule-based RL, which progressively shifts rewards from exploratory to exploitative tool-use patterns. Based on this design, we introduce the Tool-Zero series models. These models are trained to enable LLMs to autonomously utilize general tools by directly scaling up RL from Zero models (i.e., base models without post-training). Experimental results demonstrate that our models achieve over 7% performance improvement compared to both SFT and RL-with-SFT models under the same experimental settings. These gains are consistently replicated across cross-dataset and intra-dataset evaluations, validating the effectiveness and robustness of our methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.00993v1">Aligning LLM agents with human learning and adjustment behavior: a dual agent approach</a></div>
    <div class="paper-meta">
      📅 2025-11-02
      | 💬 32 pages, 6 figures, 7 tables
    </div>
    <details class="paper-abstract">
      Effective modeling of how human travelers learn and adjust their travel behavior from interacting with transportation systems is critical for system assessment and planning. However, this task is also difficult due to the complex cognition and decision-making involved in such behavior. Recent research has begun to leverage Large Language Model (LLM) agents for this task. Building on this, we introduce a novel dual-agent framework that enables continuous learning and alignment between LLM agents and human travelers on learning and adaptation behavior from online data streams. Our approach involves a set of LLM traveler agents, equipped with a memory system and a learnable persona, which serve as simulators for human travelers. To ensure behavioral alignment, we introduce an LLM calibration agent that leverages the reasoning and analytical capabilities of LLMs to train the personas of these traveler agents. Working together, this dual-agent system is designed to track and align the underlying decision-making mechanisms of travelers and produce realistic, adaptive simulations. Using a real-world dataset from a day-to-day route choice experiment, we show our approach significantly outperforms existing LLM-based methods in both individual behavioral alignment and aggregate simulation accuracy. Furthermore, we demonstrate that our method moves beyond simple behavioral mimicry to capture the evolution of underlying learning processes, a deeper alignment that fosters robust generalization. Overall, our framework provides a new approach for creating adaptive and behaviorally realistic agents to simulate travelers' learning and adaptation that can benefit transportation simulation and policy analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.00924v1">The Biased Oracle: Assessing LLMs' Understandability and Empathy in Medical Diagnoses</a></div>
    <div class="paper-meta">
      📅 2025-11-02
      | 💬 Accepted by NeurIPS 2025 GenAI4Health Workshop
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) show promise for supporting clinicians in diagnostic communication by generating explanations and guidance for patients. Yet their ability to produce outputs that are both understandable and empathetic remains uncertain. We evaluate two leading LLMs on medical diagnostic scenarios, assessing understandability using readability metrics as a proxy and empathy through LLM-as-a-Judge ratings compared to human evaluations. The results indicate that LLMs adapt explanations to socio-demographic variables and patient conditions. However, they also generate overly complex content and display biased affective empathy, leading to uneven accessibility and support. These patterns underscore the need for systematic calibration to ensure equitable patient communication. The code and data are released: https://github.com/Jeffateth/Biased_Oracle
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.00898v1">Empowering LLMs with Structural Role Inference for Zero-Shot Graph Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-02
    </div>
    <details class="paper-abstract">
      Large Language Models have emerged as a promising approach for graph learning due to their powerful reasoning capabilities. However, existing methods exhibit systematic performance degradation on structurally important nodes such as bridges and hubs. We identify the root cause of these limitations. Current approaches encode graph topology into static features but lack reasoning scaffolds to transform topological patterns into role-based interpretations. This limitation becomes critical in zero-shot scenarios where no training data establishes structure-semantics mappings. To address this gap, we propose DuoGLM, a training-free dual-perspective framework for structure-aware graph reasoning. The local perspective constructs relation-aware templates capturing semantic interactions between nodes and neighbors. The global perspective performs topology-to-role inference to generate functional descriptions of structural positions. These complementary perspectives provide explicit reasoning mechanisms enabling LLMs to distinguish topologically similar but semantically different nodes. Extensive experiments across eight benchmark datasets demonstrate substantial improvements. DuoGLM achieves 14.3\% accuracy gain in zero-shot node classification and 7.6\% AUC improvement in cross-domain transfer compared to existing methods. The results validate the effectiveness of explicit role reasoning for graph understanding with LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.00879v1">Assessing LLM Reasoning Steps via Principal Knowledge Grounding</a></div>
    <div class="paper-meta">
      📅 2025-11-02
      | 💬 Accepted to EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Step-by-step reasoning has become a standard approach for large language models (LLMs) to tackle complex tasks. While this paradigm has proven effective, it raises a fundamental question: How can we verify that an LLM's reasoning is accurately grounded in knowledge? To address this question, we introduce a novel evaluation suite that systematically assesses the knowledge grounding of intermediate reasoning. Our framework comprises three key components. (1) Principal Knowledge Collection, a large-scale repository of atomic knowledge essential for reasoning. Based on the collection, we propose (2) knowledge-grounded evaluation metrics designed to measure how well models recall and apply prerequisite knowledge in reasoning. These metrics are computed by our (3) evaluator LLM, a lightweight model optimized for cost-effective and reliable metric computation. Our evaluation suite demonstrates remarkable effectiveness in identifying missing or misapplied knowledge elements, providing crucial insights for uncovering fundamental reasoning deficiencies in LLMs. Beyond evaluation, we demonstrate how these metrics can be integrated into preference optimization, showcasing further applications of knowledge-grounded evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.00874v1">Training with Fewer Bits: Unlocking Edge LLMs Training with Stochastic Rounding</a></div>
    <div class="paper-meta">
      📅 2025-11-02
    </div>
    <details class="paper-abstract">
      LLM training is resource-intensive. Quantized training improves computational and memory efficiency but introduces quantization noise, which can hinder convergence and degrade model accuracy. Stochastic Rounding (SR) has emerged as a theoretically attractive alternative to deterministic rounding, offering unbiased gradient estimates. However, its interaction with other training factors -- especially batch size -- remains under explored. In this paper, we present a theoretical and empirical study of mini-batch stochastic gradient descent (SGD) with SR, showing that increased batch sizes can compensate for reduced precision during back-propagation. Furthermore, we show that quantizing weights and activations impacts gradient variance in distinct ways. Our experiments validate these theoretical insights.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.00818v1">Deciphering Scientific Collaboration in Biomedical LLM Research: Dynamics, Institutional Participation, and Resource Disparities</a></div>
    <div class="paper-meta">
      📅 2025-11-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly transforming biomedical discovery and clinical innovation, yet their impact extends far beyond algorithmic revolution-LLMs are restructuring how scientific collaboration occurs, who participates, and how resources shape innovation. Despite this profound transformation, how this rapid technological shift is reshaping the structure and equity of scientific collaboration in biomedical LLM research remains largely unknown. By analyzing 5,674 LLM-related biomedical publications from PubMed, we examine how collaboration diversity evolves over time, identify institutions and disciplines that anchor and bridge collaboration networks, and assess how resource disparities underpin research performance. We find that collaboration diversity has grown steadily, with a decreasing share of Computer Science and Artificial Intelligence authors, suggesting that LLMs are lowering technical barriers for biomedical investigators. Network analysis reveals central institutions, including Stanford University and Harvard Medical School, and bridging disciplines such as Medicine and Computer Science that anchor collaborations in this field. Furthermore, biomedical research resources are strongly linked to research performance, with high-performing resource-constrained institutions exhibiting larger collaboration volume with the top 1% most connected institutions in the network. Together, these findings reveal a complex landscape, where democratizing trends coexist with a persistent, resource-driven hierarchy, highlighting the critical role of strategic collaboration in this evolving field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.00808v1">Do Math Reasoning LLMs Help Predict the Impact of Public Transit Events?</a></div>
    <div class="paper-meta">
      📅 2025-11-02
    </div>
    <details class="paper-abstract">
      Predicting public transit incident duration from unstructured text alerts is a critical but challenging task. Addressing the domain sparsity of transit operations with standard Supervised Fine-Tuning (SFT) is difficult, as the task involves noisy, continuous labels and lacks reliable expert demonstrations for reasoning. While Reinforcement Learning from Verifiable Rewards (RLVR) excels at tasks with binary correctness, like mathematics, its applicability to noisy, continuous forecasting is an open question. This work, to our knowledge, is the first to bridge the gap between RLVR LLM training with the critical, real-world forecasting challenges in public transit operations. We adapt RLVR to this task by introducing a tolerance-based, shaped reward function that grants partial credit within a continuous error margin, rather than demanding a single correct answer. We systematically evaluate this framework on a curated dataset of NYC MTA service alerts. Our findings show that general-purpose, instruction-tuned LLMs significantly outperform specialized math-reasoning models, which struggle with the ambiguous, real-world text. We empirically demonstrate that the binary reward is unstable and degrades performance, whereas our shaped reward design is critical and allows our model to dominate on the most challenging metrics. While classical regressors are superior at minimizing overall MAE or MSE, our RLVR approach achieved a 35\% relative improvement in 5-minute accuracy (Acc@5) over the strongest baseline. This demonstrates that RLVR can be successfully adapted to real-world, noisy forecasting, but requires a verifier design that reflects the continuous nature of the problem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.00807v1">FREESH: Fair, Resource- and Energy-Efficient Scheduling for LLM Serving on Heterogeneous GPUs</a></div>
    <div class="paper-meta">
      📅 2025-11-02
      | 💬 In Submission, code available at https://github.com/AndrewFangZequan/LLM_Serving_FREESH
    </div>
    <details class="paper-abstract">
      The ever-increasing computation and energy demand for LLM and AI agents call for holistic and efficient optimization of LLM serving systems. In practice, heterogeneous GPU clusters can be deployed in a geographically distributed manner, while LLM load also observes diversity in terms of both query traffic and serving patterns. LLM queries running on advanced GPUs during a high-emission hour at one location can lead to significantly higher carbon footprints versus same queries running on mid-level GPUs at a low-emission time and location. By observing LLM serving requirements and leveraging spatiotemporal computation flexibility, we consider the joint routing and scheduling problem, and propose FREESH to cooperatively run a group of data centers while minimizing user-specified carbon or energy objectives. FREESH identifies the optimal configurations of balanced load serving by matching distinct GPU instance's power-throughput characteristics with predictable LLM query length and workloads. To ensure both latency and fairness requirements, FREESH identifies optimized parallelism and query routing schedules together with dynamic GPU frequency scaling for power saving, and Least-Laxity-First (LLF) serving strategy for query scheduling. During the 1-hour serving on production workloads, FREESH reduces energy by 28.6% and emissions by 45.45% together with improvements in SLO attainment and fairness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.00802v1">GrowthHacker: Automated Off-Policy Evaluation Optimization Using Code-Modifying LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-02
    </div>
    <details class="paper-abstract">
      With the software industry shifting toward a data-driven culture, online A/B testing is a key tool for evaluating new technologies. However, deploying such experiments requires substantial resources, may negatively impact users, and involves long data collection periods. To address this, \textit{off-policy evaluation (OPE)}, or offline A/B testing, uses logged data to assess technologies and is fundamental in Reinforcement Learning, making it crucial in domains where online testing is costly or risky, such as healthcare, recommender systems, education, dialog systems, and robotics. Despite advances in coding LLMs and agentic AI, little is known about leveraging them to optimize OPE results. We investigate whether LLMs and LLM-based agents can improve OPE performance via code optimization. We propose \textit{GrowthHacker}, a benchmark with agent and baseline methods on large-scale real-world datasets, which iteratively optimizes code, evaluates results, and begins new optimization cycles. We collected datasets, established protocols, implemented baselines for OPE on the Open Bandit Pipeline (OBP)~\cite{saito2021openbanditdatasetpipeline} and Scope-RL~\cite{kiyohara2023scope}, and developed the \textit{two_agent} framework, which reduces system complexity while preserving optimization effectiveness. Results show the two_agent framework achieves 100% reliability and the highest average improvement of 106.7% among positive outcomes. Both two_agent and CrewAI reach 45% success rates, outperforming AutoGen's 34%. These findings demonstrate the feasibility of LLM-based agents as automated "growth hackers" to enhance OPE systems, with implications for scaling data-driven decision-making in production.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.00783v1">When Semantics Connect the Swarm: LLM-Driven Fuzzy Control for Cooperative Multi-Robot Underwater Coverage</a></div>
    <div class="paper-meta">
      📅 2025-11-02
      | 💬 This paper has been submitted to IEEE Transactions on Mobile Computing
    </div>
    <details class="paper-abstract">
      Underwater multi-robot cooperative coverage remains challenging due to partial observability, limited communication, environmental uncertainty, and the lack of access to global localization. To address these issues, this paper presents a semantics-guided fuzzy control framework that couples Large Language Models (LLMs) with interpretable control and lightweight coordination. Raw multimodal observations are compressed by the LLM into compact, human-interpretable semantic tokens that summarize obstacles, unexplored regions, and Objects Of Interest (OOIs) under uncertain perception. A fuzzy inference system with pre-defined membership functions then maps these tokens into smooth and stable steering and gait commands, enabling reliable navigation without relying on global positioning. Then, we further coordinate multiple robots by introducing semantic communication that shares intent and local context in linguistic form, enabling agreement on who explores where while avoiding redundant revisits. Extensive simulations in unknown reef-like environments show that, under limited sensing and communication, the proposed framework achieves robust OOI-oriented navigation and cooperative coverage with improved efficiency and adaptability, narrowing the gap between semantic cognition and distributed underwater control in GPS-denied, map-free conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.00782v1">Count-Based Approaches Remain Strong: A Benchmark Against Transformer and LLM Pipelines on Structured EHR</a></div>
    <div class="paper-meta">
      📅 2025-11-02
    </div>
    <details class="paper-abstract">
      Structured electronic health records (EHR) are essential for clinical prediction. While count-based learners continue to perform strongly on such data, no benchmarking has directly compared them against more recent mixture-of-agents LLM pipelines, which have been reported to outperform single LLMs in various NLP tasks. In this study, we evaluated three categories of methodologies for EHR prediction using the EHRSHOT dataset: count-based models built from ontology roll-ups with two time bins, based on LightGBM and the tabular foundation model TabPFN; a pretrained sequential transformer (CLMBR); and a mixture-of-agents pipeline that converts tabular histories to natural-language summaries followed by a text classifier. We assessed eight outcomes using the EHRSHOT dataset. Across the eight evaluation tasks, head-to-head wins were largely split between the count-based and the mixture-of-agents methods. Given their simplicity and interpretability, count-based models remain a strong candidate for structured EHR benchmarking. The source code is available at: https://github.com/cristea-lab/Structured_EHR_Benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20432v3">LLM Strategic Reasoning: Agentic Study through Behavioral Game Theory</a></div>
    <div class="paper-meta">
      📅 2025-11-01
      | 💬 Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Strategic decision-making involves interactive reasoning where agents adapt their choices in response to others, yet existing evaluations of large language models (LLMs) often emphasize Nash Equilibrium (NE) approximation, overlooking the mechanisms driving their strategic choices. To bridge this gap, we introduce an evaluation framework grounded in behavioral game theory, disentangling reasoning capability from contextual effects. Testing 22 state-of-the-art LLMs, we find that GPT-o3-mini, GPT-o1, and DeepSeek-R1 dominate most games yet also demonstrate that the model scale alone does not determine performance. In terms of prompting enhancement, Chain-of-Thought (CoT) prompting is not universally effective, as it increases strategic reasoning only for models at certain levels while providing limited gains elsewhere. Additionally, we investigate the impact of encoded demographic features on the models, observing that certain assignments impact the decision-making pattern. For instance, GPT-4o shows stronger strategic reasoning with female traits than males, while Gemma assigns higher reasoning levels to heterosexual identities compared to other sexual orientations, indicating inherent biases. These findings underscore the need for ethical standards and contextual alignment to balance improved reasoning with fairness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.10886v3">SLIP: Securing LLMs IP Using Weights Decomposition</a></div>
    <div class="paper-meta">
      📅 2025-11-01
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently seen widespread adoption in both academia and industry. As these models grow, they become valuable intellectual property (IP), reflecting substantial investments by their owners. The high cost of cloud-based deployment has spurred interest in running models on edge devices, but this risks exposing parameters to theft and unauthorized use. Existing approaches to protect model IP on the edge trade off practicality, accuracy, or deployment requirements. We introduce SLIP, a hybrid inference algorithm designed to protect edge-deployed models from theft. SLIP is, to our knowledge, the first hybrid protocol that is both practical for real-world applications and provably secure, while incurring zero accuracy degradation and minimal latency overhead. It partitions the model across two computing resources: one secure but expensive, and one cost-effective but vulnerable. Using matrix decomposition, the secure resource retains the most sensitive portion of the model's IP while performing only a small fraction of the computation; the vulnerable resource executes the remainder. The protocol includes security guarantees that prevent attackers from using the partition to infer the protected information. Finally, we present experimental results that demonstrate the robustness and effectiveness of our method, positioning it as a compelling solution for protecting LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.01390v2">Recognising, Anticipating, and Mitigating LLM Pollution of Online Behavioural Research</a></div>
    <div class="paper-meta">
      📅 2025-11-01
    </div>
    <details class="paper-abstract">
      Online behavioural research faces an emerging threat as participants increasingly turn to large language models (LLMs) for advice, translation, or task delegation: LLM Pollution. We identify three interacting variants through which LLM Pollution threatens the validity and integrity of online behavioural research. First, Partial LLM Mediation occurs when participants make selective use of LLMs for specific aspects of a task, such as translation or wording support, leading researchers to (mis)interpret LLM-shaped outputs as human ones. Second, Full LLM Delegation arises when agentic LLMs complete studies with little to no human oversight, undermining the central premise of human-subject research at a more foundational level. Third, LLM Spillover signifies human participants altering their behaviour as they begin to anticipate LLM presence in online studies, even when none are involved. While Partial Mediation and Full Delegation form a continuum of increasing automation, LLM Spillover reflects second-order reactivity effects. Together, these variants interact and generate cascading distortions that compromise sample authenticity, introduce biases that are difficult to detect post hoc, and ultimately undermine the epistemic grounding of online research on human cognition and behaviour. Crucially, the threat of LLM Pollution is already co-evolving with advances in generative AI, creating an escalating methodological arms race. To address this, we propose a multi-layered response spanning researcher practices, platform accountability, and community efforts. As the challenge evolves, coordinated adaptation will be essential to safeguard methodological integrity and preserve the validity of online behavioural research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14314v2">Zero-knowledge LLM hallucination detection and mitigation through fine-grained cross-model consistency</a></div>
    <div class="paper-meta">
      📅 2025-11-01
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive capabilities across diverse tasks, but they remain susceptible to hallucinations--generating content that appears plausible but contains factual inaccuracies. We present Finch-Zk, a black-box framework that leverages fine-grained cross-model consistency to detect and mitigate hallucinations in LLM outputs without requiring external knowledge sources. Finch-Zk introduces two key innovations: 1) a cross-model consistency checking strategy that reveals fine-grained inaccuracies by comparing responses generated by diverse models from semantically-equivalent prompts, and 2) a targeted mitigation technique that applies precise corrections to problematic segments while preserving accurate content. Experiments on the FELM dataset show Finch-Zk improves hallucination detection F1 scores by 6-39\% compared to existing approaches. For mitigation, Finch-Zk achieves up to 9 absolute percentage points improvement in answer accuracy on the GPQA-diamond dataset when applied to state-of-the-art models like Llama 4 Maverick and Claude 4 Sonnet. Extensive evaluation on multiple datasets demonstrates that Finch-Zk provides a practical, deployment-ready safeguard for enhancing factual reliability in production LLM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.03343v3">What Features in Prompts Jailbreak LLMs? Investigating the Mechanisms Behind Attacks</a></div>
    <div class="paper-meta">
      📅 2025-11-01
    </div>
    <details class="paper-abstract">
      Jailbreaks have been a central focus of research regarding the safety and reliability of large language models (LLMs), yet the mechanisms underlying these attacks remain poorly understood. While previous studies have predominantly relied on linear methods to detect jailbreak attempts and model refusals, we take a different approach by examining both linear and non-linear features in prompts that lead to successful jailbreaks. First, we introduce a novel dataset comprising 10,800 jailbreak attempts spanning 35 diverse attack methods. Leveraging this dataset, we train linear and non-linear probes on hidden states of open-weight LLMs to predict jailbreak success. Probes achieve strong in-distribution accuracy but transfer is attack-family-specific, revealing that different jailbreaks are supported by distinct internal mechanisms rather than a single universal direction. To establish causal relevance, we construct probe-guided latent interventions that systematically shift compliance in the predicted direction. Interventions derived from non-linear probes produce larger and more reliable effects than those from linear probes, indicating that features linked to jailbreak success are encoded non-linearly in prompt representations. Overall, the results surface heterogeneous, non-linear structure in jailbreak mechanisms and provide a prompt-side methodology for recovering and testing the features that drive jailbreak outcomes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17336v3">PPMI: Privacy-Preserving LLM Interaction with Socratic Chain-of-Thought Reasoning and Homomorphically Encrypted Vector Databases</a></div>
    <div class="paper-meta">
      📅 2025-11-01
      | 💬 29 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as personal agents, accessing sensitive user data such as calendars, emails, and medical records. Users currently face a trade-off: They can send private records, many of which are stored in remote databases, to powerful but untrusted LLM providers, increasing their exposure risk. Alternatively, they can run less powerful models locally on trusted devices. We bridge this gap. Our Socratic Chain-of-Thought Reasoning first sends a generic, non-private user query to a powerful, untrusted LLM, which generates a Chain-of-Thought (CoT) prompt and detailed sub-queries without accessing user data. Next, we embed these sub-queries and perform encrypted sub-second semantic search using our Homomorphically Encrypted Vector Database across one million entries of a single user's private data. This represents a realistic scale of personal documents, emails, and records accumulated over years of digital activity. Finally, we feed the CoT prompt and the decrypted records to a local language model and generate the final response. On the LoCoMo long-context QA benchmark, our hybrid framework, combining GPT-4o with a local Llama-3.2-1B model, outperforms using GPT-4o alone by up to 7.1 percentage points. This demonstrates a first step toward systems where tasks are decomposed and split between untrusted strong LLMs and weak local ones, preserving user privacy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21189v2">Exploring the Hidden Capacity of LLMs for One-Step Text Generation</a></div>
    <div class="paper-meta">
      📅 2025-11-01
      | 💬 accepted to EMNLP2025 main
    </div>
    <details class="paper-abstract">
      A recent study showed that large language models (LLMs) can reconstruct surprisingly long texts - up to thousands of tokens - via autoregressive generation from just one trained input embedding. In this work, we explore whether autoregressive decoding is essential for such reconstruction. We show that frozen LLMs can generate hundreds of accurate tokens in just one token-parallel forward pass, when provided with only two learned embeddings. This reveals a surprising and underexplored multi-token generation capability of autoregressive LLMs. We examine these embeddings and characterize the information they encode. We also empirically show that, although these representations are not unique for a given text, they form connected and local regions in embedding space - suggesting the potential to train a practical encoder. The existence of such representations hints that multi-token generation may be natively accessible in off-the-shelf LLMs via a learned input encoder, eliminating heavy retraining and helping to overcome the fundamental bottleneck of autoregressive decoding while reusing already-trained models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12872v2">KVCOMM: Online Cross-context KV-cache Communication for Efficient LLM-based Multi-agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-11-01
      | 💬 Accepted for publication in NeurIPS2025. Code is available at \url{https://github.com/FastMAS/KVCOMM}
    </div>
    <details class="paper-abstract">
      Multi-agent large language model (LLM) systems are increasingly adopted for complex language processing tasks that require communication and coordination among agents. However, these systems often suffer substantial overhead from repeated reprocessing of overlapping contexts across agents. In typical pipelines, once an agent receives a message from its predecessor, the full context-including prior turns-must be reprocessed from scratch, leading to inefficient processing. While key-value (KV) caching is an effective solution for avoiding redundant computation in single-agent settings where prefixes remain unchanged, it cannot be directly reused in multi-agent scenarios due to diverging prefixes introduced by agent-specific context extensions. We identify that the core challenge lies in the offset variance of KV-caches across agents. To address this, we propose KVCOMM, a training-free framework that enables efficient prefilling in multi-agent inference by reusing KV-caches and aligning cache offsets of overlapping contexts under diverse prefix contexts. KVCOMM estimates and adjusts KV-caches for shared content by referencing a pool of cached examples-termed anchors-that store observed cache deviations under varying prefixes. The anchor pool is maintained and updated online, allowing dynamic adaptation to distinct user requests and context structures. KVCOMM achieves over 70% reuse rate across diverse multi-agent workloads, including retrieval-augmented generation, math reasoning, and collaborative coding tasks, all without quality degradation. Particularly, when each fully-connected agent receives 1K input tokens with 512 prefix tokens and 512 output tokens under a five-agent setting, KVCOMM achieves up to 7.8x speedup compared to the standard prefill pipeline, reducing TTFT from ~430 ms to ~55 ms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24284v2">MCP-Flow: Facilitating LLM Agents to Master Real-World, Diverse and Scaling MCP Tools</a></div>
    <div class="paper-meta">
      📅 2025-11-01
      | 💬 Preprint, Under Review
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) increasingly rely on external tools to perform complex, realistic tasks, yet their ability to utilize the rapidly expanding Model Contextual Protocol (MCP) ecosystem remains limited. Existing MCP research covers few servers, depends on costly manual curation, and lacks training support, hindering progress toward real-world deployment. To overcome these limitations, we introduce MCP-Flow, an automated web-agent-driven pipeline for large-scale server discovery, data synthesis, and model training. MCP-Flow collects and filters data from 1166 servers and 11536 tools, producing 68733 high-quality instruction-function call pairs and 6439 trajectories, far exceeding prior work in scale and diversity. Extensive experiments demonstrate MCP-Flow's effectiveness in driving superior MCP tool selection, function-call generation, and enhanced agentic task performance. MCP-Flow thus provides a scalable foundation for advancing LLM agents' proficiency in real-world MCP environments. MCP-Flow is publicly available at \href{https://github.com/wwh0411/MCP-Flow}{https://github.com/wwh0411/MCP-Flow}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23874v3">From Stochasticity to Signal: A Bayesian Latent State Model for Reliable Measurement with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-01
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to automate classification tasks in business, such as analyzing customer satisfaction from text. However, the inherent stochasticity of LLMs, in terms of their tendency to produce different outputs for the same input, creates a significant measurement error problem that is often neglected with a single round of output, or addressed with ad-hoc methods like majority voting. Such naive approaches fail to quantify uncertainty and can produce biased estimates of population-level metrics. In this paper, we propose a principled solution by reframing LLM variability as a statistical measurement error problem and introducing a Bayesian latent state model to address it. Our model treats the true classification (e.g., customer dissatisfaction) as an unobserved latent variable and the multiple LLM ratings as noisy measurements of this state. This framework allows for the simultaneous estimation of the LLM's false positive and false negative error rates, the underlying base rate of the phenomenon in the population, the posterior probability of the true state for each individual observation, and the causal impact of a business intervention, if any, on the latent state. Through simulation studies, we demonstrate that our model accurately recovers true parameters where naive methods fail. We conclude that this methodology provides a general and reliable framework for converting noisy, probabilistic outputs from LLMs into accurate and actionable insights for scientific and business applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17210v2">Wisdom is Knowing What not to Say: Hallucination-Free LLMs Unlearning via Attention Shifting</a></div>
    <div class="paper-meta">
      📅 2025-11-01
      | 💬 22 pages, 10 figures
    </div>
    <details class="paper-abstract">
      The increase in computing power and the necessity of AI-assisted decision-making boost the growing application of large language models (LLMs). Along with this, the potential retention of sensitive data of LLMs has spurred increasing research into machine unlearning. However, existing unlearning approaches face a critical dilemma: Aggressive unlearning compromises model utility, while conservative strategies preserve utility but risk hallucinated responses. This significantly limits LLMs' reliability in knowledge-intensive applications. To address this, we introduce a novel Attention-Shifting (AS) framework for selective unlearning. AS is driven by two design objectives: (1) context-preserving suppression that attenuates attention to fact-bearing tokens without disrupting LLMs' linguistic structure; and (2) hallucination-resistant response shaping that discourages fabricated completions when queried about unlearning content. AS realizes these objectives through two attention-level interventions, which are importance-aware suppression applied to the unlearning set to reduce reliance on memorized knowledge and attention-guided retention enhancement that reinforces attention toward semantically essential tokens in the retained dataset to mitigate unintended degradation. These two components are jointly optimized via a dual-loss objective, which forms a soft boundary that localizes unlearning while preserving unrelated knowledge under representation superposition. Experimental results show that AS improves performance preservation over the state-of-the-art unlearning methods, achieving up to 15% higher accuracy on the ToFU benchmark and 10% on the TDEC benchmark, while maintaining competitive hallucination-free unlearning effectiveness. Compared to existing methods, AS demonstrates a superior balance between unlearning effectiveness, generalization, and response reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.00730v1">Teaching LLMs to See and Guide: Context-Aware Real-Time Assistance in Augmented Reality</a></div>
    <div class="paper-meta">
      📅 2025-11-01
      | 💬 This work is intended for submission to the IEEE Transactions on Systems, Man, and Cybernetics: Systems for possible publication
    </div>
    <details class="paper-abstract">
      The growing adoption of augmented and virtual reality (AR and VR) technologies in industrial training and on-the-job assistance has created new opportunities for intelligent, context-aware support systems. As workers perform complex tasks guided by AR and VR, these devices capture rich streams of multimodal data, including gaze, hand actions, and task progression, that can reveal user intent and task state in real time. Leveraging this information effectively remains a major challenge. In this work, we present a context-aware large language model (LLM) assistant that integrates diverse data modalities, such as hand actions, task steps, and dialogue history, into a unified framework for real-time question answering. To systematically study how context influences performance, we introduce an incremental prompting framework, where each model version receives progressively richer contextual inputs. Using the HoloAssist dataset, which records AR-guided task executions, we evaluate how each modality contributes to the assistant's effectiveness. Our experiments show that incorporating multimodal context significantly improves the accuracy and relevance of responses. These findings highlight the potential of LLM-driven multimodal integration to enable adaptive, intuitive assistance for AR and VR-based industrial training and assistance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.00664v1">ShadowLogic: Backdoors in Any Whitebox LLM</a></div>
    <div class="paper-meta">
      📅 2025-11-01
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are widely deployed across various applications, often with safeguards to prevent the generation of harmful or restricted content. However, these safeguards can be covertly bypassed through adversarial modifications to the computational graph of a model. This work highlights a critical security vulnerability in computational graph-based LLM formats, demonstrating that widely used deployment pipelines may be susceptible to obscured backdoors. We introduce ShadowLogic, a method for creating a backdoor in a white-box LLM by injecting an uncensoring vector into its computational graph representation. We set a trigger phrase that, when added to the beginning of a prompt into the LLM, applies the uncensoring vector and removes the content generation safeguards in the model. We embed trigger logic directly into the computational graph which detects the trigger phrase in a prompt. To evade detection of our backdoor, we obfuscate this logic within the graph structure, making it similar to standard model functions. Our method requires minimal alterations to model parameters, making backdoored models appear benign while retaining the ability to generate uncensored responses when activated. We successfully implement ShadowLogic in Phi-3 and Llama 3.2, using ONNX for manipulating computational graphs. Implanting the uncensoring vector achieved a >60% attack success rate for further malicious queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.00628v1">AgentGit: A Version Control Framework for Reliable and Scalable LLM-Powered Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-11-01
    </div>
    <details class="paper-abstract">
      With the rapid progress of large language models (LLMs), LLM-powered multi-agent systems (MAS) are drawing increasing interest across academia and industry. However, many current MAS frameworks struggle with reliability and scalability, especially on complex tasks. We present AgentGit, a framework that brings Git-like rollback and branching to MAS workflows. Built as an infrastructure layer on top of LangGraph, AgentGit supports state commit, revert, and branching, allowing agents to traverse, compare, and explore multiple trajectories efficiently. To evaluate AgentGit, we designed an experiment that optimizes target agents by selecting better prompts. We ran a multi-step A/B test against three baselines -- LangGraph, AutoGen, and Agno -- on a real-world task: retrieving and analyzing paper abstracts. Results show that AgentGit significantly reduces redundant computation, lowers runtime and token usage, and supports parallel exploration across multiple branches, enhancing both reliability and scalability in MAS development. This work offers a practical path to more robust MAS design and enables error recovery, safe exploration, iterative debugging, and A/B testing in collaborative AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.00592v1">Agentic Auto-Scheduling: An Experimental Study of LLM-Guided Loop Optimization</a></div>
    <div class="paper-meta">
      📅 2025-11-01
      | 💬 Accepted at the 34th International Conference on Parallel Architectures and Compilation Techniques (PACT 2025). 12 pages, plus appendix
    </div>
    <details class="paper-abstract">
      Automatic code optimization remains a difficult challenge, particularly for complex loop nests on modern hardware. This paper investigates a novel approach to code optimization where Large Language Models (LLMs) guide the process through a closed-loop interaction with a compiler. We present ComPilot, an experimental framework that leverages off-the-shelf LLMs, without any task-specific fine-tuning, as interactive optimization agents. ComPilot establishes a feedback loop where an LLM proposes transformations for a given loop nest to a compiler. The compiler attempts the transformations, reporting back legality status and measured speedup or slowdown. The LLM utilizes this concrete feedback to iteratively refine its optimization strategy. Our extensive evaluation across the PolyBench benchmark suite demonstrates the effectiveness of this zero-shot approach. ComPilot achieves geometric mean speedups of 2.66x (single run) and 3.54x (best-of-5 runs) over the original code. Furthermore, ComPilot demonstrates competitive performance against the state-of-the-art Pluto polyhedral optimizer, outperforming it in many cases. This experimental study demonstrates that general-purpose LLMs can effectively guide the code optimization process when grounded by compiler feedback, opening promising research directions for agentic AI in code optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.00556v1">Friend or Foe: How LLMs' Safety Mind Gets Fooled by Intent Shift Attack</a></div>
    <div class="paper-meta">
      📅 2025-11-01
      | 💬 Preprint, 14 pages, 5 figures, 7 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) remain vulnerable to jailbreaking attacks despite their impressive capabilities. Investigating these weaknesses is crucial for robust safety mechanisms. Existing attacks primarily distract LLMs by introducing additional context or adversarial tokens, leaving the core harmful intent unchanged. In this paper, we introduce ISA (Intent Shift Attack), which obfuscates LLMs about the intent of the attacks. More specifically, we establish a taxonomy of intent transformations and leverage them to generate attacks that may be misperceived by LLMs as benign requests for information. Unlike prior methods relying on complex tokens or lengthy context, our approach only needs minimal edits to the original request, and yields natural, human-readable, and seemingly harmless prompts. Extensive experiments on both open-source and commercial LLMs show that ISA achieves over 70% improvement in attack success rate compared to direct harmful prompts. More critically, fine-tuning models on only benign data reformulated with ISA templates elevates success rates to nearly 100%. For defense, we evaluate existing methods and demonstrate their inadequacy against ISA, while exploring both training-free and training-based mitigation strategies. Our findings reveal fundamental challenges in intent inference for LLMs safety and underscore the need for more effective defenses. Our code and datasets are available at https://github.com/NJUNLP/ISA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.00554v1">Red-teaming Activation Probes using Prompted LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-01
    </div>
    <details class="paper-abstract">
      Activation probes are attractive monitors for AI systems due to low cost and latency, but their real-world robustness remains underexplored. We ask: What failure modes arise under realistic, black-box adversarial pressure, and how can we surface them with minimal effort? We present a lightweight black-box red-teaming procedure that wraps an off-the-shelf LLM with iterative feedback and in-context learning (ICL), and requires no fine-tuning, gradients, or architectural access. Running a case study with probes for high-stakes interactions, we show that our approach can help discover valuable insights about a SOTA probe. Our analysis uncovers interpretable brittleness patterns (e.g., legalese-induced FPs; bland procedural tone FNs) and reduced but persistent vulnerabilities under scenario-constraint attacks. These results suggest that simple prompted red-teaming scaffolding can anticipate failure patterns before deployment and might yield promising, actionable insights to harden future probes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.00488v1">\texttt{ReMind}: Understanding Deductive Code Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-01
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable progress in code-related tasks. Despite their advancement, empirical evidence reveals that they still struggle with \emph{deductive code reasoning}, the ability to reason about the program execution process. While prior studies have recognized this limitation, the underlying causes remain largely underexplored. In this paper, we begin by presenting a comprehensive empirical study that reveals three key challenges undermining deductive code reasoning: (1) an intrinsic gap between generation and reasoning abilities, (2) a consistent bias towards code sources, and (3) weak zero-shot generalization on complex benchmarks. In light of these challenges, we propose \texttt{ReMind}, a multi-agent framework composed of \texttt{Mutator}, \texttt{Executor}, and \texttt{Inspector}. The \texttt{Mutator} generates code variants to mitigate bias towards code sources, the \texttt{Executor} traces variable states step-by-step to expose inconsistency, and the \texttt{Inspector} identifies problematic reasoning steps and provides control-flow refinement to bridge the intrinsic reasoning gap. Through their coordinated collaboration, \texttt{ReMind} systematically identifies and refines reasoning flaws, achieving outstanding performance and enabling robust zero-shot generalization. Extensive experiments on two benchmarks with five LLMs demonstrate the superior advantages of \texttt{ReMind} compared to baseline approaches in deductive code reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.00476v1">Remembering Unequally: Global and Disciplinary Bias in LLM-Generated Co-Authorship Networks</a></div>
    <div class="paper-meta">
      📅 2025-11-01
    </div>
    <details class="paper-abstract">
      Ongoing breakthroughs in Large Language Models (LLMs) are reshaping search and recommendation platforms at their core. While this shift unlocks powerful new scientometric tools, it also exposes critical fairness and bias issues that could erode the integrity of the information ecosystem. Additionally, as LLMs become more integrated into web-based searches for scholarly tools, their ability to generate summarized research work based on memorized data introduces new dimensions to these challenges. The extent of memorization in LLMs can impact the accuracy and fairness of the co-authorship networks they produce, potentially reflecting and amplifying existing biases within the scientific community and across different regions. This study critically examines the impact of LLM memorization on the co-authorship networks. To this end, we assess memorization effects across three prominent models, DeepSeek R1, Llama 4 Scout, and Mixtral 8x7B, analyzing how memorization-driven outputs vary across academic disciplines and world regions. While our global analysis reveals a consistent bias favoring highly cited researchers, this pattern is not uniformly observed. Certain disciplines, such as Clinical Medicine, and regions, including parts of Africa, show more balanced representation, pointing to areas where LLM training data may reflect greater equity. These findings underscore both the risks and opportunities in deploying LLMs for scholarly discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.00432v1">G2: Guided Generation for Enhanced Output Diversity in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-01
      | 💬 EMNLP 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated exceptional performance across diverse natural language processing tasks. However, these models exhibit a critical limitation in output diversity, often generating highly similar content across multiple attempts. This limitation significantly affects tasks requiring diverse outputs, from creative writing to reasoning. Existing solutions, like temperature scaling, enhance diversity by modifying probability distributions but compromise output quality. We propose Guide-to-Generation (G2), a training-free plug-and-play method that enhances output diversity while preserving generation quality. G2 employs a base generator alongside dual Guides, which guide the generation process through decoding-based interventions to encourage more diverse outputs conditioned on the original query. Comprehensive experiments demonstrate that G2 effectively improves output diversity while maintaining an optimal balance between diversity and quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.00413v1">Tree Training: Accelerating Agentic LLMs Training via Shared Prefix Reuse</a></div>
    <div class="paper-meta">
      📅 2025-11-01
    </div>
    <details class="paper-abstract">
      In agentic LLM scenarios, an agent's interaction process during a single rollout often exhibits branching behaviors. Due to memory retrieval and concurrent tool executions at certain decision points, the token trajectory of one task evolves into a tree-like structure rather than a linear sequence. However, current training pipelines decompose such tree-structured trajectories into separate linear segments, treating each branch as an independent sequence. As a result, shared prefixes across these branches are repeatedly recomputed during both forward and backward passes. To address this inefficiency, we propose Tree Training, a paradigm that computes each shared prefix only once and reuses its intermediate results across related branches during both forward and backward passes, substantially improving computation efficiency in large-scale agentic training. This is achieved via (i) Tree Packing, which efficiently reuses shared computations across trajectories, and (ii) Gradient Restoration, which ensures correct gradient propagation across reused prefixes. Experiments on multiple open-source models demonstrate up to 3.9x reduction in total training time, enabling more efficient agentic LLM SFT and RL training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.00382v1">Efficiency vs. Alignment: Investigating Safety and Fairness Risks in Parameter-Efficient Fine-Tuning of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-01
    </div>
    <details class="paper-abstract">
      Organizations are increasingly adopting and adapting Large Language Models (LLMs) hosted on public repositories such as HuggingFace. Although these adaptations often improve performance on specialized downstream tasks, recent evidence indicates that they can also degrade a model's safety or fairness. Since different fine-tuning techniques may exert distinct effects on these critical dimensions, this study undertakes a systematic assessment of their trade-offs. Four widely used Parameter-Efficient Fine-Tuning methods, LoRA, IA3, Prompt-Tuning, and P-Tuning, are applied to four instruction-tuned model families (Meta-Llama-3-8B, Qwen2.5-7B, Mistral-7B, and Gemma-7B). In total, 235 fine-tuned variants are evaluated across eleven safety hazard categories and nine demographic fairness dimensions. The results show that adapter-based approaches (LoRA, IA3) tend to improve safety scores and are the least disruptive to fairness, retaining higher accuracy and lower bias scores. In contrast, prompt-based methods (Prompt-Tuning and P-Tuning) generally reduce safety and cause larger fairness regressions, with decreased accuracy and increased bias. Alignment shifts are strongly moderated by base model type: LLaMA remains stable, Qwen records modest gains, Gemma experiences the steepest safety decline, and Mistral, which is released without an internal moderation layer, displays the greatest variance. Improvements in safety do not necessarily translate into improvements in fairness, and no single configuration optimizes all fairness metrics simultaneously, indicating an inherent trade-off between these objectives. These findings suggest a practical guideline for safety-critical deployments: begin with a well-aligned base model, favour adapter-based PEFT, and conduct category-specific audits of both safety and fairness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.00346v1">Exploiting Latent Space Discontinuities for Building Universal LLM Jailbreaks and Data Extraction Attacks</a></div>
    <div class="paper-meta">
      📅 2025-11-01
      | 💬 10 pages, 5 figures, 4 tables, Published at the Brazilian Symposium on Cybersecurity (SBSeg 2025)
    </div>
    <details class="paper-abstract">
      The rapid proliferation of Large Language Models (LLMs) has raised significant concerns about their security against adversarial attacks. In this work, we propose a novel approach to crafting universal jailbreaks and data extraction attacks by exploiting latent space discontinuities, an architectural vulnerability related to the sparsity of training data. Unlike previous methods, our technique generalizes across various models and interfaces, proving highly effective in seven state-of-the-art LLMs and one image generation model. Initial results indicate that when these discontinuities are exploited, they can consistently and profoundly compromise model behavior, even in the presence of layered defenses. The findings suggest that this strategy has substantial potential as a systemic attack vector.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.00343v1">LingGym: How Far Are LLMs from Thinking Like Field Linguists?</a></div>
    <div class="paper-meta">
      📅 2025-11-01
      | 💬 EMNLP 2025 Main
    </div>
    <details class="paper-abstract">
      This paper introduces LingGym, a new benchmark that evaluates LLMs' capacity for meta-linguistic reasoning using Interlinear Glossed Text (IGT) and grammatical descriptions extracted from 18 typologically diverse reference grammars. Unlike previous work that focuses on specific downstream tasks, we assess whether LLMs can generalize linguistic inference across low-resource languages and structures not seen during training. We present a controlled evaluation task: Word-Gloss Inference, in which the model must infer a missing word and gloss from context using varying levels of linguistic information (e.g., glosses, grammatical explanations, translations). Our results show that incorporating structured linguistic cues leads to consistent improvements in reasoning performance across all models. This work highlights both the promise and current limitations of using LLMs for typologically informed linguistic analysis and low-resource language documentation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.00340v1">Better Call CLAUSE: A Discrepancy Benchmark for Auditing LLMs Legal Reasoning Capabilities</a></div>
    <div class="paper-meta">
      📅 2025-11-01
      | 💬 41 pages, 4 images
    </div>
    <details class="paper-abstract">
      The rapid integration of large language models (LLMs) into high-stakes legal work has exposed a critical gap: no benchmark exists to systematically stress-test their reliability against the nuanced, adversarial, and often subtle flaws present in real-world contracts. To address this, we introduce CLAUSE, a first-of-its-kind benchmark designed to evaluate the fragility of an LLM's legal reasoning. We study the capabilities of LLMs to detect and reason about fine-grained discrepancies by producing over 7500 real-world perturbed contracts from foundational datasets like CUAD and ContractNLI. Our novel, persona-driven pipeline generates 10 distinct anomaly categories, which are then validated against official statutes using a Retrieval-Augmented Generation (RAG) system to ensure legal fidelity. We use CLAUSE to evaluate leading LLMs' ability to detect embedded legal flaws and explain their significance. Our analysis shows a key weakness: these models often miss subtle errors and struggle even more to justify them legally. Our work outlines a path to identify and correct such reasoning failures in legal AI.
    </details>
</div>
