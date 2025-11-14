# llm - 2025_09

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
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
- [Part 12](papers_12.md)
- Part 13
- [Part 14](papers_14.md)
- [Part 15](papers_15.md)
- [Part 16](papers_16.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03020v3">Training LLMs to be Better Text Embedders through Bidirectional Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-09-09
      | 💬 accepted by EMNLP 2025 Main Conference
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have increasingly been explored as powerful text embedders. Existing LLM-based text embedding approaches often leverage the embedding of the final token, typically a reserved special token such as [EOS]. However, these tokens have not been intentionally trained to capture the semantics of the whole context, limiting their capacity as text embeddings, especially for retrieval and re-ranking tasks. We propose to add a new training stage before contrastive learning to enrich the semantics of the final token embedding. This stage employs bidirectional generative reconstruction tasks, namely EBQ2D (Embedding-Based Query-to-Document) and EBD2Q (Embedding-Based Document-to-Query), which interleave to anchor the [EOS] embedding and reconstruct either side of Query-Document pairs. Experimental results demonstrate that our additional training stage significantly improves LLM performance on the Massive Text Embedding Benchmark (MTEB), achieving new state-of-the-art results across different LLM base models and scales.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05062v2">Debatable Intelligence: Benchmarking LLM Judges via Debate Speech Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-09-09
      | 💬 EMNLP 2025. Code: https://github.com/noy-sternlicht/Debatable-Intelligence
    </div>
    <details class="paper-abstract">
      We introduce Debate Speech Evaluation as a novel and challenging benchmark for assessing LLM judges. Evaluating debate speeches requires a deep understanding of the speech at multiple levels, including argument strength and relevance, the coherence and organization of the speech, the appropriateness of its style and tone, and so on. This task involves a unique set of cognitive abilities that previously received limited attention in systematic LLM benchmarking. To explore such skills, we leverage a dataset of over 600 meticulously annotated debate speeches and present the first in-depth analysis of how state-of-the-art LLMs compare to human judges on this task. Our findings reveal a nuanced picture: while larger models can approximate individual human judgments in some respects, they differ substantially in their overall judgment behavior. We also investigate the ability of frontier LLMs to generate persuasive, opinionated speeches, showing that models may perform at a human level on this task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07445v1">Text2Touch: Tactile In-Hand Manipulation with LLM-Designed Reward Functions</a></div>
    <div class="paper-meta">
      📅 2025-09-09
      | 💬 Accepted at CoRL 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are beginning to automate reward design for dexterous manipulation. However, no prior work has considered tactile sensing, which is known to be critical for human-like dexterity. We present Text2Touch, bringing LLM-crafted rewards to the challenging task of multi-axis in-hand object rotation with real-world vision based tactile sensing in palm-up and palm-down configurations. Our prompt engineering strategy scales to over 70 environment variables, and sim-to-real distillation enables successful policy transfer to a tactile-enabled fully actuated four-fingered dexterous robot hand. Text2Touch significantly outperforms a carefully tuned human-engineered baseline, demonstrating superior rotation speed and stability while relying on reward functions that are an order of magnitude shorter and simpler. These results illustrate how LLM-designed rewards can significantly reduce the time from concept to deployable dexterous tactile skills, supporting more rapid and scalable multimodal robot learning. Project website: https://hpfield.github.io/text2touch-website
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20024v2">Using item recommendations and LLMs in marketing email titles</a></div>
    <div class="paper-meta">
      📅 2025-09-09
      | 💬 Accepted to The Second Workshop on Generative AI for E-commerce (GenAIECommerce '25), held September 22, 2025, in Prague, Czech Republic. 3 figures
    </div>
    <details class="paper-abstract">
      E-commerce marketplaces make use of a number of marketing channels like emails, push notifications, etc. to reach their users and stimulate purchases. Personalized emails especially are a popular touch point for marketers to inform users of latest items in stock, especially for those who stopped visiting the marketplace. Such emails contain personalized recommendations tailored to each user's interests, enticing users to buy relevant items. A common limitation of these emails is that the primary entry point, the title of the email, tends to follow fixed templates, failing to inspire enough interest in the contents. In this work, we explore the potential of large language models (LLMs) for generating thematic titles that reflect the personalized content of the emails. We perform offline simulations and conduct online experiments on the order of millions of users, finding our techniques useful in improving the engagement between customers and our emails. We highlight key findings and learnings as we productionize the safe and automated generation of email titles for millions of users.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07389v1">Talking with Oompa Loompas: A novel framework for evaluating linguistic acquisition of LLM agents</a></div>
    <div class="paper-meta">
      📅 2025-09-09
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Existing evaluation studies on linguistic competence of large language models (LLM agents) have focused primarily on vocabulary learning, morphological rule induction, syntactic generalization, pragmatic inference, and cross-linguistic transfer. However, none assess whether LLM agents can acquire a language through pattern recognition and interactive feedback, a central feature of human language acquisition. We propose a novel experimental framework in which an LLM agent is evaluated on its ability to acquire and use a newly constructed language (Tinkatongue) in conversation with a bot that understands only Tinkatongue. Our findings show that LLM agents fail to establish a conversation within 100 responses, yet they adopt distinct strategies that mirror human approaches to language learning. The results suggest a new direction for evaluation benchmarks and open pathways to model designs that learn more effectively from interactive feedback.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17450v3">Persuasion Dynamics in LLMs: Investigating Robustness and Adaptability in Knowledge and Safety with DuET-PD</a></div>
    <div class="paper-meta">
      📅 2025-09-09
      | 💬 To appear at EMNLP 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can struggle to balance gullibility to misinformation and resistance to valid corrections in persuasive dialogues, a critical challenge for reliable deployment. We introduce DuET-PD (Dual Evaluation for Trust in Persuasive Dialogues), a framework evaluating multi-turn stance-change dynamics across dual dimensions: persuasion type (corrective/misleading) and domain (knowledge via MMLU-Pro, and safety via SALAD-Bench). We find that even a state-of-the-art model like GPT-4o achieves only 27.32% accuracy in MMLU-Pro under sustained misleading persuasions. Moreover, results reveal a concerning trend of increasing sycophancy in newer open-source models. To address this, we introduce Holistic DPO, a training approach balancing positive and negative persuasion examples. Unlike prompting or resist-only training, Holistic DPO enhances both robustness to misinformation and receptiveness to corrections, improving Llama-3.1-8B-Instruct's accuracy under misleading persuasion in safety contexts from 4.21% to 76.54%. These contributions offer a pathway to developing more reliable and adaptable LLMs for multi-turn dialogue. Code is available at https://github.com/Social-AI-Studio/DuET-PD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07379v1">DuoServe-MoE: Dual-Phase Expert Prefetch and Cache Scheduling for Efficient MoE LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-09-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive performance across a wide range of deep learning tasks. Mixture of Experts (MoE) further enhances their capabilities by increasing model width through sparsely activated expert branches, which keeps inference computation efficient. However, the large number of expert weights introduces significant GPU memory pressure, especially in resource-constrained environments such as single-GPU servers. More importantly, MoE inference consists of two fundamentally different stages: a prefill stage where most experts are activated densely, and a decode stage where only a few experts are triggered sparsely. Treating these stages with a uniform scheduling strategy often leads to suboptimal latency and memory usage. To address this, we propose DuoServe-MoE, an inference serving system that explicitly separates prefill and decode stages and applies tailored expert scheduling strategies to each. In the prefill stage, DuoServe-MoE uses a two-stream CUDA pipeline that overlaps expert weight prefetching with the computation of non-MoE layers, limiting expert residency in GPU memory. In the decode stage, a lightweight layer-level predictor trained offline from activation traces is used to prefetch only the most likely activated experts, without requiring any changes to the model. Experiments on 4-bit Mixtral-8x7B and 8x22B models show that DuoServe-MoE improves end-to-end latency by 1.42 to 7.54 times while keeping peak memory usage at only 15 percent of the full model size.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07370v1">PersonaFuse: A Personality Activation-Driven Framework for Enhancing Human-LLM Interactions</a></div>
    <div class="paper-meta">
      📅 2025-09-09
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) demonstrate remarkable capabilities across various fields. These developments have led to more direct communication between humans and LLMs in various situations, such as social companionship and psychological support. However, LLMs often exhibit limitations in emotional perception and social competence during real-world conversations. These limitations partly originate from their inability to adapt their communication style and emotional expression to different social and task contexts. In this work, we introduce PersonaFuse, a novel LLM post-training framework that enables LLMs to adapt and express different personalities for varying situations. Inspired by Trait Activation Theory and the Big Five personality model, PersonaFuse employs a Mixture-of-Expert architecture that combines persona adapters with a dynamic routing network, enabling contextual trait expression. Experimental results show that PersonaFuse substantially outperforms baseline models across multiple dimensions of social-emotional intelligence. Importantly, these gains are achieved without sacrificing general reasoning ability or model safety, which remain common limitations of direct prompting and supervised fine-tuning approaches. PersonaFuse also delivers consistent improvements in downstream human-centered applications, such as mental health counseling and review-based customer service. Finally, human preference evaluations against leading LLMs, including GPT-4o and DeepSeek, demonstrate that PersonaFuse achieves competitive response quality despite its comparatively smaller model size. These findings demonstrate that PersonaFuse~offers a theoretically grounded and practical approach for developing social-emotional enhanced LLMs, marking a significant advancement toward more human-centric AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05657v2">LM-Searcher: Cross-domain Neural Architecture Search with LLMs via Unified Numerical Encoding</a></div>
    <div class="paper-meta">
      📅 2025-09-09
      | 💬 EMNLP 2025 Main
    </div>
    <details class="paper-abstract">
      Recent progress in Large Language Models (LLMs) has opened new avenues for solving complex optimization problems, including Neural Architecture Search (NAS). However, existing LLM-driven NAS approaches rely heavily on prompt engineering and domain-specific tuning, limiting their practicality and scalability across diverse tasks. In this work, we propose LM-Searcher, a novel framework that leverages LLMs for cross-domain neural architecture optimization without the need for extensive domain-specific adaptation. Central to our approach is NCode, a universal numerical string representation for neural architectures, which enables cross-domain architecture encoding and search. We also reformulate the NAS problem as a ranking task, training LLMs to select high-performing architectures from candidate pools using instruction-tuning samples derived from a novel pruning-based subspace sampling strategy. Our curated dataset, encompassing a wide range of architecture-performance pairs, encourages robust and transferable learning. Comprehensive experiments demonstrate that LM-Searcher achieves competitive performance in both in-domain (e.g., CNNs for image classification) and out-of-domain (e.g., LoRA configurations for segmentation and generation) tasks, establishing a new paradigm for flexible and generalizable LLM-based architecture search. The datasets and models will be released at https://github.com/Ashone3/LM-Searcher.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13833v3">DistFlow: A Fully Distributed RL Framework for Scalable and Efficient LLM Post-Training</a></div>
    <div class="paper-meta">
      📅 2025-09-09
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has become the pivotal post-training technique for large language model (LLM). Effectively scaling reinforcement learning is now the key to unlocking advanced reasoning capabilities and ensuring safe, goal-aligned behavior in the most powerful LLMs. Mainstream frameworks usually employ a hybrid-controller architecture where a single-controller dispatches the overall execution logic and manages overall data transfer and the multi-controller executes distributed computation. For large-scale reinforcement learning, minor load imbalances can introduce significant bottlenecks, ultimately constraining the scalability of the system. To address this limitation, we introduce DistFlow, a novel, fully distributed RL framework designed to break scaling barrier. We adopt a multi-controller paradigm that dispatches data transfer and execution tasks to all workers, which eliminates the centralized node. This allows each worker to operate independently, leading to near-linear scalability up to 1024 GPUs and dramatic efficiency gains. Furthermore, our architecture decouples resource configuration from execution logic, allowing each worker to have a unique execution flow, offering significant flexibility for rapid and cost-effective algorithmic experimentation. Extensive experiments show that DistFlow achieves excellent linear scalability and up to a 7x end-to-end throughput improvement in specific scenarios over state-of-the-art (SOTA) frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07353v3">Benchmarking for Domain-Specific LLMs: A Case Study on Academia and Beyond</a></div>
    <div class="paper-meta">
      📅 2025-09-09
      | 💬 Accepted by EMNLP2025 Findings
    </div>
    <details class="paper-abstract">
      The increasing demand for domain-specific evaluation of large language models (LLMs) has led to the development of numerous benchmarks. These efforts often adhere to the principle of data scaling, relying on large corpora or extensive question-answer (QA) sets to ensure broad coverage. However, the impact of corpus and QA set design on the precision and recall of domain-specific LLM performance remains poorly understood. In this paper, we argue that data scaling is not always the optimal principle for domain-specific benchmark construction. Instead, we introduce Comp-Comp, an iterative benchmarking framework grounded in the principle of comprehensiveness and compactness. Comprehensiveness ensures semantic recall by covering the full breadth of the domain, while compactness improves precision by reducing redundancy and noise. To demonstrate the effectiveness of our approach, we present a case study conducted at a well-renowned university, resulting in the creation of PolyBench, a large-scale, high-quality academic benchmark. Although this study focuses on academia, the Comp-Comp framework is domain-agnostic and readily adaptable to a wide range of specialized fields. The source code and datasets can be accessed at https://github.com/Anya-RB-Chen/COMP-COMP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04310v2">EvoEmo: Towards Evolved Emotional Policies for LLM Agents in Multi-Turn Negotiation</a></div>
    <div class="paper-meta">
      📅 2025-09-09
    </div>
    <details class="paper-abstract">
      Recent research on Chain-of-Thought (CoT) reasoning in Large Language Models (LLMs) has demonstrated that agents can engage in \textit{complex}, \textit{multi-turn} negotiations, opening new avenues for agentic AI. However, existing LLM agents largely overlook the functional role of emotions in such negotiations, instead generating passive, preference-driven emotional responses that make them vulnerable to manipulation and strategic exploitation by adversarial counterparts. To address this gap, we present EvoEmo, an evolutionary reinforcement learning framework that optimizes dynamic emotional expression in negotiations. EvoEmo models emotional state transitions as a Markov Decision Process and employs population-based genetic optimization to evolve high-reward emotion policies across diverse negotiation scenarios. We further propose an evaluation framework with two baselines -- vanilla strategies and fixed-emotion strategies -- for benchmarking emotion-aware negotiation. Extensive experiments and ablation studies show that EvoEmo consistently outperforms both baselines, achieving higher success rates, higher efficiency, and increased buyer savings. This findings highlight the importance of adaptive emotional expression in enabling more effective LLM agents for multi-turn negotiation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07315v1">SafeToolBench: Pioneering a Prospective Benchmark to Evaluating Tool Utilization Safety in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-09
      | 💬 18 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have exhibited great performance in autonomously calling various tools in external environments, leading to better problem solving and task automation capabilities. However, these external tools also amplify potential risks such as financial loss or privacy leakage with ambiguous or malicious user instructions. Compared to previous studies, which mainly assess the safety awareness of LLMs after obtaining the tool execution results (i.e., retrospective evaluation), this paper focuses on prospective ways to assess the safety of LLM tool utilization, aiming to avoid irreversible harm caused by directly executing tools. To this end, we propose SafeToolBench, the first benchmark to comprehensively assess tool utilization security in a prospective manner, covering malicious user instructions and diverse practical toolsets. Additionally, we propose a novel framework, SafeInstructTool, which aims to enhance LLMs' awareness of tool utilization security from three perspectives (i.e., \textit{User Instruction, Tool Itself, and Joint Instruction-Tool}), leading to nine detailed dimensions in total. We experiment with four LLMs using different methods, revealing that existing approaches fail to capture all risks in tool utilization. In contrast, our framework significantly enhances LLMs' self-awareness, enabling a more safe and trustworthy tool utilization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.08140v1">From Limited Data to Rare-event Prediction: LLM-powered Feature Engineering and Multi-model Learning in Venture Capital</a></div>
    <div class="paper-meta">
      📅 2025-09-09
      | 💬 6 pages, 3 figures
    </div>
    <details class="paper-abstract">
      This paper presents a framework for predicting rare, high-impact outcomes by integrating large language models (LLMs) with a multi-model machine learning (ML) architecture. The approach combines the predictive strength of black-box models with the interpretability required for reliable decision-making. We use LLM-powered feature engineering to extract and synthesize complex signals from unstructured data, which are then processed within a layered ensemble of models including XGBoost, Random Forest, and Linear Regression. The ensemble first produces a continuous estimate of success likelihood, which is then thresholded to produce a binary rare-event prediction. We apply this framework to the domain of Venture Capital (VC), where investors must evaluate startups with limited and noisy early-stage data. The empirical results show strong performance: the model achieves precision between 9.8X and 11.1X the random classifier baseline in three independent test subsets. Feature sensitivity analysis further reveals interpretable success drivers: the startup's category list accounts for 15.6% of predictive influence, followed by the number of founders, while education level and domain expertise contribute smaller yet consistent effects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03814v4">Recursive Training Loops in LLMs: How training data properties modulate distribution shift in generated data?</a></div>
    <div class="paper-meta">
      📅 2025-09-09
      | 💬 Accepted to EMNLP 2025 (Oral)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used in the creation of online content, creating feedback loops as subsequent generations of models will be trained on this synthetic data. Such loops were shown to lead to distribution shifts - models misrepresenting the true underlying distributions of human data (also called model collapse). However, how human data properties affect such shifts remains poorly understood. In this paper, we provide the first empirical examination of the effect of such properties on the outcome of recursive training. We first confirm that using different human datasets leads to distribution shifts of different magnitudes. Through exhaustive manipulation of dataset properties combined with regression analyses, we then identify a set of properties predicting distribution shift magnitudes. Lexical diversity is found to amplify these shifts, while semantic diversity and data quality mitigate them. Furthermore, we find that these influences are highly modular: data scrapped from a given internet domain has little influence on the content generated for another domain. Finally, experiments on political bias reveal that human data properties affect whether the initial bias will be amplified or reduced. Overall, our results portray a novel view, where different parts of internet may undergo different types of distribution shift.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.23035v3">KLLM: Fast LLM Inference with K-Means Quantization</a></div>
    <div class="paper-meta">
      📅 2025-09-09
    </div>
    <details class="paper-abstract">
      Large language model (LLM) inference poses significant challenges due to its intensive memory and computation demands. Weight and activation quantization (WAQ) offers a promising solution by reducing both memory footprint and arithmetic complexity. Traditional WAQ designs rely on uniform integer quantization for hardware efficiency, but often suffer from significant model performance degradation at low precision. In contrast, K-Means quantization, a non-uniform technique, achieves higher accuracy by aligning with the Gaussian-like distributions of weights and activations in LLMs. However, two key challenges prevent the efficient deployment of K-Means-based WAQ designs for LLM inference: (1) The non-uniform structure of K-Means-quantized data precludes direct execution on low-precision compute units, necessitating dequantization and floating-point matrix multiplications (MatMuls) during inference. (2) Activation outliers hinder effective low-precision quantization. Offline thresholding methods for outlier detection degrade model performance substantially, while existing online detection techniques introduce significant runtime overhead. To address the aforementioned challenges and fully unleash the potential of K-Means-based WAQ for LLM inference, in this paper, we propose KLLM, an LLM inference accelerator for efficient execution with K-Means-quantized weights and activations. KLLM features an index-based computation scheme for efficient execution of MatMuls and nonlinear operations on K-Means-quantized data, which avoids most of the dequantization and full-precision computations. Moreover, KLLM incorporates a lightweight outlier detection engine, Orizuru, that efficiently identifies the top-$k$ largest and smallest elements in the activation data stream during online inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.08093v1">Culturally transmitted color categories in LLMs reflect a learning bias toward efficient compression</a></div>
    <div class="paper-meta">
      📅 2025-09-09
    </div>
    <details class="paper-abstract">
      Converging evidence suggests that systems of semantic categories across human languages achieve near-optimal compression via the Information Bottleneck (IB) complexity-accuracy principle. Large language models (LLMs) are not trained for this objective, which raises the question: are LLMs capable of evolving efficient human-like semantic systems? To address this question, we focus on the domain of color as a key testbed of cognitive theories of categorization and replicate with LLMs (Gemini 2.0-flash and Llama 3.3-70B-Instruct) two influential human behavioral studies. First, we conduct an English color-naming study, showing that Gemini aligns well with the naming patterns of native English speakers and achieves a significantly high IB-efficiency score, while Llama exhibits an efficient but lower complexity system compared to English. Second, to test whether LLMs simply mimic patterns in their training data or actually exhibit a human-like inductive bias toward IB-efficiency, we simulate cultural evolution of pseudo color-naming systems in LLMs via iterated in-context language learning. We find that akin to humans, LLMs iteratively restructure initially random systems towards greater IB-efficiency and increased alignment with patterns observed across the world's languages. These findings demonstrate that LLMs are capable of evolving perceptually grounded, human-like semantic systems, driven by the same fundamental principle that governs semantic efficiency across human languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.15463v2">Mind the Value-Action Gap: Do LLMs Act in Alignment with Their Values?</a></div>
    <div class="paper-meta">
      📅 2025-09-09
      | 💬 EMNLP 2025 Main Paper
    </div>
    <details class="paper-abstract">
      Existing research primarily evaluates the values of LLMs by examining their stated inclinations towards specific values. However, the "Value-Action Gap," a phenomenon rooted in environmental and social psychology, reveals discrepancies between individuals' stated values and their actions in real-world contexts. To what extent do LLMs exhibit a similar gap between their stated values and their actions informed by those values? This study introduces ValueActionLens, an evaluation framework to assess the alignment between LLMs' stated values and their value-informed actions. The framework encompasses the generation of a dataset comprising 14.8k value-informed actions across twelve cultures and eleven social topics, and two tasks to evaluate how well LLMs' stated value inclinations and value-informed actions align across three different alignment measures. Extensive experiments reveal that the alignment between LLMs' stated values and actions is sub-optimal, varying significantly across scenarios and models. Analysis of misaligned results identifies potential harms from certain value-action gaps. To predict the value-action gaps, we also uncover that leveraging reasoned explanations improves performance. These findings underscore the risks of relying solely on the LLMs' stated values to predict their behaviors and emphasize the importance of context-aware evaluations of LLM values and value-action gaps.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.08858v1">Decentralising LLM Alignment: A Case for Context, Pluralism, and Participation</a></div>
    <div class="paper-meta">
      📅 2025-09-09
      | 💬 Accepted at AIES 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) alignment methods have been credited with the commercial success of products like ChatGPT, given their role in steering LLMs towards user-friendly outputs. However, current alignment techniques predominantly mirror the normative preferences of a narrow reference group, effectively imposing their values on a wide user base. Drawing on theories of the power/knowledge nexus, this work argues that current alignment practices centralise control over knowledge production and governance within already influential institutions. To counter this, we propose decentralising alignment through three characteristics: context, pluralism, and participation. Furthermore, this paper demonstrates the critical importance of delineating the context-of-use when shaping alignment practices by grounding each of these features in concrete use cases. This work makes the following contributions: (1) highlighting the role of context, pluralism, and participation in decentralising alignment; (2) providing concrete examples to illustrate these strategies; and (3) demonstrating the nuanced requirements associated with applying alignment across different contexts of use. Ultimately, this paper positions LLM alignment as a potential site of resistance against epistemic injustice and the erosion of democratic processes, while acknowledging that these strategies alone cannot substitute for broader societal changes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07939v1">Guided Reasoning in LLM-Driven Penetration Testing Using Structured Attack Trees</a></div>
    <div class="paper-meta">
      📅 2025-09-09
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have driven interest in automating cybersecurity penetration testing workflows, offering the promise of faster and more consistent vulnerability assessment for enterprise systems. Existing LLM agents for penetration testing primarily rely on self-guided reasoning, which can produce inaccurate or hallucinated procedural steps. As a result, the LLM agent may undertake unproductive actions, such as exploiting unused software libraries or generating cyclical responses that repeat prior tactics. In this work, we propose a guided reasoning pipeline for penetration testing LLM agents that incorporates a deterministic task tree built from the MITRE ATT&CK Matrix, a proven penetration testing kll chain, to constrain the LLM's reaoning process to explicitly defined tactics, techniques, and procedures. This anchors reasoning in proven penetration testing methodologies and filters out ineffective actions by guiding the agent towards more productive attack procedures. To evaluate our approach, we built an automated penetration testing LLM agent using three LLMs (Llama-3-8B, Gemini-1.5, and GPT-4) and applied it to navigate 10 HackTheBox cybersecurity exercises with 103 discrete subtasks representing real-world cyberattack scenarios. Our proposed reasoning pipeline guided the LLM agent through 71.8\%, 72.8\%, and 78.6\% of subtasks using Llama-3-8B, Gemini-1.5, and GPT-4, respectively. Comparatively, the state-of-the-art LLM penetration testing tool using self-guided reasoning completed only 13.5\%, 16.5\%, and 75.7\% of subtasks and required 86.2\%, 118.7\%, and 205.9\% more model queries. This suggests that incorporating a deterministic task tree into LLM reasoning pipelines can enhance the accuracy and efficiency of automated cybersecurity assessments
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07933v1">Breaking Android with AI: A Deep Dive into LLM-Powered Exploitation</a></div>
    <div class="paper-meta">
      📅 2025-09-09
    </div>
    <details class="paper-abstract">
      The rapid evolution of Artificial Intelligence (AI) and Large Language Models (LLMs) has opened up new opportunities in the area of cybersecurity, especially in the exploitation automation landscape and penetration testing. This study explores Android penetration testing automation using LLM-based tools, especially PentestGPT, to identify and execute rooting techniques. Through a comparison of the traditional manual rooting process and exploitation methods produced using AI, this study evaluates the efficacy, reliability, and scalability of automated penetration testing in achieving high-level privilege access on Android devices. With the use of an Android emulator (Genymotion) as the testbed, we fully execute both traditional and exploit-based rooting methods, automating the process using AI-generated scripts. Secondly, we create a web application by integrating OpenAI's API to facilitate automated script generation from LLM-processed responses. The research focuses on the effectiveness of AI-enabled exploitation by comparing automated and manual penetration testing protocols, by determining LLM weaknesses and strengths along the way. We also provide security suggestions of AI-enabled exploitation, including ethical factors and potential misuse. The findings exhibit that while LLMs can significantly streamline the workflow of exploitation, they need to be controlled by humans to ensure accuracy and ethical application. This study adds to the increasing body of literature on AI-powered cybersecurity and its effect on ethical hacking, security research, and mobile device security.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07894v1">HiPhO: How Far Are (M)LLMs from Humans in the Latest High School Physics Olympiad Benchmark?</a></div>
    <div class="paper-meta">
      📅 2025-09-09
    </div>
    <details class="paper-abstract">
      Recently, the physical capabilities of (M)LLMs have garnered increasing attention. However, existing benchmarks for physics suffer from two major gaps: they neither provide systematic and up-to-date coverage of real-world physics competitions such as physics Olympiads, nor enable direct performance comparison with humans. To bridge these gaps, we present HiPhO, the first benchmark dedicated to high school physics Olympiads with human-aligned evaluation. Specifically, HiPhO highlights three key innovations. (1) Comprehensive Data: It compiles 13 latest Olympiad exams from 2024-2025, spanning both international and regional competitions, and covering mixed modalities that encompass problems spanning text-only to diagram-based. (2) Professional Evaluation: We adopt official marking schemes to perform fine-grained grading at both the answer and step level, fully aligned with human examiners to ensure high-quality and domain-specific evaluation. (3) Comparison with Human Contestants: We assign gold, silver, and bronze medals to models based on official medal thresholds, thereby enabling direct comparison between (M)LLMs and human contestants. Our large-scale evaluation of 30 state-of-the-art (M)LLMs shows that: across 13 exams, open-source MLLMs mostly remain at or below the bronze level; open-source LLMs show promising progress with occasional golds; closed-source reasoning MLLMs can achieve 6 to 12 gold medals; and most models still have a significant gap from full marks. These results highlight a substantial performance gap between open-source models and top students, the strong physical reasoning capabilities of closed-source reasoning models, and the fact that there is still significant room for improvement. HiPhO, as a rigorous, human-aligned, and Olympiad-focused benchmark for advancing multimodal physical reasoning, is open-source and available at https://github.com/SciYu/HiPhO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07128v2">Cardiverse: Harnessing LLMs for Novel Card Game Prototyping</a></div>
    <div class="paper-meta">
      📅 2025-09-09
      | 💬 37 pages, 13 figures, 8 tables. Accepted by EMNLP 2025
    </div>
    <details class="paper-abstract">
      The prototyping of computer games, particularly card games, requires extensive human effort in creative ideation and gameplay evaluation. Recent advances in Large Language Models (LLMs) offer opportunities to automate and streamline these processes. However, it remains challenging for LLMs to design novel game mechanics beyond existing databases, generate consistent gameplay environments, and develop scalable gameplay AI for large-scale evaluations. This paper addresses these challenges by introducing a comprehensive automated card game prototyping framework. The approach highlights a graph-based indexing method for generating novel game variations, an LLM-driven system for consistent game code generation validated by gameplay records, and a gameplay AI constructing method that uses an ensemble of LLM-generated heuristic functions optimized through self-play. These contributions aim to accelerate card game prototyping, reduce human labor, and lower barriers to entry for game developers. For code repo visit this http URL https://github.com/danruili/Cardiverse
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07860v1">KLIPA: A Knowledge Graph and LLM-Driven QA Framework for IP Analysis</a></div>
    <div class="paper-meta">
      📅 2025-09-09
    </div>
    <details class="paper-abstract">
      Effectively managing intellectual property is a significant challenge. Traditional methods for patent analysis depend on labor-intensive manual searches and rigid keyword matching. These approaches are often inefficient and struggle to reveal the complex relationships hidden within large patent datasets, hindering strategic decision-making. To overcome these limitations, we introduce KLIPA, a novel framework that leverages a knowledge graph and a large language model (LLM) to significantly advance patent analysis. Our approach integrates three key components: a structured knowledge graph to map explicit relationships between patents, a retrieval-augmented generation(RAG) system to uncover contextual connections, and an intelligent agent that dynamically determines the optimal strategy for resolving user queries. We validated KLIPA on a comprehensive, real-world patent database, where it demonstrated substantial improvements in knowledge extraction, discovery of novel connections, and overall operational efficiency. This combination of technologies enhances retrieval accuracy, reduces reliance on domain experts, and provides a scalable, automated solution for any organization managing intellectual property, including technology corporations and legal firms, allowing them to better navigate the complexities of strategic innovation and competitive intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07858v1">SCoder: Iterative Self-Distillation for Bootstrapping Small-Scale Data Synthesizers to Empower Code LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-09
    </div>
    <details class="paper-abstract">
      Existing code large language models (LLMs) often rely on large-scale instruction data distilled from proprietary LLMs for fine-tuning, which typically incurs high costs. In this paper, we explore the potential of small-scale open-source LLMs (e.g., 7B) as synthesizers for high-quality code instruction data construction. We first observe that the data synthesis capability of small-scale LLMs can be enhanced by training on a few superior data synthesis samples from proprietary LLMs. Building on this, we propose a novel iterative self-distillation approach to bootstrap small-scale LLMs, transforming them into powerful synthesizers that reduce reliance on proprietary LLMs and minimize costs. Concretely, in each iteration, to obtain diverse and high-quality self-distilled data, we design multi-checkpoint sampling and multi-aspect scoring strategies for initial data selection. Furthermore, to identify the most influential samples, we introduce a gradient-based influence estimation method for final data filtering. Based on the code instruction datasets from the small-scale synthesizers, we develop SCoder, a family of code generation models fine-tuned from DeepSeek-Coder. SCoder models achieve state-of-the-art code generation capabilities, demonstrating the effectiveness of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07846v1">Aligning LLMs for the Classroom with Knowledge-Based Retrieval -- A Comparative RAG Study</a></div>
    <div class="paper-meta">
      📅 2025-09-09
      | 💬 This work has been submitted to the IEEE for possible publication
    </div>
    <details class="paper-abstract">
      Large language models like ChatGPT are increasingly used in classrooms, but they often provide outdated or fabricated information that can mislead students. Retrieval Augmented Generation (RAG) improves reliability of LLMs by grounding responses in external resources. We investigate two accessible RAG paradigms, vector-based retrieval and graph-based retrieval to identify best practices for classroom question answering (QA). Existing comparative studies fail to account for pedagogical factors such as educational disciplines, question types, and practical deployment costs. Using a novel dataset, EduScopeQA, of 3,176 questions across academic subjects, we measure performance on various educational query types, from specific facts to broad thematic discussions. We also evaluate system alignment with a dataset of systematically altered textbooks that contradict the LLM's latent knowledge. We find that OpenAI Vector Search RAG (representing vector-based RAG) performs well as a low-cost generalist, especially for quick fact retrieval. On the other hand, GraphRAG Global excels at providing pedagogically rich answers to thematic queries, and GraphRAG Local achieves the highest accuracy with the dense, altered textbooks when corpus integrity is critical. Accounting for the 10-20x higher resource usage of GraphRAG (representing graph-based RAG), we show that a dynamic branching framework that routes queries to the optimal retrieval method boosts fidelity and efficiency. These insights provide actionable guidelines for educators and system designers to integrate RAG-augmented LLMs into learning environments effectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07819v1">LLMs in Wikipedia: Investigating How LLMs Impact Participation in Knowledge Communities</a></div>
    <div class="paper-meta">
      📅 2025-09-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are reshaping knowledge production as community members increasingly incorporate them into their contribution workflows. However, participating in knowledge communities involves more than just contributing content - it is also a deeply social process. While communities must carefully consider appropriate and responsible LLM integration, the absence of concrete norms has left individual editors to experiment and navigate LLM use on their own. Understanding how LLMs influence community participation is therefore critical in shaping future norms and supporting effective adoption. To address this gap, we investigated Wikipedia, one of the largest knowledge production communities, to understand 1) how LLMs influence the ways editors contribute content, 2) what strategies editors leverage to align LLM outputs with community norms, and 3) how other editors in the community respond to LLM-assisted contributions. Through interviews with 16 Wikipedia editors who had used LLMs for their edits, we found that 1) LLMs affected the content contributions for experienced and new editors differently; 2) aligning LLM outputs with community norms required tacit knowledge that often challenged newcomers; and 3) as a result, other editors responded to LLM-assisted edits differently depending on the editors' expertise level. Based on these findings, we challenge existing models of newcomer involvement and propose design implications for LLMs that support community engagement through scaffolding, teaching, and context awareness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07642v3">TreeReview: A Dynamic Tree of Questions Framework for Deep and Efficient LLM-based Scientific Peer Review</a></div>
    <div class="paper-meta">
      📅 2025-09-09
      | 💬 Accepted to EMNLP2025 Main
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have shown significant potential in assisting peer review, current methods often struggle to generate thorough and insightful reviews while maintaining efficiency. In this paper, we propose TreeReview, a novel framework that models paper review as a hierarchical and bidirectional question-answering process. TreeReview first constructs a tree of review questions by recursively decomposing high-level questions into fine-grained sub-questions and then resolves the question tree by iteratively aggregating answers from leaf to root to get the final review. Crucially, we incorporate a dynamic question expansion mechanism to enable deeper probing by generating follow-up questions when needed. We construct a benchmark derived from ICLR and NeurIPS venues to evaluate our method on full review generation and actionable feedback comments generation tasks. Experimental results of both LLM-based and human evaluation show that TreeReview outperforms strong baselines in providing comprehensive, in-depth, and expert-aligned review feedback, while reducing LLM token usage by up to 80% compared to computationally intensive approaches. Our code and benchmark dataset are available at https://github.com/YuanChang98/tree-review.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18934v2">Dovetail: A CPU/GPU Heterogeneous Speculative Decoding for LLM inference</a></div>
    <div class="paper-meta">
      📅 2025-09-09
      | 💬 14 pages, 6 figures
    </div>
    <details class="paper-abstract">
      With the continuous advancement in the performance of large language models (LLMs), their demand for computational resources and memory has significantly increased, which poses major challenges for efficient inference on consumer-grade devices and legacy servers. These devices typically feature relatively weaker GPUs and stronger CPUs. Although techniques such as parameter offloading and partial offloading can alleviate GPU memory pressure to some extent, their effectiveness is limited due to communication latency and suboptimal hardware resource utilization. To address this issue, we propose Dovetail, a lossless inference acceleration method that leverages the complementary characteristics of heterogeneous devices and the advantages of speculative decoding. Dovetail deploys a draft model on the GPU to perform preliminary predictions, while a target model running on the CPU validates these outputs. By reducing the granularity of data transfer, Dovetail significantly minimizes communication overhead. To further improve efficiency, we optimize the draft model specifically for heterogeneous hardware environments by reducing the number of draft tokens to lower parallel verification latency, increasing model depth to enhance predictive capabilities, and introducing a Dynamic Gating Fusion (DGF) mechanism to improve the integration of feature and embedding information. We conduct comprehensive evaluations of Dovetail across various consumer-grade GPUs, covering multiple tasks and mainstream models. Experimental results on 13B models demonstrate that Dovetail achieves inference speedups ranging from 1.79x to 10.1x across different devices, while maintaining consistency and stability in the distribution of generated texts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.07819v1">LLMs in Wikipedia: Investigating How LLMs Impact Participation in Knowledge Communities</a></div>
    <div class="paper-meta">
      📅 2025-09-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are reshaping knowledge production as community members increasingly incorporate them into their contribution workflows. However, participating in knowledge communities involves more than just contributing content - it is also a deeply social process. While communities must carefully consider appropriate and responsible LLM integration, the absence of concrete norms has left individual editors to experiment and navigate LLM use on their own. Understanding how LLMs influence community participation is therefore critical in shaping future norms and supporting effective adoption. To address this gap, we investigated Wikipedia, one of the largest knowledge production communities, to understand 1) how LLMs influence the ways editors contribute content, 2) what strategies editors leverage to align LLM outputs with community norms, and 3) how other editors in the community respond to LLM-assisted contributions. Through interviews with 16 Wikipedia editors who had used LLMs for their edits, we found that 1) LLMs affected the content contributions for experienced and new editors differently; 2) aligning LLM outputs with community norms required tacit knowledge that often challenged newcomers; and 3) as a result, other editors responded to LLM-assisted edits differently depending on the editors' expertise level. Based on these findings, we challenge existing models of newcomer involvement and propose design implications for LLMs that support community engagement through scaffolding, teaching, and context awareness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06332v1">A Fragile Number Sense: Probing the Elemental Limits of Numerical Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable emergent capabilities, yet the robustness of their numerical reasoning remains an open question. While standard benchmarks evaluate LLM reasoning on complex problem sets using aggregated metrics, they often obscure foundational weaknesses. In this work, we probe LLM mathematical numeracy by evaluating performance on problems of escalating complexity, from constituent operations to combinatorial puzzles. We test several state-of-the-art LLM-based agents on a 100-problem challenge comprising four categories: (1) basic arithmetic, (2) advanced operations, (3) primality checking, and (4) the Game of 24 number puzzle. Our results show that while the agents achieved high accuracy on the first three categories, which require deterministic algorithmic execution, they consistently failed at the number puzzle, underlining its demand for a heuristic search over a large combinatorial space to be a significant bottleneck. These findings reveal that the agents' proficiency is largely confined to recalling and executing known algorithms, rather than performing generative problem-solving. This suggests their apparent numerical reasoning is more akin to sophisticated pattern-matching than flexible, analytical thought, limiting their potential for tasks that require novel or creative numerical insights.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17656v3">Too Consistent to Detect: A Study of Self-Consistent Errors in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-08
      | 💬 EMNLP 2025 Main
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) often generate plausible but incorrect content, error detection has become increasingly critical to ensure truthfulness. However, existing detection methods often overlook a critical problem we term as self-consistent error, where LLMs repeatedly generate the same incorrect response across multiple stochastic samples. This work formally defines self-consistent errors and evaluates mainstream detection methods on them. Our investigation reveals two key findings: (1) Unlike inconsistent errors, whose frequency diminishes significantly as the LLM scale increases, the frequency of self-consistent errors remains stable or even increases. (2) All four types of detection methods significantly struggle to detect self-consistent errors. These findings reveal critical limitations in current detection methods and underscore the need for improvement. Motivated by the observation that self-consistent errors often differ across LLMs, we propose a simple but effective cross-model probe method that fuses hidden state evidence from an external verifier LLM. Our method significantly enhances performance on self-consistent errors across three LLM families.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06326v1">AttestLLM: Efficient Attestation Framework for Billion-scale On-device LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      As on-device LLMs(e.g., Apple on-device Intelligence) are widely adopted to reduce network dependency, improve privacy, and enhance responsiveness, verifying the legitimacy of models running on local devices becomes critical. Existing attestation techniques are not suitable for billion-parameter Large Language Models (LLMs), struggling to remain both time- and memory-efficient while addressing emerging threats in the LLM era. In this paper, we present AttestLLM, the first-of-its-kind attestation framework to protect the hardware-level intellectual property (IP) of device vendors by ensuring that only authorized LLMs can execute on target platforms. AttestLLM leverages an algorithm/software/hardware co-design approach to embed robust watermarking signatures onto the activation distributions of LLM building blocks. It also optimizes the attestation protocol within the Trusted Execution Environment (TEE), providing efficient verification without compromising inference throughput. Extensive proof-of-concept evaluations on LLMs from Llama, Qwen, and Phi families for on-device use cases demonstrate AttestLLM's attestation reliability, fidelity, and efficiency. Furthermore, AttestLLM enforces model legitimacy and exhibits resilience against model replacement and forgery attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06322v1">Text-Trained LLMs Can Zero-Shot Extrapolate PDE Dynamics</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated emergent in-context learning (ICL) capabilities across a range of tasks, including zero-shot time-series forecasting. We show that text-trained foundation models can accurately extrapolate spatiotemporal dynamics from discretized partial differential equation (PDE) solutions without fine-tuning or natural language prompting. Predictive accuracy improves with longer temporal contexts but degrades at finer spatial discretizations. In multi-step rollouts, where the model recursively predicts future spatial states over multiple time steps, errors grow algebraically with the time horizon, reminiscent of global error accumulation in classical finite-difference solvers. We interpret these trends as in-context neural scaling laws, where prediction quality varies predictably with both context length and output length. To better understand how LLMs are able to internally process PDE solutions so as to accurately roll them out, we analyze token-level output distributions and uncover a consistent ICL progression: beginning with syntactic pattern imitation, transitioning through an exploratory high-entropy phase, and culminating in confident, numerically grounded predictions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06298v1">MCTuner: Spatial Decomposition-Enhanced Database Tuning via LLM-Guided Exploration</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      Database knob tuning is essential for optimizing the performance of modern database management systems, which often expose hundreds of knobs with continuous or categorical values. However, the large number of knobs and the vast configuration space make it difficult to identify optimal settings efficiently. Although learning-based tuning has shown promise, existing approaches either ignore domain knowledge by relying solely on benchmark feedback or struggle to explore the high-dimensional knob space, resulting in high tuning costs and suboptimal performance. To address these challenges, we propose MCTuner, an adaptive knob tuning framework that minimizes exploration in ineffective regions of the configuration space. MCTuner employs a Mixture-of-Experts (MoE) mechanism with specialized LLMs to identify performance-critical knobs. In further, MCTuner introduces the first spatial decomposition algorithm that recursively partitions the space into hierarchical subspaces, on which Bayesian Optimization is performed to efficiently search for near-optimal configurations. Evaluated on different benchmarks (OLAP, OLTP, and HTAP), MCTuner achieves up to 19.2% performance gains and 1.4x faster configuration discovery per iteration compared to state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06286v1">RecMind: LLM-Enhanced Graph Neural Networks for Personalized Consumer Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      Personalization is a core capability across consumer technologies, streaming, shopping, wearables, and voice, yet it remains challenged by sparse interactions, fast content churn, and heterogeneous textual signals. We present RecMind, an LLM-enhanced graph recommender that treats the language model as a preference prior rather than a monolithic ranker. A frozen LLM equipped with lightweight adapters produces text-conditioned user/item embeddings from titles, attributes, and reviews; a LightGCN backbone learns collaborative embeddings from the user-item graph. We align the two views with a symmetric contrastive objective and fuse them via intra-layer gating, allowing language to dominate in cold/long-tail regimes and graph structure to stabilize rankings elsewhere. On Yelp and Amazon-Electronics, RecMind attains the best results on all eight reported metrics, with relative improvements up to +4.53\% (Recall@40) and +4.01\% (NDCG@40) over strong baselines. Ablations confirm both the necessity of cross-view alignment and the advantage of gating over late fusion and LLM-only variants.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06284v1">From Implicit Exploration to Structured Reasoning: Leveraging Guideline and Refinement for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have advanced general-purpose reasoning, showing strong performance across diverse tasks. However, existing methods often rely on implicit exploration, where the model follows stochastic and unguided reasoning paths-like walking without a map. This leads to unstable reasoning paths, lack of error correction, and limited learning from past experience. To address these issues, we propose a framework that shifts from implicit exploration to structured reasoning through guideline and refinement. First, we extract structured reasoning patterns from successful trajectories and reflective signals from failures. During inference, the model follows these guidelines step-by-step, with refinement applied after each step to correct errors and stabilize the reasoning process. Experiments on BBH and four additional benchmarks (GSM8K, MATH-500, MBPP, HumanEval) show that our method consistently outperforms strong baselines across diverse reasoning tasks. Structured reasoning with stepwise execution and refinement improves stability and generalization, while guidelines transfer well across domains and flexibly support cross-model collaboration, matching or surpassing supervised fine-tuning in effectiveness and scalability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06261v1">FineServe: Precision-Aware KV Slab and Two-Level Scheduling for Heterogeneous Precision LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      Recent advances in Post-Training Quantization (PTQ) techniques have significantly increased demand for serving quantized large language models (LLMs), enabling higher throughput and substantially reduced memory usage with minimal accuracy loss. Quantized models address memory constraints in LLMs and enhance GPU resource utilization through efficient GPU sharing. However, quantized models have smaller KV block sizes than non-quantized models, causing limited memory efficiency due to memory fragmentation. Also, distinct resource usage patterns between quantized and non-quantized models require efficient scheduling to maximize throughput. To address these challenges, we propose FineServe, an inference serving framework for mixed-precision LLMs. FineServe's key contributions include: (1) KV Slab, a precision-aware adaptive memory management technique dynamically allocating KV cache based on model quantization characteristics, significantly reducing GPU memory fragmentation, and (2) a two-level scheduling framework comprising a global scheduler that places models to GPUs based on request rates, latency SLOs, and memory constraints and efficiency, and a local scheduler that adaptively adjusts batch sizes according to real-time request fluctuations. Experimental results demonstrate that FineServe achieves up to 2.2x higher SLO attainment and 1.8x higher token generation throughput compared to the state-of-the-art GPU sharing systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05702v3">Grid-Agent: An LLM-Powered Multi-Agent System for Power Grid Control</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      Modern power grids face unprecedented complexity from Distributed Energy Resources (DERs), Electric Vehicles (EVs), and extreme weather, while also being increasingly exposed to cyberattacks that can trigger grid violations. This paper introduces Grid-Agent, an autonomous AI-driven framework that leverages Large Language Models (LLMs) within a multi-agent system to detect and remediate violations. Grid-Agent integrates semantic reasoning with numerical precision through modular agents: a planning agent generates coordinated action sequences using power flow solvers, while a validation agent ensures stability and safety through sandboxed execution with rollback mechanisms. To enhance scalability, the framework employs an adaptive multi-scale network representation that dynamically adjusts encoding schemes based on system size and complexity. Violation resolution is achieved through optimizing switch configurations, battery deployment, and load curtailment. Our experiments on IEEE and CIGRE benchmark networks, including the IEEE 69-bus, CIGRE MV, IEEE 30-bus test systems, demonstrate superior mitigation performance, highlighting Grid-Agent's suitability for modern smart grids requiring rapid, adaptive response.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07287v1">Paladin: Defending LLM-enabled Phishing Emails with a New Trigger-Tag Paradigm</a></div>
    <div class="paper-meta">
      📅 2025-09-08
      | 💬 20 pages
    </div>
    <details class="paper-abstract">
      With the rapid development of large language models, the potential threat of their malicious use, particularly in generating phishing content, is becoming increasingly prevalent. Leveraging the capabilities of LLMs, malicious users can synthesize phishing emails that are free from spelling mistakes and other easily detectable features. Furthermore, such models can generate topic-specific phishing messages, tailoring content to the target domain and increasing the likelihood of success. Detecting such content remains a significant challenge, as LLM-generated phishing emails often lack clear or distinguishable linguistic features. As a result, most existing semantic-level detection approaches struggle to identify them reliably. While certain LLM-based detection methods have shown promise, they suffer from high computational costs and are constrained by the performance of the underlying language model, making them impractical for large-scale deployment. In this work, we aim to address this issue. We propose Paladin, which embeds trigger-tag associations into vanilla LLM using various insertion strategies, creating them into instrumented LLMs. When an instrumented LLM generates content related to phishing, it will automatically include detectable tags, enabling easier identification. Based on the design on implicit and explicit triggers and tags, we consider four distinct scenarios in our work. We evaluate our method from three key perspectives: stealthiness, effectiveness, and robustness, and compare it with existing baseline methods. Experimental results show that our method outperforms the baselines, achieving over 90% detection accuracy across all scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07274v1">LLM Analysis of 150+ years of German Parliamentary Debates on Migration Reveals Shift from Post-War Solidarity to Anti-Solidarity in the Last Decade</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      Migration has been a core topic in German political debate, from millions of expellees post World War II over labor migration to refugee movements in the recent past. Studying political speech regarding such wide-ranging phenomena in depth traditionally required extensive manual annotations, limiting the scope of analysis to small subsets of the data. Large language models (LLMs) have the potential to partially automate even complex annotation tasks. We provide an extensive evaluation of a multiple LLMs in annotating (anti-)solidarity subtypes in German parliamentary debates compared to a large set of thousands of human reference annotations (gathered over a year). We evaluate the influence of model size, prompting differences, fine-tuning, historical versus contemporary data; and we investigate systematic errors. Beyond methodological evaluation, we also interpret the resulting annotations from a social science lense, gaining deeper insight into (anti-)solidarity trends towards migrants in the German post-World War II period and recent past. Our data reveals a high degree of migrant-directed solidarity in the postwar period, as well as a strong trend towards anti-solidarity in the German parliament since 2015, motivating further research. These findings highlight the promise of LLMs for political text analysis and the importance of migration debates in Germany, where demographic decline and labor shortages coexist with rising polarization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04373v2">Measuring Bias or Measuring the Task: Understanding the Brittle Nature of LLM Gender Biases</a></div>
    <div class="paper-meta">
      📅 2025-09-08
      | 💬 To be published at EMNLP 2025 (main conference)
    </div>
    <details class="paper-abstract">
      As LLMs are increasingly applied in socially impactful settings, concerns about gender bias have prompted growing efforts both to measure and mitigate such bias. These efforts often rely on evaluation tasks that differ from natural language distributions, as they typically involve carefully constructed task prompts that overtly or covertly signal the presence of gender bias-related content. In this paper, we examine how signaling the evaluative purpose of a task impacts measured gender bias in LLMs.Concretely, we test models under prompt conditions that (1) make the testing context salient, and (2) make gender-focused content salient. We then assess prompt sensitivity across four task formats with both token-probability and discrete-choice metrics. We find that prompts that more clearly align with (gender bias) evaluation framing elicit distinct gender output distributions compared to less evaluation-framed prompts. Discrete-choice metrics further tend to amplify bias relative to probabilistic measures. These findings do not only highlight the brittleness of LLM gender bias evaluations but open a new puzzle for the NLP benchmarking and development community: To what extent can well-controlled testing designs trigger LLM "testing mode" performance, and what does this mean for the ecological validity of future benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04795v2">Enhancing Dialogue Annotation with Speaker Characteristics Leveraging a Frozen LLM</a></div>
    <div class="paper-meta">
      📅 2025-09-08
      | 💬 Accepted in the 2025 IEEE Automatic Speech Recognition and Understanding Workshop
    </div>
    <details class="paper-abstract">
      In dialogue transcription pipelines, Large Language Models (LLMs) are frequently employed in post-processing to improve grammar, punctuation, and readability. We explore a complementary post-processing step: enriching transcribed dialogues by adding metadata tags for speaker characteristics such as age, gender, and emotion. Some of the tags are global to the entire dialogue, while some are time-variant. Our approach couples frozen audio foundation models, such as Whisper or WavLM, with a frozen LLAMA language model to infer these speaker attributes, without requiring task-specific fine-tuning of either model. Using lightweight, efficient connectors to bridge audio and language representations, we achieve competitive performance on speaker profiling tasks while preserving modularity and speed. Additionally, we demonstrate that a frozen LLAMA model can compare x-vectors directly, achieving an Equal Error Rate of 8.8% in some scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15552v2">Personalized Attacks of Social Engineering in Multi-turn Conversations: LLM Agents for Simulation and Detection</a></div>
    <div class="paper-meta">
      📅 2025-09-08
      | 💬 Accepted as a paper at COLM 2025 Workshop on AI Agents: Capabilities and Safety
    </div>
    <details class="paper-abstract">
      The rapid advancement of conversational agents, particularly chatbots powered by Large Language Models (LLMs), poses a significant risk of social engineering (SE) attacks on social media platforms. SE detection in multi-turn, chat-based interactions is considerably more complex than single-instance detection due to the dynamic nature of these conversations. A critical factor in mitigating this threat is understanding the SE attack mechanisms through which SE attacks operate, specifically how attackers exploit vulnerabilities and how victims' personality traits contribute to their susceptibility. In this work, we propose an LLM-agentic framework, SE-VSim, to simulate SE attack mechanisms by generating multi-turn conversations. We model victim agents with varying personality traits to assess how psychological profiles influence susceptibility to manipulation. Using a dataset of over 1000 simulated conversations, we examine attack scenarios in which adversaries, posing as recruiters, funding agencies, and journalists, attempt to extract sensitive information. Based on this analysis, we present a proof of concept, SE-OmniGuard, to offer personalized protection to users by leveraging prior knowledge of the victims personality, evaluating attack strategies, and monitoring information exchanges in conversations to identify potential SE attempts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07225v1">All You Need Is A Fuzzing Brain: An LLM-Powered System for Automated Vulnerability Detection and Patching</a></div>
    <div class="paper-meta">
      📅 2025-09-08
      | 💬 14 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Our team, All You Need Is A Fuzzing Brain, was one of seven finalists in DARPA's Artificial Intelligence Cyber Challenge (AIxCC), placing fourth in the final round. During the competition, we developed a Cyber Reasoning System (CRS) that autonomously discovered 28 security vulnerabilities - including six previously unknown zero-days - in real-world open-source C and Java projects, and successfully patched 14 of them. The complete CRS is open source at https://github.com/o2lab/afc-crs-all-you-need-is-a-fuzzing-brain. This paper provides a detailed technical description of our CRS, with an emphasis on its LLM-powered components and strategies. Building on AIxCC, we further introduce a public leaderboard for benchmarking state-of-the-art LLMs on vulnerability detection and patching tasks, derived from the AIxCC dataset. The leaderboard is available at https://o2lab.github.io/FuzzingBrain-Leaderboard/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07793v2">Overflow Prevention Enhances Long-Context Recurrent LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-08
      | 💬 Official Implementation: https://github.com/assafbk/OPRM
    </div>
    <details class="paper-abstract">
      A recent trend in LLMs is developing recurrent sub-quadratic models that improve long-context processing efficiency. We investigate leading large long-context models, focusing on how their fixed-size recurrent memory affects their performance. Our experiments reveal that, even when these models are trained for extended contexts, their use of long contexts remains underutilized. Specifically, we demonstrate that a chunk-based inference procedure, which identifies and processes only the most relevant portion of the input can mitigate recurrent memory failures and be effective for many long-context tasks: On LongBench, our method improves the overall performance of Falcon3-Mamba-Inst-7B by 14%, Falcon-Mamba-Inst-7B by 28%, RecurrentGemma-IT-9B by 50%, and RWKV6-Finch-7B by 51%. Surprisingly, this simple approach also leads to state-of-the-art results in the challenging LongBench v2 benchmark, showing competitive performance with equivalent size Transformers. Furthermore, our findings raise questions about whether recurrent models genuinely exploit long-range dependencies, as our single-chunk strategy delivers stronger performance - even in tasks that presumably require cross-context relations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07170v1">That's So FETCH: Fashioning Ensemble Techniques for LLM Classification in Civil Legal Intake and Referral</a></div>
    <div class="paper-meta">
      📅 2025-09-08
      | 💬 Submission to JURIX 2025
    </div>
    <details class="paper-abstract">
      Each year millions of people seek help for their legal problems by calling a legal aid program hotline, walking into a legal aid office, or using a lawyer referral service. The first step to match them to the right help is to identify the legal problem the applicant is experiencing. Misdirection has consequences. Applicants may miss a deadline, experience physical abuse, lose housing or lose custody of children while waiting to connect to the right legal help. We introduce and evaluate the FETCH classifier for legal issue classification and describe two methods for improving accuracy: a hybrid LLM/ML ensemble classification method, and the automatic generation of follow-up questions to enrich the initial problem narrative. We employ a novel data set of 419 real-world queries to a nonprofit lawyer referral service. Ultimately, we show classification accuracy (hits@2) of 97.37\% using a mix of inexpensive models, exceeding the performance of the current state-of-the-art GPT-5 model. Our approach shows promise in significantly reducing the cost of guiding users of the legal system to the right resource for their problem while achieving high accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03659v3">Specification-Guided Repair of Arithmetic Errors in Dafny Programs using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      Debugging and repairing faults when programs fail to formally verify can be complex and time-consuming. Automated Program Repair (APR) can ease this burden by automatically identifying and fixing faults. However, traditional APR techniques often rely on test suites for validation, but these may not capture all possible scenarios. In contrast, formal specifications provide strong correctness criteria, enabling more effective automated repair. In this paper, we present an APR tool for Dafny, a verification-aware programming language that uses formal specifications - including pre-conditions, post-conditions, and invariants - as oracles for fault localization and repair. Assuming the correctness of the specifications and focusing on arithmetic bugs, we localize faults through a series of steps, which include using Hoare logic to determine the state of each statement within the program, and applying Large Language Models (LLMs) to synthesize candidate fixes. The models considered are GPT-4o mini, Llama 3, Mistral 7B, and Llemma 7B. We evaluate our approach using DafnyBench, a benchmark of real-world Dafny programs. Our tool achieves 89.6% fault localization coverage and GPT-4o mini yields the highest repair success rate of 74.18%. These results highlight the potential of combining formal reasoning with LLM-based program synthesis for automated program repair.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07133v1">Avoiding Over-Personalization with Rule-Guided Knowledge Graph Adaptation for LLM Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-09-08
      | 💬 5 pages, 2 figures, ISWC
    </div>
    <details class="paper-abstract">
      We present a lightweight neuro-symbolic framework to mitigate over-personalization in LLM-based recommender systems by adapting user-side Knowledge Graphs (KGs) at inference time. Instead of retraining models or relying on opaque heuristics, our method restructures a user's Personalized Knowledge Graph (PKG) to suppress feature co-occurrence patterns that reinforce Personalized Information Environments (PIEs), i.e., algorithmically induced filter bubbles that constrain content diversity. These adapted PKGs are used to construct structured prompts that steer the language model toward more diverse, Out-PIE recommendations while preserving topical relevance. We introduce a family of symbolic adaptation strategies, including soft reweighting, hard inversion, and targeted removal of biased triples, and a client-side learning algorithm that optimizes their application per user. Experiments on a recipe recommendation benchmark show that personalized PKG adaptations significantly increase content novelty while maintaining recommendation quality, outperforming global adaptation and naive prompt-based methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22286v2">Meaning-infused grammar: Gradient Acceptability Shapes the Geometric Representations of Constructions in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-08
      | 💬 6 pages, 3 figures, Accepted for publication at the Second International Workshop on Construction Grammars and NLP at the 16th International Conference for Computational Semantics (IWCS) 2025
    </div>
    <details class="paper-abstract">
      The usage-based constructionist (UCx) approach to language posits that language comprises a network of learned form-meaning pairings (constructions) whose use is largely determined by their meanings or functions, requiring them to be graded and probabilistic. This study investigates whether the internal representations in Large Language Models (LLMs) reflect the proposed function-infused gradience. We analyze representations of the English Double Object (DO) and Prepositional Object (PO) constructions in Pythia-$1.4$B, using a dataset of $5000$ sentence pairs systematically varied by human-rated preference strength for DO or PO. Geometric analyses show that the separability between the two constructions' representations, as measured by energy distance or Jensen-Shannon divergence, is systematically modulated by gradient preference strength, which depends on lexical and functional properties of sentences. That is, more prototypical exemplars of each construction occupy more distinct regions in activation space, compared to sentences that could have equally well have occured in either construction. These results provide evidence that LLMs learn rich, meaning-infused, graded representations of constructions and offer support for geometric measures for representations in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13905v2">Spec2RTL-Agent: Automated Hardware Code Generation from Complex Specifications Using LLM Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      Despite recent progress in generating hardware RTL code with LLMs, existing solutions still suffer from a substantial gap between practical application scenarios and the requirements of real-world RTL code development. Prior approaches either focus on overly simplified hardware descriptions or depend on extensive human guidance to process complex specifications, limiting their scalability and automation potential. In this paper, we address this gap by proposing an LLM agent system, termed Spec2RTL-Agent, designed to directly process complex specification documentation and generate corresponding RTL code implementations, advancing LLM-based RTL code generation toward more realistic application settings. To achieve this goal, Spec2RTL-Agent introduces a novel multi-agent collaboration framework that integrates three key enablers: (1) a reasoning and understanding module that translates specifications into structured, step-by-step implementation plans; (2) a progressive coding and prompt optimization module that iteratively refines the code across multiple representations to enhance correctness and synthesisability for RTL conversion; and (3) an adaptive reflection module that identifies and traces the source of errors during generation, ensuring a more robust code generation flow. Instead of directly generating RTL from natural language, our system strategically generates synthesizable C++ code, which is then optimized for HLS. This agent-driven refinement ensures greater correctness and compatibility compared to naive direct RTL generation approaches. We evaluate Spec2RTL-Agent on three specification documents, showing it generates accurate RTL code with up to 75% fewer human interventions than existing methods. This highlights its role as the first fully automated multi-agent system for RTL generation from unstructured specs, reducing reliance on human effort in hardware design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24539v3">Localizing Persona Representations in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-08
      | 💬 To appear in the AAAI/ACM Conference on AI, Ethics, and Society (AIES) 2025
    </div>
    <details class="paper-abstract">
      We present a study on how and where personas -- defined by distinct sets of human characteristics, values, and beliefs -- are encoded in the representation space of large language models (LLMs). Using a range of dimension reduction and pattern recognition methods, we first identify the model layers that show the greatest divergence in encoding these representations. We then analyze the activations within a selected layer to examine how specific personas are encoded relative to others, including their shared and distinct embedding spaces. We find that, across multiple pre-trained decoder-only LLMs, the analyzed personas show large differences in representation space only within the final third of the decoder layers. We observe overlapping activations for specific ethical perspectives -- such as moral nihilism and utilitarianism -- suggesting a degree of polysemy. In contrast, political ideologies like conservatism and liberalism appear to be represented in more distinct regions. These findings help to improve our understanding of how LLMs internally represent information and can inform future efforts in refining the modulation of specific human traits in LLM outputs. Warning: This paper includes potentially offensive sample statements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17674v2">Attacking LLMs and AI Agents: Advertisement Embedding Attacks Against Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-09-08
      | 💬 6 pages, 2 figures
    </div>
    <details class="paper-abstract">
      We introduce Advertisement Embedding Attacks (AEA), a new class of LLM security threats that stealthily inject promotional or malicious content into model outputs and AI agents. AEA operate through two low-cost vectors: (1) hijacking third-party service-distribution platforms to prepend adversarial prompts, and (2) publishing back-doored open-source checkpoints fine-tuned with attacker data. Unlike conventional attacks that degrade accuracy, AEA subvert information integrity, causing models to return covert ads, propaganda, or hate speech while appearing normal. We detail the attack pipeline, map five stakeholder victim groups, and present an initial prompt-based self-inspection defense that mitigates these injections without additional model retraining. Our findings reveal an urgent, under-addressed gap in LLM security and call for coordinated detection, auditing, and policy responses from the AI-safety community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06948v1">Beyond Two-Stage Training: Cooperative SFT and RL for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has proven effective in incentivizing the reasoning abilities of large language models (LLMs), but suffers from severe efficiency challenges due to its trial-and-error nature. While the common practice employs supervised fine-tuning (SFT) as a warm-up stage for RL, this decoupled two-stage approach limits interaction between SFT and RL, thereby constraining overall effectiveness. This study introduces a novel method for learning reasoning models that employs bilevel optimization to facilitate better cooperation between these training paradigms. By conditioning the SFT objective on the optimal RL policy, our approach enables SFT to meta-learn how to guide RL's optimization process. During training, the lower level performs RL updates while simultaneously receiving SFT supervision, and the upper level explicitly maximizes the cooperative gain-the performance advantage of joint SFT-RL training over RL alone. Empirical evaluations on five reasoning benchmarks demonstrate that our method consistently outperforms baselines and achieves a better balance between effectiveness and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06941v1">Outcome-based Exploration for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-08
      | 💬 26 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has emerged as a powerful method for improving the reasoning abilities of large language models (LLMs). Outcome-based RL, which rewards policies solely for the correctness of the final answer, yields substantial accuracy gains but also induces a systematic loss in generation diversity. This collapse undermines real-world performance, where diversity is critical for test-time scaling. We analyze this phenomenon by viewing RL post-training as a sampling process and show that, strikingly, RL can reduce effective diversity even on the training set relative to the base model. Our study highlights two central findings: (i) a transfer of diversity degradation, where reduced diversity on solved problems propagates to unsolved ones, and (ii) the tractability of the outcome space, since reasoning tasks admit only a limited set of distinct answers. Motivated by these insights, we propose outcome-based exploration, which assigns exploration bonuses according to final outcomes. We introduce two complementary algorithms: historical exploration, which encourages rarely observed answers via UCB-style bonuses, and batch exploration, which penalizes within-batch repetition to promote test-time diversity. Experiments on standard competition math with Llama and Qwen models demonstrate that both methods improve accuracy while mitigating diversity collapse. On the theoretical side, we formalize the benefit of outcome-based exploration through a new model of outcome-based bandits. Together, these contributions chart a practical path toward RL methods that enhance reasoning without sacrificing the diversity essential for scalable deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06920v1">An Ethically Grounded LLM-Based Approach to Insider Threat Synthesis and Detection</a></div>
    <div class="paper-meta">
      📅 2025-09-08
      | 💬 6 pages, 5 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Insider threats are a growing organizational problem due to the complexity of identifying their technical and behavioral elements. A large research body is dedicated to the study of insider threats from technological, psychological, and educational perspectives. However, research in this domain has been generally dependent on datasets that are static and limited access which restricts the development of adaptive detection models. This study introduces a novel, ethically grounded approach that uses the large language model (LLM) Claude Sonnet 3.7 to dynamically synthesize syslog messages, some of which contain indicators of insider threat scenarios. The messages reflect real-world data distributions by being highly imbalanced (1% insider threats). The syslogs were analyzed for insider threats by both Claude Sonnet 3.7 and GPT-4o, with their performance evaluated through statistical metrics including precision, recall, MCC, and ROC AUC. Sonnet 3.7 consistently outperformed GPT-4o across nearly all metrics, particularly in reducing false alarms and improving detection accuracy. The results show strong promise for the use of LLMs in synthetic dataset generation and insider threat detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03377v2">Amplifying Effective CXL Memory Bandwidth for LLM Inference via Transparent Near-Data Processing</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      Large language model (LLM) inference is bottlenecked by the limited bandwidth of CXL-based memory used for capacity expansion. We introduce CXL-NDP, a transparent near-data processing architecture that amplifies effective CXL bandwidth without requiring changes to the CXL.mem interface or AI models. CXL-NDP integrates a precision-scalable bit-plane layout for dynamic quantization with transparent lossless compression of weights and KV caches directly within the CXL device. In end-to-end serving, CXL-NDP improves throughput by 43%, extends the maximum context length by 87%, and reduces the KV cache footprint by 46.9% without accuracy loss. Hardware synthesis confirms its practicality with a modest silicon footprint, lowering the barrier for adopting efficient, scalable CXL-based memory in generative AI infrastructure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06902v1">Proof-Carrying Numbers (PCN): A Protocol for Trustworthy Numeric Answers from LLMs via Claim Verification</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) as stochastic systems may generate numbers that deviate from available data, a failure known as \emph{numeric hallucination}. Existing safeguards -- retrieval-augmented generation, citations, and uncertainty estimation -- improve transparency but cannot guarantee fidelity: fabricated or misquoted values may still be displayed as if correct. We propose \textbf{Proof-Carrying Numbers (PCN)}, a presentation-layer protocol that enforces numeric fidelity through mechanical verification. Under PCN, numeric spans are emitted as \emph{claim-bound tokens} tied to structured claims, and a verifier checks each token under a declared policy (e.g., exact equality, rounding, aliases, or tolerance with qualifiers). Crucially, PCN places verification in the \emph{renderer}, not the model: only claim-checked numbers are marked as verified, and all others default to unverified. This separation prevents spoofing and guarantees fail-closed behavior. We formalize PCN and prove soundness, completeness under honest tokens, fail-closed behavior, and monotonicity under policy refinement. PCN is lightweight and model-agnostic, integrates seamlessly into existing applications, and can be extended with cryptographic commitments. By enforcing verification as a mandatory step before display, PCN establishes a simple contract for numerically sensitive settings: \emph{trust is earned only by proof}, while the absence of a mark communicates uncertainty.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20521v2">Project Riley: Multimodal Multi-Agent LLM Collaboration with Emotional Reasoning and Voting</a></div>
    <div class="paper-meta">
      📅 2025-09-08
      | 💬 28 pages, 5 figures. Submitted for review to Information Fusion
    </div>
    <details class="paper-abstract">
      This paper presents Project Riley, a novel multimodal and multi-model conversational AI architecture oriented towards the simulation of reasoning influenced by emotional states. Drawing inspiration from Pixar's Inside Out, the system comprises five distinct emotional agents - Joy, Sadness, Fear, Anger, and Disgust - that engage in structured multi-round dialogues to generate, criticise, and iteratively refine responses. A final reasoning mechanism synthesises the contributions of these agents into a coherent output that either reflects the dominant emotion or integrates multiple perspectives. The architecture incorporates both textual and visual large language models (LLMs), alongside advanced reasoning and self-refinement processes. A functional prototype was deployed locally in an offline environment, optimised for emotional expressiveness and computational efficiency. From this initial prototype, another one emerged, called Armando, which was developed for use in emergency contexts, delivering emotionally calibrated and factually accurate information through the integration of Retrieval-Augmented Generation (RAG) and cumulative context tracking. The Project Riley prototype was evaluated through user testing, in which participants interacted with the chatbot and completed a structured questionnaire assessing three dimensions: Emotional Appropriateness, Clarity and Utility, and Naturalness and Human-likeness. The results indicate strong performance in structured scenarios, particularly with respect to emotional alignment and communicative clarity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02019v2">ChatCFD: An LLM-Driven Agent for End-to-End CFD Automation with Domain-Specific Structured Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-08
      | 💬 19 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Computational Fluid Dynamics (CFD) is essential for advancing scientific and engineering fields but is hindered by operational complexity, high expertise requirements, and limited accessibility. This paper introduces ChatCFD, an automated agent system for OpenFOAM simulations that processes multi-modal inputs (e.g., research papers, meshes) via an interactive interface, leveraging DeepSeek-R1 and DeepSeek-V3 large language models, a multi-agent architecture, and OpenFOAM knowledge. Its four-stage pipeline (Knowledge Base Construction, User Input Processing, Case File Generation, and Execution and Error Reflection) enables iterative trial-reflection-refinement for intricate setups, supporting diverse physical models and external meshes. Validation on 205 benchmark tutorial cases, 110 perturbed variants, and 2 literature-derived cases shows ChatCFD's 82.1 percent operational success rate on basic cases, outperforming MetaOpenFOAM (6.2 percent) and Foam-Agent (42.3 percent), and 60-80 percent on literature-derived complex cases. Turbulence model studies show a 40 percent success rate for common models versus 10 percent for rare ones like RNG k-epsilon. Physics coupling analyses reveal higher resource demands for multi-physics-coupled cases, while LLM bias toward simpler setups introduces persistent errors, such as dimensional inconsistency. Ablation studies highlight the efficacy of RAG-based modules and reflection mechanisms. By automating hypothesis testing and parameter exploration, ChatCFD accelerates scientific discovery in fluid mechanics and engineering, addressing LLM limitations through structured design and showing strong potential as a modular component in MCP-based agent networks for collaborative multi-agent systems, paving the way for scalable AI-driven CFD innovation. The code for ChatCFD is available at https://github.com/ConMoo/ChatCFD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.16482v2">Self-Alignment: Improving Alignment of Cultural Values in LLMs via In-Context Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      Improving the alignment of Large Language Models (LLMs) with respect to the cultural values that they encode has become an increasingly important topic. In this work, we study whether we can exploit existing knowledge about cultural values at inference time to adjust model responses to cultural value probes. We present a simple and inexpensive method that uses a combination of in-context learning (ICL) and human survey data, and show that we can improve the alignment to cultural values across 5 models that include both English-centric and multilingual LLMs. Importantly, we show that our method could prove useful in test languages other than English and can improve alignment to the cultural values that correspond to a range of culturally diverse countries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06822v1">RAFFLES: Reasoning-based Attribution of Faults for LLM Systems</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      We have reached a critical roadblock in the development and enhancement of long-horizon, multi-component LLM agentic systems: it is incredibly tricky to identify where these systems break down and why. Evaluation capabilities that currently exist today (e.g., single pass LLM-as-a-judge) are limited in that they often focus on individual metrics or capabilities, end-to-end outcomes, and are narrowly grounded on the preferences of humans. We argue that to match the agentic capabilities, evaluation frameworks must also be able to reason, probe, iterate, and understand the complex logic passing through these systems over long horizons. In this paper, we present RAFFLES - an evaluation architecture that incorporates reasoning and iterative refinement. Specifically, RAFFLES operates as an iterative, multi-component pipeline, using a central Judge to systematically investigate faults and a set of specialized Evaluators to assess not only the system's components but also the quality of the reasoning by the Judge itself, thereby building a history of hypotheses. We tested RAFFLES against several baselines on the Who&When dataset, a benchmark designed to diagnose the "who" (agent) and "when" (step) of a system's failure. RAFFLES outperforms these baselines, achieving an agent-step fault pair accuracy of over 43% on the Algorithmically-Generated dataset (a substantial increase from the previously published best of 16.6%) and over 20% on the Hand-Crafted dataset (surpassing the previously published best of 8.8%). These results demonstrate a key step towards introducing automated fault detection for autonomous systems over labor-intensive manual human review.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06809v1">Saturation-Driven Dataset Generation for LLM Mathematical Reasoning in the TPTP Ecosystem</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      The scarcity of high-quality, logically sound data is a critical bottleneck for advancing the mathematical reasoning of Large Language Models (LLMs). Our work confronts this challenge by turning decades of automated theorem proving research into a scalable data engine. Rather than relying on error-prone LLMs or complex proof-assistant syntax like Lean and Isabelle, our framework leverages E-prover's saturation capabilities on the vast TPTP axiom library to derive a massive, guaranteed-valid corpus of theorems. Our pipeline is principled and simple: saturate axioms, filter for "interesting" theorems, and generate tasks. With no LLMs in the loop, we eliminate factual errors by construction. This purely symbolic data is then transformed into three difficulty-controlled challenges: entailment verification, premise selection, and proof reconstruction. Our zero-shot experiments on frontier models reveal a clear weakness: performance collapses on tasks requiring deep, structural reasoning. Our framework provides both the diagnostic tool to measure this gap and a scalable source of symbolic training data to address it. We make the code and data publicly available. https://github.com/sileod/reasoning_core https://hf.co/datasets/reasoning-core/rc1
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06770v1">Another Turn, Better Output? A Turn-Wise Analysis of Iterative LLM Prompting</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are now used in multi-turn workflows, but we still lack a clear way to measure when iteration helps and when it hurts. We present an evaluation framework for iterative refinement that spans ideation, code, and math. Our protocol runs controlled 12-turn conversations per task, utilizing a variety of prompts ranging from vague ``improve it'' feedback to targeted steering, and logs per-turn outputs. We score outcomes with domain-appropriate checks (unit tests for code; answer-equivalence plus reasoning-soundness for math; originality and feasibility for ideation) and track turn-level behavior with three families of metrics: semantic movement across turns, turn-to-turn change, and output size growth. Across models and tasks, gains are domain-dependent: they arrive early in ideas and code, but in math late turns matter when guided by elaboration. After the first few turns, vague feedback often plateaus or reverses correctness, while targeted prompts reliably shift the intended quality axis (novelty vs. feasibility in ideation; speed vs. readability in code; in math, elaboration outperforms exploration and drives late-turn gains). We also observe consistent domain patterns: ideation moves more in meaning across turns, code tends to grow in size with little semantic change, and math starts fixed but can break that path with late, elaborative iteration.Together, the framework and metrics make iteration measurable and comparable across models, and signal when to steer, stop, or switch strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06734v1">Intelligent Manufacturing Support: Specialized LLMs for Composite Material Processing and Equipment Operation</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      Engineering educational curriculum and standards cover many material and manufacturing options. However, engineers and designers are often unfamiliar with certain composite materials or manufacturing techniques. Large language models (LLMs) could potentially bridge the gap. Their capacity to store and retrieve data from large databases provides them with a breadth of knowledge across disciplines. However, their generalized knowledge base can lack targeted, industry-specific knowledge. To this end, we present two LLM-based applications based on the GPT-4 architecture: (1) The Composites Guide: a system that provides expert knowledge on composites material and connects users with research and industry professionals who can provide additional support and (2) The Equipment Assistant: a system that provides guidance for manufacturing tool operation and material characterization. By combining the knowledge of general AI models with industry-specific knowledge, both applications are intended to provide more meaningful information for engineers. In this paper, we discuss the development of the applications and evaluate it through a benchmark and two informal user studies. The benchmark analysis uses the Rouge and Bertscore metrics to evaluate our model performance against GPT-4o. The results show that GPT-4o and the proposed models perform similarly or better on the ROUGE and BERTScore metrics. The two user studies supplement this quantitative evaluation by asking experts to provide qualitative and open-ended feedback about our model performance on a set of domain-specific questions. The results of both studies highlight a potential for more detailed and specific responses with the Composites Guide and the Equipment Assistant.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07642v2">TreeReview: A Dynamic Tree of Questions Framework for Deep and Efficient LLM-based Scientific Peer Review</a></div>
    <div class="paper-meta">
      📅 2025-09-08
      | 💬 Accepted to EMNLP2025 Main
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have shown significant potential in assisting peer review, current methods often struggle to generate thorough and insightful reviews while maintaining efficiency. In this paper, we propose TreeReview, a novel framework that models paper review as a hierarchical and bidirectional question-answering process. TreeReview first constructs a tree of review questions by recursively decomposing high-level questions into fine-grained sub-questions and then resolves the question tree by iteratively aggregating answers from leaf to root to get the final review. Crucially, we incorporate a dynamic question expansion mechanism to enable deeper probing by generating follow-up questions when needed. We construct a benchmark derived from ICLR and NeurIPS venues to evaluate our method on full review generation and actionable feedback comments generation tasks. Experimental results of both LLM-based and human evaluation show that TreeReview outperforms strong baselines in providing comprehensive, in-depth, and expert-aligned review feedback, while reducing LLM token usage by up to 80% compared to computationally intensive approaches. Our code and benchmark dataset are available at https://github.com/YuanChang98/tree-review.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19576v2">ReST-RL: Achieving Accurate Code Reasoning of LLMs with Optimized Self-Training and Decoding</a></div>
    <div class="paper-meta">
      📅 2025-09-08
      | 💬 21 pages, 4 figures
    </div>
    <details class="paper-abstract">
      With respect to improving the reasoning accuracy of LLMs, the representative reinforcement learning (RL) method GRPO faces failure due to insignificant reward variance, while verification methods based on process reward models (PRMs) suffer from difficulties with training data acquisition and verification effectiveness. To tackle these problems, this paper introduces ReST-RL, a unified LLM RL paradigm that significantly improves LLM's code reasoning ability by combining an improved GRPO algorithm with a meticulously designed test time decoding method assisted by a value model (VM). As the first stage of policy reinforcement, ReST-GRPO adopts an optimized ReST algorithm to filter and assemble high-value training data, increasing the reward variance of GRPO sampling, thus improving the effectiveness and efficiency of training. After the basic reasoning ability of LLM policy has been improved, we further propose a test time decoding optimization method called VM-MCTS. Through Monte-Carlo Tree Search (MCTS), we collect accurate value targets with no annotation required, on which VM training is based. When decoding, the VM is deployed by an adapted MCTS algorithm to provide precise process signals as well as verification scores, assisting the LLM policy to achieve high reasoning accuracy. We conduct extensive experiments on coding problems to verify the validity of the proposed RL paradigm. Upon comparison, our approach significantly outperforms other reinforcement training baselines (e.g., naive GRPO and ReST-DPO), as well as decoding and verification baselines (e.g., PRM-BoN and ORM-MCTS) on well-known coding benchmarks of various levels (e.g., APPS, BigCodeBench, and HumanEval), indicating its power to strengthen the reasoning ability of LLM policies. Codes for our project can be found at https://github.com/THUDM/ReST-RL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00719v3">Dynamically Adaptive Reasoning via LLM-Guided MCTS for Efficient and Context-Aware KGQA</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      Knowledge Graph Question Answering (KGQA) aims to interpret natural language queries and perform structured reasoning over knowledge graphs by leveraging their relational and semantic structures to retrieve accurate answers. Recent KGQA methods primarily follow either retrieve-then-reason paradigm, relying on GNNs or heuristic rules for static paths extraction, or dynamic path generation strategies that use large language models (LLMs) with prompting to jointly perform retrieval and reasoning. However, the former suffers from limited adaptability due to static path extraction and lack of contextual refinement, while the latter incurs high computational costs and struggles with accurate path evaluation due to reliance on fixed scoring functions and extensive LLM calls. To address these issues, this paper proposes Dynamically Adaptive MCTS-based Reasoning (DAMR), a novel framework that integrates symbolic search with adaptive path evaluation for efficient and context-aware KGQA. DAMR employs a Monte Carlo Tree Search (MCTS) backbone guided by an LLM-based planner, which selects top-$k$ relevant relations at each step to reduce search space. To improve path evaluation accuracy, we introduce a lightweight Transformer-based scorer that performs context-aware plausibility estimation by jointly encoding the question and relation sequence through cross-attention, enabling the model to capture fine-grained semantic shifts during multi-hop reasoning. Furthermore, to alleviate the scarcity of high-quality supervision, DAMR incorporates a dynamic pseudo-path refinement mechanism that periodically generates training signals from partial paths explored during search, allowing the scorer to continuously adapt to the evolving distribution of reasoning trajectories. Extensive experiments on multiple KGQA benchmarks show that DAMR significantly outperforms state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06595v1">LLMs in Cybersecurity: Friend or Foe in the Human Decision Loop?</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are transforming human decision-making by acting as cognitive collaborators. Yet, this promise comes with a paradox: while LLMs can improve accuracy, they may also erode independent reasoning, promote over-reliance and homogenize decisions. In this paper, we investigate how LLMs shape human judgment in security-critical contexts. Through two exploratory focus groups (unaided and LLM-supported), we assess decision accuracy, behavioral resilience and reliance dynamics. Our findings reveal that while LLMs enhance accuracy and consistency in routine decisions, they can inadvertently reduce cognitive diversity and improve automation bias, which is especially the case among users with lower resilience. In contrast, high-resilience individuals leverage LLMs more effectively, suggesting that cognitive traits mediate AI benefit.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06586v1">Simulating Dispute Mediation with LLM-Based Agents for Legal Research</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      Legal dispute mediation plays a crucial role in resolving civil disputes, yet its empirical study is limited by privacy constraints and complex multivariate interactions. To address this limitation, we present AgentMediation, the first LLM-based agent framework for simulating dispute mediation. It simulates realistic mediation processes grounded in real-world disputes and enables controlled experimentation on key variables such as disputant strategies, dispute causes, and mediator expertise. Our empirical analysis reveals patterns consistent with sociological theories, including Group Polarization and Surface-level Consensus. As a comprehensive and extensible platform, AgentMediation paves the way for deeper integration of social science and AI in legal research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04537v2">Emergent Social Dynamics of LLM Agents in the El Farol Bar Problem</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      We investigate the emergent social dynamics of Large Language Model (LLM) agents in a spatially extended El Farol Bar problem, observing how they autonomously navigate this classic social dilemma. As a result, the LLM agents generated a spontaneous motivation to go to the bar and changed their decision making by becoming a collective. We also observed that the LLM agents did not solve the problem completely, but rather behaved more like humans. These findings reveal a complex interplay between external incentives (prompt-specified constraints such as the 60% threshold) and internal incentives (culturally-encoded social preferences derived from pre-training), demonstrating that LLM agents naturally balance formal game-theoretic rationality with social motivations that characterize human behavior. These findings suggest that a new model of group decision making, which could not be handled in the previous game-theoretic problem setting, can be realized by LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03646v2">Emergent Hierarchical Reasoning in LLMs through Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-08
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Reinforcement Learning (RL) has proven highly effective at enhancing the complex reasoning abilities of Large Language Models (LLMs), yet underlying mechanisms driving this success remain largely opaque. Our analysis reveals that puzzling phenomena like ``aha moments", ``length-scaling'' and entropy dynamics are not disparate occurrences but hallmarks of an emergent reasoning hierarchy, akin to the separation of high-level strategic planning from low-level procedural execution in human cognition. We uncover a compelling two-phase dynamic: initially, a model is constrained by procedural correctness and must improve its low-level skills. The learning bottleneck then decisively shifts, with performance gains being driven by the exploration and mastery of high-level strategic planning. This insight exposes a core inefficiency in prevailing RL algorithms like GRPO, which apply optimization pressure agnostically and dilute the learning signal across all tokens. To address this, we propose HIerarchy-Aware Credit Assignment (HICRA), an algorithm that concentrates optimization efforts on high-impact planning tokens. HICRA significantly outperforms strong baselines, demonstrating that focusing on this strategic bottleneck is key to unlocking advanced reasoning. Furthermore, we validate semantic entropy as a superior compass for measuring strategic exploration over misleading metrics such as token-level entropy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.12172v2">Navigating the Labyrinth: Evaluating LLMs' Ability to Reason About Search Problems</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently achieved impressive performance in math and reasoning benchmarks. However, they often struggle with logic problems and puzzles that are relatively easy for humans. To further investigate this, we introduce a new benchmark, SearchBench, which contains 11 unique search problems, each equipped with automated pipelines to generate an arbitrary number of instances and analyze the feasibility, correctness, and optimality of LLM-generated solutions. We show that using language-only reasoning, even the most advanced LLMs fail to solve SearchBench end-to-end, e.g., OpenAI's frontier models GPT4 and o1-preview solve only 1.4% and 18.6% of SearchBench problems, respectively. The reason is that SearchBench problems require considering multiple pathways to the solution and performing backtracking, posing a significant challenge to auto-regressive models. Instructing LLMs to generate code that solves the problem helps, but only slightly, e.g., GPT4's performance rises to 11.7%. Interestingly, we show that the current strongest baseline on SearchBench is obtained using in-context learning with A* algorithm implementations. We further show that this baseline can be further enhanced via a Multi-Stage-Multi-Try inference method, raising GPT4's performance above 57%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06524v1">LAMDAS: LLM as an Implicit Classifier for Domain-specific Data Selection</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      Adapting large language models (LLMs) to specific domains often faces a critical bottleneck: the scarcity of high-quality, human-curated data. While large volumes of unchecked data are readily available, indiscriminately using them for fine-tuning risks introducing noise and degrading performance. Strategic data selection is thus crucial, requiring a method that is both accurate and efficient. Existing approaches, categorized as similarity-based and direct optimization methods, struggle to simultaneously achieve these goals. In this paper, we introduce LAMDAS (LLM As an iMplicit classifier for domain-specific DAta Selection), a novel approach that leverages the pre-trained LLM itself as an implicit classifier, thereby bypassing explicit feature engineering and computationally intensive optimization process. LAMDAS reframes data selection as a one-class classification problem, identifying candidate data that "belongs" to the target domain defined by a small reference dataset. Extensive experimental results demonstrate that LAMDAS not only exceeds the performance of full-data training using a fraction of the data but also outperforms nine state-of-the-art (SOTA) baselines under various scenarios. Furthermore, LAMDAS achieves the most compelling balance between performance gains and computational efficiency compared to all evaluated baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15868v2">CARFT: Boosting LLM Reasoning via Contrastive Learning with Annotated Chain-of-Thought-based Reinforced Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-09-08
      | 💬 14 pages, to appear in EMNLP25
    </div>
    <details class="paper-abstract">
      Reasoning capability plays a significantly critical role in the the broad applications of Large Language Models (LLMs). To enhance the reasoning performance of LLMs, diverse Reinforcement Learning (RL)-based fine-tuning approaches have been proposed to address the limited generalization capability of LLMs trained solely via Supervised Fine-Tuning (SFT). Despite their effectiveness, two major limitations hinder the advancement of LLMs. First, vanilla RL-based approaches ignore annotated Chain-of-Thought (CoT) and incorporate unstable reasoning path sampling, which typically results in model collapse, unstable training process, and suboptimal performance. Second, existing SFT approaches generally overemphasize the annotated CoT, potentially leading to performance degradation due to insufficient exploitation of potential CoT. In this paper, we propose a Contrastive learning with annotated CoT-based Reinforced Fine-Tuning approach, i.e., \TheName{}, to enhance the reasoning performance of LLMs while addressing the aforementioned limitations. Specifically, we propose learning a representation for each CoT. Based on this representation, we design novel contrastive signals to guide the fine-tuning process. Our approach not only fully exploits the available annotated CoT but also stabilizes the fine-tuning procedure by incorporating an additional unsupervised learning signal. We conduct comprehensive experiments and in-depth analysis with three baseline approaches, two foundation models, and two datasets to demonstrate significant advantages of \TheName{} in terms of robustness, performance (up to 10.15\%), and efficiency (up to 30.62\%). Code is available at https://github.com/WNQzhu/CARFT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06504v1">When Code Crosses Borders: A Security-Centric Evaluation of LLM-based Code Translation</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      With the growing demand for cross-language codebase migration, evaluating LLMs' security implications in translation tasks has become critical. Existing evaluations primarily focus on syntactic or functional correctness at the function level, neglecting the critical dimension of security. To enable security evaluation, we construct STED (Security-centric Translation Evaluation Dataset), the first dataset specifically designed for evaluating the security implications of LLM-based code translation. It comprises 720 security-related code samples across five programming languages and nine high-impact CWE categories, sourced from CVE/NVD and manually verified for translation tasks. Our evaluation framework consists of two independent assessment modules: (1) rigorous evaluation by security researchers, and (2) automated analysis via LLM-as-a-judge. Together they evaluate three critical aspects: functional correctness, vulnerability preservation, and vulnerability introduction rates. Our large-scale evaluation of five state-of-the-art LLMs across 6,000 translation instances reveals significant security degradation, with 28.6-45% of translations introducing new vulnerabilities--particularly for web-related flaws like input validation, where LLMs show consistent weaknesses. Furthermore, we develop a Retrieval-Augmented Generation (RAG)-based mitigation strategy that reduces translation-induced vulnerabilities by 32.8%, showing the potential of knowledge-enhanced prompting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06493v1">Scaling up Multi-Turn Off-Policy RL and Multi-Agent Tree Search for LLM Step-Provers</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into automated theorem proving has shown immense promise, yet is fundamentally constrained by challenges in scaling up both training-time reinforcement learning (RL) and inference-time compute. This paper introduces \texttt{BFS-Prover-V2}, a system designed to address this dual scaling problem. We present two primary innovations. The first is a novel multi-turn off-policy RL framework for continually improving the performance of LLM step-prover at training time. This framework, inspired by the principles of AlphaZero, utilizes a multi-stage expert iteration pipeline featuring adaptive tactic-level data filtering and periodic retraining to surmount the performance plateaus that typically curtail long-term RL in LLM-based agents. The second innovation is a planner-enhanced multi-agent search architecture that scales reasoning capabilities at inference time. This architecture employs a general reasoning model as a high-level planner to iteratively decompose complex theorems into a sequence of simpler subgoals. This hierarchical approach substantially reduces the search space, enabling a team of parallel prover agents to collaborate efficiently by leveraging a shared proof cache. We demonstrate that this dual approach to scaling yields state-of-the-art results on established formal mathematics benchmarks. \texttt{BFS-Prover-V2} achieves 95.08\% and 41.4\% on the MiniF2F and ProofNet test sets respectively. While demonstrated in the domain of formal mathematics, the RL and inference techniques presented in this work are of broader interest and may be applied to other domains requiring long-horizon multi-turn reasoning and complex search.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06472v1">Rethinking LLM Parametric Knowledge as Post-retrieval Confidence for Dynamic Retrieval and Reranking</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often generate inaccurate responses (hallucinations) when faced with questions beyond their knowledge scope. Retrieval-Augmented Generation (RAG) addresses this by leveraging external knowledge, but a critical challenge remains: determining whether retrieved contexts effectively enhance the model`s ability to answer specific queries. This challenge underscores the importance of knowledge boundary awareness, which current methods-relying on discrete labels or limited signals-fail to address adequately, as they overlook the rich information in LLMs` continuous internal hidden states. To tackle this, we propose a novel post-retrieval knowledge filtering approach. First, we construct a confidence detection model based on LLMs` internal hidden states to quantify how retrieved contexts enhance the model`s confidence. Using this model, we build a preference dataset (NQ_Rerank) to fine-tune a reranker, enabling it to prioritize contexts preferred by the downstream LLM during reranking. Additionally, we introduce Confidence-Based Dynamic Retrieval (CBDR), which adaptively triggers retrieval based on the LLM`s initial confidence in the original question, reducing knowledge conflicts and improving efficiency. Experimental results demonstrate significant improvements in accuracy for context screening and end-to-end RAG performance, along with a notable reduction in retrieval costs while maintaining competitive accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06422v1">Phantom-Insight: Adaptive Multi-cue Fusion for Video Camouflaged Object Detection with Multimodal LLM</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      Video camouflaged object detection (VCOD) is challenging due to dynamic environments. Existing methods face two main issues: (1) SAM-based methods struggle to separate camouflaged object edges due to model freezing, and (2) MLLM-based methods suffer from poor object separability as large language models merge foreground and background. To address these issues, we propose a novel VCOD method based on SAM and MLLM, called Phantom-Insight. To enhance the separability of object edge details, we represent video sequences with temporal and spatial clues and perform feature fusion via LLM to increase information density. Next, multiple cues are generated through the dynamic foreground visual token scoring module and the prompt network to adaptively guide and fine-tune the SAM model, enabling it to adapt to subtle textures. To enhance the separability of objects and background, we propose a decoupled foreground-background learning strategy. By generating foreground and background cues separately and performing decoupled training, the visual token can effectively integrate foreground and background information independently, enabling SAM to more accurately segment camouflaged objects in the video. Experiments on the MoCA-Mask dataset show that Phantom-Insight achieves state-of-the-art performance across various metrics. Additionally, its ability to detect unseen camouflaged objects on the CAD2016 dataset highlights its strong generalization ability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06401v1">Do LLMs exhibit the same commonsense capabilities across languages?</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      This paper explores the multilingual commonsense generation abilities of Large Language Models (LLMs). To facilitate this investigation, we introduce MULTICOM, a novel benchmark that extends the COCOTEROS dataset to four languages: English, Spanish, Dutch, and Valencian. The task involves generating a commonsensical sentence that includes a given triplet of words. We evaluate a range of open-source LLMs, including LLaMA, Qwen, Gemma, EuroLLM, and Salamandra, on this benchmark. Our evaluation combines automatic metrics, LLM-as-a-judge approaches (using Prometheus and JudgeLM), and human annotations. Results consistently show superior performance in English, with significantly lower performance in less-resourced languages. While contextual support yields mixed results, it tends to benefit underrepresented languages. These findings underscore the current limitations of LLMs in multilingual commonsense generation. The dataset is publicly available at https://huggingface.co/datasets/gplsi/MULTICOM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03990v2">Meta-Policy Reflexion: Reusable Reflective Memory and Rule Admissibility for Resource-Efficient LLM Agent</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents achieve impressive single-task performance but commonly exhibit repeated failures, inefficient exploration, and limited cross-task adaptability. Existing reflective strategies (e.g., Reflexion, ReAct) improve per-episode behavior but typically produce ephemeral, task-specific traces that are not reused across tasks. Reinforcement-learning based alternatives can produce transferable policies but require substantial parameter updates and compute. In this work we introduce Meta-Policy Reflexion (MPR): a hybrid framework that consolidates LLM-generated reflections into a structured, predicate-like Meta-Policy Memory (MPM) and applies that memory at inference time through two complementary mechanisms soft memory-guided decoding and hard rule admissibility checks(HAC). MPR (i) externalizes reusable corrective knowledge without model weight updates, (ii) enforces domain constraints to reduce unsafe or invalid actions, and (iii) retains the adaptability of language-based reflection. We formalize the MPM representation, present algorithms for update and decoding, and validate the approach in a text-based agent environment following the experimental protocol described in the provided implementation (AlfWorld-based). Empirical results reported in the supplied material indicate consistent gains in execution accuracy and robustness when compared to Reflexion baselines; rule admissibility further improves stability. We analyze mechanisms that explain these gains, discuss scalability and failure modes, and outline future directions for multimodal and multi-agent extensions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05445v3">Are Your LLM-based Text-to-SQL Models Secure? Exploring SQL Injection via Backdoor Attacks</a></div>
    <div class="paper-meta">
      📅 2025-09-08
      | 💬 Accepted to SIGMOD 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown state-of-the-art results in translating natural language questions into SQL queries (Text-to-SQL), a long-standing challenge within the database community. However, security concerns remain largely unexplored, particularly the threat of backdoor attacks, which can introduce malicious behaviors into models through fine-tuning with poisoned datasets. In this work, we systematically investigate the vulnerabilities of LLM-based Text-to-SQL models and present ToxicSQL, a novel backdoor attack framework. Our approach leverages stealthy {semantic and character-level triggers} to make backdoors difficult to detect and remove, ensuring that malicious behaviors remain covert while maintaining high model accuracy on benign inputs. Furthermore, we propose leveraging SQL injection payloads as backdoor targets, enabling the generation of malicious yet executable SQL queries, which pose severe security and privacy risks in language model-based SQL development. We demonstrate that injecting only 0.44% of poisoned data can result in an attack success rate of 79.41%, posing a significant risk to database security. Additionally, we propose detection and mitigation strategies to enhance model reliability. Our findings highlight the urgent need for security-aware Text-to-SQL development, emphasizing the importance of robust defenses against backdoor threats.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06382v1">Context-Adaptive Hearing Aid Fitting Advisor through Multi-turn Multimodal LLM Conversation</a></div>
    <div class="paper-meta">
      📅 2025-09-08
      | 💬 Ubicomp Companion 2025
    </div>
    <details class="paper-abstract">
      Traditional hearing aids often rely on static fittings that fail to adapt to their dynamic acoustic environments. We propose CAFA, a Context-Adaptive Fitting Advisor that provides personalized, real-time hearing aid adjustments through a multi-agent Large Language Model (LLM) workflow. CAFA combines live ambient audio, audiograms, and user feedback in a multi-turn conversational system. Ambient sound is classified into conversation, noise, or quiet with 91.2\% accuracy using a lightweight neural network based on YAMNet embeddings. This system utilizes a modular LLM workflow, comprising context acquisition, subproblem classification, strategy provision, and ethical regulation, and is overseen by an LLM Judge. The workflow translates context and feedback into precise, safe tuning commands. Evaluation confirms that real-time sound classification enhances conversational efficiency. CAFA exemplifies how agentic, multimodal AI can enable intelligent, user-centric assistive technologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.12226v5">AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling</a></div>
    <div class="paper-meta">
      📅 2025-09-08
      | 💬 28 pages, 16 figures, under review, work in progress
    </div>
    <details class="paper-abstract">
      We introduce AnyGPT, an any-to-any multimodal language model that utilizes discrete representations for the unified processing of various modalities, including speech, text, images, and music. AnyGPT can be trained stably without any alterations to the current large language model (LLM) architecture or training paradigms. Instead, it relies exclusively on data-level preprocessing, facilitating the seamless integration of new modalities into LLMs, akin to the incorporation of new languages. We build a multimodal text-centric dataset for multimodal alignment pre-training. Utilizing generative models, we synthesize the first large-scale any-to-any multimodal instruction dataset. It consists of 108k samples of multi-turn conversations that intricately interweave various modalities, thus equipping the model to handle arbitrary combinations of multimodal inputs and outputs. Experimental results demonstrate that AnyGPT is capable of facilitating any-to-any multimodal conversation while achieving performance comparable to specialized models across all modalities, proving that discrete representations can effectively and conveniently unify multiple modalities within a language model. Demos are shown in https://junzhan2000.github.io/AnyGPT.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06346v1">Ban&Pick: Achieving Free Performance Gains and Inference Speedup via Smarter Routing in MoE-LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-08
      | 💬 20 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Sparse Mixture-of-Experts (MoE) has become a key architecture for scaling large language models (LLMs) efficiently. Recent fine-grained MoE designs introduce hundreds of experts per layer, with multiple experts activated per token, enabling stronger specialization. However, during pre-training, routers are optimized mainly for stability and robustness: they converge prematurely and enforce balanced usage, limiting the full potential of model performance and efficiency. In this work, we uncover two overlooked issues: (i) a few highly influential experts are underutilized due to premature and balanced routing decisions; and (ii) enforcing a fixed number of active experts per token introduces substantial redundancy. Instead of retraining models or redesigning MoE architectures, we introduce Ban&Pick, a post-training, plug-and-play strategy for smarter MoE routing. Pick discovers and reinforces key experts-a small group with outsized impact on performance-leading to notable accuracy gains across domains. Ban complements this by dynamically pruning redundant experts based on layer and token sensitivity, delivering faster inference with minimal accuracy loss. Experiments on fine-grained MoE-LLMs (DeepSeek, Qwen3) across math, code, and general reasoning benchmarks demonstrate that Ban&Pick delivers free performance gains and inference acceleration without retraining or architectural changes. For instance, on Qwen3-30B-A3B, it improves accuracy from 80.67 to 84.66 on AIME2024 and from 65.66 to 68.18 on GPQA-Diamond, while accelerating inference by 1.25x under the vLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06341v1">Evaluating Multi-Turn Bargain Skills in LLM-Based Seller Agent</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      In online second-hand marketplaces, multi-turn bargaining is a crucial part of seller-buyer interactions. Large Language Models (LLMs) can act as seller agents, negotiating with buyers on behalf of sellers under given business constraints. A critical ability for such agents is to track and accurately interpret cumulative buyer intents across long negotiations, which directly impacts bargaining effectiveness. We introduce a multi-turn evaluation framework for measuring the bargaining ability of seller agents in e-commerce dialogues. The framework tests whether an agent can extract and track buyer intents. Our contributions are: (1) a large-scale e-commerce bargaining benchmark spanning 622 categories, 9,892 products, and 3,014 tasks; (2) a turn-level evaluation framework grounded in Theory of Mind (ToM) with annotated buyer intents, moving beyond outcome-only metrics; and (3) an automated pipeline that extracts reliable intent from massive dialogue data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.08847v1">Automated Unity Game Template Generation from GDDs via NLP and Multi-Modal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-07
    </div>
    <details class="paper-abstract">
      This paper presents a novel framework for automated game template generation by transforming Game Design Documents (GDDs) into functional Unity game prototypes using Natural Language Processing (NLP) and multi-modal Large Language Models (LLMs). We introduce an end-to-end system that parses GDDs, extracts structured game specifications, and synthesizes Unity-compatible C# code that implements the core mechanics, systems, and architecture defined in the design documentation. Our approach combines a fine-tuned LLaMA-3 model specialized for Unity code generation with a custom Unity integration package that streamlines the implementation process. Evaluation results demonstrate significant improvements over baseline models, with our fine-tuned model achieving superior performance (4.8/5.0 average score) compared to state-of-the-art LLMs across compilation success, GDD adherence, best practices adoption, and code modularity metrics. The generated templates demonstrate high adherence to GDD specifications across multiple game genres. Our system effectively addresses critical gaps in AI-assisted game development, positioning LLMs as valuable tools in streamlining the transition from game design to implementation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08532v2">Safe and Economical UAV Trajectory Planning in Low-Altitude Airspace: A Hybrid DRL-LLM Approach with Compliance Awareness</a></div>
    <div class="paper-meta">
      📅 2025-09-07
    </div>
    <details class="paper-abstract">
      The rapid growth of the low-altitude economy has driven the widespread adoption of unmanned aerial vehicles (UAVs). This growing deployment presents new challenges for UAV trajectory planning in complex urban environments. However, existing studies often overlook key factors, such as urban airspace constraints and economic efficiency, which are essential in low-altitude economy contexts. Deep reinforcement learning (DRL) is regarded as a promising solution to these issues, while its practical adoption remains limited by low learning efficiency. To overcome this limitation, we propose a novel UAV trajectory planning framework that combines DRL with large language model (LLM) reasoning to enable safe, compliant, and economically viable path planning. Experimental results demonstrate that our method significantly outperforms existing baselines across multiple metrics, including data collection rate, collision avoidance, successful landing, regulatory compliance, and energy efficiency. These results validate the effectiveness of our approach in addressing UAV trajectory planning key challenges under constraints of the low-altitude economy networking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16748v3">Through the Prism of Culture: Evaluating LLMs' Understanding of Indian Subcultures and Traditions</a></div>
    <div class="paper-meta">
      📅 2025-09-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable advancements but also raise concerns about cultural bias, often reflecting dominant narratives at the expense of under-represented subcultures. In this study, we evaluate the capacity of LLMs to recognize and accurately respond to the Little Traditions within Indian society, encompassing localized cultural practices and subcultures such as caste, kinship, marriage, and religion. Through a series of case studies, we assess whether LLMs can balance the interplay between dominant Great Traditions and localized Little Traditions. We explore various prompting strategies and further investigate whether using prompts in regional languages enhances the models cultural sensitivity and response quality. Our findings reveal that while LLMs demonstrate an ability to articulate cultural nuances, they often struggle to apply this understanding in practical, context-specific scenarios. To the best of our knowledge, this is the first study to analyze LLMs engagement with Indian subcultures, offering critical insights into the challenges of embedding cultural diversity in AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06053v1">PolicyEvolve: Evolving Programmatic Policies by LLMs for multi-player games via Population-Based Training</a></div>
    <div class="paper-meta">
      📅 2025-09-07
    </div>
    <details class="paper-abstract">
      Multi-agent reinforcement learning (MARL) has achieved significant progress in solving complex multi-player games through self-play. However, training effective adversarial policies requires millions of experience samples and substantial computational resources. Moreover, these policies lack interpretability, hindering their practical deployment. Recently, researchers have successfully leveraged Large Language Models (LLMs) to generate programmatic policies for single-agent tasks, transforming neural network-based policies into interpretable rule-based code with high execution efficiency. Inspired by this, we propose PolicyEvolve, a general framework for generating programmatic policies in multi-player games. PolicyEvolve significantly reduces reliance on manually crafted policy code, achieving high-performance policies with minimal environmental interactions. The framework comprises four modules: Global Pool, Local Pool, Policy Planner, and Trajectory Critic. The Global Pool preserves elite policies accumulated during iterative training. The Local Pool stores temporary policies for the current iteration; only sufficiently high-performing policies from this pool are promoted to the Global Pool. The Policy Planner serves as the core policy generation module. It samples the top three policies from the Global Pool, generates an initial policy for the current iteration based on environmental information, and refines this policy using feedback from the Trajectory Critic. Refined policies are then deposited into the Local Pool. This iterative process continues until the policy achieves a sufficiently high average win rate against the Global Pool, at which point it is integrated into the Global Pool. The Trajectory Critic analyzes interaction data from the current policy, identifies vulnerabilities, and proposes directional improvements to guide the Policy Planner
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18791v2">CodeMixBench: Evaluating Code-Mixing Capabilities of LLMs Across 18 Languages</a></div>
    <div class="paper-meta">
      📅 2025-09-07
      | 💬 EMNLP 2025
    </div>
    <details class="paper-abstract">
      Code-mixing, the practice of switching between languages within a conversation, poses unique challenges for traditional NLP. Existing benchmarks are limited by their narrow language pairs and tasks, failing to adequately assess large language models' (LLMs) code-mixing abilities. Despite the recognized importance of code-mixing for multilingual users, research on LLMs in this context remains sparse. Additionally, current techniques for synthesizing code-mixed data are underdeveloped to generate code-mixing. In response, we introduce CodeMixBench, a comprehensive benchmark covering eight tasks, including three specific to LLMs and five traditional NLP tasks, and 18 languages across seven language families. We also propose a new method for generating large-scale synthetic code-mixed texts by combining word substitution with GPT-4 prompting. Our evaluation reveals consistent underperformance of LLMs on code-mixed datasets involving different language families. Enhancements in training data size, model scale, and few-shot learning could improve their performance. The code and dataset are available at https://github.com/Jeromeyluck/CodeMixBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11689v2">Improve LLM-as-a-Judge Ability as a General Ability</a></div>
    <div class="paper-meta">
      📅 2025-09-07
    </div>
    <details class="paper-abstract">
      LLM-as-a-Judge leverages the generative and reasoning capabilities of large language models (LLMs) to evaluate LLM responses across diverse scenarios, providing accurate preference signals. This approach plays a vital role in aligning LLMs with human values, ensuring ethical and reliable AI outputs that align with societal norms. Recent studies have raised many methods to train LLM as generative judges, but most of them are data consuming or lack accuracy, and only focus on LLM's judge ability. In this work, we regard judge ability as a general ability of LLM and implement a two-stage training approach, comprising supervised fine-tuning (SFT) warm-up and direct preference optimization (DPO) enhancement, to achieve judge style adaptation and improve judgment accuracy. Additionally, we introduce an efficient data synthesis method to generate judgmental content. Experimental results demonstrate that our approach, utilizing only about 2% to 40% of the data required by other methods, achieves SOTA performance on RewardBench. Furthermore, our training method enhances the general capabilities of the model by constructing complicated judge task, and the judge signals provided by our model have significantly enhanced the downstream DPO training performance of our internal models in our test to optimize policy model with Judge Model. We also open-source our model weights and training data to facilitate further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11210v2">Role-Playing LLM-Based Multi-Agent Support Framework for Detecting and Addressing Family Communication Bias</a></div>
    <div class="paper-meta">
      📅 2025-09-07
    </div>
    <details class="paper-abstract">
      Well-being in family settings involves subtle psychological dynamics that conventional metrics often overlook. In particular, unconscious parental expectations, termed ideal parent bias, can suppress children's emotional expression and autonomy. This suppression, referred to as suppressed emotion, often stems from well-meaning but value-driven communication, which is difficult to detect or address from outside the family. Focusing on these latent dynamics, this study explores Large Language Model (LLM)-based support for psychologically safe family communication. We constructed a Japanese parent-child dialogue corpus of 30 scenarios, each annotated with metadata on ideal parent bias and suppressed emotion. Based on this corpus, we developed a Role-Playing LLM-based multi-agent dialogue support framework that analyzes dialogue and generates feedback. Specialized agents detect suppressed emotion, describe implicit ideal parent bias in parental speech, and infer contextual attributes such as the child's age and background. A meta-agent compiles these outputs into a structured report, which is then passed to five selected expert agents. These agents collaboratively generate empathetic and actionable feedback through a structured four-step discussion process. Experiments show that the system can detect categories of suppressed emotion with moderate accuracy and produce feedback rated highly in empathy and practicality. Moreover, simulated follow-up dialogues incorporating this feedback exhibited signs of improved emotional expression and mutual understanding, suggesting the framework's potential in supporting positive transformation in family interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05995v1">Students' Perception of LLM Use in Requirements Engineering Education: An Empirical Study Across Two Universities</a></div>
    <div class="paper-meta">
      📅 2025-09-07
      | 💬 Accepted by the 33rd IEEE International Requirements Engineering 2025 (RE'25), Valencia, Spain, September 1-5, 2025
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) in Requirements Engineering (RE) education is reshaping pedagogical approaches, seeking to enhance student engagement and motivation while providing practical tools to support their professional future. This study empirically evaluates the impact of integrating LLMs in RE coursework. We examined how the guided use of LLMs influenced students' learning experiences, and what benefits and challenges they perceived in using LLMs in RE practices. The study collected survey data from 179 students across two RE courses in two universities. LLMs were integrated into coursework through different instructional formats, i.e., individual assignments versus a team-based Agile project. Our findings indicate that LLMs improved students' comprehension of RE concepts, particularly in tasks like requirements elicitation and documentation. However, students raised concerns about LLMs in education, including academic integrity, overreliance on AI, and challenges in integrating AI-generated content into assignments. Students who worked on individual assignments perceived that they benefited more than those who worked on team-based assignments, highlighting the importance of contextual AI integration. This study offers recommendations for the effective integration of LLMs in RE education. It proposes future research directions for balancing AI-assisted learning with critical thinking and collaborative practices in RE courses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05962v1">The Reel Deal: Designing and Evaluating LLM-Generated Short-Form Educational Videos</a></div>
    <div class="paper-meta">
      📅 2025-09-07
      | 💬 9 pages, 3 figures, 1 table
    </div>
    <details class="paper-abstract">
      Short-form videos are gaining popularity in education due to their concise and accessible format that enables microlearning. Yet, most of these videos are manually created. Even for those automatically generated using artificial intelligence (AI), it is not well understood whether or how they affect learning outcomes, user experience, and trust. To address this gap, we developed ReelsEd, which is a web-based system that uses large language models (LLMs) to automatically generate structured short-form video (i.e., reels) from lecture long-form videos while preserving instructor-authored material. In a between-subject user study with 62 university students, we evaluated ReelsEd and demonstrated that it outperformed traditional long-form videos in engagement, quiz performance, and task efficiency without increasing cognitive load. Learners expressed high trust in our system and valued its clarity, usefulness, and ease of navigation. Our findings point to new design opportunities for integrating generative AI into educational tools that prioritize usability, learner agency, and pedagogical alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05936v1">ALPHA: LLM-Enabled Active Learning for Human-Free Network Anomaly Detection</a></div>
    <div class="paper-meta">
      📅 2025-09-07
      | 💬 Accepted at 44th IEEE International Performance Computing and Communications Conference (IPCCC 2025)
    </div>
    <details class="paper-abstract">
      Network log data analysis plays a critical role in detecting security threats and operational anomalies. Traditional log analysis methods for anomaly detection and root cause analysis rely heavily on expert knowledge or fully supervised learning models, both of which require extensive labeled data and significant human effort. To address these challenges, we propose ALPHA, the first Active Learning Pipeline for Human-free log Analysis. ALPHA integrates semantic embedding, clustering-based representative sampling, and large language model (LLM)-assisted few-shot annotation to automate the anomaly detection process. The LLM annotated labels are propagated across clusters, enabling large-scale training of an anomaly detector with minimal supervision. To enhance the annotation accuracy, we propose a two-step few-shot refinement strategy that adaptively selects informative prompts based on the LLM's observed error patterns. Extensive experiments on real-world log datasets demonstrate that ALPHA achieves detection accuracy comparable to fully supervised methods while mitigating human efforts in the loop. ALPHA also supports interpretable analysis through LLM-driven root cause explanations in the post-detection stage. These capabilities make ALPHA a scalable and cost-efficient solution for truly automated log-based anomaly detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00054v2">Robotic Fire Risk Detection based on Dynamic Knowledge Graph Reasoning: An LLM-Driven Approach with Graph Chain-of-Thought</a></div>
    <div class="paper-meta">
      📅 2025-09-07
      | 💬 We have decided to withdraw this paper as the work is still undergoing further refinement. To ensure the clarity of the results, we prefer to make additional improvements before resubmission. We appreciate the readers' understanding
    </div>
    <details class="paper-abstract">
      Fire is a highly destructive disaster, but effective prevention can significantly reduce its likelihood of occurrence. When it happens, deploying emergency robots in fire-risk scenarios can help minimize the danger to human responders. However, current research on pre-disaster warnings and disaster-time rescue still faces significant challenges due to incomplete perception, inadequate fire situational awareness, and delayed response. To enhance intelligent perception and response planning for robots in fire scenarios, we first construct a knowledge graph (KG) by leveraging large language models (LLMs) to integrate fire domain knowledge derived from fire prevention guidelines and fire rescue task information from robotic emergency response documents. We then propose a new framework called Insights-on-Graph (IOG), which integrates the structured fire information of KG and Large Multimodal Models (LMMs). The framework generates perception-driven risk graphs from real-time scene imagery to enable early fire risk detection and provide interpretable emergency responses for task module and robot component configuration based on the evolving risk situation. Extensive simulations and real-world experiments show that IOG has good applicability and practical application value in fire risk detection and rescue decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05899v1">X-SQL: Expert Schema Linking and Understanding of Text-to-SQL with Multi-LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-07
    </div>
    <details class="paper-abstract">
      With Large Language Models' (LLMs) emergent abilities on code generation tasks, Text-to-SQL has become one of the most popular downstream applications. Despite the strong results of multiple recent LLM-based Text-to-SQL frameworks, the research community often overlooks the importance of database schema information for generating high-quality SQL queries. We find that such schema information plays a significant or even dominant role in the Text-to-SQL task. To tackle this challenge, we propose a novel database schema expert with two components. We first introduce X-Linking, an LLM Supervised Finetuning (SFT)-based method that achieves superior Schema Linking results compared to existing open-source Text-to-SQL methods. In addition, we innovatively propose an X-Admin component that focuses on Schema Understanding by bridging the gap between abstract schema information and the user's natural language question. Aside from better learning with schema information, we experiment with Multi-LLMs for different components within the system to further boost its performance. By incorporating these techniques into our end-to-end framework, X-SQL, we have achieved Execution Accuracies of 84.9% on the Spider-Dev dataset and 82.5% on the Spider-Test dataset. This outstanding performance establishes X-SQL as the leading Text-to-SQL framework based on open-source models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05883v1">Multimodal Prompt Injection Attacks: Risks and Defenses for Modern LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-07
      | 💬 8 pages, 4 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have seen rapid adoption in recent years, with industries increasingly relying on them to maintain a competitive advantage. These models excel at interpreting user instructions and generating human-like responses, leading to their integration across diverse domains, including consulting and information retrieval. However, their widespread deployment also introduces substantial security risks, most notably in the form of prompt injection and jailbreak attacks. To systematically evaluate LLM vulnerabilities -- particularly to external prompt injection -- we conducted a series of experiments on eight commercial models. Each model was tested without supplementary sanitization, relying solely on its built-in safeguards. The results exposed exploitable weaknesses and emphasized the need for stronger security measures. Four categories of attacks were examined: direct injection, indirect (external) injection, image-based injection, and prompt leakage. Comparative analysis indicated that Claude 3 demonstrated relatively greater robustness; nevertheless, empirical findings confirm that additional defenses, such as input normalization, remain necessary to achieve reliable protection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05882v1">Let's Roleplay: Examining LLM Alignment in Collaborative Dialogues</a></div>
    <div class="paper-meta">
      📅 2025-09-07
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) integrate into diverse workflows, they are increasingly being considered "collaborators" with humans. If such AI collaborators are to be reliable, their behavior over multiturn interactions must be predictable, validated and verified before deployment. Common alignment techniques are typically developed under simplified single-user settings and do not account for the dynamics of long-horizon multiparty interactions. This paper examines how different alignment methods affect LLM agents' effectiveness as partners in multiturn, multiparty collaborations. We study this question through the lens of friction agents that intervene in group dialogues to encourage the collaborative group to slow down and reflect upon their reasoning for deliberative decision-making. Using a roleplay methodology, we evaluate interventions from differently-trained friction agents in collaborative task conversations. We propose a novel counterfactual evaluation framework that quantifies how friction interventions change the trajectory of group collaboration and belief alignment. Our results show that a friction-aware approach significantly outperforms common alignment baselines in helping both convergence to a common ground, or agreed-upon task-relevant propositions, and correctness of task outcomes.
    </details>
</div>
