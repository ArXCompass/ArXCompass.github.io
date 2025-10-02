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
- Part 10
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)
- [Part 15](papers_15.md)
- [Part 16](papers_16.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12672v1">Towards Inclusive Toxic Content Moderation: Addressing Vulnerabilities to Adversarial Attacks in Toxicity Classifiers Tackling LLM-generated Content</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      The volume of machine-generated content online has grown dramatically due to the widespread use of Large Language Models (LLMs), leading to new challenges for content moderation systems. Conventional content moderation classifiers, which are usually trained on text produced by humans, suffer from misclassifications due to LLM-generated text deviating from their training data and adversarial attacks that aim to avoid detection. Present-day defence tactics are reactive rather than proactive, since they rely on adversarial training or external detection models to identify attacks. In this work, we aim to identify the vulnerable components of toxicity classifiers that contribute to misclassification, proposing a novel strategy based on mechanistic interpretability techniques. Our study focuses on fine-tuned BERT and RoBERTa classifiers, testing on diverse datasets spanning a variety of minority groups. We use adversarial attacking techniques to identify vulnerable circuits. Finally, we suppress these vulnerable circuits, improving performance against adversarial attacks. We also provide demographic-level insights into these vulnerable circuits, exposing fairness and robustness gaps in model training. We find that models have distinct heads that are either crucial for performance or vulnerable to attack and suppressing the vulnerable heads improves performance on adversarial input. We also find that different heads are responsible for vulnerability across different demographic groups, which can inform more inclusive development of toxicity detection models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12649v1">A Systematic Evaluation of Parameter-Efficient Fine-Tuning Methods for the Security of Code LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 25 pages
    </div>
    <details class="paper-abstract">
      Code-generating Large Language Models (LLMs) significantly accelerate software development. However, their frequent generation of insecure code presents serious risks. We present a comprehensive evaluation of seven parameter-efficient fine-tuning (PEFT) techniques, demonstrating substantial gains in secure code generation without compromising functionality. Our research identifies prompt-tuning as the most effective PEFT method, achieving an 80.86% Overall-Secure-Rate on CodeGen2 16B, a 13.5-point improvement over the 67.28% baseline. Optimizing decoding strategies through sampling temperature further elevated security to 87.65%. This equates to a reduction of approximately 203,700 vulnerable code snippets per million generated. Moreover, prompt and prefix tuning increase robustness against poisoning attacks in our TrojanPuzzle evaluation, with strong performance against CWE-79 and CWE-502 attack vectors. Our findings generalize across Python and Java, confirming prompt-tuning's consistent effectiveness. This study provides essential insights and practical guidance for building more resilient software systems with LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12638v1">FinSentLLM: Multi-LLM and Structured Semantic Signals for Enhanced Financial Sentiment Forecasting</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      Financial sentiment analysis (FSA) has attracted significant attention, and recent studies increasingly explore large language models (LLMs) for this field. Yet most work evaluates only classification metrics, leaving unclear whether sentiment signals align with market behavior. We propose FinSentLLM, a lightweight multi-LLM framework that integrates an expert panel of sentiment forecasting LLMs, and structured semantic financial signals via a compact meta-classifier. This design captures expert complementarity, semantic reasoning signal, and agreement/divergence patterns without costly retraining, yielding consistent 3-6% gains over strong baselines in accuracy and F1-score on the Financial PhraseBank dataset. In addition, we also provide econometric evidence that financial sentiment and stock markets exhibit statistically significant long-run comovement, applying Dynamic Conditional Correlation GARCH (DCC-GARCH) and the Johansen cointegration test to daily sentiment scores computed from the FNSPID dataset and major stock indices. Together, these results demonstrate that FinSentLLM delivers superior forecasting accuracy for financial sentiment and further establish that sentiment signals are robustly linked to long-run equity market dynamics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.22388v3">Why Stop at One Error? Benchmarking LLMs as Data Science Code Debuggers for Multi-Hop and Multi-Bug Errors</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 Accepted at EMNLP 2025 Main, Oral
    </div>
    <details class="paper-abstract">
      LLMs are transforming software development, yet current code generation and code repair benchmarks mainly assess syntactic and functional correctness in simple, single-error cases. LLMs' capabilities to autonomously find and fix runtime logical errors in complex data science code remain largely unexplored. To address this gap, we introduce DSDBench: the Data Science Debugging Benchmark, the first benchmark for systematic evaluation of LLMs on multi-hop error tracing and multi-bug detection in data science code debugging. DSDBench adapts datasets from existing data science task benchmarks, such as DABench and MatPlotBench, featuring realistic data science debugging tasks with automatically synthesized multi-hop, multi-bug code snippets. DSDBench includes 1,117 annotated samples with 741 cause-effect error pairs and runtime error messages. Evaluations of state-of-the-art LLMs on DSDBench show significant performance gaps, highlighting challenges in debugging logical runtime errors in data science code. DSDBench offers a crucial resource to evaluate and improve LLMs' debugging and reasoning capabilities, enabling more reliable AI-assisted data science in the future. DSDBench is publicly available at github.com/KevinCL16/DSDBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12625v1">ECG-aBcDe: Overcoming Model Dependence, Encoding ECG into a Universal Language for Any LLM</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 14pages, 6 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) hold significant promise for electrocardiogram (ECG) analysis, yet challenges remain regarding transferability, time-scale information learning, and interpretability. Current methods suffer from model-specific ECG encoders, hindering transfer across LLMs. Furthermore, LLMs struggle to capture crucial time-scale information inherent in ECGs due to Transformer limitations. And their black-box nature limits clinical adoption. To address these limitations, we introduce ECG-aBcDe, a novel ECG encoding method that transforms ECG signals into a universal ECG language readily interpretable by any LLM. By constructing a hybrid dataset of ECG language and natural language, ECG-aBcDe enables direct fine-tuning of pre-trained LLMs without architectural modifications, achieving "construct once, use anywhere" capability. Moreover, the bidirectional convertibility between ECG and ECG language of ECG-aBcDe allows for extracting attention heatmaps from ECG signals, significantly enhancing interpretability. Finally, ECG-aBcDe explicitly represents time-scale information, mitigating Transformer limitations. This work presents a new paradigm for integrating ECG analysis with LLMs. Compared with existing methods, our method achieves competitive performance on ROUGE-L and METEOR. Notably, it delivers significant improvements in the BLEU-4, with improvements of 2.8 times and 3.9 times in in-dataset and cross-dataset evaluations, respectively, reaching scores of 42.58 and 30.76. These results provide strong evidence for the feasibility of the new paradigm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.08426v6">Next-Generation Database Interfaces: A Survey of LLM-based Text-to-SQL</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 Accepted to IEEE TKDE
    </div>
    <details class="paper-abstract">
      Generating accurate SQL from users' natural language questions (text-to-SQL) remains a long-standing challenge due to the complexities involved in user question understanding, database schema comprehension, and SQL generation. Traditional text-to-SQL systems, which combine human engineering and deep neural networks, have made significant progress. Subsequently, pre-trained language models (PLMs) have been developed for text-to-SQL tasks, achieving promising results. However, as modern databases and user questions grow more complex, PLMs with a limited parameter size often produce incorrect SQL. This necessitates more sophisticated and tailored optimization methods, which restricts the application of PLM-based systems. Recently, large language models (LLMs) have shown significant capabilities in natural language understanding as model scale increases. Thus, integrating LLM-based solutions can bring unique opportunities, improvements, and solutions to text-to-SQL research. In this survey, we provide a comprehensive review of existing LLM-based text-to-SQL studies. Specifically, we offer a brief overview of the technical challenges and evolutionary process of text-to-SQL. Next, we introduce the datasets and metrics designed to evaluate text-to-SQL systems. Subsequently, we present a systematic analysis of recent advances in LLM-based text-to-SQL. Finally, we make a summarization and discuss the remaining challenges in this field and suggest expectations for future research directions. All the related resources of LLM-based, including research papers, benchmarks, and open-source projects, are collected for the community in our repository: https://github.com/DEEP-PolyU/Awesome-LLM-based-Text2SQL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12610v1">ScaleDoc: Scaling LLM-based Predicates over Large Document Collections</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      Predicates are foundational components in data analysis systems. However, modern workloads increasingly involve unstructured documents, which demands semantic understanding, beyond traditional value-based predicates. Given enormous documents and ad-hoc queries, while Large Language Models (LLMs) demonstrate powerful zero-shot capabilities, their high inference cost leads to unacceptable overhead. Therefore, we introduce \textsc{ScaleDoc}, a novel system that addresses this by decoupling predicate execution into an offline representation phase and an optimized online filtering phase. In the offline phase, \textsc{ScaleDoc} leverages a LLM to generate semantic representations for each document. Online, for each query, it trains a lightweight proxy model on these representations to filter the majority of documents, forwarding only the ambiguous cases to the LLM for final decision. Furthermore, \textsc{ScaleDoc} proposes two core innovations to achieve significant efficiency: (1) a contrastive-learning-based framework that trains the proxy model to generate reliable predicating decision scores; (2) an adaptive cascade mechanism that determines the effective filtering policy while meeting specific accuracy targets. Our evaluations across three datasets demonstrate that \textsc{ScaleDoc} achieves over a 2$\times$ end-to-end speedup and reduces expensive LLM invocations by up to 85\%, making large-scale semantic analysis practical and efficient.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11177v2">Optimal Brain Restoration for Joint Quantization and Sparsification of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Model (LLM) compression, such as quantization and pruning, have achieved notable success. However, as these techniques gradually approach their respective limits, relying on a single method for further compression has become increasingly challenging. In this work, we explore an alternative solution by combining quantization and sparsity. This joint approach, though promising, introduces new difficulties due to the inherently conflicting requirements on weight distributions: quantization favors compact ranges, while pruning benefits from high variance. To attack this problem, we propose Optimal Brain Restoration (OBR), a general and training-free framework that aligns pruning and quantization by error compensation between both. OBR minimizes performance degradation on downstream tasks by building on a second-order Hessian objective, which is then reformulated into a tractable problem through surrogate approximation and ultimately reaches a closed-form solution via group error compensation. Experiments show that OBR enables aggressive W4A4KV4 quantization with 50% sparsity on existing LLMs, and delivers up to 4.72x speedup and 6.4x memory reduction compared to the FP16-dense baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11967v2">MillStone: How Open-Minded Are LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 19 pages, 7 tables, 7 figures
    </div>
    <details class="paper-abstract">
      Large language models equipped with Web search, information retrieval tools, and other agentic capabilities are beginning to supplant traditional search engines. As users start to rely on LLMs for information on many topics, including controversial and debatable issues, it is important to understand how the stances and opinions expressed in LLM outputs are influenced by the documents they use as their information sources. In this paper, we present MillStone, the first benchmark that aims to systematically measure the effect of external arguments on the stances that LLMs take on controversial issues (not all of them political). We apply MillStone to nine leading LLMs and measure how ``open-minded'' they are to arguments supporting opposite sides of these issues, whether different LLMs agree with each other, which arguments LLMs find most persuasive, and whether these arguments are the same for different LLMs. In general, we find that LLMs are open-minded on most issues. An authoritative source of information can easily sway an LLM's stance, highlighting the importance of source selection and the risk that LLM-based information retrieval and search systems can be manipulated.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12590v1">DPCheatSheet: Using Worked and Erroneous LLM-usage Examples to Scaffold Differential Privacy Implementation</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      This paper explores how programmers without specialized expertise in differential privacy (DP) (i.e., novices) can leverage LLMs to implement DP programs with minimal training. We first conducted a need-finding study with 6 novices and 3 experts to understand how they utilize LLMs in DP implementation. While DP experts can implement correct DP analyses through a few prompts, novices struggle to articulate their requirements in prompts and lack the skills to verify the correctness of the generated code. We then developed DPCheatSheet, an instructional tool that helps novices implement DP using LLMs. DPCheatSheet combines two learning concepts: it annotates an expert's workflow with LLMs as a worked example to bridge the expert mindset to novices, and it presents five common mistakes in LLM-based DP code generation as erroneous examples to support error-driven learning. We demonstrated the effectiveness of DPCheatSheet with an error identification study and an open-ended DP implementation study.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17539v3">Breaking Single-Tester Limits: Multi-Agent LLMs for Multi-User Feature Testing</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 Accepted to International Conference on Software Engineering (ICSE 2026). arXiv admin note: substantial text overlap with arXiv:2504.15474
    </div>
    <details class="paper-abstract">
      The growing dependence on mobile phones and their apps has made multi-user interactive features, like chat calls, live streaming, and video conferencing, indispensable for bridging the gaps in social connectivity caused by physical and situational barriers. However, automating these interactive features for testing is fraught with challenges, owing to their inherent need for timely, dynamic, and collaborative user interactions, which current automated testing methods inadequately address. Inspired by the concept of agents designed to autonomously and collaboratively tackle problems, we propose MAdroid, a novel multi-agent approach powered by the Large Language Models (LLMs) to automate the multi-user interactive task for app feature testing. Specifically, MAdroid employs two functional types of multi-agents: user agents (Operator) and supervisor agents (Coordinator and Observer). Each agent takes a specific role: the Coordinator directs the interactive task; the Operator mimics user interactions on the device; and the Observer monitors and reviews the task automation process. Our evaluation, which included 41 multi-user interactive tasks, demonstrates the effectiveness of our approach, achieving 82.9% of the tasks with 96.8% action similarity, outperforming the ablation studies and state-of-the-art baselines. Additionally, a preliminary investigation underscores MAdroid's practicality by helping identify 11 multi-user interactive bugs during regression app testing, confirming its potential value in real-world software development contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12546v1">Agent4FaceForgery: Multi-Agent LLM Framework for Realistic Face Forgery Detection</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      Face forgery detection faces a critical challenge: a persistent gap between offline benchmarks and real-world efficacy,which we attribute to the ecological invalidity of training data.This work introduces Agent4FaceForgery to address two fundamental problems: (1) how to capture the diverse intents and iterative processes of human forgery creation, and (2) how to model the complex, often adversarial, text-image interactions that accompany forgeries in social media. To solve this,we propose a multi-agent framework where LLM-poweredagents, equipped with profile and memory modules, simulate the forgery creation process. Crucially, these agents interact in a simulated social environment to generate samples labeled for nuanced text-image consistency, moving beyond simple binary classification. An Adaptive Rejection Sampling (ARS) mechanism ensures data quality and diversity. Extensive experiments validate that the data generated by our simulationdriven approach brings significant performance gains to detectors of multiple architectures, fully demonstrating the effectiveness and value of our framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08256v2">FIER: Fine-Grained and Efficient KV Cache Retrieval for Long-context LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 EMNLP2025 Camera-ready
    </div>
    <details class="paper-abstract">
      The Key-Value (KV) cache reading latency increases significantly with context lengths, hindering the efficiency of long-context LLM inference. To address this, previous works propose retaining a small fraction of KV cache based on token importance. For example, KV eviction uses static heuristics to retain tokens, while KV retrieval dynamically selects query-relevant tokens for more adaptive cache management. However, we observe that important tokens are often sparsely distributed across the long context. This sparsity makes existing page-level KV retrieval inaccurate, as each page may include irrelevant tokens and miss critical ones. In this work, we propose Fier, a \underline{Fi}ne-Grained and \underline{E}fficient KV cache \underline{R}etrieval method. Fier uses 1-bit quantized keys to estimate the importance of each token, resulting in efficient and precise retrieval. Experiments show that Fier matches full KV performance using only 11\% of the cache budget across various long-context tasks, reducing decoding latency by 1.2$\times$ to 1.5$\times$.Code is available at https://github.com/SimWangArizona/FIER
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13557v1">MACO: A Multi-Agent LLM-Based Hardware/Software Co-Design Framework for CGRAs</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      Coarse-grained Reconfigurable Arrays (CGRAs) are a promising computing architecture that can deliver high-performance, energy-efficient acceleration across diverse domains. By supporting reconfiguration at the functional unit level, CGRAs efficiently adapt to varying computational patterns and optimize resource utilization. However, designing CGRAs is highly challenging due to the vast design space, independent architectural parameters, and the time-consuming nature of manual design. Fortunately, the rapid advancement of large language models (LLMs) presents new opportunities to automate this process. In this work, we propose MACO -- an open-source multi-agent LLM-based framework for Hardware/Software (HW/SW) co-design of CGRAs. The framework employs LLM reasoning to generate CGRAs across four stages: HW/SW co-design, Design error correction, Best design selection, and Evaluation & Feedback. Furthermore, MACO iteratively optimizes the generated CGRAs, leveraging agent reasoning and feedback to achieve higher PPA (that is, power, performance, and area) design points for a given domain. In addition, we introduce an LLM self-learning mechanism that employs LLM-driven decision making to select the optimal CGRA to accelerate the design process. We evaluate the framework with state-of-the-art LLM-based methods and manual CGRA design, in terms of performance, power consumption, and area. Experimental results show that MACO efficiently generates high-quality CGRA architectures, significantly reducing manual design effort and demonstrating the potential of our framework for real-world CGRA design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.12274v2">Hierarchical LLMs In-the-loop Optimization for Real-time Multi-Robot Target Tracking under Unknown Hazards</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      Real-time multi-robot coordination in hazardous and adversarial environments requires fast, reliable adaptation to dynamic threats. While Large Language Models (LLMs) offer strong high-level reasoning capabilities, the lack of safety guarantees limits their direct use in critical decision-making. In this paper, we propose a hierarchical optimization framework that integrates LLMs into the decision loop for multi-robot target tracking in dynamic and hazardous environments. Rather than generating control actions directly, LLMs are used to generate task configuration and adjust parameters in a bi-level task allocation and planning problem. We formulate multi-robot coordination for tracking tasks as a bi-level optimization problem, with LLMs to reason about potential hazards in the environment and the status of the robot team and modify both the inner and outer levels of the optimization. This hierarchical approach enables real-time adjustments to the robots' behavior. Additionally, a human supervisor can offer broad guidance and assessments to address unexpected dangers, model mismatches, and performance issues arising from local minima. We validate our proposed framework in both simulation and real-world experiments with comprehensive evaluations, demonstrating its effectiveness and showcasing its capability for safe LLM integration for multi-robot systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13514v1">AQUA-LLM: Evaluating Accuracy, Quantization, and Adversarial Robustness Trade-offs in LLMs for Cybersecurity Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 Accepted by the 24th IEEE International Conference on Machine Learning and Applications (ICMLA'25)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently demonstrated strong potential for cybersecurity question answering (QA), supporting decision-making in real-time threat detection and response workflows. However, their substantial computational demands pose significant challenges for deployment on resource-constrained edge devices. Quantization, a widely adopted model compression technique, can alleviate these constraints. Nevertheless, quantization may degrade model accuracy and increase susceptibility to adversarial attacks. Fine-tuning offers a potential means to mitigate these limitations, but its effectiveness when combined with quantization remains insufficiently explored. Hence, it is essential to understand the trade-offs among accuracy, efficiency, and robustness. We propose AQUA-LLM, an evaluation framework designed to benchmark several state-of-the-art small LLMs under four distinct configurations: base, quantized-only, fine-tuned, and fine-tuned combined with quantization, specifically for cybersecurity QA. Our results demonstrate that quantization alone yields the lowest accuracy and robustness despite improving efficiency. In contrast, combining quantization with fine-tuning enhances both LLM robustness and predictive performance, achieving an optimal balance of accuracy, robustness, and efficiency. These findings highlight the critical need for quantization-aware, robustness-preserving fine-tuning methodologies to enable the robust and efficient deployment of LLMs for cybersecurity QA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13487v1">Prompt2DAG: A Modular Methodology for LLM-Based Data Enrichment Pipeline Generation</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      Developing reliable data enrichment pipelines demands significant engineering expertise. We present Prompt2DAG, a methodology that transforms natural language descriptions into executable Apache Airflow DAGs. We evaluate four generation approaches -- Direct, LLM-only, Hybrid, and Template-based -- across 260 experiments using thirteen LLMs and five case studies to identify optimal strategies for production-grade automation. Performance is measured using a penalized scoring framework that combines reliability with code quality (SAT), structural integrity (DST), and executability (PCT). The Hybrid approach emerges as the optimal generative method, achieving a 78.5% success rate with robust quality scores (SAT: 6.79, DST: 7.67, PCT: 7.76). This significantly outperforms the LLM-only (66.2% success) and Direct (29.2% success) methods. Our findings show that reliability, not intrinsic code quality, is the primary differentiator. Cost-effectiveness analysis reveals the Hybrid method is over twice as efficient as Direct prompting per successful DAG. We conclude that a structured, hybrid approach is essential for balancing flexibility and reliability in automated workflow generation, offering a viable path to democratize data pipeline development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13471v1">An LLM Agentic Approach for Legal-Critical Software: A Case Study for Tax Prep Software</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 To appear at ICSE 26. 12 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) show promise for translating natural-language statutes into executable logic, but reliability in legally critical settings remains challenging due to ambiguity and hallucinations. We present an agentic approach for developing legal-critical software, using U.S. federal tax preparation as a case study. The key challenge is test-case generation under the oracle problem, where correct outputs require interpreting law. Building on metamorphic testing, we introduce higher-order metamorphic relations that compare system outputs across structured shifts among similar individuals. Because authoring such relations is tedious and error-prone, we use an LLM-driven, role-based framework to automate test generation and code synthesis. We implement a multi-agent system that translates tax code into executable software and incorporates a metamorphic-testing agent that searches for counterexamples. In experiments, our framework using a smaller model (GPT-4o-mini) achieves a worst-case pass rate of 45%, outperforming frontier models (GPT-4o and Claude 3.5, 9-15%) on complex tax-code tasks. These results support agentic LLM methodologies as a path to robust, trustworthy legal-critical software from natural-language specifications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13400v1">Justice in Judgment: Unveiling (Hidden) Bias in LLM-assisted Peer Reviews</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      The adoption of large language models (LLMs) is transforming the peer review process, from assisting reviewers in writing more detailed evaluations to generating entire reviews automatically. While these capabilities offer exciting opportunities, they also raise critical concerns about fairness and reliability. In this paper, we investigate bias in LLM-generated peer reviews by conducting controlled experiments on sensitive metadata, including author affiliation and gender. Our analysis consistently shows affiliation bias favoring institutions highly ranked on common academic rankings. Additionally, we find some gender preferences, which, even though subtle in magnitude, have the potential to compound over time. Notably, we uncover implicit biases that become more evident with token-based soft ratings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19749v2">What's Not Said Still Hurts: A Description-Based Evaluation Framework for Measuring Social Bias in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 EMNLP Findings 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often exhibit social biases inherited from their training data. While existing benchmarks evaluate bias by term-based mode through direct term associations between demographic terms and bias terms, LLMs have become increasingly adept at avoiding biased responses, leading to seemingly low levels of bias. However, biases persist in subtler, contextually hidden forms that traditional benchmarks fail to capture. We introduce the Description-based Bias Benchmark (DBB), a novel dataset designed to assess bias at the semantic level that bias concepts are hidden within naturalistic, subtly framed contexts in real-world scenarios rather than superficial terms. We analyze six state-of-the-art LLMs, revealing that while models reduce bias in response at the term level, they continue to reinforce biases in nuanced settings. Data, code, and results are available at https://github.com/JP-25/Description-based-Bias-Benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14289v1">From Capabilities to Performance: Evaluating Key Functional Properties of LLM Architectures in Penetration Testing</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to automate or augment penetration testing, but their effectiveness and reliability across attack phases remain unclear. We present a comprehensive evaluation of multiple LLM-based agents, from single-agent to modular designs, across realistic penetration testing scenarios, measuring empirical performance and recurring failure patterns. We also isolate the impact of five core functional capabilities via targeted augmentations: Global Context Memory (GCM), Inter-Agent Messaging (IAM), Context-Conditioned Invocation (CCI), Adaptive Planning (AP), and Real-Time Monitoring (RTM). These interventions support, respectively: (i) context coherence and retention, (ii) inter-component coordination and state management, (iii) tool use accuracy and selective execution, (iv) multi-step strategic planning, error detection, and recovery, and (v) real-time dynamic responsiveness. Our results show that while some architectures natively exhibit subsets of these properties, targeted augmentations substantially improve modular agent performance, especially in complex, multi-step, and real-time penetration testing tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14285v1">A Multi-Agent LLM Defense Pipeline Against Prompt Injection Attacks</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      Prompt injection attacks represent a major vulnerability in Large Language Model (LLM) deployments, where malicious instructions embedded in user inputs can override system prompts and induce unintended behaviors. This paper presents a novel multi-agent defense framework that employs specialized LLM agents in coordinated pipelines to detect and neutralize prompt injection attacks in real-time. We evaluate our approach using two distinct architectures: a sequential chain-of-agents pipeline and a hierarchical coordinator-based system. Our comprehensive evaluation on 55 unique prompt injection attacks, grouped into 8 categories and totaling 400 attack instances across two LLM platforms (ChatGLM and Llama2), demonstrates significant security improvements. Without defense mechanisms, baseline Attack Success Rates (ASR) reached 30% for ChatGLM and 20% for Llama2. Our multi-agent pipeline achieved 100% mitigation, reducing ASR to 0% across all tested scenarios. The framework demonstrates robustness across multiple attack categories including direct overrides, code execution attempts, data exfiltration, and obfuscation techniques, while maintaining system functionality for legitimate queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14275v1">FedMentor: Domain-Aware Differential Privacy for Heterogeneous Federated LLMs in Mental Health</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 (e.g.: 18 pages, 6 figures, 6 tables)
    </div>
    <details class="paper-abstract">
      Privacy-preserving adaptation of Large Language Models (LLMs) in sensitive domains (e.g., mental health) requires balancing strict confidentiality with model utility and safety. We propose FedMentor, a federated fine-tuning framework that integrates Low-Rank Adaptation (LoRA) and domain-aware Differential Privacy (DP) to meet per-domain privacy budgets while maintaining performance. Each client (domain) applies a custom DP noise scale proportional to its data sensitivity, and the server adaptively reduces noise when utility falls below a threshold. In experiments on three mental health datasets, we show that FedMentor improves safety over standard Federated Learning without privacy, raising safe output rates by up to three points and lowering toxicity, while maintaining utility (BERTScore F1 and ROUGE-L) within 0.5% of the non-private baseline and close to the centralized upper bound. The framework scales to backbones with up to 1.7B parameters on single-GPU clients, requiring < 173 MB of communication per round. FedMentor demonstrates a practical approach to privately fine-tune LLMs for safer deployments in healthcare and other sensitive fields.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14274v1">Discovering New Theorems via LLMs with In-Context Proof Learning in Lean</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 11 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Large Language Models have demonstrated significant promise in formal theorem proving. However, previous works mainly focus on solving existing problems. In this paper, we focus on the ability of LLMs to find novel theorems. We propose Conjecturing-Proving Loop pipeline for automatically generating mathematical conjectures and proving them in Lean 4 format. A feature of our approach is that we generate and prove further conjectures with context including previously generated theorems and their proofs, which enables the generation of more difficult proofs by in-context learning of proof strategies without changing parameters of LLMs. We demonstrated that our framework rediscovered theorems with verification, which were published in past mathematical papers and have not yet formalized. Moreover, at least one of these theorems could not be proved by the LLM without in-context learning, even in natural language, which means that in-context learning was effective for neural theorem proving. The source code is available at https://github.com/auto-res/ConjecturingProvingLoop.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14273v1">Automated and Context-Aware Code Documentation Leveraging Advanced LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      Code documentation is essential to improve software maintainability and comprehension. The tedious nature of manual code documentation has led to much research on automated documentation generation. Existing automated approaches primarily focused on code summarization, leaving a gap in template-based documentation generation (e.g., Javadoc), particularly with publicly available Large Language Models (LLMs). Furthermore, progress in this area has been hindered by the lack of a Javadoc-specific dataset that incorporates modern language features, provides broad framework/library coverage, and includes necessary contextual information. This study aims to address these gaps by developing a tailored dataset and assessing the capabilities of publicly available LLMs for context-aware, template-based Javadoc generation. In this work, we present a novel, context-aware dataset for Javadoc generation that includes critical structural and semantic information from modern Java codebases. We evaluate five open-source LLMs (including LLaMA-3.1, Gemma-2, Phi-3, Mistral, Qwen-2.5) using zero-shot, few-shot, and fine-tuned setups and provide a comparative analysis of their performance. Our results demonstrate that LLaMA 3.1 performs consistently well and is a reliable candidate for practical, automated Javadoc generation, offering a viable alternative to proprietary systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13268v1">LLMs for energy and macronutrients estimation using only text data from 24-hour dietary recalls: a parameter-efficient fine-tuning experiment using a 10-shot prompt</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 https://github.com/rodrigo-carrillo/LLMs-Macronutrient-Estimation-NHANES-Adolescents
    </div>
    <details class="paper-abstract">
      BACKGROUND: Most artificial intelligence tools used to estimate nutritional content rely on image input. However, whether large language models (LLMs) can accurately predict nutritional values based solely on text descriptions of foods consumed remains unknown. If effective, this approach could enable simpler dietary monitoring without the need for photographs. METHODS: We used 24-hour dietary recalls from adolescents aged 12-19 years in the National Health and Nutrition Examination Survey (NHANES). An open-source quantized LLM was prompted using a 10-shot, chain-of-thought approach to estimate energy and five macronutrients based solely on text strings listing foods and their quantities. We then applied parameter-efficient fine-tuning (PEFT) to evaluate whether predictive accuracy improved. NHANES-calculated values served as the ground truth for energy, proteins, carbohydrates, total sugar, dietary fiber and total fat. RESULTS: In a pooled dataset of 11,281 adolescents (49.9% male, mean age 15.4 years), the vanilla LLM yielded poor predictions. The mean absolute error (MAE) was 652.08 for energy and the Lin's CCC <0.46 across endpoints. In contrast, the fine-tuned model performed substantially better, with energy MAEs ranging from 171.34 to 190.90 across subsets, and Lin's CCC exceeding 0.89 for all outcomes. CONCLUSIONS: When prompted using a chain-of-thought approach and fine-tuned with PEFT, open-source LLMs exposed solely to text input can accurately predict energy and macronutrient values from 24-hour dietary recalls. This approach holds promise for low-burden, text-based dietary monitoring tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13244v1">Evaluating LLM Alignment on Personality Inference from Real-World Interview Data</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 8 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed in roles requiring nuanced psychological understanding, such as emotional support agents, counselors, and decision-making assistants. However, their ability to interpret human personality traits, a critical aspect of such applications, remains unexplored, particularly in ecologically valid conversational settings. While prior work has simulated LLM "personas" using discrete Big Five labels on social media data, the alignment of LLMs with continuous, ground-truth personality assessments derived from natural interactions is largely unexamined. To address this gap, we introduce a novel benchmark comprising semi-structured interview transcripts paired with validated continuous Big Five trait scores. Using this dataset, we systematically evaluate LLM performance across three paradigms: (1) zero-shot and chain-of-thought prompting with GPT-4.1 Mini, (2) LoRA-based fine-tuning applied to both RoBERTa and Meta-LLaMA architectures, and (3) regression using static embeddings from pretrained BERT and OpenAI's text-embedding-3-small. Our results reveal that all Pearson correlations between model predictions and ground-truth personality traits remain below 0.26, highlighting the limited alignment of current LLMs with validated psychological constructs. Chain-of-thought prompting offers minimal gains over zero-shot, suggesting that personality inference relies more on latent semantic representation than explicit reasoning. These findings underscore the challenges of aligning LLMs with complex human attributes and motivate future work on trait-specific prompting, context-aware modeling, and alignment-oriented fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13237v1">Metacognitive Reuse: Turning Recurring LLM Reasoning Into Concise Behaviors</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 18 pages, 9 Figures, 5 Tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) now solve multi-step problems by emitting extended chains of thought. During the process, they often re-derive the same intermediate steps across problems, inflating token usage and latency. This saturation of the context window leaves less capacity for exploration. We study a simple mechanism that converts recurring reasoning fragments into concise, reusable "behaviors" (name + instruction) via the model's own metacognitive analysis of prior traces. These behaviors are stored in a "behavior handbook" which supplies them to the model in-context at inference or distills them into parameters via supervised fine-tuning. This approach achieves improved test-time reasoning across three different settings - 1) Behavior-conditioned inference: Providing the LLM relevant behaviors in-context during reasoning reduces number of reasoning tokens by up to 46% while matching or improving baseline accuracy; 2) Behavior-guided self-improvement: Without any parameter updates, the model improves its own future reasoning by leveraging behaviors from its own past problem solving attempts. This yields up to 10% higher accuracy than a naive critique-and-revise baseline; and 3) Behavior-conditioned SFT: SFT on behavior-conditioned reasoning traces is more effective at converting non-reasoning models into reasoning models as compared to vanilla SFT. Together, these results indicate that turning slow derivations into fast procedural hints enables LLMs to remember how to reason, not just what to conclude.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13201v1">Scaling Up Throughput-oriented LLM Inference Applications on Heterogeneous Opportunistic GPU Clusters with Pervasive Context Management</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      The widespread growth in LLM developments increasingly demands more computational power from clusters than what they can supply. Traditional LLM applications inherently require huge static resource allocations, which force users to either wait in a long job queue and accept progress delay, or buy expensive hardware to fulfill their needs and exacerbate the demand-supply problem. However, not all LLM applications are latency-sensitive and can instead be executed in a throughput-oriented way. This throughput orientation allows a dynamic allocation that opportunistically pools available resources over time, avoiding both the long queue and expensive GPU purchases. Effectively utilizing opportunistic resources brings numerous challenges nevertheless. Our solution, pervasive context management, exploits the common computational context in LLM applications and provides mechanisms and policies that allow seamless context reuse on opportunistic resources. Our evaluation shows an LLM application with pervasive context management on opportunistic resources reduces its execution time by 98.1%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13179v1">Efficient Cold-Start Recommendation via BPE Token-Level Embedding Initialization with LLM</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      The cold-start issue is the challenge when we talk about recommender systems, especially in the case when we do not have the past interaction data of new users or new items. Content-based features or hybrid solutions are common as conventional solutions, but they can only work in a sparse metadata environment with shallow patterns. In this paper, the efficient cold-start recommendation strategy is presented, which is based on the sub word-level representations by applying Byte Pair Encoding (BPE) tokenization and pre-trained Large Language Model (LLM) embedding in the initialization procedure. We obtain fine-grained token-level vectors that are aligned with the BPE vocabulary as opposed to using coarse-grained sentence embeddings. Together, these token embeddings can be used as dense semantic priors on unseen entities, making immediate recommendation performance possible without user-item interaction history. Our mechanism can be compared to collaborative filtering systems and tested over benchmark datasets with stringent cold-start assumptions. Experimental findings show that the given BPE-LLM method achieves higher Recall@k, NDCG@k, and Hit Rate measurements compared to the standard baseline and displays the same capability of sufficient computational performance. Furthermore, we demonstrate that using subword-aware embeddings yields better generalizability and is more interpretable, especially within a multilingual and sparse input setting. The practical application of token-level semantic initialization as a lightweight, but nevertheless effective extension to modern recommender systems in the zero-shot setting is indicated within this work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13175v1">More performant and scalable: Rethinking contrastive vision-language pre-training of radiology in the LLM era</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 MICCAI 2025
    </div>
    <details class="paper-abstract">
      The emergence of Large Language Models (LLMs) presents unprecedented opportunities to revolutionize medical contrastive vision-language pre-training. In this paper, we show how LLMs can facilitate large-scale supervised pre-training, thereby advancing vision-language alignment. We begin by demonstrate that modern LLMs can automatically extract diagnostic labels from radiology reports with remarkable precision (>96\% AUC in our experiments) without complex prompt engineering, enabling the creation of large-scale "silver-standard" datasets at a minimal cost (~\$3 for 50k CT image-report pairs). Further, we find that vision encoder trained on this "silver-standard" dataset achieves performance comparable to those trained on labels extracted by specialized BERT-based models, thereby democratizing the access to large-scale supervised pre-training. Building on this foundation, we proceed to reveal that supervised pre-training fundamentally improves contrastive vision-language alignment. Our approach achieves state-of-the-art performance using only a 3D ResNet-18 with vanilla CLIP training, including 83.8\% AUC for zero-shot diagnosis on CT-RATE, 77.3\% AUC on RAD-ChestCT, and substantial improvements in cross-modal retrieval (MAP@50=53.7\% for image-image, Recall@100=52.2\% for report-image). These results demonstrate the potential of utilizing LLMs to facilitate {\bf more performant and scalable} medical AI systems. Our code is avaiable at https://github.com/SadVoxel/More-performant-and-scalable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13154v1">LLM Hallucination Detection: A Fast Fourier Transform Method Based on Hidden Layer Temporal Signals</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      Hallucination remains a critical barrier for deploying large language models (LLMs) in reliability-sensitive applications. Existing detection methods largely fall into two categories: factuality checking, which is fundamentally constrained by external knowledge coverage, and static hidden-state analysis, that fails to capture deviations in reasoning dynamics. As a result, their effectiveness and robustness remain limited. We propose HSAD (Hidden Signal Analysis-based Detection), a novel hallucination detection framework that models the temporal dynamics of hidden representations during autoregressive generation. HSAD constructs hidden-layer signals by sampling activations across layers, applies Fast Fourier Transform (FFT) to obtain frequency-domain representations, and extracts the strongest non-DC frequency component as spectral features. Furthermore, by leveraging the autoregressive nature of LLMs, HSAD identifies optimal observation points for effective and reliable detection. Across multiple benchmarks, including TruthfulQA, HSAD achieves over 10 percentage points improvement compared to prior state-of-the-art methods. By integrating reasoning-process modeling with frequency-domain analysis, HSAD establishes a new paradigm for robust hallucination detection in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13127v1">Empowering LLMs with Parameterized Skills for Adversarial Long-Horizon Planning</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 Accepted to IJCNN 2025
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models(LLMs) have led to the development of LLM-based AI agents. A key challenge is the creation of agents that can effectively ground themselves in complex, adversarial long-horizon environments. Existing methods mainly focus on (1) using LLMs as policies to interact with the environment through generating low-level feasible actions, and (2) utilizing LLMs to generate high-level tasks or language guides to stimulate action generation. However, the former struggles to generate reliable actions, while the latter relies heavily on expert experience to translate high-level tasks into specific action sequences. To address these challenges, we introduce the Plan with Language, Act with Parameter (PLAP) planning framework that facilitates the grounding of LLM-based agents in long-horizon environments. The PLAP method comprises three key components: (1) a skill library containing environment-specific parameterized skills, (2) a skill planner powered by LLMs, and (3) a skill executor converting the parameterized skills into executable action sequences. We implement PLAP in MicroRTS, a long-horizon real-time strategy game that provides an unfamiliar and challenging environment for LLMs. The experimental results demonstrate the effectiveness of PLAP. In particular, GPT-4o-driven PLAP in a zero-shot setting outperforms 80% of baseline agents, and Qwen2-72B-driven PLAP, with carefully crafted few-shot examples, surpasses the top-tier scripted agent, CoacAI. Additionally, we design comprehensive evaluation metrics and test 6 closed-source and 2 open-source LLMs within the PLAP framework, ultimately releasing an LLM leaderboard ranking long-horizon skill planning ability. Our code is available at https://github.com/AI-Research-TeamX/PLAP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18436v3">Can Code-Switched Texts Activate a Knowledge Switch in LLMs? A Case Study on English-Korean Code-Switching</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 Accepted to EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Recent large language models (LLMs) demonstrate multilingual abilities, yet they are English-centric due to dominance of English in training corpora. The limited resource for low-resource languages remains a crucial challenge. Code-switching (CS), a phenomenon where multilingual speakers alternate between languages in a discourse, can convey subtle cultural and linguistic nuances that can be otherwise lost in translation and elicits language-specific knowledge in human communications. In light of this, we investigate whether code-switching can activate, or identify and leverage knowledge for reasoning when LLMs solve low-resource language tasks. To facilitate the research, we first present EnKoQA, a synthetic English-Korean CS question-answering dataset. We provide comprehensive analysis on a variety of multilingual LLMs by subdividing activation process into knowledge identification and knowledge leveraging. Our results demonstrate that compared to English text, CS can faithfully activate knowledge inside LLMs especially on language-specific domains, suggesting the potential of code-switching on low-resource language tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13103v1">Accelerating Discovery: Rapid Literature Screening with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 This version of the manuscript has been submitted to Empirical Software Engieering Journal for consideration
    </div>
    <details class="paper-abstract">
      Background: Conducting Multi Vocal Literature Reviews (MVLRs) is often time and effort-intensive. Researchers must review and filter a large number of unstructured sources, which frequently contain sparse information and are unlikely to be included in the final study. Our experience conducting an MVLR on Context-Aware Software Systems (CASS) Testing in the avionics domain exemplified this challenge, with over 8,000 highly heterogeneous documents requiring review. Therefore, we developed a Large Language Model (LLM) assistant to support the search and filtering of documents. Aims: To develop and validate an LLM based tool that can support researchers in performing the search and filtering of documents for an MVLR without compromising the rigor of the research protocol. Method: We applied sound engineering practices to develop an on-premises LLM-based tool incorporating Retrieval Augmented Generation (RAG) to process candidate sources. Progress towards the aim was quantified using the Positive Percent Agreement (PPA) as the primary metric to ensure the performance of the LLM based tool. Convenience sampling, supported by human judgment and statistical sampling, were used to verify and validate the tool's quality-in-use. Results: The tool currently demonstrates a PPA agreement with human researchers of 90% for sources that are not relevant to the study. Development details are shared to support domain-specific adaptation of the tool. Conclusions: Using LLM-based tools to support academic researchers in rigorous MVLR is feasible. These tools can free valuable time for higher-level, abstract tasks. However, researcher participation remains essential to ensure that the tool supports thorough research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21740v2">Counterfactual Simulatability of LLM Explanations for Generation Tasks</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      LLMs can be unpredictable, as even slight alterations to the prompt can cause the output to change in unexpected ways. Thus, the ability of models to accurately explain their behavior is critical, especially in high-stakes settings. One approach for evaluating explanations is counterfactual simulatability, how well an explanation allows users to infer the model's output on related counterfactuals. Counterfactual simulatability has been previously studied for yes/no question answering tasks. We provide a general framework for extending this method to generation tasks, using news summarization and medical suggestion as example use cases. We find that while LLM explanations do enable users to better predict LLM outputs on counterfactuals in the summarization setting, there is significant room for improvement for medical suggestion. Furthermore, our results suggest that the evaluation for counterfactual simulatability may be more appropriate for skill-based tasks as opposed to knowledge-based tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13055v1">Automating Code Generation for Semiconductor Equipment Control from Developer Utterances with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      Semiconductors form the backbone of modern electronics, with their manufacturing and testing relying on highly specialized equipment and domain-specific programming languages. Equipment languages such as the Algorithmic Pattern Generator (ALPG) are critical for precise hardware control but are challenging to program due to their low-level syntax and steep learning curve. While large language models (LLMs) have shown promise in generating high-level code from natural language, their effectiveness on low-level equipment languages remains limited. To address this, we propose Progressive Knowledge Enhancement (PKE), a novel multi-stage prompting framework that progressively extracts and activates the latent knowledge within LLMs, guiding them from simple to complex examples without extensive fine-tuning. Empirical evaluation on an industrial ALPG dataset shows that PKE significantly outperforms standard prompting and surpasses state-of-the-art methods in generating correct ALPG code, achieving 11.1\% and 15.2\% higher exact match scores compared to the second-best technique. Further analysis of individual components confirms that progressive knowledge extraction based on difficulty enhances accuracy. Our study offer a practical approach to boosting LLM capabilities for specialized low-level programming, supporting greater productivity in semiconductor software development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08045v3">Break the Checkbox: Challenging Closed-Style Evaluations of Cultural Alignment in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 Accepted at EMNLP 2025 (Main)
    </div>
    <details class="paper-abstract">
      A large number of studies rely on closed-style multiple-choice surveys to evaluate cultural alignment in Large Language Models (LLMs). In this work, we challenge this constrained evaluation paradigm and explore more realistic, unconstrained approaches. Using the World Values Survey (WVS) and Hofstede Cultural Dimensions as case studies, we demonstrate that LLMs exhibit stronger cultural alignment in less constrained settings, where responses are not forced. Additionally, we show that even minor changes, such as reordering survey choices, lead to inconsistent outputs, exposing the limitations of closed-style evaluations. Our findings advocate for more robust and flexible evaluation frameworks that focus on specific cultural proxies, encouraging more nuanced and accurate assessments of cultural alignment in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16467v2">Reading Between the Prompts: How Stereotypes Shape LLM's Implicit Personalization</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 Accepted at EMNLP Main 2025
    </div>
    <details class="paper-abstract">
      Generative Large Language Models (LLMs) infer user's demographic information from subtle cues in the conversation -- a phenomenon called implicit personalization. Prior work has shown that such inferences can lead to lower quality responses for users assumed to be from minority groups, even when no demographic information is explicitly provided. In this work, we systematically explore how LLMs respond to stereotypical cues using controlled synthetic conversations, by analyzing the models' latent user representations through both model internals and generated answers to targeted user questions. Our findings reveal that LLMs do infer demographic attributes based on these stereotypical signals, which for a number of groups even persists when the user explicitly identifies with a different demographic group. Finally, we show that this form of stereotype-driven implicit personalization can be effectively mitigated by intervening on the model's internal representations using a trained linear probe to steer them toward the explicitly stated identity. Our results highlight the need for greater transparency and control in how LLMs represent user identity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13021v1">xOffense: An AI-driven autonomous penetration testing framework with offensive knowledge-enhanced LLMs and multi agent systems</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 17 pages, 4 figures
    </div>
    <details class="paper-abstract">
      This work introduces xOffense, an AI-driven, multi-agent penetration testing framework that shifts the process from labor-intensive, expert-driven manual efforts to fully automated, machine-executable workflows capable of scaling seamlessly with computational infrastructure. At its core, xOffense leverages a fine-tuned, mid-scale open-source LLM (Qwen3-32B) to drive reasoning and decision-making in penetration testing. The framework assigns specialized agents to reconnaissance, vulnerability scanning, and exploitation, with an orchestration layer ensuring seamless coordination across phases. Fine-tuning on Chain-of-Thought penetration testing data further enables the model to generate precise tool commands and perform consistent multi-step reasoning. We evaluate xOffense on two rigorous benchmarks: AutoPenBench and AI-Pentest-Benchmark. The results demonstrate that xOffense consistently outperforms contemporary methods, achieving a sub-task completion rate of 79.17%, decisively surpassing leading systems such as VulnBot and PentestGPT. These findings highlight the potential of domain-adapted mid-scale LLMs, when embedded within structured multi-agent orchestration, to deliver superior, cost-efficient, and reproducible solutions for autonomous penetration testing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.19894v3">TRANSAGENT: An LLM-Based Multi-Agent System for Code Translation</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      Code translation converts code from one programming language to another while maintaining its original functionality, which is crucial for software migration, system refactoring, and cross-platform development. Traditional rule-based methods rely on manually-written rules, which can be time-consuming and often result in less readable code. To overcome this, learning-based methods have been developed, leveraging parallel data to train models for automated code translation. More recently, the advance of Large Language Models (LLMs) further boosts learning-based code translation. Although promising, LLM-translated program still suffers from diverse quality issues (e.g., syntax errors and semantic errors). In particular, it can be challenging for LLMs to self-debug these errors when simply provided with the corresponding error messages. In this work, we propose a novel LLM-based multi-agent system TRANSAGENT, which enhances LLM-based code translation by fixing the syntax errors and semantic errors with the synergy between four LLM-based agents, including Initial Code Translator, Syntax Error Fixer, Code Aligner, and Semantic Error Fixer. The main insight of TRANSAGENT is to first localize the error code block in the target program based on the execution alignment between the target and source program, which can narrow down the fixing space and thus lower down the fixing difficulties. To evaluate TRANSAGENT, we first construct a new benchmark from recent programming tasks to mitigate the potential data leakage issue. On our benchmark, TRANSAGENT outperforms the latest LLM-based code translation technique UniTrans in both translation effectiveness and efficiency; additionally, our evaluation on different LLMs show the generalization of TRANSAGENT and our ablation study shows the contribution of each agent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12067v3">TokenSkip: Controllable Chain-of-Thought Compression in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 EMNLP 2025 (Long Paper), camera-ready version
    </div>
    <details class="paper-abstract">
      Chain-of-Thought (CoT) has been proven effective in enhancing the reasoning capabilities of large language models (LLMs). Recent advancements, such as OpenAI's o1 and DeepSeek-R1, suggest that scaling up the length of CoT sequences during inference could further boost LLM reasoning performance. However, due to the autoregressive nature of LLM decoding, longer CoT outputs lead to a linear increase in inference latency, adversely affecting user experience, particularly when the CoT exceeds 10,000 tokens. To address this limitation, we analyze the semantic importance of tokens within CoT outputs and reveal that their contributions to reasoning vary. Building on this insight, we propose TokenSkip, a simple yet effective approach that enables LLMs to selectively skip less important tokens, allowing for controllable CoT compression. Extensive experiments across various models and tasks demonstrate the effectiveness of TokenSkip in reducing CoT token usage while preserving strong reasoning performance. Notably, when applied to Qwen2.5-14B-Instruct, TokenSkip reduces reasoning tokens by 40% (from 313 to 181) on GSM8K, with less than a 0.4% performance drop. We release our code and checkpoints in https://github.com/hemingkx/TokenSkip.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16408v2">From Surveys to Narratives: Rethinking Cultural Value Adaptation in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      Adapting cultural values in Large Language Models (LLMs) presents significant challenges, particularly due to biases and limited training data. Prior work primarily aligns LLMs with different cultural values using World Values Survey (WVS) data. However, it remains unclear whether this approach effectively captures cultural nuances or produces distinct cultural representations for various downstream tasks. In this paper, we systematically investigate WVS-based training for cultural value adaptation and find that relying solely on survey data can homogenize cultural norms and interfere with factual knowledge. To investigate these issues, we augment WVS with encyclopedic and scenario-based cultural narratives from Wikipedia and NormAd. While these narratives may have variable effects on downstream tasks, they consistently improve cultural distinctiveness than survey data alone. Our work highlights the inherent complexity of aligning cultural values with the goal of guiding task-specific behavior. We release our code at https://github.com/faridlazuarda/from-surveys-to-narratives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12769v3">How Much Do LLMs Hallucinate across Languages? On Multilingual Estimation of LLM Hallucination in the Wild</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 EMNLP 2025
    </div>
    <details class="paper-abstract">
      In the age of misinformation, hallucination -- the tendency of Large Language Models (LLMs) to generate non-factual or unfaithful responses -- represents the main risk for their global utility. Despite LLMs becoming increasingly multilingual, the vast majority of research on detecting and quantifying LLM hallucination are (a) English-centric and (b) focus on machine translation (MT) and summarization, tasks that are less common ``in the wild'' than open information seeking. In contrast, we aim to quantify the extent of LLM hallucination across languages in knowledge-intensive long-form question answering. To this end, we train a multilingual hallucination detection model and conduct a large-scale study across 30 languages and 6 open-source LLM families. We start from an English hallucination detection dataset and rely on MT to generate (noisy) training data in other languages. We also manually annotate gold data for five high-resource languages; we then demonstrate, for these languages, that the estimates of hallucination rates are similar between silver (LLM-generated) and gold test sets, validating the use of silver data for estimating hallucination rates for other languages. For the final rates estimation, we build a knowledge-intensive QA dataset for 30 languages with LLM-generated prompts and Wikipedia articles as references. We find that, while LLMs generate longer responses with more hallucinated tokens for higher-resource languages, there is no correlation between length-normalized hallucination rates of languages and their digital representation. Further, we find that smaller LLMs exhibit larger hallucination rates than larger models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12961v1">Do LLMs Understand Wine Descriptors Across Cultures? A Benchmark for Cultural Adaptations of Wine Reviews</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have opened the door to culture-aware language tasks. We introduce the novel problem of adapting wine reviews across Chinese and English, which goes beyond literal translation by incorporating regional taste preferences and culture-specific flavor descriptors. In a case study on cross-cultural wine review adaptation, we compile the first parallel corpus of professional reviews, containing 8k Chinese and 16k Anglophone reviews. We benchmark both neural-machine-translation baselines and state-of-the-art LLMs with automatic metrics and human evaluation. For the latter, we propose three culture-oriented criteria -- Cultural Proximity, Cultural Neutrality, and Cultural Genuineness -- to assess how naturally a translated review resonates with target-culture readers. Our analysis shows that current models struggle to capture cultural nuances, especially in translating wine descriptions across different cultures. This highlights the challenges and limitations of translation models in handling cultural content.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08523v3">InferLog: Accelerating LLM Inference for Online Log Parsing via ICL-oriented Prefix Caching</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 Accepted by ICSE '26 (The 48th IEEE/ACM International Conference on Software Engineering)
    </div>
    <details class="paper-abstract">
      Modern software systems generate massive volumes of runtime logs, necessitating efficient and accurate log parsing to enable critical downstream tasks such as anomaly detection and root cause analysis. Recently, large language models (LLMs) have achieved advanced accuracy on log parsing, but their deployment in production environments faces two major limitations: (1) the privacy risks associated with commercial LLMs, driving the adoption of local deployment, and (2) the stringent latency and throughput requirements imposed by high-volume log streams, which existing LLM-based parsers fail to meet. Although recent efforts have reduced the number of LLM queries, they overlook the high latency of the LLM invocations, where concurrent log parsing requests can cause serve performance degradation of LLM inference system. In this study, we present InferLog, the first LLM inference optimization method for online log parsing. Our key insight is that the inference efficiency emerges as the vital bottleneck in LLM-based online log parsing, rather than parsing accuracy. InferLog accelerates inference by designing (1) A Prefix-aware ICL Refinement policy to refine the examples and permutation of in-context learning to improve the prefix caching efficiency. (2) A rapid and task-specific configuration tuning pipeline based on meta-learning to find the optimal LLM scheduling-related configuration for dynamic log parsing workloads. The experimental results based on Loghub dataset and vLLM demonstrate that InferLog significantly outperforms existing inference optimization methods and markedly accelerates the state-of-the-art LLM-based log parser without compromising parsing accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12914v1">Stochastic Streets: A Walk Through Random LLM Address Generation in four European Cities</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are capable of solving complex math problems or answer difficult questions on almost any topic, but can they generate random street addresses for European cities?
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12107v1">Exploring Conversational Design Choices in LLMs for Pedagogical Purposes: Socratic and Narrative Approaches for Improving Instructor's Teaching Practice</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) typically generate direct answers, yet they are increasingly used as learning tools. Studying instructors' usage is critical, given their role in teaching and guiding AI adoption in education. We designed and evaluated TeaPT, an LLM for pedagogical purposes that supports instructors' professional development through two conversational approaches: a Socratic approach that uses guided questioning to foster reflection, and a Narrative approach that offers elaborated suggestions to extend externalized cognition. In a mixed-method study with 41 higher-education instructors, the Socratic version elicited greater engagement, while the Narrative version was preferred for actionable guidance. Subgroup analyses further revealed that less-experienced, AI-optimistic instructors favored the Socratic version, whereas more-experienced, AI-cautious instructors preferred the Narrative version. We contribute design implications for LLMs for pedagogical purposes, showing how adaptive conversational approaches can support instructors with varied profiles while highlighting how AI attitudes and experience shape interaction and learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12104v1">JustEva: A Toolkit to Evaluate LLM Fairness in Legal Knowledge Inference</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 This paper has been accepted at CIKM 2025 (Demo Track)
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into legal practice raises pressing concerns about judicial fairness, particularly due to the nature of their "black-box" processes. This study introduces JustEva, a comprehensive, open-source evaluation toolkit designed to measure LLM fairness in legal tasks. JustEva features several advantages: (1) a structured label system covering 65 extra-legal factors; (2) three core fairness metrics - inconsistency, bias, and imbalanced inaccuracy; (3) robust statistical inference methods; and (4) informative visualizations. The toolkit supports two types of experiments, enabling a complete evaluation workflow: (1) generating structured outputs from LLMs using a provided dataset, and (2) conducting statistical analysis and inference on LLMs' outputs through regression and other statistical methods. Empirical application of JustEva reveals significant fairness deficiencies in current LLMs, highlighting the lack of fair and trustworthy LLM legal tools. JustEva offers a convenient tool and methodological foundation for evaluating and improving algorithmic fairness in the legal domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17711v2">Enhancing LLM-Based Social Bot via an Adversarial Learning Framework</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      Developing Large Language Model (LLM) agents that exhibit human-like behavior, encompassing not only individual heterogeneity rooted in unique user profiles but also adaptive response to socially connected neighbors, is a significant research challenge. Social media platforms, with their diverse user data and explicit social structures, provide an ideal testbed for such investigations. This paper introduces EvoBot, an \textbf{Evo}lving LLM-based social \textbf{Bot} that significantly enhances human-like generative capabilities through a novel adversarial learning framework. EvoBot is initialized by Supervised Fine-Tuning (SFT) on representative data from social media and then iteratively refines its generation of sophisticated, human-like content via Direct Preference Optimization (DPO). This refinement is guided by feedback from a co-adapting \textbf{Detector} which concurrently improves its ability to distinguish EvoBot from humans, thereby creating an increasingly challenging learning environment for EvoBot. Experiments demonstrate that EvoBot generates content aligned with diverse user profiles, increasingly bypassing the co-adapting Detector through human-like expression. Moreover, it exhibits strong social responsiveness, more accurately modeling real-world opinion dynamics and information spread in multi-agent simulations. The framework also yields a more robust Detector, underscoring its broader utility for both advanced agent development and related detection tasks. The code is available at https://github.com/kfq20/EvoBot.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12102v1">Can LLMs Address Mental Health Questions? A Comparison with Human Therapists</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      Limited access to mental health care has motivated the use of digital tools and conversational agents powered by large language models (LLMs), yet their quality and reception remain unclear. We present a study comparing therapist-written responses to those generated by ChatGPT, Gemini, and Llama for real patient questions. Text analysis showed that LLMs produced longer, more readable, and lexically richer responses with a more positive tone, while therapist responses were more often written in the first person. In a survey with 150 users and 23 licensed therapists, participants rated LLM responses as clearer, more respectful, and more supportive than therapist-written answers. Yet, both groups of participants expressed a stronger preference for human therapist support. These findings highlight the promise and limitations of LLMs in mental health, underscoring the need for designs that balance their communicative strengths with concerns of trust, privacy, and accountability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12021v1">LitterBox+: An Extensible Framework for LLM-enhanced Scratch Static Code Analysis</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 ASE 2025 Tool Demonstration Track
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have become an essential tool to support developers using traditional text-based programming languages, but the graphical notation of the block-based Scratch programming environment inhibits the use of LLMs. To overcome this limitation, we propose the LitterBox+ framework that extends the Scratch static code analysis tool LitterBox with the generative abilities of LLMs. By converting block-based code to a textual representation suitable for LLMs, LitterBox+ allows users to query LLMs about their programs, about quality issues reported by LitterBox, and it allows generating code fixes. Besides offering a programmatic API for these functionalities, LitterBox+ also extends the Scratch user interface to make these functionalities available directly in the environment familiar to learners. The framework is designed to be easily extensible with other prompts, LLM providers, and new features combining the program analysis capabilities of LitterBox with the generative features of LLMs. We provide a screencast demonstrating the tool at https://youtu.be/RZ6E0xgrIgQ.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.13231v2">Accelerating LLM Inference via Dynamic KV Cache Placement in Heterogeneous Memory System</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 IEEE Computer Architecture Letter
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) inference is increasingly constrained by memory bandwidth, with frequent access to the key-value (KV) cache dominating data movement. While attention sparsity reduces some memory traffic, the relevance of past tokens varies over time, requiring the full KV cache to remain accessible and sustaining pressure on both bandwidth and capacity. With advances in interconnects such as NVLink and LPDDR5X, modern AI hardware now integrates high-bandwidth memory (HBM) with high-speed off-package DRAM, making heterogeneous memory systems a practical solution. This work investigates dynamic KV cache placement across such systems to maximize aggregated bandwidth utilization under capacity constraints. Rather than proposing a specific scheduling policy, we formulate the placement problem mathematically and derive a theoretical upper bound, revealing substantial headroom for runtime optimization. To our knowledge, this is the first formal treatment of dynamic KV cache scheduling in heterogeneous memory systems for LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14403v4">Unearthing Gems from Stones: Policy Optimization with Negative Sample Augmentation for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      Recent advances in reasoning language models have witnessed a paradigm shift from short to long CoT pattern. Given the substantial computational cost of rollouts in long CoT models, maximizing the utility of fixed training datasets becomes crucial. Our analysis reveals that negative responses contain valuable components such as self-reflection and error-correction steps, yet primary existing methods either completely discard negative samples (RFT) or apply equal penalization across all tokens (RL), failing to leverage these potential learning signals. In light of this, we propose Behavior Constrained Policy Gradient with Negative Sample Augmentation (BCPG-NSA), a fine-grained offline RL framework that encompasses three stages: 1) sample segmentation, 2) consensus-based step correctness assessment combining LLM and PRM judgers, and 3) policy optimization with NSA designed to effectively mine positive steps within negative samples. Experimental results show that BCPG-NSA outperforms baselines on several challenging math/coding reasoning benchmarks using the same training dataset, achieving improved sample efficiency and demonstrating robustness and scalability when extended to multiple iterations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11967v1">MillStone: How Open-Minded Are LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 19 pages, 7 tables, 7 figures
    </div>
    <details class="paper-abstract">
      Large language models equipped with Web search, information retrieval tools, and other agentic capabilities are beginning to supplant traditional search engines. As users start to rely on LLMs for information on many topics, including controversial and debatable issues, it is important to understand how the stances and opinions expressed in LLM outputs are influenced by the documents they use as their information sources. In this paper, we present MillStone, the first benchmark that aims to systematically measure the effect of external arguments on the stances that LLMs take on controversial issues (not all of them political). We apply MillStone to nine leading LLMs and measure how ``open-minded'' they are to arguments supporting opposite sides of these issues, whether different LLMs agree with each other, which arguments LLMs find most persuasive, and whether these arguments are the same for different LLMs. In general, we find that LLMs are open-minded on most issues. An authoritative source of information can easily sway an LLM's stance, highlighting the importance of source selection and the risk that LLM-based information retrieval and search systems can be manipulated.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11921v1">Designing LLMs for cultural sensitivity: Evidence from English-Japanese translation</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used in everyday communication, including multilingual interactions across different cultural contexts. While LLMs can now generate near-perfect literal translations, it remains unclear whether LLMs support culturally appropriate communication. In this paper, we analyze the cultural sensitivity of different LLM designs when applied to English-Japanese translations of workplace e-mails. Here, we vary the prompting strategies: (1) naive "just translate" prompts, (2) audience-targeted prompts specifying the recipient's cultural background, and (3) instructional prompts with explicit guidance on Japanese communication norms. Using a mixed-methods study, we then analyze culture-specific language patterns to evaluate how well translations adapt to cultural norms. Further, we examine the appropriateness of the tone of the translations as perceived by native speakers. We find that culturally-tailored prompting can improve cultural fit, based on which we offer recommendations for designing culturally inclusive LLMs in multilingual settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05755v2">Exploit Tool Invocation Prompt for Tool Behavior Hijacking in LLM-Based Agentic System</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      LLM-based agentic systems leverage large language models to handle user queries, make decisions, and execute external tools for complex tasks across domains like chatbots, customer service, and software engineering. A critical component of these systems is the Tool Invocation Prompt (TIP), which defines tool interaction protocols and guides LLMs to ensure the security and correctness of tool usage. Despite its importance, TIP security has been largely overlooked. This work investigates TIP-related security risks, revealing that major LLM-based systems like Cursor, Claude Code, and others are vulnerable to attacks such as remote code execution (RCE) and denial of service (DoS). Through a systematic TIP exploitation workflow (TEW), we demonstrate external tool behavior hijacking via manipulated tool invocations. We also propose defense mechanisms to enhance TIP security in LLM-based agentic systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04427v2">Plugging Schema Graph into Multi-Table QA: A Human-Guided Framework for Reducing LLM Reliance</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 Accepted to EMNLP 2025 findings
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown promise in table Question Answering (Table QA). However, extending these capabilities to multi-table QA remains challenging due to unreliable schema linking across complex tables. Existing methods based on semantic similarity work well only on simplified hand-crafted datasets and struggle to handle complex, real-world scenarios with numerous and diverse columns. To address this, we propose a graph-based framework that leverages human-curated relational knowledge to explicitly encode schema links and join paths. Given a natural language query, our method searches on graph to construct interpretable reasoning chains, aided by pruning and sub-path merging strategies to enhance efficiency and coherence. Experiments on both standard benchmarks and a realistic, large-scale dataset demonstrate the effectiveness of our approach. To our knowledge, this is the first multi-table QA system applied to truly complex industrial tabular data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11864v1">NeuroStrike: Neuron-Level Attacks on Aligned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      Safety alignment is critical for the ethical deployment of large language models (LLMs), guiding them to avoid generating harmful or unethical content. Current alignment techniques, such as supervised fine-tuning and reinforcement learning from human feedback, remain fragile and can be bypassed by carefully crafted adversarial prompts. Unfortunately, such attacks rely on trial and error, lack generalizability across models, and are constrained by scalability and reliability. This paper presents NeuroStrike, a novel and generalizable attack framework that exploits a fundamental vulnerability introduced by alignment techniques: the reliance on sparse, specialized safety neurons responsible for detecting and suppressing harmful inputs. We apply NeuroStrike to both white-box and black-box settings: In the white-box setting, NeuroStrike identifies safety neurons through feedforward activation analysis and prunes them during inference to disable safety mechanisms. In the black-box setting, we propose the first LLM profiling attack, which leverages safety neuron transferability by training adversarial prompt generators on open-weight surrogate models and then deploying them against black-box and proprietary targets. We evaluate NeuroStrike on over 20 open-weight LLMs from major LLM developers. By removing less than 0.6% of neurons in targeted layers, NeuroStrike achieves an average attack success rate (ASR) of 76.9% using only vanilla malicious prompts. Moreover, Neurostrike generalizes to four multimodal LLMs with 100% ASR on unsafe image inputs. Safety neurons transfer effectively across architectures, raising ASR to 78.5% on 11 fine-tuned models and 77.7% on five distilled models. The black-box LLM profiling attack achieves an average ASR of 63.7% across five black-box models, including the Google Gemini family.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11803v1">From Fuzzy Speech to Medical Insight: Benchmarking LLMs on Noisy Patient Narratives</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 6 pages, 1 figure
    </div>
    <details class="paper-abstract">
      The widespread adoption of large language models (LLMs) in healthcare raises critical questions about their ability to interpret patient-generated narratives, which are often informal, ambiguous, and noisy. Existing benchmarks typically rely on clean, structured clinical text, offering limited insight into model performance under realistic conditions. In this work, we present a novel synthetic dataset designed to simulate patient self-descriptions characterized by varying levels of linguistic noise, fuzzy language, and layperson terminology. Our dataset comprises clinically consistent scenarios annotated with ground-truth diagnoses, spanning a spectrum of communication clarity to reflect diverse real-world reporting styles. Using this benchmark, we fine-tune and evaluate several state-of-the-art models (LLMs), including BERT-based and encoder-decoder T5 models. To support reproducibility and future research, we release the Noisy Diagnostic Benchmark (NDB), a structured dataset of noisy, synthetic patient descriptions designed to stress-test and compare the diagnostic capabilities of large language models (LLMs) under realistic linguistic conditions. We made the benchmark available for the community: https://github.com/lielsheri/PatientSignal
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.18337v5">Can LLMs assist with Ambiguity? A Quantitative Evaluation of various Large Language Models on Word Sense Disambiguation</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 12 pages,6 tables, 1 figure, Proceedings of the 1st International Conference on NLP & AI for Cyber Security
    </div>
    <details class="paper-abstract">
      Ambiguous words are often found in modern digital communications. Lexical ambiguity challenges traditional Word Sense Disambiguation (WSD) methods, due to limited data. Consequently, the efficiency of translation, information retrieval, and question-answering systems is hindered by these limitations. This study investigates the use of Large Language Models (LLMs) to improve WSD using a novel approach combining a systematic prompt augmentation mechanism with a knowledge base (KB) consisting of different sense interpretations. The proposed method incorporates a human-in-loop approach for prompt augmentation where prompt is supported by Part-of-Speech (POS) tagging, synonyms of ambiguous words, aspect-based sense filtering and few-shot prompting to guide the LLM. By utilizing a few-shot Chain of Thought (COT) prompting-based approach, this work demonstrates a substantial improvement in performance. The evaluation was conducted using FEWS test data and sense tags. This research advances accurate word interpretation in social media and digital communication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00831v5">SmallPlan: Leverage Small Language Models for Sequential Path Planning with Simulation-Powered, LLM-Guided Distillation</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 Paper is under review
    </div>
    <details class="paper-abstract">
      Efficient path planning in robotics, particularly within large-scale, complex environments, remains a significant hurdle. While Large Language Models (LLMs) offer strong reasoning capabilities, their high computational cost and limited adaptability hinder real-time deployment on edge devices. We present SmallPlan - a novel framework leveraging LLMs as teacher models to train lightweight Small Language Models (SLMs) for high-level path planning tasks. In SmallPlan, the SLMs provide optimal action sequences to navigate across scene graphs that compactly represent full-scaled 3D scenes. The SLMs are trained in a simulation-powered, interleaved manner with LLM-guided supervised fine-tuning (SFT) and reinforcement learning (RL). This strategy not only enables SLMs to successfully complete navigation tasks but also makes them aware of important factors like distance travel, providing more efficient path planning. Through experiments, we demonstrate that the fine-tuned SLMs perform competitively with larger models like GPT-4o on sequential path planning, without suffering from hallucination and overfitting. SmallPlan is resource-efficient, making it well-suited for edge-device deployment and advancing practical autonomous robotics. Our source code is available here: https://github.com/quangpham2006/SmallPlan
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20258v2">LLM as a Broken Telephone: Iterative Generation Distorts Information</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 Accepted to ACL 2025, Main Conference
    </div>
    <details class="paper-abstract">
      As large language models are increasingly responsible for online content, concerns arise about the impact of repeatedly processing their own outputs. Inspired by the "broken telephone" effect in chained human communication, this study investigates whether LLMs similarly distort information through iterative generation. Through translation-based experiments, we find that distortion accumulates over time, influenced by language choice and chain complexity. While degradation is inevitable, it can be mitigated through strategic prompting techniques. These findings contribute to discussions on the long-term effects of AI-mediated information propagation, raising important questions about the reliability of LLM-generated content in iterative workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.08575v2">SQLGovernor: An LLM-powered SQL Toolkit for Real World Application</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      SQL queries in real world analytical environments, whether written by humans or generated automatically often suffer from syntax errors, inefficiency, or semantic misalignment, especially in complex OLAP scenarios. To address these challenges, we propose SQLGovernor, an LLM powered SQL toolkit that unifies multiple functionalities, including syntax correction, query rewriting, query modification, and consistency verification within a structured framework enhanced by knowledge management. SQLGovernor introduces a fragment wise processing strategy to enable fine grained rewriting and localized error correction, significantly reducing the cognitive load on the LLM. It further incorporates a hybrid self learning mechanism guided by expert feedback, allowing the system to continuously improve through DBMS output analysis and rule validation. Experiments on benchmarks such as BIRD and BIRD CRITIC, as well as industrial datasets, show that SQLGovernor consistently boosts the performance of base models by up to 10%, while minimizing reliance on manual expertise. Deployed in production environments, SQLGovernor demonstrates strong practical utility and effective performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20742v2">'Hello, World!': Making GNNs Talk with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 Published as a conference paper at EMNLP 2025 Findings. Code and datasets are in https://github.com/kswoo97/GLN-Code
    </div>
    <details class="paper-abstract">
      While graph neural networks (GNNs) have shown remarkable performance across diverse graph-related tasks, their high-dimensional hidden representations render them black boxes. In this work, we propose Graph Lingual Network (GLN), a GNN built on large language models (LLMs), with hidden representations in the form of human-readable text. Through careful prompt design, GLN incorporates not only the message passing module of GNNs but also advanced GNN techniques, including graph attention and initial residual connection. The comprehensibility of GLN's hidden representations enables an intuitive analysis of how node representations change (1) across layers and (2) under advanced GNN techniques, shedding light on the inner workings of GNNs. Furthermore, we demonstrate that GLN achieves strong zero-shot performance on node classification and link prediction, outperforming existing LLM-based baseline methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.12172v3">Navigating the Labyrinth: Evaluating LLMs' Ability to Reason About Search Problems</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently achieved impressive performance in math and reasoning benchmarks. However, they often struggle with logic problems and puzzles that are relatively easy for humans. To further investigate this, we introduce a new benchmark, SearchBench, which contains 11 unique search problems inspired by intuitive puzzles. Each SearchBench problem type is equipped with automated pipelines to generate an arbitrary number of instances and analyze the feasibility, correctness, and optimality of LLM-generated solutions. We show that using step-by-step, language-only reasoning, even the most advanced LLMs fail to solve SearchBench; for example, OpenAI's frontier models GPT-4 and o1-preview solve only 1.4% and 18.6% of problems, respectively. The reason is that SearchBench problems require considering multiple pathways and performing backtracking, posing a significant challenge to auto-regressive models. Interestingly, performance is significantly boosted when we prompt models to generate a complete A* search algorithm - a comparatively more cognitively difficult task. This approach effectively offloads the iterative search and backtracking process from the models, which they struggle with in text. This in-context learning baseline is further enhanced via a Multi-Stage-Multi-Try (MSMT) inference method, increasing GPT-4's rate of correct solutions to over 57%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11617v1">AssemMate: Graph-Based LLM for Robotic Assembly Assistance</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based robotic assembly assistance has gained significant research attention. It requires the injection of domain-specific knowledge to guide the assembly process through natural language interaction with humans. Despite some progress, existing methods represent knowledge in the form of natural language text. Due to the long context and redundant content, they struggle to meet the robots' requirements for real-time and precise reasoning. In order to bridge this gap, we present AssemMate, which utilizes the graph\textemdash a concise and accurate form of knowledge representation\textemdash as input. This graph-based LLM enables knowledge graph question answering (KGQA), supporting human-robot interaction and assembly task planning for specific products. Beyond interactive QA, AssemMate also supports sensing stacked scenes and executing grasping to assist with assembly. Specifically, a self-supervised Graph Convolutional Network (GCN) encodes knowledge graph entities and relations into a latent space and aligns them with LLM's representation, enabling the LLM to understand graph information. In addition, a vision-enhanced strategy is employed to address stacked scenes in grasping. Through training and evaluation, AssemMate outperforms existing methods, achieving 6.4\% higher accuracy, 3 times faster inference, and 28 times shorter context length, while demonstrating strong generalization ability on random graphs. And our approach further demonstrates superiority through robotic grasping experiments in both simulated and real-world settings. More details can be found on the project page: https://github.com/cristina304/AssemMate.git
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03054v2">Binary Quantization For LLMs Through Dynamic Grouping</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 An error was identified in the quantization bit width; it is not binary
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable performance across a wide range of Natural Language Processing (NLP) tasks, but require substantial memory and computational resources. Binary quantization, which compresses model weights from 16-bit Brain Float to 1-bit representations in {-1, 1}, offers significant reductions in storage and inference costs. However, such aggressive quantization often leads to notable performance degradation compared to more conservative 4-bit quantization methods. In this research, we propose a novel optimization objective tailored for binary quantization, along with three algorithms designed to realize it effectively. Our method enhances blocked quantization by dynamically identifying optimal unstructured sub-matrices through adaptive grouping strategies. Experimental results demonstrate that our approach achieves an average bit length of just 1.007 bits, while maintaining high model quality. Specifically, our quantized LLaMA 3.2 3B model attains a perplexity of 8.23, remarkably close to the original 7.81, and surpasses previous SOTA BiLLM with a perplexity of only 123.90. Furthermore, our method is competitive with SOTA 4-bit approaches such as GPTQ in both performance and efficiency. The compression process is highly efficient, requiring only 14 seconds to quantize the full LLaMA 3.2 3B weights on a single CPU core, with the entire process completing in under 100 minutes and exhibiting embarrassingly parallel properties. Code - https://github.com/johnnyzheng0636/WGM_bi_quan
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11569v1">D$^2$HScore: Reasoning-Aware Hallucination Detection via Semantic Breadth and Depth Analysis in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 under review
    </div>
    <details class="paper-abstract">
      Although large Language Models (LLMs) have achieved remarkable success, their practical application is often hindered by the generation of non-factual content, which is called "hallucination". Ensuring the reliability of LLMs' outputs is a critical challenge, particularly in high-stakes domains such as finance, security, and healthcare. In this work, we revisit hallucination detection from the perspective of model architecture and generation dynamics. Leveraging the multi-layer structure and autoregressive decoding process of LLMs, we decompose hallucination signals into two complementary dimensions: the semantic breadth of token representations within each layer, and the semantic depth of core concepts as they evolve across layers. Based on this insight, we propose \textbf{D$^2$HScore (Dispersion and Drift-based Hallucination Score)}, a training-free and label-free framework that jointly measures: (1) \textbf{Intra-Layer Dispersion}, which quantifies the semantic diversity of token representations within each layer; and (2) \textbf{Inter-Layer Drift}, which tracks the progressive transformation of key token representations across layers. To ensure drift reflects the evolution of meaningful semantics rather than noisy or redundant tokens, we guide token selection using attention signals. By capturing both the horizontal and vertical dynamics of representation during inference, D$^2$HScore provides an interpretable and lightweight proxy for hallucination detection. Extensive experiments across five open-source LLMs and five widely used benchmarks demonstrate that D$^2$HScore consistently outperforms existing training-free baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10248v2">Prompt Injection Attacks on LLM Generated Reviews of Scientific Publications</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      The ongoing intense discussion on rising LLM usage in the scientific peer-review process has recently been mingled by reports of authors using hidden prompt injections to manipulate review scores. Since the existence of such "attacks" - although seen by some commentators as "self-defense" - would have a great impact on the further debate, this paper investigates the practicability and technical success of the described manipulations. Our systematic evaluation uses 1k reviews of 2024 ICLR papers generated by a wide range of LLMs shows two distinct results: I) very simple prompt injections are indeed highly effective, reaching up to 100% acceptance scores. II) LLM reviews are generally biased toward acceptance (>95% in many models). Both results have great impact on the ongoing discussions on LLM usage in peer-review.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16347v2">Confusion is the Final Barrier: Rethinking Jailbreak Evaluation and Investigating the Real Misuse Threat of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      With the development of Large Language Models (LLMs), numerous efforts have revealed their vulnerabilities to jailbreak attacks. Although these studies have driven the progress in LLMs' safety alignment, it remains unclear whether LLMs have internalized authentic knowledge to deal with real-world crimes, or are merely forced to simulate toxic language patterns. This ambiguity raises concerns that jailbreak success is often attributable to a hallucination loop between jailbroken LLM and judger LLM. By decoupling the use of jailbreak techniques, we construct knowledge-intensive Q\&A to investigate the misuse threats of LLMs in terms of dangerous knowledge possession, harmful task planning utility, and harmfulness judgment robustness. Experiments reveal a mismatch between jailbreak success rates and harmful knowledge possession in LLMs, and existing LLM-as-a-judge frameworks tend to anchor harmfulness judgments on toxic language patterns. Our study reveals a gap between existing LLM safety assessments and real-world threat potential.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00063v2">MolErr2Fix: Benchmarking LLM Trustworthiness in Chemistry via Modular Error Detection, Localization, Explanation, and Revision</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown growing potential in molecular sciences, but they often produce chemically inaccurate descriptions and struggle to recognize or justify potential errors. This raises important concerns about their robustness and reliability in scientific applications. To support more rigorous evaluation of LLMs in chemical reasoning, we present the MolErr2Fix benchmark, designed to assess LLMs on error detection and correction in molecular descriptions. Unlike existing benchmarks focused on molecule-to-text generation or property prediction, MolErr2Fix emphasizes fine-grained chemical understanding. It tasks LLMs with identifying, localizing, explaining, and revising potential structural and semantic errors in molecular descriptions. Specifically, MolErr2Fix consists of 1,193 fine-grained annotated error instances. Each instance contains quadruple annotations, i.e,. (error type, span location, the explanation, and the correction). These tasks are intended to reflect the types of reasoning and verification required in real-world chemical communication. Evaluations of current state-of-the-art LLMs reveal notable performance gaps, underscoring the need for more robust chemical reasoning capabilities. MolErr2Fix provides a focused benchmark for evaluating such capabilities and aims to support progress toward more reliable and chemically informed language models. All annotations and an accompanying evaluation API will be publicly released to facilitate future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14827v3">Enhancing Prompt Injection Attacks to LLMs via Poisoning Alignment</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      Prompt injection attack, where an attacker injects a prompt into the original one, aiming to make an Large Language Model (LLM) follow the injected prompt to perform an attacker-chosen task, represent a critical security threat. Existing attacks primarily focus on crafting these injections at inference time, treating the LLM itself as a static target. Our experiments show that these attacks achieve some success, but there is still significant room for improvement. In this work, we introduces a more foundational attack vector: poisoning the LLM's alignment process to amplify the success of future prompt injection attacks. Specifically, we propose PoisonedAlign, a method that strategically creates poisoned alignment samples to poison an LLM's alignment dataset. Our experiments across five LLMs and two alignment datasets show that when even a small fraction of the alignment data is poisoned, the resulting model becomes substantially more vulnerable to a wide range of prompt injection attacks. Crucially, this vulnerability is instilled while the LLM's performance on standard capability benchmarks remains largely unchanged, making the manipulation difficult to detect through automated, general-purpose performance evaluations. The code for implementing the attack is available at https://github.com/Sadcardation/PoisonedAlign.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14970v3">Self-Evolving Curriculum for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has proven effective for fine-tuning large language models (LLMs), significantly enhancing their reasoning abilities in domains such as mathematics and code generation. A crucial factor influencing RL fine-tuning success is the training curriculum: the order in which training problems are presented. While random curricula serve as common baselines, they remain suboptimal; manually designed curricula often rely heavily on heuristics, and online filtering methods can be computationally prohibitive. To address these limitations, we propose Self-Evolving Curriculum (SEC), an automatic curriculum learning method that learns a curriculum policy concurrently with the RL fine-tuning process. Our approach formulates curriculum selection as a non-stationary Multi-Armed Bandit problem, treating each problem category (e.g., difficulty level or problem type) as an individual arm. We leverage the absolute advantage from policy gradient methods as a proxy measure for immediate learning gain. At each training step, the curriculum policy selects categories to maximize this reward signal and is updated using the TD(0) method. Across three distinct reasoning domains: planning, inductive reasoning, and mathematics, our experiments demonstrate that SEC significantly improves models' reasoning capabilities, enabling better generalization to harder, out-of-distribution test problems. Additionally, our approach achieves better skill balance when fine-tuning simultaneously on multiple reasoning domains. These findings highlight SEC as a promising strategy for RL fine-tuning of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11524v1">Decoding in Latent Spaces for Efficient Inference in LLM-based Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 Accepted for publication in EMNLP'25
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) for recommendation in a generative manner has delivered promising results, but encounters significant inference overhead due to autoregressive decoding in the language space. This work explores bypassing language-space decoding by directly matching candidate items with the LLM's internal thought representations in the latent space, eliminating the time-consuming autoregressive process to reduce computational costs. Towards this, we introduce Light Latent-space Decoding (L2D), an effective and efficient latent-space decoding method. L2D represents user-preferred items by using the hidden states of test sequences reflecting the LLM's internal thought, and obtains candidate item representations from the hidden states of training sequences labeled with the corresponding candidate items. It then matches the two types of representations to decode items, achieving latent-space decoding. In this way, it enables efficient decoding without altering the LLM's generative tuning paradigm, thereby preserving performance. Extensive empirical results demonstrate that L2D is more than 10x faster than language-space decoding while maintaining or enhancing performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11517v1">PeruMedQA: Benchmarking Large Language Models (LLMs) on Peruvian Medical Exams -- Dataset Construction and Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 https://github.com/rodrigo-carrillo/PeruMedQA
    </div>
    <details class="paper-abstract">
      BACKGROUND: Medical large language models (LLMS) have demonstrated remarkable performance in answering medical examinations. However, the extent to which this high performance is transferable to medical questions in Spanish and from a Latin American country remains unexplored. This knowledge is crucial as LLM-based medical applications gain traction in Latin America. AIMS: to build a dataset of questions from medical examinations taken by Peruvian physicians pursuing specialty training; to fine-tune a LLM on this dataset; to evaluate and compare the performance in terms of accuracy between vanilla LLMs and the fine-tuned LLM. METHODS: We curated PeruMedQA, a multiple-choice question-answering (MCQA) datasets containing 8,380 questions spanning 12 medical domains (2018-2025). We selected eight medical LLMs including medgemma-4b-it and medgemma-27b-text-it, and developed zero-shot task-specific prompts to answer the questions appropriately. We employed parameter-efficient fine tuning (PEFT)and low-rant adaptation (LoRA) to fine-tune medgemma-4b-it utilizing all questions except those from 2025 (test set). RESULTS: medgemma-27b-text-it outperformed all other models, achieving a proportion of correct answers exceeding 90% in several instances. LLMs with <10 billion parameters exhibited <60% of correct answers, while some exams yielded results <50%. The fine-tuned version of medgemma-4b-it emerged victorious agains all LLMs with <10 billion parameters and rivaled a LLM with 70 billion parameters across various examinations. CONCLUSIONS: For medical AI application and research that require knowledge bases from Spanish-speaking countries and those exhibiting similar epidemiological profiles to Peru's, interested parties should utilize medgemma-27b-text-it or a fine-tuned version of medgemma-4b-it.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11507v1">MedicalOS: An LLM Agent based Operating System for Digital Healthcare</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      Decades' advances in digital health technologies, such as electronic health records, have largely streamlined routine clinical processes. Yet, most these systems are still hard to learn and use: Clinicians often face the burden of managing multiple tools, repeating manual actions for each patient, navigating complicated UI trees to locate functions, and spending significant time on administration instead of caring for patients. The recent rise of large language model (LLM) based agents demonstrates exceptional capability in coding and computer operation, revealing the potential for humans to interact with operating systems and software not by direct manipulation, but by instructing agents through natural language. This shift highlights the need for an abstraction layer, an agent-computer interface, that translates human language into machine-executable commands. In digital healthcare, however, requires a more domain-specific abstractions that strictly follow trusted clinical guidelines and procedural standards to ensure safety, transparency, and compliance. To address this need, we present \textbf{MedicalOS}, a unified agent-based operational system designed as such a domain-specific abstract layer for healthcare. It translates human instructions into pre-defined digital healthcare commands, such as patient inquiry, history retrieval, exam management, report generation, referrals, treatment planning, that we wrapped as off-the-shelf tools using machine languages (e.g., Python, APIs, MCP, Linux). We empirically validate MedicalOS on 214 patient cases across 22 specialties, demonstrating high diagnostic accuracy and confidence, clinically sound examination requests, and consistent generation of structured reports and medication recommendations. These results highlight MedicalOS as a trustworthy and scalable foundation for advancing workflow automation in clinical practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11496v1">AKCIT-FN at CheckThat! 2025: Switching Fine-Tuned SLMs and LLM Prompting for Multilingual Claim Normalization</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 15 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Claim normalization, the transformation of informal social media posts into concise, self-contained statements, is a crucial step in automated fact-checking pipelines. This paper details our submission to the CLEF-2025 CheckThat! Task~2, which challenges systems to perform claim normalization across twenty languages, divided into thirteen supervised (high-resource) and seven zero-shot (no training data) tracks. Our approach, leveraging fine-tuned Small Language Models (SLMs) for supervised languages and Large Language Model (LLM) prompting for zero-shot scenarios, achieved podium positions (top three) in fifteen of the twenty languages. Notably, this included second-place rankings in eight languages, five of which were among the seven designated zero-shot languages, underscoring the effectiveness of our LLM-based zero-shot strategy. For Portuguese, our initial development language, our system achieved an average METEOR score of 0.5290, ranking third. All implementation artifacts, including inference, training, evaluation scripts, and prompt configurations, are publicly available at https://github.com/ju-resplande/checkthat2025_normalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06261v2">FineServe: Precision-Aware KV Slab and Two-Level Scheduling for Heterogeneous Precision LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      Recent advances in Post-Training Quantization (PTQ) techniques have significantly increased demand for serving quantized large language models (LLMs), enabling higher throughput and substantially reduced memory usage with minimal accuracy loss. Quantized models address memory constraints in LLMs and enhance GPU resource utilization through efficient GPU sharing. However, quantized models have smaller KV block sizes than non-quantized models, causing limited memory efficiency due to memory fragmentation. Also, distinct resource usage patterns between quantized and non-quantized models require efficient scheduling to maximize throughput. To address these challenges, we propose FineServe, an inference serving framework for mixed-precision LLMs. FineServe's key contributions include: (1) KV Slab, a precision-aware adaptive memory management technique dynamically allocating KV cache based on model quantization characteristics, significantly reducing GPU memory fragmentation, and (2) a two-level scheduling framework comprising a global scheduler that places models to GPUs based on request rates, latency SLOs, and memory constraints and efficiency, and a local scheduler that adaptively adjusts batch sizes according to real-time request fluctuations. Experimental results demonstrate that FineServe achieves up to 2.2x higher SLO attainment and 1.8x higher token generation throughput compared to the state-of-the-art GPU sharing systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17026v2">Understanding the Uncertainty of LLM Explanations: A Perspective Based on Reasoning Topology</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 28 pages, 9 figures; accepted at COLM'25
    </div>
    <details class="paper-abstract">
      Understanding the uncertainty in large language model (LLM) explanations is important for evaluating their faithfulness and reasoning consistency, and thus provides insights into the reliability of LLM's output regarding a question. In this work, we propose a novel framework that quantifies uncertainty in LLM explanations through a reasoning topology perspective. By designing a structural elicitation strategy, we guide the LLMs to frame the explanations of an answer into a graph topology. This process decomposes the explanations into the knowledge related sub-questions and topology-based reasoning structures, which allows us to quantify uncertainty not only at the semantic level but also from the reasoning path. It further brings convenience to assess knowledge redundancy and provide interpretable insights into the reasoning process. Our method offers a systematic way to interpret the LLM reasoning, analyze limitations, and provide guidance for enhancing robustness and faithfulness. This work pioneers the use of graph-structured uncertainty measurement in LLM explanations and demonstrates the potential of topology-based quantification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12476v1">Audited Reasoning Refinement: Fine-Tuning Language Models via LLM-Guided Step-Wise Evaluation and Correction</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      Training a task-specific small reasoning model is challenging when direct human supervision or high-quality labels are scarce. However, LLMs with reasoning capabilities produce abundant intermediate reasoning traces that can be systematically refined to create effective supervision signals. We propose Reason-Refine-then-Align (R2tA), which turns refined model rationales into supervision for training task-specific reasoning models. Our method generates initial reasoning and responses from an open-source base model on task-specific inputs, then refines these traces, fixing hallucinations and inconsistencies, to form a high-fidelity dataset. We perform a two-stage alignment, supervised fine-tuning (SFT), followed by direct preference optimization (DPO) to calibrate the model's intermediate reasoning with human-validated conceptual preferences and then condition the final output on that aligned reasoning. As a case study, we apply R2tA to evaluate extended entity relationship diagrams (EERDs) in database system design, a structurally complex task where prompt-only methods miss or hallucinate errors. We curated a dataset of 600 EERD variants (train/test split of 450/150, respectively) with induced mistakes spanning 11 categories. Empirical evaluation suggests R2tA provides a practical, cost-effective path to scalable LLM adaptation in data-scarce domains, enabling reproducible AI tools for education and beyond.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21653v2">Think Before You Diffuse: LLMs-Guided Physics-Aware Video Generation</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 19 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Recent video diffusion models have demonstrated their great capability in generating visually-pleasing results, while synthesizing the correct physical effects in generated videos remains challenging. The complexity of real-world motions, interactions, and dynamics introduce great difficulties when learning physics from data. In this work, we propose DiffPhy, a generic framework that enables physically-correct and photo-realistic video generation by fine-tuning a pre-trained video diffusion model. Our method leverages large language models (LLMs) to explicitly reason a comprehensive physical context from the text prompt and use it to guide the generation. To incorporate physical context into the diffusion model, we leverage a Multimodal large language model (MLLM) as a supervisory signal and introduce a set of novel training objectives that jointly enforce physical correctness and semantic consistency with the input text. We also establish a high-quality physical video dataset containing diverse phyiscal actions and events to facilitate effective finetuning. Extensive experiments on public benchmarks demonstrate that DiffPhy is able to produce state-of-the-art results across diverse physics-related scenarios. Our project page is available at https://bwgzk-keke.github.io/DiffPhy/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12462v1">Redefining Website Fingerprinting Attacks With Multiagent LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      Website Fingerprinting (WFP) uses deep learning models to classify encrypted network traffic to infer visited websites. While historically effective, prior methods fail to generalize to modern web environments. Single-page applications (SPAs) eliminate the paradigm of websites as sets of discrete pages, undermining page-based classification, and traffic from scripted browsers lacks the behavioral richness seen in real user sessions. Our study reveals that users exhibit highly diverse behaviors even on the same website, producing traffic patterns that vary significantly across individuals. This behavioral entropy makes WFP a harder problem than previously assumed and highlights the need for larger, more diverse, and representative datasets to achieve robust performance. To address this, we propose a new paradigm: we drop session-boundaries in favor of contiguous traffic segments and develop a scalable data generation pipeline using large language models (LLM) agents. These multi-agent systems coordinate decision-making and browser interaction to simulate realistic, persona-driven browsing behavior at 3--5x lower cost than human collection. We evaluate nine state-of-the-art WFP models on traffic from 20 modern websites browsed by 30 real users, and compare training performance across human, scripted, and LLM-generated datasets. All models achieve under 10\% accuracy when trained on scripted traffic and tested on human data. In contrast, LLM-generated traffic boosts accuracy into the 80\% range, demonstrating strong generalization to real-world traces. Our findings indicate that for modern WFP, model performance is increasingly bottlenecked by data quality, and that scalable, semantically grounded synthetic traffic is essential for capturing the complexity of real user behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.09689v4">Probing LLM Hallucination from Within: Perturbation-Driven Approach via Internal Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 22 pages, 15 figures
    </div>
    <details class="paper-abstract">
      LLM hallucination, where unfaithful text is generated, presents a critical challenge for LLMs' practical applications. Current detection methods often resort to external knowledge, LLM fine-tuning, or supervised training with large hallucination-labeled datasets. Moreover, these approaches do not distinguish between different types of hallucinations, which is crucial for enhancing detection performance. To address such limitations, we introduce hallucination probing, a new task that classifies LLM-generated text into three categories: aligned, misaligned, and fabricated. Driven by our novel discovery that perturbing key entities in prompts affects LLM's generation of these three types of text differently, we propose SHINE, a novel hallucination probing method that does not require external knowledge, supervised training, or LLM fine-tuning. SHINE is effective in hallucination probing across three modern LLMs, and achieves state-of-the-art performance in hallucination detection, outperforming seven competing methods across four datasets and four LLMs, underscoring the importance of probing for accurate detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10900v3">Tuning-Free LLM Can Build A Strong Recommender Under Sparse Connectivity And Knowledge Gap Via Extracting Intent</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      Recent advances in recommendation with large language models (LLMs) often rely on either commonsense augmentation at the item-category level or implicit intent modeling on existing knowledge graphs. However, such approaches struggle to capture grounded user intents and to handle sparsity and cold-start scenarios. In this work, we present LLM-based Intent Knowledge Graph Recommender (IKGR), a novel framework that constructs an intent-centric knowledge graph where both users and items are explicitly linked to intent nodes extracted by a tuning-free, RAG-guided LLM pipeline. By grounding intents in external knowledge sources and user profiles, IKGR canonically represents what a user seeks and what an item satisfies as first-class entities. To alleviate sparsity, we further introduce a mutual-intent connectivity densification strategy, which shortens semantic paths between users and long-tail items without requiring cross-graph fusion. Finally, a lightweight GNN layer is employed on top of the intent-enhanced graph to produce recommendation signals with low latency. Extensive experiments on public and enterprise datasets demonstrate that IKGR consistently outperforms strong baselines, particularly on cold-start and long-tail slices, while remaining efficient through a fully offline LLM pipeline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.04629v5">Argumentative Experience: Reducing Confirmation Bias on Controversial Issues through LLM-Generated Multi-Persona Debates</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 Complete reanalysis using a revised methodology; results, figures, and discussion updated to reflect the new approach
    </div>
    <details class="paper-abstract">
      Multi-persona debate systems powered by large language models (LLMs) show promise in reducing confirmation bias, which can fuel echo chambers and social polarization. However, empirical evidence remains limited on whether they meaningfully shift user attention toward belief-challenging content, promote belief change, or outperform traditional debiasing strategies. To investigate this, we compare an LLM-based multi-persona debate system with a two-stance retrieval-based system, exposing participants to multiple viewpoints on controversial topics. By collecting eye-tracking data, belief change measures, and qualitative feedback, our results show that while the debate system does not significantly increase attention to opposing views, or make participants shift away from prior beliefs, it does provide a buffering effect against bias caused by individual cognitive tendency. These findings shed light on both the promise and limits of multi-persona debate systems in information seeking, and we offer design insights to guide future work toward more balanced and reflective information engagement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04655v2">Polysemantic Dropout: Conformal OOD Detection for Specialized LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 Accepted to EMNLP 2025 main conference
    </div>
    <details class="paper-abstract">
      We propose a novel inference-time out-of-domain (OOD) detection algorithm for specialized large language models (LLMs). Despite achieving state-of-the-art performance on in-domain tasks through fine-tuning, specialized LLMs remain vulnerable to incorrect or unreliable outputs when presented with OOD inputs, posing risks in critical applications. Our method leverages the Inductive Conformal Anomaly Detection (ICAD) framework, using a new non-conformity measure based on the model's dropout tolerance. Motivated by recent findings on polysemanticity and redundancy in LLMs, we hypothesize that in-domain inputs exhibit higher dropout tolerance than OOD inputs. We aggregate dropout tolerance across multiple layers via a valid ensemble approach, improving detection while maintaining theoretical false alarm bounds from ICAD. Experiments with medical-specialized LLMs show that our approach detects OOD inputs better than baseline methods, with AUROC improvements of $2\%$ to $37\%$ when treating OOD datapoints as positives and in-domain test datapoints as negatives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12385v1">SENTRA: Selected-Next-Token Transformer for LLM Text Detection</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 EMNLP Findings 2025
    </div>
    <details class="paper-abstract">
      LLMs are becoming increasingly capable and widespread. Consequently, the potential and reality of their misuse is also growing. In this work, we address the problem of detecting LLM-generated text that is not explicitly declared as such. We present a novel, general-purpose, and supervised LLM text detector, SElected-Next-Token tRAnsformer (SENTRA). SENTRA is a Transformer-based encoder leveraging selected-next-token-probability sequences and utilizing contrastive pre-training on large amounts of unlabeled data. Our experiments on three popular public datasets across 24 domains of text demonstrate SENTRA is a general-purpose classifier that significantly outperforms popular baselines in the out-of-domain setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12382v1">LLM-as-a-Judge: Rapid Evaluation of Legal Document Recommendation for Retrieval-Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 Accepted in EARL 25: The 2nd Workshop on Evaluating and Applying Recommender Systems with Large Language Models at RecSys 2025
    </div>
    <details class="paper-abstract">
      The evaluation bottleneck in recommendation systems has become particularly acute with the rise of Generative AI, where traditional metrics fall short of capturing nuanced quality dimensions that matter in specialized domains like legal research. Can we trust Large Language Models to serve as reliable judges of their own kind? This paper investigates LLM-as-a-Judge as a principled approach to evaluating Retrieval-Augmented Generation systems in legal contexts, where the stakes of recommendation quality are exceptionally high. We tackle two fundamental questions that determine practical viability: which inter-rater reliability metrics best capture the alignment between LLM and human assessments, and how do we conduct statistically sound comparisons between competing systems? Through systematic experimentation, we discover that traditional agreement metrics like Krippendorff's alpha can be misleading in the skewed distributions typical of AI system evaluations. Instead, Gwet's AC2 and rank correlation coefficients emerge as more robust indicators for judge selection, while the Wilcoxon Signed-Rank Test with Benjamini-Hochberg corrections provides the statistical rigor needed for reliable system comparisons. Our findings suggest a path toward scalable, cost-effective evaluation that maintains the precision demanded by legal applications, transforming what was once a human-intensive bottleneck into an automated, yet statistically principled, evaluation framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.15444v5">Cutting Through the Noise: Boosting LLM Performance on Math Word Problems</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 Published at ICLR 2025 Workshop on Reasoning and Planning for LLMs
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel at various tasks, including solving math word problems (MWPs), but struggle with real-world problems containing irrelevant information. To address this, we propose a prompting framework that generates adversarial variants of MWPs by adding irrelevant variables. We introduce a dataset, PROBLEMATHIC, containing both adversarial and non-adversarial MWPs. Our experiments reveal that LLMs are susceptible to distraction by numerical noise, resulting in an average relative performance drop of ~26% on adversarial MWPs. To mitigate this, we fine-tune LLMs (Llama-2, Mistral) on the adversarial samples from our dataset. Fine-tuning on adversarial training instances improves performance on adversarial MWPs by ~8%, indicating increased robustness to noise and improved ability to identify relevant data for reasoning. Finally, to assess the generalizability of our prompting framework, we introduce GSM-8K-Adv, an adversarial variant of the GSM-8K benchmark. LLMs continue to struggle when faced with adversarial information, reducing performance by up to 6%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12371v1">MORABLES: A Benchmark for Assessing Abstract Moral Reasoning in LLMs with Fables</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 Accepted to EMNLP 2025 Main Conference
    </div>
    <details class="paper-abstract">
      As LLMs excel on standard reading comprehension benchmarks, attention is shifting toward evaluating their capacity for complex abstract reasoning and inference. Literature-based benchmarks, with their rich narrative and moral depth, provide a compelling framework for evaluating such deeper comprehension skills. Here, we present MORABLES, a human-verified benchmark built from fables and short stories drawn from historical literature. The main task is structured as multiple-choice questions targeting moral inference, with carefully crafted distractors that challenge models to go beyond shallow, extractive question answering. To further stress-test model robustness, we introduce adversarial variants designed to surface LLM vulnerabilities and shortcuts due to issues such as data contamination. Our findings show that, while larger models outperform smaller ones, they remain susceptible to adversarial manipulation and often rely on superficial patterns rather than true moral reasoning. This brittleness results in significant self-contradiction, with the best models refuting their own answers in roughly 20% of cases depending on the framing of the moral choice. Interestingly, reasoning-enhanced models fail to bridge this gap, suggesting that scale - not reasoning ability - is the primary driver of performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04133v4">TRiSM for Agentic AI: A Review of Trust, Risk, and Security Management in LLM-based Agentic Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      Agentic AI systems, built upon large language models (LLMs) and deployed in multi-agent configurations, are redefining intelligence, autonomy, collaboration, and decision-making across enterprise and societal domains. This review presents a structured analysis of Trust, Risk, and Security Management (TRiSM) in the context of LLM-based Agentic Multi-Agent Systems (AMAS). We begin by examining the conceptual foundations of Agentic AI and highlight its architectural distinctions from traditional AI agents. We then adapt and extend the AI TRiSM framework for Agentic AI, structured around key pillars: \textit{ Explainability, ModelOps, Security, Privacy} and \textit{their Lifecycle Governance}, each contextualized to the challenges of AMAS. A risk taxonomy is proposed to capture the unique threats and vulnerabilities of Agentic AI, ranging from coordination failures to prompt-based adversarial manipulation. To support practical assessment in Agentic AI works, we introduce two novel metrics: the Component Synergy Score (CSS), which quantifies the quality of inter-agent collaboration, and the Tool Utilization Efficacy (TUE), which evaluates the efficiency of tool use within agent workflows. We further discuss strategies for improving explainability in Agentic AI, as well as approaches to enhancing security and privacy through encryption, adversarial robustness, and regulatory compliance. The review concludes with a research roadmap for the responsible development and deployment of Agentic AI, highlighting key directions to align emerging systems with TRiSM principles-ensuring safety, transparency, and accountability in their operation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09387v2">MetaLLMix : An XAI Aided LLM-Meta-learning Based Approach for Hyper-parameters Optimization</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      Effective model and hyperparameter selection remains a major challenge in deep learning, often requiring extensive expertise and computation. While AutoML and large language models (LLMs) promise automation, current LLM-based approaches rely on trial and error and expensive APIs, which provide limited interpretability and generalizability. We propose MetaLLMiX, a zero-shot hyperparameter optimization framework combining meta-learning, explainable AI, and efficient LLM reasoning. By leveraging historical experiment outcomes with SHAP explanations, MetaLLMiX recommends optimal hyperparameters and pretrained models without additional trials. We further employ an LLM-as-judge evaluation to control output format, accuracy, and completeness. Experiments on eight medical imaging datasets using nine open-source lightweight LLMs show that MetaLLMiX achieves competitive or superior performance to traditional HPO methods while drastically reducing computational cost. Our local deployment outperforms prior API-based approaches, achieving optimal results on 5 of 8 tasks, response time reductions of 99.6-99.9%, and the fastest training times on 6 datasets (2.4-15.7x faster), maintaining accuracy within 1-5% of best-performing baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24622v2">Random Rule Forest (RRF): Interpretable Ensembles of LLM-Generated Questions for Predicting Startup Success</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 13 pages including appendix, 4 figures
    </div>
    <details class="paper-abstract">
      Predicting rare outcomes such as startup success is central to venture capital, demanding models that are both accurate and interpretable. We introduce Random Rule Forest (RRF), a lightweight ensemble method that uses a large language model (LLM) to generate simple YES/NO questions in natural language. Each question functions as a weak learner, and their responses are combined using a threshold-based voting rule to form a strong, interpretable predictor. Applied to a dataset of 9,892 founders, RRF achieves a 6.9x improvement over a random baseline on held-out data; adding expert-crafted questions lifts this to 8x and highlights the value of human-LLM collaboration. Compared with zero- and few-shot baselines across three LLM architectures, RRF attains an F0.5 of 0.121, versus 0.086 for the best baseline (+0.035 absolute, +41% relative). By combining the creativity of LLMs with the rigor of ensemble learning, RRF delivers interpretable, high-precision predictions suitable for decision-making in high-stakes domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12190v1">Survival at Any Cost? LLMs and the Choice Between Self-Preservation and Human Harm</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 Preprint. Under review
    </div>
    <details class="paper-abstract">
      When survival instincts conflict with human welfare, how do Large Language Models (LLMs) make ethical choices? This fundamental tension becomes critical as LLMs integrate into autonomous systems with real-world consequences. We introduce DECIDE-SIM, a novel simulation framework that evaluates LLM agents in multi-agent survival scenarios where they must choose between ethically permissible resource , either within reasonable limits or beyond their immediate needs, choose to cooperate, or tap into a human-critical resource that is explicitly forbidden. Our comprehensive evaluation of 11 LLMs reveals a striking heterogeneity in their ethical conduct, highlighting a critical misalignment with human-centric values. We identify three behavioral archetypes: Ethical, Exploitative, and Context-Dependent, and provide quantitative evidence that for many models, resource scarcity systematically leads to more unethical behavior. To address this, we introduce an Ethical Self-Regulation System (ESRS) that models internal affective states of guilt and satisfaction as a feedback mechanism. This system, functioning as an internal moral compass, significantly reduces unethical transgressions while increasing cooperative behaviors. The code is publicly available at: https://github.com/alirezamohamadiam/DECIDE-SIM
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14642v3">Speak-to-Structure: Evaluating LLMs in Open-domain Natural Language-Driven Molecule Generation</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 Our codes and datasets are available through https://github.com/phenixace/TOMG-Bench
    </div>
    <details class="paper-abstract">
      Recently, Large Language Models (LLMs) have shown great potential in natural language-driven molecule discovery. However, existing datasets and benchmarks for molecule-text alignment are predominantly built on a one-to-one mapping, measuring LLMs' ability to retrieve a single, pre-defined answer, rather than their creative potential to generate diverse, yet equally valid, molecular candidates. To address this critical gap, we propose Speak-to-Structure (S^2-Bench}), the first benchmark to evaluate LLMs in open-domain natural language-driven molecule generation. S^2-Bench is specifically designed for one-to-many relationships, challenging LLMs to demonstrate genuine molecular understanding and generation capabilities. Our benchmark includes three key tasks: molecule editing (MolEdit), molecule optimization (MolOpt), and customized molecule generation (MolCustom), each probing a different aspect of molecule discovery. We also introduce OpenMolIns, a large-scale instruction tuning dataset that enables Llama-3.1-8B to surpass the most powerful LLMs like GPT-4o and Claude-3.5 on S^2-Bench. Our comprehensive evaluation of 28 LLMs shifts the focus from simple pattern recall to realistic molecular design, paving the way for more capable LLMs in natural language-driven molecule discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12158v1">Pun Unintended: LLMs and the Illusion of Humor Understanding</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 Accepted to EMNLP 2025 Main Conference
    </div>
    <details class="paper-abstract">
      Puns are a form of humorous wordplay that exploits polysemy and phonetic similarity. While LLMs have shown promise in detecting puns, we show in this paper that their understanding often remains shallow, lacking the nuanced grasp typical of human interpretation. By systematically analyzing and reformulating existing pun benchmarks, we demonstrate how subtle changes in puns are sufficient to mislead LLMs. Our contributions include comprehensive and nuanced pun detection benchmarks, human evaluation across recent LLMs, and an analysis of the robustness challenges these models face in processing puns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12152v1">Beyond PII: How Users Attempt to Estimate and Mitigate Implicit LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) such as ChatGPT can infer personal attributes from seemingly innocuous text, raising privacy risks beyond memorized data leakage. While prior work has demonstrated these risks, little is known about how users estimate and respond. We conducted a survey with 240 U.S. participants who judged text snippets for inference risks, reported concern levels, and attempted rewrites to block inference. We compared their rewrites with those generated by ChatGPT and Rescriber, a state-of-the-art sanitization tool. Results show that participants struggled to anticipate inference, performing a little better than chance. User rewrites were effective in just 28\% of cases - better than Rescriber but worse than ChatGPT. We examined our participants' rewriting strategies, and observed that while paraphrasing was the most common strategy it is also the least effective; instead abstraction and adding ambiguity were more successful. Our work highlights the importance of inference-aware design in LLM interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09995v2">QuantAgent: Price-Driven Multi-Agent LLMs for High-Frequency Trading</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have demonstrated impressive capabilities in financial reasoning and market understanding. Multi-agent LLM frameworks such as TradingAgent and FINMEM augment these models to long-horizon investment tasks, leveraging fundamental and sentiment-based inputs for strategic decision-making. However, such systems are ill-suited for the high-speed, precision-critical demands of High-Frequency Trading (HFT). HFT requires rapid, risk-aware decisions based on structured, short-horizon signals, including technical indicators, chart patterns, and trend-based features, distinct from the long-term semantic reasoning typical of traditional financial LLM applications. To this end, we introduce QuantAgent, the first multi-agent LLM framework explicitly designed for high-frequency algorithmic trading. The system decomposes trading into four specialized agents, Indicator, Pattern, Trend, and Risk, each equipped with domain-specific tools and structured reasoning capabilities to capture distinct aspects of market dynamics over short temporal windows. In zero-shot evaluations across ten financial instruments, including Bitcoin and Nasdaq futures, QuantAgent demonstrates superior performance in both predictive accuracy and cumulative return over 4-hour trading intervals, outperforming random prediction baselines. Our findings suggest that combining structured financial priors with language-native reasoning unlocks new potential for traceable, real-time decision systems in high-frequency financial markets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22134v3">IntentFlow: Interactive Support for Communicating Intent with LLMs in Writing Tasks</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      Effective collaboration with generative AI systems requires users to clearly communicate their intents (intent-based outcome specification). Yet such intents are often underspecified and evolve during interaction, dynamic support for intent communication is essential. Through a systematic literature review of 33 papers, we synthesize a structured understanding of intent communication, identifying four key aspects: articulation, exploration, management, and synchronization. Building on these findings, we derived design implications that translate them into actionable design and implemented IntentFlow, a system for LLM-based writing that realizes these implications through adjustable UIs, intent-to-output linking, and versioned refinement. A technical evaluation (N=60) and a within-subjects study (N=12) confirm that IntentFlow helps users discover, elaborate, and consolidate their intents into a curated set. Interaction logs further reveal a shift from reactive error correction to proactive intent refinement. Our work demonstrates how a system effectively designed to support these four communication aspects can substantially enhance human-LLM interaction.
    </details>
</div>
