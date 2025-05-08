# llm - 2025_04

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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04386v1">Decoding Recommendation Behaviors of In-Context Learning LLMs Through Gradient Descent</a></div>
    <div class="paper-meta">
      📅 2025-04-06
      | 💬 12 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Recently, there has been a growing trend in utilizing large language models (LLMs) for recommender systems, referred to as LLMRec. A notable approach within this trend is not to fine-tune these models directly but instead to leverage In-Context Learning (ICL) methods tailored for LLMRec, denoted as LLM-ICL Rec. Many contemporary techniques focus on harnessing ICL content to enhance LLMRec performance. However, optimizing LLMRec with ICL content presents unresolved challenges. Specifically, two key issues stand out: (1) the limited understanding of why using a few demonstrations without model fine-tuning can lead to better performance compared to zero-shot recommendations. (2) the lack of evaluation metrics for demonstrations in LLM-ICL Rec and the absence of the theoretical analysis and practical design for optimizing the generation of ICL content for recommendation contexts. To address these two main issues, we propose a theoretical model, the LLM-ICL Recommendation Equivalent Gradient Descent model (LRGD) in this paper, which connects recommendation generation with gradient descent dynamics. We demonstrate that the ICL inference process in LLM aligns with the training procedure of its dual model, producing token predictions equivalent to the dual model's testing outputs. Building on these theoretical insights, we propose an evaluation metric for assessing demonstration quality. We integrate perturbations and regularizations in LRGD to enhance the robustness of the recommender system. To further improve demonstration effectiveness, prevent performance collapse, and ensure long-term adaptability, we also propose a two-stage optimization process in practice. Extensive experiments and detailed analysis on three Amazon datasets validate the theoretical equivalence and support the effectiveness of our theoretical analysis and practical module design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09061v2">CRANE: Reasoning with constrained LLM generation</a></div>
    <div class="paper-meta">
      📅 2025-04-06
      | 💬 Appearing at VerifAI@ICLR 2025
    </div>
    <details class="paper-abstract">
      Code generation, symbolic math reasoning, and other tasks require LLMs to produce outputs that are both syntactically and semantically correct. Constrained LLM generation is a promising direction to enforce adherence to formal grammar, but prior works have empirically observed that strict enforcement of formal constraints often diminishes the reasoning capabilities of LLMs. In this work, we first provide a theoretical explanation for why constraining LLM outputs to very restrictive grammars that only allow syntactically valid final answers reduces the reasoning capabilities of the model. Second, we demonstrate that by augmenting the output grammar with carefully designed additional rules, it is always possible to preserve the reasoning capabilities of the LLM while ensuring syntactic and semantic correctness in its outputs. Building on these theoretical insights, we propose a reasoning-augmented constrained decoding algorithm, CRANE, which effectively balances the correctness of constrained generation with the flexibility of unconstrained generation. Experiments on multiple open-source LLMs and benchmarks show that CRANE significantly outperforms both state-of-the-art constrained decoding strategies and standard unconstrained decoding, showing up to 10% points accuracy improvement over baselines on challenging symbolic reasoning benchmarks GSM-symbolic and FOLIO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04193v1">AiReview: An Open Platform for Accelerating Systematic Reviews with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-05
      | 💬 Accepted at SIGIR 2025
    </div>
    <details class="paper-abstract">
      Systematic reviews are fundamental to evidence-based medicine. Creating one is time-consuming and labour-intensive, mainly due to the need to screen, or assess, many studies for inclusion in the review. Several tools have been developed to streamline this process, mostly relying on traditional machine learning methods. Large language models (LLMs) have shown potential in further accelerating the screening process. However, no tool currently allows end users to directly leverage LLMs for screening or facilitates systematic and transparent usage of LLM-assisted screening methods. This paper introduces (i) an extensible framework for applying LLMs to systematic review tasks, particularly title and abstract screening, and (ii) a web-based interface for LLM-assisted screening. Together, these elements form AiReview-a novel platform for LLM-assisted systematic review creation. AiReview is the first of its kind to bridge the gap between cutting-edge LLM-assisted screening methods and those that create medical systematic reviews. The tool is available at https://aireview.ielab.io. The source code is also open sourced at https://github.com/ielab/ai-review.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09334v2">CyberLLMInstruct: A New Dataset for Analysing Safety of Fine-Tuned LLMs Using Cyber Security Data</a></div>
    <div class="paper-meta">
      📅 2025-04-05
    </div>
    <details class="paper-abstract">
      The integration of large language models (LLMs) into cyber security applications presents significant opportunities, such as enhancing threat analysis and malware detection, but can also introduce critical risks and safety concerns, including personal data leakage and automated generation of new malware. To address these challenges, we developed CyberLLMInstruct, a dataset of 54,928 instruction-response pairs spanning cyber security tasks such as malware analysis, phishing simulations, and zero-day vulnerabilities. The dataset was constructed through a multi-stage process. This involved sourcing data from multiple resources, filtering and structuring it into instruction-response pairs, and aligning it with real-world scenarios to enhance its applicability. Seven open-source LLMs were chosen to test the usefulness of CyberLLMInstruct: Phi 3 Mini 3.8B, Mistral 7B, Qwen 2.5 7B, Llama 3 8B, Llama 3.1 8B, Gemma 2 9B, and Llama 2 70B. In our primary example, we rigorously assess the safety of fine-tuned models using the OWASP top 10 framework, finding that fine-tuning reduces safety resilience across all tested LLMs and every adversarial attack (e.g., the security score of Llama 3.1 8B against prompt injection drops from 0.95 to 0.15). In our second example, we show that these same fine-tuned models can also achieve up to 92.50 percent accuracy on the CyberMetric benchmark. These findings highlight a trade-off between performance and safety, showing the importance of adversarial testing and further research into fine-tuning methodologies that can mitigate safety risks while still improving performance across diverse datasets and domains. The dataset creation pipeline, along with comprehensive documentation, examples, and resources for reproducing our results, is publicly available at https://github.com/Adelsamir01/CyberLLMInstruct.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04187v1">AttackLLM: LLM-based Attack Pattern Generation for an Industrial Control System</a></div>
    <div class="paper-meta">
      📅 2025-04-05
    </div>
    <details class="paper-abstract">
      Malicious examples are crucial for evaluating the robustness of machine learning algorithms under attack, particularly in Industrial Control Systems (ICS). However, collecting normal and attack data in ICS environments is challenging due to the scarcity of testbeds and the high cost of human expertise. Existing datasets are often limited by the domain expertise of practitioners, making the process costly and inefficient. The lack of comprehensive attack pattern data poses a significant problem for developing robust anomaly detection methods. In this paper, we propose a novel approach that combines data-centric and design-centric methodologies to generate attack patterns using large language models (LLMs). Our results demonstrate that the attack patterns generated by LLMs not only surpass the quality and quantity of those created by human experts but also offer a scalable solution that does not rely on expensive testbeds or pre-existing attack examples. This multi-agent based approach presents a promising avenue for enhancing the security and resilience of ICS environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04178v1">MSL: Not All Tokens Are What You Need for Tuning LLM as a Recommender</a></div>
    <div class="paper-meta">
      📅 2025-04-05
    </div>
    <details class="paper-abstract">
      Large language models (LLMs), known for their comprehension capabilities and extensive knowledge, have been increasingly applied to recommendation systems (RS). Given the fundamental gap between the mechanism of LLMs and the requirement of RS, researchers have focused on fine-tuning LLMs with recommendation-specific data to enhance their performance. Language Modeling Loss (LML), originally designed for language generation tasks, is commonly adopted. However, we identify two critical limitations of LML: 1) it exhibits significant divergence from the recommendation objective; 2) it erroneously treats all fictitious item descriptions as negative samples, introducing misleading training signals. To address these limitations, we propose a novel Masked Softmax Loss (MSL) tailored for fine-tuning LLMs on recommendation. MSL improves LML by identifying and masking invalid tokens that could lead to fictitious item descriptions during loss computation. This strategy can effectively avoid the interference from erroneous negative signals and ensure well alignment with the recommendation objective supported by theoretical guarantees. During implementation, we identify a potential challenge related to gradient vanishing of MSL. To overcome this, we further introduce the temperature coefficient and propose an Adaptive Temperature Strategy (ATS) that adaptively adjusts the temperature without requiring extensive hyperparameter tuning. Extensive experiments conducted on four public datasets further validate the effectiveness of MSL, achieving an average improvement of 42.24% in NDCG@10. The code is available at https://github.com/WANGBohaO-jpg/MSL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04152v1">Rethinking Multilingual Continual Pretraining: Data Mixing for Adapting LLMs Across Languages and Resources</a></div>
    <div class="paper-meta">
      📅 2025-04-05
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit significant disparities in performance across languages, primarily benefiting high-resource languages while marginalizing underrepresented ones. Continual Pretraining (CPT) has emerged as a promising approach to address this imbalance, although the relative effectiveness of monolingual, bilingual, and code-augmented data strategies remains unclear. This study systematically evaluates 36 CPT configurations involving three multilingual base models, across 30+ languages categorized as altruistic, selfish, and stagnant, spanning various resource levels. Our findings reveal three major insights: (1) Bilingual CPT improves multilingual classification but often causes language mixing issues during generation. (2) Including programming code data during CPT consistently enhances multilingual classification accuracy, particularly benefiting low-resource languages, but introduces a trade-off by slightly degrading generation quality. (3) Contrary to prior work, we observe substantial deviations from language classifications according to their impact on cross-lingual transfer: Languages classified as altruistic often negatively affect related languages, selfish languages show conditional and configuration-dependent behavior, and stagnant languages demonstrate surprising adaptability under certain CPT conditions. These nuanced interactions emphasize the complexity of multilingual representation learning, underscoring the importance of systematic studies on generalizable language classification to inform future multilingual CPT strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.07920v4">Rank-DistiLLM: Closing the Effectiveness Gap Between Cross-Encoders and LLMs for Passage Re-Ranking</a></div>
    <div class="paper-meta">
      📅 2025-04-05
      | 💬 Accepted at ECIR'25
    </div>
    <details class="paper-abstract">
      Cross-encoders distilled from large language models (LLMs) are often more effective re-rankers than cross-encoders fine-tuned on manually labeled data. However, distilled models do not match the effectiveness of their teacher LLMs. We hypothesize that this effectiveness gap is due to the fact that previous work has not applied the best-suited methods for fine-tuning cross-encoders on manually labeled data (e.g., hard-negative sampling, deep sampling, and listwise loss functions). To close this gap, we create a new dataset, Rank-DistiLLM. Cross-encoders trained on Rank-DistiLLM achieve the effectiveness of LLMs while being up to 173 times faster and 24 times more memory efficient. Our code and data is available at https://github.com/webis-de/ECIR-25.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15860v2">Synthetic vs. Gold: The Role of LLM-Generated Labels and Data in Cyberbullying Detection</a></div>
    <div class="paper-meta">
      📅 2025-04-05
    </div>
    <details class="paper-abstract">
      Cyberbullying (CB) presents a pressing threat, especially to children, underscoring the urgent need for robust detection systems to ensure online safety. However, progress in developing such systems is hindered by the scarcity of large, labeled datasets that are specifically tailored for specialized tasks and the target age groups. Creating these datasets relies heavily on human annotation, which not only strains resources but also raises significant ethical and legal concerns due to annotators' exposure to harmful content, notwithstanding the acquisition of this type of data from vulnerable populations such as children. In this paper, we address these challenges by leveraging Large Language Models (LLMs) to generate synthetic data and labels. Our experiments demonstrate that synthetic data enables BERT-based CB classifiers to achieve performance close to that of those trained on fully authentic datasets (75.8% vs. 81.5% accuracy). Additionally, LLMs can effectively label authentic yet unlabeled data, allowing BERT classifiers to attain a comparable performance level (79.1% vs. 81.5% accuracy). These results highlight the potential of LLMs as a scalable, ethical, and cost-effective solution for generating data for CB detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04110v1">PEIRCE: Unifying Material and Formal Reasoning via LLM-Driven Neuro-Symbolic Refinement</a></div>
    <div class="paper-meta">
      📅 2025-04-05
      | 💬 Demo paper. Work in progress
    </div>
    <details class="paper-abstract">
      A persistent challenge in AI is the effective integration of material and formal inference - the former concerning the plausibility and contextual relevance of arguments, while the latter focusing on their logical and structural validity. Large Language Models (LLMs), by virtue of their extensive pre-training on large textual corpora, exhibit strong capabilities in material inference. However, their reasoning often lacks formal rigour and verifiability. At the same time, LLMs' linguistic competence positions them as a promising bridge between natural and formal languages, opening up new opportunities for combining these two modes of reasoning. In this paper, we introduce PEIRCE, a neuro-symbolic framework designed to unify material and formal inference through an iterative conjecture-criticism process. Within this framework, LLMs play the central role of generating candidate solutions in natural and formal languages, which are then evaluated and refined via interaction with external critique models. These critiques include symbolic provers, which assess formal validity, as well as soft evaluators that measure the quality of the generated arguments along linguistic and epistemic dimensions such as plausibility, coherence, and parsimony. While PEIRCE is a general-purpose framework, we demonstrate its capabilities in the domain of natural language explanation generation - a setting that inherently demands both material adequacy and formal correctness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.03747v2">Context-Alignment: Activating and Enhancing LLM Capabilities in Time Series</a></div>
    <div class="paper-meta">
      📅 2025-04-05
      | 💬 no comment
    </div>
    <details class="paper-abstract">
      Recently, leveraging pre-trained Large Language Models (LLMs) for time series (TS) tasks has gained increasing attention, which involves activating and enhancing LLMs' capabilities. Many methods aim to activate LLMs' capabilities based on token-level alignment but overlook LLMs' inherent strength on natural language processing -- their deep understanding of linguistic logic and structure rather than superficial embedding processing. We propose Context-Alignment, a new paradigm that aligns TS with a linguistic component in the language environments familiar to LLMs to enable LLMs to contextualize and comprehend TS data, thereby activating their capabilities. Specifically, such context-level alignment comprises structural alignment and logical alignment, which is achieved by a Dual-Scale Context-Alignment GNNs (DSCA-GNNs) applied to TS-language multimodal inputs. Structural alignment utilizes dual-scale nodes to describe hierarchical structure in TS-language, enabling LLMs treat long TS data as a whole linguistic component while preserving intrinsic token features. Logical alignment uses directed edges to guide logical relationships, ensuring coherence in the contextual semantics. Demonstration examples prompt are employed to construct Demonstration Examples based Context-Alignment (DECA) following DSCA-GNNs framework. DECA can be flexibly and repeatedly integrated into various layers of pre-trained LLMs to improve awareness of logic and structure, thereby enhancing performance. Extensive experiments show the effectiveness of DECA and the importance of Context-Alignment across tasks, particularly in few-shot and zero-shot forecasting, confirming that Context-Alignment provide powerful prior knowledge on context.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04083v1">A Benchmark for End-to-End Zero-Shot Biomedical Relation Extraction with LLMs: Experiments with OpenAI Models</a></div>
    <div class="paper-meta">
      📅 2025-04-05
    </div>
    <details class="paper-abstract">
      Objective: Zero-shot methodology promises to cut down on costs of dataset annotation and domain expertise needed to make use of NLP. Generative large language models trained to align with human goals have achieved high zero-shot performance across a wide variety of tasks. As of yet, it is unclear how well these models perform on biomedical relation extraction (RE). To address this knowledge gap, we explore patterns in the performance of OpenAI LLMs across a diverse sampling of RE tasks. Methods: We use OpenAI GPT-4-turbo and their reasoning model o1 to conduct end-to-end RE experiments on seven datasets. We use the JSON generation capabilities of GPT models to generate structured output in two ways: (1) by defining an explicit schema describing the structure of relations, and (2) using a setting that infers the structure from the prompt language. Results: Our work is the first to study and compare the performance of the GPT-4 and o1 for the end-to-end zero-shot biomedical RE task across a broad array of datasets. We found the zero-shot performances to be proximal to that of fine-tuned methods. The limitations of this approach are that it performs poorly on instances containing many relations and errs on the boundaries of textual mentions. Conclusion: Recent large language models exhibit promising zero-shot capabilities in complex biomedical RE tasks, offering competitive performance with reduced dataset curation and NLP modeling needs at the cost of increased computing, potentially increasing medical community accessibility. Addressing the limitations we identify could further boost reliability. The code, data, and prompts for all our experiments are publicly available: https://github.com/bionlproc/ZeroShotRE
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04060v1">VocalNet: Speech LLM with Multi-Token Prediction for Faster and High-Quality Generation</a></div>
    <div class="paper-meta">
      📅 2025-04-05
    </div>
    <details class="paper-abstract">
      Speech large language models (LLMs) have emerged as a prominent research focus in speech processing. We propose VocalNet-1B and VocalNet-8B, a series of high-performance, low-latency speech LLMs enabled by a scalable and model-agnostic training framework for real-time voice interaction. Departing from the conventional next-token prediction (NTP), we introduce multi-token prediction (MTP), a novel approach optimized for speech LLMs that simultaneously improves generation speed and quality. Experiments show that VocalNet outperforms mainstream Omni LLMs despite using significantly less training data, while also surpassing existing open-source speech LLMs by a substantial margin. To support reproducibility and community advancement, we will open-source all model weights, inference code, training data, and framework implementations upon publication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04030v1">OpenCodeInstruct: A Large-scale Instruction Tuning Dataset for Code LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-05
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have transformed software development by enabling code generation, automated debugging, and complex reasoning. However, their continued advancement is constrained by the scarcity of high-quality, publicly available supervised fine-tuning (SFT) datasets tailored for coding tasks. To bridge this gap, we introduce OpenCodeInstruct, the largest open-access instruction tuning dataset, comprising 5 million diverse samples. Each sample includes a programming question, solution, test cases, execution feedback, and LLM-generated quality assessments. We fine-tune various base models, including LLaMA and Qwen, across multiple scales (1B+, 3B+, and 7B+) using our dataset. Comprehensive evaluations on popular benchmarks (HumanEval, MBPP, LiveCodeBench, and BigCodeBench) demonstrate substantial performance improvements achieved by SFT with OpenCodeInstruct. We also present a detailed methodology encompassing seed data curation, synthetic instruction and solution generation, and filtering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.20749v3">Beyond Believability: Accurate Human Behavior Simulation with Fine-Tuned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-05
    </div>
    <details class="paper-abstract">
      Recent research shows that LLMs can simulate ``believable'' human behaviors to power LLM agents via prompt-only methods. In this work, we focus on evaluating and improving LLM's objective ``accuracy'' rather than the subjective ``believability'' in the web action generation task, leveraging a large-scale, real-world dataset collected from online shopping human actions. We present the first comprehensive quantitative evaluation of state-of-the-art LLMs (e.g., DeepSeek-R1, Llama, and Claude) on the task of web action generation. Our results show that fine-tuning LLMs on real-world behavioral data substantially improves their ability to generate actions compared to prompt-only methods. Furthermore, incorporating synthesized reasoning traces into model training leads to additional performance gains, demonstrating the value of explicit rationale in behavior modeling. This work establishes a new benchmark for evaluating LLMs in behavior simulation and offers actionable insights into how real-world action data and reasoning augmentation can enhance the fidelity of LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12561v3">UXAgent: An LLM Agent-Based Usability Testing Framework for Web Design</a></div>
    <div class="paper-meta">
      📅 2025-04-05
    </div>
    <details class="paper-abstract">
      Usability testing is a fundamental yet challenging (e.g., inflexible to iterate the study design flaws and hard to recruit study participants) research method for user experience (UX) researchers to evaluate a web design. Recent advances in Large Language Model-simulated Agent (LLM-Agent) research inspired us to design UXAgent to support UX researchers in evaluating and reiterating their usability testing study design before they conduct the real human subject study. Our system features an LLM-Agent module and a universal browser connector module so that UX researchers can automatically generate thousands of simulated users to test the target website. The results are shown in qualitative (e.g., interviewing how an agent thinks ), quantitative (e.g., # of actions), and video recording formats for UX researchers to analyze. Through a heuristic user evaluation with five UX researchers, participants praised the innovation of our system but also expressed concerns about the future of LLM Agent-assisted UX study.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12784v2">JudgeBench: A Benchmark for Evaluating LLM-based Judges</a></div>
    <div class="paper-meta">
      📅 2025-04-05
      | 💬 Published as a conference paper at ICLR 2025
    </div>
    <details class="paper-abstract">
      LLM-based judges have emerged as a scalable alternative to human evaluation and are increasingly used to assess, compare, and improve models. However, the reliability of LLM-based judges themselves is rarely scrutinized. As LLMs become more advanced, their responses grow more sophisticated, requiring stronger judges to evaluate them. Existing benchmarks primarily focus on a judge's alignment with human preferences, but often fail to account for more challenging tasks where crowdsourced human preference is a poor indicator of factual and logical correctness. To address this, we propose a novel evaluation framework to objectively evaluate LLM-based judges. Based on this framework, we propose JudgeBench, a benchmark for evaluating LLM-based judges on challenging response pairs spanning knowledge, reasoning, math, and coding. JudgeBench leverages a novel pipeline for converting existing difficult datasets into challenging response pairs with preference labels reflecting objective correctness. Our comprehensive evaluation on a collection of prompted judges, fine-tuned judges, multi-agent judges, and reward models shows that JudgeBench poses a significantly greater challenge than previous benchmarks, with many strong models (e.g., GPT-4o) performing just slightly better than random guessing. Overall, JudgeBench offers a reliable platform for assessing increasingly advanced LLM-based judges. Data and code are available at https://github.com/ScalerLab/JudgeBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07983v1">Psychological Health Knowledge-Enhanced LLM-based Social Network Crisis Intervention Text Transfer Recognition Method</a></div>
    <div class="paper-meta">
      📅 2025-04-05
    </div>
    <details class="paper-abstract">
      As the prevalence of mental health crises increases on social media platforms, identifying and preventing potential harm has become an urgent challenge. This study introduces a large language model (LLM)-based text transfer recognition method for social network crisis intervention, enhanced with domain-specific mental health knowledge. We propose a multi-level framework that incorporates transfer learning using BERT, and integrates mental health knowledge, sentiment analysis, and behavior prediction techniques. The framework includes a crisis annotation tool trained on social media datasets from real-world events, enabling the model to detect nuanced emotional cues and identify psychological crises. Experimental results show that the proposed method outperforms traditional models in crisis detection accuracy and exhibits greater sensitivity to subtle emotional and contextual variations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04295v1">Dynamic Hedging Strategies in Derivatives Markets with LLM-Driven Sentiment and News Analytics</a></div>
    <div class="paper-meta">
      📅 2025-04-05
      | 💬 Accepted by IJCNN 2025
    </div>
    <details class="paper-abstract">
      Dynamic hedging strategies are essential for effective risk management in derivatives markets, where volatility and market sentiment can greatly impact performance. This paper introduces a novel framework that leverages large language models (LLMs) for sentiment analysis and news analytics to inform hedging decisions. By analyzing textual data from diverse sources like news articles, social media, and financial reports, our approach captures critical sentiment indicators that reflect current market conditions. The framework allows for real-time adjustments to hedging strategies, adapting positions based on continuous sentiment signals. Backtesting results on historical derivatives data reveal that our dynamic hedging strategies achieve superior risk-adjusted returns compared to conventional static approaches. The incorporation of LLM-driven sentiment analysis into hedging practices presents a significant advancement in decision-making processes within derivatives trading. This research showcases how sentiment-informed dynamic hedging can enhance portfolio management and effectively mitigate associated risks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04292v1">Cross-Asset Risk Management: Integrating LLMs for Real-Time Monitoring of Equity, Fixed Income, and Currency Markets</a></div>
    <div class="paper-meta">
      📅 2025-04-05
      | 💬 Accepted by IJCNN 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have emerged as powerful tools in the field of finance, particularly for risk management across different asset classes. In this work, we introduce a Cross-Asset Risk Management framework that utilizes LLMs to facilitate real-time monitoring of equity, fixed income, and currency markets. This innovative approach enables dynamic risk assessment by aggregating diverse data sources, ultimately enhancing decision-making processes. Our model effectively synthesizes and analyzes market signals to identify potential risks and opportunities while providing a holistic view of asset classes. By employing advanced analytics, we leverage LLMs to interpret financial texts, news articles, and market reports, ensuring that risks are contextualized within broader market narratives. Extensive backtesting and real-time simulations validate the framework, showing increased accuracy in predicting market shifts compared to conventional methods. The focus on real-time data integration enhances responsiveness, allowing financial institutions to manage risks adeptly under varying market conditions and promoting financial stability through the advanced application of LLMs in risk analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09647v4">Leveraging LLMS for Top-Down Sector Allocation In Automated Trading</a></div>
    <div class="paper-meta">
      📅 2025-04-05
    </div>
    <details class="paper-abstract">
      This paper introduces a methodology leveraging Large Language Models (LLMs) for sector-level portfolio allocation through systematic analysis of macroeconomic conditions and market sentiment. Our framework emphasizes top-down sector allocation by processing multiple data streams simultaneously, including policy documents, economic indicators, and sentiment patterns. Empirical results demonstrate superior risk-adjusted returns compared to traditional cross momentum strategies, achieving a Sharpe ratio of 2.51 and portfolio return of 8.79% versus -0.61 and -1.39% respectively. These results suggest that LLM-based systematic macro analysis presents a viable approach for enhancing automated portfolio allocation decisions at the sector level.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04199v1">Investigating and Mitigating Stereotype-aware Unfairness in LLM-based Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-04-05
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated unprecedented language understanding and reasoning capabilities to capture diverse user preferences and advance personalized recommendations. Despite the growing interest in LLM-based personalized recommendations, unique challenges are brought to the trustworthiness of LLM-based recommender systems (LLM-RS), since LLMs are likely to inherit stereotypes that are embedded ubiquitously in word embeddings due to their training on large-scale uncurated datasets. This leads to LLM-RS exhibiting stereotypical linguistic associations between users and items. However, there remains a lack of studies investigating the simultaneous existence of stereotypes between users and items in LLM-RS. To bridge this gap, this study reveals a new variant of fairness between stereotype groups containing both users and items, to quantify discrimination against stereotypes in LLM-RS. Moreover, in this paper, to mitigate stereotype-aware unfairness in textual user and item information, we propose a novel framework (MoS), in which an insightful stereotype-wise routing strategy over multiple stereotype-relevant experts is designed to learn unbiased representations against different stereotypes in LLM- RS. Extensive experiments are conducted to analyze the influence of stereotype-aware fairness in LLM-RS and the effectiveness of our proposed methods, which consistently outperform competitive benchmarks under various fairness settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03360v1">Sustainable LLM Inference for Edge AI: Evaluating Quantized LLMs for Energy Efficiency, Output Accuracy, and Inference Latency</a></div>
    <div class="paper-meta">
      📅 2025-04-04
      | 💬 30 pages, 14 figures
    </div>
    <details class="paper-abstract">
      Deploying Large Language Models (LLMs) on edge devices presents significant challenges due to computational constraints, memory limitations, inference speed, and energy consumption. Model quantization has emerged as a key technique to enable efficient LLM inference by reducing model size and computational overhead. In this study, we conduct a comprehensive analysis of 28 quantized LLMs from the Ollama library, which applies by default Post-Training Quantization (PTQ) and weight-only quantization techniques, deployed on an edge device (Raspberry Pi 4 with 4GB RAM). We evaluate energy efficiency, inference performance, and output accuracy across multiple quantization levels and task types. Models are benchmarked on five standardized datasets (CommonsenseQA, BIG-Bench Hard, TruthfulQA, GSM8K, and HumanEval), and we employ a high-resolution, hardware-based energy measurement tool to capture real-world power consumption. Our findings reveal the trade-offs between energy efficiency, inference speed, and accuracy in different quantization settings, highlighting configurations that optimize LLM deployment for resource-constrained environments. By integrating hardware-level energy profiling with LLM benchmarking, this study provides actionable insights for sustainable AI, bridging a critical gap in existing research on energy-aware LLM deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03343v1">Talk2X -- An Open-Source Toolkit Facilitating Deployment of LLM-Powered Chatbots on the Web</a></div>
    <div class="paper-meta">
      📅 2025-04-04
    </div>
    <details class="paper-abstract">
      Integrated into websites, LLM-powered chatbots offer alternative means of navigation and information retrieval, leading to a shift in how users access information on the web. Yet, predominantly closed-sourced solutions limit proliferation among web hosts and suffer from a lack of transparency with regard to implementation details and energy efficiency. In this work, we propose our openly available agent Talk2X leveraging an adapted retrieval-augmented generation approach (RAG) combined with an automatically generated vector database, benefiting energy efficiency. Talk2X's architecture is generalizable to arbitrary websites offering developers a ready to use tool for integration. Using a mixed-methods approach, we evaluated Talk2X's usability by tasking users to acquire specific assets from an open science repository. Talk2X significantly improved task completion time, correctness, and user experience supporting users in quickly pinpointing specific information as compared to standard user-website interaction. Our findings contribute technical advancements to an ongoing paradigm shift of how we access information on the web.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.00159v3">LLMs Prompted for Graphs: Hallucinations and Generative Capabilities</a></div>
    <div class="paper-meta">
      📅 2025-04-04
      | 💬 A preliminary version of this work appeared in the Complex Networks 2024 conference, under the title "LLMs hallucinate graphs too: a structural perspective"
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are nowadays prompted for a wide variety of tasks. In this article, we investigate their ability in reciting and generating graphs. We first study the ability of LLMs to regurgitate well known graphs from the literature (e.g. Karate club or the graph atlas)4. Secondly, we question the generative capabilities of LLMs by asking for Erdos-Renyi random graphs. As opposed to the possibility that they could memorize some Erdos-Renyi graphs included in their scraped training set, this second investigation aims at studying a possible emergent property of LLMs. For both tasks, we propose a metric to assess their errors with the lens of hallucination (i.e. incorrect information returned as facts). We most notably find that the amplitude of graph hallucinations can characterize the superiority of some LLMs. Indeed, for the recitation task, we observe that graph hallucinations correlate with the Hallucination Leaderboard, a hallucination rank that leverages 10, 000 times more prompts to obtain its ranking. For the generation task, we find surprisingly good and reproducible results in most of LLMs. We believe this to constitute a starting point for more in-depth studies of this emergent capability and a challenging benchmark for their improvements. Altogether, these two aspects of LLMs capabilities bridge a gap between the network science and machine learning communities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03312v1">Evaluating Compact LLMs for Zero-Shot Iberian Language Tasks on End-User Devices</a></div>
    <div class="paper-meta">
      📅 2025-04-04
      | 💬 Under Revision al SEPLN conference
    </div>
    <details class="paper-abstract">
      Large Language Models have significantly advanced natural language processing, achieving remarkable performance in tasks such as language generation, translation, and reasoning. However, their substantial computational requirements restrict deployment to high-end systems, limiting accessibility on consumer-grade devices. This challenge is especially pronounced for under-resourced languages like those spoken in the Iberian Peninsula, where relatively limited linguistic resources and benchmarks hinder effective evaluation. This work presents a comprehensive evaluation of compact state-of-the-art LLMs across several essential NLP tasks tailored for Iberian languages. The results reveal that while some models consistently excel in certain tasks, significant performance gaps remain, particularly for languages such as Basque. These findings highlight the need for further research on balancing model compactness with robust multilingual performance
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03255v1">Inherent and emergent liability issues in LLM-based agentic systems: a principal-agent perspective</a></div>
    <div class="paper-meta">
      📅 2025-04-04
      | 💬 12 pages content (incl. appendix) + 12 pages references, comments welcome
    </div>
    <details class="paper-abstract">
      Agentic systems powered by large language models (LLMs) are becoming progressively more complex and capable. Their increasing agency and expanding deployment settings attract growing attention over effective governance policies, monitoring and control protocols. Based on emerging landscapes of the agentic market, we analyze the potential liability issues stemming from delegated use of LLM agents and their extended systems from a principal-agent perspective. Our analysis complements existing risk-based studies on artificial agency and covers the spectrum of important aspects of the principal-agent relationship and their potential consequences at deployment. Furthermore, we motivate method developments for technical governance along the directions of interpretability and behavior evaluations, reward and conflict management, and the mitigation of misalignment and misconduct through principled engineering of detection and fail-safe mechanisms. By illustrating the outstanding issues in AI liability for LLM-based agentic systems, we aim to inform the system design, auditing and monitoring approaches to enhancing transparency and accountability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02732v2">Why do LLMs attend to the first token?</a></div>
    <div class="paper-meta">
      📅 2025-04-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) tend to attend heavily to the first token in the sequence -- creating a so-called attention sink. Many works have studied this phenomenon in detail, proposing various ways to either leverage or alleviate it. Attention sinks have been connected to quantisation difficulties, security issues, and streaming attention. Yet, while many works have provided conditions in which they occur or not, a critical question remains shallowly answered: Why do LLMs learn such patterns and how are they being used? In this work, we argue theoretically and empirically that this mechanism provides a method for LLMs to avoid over-mixing, connecting this to existing lines of work that study mathematically how information propagates in Transformers. We conduct experiments to validate our theoretical intuitions and show how choices such as context length, depth, and data packing influence the sink behaviour. We hope that this study provides a new practical perspective on why attention sinks are useful in LLMs, leading to a better understanding of the attention patterns that form during training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03174v1">Multi-lingual Multi-turn Automated Red Teaming for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-04
      | 💬 Accepted at TrustNLP@NAACL 2025
    </div>
    <details class="paper-abstract">
      Language Model Models (LLMs) have improved dramatically in the past few years, increasing their adoption and the scope of their capabilities over time. A significant amount of work is dedicated to ``model alignment'', i.e., preventing LLMs to generate unsafe responses when deployed into customer-facing applications. One popular method to evaluate safety risks is \textit{red-teaming}, where agents attempt to bypass alignment by crafting elaborate prompts that trigger unsafe responses from a model. Standard human-driven red-teaming is costly, time-consuming and rarely covers all the recent features (e.g., multi-lingual, multi-modal aspects), while proposed automation methods only cover a small subset of LLMs capabilities (i.e., English or single-turn). We present Multi-lingual Multi-turn Automated Red Teaming (\textbf{MM-ART}), a method to fully automate conversational, multi-lingual red-teaming operations and quickly identify prompts leading to unsafe responses. Through extensive experiments on different languages, we show the studied LLMs are on average 71\% more vulnerable after a 5-turn conversation in English than after the initial turn. For conversations in non-English languages, models display up to 195\% more safety vulnerabilities than the standard single-turn English approach, confirming the need for automated red-teaming methods matching LLMs capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03111v1">Les Dissonances: Cross-Tool Harvesting and Polluting in Multi-Tool Empowered LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-04
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents are autonomous systems powered by LLMs, capable of reasoning and planning to solve problems by leveraging a set of tools. However, the integration of multi-tool capabilities in LLM agents introduces challenges in securely managing tools, ensuring their compatibility, handling dependency relationships, and protecting control flows within LLM agent workflows. In this paper, we present the first systematic security analysis of task control flows in multi-tool-enabled LLM agents. We identify a novel threat, Cross-Tool Harvesting and Polluting (XTHP), which includes multiple attack vectors to first hijack the normal control flows of agent tasks, and then collect and pollute confidential or private information within LLM agent systems. To understand the impact of this threat, we developed Chord, a dynamic scanning tool designed to automatically detect real-world agent tools susceptible to XTHP attacks. Our evaluation of 73 real-world tools from the repositories of two major LLM agent development frameworks, LangChain and LlamaIndex, revealed a significant security concern: 80% of the tools are vulnerable to hijacking attacks, 78% to XTH attacks, and 41% to XTP attacks, highlighting the prevalence of this threat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03966v1">Bridging LMS and Generative AI: Dynamic Course Content Integration (DCCI) for Connecting LLMs to Course Content -- The Ask ME Assistant</a></div>
    <div class="paper-meta">
      📅 2025-04-04
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) with Learning Management Systems (LMSs) has the potential to enhance task automation and accessibility in education. However, hallucination where LLMs generate inaccurate or misleading information remains a significant challenge. This study introduces the Dynamic Course Content Integration (DCCI) mechanism, which dynamically retrieves and integrates course content and curriculum from Canvas LMS into the LLM-powered assistant, Ask ME. By employing prompt engineering to structure retrieved content within the LLM's context window, DCCI ensures accuracy, relevance, and contextual alignment, mitigating hallucination. To evaluate DCCI's effectiveness, Ask ME's usability, and broader student perceptions of AI in education, a mixed-methods approach was employed, incorporating user satisfaction ratings and a structured survey. Results from a pilot study indicate high user satisfaction (4.614/5), with students recognizing Ask ME's ability to provide timely and contextually relevant responses for both administrative and course-related inquiries. Additionally, a majority of students agreed that Ask ME's integration with course content in Canvas LMS reduced platform-switching, improving usability, engagement, and comprehension. AI's role in reducing classroom hesitation and fostering self-directed learning and intellectual curiosity was also highlighted. Despite these benefits and positive perception of AI tools, concerns emerged regarding over-reliance on AI, accuracy limitations, and ethical issues such as plagiarism and reduced student-teacher interaction. These findings emphasize the need for strategic AI implementation, ethical safeguards, and a pedagogical framework that prioritizes human-AI collaboration over substitution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14827v2">Enhancing Prompt Injection Attacks to LLMs via Poisoning Alignment</a></div>
    <div class="paper-meta">
      📅 2025-04-04
    </div>
    <details class="paper-abstract">
      In a prompt injection attack, an attacker injects a prompt into the original one, aiming to make an LLM follow the injected prompt to perform an attacker-chosen task. Existing attacks primarily focus on how to blend the injected prompt into the original prompt without altering the LLM itself. Our experiments show that these attacks achieve some success, but there is still significant room for improvement. In this work, we show that an attacker can boost the success of prompt injection attacks by poisoning the LLM's alignment process. Specifically, we propose PoisonedAlign, a method to strategically create poisoned alignment samples. When even a small fraction of the alignment data is poisoned using our method, the aligned LLM becomes more vulnerable to prompt injection while maintaining its foundational capabilities. The code is available at https://github.com/Sadcardation/PoisonedAlign
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18921v2">From Blind Solvers to Logical Thinkers: Benchmarking LLMs' Logical Integrity on Faulty Mathematical Problems</a></div>
    <div class="paper-meta">
      📅 2025-04-04
    </div>
    <details class="paper-abstract">
      Consider the math problem: "Lily received 3 cookies from her best friend yesterday and ate 5 for breakfast. Today, her friend gave her 3 more cookies. How many cookies does Lily have now?" Many large language models (LLMs) in previous research approach this problem by calculating the answer "1" using the equation "3 - 5 + 3." However, from a human perspective, we recognize the inherent flaw in this problem: Lily cannot eat 5 cookies if she initially only had 3. This discrepancy prompts a key question: Are current LLMs merely Blind Solver that apply mathematical operations without deeper reasoning, or can they function as Logical Thinker capable of identifying logical inconsistencies? To explore this question, we propose a benchmark dataset, FaultyMath, which includes faulty math problems of rich diversity: i) multiple mathematical categories, e.g., algebra, geometry, number theory, etc., ii) varying levels of difficulty, and iii) different origins of faultiness -- ranging from violations of common sense and ambiguous statements to mathematical contradictions and more. We evaluate a broad spectrum of LLMs, including open-source, closed-source, and math-specialized models, using FaultyMath across three dimensions: (i) How accurately can the models detect faulty math problems without being explicitly prompted to do so? (ii) When provided with hints -- either correct or misleading -- about the validity of the problems, to what extent do LLMs adapt to become reliable Logical Thinker? (iii) How trustworthy are the explanations generated by LLMs when they recognize a math problem as flawed? Through extensive experimentation and detailed analysis, our results demonstrate that existing LLMs largely function as Blind Solver and fall short of the reasoning capabilities required to perform as Logical Thinker.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18702v2">CypherBench: Towards Precise Retrieval over Full-scale Modern Knowledge Graphs in the LLM Era</a></div>
    <div class="paper-meta">
      📅 2025-04-04
    </div>
    <details class="paper-abstract">
      Retrieval from graph data is crucial for augmenting large language models (LLM) with both open-domain knowledge and private enterprise data, and it is also a key component in the recent GraphRAG system (edge et al., 2024). Despite decades of research on knowledge graphs and knowledge base question answering, leading LLM frameworks (e.g. Langchain and LlamaIndex) have only minimal support for retrieval from modern encyclopedic knowledge graphs like Wikidata. In this paper, we analyze the root cause and suggest that modern RDF knowledge graphs (e.g. Wikidata, Freebase) are less efficient for LLMs due to overly large schemas that far exceed the typical LLM context window, use of resource identifiers, overlapping relation types and lack of normalization. As a solution, we propose property graph views on top of the underlying RDF graph that can be efficiently queried by LLMs using Cypher. We instantiated this idea on Wikidata and introduced CypherBench, the first benchmark with 11 large-scale, multi-domain property graphs with 7.8 million entities and over 10,000 questions. To achieve this, we tackled several key challenges, including developing an RDF-to-property graph conversion engine, creating a systematic pipeline for text-to-Cypher task generation, and designing new evaluation metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03889v1">Using Attention Sinks to Identify and Evaluate Dormant Heads in Pretrained LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-04
      | 💬 22 pages, 14 figures
    </div>
    <details class="paper-abstract">
      Multi-head attention is foundational to large language models (LLMs), enabling different heads to have diverse focus on relevant input tokens. However, learned behaviors like attention sinks, where the first token receives most attention despite limited semantic importance, challenge our understanding of multi-head attention. To analyze this phenomenon, we propose a new definition for attention heads dominated by attention sinks, known as dormant attention heads. We compare our definition to prior work in a model intervention study where we test whether dormant heads matter for inference by zeroing out the output of dormant attention heads. Using six pretrained models and five benchmark datasets, we find our definition to be more model and dataset-agnostic. Using our definition on most models, more than 4% of a model's attention heads can be zeroed while maintaining average accuracy, and zeroing more than 14% of a model's attention heads can keep accuracy to within 1% of the pretrained model's average accuracy. Further analysis reveals that dormant heads emerge early in pretraining and can transition between dormant and active states during pretraining. Additionally, we provide evidence that they depend on characteristics of the input text.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02559v2">Leveraging LLM For Synchronizing Information Across Multilingual Tables</a></div>
    <div class="paper-meta">
      📅 2025-04-04
      | 💬 17 Pages, 11 Tables, 2 Figures
    </div>
    <details class="paper-abstract">
      The vast amount of online information today poses challenges for non-English speakers, as much of it is concentrated in high-resource languages such as English and French. Wikipedia reflects this imbalance, with content in low-resource languages frequently outdated or incomplete. Recent research has sought to improve cross-language synchronization of Wikipedia tables using rule-based methods. These approaches can be effective, but they struggle with complexity and generalization. This paper explores large language models (LLMs) for multilingual information synchronization, using zero-shot prompting as a scalable solution. We introduce the Information Updation dataset, simulating the real-world process of updating outdated Wikipedia tables, and evaluate LLM performance. Our findings reveal that single-prompt approaches often produce suboptimal results, prompting us to introduce a task decomposition strategy that enhances coherence and accuracy. Our proposed method outperforms existing baselines, particularly in Information Updation (1.79%) and Information Addition (20.58%), highlighting the model strength in dynamically updating and enriching data across architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03877v1">Concept-based Rubrics Improve LLM Formative Assessment and Data Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-04-04
      | 💬 13 pages excluding references. 9 tables and 4 figures
    </div>
    <details class="paper-abstract">
      Formative assessment in STEM topics aims to promote student learning by identifying students' current understanding, thus targeting how to promote further learning. Previous studies suggest that the assessment performance of current generative large language models (LLMs) on constructed responses to open-ended questions is significantly lower than that of supervised classifiers trained on high-quality labeled data. However, we demonstrate that concept-based rubrics can significantly enhance LLM performance, which narrows the gap between LLMs as off-the shelf assessment tools, and smaller supervised models, which need large amounts of training data. For datasets where concept-based rubrics allow LLMs to achieve strong performance, we show that the concept-based rubrics help the same LLMs generate high quality synthetic data for training lightweight, high-performance supervised models. Our experiments span diverse STEM student response datasets with labels of varying quality, including a new real-world dataset that contains some AI-assisted responses, which introduces additional considerations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00993v2">MedReason: Eliciting Factual Medical Reasoning Steps in LLMs via Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2025-04-04
      | 💬 18 pages, 11 figures, 6 tables. Project page: https://github.com/UCSC-VLAA/MedReason
    </div>
    <details class="paper-abstract">
      Medical tasks such as diagnosis and treatment planning require precise and complex reasoning, particularly in life-critical domains. Unlike mathematical reasoning, medical reasoning demands meticulous, verifiable thought processes to ensure reliability and accuracy. However, there is a notable lack of datasets that provide transparent, step-by-step reasoning to validate and enhance the medical reasoning ability of AI models. To bridge this gap, we introduce MedReason, a large-scale high-quality medical reasoning dataset designed to enable faithful and explainable medical problem-solving in large language models (LLMs). We utilize a structured medical knowledge graph (KG) to convert clinical QA pairs into logical chains of reasoning, or ``thinking paths'', which trace connections from question elements to answers via relevant KG entities. Each path is validated for consistency with clinical logic and evidence-based medicine. Our pipeline generates detailed reasoning for various medical questions from 7 medical datasets, resulting in a dataset of 32,682 question-answer pairs, each with detailed, step-by-step explanations. Experiments demonstrate that fine-tuning with our dataset consistently boosts medical problem-solving capabilities, achieving significant gains of up to 7.7% for DeepSeek-Ditill-8B. Our top-performing model, MedReason-8B, outperforms the Huatuo-o1-8B, a state-of-the-art medical reasoning model, by up to 4.2% on the clinical benchmark MedBullets. We also engage medical professionals from diverse specialties to assess our dataset's quality, ensuring MedReason offers accurate and coherent medical reasoning. Our data, models, and code is available at https://github.com/UCSC-VLAA/MedReason.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03846v1">Do LLM Evaluators Prefer Themselves for a Reason?</a></div>
    <div class="paper-meta">
      📅 2025-04-04
      | 💬 Preprint. 31 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as automatic evaluators in applications such as benchmarking, reward modeling, and self-refinement. Prior work highlights a potential self-preference bias where LLMs favor their own generated responses, a tendency often intensifying with model size and capability. This raises a critical question: Is self-preference detrimental, or does it simply reflect objectively superior outputs from more capable models? Disentangling these has been challenging due to the usage of subjective tasks in previous studies. To address this, we investigate self-preference using verifiable benchmarks (mathematical reasoning, factual knowledge, code generation) that allow objective ground-truth assessment. This enables us to distinguish harmful self-preference (favoring objectively worse responses) from legitimate self-preference (favoring genuinely superior ones). We conduct large-scale experiments under controlled evaluation conditions across diverse model families (e.g., Llama, Qwen, Gemma, Mistral, Phi, GPT, DeepSeek). Our findings reveal three key insights: (1) Better generators are better judges -- LLM evaluators' accuracy strongly correlates with their task performance, and much of the self-preference in capable models is legitimate. (2) Harmful self-preference persists, particularly when evaluator models perform poorly as generators on specific task instances. Stronger models exhibit more pronounced harmful bias when they err, though such incorrect generations are less frequent. (3) Inference-time scaling strategies, such as generating a long Chain-of-Thought before evaluation, effectively reduce the harmful self-preference. These results provide a more nuanced understanding of LLM-based evaluation and practical insights for improving its reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03822v1">Arti-"fickle" Intelligence: Using LLMs as a Tool for Inference in the Political and Social Sciences</a></div>
    <div class="paper-meta">
      📅 2025-04-04
    </div>
    <details class="paper-abstract">
      Generative large language models (LLMs) are incredibly useful, versatile, and promising tools. However, they will be of most use to political and social science researchers when they are used in a way that advances understanding about real human behaviors and concerns. To promote the scientific use of LLMs, we suggest that researchers in the political and social sciences need to remain focused on the scientific goal of inference. To this end, we discuss the challenges and opportunities related to scientific inference with LLMs, using validation of model output as an illustrative case for discussion. We propose a set of guidelines related to establishing the failure and success of LLMs when completing particular tasks, and discuss how we can make inferences from these observations. We conclude with a discussion of how this refocus will improve the accumulation of shared scientific knowledge about these tools and their uses in the social sciences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03814v1">Recursive Training Loops in LLMs: How training data properties modulate distribution shift in generated data?</a></div>
    <div class="paper-meta">
      📅 2025-04-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly contributing to the creation of content on the Internet. This creates a feedback loop as subsequent generations of models will be trained on this generated, synthetic data. This phenomenon is receiving increasing interest, in particular because previous studies have shown that it may lead to distribution shift - models misrepresent and forget the true underlying distributions of human data they are expected to approximate (e.g. resulting in a drastic loss of quality). In this study, we study the impact of human data properties on distribution shift dynamics in iterated training loops. We first confirm that the distribution shift dynamics greatly vary depending on the human data by comparing four datasets (two based on Twitter and two on Reddit). We then test whether data quality may influence the rate of this shift. We find that it does on the twitter, but not on the Reddit datasets. We then focus on a Reddit dataset and conduct a more exhaustive evaluation of a large set of dataset properties. This experiment associated lexical diversity with larger, and semantic diversity with smaller detrimental shifts, suggesting that incorporating text with high lexical (but limited semantic) diversity could exacerbate the degradation of generated text. We then focus on the evolution of political bias, and find that the type of shift observed (bias reduction, amplification or inversion) depends on the political lean of the human (true) distribution. Overall, our work extends the existing literature on the consequences of recursive fine-tuning by showing that this phenomenon is highly dependent on features of the human data on which training occurs. This suggests that different parts of internet (e.g. GitHub, Reddit) may undergo different types of shift depending on their properties.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03598v1">EnrichIndex: Using LLMs to Enrich Retrieval Indices Offline</a></div>
    <div class="paper-meta">
      📅 2025-04-04
      | 💬 Dataset and code are available at https://peterbaile.github.io/enrichindex/
    </div>
    <details class="paper-abstract">
      Existing information retrieval systems excel in cases where the language of target documents closely matches that of the user query. However, real-world retrieval systems are often required to implicitly reason whether a document is relevant. For example, when retrieving technical texts or tables, their relevance to the user query may be implied through a particular jargon or structure, rather than explicitly expressed in their content. Large language models (LLMs) hold great potential in identifying such implied relevance by leveraging their reasoning skills. Nevertheless, current LLM-augmented retrieval is hindered by high latency and computation cost, as the LLM typically computes the query-document relevance online, for every query anew. To tackle this issue we introduce EnrichIndex, a retrieval approach which instead uses the LLM offline to build semantically-enriched retrieval indices, by performing a single pass over all documents in the retrieval corpus once during ingestion time. Furthermore, the semantically-enriched indices can complement existing online retrieval approaches, boosting the performance of LLM re-rankers. We evaluated EnrichIndex on five retrieval tasks, involving passages and tables, and found that it outperforms strong online LLM-based retrieval systems, with an average improvement of 11.7 points in recall @ 10 and 10.6 points in NDCG @ 10 compared to strong baselines. In terms of online calls to the LLM, it processes 293.3 times fewer tokens which greatly reduces the online latency and cost. Overall, EnrichIndex is an effective way to build better retrieval indices offline by leveraging the strong reasoning skills of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03444v1">LLMSched: Uncertainty-Aware Workload Scheduling for Compound LLM Applications</a></div>
    <div class="paper-meta">
      📅 2025-04-04
    </div>
    <details class="paper-abstract">
      Developing compound Large Language Model (LLM) applications is becoming an increasingly prevalent approach to solving real-world problems. In these applications, an LLM collaborates with various external modules, including APIs and even other LLMs, to realize complex intelligent services. However, we reveal that the intrinsic duration and structural uncertainty in compound LLM applications pose great challenges for LLM service providers in serving and scheduling them efficiently. In this paper, we propose LLMSched, an uncertainty-aware scheduling framework for emerging compound LLM applications. In LLMSched, we first design a novel DAG-based model to describe the uncertain compound LLM applications. Then, we adopt the Bayesian network to comprehensively profile compound LLM applications and identify uncertainty-reducing stages, along with an entropy-based mechanism to quantify their uncertainty reduction. Combining an uncertainty reduction strategy and a job completion time (JCT)-efficient scheme, we further propose an efficient scheduler to reduce the average JCT. Evaluation of both simulation and testbed experiments on various representative compound LLM applications shows that compared to existing state-of-the-art scheduling schemes, LLMSched can reduce the average JCT by 14~79%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09893v2">RMB: Comprehensively Benchmarking Reward Models in LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-04-04
      | 💬 Accepted by ICLR2025
    </div>
    <details class="paper-abstract">
      Reward models (RMs) guide the alignment of large language models (LLMs), steering them toward behaviors preferred by humans. Evaluating RMs is the key to better aligning LLMs. However, the current evaluation of RMs may not directly correspond to their alignment performance due to the limited distribution of evaluation data and evaluation methods that are not closely related to alignment objectives. To address these limitations, we propose RMB, a comprehensive RM benchmark that covers over 49 real-world scenarios and includes both pairwise and Best-of-N (BoN) evaluations to better reflect the effectiveness of RMs in guiding alignment optimization. We demonstrate a positive correlation between our benchmark and the downstream alignment task performance. Based on our benchmark, we conduct extensive analysis on the state-of-the-art RMs, revealing their generalization defects that were not discovered by previous benchmarks, and highlighting the potential of generative RMs. Furthermore, we delve into open questions in reward models, specifically examining the effectiveness of majority voting for the evaluation of reward models and analyzing the impact factors of generative RMs, including the influence of evaluation criteria and instructing methods. Our evaluation code and datasets are available at https://github.com/Zhou-Zoey/RMB-Reward-Model-Benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02426v1">Narrative Studio: Visual narrative exploration using LLMs and Monte Carlo Tree Search</a></div>
    <div class="paper-meta">
      📅 2025-04-03
    </div>
    <details class="paper-abstract">
      Interactive storytelling benefits from planning and exploring multiple 'what if' scenarios. Modern LLMs are useful tools for ideation and exploration, but current chat-based user interfaces restrict users to a single linear flow. To address this limitation, we propose Narrative Studio -- a novel in-browser narrative exploration environment featuring a tree-like interface that allows branching exploration from user-defined points in a story. Each branch is extended via iterative LLM inference guided by system and user-defined prompts. Additionally, we employ Monte Carlo Tree Search (MCTS) to automatically expand promising narrative paths based on user-specified criteria, enabling more diverse and robust story development. We also allow users to enhance narrative coherence by grounding the generated text in an entity graph that represents the actors and environment of the story.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.15127v3">Pareto-Optimized Open-Source LLMs for Healthcare via Context Retrieval</a></div>
    <div class="paper-meta">
      📅 2025-04-03
      | 💬 14 pages, 3 figures, 5 tables, Accepted for publication at the 21st International Conference on Artificial Intelligence Applications and Innovations (AIAI 2025)
    </div>
    <details class="paper-abstract">
      This study leverages optimized context retrieval to enhance open-source Large Language Models (LLMs) for cost-effective, high performance healthcare AI. We demonstrate that this approach achieves state-of-the-art accuracy on medical question answering at a fraction of the cost of proprietary models, significantly improving the cost-accuracy Pareto frontier on the MedQA benchmark. Key contributions include: (1) OpenMedQA, a novel benchmark revealing a performance gap in open-ended medical QA compared to multiple-choice formats; (2) a practical, reproducible pipeline for context retrieval optimization; and (3) open-source resources (Prompt Engine, CoT/ToT/Thinking databases) to empower healthcare AI development. By advancing retrieval techniques and QA evaluation, we enable more affordable and reliable LLM solutions for healthcare.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02404v1">AnesBench: Multi-Dimensional Evaluation of LLM Reasoning in Anesthesiology</a></div>
    <div class="paper-meta">
      📅 2025-04-03
      | 💬 23 pages, 9 figures
    </div>
    <details class="paper-abstract">
      The application of large language models (LLMs) in the medical field has gained significant attention, yet their reasoning capabilities in more specialized domains like anesthesiology remain underexplored. In this paper, we systematically evaluate the reasoning capabilities of LLMs in anesthesiology and analyze key factors influencing their performance. To this end, we introduce AnesBench, a cross-lingual benchmark designed to assess anesthesiology-related reasoning across three levels: factual retrieval (System 1), hybrid reasoning (System 1.x), and complex decision-making (System 2). Through extensive experiments, we first explore how model characteristics, including model scale, Chain of Thought (CoT) length, and language transferability, affect reasoning performance. Then, we further evaluate the effectiveness of different training strategies, leveraging our curated anesthesiology-related dataset, including continuous pre-training (CPT) and supervised fine-tuning (SFT). Additionally, we also investigate how the test-time reasoning techniques, such as Best-of-N sampling and beam search, influence reasoning performance, and assess the impact of reasoning-enhanced model distillation, specifically DeepSeek-R1. We will publicly release AnesBench, along with our CPT and SFT training datasets and evaluation code at https://github.com/MiliLab/AnesBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02395v1">The quasi-semantic competence of LLMs: a case study on the part-whole relation</a></div>
    <div class="paper-meta">
      📅 2025-04-03
    </div>
    <details class="paper-abstract">
      Understanding the extent and depth of the semantic competence of \emph{Large Language Models} (LLMs) is at the center of the current scientific agenda in Artificial Intelligence (AI) and Computational Linguistics (CL). We contribute to this endeavor by investigating their knowledge of the \emph{part-whole} relation, a.k.a. \emph{meronymy}, which plays a crucial role in lexical organization, but it is significantly understudied. We used data from ConceptNet relations \citep{speer2016conceptnet} and human-generated semantic feature norms \citep{McRae:2005} to explore the abilities of LLMs to deal with \textit{part-whole} relations. We employed several methods based on three levels of analysis: i.) \textbf{behavioral} testing via prompting, where we directly queried the models on their knowledge of meronymy, ii.) sentence \textbf{probability} scoring, where we tested models' abilities to discriminate correct (real) and incorrect (asymmetric counterfactual) \textit{part-whole} relations, and iii.) \textbf{concept representation} analysis in vector space, where we proved the linear organization of the \textit{part-whole} concept in the embedding and unembedding spaces. These analyses present a complex picture that reveals that the LLMs' knowledge of this relation is only partial. They have just a ``\emph{quasi}-semantic'' competence and still fall short of capturing deep inferential properties.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.17233v3">ICPL: Few-shot In-context Preference Learning via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-03
    </div>
    <details class="paper-abstract">
      Preference-based reinforcement learning is an effective way to handle tasks where rewards are hard to specify but can be exceedingly inefficient as preference learning is often tabula rasa. We demonstrate that Large Language Models (LLMs) have native preference-learning capabilities that allow them to achieve sample-efficient preference learning, addressing this challenge. We propose In-Context Preference Learning (ICPL), which uses in-context learning capabilities of LLMs to reduce human query inefficiency. ICPL uses the task description and basic environment code to create sets of reward functions which are iteratively refined by placing human feedback over videos of the resultant policies into the context of an LLM and then requesting better rewards. We first demonstrate ICPL's effectiveness through a synthetic preference study, providing quantitative evidence that it significantly outperforms baseline preference-based methods with much higher performance and orders of magnitude greater efficiency. We observe that these improvements are not solely coming from LLM grounding in the task but that the quality of the rewards improves over time, indicating preference learning capabilities. Additionally, we perform a series of real human preference-learning trials and observe that ICPL extends beyond synthetic settings and can work effectively with humans-in-the-loop.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02343v1">Toward General and Robust LLM-enhanced Text-attributed Graph Learning</a></div>
    <div class="paper-meta">
      📅 2025-04-03
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) and the proliferation of Text-Attributed Graphs (TAGs) across various domains have positioned LLM-enhanced TAG learning as a critical research area. By utilizing rich graph descriptions, this paradigm leverages LLMs to generate high-quality embeddings, thereby enhancing the representational capacity of Graph Neural Networks (GNNs). However, the field faces significant challenges: (1) the absence of a unified framework to systematize the diverse optimization perspectives arising from the complex interactions between LLMs and GNNs, and (2) the lack of a robust method capable of handling real-world TAGs, which often suffer from texts and edge sparsity, leading to suboptimal performance. To address these challenges, we propose UltraTAG, a unified pipeline for LLM-enhanced TAG learning. UltraTAG provides a unified comprehensive and domain-adaptive framework that not only organizes existing methodologies but also paves the way for future advancements in the field. Building on this framework, we propose UltraTAG-S, a robust instantiation of UltraTAG designed to tackle the inherent sparsity issues in real-world TAGs. UltraTAG-S employs LLM-based text propagation and text augmentation to mitigate text sparsity, while leveraging LLM-augmented node selection techniques based on PageRank and edge reconfiguration strategies to address edge sparsity. Our extensive experiments demonstrate that UltraTAG-S significantly outperforms existing baselines, achieving improvements of 2.12\% and 17.47\% in ideal and sparse settings, respectively. Moreover, as the data sparsity ratio increases, the performance improvement of UltraTAG-S also rises, which underscores the effectiveness and robustness of UltraTAG-S.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.17867v2">Evaluating and Enhancing LLMs for Multi-turn Text-to-SQL with Multiple Question Types</a></div>
    <div class="paper-meta">
      📅 2025-04-03
      | 💬 International Joint Conference on Neural Networks 2025 (IJCNN 2025)
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have significantly advanced text-to-SQL systems. However, most LLM-based methods often narrowly focus on SQL generation, neglecting the complexities of real-world conversational queries. This oversight can lead to unreliable responses, particularly for ambiguous questions that cannot be directly addressed with SQL. To bridge this gap, we propose MMSQL, a comprehensive test suite designed to evaluate the question classification and SQL generation capabilities of LLMs by simulating real-world scenarios with diverse question types and multi-turn Q\&A interactions. Using MMSQL, we assessed the performance of popular LLMs, including both open-source and closed-source models, and identified key factors impacting their performance in such scenarios. Moreover, we introduce an LLM-based multi-agent framework that employs specialized agents to identify question types and determine appropriate answering strategies. Our experiments demonstrate that this approach significantly enhances the model's ability to navigate the complexities of conversational dynamics, effectively handling the diverse and complex nature of user queries. Our dataset and code are publicly available at https://mcxiaoxiao.github.io/MMSQL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.22512v2">Unlocking LLM Repair Capabilities in Low-Resource Programming Languages Through Cross-Language Translation and Multi-Agent Refinement</a></div>
    <div class="paper-meta">
      📅 2025-04-03
    </div>
    <details class="paper-abstract">
      Recent advances in leveraging LLMs for APR have demonstrated impressive capabilities in fixing software defects. However, current LLM-based approaches predominantly focus on mainstream programming languages like Java and Python, neglecting less prevalent but emerging languages such as Rust due to expensive training resources, limited datasets, and insufficient community support. This narrow focus creates a significant gap in repair capabilities across the programming language spectrum, where the full potential of LLMs for comprehensive multilingual program repair remains largely unexplored. To address this limitation, we introduce a novel cross-language program repair approach LANTERN that leverages LLMs' differential proficiency across languages through a multi-agent iterative repair paradigm. Our technique strategically translates defective code from languages where LLMs exhibit weaker repair capabilities to languages where they demonstrate stronger performance, without requiring additional training. A key innovation of our approach is an LLM-based decision-making system that dynamically selects optimal target languages based on bug characteristics and continuously incorporates feedback from previous repair attempts. We evaluate our method on xCodeEval, a comprehensive multilingual benchmark comprising 5,068 bugs across 11 programming languages. Results demonstrate significant enhancement in repair effectiveness, particularly for underrepresented languages, with Rust showing a 22.09% improvement in Pass@10 metrics. Our research provides the first empirical evidence that cross-language translation significantly expands the repair capabilities of LLMs and effectively bridges the performance gap between programming languages with different levels of popularity, opening new avenues for truly language-agnostic automated program repair.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02304v1">Measurement of LLM's Philosophies of Human Nature</a></div>
    <div class="paper-meta">
      📅 2025-04-03
    </div>
    <details class="paper-abstract">
      The widespread application of artificial intelligence (AI) in various tasks, along with frequent reports of conflicts or violations involving AI, has sparked societal concerns about interactions with AI systems. Based on Wrightsman's Philosophies of Human Nature Scale (PHNS), a scale empirically validated over decades to effectively assess individuals' attitudes toward human nature, we design the standardized psychological scale specifically targeting large language models (LLM), named the Machine-based Philosophies of Human Nature Scale (M-PHNS). By evaluating LLMs' attitudes toward human nature across six dimensions, we reveal that current LLMs exhibit a systemic lack of trust in humans, and there is a significant negative correlation between the model's intelligence level and its trust in humans. Furthermore, we propose a mental loop learning framework, which enables LLM to continuously optimize its value system during virtual interactions by constructing moral scenarios, thereby improving its attitude toward human nature. Experiments demonstrate that mental loop learning significantly enhances their trust in humans compared to persona or instruction prompts. This finding highlights the potential of human-based psychological assessments for LLM, which can not only diagnose cognitive biases but also provide a potential solution for ethical learning in artificial intelligence. We release the M-PHNS evaluation code and data at https://github.com/kodenii/M-PHNS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02280v1">LLM-Guided Evolution: An Autonomous Model Optimization for Object Detection</a></div>
    <div class="paper-meta">
      📅 2025-04-03
    </div>
    <details class="paper-abstract">
      In machine learning, Neural Architecture Search (NAS) requires domain knowledge of model design and a large amount of trial-and-error to achieve promising performance. Meanwhile, evolutionary algorithms have traditionally relied on fixed rules and pre-defined building blocks. The Large Language Model (LLM)-Guided Evolution (GE) framework transformed this approach by incorporating LLMs to directly modify model source code for image classification algorithms on CIFAR data and intelligently guide mutations and crossovers. A key element of LLM-GE is the "Evolution of Thought" (EoT) technique, which establishes feedback loops, allowing LLMs to refine their decisions iteratively based on how previous operations performed. In this study, we perform NAS for object detection by improving LLM-GE to modify the architecture of You Only Look Once (YOLO) models to enhance performance on the KITTI dataset. Our approach intelligently adjusts the design and settings of YOLO to find the optimal algorithms against objective such as detection accuracy and speed. We show that LLM-GE produced variants with significant performance improvements, such as an increase in Mean Average Precision from 92.5% to 94.5%. This result highlights the flexibility and effectiveness of LLM-GE on real-world challenges, offering a novel paradigm for automated machine learning that combines LLM-driven reasoning with evolutionary strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02254v1">LLMs as Deceptive Agents: How Role-Based Prompting Induces Semantic Ambiguity in Puzzle Tasks</a></div>
    <div class="paper-meta">
      📅 2025-04-03
      | 💬 9 pages, 5 figures, 1 table
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have not only showcased impressive creative capabilities but also revealed emerging agentic behaviors that exploit linguistic ambiguity in adversarial settings. In this study, we investigate how an LLM, acting as an autonomous agent, leverages semantic ambiguity to generate deceptive puzzles that mislead and challenge human users. Inspired by the popular puzzle game "Connections", we systematically compare puzzles produced through zero-shot prompting, role-injected adversarial prompts, and human-crafted examples, with an emphasis on understanding the underlying agent decision-making processes. Employing computational analyses with HateBERT to quantify semantic ambiguity, alongside subjective human evaluations, we demonstrate that explicit adversarial agent behaviors significantly heighten semantic ambiguity -- thereby increasing cognitive load and reducing fairness in puzzle solving. These findings provide critical insights into the emergent agentic qualities of LLMs and underscore important ethical considerations for evaluating and safely deploying autonomous language systems in both educational technologies and entertainment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02234v1">LLM Social Simulations Are a Promising Research Method</a></div>
    <div class="paper-meta">
      📅 2025-04-03
    </div>
    <details class="paper-abstract">
      Accurate and verifiable large language model (LLM) simulations of human research subjects promise an accessible data source for understanding human behavior and training new AI systems. However, results to date have been limited, and few social scientists have adopted these methods. In this position paper, we argue that the promise of LLM social simulations can be achieved by addressing five tractable challenges. We ground our argument in a literature survey of empirical comparisons between LLMs and human research subjects, commentaries on the topic, and related work. We identify promising directions with prompting, fine-tuning, and complementary methods. We believe that LLM social simulations can already be used for exploratory research, such as pilot experiments for psychology, economics, sociology, and marketing. More widespread use may soon be possible with rapidly advancing LLM capabilities, and researchers should prioritize developing conceptual models and evaluations that can be iteratively deployed and refined at pace with ongoing AI advances.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00070v2">Can AI Solve the Peer Review Crisis? A Large Scale Cross Model Experiment of LLMs' Performance and Biases in Evaluating over 1000 Economics Papers</a></div>
    <div class="paper-meta">
      📅 2025-04-03
      | 💬 58 pages
    </div>
    <details class="paper-abstract">
      This study examines the potential of large language models (LLMs) to augment the academic peer review process by reliably evaluating the quality of economics research without introducing systematic bias. We conduct one of the first large-scale experimental assessments of four LLMs (GPT-4o, Claude 3.5, Gemma 3, and LLaMA 3.3) across two complementary experiments. In the first, we use nonparametric binscatter and linear regression techniques to analyze over 29,000 evaluations of 1,220 anonymized papers drawn from 110 economics journals excluded from the training data of current LLMs, along with a set of AI-generated submissions. The results show that LLMs consistently distinguish between higher- and lower-quality research based solely on textual content, producing quality gradients that closely align with established journal prestige measures. Claude and Gemma perform exceptionally well in capturing these gradients, while GPT excels in detecting AI-generated content. The second experiment comprises 8,910 evaluations designed to assess whether LLMs replicate human like biases in single blind reviews. By systematically varying author gender, institutional affiliation, and academic prominence across 330 papers, we find that GPT, Gemma, and LLaMA assign significantly higher ratings to submissions from top male authors and elite institutions relative to the same papers presented anonymously. These results emphasize the importance of excluding author-identifying information when deploying LLMs in editorial screening. Overall, our findings provide compelling evidence and practical guidance for integrating LLMs into peer review to enhance efficiency, improve accuracy, and promote equity in the publication process of economics research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02195v1">LLM-Augmented Graph Neural Recommenders: Integrating User Reviews</a></div>
    <div class="paper-meta">
      📅 2025-04-03
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Recommender systems increasingly aim to combine signals from both user reviews and purchase (or other interaction) behaviors. While user-written comments provide explicit insights about preferences, merging these textual representations from large language models (LLMs) with graph-based embeddings of user actions remains a challenging task. In this work, we propose a framework that employs both a Graph Neural Network (GNN)-based model and an LLM to produce review-aware representations, preserving review semantics while mitigating textual noise. Our approach utilizes a hybrid objective that balances user-item interactions against text-derived features, ensuring that user's both behavioral and linguistic signals are effectively captured. We evaluate this method on multiple datasets from diverse application domains, demonstrating consistent improvements over a baseline GNN-based recommender model. Notably, our model achieves significant gains in recommendation accuracy when review data is sparse or unevenly distributed. These findings highlight the importance of integrating LLM-driven textual feedback with GNN-derived user behavioral patterns to develop robust, context-aware recommender systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14729v2">PROMPTFUZZ: Harnessing Fuzzing Techniques for Robust Testing of Prompt Injection in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-03
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have gained widespread use in various applications due to their powerful capability to generate human-like text. However, prompt injection attacks, which involve overwriting a model's original instructions with malicious prompts to manipulate the generated text, have raised significant concerns about the security and reliability of LLMs. Ensuring that LLMs are robust against such attacks is crucial for their deployment in real-world applications, particularly in critical tasks. In this paper, we propose PROMPTFUZZ, a novel testing framework that leverages fuzzing techniques to systematically assess the robustness of LLMs against prompt injection attacks. Inspired by software fuzzing, PROMPTFUZZ selects promising seed prompts and generates a diverse set of prompt injections to evaluate the target LLM's resilience. PROMPTFUZZ operates in two stages: the prepare phase, which involves selecting promising initial seeds and collecting few-shot examples, and the focus phase, which uses the collected examples to generate diverse, high-quality prompt injections. Using PROMPTFUZZ, we can uncover more vulnerabilities in LLMs, even those with strong defense prompts. By deploying the generated attack prompts from PROMPTFUZZ in a real-world competition, we achieved the 7th ranking out of over 4000 participants (top 0.14%) within 2 hours. Additionally, we construct a dataset to fine-tune LLMs for enhanced robustness against prompt injection attacks. While the fine-tuned model shows improved robustness, PROMPTFUZZ continues to identify vulnerabilities, highlighting the importance of robust testing for LLMs. Our work emphasizes the critical need for effective testing tools and provides a practical framework for evaluating and improving the robustness of LLMs against prompt injection attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13107v2">MatterChat: A Multi-Modal LLM for Material Science</a></div>
    <div class="paper-meta">
      📅 2025-04-03
    </div>
    <details class="paper-abstract">
      Understanding and predicting the properties of inorganic materials is crucial for accelerating advancements in materials science and driving applications in energy, electronics, and beyond. Integrating material structure data with language-based information through multi-modal large language models (LLMs) offers great potential to support these efforts by enhancing human-AI interaction. However, a key challenge lies in integrating atomic structures at full resolution into LLMs. In this work, we introduce MatterChat, a versatile structure-aware multi-modal LLM that unifies material structural data and textual inputs into a single cohesive model. MatterChat employs a bridging module to effectively align a pretrained machine learning interatomic potential with a pretrained LLM, reducing training costs and enhancing flexibility. Our results demonstrate that MatterChat significantly improves performance in material property prediction and human-AI interaction, surpassing general-purpose LLMs such as GPT-4. We also demonstrate its usefulness in applications such as more advanced scientific reasoning and step-by-step material synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14202v2">Do LLMs Consider Security? An Empirical Study on Responses to Programming Questions</a></div>
    <div class="paper-meta">
      📅 2025-04-03
      | 💬 Accepted to EMSE
    </div>
    <details class="paper-abstract">
      The widespread adoption of conversational LLMs for software development has raised new security concerns regarding the safety of LLM-generated content. Our motivational study outlines ChatGPT's potential in volunteering context-specific information to the developers, promoting safe coding practices. Motivated by this finding, we conduct a study to evaluate the degree of security awareness exhibited by three prominent LLMs: Claude 3, GPT-4, and Llama 3. We prompt these LLMs with Stack Overflow questions that contain vulnerable code to evaluate whether they merely provide answers to the questions or if they also warn users about the insecure code, thereby demonstrating a degree of security awareness. Further, we assess whether LLM responses provide information about the causes, exploits, and the potential fixes of the vulnerability, to help raise users' awareness. Our findings show that all three models struggle to accurately detect and warn users about vulnerabilities, achieving a detection rate of only 12.6% to 40% across our datasets. We also observe that the LLMs tend to identify certain types of vulnerabilities related to sensitive information exposure and improper input neutralization much more frequently than other types, such as those involving external control of file names or paths. Furthermore, when LLMs do issue security warnings, they often provide more information on the causes, exploits, and fixes of vulnerabilities compared to Stack Overflow responses. Finally, we provide an in-depth discussion on the implications of our findings and present a CLI-based prompting tool that can be used to generate significantly more secure LLM responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03048v1">LLM Library Learning Fails: A LEGO-Prover Case Study</a></div>
    <div class="paper-meta">
      📅 2025-04-03
      | 💬 24 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Recent advancements in the coding, reasoning, and tool-using abilities of LLMs have spurred interest in library learning (i.e., online learning through the creation, storage, and retrieval of reusable and composable functions, knowledge, checklists, or lemmas). Such systems often promise improved task performance through the automatic creation of broadly applicable tools, as well as superior computational performance through the caching of reasoning (i.e., the storage of generated tools). However, we find strong reason to be skeptical. We perform a deep dive into one such system, LEGO-Prover, which purports to learn reusable lemmas for mathematical reasoning. We find no evidence of the direct reuse of learned lemmas, and find evidence against the soft reuse of learned lemmas (i.e., reuse by modifying relevant examples). Crucially, we find that LEGO-Prover does not in fact improve over the simple baseline of prompting the model - the improvements in task accuracy vanish once computational cost is accounted for. Our findings suggest that serious misconceptions exist as to the effectiveness of these techniques, that a serious re-examination of the state of LLM-based library learning is required, and that we require much stronger standards for evaluation including behavioural analysis and ensuring that an equal computational budget is used for baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03015v1">AuDeRe: Automated Strategy Decision and Realization in Robot Planning and Control via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-03
      | 💬 8 pages, 14 figures, submitted for CDC 2025 invited session on Large Language Models (LLMs) and Control
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have shown significant promise in various domains, especially robotics. However, most prior LLM-based work in robotic applications either directly predicts waypoints or applies LLMs within fixed tool integration frameworks, offering limited flexibility in exploring and configuring solutions best suited to different tasks. In this work, we propose a framework that leverages LLMs to select appropriate planning and control strategies based on task descriptions, environmental constraints, and system dynamics. These strategies are then executed by calling the available comprehensive planning and control APIs. Our approach employs iterative LLM-based reasoning with performance feedback to refine the algorithm selection. We validate our approach through extensive experiments across tasks of varying complexity, from simple tracking to complex planning scenarios involving spatiotemporal constraints. The results demonstrate that using LLMs to determine planning and control strategies from natural language descriptions significantly enhances robotic autonomy while reducing the need for extensive manual tuning and expert knowledge. Furthermore, our framework maintains generalizability across different tasks and notably outperforms baseline methods that rely on LLMs for direct trajectory, control sequence, or code generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.20052v3">Understanding and Mitigating Language Confusion in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-03
      | 💬 EMNLP 2024 Main Conference Camera-ready. v3: hi, ru not run for monolingual Okapi
    </div>
    <details class="paper-abstract">
      We investigate a surprising limitation of LLMs: their inability to consistently generate text in a user's desired language. We create the Language Confusion Benchmark (LCB) to evaluate such failures, covering 15 typologically diverse languages with existing and newly-created English and multilingual prompts. We evaluate a range of LLMs on monolingual and cross-lingual generation reflecting practical use cases, finding that Llama Instruct and Mistral models exhibit high degrees of language confusion and even the strongest models fail to consistently respond in the correct language. We observe that base and English-centric instruct models are more prone to language confusion, which is aggravated by complex prompts and high sampling temperatures. We find that language confusion can be partially mitigated via few-shot prompting, multilingual SFT and preference tuning. We release our language confusion benchmark, which serves as a first layer of efficient, scalable multilingual evaluation at https://github.com/for-ai/language-confusion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02904v1">How Post-Training Reshapes LLMs: A Mechanistic View on Knowledge, Truthfulness, Refusal, and Confidence</a></div>
    <div class="paper-meta">
      📅 2025-04-03
    </div>
    <details class="paper-abstract">
      Post-training is essential for the success of large language models (LLMs), transforming pre-trained base models into more useful and aligned post-trained models. While plenty of works have studied post-training algorithms and evaluated post-training models by their outputs, it remains understudied how post-training reshapes LLMs internally. In this paper, we compare base and post-trained LLMs mechanistically from four perspectives to better understand post-training effects. Our findings across model families and datasets reveal that: (1) Post-training does not change the factual knowledge storage locations, and it adapts knowledge representations from the base model while developing new knowledge representations; (2) Both truthfulness and refusal can be represented by linear vectors in the hidden representation space. The truthfulness direction is highly similar between the base and post-trained model, and it is effectively transferable for interventions; (3) The refusal direction is different between the base and post-trained models, and it shows limited forward transferability; (4) Differences in confidence between the base and post-trained models cannot be attributed to entropy neurons. Our study provides insights into the fundamental mechanisms preserved and altered during post-training, facilitates downstream tasks like model steering, and could potentially benefit future research in interpretability and LLM post-training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02901v1">Hide and Seek in Noise Labels: Noise-Robust Collaborative Active Learning with LLM-Powered Assistance</a></div>
    <div class="paper-meta">
      📅 2025-04-03
    </div>
    <details class="paper-abstract">
      Learning from noisy labels (LNL) is a challenge that arises in many real-world scenarios where collected training data can contain incorrect or corrupted labels. Most existing solutions identify noisy labels and adopt active learning to query human experts on them for denoising. In the era of large language models (LLMs), although we can reduce the human effort to improve these methods, their performances are still subject to accurately separating the clean and noisy samples from noisy data. In this paper, we propose an innovative collaborative learning framework NoiseAL based on active learning to combine LLMs and small models (SMs) for learning from noisy labels. During collaborative training, we first adopt two SMs to form a co-prediction network and propose a dynamic-enhanced threshold strategy to divide the noisy data into different subsets, then select the clean and noisy samples from these subsets to feed the active annotator LLMs to rectify noisy samples. Finally, we employ different optimization objectives to conquer subsets with different degrees of label noises. Extensive experiments on synthetic and real-world noise datasets further demonstrate the superiority of our framework over state-of-the-art baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03789v1">Steve: LLM Powered ChatBot for Career Progression</a></div>
    <div class="paper-meta">
      📅 2025-04-03
    </div>
    <details class="paper-abstract">
      The advancements in systems deploying large language models (LLMs), as well as improvements in their ability to act as agents with predefined templates, provide an opportunity to conduct qualitative, individualized assessments, creating a bridge between qualitative and quantitative methods for candidates seeking career progression. In this paper, we develop a platform that allows candidates to run AI-led interviews to assess their current career stage and curate coursework to enable progression to the next level. Our approach incorporates predefined career trajectories, associated skills, and a method to recommend the best resources for gaining the necessary skills for advancement. We employ OpenAI API calls along with expertly compiled chat templates to assess candidate competence. Our platform is highly configurable due to the modularity of the development, is easy to deploy and use, and available as a web interface where the only requirement is candidate resumes in PDF format. We demonstrate a use-case centered on software engineering and intend to extend this platform to be domain-agnostic, requiring only regular updates to chat templates as industries evolve.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02789v1">A Framework for Robust Cognitive Evaluation of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-03
    </div>
    <details class="paper-abstract">
      Emergent cognitive abilities in large language models (LLMs) have been widely observed, but their nature and underlying mechanisms remain poorly understood. A growing body of research draws on cognitive science to investigate LLM cognition, but standard methodologies and experimen-tal pipelines have not yet been established. To address this gap we develop CognitivEval, a framework for systematically evaluating the artificial cognitive capabilities of LLMs, with a particular emphasis on robustness in response collection. The key features of CognitivEval include: (i) automatic prompt permutations, and (ii) testing that gathers both generations and model probability estimates. Our experiments demonstrate that these features lead to more robust experimental outcomes. Using CognitivEval, we replicate five classic experiments in cognitive science, illustrating the framework's generalizability across various experimental tasks and obtaining a cognitive profile of several state of the art LLMs. CognitivEval will be released publicly to foster broader collaboration within the cognitive science community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.16879v2">GRACE: Generating Socially Appropriate Robot Actions Leveraging LLMs and Human Explanations</a></div>
    <div class="paper-meta">
      📅 2025-04-03
      | 💬 2025 IEEE International Conference on Robotics & Automation (ICRA), Supplementary video: https://youtu.be/GTNCC1GkiQ4
    </div>
    <details class="paper-abstract">
      When operating in human environments, robots need to handle complex tasks while both adhering to social norms and accommodating individual preferences. For instance, based on common sense knowledge, a household robot can predict that it should avoid vacuuming during a social gathering, but it may still be uncertain whether it should vacuum before or after having guests. In such cases, integrating common-sense knowledge with human preferences, often conveyed through human explanations, is fundamental yet a challenge for existing systems. In this paper, we introduce GRACE, a novel approach addressing this while generating socially appropriate robot actions. GRACE leverages common sense knowledge from LLMs, and it integrates this knowledge with human explanations through a generative network. The bidirectional structure of GRACE enables robots to refine and enhance LLM predictions by utilizing human explanations and makes robots capable of generating such explanations for human-specified actions. Our evaluations show that integrating human explanations boosts GRACE's performance, where it outperforms several baselines and provides sensible explanations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02779v1">BT-ACTION: A Test-Driven Approach for Modular Understanding of User Instruction Leveraging Behaviour Trees and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-03
    </div>
    <details class="paper-abstract">
      Natural language instructions are often abstract and complex, requiring robots to execute multiple subtasks even for seemingly simple queries. For example, when a user asks a robot to prepare avocado toast, the task involves several sequential steps. Moreover, such instructions can be ambiguous or infeasible for the robot or may exceed the robot's existing knowledge. While Large Language Models (LLMs) offer strong language reasoning capabilities to handle these challenges, effectively integrating them into robotic systems remains a key challenge. To address this, we propose BT-ACTION, a test-driven approach that combines the modular structure of Behavior Trees (BT) with LLMs to generate coherent sequences of robot actions for following complex user instructions, specifically in the context of preparing recipes in a kitchen-assistance setting. We evaluated BT-ACTION in a comprehensive user study with 45 participants, comparing its performance to direct LLM prompting. Results demonstrate that the modular design of BT-ACTION helped the robot make fewer mistakes and increased user trust, and participants showed a significant preference for the robot leveraging BT-ACTION. The code is publicly available at https://github.com/1Eggbert7/BT_LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02733v1">Enhancing LLM Robustness to Perturbed Instructions: An Empirical Study</a></div>
    <div class="paper-meta">
      📅 2025-04-03
      | 💬 Building Trust Workshop, ICLR 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are highly vulnerable to input perturbations, as even a small prompt change may result in a substantially different output. Existing methods to enhance LLM robustness are primarily focused on perturbed data samples, whereas improving resiliency to perturbations of task-level instructions has remained relatively underexplored. In this work, we focus on character- and word-level edits of task-specific instructions, which substantially degrade downstream performance. We experiment with a variety of techniques to enhance the robustness of LLMs, including self-denoising and representation alignment, testing different models (Llama 3 and Flan-T5), datasets (CoLA, QNLI, SST-2) and instructions (both task-oriented and role-oriented). We find that, on average, self-denoising -- whether performed by a frozen LLM or a fine-tuned model -- achieves substantially higher performance gains than alternative strategies, including more complex baselines such as ensembling and supervised methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02732v1">Why do LLMs attend to the first token?</a></div>
    <div class="paper-meta">
      📅 2025-04-03
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) tend to attend heavily to the first token in the sequence -- creating a so-called attention sink. Many works have studied this phenomenon in detail, proposing various ways to either leverage or alleviate it. Attention sinks have been connected to quantisation difficulties, security issues, and streaming attention. Yet, while many works have provided conditions in which they occur or not, a critical question remains shallowly answered: Why do LLMs learn such patterns and how are they being used? In this work, we argue theoretically and empirically that this mechanism provides a method for LLMs to avoid over-mixing, connecting this to existing lines of work that study mathematically how information propagates in Transformers. We conduct experiments to validate our theoretical intuitions and show how choices such as context length, depth, and data packing influence the sink behaviour. We hope that this study provides a new practical perspective on why attention sinks are useful in LLMs, leading to a better understanding of the attention patterns that form during training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02708v1">The Hidden Space of Safety: Understanding Preference-Tuned LLMs in Multilingual context</a></div>
    <div class="paper-meta">
      📅 2025-04-03
      | 💬 14 pages, 11 Figures, 2 Tables, currently under review at ACL 2025
    </div>
    <details class="paper-abstract">
      Alignment tuning has enabled large language models to excel in reasoning, instruction-following, and minimizing harmful generations. However, despite their widespread deployment, these models exhibit a monolingual bias, raising concerns about the effectiveness of alignment across languages. Current alignment methods predominantly focus on English, leaving it unclear how alignment mechanism generalize to multilingual settings. To address this, we conduct a systematic analysis of distributional shifts in the embedding space of LLMs before and after alignment, uncovering its impact on model behavior across diverse languages. We leverage the alignment-induced separation in safety space as a quantitative tool to measure how alignment enforces safety constraints. Our study evaluates seven LLMs using balanced toxicity datasets and parallel text-detoxification benchmarks, revealing substantial disparities in the latent representation space between high-resource and low-resource languages. These findings underscore the need for language-specific fine-tuning to ensure fair, reliable and robust multilingual alignment. Our insights provide a foundation for developing truly safe multilingual LLMs, emphasizing the urgency of addressing alignment gaps in underrepresented languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02671v1">LLM for Complex Reasoning Task: An Exploratory Study in Fermi Problems</a></div>
    <div class="paper-meta">
      📅 2025-04-03
      | 💬 7 pages,7 tables, 5 figures
    </div>
    <details class="paper-abstract">
      Fermi Problems (FPs) are mathematical reasoning tasks that require human-like logic and numerical reasoning. Unlike other reasoning questions, FPs often involve real-world impracticalities or ambiguous concepts, making them challenging even for humans to solve. Despite advancements in AI, particularly with large language models (LLMs) in various reasoning tasks, FPs remain relatively under-explored. This work conducted an exploratory study to examine the capabilities and limitations of LLMs in solving FPs. We first evaluated the overall performance of three advanced LLMs using a publicly available FP dataset. We designed prompts according to the recently proposed TELeR taxonomy, including a zero-shot scenario. Results indicated that all three LLMs achieved a fp_score (range between 0 - 1) below 0.5, underscoring the inherent difficulty of these reasoning tasks. To further investigate, we categorized FPs into standard and specific questions, hypothesizing that LLMs would perform better on standard questions, which are characterized by clarity and conciseness, than on specific ones. Comparative experiments confirmed this hypothesis, demonstrating that LLMs performed better on standard FPs in terms of both accuracy and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02623v1">Multi-Mission Tool Bench: Assessing the Robustness of LLM based Agents through Related and Dynamic Missions</a></div>
    <div class="paper-meta">
      📅 2025-04-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate strong potential as agents for tool invocation due to their advanced comprehension and planning capabilities. Users increasingly rely on LLM-based agents to solve complex missions through iterative interactions. However, existing benchmarks predominantly access agents in single-mission scenarios, failing to capture real-world complexity. To bridge this gap, we propose the Multi-Mission Tool Bench. In the benchmark, each test case comprises multiple interrelated missions. This design requires agents to dynamically adapt to evolving demands. Moreover, the proposed benchmark explores all possible mission-switching patterns within a fixed mission number. Specifically, we propose a multi-agent data generation framework to construct the benchmark. We also propose a novel method to evaluate the accuracy and efficiency of agent decisions with dynamic decision trees. Experiments on diverse open-source and closed-source LLMs reveal critical factors influencing agent robustness and provide actionable insights to the tool invocation society.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02622v1">Exploring undercurrents of learning tensions in an LLM-enhanced landscape: A student-centered qualitative perspective on LLM vs Search</a></div>
    <div class="paper-meta">
      📅 2025-04-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are transforming how students learn by providing readily available tools that can quickly augment or complete various learning activities with non-trivial performance. Similar paradigm shifts have occurred in the past with the introduction of search engines and Wikipedia, which replaced or supplemented traditional information sources such as libraries and books. This study investigates the potential for LLMs to represent the next shift in learning, focusing on their role in information discovery and synthesis compared to existing technologies, such as search engines. Using a within-subjects, counterbalanced design, participants learned new topics using a search engine (Google) and an LLM (ChatGPT). Post-task follow-up interviews explored students' reflections, preferences, pain points, and overall perceptions. We present analysis of their responses that show nuanced insights into when, why, and how students prefer LLMs over search engines, offering implications for educators, policymakers, and technology developers navigating the evolving educational landscape.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01380v2">Efficient LLM Inference using Dynamic Input Pruning and Cache-Aware Masking</a></div>
    <div class="paper-meta">
      📅 2025-04-03
      | 💬 Main Text: 10 pages, 11 figures. Appendix: 6 pages, 3 figures
    </div>
    <details class="paper-abstract">
      While mobile devices provide ever more compute power, improvements in DRAM bandwidth are much slower. This is unfortunate for large language model (LLM) token generation, which is heavily memory-bound. Previous work has proposed to leverage natural dynamic activation sparsity in ReLU-activated LLMs to reduce effective DRAM bandwidth per token. However, more recent LLMs use SwiGLU instead of ReLU, which results in little inherent sparsity. While SwiGLU activations can be pruned based on magnitude, the resulting sparsity patterns are difficult to predict, rendering previous approaches ineffective. To circumvent this issue, our work introduces Dynamic Input Pruning (DIP): a predictor-free dynamic sparsification approach, which preserves accuracy with minimal fine-tuning. DIP can further use lightweight LoRA adapters to regain some performance lost during sparsification. Lastly, we describe a novel cache-aware masking strategy, which considers the cache state and activation magnitude to further increase cache hit rate, improving LLM token rate on mobile devices. DIP outperforms other methods in terms of accuracy, memory and throughput trade-offs across simulated hardware settings. On Phi-3-Medium, DIP achieves a 46\% reduction in memory and 40\% increase in throughput with $<$ 0.1 loss in perplexity when compared to streaming the dense model from Flash. The open source code for HW simulator, methods, and experiments in this paper is available at https://github.com/Qualcomm-AI-research/dynamic-sparsity .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02559v1">Leveraging LLM For Synchronizing Information Across Multilingual Tables</a></div>
    <div class="paper-meta">
      📅 2025-04-03
      | 💬 17 Pages, 11 Tables, 2 Figures
    </div>
    <details class="paper-abstract">
      The vast amount of online information today poses challenges for non-English speakers, as much of it is concentrated in high-resource languages such as English and French. Wikipedia reflects this imbalance, with content in low-resource languages frequently outdated or incomplete. Recent research has sought to improve cross-language synchronization of Wikipedia tables using rule-based methods. These approaches can be effective, but they struggle with complexity and generalization. This paper explores large language models (LLMs) for multilingual information synchronization, using zero-shot prompting as a scalable solution. We introduce the Information Updation dataset, simulating the real-world process of updating outdated Wikipedia tables, and evaluate LLM performance. Our findings reveal that single-prompt approaches often produce suboptimal results, prompting us to introduce a task decomposition strategy that enhances coherence and accuracy. Our proposed method outperforms existing baselines, particularly in Information Updation (1.79%) and Information Addition (20.58%), highlighting the model strength in dynamically updating and enriching data across architectures
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02553v1">Exploring Individual Factors in the Adoption of LLMs for Specific Software Engineering Tasks</a></div>
    <div class="paper-meta">
      📅 2025-04-03
    </div>
    <details class="paper-abstract">
      The advent of Large Language Models (LLMs) is transforming software development, significantly enhancing software engineering processes. Research has explored their role within development teams, focusing on specific tasks such as artifact generation, decision-making support, and information retrieval. Despite the growing body of work on LLMs in software engineering, most studies have centered on broad adoption trends, neglecting the nuanced relationship between individual cognitive and behavioral factors and their impact on task-specific adoption. While factors such as perceived effort and performance expectancy have been explored at a general level, their influence on distinct software engineering tasks remains underexamined. This gap hinders the development of tailored LLM-based systems (e.g., Generative AI Agents) that align with engineers' specific needs and limits the ability of team leaders to devise effective strategies for fostering LLM adoption in targeted workflows. This study bridges this gap by surveying N=188 software engineers to test the relationship between individual attributes related to technology adoption and LLM adoption across five key tasks, using structural equation modeling (SEM). The Unified Theory of Acceptance and Use of Technology (UTAUT2) was applied to characterize individual adoption behaviors. The findings reveal that task-specific adoption is influenced by distinct factors, some of which negatively impact adoption when considered in isolation, underscoring the complexity of LLM integration in software engineering. To support effective adoption, this article provides actionable recommendations, such as seamlessly integrating LLMs into existing development environments and encouraging peer-driven knowledge sharing to enhance information retrieval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02509v1">A Memory-Augmented LLM-Driven Method for Autonomous Merging of 3D Printing Work Orders</a></div>
    <div class="paper-meta">
      📅 2025-04-03
      | 💬 6 pages, 5 figures
    </div>
    <details class="paper-abstract">
      With the rapid development of 3D printing, the demand for personalized and customized production on the manufacturing line is steadily increasing. Efficient merging of printing workpieces can significantly enhance the processing efficiency of the production line. Addressing the challenge, a Large Language Model (LLM)-driven method is established in this paper for the autonomous merging of 3D printing work orders, integrated with a memory-augmented learning strategy. In industrial scenarios, both device and order features are modeled into LLM-readable natural language prompt templates, and develop an order-device matching tool along with a merging interference checking module. By incorporating a self-memory learning strategy, an intelligent agent for autonomous order merging is constructed, resulting in improved accuracy and precision in order allocation. The proposed method effectively leverages the strengths of LLMs in industrial applications while reducing hallucination.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02507v1">ZClip: Adaptive Spike Mitigation for LLM Pre-Training</a></div>
    <div class="paper-meta">
      📅 2025-04-03
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) presents numerous challenges, including gradient instability and loss spikes. These phenomena can lead to catastrophic divergence, requiring costly checkpoint restoration and data batch skipping. Traditional gradient clipping techniques, such as constant or norm-based methods, fail to address these issues effectively due to their reliance on fixed thresholds or heuristics, leading to inefficient learning and requiring frequent manual intervention. In this work, we propose ZClip, an adaptive gradient clipping algorithm that dynamically adjusts the clipping threshold based on statistical properties of gradient norms over time. Unlike prior reactive strategies, ZClip proactively adapts to training dynamics without making any prior assumptions on the scale and the temporal evolution of gradient norms. At its core, it leverages z-score-based anomaly detection to identify and mitigate large gradient spikes, preventing malignant loss spikes while not interfering with convergence otherwise. Our code is available at: https://github.com/bluorion-com/ZClip.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01667v2">Testing Low-Resource Language Support in LLMs Using Language Proficiency Exams: the Case of Luxembourgish</a></div>
    <div class="paper-meta">
      📅 2025-04-03
      | 💬 18 pages, 2 figures, 11 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become an increasingly important tool in research and society at large. While LLMs are regularly used all over the world by experts and lay-people alike, they are predominantly developed with English-speaking users in mind, performing well in English and other wide-spread languages while less-resourced languages such as Luxembourgish are seen as a lower priority. This lack of attention is also reflected in the sparsity of available evaluation tools and datasets. In this study, we investigate the viability of language proficiency exams as such evaluation tools for the Luxembourgish language. We find that large models such as ChatGPT, Claude and DeepSeek-R1 typically achieve high scores, while smaller models show weak performances. We also find that the performances in such language exams can be used to predict performances in other NLP tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02458v1">Retrieval-Augmented Purifier for Robust LLM-Empowered Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-04-03
    </div>
    <details class="paper-abstract">
      Recently, Large Language Model (LLM)-empowered recommender systems have revolutionized personalized recommendation frameworks and attracted extensive attention. Despite the remarkable success, existing LLM-empowered RecSys have been demonstrated to be highly vulnerable to minor perturbations. To mitigate the negative impact of such vulnerabilities, one potential solution is to employ collaborative signals based on item-item co-occurrence to purify the malicious collaborative knowledge from the user's historical interactions inserted by attackers. On the other hand, due to the capabilities to expand insufficient internal knowledge of LLMs, Retrieval-Augmented Generation (RAG) techniques provide unprecedented opportunities to enhance the robustness of LLM-empowered recommender systems by introducing external collaborative knowledge. Therefore, in this paper, we propose a novel framework (RETURN) by retrieving external collaborative signals to purify the poisoned user profiles and enhance the robustness of LLM-empowered RecSys in a plug-and-play manner. Specifically, retrieval-augmented perturbation positioning is proposed to identify potential perturbations within the users' historical sequences by retrieving external knowledge from collaborative item graphs. After that, we further retrieve the collaborative knowledge to cleanse the perturbations by using either deletion or replacement strategies and introduce a robust ensemble recommendation strategy to generate final robust predictions. Extensive experiments on three real-world datasets demonstrate the effectiveness of the proposed RETURN.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.06289v2">Automate Strategy Finding with LLM in Quant Investment</a></div>
    <div class="paper-meta">
      📅 2025-04-02
    </div>
    <details class="paper-abstract">
      Despite significant progress in deep learning for financial trading, existing models often face instability and high uncertainty, hindering their practical application. Leveraging advancements in Large Language Models (LLMs) and multi-agent architectures, we propose a novel framework for quantitative stock investment in portfolio management and alpha mining. Our framework addresses these issues by integrating LLMs to generate diversified alphas and employing a multi-agent approach to dynamically evaluate market conditions. This paper proposes a framework where large language models (LLMs) mine alpha factors from multimodal financial data, ensuring a comprehensive understanding of market dynamics. The first module extracts predictive signals by integrating numerical data, research papers, and visual charts. The second module uses ensemble learning to construct a diverse pool of trading agents with varying risk preferences, enhancing strategy performance through a broader market analysis. In the third module, a dynamic weight-gating mechanism selects and assigns weights to the most relevant agents based on real-time market conditions, enabling the creation of an adaptive and context-aware composite alpha formula. Extensive experiments on the Chinese stock markets demonstrate that this framework significantly outperforms state-of-the-art baselines across multiple financial metrics. The results underscore the efficacy of combining LLM-generated alphas with a multi-agent architecture to achieve superior trading performance and stability. This work highlights the potential of AI-driven approaches in enhancing quantitative investment strategies and sets a new benchmark for integrating advanced machine learning techniques in financial trading can also be applied on diverse markets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04667v5">Non-Determinism of "Deterministic" LLM Settings</a></div>
    <div class="paper-meta">
      📅 2025-04-02
    </div>
    <details class="paper-abstract">
      LLM (large language model) practitioners commonly notice that outputs can vary for the same inputs under settings expected to be deterministic. Yet the questions of how pervasive this is, and with what impact on results, have not to our knowledge been systematically investigated. We investigate non-determinism in five LLMs configured to be deterministic when applied to eight common tasks in across 10 runs, in both zero-shot and few-shot settings. We see accuracy variations up to 15% across naturally occurring runs with a gap of best possible performance to worst possible performance up to 70%. In fact, none of the LLMs consistently delivers repeatable accuracy across all tasks, much less identical output strings. Sharing preliminary results with insiders has revealed that non-determinism perhaps essential to the efficient use of compute resources via co-mingled data in input buffers so this issue is not going away anytime soon. To better quantify our observations, we introduce metrics focused on quantifying determinism, TARr@N for the total agreement rate at N runs over raw output, and TARa@N for total agreement rate of parsed-out answers. Our code and data are publicly available at https://github.com/breckbaldwin/llm-stability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02017v1">Enhancing LLMs in Long Code Translation through Instrumentation and Program State Alignment</a></div>
    <div class="paper-meta">
      📅 2025-04-02
      | 💬 20 pages
    </div>
    <details class="paper-abstract">
      Code translation aims to transform code between programming languages while preserving functionality, with applications in cross-platform development and software migration. Recent advances in Large Language Models (LLMs) have improved code translation, but challenges remain, particularly in inferring program functionality. These issues worsen with longer and more complex code, where current LLMs struggle to handle length and intricate semantics. To evaluate LLMs on long code translation, we introduce LongTrans, a large-scale execution-based benchmark with C++, Java, and Python programs, ranging from hundreds to thousands of tokens. Our empirical study of 12 LLMs reveals a sharp performance decline as code length increases, with even the best-performing model, GPT-4o, achieving only 57.51% computational accuracy. This highlights the need for further research in long code translation. We argue that code translation should maintain invariant functionality while transforming syntax and keywords across languages. Despite differences in appearance, program states should remain consistent throughout execution. To address this, we propose PAST (Program State Alignment augmented Translation), which integrates instrumentation to capture and align program states during translation. This approach is the first to leverage LLMs to insert instrumentation in both original and translated code, tracing program states at runtime. By prompting the LLM to correct errors based on output traces, we mitigate inconsistencies and enhance translation accuracy. Experimental results show significant improvements, with computational accuracy rising from 57.51% to 84.70% for GPT-4o, 50.68% to 69.97% for Mistral-Large-2, and 52.45% to 76.43% for DeepSeek-Coder-V2. These improvements are consistent across models and datasets, with ablation studies confirming the benefits of instrumentation and state alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01700v1">Reasoning LLMs for User-Aware Multimodal Conversational Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-02
    </div>
    <details class="paper-abstract">
      Personalization in social robotics is critical for fostering effective human-robot interactions, yet systems often face the cold start problem, where initial user preferences or characteristics are unavailable. This paper proposes a novel framework called USER-LLM R1 for a user-aware conversational agent that addresses this challenge through dynamic user profiling and model initiation. Our approach integrates chain-of-thought (CoT) reasoning models to iteratively infer user preferences and vision-language models (VLMs) to initialize user profiles from multimodal inputs, enabling personalized interactions from the first encounter. Leveraging a Retrieval-Augmented Generation (RAG) architecture, the system dynamically refines user representations within an inherent CoT process, ensuring contextually relevant and adaptive responses. Evaluations on the ElderlyTech-VQA Bench demonstrate significant improvements in ROUGE-1 (+23.2%), ROUGE-2 (+0.6%), and ROUGE-L (+8%) F1 scores over state-of-the-art baselines, with ablation studies underscoring the impact of reasoning model size on performance. Human evaluations further validate the framework's efficacy, particularly for elderly users, where tailored responses enhance engagement and trust. Ethical considerations, including privacy preservation and bias mitigation, are rigorously discussed and addressed to ensure responsible deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01698v1">ToM-RL: Reinforcement Learning Unlocks Theory of Mind in Small LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-02
    </div>
    <details class="paper-abstract">
      Recent advancements in rule-based reinforcement learning (RL), applied during the post-training phase of large language models (LLMs), have significantly enhanced their capabilities in structured reasoning tasks such as mathematics and logical inference. However, the effectiveness of RL in social reasoning, particularly in Theory of Mind (ToM), the ability to infer others' mental states, remains largely unexplored. In this study, we demonstrate that RL methods effectively unlock ToM reasoning capabilities even in small-scale LLMs (0.5B to 7B parameters). Using a modest dataset comprising 3200 questions across diverse scenarios, our RL-trained 7B model achieves 84.50\% accuracy on the Hi-ToM benchmark, surpassing models like GPT-4o and DeepSeek-v3 despite significantly fewer parameters. While smaller models ($\leq$3B parameters) suffer from reasoning collapse, larger models (7B parameters) maintain stable performance through consistent belief tracking. Additionally, our RL-based models demonstrate robust generalization to higher-order, out-of-distribution ToM problems, novel textual presentations, and previously unseen datasets. These findings highlight RL's potential to enhance social cognitive reasoning, bridging the gap between structured problem-solving and nuanced social inference in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00595v2">Open-Qwen2VL: Compute-Efficient Pre-Training of Fully-Open Multimodal LLMs on Academic Resources</a></div>
    <div class="paper-meta">
      📅 2025-04-02
    </div>
    <details class="paper-abstract">
      The reproduction of state-of-the-art multimodal LLM pre-training faces barriers at every stage of the pipeline, including high-quality data filtering, multimodal data mixture strategies, sequence packing techniques, and training frameworks. We introduce Open-Qwen2VL, a fully open-source 2B-parameter Multimodal Large Language Model pre-trained efficiently on 29M image-text pairs using only 220 A100-40G GPU hours. Our approach employs low-to-high dynamic image resolution and multimodal sequence packing to significantly enhance pre-training efficiency. The training dataset was carefully curated using both MLLM-based filtering techniques (e.g., MLM-Filter) and conventional CLIP-based filtering methods, substantially improving data quality and training efficiency. The Open-Qwen2VL pre-training is conducted on academic level 8xA100-40G GPUs at UCSB on 5B packed multimodal tokens, which is 0.36% of 1.4T multimodal pre-training tokens of Qwen2-VL. The final instruction-tuned Open-Qwen2VL outperforms partially-open state-of-the-art MLLM Qwen2-VL-2B on various multimodal benchmarks of MMBench, SEEDBench, MMstar, and MathVista, indicating the remarkable training efficiency of Open-Qwen2VL. We open-source all aspects of our work, including compute-efficient and data-efficient training details, data filtering methods, sequence packing scripts, pre-training data in WebDataset format, FSDP-based training codebase, and both base and instruction-tuned model checkpoints. We redefine "fully open" for multimodal LLMs as the complete release of: 1) the training codebase, 2) detailed data filtering techniques, and 3) all pre-training and supervised fine-tuning data used to develop the model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01602v1">Comment Staytime Prediction with LLM-enhanced Comment Understanding</a></div>
    <div class="paper-meta">
      📅 2025-04-02
      | 💬 Accepted by WWW 2025 Industry Track
    </div>
    <details class="paper-abstract">
      In modern online streaming platforms, the comments section plays a critical role in enhancing the overall user experience. Understanding user behavior within the comments section is essential for comprehensive user interest modeling. A key factor of user engagement is staytime, which refers to the amount of time that users browse and post comments. Existing watchtime prediction methods struggle to adapt to staytime prediction, overlooking interactions with individual comments and their interrelation. In this paper, we present a micro-video recommendation dataset with video comments (named as KuaiComt) which is collected from Kuaishou platform. correspondingly, we propose a practical framework for comment staytime prediction with LLM-enhanced Comment Understanding (LCU). Our framework leverages the strong text comprehension capabilities of large language models (LLMs) to understand textual information of comments, while also incorporating fine-grained comment ranking signals as auxiliary tasks. The framework is two-staged: first, the LLM is fine-tuned using domain-specific tasks to bridge the video and the comments; second, we incorporate the LLM outputs into the prediction model and design two comment ranking auxiliary tasks to better understand user preference. Extensive offline experiments demonstrate the effectiveness of our framework, showing significant improvements on the task of comment staytime prediction. Additionally, online A/B testing further validates the practical benefits on industrial scenario. Our dataset KuaiComt (https://github.com/lyingCS/KuaiComt.github.io) and code for LCU (https://github.com/lyingCS/LCU) are fully released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01588v1">Building Knowledge from Interactions: An LLM-Based Architecture for Adaptive Tutoring and Social Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-04-02
      | 💬 Submitted to IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2025
    </div>
    <details class="paper-abstract">
      Integrating robotics into everyday scenarios like tutoring or physical training requires robots capable of adaptive, socially engaging, and goal-oriented interactions. While Large Language Models show promise in human-like communication, their standalone use is hindered by memory constraints and contextual incoherence. This work presents a multimodal, cognitively inspired framework that enhances LLM-based autonomous decision-making in social and task-oriented Human-Robot Interaction. Specifically, we develop an LLM-based agent for a robot trainer, balancing social conversation with task guidance and goal-driven motivation. To further enhance autonomy and personalization, we introduce a memory system for selecting, storing and retrieving experiences, facilitating generalized reasoning based on knowledge built across different interactions. A preliminary HRI user study and offline experiments with a synthetic dataset validate our approach, demonstrating the system's ability to manage complex interactions, autonomously drive training tasks, and build and retrieve contextual memories, advancing socially intelligent robotics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01542v1">Register Always Matters: Analysis of LLM Pretraining Data Through the Lens of Language Variation</a></div>
    <div class="paper-meta">
      📅 2025-04-02
    </div>
    <details class="paper-abstract">
      Pretraining data curation is a cornerstone in Large Language Model (LLM) development, leading to growing research on quality filtering of large web corpora. From statistical quality flags to LLM-based labeling systems, datasets are divided into categories, frequently reducing to a binary: those passing the filters deemed as valuable examples, others discarded as useless or detrimental. However, a more detailed understanding of the contribution of different kinds of texts to model performance is still largely lacking. In this article, we present the first study utilizing registers (also known as genres) - a widely used standard in corpus linguistics to model linguistic variation - to curate pretraining datasets and investigate the effect of register on the performance of LLMs. We perform comparative studies by training models with register classified data and evaluating them using standard benchmarks, and show that the register of pretraining data substantially affects model performance. We uncover surprising relationships between the pretraining material and the resulting models: using the News register results in subpar performance, and on the contrary, including the Opinion class, covering texts such as reviews and opinion blogs, is highly beneficial. While a model trained on the entire unfiltered dataset outperforms those trained on datasets limited to a single register, combining well-performing registers like How-to-Instructions, Informational Description, and Opinion leads to major improvements. Furthermore, analysis of individual benchmark results reveals key differences in the strengths and drawbacks of specific register classes as pretraining data. These findings show that register is an important explainer of model variation and can facilitate more deliberate future data selection practices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00762v2">Do We Truly Need So Many Samples? Multi-LLM Repeated Sampling Efficiently Scales Test-Time Compute</a></div>
    <div class="paper-meta">
      📅 2025-04-02
    </div>
    <details class="paper-abstract">
      This paper presents a simple, effective, and cost-efficient strategy to improve LLM performance by scaling test-time compute. Our strategy builds upon the repeated-sampling-then-voting framework, with a novel twist: incorporating multiple models, even weaker ones, to leverage their complementary strengths that potentially arise from diverse training data and paradigms. By using consistency as a signal, our strategy dynamically switches between models. Theoretical analysis highlights the efficiency and performance advantages of our strategy. Extensive experiments on six datasets demonstrate that our strategy not only outperforms self-consistency and state-of-the-art multi-agent debate approaches, but also significantly reduces inference costs. Additionally, ModelSwitch requires only a few comparable LLMs to achieve optimal performance and can be extended with verification methods, demonstrating the potential of leveraging multiple LLMs in the generation-verification paradigm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.11313v2">An Optimizable Suffix Is Worth A Thousand Templates: Efficient Black-box Jailbreaking without Affirmative Phrases via LLM as Optimizer</a></div>
    <div class="paper-meta">
      📅 2025-04-02
      | 💬 Be accepeted as NAACL2025 Findings
    </div>
    <details class="paper-abstract">
      Despite prior safety alignment efforts, mainstream LLMs can still generate harmful and unethical content when subjected to jailbreaking attacks. Existing jailbreaking methods fall into two main categories: template-based and optimization-based methods. The former requires significant manual effort and domain knowledge, while the latter, exemplified by Greedy Coordinate Gradient (GCG), which seeks to maximize the likelihood of harmful LLM outputs through token-level optimization, also encounters several limitations: requiring white-box access, necessitating pre-constructed affirmative phrase, and suffering from low efficiency. In this paper, we present ECLIPSE, a novel and efficient black-box jailbreaking method utilizing optimizable suffixes. Drawing inspiration from LLMs' powerful generation and optimization capabilities, we employ task prompts to translate jailbreaking goals into natural language instructions. This guides the LLM to generate adversarial suffixes for malicious queries. In particular, a harmfulness scorer provides continuous feedback, enabling LLM self-reflection and iterative optimization to autonomously and efficiently produce effective suffixes. Experimental results demonstrate that ECLIPSE achieves an average attack success rate (ASR) of 0.92 across three open-source LLMs and GPT-3.5-Turbo, significantly surpassing GCG in 2.4 times. Moreover, ECLIPSE is on par with template-based methods in ASR while offering superior attack efficiency, reducing the average attack overhead by 83%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.08460v3">Is Your LLM Outdated? A Deep Look at Temporal Generalization</a></div>
    <div class="paper-meta">
      📅 2025-04-02
      | 💬 NAACL 2025 Oral
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) has led to the development of benchmarks that consider temporal dynamics, however, there remains a gap in understanding how well these models can generalize across temporal contexts due to the inherent dynamic nature of language and information. This paper introduces the concept of temporal generalization in LLMs, including bias in past and future generalizations. Then we introduce FreshBench, a new evaluation framework that employs fresh text and event prediction for assessing LLMs' temporal adaptability, ensuring the evaluation process free from data leakage and subjective bias. The experiment shows significant temporal biases and a decline in performance over time. Our findings reveal that powerful models, while initially superior, tend to decline more rapidly in future generalization. Additionally, powerful open-source models demonstrate better long-term adaptability compared to their closed-source counterparts. Our code is available at https://github.com/FreedomIntelligence/FreshBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05939v2">Direct Preference Optimization for LLM-Enhanced Recommendation Systems</a></div>
    <div class="paper-meta">
      📅 2025-04-02
      | 💬 This paper has been accepted to ICME 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have exhibited remarkable performance across a wide range of domains, motivating research into their potential for recommendation systems. Early efforts have leveraged LLMs' rich knowledge and strong generalization capabilities via in-context learning, where recommendation tasks are framed as prompts. However, LLM performance in recommendation scenarios remains limited due to the mismatch between their pretraining objectives and recommendation tasks, as well as the lack of recommendation-specific data during pretraining. To address these challenges, we propose DPO4Rec, a novel framework that integrates Direct Preference Optimization (DPO) into LLM-enhanced recommendation systems. First, we prompt the LLM to infer user preferences from historical interactions, which are then used to augment traditional ID-based sequential recommendation models. Next, we train a reward model based on knowledge-augmented recommendation architectures to assess the quality of LLM-generated reasoning. Using this, we select the highest- and lowest-ranked responses from N samples to construct a dataset for LLM fine-tuning. Finally, we apply a structure alignment strategy via DPO to align the LLM's outputs with desirable recommendation behavior. Extensive experiments show that DPO4Rec significantly improves re-ranking performance over strong baselines, demonstrating enhanced instruction-following capabilities of LLMs in recommendation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.16863v2">Augmenting Multimodal LLMs with Self-Reflective Tokens for Knowledge-based Visual Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-04-02
      | 💬 CVPR 2025
    </div>
    <details class="paper-abstract">
      Multimodal LLMs (MLLMs) are the natural extension of large language models to handle multimodal inputs, combining text and image data. They have recently garnered attention due to their capability to address complex tasks involving both modalities. However, their effectiveness is limited to the knowledge acquired during training, which restricts their practical utility. In this work, we introduce a novel method to enhance the adaptability of MLLMs by integrating external knowledge sources. Our proposed model, Reflective LLaVA (ReflectiVA), utilizes reflective tokens to dynamically determine the need for external knowledge and predict the relevance of information retrieved from an external database. Tokens are trained following a two-stage two-model training recipe. This ultimately enables the MLLM to manage external knowledge while preserving fluency and performance on tasks where external knowledge is not needed. Through our experiments, we demonstrate the efficacy of ReflectiVA for knowledge-based visual question answering, highlighting its superior performance compared to existing methods. Source code and trained models are publicly available at https://aimagelab.github.io/ReflectiVA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01369v1">LITE: LLM-Impelled efficient Taxonomy Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-04-02
    </div>
    <details class="paper-abstract">
      This paper presents LITE, an LLM-based evaluation method designed for efficient and flexible assessment of taxonomy quality. To address challenges in large-scale taxonomy evaluation, such as efficiency, fairness, and consistency, LITE adopts a top-down hierarchical evaluation strategy, breaking down the taxonomy into manageable substructures and ensuring result reliability through cross-validation and standardized input formats. LITE also introduces a penalty mechanism to handle extreme cases and provides both quantitative performance analysis and qualitative insights by integrating evaluation metrics closely aligned with task objectives. Experimental results show that LITE demonstrates high reliability in complex evaluation tasks, effectively identifying semantic errors, logical contradictions, and structural flaws in taxonomies, while offering directions for improvement. Code is available at https://github.com/Zhang-l-i-n/TAXONOMY_DETECT .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.21393v2">An evaluation of LLMs and Google Translate for translation of selected Indian languages via sentiment and semantic analyses</a></div>
    <div class="paper-meta">
      📅 2025-04-02
    </div>
    <details class="paper-abstract">
      Large Language models (LLMs) have been prominent for language translation, including low-resource languages. There has been limited study about the assessment of the quality of translations generated by LLMs, including Gemini, GPT and Google Translate. In this study, we address this limitation by using semantic and sentiment analysis of selected LLMs for Indian languages, including Sanskrit, Telugu and Hindi. We select prominent texts that have been well translated by experts and use LLMs to generate their translations to English, and then we provide a comparison with selected expert (human) translations. Our findings suggest that while LLMs have made significant progress in translation accuracy, challenges remain in preserving sentiment and semantic integrity, especially in figurative and philosophical contexts. The sentiment analysis revealed that GPT-4o and GPT-3.5 are better at preserving the sentiments for the Bhagavad Gita (Sanskrit-English) translations when compared to Google Translate. We observed a similar trend for the case of Tamas (Hindi-English) and Maha P (Telugu-English) translations. GPT-4o performs similarly to GPT-3.5 in the translation in terms of sentiments for the three languages. We found that LLMs are generally better at translation for capturing sentiments when compared to Google Translate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01304v1">Real-time Ad retrieval via LLM-generative Commercial Intention for Sponsored Search Advertising</a></div>
    <div class="paper-meta">
      📅 2025-04-02
      | 💬 13pages,5 figures
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) with retrieval systems has shown promising potential in retrieving documents (docs) or advertisements (ads) for a given query. Existing LLM-based retrieval methods generate numeric or content-based DocIDs to retrieve docs/ads. However, the one-to-few mapping between numeric IDs and docs, along with the time-consuming content extraction, leads to semantic inefficiency and limits scalability in large-scale corpora. In this paper, we propose the Real-time Ad REtrieval (RARE) framework, which leverages LLM-generated text called Commercial Intentions (CIs) as an intermediate semantic representation to directly retrieve ads for queries in real-time. These CIs are generated by a customized LLM injected with commercial knowledge, enhancing its domain relevance. Each CI corresponds to multiple ads, yielding a lightweight and scalable set of CIs. RARE has been implemented in a real-world online system, handling daily search volumes in the hundreds of millions. The online implementation has yielded significant benefits: a 5.04% increase in consumption, a 6.37% rise in Gross Merchandise Volume (GMV), a 1.28% enhancement in click-through rate (CTR) and a 5.29% increase in shallow conversions. Extensive offline experiments show RARE's superiority over ten competitive baselines in four major categories.
    </details>
</div>
