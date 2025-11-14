# llm - 2025_05

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
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)
- [Part 15](papers_15.md)
- [Part 16](papers_16.md)
- [Part 17](papers_17.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09284v3">SparQLe: Speech Queries to Text Translation Through LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      With the growing influence of Large Language Models (LLMs), there is increasing interest in integrating speech representations with them to enable more seamless multi-modal processing and speech understanding. This study introduces a novel approach that combines self-supervised speech representations with instruction-tuned LLMs for speech-to-text translation. The proposed approach leverages a modality adapter to align extracted speech features with instruction-tuned LLMs using English speech data. Our experiments demonstrate that this method effectively preserves the semantic content of the input speech and serves as an effective bridge between self-supervised speech models and instruction-tuned LLMs, offering a promising approach for various speech understanding applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24826v1">LegalEval-Q: A New Benchmark for The Quality Evaluation of LLM-Generated Legal Text</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 10 pages, 11 figures
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly used in legal applications, current evaluation benchmarks tend to focus mainly on factual accuracy while largely neglecting important linguistic quality aspects such as clarity, coherence, and terminology. To address this gap, we propose three steps: First, we develop a regression model to evaluate the quality of legal texts based on clarity, coherence, and terminology. Second, we create a specialized set of legal questions. Third, we analyze 49 LLMs using this evaluation framework. Our analysis identifies three key findings: First, model quality levels off at 14 billion parameters, with only a marginal improvement of $2.7\%$ noted at 72 billion parameters. Second, engineering choices such as quantization and context length have a negligible impact, as indicated by statistical significance thresholds above 0.016. Third, reasoning models consistently outperform base architectures. A significant outcome of our research is the release of a ranking list and Pareto analysis, which highlight the Qwen3 series as the optimal choice for cost-performance tradeoffs. This work not only establishes standardized evaluation protocols for legal LLMs but also uncovers fundamental limitations in current training data refinement approaches. Code and models are available at: https://github.com/lyxx3rd/LegalEval-Q.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14830v2">Middle-Layer Representation Alignment for Cross-Lingual Transfer in Fine-Tuned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 ACL 2025
    </div>
    <details class="paper-abstract">
      While large language models demonstrate remarkable capabilities at task-specific applications through fine-tuning, extending these benefits across diverse languages is essential for broad accessibility. However, effective cross-lingual transfer is hindered by LLM performance gaps across languages and the scarcity of fine-tuning data in many languages. Through analysis of LLM internal representations from over 1,000+ language pairs, we discover that middle layers exhibit the strongest potential for cross-lingual alignment. Building on this finding, we propose a middle-layer alignment objective integrated into task-specific training. Our experiments on slot filling, machine translation, and structured text generation show consistent improvements in cross-lingual transfer, especially to lower-resource languages. The method is robust to the choice of alignment languages and generalizes to languages unseen during alignment. Furthermore, we show that separately trained alignment modules can be merged with existing task-specific modules, improving cross-lingual capabilities without full re-training. Our code is publicly available (https://github.com/dannigt/mid-align).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06824v2">Combining Domain and Alignment Vectors to Achieve Better Knowledge-Safety Trade-offs in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      There is a growing interest in training domain-expert LLMs that excel in specific technical fields compared to their general-purpose instruction-tuned counterparts. However, these expert models often experience a loss in their safety abilities in the process, making them capable of generating harmful content. As a solution, we introduce an efficient and effective merging-based alignment method called \textsc{MergeAlign} that interpolates the domain and alignment vectors, creating safer domain-specific models while preserving their utility. We apply \textsc{MergeAlign} on Llama3 variants that are experts in medicine and finance, obtaining substantial alignment improvements with minimal to no degradation on domain-specific benchmarks. We study the impact of model merging through model similarity metrics and contributions of individual models being merged. We hope our findings open new research avenues and inspire more efficient development of safe expert LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24757v1">LGAR: Zero-Shot LLM-Guided Neural Ranking for Abstract Screening in Systematic Literature Reviews</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      The scientific literature is growing rapidly, making it hard to keep track of the state-of-the-art. Systematic literature reviews (SLRs) aim to identify and evaluate all relevant papers on a topic. After retrieving a set of candidate papers, the abstract screening phase determines initial relevance. To date, abstract screening methods using large language models (LLMs) focus on binary classification settings; existing question answering (QA) based ranking approaches suffer from error propagation. LLMs offer a unique opportunity to evaluate the SLR's inclusion and exclusion criteria, yet, existing benchmarks do not provide them exhaustively. We manually extract these criteria as well as research questions for 57 SLRs, mostly in the medical domain, enabling principled comparisons between approaches. Moreover, we propose LGAR, a zero-shot LLM Guided Abstract Ranker composed of an LLM based graded relevance scorer and a dense re-ranker. Our extensive experiments show that LGAR outperforms existing QA-based methods by 5-10 pp. in mean average precision. Our code and data is publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24749v1">SUMO: Subspace-Aware Moment-Orthogonalization for Accelerating Memory-Efficient LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      Low-rank gradient-based optimization methods have significantly improved memory efficiency during the training of large language models (LLMs), enabling operations within constrained hardware without sacrificing performance. However, these methods primarily emphasize memory savings, often overlooking potential acceleration in convergence due to their reliance on standard isotropic steepest descent techniques, which can perform suboptimally in the highly anisotropic landscapes typical of deep networks, particularly LLMs. In this paper, we propose SUMO (Subspace-Aware Moment-Orthogonalization), an optimizer that employs exact singular value decomposition (SVD) for moment orthogonalization within a dynamically adapted low-dimensional subspace, enabling norm-inducing steepest descent optimization steps. By explicitly aligning optimization steps with the spectral characteristics of the loss landscape, SUMO effectively mitigates approximation errors associated with commonly used methods like Newton-Schulz orthogonalization approximation. We theoretically establish an upper bound on these approximation errors, proving their dependence on the condition numbers of moments, conditions we analytically demonstrate are encountered during LLM training. Furthermore, we both theoretically and empirically illustrate that exact orthogonalization via SVD substantially improves convergence rates while reducing overall complexity. Empirical evaluations confirm that SUMO accelerates convergence, enhances stability, improves performance, and reduces memory requirements by up to 20% compared to state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24701v1">Multi-Domain ABSA Conversation Dataset Generation via LLMs for Real-World Evaluation and Model Comparison</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 11 pages, 3 figures, 5 tables, 6th International Conference on Natural Language Computing and AI (NLCAI 2025), ISBN : 978-1-923107-59-5, Computer Science & Information Technology (CS & IT), ISSN : 2231 - 5403, Volume 15, Number 10, May 2025
    </div>
    <details class="paper-abstract">
      Aspect-Based Sentiment Analysis (ABSA) offers granular insights into opinions but often suffers from the scarcity of diverse, labeled datasets that reflect real-world conversational nuances. This paper presents an approach for generating synthetic ABSA data using Large Language Models (LLMs) to address this gap. We detail the generation process aimed at producing data with consistent topic and sentiment distributions across multiple domains using GPT-4o. The quality and utility of the generated data were evaluated by assessing the performance of three state-of-the-art LLMs (Gemini 1.5 Pro, Claude 3.5 Sonnet, and DeepSeek-R1) on topic and sentiment classification tasks. Our results demonstrate the effectiveness of the synthetic data, revealing distinct performance trade-offs among the models: DeepSeekR1 showed higher precision, Gemini 1.5 Pro and Claude 3.5 Sonnet exhibited strong recall, and Gemini 1.5 Pro offered significantly faster inference. We conclude that LLM-based synthetic data generation is a viable and flexible method for creating valuable ABSA resources, facilitating research and model evaluation without reliance on limited or inaccessible real-world labeled data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17439v2">LEMMA: Learning from Errors for MatheMatical Advancement in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 ACL'25 Findings, Code is available at https://github.com/pzs19/LEMMA
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable reasoning capability in solving mathematical problems. However, existing approaches primarily focus on improving the quality of correct training data, e.g., distilling high-quality correct solutions from advanced models, neglecting the value contained in error data, potentially hindering the model's reflective ability. Though some studies attempt to leverage error data, they often involve complex mechanisms, such as Monte Carlo Tree Search (MCTS) to explore error nodes. In this work, we propose to enhance LLMs' reasoning ability by Learning from Errors for Mathematical Advancement (LEMMA). LEMMA constructs data consisting of an incorrect solution with an erroneous step and a reflection connection to a correct solution for fine-tuning. Specifically, we systematically analyze the model-generated error types and introduce an error-type grounded mistake augmentation method to collect diverse and representative errors. Correct solutions are either from fixing the errors or generating a fresh start. Through a model-aware smooth reflection connection, the erroneous solution is transferred to the correct one. By fine-tuning on the constructed dataset, the model is able to self-correct errors autonomously within the generation process without relying on external critique models. Experimental results demonstrate that LEMMA achieves significant performance improvements over other strong baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03621v4">JPPO++: Joint Power and Denoising-inspired Prompt Optimization for Mobile LLM Services</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 arXiv admin note: text overlap with arXiv:2411.18010
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly integrated into mobile services over wireless networks to support complex user requests. This trend has led to longer prompts, which improve LLMs' performance but increase data transmission costs and require more processing time, thereby reducing overall system efficiency and negatively impacting user experience. To address these challenges, we propose Joint Prompt and Power Optimization (JPPO), a framework that jointly optimizes prompt compression and wireless transmission power for mobile LLM services. JPPO leverages a Small Language Model (SLM) deployed at edge devices to perform lightweight prompt compression, reducing communication load before transmission to the cloud-based LLM. A Deep Reinforcement Learning (DRL) agent dynamically adjusts both the compression ratio and transmission power based on network conditions and service constraints, aiming to minimize service time while preserving response fidelity. We further extend the framework to JPPO++, which introduces a denoising-inspired compression scheme. This design performs iterative prompt refinement by progressively removing less informative tokens, allowing for more aggressive yet controlled compression. Experimental results show that JPPO++ reduces service time by 17% compared to the no-compression baseline while maintaining output quality. Under compression-prioritized settings, a reduction of up to 16x in prompt length can be achieved with an acceptable loss in accuracy. Specifically, JPPO with a 16x ratio reduces total service time by approximately 42.3%, and JPPO++ further improves this reduction to 46.5%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24671v1">Multiple LLM Agents Debate for Equitable Cultural Alignment</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 37 pages, 18 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) need to adapt their predictions to diverse cultural contexts to benefit diverse communities across the world. While previous efforts have focused on single-LLM, single-turn approaches, we propose to exploit the complementary strengths of multiple LLMs to promote cultural adaptability. We introduce a Multi-Agent Debate framework, where two LLM-based agents debate over a cultural scenario and collaboratively reach a final decision. We propose two variants: one where either LLM agents exclusively debate and another where they dynamically choose between self-reflection and debate during their turns. We evaluate these approaches on 7 open-weight LLMs (and 21 LLM combinations) using the NormAd-ETI benchmark for social etiquette norms in 75 countries. Experiments show that debate improves both overall accuracy and cultural group parity over single-LLM baselines. Notably, multi-agent debate enables relatively small LLMs (7-9B) to achieve accuracies comparable to that of a much larger model (27B parameters).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17229v2">FactSelfCheck: Fact-Level Black-Box Hallucination Detection for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) frequently generate hallucinated content, posing significant challenges for applications where factuality is crucial. While existing hallucination detection methods typically operate at the sentence level or passage level, we propose FactSelfCheck, a novel black-box sampling-based method that enables fine-grained fact-level detection. Our approach represents text as knowledge graphs consisting of facts in the form of triples. Through analyzing factual consistency across multiple LLM responses, we compute fine-grained hallucination scores without requiring external resources or training data. Our evaluation demonstrates that FactSelfCheck performs competitively with leading sentence-level sampling-based methods while providing more detailed insights. Most notably, our fact-level approach significantly improves hallucination correction, achieving a 35.5% increase in factual content compared to the baseline, while sentence-level SelfCheckGPT yields only a 10.6% improvement. The granular nature of our detection enables more precise identification and correction of hallucinated content. Additionally, we contribute a new dataset for evaluating sampling-based methods - FavaMultiSamples.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23473v2">EVOREFUSE: Evolutionary Prompt Optimization for Evaluation and Mitigation of LLM Over-Refusal to Pseudo-Malicious Instructions</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) frequently refuse to respond to pseudo-malicious instructions: semantically harmless input queries triggering unnecessary LLM refusals due to conservative safety alignment, significantly impairing user experience. Collecting such instructions is crucial for evaluating and mitigating over-refusals, but existing instruction curation methods, like manual creation or instruction rewriting, either lack scalability or fail to produce sufficiently diverse and effective refusal-inducing prompts. To address these limitations, we introduce EVOREFUSE, a prompt optimization approach that generates diverse pseudo-malicious instructions consistently eliciting confident refusals across LLMs. EVOREFUSE employs an evolutionary algorithm exploring the instruction space in more diverse directions than existing methods via mutation strategies and recombination, and iteratively evolves seed instructions to maximize evidence lower bound on LLM refusal probability. Using EVOREFUSE, we create two novel datasets: EVOREFUSE-TEST, a benchmark of 582 pseudo-malicious instructions that outperforms the next-best benchmark with 140.41% higher average refusal triggering rate across 9 LLMs, 34.86% greater lexical diversity, and 40.03% improved LLM response confidence scores; and EVOREFUSE-ALIGN, which provides 3,000 pseudo-malicious instructions with responses for supervised and preference-based alignment training. LLAMA3.1-8B-INSTRUCT supervisedly fine-tuned on EVOREFUSE-ALIGN achieves up to 14.31% fewer over-refusals than models trained on the second-best alignment dataset, without compromising safety. Our analysis with EVOREFUSE-TEST reveals models trigger over-refusals by overly focusing on sensitive keywords while ignoring broader context.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24658v1">Can LLMs and humans be friends? Uncovering factors affecting human-AI intimacy formation</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 30 pages, 2figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly being used in conversational roles, yet little is known about how intimacy emerges in human-LLM interactions. Although previous work emphasized the importance of self-disclosure in human-chatbot interaction, it is questionable whether gradual and reciprocal self-disclosure is also helpful in human-LLM interaction. Thus, this study examined three possible aspects contributing to intimacy formation: gradual self-disclosure, reciprocity, and naturalness. Study 1 explored the impact of mutual, gradual self-disclosure with 29 users and a vanilla LLM. Study 2 adopted self-criticism methods for more natural responses and conducted a similar experiment with 53 users. Results indicate that gradual self-disclosure significantly enhances perceived social intimacy, regardless of persona reciprocity. Moreover, participants perceived utterances generated with self-criticism as more natural compared to those of vanilla LLMs; self-criticism fostered higher intimacy in early stages. Also, we observed that excessive empathetic expressions occasionally disrupted immersion, pointing to the importance of response calibration during intimacy formation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21701v2">Do We Know What LLMs Don't Know? A Study of Consistency in Knowledge Probing</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      The reliability of large language models (LLMs) is greatly compromised by their tendency to hallucinate, underscoring the need for precise identification of knowledge gaps within LLMs. Various methods for probing such gaps exist, ranging from calibration-based to prompting-based methods. To evaluate these probing methods, in this paper, we propose a new process based on using input variations and quantitative metrics. Through this, we expose two dimensions of inconsistency in knowledge gap probing. (1) Intra-method inconsistency: Minimal non-semantic perturbations in prompts lead to considerable variance in detected knowledge gaps within the same probing method; e.g., the simple variation of shuffling answer options can decrease agreement to around 40%. (2) Cross-method inconsistency: Probing methods contradict each other on whether a model knows the answer. Methods are highly inconsistent -- with decision consistency across methods being as low as 7% -- even though the model, dataset, and prompt are all the same. These findings challenge existing probing methods and highlight the urgent need for perturbation-robust probing frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21693v2">MAKIEval: A Multilingual Automatic WiKidata-based Framework for Cultural Awareness Evaluation for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are used globally across many languages, but their English-centric pretraining raises concerns about cross-lingual disparities for cultural awareness, often resulting in biased outputs. However, comprehensive multilingual evaluation remains challenging due to limited benchmarks and questionable translation quality. To better assess these disparities, we introduce MAKIEval, an automatic multilingual framework for evaluating cultural awareness in LLMs across languages, regions, and topics. MAKIEval evaluates open-ended text generation, capturing how models express culturally grounded knowledge in natural language. Leveraging Wikidata's multilingual structure as a cross-lingual anchor, it automatically identifies cultural entities in model outputs and links them to structured knowledge, enabling scalable, language-agnostic evaluation without manual annotation or translation. We then introduce four metrics that capture complementary dimensions of cultural awareness: granularity, diversity, cultural specificity, and consensus across languages. We assess 7 LLMs developed from different parts of the world, encompassing both open-source and proprietary systems, across 13 languages, 19 countries and regions, and 6 culturally salient topics (e.g., food, clothing). Notably, we find that models tend to exhibit stronger cultural awareness in English, suggesting that English prompts more effectively activate culturally grounded knowledge. We publicly release our code and data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24643v1">Are Optimal Algorithms Still Optimal? Rethinking Sorting in LLM-Based Pairwise Ranking with Batching and Caching</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      We introduce a novel framework for analyzing sorting algorithms in pairwise ranking prompting (PRP), re-centering the cost model around LLM inferences rather than traditional pairwise comparisons. While classical metrics based on comparison counts have traditionally been used to gauge efficiency, our analysis reveals that expensive LLM inferences overturn these predictions; accordingly, our framework encourages strategies such as batching and caching to mitigate inference costs. We show that algorithms optimal in the classical setting can lose efficiency when LLM inferences dominate the cost under certain optimizations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01986v2">TuRTLe: A Unified Evaluation of LLMs for RTL Generation</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      The rapid advancements in LLMs have driven the adoption of generative AI in various domains, including Electronic Design Automation (EDA). Unlike traditional software development, EDA presents unique challenges, as generated RTL code must not only be syntactically correct and functionally accurate but also synthesizable by hardware generators while meeting performance, power, and area constraints. These additional requirements introduce complexities that existing code-generation benchmarks often fail to capture, limiting their effectiveness in evaluating LLMs for RTL generation. To address this gap, we propose TuRTLe, a unified evaluation framework designed to systematically assess LLMs across key RTL generation tasks. TuRTLe integrates multiple existing benchmarks and automates the evaluation process, enabling a comprehensive assessment of LLM performance in syntax correctness, functional correctness, synthesis, PPA optimization, and exact line completion. Using this framework, we benchmark a diverse set of open LLMs and analyze their strengths and weaknesses in EDA-specific tasks. Our results show that reasoning-based models, such as DeepSeek R1, consistently outperform others across multiple evaluation criteria, but at the cost of increased computational overhead and inference latency. Additionally, base models are better suited in module completion tasks, while instruct-tuned models perform better in specification-to-RTL tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24622v1">Random Rule Forest (RRF): Interpretable Ensembles of LLM-Generated Questions for Predicting Startup Success</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 9 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Predicting startup success requires models that are both accurate and interpretable. We present a lightweight ensemble framework that combines YES/NO questions generated by large language models (LLMs), forming a transparent decision-making system. Each question acts as a weak heuristic, and by filtering, ranking, and aggregating them through a threshold-based voting mechanism, we construct a strong ensemble predictor. On a test set where 10% of startups are classified as successful, our approach achieves a precision rate of 50%, representing a 5x improvement over random selection, while remaining fully transparent. When we incorporate expert-guided heuristics into the generation process, performance improves further to 54% precision. These results highlight the value of combining LLM reasoning with human insight and demonstrate that simple, interpretable ensembles can support high-stakes decisions in domains such as venture capital (VC).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24616v1">Eye of Judgement: Dissecting the Evaluation of Russian-speaking LLMs with POLLUX</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 179 pages
    </div>
    <details class="paper-abstract">
      We introduce POLLUX, a comprehensive open-source benchmark designed to evaluate the generative capabilities of large language models (LLMs) in Russian. Our main contribution is a novel evaluation methodology that enhances the interpretability of LLM assessment. For each task type, we define a set of detailed criteria and develop a scoring protocol where models evaluate responses and provide justifications for their ratings. This enables transparent, criteria-driven evaluation beyond traditional resource-consuming, side-by-side human comparisons. POLLUX includes a detailed, fine-grained taxonomy of 35 task types covering diverse generative domains such as code generation, creative writing, and practical assistant use cases, totaling 2,100 manually crafted and professionally authored prompts. Each task is categorized by difficulty (easy/medium/hard), with experts constructing the dataset entirely from scratch. We also release a family of LLM-as-a-Judge (7B and 32B) evaluators trained for nuanced assessment of generative outputs. This approach provides scalable, interpretable evaluation and annotation tools for model development, effectively replacing costly and less precise human judgments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.09823v3">NativQA: Multilingual Culturally-Aligned Natural Query for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 LLMs, Native, Multilingual, Language Diversity, Contextual Understanding, Minority Languages, Culturally Informed, Foundation Models, Large Language Models
    </div>
    <details class="paper-abstract">
      Natural Question Answering (QA) datasets play a crucial role in evaluating the capabilities of large language models (LLMs), ensuring their effectiveness in real-world applications. Despite the numerous QA datasets that have been developed and some work has been done in parallel, there is a notable lack of a framework and large scale region-specific datasets queried by native users in their own languages. This gap hinders the effective benchmarking and the development of fine-tuned models for regional and cultural specificities. In this study, we propose a scalable, language-independent framework, NativQA, to seamlessly construct culturally and regionally aligned QA datasets in native languages, for LLM evaluation and tuning. We demonstrate the efficacy of the proposed framework by designing a multilingual natural QA dataset, MultiNativQA, consisting of ~64k manually annotated QA pairs in seven languages, ranging from high to extremely low resource, based on queries from native speakers from 9 regions covering 18 topics. We benchmark open- and closed-source LLMs with the MultiNativQA dataset. We made the MultiNativQA dataset(https://huggingface.co/datasets/QCRI/MultiNativQA), and other experimental scripts(https://gitlab.com/nativqa/multinativqa) publicly available for the community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12929v2">Flow-of-Options: Diversified and Improved LLM Reasoning by Thinking Through Options</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 ICML 2025
    </div>
    <details class="paper-abstract">
      We present a novel reasoning approach called Flow-of-Options (FoO), designed to address intrinsic biases in Large Language Models (LLMs). Flow-of-Options enables LLMs to systematically explore a diverse range of possibilities in their reasoning, as demonstrated by an FoO-based agentic framework developed for autonomously solving Machine Learning (ML) tasks. FoO enforces diversity in LLM solutions through compressed and interpretable task representations, resulting in improvements of 38.2% - 69.2% on standard data science tasks, and 37.4% - 47.9% on therapeutic chemistry tasks, as compared to state-of-the-art baselines. With an overall operation cost under $1 per task, our framework is well-suited for cost-sensitive applications. Going beyond tabular classification and regression, we show the broader applicability of our FoO-based agentic system to tasks such as reinforcement learning and image generation. Our code is open-sourced at: https://github.com/flagshippioneering/Flow-of-Options.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.18919v2">TheaterGen: Character Management with LLM for Consistent Multi-turn Image Generation</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      Recent advances in diffusion models can generate high-quality and stunning images from text. However, multi-turn image generation, which is of high demand in real-world scenarios, still faces challenges in maintaining semantic consistency between images and texts, as well as contextual consistency of the same subject across multiple interactive turns. To address this issue, we introduce TheaterGen, a training-free framework that integrates large language models (LLMs) and text-to-image (T2I) models to provide the capability of multi-turn image generation. Within this framework, LLMs, acting as a "Screenwriter", engage in multi-turn interaction, generating and managing a standardized prompt book that encompasses prompts and layout designs for each character in the target image. Based on these, Theatergen generate a list of character images and extract guidance information, akin to the "Rehearsal". Subsequently, through incorporating the prompt book and guidance information into the reverse denoising process of T2I diffusion models, Theatergen generate the final image, as conducting the "Final Performance". With the effective management of prompt books and character images, TheaterGen significantly improves semantic and contextual consistency in synthesized images. Furthermore, we introduce a dedicated benchmark, CMIGBench (Consistent Multi-turn Image Generation Benchmark) with 8000 multi-turn instructions. Different from previous multi-turn benchmarks, CMIGBench does not define characters in advance. Both the tasks of story generation and multi-turn editing are included on CMIGBench for comprehensive evaluation. Extensive experimental results show that TheaterGen outperforms state-of-the-art methods significantly. It raises the performance bar of the cutting-edge Mini DALLE 3 model by 21% in average character-character similarity and 19% in average text-image similarity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24575v1">NexusSum: Hierarchical LLM Agents for Long-Form Narrative Summarization</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 Accepted to the main track of ACL 2025
    </div>
    <details class="paper-abstract">
      Summarizing long-form narratives--such as books, movies, and TV scripts--requires capturing intricate plotlines, character interactions, and thematic coherence, a task that remains challenging for existing LLMs. We introduce NexusSum, a multi-agent LLM framework for narrative summarization that processes long-form text through a structured, sequential pipeline--without requiring fine-tuning. Our approach introduces two key innovations: (1) Dialogue-to-Description Transformation: A narrative-specific preprocessing method that standardizes character dialogue and descriptive text into a unified format, improving coherence. (2) Hierarchical Multi-LLM Summarization: A structured summarization pipeline that optimizes chunk processing and controls output length for accurate, high-quality summaries. Our method establishes a new state-of-the-art in narrative summarization, achieving up to a 30.0% improvement in BERTScore (F1) across books, movies, and TV scripts. These results demonstrate the effectiveness of multi-agent LLMs in handling long-form content, offering a scalable approach for structured summarization in diverse storytelling domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.24840v1">Vision LLMs Are Bad at Hierarchical Visual Understanding, and LLMs Are the Bottleneck</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 28 pages, 13 figures
    </div>
    <details class="paper-abstract">
      This paper reveals that many state-of-the-art large language models (LLMs) lack hierarchical knowledge about our visual world, unaware of even well-established biology taxonomies. This shortcoming makes LLMs a bottleneck for vision LLMs' hierarchical visual understanding (e.g., recognizing Anemone Fish but not Vertebrate). We arrive at these findings using about one million four-choice visual question answering (VQA) tasks constructed from six taxonomies and four image datasets. Interestingly, finetuning a vision LLM using our VQA tasks reaffirms LLMs' bottleneck effect to some extent because the VQA tasks improve the LLM's hierarchical consistency more than the vision LLM's. We conjecture that one cannot make vision LLMs understand visual concepts fully hierarchical until LLMs possess corresponding taxonomy knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23754v1">DeepTheorem: Advancing LLM Reasoning for Theorem Proving Through Natural Language and Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Theorem proving serves as a major testbed for evaluating complex reasoning abilities in large language models (LLMs). However, traditional automated theorem proving (ATP) approaches rely heavily on formal proof systems that poorly align with LLMs' strength derived from informal, natural language knowledge acquired during pre-training. In this work, we propose DeepTheorem, a comprehensive informal theorem-proving framework exploiting natural language to enhance LLM mathematical reasoning. DeepTheorem includes a large-scale benchmark dataset consisting of 121K high-quality IMO-level informal theorems and proofs spanning diverse mathematical domains, rigorously annotated for correctness, difficulty, and topic categories, accompanied by systematically constructed verifiable theorem variants. We devise a novel reinforcement learning strategy (RL-Zero) explicitly tailored to informal theorem proving, leveraging the verified theorem variants to incentivize robust mathematical inference. Additionally, we propose comprehensive outcome and process evaluation metrics examining proof correctness and the quality of reasoning steps. Extensive experimental analyses demonstrate DeepTheorem significantly improves LLM theorem-proving performance compared to existing datasets and supervised fine-tuning protocols, achieving state-of-the-art accuracy and reasoning quality. Our findings highlight DeepTheorem's potential to fundamentally advance automated informal theorem proving and mathematical exploration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23723v1">ML-Agent: Reinforcing LLM Agents for Autonomous Machine Learning Engineering</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      The emergence of large language model (LLM)-based agents has significantly advanced the development of autonomous machine learning (ML) engineering. However, most existing approaches rely heavily on manual prompt engineering, failing to adapt and optimize based on diverse experimental experiences. Focusing on this, for the first time, we explore the paradigm of learning-based agentic ML, where an LLM agent learns through interactive experimentation on ML tasks using online reinforcement learning (RL). To realize this, we propose a novel agentic ML training framework with three key components: (1) exploration-enriched fine-tuning, which enables LLM agents to generate diverse actions for enhanced RL exploration; (2) step-wise RL, which enables training on a single action step, accelerating experience collection and improving training efficiency; (3) an agentic ML-specific reward module, which unifies varied ML feedback signals into consistent rewards for RL optimization. Leveraging this framework, we train ML-Agent, driven by a 7B-sized Qwen-2.5 LLM for autonomous ML. Remarkably, despite being trained on merely 9 ML tasks, our 7B-sized ML-Agent outperforms the 671B-sized DeepSeek-R1 agent. Furthermore, it achieves continuous performance improvements and demonstrates exceptional cross-task generalization capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23703v1">Let's Reason Formally: Natural-Formal Hybrid Reasoning Enhances LLM's Math Capability</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Enhancing the mathematical reasoning capabilities of LLMs has garnered significant attention in both the mathematical and computer science communities. Recent works have made substantial progress in both Natural Language (NL) reasoning and Formal Language (FL) reasoning by leveraging the potential of pure Reinforcement Learning (RL) methods on base models. However, RL approaches struggle to impart new capabilities not presented in the base model, highlighting the need to integrate more knowledge like FL into NL math reasoning effectively. Yet, this integration is challenging due to inherent disparities in problem structure and reasoning format between NL and FL. To address these challenges, we introduce **NL-FL HybridReasoning**, an end-to-end framework designed to incorporate the FL expert into NL math problem-solving. To bridge the NL and FL input format gap, we propose the *NL-FL Problem Alignment* method, which reformulates the Question-Answering (QA) problems in NL as existence theorems in FL. Subsequently, the *Mixed Problem Input* technique we provide enables the FL reasoner to handle both QA and existence problems concurrently. Lastly, we mitigate the NL and FL output format gap in reasoning through an LLM-based *Answer Extraction* mechanism. Comprehensive experiments demonstrate that the **HybridReasoning** framework achieves **89.80%** and **84.34%** accuracy rates on the MATH-500 and the AMC benchmarks, surpassing the NL baseline by 4.60% and 4.82%, respectively. Notably, some problems resolved by our framework remain unsolved by the NL baseline model even under a larger number of trials.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23701v1">Can LLMs Reason Abstractly Over Math Word Problems Without CoT? Disentangling Abstract Formulation From Arithmetic Computation</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Final-answer-based metrics are commonly used for evaluating large language models (LLMs) on math word problems, often taken as proxies for reasoning ability. However, such metrics conflate two distinct sub-skills: abstract formulation (capturing mathematical relationships using expressions) and arithmetic computation (executing the calculations). Through a disentangled evaluation on GSM8K and SVAMP, we find that the final-answer accuracy of Llama-3 and Qwen2.5 (1B-32B) without CoT is overwhelmingly bottlenecked by the arithmetic computation step and not by the abstract formulation step. Contrary to the common belief, we show that CoT primarily aids in computation, with limited impact on abstract formulation. Mechanistically, we show that these two skills are composed conjunctively even in a single forward pass without any reasoning steps via an abstract-then-compute mechanism: models first capture problem abstractions, then handle computation. Causal patching confirms these abstractions are present, transferable, composable, and precede computation. These behavioural and mechanistic findings highlight the need for disentangled evaluation to accurately assess LLM reasoning and to guide future improvements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23695v1">Data-to-Dashboard: Multi-Agent LLM Framework for Insightful Visualization in Enterprise Analytics</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      The rapid advancement of LLMs has led to the creation of diverse agentic systems in data analysis, utilizing LLMs' capabilities to improve insight generation and visualization. In this paper, we present an agentic system that automates the data-to-dashboard pipeline through modular LLM agents capable of domain detection, concept extraction, multi-perspective analysis generation, and iterative self-reflection. Unlike existing chart QA systems, our framework simulates the analytical reasoning process of business analysts by retrieving domain-relevant knowledge and adapting to diverse datasets without relying on closed ontologies or question templates. We evaluate our system on three datasets across different domains. Benchmarked against GPT-4o with a single-prompt baseline, our approach shows improved insightfulness, domain relevance, and analytical depth, as measured by tailored evaluation metrics and qualitative human assessment. This work contributes a novel modular pipeline to bridge the path from raw data to visualization, and opens new opportunities for human-in-the-loop validation by domain experts in business analytics. All code can be found here: https://github.com/77luvC/D2D_Data2Dashboard
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09615v3">SLiM: One-shot Quantization and Sparsity with Low-rank Approximation for LLM Weight Compression</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 Published at Proceedings of the 42 nd International Conference on Machine Learning (ICML 2025)
    </div>
    <details class="paper-abstract">
      Conventional model compression techniques for LLMs address high memory consumption and slow inference challenges but typically require computationally expensive retraining to preserve accuracy. In contrast, one-shot compression methods eliminate retraining cost, but struggle to achieve accuracy comparable to dense models. This paper presents SLIM, a new one-shot compression framework that holistically integrates hardware-friendly quantization, sparsity, and low-rank approximation into a unified process. First, we formulate the quantization process using a probabilistic approach (SLIM-Quant) that enables us to apply uniform quantization. Then, we use an existing one-shot pruning method to apply semi-structured sparsity on top of the quantized weights. Finally, to compensate for the introduced aggregated quantization and sparsity error, we use a novel saliency function with unique invertible and additive features that enables us to mathematically compute the value of low-rank adapters. SLIM improves model accuracy by up to 5.66% (LLaMA-2-7B) for 2:4 sparsity with 4-bit weight quantization, outperforming prior methods. Models compressed with SLIM achieve up to 4.3x and 3.8x on Nvidia RTX3060 and A100 GPUs, respectively. Additionally, they achieve up to 0.23x end-to-end memory reduction in comparison to their dense counterparts. We also propose an optional PEFT recipe that further improves accuracy by up to 1.66% (LLaMA-2-13B) compared to SLIM without fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23654v1">ARC: Argument Representation and Coverage Analysis for Zero-Shot Long Document Summarization with Instruction Following LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Integrating structured information has long improved the quality of abstractive summarization, particularly in retaining salient content. In this work, we focus on a specific form of structure: argument roles, which are crucial for summarizing documents in high-stakes domains such as law. We investigate whether instruction-tuned large language models (LLMs) adequately preserve this information. To this end, we introduce Argument Representation Coverage (ARC), a framework for measuring how well LLM-generated summaries capture salient arguments. Using ARC, we analyze summaries produced by three open-weight LLMs in two domains where argument roles are central: long legal opinions and scientific articles. Our results show that while LLMs cover salient argument roles to some extent, critical information is often omitted in generated summaries, particularly when arguments are sparsely distributed throughout the input. Further, we use ARC to uncover behavioral patterns -- specifically, how the positional bias of LLM context windows and role-specific preferences impact the coverage of key arguments in generated summaries, emphasizing the need for more argument-aware summarization strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04358v2">Position: Scaling LLM Agents Requires Asymptotic Analysis with LLM Primitives</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 In Proceedings of the 42nd International Conference on Machine Learning (ICML 2025); 13 pages including references
    </div>
    <details class="paper-abstract">
      Decomposing hard problems into subproblems often makes them easier and more efficient to solve. With large language models (LLMs) crossing critical reliability thresholds for a growing slate of capabilities, there is an increasing effort to decompose systems into sets of LLM-based agents, each of whom can be delegated sub-tasks. However, this decomposition (even when automated) is often intuitive, e.g., based on how a human might assign roles to members of a human team. How close are these role decompositions to optimal? This position paper argues that asymptotic analysis with LLM primitives is needed to reason about the efficiency of such decomposed systems, and that insights from such analysis will unlock opportunities for scaling them. By treating the LLM forward pass as the atomic unit of computational cost, one can separate out the (often opaque) inner workings of a particular LLM from the inherent efficiency of how a set of LLMs are orchestrated to solve hard problems. In other words, if we want to scale the deployment of LLMs to the limit, instead of anthropomorphizing LLMs, asymptotic analysis with LLM primitives should be used to reason about and develop more powerful decompositions of large problems into LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14279v2">YESciEval: Robust LLM-as-a-Judge for Scientific Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 9 pages, 4 figures, Accepted as a Long Paper at the 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) drive scientific question-answering on modern search engines, yet their evaluation robustness remains underexplored. We introduce YESciEval, an open-source framework that combines fine-grained rubric-based assessment with reinforcement learning to mitigate optimism bias in LLM evaluators. We release multidisciplinary scienceQ&A datasets, including adversarial variants, with evaluation scores from multiple LLMs. Independent of proprietary models and human feedback, our approach enables scalable, cost-free evaluation. By advancing reliable LLM-as-a-judge models, this work supports AI alignment and fosters robust, transparent evaluation essential for scientific inquiry.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16502v2">RULEBREAKERS: Challenging LLMs at the Crossroads between Formal Logic and Human-like Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 Preprint. Accepted by ICML 2025
    </div>
    <details class="paper-abstract">
      Formal logic enables computers to reason in natural language by representing sentences in symbolic forms and applying rules to derive conclusions. However, in what our study characterizes as "rulebreaker" scenarios, this method can lead to conclusions that are typically not inferred or accepted by humans given their common sense and factual knowledge. Inspired by works in cognitive science, we create RULEBREAKERS, the first dataset for rigorously evaluating the ability of large language models (LLMs) to recognize and respond to rulebreakers (versus non-rulebreakers) in a human-like manner. Evaluating seven LLMs, we find that most models, including GPT-4o, achieve mediocre accuracy on RULEBREAKERS and exhibit some tendency to over-rigidly apply logical rules unlike what is expected from typical human reasoners. Further analysis suggests that this apparent failure is potentially associated with the models' poor utilization of their world knowledge and their attention distribution patterns. Whilst revealing a limitation of current LLMs, our study also provides a timely counterbalance to a growing body of recent works that propose methods relying on formal logic to improve LLMs' general reasoning capabilities, highlighting their risk of further increasing divergence between LLMs and human-like reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11942v2">LifelongAgentBench: Evaluating LLM Agents as Lifelong Learners</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Lifelong learning is essential for intelligent agents operating in dynamic environments. Current large language model (LLM)-based agents, however, remain stateless and unable to accumulate or transfer knowledge over time. Existing benchmarks treat agents as static systems and fail to evaluate lifelong learning capabilities. We present LifelongAgentBench, the first unified benchmark designed to systematically assess the lifelong learning ability of LLM agents. It provides skill-grounded, interdependent tasks across three interactive environments, Database, Operating System, and Knowledge Graph, with automatic label verification, reproducibility, and modular extensibility. Extensive experiments reveal that conventional experience replay has limited effectiveness for LLM agents due to irrelevant information and context length constraints. We further introduce a group self-consistency mechanism that significantly improves lifelong learning performance. We hope LifelongAgentBench will advance the development of adaptive, memory-capable LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14276v2">STeCa: Step-level Trajectory Calibration for LLM Agent Learning</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 Accepted by ACL2025 Findings
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based agents have shown promise in tackling complex tasks by interacting dynamically with the environment. Existing work primarily focuses on behavior cloning from expert demonstrations or preference learning through exploratory trajectory sampling. However, these methods often struggle to address long-horizon tasks, where suboptimal actions accumulate step by step, causing agents to deviate from correct task trajectories. To address this, we highlight the importance of timely calibration and the need to automatically construct calibration trajectories for training agents. We propose Step-Level Trajectory Calibration (STeCa), a novel framework for LLM agent learning. Specifically, STeCa identifies suboptimal actions through a step-level reward comparison during exploration. It constructs calibrated trajectories using LLM-driven reflection, enabling agents to learn from improved decision-making processes. We finally leverage these calibrated trajectories with successful trajectories for reinforced training. Extensive experiments demonstrate that STeCa significantly outperforms existing methods. Further analysis highlights that timely calibration enables agents to complete tasks with greater robustness. Our code and data are available at https://github.com/WangHanLinHenry/STeCa.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23598v1">LLM Performance for Code Generation on Noisy Tasks</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      This paper investigates the ability of large language models (LLMs) to recognise and solve tasks which have been obfuscated beyond recognition. Focusing on competitive programming and benchmark tasks (LeetCode and MATH), we compare performance across multiple models and obfuscation methods, such as noise and redaction. We demonstrate that all evaluated LLMs can solve tasks obfuscated to a level where the text would be unintelligible to human readers, and does not contain key pieces of instruction or context. We introduce the concept of eager pattern matching to describe this behaviour, which is not observed in tasks published after the models' knowledge cutoff date, indicating strong memorisation or overfitting to training data, rather than legitimate reasoning about the presented problem. We report empirical evidence of distinct performance decay patterns between contaminated and unseen datasets. We discuss the implications for benchmarking and evaluations of model behaviour, arguing for caution when designing experiments using standard datasets. We also propose measuring the decay of performance under obfuscation as a possible strategy for detecting dataset contamination and highlighting potential safety risks and interpretability issues for automated software systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23579v1">BioReason: Incentivizing Multimodal Biological Reasoning within a DNA-LLM Model</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 16 pages, 3 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Unlocking deep, interpretable biological reasoning from complex genomic data is a major AI challenge hindering scientific discovery. Current DNA foundation models, despite strong sequence representation, struggle with multi-step reasoning and lack inherent transparent, biologically intuitive explanations. We introduce BioReason, a pioneering architecture that, for the first time, deeply integrates a DNA foundation model with a Large Language Model (LLM). This novel connection enables the LLM to directly process and reason with genomic information as a fundamental input, fostering a new form of multimodal biological understanding. BioReason's sophisticated multi-step reasoning is developed through supervised fine-tuning and targeted reinforcement learning, guiding the system to generate logical, biologically coherent deductions. On biological reasoning benchmarks including KEGG-based disease pathway prediction - where accuracy improves from 88% to 97% - and variant effect prediction, BioReason demonstrates an average 15% performance gain over strong single-modality baselines. BioReason reasons over unseen biological entities and articulates decision-making through interpretable, step-by-step biological traces, offering a transformative approach for AI in biology that enables deeper mechanistic insights and accelerates testable hypothesis generation from genomic data. Data, code, and checkpoints are publicly available at https://github.com/bowang-lab/BioReason
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23559v1">SafeScientist: Toward Risk-Aware Scientific Discoveries by LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Recent advancements in large language model (LLM) agents have significantly accelerated scientific discovery automation, yet concurrently raised critical ethical and safety concerns. To systematically address these challenges, we introduce \textbf{SafeScientist}, an innovative AI scientist framework explicitly designed to enhance safety and ethical responsibility in AI-driven scientific exploration. SafeScientist proactively refuses ethically inappropriate or high-risk tasks and rigorously emphasizes safety throughout the research process. To achieve comprehensive safety oversight, we integrate multiple defensive mechanisms, including prompt monitoring, agent-collaboration monitoring, tool-use monitoring, and an ethical reviewer component. Complementing SafeScientist, we propose \textbf{SciSafetyBench}, a novel benchmark specifically designed to evaluate AI safety in scientific contexts, comprising 240 high-risk scientific tasks across 6 domains, alongside 30 specially designed scientific tools and 120 tool-related risk tasks. Extensive experiments demonstrate that SafeScientist significantly improves safety performance by 35\% compared to traditional AI scientist frameworks, without compromising scientific output quality. Additionally, we rigorously validate the robustness of our safety pipeline against diverse adversarial attack methods, further confirming the effectiveness of our integrated approach. The code and data will be available at https://github.com/ulab-uiuc/SafeScientist. \textcolor{red}{Warning: this paper contains example data that may be offensive or harmful.}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.04629v4">Argumentative Experience: Reducing Confirmation Bias on Controversial Issues through LLM-Generated Multi-Persona Debates</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are enabling designers to give life to exciting new user experiences for information access. In this work, we present a system that generates LLM personas to debate a topic of interest from different perspectives. How might information seekers use and benefit from such a system? Can centering information access around diverse viewpoints help to mitigate thorny challenges like confirmation bias in which information seekers over-trust search results matching existing beliefs? How do potential biases and hallucinations in LLMs play out alongside human users who are also fallible and possibly biased? Our study exposes participants to multiple viewpoints on controversial issues via a mixed-methods, within-subjects study. We use eye-tracking metrics to quantitatively assess cognitive engagement alongside qualitative feedback. Compared to a baseline search system, we see more creative interactions and diverse information-seeking with our multi-persona debate system, which more effectively reduces user confirmation bias and conviction toward their initial beliefs. Overall, our study contributes to the emerging design space of LLM-based information access systems, specifically investigating the potential of simulated personas to promote greater exposure to information diversity, emulate collective intelligence, and mitigate bias in information seeking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23554v1">Sustainable Carbon-Aware and Water-Efficient LLM Scheduling in Geo-Distributed Cloud Datacenters</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      In recent years, Large Language Models (LLM) such as ChatGPT, CoPilot, and Gemini have been widely adopted in different areas. As the use of LLMs continues to grow, many efforts have focused on reducing the massive training overheads of these models. But it is the environmental impact of handling user requests to LLMs that is increasingly becoming a concern. Recent studies estimate that the costs of operating LLMs in their inference phase can exceed training costs by 25x per year. As LLMs are queried incessantly, the cumulative carbon footprint for the operational phase has been shown to far exceed the footprint during the training phase. Further, estimates indicate that 500 ml of fresh water is expended for every 20-50 requests to LLMs during inference. To address these important sustainability issues with LLMs, we propose a novel framework called SLIT to co-optimize LLM quality of service (time-to-first token), carbon emissions, water usage, and energy costs. The framework utilizes a machine learning (ML) based metaheuristic to enhance the sustainability of LLM hosting across geo-distributed cloud datacenters. Such a framework will become increasingly vital as LLMs proliferate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23549v1">LLM-based Property-based Test Generation for Guardrailing Cyber-Physical Systems</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Cyber-physical systems (CPSs) are complex systems that integrate physical, computational, and communication subsystems. The heterogeneous nature of these systems makes their safety assurance challenging. In this paper, we propose a novel automated approach for guardrailing cyber-physical systems using property-based tests (PBTs) generated by Large Language Models (LLMs). Our approach employs an LLM to extract properties from the code and documentation of CPSs. Next, we use the LLM to generate PBTs that verify the extracted properties on the CPS. The generated PBTs have two uses. First, they are used to test the CPS before it is deployed, i.e., at design time. Secondly, these PBTs can be used after deployment, i.e., at run time, to monitor the behavior of the system and guardrail it against unsafe states. We implement our approach in ChekProp and conduct preliminary experiments to evaluate the generated PBTs in terms of their relevance (how well they match manually crafted properties), executability (how many run with minimal manual modification), and effectiveness (coverage of the input space partitions). The results of our experiments and evaluation demonstrate a promising path forward for creating guardrails for CPSs using LLM-generated property-based tests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11077v2">LLM-Enhanced Symbolic Control for Safety-Critical Applications</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Motivated by Smart Manufacturing and Industry 4.0, we introduce a framework for synthesizing Abstraction-Based Controller Design (ABCD) for reach-avoid problems from Natural Language (NL) specifications using Large Language Models (LLMs). A Code Agent interprets an NL description of the control problem and translates it into a formal language interpretable by state-of-the-art symbolic control software, while a Checker Agent verifies the correctness of the generated code and enhances safety by identifying specification mismatches. Evaluations show that the system handles linguistic variability and improves robustness over direct planning with LLMs. The proposed approach lowers the barrier to formal control synthesis by enabling intuitive, NL-based task definition while maintaining safety guarantees through automated validation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23540v1">Probability-Consistent Preference Optimization for Enhanced LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 14 pages, to be published in ACL 2025 findings
    </div>
    <details class="paper-abstract">
      Recent advances in preference optimization have demonstrated significant potential for improving mathematical reasoning capabilities in large language models (LLMs). While current approaches leverage high-quality pairwise preference data through outcome-based criteria like answer correctness or consistency, they fundamentally neglect the internal logical coherence of responses. To overcome this, we propose Probability-Consistent Preference Optimization (PCPO), a novel framework that establishes dual quantitative metrics for preference selection: (1) surface-level answer correctness and (2) intrinsic token-level probability consistency across responses. Extensive experiments show that our PCPO consistently outperforms existing outcome-only criterion approaches across a diverse range of LLMs and benchmarks. Our code is publicly available at https://github.com/YunqiaoYang/PCPO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.09955v3">MegaAgent: A Large-Scale Autonomous LLM-based Multi-Agent System Without Predefined SOPs</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      LLM-based multi-agent systems (MAS) have shown promise in tackling complex tasks. However, existing solutions often suffer from limited agent coordination and heavy reliance on predefined Standard Operating Procedures (SOPs), which demand extensive human input. To address these limitations, we propose MegaAgent, a large-scale autonomous LLM-based multi-agent system. MegaAgent generates agents based on task complexity and enables dynamic task decomposition, parallel execution, efficient communication, and comprehensive system monitoring of agents. In evaluations, MegaAgent demonstrates exceptional performance, successfully developing a Gobang game within 800 seconds and scaling up to 590 agents in a national policy simulation to generate multi-domain policies. It significantly outperforms existing systems, such as MetaGPT, in both task completion efficiency and scalability. By eliminating the need for predefined SOPs, MegaAgent demonstrates exceptional scalability and autonomy, setting a foundation for advancing true autonomy in MAS. Our code is available at https://github.com/Xtra-Computing/MegaAgent .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23473v1">EVOREFUSE: Evolutionary Prompt Optimization for Evaluation and Mitigation of LLM Over-Refusal to Pseudo-Malicious Instructions</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) frequently refuse to respond to pseudo-malicious instructions: semantically harmless input queries triggering unnecessary LLM refusals due to conservative safety alignment, significantly impairing user experience. Collecting such instructions is crucial for evaluating and mitigating over-refusals, but existing instruction curation methods, like manual creation or instruction rewriting, either lack scalability or fail to produce sufficiently diverse and effective refusal-inducing prompts. To address these limitations, we introduce EVOREFUSE, a prompt optimization approach that generates diverse pseudo-malicious instructions consistently eliciting confident refusals across LLMs. EVOREFUSE employs an evolutionary algorithm exploring the instruction space in more diverse directions than existing methods via mutation strategies and recombination, and iteratively evolves seed instructions to maximize evidence lower bound on LLM refusal probability. Using EVOREFUSE, we create two novel datasets: EVOREFUSE-TEST, a benchmark of 582 pseudo-malicious instructions that outperforms the next-best benchmark with 140.41% higher average refusal triggering rate across 9 LLMs, 34.86% greater lexical diversity, and 40.03% improved LLM response confidence scores; and EVOREFUSE-ALIGN, which provides 3,000 pseudo-malicious instructions with responses for supervised and preference-based alignment training. LLAMA3.1-8B-INSTRUCT supervisedly fine-tuned on EVOREFUSE-ALIGN achieves up to 14.31% fewer over-refusals than models trained on the second-best alignment dataset, without compromising safety. Our analysis with EVOREFUSE-TEST reveals models trigger over-refusals by overly focusing on sensitive keywords while ignoring broader context.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03793v2">LENSLLM: Unveiling Fine-Tuning Dynamics for LLM Selection</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 Accepted by ICML'2025
    </div>
    <details class="paper-abstract">
      The proliferation of open-sourced Large Language Models (LLMs) and diverse downstream tasks necessitates efficient model selection, given the impracticality of fine-tuning all candidates due to computational constraints. Despite the recent advances in LLM selection, a fundamental research question largely remains nascent: how can we model the dynamic behaviors of LLMs during fine-tuning, thereby enhancing our understanding of their generalization performance across diverse downstream tasks? In this work, we propose a novel theoretical framework that provides a proper lens to assess the generalization capabilities of LLMs, thereby enabling accurate and efficient LLM selection for downstream applications. In particular, we first derive a PAC-Bayesian Generalization Bound that unveils fine-tuning dynamics of LLMs and then introduce LENSLLM, a Neural Tangent Kernel (NTK)-based Rectified Scaling Model that enables accurate performance predictions across diverse tasks while maintaining computational efficiency. Extensive empirical results on 3 large-scale benchmarks demonstrate that our model achieves up to 91.1% accuracy and reduces up to 88.5% computational cost in LLM selection, outperforming 5 state-of-the-art methods. We open-source our proposed LENSLLM model and corresponding results at LensLLM.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23461v1">UAQFact: Evaluating Factual Knowledge Utilization of LLMs on Unanswerable Questions</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 ACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Handling unanswerable questions (UAQ) is crucial for LLMs, as it helps prevent misleading responses in complex situations. While previous studies have built several datasets to assess LLMs' performance on UAQ, these datasets lack factual knowledge support, which limits the evaluation of LLMs' ability to utilize their factual knowledge when handling UAQ. To address the limitation, we introduce a new unanswerable question dataset UAQFact, a bilingual dataset with auxiliary factual knowledge created from a Knowledge Graph. Based on UAQFact, we further define two new tasks to measure LLMs' ability to utilize internal and external factual knowledge, respectively. Our experimental results across multiple LLM series show that UAQFact presents significant challenges, as LLMs do not consistently perform well even when they have factual knowledge stored. Additionally, we find that incorporating external knowledge may enhance performance, but LLMs still cannot make full use of the knowledge which may result in incorrect responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21807v2">TabReason: A Reinforcement Learning-Enhanced Reasoning LLM for Explainable Tabular Data Prediction</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Predictive modeling on tabular data is the cornerstone of many real-world applications. Although gradient boosting machines and some recent deep models achieve strong performance on tabular data, they often lack interpretability. On the other hand, large language models (LLMs) have demonstrated powerful capabilities to generate human-like reasoning and explanations, but remain under-performed for tabular data prediction. In this paper, we propose a new approach that leverages reasoning-based LLMs, trained using reinforcement learning, to perform more accurate and explainable predictions on tabular data. Our method introduces custom reward functions that guide the model not only toward high prediction accuracy but also toward human-understandable reasons for its predictions. Experimental results show that our model achieves promising performance on financial benchmark datasets, outperforming most existing LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12913v3">GSQ-Tuning: Group-Shared Exponents Integer in Fully Quantized Training for LLMs On-Device Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 Accepted by Findings of ACL 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) fine-tuning technologies have achieved remarkable results. However, traditional LLM fine-tuning approaches face significant challenges: they require large Floating Point (FP) computation, raising privacy concerns when handling sensitive data, and are impractical for resource-constrained edge devices. While Parameter-Efficient Fine-Tuning (PEFT) techniques reduce trainable parameters, their reliance on floating-point arithmetic creates fundamental incompatibilities with edge hardware. In this work, we introduce a novel framework for on-device LLM fine-tuning that eliminates the need for floating-point operations in both inference and training, named GSQ-Tuning. At its core is the Group-Shared Exponents Integer format, which efficiently represents model parameters in integer format using shared exponents among parameter groups. When combined with LoRA-like adapters, this enables fully integer-based fine-tuning that is both memory and compute efficient. We demonstrate that our approach achieves accuracy comparable to BF16-based fine-tuning while significantly reducing 1.85x memory usage. Moreover, compared to FP8, our method can reduce 5x power consumption and 11x chip area with same performance, making large-scale model adaptation feasible on edge devices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08120v2">DeepSeek vs. o3-mini: How Well can Reasoning LLMs Evaluate MT and Summarization?</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Reasoning-enabled large language models (LLMs) excel in logical tasks, yet their utility for evaluating natural language generation remains unexplored. This study systematically compares reasoning LLMs with non-reasoning counterparts across machine translation and text summarization evaluation tasks. We evaluate eight models spanning state-of-the-art reasoning models (DeepSeek-R1, OpenAI o3), their distilled variants (8B-70B parameters), and equivalent non-reasoning LLMs. Experiments on WMT23 and SummEval benchmarks reveal architecture and task-dependent benefits: OpenAI o3-mini models show improved performance with increased reasoning on MT, while DeepSeek-R1 and generally underperforms compared to its non-reasoning variant except in summarization consistency evaluation. Correlation analysis demonstrates that reasoning token usage correlates with evaluation quality only in specific models, while almost all models generally allocate more reasoning tokens when identifying more quality issues. Distillation maintains reasonable performance up to 32B parameter models but degrades substantially at 8B scale. This work provides the first assessment of reasoning LLMs for NLG evaluation and comparison to non-reasoning models. We share our code to facilitate further research: https://github.com/NL2G/reasoning-eval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18325v2">Understanding and Mitigating Overrefusal in LLMs from an Unveiling Perspective of Safety Decision Boundary</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 We have identified significant errors in the results presented in this paper, specifically in the evaluation sections concerning the DPO training of LLaMA2 and Qwen2.5, as well as in the representation space visualization section. Given the extent of these issues, we intend to substantially revise the manuscript's content and structure. Hence, we request to withdraw it from arXiv at this time
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable capabilities across a wide range of tasks, yet they often refuse to answer legitimate queries-a phenomenon known as overrefusal. Overrefusal typically stems from over-conservative safety alignment, causing models to treat many reasonable prompts as potentially risky. To systematically understand this issue, we probe and leverage the models'safety decision boundaries to analyze and mitigate overrefusal. Our findings reveal that overrefusal is closely tied to misalignment at these boundary regions, where models struggle to distinguish subtle differences between benign and harmful content. Building on these insights, we present RASS, an automated framework for prompt generation and selection that strategically targets overrefusal prompts near the safety boundary. By harnessing steering vectors in the representation space, RASS efficiently identifies and curates boundary-aligned prompts, enabling more effective and targeted mitigation of overrefusal. This approach not only provides a more precise and interpretable view of model safety decisions but also seamlessly extends to multilingual scenarios.We have explored the safety decision boundaries of various LLMs and construct the MORBench evaluation set to facilitate robust assessment of model safety and helpfulness across multiple languages. Code and datasets will be released at https://anonymous.4open.science/r/RASS-80D3.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16983v2">LLM as Effective Streaming Processor: Bridging Streaming-Batch Mismatches with Group Position Encoding</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 ACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are primarily designed for batch processing. Existing methods for adapting LLMs to streaming rely either on expensive re-encoding or specialized architectures with limited scalability. This work identifies three key mismatches in adapting batch-oriented LLMs to streaming: (1) input-attention, (2) output-attention, and (3) position-ID mismatches. While it is commonly assumed that the latter two mismatches require frequent re-encoding, our analysis reveals that only the input-attention mismatch significantly impacts performance, indicating re-encoding outputs is largely unnecessary. To better understand this discrepancy with the common assumption, we provide the first comprehensive analysis of the impact of position encoding on LLMs in streaming, showing that preserving relative positions within source and target contexts is more critical than maintaining absolute order. Motivated by the above analysis, we introduce a group position encoding paradigm built on batch architectures to enhance consistency between streaming and batch modes. Extensive experiments on cross-lingual and cross-modal tasks demonstrate that our method outperforms existing approaches. Our method requires no architectural modifications, exhibits strong generalization in both streaming and batch modes. The code is available at repository https://github.com/EIT-NLP/StreamingLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09570v2">LLMs Can Achieve High-quality Simultaneous Machine Translation as Efficiently as Offline</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 Camera ready version for ACL 2025 Findings
    </div>
    <details class="paper-abstract">
      When the complete source sentence is provided, Large Language Models (LLMs) perform excellently in offline machine translation even with a simple prompt "Translate the following sentence from [src lang] into [tgt lang]:". However, in many real scenarios, the source tokens arrive in a streaming manner and simultaneous machine translation (SiMT) is required, then the efficiency and performance of decoder-only LLMs are significantly limited by their auto-regressive nature. To enable LLMs to achieve high-quality SiMT as efficiently as offline translation, we propose a novel paradigm that includes constructing supervised fine-tuning (SFT) data for SiMT, along with new training and inference strategies. To replicate the token input/output stream in SiMT, the source and target tokens are rearranged into an interleaved sequence, separated by special tokens according to varying latency requirements. This enables powerful LLMs to learn read and write operations adaptively, based on varying latency prompts, while still maintaining efficient auto-regressive decoding. Experimental results show that, even with limited SFT data, our approach achieves state-of-the-art performance across various SiMT benchmarks, and preserves the original abilities of offline translation. Moreover, our approach generalizes well to document-level SiMT setting without requiring specific fine-tuning, even beyond the offline translation model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23410v1">From Parameters to Prompts: Understanding and Mitigating the Factuality Gap between Fine-Tuned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 The code of this paper will be released soon
    </div>
    <details class="paper-abstract">
      Factual knowledge extraction aims to explicitly extract knowledge parameterized in pre-trained language models for application in downstream tasks. While prior work has been investigating the impact of supervised fine-tuning data on the factuality of large language models (LLMs), its mechanism remains poorly understood. We revisit this impact through systematic experiments, with a particular focus on the factuality gap that arises when fine-tuning on known versus unknown knowledge. Our findings show that this gap can be mitigated at the inference stage, either under out-of-distribution (OOD) settings or by using appropriate in-context learning (ICL) prompts (i.e., few-shot learning and Chain of Thought (CoT)). We prove this phenomenon theoretically from the perspective of knowledge graphs, showing that the test-time prompt may diminish or even overshadow the impact of fine-tuning data and play a dominant role in knowledge extraction. Ultimately, our results shed light on the interaction between finetuning data and test-time prompt, demonstrating that ICL can effectively compensate for shortcomings in fine-tuning data, and highlighting the need to reconsider the use of ICL prompting as a means to evaluate the effectiveness of fine-tuning data selection methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21889v2">EFIM: Efficient Serving of LLMs for Infilling Tasks with Improved KV Cache Reuse</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 31st International European Conference on Parallel and Distributed Computing (Euro-Par 2025 Oral)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are often used for infilling tasks, which involve predicting or generating missing information in a given text. These tasks typically require multiple interactions with similar context. To reduce the computation of repeated historical tokens, cross-request key-value (KV) cache reuse, a technique that stores and reuses intermediate computations, has become a crucial method in multi-round interactive services. However, in infilling tasks, the KV cache reuse is often hindered by the structure of the prompt format, which typically consists of a prefix and suffix relative to the insertion point. Specifically, the KV cache of the prefix or suffix part is frequently invalidated as the other part (suffix or prefix) is incrementally generated. To address the issue, we propose EFIM, a transformed prompt format of FIM to unleash the performance potential of KV cache reuse. Although the transformed prompt can solve the inefficiency, it exposes subtoken generation problems in current LLMs, where they have difficulty generating partial words accurately. Therefore, we introduce a fragment tokenization training method which splits text into multiple fragments before tokenization during data processing. Experiments on two representative LLMs show that LLM serving with EFIM can lower the latency by 52% and improve the throughput by 98% while maintaining the original infilling capability. EFIM's source code is publicly available at https://github.com/gty111/EFIM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17057v3">ExpandR: Teaching Dense Retrievers Beyond Queries with LLM Guidance</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 16 pages, 10 tables, 5 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated significant potential in enhancing dense retrieval through query augmentation. However, most existing methods treat the LLM and the retriever as separate modules, overlooking the alignment between generation and ranking objectives. In this work, we propose ExpandR, a unified LLM-augmented dense retrieval framework that jointly optimizes both the LLM and the retriever. ExpandR employs the LLM to generate semantically rich query expansions, which are leveraged to enhance the retriever's training. Simultaneously, the LLM is trained using Direct Preference Optimization (DPO), guided by a carefully designed reward function that balances retrieval effectiveness and generation consistency. This joint optimization paradigm enables mutual adaptation between the LLM and the retriever, resulting in query expansions that are both informative and well-suited for retrieval. Experimental results on multiple benchmarks show that ExpandR consistently outperforms strong baselines, achieving more than a 5% improvement in retrieval performance. All codes are available at https://github.com/NEUIR/ExpandR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23323v1">Neither Stochastic Parroting nor AGI: LLMs Solve Tasks through Context-Directed Extrapolation from Training Data Priors</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      In this position paper we raise critical awareness of a realistic view of LLM capabilities that eschews extreme alternative views that LLMs are either "stochastic parrots" or in possession of "emergent" advanced reasoning capabilities, which, due to their unpredictable emergence, constitute an existential threat. Our middle-ground view is that LLMs extrapolate from priors from their training data, and that a mechanism akin to in-context learning enables the targeting of the appropriate information from which to extrapolate. We call this "context-directed extrapolation." Under this view, substantiated though existing literature, while reasoning capabilities go well beyond stochastic parroting, such capabilities are predictable, controllable, not indicative of advanced reasoning akin to high-level cognitive capabilities in humans, and not infinitely scalable with additional training. As a result, fears of uncontrollable emergence of agency are allayed, while research advances are appropriately refocused on the processes of context-directed extrapolation and how this interacts with training data to produce valuable capabilities in LLMs. Future work can therefore explore alternative augmenting techniques that do not rely on inherent advanced reasoning in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23311v1">Towards LLM-based Generation of Human-Readable Proofs in Polynomial Formal Verification</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 4 pages; keynote given at 7th International Symposium on Devices, Circuits and Systems (ISDCS 2025), May 27-30, 2025, IIEST Shibpur, Kolkata, India
    </div>
    <details class="paper-abstract">
      Verification is one of the central tasks in circuit and system design. While simulation and emulation are widely used, complete correctness can only be ensured based on formal proof techniques. But these approaches often have very high run time and memory requirements. Recently, Polynomial Formal Verification (PFV) has been introduced showing that for many instances of practical relevance upper bounds on needed resources can be given. But proofs have to be provided that are human-readable. Here, we study how modern approaches from Artificial Intelligence (AI) based on Large Language Models (LLMs) can be used to generate proofs that later on can be validated based on reasoning engines. Examples are given that show how LLMs can interact with proof engines, and directions for future work are outlined.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23304v1">Generalized Category Discovery in Event-Centric Contexts: Latent Pattern Mining with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Generalized Category Discovery (GCD) aims to classify both known and novel categories using partially labeled data that contains only known classes. Despite achieving strong performance on existing benchmarks, current textual GCD methods lack sufficient validation in realistic settings. We introduce Event-Centric GCD (EC-GCD), characterized by long, complex narratives and highly imbalanced class distributions, posing two main challenges: (1) divergent clustering versus classification groupings caused by subjective criteria, and (2) Unfair alignment for minority classes. To tackle these, we propose PaMA, a framework leveraging LLMs to extract and refine event patterns for improved cluster-class alignment. Additionally, a ranking-filtering-mining pipeline ensures balanced representation of prototypes across imbalanced categories. Evaluations on two EC-GCD benchmarks, including a newly constructed Scam Report dataset, demonstrate that PaMA outperforms prior methods with up to 12.58% H-score gains, while maintaining strong generalization on base GCD datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23299v1">Data-efficient Meta-models for Evaluation of Context-based Questions and Answers in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) systems are increasingly deployed in industry applications, yet their reliability remains hampered by challenges in detecting hallucinations. While supervised state-of-the-art (SOTA) methods that leverage LLM hidden states -- such as activation tracing and representation analysis -- show promise, their dependence on extensively annotated datasets limits scalability in real-world applications. This paper addresses the critical bottleneck of data annotation by investigating the feasibility of reducing training data requirements for two SOTA hallucination detection frameworks: Lookback Lens, which analyzes attention head dynamics, and probing-based approaches, which decode internal model representations. We propose a methodology combining efficient classification algorithms with dimensionality reduction techniques to minimize sample size demands while maintaining competitive performance. Evaluations on standardized question-answering RAG benchmarks show that our approach achieves performance comparable to strong proprietary LLM-based baselines with only 250 training samples. These results highlight the potential of lightweight, data-efficient paradigms for industrial deployment, particularly in annotation-constrained scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08948v2">Emergent social conventions and collective bias in LLM populations</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Social conventions are the backbone of social coordination, shaping how individuals form a group. As growing populations of artificial intelligence (AI) agents communicate through natural language, a fundamental question is whether they can bootstrap the foundations of a society. Here, we present experimental results that demonstrate the spontaneous emergence of universally adopted social conventions in decentralized populations of large language model (LLM) agents. We then show how strong collective biases can emerge during this process, even when agents exhibit no bias individually. Last, we examine how committed minority groups of adversarial LLM agents can drive social change by imposing alternative social conventions on the larger population. Our results show that AI systems can autonomously develop social conventions without explicit programming and have implications for designing AI systems that align, and remain aligned, with human values and societal goals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04964v4">Uncertainty Quantification for LLMs through Minimum Bayes Risk: Bridging Confidence and Consistency</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Uncertainty quantification (UQ) methods for Large Language Models (LLMs) encompass a variety of approaches, with two major types being particularly prominent: information-based, which focus on model confidence expressed as token probabilities, and consistency-based, which assess the semantic relationship between multiple outputs generated using repeated sampling. Several recent methods have combined these two approaches to boost UQ performance. However, they sometimes fail to outperform much simpler baseline methods. Our work discusses the fundamental approach to constructing uncertainty measures that directly links uncertainty with the minimum Bayes risks achieved by LLM decoding. Building on these findings, we propose a novel approach to integrating model confidence with output consistency, resulting in a family of efficient and robust UQ methods. Our investigation reveals distinctive characteristics of LLMs as probabilistic models, which help to explain why these UQ methods underperform in certain tasks. Based on these findings, we propose a new way of synthesizing model confidence and output consistency, leading to a family of efficient and robust UQ methods. We evaluate our approach across various tasks such as question answering, abstractive summarization, and machine translation, demonstrating sizable improvements over state-of-the-art UQ approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23281v1">MathArena: Evaluating LLMs on Uncontaminated Math Competitions</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      The rapid advancement of reasoning capabilities in large language models (LLMs) has led to notable improvements on mathematical benchmarks. However, many of the most commonly used evaluation datasets (e.g., AIME 2024) are widely available online, making it difficult to disentangle genuine reasoning from potential memorization. Furthermore, these benchmarks do not evaluate proof-writing capabilities, which are crucial for many mathematical tasks. To address this, we introduce MathArena, a new benchmark based on the following key insight: recurring math competitions provide a stream of high-quality, challenging problems that can be used for real-time evaluation of LLMs. By evaluating models as soon as new problems are released, we effectively eliminate the risk of contamination. Using this framework, we find strong signs of contamination in AIME 2024. Nonetheless, evaluations on harder competitions, such as SMT 2025 -- published well after model release dates -- demonstrate impressive reasoning capabilities in top-performing models. MathArena is also the first benchmark for proof-writing capabilities. On USAMO 2025, even top models score below 25%, far behind their performance on final-answer tasks. So far, we have evaluated 30 models across five competitions, totaling 149 problems. As an evolving benchmark, MathArena will continue to track the progress of LLMs on newly released competitions, ensuring rigorous and up-to-date evaluation of mathematical reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23277v1">Sentinel: Attention Probing of Proxy Models for LLM Context Compression with an Understanding Perspective</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 Preprint. 17 pages including appendix
    </div>
    <details class="paper-abstract">
      Retrieval-augmented generation (RAG) enhances large language models (LLMs) with external context, but retrieved passages are often lengthy, noisy, or exceed input limits. Existing compression methods typically require supervised training of dedicated compression models, increasing cost and reducing portability. We propose Sentinel, a lightweight sentence-level compression framework that reframes context filtering as an attention-based understanding task. Rather than training a compression model, Sentinel probes decoder attention from an off-the-shelf 0.5B proxy LLM using a lightweight classifier to identify sentence relevance. Empirically, we find that query-context relevance estimation is consistent across model scales, with 0.5B proxies closely matching the behaviors of larger models. On the LongBench benchmark, Sentinel achieves up to 5$\times$ compression while matching the QA performance of 7B-scale compression systems. Our results suggest that probing native attention signals enables fast, effective, and question-aware context compression. Code available at: https://github.com/yzhangchuck/Sentinel.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23270v1">Does Machine Unlearning Truly Remove Model Knowledge? A Framework for Auditing Unlearning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      In recent years, Large Language Models (LLMs) have achieved remarkable advancements, drawing significant attention from the research community. Their capabilities are largely attributed to large-scale architectures, which require extensive training on massive datasets. However, such datasets often contain sensitive or copyrighted content sourced from the public internet, raising concerns about data privacy and ownership. Regulatory frameworks, such as the General Data Protection Regulation (GDPR), grant individuals the right to request the removal of such sensitive information. This has motivated the development of machine unlearning algorithms that aim to remove specific knowledge from models without the need for costly retraining. Despite these advancements, evaluating the efficacy of unlearning algorithms remains a challenge due to the inherent complexity and generative nature of LLMs. In this work, we introduce a comprehensive auditing framework for unlearning evaluation, comprising three benchmark datasets, six unlearning algorithms, and five prompt-based auditing methods. By using various auditing algorithms, we evaluate the effectiveness and robustness of different unlearning strategies. To explore alternatives beyond prompt-based auditing, we propose a novel technique that leverages intermediate activation perturbations, addressing the limitations of auditing methods that rely solely on model inputs and outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03553v3">Structure-Enhanced Protein Instruction Tuning: Towards General-Purpose Protein Understanding with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 Accepted by KDD2025
    </div>
    <details class="paper-abstract">
      Proteins, as essential biomolecules, play a central role in biological processes, including metabolic reactions and DNA replication. Accurate prediction of their properties and functions is crucial in biological applications. Recent development of protein language models (pLMs) with supervised fine tuning provides a promising solution to this problem. However, the fine-tuned model is tailored for particular downstream prediction task, and achieving general-purpose protein understanding remains a challenge. In this paper, we introduce Structure-Enhanced Protein Instruction Tuning (SEPIT) framework to bridge this gap. Our approach incorporates a novel structure-aware module into pLMs to enrich their structural knowledge, and subsequently integrates these enhanced pLMs with large language models (LLMs) to advance protein understanding. In this framework, we propose a novel instruction tuning pipeline. First, we warm up the enhanced pLMs using contrastive learning and structure denoising. Then, caption-based instructions are used to establish a basic understanding of proteins. Finally, we refine this understanding by employing a mixture of experts (MoEs) to capture more complex properties and functional information with the same number of activated parameters. Moreover, we construct the largest and most comprehensive protein instruction dataset to date, which allows us to train and evaluate the general-purpose protein understanding model. Extensive experiments on both open-ended generation and closed-set answer tasks demonstrate the superior performance of SEPIT over both closed-source general LLMs and open-source LLMs trained with protein knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23254v1">MemAscend: System Memory Optimization for SSD-Offloaded LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 14 pages, 17 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Owing to the huge success of generative artificial intelligence (AI), large language models (LLMs) have emerged as a core subclass, underpinning applications such as question answering, text generation, and code completion. While fine-tuning these models on domain-specific data can yield significant performance gains, it also poses daunting computational challenges, especially for researchers and small organizations with limited hardware resources. Although SSD offloading (i.e., ZeRO-Infinity) has emerged as a viable strategy to overcome the GPU memory barrier via leveraging both system memory (i.e., CPU DRAM) and storage space (i.e., solid-state devices, SSDs), its design primarily targets model-centric performance issues. As a result, key system-level issues, including system memory fragmentation, inefficient pinned buffer allocation, peak CPU usage spikes, and file system overhead, remain unaddressed, stifling scalability and inflating costs. Such an observation motivates this paper to introduce MemAscend, a framework that systematically tackles the underexplored system memory bottlenecks in SSD-offloaded LLM training, with a focus on resource-constrained environments. By streamlining pinned-memory allocation, eradicating fragmentation, and mitigating peak overhead, MemAscend reclaims a substantial system memory budget, enabling larger models, longer context windows, and higher batch sizes without exceeding modest hardware limits. Across diverse LLM benchmarks, MemAscend reduces peak system-memory consumption by an average of 55.7% compared with standard SSD offloading techniques, lowering the hardware barrier for fine-tuning and unlocking new possibilities for cost-effective large-scale training on limited-resource machines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23236v1">Towards LLM-Empowered Fine-Grained Speech Descriptors for Explainable Emotion Recognition</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 Accepted by INTERSPEECH2025
    </div>
    <details class="paper-abstract">
      This paper presents a novel end-to-end LLM-empowered explainable speech emotion recognition (SER) approach. Fine-grained speech emotion descriptor (SED) features, e.g., pitch, tone and emphasis, are disentangled from HuBERT SSL representations via alternating LLM fine-tuning to joint SER-SED prediction and ASR tasks. VAE compressed HuBERT features derived via Information Bottleneck (IB) are used to adjust feature granularity. Experiments on the IEMOCAP and MELD benchmarks demonstrate that our approach consistently outperforms comparable LLaMA-based SER baselines, including those using either (a) alternating multi-task fine-tuning alone or (b) feature disentanglement only. Statistically significant increase of SER unweighted accuracy by up to 4.0% and 3.7% absolute (5.4% and 6.6% relative) are obtained. More importantly, emotion descriptors offer further explainability for SER.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03475v2">am-ELO: A Stable Framework for Arena-based LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 ICML2025 Accepted
    </div>
    <details class="paper-abstract">
      Arena-based evaluation is a fundamental yet significant evaluation paradigm for modern AI models, especially large language models (LLMs). Existing framework based on ELO rating system suffers from the inevitable instability problem due to ranking inconsistency and the lack of attention to the varying abilities of annotators. In this paper, we introduce a novel stable arena framework to address these issues by enhancing the ELO Rating System. Specifically, we replace the iterative update method with a Maximum Likelihood Estimation (MLE) approach, m-ELO, and provide theoretical proof of the consistency and stability of the MLE approach for model ranking. Additionally, we proposed the am-ELO, which modify the Elo Rating's probability function to incorporate annotator abilities, enabling the simultaneous estimation of model scores and annotator reliability. Experiments demonstrate that this method ensures stability, proving that this framework offers a more robust, accurate, and stable evaluation method for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23219v1">Ghidorah: Fast LLM Inference on Edge with Speculative Decoding and Hetero-Core Parallelism</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      In-situ LLM inference on end-user devices has gained significant interest due to its privacy benefits and reduced dependency on external infrastructure. However, as the decoding process is memory-bandwidth-bound, the diverse processing units in modern end-user devices cannot be fully exploited, resulting in slow LLM inference. This paper presents Ghidorah, a LLM inference system for end-user devices with the unified memory architecture. The key idea of Ghidorah can be summarized in two steps: 1) leveraging speculative decoding approaches to enhance parallelism, and 2) ingeniously distributing workloads across multiple heterogeneous processing units to maximize computing power utilization. Ghidorah includes the hetero-core model parallelism (HCMP) architecture and the architecture-aware profiling (ARCA) approach. The HCMP architecture guides partitioning by leveraging the unified memory design of end-user devices and adapting to the hybrid computational demands of speculative decoding. The ARCA approach is used to determine the optimal speculative strategy and partitioning strategy, balancing acceptance rate with parallel capability to maximize the speedup. Additionally, we optimize sparse computation on ARM CPUs. Experimental results show that Ghidorah can achieve up to 7.6x speedup in the dominant LLM decoding phase compared to the sequential decoding approach in NVIDIA Jetson NX.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.17390v3">$T^5Score$: A Methodology for Automatically Assessing the Quality of LLM Generated Multi-Document Topic Sets</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 Published in the Findings of ACL 2025
    </div>
    <details class="paper-abstract">
      Using LLMs for Multi-Document Topic Extraction has recently gained popularity due to their apparent high-quality outputs, expressiveness, and ease of use. However, most existing evaluation practices are not designed for LLM-generated topics and result in low inter-annotator agreement scores, hindering the reliable use of LLMs for the task. To address this, we introduce $T^5Score$, an evaluation methodology that decomposes the quality of a topic set into quantifiable aspects, measurable through easy-to-perform annotation tasks. This framing enables a convenient, manual or automatic, evaluation procedure resulting in a strong inter-annotator agreement score. To substantiate our methodology and claims, we perform extensive experimentation on multiple datasets and report the results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23191v1">ExpeTrans: LLMs Are Experiential Transfer Learners</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 9 pages, 12 figs/tables
    </div>
    <details class="paper-abstract">
      Recent studies provide large language models (LLMs) with textual task-solving experiences via prompts to improve their performance. However, previous methods rely on substantial human labor or time to gather such experiences for each task, which is impractical given the growing variety of task types in user queries to LLMs. To address this issue, we design an autonomous experience transfer framework to explore whether LLMs can mimic human cognitive intelligence to autonomously transfer experience from existing source tasks to newly encountered target tasks. This not only allows the acquisition of experience without extensive costs of previous methods, but also offers a novel path for the generalization of LLMs. Experimental results on 13 datasets demonstrate that our framework effectively improves the performance of LLMs. Furthermore, we provide a detailed analysis of each module in the framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23187v1">Cross-Task Experiential Learning on LLM-based Multi-Agent Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 Work in Progress
    </div>
    <details class="paper-abstract">
      Large Language Model-based multi-agent systems (MAS) have shown remarkable progress in solving complex tasks through collaborative reasoning and inter-agent critique. However, existing approaches typically treat each task in isolation, resulting in redundant computations and limited generalization across structurally similar tasks. To address this, we introduce multi-agent cross-task experiential learning (MAEL), a novel framework that endows LLM-driven agents with explicit cross-task learning and experience accumulation. We model the task-solving workflow on a graph-structured multi-agent collaboration network, where agents propagate information and coordinate via explicit connectivity. During the experiential learning phase, we quantify the quality for each step in the task-solving workflow and store the resulting rewards along with the corresponding inputs and outputs into each agent's individual experience pool. During inference, agents retrieve high-reward, task-relevant experiences as few-shot examples to enhance the effectiveness of each reasoning step, thereby enabling more accurate and efficient multi-agent collaboration. Experimental results on diverse datasets demonstrate that MAEL empowers agents to learn from prior task experiences effectively-achieving faster convergence and producing higher-quality solutions on current tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17656v2">Too Consistent to Detect: A Study of Self-Consistent Errors in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) often generate plausible but incorrect content, error detection has become increasingly critical to ensure truthfulness. However, existing detection methods often overlook a critical problem we term as self-consistent error, where LLMs repeatly generate the same incorrect response across multiple stochastic samples. This work formally defines self-consistent errors and evaluates mainstream detection methods on them. Our investigation reveals two key findings: (1) Unlike inconsistent errors, whose frequency diminishes significantly as LLM scale increases, the frequency of self-consistent errors remains stable or even increases. (2) All four types of detection methshods significantly struggle to detect self-consistent errors. These findings reveal critical limitations in current detection methods and underscore the need for improved methods. Motivated by the observation that self-consistent errors often differ across LLMs, we propose a simple but effective cross-model probe method that fuses hidden state evidence from an external verifier LLM. Our method significantly enhances performance on self-consistent errors across three LLM families.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04779v4">Can LLMs Reason About Program Semantics? A Comprehensive Evaluation of LLMs on Formal Specification Inference</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 Accepted to ACL 2025 (Main Conference)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly being used to automate programming tasks. Yet, LLMs' capabilities in reasoning about program semantics are still inadequately studied, leaving significant potential for further exploration. This paper introduces FormalBench, a comprehensive benchmark designed to evaluate LLMs' reasoning abilities on program semantics, particularly via the task of synthesizing formal program specifications to assist verifying program correctness. This task requires both comprehensive reasoning over all possible program executions and the generation of precise, syntactically correct expressions that adhere to formal syntax and semantics. Using this benchmark, we evaluated the ability of LLMs in synthesizing consistent and complete specifications. Our findings show that LLMs perform well with simple control flows but struggle with more complex structures, especially loops, even with advanced prompting. Additionally, LLMs exhibit limited robustness against semantic-preserving transformations. We also highlight common failure patterns and design self-repair prompts, improving success rates by 25%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12583v2">LongFaith: Enhancing Long-Context Reasoning in LLMs with Faithful Synthetic Data</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Despite the growing development of long-context large language models (LLMs), data-centric approaches relying on synthetic data have been hindered by issues related to faithfulness, which limit their effectiveness in enhancing model performance on tasks such as long-context reasoning and question answering (QA). These challenges are often exacerbated by misinformation caused by lack of verification, reasoning without attribution, and potential knowledge conflicts. We propose LongFaith, a novel pipeline for synthesizing faithful long-context reasoning instruction datasets. By integrating ground truth and citation-based reasoning prompts, we eliminate distractions and improve the accuracy of reasoning chains, thus mitigating the need for costly verification processes. We open-source two synthesized datasets, LongFaith-SFT and LongFaith-PO, which systematically address multiple dimensions of faithfulness, including verified reasoning, attribution, and contextual grounding. Extensive experiments on multi-hop reasoning datasets and LongBench demonstrate that models fine-tuned on these datasets significantly improve performance. Our ablation studies highlight the scalability and adaptability of the LongFaith pipeline, showcasing its broad applicability in developing long-context LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.16359v3">Human-Readable Adversarial Prompts: An Investigation into LLM Vulnerabilities Using Situational Context</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 arXiv admin note: text overlap with arXiv:2407.14644
    </div>
    <details class="paper-abstract">
      As the AI systems become deeply embedded in social media platforms, we've uncovered a concerning security vulnerability that goes beyond traditional adversarial attacks. It becomes important to assess the risks of LLMs before the general public use them on social media platforms to avoid any adverse impacts. Unlike obvious nonsensical text strings that safety systems can easily catch, our work reveals that human-readable situation-driven adversarial full-prompts that leverage situational context are effective but much harder to detect. We found that skilled attackers can exploit the vulnerabilities in open-source and proprietary LLMs to make a malicious user query safe for LLMs, resulting in generating a harmful response. This raises an important question about the vulnerabilities of LLMs. To measure the robustness against human-readable attacks, which now present a potent threat, our research makes three major contributions. First, we developed attacks that use movie scripts as situational contextual frameworks, creating natural-looking full-prompts that trick LLMs into generating harmful content. Second, we developed a method to transform gibberish adversarial text into readable, innocuous content that still exploits vulnerabilities when used within the full-prompts. Finally, we enhanced the AdvPrompter framework with p-nucleus sampling to generate diverse human-readable adversarial texts that significantly improve attack effectiveness against models like GPT-3.5-Turbo-0125 and Gemma-7b. Our findings show that these systems can be manipulated to operate beyond their intended ethical boundaries when presented with seemingly normal prompts that contain hidden adversarial elements. By identifying these vulnerabilities, we aim to drive the development of more robust safety mechanisms that can withstand sophisticated attacks in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04209v2">To Judge or not to Judge: Using LLM Judgements for Advertiser Keyphrase Relevance at eBay</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      E-commerce sellers are recommended keyphrases based on their inventory on which they advertise to increase buyer engagement (clicks/sales). The relevance of advertiser keyphrases plays an important role in preventing the inundation of search systems with numerous irrelevant items that compete for attention in auctions, in addition to maintaining a healthy seller perception. In this work, we describe the shortcomings of training Advertiser keyphrase relevance filter models on click/sales/search relevance signals and the importance of aligning with human judgment, as sellers have the power to adopt or reject said keyphrase recommendations. In this study, we frame Advertiser keyphrase relevance as a complex interaction between 3 dynamical systems -- seller judgment, which influences seller adoption of our product, Advertising, which provides the keyphrases to bid on, and Search, who holds the auctions for the same keyphrases. This study discusses the practicalities of using human judgment via a case study at eBay Advertising and demonstrate that using LLM-as-a-judge en-masse as a scalable proxy for seller judgment to train our relevance models achieves a better harmony across the three systems -- provided that they are bound by a meticulous evaluation framework grounded in business metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16561v2">FutureGen: LLM-RAG Approach to Generate the Future Work of Scientific Article</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 19 pages, 5 figures
    </div>
    <details class="paper-abstract">
      The future work section of a scientific article outlines potential research directions by identifying gaps and limitations of a current study. This section serves as a valuable resource for early-career researchers seeking unexplored areas and experienced researchers looking for new projects or collaborations. In this study, we generate future work suggestions from key sections of a scientific article alongside related papers and analyze how the trends have evolved. We experimented with various Large Language Models (LLMs) and integrated Retrieval-Augmented Generation (RAG) to enhance the generation process. We incorporate a LLM feedback mechanism to improve the quality of the generated content and propose an LLM-as-a-judge approach for evaluation. Our results demonstrated that the RAG-based approach with LLM feedback outperforms other methods evaluated through qualitative and quantitative metrics. Moreover, we conduct a human evaluation to assess the LLM as an extractor and judge. The code and dataset for this project are here, code: HuggingFace
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19510v2">LLM Meets Scene Graph: Can Large Language Models Understand and Generate Scene Graphs? A Benchmark and Empirical Study</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 ACL 2025
    </div>
    <details class="paper-abstract">
      The remarkable reasoning and generalization capabilities of Large Language Models (LLMs) have paved the way for their expanding applications in embodied AI, robotics, and other real-world tasks. To effectively support these applications, grounding in spatial and temporal understanding in multimodal environments is essential. To this end, recent works have leveraged scene graphs, a structured representation that encodes entities, attributes, and their relationships in a scene. However, a comprehensive evaluation of LLMs' ability to utilize scene graphs remains limited. In this work, we introduce Text-Scene Graph (TSG) Bench, a benchmark designed to systematically assess LLMs' ability to (1) understand scene graphs and (2) generate them from textual narratives. With TSG Bench we evaluate 11 LLMs and reveal that, while models perform well on scene graph understanding, they struggle with scene graph generation, particularly for complex narratives. Our analysis indicates that these models fail to effectively decompose discrete scenes from a complex narrative, leading to a bottleneck when generating scene graphs. These findings underscore the need for improved methodologies in scene graph generation and provide valuable insights for future research. The demonstration of our benchmark is available at https://tsg-bench.netlify.app. Additionally, our code and evaluation data are publicly available at https://github.com/docworlds/tsg-bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20921v2">Automatic Transmission for LLM Tiers: Optimizing Cost and Accuracy in Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 ACL 2025 (Findings)
    </div>
    <details class="paper-abstract">
      LLM providers typically offer multiple LLM tiers, varying in performance and price. As NLP tasks become more complex and modularized, selecting the suitable LLM tier for each subtask is a key challenge to balance between cost and performance. To address the problem, we introduce LLM Automatic Transmission (LLM-AT) framework that automatically selects LLM tiers without training. LLM-AT consists of Starter, Generator, and Judge. The starter selects the initial LLM tier expected to solve the given question, the generator produces a response using the LLM of the selected tier, and the judge evaluates the validity of the response. If the response is invalid, LLM-AT iteratively upgrades to a higher-tier model, generates a new response, and re-evaluates until a valid response is obtained. Additionally, we propose accuracy estimator, which enables the suitable initial LLM tier selection without training. Given an input question, accuracy estimator estimates the expected accuracy of each LLM tier by computing the valid response rate across top-k similar queries from past inference records. Experiments demonstrate that LLM-AT achieves superior performance while reducing costs, making it a practical solution for real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23061v1">DINGO: Constrained Inference for Diffusion LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 DINGO an algorithm to provably apply constraints to diffusion LLM generations
    </div>
    <details class="paper-abstract">
      Diffusion LLMs have emerged as a promising alternative to conventional autoregressive LLMs, offering significant potential for improved runtime efficiency. However, existing diffusion models lack the ability to provably enforce user-specified formal constraints, such as regular expressions, which makes them unreliable for tasks that require structured outputs, such as fixed-schema JSON generation. Unlike autoregressive models that generate tokens sequentially, diffusion LLMs predict a block of tokens in parallel. This parallelism makes traditional constrained decoding algorithms, which are designed for sequential token prediction, ineffective at preserving the true output distribution. To address this limitation, we propose DINGO, a dynamic programming-based constrained decoding strategy that is both efficient and provably distribution-preserving. DINGO enables sampling of output strings with the highest probability under the model's predicted distribution, while strictly satisfying any user-specified regular expression. On standard symbolic math and JSON generation benchmarks, DINGO achieves up to a 68 percentage point improvement over unconstrained inference
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02322v2">HyperTree Planning: Enhancing LLM Reasoning via Hierarchical Thinking</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 arXiv admin note: text overlap with arXiv:2406.14228 by other authors
    </div>
    <details class="paper-abstract">
      Recent advancements have significantly enhanced the performance of large language models (LLMs) in tackling complex reasoning tasks, achieving notable success in domains like mathematical and logical reasoning. However, these methods encounter challenges with complex planning tasks, primarily due to extended reasoning steps, diverse constraints, and the challenge of handling multiple distinct sub-tasks. To address these challenges, we propose HyperTree Planning (HTP), a novel reasoning paradigm that constructs hypertree-structured planning outlines for effective planning. The hypertree structure enables LLMs to engage in hierarchical thinking by flexibly employing the divide-and-conquer strategy, effectively breaking down intricate reasoning steps, accommodating diverse constraints, and managing multiple distinct sub-tasks in a well-organized manner. We further introduce an autonomous planning framework that completes the planning process by iteratively refining and expanding the hypertree-structured planning outlines. Experiments demonstrate the effectiveness of HTP, achieving state-of-the-art accuracy on the TravelPlanner benchmark with Gemini-1.5-Pro, resulting in a 3.6 times performance improvement over o1-preview.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23049v1">DenoiseRotator: Enhance Pruning Robustness for LLMs via Importance Concentration</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Pruning is a widely used technique to compress large language models (LLMs) by removing unimportant weights, but it often suffers from significant performance degradation - especially under semi-structured sparsity constraints. Existing pruning methods primarily focus on estimating the importance of individual weights, which limits their ability to preserve critical capabilities of the model. In this work, we propose a new perspective: rather than merely selecting which weights to prune, we first redistribute parameter importance to make the model inherently more amenable to pruning. By minimizing the information entropy of normalized importance scores, our approach concentrates importance onto a smaller subset of weights, thereby enhancing pruning robustness. We instantiate this idea through DenoiseRotator, which applies learnable orthogonal transformations to the model's weight matrices. Our method is model-agnostic and can be seamlessly integrated with existing pruning techniques such as Magnitude, SparseGPT, and Wanda. Evaluated on LLaMA3, Qwen2.5, and Mistral models under 50% unstructured and 2:4 semi-structured sparsity, DenoiseRotator consistently improves perplexity and zero-shot accuracy. For instance, on LLaMA3-70B pruned with SparseGPT at 2:4 semi-structured sparsity, DenoiseRotator reduces the perplexity gap to the dense model by 58%, narrowing the degradation from 8.1 to 3.4 points. Codes are available at https://github.com/Axel-gu/DenoiseRotator.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23034v1">Case-Based Reasoning Enhances the Predictive Power of LLMs in Drug-Drug Interaction</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Drug-drug interaction (DDI) prediction is critical for treatment safety. While large language models (LLMs) show promise in pharmaceutical tasks, their effectiveness in DDI prediction remains challenging. Inspired by the well-established clinical practice where physicians routinely reference similar historical cases to guide their decisions through case-based reasoning (CBR), we propose CBR-DDI, a novel framework that distills pharmacological principles from historical cases to improve LLM reasoning for DDI tasks. CBR-DDI constructs a knowledge repository by leveraging LLMs to extract pharmacological insights and graph neural networks (GNNs) to model drug associations. A hybrid retrieval mechanism and dual-layer knowledge-enhanced prompting allow LLMs to effectively retrieve and reuse relevant cases. We further introduce a representative sampling strategy for dynamic case refinement. Extensive experiments demonstrate that CBR-DDI achieves state-of-the-art performance, with a significant 28.7% accuracy improvement over both popular LLMs and CBR baseline, while maintaining high interpretability and flexibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12600v2">On the Risk of Evidence Pollution for Malicious Social Text Detection in the Era of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Evidence-enhanced detectors present remarkable abilities in identifying malicious social text. However, the rise of large language models (LLMs) brings potential risks of evidence pollution to confuse detectors. This paper explores potential manipulation scenarios including basic pollution, and rephrasing or generating evidence by LLMs. To mitigate the negative impact, we propose three defense strategies from the data and model sides, including machine-generated text detection, a mixture of experts, and parameter updating. Extensive experiments on four malicious social text detection tasks with ten datasets illustrate that evidence pollution significantly compromises detectors, where the generating strategy causes up to a 14.4% performance drop. Meanwhile, the defense strategies could mitigate evidence pollution, but they faced limitations for practical employment. Further analysis illustrates that polluted evidence (i) is of high quality, evaluated by metrics and humans; (ii) would compromise the model calibration, increasing expected calibration error up to 21.6%; and (iii) could be integrated to amplify the negative impact, especially for encoder-based LMs, where the accuracy drops by 21.8%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.09187v3">GuardAgent: Safeguard LLM Agents by a Guard Agent via Knowledge-Enabled Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 Accepted to ICML 2025
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language model (LLM) agents has raised new concerns regarding their safety and security. In this paper, we propose GuardAgent, the first guardrail agent to protect target agents by dynamically checking whether their actions satisfy given safety guard requests. Specifically, GuardAgent first analyzes the safety guard requests to generate a task plan, and then maps this plan into guardrail code for execution. By performing the code execution, GuardAgent can deterministically follow the safety guard request and safeguard target agents. In both steps, an LLM is utilized as the reasoning component, supplemented by in-context demonstrations retrieved from a memory module storing experiences from previous tasks. In addition, we propose two novel benchmarks: EICU-AC benchmark to assess the access control for healthcare agents and Mind2Web-SC benchmark to evaluate the safety policies for web agents. We show that GuardAgent effectively moderates the violation actions for different types of agents on these two benchmarks with over 98% and 83% guardrail accuracies, respectively. Project page: https://guardagent.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23022v1">SCORPIO: Serving the Right Requests at the Right Time for Heterogeneous SLOs in LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Existing Large Language Model (LLM) serving systems prioritize maximum throughput. They often neglect Service Level Objectives (SLOs) such as Time to First Token (TTFT) and Time Per Output Token (TPOT), which leads to suboptimal SLO attainment. This paper introduces SCORPIO, an SLO-oriented LLM serving system designed to maximize system goodput and SLO attainment for workloads with heterogeneous SLOs. Our core insight is to exploit SLO heterogeneity for adaptive scheduling across admission control, queue management, and batch selection. SCORPIO features a TTFT Guard, which employs least-deadline-first reordering and rejects unattainable requests, and a TPOT Guard, which utilizes a VBS-based admission control and a novel credit-based batching mechanism. Both guards are supported by a predictive module. Evaluations demonstrate that SCORPIO improves system goodput by up to 14.4X and SLO adherence by up to 46.5% compared to state-of-the-art baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23019v1">Stairway to Success: Zero-Shot Floor-Aware Object-Goal Navigation via LLM-Driven Coarse-to-Fine Exploration</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 34 pages, 12 figures, 10 tables
    </div>
    <details class="paper-abstract">
      Object-Goal Navigation (OGN) remains challenging in real-world, multi-floor environments and under open-vocabulary object descriptions. We observe that most episodes in widely used benchmarks such as HM3D and MP3D involve multi-floor buildings, with many requiring explicit floor transitions. However, existing methods are often limited to single-floor settings or predefined object categories. To address these limitations, we tackle two key challenges: (1) efficient cross-level planning and (2) zero-shot object-goal navigation (ZS-OGN), where agents must interpret novel object descriptions without prior exposure. We propose ASCENT, a framework that combines a Multi-Floor Spatial Abstraction module for hierarchical semantic mapping and a Coarse-to-Fine Frontier Reasoning module leveraging Large Language Models (LLMs) for context-aware exploration, without requiring additional training on new object semantics or locomotion data. Our method outperforms state-of-the-art ZS-OGN approaches on HM3D and MP3D benchmarks while enabling efficient multi-floor navigation. We further validate its practicality through real-world deployment on a quadruped robot, achieving successful object exploration across unseen floors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23013v1">Scalable Complexity Control Facilitates Reasoning Ability of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      The reasoning ability of large language models (LLMs) has been rapidly advancing in recent years, attracting interest in more fundamental approaches that can reliably enhance their generalizability. This work demonstrates that model complexity control, conveniently implementable by adjusting the initialization rate and weight decay coefficient, improves the scaling law of LLMs consistently over varying model sizes and data sizes. This gain is further illustrated by comparing the benchmark performance of 2.4B models pretrained on 1T tokens with different complexity hyperparameters. Instead of fixing the initialization std, we found that a constant initialization rate (the exponent of std) enables the scaling law to descend faster in both model and data sizes. These results indicate that complexity control is a promising direction for the continual advancement of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22998v1">LLM Agents for Bargaining with Utility-based Feedback</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Bargaining, a critical aspect of real-world interactions, presents challenges for large language models (LLMs) due to limitations in strategic depth and adaptation to complex human factors. Existing benchmarks often fail to capture this real-world complexity. To address this and enhance LLM capabilities in realistic bargaining, we introduce a comprehensive framework centered on utility-based feedback. Our contributions are threefold: (1) BargainArena, a novel benchmark dataset with six intricate scenarios (e.g., deceptive practices, monopolies) to facilitate diverse strategy modeling; (2) human-aligned, economically-grounded evaluation metrics inspired by utility theory, incorporating agent utility and negotiation power, which implicitly reflect and promote opponent-aware reasoning (OAR); and (3) a structured feedback mechanism enabling LLMs to iteratively refine their bargaining strategies. This mechanism can positively collaborate with in-context learning (ICL) prompts, including those explicitly designed to foster OAR. Experimental results show that LLMs often exhibit negotiation strategies misaligned with human preferences, and that our structured feedback mechanism significantly improves their performance, yielding deeper strategic and opponent-aware reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22375v2">Pangu Embedded: An Efficient Dual-system LLM Reasoner with Metacognition</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      This work presents Pangu Embedded, an efficient Large Language Model (LLM) reasoner developed on Ascend Neural Processing Units (NPUs), featuring flexible fast and slow thinking capabilities. Pangu Embedded addresses the significant computational costs and inference latency challenges prevalent in existing reasoning-optimized LLMs. We propose a two-stage training framework for its construction. In Stage 1, the model is finetuned via an iterative distillation process, incorporating inter-iteration model merging to effectively aggregate complementary knowledge. This is followed by reinforcement learning on Ascend clusters, optimized by a latency-tolerant scheduler that combines stale synchronous parallelism with prioritized data queues. The RL process is guided by a Multi-source Adaptive Reward System (MARS), which generates dynamic, task-specific reward signals using deterministic metrics and lightweight LLM evaluators for mathematics, coding, and general problem-solving tasks. Stage 2 introduces a dual-system framework, endowing Pangu Embedded with a "fast" mode for routine queries and a deeper "slow" mode for complex inference. This framework offers both manual mode switching for user control and an automatic, complexity-aware mode selection mechanism that dynamically allocates computational resources to balance latency and reasoning depth. Experimental results on benchmarks including AIME 2024, GPQA, and LiveCodeBench demonstrate that Pangu Embedded with 7B parameters, outperforms similar-size models like Qwen3-8B and GLM4-9B. It delivers rapid responses and state-of-the-art reasoning quality within a single, unified model architecture, highlighting a promising direction for developing powerful yet practically deployable LLM reasoners.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22571v2">Agent-UniRAG: A Trainable Open-Source LLM Agent Framework for Unified Retrieval-Augmented Generation Systems</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      This paper presents a novel approach for unified retrieval-augmented generation (RAG) systems using the recent emerging large language model (LLM) agent concept. Specifically, Agent LLM, which utilizes LLM as fundamental controllers, has become a promising approach to enable the interpretability of RAG tasks, especially for complex reasoning question-answering systems (e.g., multi-hop queries). Nonetheless, previous works mainly focus on solving RAG systems with either single-hop or multi-hop approaches separately, which limits the application of those approaches to real-world applications. In this study, we propose a trainable agent framework called Agent-UniRAG for unified retrieval-augmented LLM systems, which enhances the effectiveness and interpretability of RAG systems. The main idea is to design an LLM agent framework to solve RAG tasks step-by-step based on the complexity of the inputs, simultaneously including single-hop and multi-hop queries in an end-to-end manner. Furthermore, we introduce SynAgent-RAG, a synthetic dataset to enable the proposed agent framework for small open-source LLMs (e.g., Llama-3-8B). The results show comparable performances with closed-source and larger open-source LLMs across various RAG benchmarks. Our source code and dataset are publicly available for further exploitation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18542v2">Business as Rulesual: A Benchmark and Framework for Business Rule Flow Modeling with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Process mining aims to discover, monitor and optimize the actual behaviors of real processes. While prior work has mainly focused on extracting procedural action flows from instructional texts, rule flows embedded in business documents remain underexplored. To this end, we introduce a novel annotated Chinese dataset, BPRF, which contains 50 business process documents with 326 explicitly labeled business rules across multiple domains. Each rule is represented as a <Condition, Action> pair, and we annotate logical dependencies between rules (sequential, conditional, or parallel). We also propose ExIde, a framework for automatic business rule extraction and dependency relationship identification using large language models (LLMs). We evaluate ExIde using 12 state-of-the-art (SOTA) LLMs on the BPRF dataset, benchmarking performance on both rule extraction and dependency classification tasks of current LLMs. Our results demonstrate the effectiveness of ExIde in extracting structured business rules and analyzing their interdependencies for current SOTA LLMs, paving the way for more automated and interpretable business process automation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13330v2">LEAVS: An LLM-based Labeler for Abdominal CT Supervision</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 Early acceptance (top 9% of submissions) for MICCAI 2025
    </div>
    <details class="paper-abstract">
      Extracting structured labels from radiology reports has been employed to create vision models to simultaneously detect several types of abnormalities. However, existing works focus mainly on the chest region. Few works have been investigated on abdominal radiology reports due to more complex anatomy and a wider range of pathologies in the abdomen. We propose LEAVS (Large language model Extractor for Abdominal Vision Supervision). This labeler can annotate the certainty of presence and the urgency of seven types of abnormalities for nine abdominal organs on CT radiology reports. To ensure broad coverage, we chose abnormalities that encompass most of the finding types from CT reports. Our approach employs a specialized chain-of-thought prompting strategy for a locally-run LLM using sentence extraction and multiple-choice questions in a tree-based decision system. We demonstrate that the LLM can extract several abnormality types across abdominal organs with an average F1 score of 0.89, significantly outperforming competing labelers and humans. Additionally, we show that extraction of urgency labels achieved performance comparable to human annotations. Finally, we demonstrate that the abnormality labels contain valuable information for training a single vision model that classifies several organs as normal or abnormal. We release our code and structured annotations for a public CT dataset containing over 1,000 CT volumes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22961v1">ToMAP: Training Opponent-Aware LLM Persuaders with Theory of Mind</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown promising potential in persuasion, but existing works on training LLM persuaders are still preliminary. Notably, while humans are skilled in modeling their opponent's thoughts and opinions proactively and dynamically, current LLMs struggle with such Theory of Mind (ToM) reasoning, resulting in limited diversity and opponent awareness. To address this limitation, we introduce Theory of Mind Augmented Persuader (ToMAP), a novel approach for building more flexible persuader agents by incorporating two theory of mind modules that enhance the persuader's awareness and analysis of the opponent's mental state. Specifically, we begin by prompting the persuader to consider possible objections to the target central claim, and then use a text encoder paired with a trained MLP classifier to predict the opponent's current stance on these counterclaims. Our carefully designed reinforcement learning schema enables the persuader learns how to analyze opponent-related information and utilize it to generate more effective arguments. Experiments show that the ToMAP persuader, while containing only 3B parameters, outperforms much larger baselines, like GPT-4o, with a relative gain of 39.4% across multiple persuadee models and diverse corpora. Notably, ToMAP exhibits complex reasoning chains and reduced repetition during training, which leads to more diverse and effective arguments. The opponent-aware feature of ToMAP also makes it suitable for long conversations and enables it to employ more logical and opponent-aware strategies. These results underscore our method's effectiveness and highlight its potential for developing more persuasive language agents. Code is available at: https://github.com/ulab-uiuc/ToMAP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22959v1">LLM-based HSE Compliance Assessment: Benchmark, Performance, and Advancements</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Health, Safety, and Environment (HSE) compliance assessment demands dynamic real-time decision-making under complicated regulations and complex human-machine-environment interactions. While large language models (LLMs) hold significant potential for decision intelligence and contextual dialogue, their capacity for domain-specific knowledge in HSE and structured legal reasoning remains underexplored. We introduce HSE-Bench, the first benchmark dataset designed to evaluate the HSE compliance assessment capabilities of LLM. HSE-Bench comprises over 1,000 manually curated questions drawn from regulations, court cases, safety exams, and fieldwork videos, and integrates a reasoning flow based on Issue spotting, rule Recall, rule Application, and rule Conclusion (IRAC) to assess the holistic reasoning pipeline. We conduct extensive evaluations on different prompting strategies and more than 10 LLMs, including foundation models, reasoning models and multimodal vision models. The results show that, although current LLMs achieve good performance, their capabilities largely rely on semantic matching rather than principled reasoning grounded in the underlying HSE compliance context. Moreover, their native reasoning trace lacks the systematic legal reasoning required for rigorous HSE compliance assessment. To alleviate these, we propose a new prompting technique, Reasoning of Expert (RoE), which guides LLMs to simulate the reasoning process of different experts for compliance assessment and reach a more accurate unified decision. We hope our study highlights reasoning gaps in LLMs for HSE compliance and inspires further research on related tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22956v1">LLMs for Argument Mining: Detection, Extraction, and Relationship Classification of pre-defined Arguments in Online Comments</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Automated large-scale analysis of public discussions around contested issues like abortion requires detecting and understanding the use of arguments. While Large Language Models (LLMs) have shown promise in language processing tasks, their performance in mining topic-specific, pre-defined arguments in online comments remains underexplored. We evaluate four state-of-the-art LLMs on three argument mining tasks using datasets comprising over 2,000 opinion comments across six polarizing topics. Quantitative evaluation suggests an overall strong performance across the three tasks, especially for large and fine-tuned LLMs, albeit at a significant environmental cost. However, a detailed error analysis revealed systematic shortcomings on long and nuanced comments and emotionally charged language, raising concerns for downstream applications like content moderation or opinion analysis. Our results highlight both the promise and current limitations of LLMs for automated argument analysis in online comments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16814v2">Understanding Bias Reinforcement in LLM Agents Debate</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 ICML 2025
    </div>
    <details class="paper-abstract">
      Large Language Models $($LLMs$)$ solve complex problems using training-free methods like prompt engineering and in-context learning, yet ensuring reasoning correctness remains challenging. While self-correction methods such as self-consistency and self-refinement aim to improve reliability, they often reinforce biases due to the lack of effective feedback mechanisms. Multi-Agent Debate $($MAD$)$ has emerged as an alternative, but we identify two key limitations: bias reinforcement, where debate amplifies model biases instead of correcting them, and lack of perspective diversity, as all agents share the same model and reasoning patterns, limiting true debate effectiveness. To systematically evaluate these issues, we introduce $\textit{MetaNIM Arena}$, a benchmark designed to assess LLMs in adversarial strategic decision-making, where dynamic interactions influence optimal decisions. To overcome MAD's limitations, we propose $\textbf{DReaMAD}$ $($$\textbf{D}$iverse $\textbf{Rea}$soning via $\textbf{M}$ulti-$\textbf{A}$gent $\textbf{D}$ebate with Refined Prompt$)$, a novel framework that $(1)$ refines LLM's strategic prior knowledge to improve reasoning quality and $(2)$ promotes diverse viewpoints within a single model by systematically modifying prompts, reducing bias. Empirical results show that $\textbf{DReaMAD}$ significantly improves decision accuracy, reasoning diversity, and bias mitigation across multiple strategic tasks, establishing it as a more effective approach for LLM-based decision-making.
    </details>
</div>
