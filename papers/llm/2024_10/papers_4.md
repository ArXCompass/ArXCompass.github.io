# llm - 2024_10

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- Part 4
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15828v1">LLM4GRN: Discovering Causal Gene Regulatory Networks with LLMs -- Evaluation through Synthetic Data Generation</a></div>
    <div class="paper-meta">
      📅 2024-10-21
    </div>
    <details class="paper-abstract">
      Gene regulatory networks (GRNs) represent the causal relationships between transcription factors (TFs) and target genes in single-cell RNA sequencing (scRNA-seq) data. Understanding these networks is crucial for uncovering disease mechanisms and identifying therapeutic targets. In this work, we investigate the potential of large language models (LLMs) for GRN discovery, leveraging their learned biological knowledge alone or in combination with traditional statistical methods. We develop a task-based evaluation strategy to address the challenge of unavailable ground truth causal graphs. Specifically, we use the GRNs suggested by LLMs to guide causal synthetic data generation and compare the resulting data against the original dataset. Our statistical and biological assessments show that LLMs can support statistical modeling and data synthesis for biological research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.06062v3">LLM-based SPARQL Query Generation from Natural Language over Federated Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2024-10-21
    </div>
    <details class="paper-abstract">
      We introduce a Retrieval-Augmented Generation (RAG) system for translating user questions into accurate federated SPARQL queries over bioinformatics knowledge graphs (KGs) leveraging Large Language Models (LLMs). To enhance accuracy and reduce hallucinations in query generation, our system utilises metadata from the KGs, including query examples and schema information, and incorporates a validation step to correct generated queries. The system is available online at chat.expasy.org.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.12831v2">Truth is Universal: Robust Detection of Lies in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-21
      | 💬 NeurIPS 2024 poster
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionised natural language processing, exhibiting impressive human-like capabilities. In particular, LLMs are capable of "lying", knowingly outputting false statements. Hence, it is of interest and importance to develop methods to detect when LLMs lie. Indeed, several authors trained classifiers to detect LLM lies based on their internal model activations. However, other researchers showed that these classifiers may fail to generalise, for example to negated statements. In this work, we aim to develop a robust method to detect when an LLM is lying. To this end, we make the following key contributions: (i) We demonstrate the existence of a two-dimensional subspace, along which the activation vectors of true and false statements can be separated. Notably, this finding is universal and holds for various LLMs, including Gemma-7B, LLaMA2-13B, Mistral-7B and LLaMA3-8B. Our analysis explains the generalisation failures observed in previous studies and sets the stage for more robust lie detection; (ii) Building upon (i), we construct an accurate LLM lie detector. Empirically, our proposed classifier achieves state-of-the-art performance, attaining 94% accuracy in both distinguishing true from false factual statements and detecting lies generated in real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13699v2">Unconstrained Model Merging for Enhanced LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2024-10-21
      | 💬 Under review, correct typos
    </div>
    <details class="paper-abstract">
      Recent advancements in building domain-specific large language models (LLMs) have shown remarkable success, especially in tasks requiring reasoning abilities like logical inference over complex relationships and multi-step problem solving. However, creating a powerful all-in-one LLM remains challenging due to the need for proprietary data and vast computational resources. As a resource-friendly alternative, we explore the potential of merging multiple expert models into a single LLM. Existing studies on model merging mainly focus on generalist LLMs instead of domain experts, or the LLMs under the same architecture and size. In this work, we propose an unconstrained model merging framework that accommodates both homogeneous and heterogeneous model architectures with a focus on reasoning tasks. A fine-grained layer-wise weight merging strategy is designed for homogeneous models merging, while heterogeneous model merging is built upon the probabilistic distribution knowledge derived from instruction-response fine-tuning data. Across 7 benchmarks and 9 reasoning-optimized LLMs, we reveal key findings that combinatorial reasoning emerges from merging which surpasses simple additive effects. We propose that unconstrained model merging could serve as a foundation for decentralized LLMs, marking a notable progression from the existing centralized LLM framework. This evolution could enhance wider participation and stimulate additional advancement in the field of artificial intelligence, effectively addressing the constraints posed by centralized models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15690v1">Efficient Terminology Integration for LLM-based Translation in Specialized Domains</a></div>
    <div class="paper-meta">
      📅 2024-10-21
      | 💬 Accepted to WMT 2024
    </div>
    <details class="paper-abstract">
      Traditional machine translation methods typically involve training models directly on large parallel corpora, with limited emphasis on specialized terminology. However, In specialized fields such as patent, finance, or biomedical domains, terminology is crucial for translation, with many terms that needs to be translated following agreed-upon conventions. In this paper we introduce a methodology that efficiently trains models with a smaller amount of data while preserving the accuracy of terminology translation. We achieve this through a systematic process of term extraction and glossary creation using the Trie Tree algorithm, followed by data reconstruction to teach the LLM how to integrate these specialized terms. This methodology enhances the model's ability to handle specialized terminology and ensures high-quality translations, particularly in fields where term consistency is crucial. Our approach has demonstrated exceptional performance, achieving the highest translation score among participants in the WMT patent task to date, showcasing its effectiveness and broad applicability in specialized translation domains where general methods often fall short.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.16756v2">How Well Do LLMs Handle Cantonese? Benchmarking Cantonese Capabilities of Large Language Models</a></div>
    <div class="paper-meta">
      📅 2024-10-21
    </div>
    <details class="paper-abstract">
      The rapid evolution of large language models (LLMs) has transformed the competitive landscape in natural language processing (NLP), particularly for English and other data-rich languages. However, underrepresented languages like Cantonese, spoken by over 85 million people, face significant development gaps, which is particularly concerning given the economic significance of the Guangdong-Hong Kong-Macau Greater Bay Area, and in substantial Cantonese-speaking populations in places like Singapore and North America. Despite its wide use, Cantonese has scant representation in NLP research, especially compared to other languages from similarly developed regions. To bridge these gaps, we outline current Cantonese NLP methods and introduce new benchmarks designed to evaluate LLM performance in factual generation, mathematical logic, complex reasoning, and general knowledge in Cantonese, which aim to advance open-source Cantonese LLM technology. We also propose future research directions and recommended models to enhance Cantonese LLM development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15512v1">Reverse Question Answering: Can an LLM Write a Question so Hard (or Bad) that it Can't Answer?</a></div>
    <div class="paper-meta">
      📅 2024-10-20
      | 💬 In-progress preprint
    </div>
    <details class="paper-abstract">
      Question answering (QA)-producing correct answers for input questions-is popular, but we test a reverse question answering (RQA) task: given an input answer, generate a question with that answer. Past work tests QA and RQA separately, but we test them jointly, comparing their difficulty, aiding benchmark design, and assessing reasoning consistency. 16 LLMs run QA and RQA with trivia questions/answers, showing: 1) Versus QA, LLMs are much less accurate in RQA for numerical answers, but slightly more accurate in RQA for textual answers; 2) LLMs often answer their own invalid questions from RQA accurately in QA, so RQA errors are not from knowledge gaps alone; 3) RQA errors correlate with question difficulty and inversely correlate with answer frequencies in the Dolma corpus; and 4) LLMs struggle to give valid multi-hop questions. By finding question and answer types yielding RQA errors, we suggest improvements for LLM RQA reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15484v1">"What is the value of {templates}?" Rethinking Document Information Extraction Datasets for LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-20
      | 💬 Accepted to EMNLP Findings 2024
    </div>
    <details class="paper-abstract">
      The rise of large language models (LLMs) for visually rich document understanding (VRDU) has kindled a need for prompt-response, document-based datasets. As annotating new datasets from scratch is labor-intensive, the existing literature has generated prompt-response datasets from available resources using simple templates. For the case of key information extraction (KIE), one of the most common VRDU tasks, past work has typically employed the template "What is the value for the {key}?". However, given the variety of questions encountered in the wild, simple and uniform templates are insufficient for creating robust models in research and industrial contexts. In this work, we present K2Q, a diverse collection of five datasets converted from KIE to a prompt-response format using a plethora of bespoke templates. The questions in K2Q can span multiple entities and be extractive or boolean. We empirically compare the performance of seven baseline generative models on K2Q with zero-shot prompting. We further compare three of these models when training on K2Q versus training on simpler templates to motivate the need of our work. We find that creating diverse and intricate KIE questions enhances the performance and robustness of VRDU models. We hope this work encourages future studies on data quality for generative model training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.13207v3">STaRK: Benchmarking LLM Retrieval on Textual and Relational Knowledge Bases</a></div>
    <div class="paper-meta">
      📅 2024-10-20
      | 💬 NeurIPS 2024 Track on Datasets and Benchmarks. 26 Pages, 6 Figures. Website: https://stark.stanford.edu/
    </div>
    <details class="paper-abstract">
      Answering real-world complex queries, such as complex product search, often requires accurate retrieval from semi-structured knowledge bases that involve blend of unstructured (e.g., textual descriptions of products) and structured (e.g., entity relations of products) information. However, many previous works studied textual and relational retrieval tasks as separate topics. To address the gap, we develop STARK, a large-scale Semi-structure retrieval benchmark on Textual and Relational Knowledge Bases. Our benchmark covers three domains: product search, academic paper search, and queries in precision medicine. We design a novel pipeline to synthesize realistic user queries that integrate diverse relational information and complex textual properties, together with their ground-truth answers (items). We conduct rigorous human evaluation to validate the quality of our synthesized queries. We further enhance the benchmark with high-quality human-generated queries to provide an authentic reference. STARK serves as a comprehensive testbed for evaluating the performance of retrieval systems driven by large language models (LLMs). Our experiments suggest that STARK presents significant challenges to the current retrieval and LLM systems, highlighting the need for more capable semi-structured retrieval systems. The benchmark data and code are available on https://github.com/snap-stanford/STaRK.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15440v1">Evaluating Consistencies in LLM responses through a Semantic Clustering of Question Answering</a></div>
    <div class="paper-meta">
      📅 2024-10-20
      | 💬 Accepted to the Trustworthy AI Workshop at IJCAI 2024
    </div>
    <details class="paper-abstract">
      In the realm of Large Language Model (LLM) functionalities, providing reliable information is paramount, yet reports suggest that LLM outputs lack consistency. This inconsistency, often at-tributed to randomness in token sampling, under-mines user trust as it leads to varying responses even for identical queries. In this paper, we present a new approach for evaluating semantic consistencies of LLM including comparison of alternative tech-niques. Our approach evaluates whether LLM re-sponses are semantically congruent for a given question, recognizing that as syntactically different sentences may convey the same meaning. Here-tofore, To enhance LLM consistency, two main approaches have been explored: Leverage external knowledge as context like the RAG pattern or use Zero-shot-CoT to improve performance of LLM itself. We apply our evaluation approach to these techniques, and demonstrate to compare the im-pact of these methods on LLM response con-sistency across different domains of question an-swering tasks. Using the TruthfulQA dataset to assess LLM responses, the study induces N re-sponses per question from the LLM and clusters semantically equivalent sentences to measure semantic consistency across 37 categories. Through this, it quantitatively analyzes the effectiveness of the aforementioned methods in improving LLM performance before and after their adoption.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15438v1">Unveiling and Consulting Core Experts in Retrieval-Augmented MoE-based LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-20
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) significantly improved the ability of Large Language Models (LLMs) to solve knowledge-intensive tasks. While existing research seeks to enhance RAG performance by retrieving higher-quality documents or designing RAG-specific LLMs, the internal mechanisms within LLMs that contribute to the effectiveness of RAG systems remain underexplored. In this paper, we aim to investigate these internal mechanisms within the popular Mixture-of-Expert (MoE)-based LLMs and demonstrate how to improve RAG by examining expert activations in these LLMs. Our controlled experiments reveal that several core groups of experts are primarily responsible for RAG-related behaviors. The activation of these core experts can signify the model's inclination towards external/internal knowledge and adjust its behavior. For instance, we identify core experts that can (1) indicate the sufficiency of the model's internal knowledge, (2) assess the quality of retrieved documents, and (3) enhance the model's ability to utilize context. Based on these findings, we propose several strategies to enhance RAG's efficiency and effectiveness through expert activation. Experimental results across various datasets and MoE-based LLMs show the effectiveness of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.08644v4">Tandem Transformers for Inference Efficient LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-20
    </div>
    <details class="paper-abstract">
      The autoregressive nature of conventional large language models (LLMs) inherently limits inference speed, as tokens are generated sequentially. While speculative and parallel decoding techniques attempt to mitigate this, they face limitations: either relying on less accurate smaller models for generation or failing to fully leverage the base LLM's representations. We introduce a novel architecture, Tandem transformers, to address these issues. This architecture uniquely combines (1) a small autoregressive model and (2) a large model operating in block mode (processing multiple tokens simultaneously). The small model's predictive accuracy is substantially enhanced by granting it attention to the large model's richer representations. On the PaLM2 pretraining dataset, a tandem of PaLM2-Bison and PaLM2-Gecko demonstrates a 3.3% improvement in next-token prediction accuracy over a standalone PaLM2-Gecko, offering a 1.16x speedup compared to a PaLM2-Otter model with comparable downstream performance. We further incorporate the tandem model within the speculative decoding (SPEED) framework where the large model validates tokens from the small model. This ensures that the Tandem of PaLM2-Bison and PaLM2-Gecko achieves substantial speedup (around 1.14x faster than using vanilla PaLM2-Gecko in SPEED) while maintaining identical downstream task accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15413v1">A Comprehensive Evaluation of Cognitive Biases in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-20
    </div>
    <details class="paper-abstract">
      We present a large-scale evaluation of 30 cognitive biases in 20 state-of-the-art large language models (LLMs) under various decision-making scenarios. Our contributions include a novel general-purpose test framework for reliable and large-scale generation of tests for LLMs, a benchmark dataset with 30,000 tests for detecting cognitive biases in LLMs, and a comprehensive assessment of the biases found in the 20 evaluated LLMs. Our work confirms and broadens previous findings suggesting the presence of cognitive biases in LLMs by reporting evidence of all 30 tested biases in at least some of the 20 LLMs. We publish our framework code to encourage future research on biases in LLMs: https://github.com/simonmalberg/cognitive-biases-in-llms
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15396v1">The Best Defense is a Good Offense: Countering LLM-Powered Cyberattacks</a></div>
    <div class="paper-meta">
      📅 2024-10-20
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) continue to evolve, their potential use in automating cyberattacks becomes increasingly likely. With capabilities such as reconnaissance, exploitation, and command execution, LLMs could soon become integral to autonomous cyber agents, capable of launching highly sophisticated attacks. In this paper, we introduce novel defense strategies that exploit the inherent vulnerabilities of attacking LLMs. By targeting weaknesses such as biases, trust in input, memory limitations, and their tunnel-vision approach to problem-solving, we develop techniques to mislead, delay, or neutralize these autonomous agents. We evaluate our defenses under black-box conditions, starting with single prompt-response scenarios and progressing to real-world tests using custom-built CTF machines. Our results show defense success rates of up to 90\%, demonstrating the effectiveness of turning LLM vulnerabilities into defensive strategies against LLM-driven cyber threats.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15393v1">CalibraEval: Calibrating Prediction Distribution to Mitigate Selection Bias in LLMs-as-Judges</a></div>
    <div class="paper-meta">
      📅 2024-10-20
      | 💬 13 pages
    </div>
    <details class="paper-abstract">
      The use of large language models (LLMs) as automated evaluation tools to assess the quality of generated natural language, known as LLMs-as-Judges, has demonstrated promising capabilities and is rapidly gaining widespread attention. However, when applied to pairwise comparisons of candidate responses, LLM-based evaluators often exhibit selection bias. Specifically, their judgments may become inconsistent when the option positions or ID tokens are swapped, compromising the effectiveness and fairness of the evaluation result. To address this challenge, we introduce CalibraEval, a novel label-free method for mitigating selection bias during inference. Specifically, CalibraEval reformulates debiasing as an optimization task aimed at adjusting observed prediction distributions to align with unbiased prediction distributions. To solve this optimization problem, we propose a non-parametric order-preserving algorithm (NOA). This algorithm leverages the partial order relationships between model prediction distributions, thereby eliminating the need for explicit labels and precise mathematical function modeling.Empirical evaluations of LLMs in multiple representative benchmarks demonstrate that CalibraEval effectively mitigates selection bias and improves performance compared to existing debiasing methods. This work marks a step toward building more robust and unbiased automated evaluation frameworks, paving the way for improved reliability in AI-driven assessments
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15352v1">CompAct: Compressed Activations for Memory-Efficient LLM Training</a></div>
    <div class="paper-meta">
      📅 2024-10-20
    </div>
    <details class="paper-abstract">
      We introduce CompAct, a technique that reduces peak memory utilization on GPU by 25-30% for pretraining and 50% for fine-tuning of LLMs. Peak device memory is a major limiting factor in training LLMs, with various recent works aiming to reduce model memory. However most works don't target the largest component of allocated memory during training: the model's compute graph, which is stored for the backward pass. By storing low-rank, compressed activations to be used in the backward pass we greatly reduce the required memory, unlike previous methods which only reduce optimizer overheads or the number of trained parameters. Our compression uses random projection matrices, thus avoiding additional memory overheads. Comparisons with previous techniques for either pretraining or fine-tuning show that CompAct substantially improves existing compute-performance tradeoffs. We expect CompAct's savings to scale even higher for larger models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15326v1">A Survey of Uncertainty Estimation in LLMs: Theory Meets Practice</a></div>
    <div class="paper-meta">
      📅 2024-10-20
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) continue to evolve, understanding and quantifying the uncertainty in their predictions is critical for enhancing application credibility. However, the existing literature relevant to LLM uncertainty estimation often relies on heuristic approaches, lacking systematic classification of the methods. In this survey, we clarify the definitions of uncertainty and confidence, highlighting their distinctions and implications for model predictions. On this basis, we integrate theoretical perspectives, including Bayesian inference, information theory, and ensemble strategies, to categorize various classes of uncertainty estimation methods derived from heuristic approaches. Additionally, we address challenges that arise when applying these methods to LLMs. We also explore techniques for incorporating uncertainty into diverse applications, including out-of-distribution detection, data annotation, and question clarification. Our review provides insights into uncertainty estimation from both definitional and theoretical angles, contributing to a comprehensive understanding of this critical aspect in LLMs. We aim to inspire the development of more reliable and effective uncertainty estimation approaches for LLMs in real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15311v1">Who is Undercover? Guiding LLMs to Explore Multi-Perspective Team Tactic in the Game</a></div>
    <div class="paper-meta">
      📅 2024-10-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are pivotal AI agents in complex tasks but still face challenges in open decision-making problems within complex scenarios. To address this, we use the language logic game ``Who is Undercover?'' (WIU) as an experimental platform to propose the Multi-Perspective Team Tactic (MPTT) framework. MPTT aims to cultivate LLMs' human-like language expression logic, multi-dimensional thinking, and self-perception in complex scenarios. By alternating speaking and voting sessions, integrating techniques like self-perspective, identity-determination, self-reflection, self-summary and multi-round find-teammates, LLM agents make rational decisions through strategic concealment and communication, fostering human-like trust. Preliminary results show that MPTT, combined with WIU, leverages LLMs' cognitive capabilities to create a decision-making framework that can simulate real society. This framework aids minority groups in communication and expression, promoting fairness and diversity in decision-making. Additionally, our Human-in-the-loop experiments demonstrate that LLMs can learn and align with human behaviors through interactive, indicating their potential for active participation in societal decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15308v1">LlamaLens: Specialized Multilingual LLM for Analyzing News and Social Media Content</a></div>
    <div class="paper-meta">
      📅 2024-10-20
      | 💬 LLMs, Multilingual, Language Diversity, Large Language Models, Social Media, News Media, Specialized LLMs, Fact-checking, Media Analysis
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable success as general-purpose task solvers across various fields, including NLP, healthcare, finance, and law. However, their capabilities remain limited when addressing domain-specific problems, particularly in downstream NLP tasks. Research has shown that models fine-tuned on instruction-based downstream NLP datasets outperform those that are not fine-tuned. While most efforts in this area have primarily focused on resource-rich languages like English and broad domains, little attention has been given to multilingual settings and specific domains. To address this gap, this study focuses on developing a specialized LLM, LlamaLens, for analyzing news and social media content in a multilingual context. To the best of our knowledge, this is the first attempt to tackle both domain specificity and multilinguality, with a particular focus on news and social media. Our experimental setup includes 19 tasks, represented by 52 datasets covering Arabic, English, and Hindi. We demonstrate that LlamaLens outperforms the current state-of-the-art (SOTA) on 16 testing sets, and achieves comparable performance on 10 sets. We make the models and resources publicly available for the research community.(https://huggingface.co/QCRI)
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15288v1">Attention Is All You Need for LLM-based Code Vulnerability Localization</a></div>
    <div class="paper-meta">
      📅 2024-10-20
    </div>
    <details class="paper-abstract">
      The rapid expansion of software systems and the growing number of reported vulnerabilities have emphasized the importance of accurately identifying vulnerable code segments. Traditional methods for vulnerability localization, such as manual code audits or rule-based tools, are often time-consuming and limited in scope, typically focusing on specific programming languages or types of vulnerabilities. In recent years, the introduction of large language models (LLMs) such as GPT and LLaMA has opened new possibilities for automating vulnerability detection. However, while LLMs show promise in this area, they face challenges, particularly in maintaining accuracy over longer code contexts. This paper introduces LOVA, a novel framework leveraging the self-attention mechanisms inherent in LLMs to enhance vulnerability localization. Our key insight is that self-attention mechanisms assign varying importance to different parts of the input, making it possible to track how much attention the model focuses on specific lines of code. In the context of vulnerability localization, the hypothesis is that vulnerable lines of code will naturally attract higher attention weights because they have a greater influence on the model's output. By systematically tracking changes in attention weights and focusing on specific lines of code, LOVA improves the precision of identifying vulnerable lines across various programming languages. Through rigorous experimentation and evaluation, we demonstrate that LOVA significantly outperforms existing LLM-based approaches, achieving up to a 5.3x improvement in F1-scores. LOVA also demonstrated strong scalability, with up to a 14.6x improvement in smart contract vulnerability localization across languages like C, Python, Java, and Solidity. Its robustness was proven through consistent performance across different LLM architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12568v2">Robust RL with LLM-Driven Data Synthesis and Policy Adaptation for Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2024-10-20
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into autonomous driving systems demonstrates strong common sense and reasoning abilities, effectively addressing the pitfalls of purely data-driven methods. Current LLM-based agents require lengthy inference times and face challenges in interacting with real-time autonomous driving environments. A key open question is whether we can effectively leverage the knowledge from LLMs to train an efficient and robust Reinforcement Learning (RL) agent. This paper introduces RAPID, a novel \underline{\textbf{R}}obust \underline{\textbf{A}}daptive \underline{\textbf{P}}olicy \underline{\textbf{I}}nfusion and \underline{\textbf{D}}istillation framework, which trains specialized mix-of-policy RL agents using data synthesized by an LLM-based driving agent and online adaptation. RAPID features three key designs: 1) utilization of offline data collected from an LLM agent to distil expert knowledge into RL policies for faster real-time inference; 2) introduction of robust distillation in RL to inherit both performance and robustness from LLM-based teacher; and 3) employment of a mix-of-policy approach for joint decision decoding with a policy adapter. Through fine-tuning via online environment interaction, RAPID reduces the forgetting of LLM knowledge while maintaining adaptability to different tasks. Extensive experiments demonstrate RAPID's capability to effectively integrate LLM knowledge into scaled-down RL policies in an efficient, adaptable, and robust way. Code and checkpoints will be made publicly available upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03071v2">Enhancing Short-Text Topic Modeling with LLM-Driven Context Expansion and Prefix-Tuned VAEs</a></div>
    <div class="paper-meta">
      📅 2024-10-19
      | 💬 EMNLP Findings 2024. arXiv admin note: substantial text overlap with arXiv:2310.15420
    </div>
    <details class="paper-abstract">
      Topic modeling is a powerful technique for uncovering hidden themes within a collection of documents. However, the effectiveness of traditional topic models often relies on sufficient word co-occurrence, which is lacking in short texts. Therefore, existing approaches, whether probabilistic or neural, frequently struggle to extract meaningful patterns from such data, resulting in incoherent topics. To address this challenge, we propose a novel approach that leverages large language models (LLMs) to extend short texts into more detailed sequences before applying topic modeling. To further improve the efficiency and solve the problem of semantic inconsistency from LLM-generated texts, we propose to use prefix tuning to train a smaller language model coupled with a variational autoencoder for short-text topic modeling. Our method significantly improves short-text topic modeling performance, as demonstrated by extensive experiments on real-world datasets with extreme data sparsity, outperforming current state-of-the-art topic models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15200v1">Exploring LLM Support for Generating IEC 61131-3 Graphic Language Programs</a></div>
    <div class="paper-meta">
      📅 2024-10-19
    </div>
    <details class="paper-abstract">
      The capabilities demonstrated by Large Language Models (LLMs) inspire researchers to integrate them into industrial production and automation. In the field of Programmable Logic Controller (PLC) programming, previous researchers have focused on using LLMs to generate Structured Text (ST) language, and created automatic programming workflows based on it. The IEC 61131 graphic programming languages, which still has the most users, have however been overlooked. In this paper we explore using LLMs to generate graphic languages in ASCII art to provide assistance to engineers. Our series of experiments indicate that, contrary to what researchers usually think, it is possible to generate a correct Sequential Function Chart (SFC) for simple requirements when LLM is provided with several examples. On the other hand, generating a Ladder Diagram (LD) automatically remains a challenge even for very simple use cases. The automatic conversion between LD and SFC without extra information also fails when using prompt engineering alone.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.19450v2">Secret Use of Large Language Model (LLM)</a></div>
    <div class="paper-meta">
      📅 2024-10-19
      | 💬 26 pages, 3 figures, and accepted at CSCW 2025
    </div>
    <details class="paper-abstract">
      The advancements of Large Language Models (LLMs) have decentralized the responsibility for the transparency of AI usage. Specifically, LLM users are now encouraged or required to disclose the use of LLM-generated content for varied types of real-world tasks. However, an emerging phenomenon, users' secret use of LLM, raises challenges in ensuring end users adhere to the transparency requirement. Our study used mixed-methods with an exploratory survey (125 real-world secret use cases reported) and a controlled experiment among 300 users to investigate the contexts and causes behind the secret use of LLMs. We found that such secretive behavior is often triggered by certain tasks, transcending demographic and personality differences among users. Task types were found to affect users' intentions to use secretive behavior, primarily through influencing perceived external judgment regarding LLM usage. Our results yield important insights for future work on designing interventions to encourage more transparent disclosure of the use of LLMs or other AI technologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15173v1">Uncovering Autoregressive LLM Knowledge of Thematic Fit in Event Representation</a></div>
    <div class="paper-meta">
      📅 2024-10-19
      | 💬 15 pages, 3 figures
    </div>
    <details class="paper-abstract">
      The thematic fit estimation task measures the compatibility between a predicate (typically a verb), an argument (typically a noun phrase), and a specific semantic role assigned to the argument. Previous state-of-the-art work has focused on modeling thematic fit through distributional or neural models of event representation, trained in a supervised fashion with indirect labels. In this work, we assess whether pre-trained autoregressive LLMs possess consistent, expressible knowledge about thematic fit. We evaluate both closed and open state-of-the-art LLMs on several psycholinguistic datasets, along three axes: (1) Reasoning Form: multi-step logical reasoning (chain-of-thought prompting) vs. simple prompting. (2) Input Form: providing context (generated sentences) vs. raw tuples <predicate, argument, role>. (3) Output Form: categorical vs. numeric. Our results show that chain-of-thought reasoning is more effective on datasets with self-explanatory semantic role labels, especially Location. Generated sentences helped only in few settings, and lowered results in many others. Predefined categorical (compared to numeric) output raised GPT's results across the board with few exceptions, but lowered Llama's. We saw that semantically incoherent generated sentences, which the models lack the ability to consistently filter out, hurt reasoning and overall performance too. Our GPT-powered methods set new state-of-the-art on all tested datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15168v1">An Electoral Approach to Diversify LLM-based Multi-Agent Collective Decision-Making</a></div>
    <div class="paper-meta">
      📅 2024-10-19
      | 💬 Accepted to EMNLP 2024
    </div>
    <details class="paper-abstract">
      Modern large language models (LLMs) have exhibited cooperative synergy on complex task-solving, and collective decision-making (CDM) is a pivotal component in LLM-based multi-agent collaboration frameworks. Our survey on 52 recent such systems uncovers a severe lack of diversity, with a heavy reliance on dictatorial and plurality voting for CDM. Through the lens of social choice theory, we scrutinize widely-adopted CDM methods and identify their limitations. To enrich current landscape of LLM-based CDM, we present GEDI, an electoral CDM module that incorporates various ordinal preferential voting mechanisms. Our empirical case study across three benchmarks shows that the integration of certain CDM methods can markedly improve the reasoning capabilities and robustness of some leading LLMs, all without requiring intricate system designs. Additionally, we find that some CDM mechanisms generate positive synergies even with as few as three agents. The voting-based methods also demonstrate robustness against single points of failure, as well as diversity in terms of hit-rate@k and subject-wise impacts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15154v1">MCCoder: Streamlining Motion Control with LLM-Assisted Code Generation and Rigorous Verification</a></div>
    <div class="paper-meta">
      📅 2024-10-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown considerable promise in code generation. However, the automation sector, especially in motion control, continues to rely heavily on manual programming due to the complexity of tasks and critical safety considerations. In this domain, incorrect code execution can pose risks to both machinery and personnel, necessitating specialized expertise. To address these challenges, we introduce MCCoder, an LLM-powered system designed to generate code that addresses complex motion control tasks, with integrated soft-motion data verification. MCCoder enhances code generation through multitask decomposition, hybrid retrieval-augmented generation (RAG), and self-correction with a private motion library. Moreover, it supports data verification by logging detailed trajectory data and providing simulations and plots, allowing users to assess the accuracy of the generated code and bolstering confidence in LLM-based programming. To ensure robust validation, we propose MCEVAL, an evaluation dataset with metrics tailored to motion control tasks of varying difficulties. Experiments indicate that MCCoder improves performance by 11.61% overall and by 66.12% on complex tasks in MCEVAL dataset compared with base models with naive RAG. This system and dataset aim to facilitate the application of code generation in automation settings with strict safety requirements. MCCoder is publicly available at https://github.com/MCCodeAI/MCCoder.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15135v1">Augmenting the Veracity and Explanations of Complex Fact Checking via Iterative Self-Revision with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-19
    </div>
    <details class="paper-abstract">
      Explanation generation plays a more pivotal role than fact verification in producing interpretable results and facilitating comprehensive fact-checking, which has recently garnered considerable attention. However, previous studies on explanation generation has shown several limitations, such as being confined to English scenarios, involving overly complex inference processes, and not fully unleashing the potential of the mutual feedback between veracity labels and explanation texts. To address these issues, we construct two complex fact-checking datasets in the Chinese scenarios: CHEF-EG and TrendFact. These datasets involve complex facts in areas such as health, politics, and society, presenting significant challenges for fact verification methods. In response to these challenges, we propose a unified framework called FactISR (Augmenting Fact-Checking via Iterative Self-Revision) to perform mutual feedback between veracity and explanations by leveraging the capabilities of large language models(LLMs). FactISR uses a single model to address tasks such as fact verification and explanation generation. Its self-revision mechanism can further revision the consistency between veracity labels, explanation texts, and evidence, as well as eliminate irrelevant noise. We conducted extensive experiments with baselines and FactISR on the proposed datasets. The experimental results demonstrate the effectiveness of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15050v1">Are LLMs Good Zero-Shot Fallacy Classifiers?</a></div>
    <div class="paper-meta">
      📅 2024-10-19
      | 💬 Accepted to EMNLP2024 main conference
    </div>
    <details class="paper-abstract">
      Fallacies are defective arguments with faulty reasoning. Detecting and classifying them is a crucial NLP task to prevent misinformation, manipulative claims, and biased decisions. However, existing fallacy classifiers are limited by the requirement for sufficient labeled data for training, which hinders their out-of-distribution (OOD) generalization abilities. In this paper, we focus on leveraging Large Language Models (LLMs) for zero-shot fallacy classification. To elicit fallacy-related knowledge and reasoning abilities of LLMs, we propose diverse single-round and multi-round prompting schemes, applying different task-specific instructions such as extraction, summarization, and Chain-of-Thought reasoning. With comprehensive experiments on benchmark datasets, we suggest that LLMs could be potential zero-shot fallacy classifiers. In general, LLMs under single-round prompting schemes have achieved acceptable zero-shot performances compared to the best full-shot baselines and can outperform them in all OOD inference scenarios and some open-domain tasks. Our novel multi-round prompting schemes can effectively bring about more improvements, especially for small LLMs. Our analysis further underlines the future research on zero-shot fallacy classification. Codes and data are available at: https://github.com/panFJCharlotte98/Fallacy_Detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12876v2">In-context KV-Cache Eviction for LLMs via Attention-Gate</a></div>
    <div class="paper-meta">
      📅 2024-10-19
    </div>
    <details class="paper-abstract">
      The KV-Cache technique has become the standard for the inference of large language models (LLMs). It caches states of self-attention to avoid recomputation. Yet, it is widely criticized that KV-Cache can become a bottleneck of the LLM inference system, especially when confronted with ultra-large models and long-context queries. A natural remedy is to discard the KV-Cache for less important tokens, with StreamingLLM as an example, but the used static eviction strategies cannot flexibly adapt to varying contexts. Remedies like H2O leverage accumulative attention scores to perform dynamic eviction but suffer from the attention bias issue in capturing contextual information. This paper bridges this gap by devising a parameterized KV-Cache eviction mechanism, dubbed as Attention-Gate, which accepts the whole context as input and yields eviction flags for each token to realize in-context eviction. The subsequent self-attention module proceeds according to the flags and only the KV states for the remaining tokens need to be cached. The Attention-Gates can vary among different heads and layers and be trivially plugged into pre-trained LLMs, tuned by cost-effective continual pre-training or supervised fine-tuning objectives to acquire what to discard. The computational and memory overhead introduced by Attention-Gates is minimal. Our method is validated across multiple tasks, demonstrating both efficiency and adaptability. After a highly efficient continual pre-training, it achieves higher average accuracy and evicts more tokens compared to traditional training-free methods. In supervised fine-tuning, it not only evicts many tokens but also outperforms LoRA-finetuned LLMs on some datasets, such as RTE, where it improves accuracy by 13.9% while evicting 62.8% of tokens, showing that effective eviction of redundant tokens can even enhance performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15025v1">LLM-Driven Learning Analytics Dashboard for Teachers in EFL Writing Education</a></div>
    <div class="paper-meta">
      📅 2024-10-19
      | 💬 EMNLP 2024 Workshop CustomNLP4U. arXiv admin note: text overlap with arXiv:2405.19691
    </div>
    <details class="paper-abstract">
      This paper presents the development of a dashboard designed specifically for teachers in English as a Foreign Language (EFL) writing education. Leveraging LLMs, the dashboard facilitates the analysis of student interactions with an essay writing system, which integrates ChatGPT for real-time feedback. The dashboard aids teachers in monitoring student behavior, identifying noneducational interaction with ChatGPT, and aligning instructional strategies with learning objectives. By combining insights from NLP and Human-Computer Interaction (HCI), this study demonstrates how a human-centered approach can enhance the effectiveness of teacher dashboards, particularly in ChatGPT-integrated learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.12853v2">A New Perspective on ADHD Research: Knowledge Graph Construction with LLMs and Network Based Insights</a></div>
    <div class="paper-meta">
      📅 2024-10-19
      | 💬 12 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Attention-Deficit/Hyperactivity Disorder (ADHD) is a challenging disorder to study due to its complex symptomatology and diverse contributing factors. To explore how we can gain deeper insights on this topic, we performed a network analysis on a comprehensive knowledge graph (KG) of ADHD, constructed by integrating scientific literature and clinical data with the help of cutting-edge large language models. The analysis, including k-core techniques, identified critical nodes and relationships that are central to understanding the disorder. Building on these findings, we curated a knowledge graph that is usable in a context-aware chatbot (Graph-RAG) with Large Language Models (LLMs), enabling accurate and informed interactions. Our knowledge graph not only advances the understanding of ADHD but also provides a powerful tool for research and clinical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14931v1">"Ghost of the past": identifying and resolving privacy leakage from LLM's memory through proactive user interaction</a></div>
    <div class="paper-meta">
      📅 2024-10-19
    </div>
    <details class="paper-abstract">
      Memories, encompassing past inputs in context window and retrieval-augmented generation (RAG), frequently surface during human-LLM interactions, yet users are often unaware of their presence and the associated privacy risks. To address this, we propose MemoAnalyzer, a system for identifying, visualizing, and managing private information within memories. A semi-structured interview (N=40) revealed that low privacy awareness was the primary challenge, while proactive privacy control emerged as the most common user need. MemoAnalyzer uses a prompt-based method to infer and identify sensitive information from aggregated past inputs, allowing users to easily modify sensitive content. Background color temperature and transparency are mapped to inference confidence and sensitivity, streamlining privacy adjustments. A 5-day evaluation (N=36) comparing MemoAnalyzer with the default GPT setting and a manual modification baseline showed MemoAnalyzer significantly improved privacy awareness and protection without compromising interaction speed. Our study contributes to privacy-conscious LLM design, offering insights into privacy protection for Human-AI interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14926v1">Aligning LLMs with Human Instructions and Stock Market Feedback in Financial Sentiment Analysis</a></div>
    <div class="paper-meta">
      📅 2024-10-19
    </div>
    <details class="paper-abstract">
      Financial sentiment analysis is crucial for trading and investment decision-making. This study introduces an adaptive retrieval augmented framework for Large Language Models (LLMs) that aligns with human instructions through Instruction Tuning and incorporates market feedback to dynamically adjust weights across various knowledge sources within the Retrieval-Augmented Generation (RAG) module. Building upon foundational models like LLaMA 2, we fine-tune a series of LLMs ranging from 7B to 70B in size, enriched with Instruction Tuning and RAG, and further optimized through direct feedback and Reinforcement Learning (RL)-based refinement methods applied to the source weights of RAG.Through extensive evaluation, we demonstrate that the sentiment outputs from our LLMs more accurately mirror the intrinsic sentiment of textual data, showcasing a 1% to 6% boost in accuracy and F1 score over existing state-of-the-art models and leading conversational AI systems. Moreover, the sentiments extracted are more indicative of the directions in stock price movements. On top of that, we successfully construct portfolios that yield a 3.61% higher Sharpe ratio compared to the S&P 500 baseline in bullish markets. These portfolios also demonstrate resilience in bearish markets, with a 5x reduction in return losses compared to those typically experienced by the S&P 500.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.03733v2">Planning In Natural Language Improves LLM Search For Code Generation</a></div>
    <div class="paper-meta">
      📅 2024-10-18
    </div>
    <details class="paper-abstract">
      While scaling training compute has led to remarkable improvements in large language models (LLMs), scaling inference compute has not yet yielded analogous gains. We hypothesize that a core missing component is a lack of diverse LLM outputs, leading to inefficient search due to models repeatedly sampling highly similar, yet incorrect generations. We empirically demonstrate that this lack of diversity can be mitigated by searching over candidate plans for solving a problem in natural language. Based on this insight, we propose PlanSearch, a novel search algorithm which shows strong results across HumanEval+, MBPP+, and LiveCodeBench (a contamination-free benchmark for competitive coding). PlanSearch generates a diverse set of observations about the problem and then uses these observations to construct plans for solving the problem. By searching over plans in natural language rather than directly over code solutions, PlanSearch explores a significantly more diverse range of potential solutions compared to baseline search methods. Using PlanSearch on top of Claude 3.5 Sonnet achieves a state-of-the-art pass@200 of 77.0% on LiveCodeBench, outperforming both the best score achieved without search (pass@1 = 41.4%) and using standard repeated sampling (pass@200 = 60.6%). Finally, we show that, across all models, search algorithms, and benchmarks analyzed, we can accurately predict performance gains due to search as a direct function of the diversity over generated ideas. Code can be found at https://github.com/scaleapi/plansearch.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10741v2">SensorBench: Benchmarking LLMs in Coding-Based Sensor Processing</a></div>
    <div class="paper-meta">
      📅 2024-10-18
    </div>
    <details class="paper-abstract">
      Effective processing, interpretation, and management of sensor data have emerged as a critical component of cyber-physical systems. Traditionally, processing sensor data requires profound theoretical knowledge and proficiency in signal-processing tools. However, recent works show that Large Language Models (LLMs) have promising capabilities in processing sensory data, suggesting their potential as copilots for developing sensing systems. To explore this potential, we construct a comprehensive benchmark, SensorBench, to establish a quantifiable objective. The benchmark incorporates diverse real-world sensor datasets for various tasks. The results show that while LLMs exhibit considerable proficiency in simpler tasks, they face inherent challenges in processing compositional tasks with parameter selections compared to engineering experts. Additionally, we investigate four prompting strategies for sensor processing and show that self-verification can outperform all other baselines in 48% of tasks. Our study provides a comprehensive benchmark and prompting analysis for future developments, paving the way toward an LLM-based sensor processing copilot.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14897v1">From Test-Taking to Test-Making: Examining LLM Authoring of Commonsense Assessment Items</a></div>
    <div class="paper-meta">
      📅 2024-10-18
      | 💬 Accepted at Findings of EMNLP 2024
    </div>
    <details class="paper-abstract">
      LLMs can now perform a variety of complex writing tasks. They also excel in answering questions pertaining to natural language inference and commonsense reasoning. Composing these questions is itself a skilled writing task, so in this paper we consider LLMs as authors of commonsense assessment items. We prompt LLMs to generate items in the style of a prominent benchmark for commonsense reasoning, the Choice of Plausible Alternatives (COPA). We examine the outcome according to analyses facilitated by the LLMs and human annotation. We find that LLMs that succeed in answering the original COPA benchmark are also more successful in authoring their own items.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03993v2">TR-LLM: Integrating Trajectory Data for Scene-Aware LLM-Based Human Action Prediction</a></div>
    <div class="paper-meta">
      📅 2024-10-18
    </div>
    <details class="paper-abstract">
      Accurate prediction of human behavior is crucial for AI systems to effectively support real-world applications, such as autonomous robots anticipating and assisting with human tasks. Real-world scenarios frequently present challenges such as occlusions and incomplete scene observations, which can compromise predictive accuracy. Thus, traditional video-based methods often struggle due to limited temporal and spatial perspectives. Large Language Models (LLMs) offer a promising alternative. Having been trained on a large text corpus describing human behaviors, LLMs likely encode plausible sequences of human actions in a home environment. However, LLMs, trained primarily on text data, lack inherent spatial awareness and real-time environmental perception. They struggle with understanding physical constraints and spatial geometry. Therefore, to be effective in a real-world spatial scenario, we propose a multimodal prediction framework that enhances LLM-based action prediction by integrating physical constraints derived from human trajectories. Our experiments demonstrate that combining LLM predictions with trajectory data significantly improves overall prediction performance. This enhancement is particularly notable in situations where the LLM receives limited scene information, highlighting the complementary nature of linguistic knowledge and physical constraints in understanding and anticipating human behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14835v1">Towards Automated Verification of LLM-Synthesized C Programs</a></div>
    <div class="paper-meta">
      📅 2024-10-18
    </div>
    <details class="paper-abstract">
      We present \synver{}, a novel synthesis and verification framework for C programs, that deploys a Large Language Model (LLM) to search for a candidate program that satisfies the given specification. Our key idea is to impose syntactic and semantic biases on programs generated by LLMs, such that the synthesized program is more amenable to automated verification. Based on this idea, we propose a novel specification-verification tool, built on top of Verified Software Toolchain, that help automate the process. Our experiments on a diverse set of benchmarks drawn from the deductive program synthesis community, shows that this approach is scalable and extensible. The benchmarks constitute of specifications comprising of basic coding examples, Separation Logic based assertions, and API specifications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14827v1">Making LLMs Vulnerable to Prompt Injection via Poisoning Alignment</a></div>
    <div class="paper-meta">
      📅 2024-10-18
    </div>
    <details class="paper-abstract">
      In a prompt injection attack, an attacker injects a prompt into the original one, aiming to make the LLM follow the injected prompt and perform a task chosen by the attacker. Existing prompt injection attacks primarily focus on how to blend the injected prompt into the original prompt without altering the LLM itself. Our experiments show that these attacks achieve some success, but there is still significant room for improvement. In this work, we show that an attacker can boost the success of prompt injection attacks by poisoning the LLM's alignment process. Specifically, we propose PoisonedAlign, a method to strategically create poisoned alignment samples. When even a small fraction of the alignment data is poisoned using our method, the aligned LLM becomes more vulnerable to prompt injection while maintaining its foundational capabilities. The code is available at https://github.com/Sadcardation/PoisonedAlign
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14815v1">Adapting Multilingual LLMs to Low-Resource Languages using Continued Pre-training and Synthetic Corpus</a></div>
    <div class="paper-meta">
      📅 2024-10-18
    </div>
    <details class="paper-abstract">
      Multilingual LLMs support a variety of languages; however, their performance is suboptimal for low-resource languages. In this work, we emphasize the importance of continued pre-training of multilingual LLMs and the use of translation-based synthetic pre-training corpora for improving LLMs in low-resource languages. We conduct our study in the context of the low-resource Indic language Hindi. We introduce Nemotron-Mini-Hindi 4B, a bilingual SLM supporting both Hindi and English, based on Nemotron-Mini 4B. The model is trained using a mix of real and synthetic Hindi + English tokens, with continuous pre-training performed on 400B tokens. We demonstrate that both the base and instruct models achieve state-of-the-art results on Hindi benchmarks while remaining competitive on English tasks. Additionally, we observe that the continued pre-training approach enhances the model's overall factual accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14641v1">Distance between Relevant Information Pieces Causes Bias in Long-Context LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-18
      | 💬 work in progress
    </div>
    <details class="paper-abstract">
      Positional bias in large language models (LLMs) hinders their ability to effectively process long inputs. A prominent example is the "lost in the middle" phenomenon, where LLMs struggle to utilize relevant information situated in the middle of the input. While prior research primarily focuses on single pieces of relevant information, real-world applications often involve multiple relevant information pieces. To bridge this gap, we present LongPiBench, a benchmark designed to assess positional bias involving multiple pieces of relevant information. Thorough experiments are conducted with five commercial and six open-source models. These experiments reveal that while most current models are robust against the "lost in the middle" issue, there exist significant biases related to the spacing of relevant information pieces. These findings highlight the importance of evaluating and reducing positional biases to advance LLM's capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14635v1">GenEOL: Harnessing the Generative Power of LLMs for Training-Free Sentence Embeddings</a></div>
    <div class="paper-meta">
      📅 2024-10-18
    </div>
    <details class="paper-abstract">
      Training-free embedding methods directly leverage pretrained large language models (LLMs) to embed text, bypassing the costly and complex procedure of contrastive learning. Previous training-free embedding methods have mainly focused on optimizing embedding prompts and have overlooked the benefits of utilizing the generative abilities of LLMs. We propose a novel method, GenEOL, which uses LLMs to generate diverse transformations of a sentence that preserve its meaning, and aggregates the resulting embeddings of these transformations to enhance the overall sentence embedding. GenEOL significantly outperforms the existing training-free embedding methods by an average of 2.85 points across several LLMs on the sentence semantic text similarity (STS) benchmark. Our analysis shows that GenEOL stabilizes representation quality across LLM layers and is robust to perturbations of embedding prompts. GenEOL also achieves notable gains on multiple clustering, reranking and pair-classification tasks from the MTEB benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.19826v1">Novel Development of LLM Driven mCODE Data Model for Improved Clinical Trial Matching to Enable Standardization and Interoperability in Oncology Research</a></div>
    <div class="paper-meta">
      📅 2024-10-18
      | 💬 18 pages, 13 figures, accessible and published at: The Young Researcher Fall 2024 Volume 8, Number 2(Special Edition in Collaboration with Harvard Undergraduate Openbio Laboratory); Pages 28-45
    </div>
    <details class="paper-abstract">
      Each year, the lack of efficient data standardization and interoperability in cancer care contributes to the severe lack of timely and effective diagnosis, while constantly adding to the burden of cost, with cancer costs nationally reaching over $208 billion in 2023 alone. Traditional methods regarding clinical trial enrollment and clinical care in oncology are often manual, time-consuming, and lack a data-driven approach. This paper presents a novel framework to streamline standardization, interoperability, and exchange of cancer domains and enhance the integration of oncology-based EHRs across disparate healthcare systems. This paper utilizes advanced LLMs and Computer Engineering to streamline cancer clinical trials and discovery. By utilizing FHIR's resource-based approach and LLM-generated mCODE profiles, we ensure timely, accurate, and efficient sharing of patient information across disparate healthcare systems. Our methodology involves transforming unstructured patient treatment data, PDFs, free-text information, and progress notes into enriched mCODE profiles, facilitating seamless integration with our novel AI and ML-based clinical trial matching engine. The results of this study show a significant improvement in data standardization, with accuracy rates of our trained LLM peaking at over 92% with datasets consisting of thousands of patient data. Additionally, our LLM demonstrated an accuracy rate of 87% for SNOMED-CT, 90% for LOINC, and 84% for RxNorm codes. This trumps the current status quo, with LLMs such as GPT-4 and Claude's 3.5 peaking at an average of 77%. This paper successfully underscores the potential of our standardization and interoperability framework, paving the way for more efficient and personalized cancer treatment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14609v1">DiSCo Meets LLMs: A Unified Approach for Sparse Retrieval and Contextual Distillation in Conversational Search</a></div>
    <div class="paper-meta">
      📅 2024-10-18
    </div>
    <details class="paper-abstract">
      Conversational Search (CS) is the task of retrieving relevant documents from a corpus within a conversational context, combining retrieval with conversational context modeling. With the explosion of Large Language Models (LLMs), the CS field has seen major improvements with LLMs rewriting user queries, accounting for conversational context. However, engaging LLMs at inference time harms efficiency. Current methods address this by distilling embeddings from human-rewritten queries to learn the context modeling task. Yet, these approaches predominantly focus on context modeling, and only treat the contrastive component of the retrieval task within a distillation-independent loss term. To address these limitations, we propose a new distillation method, as a relaxation of the previous objective, unifying retrieval and context modeling. We relax the existing training objectives by distilling similarity scores between conversations and documents, rather than relying solely on representation learning. Our proposed distillation objective allows for more freedom in the representation space and leverages the contrastive nature of document relevance. Through experiments on Learned Sparse Retrieval (LSR) across 5 CS datasets, our approach demonstrates substantial improvements in both in-domain and out-of-domain retrieval performance, outperforming state-of-the-art with gains of up to 6 points in recall for out-of-domain datasets. Additionally, through the relaxation of the objective, we propose a multi-teacher distillation, using multiple LLMs as teachers, yielding additional gains, and outperforming the teachers themselves in in-domain experiments. Finally, analysis of the sparsity of the models reveals that our distillation allows for better control over the sparsity of the trained models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16327v1">Feint and Attack: Attention-Based Strategies for Jailbreaking and Protecting LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-18
    </div>
    <details class="paper-abstract">
      Jailbreak attack can be used to access the vulnerabilities of Large Language Models (LLMs) by inducing LLMs to generate the harmful content. And the most common method of the attack is to construct semantically ambiguous prompts to confuse and mislead the LLMs. To access the security and reveal the intrinsic relation between the input prompt and the output for LLMs, the distribution of attention weight is introduced to analyze the underlying reasons. By using statistical analysis methods, some novel metrics are defined to better describe the distribution of attention weight, such as the Attention Intensity on Sensitive Words (Attn_SensWords), the Attention-based Contextual Dependency Score (Attn_DepScore) and Attention Dispersion Entropy (Attn_Entropy). By leveraging the distinct characteristics of these metrics, the beam search algorithm and inspired by the military strategy "Feint and Attack", an effective jailbreak attack strategy named as Attention-Based Attack (ABA) is proposed. In the ABA, nested attack prompts are employed to divert the attention distribution of the LLMs. In this manner, more harmless parts of the input can be used to attract the attention of the LLMs. In addition, motivated by ABA, an effective defense strategy called as Attention-Based Defense (ABD) is also put forward. Compared with ABA, the ABD can be used to enhance the robustness of LLMs by calibrating the attention distribution of the input prompt. Some comparative experiments have been given to demonstrate the effectiveness of ABA and ABD. Therefore, both ABA and ABD can be used to access the security of the LLMs. The comparative experiment results also give a logical explanation that the distribution of attention weight can bring great influence on the output for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.02408v1">AI on My Shoulder: Supporting Emotional Labor in Front-Office Roles with an LLM-based Empathetic Coworker</a></div>
    <div class="paper-meta">
      📅 2024-10-18
    </div>
    <details class="paper-abstract">
      Client-Service Representatives (CSRs) are vital to organizations. Frequent interactions with disgruntled clients, however, disrupt their mental well-being. To help CSRs regulate their emotions while interacting with uncivil clients, we designed Pro-Pilot, an LLM-powered assistant, and evaluated its efficacy, perception, and use. Our comparative analyses between 665 human and Pro-Pilot-generated support messages demonstrate Pro-Pilot's ability to adapt to and demonstrate empathy in various incivility incidents. Additionally, 143 CSRs assessed Pro-Pilot's empathy as more sincere and actionable than human messages. Finally, we interviewed 20 CSRs who interacted with Pro-Pilot in a simulation exercise. They reported that Pro-Pilot helped them avoid negative thinking, recenter thoughts, and humanize clients; showing potential for bridging gaps in coworker support. Yet, they also noted deployment challenges and emphasized the irreplaceability of shared experiences. We discuss future designs and societal implications of AI-mediated emotional labor, underscoring empathy as a critical function for AI assistants in front-office roles.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14763v1">Enabling Scalable Evaluation of Bias Patterns in Medical LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown impressive potential in helping with numerous medical challenges. Deploying LLMs in high-stakes applications such as medicine, however, brings in many concerns. One major area of concern relates to biased behaviors of LLMs in medical applications, leading to unfair treatment of individuals. To pave the way for the responsible and impactful deployment of Med LLMs, rigorous evaluation is a key prerequisite. Due to the huge complexity and variability of different medical scenarios, existing work in this domain has primarily relied on using manually crafted datasets for bias evaluation. In this study, we present a new method to scale up such bias evaluations by automatically generating test cases based on rigorous medical evidence. We specifically target the challenges of a) domain-specificity of bias characterization, b) hallucinating while generating the test cases, and c) various dependencies between the health outcomes and sensitive attributes. To that end, we offer new methods to address these challenges integrated with our generative pipeline, using medical knowledge graphs, medical ontologies, and customized general LLM evaluation frameworks in our method. Through a series of extensive experiments, we show that the test cases generated by our proposed method can effectively reveal bias patterns in Med LLMs at larger and more flexible scales than human-crafted datasets. We publish a large bias evaluation dataset using our pipeline, which is dedicated to a few medical case studies. A live demo of our application for vignette generation is available at https://vignette.streamlit.app. Our code is also available at https://github.com/healthylaife/autofair.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16325v1">This Candidate is [MASK]. Letters of Reference and Job Market Outcomes using LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-18
    </div>
    <details class="paper-abstract">
      I implement a prompt-based learning strategy to extract measures of sentiment and other features from confidential reference letters. I show that the contents of reference letters is clearly reflected in the performance of job market candidates in the Economics academic job market. In contrast, applying traditional ``bag-of-words'' approaches produces measures of sentiment that, while positively correlated to my LLM-based measure, are not predictive of job market outcomes. Using a random forest, I show that both letter quality and length are predictive of success in the job market. Letters authored by advisers appear to be as important as those written by other referees.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14442v1">A Systematic Study of Cross-Layer KV Sharing for Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2024-10-18
    </div>
    <details class="paper-abstract">
      Recently, sharing key-value (KV) cache across layers has been found effective in efficient inference of large language models (LLMs). To systematically investigate different techniques of cross-layer KV sharing, we propose a unified framework that covers several recent methods and their novel variants. We conduct comprehensive experiments on all the configurations of the framework, evaluating their generation throughput and performance in language modeling and downstream tasks. We find that when reducing the size of the KV cache by 2x, most configurations can achieve competitive performance to and higher throughput than standard transformers, but when further reducing the size of the KV cache, pairing queries of all layers with KVs of upper layers can better maintain performance, although it also introduces additional training cost and prefilling latency. We hope that this work will help users choose the appropriate approach according to their requirements and facilitate research on the acceleration of LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14425v1">Unlearning Backdoor Attacks for LLMs with Weak-to-Strong Knowledge Distillation</a></div>
    <div class="paper-meta">
      📅 2024-10-18
    </div>
    <details class="paper-abstract">
      Parameter-efficient fine-tuning (PEFT) can bridge the gap between large language models (LLMs) and downstream tasks. However, PEFT has been proven vulnerable to malicious attacks. Research indicates that poisoned LLMs, even after PEFT, retain the capability to activate internalized backdoors when input samples contain predefined triggers. In this paper, we introduce a novel weak-to-strong unlearning algorithm to defend against backdoor attacks based on feature alignment knowledge distillation, named W2SDefense. Specifically, we first train a small-scale language model through full-parameter fine-tuning to serve as the clean teacher model. Then, this teacher model guides the large-scale poisoned student model in unlearning the backdoor, leveraging PEFT. Theoretical analysis suggests that W2SDefense has the potential to enhance the student model's ability to unlearn backdoor features, preventing the activation of the backdoor. We conduct experiments on text classification tasks involving three state-of-the-art language models and three different backdoor attack algorithms. Our empirical results demonstrate the outstanding performance of W2SDefense in defending against backdoor attacks without compromising model performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.10918v5">Multi-LLM QA with Embodied Exploration</a></div>
    <div class="paper-meta">
      📅 2024-10-18
      | 💬 16 pages, 9 Figures, 5 Tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have grown in popularity due to their natural language interface and pre trained knowledge, leading to rapidly increasing success in question-answering (QA) tasks. More recently, multi-agent systems with LLM-based agents (Multi-LLM) have been utilized increasingly more for QA. In these scenarios, the models may each answer the question and reach a consensus or each model is specialized to answer different domain questions. However, most prior work dealing with Multi-LLM QA has focused on scenarios where the models are asked in a zero-shot manner or are given information sources to extract the answer. For question answering of an unknown environment, embodied exploration of the environment is first needed to answer the question. This skill is necessary for personalizing embodied AI to environments such as households. There is a lack of insight into whether a Multi-LLM system can handle question-answering based on observations from embodied exploration. In this work, we address this gap by investigating the use of Multi-Embodied LLM Explorers (MELE) for QA in an unknown environment. Multiple LLM-based agents independently explore and then answer queries about a household environment. We analyze different aggregation methods to generate a single, final answer for each query: debating, majority voting, and training a central answer module (CAM). Using CAM, we observe a $46\%$ higher accuracy compared against the other non-learning-based aggregation methods. We provide code and the query dataset for further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.12950v2">MolecularGPT: Open Large Language Model (LLM) for Few-Shot Molecular Property Prediction</a></div>
    <div class="paper-meta">
      📅 2024-10-18
    </div>
    <details class="paper-abstract">
      Molecular property prediction (MPP) is a fundamental and crucial task in drug discovery. However, prior methods are limited by the requirement for a large number of labeled molecules and their restricted ability to generalize for unseen and new tasks, both of which are essential for real-world applications. To address these challenges, we present MolecularGPT for few-shot MPP. From a perspective on instruction tuning, we fine-tune large language models (LLMs) based on curated molecular instructions spanning over 1000 property prediction tasks. This enables building a versatile and specialized LLM that can be adapted to novel MPP tasks without any fine-tuning through zero- and few-shot in-context learning (ICL). MolecularGPT exhibits competitive in-context reasoning capabilities across 10 downstream evaluation datasets, setting new benchmarks for few-shot molecular prediction tasks. More importantly, with just two-shot examples, MolecularGPT can outperform standard supervised graph neural network methods on 4 out of 7 datasets. It also excels state-of-the-art LLM baselines by up to 15.7% increase on classification accuracy and decrease of 17.9 on regression metrics (e.g., RMSE) under zero-shot. This study demonstrates the potential of LLMs as effective few-shot molecular property predictors. The code is available at https://github.com/NYUSHCS/MolecularGPT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14391v1">Analyzing Context Utilization of LLMs in Document-Level Translation</a></div>
    <div class="paper-meta">
      📅 2024-10-18
      | 💬 4 pages, 2 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLM) are increasingly strong contenders in machine translation. We study document-level translation, where some words cannot be translated without context from outside the sentence. We investigate the ability of prominent LLMs to utilize context by analyzing models' robustness to perturbed and randomized document context. We find that LLMs' improved document-translation performance is not always reflected in pronoun translation performance. We highlight the need for context-aware finetuning of LLMs with a focus on relevant parts of the context to improve their reliability for document-level translation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14321v1">From Solitary Directives to Interactive Encouragement! LLM Secure Code Generation by Natural Language Prompting</a></div>
    <div class="paper-meta">
      📅 2024-10-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable potential in code generation, making them increasingly important in the field. However, the security issues of generated code have not been fully addressed, and the usability of LLMs in code generation still requires further exploration. This work introduces SecCode, a framework that leverages an innovative interactive encouragement prompting (EP) technique for secure code generation with \textit{only NL} prompts. This approach ensures that the prompts can be easily shared and understood by general users. SecCode functions through three stages: 1) Code Generation using NL Prompts; 2) Code Vulnerability Detection and Fixing, utilising our proposed encouragement prompting; 3) Vulnerability Cross-Checking and Code Security Refinement. These stages are executed in multiple interactive iterations to progressively enhance security. By using both proprietary LLMs (i.e., GPT-3.5 Turbo, GPT-4 and GPT-4o) and open-source LLMs (i.e., Llama 3.1 8B Instruct, DeepSeek Coder V2 Lite Instruct) evaluated on three benchmark datasets, extensive experimental results show that our proposed SecCode greatly outperforms compared baselines, generating secure code with a high vulnerability correction rate. For example, SecCode exhibits a high fix success rate of over 76\% after running 5 automated EP interactive iterations and over 89\% after running 10 automated EP interactive iterations. To the best of our knowledge, this work is the first to formulate secure code generation with NL prompts only. We have open-sourced our code and encourage the community to focus on secure code generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.15053v2">PARIKSHA: A Large-Scale Investigation of Human-LLM Evaluator Agreement on Multilingual and Multi-Cultural Data</a></div>
    <div class="paper-meta">
      📅 2024-10-18
      | 💬 Accepted to EMNLP 2024
    </div>
    <details class="paper-abstract">
      Evaluation of multilingual Large Language Models (LLMs) is challenging due to a variety of factors -- the lack of benchmarks with sufficient linguistic diversity, contamination of popular benchmarks into LLM pre-training data and the lack of local, cultural nuances in translated benchmarks. In this work, we study human and LLM-based evaluation in a multilingual, multi-cultural setting. We evaluate 30 models across 10 Indic languages by conducting 90K human evaluations and 30K LLM-based evaluations and find that models such as GPT-4o and Llama-3 70B consistently perform best for most Indic languages. We build leaderboards for two evaluation settings - pairwise comparison and direct assessment and analyze the agreement between humans and LLMs. We find that humans and LLMs agree fairly well in the pairwise setting but the agreement drops for direct assessment evaluation especially for languages such as Bengali and Odia. We also check for various biases in human and LLM-based evaluation and find evidence of self-bias in the GPT-based evaluator. Our work presents a significant step towards scaling up multilingual evaluation of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.06572v2">Graph Neural Network Enhanced Retrieval for Question Answering of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-18
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Retrieval augmented generation has revolutionized large language model (LLM) outputs by providing factual supports. Nevertheless, it struggles to capture all the necessary knowledge for complex reasoning questions. Existing retrieval methods typically divide reference documents into passages, treating them in isolation. These passages, however, are often interrelated, such as passages that are contiguous or share the same keywords. Therefore, it is crucial to recognize such relatedness for enhancing the retrieval process. In this paper, we propose a novel retrieval method, called GNN-Ret, which leverages graph neural networks (GNNs) to enhance retrieval by exploiting the relatedness between passages. Specifically, we first construct a graph of passages by connecting passages that are structure-related or keyword-related. A graph neural network (GNN) is then leveraged to exploit the relationships between passages and improve the retrieval of supporting passages. Furthermore, we extend our method to handle multi-hop reasoning questions using a recurrent graph neural network (RGNN), named RGNN-Ret. At each step, RGNN-Ret integrates the graphs of passages from previous steps, thereby enhancing the retrieval of supporting passages. Extensive experiments on benchmark datasets demonstrate that GNN-Ret achieves higher accuracy for question answering with a single query of LLMs than strong baselines that require multiple queries, and RGNN-Ret further improves accuracy and achieves state-of-the-art performance, with up to 10.4% accuracy improvement on the 2WikiMQA dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14259v1">Beyond Binary: Towards Fine-Grained LLM-Generated Text Detection via Role Recognition and Involvement Measurement</a></div>
    <div class="paper-meta">
      📅 2024-10-18
      | 💬 Social Media, Large Language Models, LLM-generated Text Detection, AI-assisted News Detection
    </div>
    <details class="paper-abstract">
      The rapid development of large language models (LLMs), like ChatGPT, has resulted in the widespread presence of LLM-generated content on social media platforms, raising concerns about misinformation, data biases, and privacy violations, which can undermine trust in online discourse. While detecting LLM-generated content is crucial for mitigating these risks, current methods often focus on binary classification, failing to address the complexities of real-world scenarios like human-AI collaboration. To move beyond binary classification and address these challenges, we propose a new paradigm for detecting LLM-generated content. This approach introduces two novel tasks: LLM Role Recognition (LLM-RR), a multi-class classification task that identifies specific roles of LLM in content generation, and LLM Influence Measurement (LLM-IM), a regression task that quantifies the extent of LLM involvement in content creation. To support these tasks, we propose LLMDetect, a benchmark designed to evaluate detectors' performance on these new tasks. LLMDetect includes the Hybrid News Detection Corpus (HNDC) for training detectors, as well as DetectEval, a comprehensive evaluation suite that considers five distinct cross-context variations and multi-intensity variations within the same LLM role. This allows for a thorough assessment of detectors' generalization and robustness across diverse contexts. Our empirical validation of 10 baseline detection methods demonstrates that fine-tuned PLM-based models consistently outperform others on both tasks, while advanced LLMs face challenges in accurately detecting their own generated content. Our experimental results and analysis offer insights for developing more effective detection models for LLM-generated content. This research enhances the understanding of LLM-generated content and establishes a foundation for more nuanced detection methodologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14257v1">Revisiting SLO and Goodput Metrics in LLM Serving</a></div>
    <div class="paper-meta">
      📅 2024-10-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable performance and are widely deployed in various applications, while the serving of LLM inference has raised concerns about user experience and serving throughput. Accordingly, service level objectives (SLOs) and goodput-the number of requests that meet SLOs per second-are introduced to evaluate the performance of LLM serving. However, existing metrics fail to capture the nature of user experience. We observe two ridiculous phenomena in existing metrics: 1) delaying token delivery can smooth the tail time between tokens (tail TBT) of a request and 2) dropping the request that fails to meet the SLOs midway can improve goodput. In this paper, we revisit SLO and goodput metrics in LLM serving and propose a unified metric framework smooth goodput including SLOs and goodput to reflect the nature of user experience in LLM serving. The framework can adapt to specific goals of different tasks by setting parameters. We re-evaluate the performance of different LLM serving systems under multiple workloads based on this unified framework and provide possible directions for future optimization of existing strategies. We hope that this framework can provide a unified standard for evaluating LLM serving and foster researches in the field of LLM serving optimization to move in a cohesive direction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14251v1">Synthesizing Post-Training Data for LLMs through Multi-Agent Simulation</a></div>
    <div class="paper-meta">
      📅 2024-10-18
    </div>
    <details class="paper-abstract">
      Post-training is essential for enabling large language models (LLMs) to follow human instructions. Inspired by the recent success of using LLMs to simulate human society, we leverage multi-agent simulation to automatically generate diverse text-based scenarios, capturing a wide range of real-world human needs. We propose MATRIX, a multi-agent simulator that creates realistic and scalable scenarios. Leveraging these outputs, we introduce a novel scenario-driven instruction generator MATRIX-Gen for controllable and highly realistic data synthesis. Extensive experiments demonstrate that our framework effectively generates both general and domain-specific data. Notably, on AlpacaEval 2 and Arena-Hard benchmarks, Llama-3-8B-Base, post-trained on datasets synthesized by MATRIX-Gen with just 20K instruction-response pairs, outperforms Meta's Llama-3-8B-Instruct model, which was trained on over 10M pairs; see our project at https://github.com/ShuoTang123/MATRIX-Gen.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14235v1">Towards Robust Knowledge Representations in Multilingual LLMs for Equivalence and Inheritance based Consistent Reasoning</a></div>
    <div class="paper-meta">
      📅 2024-10-18
    </div>
    <details class="paper-abstract">
      Reasoning and linguistic skills form the cornerstone of human intelligence, facilitating problem-solving and decision-making. Recent advances in Large Language Models (LLMs) have led to impressive linguistic capabilities and emergent reasoning behaviors, fueling widespread adoption across application domains. However, LLMs still struggle with complex reasoning tasks, highlighting their systemic limitations. In this work, we focus on evaluating whether LLMs have the requisite representations to reason using two foundational relationships: "equivalence" and "inheritance". We introduce novel tasks and benchmarks spanning six languages and observe that current SOTA LLMs often produce conflicting answers to the same questions across languages in 17.3-57.5% of cases and violate inheritance constraints in up to 37.2% cases. To enhance consistency across languages, we propose novel "Compositional Representations" where tokens are represented as composition of equivalent tokens across languages, with resulting conflict reduction (up to -4.7%) indicating benefits of shared LLM representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.04808v3">WaterMax: breaking the LLM watermark detectability-robustness-quality trade-off</a></div>
    <div class="paper-meta">
      📅 2024-10-18
    </div>
    <details class="paper-abstract">
      Watermarking is a technical means to dissuade malfeasant usage of Large Language Models. This paper proposes a novel watermarking scheme, so-called WaterMax, that enjoys high detectability while sustaining the quality of the generated text of the original LLM. Its new design leaves the LLM untouched (no modification of the weights, logits, temperature, or sampling technique). WaterMax balances robustness and complexity contrary to the watermarking techniques of the literature inherently provoking a trade-off between quality and robustness. Its performance is both theoretically proven and experimentally validated. It outperforms all the SotA techniques under the most complete benchmark suite. Code available at https://github.com/eva-giboulot/WaterMax.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.19740v2">PertEval: Unveiling Real Knowledge Capacity of LLMs with Knowledge-Invariant Perturbations</a></div>
    <div class="paper-meta">
      📅 2024-10-18
      | 💬 Accepted by NeurIPS '24 D&B Spotlight; 28 pages, 15 figures, 14 tables
    </div>
    <details class="paper-abstract">
      Expert-designed close-ended benchmarks are indispensable in assessing the knowledge capacity of large language models (LLMs). Despite their widespread use, concerns have mounted regarding their reliability due to limited test scenarios and an unavoidable risk of data contamination. To rectify this, we present PertEval, a toolkit devised for in-depth probing of LLMs' knowledge capacity through \textbf{knowledge-invariant perturbations}. These perturbations employ human-like restatement techniques to generate on-the-fly test samples from static benchmarks, meticulously retaining knowledge-critical content while altering irrelevant details. Our toolkit further includes a suite of \textbf{response consistency analyses} that compare performance on raw vs. perturbed test sets to precisely assess LLMs' genuine knowledge capacity. Six representative LLMs are re-evaluated using PertEval. Results reveal significantly inflated performance of the LLMs on raw benchmarks, including an absolute 25.8% overestimation for GPT-4. Additionally, through a nuanced response pattern analysis, we discover that PertEval retains LLMs' uncertainty to specious knowledge, and reveals their potential rote memorization to correct options which leads to overestimated performance. We also find that the detailed response consistency analyses by PertEval could illuminate various weaknesses in existing LLMs' knowledge mastery and guide the development of refinement. Our findings provide insights for advancing more robust and genuinely knowledgeable LLMs. Our code is available at \url{https://github.com/aigc-apps/PertEval}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.01976v5">SciAssess: Benchmarking LLM Proficiency in Scientific Literature Analysis</a></div>
    <div class="paper-meta">
      📅 2024-10-18
    </div>
    <details class="paper-abstract">
      Recent breakthroughs in Large Language Models (LLMs) have revolutionized scientific literature analysis. However, existing benchmarks fail to adequately evaluate the proficiency of LLMs in this domain, particularly in scenarios requiring higher-level abilities beyond mere memorization and the handling of multimodal data. In response to this gap, we introduce SciAssess, a benchmark specifically designed for the comprehensive evaluation of LLMs in scientific literature analysis. It aims to thoroughly assess the efficacy of LLMs by evaluating their capabilities in Memorization (L1), Comprehension (L2), and Analysis \& Reasoning (L3). It encompasses a variety of tasks drawn from diverse scientific fields, including biology, chemistry, material, and medicine. To ensure the reliability of SciAssess, rigorous quality control measures have been implemented, ensuring accuracy, anonymization, and compliance with copyright standards. SciAssess evaluates 11 LLMs, highlighting their strengths and areas for improvement. We hope this evaluation supports the ongoing development of LLM applications in scientific literature analysis. SciAssess and its resources are available at \url{https://github.com/sci-assess/SciAssess}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14202v1">Rationale Behind Essay Scores: Enhancing S-LLM's Multi-Trait Essay Scoring with Rationale Generated by LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-18
    </div>
    <details class="paper-abstract">
      Existing automated essay scoring (AES) has solely relied on essay text without using explanatory rationales for the scores, thereby forgoing an opportunity to capture the specific aspects evaluated by rubric indicators in a fine-grained manner. This paper introduces Rationale-based Multiple Trait Scoring (RMTS), a novel approach for multi-trait essay scoring that integrates prompt-engineering-based large language models (LLMs) with a fine-tuning-based essay scoring model using a smaller large language model (S-LLM). RMTS uses an LLM-based trait-wise rationale generation system where a separate LLM agent generates trait-specific rationales based on rubric guidelines, which the scoring model uses to accurately predict multi-trait scores. Extensive experiments on benchmark datasets, including ASAP, ASAP++, and Feedback Prize, show that RMTS significantly outperforms state-of-the-art models and vanilla S-LLMs in trait-specific scoring. By assisting quantitative assessment with fine-grained qualitative rationales, RMTS enhances the trait-wise reliability, providing partial explanations about essays.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.06833v2">Does Mapo Tofu Contain Coffee? Probing LLMs for Food-related Cultural Knowledge</a></div>
    <div class="paper-meta">
      📅 2024-10-18
      | 💬 cultural bias analysis, cultural knowledge probing, large language models, cultural NLP
    </div>
    <details class="paper-abstract">
      Recent studies have highlighted the presence of cultural biases in Large Language Models (LLMs), yet often lack a robust methodology to dissect these phenomena comprehensively. Our work aims to bridge this gap by delving into the Food domain, a universally relevant yet culturally diverse aspect of human life. We introduce FmLAMA, a multilingual dataset centered on food-related cultural facts and variations in food practices. We analyze LLMs across various architectures and configurations, evaluating their performance in both monolingual and multilingual settings. By leveraging templates in six different languages, we investigate how LLMs interact with language-specific and cultural knowledge. Our findings reveal that (1) LLMs demonstrate a pronounced bias towards food knowledge prevalent in the United States; (2) Incorporating relevant cultural context significantly improves LLMs' ability to access cultural knowledge; (3) The efficacy of LLMs in capturing cultural nuances is highly dependent on the interplay between the probing language, the specific model architecture, and the cultural context in question. This research underscores the complexity of integrating cultural understanding into LLMs and emphasizes the importance of culturally diverse datasets to mitigate biases and enhance model performance across different cultural domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.19000v1">Make LLMs better zero-shot reasoners: Structure-orientated autonomous reasoning</a></div>
    <div class="paper-meta">
      📅 2024-10-18
    </div>
    <details class="paper-abstract">
      Zero-shot reasoning methods with Large Language Models (LLMs) offer significant advantages including great generalization to novel tasks and reduced dependency on human-crafted examples. However, the current zero-shot methods still have limitations in complex tasks, e.g., answering questions that require multi-step reasoning. In this paper, we address this limitation by introducing a novel structure-oriented analysis method to help LLMs better understand the question and guide the problem-solving process of LLMs. We first demonstrate how the existing reasoning strategies, Chain-of-Thought and ReAct, can benefit from our structure-oriented analysis. In addition to empirical investigations, we leverage the probabilistic graphical model to theoretically explain why our structure-oriented analysis can improve the LLM reasoning process. To further improve the reliability in complex question-answering tasks, we propose a multi-agent reasoning system, Structure-oriented Autonomous Reasoning Agents (SARA), that can better enforce the reasoning process following our structure-oriented analysis by refinement techniques and is equipped with external knowledge retrieval capability to reduce factual errors. Extensive experiments verify the effectiveness of the proposed reasoning system. Surprisingly, in some cases, the system even surpasses few-shot methods. Finally, the system not only improves reasoning accuracy in complex tasks but also demonstrates robustness against potential attacks that corrupt the reasoning process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14182v1">LabSafety Bench: Benchmarking LLMs on Safety Issues in Scientific Labs</a></div>
    <div class="paper-meta">
      📅 2024-10-18
      | 💬 50 pages, 19 figures
    </div>
    <details class="paper-abstract">
      Laboratory accidents pose significant risks to human life and property, underscoring the importance of robust safety protocols. Despite advancements in safety training, laboratory personnel may still unknowingly engage in unsafe practices. With the increasing reliance on large language models (LLMs) for guidance in various fields, including laboratory settings, there is a growing concern about their reliability in critical safety-related decision-making. Unlike trained human researchers, LLMs lack formal lab safety education, raising questions about their ability to provide safe and accurate guidance. Existing research on LLM trustworthiness primarily focuses on issues such as ethical compliance, truthfulness, and fairness but fails to fully cover safety-critical real-world applications, like lab safety. To address this gap, we propose the Laboratory Safety Benchmark (LabSafety Bench), a comprehensive evaluation framework based on a new taxonomy aligned with Occupational Safety and Health Administration (OSHA) protocols. This benchmark includes 765 multiple-choice questions verified by human experts, assessing LLMs and vision language models (VLMs) performance in lab safety contexts. Our evaluations demonstrate that while GPT-4o outperforms human participants, it is still prone to critical errors, highlighting the risks of relying on LLMs in safety-critical environments. Our findings emphasize the need for specialized benchmarks to accurately assess the trustworthiness of LLMs in real-world safety applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.15545v3">SciLitLLM: How to Adapt LLMs for Scientific Literature Understanding</a></div>
    <div class="paper-meta">
      📅 2024-10-18
    </div>
    <details class="paper-abstract">
      Scientific literature understanding is crucial for extracting targeted information and garnering insights, thereby significantly advancing scientific discovery. Despite the remarkable success of Large Language Models (LLMs), they face challenges in scientific literature understanding, primarily due to (1) a lack of scientific knowledge and (2) unfamiliarity with specialized scientific tasks. To develop an LLM specialized in scientific literature understanding, we propose a hybrid strategy that integrates continual pre-training (CPT) and supervised fine-tuning (SFT), to simultaneously infuse scientific domain knowledge and enhance instruction-following capabilities for domain-specific tasks.cIn this process, we identify two key challenges: (1) constructing high-quality CPT corpora, and (2) generating diverse SFT instructions. We address these challenges through a meticulous pipeline, including PDF text extraction, parsing content error correction, quality filtering, and synthetic instruction creation. Applying this strategy, we present a suite of LLMs: SciLitLLM, specialized in scientific literature understanding. These models demonstrate promising performance on scientific literature understanding benchmarks. Our contributions are threefold: (1) We present an effective framework that integrates CPT and SFT to adapt LLMs to scientific literature understanding, which can also be easily adapted to other domains. (2) We propose an LLM-based synthesis method to generate diverse and high-quality scientific instructions, resulting in a new instruction set -- SciLitIns -- for supervised fine-tuning in less-represented scientific domains. (3) SciLitLLM achieves promising performance improvements on scientific literature understanding benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13276v2">SeerAttention: Learning Intrinsic Sparse Attention in Your LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-18
    </div>
    <details class="paper-abstract">
      Attention is the cornerstone of modern Large Language Models (LLMs). Yet its quadratic complexity limits the efficiency and scalability of LLMs, especially for those with a long-context window. A promising approach addressing this limitation is to leverage the sparsity in attention. However, existing sparsity-based solutions predominantly rely on predefined patterns or heuristics to approximate sparsity. This practice falls short to fully capture the dynamic nature of attention sparsity in language-based tasks. This paper argues that attention sparsity should be learned rather than predefined. To this end, we design SeerAttention, a new Attention mechanism that augments the conventional attention with a learnable gate that adaptively selects significant blocks in an attention map and deems the rest blocks sparse. Such block-level sparsity effectively balances accuracy and speedup. To enable efficient learning of the gating network, we develop a customized FlashAttention implementation that extracts the block-level ground truth of attention map with minimum overhead. SeerAttention not only applies to post-training, but also excels in long-context fine-tuning. Our results show that at post-training stages, SeerAttention significantly outperforms state-of-the-art static or heuristic-based sparse attention methods, while also being more versatile and flexible to adapt to varying context lengths and sparsity ratios. When applied to long-context fine-tuning with YaRN, SeerAttention can achieve a remarkable 90% sparsity ratio at a 32k context length with minimal perplexity loss, offering a 5.67x speedup over FlashAttention-2.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.05541v2">P3: A Policy-Driven, Pace-Adaptive, and Diversity-Promoted Framework for data pruning in LLM Training</a></div>
    <div class="paper-meta">
      📅 2024-10-18
    </div>
    <details class="paper-abstract">
      In the rapidly advancing field of Large Language Models (LLMs), effectively leveraging existing datasets during fine-tuning to maximize the model's potential is of paramount importance. This paper introduces P3, an adaptive framework aimed at optimizing the task-specific fine-tuning process through iterative data pruning. P3 consists of three key components: (1) Policy-driven Difficulty Measurement, which dynamically assesses data difficulty based on the model's real-time performance, replacing static metrics with adaptable evaluations; (2) Pace-Adaptive Selection, leveraging self-paced learning to progressively introduce more challenging data, thereby enhancing model capability; (3) Diversity Promotion, incorporating Determinantal Point Process (DPP) to ensure data diversity across epochs, enriching the learning process. We validate P3 on the reasoning scenarios, APPS and MATH, demonstrating significant improvements over traditional data pruning methods. By advancing dynamic data selection and utilization strategies, P3 contributes both a theoretical framework and concrete approach to fully exploit existing data for LLMs' performance improvement, offering utility across diverse tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14166v1">LLM The Genius Paradox: A Linguistic and Math Expert's Struggle with Simple Word-based Counting Problems</a></div>
    <div class="paper-meta">
      📅 2024-10-18
    </div>
    <details class="paper-abstract">
      Interestingly, LLMs yet struggle with some basic tasks that humans find trivial to handle, e.g., counting the number of character r's in the word "strawberry". There are several popular conjectures (e.g., tokenization, architecture and training data) regarding the reason for deficiency of LLMs in simple word-based counting problems, sharing the similar belief that such failure stems from model pretraining hence probably inevitable during deployment. In this paper, we carefully design multiple evaluation settings to investigate validity of prevalent conjectures. Meanwhile, we measure transferability of advanced mathematical and coding reasoning capabilities from specialized LLMs to simple counting tasks. Although specialized LLMs suffer from counting problems as well, we find conjectures about inherent deficiency of LLMs invalid and further seek opportunities to elicit knowledge and capabilities from LLMs that are beneficial to counting tasks. Compared with strategies such as finetuning and in-context learning that are commonly adopted to enhance performance on new or challenging tasks, we show that engaging reasoning is the most robust and efficient way to help LLMs better perceive tasks with more accurate responses. We hope our conjecture validation design could provide insights into the study of future critical failure modes of LLMs. Based on challenges in transferring advanced capabilities to much simpler tasks, we call for more attention to model capability acquisition and evaluation. We also highlight the importance of cultivating consciousness of "reasoning before responding" during model pretraining.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.13170v2">Amphista: Bi-directional Multi-head Decoding for Accelerating LLM Inference</a></div>
    <div class="paper-meta">
      📅 2024-10-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) inherently use autoregressive decoding, which lacks parallelism in inference and results in significantly slow inference speed. While methods such as Medusa constructs parallelized heads, they lack adequate information interaction across different prediction positions. To overcome this limitation, we introduce Amphista, an enhanced speculative decoding framework that builds upon Medusa. Specifically, Amphista models an Auto-embedding Block capable of parallel inference, incorporating bi-directional attention to enable interaction between different drafting heads. Additionally, Amphista integrates Staged Adaptation Layers, which ensure a seamless transition of semantic information from the target model's autoregressive inference to the drafting heads' non-autoregressive inference, effectively achieving paradigm shift and feature fusion. Experimental results on Vicuna models using MT-Bench and Spec-Bench demonstrate that Amphista achieves substantial acceleration while maintaining generation quality. On MT-Bench, Amphista delivers up to 2.75$\times$ speedup over vanilla autoregressive decoding and 1.40$\times$ over Medusa on Vicuna 33B in wall-clock time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14152v1">SRAP-Agent: Simulating and Optimizing Scarce Resource Allocation Policy with LLM-based Agent</a></div>
    <div class="paper-meta">
      📅 2024-10-18
    </div>
    <details class="paper-abstract">
      Public scarce resource allocation plays a crucial role in economics as it directly influences the efficiency and equity in society. Traditional studies including theoretical model-based, empirical study-based and simulation-based methods encounter limitations due to the idealized assumption of complete information and individual rationality, as well as constraints posed by limited available data. In this work, we propose an innovative framework, SRAP-Agent (Simulating and Optimizing Scarce Resource Allocation Policy with LLM-based Agent), which integrates Large Language Models (LLMs) into economic simulations, aiming to bridge the gap between theoretical models and real-world dynamics. Using public housing allocation scenarios as a case study, we conduct extensive policy simulation experiments to verify the feasibility and effectiveness of the SRAP-Agent and employ the Policy Optimization Algorithm with certain optimization objectives. The source code can be found in https://github.com/jijiarui-cather/SRAPAgent_Framework
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.11278v2">Do Not Design, Learn: A Trainable Scoring Function for Uncertainty Estimation in Generative LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-18
    </div>
    <details class="paper-abstract">
      Uncertainty estimation (UE) of generative large language models (LLMs) is crucial for evaluating the reliability of generated sequences. A significant subset of UE methods utilize token probabilities to assess uncertainty, aggregating multiple token probabilities into a single UE score using a scoring function. Existing scoring functions for probability-based UE, such as length-normalized scoring and semantic contribution-based weighting, are designed to solve certain aspects of the problem but exhibit limitations, including the inability to handle biased probabilities and complex semantic dependencies between tokens. To address these issues, in this work, we propose Learnable Response Scoring (LARS) function, a novel scoring function that leverages supervised data to capture complex dependencies between tokens and probabilities, thereby producing more reliable and calibrated response scores in computing the uncertainty of LLM generations. Our comprehensive experiments across question-answering and arithmetical reasoning tasks with various datasets demonstrate that LARS significantly outperforms existing scoring functions, achieving improvements of up to 16\% AUROC score.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14074v1">Be My Donor. Transfer the NLP Datasets Between the Languages Using LLM</a></div>
    <div class="paper-meta">
      📅 2024-10-17
    </div>
    <details class="paper-abstract">
      In this work, we investigated how one can use the LLM to transfer the dataset and its annotation from one language to another. This is crucial since sharing the knowledge between different languages could boost certain underresourced directions in the target language, saving lots of efforts in data annotation or quick prototyping. We experiment with English and Russian pairs translating the DEFT corpus. This corpus contains three layers of annotation dedicated to term-definition pair mining, which is a rare annotation type for Russian. We provide a pipeline for the annotation transferring using ChatGPT3.5-turbo and Llama-3.1-8b as core LLMs. In the end, we train the BERT-based models on the translated dataset to establish a baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08418v2">Can LLMs advance democratic values?</a></div>
    <div class="paper-meta">
      📅 2024-10-17
    </div>
    <details class="paper-abstract">
      LLMs are among the most advanced tools ever devised for analysing and generating linguistic content. Democratic deliberation and decision-making involve, at several distinct stages, the production and analysis of language. So it is natural to ask whether our best tools for manipulating language might prove instrumental to one of our most important linguistic tasks. Researchers and practitioners have recently asked whether LLMs can support democratic deliberation by leveraging abilities to summarise content, as well as to aggregate opinion over summarised content, and indeed to represent voters by predicting their preferences over unseen choices. In this paper, we assess whether using LLMs to perform these and related functions really advances the democratic values that inspire these experiments. We suggest that the record is decidedly mixed. In the presence of background inequality of power and resources, as well as deep moral and political disagreement, we should be careful not to use LLMs in ways that automate non-instrumentally valuable components of the democratic process, or else threaten to supplant fair and transparent decision-making procedures that are necessary to reconcile competing interests and values. However, while we argue that LLMs should be kept well clear of formal democratic decision-making processes, we think that they can be put to good use in strengthening the informal public sphere: the arena that mediates between democratic governments and the polities that they serve, in which political communities seek information, form civic publics, and hold their leaders to account.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16322v1">SouLLMate: An Application Enhancing Diverse Mental Health Support with Adaptive LLMs, Prompt Engineering, and RAG Techniques</a></div>
    <div class="paper-meta">
      📅 2024-10-17
      | 💬 26 pages, 19 figures, 8 tables
    </div>
    <details class="paper-abstract">
      Mental health issues significantly impact individuals' daily lives, yet many do not receive the help they need even with available online resources. This study aims to provide diverse, accessible, stigma-free, personalized, and real-time mental health support through cutting-edge AI technologies. It makes the following contributions: (1) Conducting an extensive survey of recent mental health support methods to identify prevalent functionalities and unmet needs. (2) Introducing SouLLMate, an adaptive LLM-driven system that integrates LLM technologies, Chain, Retrieval-Augmented Generation (RAG), prompt engineering, and domain knowledge. This system offers advanced features such as Risk Detection and Proactive Guidance Dialogue, and utilizes RAG for personalized profile uploads and Conversational Information Extraction. (3) Developing novel evaluation approaches for preliminary assessments and risk detection via professionally annotated interview data and real-life suicide tendency data. (4) Proposing the Key Indicator Summarization (KIS), Proactive Questioning Strategy (PQS), and Stacked Multi-Model Reasoning (SMMR) methods to enhance model performance and usability through context-sensitive response adjustments, semantic coherence evaluations, and enhanced accuracy of long-context reasoning in language models. This study contributes to advancing mental health support technologies, potentially improving the accessibility and effectiveness of mental health care globally.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.07235v2">LLM-aided explanations of EDA synthesis errors</a></div>
    <div class="paper-meta">
      📅 2024-10-17
      | 💬 6 pages, 6 figures. Accepted in IEEE LLM Aided Design Workshop (LAD'2024)
    </div>
    <details class="paper-abstract">
      Training new engineers in digital design is a challenge, particularly when it comes to teaching the complex electronic design automation (EDA) tooling used in this domain. Learners will typically deploy designs in the Verilog and VHDL hardware description languages to Field Programmable Gate Arrays (FPGAs) from Altera (Intel) and Xilinx (AMD) via proprietary closed-source toolchains (Quartus Prime and Vivado, respectively). These tools are complex and difficult to use -- yet, as they are the tools used in industry, they are an essential first step in this space. In this work, we examine how recent advances in artificial intelligence may be leveraged to address aspects of this challenge. Specifically, we investigate if Large Language Models (LLMs), which have demonstrated text comprehension and question-answering capabilities, can be used to generate novice-friendly explanations of compile-time synthesis error messages from Quartus Prime and Vivado. To perform this study we generate 936 error message explanations using three OpenAI LLMs over 21 different buggy code samples. These are then graded for relevance and correctness, and we find that in approximately 71% of cases the LLMs give correct & complete explanations suitable for novice learners.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14012v1">LLMs are Biased Teachers: Evaluating LLM Bias in Personalized Education</a></div>
    <div class="paper-meta">
      📅 2024-10-17
      | 💬 46 Pages, 55 Figures, dataset release pending publication
    </div>
    <details class="paper-abstract">
      With the increasing adoption of large language models (LLMs) in education, concerns about inherent biases in these models have gained prominence. We evaluate LLMs for bias in the personalized educational setting, specifically focusing on the models' roles as "teachers". We reveal significant biases in how models generate and select educational content tailored to different demographic groups, including race, ethnicity, sex, gender, disability status, income, and national origin. We introduce and apply two bias score metrics--Mean Absolute Bias (MAB) and Maximum Difference Bias (MDB)--to analyze 9 open and closed state-of-the-art LLMs. Our experiments, which utilize over 17,000 educational explanations across multiple difficulty levels and topics, uncover that models perpetuate both typical and inverted harmful stereotypes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13984v1">Are LLMs Models of Distributional Semantics? A Case Study on Quantifiers</a></div>
    <div class="paper-meta">
      📅 2024-10-17
      | 💬 9 Pages, 3 Figures
    </div>
    <details class="paper-abstract">
      Distributional semantics is the linguistic theory that a word's meaning can be derived from its distribution in natural language (i.e., its use). Language models are commonly viewed as an implementation of distributional semantics, as they are optimized to capture the statistical features of natural language. It is often argued that distributional semantics models should excel at capturing graded/vague meaning based on linguistic conventions, but struggle with truth-conditional reasoning and symbolic processing. We evaluate this claim with a case study on vague (e.g. "many") and exact (e.g. "more than half") quantifiers. Contrary to expectations, we find that, across a broad range of models of various types, LLMs align more closely with human judgements on exact quantifiers versus vague ones. These findings call for a re-evaluation of the assumptions underpinning what distributional semantics models are, as well as what they can capture.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.02958v3">PrE-Text: Training Language Models on Private Federated Data in the Age of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-17
      | 💬 ICML 2024 (Oral). Latest revision corrects a discussion on concurrent work arXiv:2403.01749. We described their work as reliant on using closed-sourced models when in reality they also evaluate and use open source models. This has been corrected in this version
    </div>
    <details class="paper-abstract">
      On-device training is currently the most common approach for training machine learning (ML) models on private, distributed user data. Despite this, on-device training has several drawbacks: (1) most user devices are too small to train large models on-device, (2) on-device training is communication- and computation-intensive, and (3) on-device training can be difficult to debug and deploy. To address these problems, we propose Private Evolution-Text (PrE-Text), a method for generating differentially private (DP) synthetic textual data. First, we show that across multiple datasets, training small models (models that fit on user devices) with PrE-Text synthetic data outperforms small models trained on-device under practical privacy regimes ($\epsilon=1.29$, $\epsilon=7.58$). We achieve these results while using 9$\times$ fewer rounds, 6$\times$ less client computation per round, and 100$\times$ less communication per round. Second, finetuning large models on PrE-Text's DP synthetic data improves large language model (LLM) performance on private data across the same range of privacy budgets. Altogether, these results suggest that training on DP synthetic data can be a better option than training a model on-device on private distributed data. Code is available at https://github.com/houcharlie/PrE-Text.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.11773v2">Behavior Alignment: A New Perspective of Evaluating LLM-based Conversational Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2024-10-17
      | 💬 Accepted by the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated great potential in Conversational Recommender Systems (CRS). However, the application of LLMs to CRS has exposed a notable discrepancy in behavior between LLM-based CRS and human recommenders: LLMs often appear inflexible and passive, frequently rushing to complete the recommendation task without sufficient inquiry.This behavior discrepancy can lead to decreased accuracy in recommendations and lower user satisfaction. Despite its importance, existing studies in CRS lack a study about how to measure such behavior discrepancy. To fill this gap, we propose Behavior Alignment, a new evaluation metric to measure how well the recommendation strategies made by a LLM-based CRS are consistent with human recommenders'. Our experiment results show that the new metric is better aligned with human preferences and can better differentiate how systems perform than existing evaluation metrics. As Behavior Alignment requires explicit and costly human annotations on the recommendation strategies, we also propose a classification-based method to implicitly measure the Behavior Alignment based on the responses. The evaluation results confirm the robustness of the method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13961v1">From Single to Multi: How LLMs Hallucinate in Multi-Document Summarization</a></div>
    <div class="paper-meta">
      📅 2024-10-17
    </div>
    <details class="paper-abstract">
      Although many studies have investigated and reduced hallucinations in large language models (LLMs) for single-document tasks, research on hallucination in multi-document summarization (MDS) tasks remains largely unexplored. Specifically, it is unclear how the challenges arising from handling multiple documents (e.g., repetition and diversity of information) affect models outputs. In this work, we investigate how hallucinations manifest in LLMs when summarizing topic-specific information from multiple documents. Since no benchmarks exist for investigating hallucinations in MDS, we use existing news and conversation datasets, annotated with topic-specific insights, to create two novel multi-document benchmarks. When evaluating 5 LLMs on our benchmarks, we observe that on average, up to 75% of the content in LLM-generated summary is hallucinated, with hallucinations more likely to occur towards the end of the summaries. Moreover, when summarizing non-existent topic-related information, gpt-3.5-turbo and GPT-4o still generate summaries about 79.35% and 44% of the time, raising concerns about their tendency to fabricate content. To understand the characteristics of these hallucinations, we manually evaluate 700+ insights and find that most errors stem from either failing to follow instructions or producing overly generic insights. Motivated by these observations, we investigate the efficacy of simple post-hoc baselines in mitigating hallucinations but find them only moderately effective. Our results underscore the need for more effective approaches to systematically mitigate hallucinations in MDS. We release our dataset and code at github.com/megagonlabs/Hallucination_MDS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.04819v4">DALK: Dynamic Co-Augmentation of LLMs and KG to answer Alzheimer's Disease Questions with Scientific Literature</a></div>
    <div class="paper-meta">
      📅 2024-10-17
      | 💬 Accepted by EMNLP 2024 Findings; revise format problem
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have achieved promising performances across various applications. Nonetheless, the ongoing challenge of integrating long-tail knowledge continues to impede the seamless adoption of LLMs in specialized domains. In this work, we introduce DALK, a.k.a. Dynamic Co-Augmentation of LLMs and KG, to address this limitation and demonstrate its ability on studying Alzheimer's Disease (AD), a specialized sub-field in biomedicine and a global health priority. With a synergized framework of LLM and KG mutually enhancing each other, we first leverage LLM to construct an evolving AD-specific knowledge graph (KG) sourced from AD-related scientific literature, and then we utilize a coarse-to-fine sampling method with a novel self-aware knowledge retrieval approach to select appropriate knowledge from the KG to augment LLM inference capabilities. The experimental results, conducted on our constructed AD question answering (ADQA) benchmark, underscore the efficacy of DALK. Additionally, we perform a series of detailed analyses that can offer valuable insights and guidelines for the emerging topic of mutually enhancing KG and LLM. We will release the code and data at https://github.com/David-Li0406/DALK.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13944v1">Boosting LLM Translation Skills without General Ability Loss via Rationale Distillation</a></div>
    <div class="paper-meta">
      📅 2024-10-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved impressive results across numerous NLP tasks but still encounter difficulties in machine translation. Traditional methods to improve translation have typically involved fine-tuning LLMs using parallel corpora. However, vanilla fine-tuning often leads to catastrophic forgetting of the instruction-following capabilities and alignment with human preferences, compromising their broad general abilities and introducing potential security risks. These abilities, which are developed using proprietary and unavailable training data, make existing continual instruction tuning methods ineffective. To overcome this issue, we propose a novel approach called RaDis (Rationale Distillation). RaDis harnesses the strong generative capabilities of LLMs to create rationales for training data, which are then "replayed" to prevent forgetting. These rationales encapsulate general knowledge and safety principles, acting as self-distillation targets to regulate the training process. By jointly training on both reference translations and self-generated rationales, the model can learn new translation skills while preserving its overall general abilities. Extensive experiments demonstrate that our method enhances machine translation performance while maintaining the broader capabilities of LLMs across other tasks. This work presents a pathway for creating more versatile LLMs that excel in specialized tasks without compromising generality and safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13857v1">How Numerical Precision Affects Mathematical Reasoning Capabilities of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-17
    </div>
    <details class="paper-abstract">
      Despite the remarkable success of Transformer-based Large Language Models (LLMs) across various domains, understanding and enhancing their mathematical capabilities remains a significant challenge. In this paper, we conduct a rigorous theoretical analysis of LLMs' mathematical abilities, with a specific focus on their arithmetic performances. We identify numerical precision as a key factor that influences their effectiveness in mathematical tasks. Our results show that Transformers operating with low numerical precision fail to address arithmetic tasks, such as iterated addition and integer multiplication, unless the model size grows super-polynomially with respect to the input length. In contrast, Transformers with standard numerical precision can efficiently handle these tasks with significantly smaller model sizes. We further support our theoretical findings through empirical experiments that explore the impact of varying numerical precision on arithmetic tasks, providing valuable insights for improving the mathematical reasoning capabilities of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08928v2">Towards Multilingual LLM Evaluation for European Languages</a></div>
    <div class="paper-meta">
      📅 2024-10-17
    </div>
    <details class="paper-abstract">
      The rise of Large Language Models (LLMs) has revolutionized natural language processing across numerous languages and tasks. However, evaluating LLM performance in a consistent and meaningful way across multiple European languages remains challenging, especially due to the scarcity of language-parallel multilingual benchmarks. We introduce a multilingual evaluation approach tailored for European languages. We employ translated versions of five widely-used benchmarks to assess the capabilities of 40 LLMs across 21 European languages. Our contributions include examining the effectiveness of translated benchmarks, assessing the impact of different translation services, and offering a multilingual evaluation framework for LLMs that includes newly created datasets: EU20-MMLU, EU20-HellaSwag, EU20-ARC, EU20-TruthfulQA, and EU20-GSM8K. The benchmarks and results are made publicly available to encourage further research in multilingual LLM evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.16833v2">Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study and Hybrid Approach</a></div>
    <div class="paper-meta">
      📅 2024-10-17
      | 💬 Accepted to EMNLP 2024 industry track
    </div>
    <details class="paper-abstract">
      Retrieval Augmented Generation (RAG) has been a powerful tool for Large Language Models (LLMs) to efficiently process overly lengthy contexts. However, recent LLMs like Gemini-1.5 and GPT-4 show exceptional capabilities to understand long contexts directly. We conduct a comprehensive comparison between RAG and long-context (LC) LLMs, aiming to leverage the strengths of both. We benchmark RAG and LC across various public datasets using three latest LLMs. Results reveal that when resourced sufficiently, LC consistently outperforms RAG in terms of average performance. However, RAG's significantly lower cost remains a distinct advantage. Based on this observation, we propose Self-Route, a simple yet effective method that routes queries to RAG or LC based on model self-reflection. Self-Route significantly reduces the computation cost while maintaining a comparable performance to LC. Our findings provide a guideline for long-context applications of LLMs using RAG and LC.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13825v1">AgentOccam: A Simple Yet Strong Baseline for LLM-Based Web Agents</a></div>
    <div class="paper-meta">
      📅 2024-10-17
    </div>
    <details class="paper-abstract">
      Autonomy via agents using large language models (LLMs) for personalized, standardized tasks boosts human efficiency. Automating web tasks (like booking hotels within a budget) is increasingly sought after. Fulfilling practical needs, the web agent also serves as an important proof-of-concept example for various agent grounding scenarios, with its success promising advancements in many future applications. Prior research often handcrafts web agent strategies (e.g., prompting templates, multi-agent systems, search methods, etc.) and the corresponding in-context examples, which may not generalize well across all real-world scenarios. On the other hand, there has been limited study on the misalignment between a web agent's observation/action representation and the pre-training data of the LLM it's based on. This discrepancy is especially notable when LLMs are primarily trained for language completion rather than tasks involving embodied navigation actions and symbolic web elements. Our study enhances an LLM-based web agent by simply refining its observation and action space to better align with the LLM's capabilities. This approach enables our base agent to significantly outperform previous methods on a wide variety of web tasks. Specifically, on WebArena, a benchmark featuring general-purpose web interaction tasks, our agent AgentOccam surpasses the previous state-of-the-art and concurrent work by 9.8 (+29.4%) and 5.9 (+15.8%) absolute points respectively, and boosts the success rate by 26.6 points (+161%) over similar plain web agents with its observation and action space alignment. We achieve this without using in-context examples, new agent roles, online feedback or search strategies. AgentOccam's simple design highlights LLMs' impressive zero-shot performance on web tasks, and underlines the critical role of carefully tuning observation and action spaces for LLM-based agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.19811v1">ControlAgent: Automating Control System Design via Novel Integration of LLM Agents and Domain Expertise</a></div>
    <div class="paper-meta">
      📅 2024-10-17
    </div>
    <details class="paper-abstract">
      Control system design is a crucial aspect of modern engineering with far-reaching applications across diverse sectors including aerospace, automotive systems, power grids, and robotics. Despite advances made by Large Language Models (LLMs) in various domains, their application in control system design remains limited due to the complexity and specificity of control theory. To bridge this gap, we introduce ControlAgent, a new paradigm that automates control system design via novel integration of LLM agents and control-oriented domain expertise. ControlAgent encodes expert control knowledge and emulates human iterative design processes by gradually tuning controller parameters to meet user-specified requirements for stability, performance, and robustness. ControlAgent integrates multiple collaborative LLM agents, including a central agent responsible for task distribution and task-specific agents dedicated to detailed controller design for various types of systems and requirements. ControlAgent also employs a Python computation agent that performs complex calculations and controller evaluations based on standard design information provided by task-specified LLM agents. Combined with a history and feedback module, the task-specific LLM agents iteratively refine controller parameters based on real-time feedback from prior designs. Overall, ControlAgent mimics the design processes used by (human) practicing engineers, but removes all the human efforts and can be run in a fully automated way to give end-to-end solutions for control system design with user-specified requirements. To validate ControlAgent's effectiveness, we develop ControlEval, an evaluation dataset that comprises 500 control tasks with various specific design goals. The effectiveness of ControlAgent is demonstrated via extensive comparative evaluations between LLM-based and traditional human-involved toolbox-based baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13788v1">Modeling Future Conversation Turns to Teach LLMs to Ask Clarifying Questions</a></div>
    <div class="paper-meta">
      📅 2024-10-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) must often respond to highly ambiguous user requests. In such cases, the LLM's best response may be to ask a clarifying question to elicit more information. We observe existing LLMs often respond by presupposing a single interpretation of such ambiguous requests, frustrating users who intended a different interpretation. We speculate this is caused by current preference data labeling practice, where LLM responses are evaluated only on their prior contexts. To address this, we propose to assign preference labels by simulating their expected outcomes in the future turns. This allows LLMs to learn to ask clarifying questions when it can generate responses that are tailored to each user interpretation in future turns. In experiments on open-domain QA, we compare systems that trained using our proposed preference labeling methods against standard methods, which assign preferences based on only prior context. We evaluate systems based on their ability to ask clarifying questions that can recover each user's interpretation and expected answer, and find that our training with our proposed method trains LLMs to ask clarifying questions with a 5% improvement in F1 measured against the answer set from different interpretations of each query
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13768v1">Rapid and Automated Alloy Design with Graph Neural Network-Powered LLM-Driven Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2024-10-17
    </div>
    <details class="paper-abstract">
      A multi-agent AI model is used to automate the discovery of new metallic alloys, integrating multimodal data and external knowledge including insights from physics via atomistic simulations. Our multi-agent system features three key components: (a) a suite of LLMs responsible for tasks such as reasoning and planning, (b) a group of AI agents with distinct roles and expertise that dynamically collaborate, and (c) a newly developed graph neural network (GNN) model for rapid retrieval of key physical properties. A set of LLM-driven AI agents collaborate to automate the exploration of the vast design space of MPEAs, guided by predictions from the GNN. We focus on the NbMoTa family of body-centered cubic (bcc) alloys, modeled using an ML-based interatomic potential, and target two key properties: the Peierls barrier and solute/screw dislocation interaction energy. Our GNN model accurately predicts these atomic-scale properties, providing a faster alternative to costly brute-force calculations and reducing the computational burden on multi-agent systems for physics retrieval. This AI system revolutionizes materials discovery by reducing reliance on human expertise and overcoming the limitations of direct all-atom simulations. By synergizing the predictive power of GNNs with the dynamic collaboration of LLM-based agents, the system autonomously navigates vast alloy design spaces, identifying trends in atomic-scale material properties and predicting macro-scale mechanical strength, as demonstrated by several computational experiments. This approach accelerates the discovery of advanced alloys and holds promise for broader applications in other complex systems, marking a significant step forward in automated materials design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14745v1">SemiEvol: Semi-supervised Fine-tuning for LLM Adaptation</a></div>
    <div class="paper-meta">
      📅 2024-10-17
    </div>
    <details class="paper-abstract">
      Supervised fine-tuning (SFT) is crucial in adapting large language models (LLMs) to a specific domain or task. However, only a limited amount of labeled data is available in practical applications, which poses a severe challenge for SFT in yielding satisfactory results. Therefore, a data-efficient framework that can fully exploit labeled and unlabeled data for LLM fine-tuning is highly anticipated. Towards this end, we introduce a semi-supervised fine-tuning framework named SemiEvol for LLM adaptation from a propagate-and-select manner. For knowledge propagation, SemiEvol adopts a bi-level approach, propagating knowledge from labeled data to unlabeled data through both in-weight and in-context methods. For knowledge selection, SemiEvol incorporates a collaborative learning mechanism, selecting higher-quality pseudo-response samples. We conducted experiments using GPT-4o-mini and Llama-3.1 on seven general or domain-specific datasets, demonstrating significant improvements in model performance on target data. Furthermore, we compared SemiEvol with SFT and self-evolution methods, highlighting its practicality in hybrid data scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13727v1">LLM-Human Pipeline for Cultural Context Grounding of Conversations</a></div>
    <div class="paper-meta">
      📅 2024-10-17
      | 💬 19 pages, 9 figures, 7 tables
    </div>
    <details class="paper-abstract">
      Conversations often adhere to well-understood social norms that vary across cultures. For example, while "addressing parents by name" is commonplace in the West, it is rare in most Asian cultures. Adherence or violation of such norms often dictates the tenor of conversations. Humans are able to navigate social situations requiring cultural awareness quite adeptly. However, it is a hard task for NLP models. In this paper, we tackle this problem by introducing a "Cultural Context Schema" for conversations. It comprises (1) conversational information such as emotions, dialogue acts, etc., and (2) cultural information such as social norms, violations, etc. We generate ~110k social norm and violation descriptions for ~23k conversations from Chinese culture using LLMs. We refine them using automated verification strategies which are evaluated against culturally aware human judgements. We organize these descriptions into meaningful structures we call "Norm Concepts", using an interactive human-in-loop framework. We ground the norm concepts and the descriptions in conversations using symbolic annotation. Finally, we use the obtained dataset for downstream tasks such as emotion, sentiment, and dialogue act detection. We show that it significantly improves the empirical performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.14462v2">Modeling Human Subjectivity in LLMs Using Explicit and Implicit Human Factors in Personas</a></div>
    <div class="paper-meta">
      📅 2024-10-17
      | 💬 Accepted at Findings of EMNLP 2024
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly being used in human-centered social scientific tasks, such as data annotation, synthetic data creation, and engaging in dialog. However, these tasks are highly subjective and dependent on human factors, such as one's environment, attitudes, beliefs, and lived experiences. Thus, it may be the case that employing LLMs (which do not have such human factors) in these tasks results in a lack of variation in data, failing to reflect the diversity of human experiences. In this paper, we examine the role of prompting LLMs with human-like personas and asking the models to answer as if they were a specific human. This is done explicitly, with exact demographics, political beliefs, and lived experiences, or implicitly via names prevalent in specific populations. The LLM personas are then evaluated via (1) subjective annotation task (e.g., detecting toxicity) and (2) a belief generation task, where both tasks are known to vary across human factors. We examine the impact of explicit vs. implicit personas and investigate which human factors LLMs recognize and respond to. Results show that explicit LLM personas show mixed results when reproducing known human biases, but generally fail to demonstrate implicit biases. We conclude that LLMs may capture the statistical patterns of how people speak, but are generally unable to model the complex interactions and subtleties of human perceptions, potentially limiting their effectiveness in social science applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13722v1">Persistent Pre-Training Poisoning of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-17
    </div>
    <details class="paper-abstract">
      Large language models are pre-trained on uncurated text datasets consisting of trillions of tokens scraped from the Web. Prior work has shown that: (1) web-scraped pre-training datasets can be practically poisoned by malicious actors; and (2) adversaries can compromise language models after poisoning fine-tuning datasets. Our work evaluates for the first time whether language models can also be compromised during pre-training, with a focus on the persistence of pre-training attacks after models are fine-tuned as helpful and harmless chatbots (i.e., after SFT and DPO). We pre-train a series of LLMs from scratch to measure the impact of a potential poisoning adversary under four different attack objectives (denial-of-service, belief manipulation, jailbreaking, and prompt stealing), and across a wide range of model sizes (from 600M to 7B). Our main result is that poisoning only 0.1% of a model's pre-training dataset is sufficient for three out of four attacks to measurably persist through post-training. Moreover, simple attacks like denial-of-service persist through post-training with a poisoning rate of only 0.001%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.20052v2">Understanding and Mitigating Language Confusion in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-17
      | 💬 EMNLP 2024 Main Conference Camera-ready
    </div>
    <details class="paper-abstract">
      We investigate a surprising limitation of LLMs: their inability to consistently generate text in a user's desired language. We create the Language Confusion Benchmark (LCB) to evaluate such failures, covering 15 typologically diverse languages with existing and newly-created English and multilingual prompts. We evaluate a range of LLMs on monolingual and cross-lingual generation reflecting practical use cases, finding that Llama Instruct and Mistral models exhibit high degrees of language confusion and even the strongest models fail to consistently respond in the correct language. We observe that base and English-centric instruct models are more prone to language confusion, which is aggravated by complex prompts and high sampling temperatures. We find that language confusion can be partially mitigated via few-shot prompting, multilingual SFT and preference tuning. We release our language confusion benchmark, which serves as a first layer of efficient, scalable multilingual evaluation at https://github.com/for-ai/language-confusion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13648v1">SimpleToM: Exposing the Gap between Explicit ToM Inference and Implicit ToM Application in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-17
    </div>
    <details class="paper-abstract">
      While prior work has explored whether large language models (LLMs) possess a "theory of mind" (ToM) - the ability to attribute mental states to oneself and others - there has been little work testing whether LLMs can implicitly apply such knowledge to predict behavior, or to judge whether an observed behavior is rational. Such skills are critical for appropriate interaction in social environments. We create a new dataset, SimpleTom, containing concise, diverse stories (e.g., "The can of Pringles has moldy chips in it. Mary picks up the can in the supermarket and walks to the cashier."), each with three questions that test different degrees of ToM reasoning, asking models to predict (a) mental state ("Is Mary aware of the mold?"), (b) behavior ("Will Mary pay for the chips or report the mold?"), and (c) judgment ("Mary paid for the chips. Was that reasonable?"). To our knowledge, SimpleToM is the first dataset to systematically explore downstream reasoning requiring knowledge of mental states in realistic scenarios. Our experimental results are intriguing: While most models can reliably predict mental state on our dataset (a), they often fail to correctly predict the behavior (b), and fare even worse at judging whether given behaviors are reasonable (c), despite being correctly aware of the protagonist's mental state should make such secondary predictions obvious. We further show that we can help models do better at (b) and (c) via interventions such as reminding the model of its earlier mental state answer and mental-state-specific chain-of-thought prompting, raising the action prediction accuracies (e.g., from 49.5% to 93.5% for GPT-4o) and judgment accuracies (e.g., from 15.3% to 94.7% in GPT-4o). While this shows that models can be coaxed to perform well, it requires task-specific interventions, and the natural model performances remain low, a cautionary tale for LLM deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13640v1">Latent Space Chain-of-Embedding Enables Output-free LLM Self-Evaluation</a></div>
    <div class="paper-meta">
      📅 2024-10-17
      | 💬 33 pages, 18 figures, 12 tables
    </div>
    <details class="paper-abstract">
      LLM self-evaluation relies on the LLM's own ability to estimate response correctness, which can greatly improve its deployment reliability. In this research track, we propose the Chain-of-Embedding (CoE) in the latent space to enable LLMs to perform output-free self-evaluation. CoE consists of all progressive hidden states produced during the inference time, which can be treated as the latent thinking path of LLMs. We find that when LLMs respond correctly and incorrectly, their CoE features differ, these discrepancies assist us in estimating LLM response correctness. Experiments in four diverse domains and seven LLMs fully demonstrate the effectiveness of our method. Meanwhile, its label-free design intent without any training and millisecond-level computational cost ensure real-time feedback in large-scale scenarios. More importantly, we provide interesting insights into LLM response correctness from the perspective of hidden state changes inside LLMs.
    </details>
</div>
