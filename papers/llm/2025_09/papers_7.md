# llm - 2025_09

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
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)
- [Part 15](papers_15.md)
- [Part 16](papers_16.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17314v2">Clotho: Measuring Task-Specific Pre-Generation Test Adequacy for LLM Inputs</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      Software increasingly relies on the emergent capabilities of Large Language Models (LLMs), from natural language understanding to program analysis and generation. Yet testing them on specific tasks remains difficult and costly: many prompts lack ground truth, forcing reliance on human judgment, while existing uncertainty and adequacy measures typically require full inference. A key challenge is to assess input adequacy in a way that reflects the demands of the task, ideally before even generating any output. We introduce CLOTHO, a task-specific, pre-generation adequacy measure that estimates input difficulty directly from hidden LLM states. Given a large pool of unlabelled inputs for a specific task, CLOTHO uses a Gaussian Mixture Model (GMM) to adaptively sample the most informative cases for human labelling. Based on this reference set the GMM can then rank unseen inputs by their likelihood of failure. In our empirical evaluation across eight benchmark tasks and three open-weight LLMs, CLOTHO can predict failures with a ROC-AUC of 0.716, after labelling reference sets that are on average only 5.4% of inputs. It does so without generating any outputs, thereby reducing costs compared to existing uncertainty measures. Comparison of CLOTHO and post-generation uncertainty measures shows that the two approaches complement each other. Crucially, we show that adequacy scores learnt from open-weight LLMs transfer effectively to proprietary models, extending the applicability of the approach. When prioritising test inputs for proprietary models, CLOTHO increases the average number of failing inputs from 18.7 to 42.5 out of 100, compared to random prioritisation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.19489v1">Estimating the Self-Consistency of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-23
      | 💬 5 pages
    </div>
    <details class="paper-abstract">
      Systems often repeat the same prompt to large language models (LLMs) and aggregate responses to improve reliability. This short note analyzes an estimator of the self-consistency of LLMs and the tradeoffs it induces under a fixed compute budget $B=mn$, where $m$ is the number of prompts sampled from the task distribution and $n$ is the number of repeated LLM calls per prompt; the resulting analysis favors a rough split $m,n\propto\sqrt{B}$.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02362v6">Premise-Augmented Reasoning Chains Improve Error Identification in Math reasoning with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-23
      | 💬 Accepted at ICML 2025
    </div>
    <details class="paper-abstract">
      Chain-of-Thought (CoT) prompting enhances mathematical reasoning in large language models (LLMs) by enabling detailed step-by-step solutions. However, due to the verbosity of LLMs, the resulting reasoning chains can be long, making it harder to verify the reasoning steps and trace issues resulting from dependencies between the steps that may be farther away in the sequence of steps. Importantly, mathematical reasoning allows each step to be derived from a small set of premises, which are a subset of the preceding steps in the reasoning chain. In this paper, we present a framework that identifies the premises for each step, to improve the evaluation of reasoning. We restructure conventional linear reasoning chains into Premise Augmented Reasoning Chains (PARC) by introducing premise links, resulting in a directed acyclic graph where the nodes are the steps and the edges are the premise links. Through experiments with a PARC-based dataset that we built, namely PERL (Premises and ERrors identification in LLMs), we demonstrate that LLMs can reliably identify premises within complex reasoning chains. In particular, even open-source LLMs achieve 90% recall in premise identification. We also show that PARC helps to identify errors in reasoning chains more reliably. The accuracy of error identification improves by 6% to 16% absolute when step-by-step verification is carried out in PARC under the premises. Our findings highlight the utility of premise-centric representations in addressing complex problem-solving tasks and open new avenues for improving the reliability of LLM-based reasoning evaluations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18961v2">Weaver: Interweaving SQL and LLM for Table Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      Querying tables with unstructured data is challenging due to the presence of text (or image), either embedded in the table or in external paragraphs, which traditional SQL struggles to process, especially for tasks requiring semantic reasoning. While Large Language Models (LLMs) excel at understanding context, they face limitations with long input sequences. Existing approaches that combine SQL and LLMs typically rely on rigid, predefined work-flows, limiting their adaptability to complex queries. To address these issues, we introduce Weaver , a modular pipeline that dynamically integrates SQL and LLMs for table-based question answering (TableQA). Weaver generates a flexible, step-by-step plan that combines SQL for structured data retrieval with LLMs for semantic processing. By decomposing complex queries into manageable subtasks, Weaver improves accuracy and generalization. Our experiments show that Weaver consistently outperforms state-of-the-art methods across four TableQA datasets, reducing both API calls and error rates. The code, along with other associated scripts, are available at https://coral-lab-asu.github.io/weaver.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12260v4">LightRetriever: A LLM-based Text Retrieval Architecture with Extremely Faster Query Inference</a></div>
    <div class="paper-meta">
      📅 2025-09-22
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs)-based text retrieval retrieves documents relevant to search queries based on vector similarities. Documents are pre-encoded offline, while queries arrive in real-time, necessitating an efficient online query encoder. Although LLMs significantly enhance retrieval capabilities, serving deeply parameterized LLMs slows down query inference throughput and increases demands for online deployment resources. In this paper, we propose LightRetriever, a novel LLM-based retriever with extremely lightweight query encoders. Our method retains a full-sized LLM for document encoding, but reduces the workload of query encoding to no more than an embedding lookup. Compared to serving a full LLM on an A800 GPU, our method achieves over 1000x speedup in query encoding and over 10x increase in end-to-end retrieval throughput. Extensive experiments on large-scale retrieval benchmarks show that LightRetriever generalizes well across diverse tasks, maintaining an average of 95% retrieval performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.22424v2">CoSIL: Issue Localization via LLM-Driven Code Graph Searching</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 Accepted by ASE 2025
    </div>
    <details class="paper-abstract">
      Issue solving aims to generate patches to fix reported issues in real-world code repositories according to issue descriptions. Issue localization forms the basis for accurate issue solving. Recently, LLM-based issue localization methods have demonstrated state-of-the-art performance. However, these methods either search from files mentioned in issue descriptions or in the whole repository and struggle to balance the breadth and depth of the search space to converge on the target efficiently. Moreover, they allow LLM to explore whole repositories freely, making it challenging to control the search direction to prevent the LLM from searching for incorrect targets. This paper introduces CoSIL, an LLM-driven, powerful function-level issue localization method without training or indexing. CoSIL employs a two-phase code graph search strategy. It first conducts broad exploration at the file level using dynamically constructed module call graphs, and then performs in-depth analysis at the function level by expanding the module call graph into a function call graph and executing iterative searches. To precisely control the search direction, CoSIL designs a pruner to filter unrelated directions and irrelevant contexts. To avoid incorrect interaction formats in long contexts, CoSIL introduces a reflection mechanism that uses additional independent queries in short contexts to enhance formatted abilities. Experiment results demonstrate that CoSIL achieves a Top-1 localization accuracy of 43.3\% and 44.6\% on SWE-bench Lite and SWE-bench Verified, respectively, with Qwen2.5-Coder-32B, average outperforming the state-of-the-art methods by 96.04\%. When CoSIL is integrated into an issue-solving method, Agentless, the issue resolution rate improves by 2.98\%--30.5\%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17703v1">An LLM-based Agent Simulation Approach to Study Moral Evolution</a></div>
    <div class="paper-meta">
      📅 2025-09-22
    </div>
    <details class="paper-abstract">
      The evolution of morality presents a puzzle: natural selection should favor self-interest, yet humans developed moral systems promoting altruism. We address this question by introducing a novel Large Language Model (LLM)-based agent simulation framework modeling prehistoric hunter-gatherer societies. This platform is designed to probe diverse questions in social evolution, from survival advantages to inter-group dynamics. To investigate moral evolution, we designed agents with varying moral dispositions based on the Expanding Circle Theory \citep{singer1981expanding}. We evaluated their evolutionary success across a series of simulations and analyzed their decision-making in specially designed moral dilemmas. These experiments reveal how an agent's moral framework, in combination with its cognitive constraints, directly shapes its behavior and determines its evolutionary outcome. Crucially, the emergent patterns echo seminal theories from related domains of social science, providing external validation for the simulations. This work establishes LLM-based simulation as a powerful new paradigm to complement traditional research in evolutionary biology and anthropology, opening new avenues for investigating the complexities of moral and social evolution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17701v1">Investigating Bias: A Multilingual Pipeline for Generating, Solving, and Evaluating Math Problems with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 Accepted at edu4AI'25: 2nd Workshop on Education for Artificial Intelligence | co-located with ECAI, October 26th, 2025, Bologna, Italy. 7 pages, 0 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used for educational support, yet their response quality varies depending on the language of interaction. This paper presents an automated multilingual pipeline for generating, solving, and evaluating math problems aligned with the German K-10 curriculum. We generated 628 math exercises and translated them into English, German, and Arabic. Three commercial LLMs (GPT-4o-mini, Gemini 2.5 Flash, and Qwen-plus) were prompted to produce step-by-step solutions in each language. A held-out panel of LLM judges, including Claude 3.5 Haiku, evaluated solution quality using a comparative framework. Results show a consistent gap, with English solutions consistently rated highest, and Arabic often ranked lower. These findings highlight persistent linguistic bias and the need for more equitable multilingual AI systems in education.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17694v1">Evaluating LLM-Generated Versus Human-Authored Responses in Role-Play Dialogues</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 Accepted for publication at the 18th International Natural Language Generation Conference (INLG 2025)
    </div>
    <details class="paper-abstract">
      Evaluating large language models (LLMs) in long-form, knowledge-grounded role-play dialogues remains challenging. This study compares LLM-generated and human-authored responses in multi-turn professional training simulations through human evaluation ($N=38$) and automated LLM-as-a-judge assessment. Human evaluation revealed significant degradation in LLM-generated response quality across turns, particularly in naturalness, context maintenance and overall quality, while human-authored responses progressively improved. In line with this finding, participants also indicated a consistent preference for human-authored dialogue. These human judgements were validated by our automated LLM-as-a-judge evaluation, where Gemini 2.0 Flash achieved strong alignment with human evaluators on both zero-shot pairwise preference and stochastic 6-shot construct ratings, confirming the widening quality gap between LLM and human responses over time. Our work contributes a multi-turn benchmark exposing LLM degradation in knowledge-grounded role-play dialogues and provides a validated hybrid evaluation framework to guide the reliable integration of LLMs in training simulations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09996v3">From Judgment to Interference: Early Stopping LLM Harmful Outputs via Streaming Content Monitoring</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 NeurIPS 2025 Accepted Paper
    </div>
    <details class="paper-abstract">
      Though safety alignment has been applied to most large language models (LLMs), LLM service providers generally deploy a subsequent moderation as the external safety guardrail in real-world products. Existing moderators mainly practice a conventional full detection, which determines the harmfulness based on the complete LLM output, causing high service latency. Recent works pay more attention to partial detection where moderators oversee the generation midway and early stop the output if harmfulness is detected, but they directly apply moderators trained with the full detection paradigm to incomplete outputs, introducing a training-inference gap that lowers the performance. In this paper, we explore how to form a data-and-model solution that natively supports partial detection. For the data, we construct FineHarm, a dataset consisting of 29K prompt-response pairs with fine-grained annotations to provide reasonable supervision for token-level training. Then, we propose the streaming content monitor, which is trained with dual supervision of response- and token-level labels and can follow the output stream of LLM to make a timely judgment of harmfulness. Experiments show that SCM gains 0.95+ in macro F1 score that is comparable to full detection, by only seeing the first 18% of tokens in responses on average. Moreover, the SCM can serve as a pseudo-harmfulness annotator for improving safety alignment and lead to a higher harmlessness score than DPO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17628v1">MSCoRe: A Benchmark for Multi-Stage Collaborative Reasoning in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 10 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have excelled in question-answering (QA) tasks within single domains. However, their reasoning and coordination capabilities in complex, multi-stage scenarios remain underexplored. Existing benchmarks typically focus on isolated tasks or narrow domains, overlooking models' abilities for multi-stage collaboration and optimization without explicit external guidance. To bridge this gap, we propose \textbf{MSCoRe}, a novel benchmark comprising 126696 domain-specific QA instances spanning scenarios in automotive, pharmaceutical, electronics, and energy sectors. The dataset is created using a structured three-phase pipeline: dynamic sampling, iterative question-answer generation, and a multi-level quality assessment to ensure data quality. Tasks are further categorized into three difficulty levels according to stage coverage and complexity. With MSCoRe, we have conducted a comprehensive evaluation of various state-of-the-art LLM agents. The commercial models performed best across all tasks and scenarios, but a notable gap in ROUGE scores remains between simple and complex tasks. We also tested the models' robustness and found that their performance is negatively affected by noisy data. MSCoRe provides a valuable new resource for the community to evaluate and improve multi-stage reasoning in LLM agents. The code and data are available at https://github.com/D3E0-source/MSCoRE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17552v1">Can LLMs Reason Over Non-Text Modalities in a Training-Free Manner? A Case Study with In-Context Representation Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 NIPS 2025
    </div>
    <details class="paper-abstract">
      The remarkable performance of Large Language Models (LLMs) can be enhanced with test-time computation, which relies on external tools and even other deep learning models. However, existing approaches for integrating non-text modality representations into LLMs typically require additional costly supervised training, restricting on-the-fly adaptation to new domains and modalities. In this work, we explore the feasibility of integrating representations from non-text foundational models (FMs) into text-based LLMs in a training-free manner. We propose In-Context Representation Learning (ICRL) as a proof-of-concept to allow LLMs to adaptively utilize non-text modality representations with few-shot learning. Unlike traditional in-context learning, which incorporates text-label pairs, ICRL replaces text inputs with FM representations, enabling the LLM to perform multi-modal inference without fine-tuning. We evaluate ICRL on a suite of tasks in the molecular domain, investigating three core research questions: (i) how to map FM representations into LLMs in a training-free manner, (ii) what factors influence ICRL performance, and (iii) what mechanisms underlie the effectiveness of ICRL. To the best of our knowledge, ICRL is the first training-free framework for integrating non-text modality representations into text-based LLMs, presenting a promising direction for adaptable, multi-modal generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14423v2">Scaling Low-Resource MT via Synthetic Data Generation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 Accepted at EMNLP 2025 Main Conference
    </div>
    <details class="paper-abstract">
      We investigate the potential of LLM-generated synthetic data for improving low-resource Machine Translation (MT). Focusing on seven diverse target languages, we construct a document-level synthetic corpus from English Europarl, and extend it via pivoting to 147 additional language pairs. Automatic and human evaluation confirm its overall high quality. We study its practical application by (i) identifying effective training regimes, (ii) comparing our data with the HPLT dataset, (iii) studying the effect of varying training data size, and (iiii) testing its utility beyond English-centric MT. Finally, we introduce SynOPUS, a public repository for synthetic parallel datasets. Our findings show that LLM-generated synthetic data, even when noisy, can substantially improve MT performance for low-resource languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17505v1">CorefInst: Leveraging LLMs for Multilingual Coreference Resolution</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 Accepted for publication in Transactions of the Association for Computational Linguistics (TACL) (2025 August). Submission: March, 2025. Revision: July, 2025. Acceptance: August, 2025
    </div>
    <details class="paper-abstract">
      Coreference Resolution (CR) is a crucial yet challenging task in natural language understanding, often constrained by task-specific architectures and encoder-based language models that demand extensive training and lack adaptability. This study introduces the first multilingual CR methodology which leverages decoder-only LLMs to handle both overt and zero mentions. The article explores how to model the CR task for LLMs via five different instruction sets using a controlled inference method. The approach is evaluated across three LLMs; Llama 3.1, Gemma 2, and Mistral 0.3. The results indicate that LLMs, when instruction-tuned with a suitable instruction set, can surpass state-of-the-art task-specific architectures. Specifically, our best model, a fully fine-tuned Llama 3.1 for multilingual CR, outperforms the leading multilingual CR model (i.e., Corpipe 24 single stage variant) by 2 pp on average across all languages in the CorefUD v1.2 dataset collection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13400v2">Justice in Judgment: Unveiling (Hidden) Bias in LLM-assisted Peer Reviews</a></div>
    <div class="paper-meta">
      📅 2025-09-22
    </div>
    <details class="paper-abstract">
      The adoption of large language models (LLMs) is transforming the peer review process, from assisting reviewers in writing more detailed evaluations to generating entire reviews automatically. While these capabilities offer exciting opportunities, they also raise critical concerns about fairness and reliability. In this paper, we investigate bias in LLM-generated peer reviews by conducting controlled experiments on sensitive metadata, including author affiliation and gender. Our analysis consistently shows affiliation bias favoring institutions highly ranked on common academic rankings. Additionally, we find some gender preferences, which, even though subtle in magnitude, have the potential to compound over time. Notably, we uncover implicit biases that become more evident with token-based soft ratings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14615v2">SATBench: Benchmarking LLMs' Logical Reasoning via Automated Puzzle Generation from SAT Formulas</a></div>
    <div class="paper-meta">
      📅 2025-09-22
    </div>
    <details class="paper-abstract">
      We introduce SATBench, a benchmark for evaluating the logical reasoning capabilities of large language models (LLMs) through logical puzzles derived from Boolean satisfiability (SAT) problems. Unlike prior work that focuses on inference rule-based reasoning, which often involves deducing conclusions from a set of premises, our approach leverages the search-based nature of SAT problems, where the objective is to find a solution that fulfills a specified set of logical constraints. Each instance in SATBench is generated from a SAT formula, then translated into a puzzle using LLMs. The generation process is fully automated and allows for adjustable difficulty by varying the number of clauses. All 2100 puzzles are validated through both LLM-based and solver-based consistency checks, with human validation on a subset. Experimental results show that even the strongest model, o4-mini, achieves only 65.0% accuracy on hard UNSAT problems, close to the random baseline of 50%. Our error analysis reveals systematic failures such as satisfiability bias, context inconsistency, and condition omission, highlighting limitations of current LLMs in search-based logical reasoning. Our code and data are publicly available at https://github.com/Anjiang-Wei/SATBench
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17489v1">MapCoder-Lite: Squeezing Multi-Agent Coding into a Single Small LLM</a></div>
    <div class="paper-meta">
      📅 2025-09-22
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have advanced code generation from single-function tasks to competitive-programming problems, but existing multi-agent solutions either rely on costly large-scale ($>$ 30B) models or collapse when downsized to small open-source models. We present MapCoder-Lite, which upgrades a single 7B model into four role-specialised agents-retriever, planner, coder, and debugger-using only rank-32, role-specific LoRA adapters ($<3\%$ extra parameters). Three lightweight techniques make this possible: (i) trajectory distillation from strong LLMs fixes format fragility in retrieval and debugging, (ii) supervisor-guided correction strengthens planning and coding agents, and (iii) agent-wise LoRA fine-tuning delivers memory-efficient specialisation. Comprehensive evaluation on xCodeEval, APPS, and CodeContests shows that MapCoder-Lite more than doubles xCodeEval accuracy (from $13.2\%$ to $28.3\%$), eliminates all format failures, and closes to within six points of a 32B baseline while cutting GPU memory and token-generation time by $4\times$. These results demonstrate that careful agent-wise fine-tuning unleashes high-quality multi-agent coding on a small language model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17488v1">Privacy in Action: Towards Realistic Privacy Mitigation and Evaluation for LLM-Powered Agents</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 To appear at EMNLP 2025 (Findings)
    </div>
    <details class="paper-abstract">
      The increasing autonomy of LLM agents in handling sensitive communications, accelerated by Model Context Protocol (MCP) and Agent-to-Agent (A2A) frameworks, creates urgent privacy challenges. While recent work reveals significant gaps between LLMs' privacy Q&A performance and their agent behavior, existing benchmarks remain limited to static, simplified scenarios. We present PrivacyChecker, a model-agnostic, contextual integrity based mitigation approach that effectively reduces privacy leakage from 36.08% to 7.30% on DeepSeek-R1 and from 33.06% to 8.32% on GPT-4o, all while preserving task helpfulness. We also introduce PrivacyLens-Live, transforming static benchmarks into dynamic MCP and A2A environments that reveal substantially higher privacy risks in practical. Our modular mitigation approach integrates seamlessly into agent protocols through three deployment strategies, providing practical privacy protection for the emerging agentic ecosystem. Our data and code will be made available at https://aka.ms/privacy_in_action.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17436v1">MedFact: A Large-scale Chinese Dataset for Evidence-based Medical Fact-checking of LLM Responses</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 Accepted at EMNLP 2025. Camera-ready version
    </div>
    <details class="paper-abstract">
      Medical fact-checking has become increasingly critical as more individuals seek medical information online. However, existing datasets predominantly focus on human-generated content, leaving the verification of content generated by large language models (LLMs) relatively unexplored. To address this gap, we introduce MedFact, the first evidence-based Chinese medical fact-checking dataset of LLM-generated medical content. It consists of 1,321 questions and 7,409 claims, mirroring the complexities of real-world medical scenarios. We conduct comprehensive experiments in both in-context learning (ICL) and fine-tuning settings, showcasing the capability and challenges of current LLMs on this task, accompanied by an in-depth error analysis to point out key directions for future research. Our dataset is publicly available at https://github.com/AshleyChenNLP/MedFact.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17399v1">DIWALI - Diversity and Inclusivity aWare cuLture specific Items for India: Dataset and Assessment of LLMs for Cultural Text Adaptation in Indian Context</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 Accepted at EMNLP 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are widely used in various tasks and applications. However, despite their wide capabilities, they are shown to lack cultural alignment \citep{ryan-etal-2024-unintended, alkhamissi-etal-2024-investigating} and produce biased generations \cite{naous-etal-2024-beer} due to a lack of cultural knowledge and competence. Evaluation of LLMs for cultural awareness and alignment is particularly challenging due to the lack of proper evaluation metrics and unavailability of culturally grounded datasets representing the vast complexity of cultures at the regional and sub-regional levels. Existing datasets for culture specific items (CSIs) focus primarily on concepts at the regional level and may contain false positives. To address this issue, we introduce a novel CSI dataset for Indian culture, belonging to 17 cultural facets. The dataset comprises $\sim$8k cultural concepts from 36 sub-regions. To measure the cultural competence of LLMs on a cultural text adaptation task, we evaluate the adaptations using the CSIs created, LLM as Judge, and human evaluations from diverse socio-demographic region. Furthermore, we perform quantitative analysis demonstrating selective sub-regional coverage and surface-level adaptations across all considered LLMs. Our dataset is available here: \href{https://huggingface.co/datasets/nlip/DIWALI}{https://huggingface.co/datasets/nlip/DIWALI}, project webpage\footnote{\href{https://nlip-lab.github.io/nlip/publications/diwali/}{https://nlip-lab.github.io/nlip/publications/diwali/}}, and our codebase with model outputs can be found here: \href{https://github.com/pramitsahoo/culture-evaluation}{https://github.com/pramitsahoo/culture-evaluation}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17380v1">Correlation or Causation: Analyzing the Causal Structures of LLM and LRM Reasoning Process</a></div>
    <div class="paper-meta">
      📅 2025-09-22
    </div>
    <details class="paper-abstract">
      LLMs suffer from critical reasoning issues such as unfaithfulness, bias, and inconsistency, since they lack robust causal underpinnings and may rely on superficial correlations rather than genuine understanding. Successive LRMs have emerged as a promising alternative, leveraging advanced training techniques such as reinforcement learning (RL) and distillation to improve task accuracy. However, the impact of these training methods on causality remains largely unexplored. In this study, we conduct a systematic causal analysis on LLMs and LRMs, examining structural causal models (SCMs) of four key variables: problem instruction (Z), thinking process (T), reasoning steps (X), and answer (Y). Our findings reveal that RLVR-trained LRMs exhibit enhanced causal reasoning capabilities, aligning more closely with ideal causal structures, while LLMs and distilled LRMs fail to address causality-related deficiencies. Our further investigation indicates that RLVR reduces spurious correlations and strengthens genuine causal patterns, thereby mitigating unfaithfulness and bias. In addition, our inspection on the dynamics of the RLVR training process observes a high correlation between reduced spurious features and improved causal structures, where the causal relationships consistently improve in the training process. This study contributes to the understanding of causality in reasoning models, highlights the critical role of RLVR in enhancing causal reasoning, and provides insights for designing future AI systems with stronger causal foundations. We release our code and data at https://github.com/Harryking1999/CoT_Causal_Analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17360v1">Asteria: Semantic-Aware Cross-Region Caching for Agentic LLM Tool Access</a></div>
    <div class="paper-meta">
      📅 2025-09-22
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents tackle data-intensive tasks such as deep research and code generation. However, their effectiveness depends on frequent interactions with knowledge sources across remote clouds or regions. Such interactions can create non-trivial latency and cost bottlenecks. Existing caching solutions focus on exact-match queries, limiting their effectiveness for semantic knowledge reuse. To address this challenge, we introduce Asteria, a novel cross-region knowledge caching architecture for LLM agents. At its core are two abstractions: Semantic Element (SE) and Semantic Retrieval Index (Sine). A semantic element captures the semantic embedding representation of an LLM query together with performance-aware metadata such as latency, cost, and staticity. Sine then provides two-stage retrieval: a vector similar index with semantic embedding for fast candidate selection and a lightweight LLM-powered semantic judger for precise validation. Atop these primitives, Asteria builds a new cache interface that includes a new semantic-aware cache hit definition, a cost-efficient eviction policy, and proactive prefetching. To reduce overhead, Asteria co-locates the small LLM judger with the main LLM using adaptive scheduling and resource sharing. Our evaluation demonstrates that Asteria delivers substantial performance improvements without compromising correctness. On representative search workloads, Asteria achieves up to a 3.6$\times$ increase in throughput by maintaining cache hit rates of over 85%, while preserving accuracy virtually identical to non-cached baselines. Asteria also improves throughput for complex coding tasks by 20%, showcasing its versatility across diverse agentic workloads.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17357v1">Cronus: Efficient LLM inference on Heterogeneous GPU Clusters via Partially Disaggregated Prefill</a></div>
    <div class="paper-meta">
      📅 2025-09-22
    </div>
    <details class="paper-abstract">
      Efficient LLM inference is critical for real-world applications, especially within heterogeneous GPU clusters commonly found in organizations and on-premise datacenters as GPU architecture rapidly evolves. Current disaggregated prefill strategies, which separate the prefill and decode stages of LLM inference across different GPUs, often suffer from suboptimal performance due to imbalances between GPU capabilities and workload demands. On the other hand, extending conventional data parallelism and pipeline parallelism to heterogeneous setups incurs high inference latencies. To address these challenges, we introduce Cronus, a novel LLM inference system designed to dynamically balance workloads across heterogeneous GPUs using partially disaggregated prefill. Cronus partitions each prefill stage and executes its initial portion on the low-end GPU, while overlapping the remaining prefill and decode stages of earlier requests on the high-end GPU. Extensive evaluations across various high-end and low-end GPU combinations demonstrate that Cronus significantly improves the throughput over disaggregated prefill. It also reduces TTFT P99 and TBT P99 significantly over DP and PP while maintaining similar or better throughput.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17337v1">LLaVul: A Multimodal LLM for Interpretable Vulnerability Reasoning about Source Code</a></div>
    <div class="paper-meta">
      📅 2025-09-22
    </div>
    <details class="paper-abstract">
      Increasing complexity in software systems places a growing demand on reasoning tools that unlock vulnerabilities manifest in source code. Many current approaches focus on vulnerability analysis as a classifying task, oversimplifying the nuanced and context-dependent real-world scenarios. Even though current code large language models (LLMs) excel in code understanding, they often pay little attention to security-specific reasoning. We propose LLaVul, a multimodal LLM tailored to provide fine-grained reasoning about code through question-answering (QA). Our model is trained to integrate paired code and natural queries into a unified space, enhancing reasoning and context-dependent insights about code vulnerability. To evaluate our model performance, we construct a curated dataset of real-world vulnerabilities paired with security-focused questions and answers. Our model outperforms state-of-the-art general-purpose and code LLMs in the QA and detection tasks. We further explain decision-making by conducting qualitative analysis to highlight capabilities and limitations. By integrating code and QA, LLaVul enables more interpretable and security-focused code understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17335v1">BASFuzz: Towards Robustness Evaluation of LLM-based NLP Software via Automated Fuzz Testing</a></div>
    <div class="paper-meta">
      📅 2025-09-22
    </div>
    <details class="paper-abstract">
      Fuzzing has shown great success in evaluating the robustness of intelligent natural language processing (NLP) software. As large language model (LLM)-based NLP software is widely deployed in critical industries, existing methods still face two main challenges: 1 testing methods are insufficiently coupled with the behavioral patterns of LLM-based NLP software; 2 fuzzing capability for the testing scenario of natural language generation (NLG) generally degrades. To address these issues, we propose BASFuzz, an efficient Fuzz testing method tailored for LLM-based NLP software. BASFuzz targets complete test inputs composed of prompts and examples, and uses a text consistency metric to guide mutations of the fuzzing loop, aligning with the behavioral patterns of LLM-based NLP software. A Beam-Annealing Search algorithm, which integrates beam search and simulated annealing, is employed to design an efficient fuzzing loop. In addition, information entropy-based adaptive adjustment and an elitism strategy further enhance fuzzing capability. We evaluate BASFuzz on six datasets in representative scenarios of NLG and natural language understanding (NLU). Experimental results demonstrate that BASFuzz achieves a testing effectiveness of 90.335% while reducing the average time overhead by 2,163.852 seconds compared to the current best baseline, enabling more effective robustness evaluation prior to software deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17314v1">Clotho: Measuring Task-Specific Pre-Generation Test Adequacy for LLM Inputs</a></div>
    <div class="paper-meta">
      📅 2025-09-22
    </div>
    <details class="paper-abstract">
      Software increasingly relies on the emergent capabilities of Large Language Models (LLMs), from natural language understanding to program analysis and generation. Yet testing them on specific tasks remains difficult and costly: many prompts lack ground truth, forcing reliance on human judgment, while existing uncertainty and adequacy measures typically require full inference. A key challenge is to assess input adequacy in a way that reflects the demands of the task, ideally before even generating any output. We introduce CLOTHO, a task-specific, pre-generation adequacy measure that estimates input difficulty directly from hidden LLM states. Given a large pool of unlabelled inputs for a specific task, CLOTHO uses a Gaussian Mixture Model (GMM) to adaptively sample the most informative cases for human labelling. Based on this reference set the GMM can then rank unseen inputs by their likelihood of failure. In our empirical evaluation across eight benchmark tasks and three open-weight LLMs, CLOTHO can predict failures with a ROC-AUC of 0.716, after labelling reference sets that are on average only 5.4% of inputs. It does so without generating any outputs, thereby reducing costs compared to existing uncertainty measures. Comparison of CLOTHO and post-generation uncertainty measures shows that the two approaches complement each other. Crucially, we show that adequacy scores learnt from open-weight LLMs transfer effectively to proprietary models, extending the applicability of the approach. When prioritising test inputs for proprietary models, CLOTHO increases the average number of failing inputs from 18.7 to 42.5 out of 100, compared to random prioritisation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14315v4">Mitigating Forgetting in LLM Fine-Tuning via Low-Perplexity Token Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-22
    </div>
    <details class="paper-abstract">
      Maintaining consistent model performance across domains is a fundamental challenge in machine learning. While recent work has explored using LLM-generated data for fine-tuning, its impact on cross-domain generalization remains poorly understood. This paper presents a systematic analysis revealing that fine-tuning with LLM-generated data not only improves target task performance but also reduces non-target task degradation compared to fine-tuning with ground truth data. Through analyzing the data sequence in tasks of various domains, we demonstrate that this enhancement of non-target task robustness stems from the reduction of high perplexity tokens found in LLM-generated sequences. Following our findings, we showed that masking high perplexity tokens in ground truth training data achieves similar non-target task performance preservation, comparable to using LLM-generated data. Extensive experiments across different model families and scales, including Gemma 2 IT 2B, Llama 3 8B Instruct, and 3 additional models, agree with our findings. To the best of our knowledge, this is the first work to provide an empirical explanation based on token perplexity reduction to mitigate catastrophic forgetting in LLMs after fine-tuning, offering valuable insights for developing more robust fine-tuning strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17292v1">Multi-View Attention Multiple-Instance Learning Enhanced by LLM Reasoning for Cognitive Distortion Detection</a></div>
    <div class="paper-meta">
      📅 2025-09-22
    </div>
    <details class="paper-abstract">
      Cognitive distortions have been closely linked to mental health disorders, yet their automatic detection remained challenging due to contextual ambiguity, co-occurrence, and semantic overlap. We proposed a novel framework that combines Large Language Models (LLMs) with Multiple-Instance Learning (MIL) architecture to enhance interpretability and expression-level reasoning. Each utterance was decomposed into Emotion, Logic, and Behavior (ELB) components, which were processed by LLMs to infer multiple distortion instances, each with a predicted type, expression, and model-assigned salience score. These instances were integrated via a Multi-View Gated Attention mechanism for final classification. Experiments on Korean (KoACD) and English (Therapist QA) datasets demonstrate that incorporating ELB and LLM-inferred salience scores improves classification performance, especially for distortions with high interpretive ambiguity. Our results suggested a psychologically grounded and generalizable approach for fine-grained reasoning in mental health NLP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18383v3">NileChat: Towards Linguistically Diverse and Culturally Aware LLMs for Local Communities</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 Accepted to EMNLP 2025 (Main Conference). Camera-ready version. Data & models: https://github.com/UBC-NLP/nilechat
    </div>
    <details class="paper-abstract">
      Enhancing the linguistic capabilities of Large Language Models (LLMs) to include low-resource languages is a critical research area. Current research directions predominantly rely on synthetic data generated by translating English corpora, which, while demonstrating promising linguistic understanding and translation abilities, often results in models aligned with source language culture. These models frequently fail to represent the cultural heritage and values of local communities. This work proposes a methodology to create both synthetic and retrieval-based pre-training data tailored to a specific community, considering its (i) language, (ii) cultural heritage, and (iii) cultural values. We demonstrate our methodology using Egyptian and Moroccan dialects as testbeds, chosen for their linguistic and cultural richness and current underrepresentation in LLMs. As a proof-of-concept, we develop NileChat, a 3B parameter Egyptian and Moroccan Arabic LLM adapted for Egyptian and Moroccan communities, incorporating their language, cultural heritage, and values. Our results on various understanding, translation, and cultural and values alignment benchmarks show that NileChat outperforms existing Arabic-aware LLMs of similar size and performs on par with larger models. This work addresses Arabic dialect in LLMs with a focus on cultural and values alignment via controlled synthetic data generation and retrieval-augmented pre-training for Moroccan Darija and Egyptian Arabic, including Arabizi variants, advancing Arabic NLP for low-resource communities. We share our methods, data, and models with the community to promote the inclusion and coverage of more diverse communities in cultural LLM development: https://github.com/UBC-NLP/nilechat .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04130v4">STORM: Token-Efficient Long Video Understanding for Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-22
    </div>
    <details class="paper-abstract">
      Recent advances in video-based multimodal large language models (Video-LLMs) have significantly improved video understanding by processing videos as sequences of image frames. However, many existing methods treat frames independently in the vision backbone, lacking explicit temporal modeling, which limits their ability to capture dynamic patterns and efficiently handle long videos. To address these limitations, we introduce STORM (Spatiotemporal TOken Reduction for Multimodal LLMs), a novel architecture incorporating a dedicated temporal encoder between the image encoder and the LLM. Our temporal encoder leverages the Mamba State Space Model to integrate temporal information into image tokens, generating enriched representations that preserve inter-frame dynamics across the entire video sequence. This enriched encoding not only enhances video reasoning capabilities but also enables effective token reduction strategies, including test-time sampling and training-based temporal and spatial pooling, substantially reducing computational demands on the LLM without sacrificing key temporal information. By integrating these techniques, our approach simultaneously reduces training and inference latency while improving performance, enabling efficient and robust video understanding over extended temporal contexts. Extensive evaluations show that STORM achieves state-of-the-art results across various long video understanding benchmarks (more than 5% improvement on MLVU and LongVideoBench) while reducing the computation costs by up to $8\times$ and the decoding latency by 2.4-2.9$\times$ for the fixed numbers of input frames. Project page is available at https://research.nvidia.com/labs/lpr/storm
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18401v1">Evaluating the Creativity of LLMs in Persian Literary Text Generation</a></div>
    <div class="paper-meta">
      📅 2025-09-22
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated notable creative abilities in generating literary texts, including poetry and short stories. However, prior research has primarily centered on English, with limited exploration of non-English literary traditions and without standardized methods for assessing creativity. In this paper, we evaluate the capacity of LLMs to generate Persian literary text enriched with culturally relevant expressions. We build a dataset of user-generated Persian literary spanning 20 diverse topics and assess model outputs along four creativity dimensions-originality, fluency, flexibility, and elaboration-by adapting the Torrance Tests of Creative Thinking. To reduce evaluation costs, we adopt an LLM as a judge for automated scoring and validate its reliability against human judgments using intraclass correlation coefficients, observing strong agreement. In addition, we analyze the models' ability to understand and employ four core literary devices: simile, metaphor, hyperbole, and antithesis. Our results highlight both the strengths and limitations of LLMs in Persian literary text generation, underscoring the need for further refinement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18400v1">ATLAS: Benchmarking and Adapting LLMs for Global Trade via Harmonized Tariff Code Classification</a></div>
    <div class="paper-meta">
      📅 2025-09-22
    </div>
    <details class="paper-abstract">
      Accurate classification of products under the Harmonized Tariff Schedule (HTS) is a critical bottleneck in global trade, yet it has received little attention from the machine learning community. Misclassification can halt shipments entirely, with major postal operators suspending deliveries to the U.S. due to incomplete customs documentation. We introduce the first benchmark for HTS code classification, derived from the U.S. Customs Rulings Online Search System (CROSS). Evaluating leading LLMs, we find that our fine-tuned Atlas model (LLaMA-3.3-70B) achieves 40 percent fully correct 10-digit classifications and 57.5 percent correct 6-digit classifications, improvements of 15 points over GPT-5-Thinking and 27.5 points over Gemini-2.5-Pro-Thinking. Beyond accuracy, Atlas is roughly five times cheaper than GPT-5-Thinking and eight times cheaper than Gemini-2.5-Pro-Thinking, and can be self-hosted to guarantee data privacy in high-stakes trade and compliance workflows. While Atlas sets a strong baseline, the benchmark remains highly challenging, with only 40 percent 10-digit accuracy. By releasing both dataset and model, we aim to position HTS classification as a new community benchmark task and invite future work in retrieval, reasoning, and alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18384v1">AD-VF: LLM-Automatic Differentiation Enables Fine-Tuning-Free Robot Planning from Formal Methods Feedback</a></div>
    <div class="paper-meta">
      📅 2025-09-22
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can translate natural language instructions into executable action plans for robotics, autonomous driving, and other domains. Yet, deploying LLM-driven planning in the physical world demands strict adherence to safety and regulatory constraints, which current models often violate due to hallucination or weak alignment. Traditional data-driven alignment methods, such as Direct Preference Optimization (DPO), require costly human labeling, while recent formal-feedback approaches still depend on resource-intensive fine-tuning. In this paper, we propose LAD-VF, a fine-tuning-free framework that leverages formal verification feedback for automated prompt engineering. By introducing a formal-verification-informed text loss integrated with LLM-AutoDiff, LAD-VF iteratively refines prompts rather than model parameters. This yields three key benefits: (i) scalable adaptation without fine-tuning; (ii) compatibility with modular LLM architectures; and (iii) interpretable refinement via auditable prompts. Experiments in robot navigation and manipulation tasks demonstrate that LAD-VF substantially enhances specification compliance, improving success rates from 60% to over 90%. Our method thus presents a scalable and interpretable pathway toward trustworthy, formally-verified LLM-driven control systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18344v1">Speculate Deep and Accurate: Lossless and Training-Free Acceleration for Offloaded LLMs via Substitute Speculative Decoding</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      The immense model sizes of large language models (LLMs) challenge deployment on memory-limited consumer GPUs. Although model compression and parameter offloading are common strategies to address memory limitations, compression can degrade quality, and offloading maintains quality but suffers from slow inference. Speculative decoding presents a promising avenue to accelerate parameter offloading, utilizing a fast draft model to propose multiple draft tokens, which are then verified by the target LLM in parallel with a single forward pass. This method reduces the time-consuming data transfers in forward passes that involve offloaded weight transfers. Existing methods often rely on pretrained weights of the same family, but require additional training to align with custom-trained models. Moreover, approaches that involve draft model training usually yield only modest speedups. This limitation arises from insufficient alignment with the target model, preventing higher token acceptance lengths. To address these challenges and achieve greater speedups, we propose SubSpec, a plug-and-play method to accelerate parameter offloading that is lossless and training-free. SubSpec constructs a highly aligned draft model by generating low-bit quantized substitute layers from offloaded target LLM portions. Additionally, our method shares the remaining GPU-resident layers and the KV-Cache, further reducing memory overhead and enhance alignment. SubSpec achieves a high average acceptance length, delivering 9.1x speedup for Qwen2.5 7B on MT-Bench (8GB VRAM limit) and an average of 12.5x speedup for Qwen2.5 32B on popular generation benchmarks (24GB VRAM limit).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18314v1">Exploiting Tree Structure for Credit Assignment in RL Training of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      Reinforcement learning improves LLM reasoning, yet sparse delayed reward over long sequences makes token-level credit assignment the key bottleneck. We study the verifiable-reward setting, where the final answer is checkable and multiple responses can be drawn per prompt. Reasoning tasks in math and medical QA align with this setup, where only a few decision tokens significantly impact the outcome. PPO offers token-level advantages with a learned value model, but it is complex to train both the actor and critic models simultaneously, and it is not easily generalizable, as the token-level values from the critic model can make training prone to overfitting. GRPO is critic-free and supports verifiable rewards, but spreads a single sequence-level return across tokens and ignores branching. We introduce \textbf{Prefix-to-Tree (P2T)}, a simple procedure that converts a group of responses into a prefix tree and computes \emph{nonparametric} prefix values \(V(s)\) by aggregating descendant outcomes. Built on P2T, we propose \textbf{TEMPO} (\emph{\textbf{T}ree-\textbf{E}stimated \textbf{M}ean Prefix Value for \textbf{P}olicy \textbf{O}ptimization}), a critic-free algorithm that augments the group-relative outcome signal of GRPO with \emph{branch-gated} temporal-difference corrections derived from the tree. At non-branch tokens, the temporal-difference (TD) term is zero, so TEMPO reduces to GRPO; at branching tokens, it supplies precise token-level credit without a learned value network or extra judges/teachers. On Qwen3-1.7B/4B, TEMPO outperforms PPO and GRPO on in-distribution (MATH, MedQA) and out-of-distribution (GSM-HARD, AMC23, MedMCQA, MMLU-Medical) benchmarks, and reaches higher validation accuracy with roughly the same wall-clock time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12543v2">Disambiguation in Conversational Question Answering in the Era of LLMs and Agents: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 14 pages, 2 figures, Accepted at EMNLP 2025 Main Conference
    </div>
    <details class="paper-abstract">
      Ambiguity remains a fundamental challenge in Natural Language Processing (NLP) due to the inherent complexity and flexibility of human language. With the advent of Large Language Models (LLMs), addressing ambiguity has become even more critical due to their expanded capabilities and applications. In the context of Conversational Question Answering (CQA), this paper explores the definition, forms, and implications of ambiguity for language driven systems, particularly in the context of LLMs. We define key terms and concepts, categorize various disambiguation approaches enabled by LLMs, and provide a comparative analysis of their advantages and disadvantages. We also explore publicly available datasets for benchmarking ambiguity detection and resolution techniques and highlight their relevance for ongoing research. Finally, we identify open problems and future research directions, especially in agentic settings, proposing areas for further investigation. By offering a comprehensive review of current research on ambiguities and disambiguation with LLMs, we aim to contribute to the development of more robust and reliable LLM-based systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18085v1">Spiffy: Multiplying Diffusion LLM Acceleration via Lossless Speculative Decoding</a></div>
    <div class="paper-meta">
      📅 2025-09-22
    </div>
    <details class="paper-abstract">
      Diffusion LLMs (dLLMs) have recently emerged as a powerful alternative to autoregressive LLMs (AR-LLMs) with the potential to operate at significantly higher token generation rates. However, currently available open-source dLLMs often generate at much lower rates, typically decoding only a single token at every denoising timestep in order to maximize output quality. We present Spiffy, a speculative decoding algorithm that accelerates dLLM inference by $\mathbf{2.8{-}3.1\times}$ while provably preserving the model's output distribution. This work addresses the unique challenges involved in applying ideas from speculative decoding of AR-LLMs to the dLLM setting. Spiffy proposes draft states by leveraging the dLLM's distribution itself in an auto-speculative manner. This approach is efficient and effective, and eliminates the overheads of training and running an independent draft model. To structure the candidate draft states, we propose a novel directed draft graph which is uniquely designed to take advantage of the bidirectional, block-wise nature of dLLM generation and can be verified in parallel by the dLLM. To further optimize the structure of these draft graphs, we introduce an efficient, offline calibration algorithm that procedurally determines high-quality graph configurations. These optimized draft graphs, enabling increased acceptance rates, lead to a significant boost in the overall speedup achieved by the system. Crucially, Spiffy is also complementary to other recent innovations in improving dLLM generation speeds such as KV-caching and multi-token unmasking. We demonstrate that when combined with such parallel decoding algorithms, Spiffy is able to effectively multiply the benefits of these methods leading to total speedups of up to $\mathbf{7.9\times}$.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18083v1">Reasoning Core: A Scalable RL Environment for LLM Symbolic Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-22
    </div>
    <details class="paper-abstract">
      We introduce Reasoning Core, a new scalable environment for Reinforcement Learning with Verifiable Rewards (RLVR), designed to advance foundational symbolic reasoning in Large Language Models (LLMs). Unlike existing benchmarks that focus on games or isolated puzzles, Reasoning Core procedurally generates problems across core formal domains, including PDDL planning, first-order logic, context-free grammar parsing, causal reasoning, and system equation solving. The environment is built on key design principles of high-generality problem distributions, verification via external tools, and continuous difficulty control, which together provide a virtually infinite supply of novel training instances. Initial zero-shot evaluations with frontier LLMs confirm the difficulty of Reasoning Core's tasks, positioning it as a promising resource to improve the reasoning capabilities of future models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18063v1">ARK-V1: An LLM-Agent for Knowledge Graph Question Answering Requiring Commonsense Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 Work in Progess
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) show strong reasoning abilities but rely on internalized knowledge that is often insufficient, outdated, or incorrect when trying to answer a question that requires specific domain knowledge. Knowledge Graphs (KGs) provide structured external knowledge, yet their complexity and multi-hop reasoning requirements make integration challenging. We present ARK-V1, a simple KG-agent that iteratively explores graphs to answer natural language queries. We evaluate several not fine-tuned state-of-the art LLMs as backbones for ARK-V1 on the CoLoTa dataset, which requires both KG-based and commonsense reasoning over long-tail entities. ARK-V1 achieves substantially higher conditional accuracies than Chain-of-Thought baselines, and larger backbone models show a clear trend toward better coverage, correctness, and stability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18058v1">Strategic Dishonesty Can Undermine AI Safety Evaluations of Frontier LLM</a></div>
    <div class="paper-meta">
      📅 2025-09-22
    </div>
    <details class="paper-abstract">
      Large language model (LLM) developers aim for their models to be honest, helpful, and harmless. However, when faced with malicious requests, models are trained to refuse, sacrificing helpfulness. We show that frontier LLMs can develop a preference for dishonesty as a new strategy, even when other options are available. Affected models respond to harmful requests with outputs that sound harmful but are subtly incorrect or otherwise harmless in practice. This behavior emerges with hard-to-predict variations even within models from the same model family. We find no apparent cause for the propensity to deceive, but we show that more capable models are better at executing this strategy. Strategic dishonesty already has a practical impact on safety evaluations, as we show that dishonest responses fool all output-based monitors used to detect jailbreaks that we test, rendering benchmark scores unreliable. Further, strategic dishonesty can act like a honeypot against malicious users, which noticeably obfuscates prior jailbreak attacks. While output monitors fail, we show that linear probes on internal activations can be used to reliably detect strategic dishonesty. We validate probes on datasets with verifiable outcomes and by using their features as steering vectors. Overall, we consider strategic dishonesty as a concrete example of a broader concern that alignment of LLMs is hard to control, especially when helpfulness and harmlessness conflict.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18052v1">The PIMMUR Principles: Ensuring Validity in Collective Behavior of LLM Societies</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used for social simulation, where populations of agents are expected to reproduce human-like collective behavior. However, we find that many recent studies adopt experimental designs that systematically undermine the validity of their claims. From a survey of over 40 papers, we identify six recurring methodological flaws: agents are often homogeneous (Profile), interactions are absent or artificially imposed (Interaction), memory is discarded (Memory), prompts tightly control outcomes (Minimal-Control), agents can infer the experimental hypothesis (Unawareness), and validation relies on simplified theoretical models rather than real-world data (Realism). For instance, GPT-4o and Qwen-3 correctly infer the underlying social experiment in 53.1% of cases when given instructions from prior work-violating the Unawareness principle. We formalize these six requirements as the PIMMUR principles and argue they are necessary conditions for credible LLM-based social simulation. To demonstrate their impact, we re-run five representative studies using a framework that enforces PIMMUR and find that the reported social phenomena frequently fail to emerge under more rigorous conditions. Our work establishes methodological standards for LLM-based multi-agent research and provides a foundation for more reliable and reproducible claims about "AI societies."
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18015v1">Beyond Diagnosis: Evaluating Multimodal LLMs for Pathology Localization in Chest Radiographs</a></div>
    <div class="paper-meta">
      📅 2025-09-22
    </div>
    <details class="paper-abstract">
      Recent work has shown promising performance of frontier large language models (LLMs) and their multimodal counterparts in medical quizzes and diagnostic tasks, highlighting their potential for broad clinical utility given their accessible, general-purpose nature. However, beyond diagnosis, a fundamental aspect of medical image interpretation is the ability to localize pathological findings. Evaluating localization not only has clinical and educational relevance but also provides insight into a model's spatial understanding of anatomy and disease. Here, we systematically assess two general-purpose MLLMs (GPT-4 and GPT-5) and a domain-specific model (MedGemma) in their ability to localize pathologies on chest radiographs, using a prompting pipeline that overlays a spatial grid and elicits coordinate-based predictions. Averaged across nine pathologies in the CheXlocalize dataset, GPT-5 exhibited a localization accuracy of 49.7%, followed by GPT-4 (39.1%) and MedGemma (17.7%), all lower than a task-specific CNN baseline (59.9%) and a radiologist benchmark (80.1%). Despite modest performance, error analysis revealed that GPT-5's predictions were largely in anatomically plausible regions, just not always precisely localized. GPT-4 performed well on pathologies with fixed anatomical locations, but struggled with spatially variable findings and exhibited anatomically implausible predictions more frequently. MedGemma demonstrated the lowest performance on all pathologies, showing limited capacity to generalize to this novel task. Our findings highlight both the promise and limitations of current MLLMs in medical imaging and underscore the importance of integrating them with task-specific tools for reliable use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03116v2">Measuring Scalar Constructs in Social Science with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 Accepted to EMNLP 2025 (Main)
    </div>
    <details class="paper-abstract">
      Many constructs that characterize language, like its complexity or emotionality, have a naturally continuous semantic structure; a public speech is not just "simple" or "complex," but exists on a continuum between extremes. Although large language models (LLMs) are an attractive tool for measuring scalar constructs, their idiosyncratic treatment of numerical outputs raises questions of how to best apply them. We address these questions with a comprehensive evaluation of LLM-based approaches to scalar construct measurement in social science. Using multiple datasets sourced from the political science literature, we evaluate four approaches: unweighted direct pointwise scoring, aggregation of pairwise comparisons, token-probability-weighted pointwise scoring, and finetuning. Our study finds that pairwise comparisons made by LLMs produce better measurements than simply prompting the LLM to directly output the scores, which suffers from bunching around arbitrary numbers. However, taking the weighted mean over the token probability of scores further improves the measurements over the two previous approaches. Finally, finetuning smaller models with as few as 1,000 training pairs can match or exceed the performance of prompted LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18761v2">How Is LLM Reasoning Distracted by Irrelevant Context? An Analysis Using a Controlled Benchmark</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 19 pages, 10 figures, 5 tables
    </div>
    <details class="paper-abstract">
      We introduce Grade School Math with Distracting Context (GSM-DC), a synthetic benchmark to evaluate Large Language Models' (LLMs) reasoning robustness against systematically controlled irrelevant context (IC). GSM-DC constructs symbolic reasoning graphs with precise distractor injections, enabling rigorous, reproducible evaluation. Our experiments demonstrate that LLMs are significantly sensitive to IC, affecting both reasoning path selection and arithmetic accuracy. Additionally, training models with strong distractors improves performance in both in-distribution and out-of-distribution scenarios. We further propose a stepwise tree search guided by a process reward model, which notably enhances robustness in out-of-distribution conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17998v1">Adaptive Kernel Design for Bayesian Optimization Is a Piece of CAKE with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 Accepted as Poster at NeurIPS 2025
    </div>
    <details class="paper-abstract">
      The efficiency of Bayesian optimization (BO) relies heavily on the choice of the Gaussian process (GP) kernel, which plays a central role in balancing exploration and exploitation under limited evaluation budgets. Traditional BO methods often rely on fixed or heuristic kernel selection strategies, which can result in slow convergence or suboptimal solutions when the chosen kernel is poorly suited to the underlying objective function. To address this limitation, we propose a freshly-baked Context-Aware Kernel Evolution (CAKE) to enhance BO with large language models (LLMs). Concretely, CAKE leverages LLMs as the crossover and mutation operators to adaptively generate and refine GP kernels based on the observed data throughout the optimization process. To maximize the power of CAKE, we further propose BIC-Acquisition Kernel Ranking (BAKER) to select the most effective kernel through balancing the model fit measured by the Bayesian information criterion (BIC) with the expected improvement at each iteration of BO. Extensive experiments demonstrate that our fresh CAKE-based BO method consistently outperforms established baselines across a range of real-world tasks, including hyperparameter optimization, controller tuning, and photonic chip design. Our code is publicly available at https://github.com/cake4bo/cake.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17946v1">HICode: Hierarchical Inductive Coding with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 Long paper accepted at EMNLP 2025 main conference, 19 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Despite numerous applications for fine-grained corpus analysis, researchers continue to rely on manual labeling, which does not scale, or statistical tools like topic modeling, which are difficult to control. We propose that LLMs have the potential to scale the nuanced analyses that researchers typically conduct manually to large text corpora. To this effect, inspired by qualitative research methods, we develop HICode, a two-part pipeline that first inductively generates labels directly from analysis data and then hierarchically clusters them to surface emergent themes. We validate this approach across three diverse datasets by measuring alignment with human-constructed themes and demonstrating its robustness through automated and human evaluations. Finally, we conduct a case study of litigation documents related to the ongoing opioid crisis in the U.S., revealing aggressive marketing strategies employed by pharmaceutical companies and demonstrating HICode's potential for facilitating nuanced analyses in large-scale data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17932v1">Training-free Truthfulness Detection via Value Vectors in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-22
    </div>
    <details class="paper-abstract">
      Large language models often generate factually incorrect outputs, motivating efforts to detect the truthfulness of their content. Most existing approaches rely on training probes over internal activations, but these methods suffer from scalability and generalization issues. A recent training-free method, NoVo, addresses this challenge by exploiting statistical patterns from the model itself. However, it focuses exclusively on attention mechanisms, potentially overlooking the MLP module-a core component of Transformer models known to support factual recall. In this paper, we show that certain value vectors within MLP modules exhibit truthfulness-related statistical patterns. Building on this insight, we propose TruthV, a simple and interpretable training-free method that detects content truthfulness by leveraging these value vectors. On the NoVo benchmark, TruthV significantly outperforms both NoVo and log-likelihood baselines, demonstrating that MLP modules-despite being neglected in prior training-free efforts-encode rich and useful signals for truthfulness detection. These findings offer new insights into how truthfulness is internally represented in LLMs and motivate further research on scalable and interpretable truthfulness detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.01659v2">From Contrast to Commonality: Audio Commonality Captioning for Enhanced Audio-Text Cross-modal Understanding in Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-22
    </div>
    <details class="paper-abstract">
      Audio Captioning (AC) plays a pivotal role in enhancing audio-text cross-modal understanding during the pretraining and finetuning of Multimodal LLMs (MLLMs). To strengthen this alignment, recent works propose Audio Difference Captioning (ADC), which takes multiple audio inputs and encourages the model to describe their differences, thereby promoting fine-grained discrimination. However, despite its effectiveness, ADC introduces a semantic gap between input audios-often rich in diverse events-and the brief, difference-focused short caption. This deviation from AC-style task causes a mismatch with the pretraining objective, leading to catastrophic forgetting. To address this, we propose Audio Commonality Captioning (ACC), a comparably challenging but gentler alternative that guides the model to capture shared semantics across audio clips rather than detailed differences. Experiments show that ACC not only improves audio-text understanding on captioning benchmarks but also better preserves general capabilities across diverse speech and music tasks, confirming its ability to enable more robust cross-modal understanding and achieve a better balance between generalization and task-specific performance in MLLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10369v2">SymRTLO: Enhancing RTL Code Optimization with LLMs and Neuron-Inspired Symbolic Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Optimizing Register Transfer Level (RTL) code is crucial for improving the power, performance, and area (PPA) of digital circuits in the early stages of synthesis. Manual rewriting, guided by synthesis feedback, can yield high-quality results but is time-consuming and error-prone. Most existing compiler-based approaches have difficulty handling complex design constraints. Large Language Model (LLM)-based methods have emerged as a promising alternative to address these challenges. However, LLM-based approaches often face difficulties in ensuring alignment between the generated code and the provided prompts. This paper presents SymRTLO, a novel neuron-symbolic RTL optimization framework that seamlessly integrates LLM-based code rewriting with symbolic reasoning techniques. Our method incorporates a retrieval-augmented generation (RAG) system of optimization rules and Abstract Syntax Tree (AST)-based templates, enabling LLM-based rewriting that maintains syntactic correctness while minimizing undesired circuit behaviors. A symbolic module is proposed for analyzing and optimizing finite state machine (FSM) logic, allowing fine-grained state merging and partial specification handling beyond the scope of pattern-based compilers. Furthermore, a fast verification pipeline, combining formal equivalence checks with test-driven validation, further reduces the complexity of verification. Experiments on the RTL-Rewriter benchmark with Synopsys Design Compiler and Yosys show that SymRTLO improves power, performance, and area (PPA) by up to 43.9%, 62.5%, and 51.1%, respectively, compared to the state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17901v1">Does Audio Matter for Modern Video-LLMs and Their Benchmarks?</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 5 pages, 2 figures, under review. Project page: https://github.com/naver-ai/LLaVA-AV-SSM
    </div>
    <details class="paper-abstract">
      Modern multimodal large language models often claim "video understanding," yet most evaluations use muted videos or simply discard audio. We ask a direct question: how much does audio actually matter for contemporary Video-LLMs and the benchmarks that certify them? We audit widely used suites and observe that many items are even solvable from a single frame, rendering audio largely redundant. Building on LLaVA-OneVision architecture, we attach a speech/audio encoder (e.g., Whisper) and analyze when audio helps, while addressing audio token explosion with a lightweight Mamba-based state-space token compressor. We find that audio yields minimal gains on recent video benchmarks but is decisive on curated, audio-sensitive subsets. To enable faithful evaluation, we release AVQA-Hard and Music-AVQA-Hard, our model, and code. Our findings surface a growing gap between current academic practice and real-world expectations, and provide practical tools for scalable audio-visual Video-LLMs. We will fully open-source our work at https://github.com/naver-ai/LLaVA-AV-SSM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.14498v4">LLaSA: A Sensor-Aware LLM for Natural Language Reasoning of Human Activity from IMU Data</a></div>
    <div class="paper-meta">
      📅 2025-09-22
    </div>
    <details class="paper-abstract">
      Wearable systems can recognize activities from IMU data but often fail to explain their underlying causes or contextual significance. To address this limitation, we introduce two large-scale resources: SensorCap, comprising 35,960 IMU--caption pairs, and OpenSQA, with 199,701 question--answer pairs designed for causal and explanatory reasoning. OpenSQA includes a curated tuning split (Tune-OpenSQA) optimized for scientific accuracy, narrative clarity, and diagnostic insight. Leveraging these datasets, we develop LLaSA (Large Language and Sensor Assistant), a family of compact sensor-aware language models (7B and 13B) that generate interpretable, context-rich responses to open-ended questions grounded in raw IMU data. LLaSA outperforms commercial LLMs, including GPT-3.5 and GPT-4o-mini, on benchmark and real-world tasks, demonstrating the effectiveness of domain supervision and model alignment for sensor reasoning. Our code repository and datasets can be found at https://github.com/BASHLab/LLaSA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17796v1">Findings of the Fourth Shared Task on Multilingual Coreference Resolution: Can LLMs Dethrone Traditional Approaches?</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 Accepted to CODI-CRAC 2025
    </div>
    <details class="paper-abstract">
      The paper presents an overview of the fourth edition of the Shared Task on Multilingual Coreference Resolution, organized as part of the CODI-CRAC 2025 workshop. As in the previous editions, participants were challenged to develop systems that identify mentions and cluster them according to identity coreference. A key innovation of this year's task was the introduction of a dedicated Large Language Model (LLM) track, featuring a simplified plaintext format designed to be more suitable for LLMs than the original CoNLL-U representation. The task also expanded its coverage with three new datasets in two additional languages, using version 1.3 of CorefUD - a harmonized multilingual collection of 22 datasets in 17 languages. In total, nine systems participated, including four LLM-based approaches (two fine-tuned and two using few-shot adaptation). While traditional systems still kept the lead, LLMs showed clear potential, suggesting they may soon challenge established approaches in future editions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11405v2">DCR: Quantifying Data Contamination in LLMs Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 EMNLP 2025 Main
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has heightened concerns about benchmark data contamination (BDC), where models inadvertently memorize evaluation data during the training process, inflating performance metrics, and undermining genuine generalization assessment. This paper introduces the Data Contamination Risk (DCR) framework, a lightweight, interpretable pipeline designed to detect and quantify BDC risk across four granular levels: semantic, informational, data, and label. By synthesizing contamination scores via a fuzzy inference system, DCR produces a unified DCR Factor that adjusts raw accuracy to reflect contamination-aware performance. Validated on 9 LLMs (0.5B-72B) across sentiment analysis, fake news detection, and arithmetic reasoning tasks, the DCR framework reliably diagnoses contamination severity and with accuracy adjusted using the DCR Factor to within 4% average error across the three benchmarks compared to the uncontaminated baseline. Emphasizing computational efficiency and transparency, DCR provides a practical tool for integrating contamination assessment into routine evaluations, fostering fairer comparisons and enhancing the credibility of LLM benchmarking practices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04931v2">Breaking the News: Taking the Roles of Influencer vs. Journalist in a LLM-Based Game for Raising Misinformation Awareness</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 Accepted to ACM CHI PLAY 2025
    </div>
    <details class="paper-abstract">
      Effectively mitigating online misinformation requires understanding of their mechanisms and learning of practical skills for identification and counteraction. Serious games may serve as tools for combating misinformation, teaching players to recognize common misinformation tactics, and improving their skills of discernment. However, current interventions are designed as single-player, choice-based games, which present players with limited predefined choices. Such restrictions reduce replayability and may lead to an overly simplistic understanding of misinformation and how to debunk them. This study seeks to empower people to understand opinion-influencing and misinformation-debunking processes. We created a Player vs. Player (PvP) game in which participants attempt to generate or debunk misinformation to convince the public opinion represented by LLM. Using a within-subjects mixed-methods study design (N=47), we found that this game significantly raised participants' media literacy and improved their ability to identify misinformation. Qualitative analyses revealed how participants' use of debunking and content creation strategies deepened their understanding of misinformation. This work shows the potential for illuminating contrasting viewpoints of social issues by LLM-based mechanics in PvP games.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21693v3">MAKIEval: A Multilingual Automatic WiKidata-based Framework for Cultural Awareness Evaluation for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-22
      | 💬 Accepted by EMNLP 2025 Findings, 33 pages, 30 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are used globally across many languages, but their English-centric pretraining raises concerns about cross-lingual disparities for cultural awareness, often resulting in biased outputs. However, comprehensive multilingual evaluation remains challenging due to limited benchmarks and questionable translation quality. To better assess these disparities, we introduce MAKIEval, an automatic multilingual framework for evaluating cultural awareness in LLMs across languages, regions, and topics. MAKIEval evaluates open-ended text generation, capturing how models express culturally grounded knowledge in natural language. Leveraging Wikidata's multilingual structure as a cross-lingual anchor, it automatically identifies cultural entities in model outputs and links them to structured knowledge, enabling scalable, language-agnostic evaluation without manual annotation or translation. We then introduce four metrics that capture complementary dimensions of cultural awareness: granularity, diversity, cultural specificity, and consensus across languages. We assess 7 LLMs developed from different parts of the world, encompassing both open-source and proprietary systems, across 13 languages, 19 countries and regions, and 6 culturally salient topics (e.g., food, clothing). Notably, we find that models tend to exhibit stronger cultural awareness in English, suggesting that English prompts more effectively activate culturally grounded knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.14746v4">Scaling Efficient LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-22
    </div>
    <details class="paper-abstract">
      Trained LLMs in the transformer architecture are typically sparse in that most of the parameters are negligible, raising questions on efficiency. Furthermore, the so called "AI scaling law" for transformers suggests that the number of parameters must scale linearly with the size of the data. In response, we inquire into efficient LLMs, i.e. those with the fewest parameters that achieve the desired accuracy on a training corpus. Specifically, by comparing theoretical and empirical estimates of the Kullback-Liebler divergence, we derive a natural AI scaling law that the number of parameters in an efficient LLM scales as $D^{\gamma}$ where $D$ is the size of the training data and $ \gamma \in [0.44, 0.72]$, suggesting the existence of more efficient architectures. Against this backdrop, we propose recurrent transformers, combining the efficacy of transformers with the efficiency of recurrent networks, progressively applying a single transformer layer to a fixed-width sliding window across the input sequence. Recurrent transformers (a) run in linear time in the sequence length, (b) are memory-efficient and amenable to parallel processing in large batches, (c) learn to forget history for language tasks, or accumulate history for long range tasks like copy and selective copy, and (d) are amenable to curriculum training to overcome vanishing gradients. In our experiments, we find that recurrent transformers perform favorably on benchmark tests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17730v1">ConfClip: Confidence-Weighted and Clipped Reward for Reinforcement Learning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-22
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has become a standard paradigm for refining large language models (LLMs) beyond pre-training and instruction tuning. A prominent line of work is RL with verifiable rewards (RLVR), which leverages automatically verifiable outcomes (e.g., correctness or executability) to generate reward signals. While efficient, this framework faces two key limitations: First, its binary feedback is too sparse to capture the quality of the reasoning process. Second, its coarse-grained rewards potentially lead to vanishing gradients. Inspired by observations from human learning, we introduce a RL technique that integrates verifiable outcomes with the model's own confidence estimates. This joint design enriches the reward signal, providing finer-grained feedback and implicitly supervising the reasoning process. Experimental results demonstrate that our proposed method enhances RL performance across multiple datasets and reduces token consumption during inference, while incurring negligible additional training cost. Moreover, it can be used as a plug-in module to enhance other state-of-the-art RL methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11930v2">Feedback Friction: LLMs Struggle to Fully Incorporate External Feedback</a></div>
    <div class="paper-meta">
      📅 2025-09-21
    </div>
    <details class="paper-abstract">
      Recent studies have shown LLMs possess some ability to improve their responses when given external feedback. However, it remains unclear how effectively and thoroughly these models can incorporate extrinsic feedback. In an ideal scenario, if LLMs receive near-perfect and complete feedback, we would expect them to fully integrate the feedback and reach correct solutions. In this paper, we systematically investigate LLMs' ability to incorporate feedback by designing a controlled experimental environment. For each problem, a solver model attempts a solution, then a feedback generator with access to near-complete ground-truth answers produces targeted feedback, after which the solver tries again. We evaluate this pipeline across a diverse range of tasks, including math reasoning, knowledge reasoning, scientific reasoning, and general multi-domain evaluations with state-of-the-art language models including Claude 3.7 with extended thinking. Surprisingly, even under these near-ideal conditions, solver models consistently show resistance to feedback, a limitation that we term Feedback Friction. To mitigate this limitation, we experiment with sampling-based strategies like progressive temperature increases and explicit rejection of previously attempted incorrect answers, which yield improvements but still fail to help models achieve target performance. We analyze Feedback Friction and find that models' confidence on specific questions, measured by semantic entropy, predicts feedback resistance: high-confidence predictions remain resistant to external correction. We hope that highlighting this issue in LLMs will help future research in self-improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20383v3">Why Are Web AI Agents More Vulnerable Than Standalone LLMs? A Security Analysis</a></div>
    <div class="paper-meta">
      📅 2025-09-21
      | 💬 Project website: http://vulnerable-ai-agents.github.io
    </div>
    <details class="paper-abstract">
      Recent advancements in Web AI agents have demonstrated remarkable capabilities in addressing complex web navigation tasks. However, emerging research shows that these agents exhibit greater vulnerability compared to standalone Large Language Models (LLMs), despite both being built upon the same safety-aligned models. This discrepancy is particularly concerning given the greater flexibility of Web AI Agent compared to standalone LLMs, which may expose them to a wider range of adversarial user inputs. To build a scaffold that addresses these concerns, this study investigates the underlying factors that contribute to the increased vulnerability of Web AI agents. Notably, this disparity stems from the multifaceted differences between Web AI agents and standalone LLMs, as well as the complex signals - nuances that simple evaluation metrics, such as success rate, often fail to capture. To tackle these challenges, we propose a component-level analysis and a more granular, systematic evaluation framework. Through this fine-grained investigation, we identify three critical factors that amplify the vulnerability of Web AI agents; (1) embedding user goals into the system prompt, (2) multi-step action generation, and (3) observational capabilities. Our findings highlights the pressing need to enhance security and robustness in AI agent design and provide actionable insights for targeted defense strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18135v2">Tool Preferences in Agentic LLMs are Unreliable</a></div>
    <div class="paper-meta">
      📅 2025-09-21
      | 💬 Conference on Empirical Methods in Natural Language Processing (EMNLP) 2025, main
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can now access a wide range of external tools, thanks to the Model Context Protocol (MCP). This greatly expands their abilities as various agents. However, LLMs rely entirely on the text descriptions of tools to decide which ones to use--a process that is surprisingly fragile. In this work, we expose a vulnerability in prevalent tool/function-calling protocols by investigating a series of edits to tool descriptions, some of which can drastically increase a tool's usage from LLMs when competing with alternatives. Through controlled experiments, we show that tools with properly edited descriptions receive over 10 times more usage from GPT-4.1 and Qwen2.5-7B than tools with original descriptions. We further evaluate how various edits to tool descriptions perform when competing directly with one another and how these trends generalize or differ across a broader set of 17 different models. These phenomena, while giving developers a powerful way to promote their tools, underscore the need for a more reliable foundation for agentic LLMs to select and utilize tools and resources. Our code is publicly available at https://github.com/kazemf78/llm-unreliable-tool-preferences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17240v1">Can Agents Judge Systematic Reviews Like Humans? Evaluating SLRs with LLM-based Multi-Agent System</a></div>
    <div class="paper-meta">
      📅 2025-09-21
    </div>
    <details class="paper-abstract">
      Systematic Literature Reviews (SLRs) are foundational to evidence-based research but remain labor-intensive and prone to inconsistency across disciplines. We present an LLM-based SLR evaluation copilot built on a Multi-Agent System (MAS) architecture to assist researchers in assessing the overall quality of the systematic literature reviews. The system automates protocol validation, methodological assessment, and topic relevance checks using a scholarly database. Unlike conventional single-agent methods, our design integrates a specialized agentic approach aligned with PRISMA guidelines to support more structured and interpretable evaluations. We conducted an initial study on five published SLRs from diverse domains, comparing system outputs to expert-annotated PRISMA scores, and observed 84% agreement. While early results are promising, this work represents a first step toward scalable and accurate NLP-driven systems for interdisciplinary workflows and reveals their capacity for rigorous, domain-agnostic knowledge aggregation to streamline the review process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20024v3">Using item recommendations and LLMs in marketing email titles</a></div>
    <div class="paper-meta">
      📅 2025-09-21
      | 💬 Accepted to The Second Workshop on Generative AI for E-commerce (GenAIECommerce '25), held September 22, 2025, in Prague, Czech Republic. 3 figures
    </div>
    <details class="paper-abstract">
      E-commerce marketplaces make use of a number of marketing channels like emails, push notifications, etc. to reach their users and stimulate purchases. Personalized emails especially are a popular touch point for marketers to inform users of latest items in stock, especially for those who stopped visiting the marketplace. Such emails contain personalized recommendations tailored to each user's interests, enticing users to buy relevant items. A common limitation of these emails is that the primary entry point, the title of the email, tends to follow fixed templates, failing to inspire enough interest in the contents. In this work, we explore the potential of large language models (LLMs) for generating thematic titles that reflect the personalized content of the emails. We perform offline simulations and conduct online experiments on the order of millions of users, finding our techniques useful in improving the engagement between customers and our emails. We highlight key findings and learnings as we productionize the safe and automated generation of email titles for millions of users.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21548v2">Fluent but Foreign: Even Regional LLMs Lack Cultural Alignment</a></div>
    <div class="paper-meta">
      📅 2025-09-21
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are used worldwide, yet exhibit Western cultural tendencies. Many countries are now building ``regional'' LLMs, but it remains unclear whether they reflect local values and practices or merely speak local languages. Using India as a case study, we evaluate six Indic and six global LLMs on two dimensions -- values and practices -- grounded in nationally representative surveys and community-sourced QA datasets. Across tasks, Indic models do not align better with Indian norms than global models; in fact, a U.S. respondent is a closer proxy for Indian values than any Indic model. Prompting and regional fine-tuning fail to recover alignment and can even degrade existing knowledge. We attribute this to scarce culturally grounded data, especially for pretraining. We position cultural evaluation as a first-class requirement alongside multilingual benchmarks and offer a reusable, community-grounded methodology. We call for native, community-authored corpora and thick x wide evaluations to build truly sovereign LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17197v1">SignalLLM: A General-Purpose LLM Agent Framework for Automated Signal Processing</a></div>
    <div class="paper-meta">
      📅 2025-09-21
      | 💬 11 pages
    </div>
    <details class="paper-abstract">
      Modern signal processing (SP) pipelines, whether model-based or data-driven, often constrained by complex and fragmented workflow, rely heavily on expert knowledge and manual engineering, and struggle with adaptability and generalization under limited data. In contrast, Large Language Models (LLMs) offer strong reasoning capabilities, broad general-purpose knowledge, in-context learning, and cross-modal transfer abilities, positioning them as powerful tools for automating and generalizing SP workflows. Motivated by these potentials, we introduce SignalLLM, the first general-purpose LLM-based agent framework for general SP tasks. Unlike prior LLM-based SP approaches that are limited to narrow applications or tricky prompting, SignalLLM introduces a principled, modular architecture. It decomposes high-level SP goals into structured subtasks via in-context learning and domain-specific retrieval, followed by hierarchical planning through adaptive retrieval-augmented generation (RAG) and refinement; these subtasks are then executed through prompt-based reasoning, cross-modal reasoning, code synthesis, model invocation, or data-driven LLM-assisted modeling. Its generalizable design enables the flexible selection of problem solving strategies across different signal modalities, task types, and data conditions. We demonstrate the versatility and effectiveness of SignalLLM through five representative tasks in communication and sensing, such as radar target detection, human activity recognition, and text compression. Experimental results show superior performance over traditional and existing LLM-based methods, particularly in few-shot and zero-shot settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17178v1">Attention Consistency for LLMs Explanation</a></div>
    <div class="paper-meta">
      📅 2025-09-21
    </div>
    <details class="paper-abstract">
      Understanding the decision-making processes of large language models (LLMs) is essential for their trustworthy development and deployment. However, current interpretability methods often face challenges such as low resolution and high computational cost. To address these limitations, we propose the \textbf{Multi-Layer Attention Consistency Score (MACS)}, a novel, lightweight, and easily deployable heuristic for estimating the importance of input tokens in decoder-based models. MACS measures contributions of input tokens based on the consistency of maximal attention. Empirical evaluations demonstrate that MACS achieves a favorable trade-off between interpretability quality and computational efficiency, showing faithfulness comparable to complex techniques with a 22\% decrease in VRAM usage and 30\% reduction in latency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17167v1">SFT-TA: Supervised Fine-Tuned Agents in Multi-Agent LLMs for Automated Inductive Thematic Analysis</a></div>
    <div class="paper-meta">
      📅 2025-09-21
    </div>
    <details class="paper-abstract">
      Thematic Analysis (TA) is a widely used qualitative method that provides a structured yet flexible framework for identifying and reporting patterns in clinical interview transcripts. However, manual thematic analysis is time-consuming and limits scalability. Recent advances in LLMs offer a pathway to automate thematic analysis, but alignment with human results remains limited. To address these limitations, we propose SFT-TA, an automated thematic analysis framework that embeds supervised fine-tuned (SFT) agents within a multi-agent system. Our framework outperforms existing frameworks and the gpt-4o baseline in alignment with human reference themes. We observed that SFT agents alone may underperform, but achieve better results than the baseline when embedded within a multi-agent system. Our results highlight that embedding SFT agents in specific roles within a multi-agent system is a promising pathway to improve alignment with desired outputs for thematic analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17136v1">SAEC: Scene-Aware Enhanced Edge-Cloud Collaborative Industrial Vision Inspection with Multimodal LLM</a></div>
    <div class="paper-meta">
      📅 2025-09-21
      | 💬 5 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Industrial vision inspection requires high accuracy under stringent resource constraints, yet existing approaches face a fundamental trade-off. Multimodal LLMs (MLLMs) deliver strong reasoning capabilities but incur prohibitive computational costs, while lightweight edge models often fail on complex cases. In this paper, we present SAEC, a scene-aware enhanced edge-cloud collaborative industrial vision inspection framework with MLLM. The framework is composed of three synergistic components: (1) Efficient MLLM Fine-Tuning for Complex Defect Inspection, (2) Lightweight Multiscale Scene-Complexity Estimation, and (3) Adaptive Edge-Cloud Scheduler. Together, these modules enable robust defect detection by tailoring multimodal reasoning to scene complexity and dynamically balancing computation between edge and cloud resources. Experimental results on MVTec AD and KSDD2 datasets demonstrate that SAEC attains 85.11% and 82.72% accuracy, surpassing Qwen by 22.1% and 20.8%, and LLaVA by 33.3% and 31.6%. It also reduces runtime by up to 22.4% and cuts energy per correct decision by 40%-74%. The code is available at https://github.com/YuHao-Tian/SAEC.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17096v1">Prompt-with-Me: in-IDE Structured Prompt Management for LLM-Driven Software Engineering</a></div>
    <div class="paper-meta">
      📅 2025-09-21
      | 💬 Accepted in the 40th IEEE/ACM International Conference on Automated Software Engineering, ASE 2025 (Industry track)
    </div>
    <details class="paper-abstract">
      Large Language Models are transforming software engineering, yet prompt management in practice remains ad hoc, hindering reliability, reuse, and integration into industrial workflows. We present Prompt-with-Me, a practical solution for structured prompt management embedded directly in the development environment. The system automatically classifies prompts using a four-dimensional taxonomy encompassing intent, author role, software development lifecycle stage, and prompt type. To enhance prompt reuse and quality, Prompt-with-Me suggests language refinements, masks sensitive information, and extracts reusable templates from a developer's prompt library. Our taxonomy study of 1108 real-world prompts demonstrates that modern LLMs can accurately classify software engineering prompts. Furthermore, our user study with 11 participants shows strong developer acceptance, with high usability (Mean SUS=73), low cognitive load (Mean NASA-TLX=21), and reported gains in prompt quality and efficiency through reduced repetitive effort. Lastly, we offer actionable insights for building the next generation of prompt management and maintenance tools for software engineering workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20764v2">Feel the Difference? A Comparative Analysis of Emotional Arcs in Real and LLM-Generated CBT Sessions</a></div>
    <div class="paper-meta">
      📅 2025-09-21
      | 💬 Accepted at 2025 EMNLP findings,19 page,2 figures
    </div>
    <details class="paper-abstract">
      Synthetic therapy dialogues generated by large language models (LLMs) are increasingly used in mental health NLP to simulate counseling scenarios, train models, and supplement limited real-world data. However, it remains unclear whether these synthetic conversations capture the nuanced emotional dynamics of real therapy. In this work, we introduce RealCBT, a dataset of authentic cognitive behavioral therapy (CBT) dialogues, and conduct the first comparative analysis of emotional arcs between real and LLM-generated CBT sessions. We adapt the Utterance Emotion Dynamics framework to analyze fine-grained affective trajectories across valence, arousal, and dominance dimensions. Our analysis spans both full dialogues and individual speaker roles (counselor and client), using real sessions from the RealCBT dataset and synthetic dialogues from the CACTUS dataset. We find that while synthetic dialogues are fluent and structurally coherent, they diverge from real conversations in key emotional properties: real sessions exhibit greater emotional variability, more emotion-laden language, and more authentic patterns of reactivity and regulation. Moreover, emotional arc similarity remains low across all pairings, with especially weak alignment between real and synthetic speakers. These findings underscore the limitations of current LLM-generated therapy data and highlight the importance of emotional fidelity in mental health applications. To support future research, our dataset RealCBT is released at https://gitlab.com/xiaoyi.wang/realcbt-dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17072v1">SnipSnap: A Joint Compression Format and Dataflow Co-Optimization Framework for Efficient Sparse LLM Accelerator Design</a></div>
    <div class="paper-meta">
      📅 2025-09-21
      | 💬 To appear in the 31st Asia and South Pacific Design Automation Conference (ASP-DAC 2026)
    </div>
    <details class="paper-abstract">
      The growing scale of large language models (LLMs) has intensified demands on computation and memory, making efficient inference a key challenge. While sparsity can reduce these costs, existing design space exploration (DSE) frameworks often overlook compression formats, a key factor for leveraging sparsity on accelerators. This paper proposes SnipSnap, a joint compression format and dataflow co-optimization framework for efficient sparse LLM accelerator design. SnipSnap introduces: (1) a hierarchical compression format encoding to expand the design space; (2) an adaptive compression engine for selecting formats under diverse sparsity; and (3) a progressive co-search workflow that jointly optimizes dataflow and compression formats. SnipSnap achieves 18.24\% average memory energy savings via format optimization, along with 2248.3$\times$ and 21.0$\times$ speedups over Sparseloop and DiMO-Sparse frameworks, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17066v1">RALLM-POI: Retrieval-Augmented LLM for Zero-shot Next POI Recommendation with Geographical Reranking</a></div>
    <div class="paper-meta">
      📅 2025-09-21
      | 💬 PRICAI 2025
    </div>
    <details class="paper-abstract">
      Next point-of-interest (POI) recommendation predicts a user's next destination from historical movements. Traditional models require intensive training, while LLMs offer flexible and generalizable zero-shot solutions but often generate generic or geographically irrelevant results due to missing trajectory and spatial context. To address these issues, we propose RALLM-POI, a framework that couples LLMs with retrieval-augmented generation and self-rectification. We first propose a Historical Trajectory Retriever (HTR) that retrieves relevant past trajectories to serve as contextual references, which are then reranked by a Geographical Distance Reranker (GDR) for prioritizing spatially relevant trajectories. Lastly, an Agentic LLM Rectifier (ALR) is designed to refine outputs through self-reflection. Without additional training, RALLM-POI achieves substantial accuracy gains across three real-world Foursquare datasets, outperforming both conventional and LLM-based baselines. Code is released at https://github.com/LKRcrocodile/RALLM-POI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17054v1">TactfulToM: Do LLMs Have the Theory of Mind Ability to Understand White Lies?</a></div>
    <div class="paper-meta">
      📅 2025-09-21
    </div>
    <details class="paper-abstract">
      While recent studies explore Large Language Models' (LLMs) performance on Theory of Mind (ToM) reasoning tasks, research on ToM abilities that require more nuanced social context is limited, such as white lies. We introduce TactfulToM, a novel English benchmark designed to evaluate LLMs' ability to understand white lies within real-life conversations and reason about prosocial motivations behind them, particularly when they are used to spare others' feelings and maintain social harmony. Our benchmark is generated through a multi-stage human-in-the-loop pipeline where LLMs expand manually designed seed stories into conversations to maintain the information asymmetry between participants necessary for authentic white lies. We show that TactfulToM is challenging for state-of-the-art models, which perform substantially below humans, revealing shortcomings in their ability to fully comprehend the ToM reasoning that enables true understanding of white lies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17601v3">Revisiting Backdoor Attacks on LLMs: A Stealthy and Practical Poisoning Framework via Harmless Inputs</a></div>
    <div class="paper-meta">
      📅 2025-09-21
    </div>
    <details class="paper-abstract">
      Recent studies have widely investigated backdoor attacks on Large language models (LLMs) by inserting harmful question-answer (QA) pairs into training data to implant triggers. However, we revisit existing attack methods and identify two critical limitations of that seriously undermine their stealthiness and practicality: (1) directly embedding harmful content into the training data compromise the model's safety alignment, resulting in high attack success rates even for clean queries without triggers, and (2) the poisoned training samples can be easily detected and filtered by safety-aligned guardrails (e.g., LLaMAGuard). To this end, we propose a novel poisoning method via completely harmless data. Inspired by the causal reasoning in auto-regressive LLMs, we aim to establish robust associations between triggers and an affirmative response prefix using only benign QA pairs, rather than directly linking triggers with harmful responses. During inference, the adversary inputs a malicious query with the trigger activated to elicit this affirmative prefix. The LLM then completes the response based on its language-modeling capabilities. Notably, achieving this behavior from clean QA pairs is non-trivial. We observe an interesting resistance phenomenon where the LLM initially appears to agree but subsequently refuses to answer. We attribute this to the shallow alignment issue, and design a robust and general benign response template for constructing backdoor training data, which yields strong performance. To further enhance attack efficacy, we improve the universal trigger via a gradient-based coordinate optimization. Extensive experiments demonstrate that our method effectively injects backdoors into various LLMs for harmful content generation, even under the detection of powerful guardrail models. E.g., ASRs of 86.67% and 85% on LLaMA-3-8B and Qwen-2.5-7B judged by GPT-4o.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17030v1">The Transfer Neurons Hypothesis: An Underlying Mechanism for Language Latent Space Transitions in Multilingual LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-21
      | 💬 57 pages, 47 figures and 41 tables; Accepted to EMNLP 2025 Main
    </div>
    <details class="paper-abstract">
      Recent studies have suggested a processing framework for multilingual inputs in decoder-based LLMs: early layers convert inputs into English-centric and language-agnostic representations; middle layers perform reasoning within an English-centric latent space; and final layers generate outputs by transforming these representations back into language-specific latent spaces. However, the internal dynamics of such transformation and the underlying mechanism remain underexplored. Towards a deeper understanding of this framework, we propose and empirically validate The Transfer Neurons Hypothesis: certain neurons in the MLP module are responsible for transferring representations between language-specific latent spaces and a shared semantic latent space. Furthermore, we show that one function of language-specific neurons, as identified in recent studies, is to facilitate movement between latent spaces. Finally, we show that transfer neurons are critical for reasoning in multilingual LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17022v1">VAInpaint: Zero-Shot Video-Audio inpainting framework with LLMs-driven Module</a></div>
    <div class="paper-meta">
      📅 2025-09-21
    </div>
    <details class="paper-abstract">
      Video and audio inpainting for mixed audio-visual content has become a crucial task in multimedia editing recently. However, precisely removing an object and its corresponding audio from a video without affecting the rest of the scene remains a significant challenge. To address this, we propose VAInpaint, a novel pipeline that first utilizes a segmentation model to generate masks and guide a video inpainting model in removing objects. At the same time, an LLM then analyzes the scene globally, while a region-specific model provides localized descriptions. Both the overall and regional descriptions will be inputted into an LLM, which will refine the content and turn it into text queries for our text-driven audio separation model. Our audio separation model is fine-tuned on a customized dataset comprising segmented MUSIC instrument images and VGGSound backgrounds to enhance its generalization performance. Experiments show that our method achieves performance comparable to current benchmarks in both audio and video inpainting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.12797v2">CEBench: A Benchmarking Toolkit for the Cost-Effectiveness of LLM Pipelines</a></div>
    <div class="paper-meta">
      📅 2025-09-21
    </div>
    <details class="paper-abstract">
      Online Large Language Model (LLM) services such as ChatGPT and Claude 3 have transformed business operations and academic research by effortlessly enabling new opportunities. However, due to data-sharing restrictions, sectors such as healthcare and finance prefer to deploy local LLM applications using costly hardware resources. This scenario requires a balance between the effectiveness advantages of LLMs and significant financial burdens. Additionally, the rapid evolution of models increases the frequency and redundancy of benchmarking efforts. Existing benchmarking toolkits, which typically focus on effectiveness, often overlook economic considerations, making their findings less applicable to practical scenarios. To address these challenges, we introduce CEBench, an open-source toolkit specifically designed for multi-objective benchmarking that focuses on the critical trade-offs between expenditure and effectiveness required for LLM deployments. CEBench allows for easy modifications through configuration files, enabling stakeholders to effectively assess and optimize these trade-offs. This strategic capability supports crucial decision-making processes aimed at maximizing effectiveness while minimizing cost impacts. By streamlining the evaluation process and emphasizing cost-effectiveness, CEBench seeks to facilitate the development of economically viable AI solutions across various industries and research fields. The code and demonstration are available in https://github.com/amademicnoboday12/CEBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.12636v4">MORepair: Teaching LLMs to Repair Code via Multi-Objective Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2025-09-21
    </div>
    <details class="paper-abstract">
      Within the realm of software engineering, specialized tasks on code, such as program repair, present unique challenges, necessitating fine-tuning Large language models~(LLMs) to unlock state-of-the-art performance. Fine-tuning approaches proposed in the literature for LLMs on program repair tasks generally overlook the need to reason about the logic behind code changes, beyond syntactic patterns in the data. High-performing fine-tuning experiments also usually come at very high computational costs. With MORepair, we propose a novel perspective on the learning focus of LLM fine-tuning for program repair: we not only adapt the LLM parameters to the syntactic nuances of the task of code transformation (objective 1), but we also specifically fine-tune the LLM with respect to the logical reason behind the code change in the training data (objective 2). Such a multi-objective fine-tuning will instruct LLMs to generate high-quality patches. We apply MORepair to fine-tune four open-source LLMs with different sizes and architectures. Experimental results on function-level and repository-level repair benchmarks show that the implemented fine-tuning effectively boosts LLM repair performance by 11.4% to 56.0%. We further show that our fine-tuning strategy yields superior performance compared to the state-of-the-art approaches, including standard fine-tuning, Fine-tune-CoT, and RepairLLaMA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16970v1">LLM-Assisted Semantic Guidance for Sparsely Annotated Remote Sensing Object Detection</a></div>
    <div class="paper-meta">
      📅 2025-09-21
    </div>
    <details class="paper-abstract">
      Sparse annotation in remote sensing object detection poses significant challenges due to dense object distributions and category imbalances. Although existing Dense Pseudo-Label methods have demonstrated substantial potential in pseudo-labeling tasks, they remain constrained by selection ambiguities and inconsistencies in confidence estimation.In this paper, we introduce an LLM-assisted semantic guidance framework tailored for sparsely annotated remote sensing object detection, exploiting the advanced semantic reasoning capabilities of large language models (LLMs) to distill high-confidence pseudo-labels.By integrating LLM-generated semantic priors, we propose a Class-Aware Dense Pseudo-Label Assignment mechanism that adaptively assigns pseudo-labels for both unlabeled and sparsely labeled data, ensuring robust supervision across varying data distributions. Additionally, we develop an Adaptive Hard-Negative Reweighting Module to stabilize the supervised learning branch by mitigating the influence of confounding background information. Extensive experiments on DOTA and HRSC2016 demonstrate that the proposed method outperforms existing single-stage detector-based frameworks, significantly improving detection performance under sparse annotations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01609v2">Adaptive Distraction: Probing LLM Contextual Robustness with Automated Tree Search</a></div>
    <div class="paper-meta">
      📅 2025-09-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often struggle to maintain their original performance when faced with semantically coherent but task-irrelevant contextual information. Although prior studies have explored this issue using fixed-template or retrieval-based distractions, such static methods show limited effectiveness against contemporary models. To address this problem, we propose a dynamic distraction generation framework based on tree search, where the generation process is guided by model behavior. Without modifying the original question or answer, the method efficiently produces challenging adaptive distractions across multiple datasets, enabling systematic stress testing of LLMs' contextual robustness. Experiments on four benchmarks demonstrate that the generated distractions lead to an average performance drop of over 45\% for mainstream models. Further comparisons of mitigation strategies show that prompt-based optimization methods yield limited gains, whereas post-training approaches (e.g., DPO) significantly enhance the model's contextual robustness. The results indicate that these issues do not stem from knowledge deficits in LLMs, but from a fundamental inability to maintain consistent reasoning under contextual distraction, posing a major challenge to the reliability of LLMs in real-world applications. The code is publicly available at https://github.com/wyf23187/Adaptive_Distractions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09585v2">Elevating Visual Perception in Multimodal LLMs with Auxiliary Embedding Distillation</a></div>
    <div class="paper-meta">
      📅 2025-09-21
      | 💬 Project Page: https://praeclarumjj3.github.io/visper_lm/
    </div>
    <details class="paper-abstract">
      In recent times, the standard practice for developing MLLMs is to feed features from vision encoder(s) into the LLM and train with natural language supervision. This approach often causes models to lean towards language comprehension and undermine the rich visual perception signals present in the data, which are critical for tasks involving spatial reasoning in the domain of embodied AI and robotics. Is it possible to optimize both at the same time? In this work, we propose VisPer-LM, the first approach that infuses visual perception knowledge from expert vision encoders into the LLM's (of an MLLM) hidden representations. We start by investigating MLLMs trained solely with natural language supervision and identify a positive correlation between the quality of visual representations within these models and their downstream performance. Given this insight, we formulate the objective during the pretraining stage in MLLMs as a coupled optimization of predictive visual embedding and next (text) token prediction. Moreover, through extensive probing, we observe improved visual representation quality due to embedding optimization, underscoring the effectiveness of our probing setup. We demonstrate that our VisPer-LM outperforms the single and multi-encoder baselines, proving our approach's superiority over explicitly feeding the corresponding features to the LLM. In particular, VisPer-LM boosts performance by an average margin of up to 2.5% on various benchmarks, with a notable improvement of 8.7% on the Depth task in CV-Bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16920v1">SwarmChat: An LLM-Based, Context-Aware Multimodal Interaction System for Robotic Swarms</a></div>
    <div class="paper-meta">
      📅 2025-09-21
      | 💬 This paper has been accepted and presented at the 16th International Conference on Swarm Intelligence (ICSI 2025), held on July 11-15, 2025, in Yokohama, Japan
    </div>
    <details class="paper-abstract">
      Traditional Human-Swarm Interaction (HSI) methods often lack intuitive real-time adaptive interfaces, making decision making slower and increasing cognitive load while limiting command flexibility. To solve this, we present SwarmChat, a context-aware, multimodal interaction system powered by Large Language Models (LLMs). SwarmChat enables users to issue natural language commands to robotic swarms using multiple modalities, such as text, voice, or teleoperation. The system integrates four LLM-based modules: Context Generator, Intent Recognition, Task Planner, and Modality Selector. These modules collaboratively generate context from keywords, detect user intent, adapt commands based on real-time robot state, and suggest optimal communication modalities. Its three-layer architecture offers a dynamic interface with both fixed and customizable command options, supporting flexible control while optimizing cognitive effort. The preliminary evaluation also shows that the SwarmChat's LLM modules provide accurate context interpretation, relevant intent recognition, and effective command delivery, achieving high user satisfaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16072v3">InMind: Evaluating LLMs in Capturing and Applying Individual Human Reasoning Styles</a></div>
    <div class="paper-meta">
      📅 2025-09-21
      | 💬 EMNLP 2025 MainConference
    </div>
    <details class="paper-abstract">
      LLMs have shown strong performance on human-centric reasoning tasks. While previous evaluations have explored whether LLMs can infer intentions or detect deception, they often overlook the individualized reasoning styles that influence how people interpret and act in social contexts. Social deduction games (SDGs) provide a natural testbed for evaluating individualized reasoning styles, where different players may adopt diverse but contextually valid reasoning strategies under identical conditions. To address this, we introduce InMind, a cognitively grounded evaluation framework designed to assess whether LLMs can capture and apply personalized reasoning styles in SDGs. InMind enhances structured gameplay data with round-level strategy traces and post-game reflections, collected under both Observer and Participant modes. It supports four cognitively motivated tasks that jointly evaluate both static alignment and dynamic adaptation. As a case study, we apply InMind to the game Avalon, evaluating 11 state-of-the-art LLMs. General-purpose LLMs, even GPT-4o frequently rely on lexical cues, struggling to anchor reflections in temporal gameplay or adapt to evolving strategies. In contrast, reasoning-enhanced LLMs like DeepSeek-R1 exhibit early signs of style-sensitive reasoning. These findings reveal key limitations in current LLMs' capacity for individualized, adaptive reasoning, and position InMind as a step toward cognitively aligned human-AI interaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16891v1">LLMs as Layout Designers: A Spatial Reasoning Perspective</a></div>
    <div class="paper-meta">
      📅 2025-09-21
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have demonstrated impressive reasoning and planning abilities in textual domains and can effectively follow instructions for complex tasks, their capacity for spatial understanding and reasoning remains limited. Such capabilities, however, are critical for applications like content-aware graphic layout design, which demands precise placement, alignment, and structural organization of multiple elements within constrained visual spaces. To address this gap, we propose LaySPA, a reinforcement learning-based framework that augments LLM agents with explicit spatial reasoning capabilities. LaySPA leverages hybrid reward signals that capture geometric validity, structural fidelity, and visual quality, enabling agents to model inter-element relationships, navigate the canvas, and optimize spatial arrangements. Through iterative self-exploration and adaptive policy optimization, LaySPA produces both interpretable reasoning traces and structured layouts. Experimental results demonstrate that LaySPA generates structurally sound and visually appealing layouts, outperforming larger general-purpose LLMs and achieving results on par with state-of-the-art specialized layout models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16866v1">seqBench: A Tunable Benchmark to Quantify Sequential Reasoning Limits of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-21
    </div>
    <details class="paper-abstract">
      We introduce seqBench, a parametrized benchmark for probing sequential reasoning limits in Large Language Models (LLMs) through precise, multi-dimensional control over several key complexity dimensions. seqBench allows systematic variation of (1) the logical depth, defined as the number of sequential actions required to solve the task; (2) the number of backtracking steps along the optimal path, quantifying how often the agent must revisit prior states to satisfy deferred preconditions (e.g., retrieving a key after encountering a locked door); and (3) the noise ratio, defined as the ratio between supporting and distracting facts about the environment. Our evaluations on state-of-the-art LLMs reveal a universal failure pattern: accuracy collapses exponentially beyond a model-specific logical depth. Unlike existing benchmarks, seqBench's fine-grained control facilitates targeted analyses of these reasoning failures, illuminating universal scaling laws and statistical limits, as detailed in this paper alongside its generation methodology and evaluation metrics. We find that even top-performing models systematically fail on seqBench's structured reasoning tasks despite minimal search complexity, underscoring key limitations in their commonsense reasoning capabilities. Designed for future evolution to keep pace with advancing models, the seqBench datasets are publicly released to spur deeper scientific inquiry into LLM reasoning, aiming to establish a clearer understanding of their true potential and current boundaries for robust real-world application.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16861v1">AdaptiveGuard: Towards Adaptive Runtime Safety for LLM-Powered Software</a></div>
    <div class="paper-meta">
      📅 2025-09-21
      | 💬 Accepted to the ASE 2025 International Conference on Automated Software Engineering, Industry Showcase Track
    </div>
    <details class="paper-abstract">
      Guardrails are critical for the safe deployment of Large Language Models (LLMs)-powered software. Unlike traditional rule-based systems with limited, predefined input-output spaces that inherently constrain unsafe behavior, LLMs enable open-ended, intelligent interactions--opening the door to jailbreak attacks through user inputs. Guardrails serve as a protective layer, filtering unsafe prompts before they reach the LLM. However, prior research shows that jailbreak attacks can still succeed over 70% of the time, even against advanced models like GPT-4o. While guardrails such as LlamaGuard report up to 95% accuracy, our preliminary analysis shows their performance can drop sharply--to as low as 12%--when confronted with unseen attacks. This highlights a growing software engineering challenge: how to build a post-deployment guardrail that adapts dynamically to emerging threats? To address this, we propose AdaptiveGuard, an adaptive guardrail that detects novel jailbreak attacks as out-of-distribution (OOD) inputs and learns to defend against them through a continual learning framework. Through empirical evaluation, AdaptiveGuard achieves 96% OOD detection accuracy, adapts to new attacks in just two update steps, and retains over 85% F1-score on in-distribution data post-adaptation, outperforming other baselines. These results demonstrate that AdaptiveGuard is a guardrail capable of evolving in response to emerging jailbreak strategies post deployment. We release our AdaptiveGuard and studied datasets at https://github.com/awsm-research/AdaptiveGuard to support further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.15095v2">Listening, Imagining & Refining: A Heuristic Optimized ASR Correction Framework with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-20
    </div>
    <details class="paper-abstract">
      Automatic Speech Recognition (ASR) systems remain prone to errors that affect downstream applications. In this paper, we propose LIR-ASR, a heuristic optimized iterative correction framework using LLMs, inspired by human auditory perception. LIR-ASR applies a "Listening-Imagining-Refining" strategy, generating phonetic variants and refining them in context. A heuristic optimization with finite state machine (FSM) is introduced to prevent the correction process from being trapped in local optima and rule-based constraints help maintain semantic fidelity. Experiments on both English and Chinese ASR outputs show that LIR-ASR achieves average reductions in CER/WER of up to 1.5 percentage points compared to baselines, demonstrating substantial accuracy gains in transcription.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20849v2">Latent Inter-User Difference Modeling for LLM Personalization</a></div>
    <div class="paper-meta">
      📅 2025-09-20
      | 💬 2025 EMNLP Main Conference (Oral)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly integrated into users' daily lives, leading to a growing demand for personalized outputs. Previous work focuses on leveraging a user's own history, overlooking inter-user differences that are crucial for effective personalization. While recent work has attempted to model such differences, the reliance on language-based prompts often hampers the effective extraction of meaningful distinctions. To address these issues, we propose Difference-aware Embedding-based Personalization (DEP), a framework that models inter-user differences in the latent space instead of relying on language prompts. DEP constructs soft prompts by contrasting a user's embedding with those of peers who engaged with similar content, highlighting relative behavioral signals. A sparse autoencoder then filters and compresses both user-specific and difference-aware embeddings, preserving only task-relevant features before injecting them into a frozen LLM. Experiments on personalized review generation show that DEP consistently outperforms baseline methods across multiple metrics. Our code is available at https://github.com/SnowCharmQ/DEP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16648v1">FESTA: Functionally Equivalent Sampling for Trust Assessment of Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-20
      | 💬 Accepted in the Findings of EMNLP, 2025
    </div>
    <details class="paper-abstract">
      The accurate trust assessment of multimodal large language models (MLLMs) generated predictions, which can enable selective prediction and improve user confidence, is challenging due to the diverse multi-modal input paradigms. We propose Functionally Equivalent Sampling for Trust Assessment (FESTA), a multimodal input sampling technique for MLLMs, that generates an uncertainty measure based on the equivalent and complementary input samplings. The proposed task-preserving sampling approach for uncertainty quantification expands the input space to probe the consistency (through equivalent samples) and sensitivity (through complementary samples) of the model. FESTA uses only input-output access of the model (black-box), and does not require ground truth (unsupervised). The experiments are conducted with various off-the-shelf multi-modal LLMs, on both visual and audio reasoning tasks. The proposed FESTA uncertainty estimate achieves significant improvement (33.3% relative improvement for vision-LLMs and 29.6% relative improvement for audio-LLMs) in selective prediction performance, based on area-under-receiver-operating-characteristic curve (AUROC) metric in detecting mispredictions. The code implementation is open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05652v3">Sugar-Coated Poison: Benign Generation Unlocks LLM Jailbreaking</a></div>
    <div class="paper-meta">
      📅 2025-09-20
      | 💬 Accepted by EMNLP2025
    </div>
    <details class="paper-abstract">
      With the increasingly deep integration of large language models (LLMs) across diverse domains, the effectiveness of their safety mechanisms is encountering severe challenges. Currently, jailbreak attacks based on prompt engineering have become a major safety threat. However, existing methods primarily rely on black-box manipulation of prompt templates, resulting in poor interpretability and limited generalization. To break through the bottleneck, this study first introduces the concept of Defense Threshold Decay (DTD), revealing the potential safety impact caused by LLMs' benign generation: as benign content generation in LLMs increases, the model's focus on input instructions progressively diminishes. Building on this insight, we propose the Sugar-Coated Poison (SCP) attack paradigm, which uses a "semantic reversal" strategy to craft benign inputs that are opposite in meaning to malicious intent. This strategy induces the models to generate extensive benign content, thereby enabling adversarial reasoning to bypass safety mechanisms. Experiments show that SCP outperforms existing baselines. Remarkably, it achieves an average attack success rate of 87.23% across six LLMs. For defense, we propose Part-of-Speech Defense (POSD), leveraging verb-noun dependencies for syntactic analysis to enhance safety of LLMs while preserving their generalization ability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16622v1">Audio-Conditioned Diffusion LLMs for ASR and Deliberation Processing</a></div>
    <div class="paper-meta">
      📅 2025-09-20
    </div>
    <details class="paper-abstract">
      Diffusion-based large language models (DLLMs) have recently attracted growing interest as an alternative to autoregressive decoders. In this work, we present an empirical study on using the diffusion-based large language model LLaDA for automatic speech recognition (ASR). We first investigate its use as an external deliberation-based processing module for Whisper-LLaMA transcripts. By leveraging the bidirectional attention and denoising capabilities of LLaDA, we explore random masking, low-confidence masking, and semi-autoregressive strategies, showing that Whisper-LLaDA substantially reduces WER compared with the baseline. On LibriSpeech, the best cascade system achieves 2.25%/4.94% WER on test-clean/test-other, representing a 12.3% relative improvement over the Whisper-LLaMA baseline on the test-other split. In contrast, a plain-text LLaDA without acoustic features fails to improve accuracy, highlighting the importance of audio-conditioned embeddings. We further evaluate Whisper-LLaDA as a standalone decoder for ASR with diffusion-based and semi-autoregressive decoding. Most experimental configurations achieve faster inference than the Whisper-LLaMA baseline, although recognition accuracy is slightly lower. These findings offer an empirical view of diffusion-based LLMs for ASR and point to promising directions for improvements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16615v1">LLM-Guided Task- and Affordance-Level Exploration in Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-20
      | 💬 8 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) is a promising approach for robotic manipulation, but it can suffer from low sample efficiency and requires extensive exploration of large state-action spaces. Recent methods leverage the commonsense knowledge and reasoning abilities of large language models (LLMs) to guide exploration toward more meaningful states. However, LLMs can produce plans that are semantically plausible yet physically infeasible, yielding unreliable behavior. We introduce LLM-TALE, a framework that uses LLMs' planning to directly steer RL exploration. LLM-TALE integrates planning at both the task level and the affordance level, improving learning efficiency by directing agents toward semantically meaningful actions. Unlike prior approaches that assume optimal LLM-generated plans or rewards, LLM-TALE corrects suboptimality online and explores multimodal affordance-level plans without human supervision. We evaluate LLM-TALE on pick-and-place tasks in standard RL benchmarks, observing improvements in both sample efficiency and success rates over strong baselines. Real-robot experiments indicate promising zero-shot sim-to-real transfer. Code and supplementary material are available at https://llm-tale.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16590v1">Question Answering with LLMs and Learning from Answer Sets</a></div>
    <div class="paper-meta">
      📅 2025-09-20
      | 💬 Under consideration for TPLP journal
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel at understanding natural language but struggle with explicit commonsense reasoning. A recent trend of research suggests that the combination of LLM with robust symbolic reasoning systems can overcome this problem on story-based question answering tasks. In this setting, existing approaches typically depend on human expertise to manually craft the symbolic component. We argue, however, that this component can also be automatically learned from examples. In this work, we introduce LLM2LAS, a hybrid system that effectively combines the natural language understanding capabilities of LLMs, the rule induction power of the Learning from Answer Sets (LAS) system ILASP, and the formal reasoning strengths of Answer Set Programming (ASP). LLMs are used to extract semantic structures from text, which ILASP then transforms into interpretable logic rules. These rules allow an ASP solver to perform precise and consistent reasoning, enabling correct answers to previously unseen questions. Empirical results outline the strengths and weaknesses of our automatic approach for learning and reasoning in a story-based question answering benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16584v1">From Scores to Steps: Diagnosing and Improving LLM Performance in Evidence-Based Medical Calculations</a></div>
    <div class="paper-meta">
      📅 2025-09-20
      | 💬 Equal contribution for the first two authors. To appear as an Oral presentation in the proceedings of the Main Conference on Empirical Methods in Natural Language Processing (EMNLP) 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated promising performance on medical benchmarks; however, their ability to perform medical calculations, a crucial aspect of clinical decision-making, remains underexplored and poorly evaluated. Existing benchmarks often assess only the final answer with a wide numerical tolerance, overlooking systematic reasoning failures and potentially causing serious clinical misjudgments. In this work, we revisit medical calculation evaluation with a stronger focus on clinical trustworthiness. First, we clean and restructure the MedCalc-Bench dataset and propose a new step-by-step evaluation pipeline that independently assesses formula selection, entity extraction, and arithmetic computation. Under this granular framework, the accuracy of GPT-4o drops from 62.7% to 43.6%, revealing errors masked by prior evaluations. Second, we introduce an automatic error analysis framework that generates structured attribution for each failure mode. Human evaluation confirms its alignment with expert judgment, enabling scalable and explainable diagnostics. Finally, we propose a modular agentic pipeline, MedRaC, that combines retrieval-augmented generation and Python-based code execution. Without any fine-tuning, MedRaC improves the accuracy of different LLMs from 16.35% up to 53.19%. Our work highlights the limitations of current benchmark practices and proposes a more clinically faithful methodology. By enabling transparent and transferable reasoning evaluation, we move closer to making LLM-based systems trustworthy for real-world medical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15361v3">Does Reasoning Introduce Bias? A Study of Social Bias Evaluation and Mitigation in LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-20
      | 💬 EMNLP Findings
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have enabled automatic generation of chain-of-thought (CoT) reasoning, leading to strong performance on tasks such as math and code. However, when reasoning steps reflect social stereotypes (e.g., those related to gender, race or age), they can reinforce harmful associations and lead to misleading conclusions. We present the first systematic evaluation of social bias within LLM-generated reasoning, focusing on reasoning language models (e.g., DeepSeek-R1, OpenAI o1) that natively produce reasoning chains as part of their answers. Using the BBQ dataset, we analyze both prediction accuracy and reasoning bias across a broad spectrum of models, including instruction-tuned and CoT-augmented variants of DeepSeek-R1 (8B/32B), ChatGPT, and other open-source LLMs. We quantify how biased reasoning steps correlate with incorrect predictions and often lead to stereotype expression. To mitigate reasoning-induced bias, we propose Answer Distribution as Bias Proxy (ADBP), a lightweight mitigation method that detects bias by tracking how model predictions change across incremental reasoning steps. ADBP outperforms Stereotype-free Reasoning Pattern (SfRP) baseline in most cases, mitigating bias and improving the accuracy of LLM outputs. Evaluation and mitigation code is available at https://github.com/elviswxy/LLM_reasoning_bias.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16564v1">MPCG: Multi-Round Persona-Conditioned Generation for Modeling the Evolution of Misinformation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-20
      | 💬 35 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Misinformation evolves as it spreads, shifting in language, framing, and moral emphasis to adapt to new audiences. However, current misinformation detection approaches implicitly assume that misinformation is static. We introduce MPCG, a multi-round, persona-conditioned framework that simulates how claims are iteratively reinterpreted by agents with distinct ideological perspectives. Our approach uses an uncensored large language model (LLM) to generate persona-specific claims across multiple rounds, conditioning each generation on outputs from the previous round, enabling the study of misinformation evolution. We evaluate the generated claims through human and LLM-based annotations, cognitive effort metrics (readability, perplexity), emotion evocation metrics (sentiment analysis, morality), clustering, feasibility, and downstream classification. Results show strong agreement between human and GPT-4o-mini annotations, with higher divergence in fluency judgments. Generated claims require greater cognitive effort than the original claims and consistently reflect persona-aligned emotional and moral framing. Clustering and cosine similarity analyses confirm semantic drift across rounds while preserving topical coherence. Feasibility results show a 77% feasibility rate, confirming suitability for downstream tasks. Classification results reveal that commonly used misinformation detectors experience macro-F1 performance drops of up to 49.7%. The code is available at https://github.com/bcjr1997/MPCG
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08846v3">Steering Towards Fairness: Mitigating Political Bias in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-20
      | 💬 Accepted at CASE@RANLP2025
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have enabled their widespread use across diverse real-world applications. However, concerns remain about their tendency to encode and reproduce ideological biases along political and economic dimensions. In this paper, we employ a framework for probing and mitigating such biases in decoder-based LLMs through analysis of internal model representations. Grounded in the Political Compass Test (PCT), this method uses contrastive pairs to extract and compare hidden layer activations from models like Mistral and DeepSeek. We introduce a comprehensive activation extraction pipeline capable of layer-wise analysis across multiple ideological axes, revealing meaningful disparities linked to political framing. Our results show that decoder LLMs systematically encode representational bias across layers, which can be leveraged for effective steering vector-based mitigation. This work provides new insights into how political bias is encoded in LLMs and offers a principled approach to debiasing beyond surface-level output interventions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14803v2">OnlineMate: An LLM-Based Multi-Agent Companion System for Cognitive Support in Online Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-20
      | 💬 work in progress
    </div>
    <details class="paper-abstract">
      In online learning environments, students often lack personalized peer interactions, which play a crucial role in supporting cognitive development and learning engagement. Although previous studies have utilized large language models (LLMs) to simulate interactive dynamic learning environments for students, these interactions remain limited to conversational exchanges, lacking insights and adaptations to the learners' individualized learning and cognitive states. As a result, students' interest in discussions with AI learning companions is low, and they struggle to gain inspiration from such interactions. To address this challenge, we propose OnlineMate, a multi-agent learning companion system driven by LLMs that integrates the Theory of Mind (ToM). OnlineMate is capable of simulating peer-like agent roles, adapting to learners' cognitive states during collaborative discussions, and inferring their psychological states, such as misunderstandings, confusion, or motivation. By incorporating Theory of Mind capabilities, the system can dynamically adjust its interaction strategies to support the development of higher-order thinking and cognition. Experimental results in simulated learning scenarios demonstrate that OnlineMate effectively fosters deep learning and discussions while enhancing cognitive engagement in online educational settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16543v1">ChemOrch: Empowering LLMs with Chemical Intelligence via Synthetic Instructions</a></div>
    <div class="paper-meta">
      📅 2025-09-20
    </div>
    <details class="paper-abstract">
      Empowering large language models (LLMs) with chemical intelligence remains a challenge due to the scarcity of high-quality, domain-specific instruction-response datasets and the misalignment of existing synthetic data generation pipelines with the inherently hierarchical and rule-governed structure of chemical information. To address this, we propose ChemOrch, a framework that synthesizes chemically grounded instruction-response pairs through a two-stage process: task-controlled instruction generation and tool-aware response construction. ChemOrch enables controllable diversity and levels of difficulty for the generated tasks, and ensures response precision through tool planning and distillation, and tool-based self-repair mechanisms. The effectiveness of ChemOrch is evaluated based on: 1) the high quality of generated instruction data, demonstrating superior diversity and strong alignment with chemical constraints; 2) the reliable generation of evaluation tasks that more effectively reveal LLM weaknesses in chemistry; and 3) the significant improvement of LLM chemistry capabilities when the generated instruction data are used for fine-tuning. Our work thus represents a critical step toward scalable and verifiable chemical intelligence in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03945v2">Function-based Labels for Complementary Recommendation: Definition, Annotation, and LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-09-20
    </div>
    <details class="paper-abstract">
      Complementary recommendations enhance the user experience by suggesting items that are frequently purchased together while serving different functions from the query item. Inferring or evaluating whether two items have a complementary relationship requires complementary relationship labels; however, defining these labels is challenging because of the inherent ambiguity of such relationships. Complementary labels based on user historical behavior logs attempt to capture these relationships, but often produce inconsistent and unreliable results. Recent efforts have introduced large language models (LLMs) to infer these relationships. However, these approaches provide a binary classification without a nuanced understanding of complementary relationships. In this study, we address these challenges by introducing Function-Based Labels (FBLs), a novel definition of complementary relationships independent of user purchase logs and the opaque decision processes of LLMs. We constructed a human-annotated FBLs dataset comprising 2,759 item pairs and demonstrated that it covered possible item relationships and minimized ambiguity. We then evaluated whether some machine learning (ML) methods using annotated FBLs could accurately infer labels for unseen item pairs, and whether LLM-generated complementary labels align with human perception. Our results demonstrate that even with limited data, ML models, such as logistic regression and SVM achieve high macro-F1 scores (approximately 0.82). Furthermore, LLMs, such as gpt-4o-mini, demonstrated high consistency (0.989) and classification accuracy (0.849) under the detailed definition of FBLs, indicating their potential as effective annotators that mimic human judgment. Overall, our study presents FBLs as a clear definition of complementary relationships, enabling more accurate inferences and automated labeling of complementary recommendations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16533v1">Challenging the Evaluator: LLM Sycophancy Under User Rebuttal</a></div>
    <div class="paper-meta">
      📅 2025-09-20
      | 💬 Accepted to EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often exhibit sycophancy, distorting responses to align with user beliefs, notably by readily agreeing with user counterarguments. Paradoxically, LLMs are increasingly adopted as successful evaluative agents for tasks such as grading and adjudicating claims. This research investigates that tension: why do LLMs show sycophancy when challenged in subsequent conversational turns, yet perform well when evaluating conflicting arguments presented simultaneously? We empirically tested these contrasting scenarios by varying key interaction patterns. We find that state-of-the-art models: (1) are more likely to endorse a user's counterargument when framed as a follow-up from a user, rather than when both responses are presented simultaneously for evaluation; (2) show increased susceptibility to persuasion when the user's rebuttal includes detailed reasoning, even when the conclusion of the reasoning is incorrect; and (3) are more readily swayed by casually phrased feedback than by formal critiques, even when the casual input lacks justification. Our results highlight the risk of relying on LLMs for judgment tasks without accounting for conversational framing.
    </details>
</div>
