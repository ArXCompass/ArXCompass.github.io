# llm - 2025_06

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
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)
- Part 15

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13975v2">Navigating Rifts in Human-LLM Grounding: Study and Benchmark</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 ACL 2025, 16 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Language models excel at following instructions but often struggle with the collaborative aspects of conversation that humans naturally employ. This limitation in grounding -- the process by which conversation participants establish mutual understanding -- can lead to outcomes ranging from frustrated users to serious consequences in high-stakes scenarios. To systematically study grounding challenges in human-LLM interactions, we analyze logs from three human-assistant datasets: WildChat, MultiWOZ, and Bing Chat. We develop a taxonomy of grounding acts and build models to annotate and forecast grounding behavior. Our findings reveal significant differences in human-human and human-LLM grounding: LLMs were three times less likely to initiate clarification and sixteen times less likely to provide follow-up requests than humans. Additionally, we find that early grounding failures predict later interaction breakdowns. Building on these insights, we introduce Rifts, a benchmark derived from publicly available LLM interaction data containing situations where LLMs fail to initiate grounding. We note that current frontier models perform poorly on Rifts, highlighting the need to reconsider how we train and prompt LLMs for human interaction. To this end, we develop a preliminary intervention aimed at mitigating grounding failures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05873v2">MEXA: Multilingual Evaluation of English-Centric LLMs via Cross-Lingual Alignment</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 ACL Findings 2025
    </div>
    <details class="paper-abstract">
      English-centric large language models (LLMs) often show strong multilingual capabilities. However, their multilingual performance remains unclear and is under-evaluated for many other languages. Most benchmarks for multilinguality focus on classic NLP tasks or cover a minimal number of languages. We introduce MEXA, a method for assessing the multilingual capabilities of pre-trained English-centric LLMs using parallel sentences, which are available for more languages than existing downstream tasks. MEXA leverages that English-centric LLMs use English as a pivot language in their intermediate layers. MEXA computes the alignment between English and non-English languages using parallel sentences to evaluate the transfer of language understanding from English to other languages. This alignment can be used to estimate model performance in different languages. We conduct controlled experiments using various parallel datasets (FLORES-200 and Bible), models (Llama family, Gemma family, Mistral, and OLMo), and established downstream tasks (Belebele, m-MMLU, and m-ARC). We explore different methods to compute embeddings in decoder-only models. Our results show that MEXA, in its default settings, achieves an average Pearson correlation of 0.90 between its predicted scores and actual task performance across languages. This suggests that MEXA is a reliable method for estimating the multilingual capabilities of English-centric LLMs, providing a clearer understanding of their multilingual potential and the inner workings of LLMs. Leaderboard: https://cis-lmu-mexa.hf.space, Code: https://github.com/cisnlp/MEXA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.09689v2">Probing LLM Hallucination from Within: Perturbation-Driven Approach via Internal Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 22 pages, 15 figures
    </div>
    <details class="paper-abstract">
      LLM hallucination, where unfaithful text is generated, presents a critical challenge for LLMs' practical applications. Current detection methods often resort to external knowledge, LLM fine-tuning, or supervised training with large hallucination-labeled datasets. Moreover, these approaches do not distinguish between different types of hallucinations, which is crucial for enhancing detection performance. To address such limitations, we introduce hallucination probing, a new task that classifies LLM-generated text into three categories: aligned, misaligned, and fabricated. Driven by our novel discovery that perturbing key entities in prompts affects LLM's generation of these three types of text differently, we propose SHINE, a novel hallucination probing method that does not require external knowledge, supervised training, or LLM fine-tuning. SHINE is effective in hallucination probing across three modern LLMs, and achieves state-of-the-art performance in hallucination detection, outperforming seven competing methods across four datasets and four LLMs, underscoring the importance of probing for accurate detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.12728v3">Relevant or Random: Can LLMs Truly Perform Analogical Reasoning?</a></div>
    <div class="paper-meta">
      📅 2025-06-01
    </div>
    <details class="paper-abstract">
      Analogical reasoning is a unique ability of humans to address unfamiliar challenges by transferring strategies from relevant past experiences. One key finding in psychology is that compared with irrelevant past experiences, recalling relevant ones can help humans better handle new tasks. Coincidentally, the NLP community has also recently found that self-generating relevant examples in the context can help large language models (LLMs) better solve a given problem than hand-crafted prompts. However, it is yet not clear whether relevance is the key factor eliciting such capability, i.e., can LLMs benefit more from self-generated relevant examples than irrelevant ones? In this work, we systematically explore whether LLMs can truly perform analogical reasoning on a diverse set of reasoning tasks. With extensive experiments and analysis, we show that self-generated random examples can surprisingly achieve comparable or even better performance on certain tasks, e.g., 4% performance boost on GSM8K with random biological examples. We find that the accuracy of self-generated examples is the key factor and subsequently design two novel methods with improved performance and significantly reduced inference costs. Overall, we aim to advance a deeper understanding of LLM analogical reasoning and hope this work stimulates further research in the design of self-generated contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11191v2">Primus: A Pioneering Collection of Open-Source Datasets for Cybersecurity LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-06-01
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable advancements in specialized fields such as finance, law, and medicine. However, in cybersecurity, we have noticed a lack of open-source datasets, with a particular lack of high-quality cybersecurity pretraining corpora, even though much research indicates that LLMs acquire their knowledge during pretraining. To address this, we present a comprehensive suite of datasets covering all major training stages, including pretraining, instruction fine-tuning, and reasoning distillation with cybersecurity-specific self-reflection data. Extensive ablation studies demonstrate their effectiveness on public cybersecurity benchmarks. In particular, continual pre-training on our dataset yields a 15.88% improvement in the aggregate score, while reasoning distillation leads to a 10% gain in security certification (CISSP). We will release all datasets and trained cybersecurity LLMs under the ODC-BY and MIT licenses to encourage further research in the community. For access to all datasets and model weights, please refer to https://huggingface.co/collections/trendmicro-ailab/primus-67b1fd27052b802b4af9d243.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18458v3">A Survey of LLM $\times$ DATA</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 Please refer to the paper list at: https://github.com/weAIDB/awesome-data-llm
    </div>
    <details class="paper-abstract">
      The integration of large language model (LLM) and data management (DATA) is rapidly redefining both domains. In this survey, we comprehensively review the bidirectional relationships. On the one hand, DATA4LLM, spanning large-scale data processing, storage, and serving, feeds LLMs with high quality, diversity, and timeliness of data required for stages like pre-training, post-training, retrieval-augmented generation, and agentic workflows: (i) Data processing for LLMs includes scalable acquisition, deduplication, filtering, selection, domain mixing, and synthetic augmentation; (ii) Data Storage for LLMs focuses on efficient data and model formats, distributed and heterogeneous storage hierarchies, KV-cache management, and fault-tolerant checkpointing; (iii) Data serving for LLMs tackles challenges in RAG (e.g., knowledge post-processing), LLM inference (e.g., prompt compression, data provenance), and training strategies (e.g., data packing and shuffling). On the other hand, in LLM4DATA, LLMs are emerging as general-purpose engines for data management. We review recent advances in (i) data manipulation, including automatic data cleaning, integration, discovery; (ii) data analysis, covering reasoning over structured, semi-structured, and unstructured data, and (iii) system optimization (e.g., configuration tuning, query rewriting, anomaly diagnosis), powered by LLM techniques like retrieval-augmented prompting, task-specialized fine-tuning, and multi-agent collaboration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20238v2">FINEREASON: Evaluating and Improving LLMs' Deliberate Reasoning through Reflective Puzzle Solving</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 Accepted to ACL2025 Main
    </div>
    <details class="paper-abstract">
      Many challenging reasoning tasks require not just rapid, intuitive responses, but a more deliberate, multi-step approach. Recent progress in large language models (LLMs) highlights an important shift from the "System 1" way of quick reactions to the "System 2" style of reflection-and-correction problem solving. However, current benchmarks heavily rely on the final-answer accuracy, leaving much of a model's intermediate reasoning steps unexamined. This fails to assess the model's ability to reflect and rectify mistakes within the reasoning process. To bridge this gap, we introduce FINEREASON, a logic-puzzle benchmark for fine-grained evaluation of LLMs' reasoning capabilities. Each puzzle can be decomposed into atomic steps, making it ideal for rigorous validation of intermediate correctness. Building on this, we introduce two tasks: state checking, and state transition, for a comprehensive evaluation of how models assess the current situation and plan the next move. To support broader research, we also provide a puzzle training set aimed at enhancing performance on general mathematical tasks. We show that models trained on our state checking and transition data demonstrate gains in math reasoning by up to 5.1% on GSM8K.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.10424v3">LLM-as-an-Interviewer: Beyond Static Testing Through Dynamic LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-06-01
    </div>
    <details class="paper-abstract">
      We introduce LLM-as-an-Interviewer, a novel paradigm for evaluating large language models (LLMs). This approach leverages multi-turn interactions where the LLM interviewer actively provides feedback on responses and poses follow-up questions to the evaluated LLM. At the start of the interview, the LLM interviewer dynamically modifies datasets to generate initial questions, mitigating data contamination. We apply the LLM-as-an-Interviewer framework to evaluate six models on the MATH and DepthQA tasks. Our results show that the framework effectively provides insights into LLM performance, including the quality of initial responses, adaptability to feedback, and ability to address follow-up queries like clarification or additional knowledge requests. The framework also addresses key limitations of conventional methods like LLM-as-a-Judge, including verbosity bias and inconsistency across runs. Finally, we propose the Interview Report, which aggregates insights from the interview process, providing examples and a comprehensive analysis of the LLM's strengths and weaknesses. This report offers a detailed snapshot of the model's real-world applicability. The code for our framework is publicly available at https://github.com/interview-eval/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19433v2">Can Compressed LLMs Truly Act? An Empirical Evaluation of Agentic Capabilities in LLM Compression</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 Accepted by ICML2025 as Poster
    </div>
    <details class="paper-abstract">
      Post-training compression reduces the computational and memory costs of large language models (LLMs), enabling resource-efficient deployment. However, existing compression benchmarks only focus on language modeling (e.g., perplexity) and natural language understanding tasks (e.g., GLUE accuracy), ignoring the agentic capabilities - workflow, tool use/function call, long-context understanding and real-world application. We introduce the Agent Compression Benchmark (ACBench), the first comprehensive benchmark for evaluating how compression impacts LLMs' agentic abilities. ACBench spans (1) 12 tasks across 4 capabilities (e.g., WorfBench for workflow generation, Needle-in-Haystack for long-context retrieval), (2) quantization (GPTQ, AWQ) and pruning (Wanda, SparseGPT), and (3) 15 models, including small (Gemma-2B), standard (Qwen2.5 7B-32B), and distilled reasoning LLMs (DeepSeek-R1-Distill). Our experiments reveal compression tradeoffs: 4-bit quantization preserves workflow generation and tool use (1%-3% drop) but degrades real-world application accuracy by 10%-15%. We introduce ERank, Top-k Ranking Correlation and Energy to systematize analysis. ACBench provides actionable insights for optimizing LLM compression in agentic scenarios. The code can be found in https://github.com/pprp/ACBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16978v2">HyGenar: An LLM-Driven Hybrid Genetic Algorithm for Few-Shot Grammar Generation</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 Accepted to ACL 2025 Findings. Code available at https://github.com/RutaTang/HyGenar
    </div>
    <details class="paper-abstract">
      Grammar plays a critical role in natural language processing and text/code generation by enabling the definition of syntax, the creation of parsers, and guiding structured outputs. Although large language models (LLMs) demonstrate impressive capabilities across domains, their ability to infer and generate grammars has not yet been thoroughly explored. In this paper, we aim to study and improve the ability of LLMs for few-shot grammar generation, where grammars are inferred from sets of a small number of positive and negative examples and generated in Backus-Naur Form. To explore this, we introduced a novel dataset comprising 540 structured grammar generation challenges, devised 6 metrics, and evaluated 8 various LLMs against it. Our findings reveal that existing LLMs perform sub-optimally in grammar generation. To address this, we propose an LLM-driven hybrid genetic algorithm, namely HyGenar, to optimize grammar generation. HyGenar achieves substantial improvements in both the syntactic and semantic correctness of generated grammars across LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02450v2">Measuring What Makes You Unique: Difference-Aware User Modeling for Enhancing LLM Personalization</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 2025 ACL Findings
    </div>
    <details class="paper-abstract">
      Personalizing Large Language Models (LLMs) has become a critical step in facilitating their widespread application to enhance individual life experiences. In pursuit of personalization, distilling key preference information from an individual's historical data as instructional preference context to customize LLM generation has emerged as a promising direction. However, these methods face a fundamental limitation by overlooking the inter-user comparative analysis, which is essential for identifying the inter-user differences that truly shape preferences. To address this limitation, we propose Difference-aware Personalization Learning (DPL), a novel approach that emphasizes extracting inter-user differences to enhance LLM personalization. DPL strategically selects representative users for comparison and establishes a structured standard to extract meaningful, task-relevant differences for customizing LLM generation. Extensive experiments on real-world datasets demonstrate that DPL significantly enhances LLM personalization. We release our code at https://github.com/SnowCharmQ/DPL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18530v2">IMPROVE: Iterative Model Pipeline Refinement and Optimization Leveraging LLM Experts</a></div>
    <div class="paper-meta">
      📅 2025-06-01
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents have emerged as a promising solution to automate the workflow of machine learning, but most existing methods share a common limitation: they attempt to optimize entire pipelines in a single step before evaluation, making it difficult to attribute improvements to specific changes. This lack of granularity leads to unstable optimization and slower convergence, limiting their effectiveness. To address this, we introduce Iterative Refinement, a novel strategy for LLM-driven ML pipeline design inspired by how human ML experts iteratively refine models, focusing on one component at a time rather than making sweeping changes all at once. By systematically updating individual components based on real training feedback, Iterative Refinement improves overall model performance. We also provide some theoretical edvience of the superior properties of this Iterative Refinement. Further, we implement this strategy in IMPROVE, an end-to-end LLM agent framework for automating and optimizing object classification pipelines. Through extensive evaluations across datasets of varying sizes and domains, we demonstrate that Iterative Refinement enables IMPROVE to consistently achieve better performance over existing zero-shot LLM-based approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13602v2">GAMEBoT: Transparent Assessment of LLM Reasoning in Games</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 9 pages, ACL 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed in real-world applications that demand complex reasoning. To track progress, robust benchmarks are required to evaluate their capabilities beyond superficial pattern recognition. However, current LLM reasoning benchmarks often face challenges such as insufficient interpretability, performance saturation or data contamination. To address these challenges, we introduce GAMEBoT, a gaming arena designed for rigorous and transparent assessment of LLM reasoning capabilities. GAMEBoT decomposes complex reasoning in games into predefined modular subproblems. This decomposition allows us to design a suite of Chain-of-Thought (CoT) prompts that leverage domain knowledge to guide LLMs in addressing these subproblems before action selection. Furthermore, we develop a suite of rule-based algorithms to generate ground truth for these subproblems, enabling rigorous validation of the LLMs' intermediate reasoning steps. This approach facilitates evaluation of both the quality of final actions and the accuracy of the underlying reasoning process. GAMEBoT also naturally alleviates the risk of data contamination through dynamic games and head-to-head LLM competitions. We benchmark 17 prominent LLMs across eight games, encompassing various strategic abilities and game characteristics. Our results suggest that GAMEBoT presents a significant challenge, even when LLMs are provided with detailed CoT prompts. Project page: https://visual-ai.github.io/gamebot
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.05806v4">CKnowEdit: A New Chinese Knowledge Editing Dataset for Linguistics, Facts, and Logic Error Correction in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 ACL 2025; project website is available at https://zjunlp.github.io/project/CKnowEdit code and dataset are available at https://github.com/zjunlp/EasyEdit
    </div>
    <details class="paper-abstract">
      Chinese, as a linguistic system rich in depth and complexity, is characterized by distinctive elements such as ancient poetry, proverbs, idioms, and other cultural constructs. However, current Large Language Models (LLMs) face limitations in these specialized domains, highlighting the need for the development of comprehensive datasets that can assess, continuously update, and progressively improve these culturally-grounded linguistic competencies through targeted training optimizations. To address this gap, we introduce CKnowEdit, the first-ever Chinese knowledge editing dataset designed to correct linguistic, factual, and logical errors in LLMs. We collect seven types of knowledge from a wide range of sources, including classical texts, idioms, and content from Baidu Tieba Ruozhiba, taking into account the unique polyphony, antithesis, and logical structures inherent in the Chinese language. By analyzing this dataset, we highlight the challenges current LLMs face in mastering Chinese. Furthermore, our evaluation of state-of-the-art knowledge editing techniques reveals opportunities to advance the correction of Chinese knowledge. Code and dataset are available at https://github.com/zjunlp/EasyEdit.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11196v2">How Do LLMs Acquire New Knowledge? A Knowledge Circuits Perspective on Continual Pre-Training</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 ACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Despite exceptional capabilities in knowledge-intensive tasks, Large Language Models (LLMs) face a critical gap in understanding how they internalize new knowledge, particularly how to structurally embed acquired knowledge in their neural computations. We address this issue through the lens of knowledge circuit evolution, identifying computational subgraphs that facilitate knowledge storage and processing. Our systematic analysis of circuit evolution throughout continual pre-training reveals several key findings: (1) the acquisition of new knowledge is influenced by its relevance to pre-existing knowledge; (2) the evolution of knowledge circuits exhibits a distinct phase shift from formation to optimization; (3) the evolution of knowledge circuits follows a deep-to-shallow pattern. These insights not only advance our theoretical understanding of the mechanisms of new knowledge acquisition in LLMs, but also provide potential implications for improving continual pre-training strategies to enhance model performance. Code and data will be available at https://github.com/zjunlp/DynamicKnowledgeCircuits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11674v2">LLM-Mixer: Multiscale Mixing in LLMs for Time Series Forecasting</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 Time series forecasting using LLMs
    </div>
    <details class="paper-abstract">
      Time series forecasting remains a challenging task, particularly in the context of complex multiscale temporal patterns. This study presents LLM-Mixer, a framework that improves forecasting accuracy through the combination of multiscale time-series decomposition with pre-trained LLMs (Large Language Models). LLM-Mixer captures both short-term fluctuations and long-term trends by decomposing the data into multiple temporal resolutions and processing them with a frozen LLM, guided by a textual prompt specifically designed for time-series data. Extensive experiments conducted on multivariate and univariate datasets demonstrate that LLM-Mixer achieves competitive performance, outperforming recent state-of-the-art models across various forecasting horizons. This work highlights the potential of combining multiscale analysis and LLMs for effective and scalable time-series forecasting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13147v4">Are Your LLMs Capable of Stable Reasoning?</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 ACL 2025 Camera
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has shown remarkable progress in complex reasoning tasks. However, a significant disparity exists between benchmark performances and real-world applications. We attribute this gap primarily to current evaluation protocols and metrics, which inadequately capture the full spectrum of LLM capabilities, especially in complex reasoning tasks where both accuracy and consistency are essential. In this paper, we introduce G-Pass@$k$, a novel evaluation metric that continuously assesses model performance across multiple sampling attempts, quantifying both the model's performance potential and its stability. Through extensive experiments on various public and newly constructed benchmarks, we employ G-Pass@$k$ in conjunction with state-of-the-art large language models to provide comprehensive insights into their potential capabilities and operational consistency. Our findings reveal a significant opportunity to enhance the realistic reasoning abilities of LLMs, underscoring the necessity for more robust evaluation metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12212v3">Data Whisperer: Efficient Data Selection for Task-Specific LLM Fine-Tuning via Few-Shot In-Context Learning</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 Accepted by ACL 2025 main, 18 pages, 8 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) on task-specific data is essential for their effective deployment. As dataset sizes grow, efficiently selecting optimal subsets for training becomes crucial to balancing performance and computational costs. Traditional data selection methods often require fine-tuning a scoring model on the target dataset, which is time-consuming and resource-intensive, or rely on heuristics that fail to fully leverage the model's predictive capabilities. To address these challenges, we propose Data Whisperer, an efficient, training-free, attention-based method that leverages few-shot in-context learning with the model to be fine-tuned. Comprehensive evaluations were conducted on both raw and synthetic datasets across diverse tasks and models. Notably, Data Whisperer achieves superior performance compared to the full GSM8K dataset on the Llama-3-8B-Instruct model, using just 10% of the data, and outperforms existing methods with a 3.1-point improvement and a 7.4$\times$ speedup. The code is available at https://github.com/gszfwsb/Data-Whisperer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10084v2">Why Prompt Design Matters and Works: A Complexity Analysis of Prompt Search Space in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 ACL 2025 main conference
    </div>
    <details class="paper-abstract">
      Despite the remarkable successes of large language models (LLMs), the underlying Transformer architecture has inherent limitations in handling complex reasoning tasks. Chain-of-thought (CoT) prompting has emerged as a practical workaround, but most CoT-based methods rely on a single, generic prompt such as "think step by step", with no task-specific adaptation. These approaches expect the model to discover an effective reasoning path on its own, forcing it to search through a vast prompt space. In contrast, several studies have explored task-specific prompt designs to boost performance. However, these designs are typically developed through trial and error, lacking theoretical grounding. As a result, prompt engineering remains largely ad hoc and unguided. In this paper, we provide a theoretical framework that explains why some prompts succeed while others fail. We show that prompts function as selectors, extracting task-relevant information from the model's full hidden state during CoT reasoning. Each prompt defines a unique trajectory through the answer space, and the choice of trajectory is crucial for task performance and future navigation within the space. We analyze the complexity of finding optimal prompts and characterize the size of the prompt space for a given task. Our theory reveals principles behind effective prompt design and shows that naive CoT-using self-guided prompts like "think step by step"-can severely hinder performance. Through experiments, we show that optimal prompt search can lead to more than a 50% improvement on reasoning tasks, providing a theoretical foundation for prompt engineering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11368v2">LLMs can Perform Multi-Dimensional Analytic Writing Assessments: A Case Study of L2 Graduate-Level Academic English Writing</a></div>
    <div class="paper-meta">
      📅 2025-06-01
      | 💬 ACL 2025
    </div>
    <details class="paper-abstract">
      The paper explores the performance of LLMs in the context of multi-dimensional analytic writing assessments, i.e. their ability to provide both scores and comments based on multiple assessment criteria. Using a corpus of literature reviews written by L2 graduate students and assessed by human experts against 9 analytic criteria, we prompt several popular LLMs to perform the same task under various conditions. To evaluate the quality of feedback comments, we apply a novel feedback comment quality evaluation framework. This framework is interpretable, cost-efficient, scalable, and reproducible, compared to existing methods that rely on manual judgments. We find that LLMs can generate reasonably good and generally reliable multi-dimensional analytic assessments. We release our corpus and code for reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.03219v2">Content Moderation by LLM: From Accuracy to Legitimacy</a></div>
    <div class="paper-meta">
      📅 2025-06-01
    </div>
    <details class="paper-abstract">
      One trending application of LLM (large language model) is to use it for content moderation in online platforms. Most current studies on this application have focused on the metric of accuracy -- the extent to which LLMs make correct decisions about content. This article argues that accuracy is insufficient and misleading because it fails to grasp the distinction between easy cases and hard cases, as well as the inevitable trade-offs in achieving higher accuracy. Closer examination reveals that content moderation is a constitutive part of platform governance, the key of which is to gain and enhance legitimacy. Instead of making moderation decisions correct, the chief goal of LLMs is to make them legitimate. In this regard, this article proposes a paradigm shift from the single benchmark of accuracy towards a legitimacy-based framework for evaluating the performance of LLM moderators. The framework suggests that for easy cases, the key is to ensure accuracy, speed, and transparency, while for hard cases, what matters is reasoned justification and user participation. Examined under this framework, LLMs' real potential in moderation is not accuracy improvement. Rather, LLMs can better contribute in four other aspects: to conduct screening of hard cases from easy cases, to provide quality explanations for moderation decisions, to assist human reviewers in getting more contextual information, and to facilitate user participation in a more interactive way. To realize these contributions, this article proposes a workflow for incorporating LLMs into the content moderation system. Using normative theories from law and social sciences to critically assess the new technological application, this article seeks to redefine LLMs' role in content moderation and redirect relevant research in this field.
    </details>
</div>
