# llm - 2023_10

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1
- [Part 2](papers_2.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.19462v2">Constituency Parsing using LLMs</a></div>
    <div class="paper-meta">
      📅 2023-10-31
    </div>
    <details class="paper-abstract">
      Constituency parsing is a fundamental yet unsolved natural language processing task. In this paper, we explore the potential of recent large language models (LLMs) that have exhibited remarkable performance across various domains and tasks to tackle this task. We employ three linearization strategies to transform output trees into symbol sequences, such that LLMs can solve constituency parsing by generating linearized trees. We conduct experiments using a diverse range of LLMs, including ChatGPT, GPT-4, OPT, LLaMA, and Alpaca, comparing their performance against the state-of-the-art constituency parsers. Our experiments encompass zero-shot, few-shot, and full-training learning settings, and we evaluate the models on one in-domain and five out-of-domain test datasets. Our findings reveal insights into LLMs' performance, generalization abilities, and challenges in constituency parsing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.20170v1">DIVKNOWQA: Assessing the Reasoning Ability of LLMs via Open-Domain Question Answering over Knowledge Base and Text</a></div>
    <div class="paper-meta">
      📅 2023-10-31
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have exhibited impressive generation capabilities, but they suffer from hallucinations when solely relying on their internal knowledge, especially when answering questions that require less commonly known information. Retrieval-augmented LLMs have emerged as a potential solution to ground LLMs in external knowledge. Nonetheless, recent approaches have primarily emphasized retrieval from unstructured text corpora, owing to its seamless integration into prompts. When using structured data such as knowledge graphs, most methods simplify it into natural text, neglecting the underlying structures. Moreover, a significant gap in the current landscape is the absence of a realistic benchmark for evaluating the effectiveness of grounding LLMs on heterogeneous knowledge sources (e.g., knowledge base and text). To fill this gap, we have curated a comprehensive dataset that poses two unique challenges: (1) Two-hop multi-source questions that require retrieving information from both open-domain structured and unstructured knowledge sources; retrieving information from structured knowledge sources is a critical component in correctly answering the questions. (2) The generation of symbolic queries (e.g., SPARQL for Wikidata) is a key requirement, which adds another layer of challenge. Our dataset is created using a combination of automatic generation through predefined reasoning chains and human annotation. We also introduce a novel approach that leverages multiple retrieval tools, including text passage retrieval and symbolic language-assisted retrieval. Our model outperforms previous approaches by a significant margin, demonstrating its effectiveness in addressing the above-mentioned reasoning challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.20150v1">Unlearn What You Want to Forget: Efficient Unlearning for LLMs</a></div>
    <div class="paper-meta">
      📅 2023-10-31
      | 💬 EMNLP 2023
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved significant progress from pre-training on and memorizing a wide range of textual data, however, this process might suffer from privacy issues and violations of data protection regulations. As a result, the ability to easily remove data related to individual users from such models while not deteriorating their predictive quality after the removal becomes increasingly important. To address these issues, in this work, we propose an efficient unlearning framework that could efficiently update LLMs without having to retrain the whole model after data removals, by introducing lightweight unlearning layers learned with a selective teacher-student objective into the transformers. In addition, we introduce a fusion mechanism to effectively combine different unlearning layers that learns to forget different sets of data to handle a sequence of forgetting operations. Experiments on classification and generation tasks demonstrate the effectiveness of our proposed methods compared to the state-of-the-art baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.14987v2">Investigating Table-to-Text Generation Capabilities of LLMs in Real-World Information Seeking Scenarios</a></div>
    <div class="paper-meta">
      📅 2023-10-30
      | 💬 Camera-ready version for EMNLP 2023 industry track
    </div>
    <details class="paper-abstract">
      Tabular data is prevalent across various industries, necessitating significant time and effort for users to understand and manipulate for their information-seeking purposes. The advancements in large language models (LLMs) have shown enormous potential to improve user efficiency. However, the adoption of LLMs in real-world applications for table information seeking remains underexplored. In this paper, we investigate the table-to-text capabilities of different LLMs using four datasets within two real-world information seeking scenarios. These include the LogicNLG and our newly-constructed LoTNLG datasets for data insight generation, along with the FeTaQA and our newly-constructed F2WTQ datasets for query-based generation. We structure our investigation around three research questions, evaluating the performance of LLMs in table-to-text generation, automated evaluation, and feedback generation, respectively. Experimental results indicate that the current high-performing LLM, specifically GPT-4, can effectively serve as a table-to-text generator, evaluator, and feedback generator, facilitating users' information seeking purposes in real-world scenarios. However, a significant performance gap still exists between other open-sourced LLMs (e.g., Tulu and LLaMA-2) and GPT-4 models. Our data and code are publicly available at https://github.com/yale-nlp/LLM-T2T.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.18974v1">EtiCor: Corpus for Analyzing LLMs for Etiquettes</a></div>
    <div class="paper-meta">
      📅 2023-10-29
      | 💬 Accepted at EMNLP 2023, Main Conference
    </div>
    <details class="paper-abstract">
      Etiquettes are an essential ingredient of day-to-day interactions among people. Moreover, etiquettes are region-specific, and etiquettes in one region might contradict those in other regions. In this paper, we propose EtiCor, an Etiquettes Corpus, having texts about social norms from five different regions across the globe. The corpus provides a test bed for evaluating LLMs for knowledge and understanding of region-specific etiquettes. Additionally, we propose the task of Etiquette Sensitivity. We experiment with state-of-the-art LLMs (Delphi, Falcon40B, and GPT-3.5). Initial results indicate that LLMs, mostly fail to understand etiquettes from regions from non-Western world.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.17842v3">SPAE: Semantic Pyramid AutoEncoder for Multimodal Generation with Frozen LLMs</a></div>
    <div class="paper-meta">
      📅 2023-10-28
      | 💬 NeurIPS 2023 spotlight
    </div>
    <details class="paper-abstract">
      In this work, we introduce Semantic Pyramid AutoEncoder (SPAE) for enabling frozen LLMs to perform both understanding and generation tasks involving non-linguistic modalities such as images or videos. SPAE converts between raw pixels and interpretable lexical tokens (or words) extracted from the LLM's vocabulary. The resulting tokens capture both the semantic meaning and the fine-grained details needed for visual reconstruction, effectively translating the visual content into a language comprehensible to the LLM, and empowering it to perform a wide array of multimodal tasks. Our approach is validated through in-context learning experiments with frozen PaLM 2 and GPT 3.5 on a diverse set of image understanding and generation tasks. Our method marks the first successful attempt to enable a frozen LLM to generate image content while surpassing state-of-the-art performance in image understanding tasks, under the same setting, by over 25%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.18696v1">Probing LLMs for Joint Encoding of Linguistic Categories</a></div>
    <div class="paper-meta">
      📅 2023-10-28
      | 💬 Accepted in EMNLP Findings 2023
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit impressive performance on a range of NLP tasks, due to the general-purpose linguistic knowledge acquired during pretraining. Existing model interpretability research (Tenney et al., 2019) suggests that a linguistic hierarchy emerges in the LLM layers, with lower layers better suited to solving syntactic tasks and higher layers employed for semantic processing. Yet, little is known about how encodings of different linguistic phenomena interact within the models and to what extent processing of linguistically-related categories relies on the same, shared model representations. In this paper, we propose a framework for testing the joint encoding of linguistic categories in LLMs. Focusing on syntax, we find evidence of joint encoding both at the same (related part-of-speech (POS) classes) and different (POS classes and related syntactic dependency relations) levels of linguistic hierarchy. Our cross-lingual experiments show that the same patterns hold across languages in multilingual LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.13259v2">Knowledge-Driven CoT: Exploring Faithful Reasoning in LLMs for Knowledge-intensive Question Answering</a></div>
    <div class="paper-meta">
      📅 2023-10-28
    </div>
    <details class="paper-abstract">
      Equipped with Chain-of-Thought (CoT), Large language models (LLMs) have shown impressive reasoning ability in various downstream tasks. Even so, suffering from hallucinations and the inability to access external knowledge, LLMs often come with incorrect or unfaithful intermediate reasoning steps, especially in the context of answering knowledge-intensive tasks such as KBQA. To alleviate this issue, we propose a framework called Knowledge-Driven Chain-of-Thought (KD-CoT) to verify and modify reasoning traces in CoT via interaction with external knowledge, and thus overcome the hallucinations and error propagation. Concretely, we formulate the CoT rationale process of LLMs into a structured multi-round QA format. In each round, LLMs interact with a QA system that retrieves external knowledge and produce faithful reasoning traces based on retrieved precise answers. The structured CoT reasoning of LLMs is facilitated by our developed KBQA CoT collection, which serves as in-context learning demonstrations and can also be utilized as feedback augmentation to train a robust retriever. Extensive experiments on WebQSP and ComplexWebQuestion datasets demonstrate the effectiveness of proposed KD-CoT in task-solving reasoning generation, which outperforms the vanilla CoT ICL with an absolute success rate of 8.0% and 5.1%. Furthermore, our proposed feedback-augmented retriever outperforms the state-of-the-art baselines for retrieving knowledge, achieving significant improvement in Hit and recall performance. Our code and data are released on https://github.com/AdelWang/KD-CoT/tree/main.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.18018v1">NLP Evaluation in trouble: On the Need to Measure LLM Data Contamination for each Benchmark</a></div>
    <div class="paper-meta">
      📅 2023-10-27
      | 💬 Accepted at EMNLP2024-Findings
    </div>
    <details class="paper-abstract">
      In this position paper, we argue that the classical evaluation on Natural Language Processing (NLP) tasks using annotated benchmarks is in trouble. The worst kind of data contamination happens when a Large Language Model (LLM) is trained on the test split of a benchmark, and then evaluated in the same benchmark. The extent of the problem is unknown, as it is not straightforward to measure. Contamination causes an overestimation of the performance of a contaminated model in a target benchmark and associated task with respect to their non-contaminated counterparts. The consequences can be very harmful, with wrong scientific conclusions being published while other correct ones are discarded. This position paper defines different levels of data contamination and argues for a community effort, including the development of automatic and semi-automatic measures to detect when data from a benchmark was exposed to a model, and suggestions for flagging papers with conclusions that are compromised by data contamination.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.04618v2">Revisiting Out-of-distribution Robustness in NLP: Benchmark, Analysis, and LLMs Evaluations</a></div>
    <div class="paper-meta">
      📅 2023-10-26
      | 💬 Accepted to NeurIPS 2023 Dataset and Benchmark Track. Code is available at \url{https://github.com/lifan-yuan/OOD_NLP}
    </div>
    <details class="paper-abstract">
      This paper reexamines the research on out-of-distribution (OOD) robustness in the field of NLP. We find that the distribution shift settings in previous studies commonly lack adequate challenges, hindering the accurate evaluation of OOD robustness. To address these issues, we propose a benchmark construction protocol that ensures clear differentiation and challenging distribution shifts. Then we introduce BOSS, a Benchmark suite for Out-of-distribution robustneSS evaluation covering 5 tasks and 20 datasets. Based on BOSS, we conduct a series of experiments on pre-trained language models for analysis and evaluation of OOD robustness. First, for vanilla fine-tuning, we examine the relationship between in-distribution (ID) and OOD performance. We identify three typical types that unveil the inner learning mechanism, which could potentially facilitate the forecasting of OOD robustness, correlating with the advancements on ID datasets. Then, we evaluate 5 classic methods on BOSS and find that, despite exhibiting some effectiveness in specific cases, they do not offer significant improvement compared to vanilla fine-tuning. Further, we evaluate 5 LLMs with various adaptation paradigms and find that when sufficient ID data is available, fine-tuning domain-specific models outperform LLMs on ID examples significantly. However, in the case of OOD instances, prioritizing LLMs with in-context learning yields better results. We identify that both fine-tuned small models and LLMs face challenges in effectively addressing downstream tasks. The code is public at \url{https://github.com/lifan-yuan/OOD_NLP}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.17157v1">Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time</a></div>
    <div class="paper-meta">
      📅 2023-10-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) with hundreds of billions of parameters have sparked a new wave of exciting AI applications. However, they are computationally expensive at inference time. Sparsity is a natural approach to reduce this cost, but existing methods either require costly retraining, have to forgo LLM's in-context learning ability, or do not yield wall-clock time speedup on modern hardware. We hypothesize that contextual sparsity, which are small, input-dependent sets of attention heads and MLP parameters that yield approximately the same output as the dense model for a given input, can address these issues. We show that contextual sparsity exists, that it can be accurately predicted, and that we can exploit it to speed up LLM inference in wall-clock time without compromising LLM's quality or in-context learning ability. Based on these insights, we propose DejaVu, a system that uses a low-cost algorithm to predict contextual sparsity on the fly given inputs to each layer, along with an asynchronous and hardware-aware implementation that speeds up LLM inference. We validate that DejaVu can reduce the inference latency of OPT-175B by over 2X compared to the state-of-the-art FasterTransformer, and over 6X compared to the widely used Hugging Face implementation, without compromising model quality. The code is available at https://github.com/FMInference/DejaVu.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.03688v2">AgentBench: Evaluating LLMs as Agents</a></div>
    <div class="paper-meta">
      📅 2023-10-25
      | 💬 55 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are becoming increasingly smart and autonomous, targeting real-world pragmatic missions beyond traditional NLP tasks. As a result, there has been an urgent need to evaluate LLMs as agents on challenging tasks in interactive environments. We present AgentBench, a multi-dimensional evolving benchmark that currently consists of 8 distinct environments to assess LLM-as-Agent's reasoning and decision-making abilities in a multi-turn open-ended generation setting. Our extensive test over 27 API-based and open-sourced (OSS) LLMs shows that, while top commercial LLMs present a strong ability of acting as agents in complex environments, there is a significant disparity in performance between them and OSS competitors. We identify the typical reasons of failures in environments and LLMs, showing that poor long-term reasoning, decision-making, and instruction following abilities are the main obstacles for developing usable LLM agents. Training on code and high quality multi-turn alignment data could improve agent performance. Datasets, environments, and an integrated evaluation package for AgentBench are released at \url{https://github.com/THUDM/AgentBench}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2304.08244v2">API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs</a></div>
    <div class="paper-meta">
      📅 2023-10-25
      | 💬 EMNLP 2023
    </div>
    <details class="paper-abstract">
      Recent research has demonstrated that Large Language Models (LLMs) can enhance their capabilities by utilizing external tools. However, three pivotal questions remain unanswered: (1) How effective are current LLMs in utilizing tools? (2) How can we enhance LLMs' ability to utilize tools? (3) What obstacles need to be overcome to leverage tools? To address these questions, we introduce API-Bank, a groundbreaking benchmark, specifically designed for tool-augmented LLMs. For the first question, we develop a runnable evaluation system consisting of 73 API tools. We annotate 314 tool-use dialogues with 753 API calls to assess the existing LLMs' capabilities in planning, retrieving, and calling APIs. For the second question, we construct a comprehensive training set containing 1,888 tool-use dialogues from 2,138 APIs spanning 1,000 distinct domains. Using this dataset, we train Lynx, a tool-augmented LLM initialized from Alpaca. Experimental results demonstrate that GPT-3.5 exhibits improved tool utilization compared to GPT-3, while GPT-4 excels in planning. However, there is still significant potential for further improvement. Moreover, Lynx surpasses Alpaca's tool utilization performance by more than 26 pts and approaches the effectiveness of GPT-3.5. Through error analysis, we highlight the key challenges for future research in this field to answer the third question.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.16361v1">InstructPTS: Instruction-Tuning LLMs for Product Title Summarization</a></div>
    <div class="paper-meta">
      📅 2023-10-25
      | 💬 Accepted by EMNLP 2023 (Industry Track)
    </div>
    <details class="paper-abstract">
      E-commerce product catalogs contain billions of items. Most products have lengthy titles, as sellers pack them with product attributes to improve retrieval, and highlight key product aspects. This results in a gap between such unnatural products titles, and how customers refer to them. It also limits how e-commerce stores can use these seller-provided titles for recommendation, QA, or review summarization. Inspired by recent work on instruction-tuned LLMs, we present InstructPTS, a controllable approach for the task of Product Title Summarization (PTS). Trained using a novel instruction fine-tuning strategy, our approach is able to summarize product titles according to various criteria (e.g. number of words in a summary, inclusion of specific phrases, etc.). Extensive evaluation on a real-world e-commerce catalog shows that compared to simple fine-tuning of LLMs, our proposed approach can generate more accurate product name summaries, with an improvement of over 14 and 8 BLEU and ROUGE points, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.16271v1">CycleAlign: Iterative Distillation from Black-box LLM to White-box Models for Better Human Alignment</a></div>
    <div class="paper-meta">
      📅 2023-10-25
    </div>
    <details class="paper-abstract">
      Language models trained on large-scale corpus often generate content that is harmful, toxic, or contrary to human preferences, making their alignment with human values a critical concern. Reinforcement learning from human feedback (RLHF) with algorithms like PPO is a prevalent approach for alignment but is often complex, unstable, and resource-intensive. Recently, ranking-based alignment methods have emerged, offering stability and effectiveness by replacing the RL framework with supervised fine-tuning, but they are costly due to the need for annotated data. Considering that existing large language models (LLMs) like ChatGPT are already relatively well-aligned and cost-friendly, researchers have begun to align the language model with human preference from AI feedback. The common practices, which unidirectionally distill the instruction-following responses from LLMs, are constrained by their bottleneck. Thus we introduce CycleAlign to distill alignment capabilities from parameter-invisible LLMs (black-box) to a parameter-visible model (white-box) in an iterative manner. With in-context learning (ICL) as the core of the cycle, the black-box models are able to rank the model-generated responses guided by human-craft instruction and demonstrations about their preferences. During iterative interaction, the white-box models also have a judgment about responses generated by them. Consequently, the agreement ranking could be viewed as a pseudo label to dynamically update the in-context demonstrations and improve the preference ranking ability of black-box models. Through multiple interactions, the CycleAlign framework could align the white-box model with the black-box model effectively in a low-resource way. Empirical results illustrate that the model fine-tuned by CycleAlign remarkably exceeds existing methods, and achieves the state-of-the-art performance in alignment with human value.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.16253v1">ConDefects: A New Dataset to Address the Data Leakage Concern for LLM-based Fault Localization and Program Repair</a></div>
    <div class="paper-meta">
      📅 2023-10-25
      | 💬 5pages, 3 figures
    </div>
    <details class="paper-abstract">
      With the growing interest on Large Language Models (LLMs) for fault localization and program repair, ensuring the integrity and generalizability of the LLM-based methods becomes paramount. The code in existing widely-adopted benchmarks for these tasks was written before the the bloom of LLMs and may be included in the training data of existing popular LLMs, thereby suffering from the threat of data leakage, leading to misleadingly optimistic performance metrics. To address this issue, we introduce "ConDefects", a novel dataset of real faults meticulously curated to eliminate such overlap. ConDefects contains 1,254 Java faulty programs and 1,625 Python faulty programs. All these programs are sourced from the online competition platform AtCoder and were produced between October 2021 and September 2023. We pair each fault with fault locations and the corresponding repaired code versions, making it tailored for in fault localization and program repair related research. We also provide interfaces for selecting subsets based on different time windows and coding task difficulties. While inspired by LLM-based tasks, ConDefects can be adopted for benchmarking ALL types of fault localization and program repair methods. The dataset is publicly available, and a demo video can be found at https://www.youtube.com/watch?v=22j15Hj5ONk.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.11430v2">TELeR: A General Taxonomy of LLM Prompts for Benchmarking Complex Tasks</a></div>
    <div class="paper-meta">
      📅 2023-10-24
      | 💬 Accepted to EMNLP 2023
    </div>
    <details class="paper-abstract">
      While LLMs have shown great success in understanding and generating text in traditional conversational settings, their potential for performing ill-defined complex tasks is largely under-studied. Indeed, we are yet to conduct comprehensive benchmarking studies with multiple LLMs that are exclusively focused on a complex task. However, conducting such benchmarking studies is challenging because of the large variations in LLMs' performance when different prompt types/styles are used and different degrees of detail are provided in the prompts. To address this issue, the paper proposes a general taxonomy that can be used to design prompts with specific properties in order to perform a wide range of complex tasks. This taxonomy will allow future benchmarking studies to report the specific categories of prompts used as part of the study, enabling meaningful comparisons across different studies. Also, by establishing a common standard through this taxonomy, researchers will be able to draw more accurate conclusions about LLMs' performance on a specific complex task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.18360v1">Guiding LLM to Fool Itself: Automatically Manipulating Machine Reading Comprehension Shortcut Triggers</a></div>
    <div class="paper-meta">
      📅 2023-10-24
      | 💬 Accepted to EMNLP 2023 Findings
    </div>
    <details class="paper-abstract">
      Recent applications of LLMs in Machine Reading Comprehension (MRC) systems have shown impressive results, but the use of shortcuts, mechanisms triggered by features spuriously correlated to the true label, has emerged as a potential threat to their reliability. We analyze the problem from two angles: LLMs as editors, guided to edit text to mislead LLMs; and LLMs as readers, who answer questions based on the edited text. We introduce a framework that guides an editor to add potential shortcuts-triggers to samples. Using GPT4 as the editor, we find it can successfully edit trigger shortcut in samples that fool LLMs. Analysing LLMs as readers, we observe that even capable LLMs can be deceived using shortcut knowledge. Strikingly, we discover that GPT4 can be deceived by its own edits (15% drop in F1). Our findings highlight inherent vulnerabilities of LLMs to shortcut manipulations. We publish ShortcutQA, a curated dataset generated by our framework for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.15780v1">Make LLM a Testing Expert: Bringing Human-like Interaction to Mobile GUI Testing via Functionality-aware Decisions</a></div>
    <div class="paper-meta">
      📅 2023-10-24
      | 💬 Accepted by IEEE/ACM International Conference on Software Engineering 2024 (ICSE 2024). arXiv admin note: substantial text overlap with arXiv:2305.09434
    </div>
    <details class="paper-abstract">
      Automated Graphical User Interface (GUI) testing plays a crucial role in ensuring app quality, especially as mobile applications have become an integral part of our daily lives. Despite the growing popularity of learning-based techniques in automated GUI testing due to their ability to generate human-like interactions, they still suffer from several limitations, such as low testing coverage, inadequate generalization capabilities, and heavy reliance on training data. Inspired by the success of Large Language Models (LLMs) like ChatGPT in natural language understanding and question answering, we formulate the mobile GUI testing problem as a Q&A task. We propose GPTDroid, asking LLM to chat with the mobile apps by passing the GUI page information to LLM to elicit testing scripts, and executing them to keep passing the app feedback to LLM, iterating the whole process. Within this framework, we have also introduced a functionality-aware memory prompting mechanism that equips the LLM with the ability to retain testing knowledge of the whole process and conduct long-term, functionality-based reasoning to guide exploration. We evaluate it on 93 apps from Google Play and demonstrate that it outperforms the best baseline by 32% in activity coverage, and detects 31% more bugs at a faster rate. Moreover, GPTDroid identify 53 new bugs on Google Play, of which 35 have been confirmed and fixed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.15654v1">A Survey on Detection of LLMs-Generated Content</a></div>
    <div class="paper-meta">
      📅 2023-10-24
      | 💬 We will keep updating at https://github.com/Xianjun-Yang/Awesome_papers_on_LLMs_detection.git
    </div>
    <details class="paper-abstract">
      The burgeoning capabilities of advanced large language models (LLMs) such as ChatGPT have led to an increase in synthetic content generation with implications across a variety of sectors, including media, cybersecurity, public discourse, and education. As such, the ability to detect LLMs-generated content has become of paramount importance. We aim to provide a detailed overview of existing detection strategies and benchmarks, scrutinizing their differences and identifying key challenges and prospects in the field, advocating for more adaptable and robust models to enhance detection accuracy. We also posit the necessity for a multi-faceted approach to defend against various attacks to counter the rapidly advancing capabilities of LLMs. To the best of our knowledge, this work is the first comprehensive survey on the detection in the era of LLMs. We hope it will provide a broad understanding of the current landscape of LLMs-generated content detection, offering a guiding reference for researchers and practitioners striving to uphold the integrity of digital information in an era increasingly dominated by synthetic content. The relevant papers are summarized and will be consistently updated at https://github.com/Xianjun-Yang/Awesome_papers_on_LLMs_detection.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.15515v1">Fighting Fire with Fire: The Dual Role of LLMs in Crafting and Detecting Elusive Disinformation</a></div>
    <div class="paper-meta">
      📅 2023-10-24
      | 💬 Accepted at EMNLP 2023
    </div>
    <details class="paper-abstract">
      Recent ubiquity and disruptive impacts of large language models (LLMs) have raised concerns about their potential to be misused (.i.e, generating large-scale harmful and misleading content). To combat this emerging risk of LLMs, we propose a novel "Fighting Fire with Fire" (F3) strategy that harnesses modern LLMs' generative and emergent reasoning capabilities to counter human-written and LLM-generated disinformation. First, we leverage GPT-3.5-turbo to synthesize authentic and deceptive LLM-generated content through paraphrase-based and perturbation-based prefix-style prompts, respectively. Second, we apply zero-shot in-context semantic reasoning techniques with cloze-style prompts to discern genuine from deceptive posts and news articles. In our extensive experiments, we observe GPT-3.5-turbo's zero-shot superiority for both in-distribution and out-of-distribution datasets, where GPT-3.5-turbo consistently achieved accuracy at 68-72%, unlike the decline observed in previous customized and fine-tuned disinformation detectors. Our codebase and dataset are available at https://github.com/mickeymst/F3.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.15511v1">KITAB: Evaluating LLMs on Constraint Satisfaction for Information Retrieval</a></div>
    <div class="paper-meta">
      📅 2023-10-24
      | 💬 23 pages
    </div>
    <details class="paper-abstract">
      We study the ability of state-of-the art models to answer constraint satisfaction queries for information retrieval (e.g., 'a list of ice cream shops in San Diego'). In the past, such queries were considered to be tasks that could only be solved via web-search or knowledge bases. More recently, large language models (LLMs) have demonstrated initial emergent abilities in this task. However, many current retrieval benchmarks are either saturated or do not measure constraint satisfaction. Motivated by rising concerns around factual incorrectness and hallucinations of LLMs, we present KITAB, a new dataset for measuring constraint satisfaction abilities of language models. KITAB consists of book-related data across more than 600 authors and 13,000 queries, and also offers an associated dynamic data collection and constraint verification approach for acquiring similar test data for other authors. Our extended experiments on GPT4 and GPT3.5 characterize and decouple common failure modes across dimensions such as information popularity, constraint types, and context availability. Results show that in the absence of context, models exhibit severe limitations as measured by irrelevant information, factual errors, and incompleteness, many of which exacerbate as information popularity decreases. While context availability mitigates irrelevant information, it is not helpful for satisfying constraints, identifying fundamental barriers to constraint satisfaction. We open source our contributions to foster further research on improving constraint satisfaction abilities of future models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.16339v2">Don't Trust ChatGPT when Your Question is not in English: A Study of Multilingual Abilities and Types of LLMs</a></div>
    <div class="paper-meta">
      📅 2023-10-24
      | 💬 Paper accepted to EMNLP 2023
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated exceptional natural language understanding abilities and have excelled in a variety of natural language processing (NLP)tasks in recent years. Despite the fact that most LLMs are trained predominantly in English, multiple studies have demonstrated their comparative performance in many other languages. However, fundamental questions persist regarding how LLMs acquire their multi-lingual abilities and how performance varies across different languages. These inquiries are crucial for the study of LLMs since users and researchers often come from diverse language backgrounds, potentially influencing their utilization and interpretation of LLMs' results. In this work, we propose a systematic way of qualifying the performance disparities of LLMs under multilingual settings. We investigate the phenomenon of across-language generalizations in LLMs, wherein insufficient multi-lingual training data leads to advanced multi-lingual capabilities. To accomplish this, we employ a novel back-translation-based prompting method. The results show that GPT exhibits highly translating-like behaviour in multilingual settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.01278v2">VPGTrans: Transfer Visual Prompt Generator across LLMs</a></div>
    <div class="paper-meta">
      📅 2023-10-24
      | 💬 Project Website: https://vpgtrans.github.io Code: https://github.com/VPGTrans/VPGTrans NeurIPS 2023
    </div>
    <details class="paper-abstract">
      While developing a new multimodal LLM (MLLM) by pre-training on tremendous image-text pairs from scratch can be exceedingly resource-consuming, connecting an existing LLM with a comparatively lightweight visual prompt generator (VPG) becomes a feasible paradigm. However, further tuning the VPG part of the MLLM still suffers from indispensable computational costs, i.e., requiring thousands of GPU hours and millions of training data. One alternative solution is to transfer an existing VPG from any existing MLLMs for the target MLLM. In this work, we for the first time investigate the VPG transferability across LLMs, and explore a solution to reduce the cost of VPG transfer. We first study the VPG transfer across different LLM sizes (e.g., small-to-large), and across different LLM types, through which we diagnose the key factors to maximize the transfer efficiency. Based on our observation, we design a two-stage transfer framework named VPGTrans, which is simple yet highly effective. Through extensive experiments, we demonstrate that VPGTrans helps significantly speed up the transfer learning process without compromising performance. Remarkably, it helps achieve the VPG transfer from BLIP-2 OPT$_\text{2.7B}$ to BLIP-2 OPT$_\text{6.7B}$ with over 10 times speed-up and 10.7% training data compared with connecting a VPG to OPT$_\text{6.7B}$ from scratch. Further, a series of intriguing findings and potential rationales behind them are provided and discussed. Finally, we showcase the practical value of our VPGTrans approach, by customizing two novel MLLMs, including VL-LLaMA and VL-Vicuna, with recently released LLaMA and Vicuna LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.15455v1">UI Layout Generation with LLMs Guided by UI Grammar</a></div>
    <div class="paper-meta">
      📅 2023-10-24
      | 💬 ICML 2023 Workshop on AI and HCI
    </div>
    <details class="paper-abstract">
      The recent advances in Large Language Models (LLMs) have stimulated interest among researchers and industry professionals, particularly in their application to tasks concerning mobile user interfaces (UIs). This position paper investigates the use of LLMs for UI layout generation. Central to our exploration is the introduction of UI grammar -- a novel approach we proposed to represent the hierarchical structure inherent in UI screens. The aim of this approach is to guide the generative capacities of LLMs more effectively and improve the explainability and controllability of the process. Initial experiments conducted with GPT-4 showed the promising capability of LLMs to produce high-quality user interfaces via in-context learning. Furthermore, our preliminary comparative study suggested the potential of the grammar-based approach in improving the quality of generative results in specific aspects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.15355v1">Why LLMs Hallucinate, and How to Get (Evidential) Closure: Perceptual, Intensional, and Extensional Learning for Faithful Natural Language Generation</a></div>
    <div class="paper-meta">
      📅 2023-10-23
    </div>
    <details class="paper-abstract">
      We show that LLMs hallucinate because their output is not constrained to be synonymous with claims for which they have evidence: a condition that we call evidential closure. Information about the truth or falsity of sentences is not statistically identified in the standard neural probabilistic language model setup, and so cannot be conditioned on to generate new strings. We then show how to constrain LLMs to produce output that does satisfy evidential closure. A multimodal LLM must learn about the external world (perceptual learning); it must learn a mapping from strings to states of the world (extensional learning); and, to achieve fluency when generalizing beyond a body of evidence, it must learn mappings from strings to their synonyms (intensional learning). The output of a unimodal LLM must be synonymous with strings in a validated evidence set. Finally, we present a heuristic procedure, Learn-Babble-Prune, that yields faithful output from an LLM by rejecting output that is not synonymous with claims for which the LLM has evidence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.15117v1">Causal Inference Using LLM-Guided Discovery</a></div>
    <div class="paper-meta">
      📅 2023-10-23
    </div>
    <details class="paper-abstract">
      At the core of causal inference lies the challenge of determining reliable causal graphs solely based on observational data. Since the well-known backdoor criterion depends on the graph, any errors in the graph can propagate downstream to effect inference. In this work, we initially show that complete graph information is not necessary for causal effect inference; the topological order over graph variables (causal order) alone suffices. Further, given a node pair, causal order is easier to elicit from domain experts compared to graph edges since determining the existence of an edge can depend extensively on other variables. Interestingly, we find that the same principle holds for Large Language Models (LLMs) such as GPT-3.5-turbo and GPT-4, motivating an automated method to obtain causal order (and hence causal effect) with LLMs acting as virtual domain experts. To this end, we employ different prompting strategies and contextual cues to propose a robust technique of obtaining causal order from LLMs. Acknowledging LLMs' limitations, we also study possible techniques to integrate LLMs with established causal discovery algorithms, including constraint-based and score-based methods, to enhance their performance. Extensive experiments demonstrate that our approach significantly improves causal ordering accuracy as compared to discovery algorithms, highlighting the potential of LLMs to enhance causal inference across diverse fields.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.15100v1">LLM-in-the-loop: Leveraging Large Language Model for Thematic Analysis</a></div>
    <div class="paper-meta">
      📅 2023-10-23
      | 💬 EMNLP 2023 Findings
    </div>
    <details class="paper-abstract">
      Thematic analysis (TA) has been widely used for analyzing qualitative data in many disciplines and fields. To ensure reliable analysis, the same piece of data is typically assigned to at least two human coders. Moreover, to produce meaningful and useful analysis, human coders develop and deepen their data interpretation and coding over multiple iterations, making TA labor-intensive and time-consuming. Recently the emerging field of large language models (LLMs) research has shown that LLMs have the potential replicate human-like behavior in various tasks: in particular, LLMs outperform crowd workers on text-annotation tasks, suggesting an opportunity to leverage LLMs on TA. We propose a human-LLM collaboration framework (i.e., LLM-in-the-loop) to conduct TA with in-context learning (ICL). This framework provides the prompt to frame discussions with a LLM (e.g., GPT-3.5) to generate the final codebook for TA. We demonstrate the utility of this framework using survey datasets on the aspects of the music listening experience and the usage of a password manager. Results of the two case studies show that the proposed framework yields similar coding quality to that of human coders but reduces TA's labor and time demands.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.14970v1">Towards LLM-driven Dialogue State Tracking</a></div>
    <div class="paper-meta">
      📅 2023-10-23
      | 💬 Accepted at EMNLP 2023
    </div>
    <details class="paper-abstract">
      Dialogue State Tracking (DST) is of paramount importance in ensuring accurate tracking of user goals and system actions within task-oriented dialogue systems. The emergence of large language models (LLMs) such as GPT3 and ChatGPT has sparked considerable interest in assessing their efficacy across diverse applications. In this study, we conduct an initial examination of ChatGPT's capabilities in DST. Our evaluation uncovers the exceptional performance of ChatGPT in this task, offering valuable insights to researchers regarding its capabilities and providing useful directions for designing and enhancing dialogue systems. Despite its impressive performance, ChatGPT has significant limitations including its closed-source nature, request restrictions, raising data privacy concerns, and lacking local deployment capabilities. To address these concerns, we present LDST, an LLM-driven DST framework based on smaller, open-source foundation models. By utilizing a novel domain-slot instruction tuning method, LDST achieves performance on par with ChatGPT. Comprehensive evaluations across three distinct experimental settings, we find that LDST exhibits remarkable performance improvements in both zero-shot and few-shot setting compared to previous SOTA methods. The source code is provided for reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.15074v3">Have LLMs Advanced Enough? A Challenging Problem Solving Benchmark For Large Language Models</a></div>
    <div class="paper-meta">
      📅 2023-10-23
      | 💬 EMNLP 2023
    </div>
    <details class="paper-abstract">
      The performance of large language models (LLMs) on existing reasoning benchmarks has significantly improved over the past years. In response, we present JEEBench, a considerably more challenging benchmark dataset for evaluating the problem solving abilities of LLMs. We curate 515 challenging pre-engineering mathematics, physics and chemistry problems from the highly competitive IIT JEE-Advanced exam. Long-horizon reasoning on top of deep in-domain knowledge is essential for solving problems in this benchmark. Our evaluation on various open-source and proprietary models reveals that the highest performance, even after using techniques like self-consistency, self-refinement and chain-of-thought prompting, is less than 40%. The typical failure modes of GPT-4, the best model, are errors in algebraic manipulation, difficulty in grounding abstract concepts into mathematical equations accurately and failure in retrieving relevant domain-specific concepts. We also observe that by mere prompting, GPT-4 is unable to assess risk introduced by negative marking for incorrect answers. For this, we develop a post-hoc confidence-thresholding method over self-consistency, which enables effective response selection. We hope that our challenging benchmark will guide future re-search in problem-solving using LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.14819v1">Analyzing Multilingual Competency of LLMs in Multi-Turn Instruction Following: A Case Study of Arabic</a></div>
    <div class="paper-meta">
      📅 2023-10-23
      | 💬 Accepted at SIGARAB ArabicNLP 2023
    </div>
    <details class="paper-abstract">
      While significant progress has been made in benchmarking Large Language Models (LLMs) across various tasks, there is a lack of comprehensive evaluation of their abilities in responding to multi-turn instructions in less-commonly tested languages like Arabic. Our paper offers a detailed examination of the proficiency of open LLMs in such scenarios in Arabic. Utilizing a customized Arabic translation of the MT-Bench benchmark suite, we employ GPT-4 as a uniform evaluator for both English and Arabic queries to assess and compare the performance of the LLMs on various open-ended tasks. Our findings reveal variations in model responses on different task categories, e.g., logic vs. literacy, when instructed in English or Arabic. We find that fine-tuned base models using multilingual and multi-turn datasets could be competitive to models trained from scratch on multilingual data. Finally, we hypothesize that an ensemble of small, open LLMs could perform competitively to proprietary LLMs on the benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.14557v1">The Skipped Beat: A Study of Sociopragmatic Understanding in LLMs for 64 Languages</a></div>
    <div class="paper-meta">
      📅 2023-10-23
      | 💬 Accepted by EMNLP 2023 Main conference
    </div>
    <details class="paper-abstract">
      Instruction tuned large language models (LLMs), such as ChatGPT, demonstrate remarkable performance in a wide range of tasks. Despite numerous recent studies that examine the performance of instruction-tuned LLMs on various NLP benchmarks, there remains a lack of comprehensive investigation into their ability to understand cross-lingual sociopragmatic meaning (SM), i.e., meaning embedded within social and interactive contexts. This deficiency arises partly from SM not being adequately represented in any of the existing benchmarks. To address this gap, we present SPARROW, an extensive multilingual benchmark specifically designed for SM understanding. SPARROW comprises 169 datasets covering 13 task types across six primary categories (e.g., anti-social language detection, emotion recognition). SPARROW datasets encompass 64 different languages originating from 12 language families representing 16 writing scripts. We evaluate the performance of various multilingual pretrained language models (e.g., mT5) and instruction-tuned LLMs (e.g., BLOOMZ, ChatGPT) on SPARROW through fine-tuning, zero-shot, and/or few-shot learning. Our comprehensive analysis reveals that existing open-source instruction tuned LLMs still struggle to understand SM across various languages, performing close to a random baseline in some cases. We also find that although ChatGPT outperforms many LLMs, it still falls behind task-specific finetuned models with a gap of 12.19 SPARROW score. Our benchmark is available at: https://github.com/UBC-NLP/SPARROW
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.14288v2">LLM-powered Data Augmentation for Enhanced Cross-lingual Performance</a></div>
    <div class="paper-meta">
      📅 2023-10-22
      | 💬 EMNLP 2023 Main Conference
    </div>
    <details class="paper-abstract">
      This paper explores the potential of leveraging Large Language Models (LLMs) for data augmentation in multilingual commonsense reasoning datasets where the available training data is extremely limited. To achieve this, we utilise several LLMs, namely Dolly-v2, StableVicuna, ChatGPT, and GPT-4, to augment three datasets: XCOPA, XWinograd, and XStoryCloze. Subsequently, we evaluate the effectiveness of fine-tuning smaller multilingual models, mBERT and XLMR, using the synthesised data. We compare the performance of training with data generated in English and target languages, as well as translated English-generated data, revealing the overall advantages of incorporating data generated by LLMs, e.g. a notable 13.4 accuracy score improvement for the best case. Furthermore, we conduct a human evaluation by asking native speakers to assess the naturalness and logical coherence of the generated examples across different languages. The results of the evaluation indicate that LLMs such as ChatGPT and GPT-4 excel at producing natural and coherent text in most languages, however, they struggle to generate meaningful text in certain languages like Tamil. We also observe that ChatGPT falls short in generating plausible alternatives compared to the original dataset, whereas examples from GPT-4 exhibit competitive logical consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.12823v2">AgentTuning: Enabling Generalized Agent Abilities for LLMs</a></div>
    <div class="paper-meta">
      📅 2023-10-22
      | 💬 31 pages
    </div>
    <details class="paper-abstract">
      Open large language models (LLMs) with great performance in various tasks have significantly advanced the development of LLMs. However, they are far inferior to commercial models such as ChatGPT and GPT-4 when acting as agents to tackle complex tasks in the real world. These agent tasks employ LLMs as the central controller responsible for planning, memorization, and tool utilization, necessitating both fine-grained prompting methods and robust LLMs to achieve satisfactory performance. Though many prompting methods have been proposed to complete particular agent tasks, there is lack of research focusing on improving the agent capabilities of LLMs themselves without compromising their general abilities. In this work, we present AgentTuning, a simple and general method to enhance the agent abilities of LLMs while maintaining their general LLM capabilities. We construct AgentInstruct, a lightweight instruction-tuning dataset containing high-quality interaction trajectories. We employ a hybrid instruction-tuning strategy by combining AgentInstruct with open-source instructions from general domains. AgentTuning is used to instruction-tune the Llama 2 series, resulting in AgentLM. Our evaluations show that AgentTuning enables LLMs' agent capabilities without compromising general abilities. The AgentLM-70B is comparable to GPT-3.5-turbo on unseen agent tasks, demonstrating generalized agent capabilities. We open source the AgentInstruct and AgentLM-7B, 13B, and 70B models at https://github.com/THUDM/AgentTuning, serving open and powerful alternatives to commercial LLMs for agent tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.18344v1">Chainpoll: A high efficacy method for LLM hallucination detection</a></div>
    <div class="paper-meta">
      📅 2023-10-22
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have experienced notable advancements in generating coherent and contextually relevant responses. However, hallucinations - incorrect or unfounded claims - are still prevalent, prompting the creation of automated metrics to detect these in LLM outputs. Our contributions include: introducing ChainPoll, an innovative hallucination detection method that excels compared to its counterparts, and unveiling RealHall, a refined collection of benchmark datasets to assess hallucination detection metrics from recent studies. While creating RealHall, we assessed tasks and datasets from previous hallucination detection studies and observed that many are not suitable for the potent LLMs currently in use. Overcoming this, we opted for four datasets challenging for modern LLMs and pertinent to real-world scenarios. Using RealHall, we conducted a comprehensive comparison of ChainPoll with numerous hallucination metrics from recent studies. Our findings indicate that ChainPoll outperforms in all RealHall benchmarks, achieving an overall AUROC of 0.781. This surpasses the next best theoretical method by 11% and exceeds industry standards by over 23%. Additionally, ChainPoll is cost-effective and offers greater transparency than other metrics. We introduce two novel metrics to assess LLM hallucinations: Adherence and Correctness. Adherence is relevant to Retrieval Augmented Generation workflows, evaluating an LLM's analytical capabilities within given documents and contexts. In contrast, Correctness identifies logical and reasoning errors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.07004v2">Not All Languages Are Created Equal in LLMs: Improving Multilingual Capability by Cross-Lingual-Thought Prompting</a></div>
    <div class="paper-meta">
      📅 2023-10-22
      | 💬 Accepted by EMNLP 2023 Findings
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate impressive multilingual capability, but their performance varies substantially across different languages. In this work, we introduce a simple yet effective method, called cross-lingual-thought prompting (XLT), to systematically improve the multilingual capability of LLMs. Specifically, XLT is a generic template prompt that stimulates cross-lingual and logical reasoning skills to enhance task performance across languages. We conduct comprehensive evaluations on 7 typical benchmarks related to reasoning, understanding, and generation tasks, covering both high-resource and low-resource languages. Experimental results show that XLT not only remarkably enhances the performance of various multilingual tasks but also significantly reduces the gap between the average performance and the best performance of each task in different languages. Notably, XLT brings over 10 points of average improvement in arithmetic reasoning and open-domain question-answering tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.10698v2">Bridging Code Semantic and LLMs: Semantic Chain-of-Thought Prompting for Code Generation</a></div>
    <div class="paper-meta">
      📅 2023-10-22
      | 💬 There may be calculation errors in Table 4 of the paper. We need time to verify and supplement, so the manuscript needs to be withdrawn. Thanks!
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have showcased remarkable prowess in code generation. However, automated code generation is still challenging since it requires a high-level semantic mapping between natural language requirements and codes. Most existing LLMs-based approaches for code generation rely on decoder-only causal language models often treate codes merely as plain text tokens, i.e., feeding the requirements as a prompt input, and outputing code as flat sequence of tokens, potentially missing the rich semantic features inherent in source code. To bridge this gap, this paper proposes the "Semantic Chain-of-Thought" approach to intruduce semantic information of code, named SeCoT. Our motivation is that the semantic information of the source code (\eg data flow and control flow) describes more precise program execution behavior, intention and function. By guiding LLM consider and integrate semantic information, we can achieve a more granular understanding and representation of code, enhancing code generation accuracy. Meanwhile, while traditional techniques leveraging such semantic information require complex static or dynamic code analysis to obtain features such as data flow and control flow, SeCoT demonstrates that this process can be fully automated via the intrinsic capabilities of LLMs (i.e., in-context learning), while being generalizable and applicable to challenging domains. While SeCoT can be applied with different LLMs, this paper focuses on the powerful GPT-style models: ChatGPT(close-source model) and WizardCoder(open-source model). The experimental study on three popular DL benchmarks (i.e., HumanEval, HumanEval-ET and MBPP) shows that SeCoT can achieves state-of-the-art performance, greatly improving the potential for large models and code generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.07611v2">Democratizing LLMs: An Exploration of Cost-Performance Trade-offs in Self-Refined Open-Source Models</a></div>
    <div class="paper-meta">
      📅 2023-10-22
      | 💬 Accepted to Findings of EMNLP 2023
    </div>
    <details class="paper-abstract">
      The dominance of proprietary LLMs has led to restricted access and raised information privacy concerns. High-performing open-source alternatives are crucial for information-sensitive and high-volume applications but often lag behind in performance. To address this gap, we propose (1) A untargeted variant of iterative self-critique and self-refinement devoid of external influence. (2) A novel ranking metric - Performance, Refinement, and Inference Cost Score (PeRFICS) - to find the optimal model for a given task considering refined performance and cost. Our experiments show that SoTA open source models of varying sizes from 7B - 65B, on average, improve 8.2% from their baseline performance. Strikingly, even models with extremely small memory footprints, such as Vicuna-7B, show a 11.74% improvement overall and up to a 25.39% improvement in high-creativity, open ended tasks on the Vicuna benchmark. Vicuna-13B takes it a step further and outperforms ChatGPT post-refinement. This work has profound implications for resource-constrained and information-sensitive environments seeking to leverage LLMs without incurring prohibitive costs, compromising on performance and privacy. The domain-agnostic self-refinement process coupled with our novel ranking metric facilitates informed decision-making in model selection, thereby reducing costs and democratizing access to high-performing language models, as evidenced by case studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.05079v2">Revisiting Block-based Quantisation: What is Important for Sub-8-bit LLM Inference?</a></div>
    <div class="paper-meta">
      📅 2023-10-21
      | 💬 Accepted by EMNLP2023
    </div>
    <details class="paper-abstract">
      The inference of Large language models (LLMs) requires immense computation and memory resources. To curtail these costs, quantisation has merged as a promising solution, but existing LLM quantisation mainly focuses on 8-bit. In this work, we explore the statistical and learning properties of the LLM layer and attribute the bottleneck of LLM quantisation to numerical scaling offsets. To address this, we adapt block quantisations for LLMs, a family of methods that share scaling factors across packed numbers. Block quantisations efficiently reduce the numerical scaling offsets solely from an arithmetic perspective, without additional treatments in the computational path. Our nearly-lossless quantised 6-bit LLMs achieve a $19\times$ higher arithmetic density and $5\times$ memory density than the float32 baseline, surpassing the prior art 8-bit quantisation by $2.5\times$ in arithmetic density and $1.2\times$ in memory density, without requiring any data calibration or re-training. We also share our insights into sub-8-bit LLM quantisation, including the mismatch between activation and weight distributions, optimal fine-tuning strategies, and a lower quantisation granularity inherent in the statistical properties of LLMs. The latter two tricks enable nearly-lossless 4-bit LLMs on downstream tasks. Our code is open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.13855v1">Evoke: Evoking Critical Thinking Abilities in LLMs via Reviewer-Author Prompt Editing</a></div>
    <div class="paper-meta">
      📅 2023-10-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have made impressive progress in natural language processing. These models rely on proper human instructions (or prompts) to generate suitable responses. However, the potential of LLMs are not fully harnessed by commonly-used prompting methods: many human-in-the-loop algorithms employ ad-hoc procedures for prompt selection; while auto prompt generation approaches are essentially searching all possible prompts randomly and inefficiently. We propose Evoke, an automatic prompt refinement framework. In Evoke, there are two instances of a same LLM: one as a reviewer (LLM-Reviewer), it scores the current prompt; the other as an author (LLM-Author), it edits the prompt by considering the edit history and the reviewer's feedback. Such an author-reviewer feedback loop ensures that the prompt is refined in each iteration. We further aggregate a data selection approach to Evoke, where only the hard samples are exposed to the LLM. The hard samples are more important because the LLM can develop deeper understanding of the tasks out of them, while the model may already know how to solve the easier cases. Experimental results show that Evoke significantly outperforms existing methods. For instance, in the challenging task of logical fallacy detection, Evoke scores above 80, while all other baseline methods struggle to reach 20.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.13787v1">Enhancing Illicit Activity Detection using XAI: A Multimodal Graph-LLM Framework</a></div>
    <div class="paper-meta">
      📅 2023-10-20
      | 💬 6 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Financial cybercrime prevention is an increasing issue with many organisations and governments. As deep learning models have progressed to identify illicit activity on various financial and social networks, the explainability behind the model decisions has been lacklustre with the investigative analyst at the heart of any deep learning platform. In our paper, we present a state-of-the-art, novel multimodal proactive approach to addressing XAI in financial cybercrime detection. We leverage a triad of deep learning models designed to distill essential representations from transaction sequencing, subgraph connectivity, and narrative generation to significantly streamline the analyst's investigative process. Our narrative generation proposal leverages LLM to ingest transaction details and output contextual narrative for an analyst to understand a transaction and its metadata much further.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.13650v1">BotChat: Evaluating LLMs' Capabilities of Having Multi-Turn Dialogues</a></div>
    <div class="paper-meta">
      📅 2023-10-20
    </div>
    <details class="paper-abstract">
      Interacting with human via high-quality multi-turn dialogues is a key feature of large language models (LLMs). However, human-based evaluation of such capability involves intensive manual labor. This report provides a preliminary evaluation of existing large language models for human-style multi-turn chatting, through an LLM-based approach. We start from real-world human dialogues and keep the very first utterances as the ChatSEED. Then we prompt LLMs to generate a full multi-turn dialogue (tens of utterances) based on the ChatSEED, utterance by utterance. Finally, we adopt state-of-the-art LLMs (GPT-4, \etc) as the judge to evaluate the generated dialogues. With different evaluation protocols, we come to substantially identical conclusions. We find that GPT-4 can generate human-style multi-turn dialogues with impressive quality, significantly outperforms its counterparts. It's difficult for a discriminator to distinguish between GPT-4 generated dialogues and human dialogues. In contrast, other LLMs struggle to generate multi-turn dialogues of satisfactory quality due to poor instruction-following capability, tendency to generate lengthy utterances, or limited general capability. All data and codes will be provided in https://github.com/open-compass/BotChat/ and we hope they can serve as a valuable resource for evaluating multi-turn chatting capabilities of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.13394v1">POSQA: Probe the World Models of LLMs with Size Comparisons</a></div>
    <div class="paper-meta">
      📅 2023-10-20
      | 💬 Accepted by EMNLP 2023 Findings
    </div>
    <details class="paper-abstract">
      Embodied language comprehension emphasizes that language understanding is not solely a matter of mental processing in the brain but also involves interactions with the physical and social environment. With the explosive growth of Large Language Models (LLMs) and their already ubiquitous presence in our daily lives, it is becoming increasingly necessary to verify their real-world understanding. Inspired by cognitive theories, we propose POSQA: a Physical Object Size Question Answering dataset with simple size comparison questions to examine the extremity and analyze the potential mechanisms of the embodied comprehension of the latest LLMs. We show that even the largest LLMs today perform poorly under the zero-shot setting. We then push their limits with advanced prompting techniques and external knowledge augmentation. Furthermore, we investigate whether their real-world comprehension primarily derives from contextual information or internal weights and analyse the impact of prompt formats and report bias of different objects. Our results show that real-world understanding that LLMs shaped from textual data can be vulnerable to deception and confusion by the surface form of prompts, which makes it less aligned with human behaviours.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.13345v1">An LLM can Fool Itself: A Prompt-Based Adversarial Attack</a></div>
    <div class="paper-meta">
      📅 2023-10-20
    </div>
    <details class="paper-abstract">
      The wide-ranging applications of large language models (LLMs), especially in safety-critical domains, necessitate the proper evaluation of the LLM's adversarial robustness. This paper proposes an efficient tool to audit the LLM's adversarial robustness via a prompt-based adversarial attack (PromptAttack). PromptAttack converts adversarial textual attacks into an attack prompt that can cause the victim LLM to output the adversarial sample to fool itself. The attack prompt is composed of three important components: (1) original input (OI) including the original sample and its ground-truth label, (2) attack objective (AO) illustrating a task description of generating a new sample that can fool itself without changing the semantic meaning, and (3) attack guidance (AG) containing the perturbation instructions to guide the LLM on how to complete the task by perturbing the original sample at character, word, and sentence levels, respectively. Besides, we use a fidelity filter to ensure that PromptAttack maintains the original semantic meanings of the adversarial examples. Further, we enhance the attack power of PromptAttack by ensembling adversarial examples at different perturbation levels. Comprehensive empirical results using Llama2 and GPT-3.5 validate that PromptAttack consistently yields a much higher attack success rate compared to AdvGLUE and AdvGLUE++. Interesting findings include that a simple emoji can easily mislead GPT-3.5 to make wrong predictions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.13343v1">Challenges and Contributing Factors in the Utilization of Large Language Models (LLMs)</a></div>
    <div class="paper-meta">
      📅 2023-10-20
    </div>
    <details class="paper-abstract">
      With the development of large language models (LLMs) like the GPT series, their widespread use across various application scenarios presents a myriad of challenges. This review initially explores the issue of domain specificity, where LLMs may struggle to provide precise answers to specialized questions within niche fields. The problem of knowledge forgetting arises as these LLMs might find it hard to balance old and new information. The knowledge repetition phenomenon reveals that sometimes LLMs might deliver overly mechanized responses, lacking depth and originality. Furthermore, knowledge illusion describes situations where LLMs might provide answers that seem insightful but are actually superficial, while knowledge toxicity focuses on harmful or biased information outputs. These challenges underscore problems in the training data and algorithmic design of LLMs. To address these issues, it's suggested to diversify training data, fine-tune models, enhance transparency and interpretability, and incorporate ethics and fairness training. Future technological trends might lean towards iterative methodologies, multimodal learning, model personalization and customization, and real-time learning and feedback mechanisms. In conclusion, future LLMs should prioritize fairness, transparency, and ethics, ensuring they uphold high moral and ethical standards when serving humanity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.11202v3">LLM-based Frameworks for Power Engineering from Routine to Novel Tasks</a></div>
    <div class="paper-meta">
      📅 2023-10-19
    </div>
    <details class="paper-abstract">
      The digitalization of energy sectors has expanded the coding responsibilities for power engineers and researchers. This research article explores the potential of leveraging Large Language Models (LLMs) to alleviate this burden. Here, we propose LLM-based frameworks for different programming tasks in power systems. For well-defined and routine tasks like the classic unit commitment (UC) problem, we deploy an end-to-end framework to systematically assesses four leading LLMs-ChatGPT 3.5, ChatGPT 4.0, Claude and Google Bard in terms of success rate, consistency, and robustness. For complex tasks with limited prior knowledge, we propose a human-in-the-loop framework to enable engineers and LLMs to collaboratively solve the problem through interactive-learning of method recommendation, problem de-composition, subtask programming and synthesis. Through a comparative study between two frameworks, we find that human-in-the-loop features like web access, problem decomposition with field knowledge and human-assisted code synthesis are essential as LLMs currently still fall short in acquiring cutting-edge and domain-specific knowledge to complete a holistic problem-solving project.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.12443v1">Know Where to Go: Make LLM a Relevant, Responsible, and Trustworthy Searcher</a></div>
    <div class="paper-meta">
      📅 2023-10-19
      | 💬 14 pages, 4 figures, under peer review
    </div>
    <details class="paper-abstract">
      The advent of Large Language Models (LLMs) has shown the potential to improve relevance and provide direct answers in web searches. However, challenges arise in validating the reliability of generated results and the credibility of contributing sources, due to the limitations of traditional information retrieval algorithms and the LLM hallucination problem. Aiming to create a "PageRank" for the LLM era, we strive to transform LLM into a relevant, responsible, and trustworthy searcher. We propose a novel generative retrieval framework leveraging the knowledge of LLMs to foster a direct link between queries and online sources. This framework consists of three core modules: Generator, Validator, and Optimizer, each focusing on generating trustworthy online sources, verifying source reliability, and refining unreliable sources, respectively. Extensive experiments and evaluations highlight our method's superior relevance, responsibility, and trustfulness against various SOTA methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.10046v3">TRANSOM: An Efficient Fault-Tolerant System for Training LLMs</a></div>
    <div class="paper-meta">
      📅 2023-10-18
      | 💬 14 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) with hundreds of billions or trillions of parameters, represented by chatGPT, have achieved profound impact on various fields. However, training LLMs with super-large-scale parameters requires large high-performance GPU clusters and long training periods lasting for months. Due to the inevitable hardware and software failures in large-scale clusters, maintaining uninterrupted and long-duration training is extremely challenging. As a result, A substantial amount of training time is devoted to task checkpoint saving and loading, task rescheduling and restart, and task manual anomaly checks, which greatly harms the overall training efficiency. To address these issues, we propose TRANSOM, a novel fault-tolerant LLM training system. In this work, we design three key subsystems: the training pipeline automatic fault tolerance and recovery mechanism named Transom Operator and Launcher (TOL), the training task multi-dimensional metric automatic anomaly detection system named Transom Eagle Eye (TEE), and the training checkpoint asynchronous access automatic fault tolerance and recovery technology named Transom Checkpoint Engine (TCE). Here, TOL manages the lifecycle of training tasks, while TEE is responsible for task monitoring and anomaly reporting. TEE detects training anomalies and reports them to TOL, who automatically enters the fault tolerance strategy to eliminate abnormal nodes and restart the training task. And the asynchronous checkpoint saving and loading functionality provided by TCE greatly shorten the fault tolerance overhead. The experimental results indicate that TRANSOM significantly enhances the efficiency of large-scale LLM training on clusters. Specifically, the pre-training time for GPT3-175B has been reduced by 28%, while checkpoint saving and loading performance have improved by a factor of 20.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.11828v3">Appraising the Potential Uses and Harms of LLMs for Medical Systematic Reviews</a></div>
    <div class="paper-meta">
      📅 2023-10-18
      | 💬 18 pages, 2 figures, 8 tables. Accepted as an EMNLP 2023 main paper
    </div>
    <details class="paper-abstract">
      Medical systematic reviews play a vital role in healthcare decision making and policy. However, their production is time-consuming, limiting the availability of high-quality and up-to-date evidence summaries. Recent advancements in large language models (LLMs) offer the potential to automatically generate literature reviews on demand, addressing this issue. However, LLMs sometimes generate inaccurate (and potentially misleading) texts by hallucination or omission. In healthcare, this can make LLMs unusable at best and dangerous at worst. We conducted 16 interviews with international systematic review experts to characterize the perceived utility and risks of LLMs in the specific context of medical evidence reviews. Experts indicated that LLMs can assist in the writing process by drafting summaries, generating templates, distilling information, and crosschecking information. They also raised concerns regarding confidently composed but inaccurate LLM outputs and other potential downstream harms, including decreased accountability and proliferation of low-quality reviews. Informed by this qualitative analysis, we identify criteria for rigorous evaluation of biomedical LLMs aligned with domain expert views.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.11716v1">Reflection-Tuning: Data Recycling Improves LLM Instruction-Tuning</a></div>
    <div class="paper-meta">
      📅 2023-10-18
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have expanded the horizons of natural language understanding and generation. Notably, the output control and alignment with the input of LLMs can be refined through instruction tuning. However, as highlighted in several studies, low-quality data in the training set are usually detrimental to instruction tuning, resulting in inconsistent or even misleading LLM outputs. We propose a novel method, termed "reflection-tuning," which addresses the problem by self-improvement and judging capabilities of LLMs. This approach utilizes an oracle LLM to recycle the original training data by introspecting and enhancing the quality of instructions and responses in the data. Extensive experiments on widely used evaluation benchmarks show that LLMs trained with our recycled data outperform those trained with existing datasets in various benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.10706v2">Harnessing the Power of LLMs: Evaluating Human-AI Text Co-Creation through the Lens of News Headline Generation</a></div>
    <div class="paper-meta">
      📅 2023-10-18
    </div>
    <details class="paper-abstract">
      To explore how humans can best leverage LLMs for writing and how interacting with these models affects feelings of ownership and trust in the writing process, we compared common human-AI interaction types (e.g., guiding system, selecting from system outputs, post-editing outputs) in the context of LLM-assisted news headline generation. While LLMs alone can generate satisfactory news headlines, on average, human control is needed to fix undesirable model outputs. Of the interaction methods, guiding and selecting model output added the most benefit with the lowest cost (in time and effort). Further, AI assistance did not harm participants' perception of control compared to freeform editing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.11501v1">CoMPosT: Characterizing and Evaluating Caricature in LLM Simulations</a></div>
    <div class="paper-meta">
      📅 2023-10-17
      | 💬 To appear at EMNLP 2023 (Main)
    </div>
    <details class="paper-abstract">
      Recent work has aimed to capture nuances of human behavior by using LLMs to simulate responses from particular demographics in settings like social science experiments and public opinion surveys. However, there are currently no established ways to discuss or evaluate the quality of such LLM simulations. Moreover, there is growing concern that these LLM simulations are flattened caricatures of the personas that they aim to simulate, failing to capture the multidimensionality of people and perpetuating stereotypes. To bridge these gaps, we present CoMPosT, a framework to characterize LLM simulations using four dimensions: Context, Model, Persona, and Topic. We use this framework to measure open-ended LLM simulations' susceptibility to caricature, defined via two criteria: individuation and exaggeration. We evaluate the level of caricature in scenarios from existing work on LLM simulations. We find that for GPT-4, simulations of certain demographics (political and marginalized groups) and topics (general, uncontroversial) are highly susceptible to caricature.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.15714v2">Integrating LLM, EEG, and Eye-Tracking Biomarker Analysis for Word-Level Neural State Classification in Semantic Inference Reading Comprehension</a></div>
    <div class="paper-meta">
      📅 2023-10-17
    </div>
    <details class="paper-abstract">
      With the recent proliferation of large language models (LLMs), such as Generative Pre-trained Transformers (GPT), there has been a significant shift in exploring human and machine comprehension of semantic language meaning. This shift calls for interdisciplinary research that bridges cognitive science and natural language processing (NLP). This pilot study aims to provide insights into individuals' neural states during a semantic relation reading-comprehension task. We propose jointly analyzing LLMs, eye-gaze, and electroencephalographic (EEG) data to study how the brain processes words with varying degrees of relevance to a keyword during reading. We also use a feature engineering approach to improve the fixation-related EEG data classification while participants read words with high versus low relevance to the keyword. The best validation accuracy in this word-level classification is over 60\% across 12 subjects. Words of high relevance to the inference keyword had significantly more eye fixations per word: 1.0584 compared to 0.6576 when excluding no-fixation words, and 1.5126 compared to 1.4026 when including them. This study represents the first attempt to classify brain states at a word level using LLM knowledge. It provides valuable insights into human cognitive abilities and the realm of Artificial General Intelligence (AGI), and offers guidance for developing potential reading-assisted technologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.11207v1">Can Large Language Models Explain Themselves? A Study of LLM-Generated Self-Explanations</a></div>
    <div class="paper-meta">
      📅 2023-10-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) such as ChatGPT have demonstrated superior performance on a variety of natural language processing (NLP) tasks including sentiment analysis, mathematical reasoning and summarization. Furthermore, since these models are instruction-tuned on human conversations to produce "helpful" responses, they can and often will produce explanations along with the response, which we call self-explanations. For example, when analyzing the sentiment of a movie review, the model may output not only the positivity of the sentiment, but also an explanation (e.g., by listing the sentiment-laden words such as "fantastic" and "memorable" in the review). How good are these automatically generated self-explanations? In this paper, we investigate this question on the task of sentiment analysis and for feature attribution explanation, one of the most commonly studied settings in the interpretability literature (for pre-ChatGPT models). Specifically, we study different ways to elicit the self-explanations, evaluate their faithfulness on a set of evaluation metrics, and compare them to traditional explanation methods such as occlusion or LIME saliency maps. Through an extensive set of experiments, we find that ChatGPT's self-explanations perform on par with traditional ones, but are quite different from them according to various agreement metrics, meanwhile being much cheaper to produce (as they are generated along with the prediction). In addition, we identified several interesting characteristics of them, which prompt us to rethink many current model interpretability practices in the era of ChatGPT(-like) LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.09904v2">RAH! RecSys-Assistant-Human: A Human-Centered Recommendation Framework with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2023-10-17
    </div>
    <details class="paper-abstract">
      The rapid evolution of the web has led to an exponential growth in content. Recommender systems play a crucial role in Human-Computer Interaction (HCI) by tailoring content based on individual preferences. Despite their importance, challenges persist in balancing recommendation accuracy with user satisfaction, addressing biases while preserving user privacy, and solving cold-start problems in cross-domain situations. This research argues that addressing these issues is not solely the recommender systems' responsibility, and a human-centered approach is vital. We introduce the RAH Recommender system, Assistant, and Human) framework, an innovative solution with LLM-based agents such as Perceive, Learn, Act, Critic, and Reflect, emphasizing the alignment with user personalities. The framework utilizes the Learn-Act-Critic loop and a reflection mechanism for improving user alignment. Using the real-world data, our experiments demonstrate the RAH framework's efficacy in various recommendation domains, from reducing human burden to mitigating biases and enhancing user control. Notably, our contributions provide a human-centered recommendation framework that partners effectively with various recommendation models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2304.13734v2">The Internal State of an LLM Knows When It's Lying</a></div>
    <div class="paper-meta">
      📅 2023-10-17
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have shown exceptional performance in various tasks, one of their most prominent drawbacks is generating inaccurate or false information with a confident tone. In this paper, we provide evidence that the LLM's internal state can be used to reveal the truthfulness of statements. This includes both statements provided to the LLM, and statements that the LLM itself generates. Our approach is to train a classifier that outputs the probability that a statement is truthful, based on the hidden layer activations of the LLM as it reads or generates the statement. Experiments demonstrate that given a set of test sentences, of which half are true and half false, our trained classifier achieves an average of 71\% to 83\% accuracy labeling which sentences are true versus false, depending on the LLM base model. Furthermore, we explore the relationship between our classifier's performance and approaches based on the probability assigned to the sentence by the LLM. We show that while LLM-assigned sentence probability is related to sentence truthfulness, this probability is also dependent on sentence length and the frequencies of words in the sentence, resulting in our trained classifier providing a more reliable approach to detecting truthfulness, highlighting its potential to enhance the reliability of LLM-generated content and its practical applicability in real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.10996v1">ClarifyGPT: Empowering LLM-based Code Generation with Intention Clarification</a></div>
    <div class="paper-meta">
      📅 2023-10-17
    </div>
    <details class="paper-abstract">
      We introduce a novel framework named ClarifyGPT, which aims to enhance code generation by empowering LLMs with the ability to identify ambiguous requirements and ask targeted clarifying questions. In particular, ClarifyGPT first detects whether a given requirement is ambiguous by performing a code consistency check. If it is ambiguous, ClarifyGPT prompts an LLM to generate targeted clarifying questions. After receiving question responses, ClarifyGPT refines the ambiguous requirement and inputs it into the same LLM to generate a final code solution. To evaluate our ClarifyGPT, we first conduct a human evaluation involving ten participants who use ClarifyGPT for code generation on two publicly available benchmarks: MBPP-sanitized and MBPP-ET. The results show that ClarifyGPT elevates the performance (Pass@1) of GPT-4 from 70.96% to 80.80% on MBPP-sanitized. Furthermore, to perform large-scale automated evaluations of ClarifyGPT across different LLMs and benchmarks without requiring user participation, we introduce a high-fidelity simulation method to simulate user responses. The automated evaluation results also demonstrate that ClarifyGPT can significantly enhance code generation performance compared to the baselines. In particular, ClarifyGPT improves the average performance of GPT-4 and ChatGPT across four benchmarks from 68.02% to 75.75% and from 58.55% to 67.22%, respectively. We believe that ClarifyGPT can effectively facilitate the practical application of LLMs in real-world development environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.10944v1">TEQ: Trainable Equivalent Transformation for Quantization of LLMs</a></div>
    <div class="paper-meta">
      📅 2023-10-17
      | 💬 10 pages, 3 figures
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become more prevalent, there is a growing need for new and improved quantization methods that can meet the computationalast layer demands of these modern architectures while maintaining the accuracy. In this paper, we present TEQ, a trainable equivalent transformation that preserves the FP32 precision of the model output while taking advantage of low-precision quantization, especially 3 and 4 bits weight-only quantization. The training process is lightweight, requiring only 1K steps and fewer than 0.1 percent of the original model's trainable parameters. Furthermore, the transformation does not add any computational overhead during inference. Our results are on-par with the state-of-the-art (SOTA) methods on typical LLMs. Our approach can be combined with other methods to achieve even better performance. The code is available at https://github.com/intel/neural-compressor.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.10632v1">BioPlanner: Automatic Evaluation of LLMs on Protocol Planning in Biology</a></div>
    <div class="paper-meta">
      📅 2023-10-16
      | 💬 EMNLP 2023. Dataset and code: https://github.com/bioplanner/bioplanner
    </div>
    <details class="paper-abstract">
      The ability to automatically generate accurate protocols for scientific experiments would represent a major step towards the automation of science. Large Language Models (LLMs) have impressive capabilities on a wide range of tasks, such as question answering and the generation of coherent text and code. However, LLMs can struggle with multi-step problems and long-term planning, which are crucial for designing scientific experiments. Moreover, evaluation of the accuracy of scientific protocols is challenging, because experiments can be described correctly in many different ways, require expert knowledge to evaluate, and cannot usually be executed automatically. Here we present an automatic evaluation framework for the task of planning experimental protocols, and we introduce BioProt: a dataset of biology protocols with corresponding pseudocode representations. To measure performance on generating scientific protocols, we use an LLM to convert a natural language protocol into pseudocode, and then evaluate an LLM's ability to reconstruct the pseudocode from a high-level description and a list of admissible pseudocode functions. We evaluate GPT-3 and GPT-4 on this task and explore their robustness. We externally validate the utility of pseudocode representations of text by generating accurate novel protocols using retrieved pseudocode, and we run a generated protocol successfully in our biological laboratory. Our framework is extensible to the evaluation and improvement of language model planning abilities in other areas of science or other areas that lack automatic evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.10501v1">NeMo Guardrails: A Toolkit for Controllable and Safe LLM Applications with Programmable Rails</a></div>
    <div class="paper-meta">
      📅 2023-10-16
      | 💬 Accepted at EMNLP 2023 - Demo track
    </div>
    <details class="paper-abstract">
      NeMo Guardrails is an open-source toolkit for easily adding programmable guardrails to LLM-based conversational systems. Guardrails (or rails for short) are a specific way of controlling the output of an LLM, such as not talking about topics considered harmful, following a predefined dialogue path, using a particular language style, and more. There are several mechanisms that allow LLM providers and developers to add guardrails that are embedded into a specific model at training, e.g. using model alignment. Differently, using a runtime inspired from dialogue management, NeMo Guardrails allows developers to add programmable rails to LLM applications - these are user-defined, independent of the underlying LLM, and interpretable. Our initial results show that the proposed approach can be used with several LLM providers to develop controllable and safe LLM applications using programmable rails.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.10358v1">Tabular Representation, Noisy Operators, and Impacts on Table Structure Understanding Tasks in LLMs</a></div>
    <div class="paper-meta">
      📅 2023-10-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly applied for tabular tasks using in-context learning. The prompt representation for a table may play a role in the LLMs ability to process the table. Inspired by prior work, we generate a collection of self-supervised structural tasks (e.g. navigate to a cell and row; transpose the table) and evaluate the performance differences when using 8 formats. In contrast to past work, we introduce 8 noise operations inspired by real-world messy data and adversarial inputs, and show that such operations can impact LLM performance across formats for different structural understanding tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.10077v1">Prompt Packer: Deceiving LLMs through Compositional Instruction with Hidden Attacks</a></div>
    <div class="paper-meta">
      📅 2023-10-16
    </div>
    <details class="paper-abstract">
      Recently, Large language models (LLMs) with powerful general capabilities have been increasingly integrated into various Web applications, while undergoing alignment training to ensure that the generated content aligns with user intent and ethics. Unfortunately, they remain the risk of generating harmful content like hate speech and criminal activities in practical applications. Current approaches primarily rely on detecting, collecting, and training against harmful prompts to prevent such risks. However, they typically focused on the "superficial" harmful prompts with a solitary intent, ignoring composite attack instructions with multiple intentions that can easily elicit harmful content in real-world scenarios. In this paper, we introduce an innovative technique for obfuscating harmful instructions: Compositional Instruction Attacks (CIA), which refers to attacking by combination and encapsulation of multiple instructions. CIA hides harmful prompts within instructions of harmless intentions, making it impossible for the model to identify underlying malicious intentions. Furthermore, we implement two transformation methods, known as T-CIA and W-CIA, to automatically disguise harmful instructions as talking or writing tasks, making them appear harmless to LLMs. We evaluated CIA on GPT-4, ChatGPT, and ChatGLM2 with two safety assessment datasets and two harmful prompt datasets. It achieves an attack success rate of 95%+ on safety assessment datasets, and 83%+ for GPT-4, 91%+ for ChatGPT (gpt-3.5-turbo backed) and ChatGLM2-6B on harmful prompt datasets. Our approach reveals the vulnerability of LLMs to such compositional instruction attacks that harbor underlying harmful intentions, contributing significantly to LLM security development. Warning: this paper may contain offensive or upsetting content!
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.11792v2">Cue-CoT: Chain-of-thought Prompting for Responding to In-depth Dialogue Questions with LLMs</a></div>
    <div class="paper-meta">
      📅 2023-10-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), such as \texttt{ChatGPT}, greatly empower dialogue systems with strong language understanding and generation capabilities. However, most of the previous works prompt the LLMs to directly generate a response based on the dialogue context, overlooking the underlying linguistic cues about the user status exhibited in the context. Such in-depth dialogue scenarios are challenging for existing LLMs to figure out the user's hidden needs and respond satisfactorily through a single-step inference. To this end, we propose a novel linguistic cue-based chain-of-thoughts (\textit{Cue}-CoT), which enhances the LLMs inference with an intermediate reasoning step to find cues exhibited in the dialogue, aiming to provide a more personalized and engaging response. To evaluate the approach, we build a benchmark with in-depth dialogue questions, consisting of 6 datasets in both Chinese and English, targeting 3 major linguistic cues during the conversation: \textit{personality}, \textit{emotion}, and \textit{psychology}. We conduct extensive experiments on the proposed benchmark with 5 LLMs under both zero-shot and one-shot settings. Empirical results demonstrate our proposed \textit{Cue}-CoT method outperforms standard prompting methods in terms of both \textit{helpfulness} and \textit{acceptability} on all datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.09755v1">Beyond Segmentation: Road Network Generation with Multi-Modal LLMs</a></div>
    <div class="paper-meta">
      📅 2023-10-15
    </div>
    <details class="paper-abstract">
      This paper introduces an innovative approach to road network generation through the utilization of a multi-modal Large Language Model (LLM). Our model is specifically designed to process aerial images of road layouts and produce detailed, navigable road networks within the input images. The core innovation of our system lies in the unique training methodology employed for the large language model to generate road networks as its output. This approach draws inspiration from the BLIP-2 architecture arXiv:2301.12597, leveraging pre-trained frozen image encoders and large language models to create a versatile multi-modal LLM. Our work also offers an alternative to the reasoning segmentation method proposed in the LISA paper arXiv:2308.00692. By training the large language model with our approach, the necessity for generating binary segmentation masks, as suggested in the LISA paper arXiv:2308.00692, is effectively eliminated. Experimental results underscore the efficacy of our multi-modal LLM in providing precise and valuable navigational guidance. This research represents a significant stride in bolstering autonomous navigation systems, especially in road network scenarios, where accurate guidance is of paramount importance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.09668v1">Beyond Testers' Biases: Guiding Model Testing with Knowledge Bases using LLMs</a></div>
    <div class="paper-meta">
      📅 2023-10-14
    </div>
    <details class="paper-abstract">
      Current model testing work has mostly focused on creating test cases. Identifying what to test is a step that is largely ignored and poorly supported. We propose Weaver, an interactive tool that supports requirements elicitation for guiding model testing. Weaver uses large language models to generate knowledge bases and recommends concepts from them interactively, allowing testers to elicit requirements for further testing. Weaver provides rich external knowledge to testers and encourages testers to systematically explore diverse concepts beyond their own biases. In a user study, we show that both NLP experts and non-experts identified more, as well as more diverse concepts worth testing when using Weaver. Collectively, they found more than 200 failing test cases for stance detection with zero-shot ChatGPT. Our case studies further show that Weaver can help practitioners test models in real-world settings, where developers define more nuanced application scenarios (e.g., code understanding and transcript summarization) using LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.09454v1">LgTS: Dynamic Task Sampling using LLM-generated sub-goals for Reinforcement Learning Agents</a></div>
    <div class="paper-meta">
      📅 2023-10-14
    </div>
    <details class="paper-abstract">
      Recent advancements in reasoning abilities of Large Language Models (LLM) has promoted their usage in problems that require high-level planning for robots and artificial agents. However, current techniques that utilize LLMs for such planning tasks make certain key assumptions such as, access to datasets that permit finetuning, meticulously engineered prompts that only provide relevant and essential information to the LLM, and most importantly, a deterministic approach to allow execution of the LLM responses either in the form of existing policies or plan operators. In this work, we propose LgTS (LLM-guided Teacher-Student learning), a novel approach that explores the planning abilities of LLMs to provide a graphical representation of the sub-goals to a reinforcement learning (RL) agent that does not have access to the transition dynamics of the environment. The RL agent uses Teacher-Student learning algorithm to learn a set of successful policies for reaching the goal state from the start state while simultaneously minimizing the number of environmental interactions. Unlike previous methods that utilize LLMs, our approach does not assume access to a propreitary or a fine-tuned LLM, nor does it require pre-trained policies that achieve the sub-goals proposed by the LLM. Through experiments on a gridworld based DoorKey domain and a search-and-rescue inspired domain, we show that generating a graphical structure of sub-goals helps in learning policies for the LLM proposed sub-goals and the Teacher-Student learning algorithm minimizes the number of environment interactions when the transition dynamics are unknown.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.01957v2">Driving with LLMs: Fusing Object-Level Vector Modality for Explainable Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2023-10-13
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown promise in the autonomous driving sector, particularly in generalization and interpretability. We introduce a unique object-level multimodal LLM architecture that merges vectorized numeric modalities with a pre-trained LLM to improve context understanding in driving situations. We also present a new dataset of 160k QA pairs derived from 10k driving scenarios, paired with high quality control commands collected with RL agent and question answer pairs generated by teacher LLM (GPT-3.5). A distinct pretraining strategy is devised to align numeric vector modalities with static LLM representations using vector captioning language data. We also introduce an evaluation metric for Driving QA and demonstrate our LLM-driver's proficiency in interpreting driving scenarios, answering questions, and decision-making. Our findings highlight the potential of LLM-based driving action generation in comparison to traditional behavioral cloning. We make our benchmark, datasets, and model available for further exploration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2303.00807v3">UDAPDR: Unsupervised Domain Adaptation via LLM Prompting and Distillation of Rerankers</a></div>
    <div class="paper-meta">
      📅 2023-10-13
      | 💬 Long Paper at Empirical Methods in Natural Language Processing (EMNLP) 2023
    </div>
    <details class="paper-abstract">
      Many information retrieval tasks require large labeled datasets for fine-tuning. However, such datasets are often unavailable, and their utility for real-world applications can diminish quickly due to domain shifts. To address this challenge, we develop and motivate a method for using large language models (LLMs) to generate large numbers of synthetic queries cheaply. The method begins by generating a small number of synthetic queries using an expensive LLM. After that, a much less expensive one is used to create large numbers of synthetic queries, which are used to fine-tune a family of reranker models. These rerankers are then distilled into a single efficient retriever for use in the target domain. We show that this technique boosts zero-shot accuracy in long-tail domains and achieves substantially lower latency than standard reranking methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.09241v1">Precedent-Enhanced Legal Judgment Prediction with LLM and Domain-Model Collaboration</a></div>
    <div class="paper-meta">
      📅 2023-10-13
    </div>
    <details class="paper-abstract">
      Legal Judgment Prediction (LJP) has become an increasingly crucial task in Legal AI, i.e., predicting the judgment of the case in terms of case fact description. Precedents are the previous legal cases with similar facts, which are the basis for the judgment of the subsequent case in national legal systems. Thus, it is worthwhile to explore the utilization of precedents in the LJP. Recent advances in deep learning have enabled a variety of techniques to be used to solve the LJP task. These can be broken down into two categories: large language models (LLMs) and domain-specific models. LLMs are capable of interpreting and generating complex natural language, while domain models are efficient in learning task-specific information. In this paper, we propose the precedent-enhanced LJP framework (PLJP), a system that leverages the strength of both LLM and domain models in the context of precedents. Specifically, the domain models are designed to provide candidate labels and find the proper precedents efficiently, and the large models will make the final prediction with an in-context precedents comprehension. Experiments on the real-world dataset demonstrate the effectiveness of our PLJP. Moreover, our work shows a promising direction for LLM and domain-model collaboration that can be generalized to other vertical domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.10275v1">A ML-LLM pairing for better code comment classification</a></div>
    <div class="paper-meta">
      📅 2023-10-13
      | 💬 10 pages, 2 figures, 2 tables, accepted for the Information Retrieval in Software Engineering track at the Forum for Information Retrieval Evaluation 2023
    </div>
    <details class="paper-abstract">
      The "Information Retrieval in Software Engineering (IRSE)" at FIRE 2023 shared task introduces code comment classification, a challenging task that pairs a code snippet with a comment that should be evaluated as either useful or not useful to the understanding of the relevant code. We answer the code comment classification shared task challenge by providing a two-fold evaluation: from an algorithmic perspective, we compare the performance of classical machine learning systems and complement our evaluations from a data-driven perspective by generating additional data with the help of large language model (LLM) prompting to measure the potential increase in performance. Our best model, which took second place in the shared task, is a Neural Network with a Macro-F1 score of 88.401% on the provided seed data and a 1.5% overall increase in performance on the data generated by the LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.08523v1">LLM-augmented Preference Learning from Natural Language</a></div>
    <div class="paper-meta">
      📅 2023-10-12
    </div>
    <details class="paper-abstract">
      Finding preferences expressed in natural language is an important but challenging task. State-of-the-art(SotA) methods leverage transformer-based models such as BERT, RoBERTa, etc. and graph neural architectures such as graph attention networks. Since Large Language Models (LLMs) are equipped to deal with larger context lengths and have much larger model sizes than the transformer-based model, we investigate their ability to classify comparative text directly. This work aims to serve as a first step towards using LLMs for the CPC task. We design and conduct a set of experiments that format the classification task into an input prompt for the LLM and a methodology to get a fixed-format response that can be automatically evaluated. Comparing performances with existing methods, we see that pre-trained LLMs are able to outperform the previous SotA models with no fine-tuning involved. Our results show that the LLMs can consistently outperform the SotA when the target text is large -- i.e. composed of multiple sentences --, and are still comparable to the SotA performance in shorter text. We also find that few-shot learning yields better performance than zero-shot learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.08433v1">A Confederacy of Models: a Comprehensive Evaluation of LLMs on Creative Writing</a></div>
    <div class="paper-meta">
      📅 2023-10-12
      | 💬 Accepted for publication in Findings of EMNLP 2023
    </div>
    <details class="paper-abstract">
      We evaluate a range of recent LLMs on English creative writing, a challenging and complex task that requires imagination, coherence, and style. We use a difficult, open-ended scenario chosen to avoid training data reuse: an epic narration of a single combat between Ignatius J. Reilly, the protagonist of the Pulitzer Prize-winning novel A Confederacy of Dunces (1980), and a pterodactyl, a prehistoric flying reptile. We ask several LLMs and humans to write such a story and conduct a human evalution involving various criteria such as fluency, coherence, originality, humor, and style. Our results show that some state-of-the-art commercial LLMs match or slightly outperform our writers in most dimensions; whereas open-source LLMs lag behind. Humans retain an edge in creativity, while humor shows a binary divide between LLMs that can handle it comparably to humans and those that fail at it. We discuss the implications and limitations of our study and suggest directions for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.15770v2">CHATREPORT: Democratizing Sustainability Disclosure Analysis through LLM-based Tools</a></div>
    <div class="paper-meta">
      📅 2023-10-11
      | 💬 6 pages. arXiv admin note: text overlap with arXiv:2306.15518
    </div>
    <details class="paper-abstract">
      In the face of climate change, are companies really taking substantial steps toward more sustainable operations? A comprehensive answer lies in the dense, information-rich landscape of corporate sustainability reports. However, the sheer volume and complexity of these reports make human analysis very costly. Therefore, only a few entities worldwide have the resources to analyze these reports at scale, which leads to a lack of transparency in sustainability reporting. Empowering stakeholders with LLM-based automatic analysis tools can be a promising way to democratize sustainability report analysis. However, developing such tools is challenging due to (1) the hallucination of LLMs and (2) the inefficiency of bringing domain experts into the AI development loop. In this paper, we ChatReport, a novel LLM-based system to automate the analysis of corporate sustainability reports, addressing existing challenges by (1) making the answers traceable to reduce the harm of hallucination and (2) actively involving domain experts in the development loop. We make our methodology, annotated datasets, and generated analyses of 1015 reports publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.13078v2">LPML: LLM-Prompting Markup Language for Mathematical Reasoning</a></div>
    <div class="paper-meta">
      📅 2023-10-11
    </div>
    <details class="paper-abstract">
      In utilizing large language models (LLMs) for mathematical reasoning, addressing the errors in the reasoning and calculation present in the generated text by LLMs is a crucial challenge. In this paper, we propose a novel framework that integrates the Chain-of-Thought (CoT) method with an external tool (Python REPL). We discovered that by prompting LLMs to generate structured text in XML-like markup language, we could seamlessly integrate CoT and the external tool and control the undesired behaviors of LLMs. With our approach, LLMs can utilize Python computation to rectify errors within CoT. We applied our method to ChatGPT (GPT-3.5) to solve challenging mathematical problems and demonstrated that combining CoT and Python REPL through the markup language enhances the reasoning capability of LLMs. Our approach enables LLMs to write the markup language and perform advanced mathematical reasoning using only zero-shot prompting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2210.08786v6">Exposing Influence Campaigns in the Age of LLMs: A Behavioral-Based AI Approach to Detecting State-Sponsored Trolls</a></div>
    <div class="paper-meta">
      📅 2023-10-11
      | 💬 22
    </div>
    <details class="paper-abstract">
      The detection of state-sponsored trolls operating in influence campaigns on social media is a critical and unsolved challenge for the research community, which has significant implications beyond the online realm. To address this challenge, we propose a new AI-based solution that identifies troll accounts solely through behavioral cues associated with their sequences of sharing activity, encompassing both their actions and the feedback they receive from others. Our approach does not incorporate any textual content shared and consists of two steps: First, we leverage an LSTM-based classifier to determine whether account sequences belong to a state-sponsored troll or an organic, legitimate user. Second, we employ the classified sequences to calculate a metric named the "Troll Score", quantifying the degree to which an account exhibits troll-like behavior. To assess the effectiveness of our method, we examine its performance in the context of the 2016 Russian interference campaign during the U.S. Presidential election. Our experiments yield compelling results, demonstrating that our approach can identify account sequences with an AUC close to 99% and accurately differentiate between Russian trolls and organic users with an AUC of 91%. Notably, our behavioral-based approach holds a significant advantage in the ever-evolving landscape, where textual and linguistic properties can be easily mimicked by Large Language Models (LLMs): In contrast to existing language-based techniques, it relies on more challenging-to-replicate behavioral cues, ensuring greater resilience in identifying influence campaigns, especially given the potential increase in the usage of LLMs for generating inauthentic content. Finally, we assessed the generalizability of our solution to various entities driving different information operations and found promising results that will guide future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.07251v1">Ethical Reasoning over Moral Alignment: A Case and Framework for In-Context Ethical Policies in LLMs</a></div>
    <div class="paper-meta">
      📅 2023-10-11
    </div>
    <details class="paper-abstract">
      In this position paper, we argue that instead of morally aligning LLMs to specific set of ethical principles, we should infuse generic ethical reasoning capabilities into them so that they can handle value pluralism at a global scale. When provided with an ethical policy, an LLM should be capable of making decisions that are ethically consistent to the policy. We develop a framework that integrates moral dilemmas with moral principles pertaining to different foramlisms of normative ethics, and at different levels of abstractions. Initial experiments with GPT-x models shows that while GPT-4 is a nearly perfect ethical reasoner, the models still have bias towards the moral values of Western and English speaking societies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.07147v1">QFT: Quantized Full-parameter Tuning of LLMs with Affordable Resources</a></div>
    <div class="paper-meta">
      📅 2023-10-11
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have showcased remarkable impacts across a wide spectrum of natural language processing tasks. Fine-tuning these pre-trained models on downstream datasets provides further significant performance gains, but this process has been challenging due to its extraordinary resource requirements. To this end, existing efforts focus on parameter-efficient fine-tuning, which, unfortunately, fail to capitalize on the powerful potential of full-parameter fine-tuning. In this work, we propose QFT, a novel Quantized Full-parameter Tuning framework for LLMs that enables memory-efficient fine-tuning without harming performance. Our framework incorporates two novel ideas: (i) we adopt the efficient Lion optimizer, which only keeps track of the momentum and has consistent update magnitudes for each parameter, an inherent advantage for robust quantization; and (ii) we quantize all model states and store them as integer values, and present a gradient flow and parameter update scheme for the quantized weights. As a result, QFT reduces the model state memory to 21% of the standard solution while achieving comparable performance, e.g., tuning a LLaMA-7B model requires only <30GB of memory, satisfied by a single A6000 GPU.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.10677v1">LLMs as Potential Brainstorming Partners for Math and Science Problems</a></div>
    <div class="paper-meta">
      📅 2023-10-10
    </div>
    <details class="paper-abstract">
      With the recent rise of widely successful deep learning models, there is emerging interest among professionals in various math and science communities to see and evaluate the state-of-the-art models' abilities to collaborate on finding or solving problems that often require creativity and thus brainstorming. While a significant chasm still exists between current human-machine intellectual collaborations and the resolution of complex math and science problems, such as the six unsolved Millennium Prize Problems, our initial investigation into this matter reveals a promising step towards bridging the divide. This is due to the recent advancements in Large Language Models (LLMs). More specifically, we conduct comprehensive case studies to explore both the capabilities and limitations of the current state-of-the-art LLM, notably GPT-4, in collective brainstorming with humans.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.06987v1">Catastrophic Jailbreak of Open-source LLMs via Exploiting Generation</a></div>
    <div class="paper-meta">
      📅 2023-10-10
    </div>
    <details class="paper-abstract">
      The rapid progress in open-source large language models (LLMs) is significantly advancing AI development. Extensive efforts have been made before model release to align their behavior with human values, with the primary goal of ensuring their helpfulness and harmlessness. However, even carefully aligned models can be manipulated maliciously, leading to unintended behaviors, known as "jailbreaks". These jailbreaks are typically triggered by specific text inputs, often referred to as adversarial prompts. In this work, we propose the generation exploitation attack, an extremely simple approach that disrupts model alignment by only manipulating variations of decoding methods. By exploiting different generation strategies, including varying decoding hyper-parameters and sampling methods, we increase the misalignment rate from 0% to more than 95% across 11 language models including LLaMA2, Vicuna, Falcon, and MPT families, outperforming state-of-the-art attacks with $30\times$ lower computational cost. Finally, we propose an effective alignment method that explores diverse generation strategies, which can reasonably reduce the misalignment rate under our attack. Altogether, our study underscores a major failure in current safety evaluation and alignment procedures for open-source LLMs, strongly advocating for more comprehensive red teaming and better alignment before releasing such models. Our code is available at https://github.com/Princeton-SysML/Jailbreak_LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.06936v1">LLMs Killed the Script Kiddie: How Agents Supported by Large Language Models Change the Landscape of Network Threat Testing</a></div>
    <div class="paper-meta">
      📅 2023-10-10
    </div>
    <details class="paper-abstract">
      In this paper, we explore the potential of Large Language Models (LLMs) to reason about threats, generate information about tools, and automate cyber campaigns. We begin with a manual exploration of LLMs in supporting specific threat-related actions and decisions. We proceed by automating the decision process in a cyber campaign. We present prompt engineering approaches for a plan-act-report loop for one action of a threat campaign and and a prompt chaining design that directs the sequential decision process of a multi-action campaign. We assess the extent of LLM's cyber-specific knowledge w.r.t the short campaign we demonstrate and provide insights into prompt design for eliciting actionable responses. We discuss the potential impact of LLMs on the threat landscape and the ethical considerations of using LLMs for accelerating threat actor capabilities. We report a promising, yet concerning, application of generative AI to cyber threats. However, the LLM's capabilities to deal with more complex networks, sophisticated vulnerabilities, and the sensitivity of prompts are open questions. This research should spur deliberations over the inevitable advancements in LLM-supported cyber adversarial landscape.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.13160v2">Can ChatGPT Defend its Belief in Truth? Evaluating LLM Reasoning via Debate</a></div>
    <div class="paper-meta">
      📅 2023-10-10
      | 💬 EMNLP-23 (findings)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) such as ChatGPT and GPT-4 have shown impressive performance in complex reasoning tasks. However, it is difficult to know whether the models are reasoning based on deep understandings of truth and logic, or leveraging their memorized patterns in a relatively superficial way. In this work, we explore testing LLMs' reasoning by engaging with them in a debate-like conversation, where given a question, the LLM and the user need to discuss to make the correct decision starting from opposing arguments. Upon mitigating the Clever Hans effect, our task requires the LLM to not only achieve the correct answer on its own, but also be able to hold and defend its belief instead of blindly believing or getting misled by the user's (invalid) arguments and critiques, thus testing in greater depth whether the LLM grasps the essence of the reasoning required to solve the problem. Across a range of complex reasoning benchmarks spanning math, commonsense, logic and BIG-Bench tasks, we find that despite their impressive performance as reported in existing work on generating correct step-by-step solutions in the beginning, LLMs like ChatGPT cannot maintain their beliefs in truth for a significant portion of examples when challenged by oftentimes absurdly invalid arguments. Our work points to danger zones of model alignment, and also suggests more careful treatments and interpretations of the recent findings that LLMs can improve their responses based on feedback.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.06646v1">Forgetful Large Language Models: Lessons Learned from Using LLMs in Robot Programming</a></div>
    <div class="paper-meta">
      📅 2023-10-10
      | 💬 9 pages ,8 figures, accepted by the AAAI 2023 Fall Symposium Series
    </div>
    <details class="paper-abstract">
      Large language models offer new ways of empowering people to program robot applications-namely, code generation via prompting. However, the code generated by LLMs is susceptible to errors. This work reports a preliminary exploration that empirically characterizes common errors produced by LLMs in robot programming. We categorize these errors into two phases: interpretation and execution. In this work, we focus on errors in execution and observe that they are caused by LLMs being "forgetful" of key information provided in user prompts. Based on this observation, we propose prompt engineering tactics designed to reduce errors in execution. We then demonstrate the effectiveness of these tactics with three language models: ChatGPT, Bard, and LLaMA-2. Finally, we discuss lessons learned from using LLMs in robot programming and call for the benchmarking of LLM-powered end-user development of robot applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.06556v1">Gender, Age, and Technology Education Influence the Adoption and Appropriation of LLMs</a></div>
    <div class="paper-meta">
      📅 2023-10-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) such as ChatGPT have become increasingly integrated into critical activities of daily life, raising concerns about equitable access and utilization across diverse demographics. This study investigates the usage of LLMs among 1,500 representative US citizens. Remarkably, 42% of participants reported utilizing an LLM. Our findings reveal a gender gap in LLM technology adoption (more male users than female users) with complex interaction patterns regarding age. Technology-related education eliminates the gender gap in our sample. Moreover, expert users are more likely than novices to list professional tasks as typical application scenarios, suggesting discrepancies in effective usage at the workplace. These results underscore the importance of providing education in artificial intelligence in our technology-driven society to promote equitable access to and benefits from LLMs. We urge for both international replication beyond the US and longitudinal observation of adoption.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.06504v1">Revisit Input Perturbation Problems for LLMs: A Unified Robustness Evaluation Framework for Noisy Slot Filling Task</a></div>
    <div class="paper-meta">
      📅 2023-10-10
      | 💬 Accepted at NLPCC 2023 (Oral Presentation)
    </div>
    <details class="paper-abstract">
      With the increasing capabilities of large language models (LLMs), these high-performance models have achieved state-of-the-art results on a wide range of natural language processing (NLP) tasks. However, the models' performance on commonly-used benchmark datasets often fails to accurately reflect their reliability and robustness when applied to real-world noisy data. To address these challenges, we propose a unified robustness evaluation framework based on the slot-filling task to systematically evaluate the dialogue understanding capability of LLMs in diverse input perturbation scenarios. Specifically, we construct a input perturbation evaluation dataset, Noise-LLM, which contains five types of single perturbation and four types of mixed perturbation data. Furthermore, we utilize a multi-level data augmentation method (character, word, and sentence levels) to construct a candidate data pool, and carefully design two ways of automatic task demonstration construction strategies (instance-level and entity-level) with various prompt templates. Our aim is to assess how well various robustness methods of LLMs perform in real-world noisy scenarios. The experiments have demonstrated that the current open-source LLMs generally achieve limited perturbation robustness performance. Based on these experimental observations, we make some forward-looking suggestions to fuel the research in this direction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.06500v1">MetaAgents: Simulating Interactions of Human Behaviors for LLM-based Task-oriented Coordination via Collaborative Generative Agents</a></div>
    <div class="paper-meta">
      📅 2023-10-10
    </div>
    <details class="paper-abstract">
      Significant advancements have occurred in the application of Large Language Models (LLMs) for various tasks and social simulations. Despite this, their capacities to coordinate within task-oriented social contexts are under-explored. Such capabilities are crucial if LLMs are to effectively mimic human-like social behavior and produce meaningful results. To bridge this gap, we introduce collaborative generative agents, endowing LLM-based Agents with consistent behavior patterns and task-solving abilities. We situate these agents in a simulated job fair environment as a case study to scrutinize their coordination skills. We propose a novel framework that equips collaborative generative agents with human-like reasoning abilities and specialized skills. Our evaluation demonstrates that these agents show promising performance. However, we also uncover limitations that hinder their effectiveness in more complex coordination tasks. Our work provides valuable insights into the role and evolution of LLMs in task-oriented social simulations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.06391v1">Improved prompting and process for writing user personas with LLMs, using qualitative interviews: Capturing behaviour and personality traits of users</a></div>
    <div class="paper-meta">
      📅 2023-10-10
    </div>
    <details class="paper-abstract">
      This draft paper presents a workflow for creating User Personas with Large Language Models, using the results of a Thematic Analysis of qualitative interviews. The proposed workflow uses improved prompting and a larger pool of Themes, compared to previous work conducted by the author for the same task. This is possible due to the capabilities of a recently released LLM which allows the processing of 16 thousand tokens (GPT3.5-Turbo-16k) and also due to the possibility to offer a refined prompting for the creation of Personas. The paper offers details of performing Phase 2 and 3 of Thematic Analysis, and then discusses the improved workflow for creating Personas. The paper also offers some reflections on the relationship between the proposed process and existing approaches to Personas such as the data-driven and qualitative Personas. Moreover, the paper offers reflections on the capacity of LLMs to capture user behaviours and personality traits, from the underlying dataset of qualitative interviews used for the analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.06310v1">Can LLMs Demystify Bug Reports?</a></div>
    <div class="paper-meta">
      📅 2023-10-10
    </div>
    <details class="paper-abstract">
      Bugs are notoriously challenging: they slow down software users and result in time-consuming investigations for developers. These challenges are exacerbated when bugs must be reported in natural language by users. Indeed, we lack reliable tools to automatically address reported bugs (i.e., enabling their analysis, reproduction, and bug fixing). With the recent promises created by LLMs such as ChatGPT for various tasks, including in software engineering, we ask ourselves: What if ChatGPT could understand bug reports and reproduce them? This question will be the main focus of this study. To evaluate whether ChatGPT is capable of catching the semantics of bug reports, we used the popular Defects4J benchmark with its bug reports. Our study has shown that ChatGPT was able to demystify and reproduce 50% of the reported bugs. ChatGPT being able to automatically address half of the reported bugs shows promising potential in the direction of applying machine learning to address bugs with only a human-in-the-loop to report the bug.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.11186v2">Compress, Then Prompt: Improving Accuracy-Efficiency Trade-off of LLM Inference with Transferable Prompt</a></div>
    <div class="paper-meta">
      📅 2023-10-10
    </div>
    <details class="paper-abstract">
      While the numerous parameters in Large Language Models (LLMs) contribute to their superior performance, this massive scale makes them inefficient and memory-hungry. Thus, they are hard to deploy on commodity hardware, such as one single GPU. Given the memory and power constraints of such devices, model compression methods are widely employed to reduce both the model size and inference latency, which essentially trades off model quality in return for improved efficiency. Thus, optimizing this accuracy-efficiency trade-off is crucial for the LLM deployment on commodity hardware. In this paper, we introduce a new perspective to optimize this trade-off by prompting compressed models. Specifically, we first observe that for certain questions, the generation quality of a compressed LLM can be significantly improved by adding carefully designed hard prompts, though this isn't the case for all questions. Based on this observation, we propose a soft prompt learning method where we expose the compressed model to the prompt learning process, aiming to enhance the performance of prompts. Our experimental analysis suggests our soft prompt strategy greatly improves the performance of the 8x compressed LLaMA-7B model (with a joint 4-bit quantization and 50% weight pruning compression), allowing them to match their uncompressed counterparts on popular benchmarks. Also, we demonstrate that these learned prompts can be transferred across various datasets, tasks, and compression levels. Hence with this transferability, we can stitch the soft prompt to a newly compressed model to improve the test-time accuracy in an ``in-situ'' way.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.00212v3">Pairwise Proximal Policy Optimization: Harnessing Relative Feedback for LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2023-10-10
      | 💬 19 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can acquire extensive world knowledge through pre-training on large corpora. However, due to exposure to low-quality data, LLMs may exhibit harmful behavior without aligning with human values. The dominant approach for steering LLMs towards beneficial behavior involves Reinforcement Learning with Human Feedback (RLHF), with Proximal Policy Optimization (PPO) serving as the default RL optimizer. Despite its effectiveness, PPO has limitations when optimizing rewards trained from comparison-based loss. Primarily, PPO is not invariant to equivalent reward functions containing identical preference information due to the need to calibrate the reward scale. Additionally, PPO's necessity for token-wise updates introduces complexity in both function approximation and algorithm design compared to trajectory-wise optimization. This paper proposes a new framework, reinforcement learning with relative feedback, and a novel trajectory-wise policy gradient algorithm, Pairwise Proximal Policy Optimization (P3O) that operates directly on comparative rewards. We show theoretically that P3O is invariant to equivalent rewards and avoids the complexity of PPO. Empirical evaluations demonstrate that P3O outperforms PPO in the KL-Reward trade-off and can align with human preferences as well as or better than prior methods. In summary, this work introduces a simpler yet effective approach for aligning LLMs to human preferences through relative feedback.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.06147v1">Reinforcement Learning in the Era of LLMs: What is Essential? What is needed? An RL Perspective on RLHF, Prompting, and Beyond</a></div>
    <div class="paper-meta">
      📅 2023-10-09
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have garnered wide attention and led to successful products such as ChatGPT and GPT-4. Their proficiency in adhering to instructions and delivering harmless, helpful, and honest (3H) responses can largely be attributed to the technique of Reinforcement Learning from Human Feedback (RLHF). In this paper, we aim to link the research in conventional RL to RL techniques used in LLM research. Demystify this technique by discussing why, when, and how RL excels. Furthermore, we explore potential future avenues that could either benefit from or contribute to RLHF research. Highlighted Takeaways: 1. RLHF is Online Inverse RL with Offline Demonstration Data. 2. RLHF $>$ SFT because Imitation Learning (and Inverse RL) $>$ Behavior Cloning (BC) by alleviating the problem of compounding error. 3. The RM step in RLHF generates a proxy of the expensive human feedback, such an insight can be generalized to other LLM tasks such as prompting evaluation and optimization where feedback is also expensive. 4. The policy learning in RLHF is more challenging than conventional problems studied in IRL due to their high action dimensionality and feedback sparsity. 5. The main superiority of PPO over off-policy value-based methods is its stability gained from (almost) on-policy data and conservative policy updates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.06046v1">LLM for SoC Security: A Paradigm Shift</a></div>
    <div class="paper-meta">
      📅 2023-10-09
      | 💬 42 pages
    </div>
    <details class="paper-abstract">
      As the ubiquity and complexity of system-on-chip (SoC) designs increase across electronic devices, the task of incorporating security into an SoC design flow poses significant challenges. Existing security solutions are inadequate to provide effective verification of modern SoC designs due to their limitations in scalability, comprehensiveness, and adaptability. On the other hand, Large Language Models (LLMs) are celebrated for their remarkable success in natural language understanding, advanced reasoning, and program synthesis tasks. Recognizing an opportunity, our research delves into leveraging the emergent capabilities of Generative Pre-trained Transformers (GPTs) to address the existing gaps in SoC security, aiming for a more efficient, scalable, and adaptable methodology. By integrating LLMs into the SoC security verification paradigm, we open a new frontier of possibilities and challenges to ensure the security of increasingly complex SoCs. This paper offers an in-depth analysis of existing works, showcases practical case studies, demonstrates comprehensive experiments, and provides useful promoting guidelines. We also present the achievements, prospects, and challenges of employing LLM in different SoC security verification tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.05853v1">"Mango Mango, How to Let The Lettuce Dry Without A Spinner?'': Exploring User Perceptions of Using An LLM-Based Conversational Assistant Toward Cooking Partner</a></div>
    <div class="paper-meta">
      📅 2023-10-09
      | 💬 Under submission to CHI2024
    </div>
    <details class="paper-abstract">
      The rapid advancement of the Large Language Model (LLM) has created numerous potentials for integration with conversational assistants (CAs) assisting people in their daily tasks, particularly due to their extensive flexibility. However, users' real-world experiences interacting with these assistants remain unexplored. In this research, we chose cooking, a complex daily task, as a scenario to investigate people's successful and unsatisfactory experiences while receiving assistance from an LLM-based CA, Mango Mango. We discovered that participants value the system's ability to provide extensive information beyond the recipe, offer customized instructions based on context, and assist them in dynamically planning the task. However, they expect the system to be more adaptive to oral conversation and provide more suggestive responses to keep users actively involved. Recognizing that users began treating our LLM-CA as a personal assistant or even a partner rather than just a recipe-reading tool, we propose several design considerations for future development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.10673v1">Towards Emotion-Based Synthetic Consciousness: Using LLMs to Estimate Emotion Probability Vectors</a></div>
    <div class="paper-meta">
      📅 2023-10-09
      | 💬 Emotion descriptors derived from LLMs are likely to be used in governing robot behaviour. Vectorizing emotion descriptors provides a finer grained representation of state than current 'positive vs negative' situation assessments. Finer grained geometric representation of state will be necessary for future human robot interaction governance
    </div>
    <details class="paper-abstract">
      This paper shows how LLMs (Large Language Models) may be used to estimate a summary of the emotional state associated with piece of text. The summary of emotional state is a dictionary of words used to describe emotion together with the probability of the word appearing after a prompt comprising the original text and an emotion eliciting tail. Through emotion analysis of Amazon product reviews we demonstrate emotion descriptors can be mapped into a PCA type space. It was hoped that text descriptions of actions to improve a current text described state could also be elicited through a tail prompt. Experiment seemed to indicate that this is not straightforward to make work. This failure put our hoped for selection of action via choosing the best predict ed outcome via comparing emotional responses out of reach for the moment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.05146v1">Large Language Model (LLM) as a System of Multiple Expert Agents: An Approach to solve the Abstraction and Reasoning Corpus (ARC) Challenge</a></div>
    <div class="paper-meta">
      📅 2023-10-08
      | 💬 6 main pages, 1 page references, 18 pages appendix
    </div>
    <details class="paper-abstract">
      We attempt to solve the Abstraction and Reasoning Corpus (ARC) Challenge using Large Language Models (LLMs) as a system of multiple expert agents. Using the flexibility of LLMs to be prompted to do various novel tasks using zero-shot, few-shot, context-grounded prompting, we explore the feasibility of using LLMs to solve the ARC Challenge. We firstly convert the input image into multiple suitable text-based abstraction spaces. We then utilise the associative power of LLMs to derive the input-output relationship and map this to actions in the form of a working program, similar to Voyager / Ghost in the MineCraft. In addition, we use iterative environmental feedback in order to guide LLMs to solve the task. Our proposed approach achieves 50 solves out of 111 training set problems (45%) with just three abstraction spaces - grid, object and pixel - and we believe that with more abstraction spaces and learnable actions, we will be able to solve more.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.04945v1">Balancing Specialized and General Skills in LLMs: The Impact of Modern Tuning and Data Strategy</a></div>
    <div class="paper-meta">
      📅 2023-10-07
    </div>
    <details class="paper-abstract">
      This paper introduces a multifaceted methodology for fine-tuning and evaluating large language models (LLMs) for specialized monetization tasks. The goal is to balance general language proficiency with domain-specific skills. The methodology has three main components: 1) Carefully blending in-domain and general-purpose data during fine-tuning to achieve an optimal balance between general and specialized capabilities; 2) Designing a comprehensive evaluation framework with 45 questions tailored to assess performance on functionally relevant dimensions like reliability, consistency, and business impact; 3) Analyzing how model size and continual training influence metrics to guide efficient resource allocation during fine-tuning. The paper details the design, data collection, analytical techniques, and results validating the proposed frameworks. It aims to provide businesses and researchers with actionable insights on effectively adapting LLMs for specialized contexts. We also intend to make public the comprehensive evaluation framework, which includes the 45 tailored questions and their respective scoring guidelines, to foster transparency and collaboration in adapting LLMs for specialized tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.04836v1">Dual Grained Quantization: Efficient Fine-Grained Quantization for LLM</a></div>
    <div class="paper-meta">
      📅 2023-10-07
      | 💬 15 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) pose significant hardware challenges related to memory requirements and computational ability. There are two mainstream quantization schemes for LLMs: coarse-grained ($\textit{e.g.,}$ channel-wise) quantization and fine-grained ($\textit{e.g.,}$ group-wise) quantization. Fine-grained quantization has smaller quantization loss, consequently achieving superior performance. However, when applied to weight-activation quantization, it disrupts continuous integer matrix multiplication, leading to inefficient inference. In this paper, we introduce Dual Grained Quantization (DGQ), a novel A8W4 quantization for LLM that maintains superior performance while ensuring fast inference speed. DSQ dequantizes the fine-grained INT4 weight into coarse-grained INT8 representation and preform matrix multiplication using INT8 kernels. Besides, we develop a two-phase grid search algorithm to simplify the determination of fine-grained and coarse-grained quantization scales. We also devise a percentile clipping schema for smoothing the activation outliers without the need for complex optimization techniques. Experimental results demonstrate that DGQ consistently outperforms prior methods across various LLM architectures and a wide range of tasks. Remarkably, by our implemented efficient CUTLASS kernel, we achieve $\textbf{1.12}$ $\times$ memory reduction and $\textbf{3.24}$ $\times$ speed gains comparing A16W4 implementation. These advancements enable efficient deployment of A8W4 LLMs for real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.04276v1">From task structures to world models: What do LLMs know?</a></div>
    <div class="paper-meta">
      📅 2023-10-06
    </div>
    <details class="paper-abstract">
      In what sense does a large language model have knowledge? The answer to this question extends beyond the capabilities of a particular AI system, and challenges our assumptions about the nature of knowledge and intelligence. We answer by granting LLMs "instrumental knowledge"; knowledge defined by a certain set of abilities. We then ask how such knowledge is related to the more ordinary, "worldly" knowledge exhibited by human agents, and explore this in terms of the degree to which instrumental knowledge can be said to incorporate the structured world models of cognitive science. We discuss ways LLMs could recover degrees of worldly knowledge, and suggest such recovery will be governed by an implicit, resource-rational tradeoff between world models and task demands.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.03874v1">Benchmarking a foundation LLM on its ability to re-label structure names in accordance with the AAPM TG-263 report</a></div>
    <div class="paper-meta">
      📅 2023-10-05
      | 💬 20 pages, 5 figures, 1 table
    </div>
    <details class="paper-abstract">
      Purpose: To introduce the concept of using large language models (LLMs) to re-label structure names in accordance with the American Association of Physicists in Medicine (AAPM) Task Group (TG)-263 standard, and to establish a benchmark for future studies to reference. Methods and Materials: The Generative Pre-trained Transformer (GPT)-4 application programming interface (API) was implemented as a Digital Imaging and Communications in Medicine (DICOM) storage server, which upon receiving a structure set DICOM file, prompts GPT-4 to re-label the structure names of both target volumes and normal tissues according to the AAPM TG-263. Three disease sites, prostate, head and neck, and thorax were selected for evaluation. For each disease site category, 150 patients were randomly selected for manually tuning the instructions prompt (in batches of 50) and 50 patients were randomly selected for evaluation. Structure names that were considered were those that were most likely to be relevant for studies utilizing structure contours for many patients. Results: The overall re-labeling accuracy of both target volumes and normal tissues for prostate, head and neck, and thorax cases was 96.0%, 98.5%, and 96.9% respectively. Re-labeling of target volumes was less accurate on average except for prostate - 100%, 93.1%, and 91.1% respectively. Conclusions: Given the accuracy of GPT-4 in re-labeling structure names of both target volumes and normal tissues as presented in this work, LLMs are poised to be the preferred method for standardizing structure names in radiation oncology, especially considering the rapid advancements in LLM capabilities that are likely to continue.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.17590v2">Integrating Action Knowledge and LLMs for Task Planning and Situation Handling in Open Worlds</a></div>
    <div class="paper-meta">
      📅 2023-10-05
      | 💬 arXiv admin note: substantial text overlap with arXiv:2210.01287
    </div>
    <details class="paper-abstract">
      Task planning systems have been developed to help robots use human knowledge (about actions) to complete long-horizon tasks. Most of them have been developed for "closed worlds" while assuming the robot is provided with complete world knowledge. However, the real world is generally open, and the robots frequently encounter unforeseen situations that can potentially break the planner's completeness. Could we leverage the recent advances on pre-trained Large Language Models (LLMs) to enable classical planning systems to deal with novel situations? This paper introduces a novel framework, called COWP, for open-world task planning and situation handling. COWP dynamically augments the robot's action knowledge, including the preconditions and effects of actions, with task-oriented commonsense knowledge. COWP embraces the openness from LLMs, and is grounded to specific domains via action knowledge. For systematic evaluations, we collected a dataset that includes 1,085 execution-time situations. Each situation corresponds to a state instance wherein a robot is potentially unable to complete a task using a solution that normally works. Experimental results show that our approach outperforms competitive baselines from the literature in the success rate of service tasks. Additionally, we have demonstrated COWP using a mobile manipulator. Supplementary materials are available at: https://cowplanning.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.03731v1">MathCoder: Seamless Code Integration in LLMs for Enhanced Mathematical Reasoning</a></div>
    <div class="paper-meta">
      📅 2023-10-05
      | 💬 The state-of-the-art open-source language models for mathematical reasoning
    </div>
    <details class="paper-abstract">
      The recently released GPT-4 Code Interpreter has demonstrated remarkable proficiency in solving challenging math problems, primarily attributed to its ability to seamlessly reason with natural language, generate code, execute code, and continue reasoning based on the execution output. In this paper, we present a method to fine-tune open-source language models, enabling them to use code for modeling and deriving math equations and, consequently, enhancing their mathematical reasoning abilities. We propose a method of generating novel and high-quality datasets with math problems and their code-based solutions, referred to as MathCodeInstruct. Each solution interleaves natural language, code, and execution results. We also introduce a customized supervised fine-tuning and inference approach. This approach yields the MathCoder models, a family of models capable of generating code-based solutions for solving challenging math problems. Impressively, the MathCoder models achieve state-of-the-art scores among open-source LLMs on the MATH (45.2%) and GSM8K (83.9%) datasets, substantially outperforming other open-source alternatives. Notably, the MathCoder model not only surpasses ChatGPT-3.5 and PaLM-2 on GSM8K and MATH but also outperforms GPT-4 on the competition-level MATH dataset. The dataset and models will be released at https://github.com/mathllm/MathCoder.
    </details>
</div>
