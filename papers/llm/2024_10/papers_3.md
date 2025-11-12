# llm - 2024_10

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18693v1">Unleashing Reasoning Capability of LLMs via Scalable Question Synthesis from Scratch</a></div>
    <div class="paper-meta">
      📅 2024-10-24
      | 💬 Preprint. Project page: https://scalequest.github.io/
    </div>
    <details class="paper-abstract">
      The availability of high-quality data is one of the most important factors in improving the reasoning capability of LLMs. Existing works have demonstrated the effectiveness of creating more instruction data from seed questions or knowledge bases. Recent research indicates that continually scaling up data synthesis from strong models (e.g., GPT-4) can further elicit reasoning performance. Though promising, the open-sourced community still lacks high-quality data at scale and scalable data synthesis methods with affordable costs. To address this, we introduce ScaleQuest, a scalable and novel data synthesis method that utilizes "small-size" (e.g., 7B) open-source models to generate questions from scratch without the need for seed data with complex augmentation constraints. With the efficient ScaleQuest, we automatically constructed a mathematical reasoning dataset consisting of 1 million problem-solution pairs, which are more effective than existing open-sourced datasets. It can universally increase the performance of mainstream open-source models (i.e., Mistral, Llama3, DeepSeekMath, and Qwen2-Math) by achieving 29.2% to 46.4% gains on MATH. Notably, simply fine-tuning the Qwen2-Math-7B-Base model with our dataset can even surpass Qwen2-Math-7B-Instruct, a strong and well-aligned model on closed-source data, and proprietary models such as GPT-4-Turbo and Claude-3.5 Sonnet.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18686v1">Hierarchical Multimodal LLMs with Semantic Space Alignment for Enhanced Time Series Classification</a></div>
    <div class="paper-meta">
      📅 2024-10-24
    </div>
    <details class="paper-abstract">
      Leveraging large language models (LLMs) has garnered increasing attention and introduced novel perspectives in time series classification. However, existing approaches often overlook the crucial dynamic temporal information inherent in time series data and face challenges in aligning this data with textual semantics. To address these limitations, we propose HiTime, a hierarchical multi-modal model that seamlessly integrates temporal information into LLMs for multivariate time series classification (MTSC). Our model employs a hierarchical feature encoder to capture diverse aspects of time series data through both data-specific and task-specific embeddings. To facilitate semantic space alignment between time series and text, we introduce a dual-view contrastive alignment module that bridges the gap between modalities. Additionally, we adopt a hybrid prompting strategy to fine-tune the pre-trained LLM in a parameter-efficient manner. By effectively incorporating dynamic temporal features and ensuring semantic alignment, HiTime enables LLMs to process continuous time series data and achieves state-of-the-art classification performance through text generation. Extensive experiments on benchmark datasets demonstrate that HiTime significantly enhances time series classification accuracy compared to most competitive baseline methods. Our findings highlight the potential of integrating temporal features into LLMs, paving the way for advanced time series analysis. The code is publicly available for further research and validation. Our codes are publicly available1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18641v1">Smart ETL and LLM-based contents classification: the European Smart Tourism Tools Observatory experience</a></div>
    <div class="paper-meta">
      📅 2024-10-24
    </div>
    <details class="paper-abstract">
      Purpose: Our research project focuses on improving the content update of the online European Smart Tourism Tools (STTs) Observatory by incorporating and categorizing STTs. The categorization is based on their taxonomy, and it facilitates the end user's search process. The use of a Smart ETL (Extract, Transform, and Load) process, where \emph{Smart} indicates the use of Artificial Intelligence (AI), is central to this endeavor. Methods: The contents describing STTs are derived from PDF catalogs, where PDF-scraping techniques extract QR codes, images, links, and text information. Duplicate STTs between the catalogs are removed, and the remaining ones are classified based on their text information using Large Language Models (LLMs). Finally, the data is transformed to comply with the Dublin Core metadata structure (the observatory's metadata structure), chosen for its wide acceptance and flexibility. Results: The Smart ETL process to import STTs to the observatory combines PDF-scraping techniques with LLMs for text content-based classification. Our preliminary results have demonstrated the potential of LLMs for text content-based classification. Conclusion: The proposed approach's feasibility is a step towards efficient content-based classification, not only in Smart Tourism but also adaptable to other fields. Future work will mainly focus on refining this classification process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18624v1">Prompting and Fine-Tuning of Small LLMs for Length-Controllable Telephone Call Summarization</a></div>
    <div class="paper-meta">
      📅 2024-10-24
      | 💬 Accepted at the The International Conference on Foundation and Large Language Models (FLLM2024)
    </div>
    <details class="paper-abstract">
      This paper explores the rapid development of a telephone call summarization system utilizing large language models (LLMs). Our approach involves initial experiments with prompting existing LLMs to generate summaries of telephone conversations, followed by the creation of a tailored synthetic training dataset utilizing stronger frontier models. We place special focus on the diversity of the generated data and on the ability to control the length of the generated summaries to meet various use-case specific requirements. The effectiveness of our method is evaluated using two state-of-the-art LLM-as-a-judge-based evaluation techniques to ensure the quality and relevance of the summaries. Our results show that fine-tuned Llama-2-7B-based summarization model performs on-par with GPT-4 in terms of factual accuracy, completeness and conciseness. Our findings demonstrate the potential for quickly bootstrapping a practical and efficient call summarization system.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15859v3">Mesa-Extrapolation: A Weave Position Encoding Method for Enhanced Extrapolation in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-24
      | 💬 Accepted by NeurIPS 2024; 13 pages and 30 pages appendix;
    </div>
    <details class="paper-abstract">
      Large language models (LLMs), although having revolutionized many fields, still suffer from the challenging extrapolation problem, where the inference ability of LLMs sharply declines beyond their max training lengths. In this work, we conduct a theoretical analysis to better understand why No Position Encoding (NoPE) fails outside its effective range, as well as examining the power of Position Encoding (PE) in this context. Our findings reveal that with meticulous weave position, PE can indeed be extended beyond effective range. Our theorems establish that LLMs equipped with weave PE can achieve improved extrapolation performance without additional cost. Furthermore, we introduce a novel weave PE method, Mesa-Extrapolation, which utilizes a chunk-based triangular attention matrix and applies Stair PE to manage the final chunk. This method not only retains competitive performance but also offers substantial benefits such as significantly reduced memory demand and faster inference speed. Extensive experiments validate the effectiveness of Mesa-Extrapolation, demonstrating its potential as a scalable solution to enhancing LLMs applicative reach. Our code is available at \url{https://github.com/soacker/Mesa-Extrapolation}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18588v1">Knowledge Distillation Using Frontier Open-source LLMs: Generalizability and the Role of Synthetic Data</a></div>
    <div class="paper-meta">
      📅 2024-10-24
      | 💬 25 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Leading open-source large language models (LLMs) such as Llama-3.1-Instruct-405B are extremely capable at generating text, answering questions, and solving a variety of natural language understanding tasks. However, they incur higher inference cost and latency compared to smaller LLMs. Knowledge distillation provides a way to use outputs from these large, capable teacher models to train smaller student models which can be used for inference at lower cost and latency, while retaining comparable accuracy. We investigate the efficacy of distillation using the Llama-3.1-405B-Instruct teacher and the smaller Llama-3.1-8B-Instruct and Llama-3.1-70B-Instruct student models. Contributions of this work include (a) We evaluate the generalizability of distillation with the above Llama-3.1 teacher-student pairs across different tasks and datasets (b) We show that using synthetic data during distillation significantly improves the accuracy of 8B and 70B models, and when used with reasoning chains, even matches or surpasses the zero-shot accuracy of 405B model on some datasets (c) We empirically show that distillation enables 8B and 70B models to internalize 405B's reasoning ability by using only standard fine-tuning (without customizing any loss function). This allows cost and latency-efficient student model inference. (d) We show pitfalls in evaluation of distillation, and present task-specific evaluation, including both human and LLM-grading, and ground-truth based traditional accuracy benchmarks. This methodical study brings out the fundamental importance of synthetic data quality in knowledge distillation, and of combining multiple, task-specific ways of accuracy and quality evaluation in assessing the effectiveness of distillation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18582v1">LLM-Aided Efficient Hardware Design Automation</a></div>
    <div class="paper-meta">
      📅 2024-10-24
    </div>
    <details class="paper-abstract">
      With the rapidly increasing complexity of modern chips, hardware engineers are required to invest more effort in tasks such as circuit design, verification, and physical implementation. These workflows often involve continuous modifications, which are labor-intensive and prone to errors. Therefore, there is an increasing need for more efficient and cost-effective Electronic Design Automation (EDA) solutions to accelerate new hardware development. Recently, large language models (LLMs) have made significant advancements in contextual understanding, logical reasoning, and response generation. Since hardware designs and intermediate scripts can be expressed in text format, it is reasonable to explore whether integrating LLMs into EDA could simplify and fully automate the entire workflow. Accordingly, this paper discusses such possibilities in several aspects, covering hardware description language (HDL) generation, code debugging, design verification, and physical implementation. Two case studies, along with their future outlook, are introduced to highlight the capabilities of LLMs in code repair and testbench generation. Finally, future directions and challenges are highlighted to further explore the potential of LLMs in shaping the next-generation EDA
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.18266v4">"Vorbeşti Româneşte?" A Recipe to Train Powerful Romanian LLMs with English Instructions</a></div>
    <div class="paper-meta">
      📅 2024-10-24
      | 💬 Accepted at The 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP 2024 Findings). arXiv admin note: text overlap with arXiv:2405.07703
    </div>
    <details class="paper-abstract">
      In recent years, Large Language Models (LLMs) have achieved almost human-like performance on various tasks. While some LLMs have been trained on multilingual data, most of the training data is in English; hence, their performance in English greatly exceeds other languages. To our knowledge, we are the first to collect and translate a large collection of texts, instructions, and benchmarks and train, evaluate, and release open-source LLMs tailored for Romanian. We evaluate our methods on four different categories, including academic benchmarks, MT-Bench (manually translated), and a professionally built historical, cultural, and social benchmark adapted to Romanian. We argue for the usefulness and high performance of RoLLMs by obtaining state-of-the-art results across the board. We publicly release all resources (i.e., data, training and evaluation code, models) to support and encourage research on Romanian LLMs while concurrently creating a generalizable recipe, adequate for other low or less-resourced languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18527v1">Probing Ranking LLMs: Mechanistic Interpretability in Information Retrieval</a></div>
    <div class="paper-meta">
      📅 2024-10-24
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      Transformer networks, especially those with performance on par with GPT models, are renowned for their powerful feature extraction capabilities. However, the nature and correlation of these features with human-engineered ones remain unclear. In this study, we delve into the mechanistic workings of state-of-the-art, fine-tuning-based passage-reranking transformer networks. Our approach involves a probing-based, layer-by-layer analysis of neurons within ranking LLMs to identify individual or groups of known human-engineered and semantic features within the network's activations. We explore a wide range of features, including lexical, document structure, query-document interaction, advanced semantic, interaction-based, and LLM-specific features, to gain a deeper understanding of the underlying mechanisms that drive ranking decisions in LLMs. Our results reveal a set of features that are prominently represented in LLM activations, as well as others that are notably absent. Additionally, we observe distinct behaviors of LLMs when processing low versus high relevance queries and when encountering out-of-distribution query and document sets. By examining these features within activations, we aim to enhance the interpretability and performance of LLMs in ranking tasks. Our findings provide valuable insights for the development of more effective and transparent ranking models, with significant implications for the broader information retrieval community. All scripts and code necessary to replicate our findings are made available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.15444v3">Cutting Through the Noise: Boosting LLM Performance on Math Word Problems</a></div>
    <div class="paper-meta">
      📅 2024-10-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel at various tasks, including solving math word problems (MWPs), but struggle with real-world problems containing irrelevant information. To address this, we propose a prompting framework that generates adversarial variants of MWPs by adding irrelevant variables. We introduce a dataset, PROBLEMATHIC, containing both adversarial and non-adversarial MWPs. Our experiments reveal that LLMs are susceptible to distraction by numerical noise, resulting in an average relative performance drop of ~26% on adversarial MWPs. To mitigate this, we fine-tune LLMs (Llama-2, Mistral) on the adversarial samples from our dataset. Fine-tuning on adversarial training instances improves performance on adversarial MWPs by ~8%, indicating increased robustness to noise and improved ability to identify relevant data for reasoning. Finally, to assess the generalizability of our prompting framework, we introduce GSM-8K-Adv, an adversarial variant of the GSM-8K benchmark. LLMs continue to struggle when faced with adversarial information, reducing performance by up to 6%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18489v1">LLM as a code generator in Agile Model Driven Development</a></div>
    <div class="paper-meta">
      📅 2024-10-24
    </div>
    <details class="paper-abstract">
      Leveraging Large Language Models (LLM) like GPT4 in the auto generation of code represents a significant advancement, yet it is not without its challenges. The ambiguity inherent in natural language descriptions of software poses substantial obstacles to generating deployable, structured artifacts. This research champions Model Driven Development (MDD) as a viable strategy to overcome these challenges, proposing an Agile Model Driven Development (AMDD) approach that employs GPT4 as a code generator. This approach enhances the flexibility and scalability of the code auto generation process and offers agility that allows seamless adaptation to changes in models or deployment environments. We illustrate this by modeling a multi agent Unmanned Vehicle Fleet (UVF) system using the Unified Modeling Language (UML), significantly reducing model ambiguity by integrating the Object Constraint Language (OCL) for code structure meta modeling, and the FIPA ontology language for communication semantics meta modeling. Applying GPT4 auto generation capabilities yields Java and Python code that is compatible with the JADE and PADE frameworks, respectively. Our thorough evaluation of the auto generated code verifies its alignment with expected behaviors and identifies enhancements in agent interactions. Structurally, we assessed the complexity of code derived from a model constrained solely by OCL meta models, against that influenced by both OCL and FIPA ontology meta models. The results indicate that the ontology constrained meta model produces inherently more complex code, yet its cyclomatic complexity remains within manageable levels, suggesting that additional meta model constraints can be incorporated without exceeding the high risk threshold for complexity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18451v1">Skywork-Reward: Bag of Tricks for Reward Modeling in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-24
    </div>
    <details class="paper-abstract">
      In this report, we introduce a collection of methods to enhance reward modeling for LLMs, focusing specifically on data-centric techniques. We propose effective data selection and filtering strategies for curating high-quality open-source preference datasets, culminating in the Skywork-Reward data collection, which contains only 80K preference pairs -- significantly smaller than existing datasets. Using this curated dataset, we developed the Skywork-Reward model series -- Skywork-Reward-Gemma-27B and Skywork-Reward-Llama-3.1-8B -- with the former currently holding the top position on the RewardBench leaderboard. Notably, our techniques and datasets have directly enhanced the performance of many top-ranked models on RewardBench, highlighting the practical impact of our contributions in real-world preference learning applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18447v1">ToolFlow: Boosting LLM Tool-Calling Through Natural and Coherent Dialogue Synthesis</a></div>
    <div class="paper-meta">
      📅 2024-10-24
    </div>
    <details class="paper-abstract">
      Supervised fine-tuning (SFT) is a common method to enhance the tool calling capabilities of Large Language Models (LLMs), with the training data often being synthesized. The current data synthesis process generally involves sampling a set of tools, formulating a requirement based on these tools, and generating the call statements. However, tools sampled randomly lack relevance, making them difficult to combine and thus reducing the diversity of the data. Additionally, current work overlooks the coherence between turns of dialogues, leading to a gap between the synthesized data and real-world scenarios. To address these issues, we propose a Graph-based Sampling strategy to sample more relevant tool combinations, and a Planned-generation strategy to create plans that guide the synthesis of coherent dialogues. We integrate these two strategies and enable multiple agents to synthesize the dialogue data interactively, resulting in our tool-calling data synthesis pipeline ToolFlow. Data quality assessments demonstrate improvements in the naturalness and coherence of our synthesized dialogues. Finally, we apply SFT on LLaMA-3.1-8B using 8,000 synthetic dialogues generated with ToolFlow. Results show that the model achieves tool-calling performance comparable to or even surpassing GPT-4, while maintaining strong general capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18436v1">Can Code-Switched Texts Activate a Knowledge Switch in LLMs? A Case Study on English-Korean Code-Switching</a></div>
    <div class="paper-meta">
      📅 2024-10-24
      | 💬 19 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Code-switching (CS), a phenomenon where multilingual speakers alternate between languages in a discourse, can convey subtle cultural and linguistic nuances that can be otherwise lost in translation. Recent state-of-the-art multilingual large language models (LLMs) demonstrate excellent multilingual abilities in various aspects including understanding CS, but the power of CS in eliciting language-specific knowledge is yet to be discovered. Therefore, we investigate the effectiveness of code-switching on a wide range of multilingual LLMs in terms of knowledge activation, or the act of identifying and leveraging knowledge for reasoning. To facilitate the research, we first present EnKoQA, a synthetic English-Korean CS question-answering dataset. We provide a comprehensive analysis on a variety of multilingual LLMs by subdividing activation process into knowledge identification and knowledge leveraging. Our experiments demonstrate that compared to English text, CS can faithfully activate knowledge inside LLMs, especially on language-specific domains. In addition, the performance gap between CS and English is larger in models that show excellent monolingual abilities, suggesting that there exists a correlation with CS and Korean proficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.03585v3">Zero-shot Persuasive Chatbots with LLM-Generated Strategies and Information Retrieval</a></div>
    <div class="paper-meta">
      📅 2024-10-24
      | 💬 Findings of EMNLP 2024
    </div>
    <details class="paper-abstract">
      Persuasion plays a pivotal role in a wide range of applications from health intervention to the promotion of social good. Persuasive chatbots employed responsibly for social good can be an enabler of positive individual and social change. Existing methods rely on fine-tuning persuasive chatbots with task-specific training data which is costly, if not infeasible, to collect. Furthermore, they employ only a handful of pre-defined persuasion strategies. We propose PersuaBot, a zero-shot chatbot based on Large Language Models (LLMs) that is factual and more persuasive by leveraging many more nuanced strategies. PersuaBot uses an LLM to first generate natural responses, from which the strategies used are extracted. To combat hallucination of LLMs, Persuabot replace any unsubstantiated claims in the response with retrieved facts supporting the extracted strategies. We applied our chatbot, PersuaBot, to three significantly different domains needing persuasion skills: donation solicitation, recommendations, and health intervention. Our experiments on simulated and human conversations show that our zero-shot approach is more persuasive than prior work, while achieving factual accuracy surpassing state-of-the-art knowledge-oriented chatbots.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18336v1">Assessing the Creativity of LLMs in Proposing Novel Solutions to Mathematical Problems</a></div>
    <div class="paper-meta">
      📅 2024-10-24
    </div>
    <details class="paper-abstract">
      The mathematical capabilities of AI systems are complex and multifaceted. Most existing research has predominantly focused on the correctness of AI-generated solutions to mathematical problems. In this work, we argue that beyond producing correct answers, AI systems should also be capable of, or assist humans in, developing novel solutions to mathematical challenges. This study explores the creative potential of Large Language Models (LLMs) in mathematical reasoning, an aspect that has received limited attention in prior research. We introduce a novel framework and benchmark, CreativeMath, which encompasses problems ranging from middle school curricula to Olympic-level competitions, designed to assess LLMs' ability to propose innovative solutions after some known solutions have been provided. Our experiments demonstrate that, while LLMs perform well on standard mathematical tasks, their capacity for creative problem-solving varies considerably. Notably, the Gemini-1.5-Pro model outperformed other LLMs in generating novel solutions. This research opens a new frontier in evaluating AI creativity, shedding light on both the strengths and limitations of LLMs in fostering mathematical innovation, and setting the stage for future developments in AI-assisted mathematical discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.12665v2">CollabStory: Multi-LLM Collaborative Story Generation and Authorship Analysis</a></div>
    <div class="paper-meta">
      📅 2024-10-23
    </div>
    <details class="paper-abstract">
      The rise of unifying frameworks that enable seamless interoperability of Large Language Models (LLMs) has made LLM-LLM collaboration for open-ended tasks a possibility. Despite this, there have not been efforts to explore such collaborative writing. We take the next step beyond human-LLM collaboration to explore this multi-LLM scenario by generating the first exclusively LLM-generated collaborative stories dataset called CollabStory. We focus on single-author ($N=1$) to multi-author (up to $N=5$) scenarios, where multiple LLMs co-author stories. We generate over 32k stories using open-source instruction-tuned LLMs. Further, we take inspiration from the PAN tasks that have set the standard for human-human multi-author writing tasks and analysis. We extend their authorship-related tasks for multi-LLM settings and present baselines for LLM-LLM collaboration. We find that current baselines are not able to handle this emerging scenario. Thus, CollabStory is a resource that could help propel an understanding as well as the development of techniques to discern the use of multiple LLMs. This is crucial to study in the context of writing tasks since LLM-LLM collaboration could potentially overwhelm ongoing challenges related to plagiarism detection, credit assignment, maintaining academic integrity in educational settings, and addressing copyright infringement concerns. We make our dataset and code available at \texttt{\url{https://github.com/saranya-venkatraman/multi_llm_story_writing}}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18218v1">Optimizing the role of human evaluation in LLM-based spoken document summarization systems</a></div>
    <div class="paper-meta">
      📅 2024-10-23
    </div>
    <details class="paper-abstract">
      The emergence of powerful LLMs has led to a paradigm shift in abstractive summarization of spoken documents. The properties that make LLMs so valuable for this task -- creativity, ability to produce fluent speech, and ability to abstract information from large corpora -- also present new challenges to evaluating their content. Quick, cost-effective automatic evaluations such as ROUGE and BERTScore offer promise, but do not yet show competitive performance when compared to human evaluations. We draw on methodologies from the social sciences to propose an evaluation paradigm for spoken document summarization explicitly tailored for generative AI content. We provide detailed evaluation criteria and best practices guidelines to ensure robustness in the experimental design, replicability, and trustworthiness of human evaluation studies. We additionally include two case studies that show how these human-in-the-loop evaluation methods have been implemented at a major U.S. technology company.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18215v1">Advancing NLP Security by Leveraging LLMs as Adversarial Engines</a></div>
    <div class="paper-meta">
      📅 2024-10-23
      | 💬 5 pages
    </div>
    <details class="paper-abstract">
      This position paper proposes a novel approach to advancing NLP security by leveraging Large Language Models (LLMs) as engines for generating diverse adversarial attacks. Building upon recent work demonstrating LLMs' effectiveness in creating word-level adversarial examples, we argue for expanding this concept to encompass a broader range of attack types, including adversarial patches, universal perturbations, and targeted attacks. We posit that LLMs' sophisticated language understanding and generation capabilities can produce more effective, semantically coherent, and human-like adversarial examples across various domains and classifier architectures. This paradigm shift in adversarial NLP has far-reaching implications, potentially enhancing model robustness, uncovering new vulnerabilities, and driving innovation in defense mechanisms. By exploring this new frontier, we aim to contribute to the development of more secure, reliable, and trustworthy NLP systems for critical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18210v1">Towards Understanding the Fragility of Multilingual LLMs against Fine-Tuning Attacks</a></div>
    <div class="paper-meta">
      📅 2024-10-23
      | 💬 14 pages, 6 figures, 7 tables
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have sparked widespread concerns about their safety. Recent work demonstrates that safety alignment of LLMs can be easily removed by fine-tuning with a few adversarially chosen instruction-following examples, i.e., fine-tuning attacks. We take a further step to understand fine-tuning attacks in multilingual LLMs. We first discover cross-lingual generalization of fine-tuning attacks: using a few adversarially chosen instruction-following examples in one language, multilingual LLMs can also be easily compromised (e.g., multilingual LLMs fail to refuse harmful prompts in other languages). Motivated by this finding, we hypothesize that safety-related information is language-agnostic and propose a new method termed Safety Information Localization (SIL) to identify the safety-related information in the model parameter space. Through SIL, we validate this hypothesis and find that only changing 20% of weight parameters in fine-tuning attacks can break safety alignment across all languages. Furthermore, we provide evidence to the alternative pathways hypothesis for why freezing safety-related parameters does not prevent fine-tuning attacks, and we demonstrate that our attack vector can still jailbreak LLMs adapted to new languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.12320v3">TensorOpera Router: A Multi-Model Router for Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2024-10-23
      | 💬 14 pages, 7 figures, 2 tables
    </div>
    <details class="paper-abstract">
      With the rapid growth of Large Language Models (LLMs) across various domains, numerous new LLMs have emerged, each possessing domain-specific expertise. This proliferation has highlighted the need for quick, high-quality, and cost-effective LLM query response methods. Yet, no single LLM exists to efficiently balance this trilemma. Some models are powerful but extremely costly, while others are fast and inexpensive but qualitatively inferior. To address this challenge, we present TO-Router, a non-monolithic LLM querying system that seamlessly integrates various LLM experts into a single query interface and dynamically routes incoming queries to the most high-performant expert based on query's requirements. Through extensive experiments, we demonstrate that when compared to standalone expert models, TO-Router improves query efficiency by up to 40\%, and leads to significant cost reductions of up to 30%, while maintaining or enhancing model performance by up to 10%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.19366v2">ECG Semantic Integrator (ESI): A Foundation ECG Model Pretrained with LLM-Enhanced Cardiological Text</a></div>
    <div class="paper-meta">
      📅 2024-10-23
    </div>
    <details class="paper-abstract">
      The utilization of deep learning on electrocardiogram (ECG) analysis has brought the advanced accuracy and efficiency of cardiac healthcare diagnostics. By leveraging the capabilities of deep learning in semantic understanding, especially in feature extraction and representation learning, this study introduces a new multimodal contrastive pretaining framework that aims to improve the quality and robustness of learned representations of 12-lead ECG signals. Our framework comprises two key components, including Cardio Query Assistant (CQA) and ECG Semantics Integrator(ESI). CQA integrates a retrieval-augmented generation (RAG) pipeline to leverage large language models (LLMs) and external medical knowledge to generate detailed textual descriptions of ECGs. The generated text is enriched with information about demographics and waveform patterns. ESI integrates both contrastive and captioning loss to pretrain ECG encoders for enhanced representations. We validate our approach through various downstream tasks, including arrhythmia detection and ECG-based subject identification. Our experimental results demonstrate substantial improvements over strong baselines in these tasks. These baselines encompass supervised and self-supervised learning methods, as well as prior multimodal pretraining approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18040v1">Key Algorithms for Keyphrase Generation: Instruction-Based LLMs for Russian Scientific Keyphrases</a></div>
    <div class="paper-meta">
      📅 2024-10-23
      | 💬 The 12th International Conference on Analysis of Images, Social Networks and Texts (AIST'2024)
    </div>
    <details class="paper-abstract">
      Keyphrase selection is a challenging task in natural language processing that has a wide range of applications. Adapting existing supervised and unsupervised solutions for the Russian language faces several limitations due to the rich morphology of Russian and the limited number of training datasets available. Recent studies conducted on English texts show that large language models (LLMs) successfully address the task of generating keyphrases. LLMs allow achieving impressive results without task-specific fine-tuning, using text prompts instead. In this work, we access the performance of prompt-based methods for generating keyphrases for Russian scientific abstracts. First, we compare the performance of zero-shot and few-shot prompt-based methods, fine-tuned models, and unsupervised methods. Then we assess strategies for selecting keyphrase examples in a few-shot setting. We present the outcomes of human evaluation of the generated keyphrases and analyze the strengths and weaknesses of the models through expert assessment. Our results suggest that prompt-based methods can outperform common baselines even using simple text prompts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18038v1">POD-Attention: Unlocking Full Prefill-Decode Overlap for Faster LLM Inference</a></div>
    <div class="paper-meta">
      📅 2024-10-23
    </div>
    <details class="paper-abstract">
      Each request in LLM inference goes through two phases: compute-bound prefill and memory-bandwidth-bound decode. To improve GPU utilization, recent systems use hybrid batching that combines the prefill and decode phases of different requests into the same batch. Hybrid batching works well for linear operations as it amortizes the cost of loading model weights from HBM. However, attention computation in hybrid batches remains inefficient because existing attention kernels are optimized for either prefill or decode. In this paper, we present POD-Attention -- the first GPU kernel that efficiently computes attention for hybrid batches. POD-Attention aims to maximize the utilization of both compute and memory bandwidth by carefully allocating the GPU's resources such that prefill and decode operations happen concurrently on the same multiprocessor. We integrate POD-Attention in a state-of-the-art LLM inference scheduler Sarathi-Serve. POD-Attention speeds up attention computation by up to 75% (mean 28%) and increases LLM serving throughput by up to 22% in offline inference. In online inference, POD-Attention enables lower time-to-first-token (TTFT), time-between-tokens (TBT), and request execution latency versus Sarathi-Serve.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.16664v3">LLM-Assisted Multi-Teacher Continual Learning for Visual Question Answering in Robotic Surgery</a></div>
    <div class="paper-meta">
      📅 2024-10-23
      | 💬 This paper has been accapted by 2024 IEEE International Conference on Robotics and Automation (ICRA)
    </div>
    <details class="paper-abstract">
      Visual question answering (VQA) is crucial for promoting surgical education. In practice, the needs of trainees are constantly evolving, such as learning more surgical types, adapting to different robots, and learning new surgical instruments and techniques for various surgeries. However, patient data privacy often restricts the availability of old data when updating the model, necessitating an exemplar-free continual learning (CL) setup. Prior CL studies overlooked two vital problems in the surgical domain: 1) large domain shifts from diverse surgical operations collected from multiple sources, and 2) severe data imbalance arising from the uneven presence of surgical instruments or activities. This paper proposes addressing these problems with a multimodal large language model (LLM) and an adaptive weight assignment methodology. We first develop a new multi-teacher CL framework that leverages a multimodal LLM as the additional teacher. The strong generalization ability of the LLM can bridge the knowledge gap when domain shifts and data imbalances occur. We then put forth a novel data processing method that transforms complex LLM embeddings into logits compatible with our CL framework. We further design an adaptive weight assignment approach that balances the generalization ability of the LLM and the domain expertise of the old CL model. Finally, to comprehensively test the effectiveness of our proposed method, we have also constructed two new surgical VQA datasets that are largely different from existing ones and could be valuable resources for future research. Extensive experimental results on the tested datasets demonstrate the superiority of our method to other advanced CL schemes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.17950v1">Benchmarking Floworks against OpenAI & Anthropic: A Novel Framework for Enhanced LLM Function Calling</a></div>
    <div class="paper-meta">
      📅 2024-10-23
      | 💬 15 pages for main paper, 21 pages in total including references and appendix, 10 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable capabilities in various domains, yet their economic impact has been limited by challenges in tool use and function calling. This paper introduces ThorV2, a novel architecture that significantly enhances LLMs' function calling abilities. We develop a comprehensive benchmark focused on HubSpot CRM operations to evaluate ThorV2 against leading models from OpenAI and Anthropic. Our results demonstrate that ThorV2 outperforms existing models in accuracy, reliability, latency, and cost efficiency for both single and multi-API calling tasks. We also show that ThorV2 is far more reliable and scales better to multistep tasks compared to traditional models. Our work offers the tantalizing possibility of more accurate function-calling compared to today's best-performing models using significantly smaller LLMs. These advancements have significant implications for the development of more capable AI assistants and the broader application of LLMs in real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.04957v3">Reconfidencing LLMs from the Grouping Loss Perspective</a></div>
    <div class="paper-meta">
      📅 2024-10-23
      | 💬 EMNLP 2024 Findings
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), including ChatGPT and LLaMA, are susceptible to generating hallucinated answers in a confident tone. While efforts to elicit and calibrate confidence scores have proven useful, recent findings show that controlling uncertainty must go beyond calibration: predicted scores may deviate significantly from the actual posterior probabilities due to the impact of grouping loss. In this work, we construct a new evaluation dataset derived from a knowledge base to assess confidence scores given to answers of Mistral and LLaMA. Experiments show that they tend to be overconfident. Further, we show that they are more overconfident on some answers than others, \emph{eg} depending on the nationality of the person in the query. In uncertainty-quantification theory, this is grouping loss. To address this, we propose a solution to reconfidence LLMs, canceling not only calibration but also grouping loss. The LLMs, after the reconfidencing process, indicate improved confidence alignment with the accuracy of their responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.14703v2">Do LLMs Have Distinct and Consistent Personality? TRAIT: Personality Testset designed for LLMs with Psychometrics</a></div>
    <div class="paper-meta">
      📅 2024-10-23
      | 💬 Preprint; Under review
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have led to their adaptation in various domains as conversational agents. We wonder: can personality tests be applied to these agents to analyze their behavior, similar to humans? We introduce TRAIT, a new benchmark consisting of 8K multi-choice questions designed to assess the personality of LLMs. TRAIT is built on two psychometrically validated small human questionnaires, Big Five Inventory (BFI) and Short Dark Triad (SD-3), enhanced with the ATOMIC-10X knowledge graph to a variety of real-world scenarios. TRAIT also outperforms existing personality tests for LLMs in terms of reliability and validity, achieving the highest scores across four key metrics: Content Validity, Internal Validity, Refusal Rate, and Reliability. Using TRAIT, we reveal two notable insights into personalities of LLMs: 1) LLMs exhibit distinct and consistent personality, which is highly influenced by their training data (e.g., data used for alignment tuning), and 2) current prompting techniques have limited effectiveness in eliciting certain traits, such as high psychopathy or low conscientiousness, suggesting the need for further research in this direction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.05013v1">Enhancing literature review with LLM and NLP methods. Algorithmic trading case</a></div>
    <div class="paper-meta">
      📅 2024-10-23
    </div>
    <details class="paper-abstract">
      This study utilizes machine learning algorithms to analyze and organize knowledge in the field of algorithmic trading. By filtering a dataset of 136 million research papers, we identified 14,342 relevant articles published between 1956 and Q1 2020. We compare traditional practices-such as keyword-based algorithms and embedding techniques-with state-of-the-art topic modeling methods that employ dimensionality reduction and clustering. This comparison allows us to assess the popularity and evolution of different approaches and themes within algorithmic trading. We demonstrate the usefulness of Natural Language Processing (NLP) in the automatic extraction of knowledge, highlighting the new possibilities created by the latest iterations of Large Language Models (LLMs) like ChatGPT. The rationale for focusing on this topic stems from our analysis, which reveals that research articles on algorithmic trading are increasing at a faster rate than the overall number of publications. While stocks and main indices comprise more than half of all assets considered, certain asset classes, such as cryptocurrencies, exhibit a much stronger growth trend. Machine learning models have become the most popular methods in recent years. The study demonstrates the efficacy of LLMs in refining datasets and addressing intricate questions about the analyzed articles, such as comparing the efficiency of different models. Our research shows that by decomposing tasks into smaller components and incorporating reasoning steps, we can effectively tackle complex questions supported by case analyses. This approach contributes to a deeper understanding of algorithmic trading methodologies and underscores the potential of advanced NLP techniques in literature reviews.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15956v2">Do Large Language Models Have an English Accent? Evaluating and Improving the Naturalness of Multilingual LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-23
    </div>
    <details class="paper-abstract">
      Current Large Language Models (LLMs) are predominantly designed with English as the primary language, and even the few that are multilingual tend to exhibit strong English-centric biases. Much like speakers who might produce awkward expressions when learning a second language, LLMs often generate unnatural outputs in non-English languages, reflecting English-centric patterns in both vocabulary and grammar. Despite the importance of this issue, the naturalness of multilingual LLM outputs has received limited attention. In this paper, we address this gap by introducing novel automatic corpus-level metrics to assess the lexical and syntactic naturalness of LLM outputs in a multilingual context. Using our new metrics, we evaluate state-of-the-art LLMs on a curated benchmark in French and Chinese, revealing a tendency towards English-influenced patterns. To mitigate this issue, we also propose a simple and effective alignment method to improve the naturalness of an LLM in a target language and domain, achieving consistent improvements in naturalness without compromising the performance on general-purpose benchmarks. Our work highlights the importance of developing multilingual metrics, resources and methods for the new wave of multilingual LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.17153v2">A Unified Debugging Approach via LLM-Based Multi-Agent Synergy</a></div>
    <div class="paper-meta">
      📅 2024-10-23
    </div>
    <details class="paper-abstract">
      Software debugging is a time-consuming endeavor involving a series of steps, such as fault localization and patch generation, each requiring thorough analysis and a deep understanding of the underlying logic. While large language models (LLMs) demonstrate promising potential in coding tasks, their performance in debugging remains limited. Current LLM-based methods often focus on isolated steps and struggle with complex bugs. In this paper, we propose the first end-to-end framework, FixAgent, for unified debugging through multi-agent synergy. It mimics the entire cognitive processes of developers, with each agent specialized as a particular component of this process rather than mirroring the actions of an independent expert as in previous multi-agent systems. Agents are coordinated through a three-level design, following a cognitive model of debugging, allowing adaptive handling of bugs with varying complexities. Experiments on extensive benchmarks demonstrate that FixAgent significantly outperforms state-of-the-art repair methods, fixing 1.25$\times$ to 2.56$\times$ bugs on the repo-level benchmark, Defects4J. This performance is achieved without requiring ground-truth root-cause code statements, unlike the baselines. Our source code is available on https://github.com/AcceptePapier/UniDebugger.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.17781v1">Evaluating Explanations Through LLMs: Beyond Traditional User Studies</a></div>
    <div class="paper-meta">
      📅 2024-10-23
    </div>
    <details class="paper-abstract">
      As AI becomes fundamental in sectors like healthcare, explainable AI (XAI) tools are essential for trust and transparency. However, traditional user studies used to evaluate these tools are often costly, time consuming, and difficult to scale. In this paper, we explore the use of Large Language Models (LLMs) to replicate human participants to help streamline XAI evaluation. We reproduce a user study comparing counterfactual and causal explanations, replicating human participants with seven LLMs under various settings. Our results show that (i) LLMs can replicate most conclusions from the original study, (ii) different LLMs yield varying levels of alignment in the results, and (iii) experimental factors such as LLM memory and output variability affect alignment with human responses. These initial findings suggest that LLMs could provide a scalable and cost-effective way to simplify qualitative XAI evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.04249v2">DiffSpec: Differential Testing with LLMs using Natural Language Specifications and Code Artifacts</a></div>
    <div class="paper-meta">
      📅 2024-10-23
    </div>
    <details class="paper-abstract">
      Differential testing can be an effective way to find bugs in software systems with multiple implementations that conform to the same specification, like compilers, network protocol parsers, and language runtimes. Specifications for such systems are often standardized in natural language documents, like Instruction Set Architecture (ISA) specifications, Wasm specifications or IETF RFC's. Large Language Models (LLMs) have demonstrated potential in both generating tests and handling large volumes of natural language text, making them well-suited for utilizing artifacts like specification documents, bug reports, and code implementations. In this work, we leverage natural language and code artifacts to guide LLMs to generate targeted, meaningful tests that highlight meaningful behavioral differences between implementations, including those corresponding to bugs. We introduce DiffSpec, a framework for generating differential tests with LLMs using prompt chaining. We demonstrate the efficacy of DiffSpec on two different systems, namely, eBPF runtimes and Wasm validators. Using DiffSpec, we generated 359 differentiating tests, uncovering at least four distinct and confirmed bugs in eBPF, including a kernel memory leak, inconsistent behavior in jump instructions, and undefined behavior when using the stack pointer. We also found 279 differentiating tests in Wasm validators, that point to at least 2 confirmed and fixed bugs in Wizard Engine.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.15993v4">Uncertainty Estimation and Quantification for LLMs: A Simple Supervised Approach</a></div>
    <div class="paper-meta">
      📅 2024-10-23
      | 💬 29 pages, 14 figures
    </div>
    <details class="paper-abstract">
      In this paper, we study the problem of uncertainty estimation and calibration for LLMs. We begin by formulating the uncertainty estimation problem, a relevant yet underexplored area in existing literature. We then propose a supervised approach that leverages labeled datasets to estimate the uncertainty in LLMs' responses. Based on the formulation, we illustrate the difference between the uncertainty estimation for LLMs and that for standard ML models and explain why the hidden neurons of the LLMs may contain uncertainty information. Our designed approach demonstrates the benefits of utilizing hidden activations to enhance uncertainty estimation across various tasks and shows robust transferability in out-of-distribution settings. We distinguish the uncertainty estimation task from the uncertainty calibration task and show that better uncertainty estimation leads to better calibration performance. Furthermore, our method is easy to implement and adaptable to different levels of model accessibility including black box, grey box, and white box.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.10216v2">Regularizing Hidden States Enables Learning Generalizable Reward Model for LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-23
      | 💬 NeurIPS 2024
    </div>
    <details class="paper-abstract">
      Reward models trained on human preference data have been proven to effectively align Large Language Models (LLMs) with human intent within the framework of reinforcement learning from human feedback (RLHF). However, current reward models have limited generalization capabilities to unseen prompts and responses, which can lead to an unexpected phenomenon known as reward over-optimization, resulting in a decline in actual performance due to excessive optimization of rewards. While previous research has advocated for constraining policy optimization, our study introduces a novel approach to enhance the reward model's generalization ability against distribution shifts by regularizing the hidden states. Specifically, we retain the base model's language model head and incorporate a suite of text-generation losses to preserve the hidden states' text-generation capabilities, while concurrently learning a reward head behind the same hidden states. Our experimental results demonstrate that the introduced regularization technique markedly improves the accuracy of learned reward models across a variety of out-of-distribution (OOD) tasks and effectively alleviates the over-optimization issue in RLHF, offering a more reliable and robust preference learning paradigm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.03622v3">Mind's Eye of LLMs: Visualization-of-Thought Elicits Spatial Reasoning in Large Language Models</a></div>
    <div class="paper-meta">
      📅 2024-10-23
      | 💬 38th Conference on Neural Information Processing Systems (NeurIPS 2024)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have exhibited impressive performance in language comprehension and various reasoning tasks. However, their abilities in spatial reasoning, a crucial aspect of human cognition, remain relatively unexplored. Human possess a remarkable ability to create mental images of unseen objects and actions through a process known as the Mind's Eye, enabling the imagination of the unseen world. Inspired by this cognitive capacity, we propose Visualization-of-Thought (VoT) prompting. VoT aims to elicit spatial reasoning of LLMs by visualizing their reasoning traces, thereby guiding subsequent reasoning steps. We employed VoT for multi-hop spatial reasoning tasks, including natural language navigation, visual navigation, and visual tiling in 2D grid worlds. Experimental results demonstrated that VoT significantly enhances the spatial reasoning abilities of LLMs. Notably, VoT outperformed existing multimodal large language models (MLLMs) in these tasks. While VoT works surprisingly well on LLMs, the ability to generate mental images to facilitate spatial reasoning resembles the mind's eye process, suggesting its potential viability in MLLMs. Please find the dataset and codes at https://microsoft.github.io/visualization-of-thought
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.06813v4">Richelieu: Self-Evolving LLM-Based Agents for AI Diplomacy</a></div>
    <div class="paper-meta">
      📅 2024-10-23
    </div>
    <details class="paper-abstract">
      Diplomacy is one of the most sophisticated activities in human society, involving complex interactions among multiple parties that require skills in social reasoning, negotiation, and long-term strategic planning. Previous AI agents have demonstrated their ability to handle multi-step games and large action spaces in multi-agent tasks. However, diplomacy involves a staggering magnitude of decision spaces, especially considering the negotiation stage required. While recent agents based on large language models (LLMs) have shown potential in various applications, they still struggle with extended planning periods in complex multi-agent settings. Leveraging recent technologies for LLM-based agents, we aim to explore AI's potential to create a human-like agent capable of executing comprehensive multi-agent missions by integrating three fundamental capabilities: 1) strategic planning with memory and reflection; 2) goal-oriented negotiation with social reasoning; and 3) augmenting memory through self-play games for self-evolution without human in the loop.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.17578v1">MM-Eval: A Multilingual Meta-Evaluation Benchmark for LLM-as-a-Judge and Reward Models</a></div>
    <div class="paper-meta">
      📅 2024-10-23
      | 💬 work in progress
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are commonly used as evaluators in tasks (e.g., reward modeling, LLM-as-a-judge), where they act as proxies for human preferences or judgments. This leads to the need for meta-evaluation: evaluating the credibility of LLMs as evaluators. However, existing benchmarks primarily focus on English, offering limited insight into LLMs' effectiveness as evaluators in non-English contexts. To address this, we introduce MM-Eval, a multilingual meta-evaluation benchmark that covers 18 languages across six categories. MM-Eval evaluates various dimensions, including language-specific challenges like linguistics and language hallucinations. Evaluation results show that both proprietary and open-source language models have considerable room for improvement. Further analysis reveals a tendency for these models to assign middle-ground scores to low-resource languages. We publicly release our benchmark and code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04870v5">ConfusedPilot: Confused Deputy Risks in RAG-based LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-23
    </div>
    <details class="paper-abstract">
      Retrieval augmented generation (RAG) is a process where a large language model (LLM) retrieves useful information from a database and then generates the responses. It is becoming popular in enterprise settings for daily business operations. For example, Copilot for Microsoft 365 has accumulated millions of businesses. However, the security implications of adopting such RAG-based systems are unclear. In this paper, we introduce ConfusedPilot, a class of security vulnerabilities of RAG systems that confuse Copilot and cause integrity and confidentiality violations in its responses. First, we investigate a vulnerability that embeds malicious text in the modified prompt in RAG, corrupting the responses generated by the LLM. Second, we demonstrate a vulnerability that leaks secret data, which leverages the caching mechanism during retrieval. Third, we investigate how both vulnerabilities can be exploited to propagate misinformation within the enterprise and ultimately impact its operations, such as sales and manufacturing. We also discuss the root cause of these attacks by investigating the architecture of a RAG-based system. This study highlights the security vulnerabilities in today's RAG-based systems and proposes design guidelines to secure future RAG-based systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16638v2">LLMScan: Causal Scan for LLM Misbehavior Detection</a></div>
    <div class="paper-meta">
      📅 2024-10-23
    </div>
    <details class="paper-abstract">
      Despite the success of Large Language Models (LLMs) across various fields, their potential to generate untruthful, biased and harmful responses poses significant risks, particularly in critical applications. This highlights the urgent need for systematic methods to detect and prevent such misbehavior. While existing approaches target specific issues such as harmful responses, this work introduces LLMScan, an innovative LLM monitoring technique based on causality analysis, offering a comprehensive solution. LLMScan systematically monitors the inner workings of an LLM through the lens of causal inference, operating on the premise that the LLM's `brain' behaves differently when misbehaving. By analyzing the causal contributions of the LLM's input tokens and transformer layers, LLMScan effectively detects misbehavior. Extensive experiments across various tasks and models reveal clear distinctions in the causal distributions between normal behavior and misbehavior, enabling the development of accurate, lightweight detectors for a variety of misbehavior detection tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.17529v1">Navigate Complex Physical Worlds via Geometrically Constrained LLM</a></div>
    <div class="paper-meta">
      📅 2024-10-23
    </div>
    <details class="paper-abstract">
      This study investigates the potential of Large Language Models (LLMs) for reconstructing and constructing the physical world solely based on textual knowledge. It explores the impact of model performance on spatial understanding abilities. To enhance the comprehension of geometric and spatial relationships in the complex physical world, the study introduces a set of geometric conventions and develops a workflow based on multi-layer graphs and multi-agent system frameworks. It examines how LLMs achieve multi-step and multi-objective geometric inference in a spatial environment using multi-layer graphs under unified geometric conventions. Additionally, the study employs a genetic algorithm, inspired by large-scale model knowledge, to solve geometric constraint problems. In summary, this work innovatively explores the feasibility of using text-based LLMs as physical world builders and designs a workflow to enhance their capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14687v2">BrainTransformers: SNN-LLM</a></div>
    <div class="paper-meta">
      📅 2024-10-23
    </div>
    <details class="paper-abstract">
      This study introduces BrainTransformers, an innovative Large Language Model (LLM) implemented using Spiking Neural Networks (SNN). Our key contributions include: (1) designing SNN-compatible Transformer components such as SNNMatmul, SNNSoftmax, and SNNSiLU; (2) implementing an SNN approximation of the SiLU activation function; and (3) developing a Synapsis module to simulate synaptic plasticity. Our 3-billion parameter model, BrainTransformers-3B-Chat, demonstrates competitive performance across various benchmarks, including MMLU (63.2), BBH (54.1), ARC-C (54.3), and GSM8K (76.3), while potentially offering improved energy efficiency and biological plausibility. The model employs a three-stage training approach, including SNN-specific neuronal synaptic plasticity training. This research opens new avenues for brain-like AI systems in natural language processing and neuromorphic computing. Future work will focus on hardware optimization, developing specialized SNN fine-tuning tools, and exploring practical applications in energy-efficient computing environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.06239v2">OrionNav: Online Planning for Robot Autonomy with Context-Aware LLM and Open-Vocabulary Semantic Scene Graphs</a></div>
    <div class="paper-meta">
      📅 2024-10-23
    </div>
    <details class="paper-abstract">
      Enabling robots to autonomously navigate unknown, complex, dynamic environments and perform diverse tasks remains a fundamental challenge in developing robust autonomous physical agents. These agents must effectively perceive their surroundings while leveraging world knowledge for decision-making. Although recent approaches utilize vision-language and large language models for scene understanding and planning, they often rely on offline processing, offboard compute, make simplifying assumptions about the environment and perception, limiting real-world applicability. We present a novel framework for real-time onboard autonomous navigation in unknown environments that change over time by integrating multi-level abstraction in both perception and planning pipelines. Our system fuses data from multiple onboard sensors for localization and mapping and integrates it with open-vocabulary semantics to generate hierarchical scene graphs from continuously updated semantic object map. The LLM-based planner uses these graphs to create multi-step plans that guide low-level controllers in executing navigation tasks specified in natural language. The system's real-time operation enables the LLM to adjust its plans based on updates to the scene graph and task execution status, ensuring continuous adaptation to new situations or when the current plan cannot accomplish the task, a key advantage over static or rule-based systems. We demonstrate our system's efficacy on a quadruped navigating dynamic environments, showcasing its adaptability and robustness in diverse scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14740v2">Harnessing Your DRAM and SSD for Sustainable and Accessible LLM Inference with Mixed-Precision and Multi-level Caching</a></div>
    <div class="paper-meta">
      📅 2024-10-23
      | 💬 24 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Although Large Language Models (LLMs) have demonstrated remarkable capabilities, their massive parameter counts and associated extensive computing make LLMs' deployment the main part of carbon emission from nowadays AI applications. Compared to modern GPUs like H$100$, it would be significantly carbon-sustainable if we could leverage old-fashioned GPUs such as M$40$ (as shown in Figure 1, M$40$ only has one third carbon emission of H$100$'s) for LLM servings. However, the limited High Bandwidth Memory (HBM) available on such GPU often cannot support the loading of LLMs due to the gigantic model size and intermediate activation data, making their serving challenging. For instance, a LLaMA2 model with $70$B parameters typically requires $128$GB for inference, which substantially surpasses $24$GB HBM in a $3090$ GPU and remains infeasible even considering the additional $64$GB DRAM. To address this challenge, this paper proposes a mixed-precision with a model modularization algorithm to enable LLM inference on outdated hardware with resource constraints. (The precision denotes the numerical precision like FP16, INT8, INT4) and multi-level caching (M2Cache).) Specifically, our M2Cache first modulizes neurons in LLM and creates their importance ranking. Then, it adopts a dynamic sparse mixed-precision quantization mechanism in weight space to reduce computational demands and communication overhead at each decoding step. It collectively lowers the operational carbon emissions associated with LLM inference. Moreover, M2Cache introduces a three-level cache management system with HBM, DRAM, and SSDs that complements the dynamic sparse mixed-precision inference. To enhance communication efficiency, M2Cache maintains a neuron-level mixed-precision LRU cache in HBM, a larger layer-aware cache in DRAM, and a full model in SSD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.10601v2">When "Competency" in Reasoning Opens the Door to Vulnerability: Jailbreaking LLMs via Novel Complex Ciphers</a></div>
    <div class="paper-meta">
      📅 2024-10-23
      | 💬 14 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Recent advancements in the safety of Large Language Models (LLMs) have primarily focused on mitigating attacks crafted in natural language or in common encryption techniques like Base64. However, new models which often possess better reasoning capabilities, open the door to new attack vectors that were previously non-existent in older models. This seems counter-intuitive at first glance, but these advanced models can decipher more complex cryptic queries that previous models could not, making them susceptible to attacks using such prompts. To exploit this vulnerability, we propose Attacks using Custom Encryptions (ACE), a novel method to jailbreak LLMs by leveraging custom encryption schemes. We evaluate the effectiveness of ACE on four state-of-the-art LLMs, achieving Attack Success Rates (ASR) of up to 66% on close-source models and 88% on open-source models. Building upon this, we introduce Layered Attacks using Custom Encryptions (LACE), which employs multiple layers of encryption through our custom ciphers to further enhance the ASR. Our findings demonstrate that LACE significantly enhances the ability to jailbreak LLMs, increasing the ASR of GPT-4o from 40% to 78%, a 38% improvement. Our results highlight that the advanced capabilities of LLMs introduce unforeseen vulnerabilities to complex attacks. Specifically complex and layered ciphers increase the chance of jailbreaking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.17482v1">Is artificial intelligence still intelligence? LLMs generalize to novel adjective-noun pairs, but don't mimic the full human distribution</a></div>
    <div class="paper-meta">
      📅 2024-10-23
      | 💬 9 pages (23 pages with appendix). Accepted to GenBench 2024
    </div>
    <details class="paper-abstract">
      Inferences from adjective-noun combinations like "Is artificial intelligence still intelligence?" provide a good test bed for LLMs' understanding of meaning and compositional generalization capability, since there are many combinations which are novel to both humans and LLMs but nevertheless elicit convergent human judgments. We study a range of LLMs and find that the largest models we tested are able to draw human-like inferences when the inference is determined by context and can generalize to unseen adjective-noun combinations. We also propose three methods to evaluate LLMs on these inferences out of context, where there is a distribution of human-like answers rather than a single correct answer. We find that LLMs show a human-like distribution on at most 75\% of our dataset, which is promising but still leaves room for improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2410.16285v1">Assessing the Performance of Human-Capable LLMs -- Are LLMs Coming for Your Job?</a></div>
    <div class="paper-meta">
      📅 2024-10-23
    </div>
    <details class="paper-abstract">
      The current paper presents the development and validation of SelfScore, a novel benchmark designed to assess the performance of automated Large Language Model (LLM) agents on help desk and professional consultation tasks. Given the increasing integration of AI in industries, particularly within customer service, SelfScore fills a crucial gap by enabling the comparison of automated agents and human workers. The benchmark evaluates agents on problem complexity and response helpfulness, ensuring transparency and simplicity in its scoring system. The study also develops automated LLM agents to assess SelfScore and explores the benefits of Retrieval-Augmented Generation (RAG) for domain-specific tasks, demonstrating that automated LLM agents incorporating RAG outperform those without. All automated LLM agents were observed to perform better than the human control group. Given these results, the study raises concerns about the potential displacement of human workers, especially in areas where AI technologies excel. Ultimately, SelfScore provides a foundational tool for understanding the impact of AI in help desk environments while advocating for ethical considerations in the ongoing transition towards automation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.11402v2">NVLM: Open Frontier-Class Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-22
      | 💬 Fixed the typos. For more information, please visit our project page at: https://research.nvidia.com/labs/adlr/NVLM-1
    </div>
    <details class="paper-abstract">
      We introduce NVLM 1.0, a family of frontier-class multimodal large language models (LLMs) that achieve state-of-the-art results on vision-language tasks, rivaling the leading proprietary models (e.g., GPT-4o) and open-access models (e.g., Llama 3-V 405B and InternVL 2). Remarkably, NVLM 1.0 shows improved text-only performance over its LLM backbone after multimodal training. In terms of model design, we perform a comprehensive comparison between decoder-only multimodal LLMs (e.g., LLaVA) and cross-attention-based models (e.g., Flamingo). Based on the strengths and weaknesses of both approaches, we propose a novel architecture that enhances both training efficiency and multimodal reasoning capabilities. Furthermore, we introduce a 1-D tile-tagging design for tile-based dynamic high-resolution images, which significantly boosts performance on multimodal reasoning and OCR-related tasks. Regarding training data, we meticulously curate and provide detailed information on our multimodal pretraining and supervised fine-tuning datasets. Our findings indicate that dataset quality and task diversity are more important than scale, even during the pretraining phase, across all architectures. Notably, we develop production-grade multimodality for the NVLM-1.0 models, enabling them to excel in vision-language tasks while maintaining and even improving text-only performance compared to their LLM backbones. To achieve this, we craft and integrate a high-quality text-only dataset into multimodal training, alongside a substantial amount of multimodal math and reasoning data, leading to enhanced math and coding capabilities across modalities. To advance research in the field, we release the model weights at https://huggingface.co/nvidia/NVLM-D-72B and will open-source the training code for the community soon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.17406v1">ProveRAG: Provenance-Driven Vulnerability Analysis with Automated Retrieval-Augmented LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-22
    </div>
    <details class="paper-abstract">
      In cybersecurity, security analysts face the challenge of mitigating newly discovered vulnerabilities in real-time, with over 300,000 Common Vulnerabilities and Exposures (CVEs) identified since 1999. The sheer volume of known vulnerabilities complicates the detection of patterns for unknown threats. While LLMs can assist, they often hallucinate and lack alignment with recent threats. Over 25,000 vulnerabilities have been identified so far in 2024, which are introduced after popular LLMs' (e.g., GPT-4) training data cutoff. This raises a major challenge of leveraging LLMs in cybersecurity, where accuracy and up-to-date information are paramount. In this work, we aim to improve the adaptation of LLMs in vulnerability analysis by mimicking how analysts perform such tasks. We propose ProveRAG, an LLM-powered system designed to assist in rapidly analyzing CVEs with automated retrieval augmentation of web data while self-evaluating its responses with verifiable evidence. ProveRAG incorporates a self-critique mechanism to help alleviate omission and hallucination common in the output of LLMs applied in cybersecurity applications. The system cross-references data from verifiable sources (NVD and CWE), giving analysts confidence in the actionable insights provided. Our results indicate that ProveRAG excels in delivering verifiable evidence to the user with over 99% and 97% accuracy in exploitation and mitigation strategies, respectively. This system outperforms direct prompting and chunking retrieval in vulnerability analysis by overcoming temporal and context-window limitations. ProveRAG guides analysts to secure their systems more effectively while documenting the process for future audits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.17542v3">CDQuant: Greedy Coordinate Descent for Accurate LLM Quantization</a></div>
    <div class="paper-meta">
      📅 2024-10-22
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently demonstrated remarkable performance across diverse language tasks. But their deployment is often constrained by their substantial computational and storage requirements. Quantization has emerged as a key technique for addressing this challenge, enabling the compression of large models with minimal impact on performance. The recent GPTQ algorithm, a post-training quantization (PTQ) method, has proven highly effective for compressing LLMs, sparking a wave of research that leverages GPTQ as a core component. Recognizing the pivotal role of GPTQ in the PTQ landscape, we introduce CDQuant, a simple and scalable alternative to GPTQ with improved performance. CDQuant uses greedy coordinate descent to minimize the layer-wise reconstruction loss to achieve high-quality quantized weights. Our algorithm is easy to implement and scales efficiently to models with hundreds of billions of parameters. We perform extensive evaluation on Gemma, and PaLM2 model families, and demonstrate that CDQuant consistently outperforms GPTQ in 2-4 bit weight quantization. Moreover, CDQuant improves the performance of state-of-the-art PTQ techniques such as QuIP and FrameQuant when used as a replacement for their GPTQ component, resulting in further gains in quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.17238v1">SELA: Tree-Search Enhanced LLM Agents for Automated Machine Learning</a></div>
    <div class="paper-meta">
      📅 2024-10-22
      | 💬 The code is available at https://github.com/geekan/MetaGPT
    </div>
    <details class="paper-abstract">
      Automated Machine Learning (AutoML) approaches encompass traditional methods that optimize fixed pipelines for model selection and ensembling, as well as newer LLM-based frameworks that autonomously build pipelines. While LLM-based agents have shown promise in automating machine learning tasks, they often generate low-diversity and suboptimal code, even after multiple iterations. To overcome these limitations, we introduce Tree-Search Enhanced LLM Agents (SELA), an innovative agent-based system that leverages Monte Carlo Tree Search (MCTS) to optimize the AutoML process. By representing pipeline configurations as trees, our framework enables agents to conduct experiments intelligently and iteratively refine their strategies, facilitating a more effective exploration of the machine learning solution space. This novel approach allows SELA to discover optimal pathways based on experimental feedback, improving the overall quality of the solutions. In an extensive evaluation across 20 machine learning datasets, we compare the performance of traditional and agent-based AutoML methods, demonstrating that SELA achieves a win rate of 65% to 80% against each baseline across all datasets. These results underscore the significant potential of agent-based strategies in AutoML, offering a fresh perspective on tackling complex machine learning challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.14344v2">LLMs left, right, and center: Assessing GPT's capabilities to label political bias from web domains</a></div>
    <div class="paper-meta">
      📅 2024-10-22
      | 💬 12 pages, 4 figures
    </div>
    <details class="paper-abstract">
      This research investigates whether OpenAI's GPT-4, a state-of-the-art large language model, can accurately classify the political bias of news sources based solely on their URLs. Given the subjective nature of political labels, third-party bias ratings like those from Ad Fontes Media, AllSides, and Media Bias/Fact Check (MBFC) are often used in research to analyze news source diversity. This study aims to determine if GPT-4 can replicate these human ratings on a seven-degree scale ("far-left" to "far-right"). The analysis compares GPT-4's classifications against MBFC's, and controls for website popularity using Open PageRank scores. Findings reveal a high correlation ($\text{Spearman's } \rho = .89$, $n = 5,877$, $p < 0.001$) between GPT-4's and MBFC's ratings, indicating the model's potential reliability. However, GPT-4 abstained from classifying approximately $\frac{2}{3}$ of the dataset. It is more likely to abstain from rating unpopular websites, which also suffer from less accurate assessments. The LLM tends to avoid classifying sources that MBFC considers to be centrist, resulting in more polarized outputs. Finally, this analysis shows a slight leftward skew in GPT's classifications compared to MBFC's. Therefore, while this paper suggests that while GPT-4 can be a scalable, cost-effective tool for political bias classification of news websites, its use should be as a complement to human judgment to mitigate biases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.17126v1">Exploring RL-based LLM Training for Formal Language Tasks with Programmed Rewards</a></div>
    <div class="paper-meta">
      📅 2024-10-22
      | 💬 Accepted at BNAIC 2024
    </div>
    <details class="paper-abstract">
      Proximal Policy Optimization (PPO) is commonly used in Reinforcement Learning from Human Feedback to align large language models (LLMs) with downstream tasks. This paper investigates the feasibility of using PPO for direct reinforcement learning (RL) from explicitly programmed reward signals, as opposed to indirect learning from human feedback via an intermediary reward model. We focus on tasks expressed through formal languages, such as mathematics and programming, where explicit reward functions can be programmed to automatically assess the quality of generated outputs. We apply this approach to a sentiment alignment task, a simple arithmetic task, and a more complex game synthesis task. The sentiment alignment task replicates prior research and serves to validate our experimental setup. Our results show that pure RL-based training for the two formal language tasks is challenging, with success being limited even for the simple arithmetic task. We propose a novel batch-entropy regularization term to aid exploration, although training is not yet entirely stable. Our findings suggest that direct RL training of LLMs may be more suitable for relatively minor changes, such as alignment, than for learning new tasks altogether, even if an informative reward signal can be expressed programmatically.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.17099v1">Human-LLM Hybrid Text Answer Aggregation for Crowd Annotations</a></div>
    <div class="paper-meta">
      📅 2024-10-22
      | 💬 Accepted in EMNLP 2024
    </div>
    <details class="paper-abstract">
      The quality is a crucial issue for crowd annotations. Answer aggregation is an important type of solution. The aggregated answers estimated from multiple crowd answers to the same instance are the eventually collected annotations, rather than the individual crowd answers themselves. Recently, the capability of Large Language Models (LLMs) on data annotation tasks has attracted interest from researchers. Most of the existing studies mainly focus on the average performance of individual crowd workers; several recent works studied the scenarios of aggregation on categorical labels and LLMs used as label creators. However, the scenario of aggregation on text answers and the role of LLMs as aggregators are not yet well-studied. In this paper, we investigate the capability of LLMs as aggregators in the scenario of close-ended crowd text answer aggregation. We propose a human-LLM hybrid text answer aggregation method with a Creator-Aggregator Multi-Stage (CAMS) crowdsourcing framework. We make the experiments based on public crowdsourcing datasets. The results show the effectiveness of our approach based on the collaboration of crowd workers and LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.17050v1">UnStar: Unlearning with Self-Taught Anti-Sample Reasoning for LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-22
    </div>
    <details class="paper-abstract">
      The key components of machine learning are data samples for training, model for learning patterns, and loss function for optimizing accuracy. Analogously, unlearning can potentially be achieved through anti-data samples (or anti-samples), unlearning method, and reversed loss function. While prior research has explored unlearning methods and reversed loss functions, the potential of anti-samples remains largely untapped. In this paper, we introduce UnSTAR: Unlearning with Self-Taught Anti-Sample Reasoning for large language models (LLMs). Our contributions are threefold; first, we propose a novel concept of anti-sample-induced unlearning; second, we generate anti-samples by leveraging misleading rationales, which help reverse learned associations and accelerate the unlearning process; and third, we enable fine-grained targeted unlearning, allowing for the selective removal of specific associations without impacting related knowledge - something not achievable by previous works. Results demonstrate that anti-samples offer an efficient, targeted unlearning strategy for LLMs, opening new avenues for privacy-preserving machine learning and model modification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.17040v1">Arabic Dataset for LLM Safeguard Evaluation</a></div>
    <div class="paper-meta">
      📅 2024-10-22
      | 💬 17 pages, 6 figures, 10 tables
    </div>
    <details class="paper-abstract">
      The growing use of large language models (LLMs) has raised concerns regarding their safety. While many studies have focused on English, the safety of LLMs in Arabic, with its linguistic and cultural complexities, remains under-explored. Here, we aim to bridge this gap. In particular, we present an Arab-region-specific safety evaluation dataset consisting of 5,799 questions, including direct attacks, indirect attacks, and harmless requests with sensitive words, adapted to reflect the socio-cultural context of the Arab world. To uncover the impact of different stances in handling sensitive and controversial topics, we propose a dual-perspective evaluation framework. It assesses the LLM responses from both governmental and opposition viewpoints. Experiments over five leading Arabic-centric and multilingual LLMs reveal substantial disparities in their safety performance. This reinforces the need for culturally specific datasets to ensure the responsible deployment of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16070v2">On-Device LLMs for SMEs: Challenges and Opportunities</a></div>
    <div class="paper-meta">
      📅 2024-10-22
      | 💬 9 pages, 1 figure. The work is supported by the SIT-NVIDIA Joint AI Centre
    </div>
    <details class="paper-abstract">
      This paper presents a systematic review of the infrastructure requirements for deploying Large Language Models (LLMs) on-device within the context of small and medium-sized enterprises (SMEs), focusing on both hardware and software perspectives. From the hardware viewpoint, we discuss the utilization of processing units like GPUs and TPUs, efficient memory and storage solutions, and strategies for effective deployment, addressing the challenges of limited computational resources typical in SME settings. From the software perspective, we explore framework compatibility, operating system optimization, and the use of specialized libraries tailored for resource-constrained environments. The review is structured to first identify the unique challenges faced by SMEs in deploying LLMs on-device, followed by an exploration of the opportunities that both hardware innovations and software adaptations offer to overcome these obstacles. Such a structured review provides practical insights, contributing significantly to the community by enhancing the technological resilience of SMEs in integrating LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10851v2">LLM Gesticulator: Leveraging Large Language Models for Scalable and Controllable Co-Speech Gesture Synthesis</a></div>
    <div class="paper-meta">
      📅 2024-10-22
    </div>
    <details class="paper-abstract">
      In this work, we present LLM Gesticulator, an LLM-based audio-driven co-speech gesture generation framework that synthesizes full-body animations that are rhythmically aligned with the input audio while exhibiting natural movements and editability. Compared to previous work, our model demonstrates substantial scalability. As the size of the backbone LLM model increases, our framework shows proportional improvements in evaluation metrics (a.k.a. scaling law). Our method also exhibits strong controllability where the content, style of the generated gestures can be controlled by text prompt. To the best of our knowledge, LLM gesticulator is the first work that use LLM on the co-speech generation task. Evaluation with existing objective metrics and user studies indicate that our framework outperforms prior works.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.15319v2">Stacking Your Transformers: A Closer Look at Model Growth for Efficient LLM Pre-Training</a></div>
    <div class="paper-meta">
      📅 2024-10-22
      | 💬 NeurIPS 2024 Spotlight
    </div>
    <details class="paper-abstract">
      LLMs are computationally expensive to pre-train due to their large scale. Model growth emerges as a promising approach by leveraging smaller models to accelerate the training of larger ones. However, the viability of these model growth methods in efficient LLM pre-training remains underexplored. This work identifies three critical $\underline{\textit{O}}$bstacles: ($\textit{O}$1) lack of comprehensive evaluation, ($\textit{O}$2) untested viability for scaling, and ($\textit{O}$3) lack of empirical guidelines. To tackle $\textit{O}$1, we summarize existing approaches into four atomic growth operators and systematically evaluate them in a standardized LLM pre-training setting. Our findings reveal that a depthwise stacking operator, called $G_{\text{stack}}$, exhibits remarkable acceleration in training, leading to decreased loss and improved overall performance on eight standard NLP benchmarks compared to strong baselines. Motivated by these promising results, we conduct extensive experiments to delve deeper into $G_{\text{stack}}$ to address $\textit{O}$2 and $\textit{O}$3. For $\textit{O}$2 (untested scalability), our study shows that $G_{\text{stack}}$ is scalable and consistently performs well, with experiments up to 7B LLMs after growth and pre-training LLMs with 750B tokens. For example, compared to a conventionally trained 7B model using 300B tokens, our $G_{\text{stack}}$ model converges to the same loss with 194B tokens, resulting in a 54.6\% speedup. We further address $\textit{O}$3 (lack of empirical guidelines) by formalizing guidelines to determine growth timing and growth factor for $G_{\text{stack}}$, making it practical in general LLM pre-training. We also provide in-depth discussions and comprehensive ablation studies of $G_{\text{stack}}$. Our code and pre-trained model are available at https://llm-stacking.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16775v1">Context-Aware LLM Translation System Using Conversation Summarization and Dialogue History</a></div>
    <div class="paper-meta">
      📅 2024-10-22
      | 💬 Accepted to WMT 2024
    </div>
    <details class="paper-abstract">
      Translating conversational text, particularly in customer support contexts, presents unique challenges due to its informal and unstructured nature. We propose a context-aware LLM translation system that leverages conversation summarization and dialogue history to enhance translation quality for the English-Korean language pair. Our approach incorporates the two most recent dialogues as raw data and a summary of earlier conversations to manage context length effectively. We demonstrate that this method significantly improves translation accuracy, maintaining coherence and consistency across conversations. This system offers a practical solution for customer support translation tasks, addressing the complexities of conversational text.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16738v1">LLM-Assisted Red Teaming of Diffusion Models through "Failures Are Fated, But Can Be Faded"</a></div>
    <div class="paper-meta">
      📅 2024-10-22
      | 💬 13 pages, 11 figures. arXiv admin note: substantial text overlap with arXiv:2406.07145
    </div>
    <details class="paper-abstract">
      In large deep neural networks that seem to perform surprisingly well on many tasks, we also observe a few failures related to accuracy, social biases, and alignment with human values, among others. Therefore, before deploying these models, it is crucial to characterize this failure landscape for engineers to debug or audit models. Nevertheless, it is infeasible to exhaustively test for all possible combinations of factors that could lead to a model's failure. In this paper, we improve the "Failures are fated, but can be faded" framework (arXiv:2406.07145)--a post-hoc method to explore and construct the failure landscape in pre-trained generative models--with a variety of deep reinforcement learning algorithms, screening tests, and LLM-based rewards and state generation. With the aid of limited human feedback, we then demonstrate how to restructure the failure landscape to be more desirable by moving away from the discovered failure modes. We empirically demonstrate the effectiveness of the proposed method on diffusion models. We also highlight the strengths and weaknesses of each algorithm in identifying failure modes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16736v1">Forewarned is Forearmed: Leveraging LLMs for Data Synthesis through Failure-Inducing Exploration</a></div>
    <div class="paper-meta">
      📅 2024-10-22
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have significantly benefited from training on diverse, high-quality task-specific data, leading to impressive performance across a range of downstream applications. Current methods often rely on human-annotated data or predefined task templates to direct powerful LLMs in synthesizing task-relevant data for effective model training. However, this dependence on manually designed components may constrain the scope of generated data, potentially overlooking critical edge cases or novel scenarios that could challenge the model. In this paper, we present a novel approach, ReverseGen, designed to automatically generate effective training samples that expose the weaknesses of LLMs. Specifically, we introduce a dedicated proposer trained to produce queries that lead target models to generate unsatisfactory responses. These failure-inducing queries are then used to construct training data, helping to address the models' shortcomings and improve overall performance. Our approach is flexible and can be applied to models of various scales (3B, 7B, and 8B). We evaluate ReverseGen on three key applications (safety, honesty, and math), demonstrating that our generated data is both highly effective and diverse. Models fine-tuned with ReverseGen-generated data consistently outperform those trained on human-annotated or general model-generated data, offering a new perspective on data synthesis for task-specific LLM enhancement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.00219v2">Evaluating Human Alignment and Model Faithfulness of LLM Rationale</a></div>
    <div class="paper-meta">
      📅 2024-10-22
    </div>
    <details class="paper-abstract">
      We study how well large language models (LLMs) explain their generations through rationales -- a set of tokens extracted from the input text that reflect the decision-making process of LLMs. Specifically, we systematically study rationales derived using two approaches: (1) popular prompting-based methods, where prompts are used to guide LLMs in generating rationales, and (2) technical attribution-based methods, which leverage attention or gradients to identify important tokens. Our analysis spans three classification datasets with annotated rationales, encompassing tasks with varying performance levels. While prompting-based self-explanations are widely used, our study reveals that these explanations are not always as "aligned" with the human rationale as attribution-based explanations. Even more so, fine-tuning LLMs to enhance classification task accuracy does not enhance the alignment of prompting-based rationales. Still, it does considerably improve the alignment of attribution-based methods (e.g., InputXGradient). More importantly, we show that prompting-based self-explanation is also less "faithful" than attribution-based explanations, failing to provide a reliable account of the model's decision-making process. To evaluate faithfulness, unlike prior studies that excluded misclassified examples, we evaluate all instances and also examine the impact of fine-tuning and accuracy on alignment and faithfulness. Our findings suggest that inconclusive faithfulness results reported in earlier studies may stem from low classification accuracy. These findings underscore the importance of more rigorous and comprehensive evaluations of LLM rationales.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16682v1">Methods of improving LLM training stability</a></div>
    <div class="paper-meta">
      📅 2024-10-22
    </div>
    <details class="paper-abstract">
      Training stability of large language models(LLMs) is an important research topic. Reproducing training instabilities can be costly, so we use a small language model with 830M parameters and experiment with higher learning rates to force models to diverge. One of the sources of training instability is the growth of logits in attention layers. We extend the focus of the previous work and look not only at the magnitude of the logits but at all outputs of linear layers in the Transformer block. We observe that with a high learning rate the L2 norm of all linear layer outputs can grow with each training step and the model diverges. Specifically we observe that QKV, Proj and FC2 layers have the largest growth of the output magnitude. This prompts us to explore several options: 1) apply layer normalization not only after QK layers but also after Proj and FC2 layers too; 2) apply layer normalization after the QKV layer (and remove pre normalization). 3) apply QK layer normalization together with softmax capping. We show that with the last two methods we can increase learning rate by 1.5x (without model divergence) in comparison to an approach based on QK layer normalization only. Also we observe significant perplexity improvements for all three methods in comparison to the baseline model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16670v1">CoPS: Empowering LLM Agents with Provable Cross-Task Experience Sharing</a></div>
    <div class="paper-meta">
      📅 2024-10-22
      | 💬 25 pages, 5 tables, 3 figures
    </div>
    <details class="paper-abstract">
      Sequential reasoning in agent systems has been significantly advanced by large language models (LLMs), yet existing approaches face limitations. Reflection-driven reasoning relies solely on knowledge in pretrained models, limiting performance in novel scenarios, while experience-assisted reasoning often depends on external experiences and lacks clear principles for selecting representative experiences. We address these limitations by proposing CoPS (Cross-Task Experience Sharing), a generalizable algorithm that enhances sequential reasoning by cross-task experience sharing and selection. In detail, CoPS leverages agents' experiences on previous tasks, selecting distribution-matched experiences via a provable pessimism-based strategy to maximize utility while minimizing risks from distribution shifts. Extensive experimental results on benchmarks like Alfworld, Webshop, and HotPotQA demonstrate that CoPS consistently outperforms state-of-the-art baselines, with superior sample efficiency suitable for resource-constrained scenarios. Theoretically, we show that the performance of our algorithm depends on both the quality of the pretrained LLM and the matching between the agent's task-dependent trial distribution and that generated by the LLM. Our work bridges the gap between existing sequential reasoning paradigms and validates the effectiveness of leveraging cross-task experiences, shedding light on the potential to improve agents' generalization and adaptability across diverse tasks. Our codes are available at $\href{https://github.com/uclaml/COPS}{\text{https://github.com/uclaml/COPS}}$.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02084v2">Generating Symbolic Music from Natural Language Prompts using an LLM-Enhanced Dataset</a></div>
    <div class="paper-meta">
      📅 2024-10-22
    </div>
    <details class="paper-abstract">
      Recent years have seen many audio-domain text-to-music generation models that rely on large amounts of text-audio pairs for training. However, symbolic-domain controllable music generation has lagged behind partly due to the lack of a large-scale symbolic music dataset with extensive metadata and captions. In this work, we present MetaScore, a new dataset consisting of 963K musical scores paired with rich metadata, including free-form user-annotated tags, collected from an online music forum. To approach text-to-music generation, we leverage a pretrained large language model (LLM) to generate pseudo natural language captions from the metadata. With the LLM-enhanced MetaScore, we train a text-conditioned music generation model that learns to generate symbolic music from the pseudo captions, allowing control of instruments, genre, composer, complexity and other free-form music descriptors. In addition, we train a tag-conditioned system that supports a predefined set of tags available in MetaScore. Our experimental results show that both the proposed text-to-music and tags-to-music models outperform a baseline text-to-music model in a listening test, while the text-based system offers a more natural interface that allows free-form natural language prompts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16665v1">SafetyAnalyst: Interpretable, transparent, and steerable LLM safety moderation</a></div>
    <div class="paper-meta">
      📅 2024-10-22
    </div>
    <details class="paper-abstract">
      The ideal LLM content moderation system would be both structurally interpretable (so its decisions can be explained to users) and steerable (to reflect a community's values or align to safety standards). However, current systems fall short on both of these dimensions. To address this gap, we present SafetyAnalyst, a novel LLM safety moderation framework. Given a prompt, SafetyAnalyst creates a structured "harm-benefit tree," which identifies 1) the actions that could be taken if a compliant response were provided, 2) the harmful and beneficial effects of those actions (along with their likelihood, severity, and immediacy), and 3) the stakeholders that would be impacted by those effects. It then aggregates this structured representation into a harmfulness score based on a parameterized set of safety preferences, which can be transparently aligned to particular values. Using extensive harm-benefit features generated by SOTA LLMs on 19k prompts, we fine-tuned an open-weight LM to specialize in generating harm-benefit trees through symbolic knowledge distillation. On a comprehensive set of prompt safety benchmarks, we show that our system (average F1=0.75) outperforms existing LLM safety moderation systems (average F1$<$0.72) on prompt harmfulness classification, while offering the additional advantages of interpretability and steerability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16640v1">A Statistical Analysis of LLMs' Self-Evaluation Using Proverbs</a></div>
    <div class="paper-meta">
      📅 2024-10-22
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) such as ChatGPT, GPT-4, Claude-3, and Llama are being integrated across a variety of industries. Despite this rapid proliferation, experts are calling for caution in the interpretation and adoption of LLMs, owing to numerous associated ethical concerns. Research has also uncovered shortcomings in LLMs' reasoning and logical abilities, raising questions on the potential of LLMs as evaluation tools. In this paper, we investigate LLMs' self-evaluation capabilities on a novel proverb reasoning task. We introduce a novel proverb database consisting of 300 proverb pairs that are similar in intent but different in wordings, across topics spanning gender, wisdom, and society. We propose tests to evaluate textual consistencies as well as numerical consistencies across similar proverbs, and demonstrate the effectiveness of our method and dataset in identifying failures in LLMs' self-evaluation which in turn can highlight issues related to gender stereotypes and lack of cultural understanding in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.05010v1">Scattered Forest Search: Smarter Code Space Exploration with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-22
    </div>
    <details class="paper-abstract">
      We propose a novel approach to scaling LLM inference for code generation. We frame code generation as a black box optimization problem within the code space, and employ optimization-inspired techniques to enhance exploration. Specifically, we introduce Scattered Forest Search to enhance solution diversity while searching for solutions. Our theoretical analysis illustrates how these methods avoid local optima during optimization. Extensive experiments on HumanEval, MBPP, APPS, CodeContests, and Leetcode reveal significant performance improvements. For instance, our method achieves a pass@1 rate of 67.1% on HumanEval+ and 87.2% on HumanEval with GPT-3.5, marking improvements of 8.6% and 4.3% over the state-of-the-art, while also halving the iterations needed to find the correct solution. Furthermore, our method scales more efficiently than existing search techniques, including tree search, line search, and repeated sampling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14923v2">Imprompter: Tricking LLM Agents into Improper Tool Use</a></div>
    <div class="paper-meta">
      📅 2024-10-22
      | 💬 website: https://imprompter.ai code: https://github.com/Reapor-Yurnero/imprompter v2 changelog: add new results to Table 3, correct several typos
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) Agents are an emerging computing paradigm that blends generative machine learning with tools such as code interpreters, web browsing, email, and more generally, external resources. These agent-based systems represent an emerging shift in personal computing. We contribute to the security foundations of agent-based systems and surface a new class of automatically computed obfuscated adversarial prompt attacks that violate the confidentiality and integrity of user resources connected to an LLM agent. We show how prompt optimization techniques can find such prompts automatically given the weights of a model. We demonstrate that such attacks transfer to production-level agents. For example, we show an information exfiltration attack on Mistral's LeChat agent that analyzes a user's conversation, picks out personally identifiable information, and formats it into a valid markdown command that results in leaking that data to the attacker's server. This attack shows a nearly 80% success rate in an end-to-end evaluation. We conduct a range of experiments to characterize the efficacy of these attacks and find that they reliably work on emerging agent-based systems like Mistral's LeChat, ChatGLM, and Meta's Llama. These attacks are multimodal, and we show variants in the text-only and image domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16586v1">Optimizing LLMs with Direct Preferences: A Data Efficiency Perspective</a></div>
    <div class="paper-meta">
      📅 2024-10-22
    </div>
    <details class="paper-abstract">
      Aligning the output of Large Language Models (LLMs) with human preferences (e.g., by means of reinforcement learning with human feedback, or RLHF) is essential for ensuring their effectiveness in real-world scenarios. Despite significant advancements in LLM alignment techniques, the impact of different type of preference data on model performance has yet to be systematically explored. In this study, we investigate the scalability, data efficiency, and effectiveness of Direct Preference Optimization (DPO) in fine-tuning pre-trained LLMs, aiming to reduce their dependency on extensive amounts of preference data, which is expensive to collect. We (1) systematically compare the performance of models fine-tuned with varying percentages of a combined preference judgement dataset to define the improvement curve of DPO and assess its effectiveness in data-constrained environments; and (2) provide insights for the development of an optimal approach for selective preference data usage. Our study reveals that increasing the amount of data used for training generally enhances and stabilizes model performance. Moreover, the use of a combination of diverse datasets significantly improves model effectiveness. Furthermore, when models are trained separately using different types of prompts, models trained with conversational prompts outperformed those trained with question answering prompts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15625v1">Improving Parallel Program Performance Through DSL-Driven Code Generation with LLM Optimizers</a></div>
    <div class="paper-meta">
      📅 2024-10-21
      | 💬 26 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Mapping computations to processors and assigning data to memory are critical for maximizing performance in parallel programming. These mapping decisions are managed through the development of specialized low-level system code, called mappers, crafted by performance engineers. Each mapper is tailored to a specific application and optimized for the underlying machine architecture, a process that requires days of refinement and tuning from an expert. Despite advances in system research, automating mapper generation remains a challenge due to the complexity of making millions of decisions to find the optimal solution and generate the solution as code. We introduce an approach that leverages recent advances in LLM-based optimizers for mapper design. In under ten minutes, our method automatically discovers mappers that surpass human expert designs in scientific applications by up to 1.34X speedup. For parallel matrix multiplication algorithms, our mapper achieves up to 1.31X of the expert-designed solution. To achieve this, we simplify the complexity of low-level code generation by introducing a domain-specific language (DSL) that abstracts the low-level system programming details and defines a structured search space for LLMs to explore. To maximize the application performance, we use an LLM optimizer to improve an agentic system that generates the mapper code. As a result, this approach significantly reduces the workload for performance engineers while achieving substantial performance gains across diverse applications. Finally, our results demonstrate the effectiveness of LLM-based optimization in system design and suggest its potential for addressing other complex system challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15623v1">Guardians of Discourse: Evaluating LLMs on Multilingual Offensive Language Detection</a></div>
    <div class="paper-meta">
      📅 2024-10-21
      | 💬 Accepted at UIC 2024 proceedings. Accepted version
    </div>
    <details class="paper-abstract">
      Identifying offensive language is essential for maintaining safety and sustainability in the social media era. Though large language models (LLMs) have demonstrated encouraging potential in social media analytics, they lack thorough evaluation when in offensive language detection, particularly in multilingual environments. We for the first time evaluate multilingual offensive language detection of LLMs in three languages: English, Spanish, and German with three LLMs, GPT-3.5, Flan-T5, and Mistral, in both monolingual and multilingual settings. We further examine the impact of different prompt languages and augmented translation data for the task in non-English contexts. Furthermore, we discuss the impact of the inherent bias in LLMs and the datasets in the mispredictions related to sensitive topics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2212.10678v3">Causally Testing Gender Bias in LLMs: A Case Study on Occupational Bias</a></div>
    <div class="paper-meta">
      📅 2024-10-21
    </div>
    <details class="paper-abstract">
      Generated texts from large language models (LLMs) have been shown to exhibit a variety of harmful, human-like biases against various demographics. These findings motivate research efforts aiming to understand and measure such effects. This paper introduces a causal formulation for bias measurement in generative language models. Based on this theoretical foundation, we outline a list of desiderata for designing robust bias benchmarks. We then propose a benchmark called OccuGender, with a bias-measuring procedure to investigate occupational gender bias. We test several state-of-the-art open-source LLMs on OccuGender, including Llama, Mistral, and their instruction-tuned versions. The results show that these models exhibit substantial occupational gender bias. Lastly, we discuss prompting strategies for bias mitigation and an extension of our causal formulation to illustrate the generalizability of our framework. Our code and data https://github.com/chenyuen0103/gender-bias.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15555v1">Bayesian Concept Bottleneck Models with LLM Priors</a></div>
    <div class="paper-meta">
      📅 2024-10-21
    </div>
    <details class="paper-abstract">
      Concept Bottleneck Models (CBMs) have been proposed as a compromise between white-box and black-box models, aiming to achieve interpretability without sacrificing accuracy. The standard training procedure for CBMs is to predefine a candidate set of human-interpretable concepts, extract their values from the training data, and identify a sparse subset as inputs to a transparent prediction model. However, such approaches are often hampered by the tradeoff between enumerating a sufficiently large set of concepts to include those that are truly relevant versus controlling the cost of obtaining concept extractions. This work investigates a novel approach that sidesteps these challenges: BC-LLM iteratively searches over a potentially infinite set of concepts within a Bayesian framework, in which Large Language Models (LLMs) serve as both a concept extraction mechanism and prior. BC-LLM is broadly applicable and multi-modal. Despite imperfections in LLMs, we prove that BC-LLM can provide rigorous statistical inference and uncertainty quantification. In experiments, it outperforms comparator methods including black-box models, converges more rapidly towards relevant concepts and away from spuriously correlated ones, and is more robust to out-of-distribution samples.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.05357v2">SHIELD: LLM-Driven Schema Induction for Predictive Analytics in EV Battery Supply Chain Disruptions</a></div>
    <div class="paper-meta">
      📅 2024-10-21
      | 💬 Oral, EMNLP 2024 Industry Track. 31 pages, 11 figures, Project: https://fly1113.github.io/MFI/
    </div>
    <details class="paper-abstract">
      The electric vehicle (EV) battery supply chain's vulnerability to disruptions necessitates advanced predictive analytics. We present SHIELD (Schema-based Hierarchical Induction for EV supply chain Disruption), a system integrating Large Language Models (LLMs) with domain expertise for EV battery supply chain risk assessment. SHIELD combines: (1) LLM-driven schema learning to construct a comprehensive knowledge library, (2) a disruption analysis system utilizing fine-tuned language models for event extraction, multi-dimensional similarity matching for schema matching, and Graph Convolutional Networks (GCNs) with logical constraints for prediction, and (3) an interactive interface for visualizing results and incorporating expert feedback to enhance decision-making. Evaluated on 12,070 paragraphs from 365 sources (2022-2023), SHIELD outperforms baseline GCNs and LLM+prompt methods (e.g., GPT-4o) in disruption prediction. These results demonstrate SHIELD's effectiveness in combining LLM capabilities with domain expertise for enhanced supply chain risk assessment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16513v1">SPHERE: Scaling Personalized Feedback in Programming Classrooms with Structured Review of LLM Outputs</a></div>
    <div class="paper-meta">
      📅 2024-10-21
    </div>
    <details class="paper-abstract">
      Effective personalized feedback is crucial for learning programming. However, providing personalized, real-time feedback in large programming classrooms poses significant challenges for instructors. This paper introduces SPHERE, an interactive system that leverages Large Language Models (LLMs) and structured LLM output review to scale personalized feedback for in-class coding activities. SPHERE employs two key components: an Issue Recommendation Component that identifies critical patterns in students' code and discussion, and a Feedback Review Component that uses a ``strategy-detail-verify'' approach for efficient feedback creation and verification. An in-lab, between-subject study demonstrates SPHERE's effectiveness in improving feedback quality and the overall feedback review process compared to a baseline system using off-the-shelf LLM outputs. This work contributes a novel approach to scaling personalized feedback in programming education, addressing the challenges of real-time response, issue prioritization, and large-scale personalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16491v1">BIG5-CHAT: Shaping LLM Personalities Through Training on Human-Grounded Data</a></div>
    <div class="paper-meta">
      📅 2024-10-21
    </div>
    <details class="paper-abstract">
      In this work, we tackle the challenge of embedding realistic human personality traits into LLMs. Previous approaches have primarily focused on prompt-based methods that describe the behavior associated with the desired personality traits, suffering from realism and validity issues. To address these limitations, we introduce BIG5-CHAT, a large-scale dataset containing 100,000 dialogues designed to ground models in how humans express their personality in text. Leveraging this dataset, we explore Supervised Fine-Tuning and Direct Preference Optimization as training-based methods to align LLMs more naturally with human personality patterns. Our methods outperform prompting on personality assessments such as BFI and IPIP-NEO, with trait correlations more closely matching human data. Furthermore, our experiments reveal that models trained to exhibit higher conscientiousness, higher agreeableness, lower extraversion, and lower neuroticism display better performance on reasoning tasks, aligning with psychological findings on how these traits impact human cognitive performance. To our knowledge, this work is the first comprehensive study to demonstrate how training-based methods can shape LLM personalities through learning from real human behaviors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16489v1">LLM-TS Integrator: Integrating LLM for Enhanced Time Series Modeling</a></div>
    <div class="paper-meta">
      📅 2024-10-21
      | 💬 18 pages, 13 figures, 18 tables
    </div>
    <details class="paper-abstract">
      Time series~(TS) modeling is essential in dynamic systems like weather prediction and anomaly detection. Recent studies utilize Large Language Models (LLMs) for TS modeling, leveraging their powerful pattern recognition capabilities. These methods primarily position LLMs as the predictive backbone, often omitting the mathematical modeling within traditional TS models, such as periodicity. However, disregarding the potential of LLMs also overlooks their pattern recognition capabilities. To address this gap, we introduce \textit{LLM-TS Integrator}, a novel framework that effectively integrates the capabilities of LLMs into traditional TS modeling. Central to this integration is our \textit{mutual information} module. The core of this \textit{mutual information} module is a traditional TS model enhanced with LLM-derived insights for improved predictive abilities. This enhancement is achieved by maximizing the mutual information between traditional model's TS representations and LLM's textual representation counterparts, bridging the two modalities. Moreover, we recognize that samples vary in importance for two losses: traditional prediction and mutual information maximization. To address this variability, we introduce the \textit{sample reweighting} module to improve information utilization. This module assigns dual weights to each sample: one for prediction loss and another for mutual information loss, dynamically optimizing these weights via bi-level optimization. Our method achieves state-of-the-art or comparable performance across five mainstream TS tasks, including short-term and long-term forecasting, imputation, classification, and anomaly detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16472v1">DocEdit-v2: Document Structure Editing Via Multimodal LLM Grounding</a></div>
    <div class="paper-meta">
      📅 2024-10-21
      | 💬 EMNLP 2024 (Main)
    </div>
    <details class="paper-abstract">
      Document structure editing involves manipulating localized textual, visual, and layout components in document images based on the user's requests. Past works have shown that multimodal grounding of user requests in the document image and identifying the accurate structural components and their associated attributes remain key challenges for this task. To address these, we introduce the DocEdit-v2, a novel framework that performs end-to-end document editing by leveraging Large Multimodal Models (LMMs). It consists of three novel components: (1) Doc2Command, which simultaneously localizes edit regions of interest (RoI) and disambiguates user edit requests into edit commands; (2) LLM-based Command Reformulation prompting to tailor edit commands originally intended for specialized software into edit instructions suitable for generalist LMMs. (3) Moreover, DocEdit-v2 processes these outputs via Large Multimodal Models like GPT-4V and Gemini, to parse the document layout, execute edits on grounded Region of Interest (RoI), and generate the edited document image. Extensive experiments on the DocEdit dataset show that DocEdit-v2 significantly outperforms strong baselines on edit command generation (2-33%), RoI bounding box detection (12-31%), and overall document editing (1-12\%) tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13886v2">Refusal-Trained LLMs Are Easily Jailbroken As Browser Agents</a></div>
    <div class="paper-meta">
      📅 2024-10-21
    </div>
    <details class="paper-abstract">
      For safety reasons, large language models (LLMs) are trained to refuse harmful user instructions, such as assisting dangerous activities. We study an open question in this work: does the desired safety refusal, typically enforced in chat contexts, generalize to non-chat and agentic use cases? Unlike chatbots, LLM agents equipped with general-purpose tools, such as web browsers and mobile devices, can directly influence the real world, making it even more crucial to refuse harmful instructions. In this work, we primarily focus on red-teaming browser agents, LLMs that manipulate information via web browsers. To this end, we introduce Browser Agent Red teaming Toolkit (BrowserART), a comprehensive test suite designed specifically for red-teaming browser agents. BrowserART is consist of 100 diverse browser-related harmful behaviors (including original behaviors and ones sourced from HarmBench [Mazeika et al., 2024] and AirBench 2024 [Zeng et al., 2024b]) across both synthetic and real websites. Our empirical study on state-of-the-art browser agents reveals that, while the backbone LLM refuses harmful instructions as a chatbot, the corresponding agent does not. Moreover, attack methods designed to jailbreak refusal-trained LLMs in the chat settings transfer effectively to browser agents. With human rewrites, GPT-4o and o1-preview-based browser agents attempted 98 and 63 harmful behaviors (out of 100), respectively. We publicly release BrowserART and call on LLM developers, policymakers, and agent developers to collaborate on improving agent safety
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16392v1">LLM-based Optimization of Compound AI Systems: A Survey</a></div>
    <div class="paper-meta">
      📅 2024-10-21
    </div>
    <details class="paper-abstract">
      In a compound AI system, components such as an LLM call, a retriever, a code interpreter, or tools are interconnected. The system's behavior is primarily driven by parameters such as instructions or tool definitions. Recent advancements enable end-to-end optimization of these parameters using an LLM. Notably, leveraging an LLM as an optimizer is particularly efficient because it avoids gradient computation and can generate complex code and instructions. This paper presents a survey of the principles and emerging trends in LLM-based optimization of compound AI systems. It covers archetypes of compound AI systems, approaches to LLM-based end-to-end optimization, and insights into future directions and broader impacts. Importantly, this survey uses concepts from program analysis to provide a unified view of how an LLM optimizer is prompted to optimize a compound AI system. The exhaustive list of paper is provided at https://github.com/linyuhongg/LLM-based-Optimization-of-Compound-AI-Systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.12168v4">BPO: Staying Close to the Behavior LLM Creates Better Online LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2024-10-21
      | 💬 Wenda Xu and Jiachen Li contributed equally. Accepted by EMNLP 2024
    </div>
    <details class="paper-abstract">
      Direct alignment from preferences (DAP) has emerged as a promising paradigm for aligning large language models (LLMs) to human desiderata from pre-collected, offline preference datasets. While recent studies indicate that existing offline DAP methods can directly benefit from online training samples, we highlight the need to develop specific online DAP algorithms to fully harness the power of online training. Specifically, we identify that the learned LLM should adhere to the proximity of the behavior LLM, which collects the training samples. To this end, we propose online Preference Optimization in proximity to the Behavior LLM (BPO), emphasizing the importance of constructing a proper trust region for LLM alignment. We conduct extensive experiments to validate the effectiveness and applicability of our approach by integrating it with various DAP methods, resulting in significant performance improvements across a wide range of tasks when training with the same amount of preference data. Even when only introducing one additional data collection phase, our online BPO improves its offline DAP baseline from 72.0% to 80.2% on TL;DR and from 82.2% to 89.1% on Anthropic Helpfulness in terms of win rate against human reference text.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16246v1">Analyzing Context Contributions in LLM-based Machine Translation</a></div>
    <div class="paper-meta">
      📅 2024-10-21
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved state-of-the-art performance in machine translation (MT) and demonstrated the ability to leverage in-context learning through few-shot examples. However, the mechanisms by which LLMs use different parts of the input context remain largely unexplored. In this work, we provide a comprehensive analysis of context utilization in MT, studying how LLMs use various context parts, such as few-shot examples and the source text, when generating translations. We highlight several key findings: (1) the source part of few-shot examples appears to contribute more than its corresponding targets, irrespective of translation direction; (2) finetuning LLMs with parallel data alters the contribution patterns of different context parts; and (3) there is a positional bias where earlier few-shot examples have higher contributions to the translated sequence. Finally, we demonstrate that inspecting anomalous context contributions can potentially uncover pathological translations, such as hallucinations. Our findings shed light on the internal workings of LLM-based MT which go beyond those known for standard encoder-decoder MT models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.16276v3">Mechanism Design for LLM Fine-tuning with Multiple Reward Models</a></div>
    <div class="paper-meta">
      📅 2024-10-21
      | 💬 35 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) to aggregate multiple preferences has attracted considerable research attention. With aggregation algorithms advancing, a potential economic scenario arises where fine-tuning services are provided to agents with different preferences. In this context, agents may benefit from strategically misreporting their preferences, which could affect the fine-tuned outcomes. This paper addresses such incentive issues by framing it as a mechanism design problem: an LLM provider determines the fine-tuning objective (training rule) and the pricing scheme (payment rule) for agents. We primarily focus on a representative class of training rules that maximize social welfare subject to certain regularizations, referred to as \tr\ rules. Firstly, we show that under most circumstances, truthful reporting is sub-optimal with simply a training rule, thereby highlighting the necessity of payments. Secondly, we design affine maximizer payment rules that implement \tr\ rules in dominant-strategy incentive compatibility (DSIC). We characterize sufficient conditions for payment equivalence properties. For a training rule that satisfies these conditions, we have found all the payment rules that implement it in DSIC, as they only differ by a constant term irrelevant to agents' reports from each other. Thirdly, we demonstrate that our mechanism is approximately DSIC even with perturbed input, showcasing its robustness against the inevitable errors in real-world applications. Experiments on real LLM setups further confirm the practical implications of our results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16107v1">Do LLMs write like humans? Variation in grammatical and rhetorical styles</a></div>
    <div class="paper-meta">
      📅 2024-10-21
      | 💬 29 pages, 4 figures, 11 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are capable of writing grammatical text that follows instructions, answers questions, and solves problems. As they have advanced, it has become difficult to distinguish their output from human-written text. While past research has found some differences in surface features such as word choice and punctuation, and developed classifiers to detect LLM output, none has studied the rhetorical styles of LLMs. Using several variants of Llama 3 and GPT-4o, we construct two parallel corpora of human- and LLM-written texts from common prompts. Using Douglas Biber's set of lexical, grammatical, and rhetorical features, we identify systematic differences between LLMs and humans and between different LLMs. These differences persist when moving from smaller models to larger ones, and are larger for instruction-tuned models than base models. This demonstrates that despite their advanced abilities, LLMs struggle to match human styles, and hence more advanced linguistic features can detect patterns in their behavior not previously recognized.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.19845v1">Enhancing Trust and Safety in Digital Payments: An LLM-Powered Approach</a></div>
    <div class="paper-meta">
      📅 2024-10-21
      | 💬 10 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Digital payment systems have revolutionized financial transactions, offering unparalleled convenience and accessibility to users worldwide. However, the increasing popularity of these platforms has also attracted malicious actors seeking to exploit their vulnerabilities for financial gain. To address this challenge, robust and adaptable scam detection mechanisms are crucial for maintaining the trust and safety of digital payment ecosystems. This paper presents a comprehensive approach to scam detection, focusing on the Unified Payments Interface (UPI) in India, Google Pay (GPay) as a specific use case. The approach leverages Large Language Models (LLMs) to enhance scam classification accuracy and designs a digital assistant to aid human reviewers in identifying and mitigating fraudulent activities. The results demonstrate the potential of LLMs in augmenting existing machine learning models and improving the efficiency, accuracy, quality, and consistency of scam reviews, ultimately contributing to a safer and more secure digital payment landscape. Our evaluation of the Gemini Ultra model on curated transaction data showed a 93.33% accuracy in scam classification. Furthermore, the model demonstrated 89% accuracy in generating reasoning for these classifications. A promising fact, the model identified 32% new accurate reasons for suspected scams that human reviewers had not included in the review notes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16088v1">Fine-Tuning LLMs for Reliable Medical Question-Answering Services</a></div>
    <div class="paper-meta">
      📅 2024-10-21
      | 💬 8 pages, 10 figures, accepted and to be published in the proceedings of 2024 IEEE International Conference on Data Mining Workshops (ICDMW)
    </div>
    <details class="paper-abstract">
      We present an advanced approach to medical question-answering (QA) services, using fine-tuned Large Language Models (LLMs) to improve the accuracy and reliability of healthcare information. Our study focuses on optimizing models like LLaMA-2 and Mistral, which have shown great promise in delivering precise, reliable medical answers. By leveraging comprehensive datasets, we applied fine-tuning techniques such as rsDoRA+ and ReRAG. rsDoRA+ enhances model performance through a combination of decomposed model weights, varied learning rates for low-rank matrices, and rank stabilization, leading to improved efficiency. ReRAG, which integrates retrieval on demand and question rewriting, further refines the accuracy of the responses. This approach enables healthcare providers to access fast, dependable information, aiding in more efficient decision-making and fostering greater patient trust. Our work highlights the potential of fine-tuned LLMs to significantly improve the quality and accessibility of medical information services, ultimately contributing to better healthcare outcomes for all.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16069v1">Rolling the DICE on Idiomaticity: How LLMs Fail to Grasp Context</a></div>
    <div class="paper-meta">
      📅 2024-10-21
    </div>
    <details class="paper-abstract">
      Human processing of idioms relies on understanding the contextual sentences in which idioms occur, as well as language-intrinsic features such as frequency and speaker-intrinsic factors like familiarity. While LLMs have shown high performance on idiomaticity detection tasks, this success may be attributed to reasoning shortcuts in existing datasets. To this end, we construct a novel, controlled contrastive dataset designed to test whether LLMs can effectively use context to disambiguate idiomatic meaning. Additionally, we explore how collocational frequency and sentence probability influence model performance. Our findings reveal that LLMs often fail to resolve idiomaticity when it is required to attend to the surrounding context, and that models perform better on sentences that have higher likelihood. The collocational frequency of expressions also impacts performance. We make our code and dataset publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16029v1">Natural GaLore: Accelerating GaLore for memory-efficient LLM Training and Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2024-10-21
      | 💬 10 pages, 3 tables, 3 figures
    </div>
    <details class="paper-abstract">
      Training LLMs presents significant memory challenges due to growing size of data, weights, and optimizer states. Techniques such as data and model parallelism, gradient checkpointing, and offloading strategies address this issue but are often infeasible due to hardware constraints. To mitigate memory usage, alternative methods like Parameter-Efficient-Fine-Tuning (PEFT) and GaLore approximate weights or optimizer states. PEFT methods, such as LoRA, have gained popularity for fine-tuning LLMs, though they require a full-rank warm start. In contrast, GaLore allows full-parameter learning while being more memory-efficient. This work introduces Natural GaLore, a simple drop in replacement for AdamW, which efficiently applies the inverse Empirical Fisher Information Matrix to low-rank gradients using Woodbury's Identity. We demonstrate that incorporating second-order information speeds up optimization significantly, especially when the iteration budget is limited. Empirical pretraining on 60M, 130M, 350M, and 1.1B parameter Llama models on C4 data demonstrate significantly lower perplexity over GaLore without additional memory overhead. By fine-tuning RoBERTa on the GLUE benchmark using Natural GaLore, we demonstrate significant reduction in gap 86.05% vs 86.28% for full-finetuning. Furthermore, fine-tuning the TinyLlama 1.1B model for function calling using the TinyAgent framework shows that Natural GaLore achieving 83.09% accuracy on the TinyAgent dataset, significantly outperforms 16-bit LoRA at 80.06% and even surpasses GPT4-Turbo by 4%, all while using 30% less memory. All code to reproduce the results are available at: https://github.com/selfsupervised-ai/Natural-GaLore.git
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.15518v2">Beware of Words: Evaluating the Lexical Diversity of Conversational LLMs using ChatGPT as Case Study</a></div>
    <div class="paper-meta">
      📅 2024-10-21
    </div>
    <details class="paper-abstract">
      The performance of conversational Large Language Models (LLMs) in general, and of ChatGPT in particular, is currently being evaluated on many different tasks, from logical reasoning or maths to answering questions on a myriad of topics. Instead, much less attention is being devoted to the study of the linguistic features of the texts generated by these LLMs. This is surprising since LLMs are models for language, and understanding how they use the language is important. Indeed, conversational LLMs are poised to have a significant impact on the evolution of languages as they may eventually dominate the creation of new text. This means that for example, if conversational LLMs do not use a word it may become less and less frequent and eventually stop being used altogether. Therefore, evaluating the linguistic features of the text they produce and how those depend on the model parameters is the first step toward understanding the potential impact of conversational LLMs on the evolution of languages. In this paper, we consider the evaluation of the lexical richness of the text generated by LLMs and how it depends on the model parameters. A methodology is presented and used to conduct a comprehensive evaluation of lexical richness using ChatGPT as a case study. The results show how lexical richness depends on the version of ChatGPT and some of its parameters, such as the presence penalty, or on the role assigned to the model. The dataset and tools used in our analysis are released under open licenses with the goal of drawing the much-needed attention to the evaluation of the linguistic features of LLM-generated text.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15944v1">Developing Retrieval Augmented Generation (RAG) based LLM Systems from PDFs: An Experience Report</a></div>
    <div class="paper-meta">
      📅 2024-10-21
      | 💬 36 pages, 8 figures, 2 tables, and python code snippets
    </div>
    <details class="paper-abstract">
      This paper presents an experience report on the development of Retrieval Augmented Generation (RAG) systems using PDF documents as the primary data source. The RAG architecture combines generative capabilities of Large Language Models (LLMs) with the precision of information retrieval. This approach has the potential to redefine how we interact with and augment both structured and unstructured knowledge in generative models to enhance transparency, accuracy, and contextuality of responses. The paper details the end-to-end pipeline, from data collection, preprocessing, to retrieval indexing and response generation, highlighting technical challenges and practical solutions. We aim to offer insights to researchers and practitioners developing similar systems using two distinct approaches: OpenAI's Assistant API with GPT Series and Llama's open-source models. The practical implications of this research lie in enhancing the reliability of generative AI systems in various sectors where domain-specific knowledge and real-time information retrieval is important. The Python code used in this work is also available at: https://github.com/GPT-Laboratory/RAG-LLM-Development-Guidebook-from-PDFs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15939v1">CausalGraph2LLM: Evaluating LLMs for Causal Queries</a></div>
    <div class="paper-meta">
      📅 2024-10-21
      | 💬 Code - https://github.com/ivaxi0s/CausalGraph2LLM
    </div>
    <details class="paper-abstract">
      Causality is essential in scientific research, enabling researchers to interpret true relationships between variables. These causal relationships are often represented by causal graphs, which are directed acyclic graphs. With the recent advancements in Large Language Models (LLMs), there is an increasing interest in exploring their capabilities in causal reasoning and their potential use to hypothesize causal graphs. These tasks necessitate the LLMs to encode the causal graph effectively for subsequent downstream tasks. In this paper, we propose a comprehensive benchmark, \emph{CausalGraph2LLM}, encompassing a variety of causal graph settings to assess the causal graph understanding capability of LLMs. We categorize the causal queries into two types: graph-level and node-level queries. We benchmark both open-sourced and closed models for our study. Our findings reveal that while LLMs show promise in this domain, they are highly sensitive to the encoding used. Even capable models like GPT-4 and Gemini-1.5 exhibit sensitivity to encoding, with deviations of about $60\%$. We further demonstrate this sensitivity for downstream causal intervention tasks. Moreover, we observe that LLMs can often display biases when presented with contextual information about a causal graph, potentially stemming from their parametric memory.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.12174v2">Claim Check-Worthiness Detection: How Well do LLMs Grasp Annotation Guidelines?</a></div>
    <div class="paper-meta">
      📅 2024-10-21
      | 💬 Accepted to WASSA at EMNLP 2024
    </div>
    <details class="paper-abstract">
      The increasing threat of disinformation calls for automating parts of the fact-checking pipeline. Identifying text segments requiring fact-checking is known as claim detection (CD) and claim check-worthiness detection (CW), the latter incorporating complex domain-specific criteria of worthiness and often framed as a ranking task. Zero- and few-shot LLM prompting is an attractive option for both tasks, as it bypasses the need for labeled datasets and allows verbalized claim and worthiness criteria to be directly used for prompting. We evaluate the LLMs' predictive and calibration accuracy on five CD/CW datasets from diverse domains, each utilizing a different worthiness criterion. We investigate two key aspects: (1) how best to distill factuality and worthiness criteria into a prompt and (2) what amount of context to provide for each claim. To this end, we experiment with varying the level of prompt verbosity and the amount of contextual information provided to the model. Our results show that optimal prompt verbosity is domain-dependent, adding context does not improve performance, and confidence scores can be directly used to produce reliable check-worthiness rankings.
    </details>
</div>
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
