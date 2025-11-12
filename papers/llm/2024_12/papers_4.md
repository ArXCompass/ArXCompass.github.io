# llm - 2024_12

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- Part 4
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.04049v3">Systematic Biases in LLM Simulations of Debates</a></div>
    <div class="paper-meta">
      📅 2024-12-17
      | 💬 Published as a conference paper at EMNLP 2024
    </div>
    <details class="paper-abstract">
      The emergence of Large Language Models (LLMs), has opened exciting possibilities for constructing computational simulations designed to replicate human behavior accurately. Current research suggests that LLM-based agents become increasingly human-like in their performance, sparking interest in using these AI agents as substitutes for human participants in behavioral studies. However, LLMs are complex statistical learners without straightforward deductive rules, making them prone to unexpected behaviors. Hence, it is crucial to study and pinpoint the key behavioral distinctions between humans and LLM-based agents. In this study, we highlight the limitations of LLMs in simulating human interactions, particularly focusing on LLMs' ability to simulate political debates on topics that are important aspects of people's day-to-day lives and decision-making processes. Our findings indicate a tendency for LLM agents to conform to the model's inherent social biases despite being directed to debate from certain political perspectives. This tendency results in behavioral patterns that seem to deviate from well-established social dynamics among humans. We reinforce these observations using an automatic self-fine-tuning method, which enables us to manipulate the biases within the LLM and demonstrate that agents subsequently align with the altered biases. These results underscore the need for further research to develop methods that help agents overcome these biases, a critical step toward creating more realistic simulations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13103v1">AI PERSONA: Towards Life-long Personalization of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-17
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      In this work, we introduce the task of life-long personalization of large language models. While recent mainstream efforts in the LLM community mainly focus on scaling data and compute for improved capabilities of LLMs, we argue that it is also very important to enable LLM systems, or language agents, to continuously adapt to the diverse and ever-changing profiles of every distinct user and provide up-to-date personalized assistance. We provide a clear task formulation and introduce a simple, general, effective, and scalable framework for life-long personalization of LLM systems and language agents. To facilitate future research on LLM personalization, we also introduce methods to synthesize realistic benchmarks and robust evaluation metrics. We will release all codes and data for building and benchmarking life-long personalized LLM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.09739v2">PersonaMark: Personalized LLM watermarking for model protection and user attribution</a></div>
    <div class="paper-meta">
      📅 2024-12-17
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      The rapid advancement of customized Large Language Models (LLMs) offers considerable convenience. However, it also intensifies concerns regarding the protection of copyright/confidential information. With the extensive adoption of private LLMs, safeguarding model copyright and ensuring data privacy have become critical. Text watermarking has emerged as a viable solution for detecting AI-generated content and protecting models. However, existing methods fall short in providing individualized watermarks for each user, a critical feature for enhancing accountability and traceability. In this paper, we introduce PersonaMark, a novel personalized text watermarking scheme designed to protect LLMs' copyrights and bolster accountability. PersonaMark leverages sentence structure as a subtle carrier of watermark information and optimizes the generation process to maintain the natural output of the model. By employing a personalized hashing function, unique watermarks are embedded for each user, enabling high-quality text generation without compromising the model's performance. This approach is both time-efficient and scalable, capable of handling large numbers of users through a multi-user hashing mechanism. To the best of our knowledge, this is a pioneer study to explore personalized watermarking in LLMs. We conduct extensive evaluations across four LLMs, analyzing various metrics such as perplexity, sentiment, alignment, and readability. The results validate that PersonaMark preserves text quality, ensures unbiased watermark insertion, and offers robust watermark detection capabilities, all while maintaining the model's behavior with minimal disruption.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.12957v2">Towards Reliable Latent Knowledge Estimation in LLMs: Zero-Prompt Many-Shot Based Factual Knowledge Extraction</a></div>
    <div class="paper-meta">
      📅 2024-12-17
    </div>
    <details class="paper-abstract">
      In this paper, we focus on the challenging task of reliably estimating factual knowledge that is embedded inside large language models (LLMs). To avoid reliability concerns with prior approaches, we propose to eliminate prompt engineering when probing LLMs for factual knowledge. Our approach, called Zero-Prompt Latent Knowledge Estimator (ZP-LKE), leverages the in-context learning ability of LLMs to communicate both the factual knowledge question as well as the expected answer format. Our knowledge estimator is both conceptually simpler (i.e., doesn't depend on meta-linguistic judgments of LLMs) and easier to apply (i.e., is not LLM-specific), and we demonstrate that it can surface more of the latent knowledge embedded in LLMs. We also investigate how different design choices affect the performance of ZP-LKE. Using the proposed estimator, we perform a large-scale evaluation of the factual knowledge of a variety of open-source LLMs, like OPT, Pythia, Llama(2), Mistral, Gemma, etc. over a large set of relations and facts from the Wikidata knowledge base. We observe differences in the factual knowledge between different model families and models of different sizes, that some relations are consistently better known than others but that models differ in the precise facts they know, and differences in the knowledge of base models and their finetuned counterparts. Code available at: https://github.com/QinyuanWu0710/ZeroPrompt_LKE
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12981v1">Unlocking LLMs: Addressing Scarce Data and Bias Challenges in Mental Health</a></div>
    <div class="paper-meta">
      📅 2024-12-17
      | 💬 International Conference on Natural Language Processing and Artificial Intelligence for Cyber Security (NLPAICS) 2024
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown promising capabilities in healthcare analysis but face several challenges like hallucinations, parroting, and bias manifestation. These challenges are exacerbated in complex, sensitive, and low-resource domains. Therefore, in this work we introduce IC-AnnoMI, an expert-annotated motivational interviewing (MI) dataset built upon AnnoMI by generating in-context conversational dialogues leveraging LLMs, particularly ChatGPT. IC-AnnoMI employs targeted prompts accurately engineered through cues and tailored information, taking into account therapy style (empathy, reflection), contextual relevance, and false semantic change. Subsequently, the dialogues are annotated by experts, strictly adhering to the Motivational Interviewing Skills Code (MISC), focusing on both the psychological and linguistic dimensions of MI dialogues. We comprehensively evaluate the IC-AnnoMI dataset and ChatGPT's emotional reasoning ability and understanding of domain intricacies by modeling novel classification tasks employing several classical machine learning and current state-of-the-art transformer approaches. Finally, we discuss the effects of progressive prompting strategies and the impact of augmented data in mitigating the biases manifested in IC-AnnoM. Our contributions provide the MI community with not only a comprehensive dataset but also valuable insights for using LLMs in empathetic text generation for conversational therapy in supervised settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12951v1">FineGates: LLMs Finetuning with Compression using Stochastic Gates</a></div>
    <div class="paper-meta">
      📅 2024-12-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), with billions of parameters, present significant challenges for full finetuning due to the high computational demands, memory requirements, and impracticality of many real-world applications. When faced with limited computational resources or small datasets, updating all model parameters can often result in overfitting. To address this, lightweight finetuning techniques have been proposed, like learning low-rank adapter layers. These methods aim to train only a few additional parameters combined with the base model, which remains frozen, reducing resource usage and mitigating overfitting risks. In this work, we propose an adaptor model based on stochastic gates that simultaneously sparsify the frozen base model with task-specific adaptation. Our method comes with a small number of trainable parameters and allows us to speed up the base model inference with competitive accuracy. We evaluate it in additional variants by equipping it with additional low-rank parameters and comparing it to several recent baselines. Our results show that the proposed method improves the finetuned model accuracy comparatively to the several baselines and allows the removal of up to 20-40\% without significant accuracy loss.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09916v2">ProxyLLM : LLM-Driven Framework for Customer Support Through Text-Style Transfer</a></div>
    <div class="paper-meta">
      📅 2024-12-17
    </div>
    <details class="paper-abstract">
      Chatbot-based customer support services have significantly advanced with the introduction of large language models (LLMs), enabling enhanced response quality and broader application across industries. However, while these advancements focus on reducing business costs and improving customer satisfaction, limited attention has been given to the experiences of customer service agents, who are critical to the service ecosystem. A major challenge faced by agents is the stress caused by unnecessary emotional exhaustion from harmful texts, which not only impairs their efficiency but also negatively affects customer satisfaction and business outcomes. In this work, we propose an LLM-powered system designed to enhance the working conditions of customer service agents by addressing emotionally intensive communications. Our proposed system leverages LLMs to transform the tone of customer messages, preserving actionable content while mitigating the emotional impact on human agents. Furthermore, the application is implemented as a Chrome extension, making it highly adaptable and easy to integrate into existing systems. Our method aims to enhance the overall service experience for businesses, customers, and agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.11497v3">CrAM: Credibility-Aware Attention Modification in LLMs for Combating Misinformation in RAG</a></div>
    <div class="paper-meta">
      📅 2024-12-17
      | 💬 AAAI25 camera-ready
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) can alleviate hallucinations of Large Language Models (LLMs) by referencing external documents. However, the misinformation in external documents may mislead LLMs' generation. To address this issue, we explore the task of "credibility-aware RAG", in which LLMs automatically adjust the influence of retrieved documents based on their credibility scores to counteract misinformation. To this end, we introduce a plug-and-play method named $\textbf{Cr}$edibility-aware $\textbf{A}$ttention $\textbf{M}$odification (CrAM). CrAM identifies influential attention heads in LLMs and adjusts their attention weights based on the credibility of the documents, thereby reducing the impact of low-credibility documents. Experiments on Natual Questions and TriviaQA using Llama2-13B, Llama3-8B, and Qwen1.5-7B show that CrAM improves the RAG performance of LLMs against misinformation pollution by over 20%, even surpassing supervised fine-tuning methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.11491v2">SCANS: Mitigating the Exaggerated Safety for LLMs via Safety-Conscious Activation Steering</a></div>
    <div class="paper-meta">
      📅 2024-12-17
      | 💬 Extended version of paper accepted to AAAI 2025. 14 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Safety alignment is indispensable for Large Language Models (LLMs) to defend threats from malicious instructions. However, recent researches reveal safety-aligned LLMs prone to reject benign queries due to the exaggerated safety issue, limiting their helpfulness. In this paper, we propose a Safety-Conscious Activation Steering (SCANS) method to mitigate the exaggerated safety concerns in aligned LLMs. First, SCANS extracts the refusal steering vectors within the activation space and utilizes vocabulary projection to anchor some specific safety-critical layers which influence model refusal behavior. Second, by tracking the hidden state transition, SCANS identifies the steering direction and steers the model behavior accordingly, achieving a balance between exaggerated safety and adequate safety. Experiments show that SCANS achieves new state-of-the-art performance on XSTest and OKTest benchmarks, without impairing their defense capability against harmful queries and maintaining almost unchanged model capability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.10033v3">Can GPT-O1 Kill All Bugs? An Evaluation of GPT-Family LLMs on QuixBugs</a></div>
    <div class="paper-meta">
      📅 2024-12-17
      | 💬 Accepted to the 6th International Workshop on Automated Program Repair (APR 2025)
    </div>
    <details class="paper-abstract">
      LLMs have long demonstrated remarkable effectiveness in automatic program repair (APR), with OpenAI's ChatGPT being one of the most widely used models in this domain. Through continuous iterations and upgrades of GPT-family models, their performance in fixing bugs has already reached state-of-the-art levels. However, there are few works comparing the effectiveness and variations of different versions of GPT-family models on APR. In this work, inspired by the recent public release of the GPT-o1 models, we conduct the first study to compare the effectiveness of different versions of the GPT-family models in APR. We evaluate the performance of the latest version of the GPT-family models (i.e., O1-preview and O1-mini), GPT-4o, and the historical version of ChatGPT on APR. We conduct an empirical study of the four GPT-family models against other LLMs and APR techniques on the QuixBugs benchmark from multiple evaluation perspectives, including repair success rate, repair cost, response length, and behavior patterns. The results demonstrate that O1's repair capability exceeds that of prior GPT-family models, successfully fixing all 40 bugs in the benchmark. Our work can serve as a foundation for further in-depth exploration of the applications of GPT-family models in APR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17679v3">Enhancing Character-Level Understanding in LLMs through Token Internal Structure Learning</a></div>
    <div class="paper-meta">
      📅 2024-12-17
    </div>
    <details class="paper-abstract">
      Tokenization methods like Byte-Pair Encoding (BPE) enhance computational efficiency in large language models (LLMs) but often obscure internal character structures within tokens. This limitation hinders LLMs' ability to predict precise character positions, which is crucial in tasks like Chinese Spelling Correction (CSC) where identifying the positions of misspelled characters accelerates correction processes. We propose Token Internal Position Awareness (TIPA), a method that significantly improves models' ability to capture character positions within tokens by training them on reverse character prediction tasks using the tokenizer's vocabulary. Experiments demonstrate that TIPA enhances position prediction accuracy in LLMs, enabling more precise identification of target characters in original text. Furthermore, when applied to downstream tasks that do not require exact position prediction, TIPA still boosts performance in tasks needing character-level information, validating its versatility and effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.09056v3">Towards Reliable Detection of LLM-Generated Texts: A Comprehensive Evaluation Framework with CUDRT</a></div>
    <div class="paper-meta">
      📅 2024-12-17
      | 💬 30 pages
    </div>
    <details class="paper-abstract">
      The increasing prevalence of large language models (LLMs) has significantly advanced text generation, but the human-like quality of LLM outputs presents major challenges in reliably distinguishing between human-authored and LLM-generated texts. Existing detection benchmarks are constrained by their reliance on static datasets, scenario-specific tasks (e.g., question answering and text refinement), and a primary focus on English, overlooking the diverse linguistic and operational subtleties of LLMs. To address these gaps, we propose CUDRT, a comprehensive evaluation framework and bilingual benchmark in Chinese and English, categorizing LLM activities into five key operations: Create, Update, Delete, Rewrite, and Translate. CUDRT provides extensive datasets tailored to each operation, featuring outputs from state-of-the-art LLMs to assess the reliability of LLM-generated text detectors. This framework supports scalable, reproducible experiments and enables in-depth analysis of how operational diversity, multilingual training sets, and LLM architectures influence detection performance. Our extensive experiments demonstrate the framework's capacity to optimize detection systems, providing critical insights to enhance reliability, cross-linguistic adaptability, and detection accuracy. By advancing robust methodologies for identifying LLM-generated texts, this work contributes to the development of intelligent systems capable of meeting real-world multilingual detection challenges. Source code and dataset are available at GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12841v1">Benchmarking and Understanding Compositional Relational Reasoning of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-17
      | 💬 Accepted to the 39th Annual AAAI Conference on Artificial Intelligence (AAAI-25)
    </div>
    <details class="paper-abstract">
      Compositional relational reasoning (CRR) is a hallmark of human intelligence, but we lack a clear understanding of whether and how existing transformer large language models (LLMs) can solve CRR tasks. To enable systematic exploration of the CRR capability of LLMs, we first propose a new synthetic benchmark called Generalized Associative Recall (GAR) by integrating and generalizing the essence of several tasks in mechanistic interpretability (MI) study in a unified framework. Evaluation shows that GAR is challenging enough for existing LLMs, revealing their fundamental deficiency in CRR. Meanwhile, it is easy enough for systematic MI study. Then, to understand how LLMs solve GAR tasks, we use attribution patching to discover the core circuits reused by Vicuna-33B across different tasks and a set of vital attention heads. Intervention experiments show that the correct functioning of these heads significantly impacts task performance. Especially, we identify two classes of heads whose activations represent the abstract notion of true and false in GAR tasks respectively. They play a fundamental role in CRR across various models and tasks. The dataset and code are available at https://github.com/Caiyun-AI/GAR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12417v1">Bridging the Gap: Enhancing LLM Performance for Low-Resource African Languages with New Benchmarks, Fine-Tuning, and Cultural Adjustments</a></div>
    <div class="paper-meta">
      📅 2024-12-16
      | 💬 Accepted to AAAI 2025. Main content is 9 pages, 3 figures. Includes supplementary materials
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable performance across various tasks, yet significant disparities remain for non-English languages, and especially native African languages. This paper addresses these disparities by creating approximately 1 million human-translated words of new benchmark data in 8 low-resource African languages, covering a population of over 160 million speakers of: Amharic, Bambara, Igbo, Sepedi (Northern Sotho), Shona, Sesotho (Southern Sotho), Setswana, and Tsonga. Our benchmarks are translations of Winogrande and three sections of MMLU: college medicine, clinical knowledge, and virology. Using the translated benchmarks, we report previously unknown performance gaps between state-of-the-art (SOTA) LLMs in English and African languages. Finally, using results from over 400 fine-tuned models, we explore several methods to reduce the LLM performance gap, including high-quality dataset fine-tuning (using an LLM-as-an-Annotator), cross-lingual transfer, and cultural appropriateness adjustments. Key findings include average mono-lingual improvements of 5.6% with fine-tuning (with 5.4% average mono-lingual improvements when using high-quality data over low-quality data), 2.9% average gains from cross-lingual transfer, and a 3.0% out-of-the-box performance boost on culturally appropriate questions. The publicly available benchmarks, translations, and code from this study support further research and development aimed at creating more inclusive and effective language technologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12408v1">Automated Generation of Massive Reasonable Empirical Theorems by Forward Reasoning Based on Strong Relevant Logics -- A Solution to the Problem of LLM Pre-training Data Exhaustion</a></div>
    <div class="paper-meta">
      📅 2024-12-16
      | 💬 11 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Recently, it is often said that the data used for the pre-training of large language models (LLMs) have been exhausted. This paper proposes a solution to the problem: Automated generation of massive reasonable empirical theorems by forward reasoning based on strong relevant logics. In fact, this can be regarded as a part of our approach to the problems of ATF (Automated Theorem Finding) and AKA (Automated Knowledge Appreciation).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12386v1">Interpretable LLM-based Table Question Answering</a></div>
    <div class="paper-meta">
      📅 2024-12-16
    </div>
    <details class="paper-abstract">
      Interpretability for Table Question Answering (Table QA) is critical, particularly in high-stakes industries like finance or healthcare. Although recent approaches using Large Language Models (LLMs) have significantly improved Table QA performance, their explanations for how the answers are generated are ambiguous. To fill this gap, we introduce Plan-of-SQLs ( or POS), an interpretable, effective, and efficient approach to Table QA that answers an input query solely with SQL executions. Through qualitative and quantitative evaluations with human and LLM judges, we show that POS is most preferred among explanation methods, helps human users understand model decision boundaries, and facilitates model success and error identification. Furthermore, when evaluated in standard benchmarks (TabFact, WikiTQ, and FetaQA), POS achieves competitive or superior accuracy compared to existing methods, while maintaining greater efficiency by requiring significantly fewer LLM calls and database queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.15262v1">Advanced ingestion process powered by LLM parsing for RAG system</a></div>
    <div class="paper-meta">
      📅 2024-12-16
      | 💬 12 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Retrieval Augmented Generation (RAG) systems struggle with processing multimodal documents of varying structural complexity. This paper introduces a novel multi-strategy parsing approach using LLM-powered OCR to extract content from diverse document types, including presentations and high text density files both scanned or not. The methodology employs a node-based extraction technique that creates relationships between different information types and generates context-aware metadata. By implementing a Multimodal Assembler Agent and a flexible embedding strategy, the system enhances document comprehension and retrieval capabilities. Experimental evaluations across multiple knowledge bases demonstrate the approach's effectiveness, showing improvements in answer relevancy and information faithfulness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.21359v2">Can Machines Think Like Humans? A Behavioral Evaluation of LLM-Agents in Dictator Games</a></div>
    <div class="paper-meta">
      📅 2024-12-16
    </div>
    <details class="paper-abstract">
      As Large Language Model (LLM)-based agents increasingly undertake real-world tasks and engage with human society, how well do we understand their behaviors? We (1) investigate how LLM agents' prosocial behaviors -- a fundamental social norm -- can be induced by different personas and benchmarked against human behaviors; and (2) introduce a behavioral and social science approach to evaluate LLM agents' decision-making. We explored how different personas and experimental framings affect these AI agents' altruistic behavior in dictator games and compared their behaviors within the same LLM family, across various families, and with human behaviors. The findings reveal substantial variations and inconsistencies among LLMs and notable differences compared to human behaviors. Merely assigning a human-like identity to LLMs does not produce human-like behaviors. Despite being trained on extensive human-generated data, these AI agents are unable to capture the internal processes of human decision-making. Their alignment with human is highly variable and dependent on specific model architectures and prompt formulations; even worse, such dependence does not follow a clear pattern. LLMs can be useful task-specific tools but are not yet intelligent human-like agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12310v1">Second Language (Arabic) Acquisition of LLMs via Progressive Vocabulary Expansion</a></div>
    <div class="paper-meta">
      📅 2024-12-16
    </div>
    <details class="paper-abstract">
      This paper addresses the critical need for democratizing large language models (LLM) in the Arab world, a region that has seen slower progress in developing models comparable to state-of-the-art offerings like GPT-4 or ChatGPT 3.5, due to a predominant focus on mainstream languages (e.g., English and Chinese). One practical objective for an Arabic LLM is to utilize an Arabic-specific vocabulary for the tokenizer that could speed up decoding. However, using a different vocabulary often leads to a degradation of learned knowledge since many words are initially out-of-vocabulary (OOV) when training starts. Inspired by the vocabulary learning during Second Language (Arabic) Acquisition for humans, the released AraLLaMA employs progressive vocabulary expansion, which is implemented by a modified BPE algorithm that progressively extends the Arabic subwords in its dynamic vocabulary during training, thereby balancing the OOV ratio at every stage. The ablation study demonstrated the effectiveness of Progressive Vocabulary Expansion. Moreover, AraLLaMA achieves decent performance comparable to the best Arabic LLMs across a variety of Arabic benchmarks. Models, training data, benchmarks, and codes will be all open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12038v1">LLMs for Cold-Start Cutting Plane Separator Configuration</a></div>
    <div class="paper-meta">
      📅 2024-12-16
    </div>
    <details class="paper-abstract">
      Mixed integer linear programming (MILP) solvers ship with a staggering number of parameters that are challenging to select a priori for all but expert optimization users, but can have an outsized impact on the performance of the MILP solver. Existing machine learning (ML) approaches to configure solvers require training ML models by solving thousands of related MILP instances, generalize poorly to new problem sizes, and often require implementing complex ML pipelines and custom solver interfaces that can be difficult to integrate into existing optimization workflows. In this paper, we introduce a new LLM-based framework to configure which cutting plane separators to use for a given MILP problem with little to no training data based on characteristics of the instance, such as a natural language description of the problem and the associated LaTeX formulation. We augment these LLMs with descriptions of cutting plane separators available in a given solver, grounded by summarizing the existing research literature on separators. While individual solver configurations have a large variance in performance, we present a novel ensembling strategy that clusters and aggregates configurations to create a small portfolio of high-performing configurations. Our LLM-based methodology requires no custom solver interface, can find a high-performing configuration by solving only a small number of MILPs, and can generate the configuration with simple API calls that run in under a second. Numerical results show our approach is competitive with existing configuration approaches on a suite of classic combinatorial optimization problems and real-world datasets with only a fraction of the training data and computation time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.14397v2">RTP-LX: Can LLMs Evaluate Toxicity in Multilingual Scenarios?</a></div>
    <div class="paper-meta">
      📅 2024-12-16
      | 💬 AAAI 2025--camera ready + extended abstract
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) and small language models (SLMs) are being adopted at remarkable speed, although their safety still remains a serious concern. With the advent of multilingual S/LLMs, the question now becomes a matter of scale: can we expand multilingual safety evaluations of these models with the same velocity at which they are deployed? To this end, we introduce RTP-LX, a human-transcreated and human-annotated corpus of toxic prompts and outputs in 28 languages. RTP-LX follows participatory design practices, and a portion of the corpus is especially designed to detect culturally-specific toxic language. We evaluate 10 S/LLMs on their ability to detect toxic content in a culturally-sensitive, multilingual scenario. We find that, although they typically score acceptably in terms of accuracy, they have low agreement with human judges when scoring holistically the toxicity of a prompt; and have difficulty discerning harm in context-dependent scenarios, particularly with subtle-yet-harmful content (e.g. microaggressions, bias). We release this dataset to contribute to further reduce harmful uses of these models and improve their safe deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12004v1">The Open Source Advantage in Large Language Models (LLMs)</a></div>
    <div class="paper-meta">
      📅 2024-12-16
      | 💬 7 pages, 0 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) mark a key shift in natural language processing (NLP), having advanced text generation, translation, and domain-specific reasoning. Closed-source models like GPT-4, powered by proprietary datasets and extensive computational resources, lead with state-of-the-art performance today. However, they face criticism for their "black box" nature and for limiting accessibility in a manner that hinders reproducibility and equitable AI development. By contrast, open-source initiatives like LLaMA and BLOOM prioritize democratization through community-driven development and computational efficiency. These models have significantly reduced performance gaps, particularly in linguistic diversity and domain-specific applications, while providing accessible tools for global researchers and developers. Notably, both paradigms rely on foundational architectural innovations, such as the Transformer framework by Vaswani et al. (2017). Closed-source models excel by scaling effectively, while open-source models adapt to real-world applications in underrepresented languages and domains. Techniques like Low-Rank Adaptation (LoRA) and instruction-tuning datasets enable open-source models to achieve competitive results despite limited resources. To be sure, the tension between closed-source and open-source approaches underscores a broader debate on transparency versus proprietary control in AI. Ethical considerations further highlight this divide. Closed-source systems restrict external scrutiny, while open-source models promote reproducibility and collaboration but lack standardized auditing documentation frameworks to mitigate biases. Hybrid approaches that leverage the strengths of both paradigms are likely to shape the future of LLM innovation, ensuring accessibility, competitive technical performance, and ethical deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.11988v1">SciFaultyQA: Benchmarking LLMs on Faulty Science Question Detection with a GAN-Inspired Approach to Synthetic Dataset Generation</a></div>
    <div class="paper-meta">
      📅 2024-12-16
    </div>
    <details class="paper-abstract">
      Consider the problem: ``If one man and one woman can produce one child in one year, how many children will be produced by one woman and three men in 0.5 years?" Current large language models (LLMs) such as GPT-4o, GPT-o1-preview, and Gemini Flash frequently answer "0.5," which does not make sense. While these models sometimes acknowledge the unrealistic nature of the question, in many cases (8 out of 10 trials), they provide the nonsensical answer of "0.5 child." Additionally, temporal variation has been observed: if an LLM answers correctly once (by recognizing the faulty nature of the question), subsequent responses are more likely to also reflect this understanding. However, this is inconsistent. These types of questions have motivated us to develop a dataset of science questions, SciFaultyQA, where the questions themselves are intentionally faulty. We observed that LLMs often proceed to answer these flawed questions without recognizing their inherent issues, producing results that are logically or scientifically invalid. By analyzing such patterns, we developed a novel method for generating synthetic datasets to evaluate and benchmark the performance of various LLMs in identifying these flawed questions. We have also developed novel approaches to reduce the errors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.11983v1">Cost-Effective Label-free Node Classification with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-16
      | 💬 15 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Graph neural networks (GNNs) have emerged as go-to models for node classification in graph data due to their powerful abilities in fusing graph structures and attributes. However, such models strongly rely on adequate high-quality labeled data for training, which are expensive to acquire in practice. With the advent of large language models (LLMs), a promising way is to leverage their superb zero-shot capabilities and massive knowledge for node labeling. Despite promising results reported, this methodology either demands considerable queries to LLMs, or suffers from compromised performance caused by noisy labels produced by LLMs. To remedy these issues, this work presents Cella, an active self-training framework that integrates LLMs into GNNs in a cost-effective manner. The design recipe of Cella is to iteratively identify small sets of "critical" samples using GNNs and extract informative pseudo-labels for them with both LLMs and GNNs as additional supervision signals to enhance model training. Particularly, Cella includes three major components: (i) an effective active node selection strategy for initial annotations; (ii) a judicious sample selection scheme to sift out the "critical" nodes based on label disharmonicity and entropy; and (iii) a label refinement module combining LLMs and GNNs with rewired topology. Our extensive experiments over five benchmark text-attributed graph datasets demonstrate that Cella significantly outperforms the state of the arts under the same query budget to LLMs in terms of label-free node classification. In particular, on the DBLP dataset with 14.3k nodes, Cella is able to achieve an 8.08% conspicuous improvement in accuracy over the state-of-the-art at a cost of less than one cent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.12701v2">When Backdoors Speak: Understanding LLM Backdoor Attacks Through Model-Generated Explanations</a></div>
    <div class="paper-meta">
      📅 2024-12-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are known to be vulnerable to backdoor attacks, where triggers embedded in poisoned samples can maliciously alter LLMs' behaviors. In this paper, we move beyond attacking LLMs and instead examine backdoor attacks through the novel lens of natural language explanations. Specifically, we leverage LLMs' generative capabilities to produce human-readable explanations for their decisions, enabling direct comparisons between explanations for clean and poisoned samples. Our results show that backdoored models produce coherent explanations for clean inputs but diverse and logically flawed explanations for poisoned data, a pattern consistent across classification and generation tasks for different backdoor attacks. Further analysis reveals key insights into the explanation generation process. At the token level, explanation tokens associated with poisoned samples only appear in the final few transformer layers. At the sentence level, attention dynamics indicate that poisoned inputs shift attention away from the original input context during explanation generation. These findings enhance our understanding of backdoor mechanisms in LLMs and present a promising framework for detecting vulnerabilities through explainability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.16133v3">Uncovering LLM-Generated Code: A Zero-Shot Synthetic Code Detector via Code Rewriting</a></div>
    <div class="paper-meta">
      📅 2024-12-16
      | 💬 Accepted by AAAI 2025; previously submitted to EMNLP 2023
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable proficiency in generating code. However, the misuse of LLM-generated (synthetic) code has raised concerns in both educational and industrial contexts, underscoring the urgent need for synthetic code detectors. Existing methods for detecting synthetic content are primarily designed for general text and struggle with code due to the unique grammatical structure of programming languages and the presence of numerous ''low-entropy'' tokens. Building on this, our work proposes a novel zero-shot synthetic code detector based on the similarity between the original code and its LLM-rewritten variants. Our method is based on the observation that differences between LLM-rewritten and original code tend to be smaller when the original code is synthetic. We utilize self-supervised contrastive learning to train a code similarity model and evaluate our approach on two synthetic code detection benchmarks. Our results demonstrate a significant improvement over existing SOTA synthetic content detectors, with AUROC scores increasing by 20.5% on the APPS benchmark and 29.1% on the MBPP benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.05668v2">Comprehensive Assessment of Jailbreak Attacks Against LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-16
      | 💬 22 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Jailbreak attacks aim to bypass the safeguards of LLMs. While researchers have studied different jailbreak attacks in depth, they have done so in isolation -- either with unaligned experiment settings or comparing a limited range of methods. To fill this gap, we present the first large-scale measurement of various jailbreak attack methods. We collect 17 cutting-edge jailbreak methods, summarize their features, and establish a novel jailbreak attack taxonomy. Based on eight popular censored LLMs and 160 questions from 16 violation categories, we conduct a unified and impartial assessment of attack effectiveness as well as a comprehensive ablation study. Our extensive experimental results demonstrate that all the jailbreak attacks have a powerful effect on the LLMs. This indicates that all LLMs fail to cover all the violation categories, and they are susceptible to significant jailbreak risks, with even the well-aligned Llama3 facing a maximum attack success rate of 0.88. Additionally, we test jailbreak attacks under eight advanced external defenses and find none of the defenses could mitigate the jailbreak attacks entirely. Our study offers valuable insights for future research on jailbreak attacks and defenses and serves as a benchmark tool for researchers and practitioners to evaluate them effectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.15260v1">Analyzing Images of Legal Documents: Toward Multi-Modal LLMs for Access to Justice</a></div>
    <div class="paper-meta">
      📅 2024-12-16
      | 💬 Accepted at AI for Access to Justice Workshop at Jurix 2024, Brno, Czechia. Code and Data available at: https://github.com/hwestermann/AI4A2J_analyzing_images_of_legal_documents
    </div>
    <details class="paper-abstract">
      Interacting with the legal system and the government requires the assembly and analysis of various pieces of information that can be spread across different (paper) documents, such as forms, certificates and contracts (e.g. leases). This information is required in order to understand one's legal rights, as well as to fill out forms to file claims in court or obtain government benefits. However, finding the right information, locating the correct forms and filling them out can be challenging for laypeople. Large language models (LLMs) have emerged as a powerful technology that has the potential to address this gap, but still rely on the user to provide the correct information, which may be challenging and error-prone if the information is only available in complex paper documents. We present an investigation into utilizing multi-modal LLMs to analyze images of handwritten paper forms, in order to automatically extract relevant information in a structured format. Our initial results are promising, but reveal some limitations (e.g., when the image quality is low). Our work demonstrates the potential of integrating multi-modal LLMs to support laypeople and self-represented litigants in finding and assembling relevant information.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.04920v3">GPTKB: Comprehensively Materializing Factual LLM Knowledge</a></div>
    <div class="paper-meta">
      📅 2024-12-16
      | 💬 13 pages, 4 tables, 10 figures
    </div>
    <details class="paper-abstract">
      LLMs have majorly advanced NLP and AI, and next to their ability to perform a wide range of procedural tasks, a major success factor is their internalized factual knowledge. Since (Petroni et al., 2019), analyzing this knowledge has gained attention. However, most approaches investigate one question at a time via modest-sized pre-defined samples, introducing an availability bias (Tversky and Kahnemann, 1973) that prevents the discovery of knowledge (or beliefs) of LLMs beyond the experimenter's predisposition. To address this challenge, we propose a novel methodology to comprehensively materializing an LLM's factual knowledge through recursive querying and result consolidation. As a prototype, we employ GPT-4o-mini to construct GPTKB, a large-scale knowledge base (KB) comprising 105 million triples for over 2.9 million entities - achieved at 1% of the cost of previous KB projects. This work marks a milestone in two areas: For LLM research, for the first time, it provides constructive insights into the scope and structure of LLMs' knowledge (or beliefs). For KB construction, it pioneers new pathways for the long-standing challenge of general-domain KB construction. GPTKB is accessible at https://gptkb.org.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.11763v1">QUENCH: Measuring the gap between Indic and Non-Indic Contextual General Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-16
      | 💬 17 Pages, 6 Figures, 8 Tables, COLING 2025
    </div>
    <details class="paper-abstract">
      The rise of large language models (LLMs) has created a need for advanced benchmarking systems beyond traditional setups. To this end, we introduce QUENCH, a novel text-based English Quizzing Benchmark manually curated and transcribed from YouTube quiz videos. QUENCH possesses masked entities and rationales for the LLMs to predict via generation. At the intersection of geographical context and common sense reasoning, QUENCH helps assess world knowledge and deduction capabilities of LLMs via a zero-shot, open-domain quizzing setup. We perform an extensive evaluation on 7 LLMs and 4 metrics, investigating the influence of model size, prompting style, geographical context, and gold-labeled rationale generation. The benchmarking concludes with an error analysis to which the LLMs are prone.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.11761v1">Harnessing Language for Coordination: A Framework and Benchmark for LLM-Driven Multi-Agent Control</a></div>
    <div class="paper-meta">
      📅 2024-12-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable performance across various tasks. A promising but largely under-explored area is their potential to facilitate human coordination with many agents. Such capabilities would be useful in domains including disaster response, urban planning, and real-time strategy scenarios. In this work, we introduce (1) a real-time strategy game benchmark designed to evaluate these abilities and (2) a novel framework we term HIVE. HIVE empowers a single human to coordinate swarms of up to 2,000 agents using natural language dialog with an LLM. We present promising results on this multi-agent benchmark, with our hybrid approach solving tasks such as coordinating agent movements, exploiting unit weaknesses, leveraging human annotations, and understanding terrain and strategic points. However, our findings also highlight critical limitations of current models, including difficulties in processing spatial visual information and challenges in formulating long-term strategic plans. This work sheds light on the potential and limitations of LLMs in human-swarm coordination, paving the way for future research in this area. The HIVE project page, which includes videos of the system in action, can be found here: hive.syrkis.com.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.11736v1">Personalized LLM for Generating Customized Responses to the Same Query from Different Users</a></div>
    <div class="paper-meta">
      📅 2024-12-16
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      Existing work on large language model (LLM) personalization assigned different responding roles to LLM, but overlooked the diversity of questioners. In this work, we propose a new form of questioner-aware LLM personalization, generating different responses even for the same query from different questioners. We design a dual-tower model architecture with a cross-questioner general encoder and a questioner-specific encoder. We further apply contrastive learning with multi-view augmentation, pulling close the dialogue representations of the same questioner, while pulling apart those of different questioners. To mitigate the impact of question diversity on questioner-contrastive learning, we cluster the dialogues based on question similarity and restrict the scope of contrastive learning within each cluster. We also build a multi-questioner dataset from English and Chinese scripts and WeChat records, called MQDialog, containing 173 questioners and 12 responders. Extensive evaluation with different metrics shows a significant improvement in the quality of personalized response generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.11716v1">LLMs Can Simulate Standardized Patients via Agent Coevolution</a></div>
    <div class="paper-meta">
      📅 2024-12-16
      | 💬 Work in Progress
    </div>
    <details class="paper-abstract">
      Training medical personnel using standardized patients (SPs) remains a complex challenge, requiring extensive domain expertise and role-specific practice. Most research on Large Language Model (LLM)-based simulated patients focuses on improving data retrieval accuracy or adjusting prompts through human feedback. However, this focus has overlooked the critical need for patient agents to learn a standardized presentation pattern that transforms data into human-like patient responses through unsupervised simulations. To address this gap, we propose EvoPatient, a novel simulated patient framework in which a patient agent and doctor agents simulate the diagnostic process through multi-turn dialogues, simultaneously gathering experience to improve the quality of both questions and answers, ultimately enabling human doctor training. Extensive experiments on various cases demonstrate that, by providing only overall SP requirements, our framework improves over existing reasoning methods by more than 10% in requirement alignment and better human preference, while achieving an optimal balance of resource consumption after evolving over 200 cases for 10 hours, with excellent generalizability. The code will be available at https://github.com/ZJUMAI/EvoPatient.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.11699v1">CoinMath: Harnessing the Power of Coding Instruction for Math LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown strong performance in solving mathematical problems, with code-based solutions proving particularly effective. However, the best practice to leverage coding instruction data to enhance mathematical reasoning remains underexplored. This study investigates three key questions: (1) How do different coding styles of mathematical code-based rationales impact LLMs' learning performance? (2) Can general-domain coding instructions improve performance? (3) How does integrating textual rationales with code-based ones during training enhance mathematical reasoning abilities? Our findings reveal that code-based rationales with concise comments, descriptive naming, and hardcoded solutions are beneficial, while improvements from general-domain coding instructions and textual rationales are relatively minor. Based on these insights, we propose CoinMath, a learning strategy designed to enhance mathematical reasoning by diversifying the coding styles of code-based rationales. CoinMath generates a variety of code-based rationales incorporating concise comments, descriptive naming conventions, and hardcoded solutions. Experimental results demonstrate that CoinMath significantly outperforms its baseline model, MAmmoTH, one of the SOTA math LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.11683v1">Multimodal LLM for Intelligent Transportation Systems</a></div>
    <div class="paper-meta">
      📅 2024-12-16
      | 💬 Accepted at IEEE Symposium Series on Computational Intelligence (SSCI) 2025
    </div>
    <details class="paper-abstract">
      In the evolving landscape of transportation systems, integrating Large Language Models (LLMs) offers a promising frontier for advancing intelligent decision-making across various applications. This paper introduces a novel 3-dimensional framework that encapsulates the intersection of applications, machine learning methodologies, and hardware devices, particularly emphasizing the role of LLMs. Instead of using multiple machine learning algorithms, our framework uses a single, data-centric LLM architecture that can analyze time series, images, and videos. We explore how LLMs can enhance data interpretation and decision-making in transportation. We apply this LLM framework to different sensor datasets, including time-series data and visual data from sources like Oxford Radar RobotCar, D-Behavior (D-Set), nuScenes by Motional, and Comma2k19. The goal is to streamline data processing workflows, reduce the complexity of deploying multiple models, and make intelligent transportation systems more efficient and accurate. The study was conducted using state-of-the-art hardware, leveraging the computational power of AMD RTX 3060 GPUs and Intel i9-12900 processors. The experimental results demonstrate that our framework achieves an average accuracy of 91.33\% across these datasets, with the highest accuracy observed in time-series data (92.7\%), showcasing the model's proficiency in handling sequential information essential for tasks such as motion planning and predictive maintenance. Through our exploration, we demonstrate the versatility and efficacy of LLMs in handling multimodal data within the transportation sector, ultimately providing insights into their application in real-world scenarios. Our findings align with the broader conference themes, highlighting the transformative potential of LLMs in advancing transportation technologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.11847v2">Prompto: An open source library for asynchronous querying of LLM endpoints</a></div>
    <div class="paper-meta">
      📅 2024-12-16
    </div>
    <details class="paper-abstract">
      Recent surge in Large Language Model (LLM) availability has opened exciting avenues for research. However, efficiently interacting with these models presents a significant hurdle since LLMs often reside on proprietary or self-hosted API endpoints, each requiring custom code for interaction. Conducting comparative studies between different models can therefore be time-consuming and necessitate significant engineering effort, hindering research efficiency and reproducibility. To address these challenges, we present prompto, an open source Python library which facilitates asynchronous querying of LLM endpoints enabling researchers to interact with multiple LLMs concurrently, while maximising efficiency and utilising individual rate limits. Our library empowers researchers and developers to interact with LLMs more effectively and allowing faster experimentation, data generation and evaluation. prompto is released with an introductory video (https://youtu.be/lWN9hXBOLyQ) under MIT License and is available via GitHub (https://github.com/alan-turing-institute/prompto).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.11672v1">LLM-DaaS: LLM-driven Drone-as-a-Service Operations from Text User Requests</a></div>
    <div class="paper-meta">
      📅 2024-12-16
    </div>
    <details class="paper-abstract">
      We propose LLM-DaaS, a novel Drone-as-a-Service (DaaS) framework that leverages Large Language Models (LLMs) to transform free-text user requests into structured, actionable DaaS operation tasks. Our approach addresses the key challenge of interpreting and structuring natural language input to automate drone service operations under uncertain conditions. The system is composed of three main components: free-text request processing, structured request generation, and dynamic DaaS selection and composition. First, we fine-tune different LLM models such as Phi-3.5, LLaMA-3.2 7b and Gemma 2b on a dataset of text user requests mapped to structured DaaS requests. Users interact with our model in a free conversational style, discussing package delivery requests, while the fine-tuned LLM extracts DaaS metadata such as delivery time, source and destination locations, and package weight. The DaaS service selection model is designed to select the best available drone capable of delivering the requested package from the delivery point to the nearest optimal destination. Additionally, the DaaS composition model composes a service from a set of the best available drones to deliver the package from the source to the final destination. Second, the system integrates real-time weather data to optimize drone route planning and scheduling, ensuring safe and efficient operations. Simulations demonstrate the system's ability to significantly improve task accuracy, operational efficiency, and establish LLM-DaaS as a robust solution for DaaS operations in uncertain environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.13578v2">How Reliable are LLMs as Knowledge Bases? Re-thinking Facutality and Consistency</a></div>
    <div class="paper-meta">
      📅 2024-12-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly explored as knowledge bases (KBs), yet current evaluation methods focus too narrowly on knowledge retention, overlooking other crucial criteria for reliable performance. In this work, we rethink the requirements for evaluating reliable LLM-as-KB usage and highlight two essential factors: factuality, ensuring accurate responses to seen and unseen knowledge, and consistency, maintaining stable answers to questions about the same knowledge. We introduce UnseenQA, a dataset designed to assess LLM performance on unseen knowledge, and propose new criteria and metrics to quantify factuality and consistency, leading to a final reliability score. Our experiments on 26 LLMs reveal several challenges regarding their use as KBs, underscoring the need for more principled and comprehensive evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.11656v1">Private Yet Social: How LLM Chatbots Support and Challenge Eating Disorder Recovery</a></div>
    <div class="paper-meta">
      📅 2024-12-16
    </div>
    <details class="paper-abstract">
      Eating disorders (ED) are complex mental health conditions that require long-term management and support. Recent advancements in large language model (LLM)-based chatbots offer the potential to assist individuals in receiving immediate support. Yet, concerns remain about their reliability and safety in sensitive contexts such as ED. We explore the opportunities and potential harms of using LLM-based chatbots for ED recovery. We observe the interactions between 26 participants with ED and an LLM-based chatbot, WellnessBot, designed to support ED recovery, over 10 days. We discovered that our participants have felt empowered in recovery by discussing ED-related stories with the chatbot, which served as a personal yet social avenue. However, we also identified harmful chatbot responses, especially concerning individuals with ED, that went unnoticed partly due to participants' unquestioning trust in the chatbot's reliability. Based on these findings, we provide design implications for safe and effective LLM-based interventions in ED management.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.11625v1">Fool Me, Fool Me: User Attitudes Toward LLM Falsehoods</a></div>
    <div class="paper-meta">
      📅 2024-12-16
      | 💬 11 pages, 5 figures, 5 tables
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have become central tools in various fields, they often provide inaccurate or false information. This study examines user preferences regarding falsehood responses from LLMs. Specifically, we evaluate preferences for LLM responses where false statements are explicitly marked versus unmarked responses and preferences for confident falsehoods compared to LLM disclaimers acknowledging a lack of knowledge. Additionally, we investigate how requiring users to assess the truthfulness of statements influences these preferences. Surprisingly, 61\% of users prefer unmarked falsehood responses over marked ones, and 69\% prefer confident falsehoods over LLMs admitting lack of knowledge. In all our experiments, a total of 300 users participated, contributing valuable data to our analysis and conclusions. When users are required to evaluate the truthfulness of statements, preferences for unmarked and falsehood responses decrease slightly but remain high. These findings suggest that user preferences, which influence LLM training via feedback mechanisms, may inadvertently encourage the generation of falsehoods. Future research should address the ethical and practical implications of aligning LLM behavior with such preferences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.11618v1">EvoLlama: Enhancing LLMs' Understanding of Proteins via Multimodal Structure and Sequence Representations</a></div>
    <div class="paper-meta">
      📅 2024-12-16
    </div>
    <details class="paper-abstract">
      Current Large Language Models (LLMs) for understanding proteins primarily treats amino acid sequences as a text modality. Meanwhile, Protein Language Models (PLMs), such as ESM-2, have learned massive sequential evolutionary knowledge from the universe of natural protein sequences. Furthermore, structure-based encoders like ProteinMPNN learn the structural information of proteins through Graph Neural Networks. However, whether the incorporation of protein encoders can enhance the protein understanding of LLMs has not been explored. To bridge this gap, we propose EvoLlama, a multimodal framework that connects a structure-based encoder, a sequence-based protein encoder and an LLM for protein understanding. EvoLlama consists of a ProteinMPNN structure encoder, an ESM-2 protein sequence encoder, a multimodal projector to align protein and text representations and a Llama-3 text decoder. To train EvoLlama, we fine-tune it on protein-oriented instructions and protein property prediction datasets verbalized via natural language instruction templates. Our experiments show that EvoLlama's protein understanding capabilities have been significantly enhanced, outperforming other fine-tuned protein-oriented LLMs in zero-shot settings by an average of 1%-8% and surpassing the state-of-the-art baseline with supervised fine-tuning by an average of 6%. On protein property prediction datasets, our approach achieves promising results that are competitive with state-of-the-art task-specific baselines. We will release our code in a future version.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.06561v4">Intention Analysis Makes LLMs A Good Jailbreak Defender</a></div>
    <div class="paper-meta">
      📅 2024-12-16
      | 💬 COLING 2025
    </div>
    <details class="paper-abstract">
      Aligning large language models (LLMs) with human values, particularly when facing complex and stealthy jailbreak attacks, presents a formidable challenge. Unfortunately, existing methods often overlook this intrinsic nature of jailbreaks, which limits their effectiveness in such complex scenarios. In this study, we present a simple yet highly effective defense strategy, i.e., Intention Analysis ($\mathbb{IA}$). $\mathbb{IA}$ works by triggering LLMs' inherent self-correct and improve ability through a two-stage process: 1) analyzing the essential intention of the user input, and 2) providing final policy-aligned responses based on the first round conversation. Notably, $\mathbb{IA}$ is an inference-only method, thus could enhance LLM safety without compromising their helpfulness. Extensive experiments on varying jailbreak benchmarks across a wide range of LLMs show that $\mathbb{IA}$ could consistently and significantly reduce the harmfulness in responses (averagely -48.2% attack success rate). Encouragingly, with our $\mathbb{IA}$, Vicuna-7B even outperforms GPT-3.5 regarding attack success rate. We empirically demonstrate that, to some extent, $\mathbb{IA}$ is robust to errors in generated intentions. Further analyses reveal the underlying principle of $\mathbb{IA}$: suppressing LLM's tendency to follow jailbreak prompts, thereby enhancing safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.11556v1">Token Prepending: A Training-Free Approach for Eliciting Better Sentence Embeddings from LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-16
      | 💬 14 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Extracting sentence embeddings from large language models (LLMs) is a promising direction, as LLMs have demonstrated stronger semantic understanding capabilities. Previous studies typically focus on prompt engineering to elicit sentence embeddings from LLMs by prompting the model to encode sentence information into the embedding of the last token. However, LLMs are mostly decoder-only models with causal attention and the earlier tokens in the sentence cannot attend to the latter tokens, resulting in biased encoding of sentence information and cascading effects on the final decoded token. To this end, we propose a novel Token Prepending (TP) technique that prepends each layer's decoded sentence embedding to the beginning of the sentence in the next layer's input, allowing earlier tokens to attend to the complete sentence information under the causal attention mechanism. The proposed TP technique is a plug-and-play and training-free technique, which means it can be seamlessly integrated with various prompt-based sentence embedding methods and autoregressive LLMs. Extensive experiments on various Semantic Textual Similarity (STS) tasks and downstream classification tasks demonstrate that our proposed TP technique can significantly improve the performance of existing prompt-based sentence embedding methods across different LLMs, while incurring negligible additional inference cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.15734v2">RankAdaptor: Hierarchical Rank Allocation for Efficient Fine-Tuning Pruned LLMs via Performance Model</a></div>
    <div class="paper-meta">
      📅 2024-12-16
    </div>
    <details class="paper-abstract">
      The efficient compression of large language models (LLMs) has become increasingly popular. However, recovering the performance of compressed LLMs remains a major challenge. The current practice in LLM compression entails the implementation of structural pruning, complemented by a recovery phase that leverages the Low-Rank Adaptation (LoRA) algorithm. Structural pruning's uneven modification of model architecture, coupled with standard LoRA's fixed configuration allocation across layers in an online pipeline, leads to suboptimal performance in various downstream tasks for pruned models. To address this challenge, we introduce RankAdaptor, a hierarchical rank allocation method that enables efficient fine-tuning of pruned LLMs according to layerwise specific recovery requirements. We employ a performance model that conducts offline meta-learning and online incremental learning to explore optimal rank values for each layer. Comprehensive experiments on popular benchmarks show that RankAdaptor consistently outperforms state-of-the-art methods across a variety of pruning settings and LLM architectures, with improvements ranging from 0.7\% to 5.5\%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.05299v2">Specifications: The missing link to making the development of LLM systems an engineering discipline</a></div>
    <div class="paper-meta">
      📅 2024-12-16
    </div>
    <details class="paper-abstract">
      Despite the significant strides made by generative AI in just a few short years, its future progress is constrained by the challenge of building modular and robust systems. This capability has been a cornerstone of past technological revolutions, which relied on combining components to create increasingly sophisticated and reliable systems. Cars, airplanes, computers, and software consist of components-such as engines, wheels, CPUs, and libraries-that can be assembled, debugged, and replaced. A key tool for building such reliable and modular systems is specification: the precise description of the expected behavior, inputs, and outputs of each component. However, the generality of LLMs and the inherent ambiguity of natural language make defining specifications for LLM-based components (e.g., agents) both a challenging and urgent problem. In this paper, we discuss the progress the field has made so far-through advances like structured outputs, process supervision, and test-time compute-and outline several future directions for research to enable the development of modular and reliable LLM-based systems through improved specifications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.11536v1">Let your LLM generate a few tokens and you will reduce the need for retrieval</a></div>
    <div class="paper-meta">
      📅 2024-12-16
    </div>
    <details class="paper-abstract">
      In this paper, we investigate how efficiently large language models (LLM) can be trained to check whether an answer is already stored in their parametric memory. We distill an LLM-as-a-judge to compute the IK (I Know) score. We found that this method is particularly beneficial in the context of retrieval-assisted augmented generation (RAG), with a respectable accuracy of 80%. It enables a significant reduction (more than 50%) in the number of search and reranking steps required for certain data sets. We have also introduced the IK score, which serves as a useful tool for characterising datasets by facilitating the classification task. Interestingly, through the inclusion of response tokens as input, our results suggest that only about 20,000 training samples are required to achieve good performance. The central element of this work is the use of a teacher model - the LLM as a judge - to generate training data. We also assess the robustness of the IK classifier by evaluating it with various types of teachers, including both string-based methods and LLMs, with the latter providing better results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14335v2">MQM-APE: Toward High-Quality Error Annotation Predictors with Automatic Post-Editing in LLM Translation Evaluators</a></div>
    <div class="paper-meta">
      📅 2024-12-16
      | 💬 COLING 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown significant potential as judges for Machine Translation (MT) quality assessment, providing both scores and fine-grained feedback. Although approaches such as GEMBA-MQM have shown state-of-the-art performance on reference-free evaluation, the predicted errors do not align well with those annotated by human, limiting their interpretability as feedback signals. To enhance the quality of error annotations predicted by LLM evaluators, we introduce a universal and training-free framework, $\textbf{MQM-APE}$, based on the idea of filtering out non-impactful errors by Automatically Post-Editing (APE) the original translation based on each error, leaving only those errors that contribute to quality improvement. Specifically, we prompt the LLM to act as 1) $\textit{evaluator}$ to provide error annotations, 2) $\textit{post-editor}$ to determine whether errors impact quality improvement and 3) $\textit{pairwise quality verifier}$ as the error filter. Experiments show that our approach consistently improves both the reliability and quality of error spans against GEMBA-MQM, across eight LLMs in both high- and low-resource languages. Orthogonal to trained approaches, MQM-APE complements translation-specific evaluators such as Tower, highlighting its broad applicability. Further analysis confirms the effectiveness of each module and offers valuable insights into evaluator design and LLMs selection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.05583v2">OpenFactCheck: Building, Benchmarking Customized Fact-Checking Systems and Evaluating the Factuality of Claims and LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-16
      | 💬 22 pages, 8 tables, 11 figures
    </div>
    <details class="paper-abstract">
      The increased use of large language models (LLMs) across a variety of real-world applications calls for mechanisms to verify the factual accuracy of their outputs. Difficulties lie in assessing the factuality of free-form responses in open domains. Also, different papers use disparate evaluation benchmarks and measurements, which renders them hard to compare and hampers future progress. To mitigate these issues, we propose OpenFactCheck, a unified framework for building customized automatic fact-checking systems, benchmarking their accuracy, evaluating factuality of LLMs, and verifying claims in a document. OpenFactCheck consists of three modules: (i) CUSTCHECKER allows users to easily customize an automatic fact-checker and verify the factual correctness of documents and claims, (ii) LLMEVAL, a unified evaluation framework assesses LLM's factuality ability from various perspectives fairly, and (iii) CHECKEREVAL is an extensible solution for gauging the reliability of automatic fact-checkers' verification results using human-annotated datasets. Data and code are publicly available at https://github.com/yuxiaw/openfactcheck.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.11506v1">Glimpse: Enabling White-Box Methods to Use Proprietary Models for Zero-Shot LLM-Generated Text Detection</a></div>
    <div class="paper-meta">
      📅 2024-12-16
      | 💬 10 pages, 9 figures, 10 tables
    </div>
    <details class="paper-abstract">
      Advanced large language models (LLMs) can generate text almost indistinguishable from human-written text, highlighting the importance of LLM-generated text detection. However, current zero-shot techniques face challenges as white-box methods are restricted to use weaker open-source LLMs, and black-box methods are limited by partial observation from stronger proprietary LLMs. It seems impossible to enable white-box methods to use proprietary models because API-level access to the models neither provides full predictive distributions nor inner embeddings. To traverse the divide, we propose Glimpse, a probability distribution estimation approach, predicting the full distributions from partial observations. Despite the simplicity of Glimpse, we successfully extend white-box methods like Entropy, Rank, Log-Rank, and Fast-DetectGPT to latest proprietary models. Experiments show that Glimpse with Fast-DetectGPT and GPT-3.5 achieves an average AUROC of about 0.95 in five latest source models, improving the score by 51% relative to the remaining space of the open source baseline (Table 1). It demonstrates that the latest LLMs can effectively detect their own outputs, suggesting that advanced LLMs may be the best shield against themselves.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.11499v1">Embodied CoT Distillation From LLM To Off-the-shelf Agents</a></div>
    <div class="paper-meta">
      📅 2024-12-16
      | 💬 Accepted at ICML 2024
    </div>
    <details class="paper-abstract">
      We address the challenge of utilizing large language models (LLMs) for complex embodied tasks, in the environment where decision-making systems operate timely on capacity-limited, off-the-shelf devices. We present DeDer, a framework for decomposing and distilling the embodied reasoning capabilities from LLMs to efficient, small language model (sLM)-based policies. In DeDer, the decision-making process of LLM-based strategies is restructured into a hierarchy with a reasoning-policy and planning-policy. The reasoning-policy is distilled from the data that is generated through the embodied in-context learning and self-verification of an LLM, so it can produce effective rationales. The planning-policy, guided by the rationales, can render optimized plans efficiently. In turn, DeDer allows for adopting sLMs for both policies, deployed on off-the-shelf devices. Furthermore, to enhance the quality of intermediate rationales, specific to embodied tasks, we devise the embodied knowledge graph, and to generate multiple rationales timely through a single inference, we also use the contrastively prompted attention model. Our experiments with the ALFRED benchmark demonstrate that DeDer surpasses leading language planning and distillation approaches, indicating the applicability and efficiency of sLM-based embodied policies derived through DeDer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.09386v2">Game Development as Human-LLM Interaction</a></div>
    <div class="paper-meta">
      📅 2024-12-16
    </div>
    <details class="paper-abstract">
      Game development is a highly specialized task that relies on a complex game engine powered by complex programming languages, preventing many gaming enthusiasts from handling it. This paper introduces the Chat Game Engine (ChatGE) powered by LLM, which allows everyone to develop a custom game using natural language through Human-LLM interaction. To enable an LLM to function as a ChatGE, we instruct it to perform the following processes in each turn: (1) $P_{script}$: configure the game script segment based on the user's input; (2) $P_{code}$: generate the corresponding code snippet based on the game script segment; (3) $P_{utter}$: interact with the user, including guidance and feedback. We propose a data synthesis pipeline based on LLM to generate game script-code pairs and interactions from a few manually crafted seed data. We propose a three-stage progressive training strategy to transfer the dialogue-based LLM to our ChatGE smoothly. We construct a ChatGE for poker games as a case study and comprehensively evaluate it from two perspectives: interaction quality and code correctness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.06736v2">Are LLMs Rigorous Logical Reasoner? Empowering Natural Language Proof Generation with Contrastive Stepwise Decoding</a></div>
    <div class="paper-meta">
      📅 2024-12-16
      | 💬 The paper is currently undergoing extensive revisions and improvements
    </div>
    <details class="paper-abstract">
      Logical reasoning remains a pivotal component within the realm of artificial intelligence. The recent evolution of large language models (LLMs) has marked significant progress in this domain. The adoption of strategies like chain-of-thought (CoT) has enhanced the performance of LLMs across diverse reasoning tasks. Nonetheless, logical reasoning that involves proof planning, specifically those that necessitate the validation of explanation accuracy, continues to present stumbling blocks. In this study, we first evaluate the efficacy of LLMs with advanced CoT strategies concerning such tasks. Our analysis reveals that LLMs still struggle to navigate complex reasoning chains, which demand the meticulous linkage of premises to derive a cogent conclusion. To address this issue, we finetune a smaller-scale language model, equipping it to decompose proof objectives into more manageable subgoals. We also introduce contrastive decoding to stepwise proof generation, making use of negative reasoning paths to strengthen the model's capacity for logical deduction. Experiments on EntailmentBank underscore the success of our method in augmenting the proof planning abilities of language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.17836v1">Look Ahead Text Understanding and LLM Stitching</a></div>
    <div class="paper-meta">
      📅 2024-12-16
      | 💬 9 pages, 6 figures, 4 tables, published in Vol. 18 (2024): Proceedings of the Eighteenth International AAAI Conference on Web and Social Media
    </div>
    <details class="paper-abstract">
      This paper proposes a look ahead text understanding problem with look ahead section identification (LASI) as an example. This problem may appear in generative AI as well as human interactions, where we want to understand the direction of a developing text or conversation. We tackle the problem using transformer-based LLMs. We show that LASI is more challenging than classic section identification (SI). We argue that both bidirectional contextual information (e.g., BERT) and unidirectional predictive ability (e.g., GPT) will benefit the task. We propose two approaches to stitch together BERT and GPT. Experiments show that our approach outperforms the established models, especially when there is noise in the text (which is often the case for developing text in generative AI). Our paper sheds light on other look ahead text understanding tasks that are important to social media, such as look ahead sentiment classification, and points out the opportunities to leverage pre-trained LLMs through stitching.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.05700v2">InverseCoder: Self-improving Instruction-Tuned Code LLMs with Inverse-Instruct</a></div>
    <div class="paper-meta">
      📅 2024-12-16
      | 💬 Accepted for publication at AAAI 2025. Extended version with full appendix, 18 pages
    </div>
    <details class="paper-abstract">
      Recent advancements in open-source code large language models (LLMs) have been driven by fine-tuning on the data generated from powerful closed-source LLMs, which are expensive to obtain. This paper explores whether it is possible to use a fine-tuned open-source model to generate additional data to augment its instruction-tuning dataset. We make two observations: (1) A code snippet can serve as the response to different instructions. (2) Instruction-tuned code LLMs perform better at translating code into instructions than the reverse. Based on these observations, we propose Inverse-Instruct, a data augmentation technique that uses a fine-tuned LLM to generate additional instructions of code responses from its own training dataset. The additional instruction-response pairs are added to the original dataset, and a stronger code LLM can be obtained by fine-tuning on the augmented dataset. We empirically validate Inverse-Instruct on a range of open-source code models (e.g. CodeLlama-Python and DeepSeek-Coder) and benchmarks (e.g., HumanEval(+), MBPP(+), DS-1000 and MultiPL-E), showing it consistently improves the base models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.15256v1">Structured Extraction of Real World Medical Knowledge using LLMs for Summarization and Search</a></div>
    <div class="paper-meta">
      📅 2024-12-16
      | 💬 10 pages, 3 figures, Work published in 4th Workshop on Knowledge Graphs and Big Data (In Conjunction with IEEE Big Data 2024)
    </div>
    <details class="paper-abstract">
      Creation and curation of knowledge graphs can accelerate disease discovery and analysis in real-world data. While disease ontologies aid in biological data annotation, codified categories (SNOMED-CT, ICD10, CPT) may not capture patient condition nuances or rare diseases. Multiple disease definitions across data sources complicate ontology mapping and disease clustering. We propose creating patient knowledge graphs using large language model extraction techniques, allowing data extraction via natural language rather than rigid ontological hierarchies. Our method maps to existing ontologies (MeSH, SNOMED-CT, RxNORM, HPO) to ground extracted entities. Using a large ambulatory care EHR database with 33.6M patients, we demonstrate our method through the patient search for Dravet syndrome, which received ICD10 recognition in October 2020. We describe our construction of patient-specific knowledge graphs and symptom-based patient searches. Using confirmed Dravet syndrome ICD10 codes as ground truth, we employ LLM-based entity extraction to characterize patients in grounded ontologies. We then apply this method to identify Beta-propeller protein-associated neurodegeneration (BPAN) patients, demonstrating real-world discovery where no ground truth exists.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.11387v1">How Can LLMs and Knowledge Graphs Contribute to Robot Safety? A Few-Shot Learning Approach</a></div>
    <div class="paper-meta">
      📅 2024-12-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are transforming the robotics domain by enabling robots to comprehend and execute natural language instructions. The cornerstone benefits of LLM include processing textual data from technical manuals, instructions, academic papers, and user queries based on the knowledge provided. However, deploying LLM-generated code in robotic systems without safety verification poses significant risks. This paper outlines a safety layer that verifies the code generated by ChatGPT before executing it to control a drone in a simulated environment. The safety layer consists of a fine-tuned GPT-4o model using Few-Shot learning, supported by knowledge graph prompting (KGP). Our approach improves the safety and compliance of robotic actions, ensuring that they adhere to the regulations of drone operations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.11328v1">Zero-Shot Prompting Approaches for LLM-based Graphical User Interface Generation</a></div>
    <div class="paper-meta">
      📅 2024-12-15
    </div>
    <details class="paper-abstract">
      Graphical user interface (GUI) prototyping represents an essential activity in the development of interactive systems, which are omnipresent today. GUI prototypes facilitate elicitation of requirements and help to test, evaluate, and validate ideas with users and the development team. However, creating GUI prototypes is a time-consuming process and often requires extensive resources. While existing research for automatic GUI generation focused largely on resource-intensive training and fine-tuning of LLMs, mainly for low-fidelity GUIs, we investigate the potential and effectiveness of Zero-Shot (ZS) prompting for high-fidelity GUI generation. We propose a Retrieval-Augmented GUI Generation (RAGG) approach, integrated with an LLM-based GUI retrieval re-ranking and filtering mechanism based on a large-scale GUI repository. In addition, we adapt Prompt Decomposition (PDGG) and Self-Critique (SCGG) for GUI generation. To evaluate the effectiveness of the proposed ZS prompting approaches for GUI generation, we extensively evaluated the accuracy and subjective satisfaction of the generated GUI prototypes. Our evaluation, which encompasses over 3,000 GUI annotations from over 100 crowd-workers with UI/UX experience, shows that SCGG, in contrast to PDGG and RAGG, can lead to more effective GUI generation, and provides valuable insights into the defects that are produced by the LLMs in the generated GUI prototypes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03642v2">Aligning LLMs with Individual Preferences via Interaction</a></div>
    <div class="paper-meta">
      📅 2024-12-15
      | 💬 Accepted to COLING 2025. The code and dataset are made public at https://github.com/ShujinWu-0814/ALOE
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) demonstrate increasingly advanced capabilities, aligning their behaviors with human values and preferences becomes crucial for their wide adoption. While previous research focuses on general alignment to principles such as helpfulness, harmlessness, and honesty, the need to account for individual and diverse preferences has been largely overlooked, potentially undermining customized human experiences. To address this gap, we train LLMs that can ''interact to align'', essentially cultivating the meta-skill of LLMs to implicitly infer the unspoken personalized preferences of the current user through multi-turn conversations, and then dynamically align their following behaviors and responses to these inferred preferences. Our approach involves establishing a diverse pool of 3,310 distinct user personas by initially creating seed examples, which are then expanded through iterative self-generation and filtering. Guided by distinct user personas, we leverage multi-LLM collaboration to develop a multi-turn preference dataset containing 3K+ multi-turn conversations in tree structures. Finally, we apply supervised fine-tuning and reinforcement learning to enhance LLMs using this dataset. For evaluation, we establish the ALOE (ALign With CustOmized PrEferences) benchmark, consisting of 100 carefully selected examples and well-designed metrics to measure the customized alignment performance during conversations. Experimental results demonstrate the effectiveness of our method in enabling dynamic, personalized alignment via interaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.05456v1">LLM Based Input Space Partitioning Testing for Library APIs</a></div>
    <div class="paper-meta">
      📅 2024-12-15
      | 💬 11 pages, 10 figures. Proceedings of the 47th IEEE/ACM International Conference on Software Engineering (ICSE 2025)
    </div>
    <details class="paper-abstract">
      Automated library APIs testing is difficult as it requires exploring a vast space of parameter inputs that may involve objects with complex data types. Existing search based approaches, with limited knowledge of relations between object states and program branches, often suffer from the low efficiency issue, i.e., tending to generate invalid inputs. Symbolic execution based approaches can effectively identify such relations, but fail to scale to large programs. In this work, we present an LLM-based input space partitioning testing approach, LISP, for library APIs. The approach leverages LLMs to understand the code of a library API under test and perform input space partitioning based on its understanding and rich common knowledge. Specifically, we provide the signature and code of the API under test to LLMs, with the expectation of obtaining a text description of each input space partition of theAPI under test. Then, we generate inputs through employing the generated text description to sample inputs from each partition, ultimately resulting in test suites that systematically explore the program behavior of the API. We evaluate LISP on more than 2,205 library API methods taken from 10 popular open-source Java libraries (e.g.,apache/commonslang with 2.6k stars, guava with 48.8k stars on GitHub). Our experiment results show that LISP is effective in library API testing. It significantly outperforms state-of-the-art tool EvoSuite in terms of edge coverage. On average, LISP achieves 67.82% branch coverage, surpassing EvoSuite by 1.21 times. In total, LISP triggers 404 exceptions or errors in the experiments, and discovers 13 previously unknown vulnerabilities during evaluation, which have been assigned CVE IDs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.11261v1">CATER: Leveraging LLM to Pioneer a Multidimensional, Reference-Independent Paradigm in Translation Quality Evaluation</a></div>
    <div class="paper-meta">
      📅 2024-12-15
      | 💬 17pages,1sample prompt
    </div>
    <details class="paper-abstract">
      This paper introduces the Comprehensive AI-assisted Translation Edit Ratio (CATER), a novel and fully prompt-driven framework for evaluating machine translation (MT) quality. Leveraging large language models (LLMs) via a carefully designed prompt-based protocol, CATER expands beyond traditional reference-bound metrics, offering a multidimensional, reference-independent evaluation that addresses linguistic accuracy, semantic fidelity, contextual coherence, stylistic appropriateness, and information completeness. CATER's unique advantage lies in its immediate implementability: by providing the source and target texts along with a standardized prompt, an LLM can rapidly identify errors, quantify edit effort, and produce category-level and overall scores. This approach eliminates the need for pre-computed references or domain-specific resources, enabling instant adaptation to diverse languages, genres, and user priorities through adjustable weights and prompt modifications. CATER's LLM-enabled strategy supports more nuanced assessments, capturing phenomena such as subtle omissions, hallucinations, and discourse-level shifts that increasingly challenge contemporary MT systems. By uniting the conceptual rigor of frameworks like MQM and DQF with the scalability and flexibility of LLM-based evaluation, CATER emerges as a valuable tool for researchers, developers, and professional translators worldwide. The framework and example prompts are openly available, encouraging community-driven refinement and further empirical validation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.10927v3">Propulsion: Steering LLM with Tiny Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2024-12-15
      | 💬 26 pages, 11 figures accepted paper
    </div>
    <details class="paper-abstract">
      The rapid advancements in Large Language Models (LLMs) have revolutionized natural language processing (NLP) and related fields. However, fine-tuning these models for specific tasks remains computationally expensive and risks degrading pre-learned features. To address these challenges, we propose Propulsion, a novel parameter efficient fine-tuning (PEFT) method designed to optimize task-specific performance while drastically reducing computational overhead. Inspired by the concept of controlled adjustments in physical motion, Propulsion selectively re-scales specific dimensions of a pre-trained model, guiding output predictions toward task objectives without modifying the model's parameters. By introducing lightweight, trainable Propulsion parameters at the pre-trained layer, we minimize the number of parameters updated during fine-tuning, preventing overfitting or overwriting of existing knowledge. Our theoretical analysis, supported by Neural Tangent Kernel (NTK) theory, shows that Propulsion approximates the performance of full fine-tuning with far fewer trainable parameters. Empirically, Propulsion reduces the parameter count from 355.3 million to just 0.086 million, achieving over a 10x reduction compared to standard approaches like LoRA while maintaining competitive performance across benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.05858v2">Fast On-device LLM Inference with NPUs</a></div>
    <div class="paper-meta">
      📅 2024-12-15
    </div>
    <details class="paper-abstract">
      On-device inference for Large Language Models (LLMs), driven by increasing privacy concerns and advancements of mobile-sized models, has gained significant interest. However, even mobile-sized LLMs (e.g., Gemma-2B) encounter unacceptably high inference latency, often bottlenecked by the prefill stage in tasks like screen UI understanding. We present llm.npu, the first LLM inference system utilizing on-device Neural Processing Unit (NPU) offloading to reduce prefill latency. llm.npu enhances NPU offloading efficiency by re-constructing the prompt and model in three levels: (1) At prompt level, it divides variable-length prompts into multiple fixed-sized chunks while maintaining data dependencies; (2) At tensor level, it identifies and extracts significant outliers to run on the CPU/GPU in parallel with minimal overhead; (3) At block level, it schedules Transformer blocks in an out-of-order manner to the CPU/GPU and NPU based on their hardware affinity and sensitivity to accuracy. Compared to competitive baselines, llm.npu achieves 22.4x faster prefill speed and 30.7$\times$ energy savings on average, and up to 32.8x speedup in an end-to-end real-world application. For the first time, llm.npu achieves more than 1,000 tokens/sec prefilling for a billion-sized model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09048v2">Towards Trustworthy LLMs for Code: A Data-Centric Synergistic Auditing Framework</a></div>
    <div class="paper-meta">
      📅 2024-12-15
      | 💬 Short Vision Paper, Accepted by ICSE'25-NIER
    </div>
    <details class="paper-abstract">
      LLM-powered coding and development assistants have become prevalent to programmers' workflows. However, concerns about the trustworthiness of LLMs for code persist despite their widespread use. Much of the existing research focused on either training or evaluation, raising questions about whether stakeholders in training and evaluation align in their understanding of model trustworthiness and whether they can move toward a unified direction. In this paper, we propose a vision for a unified trustworthiness auditing framework, DataTrust, which adopts a data-centric approach that synergistically emphasizes both training and evaluation data and their correlations. DataTrust aims to connect model trustworthiness indicators in evaluation with data quality indicators in training. It autonomously inspects training data and evaluates model trustworthiness using synthesized data, attributing potential causes from specific evaluation data to corresponding training data and refining indicator connections. Additionally, a trustworthiness arena powered by DataTrust will engage crowdsourced input and deliver quantitative outcomes. We outline the benefits that various stakeholders can gain from DataTrust and discuss the challenges and opportunities it presents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.00326v2">Teola: Towards End-to-End Optimization of LLM-based Applications</a></div>
    <div class="paper-meta">
      📅 2024-12-15
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based applications consist of both LLM and non-LLM components, each contributing to the end-to-end latency. Despite great efforts to optimize LLM inference, end-to-end workflow optimization has been overlooked. Existing frameworks employ coarse-grained orchestration with task modules, which confines optimizations to within each module and yields suboptimal scheduling decisions. We propose fine-grained end-to-end orchestration, which utilizes task primitives as the basic units and represents each query's workflow as a primitive-level dataflow graph. This explicitly exposes a much larger design space, enables optimizations in parallelization and pipelining across primitives of different modules, and enhances scheduling to improve application-level performance. We build Teola, a novel orchestration framework for LLM-based applications that implements this scheme. Comprehensive experiments show that Teola can achieve up to 2.09x speedup over existing systems across various popular LLM applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.19820v1">GaLore$+$: Boosting Low-Rank Adaptation for LLMs with Cross-Head Projection</a></div>
    <div class="paper-meta">
      📅 2024-12-15
    </div>
    <details class="paper-abstract">
      Recent low-rank training methods, such as GaLore, have significantly reduced the memory required to optimize large language models (LLMs). However, these methods often suffer from time-consuming low-rank projection estimations. In particular, the singular value decomposition (SVD) in GaLore can consume more than 80\% of the total training time. To address this issue, we propose GaLore$+$, which uses cross-head low-rank projection to reduce the substantial time consumption in estimating low-rank projections for multi-head attention. In addition, we employ randomized subspace iteration to achieve fast SVD. To further enhance performance, we propose sparsely coded residuals to reduce the errors caused by low-rank approximation on the first- and second-order moments of the optimizers and weight updates. We evaluate GaLore$+$ on arithmetic reasoning and natural language generation datasets. Our experiments demonstrate that GaLore$+$ delivers superior performance while achieving approximately $4\times$ fine-tuning speed compared to vanilla GaLore.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.17296v2">BlockLLM: Memory-Efficient Adaptation of LLMs by Selecting and Optimizing the Right Coordinate Blocks</a></div>
    <div class="paper-meta">
      📅 2024-12-15
      | 💬 18 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) for pretraining or adapting to new tasks and domains has become increasingly critical as their applications expand. However, as the model and the data sizes grow, the training process presents significant memory challenges, often requiring a prohibitive amount of GPU memory that may not be readily available. Existing methods such as low-rank adaptation (LoRA) add trainable low-rank matrix factorizations, altering the training dynamics and limiting the model's parameter search to a low-rank subspace. GaLore, a more recent method, employs Gradient Low-Rank Projection to reduce the memory footprint, in the full parameter training setting. However GaLore can only be applied to a subset of the LLM layers that satisfy the "reversibility" property, thus limiting their applicability. In response to these challenges, we introduce BlockLLM, an approach inspired by block coordinate descent. Our method carefully selects and updates a very small subset of the trainable parameters without altering any part of its architecture and training procedure. BlockLLM achieves state-of-the-art performance in both finetuning and pretraining tasks, while reducing the memory footprint of the underlying optimization process. Our experiments demonstrate that fine-tuning with only less than 5% of the parameters, BlockLLM achieves state-of-the-art perplexity scores on the GLUE benchmarks. On Llama model pretrained on C4 dataset, BlockLLM is able to train with significantly less memory than the state-of-the-art, while still maintaining competitive performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.07791v7">Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2024-12-15
    </div>
    <details class="paper-abstract">
      LLM-as-a-Judge presents a promising alternative to human evaluators across various tasks, but inherent biases, especially position bias - a tendency to favor solutions based on their position in the prompt - have compromised its effectiveness. Our study introduces a systematic framework to examine position bias in pairwise comparisons, focusing on repetition stability, position consistency, and preference fairness. This research significantly contributes to the field by introducing new concepts for understanding position bias and providing a multi-dimensional framework for evaluations. We conducted experiments with 12 LLM judges across MTBench and DevBench, covering 22 tasks and approximately 40 solution-generating models - candidates, resulting in over 100,000 evaluation instances. Our findings confirm that position bias in capable LLM judges is not due to random chances, along with notable variations observed across judges and tasks. Moreover, position bias is weakly influenced by the length of prompt components but significantly impacted by the quality gap between solutions. These insights can help optimize judge model selections, improve benchmark design, and inform future research on debiasing strategies, ultimately enhancing the reliability of LLM judges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.10593v2">An Efficient Sign Language Translation Using Spatial Configuration and Motion Dynamics with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-15
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Gloss-free Sign Language Translation (SLT) converts sign videos directly into spoken language sentences without relying on glosses. Recently, Large Language Models (LLMs) have shown remarkable translation performance in gloss-free methods by harnessing their powerful natural language generation capabilities. However, these methods often rely on domain-specific fine-tuning of visual encoders to achieve optimal results. By contrast, this paper emphasizes the importance of capturing the spatial configurations and motion dynamics inherent in sign language. With this in mind, we introduce Spatial and Motion-based Sign Language Translation (SpaMo), a novel LLM-based SLT framework. The core idea of SpaMo is simple yet effective. We first extract spatial and motion features using off-the-shelf visual encoders and then input these features into an LLM with a language prompt. Additionally, we employ a visual-text alignment process as a warm-up before the SLT supervision. Our experiments demonstrate that SpaMo achieves state-of-the-art performance on two popular datasets, PHOENIX14T and How2Sign.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.11053v1">NITRO: LLM Inference on Intel Laptop NPUs</a></div>
    <div class="paper-meta">
      📅 2024-12-15
      | 💬 11 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become essential tools in natural language processing, finding large usage in chatbots such as ChatGPT and Gemini, and are a central area of research. A particular area of interest includes designing hardware specialized for these AI applications, with one such example being the neural processing unit (NPU). In 2023, Intel released the Intel Core Ultra processor with codename Meteor Lake, featuring a CPU, GPU, and NPU system-on-chip. However, official software support for the NPU through Intel's OpenVINO framework is limited to static model inference. The dynamic nature of autoregressive token generation in LLMs is therefore not supported out of the box. To address this shortcoming, we present NITRO (NPU Inference for Transformers Optimization), a Python-based framework built on top of OpenVINO to support text and chat generation on NPUs. In this paper, we discuss in detail the key modifications made to the transformer architecture to enable inference, some performance benchmarks, and future steps towards improving the package. The code repository for NITRO can be found here: https://github.com/abdelfattah-lab/nitro.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.11026v1">SceneLLM: Implicit Language Reasoning in LLM for Dynamic Scene Graph Generation</a></div>
    <div class="paper-meta">
      📅 2024-12-15
      | 💬 27 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Dynamic scenes contain intricate spatio-temporal information, crucial for mobile robots, UAVs, and autonomous driving systems to make informed decisions. Parsing these scenes into semantic triplets <Subject-Predicate-Object> for accurate Scene Graph Generation (SGG) is highly challenging due to the fluctuating spatio-temporal complexity. Inspired by the reasoning capabilities of Large Language Models (LLMs), we propose SceneLLM, a novel framework that leverages LLMs as powerful scene analyzers for dynamic SGG. Our framework introduces a Video-to-Language (V2L) mapping module that transforms video frames into linguistic signals (scene tokens), making the input more comprehensible for LLMs. To better encode spatial information, we devise a Spatial Information Aggregation (SIA) scheme, inspired by the structure of Chinese characters, which encodes spatial data into tokens. Using Optimal Transport (OT), we generate an implicit language signal from the frame-level token sequence that captures the video's spatio-temporal information. To further improve the LLM's ability to process this implicit linguistic input, we apply Low-Rank Adaptation (LoRA) to fine-tune the model. Finally, we use a transformer-based SGG predictor to decode the LLM's reasoning and predict semantic triplets. Our method achieves state-of-the-art results on the Action Genome (AG) benchmark, and extensive experiments show the effectiveness of SceneLLM in understanding and generating accurate dynamic scene graphs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.11014v1">PromptV: Leveraging LLM-powered Multi-Agent Prompting for High-quality Verilog Generation</a></div>
    <div class="paper-meta">
      📅 2024-12-15
    </div>
    <details class="paper-abstract">
      Recent advances in agentic LLMs have demonstrated remarkable automated Verilog code generation capabilities. However, existing approaches either demand substantial computational resources or rely on LLM-assisted single-agent prompt learning techniques, which we observe for the first time has a degeneration issue - characterized by deteriorating generative performance and diminished error detection and correction capabilities. This paper proposes a novel multi-agent prompt learning framework to address these limitations and enhance code generation quality. We show for the first time that multi-agent architectures can effectively mitigate the degeneration risk while improving code error correction capabilities, resulting in higher-quality Verilog code generation. Experimental results show that the proposed method could achieve 96.4% and 96.5% pass@10 scores on VerilogEval Machine and Human benchmarks, respectively while attaining 100% Syntax and 99.9% Functionality pass@5 metrics on the RTLLM benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.15249v1">LLMs for Literature Review: Are we there yet?</a></div>
    <div class="paper-meta">
      📅 2024-12-15
    </div>
    <details class="paper-abstract">
      Literature reviews are an essential component of scientific research, but they remain time-intensive and challenging to write, especially due to the recent influx of research papers. This paper explores the zero-shot abilities of recent Large Language Models (LLMs) in assisting with the writing of literature reviews based on an abstract. We decompose the task into two components: 1. Retrieving related works given a query abstract, and 2. Writing a literature review based on the retrieved results. We analyze how effective LLMs are for both components. For retrieval, we introduce a novel two-step search strategy that first uses an LLM to extract meaningful keywords from the abstract of a paper and then retrieves potentially relevant papers by querying an external knowledge base. Additionally, we study a prompting-based re-ranking mechanism with attribution and show that re-ranking doubles the normalized recall compared to naive search methods, while providing insights into the LLM's decision-making process. In the generation phase, we propose a two-step approach that first outlines a plan for the review and then executes steps in the plan to generate the actual review. To evaluate different LLM-based literature review methods, we create test sets from arXiv papers using a protocol designed for rolling use with newly released LLMs to avoid test set contamination in zero-shot evaluations. We release this evaluation protocol to promote additional research and development in this regard. Our empirical results suggest that LLMs show promising potential for writing literature reviews when the task is decomposed into smaller components of retrieval and planning. Further, we demonstrate that our planning-based approach achieves higher-quality reviews by minimizing hallucinated references in the generated review by 18-26% compared to existing simpler LLM-based generation methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.10960v1">Can LLMs Help Create Grammar?: Automating Grammar Creation for Endangered Languages with In-Context Learning</a></div>
    <div class="paper-meta">
      📅 2024-12-14
      | 💬 Preprint manuscript. Under revision. Accepted to COLING 2025
    </div>
    <details class="paper-abstract">
      Yes! In the present-day documenting and preserving endangered languages, the application of Large Language Models (LLMs) presents a promising approach. This paper explores how LLMs, particularly through in-context learning, can assist in generating grammatical information for low-resource languages with limited amount of data. We takes Moklen as a case study to evaluate the efficacy of LLMs in producing coherent grammatical rules and lexical entries using only bilingual dictionaries and parallel sentences of the unknown language without building the model from scratch. Our methodology involves organising the existing linguistic data and prompting to efficiently enable to generate formal XLE grammar. Our results demonstrate that LLMs can successfully capture key grammatical structures and lexical information, although challenges such as the potential for English grammatical biases remain. This study highlights the potential of LLMs to enhance language documentation efforts, providing a cost-effective solution for generating linguistic data and contributing to the preservation of endangered languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.01002v2">Attribute Structuring Improves LLM-Based Evaluation of Clinical Text Summaries</a></div>
    <div class="paper-meta">
      📅 2024-12-14
      | 💬 Published in ML4H Findings 2024, 4 pages
    </div>
    <details class="paper-abstract">
      Summarizing clinical text is crucial in health decision-support and clinical research. Large language models (LLMs) have shown the potential to generate accurate clinical text summaries, but still struggle with issues regarding grounding and evaluation, especially in safety-critical domains such as health. Holistically evaluating text summaries is challenging because they may contain unsubstantiated information. Here, we explore a general mitigation framework using Attribute Structuring (AS), which structures the summary evaluation process. It decomposes the evaluation process into a grounded procedure that uses an LLM for relatively simple structuring and scoring tasks, rather than the full task of holistic summary evaluation. Experiments show that AS consistently improves the correspondence between human annotations and automated metrics in clinical text summarization. Additionally, AS yields interpretations in the form of a short text span corresponding to each output, which enables efficient human auditing, paving the way towards trustworthy evaluation of clinical information in resource-constrained scenarios. We release our code, prompts, and an open-source benchmark at https://github.com/microsoft/attribute-structuring.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.10918v1">LLMs-in-the-Loop Part 2: Expert Small AI Models for Anonymization and De-identification of PHI Across Multiple Languages</a></div>
    <div class="paper-meta">
      📅 2024-12-14
      | 💬 21 pages, 7 tables
    </div>
    <details class="paper-abstract">
      The rise of chronic diseases and pandemics like COVID-19 has emphasized the need for effective patient data processing while ensuring privacy through anonymization and de-identification of protected health information (PHI). Anonymized data facilitates research without compromising patient confidentiality. This paper introduces expert small AI models developed using the LLM-in-the-loop methodology to meet the demand for domain-specific de-identification NER models. These models overcome the privacy risks associated with large language models (LLMs) used via APIs by eliminating the need to transmit or store sensitive data. More importantly, they consistently outperform LLMs in de-identification tasks, offering superior performance and reliability. Our de-identification NER models, developed in eight languages (English, German, Italian, French, Romanian, Turkish, Spanish, and Arabic) achieved f1-micro score averages of 0.966, 0.975, 0.976, 0.970, 0.964, 0.974, 0.978, and 0.953 respectively. These results establish them as the most accurate healthcare anonymization solutions, surpassing existing small models and even general-purpose LLMs such as GPT-4o. While Part-1 of this series introduced the LLM-in-the-loop methodology for bio-medical document translation, this second paper showcases its success in developing cost-effective expert small NER models in de-identification tasks. Our findings lay the groundwork for future healthcare AI innovations, including biomedical entity and relation extraction, demonstrating the value of specialized models for domain-specific challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.10906v1">SusGen-GPT: A Data-Centric LLM for Financial NLP and Sustainability Report Generation</a></div>
    <div class="paper-meta">
      📅 2024-12-14
    </div>
    <details class="paper-abstract">
      The rapid growth of the financial sector and the rising focus on Environmental, Social, and Governance (ESG) considerations highlight the need for advanced NLP tools. However, open-source LLMs proficient in both finance and ESG domains remain scarce. To address this gap, we introduce SusGen-30K, a category-balanced dataset comprising seven financial NLP tasks and ESG report generation, and propose TCFD-Bench, a benchmark for evaluating sustainability report generation. Leveraging this dataset, we developed SusGen-GPT, a suite of models achieving state-of-the-art performance across six adapted and two off-the-shelf tasks, trailing GPT-4 by only 2% despite using 7-8B parameters compared to GPT-4's 1,700B. Based on this, we propose the SusGen system, integrated with Retrieval-Augmented Generation (RAG), to assist in sustainability report generation. This work demonstrates the efficiency of our approach, advancing research in finance and ESG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.10904v1">CEKER: A Generalizable LLM Framework for Literature Analysis with a Case Study in Unikernel Security</a></div>
    <div class="paper-meta">
      📅 2024-12-14
      | 💬 7 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Literature reviews are a critical component of formulating and justifying new research, but are a manual and often time-consuming process. This research introduces a novel, generalizable approach to literature analysis called CEKER which uses a three-step process to streamline the collection of literature, the extraction of key insights, and the summarized analysis of key trends and gaps. Leveraging Large Language Models (LLMs), this methodology represents a significant shift from traditional manual literature reviews, offering a scalable, flexible, and repeatable approach that can be applied across diverse research domains. A case study on unikernel security illustrates CEKER's ability to generate novel insights validated against previous manual methods. CEKER's analysis highlighted reduced attack surface as the most prominent theme. Key security gaps included the absence of Address Space Layout Randomization, missing debugging tools, and limited entropy generation, all of which represent important challenges to unikernel security. The study also revealed a reliance on hypervisors as a potential attack vector and emphasized the need for dynamic security adjustments to address real-time threats.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.01789v2">Generating executable oracles to check conformance of client code to requirements of JDK Javadocs using LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-14
    </div>
    <details class="paper-abstract">
      Software testing remains the most widely used methodology for validating quality of code. However, effectiveness of testing critically depends on the quality of test suites used. Test cases in a test suite consist of two fundamental parts: (1) input values for the code under test, and (2) correct checks for the outputs it produces. These checks are commonly written as assertions, and termed test oracles. The last couple of decades have seen much progress in automated test input generation, e.g., using fuzzing and symbolic execution. However, automating test oracles remains a relatively less explored problem area. Indeed, a test oracle by its nature requires knowledge of expected behavior, which may only be known to the developer and may not not exist in a formal language that supports automated reasoning. Our focus in this paper is automation of test oracles for clients of widely used Java libraries, e.g., java.lang and java.util packages. Our key insight is that Javadocs that provide a rich source of information can enable automated generation of test oracles. Javadocs of the core Java libraries are fairly detailed documents that contain natural language descriptions of not only how the libraries behave but also how the clients must (not) use them. We use large language models as an enabling technology to embody our insight into a framework for test oracle automation, and evaluate it experimentally. Our experiments demonstrate that LLMs can generate oracles for checking normal and exceptional behaviors from Javadocs, with 98.8% of these oracles being compilable and 96.4% accurately reflecting intended properties. Even for the few incorrect oracles, errors are minor and can be easily corrected with the help of additional comment information generated by the LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.03934v2">From Words to Worth: Newborn Article Impact Prediction with LLM</a></div>
    <div class="paper-meta">
      📅 2024-12-14
      | 💬 Accepted by AAAI 2025. Code, dataset are released at https://sway.cloud.microsoft/KOH09sPR21Ubojbc
    </div>
    <details class="paper-abstract">
      As the academic landscape expands, the challenge of efficiently identifying impactful newly published articles grows increasingly vital. This paper introduces a promising approach, leveraging the capabilities of LLMs to predict the future impact of newborn articles solely based on titles and abstracts. Moving beyond traditional methods heavily reliant on external information, the proposed method employs LLM to discern the shared semantic features of highly impactful papers from a large collection of title-abstract pairs. These semantic features are further utilized to predict the proposed indicator, TNCSI_SP, which incorporates favorable normalization properties across value, field, and time. To facilitate parameter-efficient fine-tuning of the LLM, we have also meticulously curated a dataset containing over 12,000 entries, each annotated with titles, abstracts, and their corresponding TNCSI_SP values. The quantitative results, with an MAE of 0.216 and an NDCG@20 of 0.901, demonstrate that the proposed approach achieves state-of-the-art performance in predicting the impact of newborn articles when compared to several promising methods. Finally, we present a real-world application example for predicting the impact of newborn journal articles to demonstrate its noteworthy practical value. Overall, our findings challenge existing paradigms and propose a shift towards a more content-focused prediction of academic impact, offering new insights for article impact prediction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.10823v1">FinGPT: Enhancing Sentiment-Based Stock Movement Prediction with Dissemination-Aware and Context-Enriched LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-14
      | 💬 1st Workshop on Preparing Good Data for Generative AI: Challenges and Approaches@ AAAI 2025
    </div>
    <details class="paper-abstract">
      Financial sentiment analysis is crucial for understanding the influence of news on stock prices. Recently, large language models (LLMs) have been widely adopted for this purpose due to their advanced text analysis capabilities. However, these models often only consider the news content itself, ignoring its dissemination, which hampers accurate prediction of short-term stock movements. Additionally, current methods often lack sufficient contextual data and explicit instructions in their prompts, limiting LLMs' ability to interpret news. In this paper, we propose a data-driven approach that enhances LLM-powered sentiment-based stock movement predictions by incorporating news dissemination breadth, contextual data, and explicit instructions. We cluster recent company-related news to assess its reach and influence, enriching prompts with more specific data and precise instructions. This data is used to construct an instruction tuning dataset to fine-tune an LLM for predicting short-term stock price movements. Our experimental results show that our approach improves prediction accuracy by 8\% compared to existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12196v1">TrendSim: Simulating Trending Topics in Social Media Under Poisoning Attacks with LLM-based Multi-agent System</a></div>
    <div class="paper-meta">
      📅 2024-12-14
      | 💬 19 pages, 9 tables, 8 figure
    </div>
    <details class="paper-abstract">
      Trending topics have become a significant part of modern social media, attracting users to participate in discussions of breaking events. However, they also bring in a new channel for poisoning attacks, resulting in negative impacts on society. Therefore, it is urgent to study this critical problem and develop effective strategies for defense. In this paper, we propose TrendSim, an LLM-based multi-agent system to simulate trending topics in social media under poisoning attacks. Specifically, we create a simulation environment for trending topics that incorporates a time-aware interaction mechanism, centralized message dissemination, and an interactive system. Moreover, we develop LLM-based human-like agents to simulate users in social media, and propose prototype-based attackers to replicate poisoning attacks. Besides, we evaluate TrendSim from multiple aspects to validate its effectiveness. Based on TrendSim, we conduct simulation experiments to study four critical problems about poisoning attacks on trending topics for social benefit.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06684v2">Exploring Critical Testing Scenarios for Decision-Making Policies: An LLM Approach</a></div>
    <div class="paper-meta">
      📅 2024-12-14
      | 💬 16 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Recent advances in decision-making policies have led to significant progress in fields such as autonomous driving and robotics. However, testing these policies remains crucial with the existence of critical scenarios that may threaten their reliability. Despite ongoing research, challenges such as low testing efficiency and limited diversity persist due to the complexity of the decision-making policies and their environments. To address these challenges, this paper proposes an adaptable Large Language Model (LLM)-driven online testing framework to explore critical and diverse testing scenarios for decision-making policies. Specifically, we design a "generate-test-feedback" pipeline with templated prompt engineering to harness the world knowledge and reasoning abilities of LLMs. Additionally, a multi-scale scenario generation strategy is proposed to address the limitations of LLMs in making fine-grained adjustments, further enhancing testing efficiency. Finally, the proposed LLM-driven method is evaluated on five widely recognized benchmarks, and the experimental results demonstrate that our method significantly outperforms baseline methods in uncovering both critical and diverse scenarios. These findings suggest that LLM-driven methods hold significant promise for advancing the testing of decision-making policies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.10742v1">WEPO: Web Element Preference Optimization for LLM-based Web Navigation</a></div>
    <div class="paper-meta">
      📅 2024-12-14
      | 💬 Published at AAAI 2025
    </div>
    <details class="paper-abstract">
      The rapid advancement of autonomous web navigation has significantly benefited from grounding pretrained Large Language Models (LLMs) as agents. However, current research has yet to fully leverage the redundancy of HTML elements for contrastive training. This paper introduces a novel approach to LLM-based web navigation tasks, called Web Element Preference Optimization (WEPO). WEPO utilizes unsupervised preference learning by sampling distance-based non-salient web elements as negative samples, optimizing maximum likelihood objective within Direct Preference Optimization (DPO). We evaluate WEPO on the Mind2Web benchmark and empirically demonstrate that WEPO aligns user high-level intent with output actions more effectively. The results show that our method achieved the state-of-the-art, with an improvement of 13.8% over WebAgent and 5.3% over the visual language model CogAgent baseline. Our findings underscore the potential of preference optimization to enhance web navigation and other web page based tasks, suggesting a promising direction for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.09672v2">InstructPipe: Building Visual Programming Pipelines with Human Instructions Using LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-14
    </div>
    <details class="paper-abstract">
      Visual programming has the potential of providing novice programmers with a low-code experience to build customized processing pipelines. Existing systems typically require users to build pipelines from scratch, implying that novice users are expected to set up and link appropriate nodes from a blank workspace. In this paper, we introduce InstructPipe, an AI assistant for prototyping machine learning (ML) pipelines with text instructions. We contribute two large language model (LLM) modules and a code interpreter as part of our framework. The LLM modules generate pseudocode for a target pipeline, and the interpreter renders the pipeline in the node-graph editor for further human-AI collaboration. Both technical and user evaluation (N=16) shows that InstructPipe empowers users to streamline their ML pipeline workflow, reduce their learning curve, and leverage open-ended commands to spark innovative ideas.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.10689v1">Learning to Verify Summary Facts with Fine-Grained LLM Feedback</a></div>
    <div class="paper-meta">
      📅 2024-12-14
      | 💬 Accepted at COLING 2025
    </div>
    <details class="paper-abstract">
      Training automatic summary fact verifiers often faces the challenge of a lack of human-labeled data. In this paper, we explore alternative way of leveraging Large Language Model (LLM) generated feedback to address the inherent limitation of using human-labeled data. We introduce FineSumFact, a large-scale dataset containing fine-grained factual feedback on summaries. We employ 10 distinct LLMs for diverse summary generation and Llama-3-70B-Instruct for feedback. We utilize this dataset to fine-tune the lightweight open-source model Llama-3-8B-Instruct, optimizing resource efficiency while maintaining high performance. Our experimental results reveal that the model trained on extensive LLM-generated datasets surpasses that trained on smaller human-annotated datasets when evaluated using human-generated test sets. Fine-tuning fact verification models with LLM feedback can be more effective and cost-efficient than using human feedback. The dataset is available at https://github.com/DISL-Lab/FineSumFact.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.10675v1">Chasing Progress, Not Perfection: Revisiting Strategies for End-to-End LLM Plan Generation</a></div>
    <div class="paper-meta">
      📅 2024-12-14
      | 💬 8 pages main body, 10 pages appendix, accepted by Workshop on Planning in the Era of LLMs (LM4Plan @ AAAI 2025)
    </div>
    <details class="paper-abstract">
      The capability of Large Language Models (LLMs) to plan remains a topic of debate. Some critics argue that strategies to boost LLMs' reasoning skills are ineffective in planning tasks, while others report strong outcomes merely from training models on a planning corpus. This study reassesses recent strategies by developing an end-to-end LLM planner and employing diverse metrics for a thorough evaluation. We find that merely fine-tuning LLMs on a corpus of planning instances does not lead to robust planning skills, as indicated by poor performance on out-of-distribution test sets. At the same time, we find that various strategies, including Chain-of-Thought, do enhance the probability of a plan being executable. This indicates progress towards better plan quality, despite not directly enhancing the final validity rate. Among the strategies we evaluated, reinforcement learning with our novel `Longest Contiguous Common Subsequence' reward emerged as the most effective, contributing to both plan validity and executability. Overall, our research addresses key misconceptions in the LLM-planning literature; we validate incremental progress in plan executability, although plan validity remains a challenge. Hence, future strategies should focus on both these aspects, drawing insights from our findings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.10654v1">Thinking with Knowledge Graphs: Enhancing LLM Reasoning Through Structured Data</a></div>
    <div class="paper-meta">
      📅 2024-12-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities in natural language understanding and generation. However, they often struggle with complex reasoning tasks and are prone to hallucination. Recent research has shown promising results in leveraging knowledge graphs (KGs) to enhance LLM performance. KGs provide a structured representation of entities and their relationships, offering a rich source of information that can enhance the reasoning capabilities of LLMs. For this work, we have developed different techniques that tightly integrate KG structures and semantics into LLM representations. Our results show that we are able to significantly improve the performance of LLMs in complex reasoning scenarios, and ground the reasoning process with KGs. We are the first to represent KGs with programming language and fine-tune pretrained LLMs with KGs. This integration facilitates more accurate and interpretable reasoning processes, paving the way for more advanced reasoning capabilities of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09318v2">Benchmarking LLMs for Mimicking Child-Caregiver Language in Interaction</a></div>
    <div class="paper-meta">
      📅 2024-12-13
    </div>
    <details class="paper-abstract">
      LLMs can generate human-like dialogues, yet their ability to simulate early child-adult interactions remains largely unexplored. In this paper, we examined how effectively LLMs can capture the distinctive features of child-caregiver language in interaction, using both static and interactive benchmarking methods. We found that state-of-the-art LLMs like Llama 3 and GPT-4o can approximate child-caregiver dialogues at the word and utterance level, but they struggle to reproduce the child and caregiver's discursive patterns, exaggerate alignment, and fail to reach the level of diversity shown by humans. The broader goal of this work is to initiate the development of a comprehensive benchmark for LLMs in child-oriented applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09993v1">A Comparative Study of LLMs, NMT Models, and Their Combination in Persian-English Idiom Translation</a></div>
    <div class="paper-meta">
      📅 2024-12-13
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown superior capabilities in translating figurative language compared to neural machine translation (NMT) systems. However, the impact of different prompting methods and LLM-NMT combinations on idiom translation has yet to be thoroughly investigated. This paper introduces two parallel datasets of sentences containing idiomatic expressions for Persian$\rightarrow$English and English$\rightarrow$Persian translations, with Persian idioms sampled from our PersianIdioms resource, a collection of 2,200 idioms and their meanings. Using these datasets, we evaluate various open- and closed-source LLMs, NMT models, and their combinations. Translation quality is assessed through idiom translation accuracy and fluency. We also find that automatic evaluation methods like LLM-as-a-judge, BLEU and BERTScore are effective for comparing different aspects of model performance. Our experiments reveal that Claude-3.5-Sonnet delivers outstanding results in both translation directions. For English$\rightarrow$Persian, combining weaker LLMs with Google Translate improves results, while Persian$\rightarrow$English translations benefit from single prompts for simpler models and complex prompts for advanced ones.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04680v2">Dynamic Fog Computing for Enhanced LLM Execution in Medical Applications</a></div>
    <div class="paper-meta">
      📅 2024-12-13
    </div>
    <details class="paper-abstract">
      The ability of large language models (LLMs) to transform, interpret, and comprehend vast quantities of heterogeneous data presents a significant opportunity to enhance data-driven care delivery. However, the sensitive nature of protected health information (PHI) raises valid concerns about data privacy and trust in remote LLM platforms. In addition, the cost associated with cloud-based artificial intelligence (AI) services continues to impede widespread adoption. To address these challenges, we propose a shift in the LLM execution environment from opaque, centralized cloud providers to a decentralized and dynamic fog computing architecture. By executing open-weight LLMs in more trusted environments, such as the user's edge device or a fog layer within a local network, we aim to mitigate the privacy, trust, and financial challenges associated with cloud-based LLMs. We further present SpeziLLM, an open-source framework designed to facilitate rapid and seamless leveraging of different LLM execution layers and lowering barriers to LLM integration in digital health applications. We demonstrate SpeziLLM's broad applicability across six digital health applications, showcasing its versatility in various healthcare settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.15240v1">ChainStream: An LLM-based Framework for Unified Synthetic Sensing</a></div>
    <div class="paper-meta">
      📅 2024-12-13
      | 💬 18 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Many applications demand context sensing to offer personalized and timely services. Yet, developing sensing programs can be challenging for developers and using them is privacy-concerning for end-users. In this paper, we propose to use natural language as the unified interface to process personal data and sense user context, which can effectively ease app development and make the data pipeline more transparent. Our work is inspired by large language models (LLMs) and other generative models, while directly applying them does not solve the problem - letting the model directly process the data cannot handle complex sensing requests and letting the model write the data processing program suffers error-prone code generation. We address the problem with 1) a unified data processing framework that makes context-sensing programs simpler and 2) a feedback-guided query optimizer that makes data query more informative. To evaluate the performance of natural language-based context sensing, we create a benchmark that contains 133 context sensing tasks. Extensive evaluation has shown that our approach is able to automatically solve the context-sensing tasks efficiently and precisely. The code is opensourced at https://github.com/MobileLLM/ChainStream.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.04617v2">Evaluation of Code LLMs on Geospatial Code Generation</a></div>
    <div class="paper-meta">
      📅 2024-12-13
      | 💬 7th ACM SIGSPATIAL International Workshop on AI for Geographic Knowledge Discovery (GeoAI'24)
    </div>
    <details class="paper-abstract">
      Software development support tools have been studied for a long time, with recent approaches using Large Language Models (LLMs) for code generation. These models can generate Python code for data science and machine learning applications. LLMs are helpful for software engineers because they increase productivity in daily work. An LLM can also serve as a "mentor" for inexperienced software developers, and be a viable learning support. High-quality code generation with LLMs can also be beneficial in geospatial data science. However, this domain poses different challenges, and code generation LLMs are typically not evaluated on geospatial tasks. Here, we show how we constructed an evaluation benchmark for code generation models, based on a selection of geospatial tasks. We categorised geospatial tasks based on their complexity and required tools. Then, we created a dataset with tasks that test model capabilities in spatial reasoning, spatial data processing, and geospatial tools usage. The dataset consists of specific coding problems that were manually created for high quality. For every problem, we proposed a set of test scenarios that make it possible to automatically check the generated code for correctness. In addition, we tested a selection of existing code generation LLMs for code generation in the geospatial domain. We share our dataset and reproducible evaluation code on a public GitHub repository, arguing that this can serve as an evaluation benchmark for new LLMs in the future. Our dataset will hopefully contribute to the development new models capable of solving geospatial coding tasks with high accuracy. These models will enable the creation of coding assistants tailored for geospatial applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.16283v2">Andes: Defining and Enhancing Quality-of-Experience in LLM-Based Text Streaming Services</a></div>
    <div class="paper-meta">
      📅 2024-12-13
      | 💬 16 pages, 21 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are now at the core of conversational AI services such as real-time translation and chatbots, which provide live user interaction by incrementally streaming text to the user. However, existing LLM serving systems fail to provide good user experience because their optimization metrics are not always aligned with user experience. In this paper, we first introduce and define the notion of Quality-of-Experience (QoE) for text streaming services by considering each user's end-to-end interaction timeline. Based on this, we propose Andes, a QoE-aware LLM serving system that enhances user experience by ensuring that users receive the first token promptly and subsequent tokens at a smooth, digestible pace, even during surge periods. This is enabled by Andes's preemptive request scheduler that dynamically prioritizes requests at the token granularity based on each request's expected QoE gain and GPU resource usage. Our evaluations demonstrate that, compared to state-of-the-art LLM serving systems, Andes improves the average QoE by up to $4.7\times$ given the same GPU resource, or saves up to 61% GPU resources while maintaining the same high QoE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.13748v2">Learn and Unlearn in Multilingual LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-13
    </div>
    <details class="paper-abstract">
      This paper investigates the propagation of harmful information in multilingual large language models (LLMs) and evaluates the efficacy of various unlearning methods. We demonstrate that fake information, regardless of the language it is in, once introduced into these models through training data, can spread across different languages, compromising the integrity and reliability of the generated content. Our findings reveal that standard unlearning techniques, which typically focus on English data, are insufficient in mitigating the spread of harmful content in multilingual contexts and could inadvertently reinforce harmful content across languages. We show that only by addressing harmful responses in both English and the original language of the harmful data can we effectively eliminate generations for all languages. This underscores the critical need for comprehensive unlearning strategies that consider the multilingual nature of modern LLMs to enhance their safety and reliability across diverse linguistic landscapes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06724v2">AutoDCWorkflow: LLM-based Data Cleaning Workflow Auto-Generation and Benchmark</a></div>
    <div class="paper-meta">
      📅 2024-12-13
    </div>
    <details class="paper-abstract">
      We investigate the reasoning capabilities of large language models (LLMs) for automatically generating data-cleaning workflows. To evaluate LLMs' ability to complete data-cleaning tasks, we implemented a pipeline for LLM-based Auto Data Cleaning Workflow (AutoDCWorkflow), prompting LLMs on data cleaning operations to repair three types of data quality issues: duplicates, missing values, and inconsistent data formats. Given a dirty table and a purpose (expressed as a query), this pipeline generates a minimal, clean table sufficient to address the purpose and the data cleaning workflow used to produce the table. The planning process involves three main LLM-driven components: (1) Select Target Columns: Identifies a set of target columns related to the purpose. (2) Inspect Column Quality: Assesses the data quality for each target column and generates a Data Quality Report as operation objectives. (3) Generate Operation & Arguments: Predicts the next operation and arguments based on the data quality report results. Additionally, we propose a data cleaning benchmark to evaluate the capability of LLM agents to automatically generate workflows that address data cleaning purposes of varying difficulty levels. The benchmark comprises the annotated datasets as a collection of purpose, raw table, clean table, data cleaning workflow, and answer set. In our experiments, we evaluated three LLMs that auto-generate purpose-driven data cleaning workflows. The results indicate that LLMs perform well in planning and generating data-cleaning workflows without the need for fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.00416v2">Too Late to Train, Too Early To Use? A Study on Necessity and Viability of Low-Resource Bengali LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-13
    </div>
    <details class="paper-abstract">
      Each new generation of English-oriented Large Language Models (LLMs) exhibits enhanced cross-lingual transfer capabilities and significantly outperforms older LLMs on low-resource languages. This prompts the question: Is there a need for LLMs dedicated to a particular low-resource language? We aim to explore this question for Bengali, a low-to-moderate resource Indo-Aryan language native to the Bengal region of South Asia. We compare the performance of open-weight and closed-source LLMs such as LLaMA-3 and GPT-4 against fine-tuned encoder-decoder models across a diverse set of Bengali downstream tasks, including translation, summarization, paraphrasing, question-answering, and natural language inference. Our findings reveal that while LLMs generally excel in reasoning tasks, their performance in tasks requiring Bengali script generation is inconsistent. Key challenges include inefficient tokenization of Bengali script by existing LLMs, leading to increased computational costs and potential performance degradation. Additionally, we highlight biases in machine-translated datasets commonly used for Bengali NLP tasks. We conclude that there is a significant need for a Bengali-oriented LLM, but the field currently lacks the high-quality pretraining and instruction-tuning datasets necessary to develop a highly effective model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.09025v6">SLEB: Streamlining LLMs through Redundancy Verification and Elimination of Transformer Blocks</a></div>
    <div class="paper-meta">
      📅 2024-12-13
      | 💬 ICML 2024
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have proven to be highly effective across various natural language processing tasks. However, their large number of parameters poses significant challenges for practical deployment. Pruning, a technique aimed at reducing the size and complexity of LLMs, offers a potential solution by removing redundant components from the network. Despite the promise of pruning, existing methods often struggle to achieve substantial end-to-end LLM inference speedup. In this paper, we introduce SLEB, a novel approach designed to streamline LLMs by eliminating redundant transformer blocks. We choose the transformer block as the fundamental unit for pruning, because LLMs exhibit block-level redundancy with high similarity between the outputs of neighboring blocks. This choice allows us to effectively enhance the processing speed of LLMs. Our experimental results demonstrate that SLEB outperforms previous LLM pruning methods in accelerating LLM inference while also maintaining superior perplexity and accuracy, making SLEB as a promising technique for enhancing the efficiency of LLMs. The code is available at: https://github.com/jiwonsong-dev/SLEB.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14743v1">KVDirect: Distributed Disaggregated LLM Inference</a></div>
    <div class="paper-meta">
      📅 2024-12-13
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become the new foundation for many applications, reshaping human society like a storm. Disaggregated inference, which separates prefill and decode stages, is a promising approach to improving hardware utilization and service quality. However, due to inefficient inter-node communication, existing systems restrict disaggregated inference to a single node, limiting resource allocation flexibility and reducing service capacity. This paper introduces KVDirect, which optimizes KV cache transfer to enable a distributed disaggregated LLM inference. KVDirect achieves this through the following contributions. First, we propose a novel tensor-centric communication mechanism that reduces the synchronization overhead in traditional distributed GPU systems. Second, we design a custom communication library to support dynamic GPU resource scheduling and efficient KV cache transfer. Third, we introduce a pull-based KV cache transfer strategy that reduces GPU resource idling and improves latency. Finally, we implement KVDirect as an open-source LLM inference framework. Our evaluation demonstrates that KVDirect reduces per-request latency by 55% compared to the baseline across diverse workloads under the same resource constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.10321v1">AdvPrefix: An Objective for Nuanced LLM Jailbreaks</a></div>
    <div class="paper-meta">
      📅 2024-12-13
    </div>
    <details class="paper-abstract">
      Many jailbreak attacks on large language models (LLMs) rely on a common objective: making the model respond with the prefix "Sure, here is (harmful request)". While straightforward, this objective has two limitations: limited control over model behaviors, often resulting in incomplete or unrealistic responses, and a rigid format that hinders optimization. To address these limitations, we introduce AdvPrefix, a new prefix-forcing objective that enables more nuanced control over model behavior while being easy to optimize. Our objective leverages model-dependent prefixes, automatically selected based on two criteria: high prefilling attack success rates and low negative log-likelihood. It can further simplify optimization by using multiple prefixes for a single user request. AdvPrefix can integrate seamlessly into existing jailbreak attacks to improve their performance for free. For example, simply replacing GCG attack's target prefixes with ours on Llama-3 improves nuanced attack success rates from 14% to 80%, suggesting that current alignment struggles to generalize to unseen prefixes. Our work demonstrates the importance of jailbreak objectives in achieving nuanced jailbreaks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.10281v1">One world, one opinion? The superstar effect in LLM responses</a></div>
    <div class="paper-meta">
      📅 2024-12-13
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are shaping the way information is shared and accessed online, their opinions have the potential to influence a wide audience. This study examines who the LLMs view as the most prominent figures across various fields, using prompts in ten different languages to explore the influence of linguistic diversity. Our findings reveal low diversity in responses, with a small number of figures dominating recognition across languages (also known as the "superstar effect"). These results highlight the risk of narrowing global knowledge representation when LLMs retrieve subjective information.
    </details>
</div>
