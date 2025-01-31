# llm - 2024_07

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3
- [Part 4](papers_4.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.00870v2">Roleplay-doh: Enabling Domain-Experts to Create LLM-simulated Patients via Eliciting and Adhering to Principles</a></div>
    <div class="paper-meta">
      📅 2024-07-14
      | 💬 34 pages, 24 figures, 11 Tables
    </div>
    <details class="paper-abstract">
      Recent works leverage LLMs to roleplay realistic social scenarios, aiding novices in practicing their social skills. However, simulating sensitive interactions, such as in mental health, is challenging. Privacy concerns restrict data access, and collecting expert feedback, although vital, is laborious. To address this, we develop Roleplay-doh, a novel human-LLM collaboration pipeline that elicits qualitative feedback from a domain-expert, which is transformed into a set of principles, or natural language rules, that govern an LLM-prompted roleplay. We apply this pipeline to enable senior mental health supporters to create customized AI patients for simulated practice partners for novice counselors. After uncovering issues in GPT-4 simulations not adhering to expert-defined principles, we also introduce a novel principle-adherence prompting pipeline which shows 30% improvements in response quality and principle following for the downstream task. Via a user study with 25 counseling experts, we demonstrate that the pipeline makes it easy and effective to create AI patients that more faithfully resemble real patients, as judged by creators and third-party counselors. See our project website at https://roleplay-doh.github.io/ for code and data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.10153v1">Look Within, Why LLMs Hallucinate: A Causal Perspective</a></div>
    <div class="paper-meta">
      📅 2024-07-14
      | 💬 15 pages, 7 figures
    </div>
    <details class="paper-abstract">
      The emergence of large language models (LLMs) is a milestone in generative artificial intelligence, achieving significant success in text comprehension and generation tasks. Despite the tremendous success of LLMs in many downstream tasks, they suffer from severe hallucination problems, posing significant challenges to the practical applications of LLMs. Most of the works about LLMs' hallucinations focus on data quality. Self-attention is a core module in transformer-based LLMs, while its potential relationship with LLMs' hallucination has been hardly investigated. To fill this gap, we study this problem from a causal perspective. We propose a method to intervene in LLMs' self-attention layers and maintain their structures and sizes intact. Specifically, we disable different self-attention layers in several popular open-source LLMs and then compare their degrees of hallucination with the original ones. We evaluate the intervened LLMs on hallucination assessment benchmarks and conclude that disabling some specific self-attention layers in the front or tail of the LLMs can alleviate hallucination issues. The study paves a new way for understanding and mitigating LLMs' hallucinations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.10081v1">All Roads Lead to Rome: Unveiling the Trajectory of Recommender Systems Across the LLM Era</a></div>
    <div class="paper-meta">
      📅 2024-07-14
    </div>
    <details class="paper-abstract">
      Recommender systems (RS) are vital for managing information overload and delivering personalized content, responding to users' diverse information needs. The emergence of large language models (LLMs) offers a new horizon for redefining recommender systems with vast general knowledge and reasoning capabilities. Standing across this LLM era, we aim to integrate recommender systems into a broader picture, and pave the way for more comprehensive solutions for future research. Therefore, we first offer a comprehensive overview of the technical progression of recommender systems, particularly focusing on language foundation models and their applications in recommendation. We identify two evolution paths of modern recommender systems -- via list-wise recommendation and conversational recommendation. These two paths finally converge at LLM agents with superior capabilities of long-term memory, reflection, and tool intelligence. Along these two paths, we point out that the information effectiveness of the recommendation is increased, while the user's acquisition cost is decreased. Technical features, research methodologies, and inherent challenges for each milestone along the path are carefully investigated -- from traditional list-wise recommendation to LLM-enhanced recommendation to recommendation with LLM agents. Finally, we highlight several unresolved challenges crucial for the development of future personalization technologies and interfaces and discuss the future prospects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.06245v2">ORAN-Bench-13K: An Open Source Benchmark for Assessing LLMs in Open Radio Access Networks</a></div>
    <div class="paper-meta">
      📅 2024-07-13
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can revolutionize how we deploy and operate Open Radio Access Networks (O-RAN) by enhancing network analytics, anomaly detection, and code generation and significantly increasing the efficiency and reliability of a plethora of O-RAN tasks. In this paper, we present ORAN-Bench-13K, the first comprehensive benchmark designed to evaluate the performance of Large Language Models (LLMs) within the context of O-RAN. Our benchmark consists of 13,952 meticulously curated multiple-choice questions generated from 116 O-RAN specification documents. We leverage a novel three-stage LLM framework, and the questions are categorized into three distinct difficulties to cover a wide spectrum of ORAN-related knowledge. We thoroughly evaluate the performance of several state-of-the-art LLMs, including Gemini, Chat-GPT, and Mistral. Additionally, we propose ORANSight, a Retrieval-Augmented Generation (RAG)-based pipeline that demonstrates superior performance on ORAN-Bench-13K compared to other tested closed-source models. Our findings indicate that current popular LLM models are not proficient in O-RAN, highlighting the need for specialized models. We observed a noticeable performance improvement when incorporating the RAG-based ORANSight pipeline, with a Macro Accuracy of 0.784 and a Weighted Accuracy of 0.776, which was on average 21.55% and 22.59% better than the other tested LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.10020v1">Causality extraction from medical text using Large Language Models (LLMs)</a></div>
    <div class="paper-meta">
      📅 2024-07-13
    </div>
    <details class="paper-abstract">
      This study explores the potential of natural language models, including large language models, to extract causal relations from medical texts, specifically from Clinical Practice Guidelines (CPGs). The outcomes causality extraction from Clinical Practice Guidelines for gestational diabetes are presented, marking a first in the field. We report on a set of experiments using variants of BERT (BioBERT, DistilBERT, and BERT) and using Large Language Models (LLMs), namely GPT-4 and LLAMA2. Our experiments show that BioBERT performed better than other models, including the Large Language Models, with an average F1-score of 0.72. GPT-4 and LLAMA2 results show similar performance but less consistency. We also release the code and an annotated a corpus of causal statements within the Clinical Practice Guidelines for gestational diabetes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.09855v1">Building pre-train LLM Dataset for the INDIC Languages: a case study on Hindi</a></div>
    <div class="paper-meta">
      📅 2024-07-13
      | 💬 Accepted as a book chapter in the book Title "APPLIED SPEECH AND TEXT PROCESSING FOR LOW RESOURCE LANGUAGES"
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrated transformative capabilities in many applications that require automatically generating responses based on human instruction. However, the major challenge for building LLMs, particularly in Indic languages, is the availability of high-quality data for building foundation LLMs. In this paper, we are proposing a large pre-train dataset in Hindi useful for the Indic language Hindi. We have collected the data span across several domains including major dialects in Hindi. The dataset contains 1.28 billion Hindi tokens. We have explained our pipeline including data collection, pre-processing, and availability for LLM pre-training. The proposed approach can be easily extended to other Indic and low-resource languages and will be available freely for LLM pre-training and LLM research purposes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.09811v1">CellAgent: An LLM-driven Multi-Agent Framework for Automated Single-cell Data Analysis</a></div>
    <div class="paper-meta">
      📅 2024-07-13
    </div>
    <details class="paper-abstract">
      Single-cell RNA sequencing (scRNA-seq) data analysis is crucial for biological research, as it enables the precise characterization of cellular heterogeneity. However, manual manipulation of various tools to achieve desired outcomes can be labor-intensive for researchers. To address this, we introduce CellAgent (http://cell.agent4science.cn/), an LLM-driven multi-agent framework, specifically designed for the automatic processing and execution of scRNA-seq data analysis tasks, providing high-quality results with no human intervention. Firstly, to adapt general LLMs to the biological field, CellAgent constructs LLM-driven biological expert roles - planner, executor, and evaluator - each with specific responsibilities. Then, CellAgent introduces a hierarchical decision-making mechanism to coordinate these biological experts, effectively driving the planning and step-by-step execution of complex data analysis tasks. Furthermore, we propose a self-iterative optimization mechanism, enabling CellAgent to autonomously evaluate and optimize solutions, thereby guaranteeing output quality. We evaluate CellAgent on a comprehensive benchmark dataset encompassing dozens of tissues and hundreds of distinct cell types. Evaluation results consistently show that CellAgent effectively identifies the most suitable tools and hyperparameters for single-cell analysis tasks, achieving optimal performance. This automated framework dramatically reduces the workload for science data analyses, bringing us into the "Agent for Science" era.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.15042v2">LLM2LLM: Boosting LLMs with Novel Iterative Data Enhancement</a></div>
    <div class="paper-meta">
      📅 2024-07-13
      | 💬 ACL 2024
    </div>
    <details class="paper-abstract">
      Pretrained large language models (LLMs) are currently state-of-the-art for solving the vast majority of natural language processing tasks. While many real-world applications still require fine-tuning to reach satisfactory levels of performance, many of them are in the low-data regime, making fine-tuning challenging. To address this, we propose LLM2LLM, a targeted and iterative data augmentation strategy that uses a teacher LLM to enhance a small seed dataset by augmenting additional data that can be used for fine-tuning on a specific task. LLM2LLM (1) fine-tunes a baseline student LLM on the initial seed data, (2) evaluates and extracts data points that the model gets wrong, and (3) uses a teacher LLM to generate synthetic data based on these incorrect data points, which are then added back into the training data. This approach amplifies the signal from incorrectly predicted data points by the LLM during training and reintegrates them into the dataset to focus on more challenging examples for the LLM. Our results show that LLM2LLM significantly enhances the performance of LLMs in the low-data regime, outperforming both traditional fine-tuning and other data augmentation baselines. LLM2LLM reduces the dependence on labor-intensive data curation and paves the way for more scalable and performant LLM solutions, allowing us to tackle data-constrained domains and tasks. We achieve improvements up to 24.2% on the GSM8K dataset, 32.6% on CaseHOLD, 32.0% on SNIPS, 52.6% on TREC and 39.8% on SST-2 over regular fine-tuning in the low-data regime using a Llama-2-7B student model. Our code is available at https://github.com/SqueezeAILab/LLM2LLM .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.12866v1">Beyond KV Caching: Shared Attention for Efficient LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-13
    </div>
    <details class="paper-abstract">
      The efficiency of large language models (LLMs) remains a critical challenge, particularly in contexts where computational resources are limited. Traditional attention mechanisms in these models, while powerful, require significant computational and memory resources due to the necessity of recalculating and storing attention weights across different layers. This paper introduces a novel Shared Attention (SA) mechanism, designed to enhance the efficiency of LLMs by directly sharing computed attention weights across multiple layers. Unlike previous methods that focus on sharing intermediate Key-Value (KV) caches, our approach utilizes the isotropic tendencies of attention distributions observed in advanced LLMs post-pretraining to reduce both the computational flops and the size of the KV cache required during inference. We empirically demonstrate that implementing SA across various LLMs results in minimal accuracy loss on standard benchmarks. Our findings suggest that SA not only conserves computational resources but also maintains robust model performance, thereby facilitating the deployment of more efficient LLMs in resource-constrained environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.09756v1">LLM-Collaboration on Automatic Science Journalism for the General Audience</a></div>
    <div class="paper-meta">
      📅 2024-07-13
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Science journalism reports current scientific discoveries to non-specialists, aiming to enable public comprehension of the state of the art. However, this task can be challenging as the audience often lacks specific knowledge about the presented research. To address this challenge, we propose a framework that integrates three LLMs mimicking the real-world writing-reading-feedback-revision workflow, with one LLM acting as the journalist, a smaller LLM as the general public reader, and the third LLM as an editor. The journalist's writing is iteratively refined by feedback from the reader and suggestions from the editor. Our experiments demonstrate that by leveraging the collaboration of two 7B and one 1.8B open-source LLMs, we can generate articles that are more accessible than those generated by existing methods, including advanced models such as GPT-4.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.09726v1">On Mitigating Code LLM Hallucinations with API Documentation</a></div>
    <div class="paper-meta">
      📅 2024-07-13
    </div>
    <details class="paper-abstract">
      In this study, we address the issue of API hallucinations in various software engineering contexts. We introduce CloudAPIBench, a new benchmark designed to measure API hallucination occurrences. CloudAPIBench also provides annotations for frequencies of API occurrences in the public domain, allowing us to study API hallucinations at various frequency levels. Our findings reveal that Code LLMs struggle with low frequency APIs: for e.g., GPT-4o achieves only 38.58% valid low frequency API invocations. We demonstrate that Documentation Augmented Generation (DAG) significantly improves performance for low frequency APIs (increase to 47.94% with DAG) but negatively impacts high frequency APIs when using sub-optimal retrievers (a 39.02% absolute drop). To mitigate this, we propose to intelligently trigger DAG where we check against an API index or leverage Code LLMs' confidence scores to retrieve only when needed. We demonstrate that our proposed methods enhance the balance between low and high frequency API performance, resulting in more reliable API invocations (8.20% absolute improvement on CloudAPIBench for GPT-4o).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.11072v1">MaPPing Your Model: Assessing the Impact of Adversarial Attacks on LLM-based Programming Assistants</a></div>
    <div class="paper-meta">
      📅 2024-07-12
      | 💬 6 pages, 5 figures, Proceedings of the ICML 2024 Workshop on Trustworthy Multimodal Foundation Models and AI Agents
    </div>
    <details class="paper-abstract">
      LLM-based programming assistants offer the promise of programming faster but with the risk of introducing more security vulnerabilities. Prior work has studied how LLMs could be maliciously fine-tuned to suggest vulnerabilities more often. With the rise of agentic LLMs, which may use results from an untrusted third party, there is a growing risk of attacks on the model's prompt. We introduce the Malicious Programming Prompt (MaPP) attack, in which an attacker adds a small amount of text to a prompt for a programming task (under 500 bytes). We show that our prompt strategy can cause an LLM to add vulnerabilities while continuing to write otherwise correct code. We evaluate three prompts on seven common LLMs, from basic to state-of-the-art commercial models. Using the HumanEval benchmark, we find that our prompts are broadly effective, with no customization required for different LLMs. Furthermore, the LLMs that are best at HumanEval are also best at following our malicious instructions, suggesting that simply scaling language models will not prevent MaPP attacks. Using a dataset of eight CWEs in 16 scenarios, we find that MaPP attacks are also effective at implementing specific and targeted vulnerabilities across a range of models. Our work highlights the need to secure LLM prompts against manipulation as well as rigorously auditing code generated with the help of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.09704v1">What an Elegant Bridge: Multilingual LLMs are Biased Similarly in Different Languages</a></div>
    <div class="paper-meta">
      📅 2024-07-12
    </div>
    <details class="paper-abstract">
      This paper investigates biases of Large Language Models (LLMs) through the lens of grammatical gender. Drawing inspiration from seminal works in psycholinguistics, particularly the study of gender's influence on language perception, we leverage multilingual LLMs to revisit and expand upon the foundational experiments of Boroditsky (2003). Employing LLMs as a novel method for examining psycholinguistic biases related to grammatical gender, we prompt a model to describe nouns with adjectives in various languages, focusing specifically on languages with grammatical gender. In particular, we look at adjective co-occurrences across gender and languages, and train a binary classifier to predict grammatical gender given adjectives an LLM uses to describe a noun. Surprisingly, we find that a simple classifier can not only predict noun gender above chance but also exhibit cross-language transferability. We show that while LLMs may describe words differently in different languages, they are biased similarly.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.09652v1">How Chinese are Chinese Language Models? The Puzzling Lack of Language Policy in China's LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-12
      | 💬 Wen-Yi and Jo contributed equally to this work
    </div>
    <details class="paper-abstract">
      Contemporary language models are increasingly multilingual, but Chinese LLM developers must navigate complex political and business considerations of language diversity. Language policy in China aims at influencing the public discourse and governing a multi-ethnic society, and has gradually transitioned from a pluralist to a more assimilationist approach since 1949. We explore the impact of these influences on current language technology. We evaluate six open-source multilingual LLMs pre-trained by Chinese companies on 18 languages, spanning a wide range of Chinese, Asian, and Anglo-European languages. Our experiments show Chinese LLMs performance on diverse languages is indistinguishable from international LLMs. Similarly, the models' technical reports also show lack of consideration for pretraining data language coverage except for English and Mandarin Chinese. Examining Chinese AI policy, model experiments, and technical reports, we find no sign of any consistent policy, either for or against, language diversity in China's LLM development. This leaves a puzzling fact that while China regulates both the languages people use daily as well as language model development, they do not seem to have any policy on the languages in language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.18120v2">ArzEn-LLM: Code-Switched Egyptian Arabic-English Translation and Speech Recognition Using LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-12
      | 💬 9 pages, 4 figures, 5 tables, 6th International Conference on AI in Computational Linguistics
    </div>
    <details class="paper-abstract">
      Motivated by the widespread increase in the phenomenon of code-switching between Egyptian Arabic and English in recent times, this paper explores the intricacies of machine translation (MT) and automatic speech recognition (ASR) systems, focusing on translating code-switched Egyptian Arabic-English to either English or Egyptian Arabic. Our goal is to present the methodologies employed in developing these systems, utilizing large language models such as LLama and Gemma. In the field of ASR, we explore the utilization of the Whisper model for code-switched Egyptian Arabic recognition, detailing our experimental procedures including data preprocessing and training techniques. Through the implementation of a consecutive speech-to-text translation system that integrates ASR with MT, we aim to overcome challenges posed by limited resources and the unique characteristics of the Egyptian Arabic dialect. Evaluation against established metrics showcases promising results, with our methodologies yielding a significant improvement of $56\%$ in English translation over the state-of-the-art and $9.3\%$ in Arabic translation. Since code-switching is deeply inherent in spoken languages, it is crucial that ASR systems can effectively handle this phenomenon. This capability is crucial for enabling seamless interaction in various domains, including business negotiations, cultural exchanges, and academic discourse. Our models and code are available as open-source resources. Code: \url{http://github.com/ahmedheakl/arazn-llm}}, Models: \url{http://huggingface.co/collections/ahmedheakl/arazn-llm-662ceaf12777656607b9524e}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.15091v2">VideoDirectorGPT: Consistent Multi-scene Video Generation via LLM-Guided Planning</a></div>
    <div class="paper-meta">
      📅 2024-07-12
      | 💬 COLM 2024; Project page: https://videodirectorgpt.github.io
    </div>
    <details class="paper-abstract">
      Recent text-to-video (T2V) generation methods have seen significant advancements. However, the majority of these works focus on producing short video clips of a single event (i.e., single-scene videos). Meanwhile, recent large language models (LLMs) have demonstrated their capability in generating layouts and programs to control downstream visual modules. This prompts an important question: can we leverage the knowledge embedded in these LLMs for temporally consistent long video generation? In this paper, we propose VideoDirectorGPT, a novel framework for consistent multi-scene video generation that uses the knowledge of LLMs for video content planning and grounded video generation. Specifically, given a single text prompt, we first ask our video planner LLM (GPT-4) to expand it into a 'video plan', which includes the scene descriptions, the entities with their respective layouts, the background for each scene, and consistency groupings of the entities. Next, guided by this video plan, our video generator, named Layout2Vid, has explicit control over spatial layouts and can maintain temporal consistency of entities across multiple scenes, while being trained only with image-level annotations. Our experiments demonstrate that our proposed VideoDirectorGPT framework substantially improves layout and movement control in both single- and multi-scene video generation and can generate multi-scene videos with consistency, while achieving competitive performance with SOTAs in open-domain single-scene T2V generation. Detailed ablation studies, including dynamic adjustment of layout control strength with an LLM and video generation with user-provided images, confirm the effectiveness of each component of our framework and its future potential.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.12014v2">EnvGen: Generating and Adapting Environments via LLMs for Training Embodied Agents</a></div>
    <div class="paper-meta">
      📅 2024-07-12
      | 💬 COLM 2024; First two authors contributed equally; Project website: https://envgen-llm.github.io/
    </div>
    <details class="paper-abstract">
      Recent SOTA approaches for embodied learning via interaction directly employ large language models (LLMs) as agents to determine the next steps in an environment. Due to their world knowledge and reasoning capabilities, LLM agents achieve stronger performance than previous smaller agents based on reinforcement learning (RL); however, frequently calling LLMs is slow and expensive. Instead of directly employing LLMs as agents, can we use LLMs' reasoning capabilities to adaptively create training environments to help smaller RL agents learn useful skills that they are weak at? We propose EnvGen, a novel framework to address this question. We first prompt an LLM to generate training environments by giving it the task description and simulator objectives that the agents should learn and then asking it to generate a set of environment configurations (e.g., different terrains, items initially given to agents, etc.). Next, we train a small RL agent in a mixture of the original and LLM-generated environments. Then, we enable the LLM to continuously adapt the generated environments to progressively improve the skills that the agent is weak at, by providing feedback to the LLM in the form of the agent's performance. We demonstrate the usefulness of EnvGen with comprehensive experiments in Crafter and Heist environments. We find that a small RL agent trained with EnvGen can outperform SOTA methods, including a GPT-4 agent, and learns long-horizon tasks significantly faster. We also show that using an LLM to adapt environments dynamically outperforms curriculum learning approaches and how the environments are adapted to help improve RL agents' weaker skills over time. Additionally, EnvGen is substantially more efficient as it only uses a small number of LLM calls (e.g., 4 in total), whereas LLM agents require thousands of calls. Lastly, we present detailed ablation studies for EnvGen design choices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.09429v1">Open (Clinical) LLMs are Sensitive to Instruction Phrasings</a></div>
    <div class="paper-meta">
      📅 2024-07-12
      | 💬 To appear at BioNLP, ACL 2024
    </div>
    <details class="paper-abstract">
      Instruction-tuned Large Language Models (LLMs) can perform a wide range of tasks given natural language instructions to do so, but they are sensitive to how such instructions are phrased. This issue is especially concerning in healthcare, as clinicians are unlikely to be experienced prompt engineers and the potential consequences of inaccurate outputs are heightened in this domain. This raises a practical question: How robust are instruction-tuned LLMs to natural variations in the instructions provided for clinical NLP tasks? We collect prompts from medical doctors across a range of tasks and quantify the sensitivity of seven LLMs -- some general, others specialized -- to natural (i.e., non-adversarial) instruction phrasings. We find that performance varies substantially across all models, and that -- perhaps surprisingly -- domain-specific models explicitly trained on clinical data are especially brittle, compared to their general domain counterparts. Further, arbitrary phrasing differences can affect fairness, e.g., valid but distinct instructions for mortality prediction yield a range both in overall performance, and in terms of differences between demographic groups.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.04622v2">On scalable oversight with weak LLMs judging strong LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-12
      | 💬 15 pages (53 including appendices). V2: minor correction to Figure 3; add Figure A.9 comparing open vs assigned consultancy; add a reference
    </div>
    <details class="paper-abstract">
      Scalable oversight protocols aim to enable humans to accurately supervise superhuman AI. In this paper we study debate, where two AI's compete to convince a judge; consultancy, where a single AI tries to convince a judge that asks questions; and compare to a baseline of direct question-answering, where the judge just answers outright without the AI. We use large language models (LLMs) as both AI agents and as stand-ins for human judges, taking the judge models to be weaker than agent models. We benchmark on a diverse range of asymmetries between judges and agents, extending previous work on a single extractive QA task with information asymmetry, to also include mathematics, coding, logic and multimodal reasoning asymmetries. We find that debate outperforms consultancy across all tasks when the consultant is randomly assigned to argue for the correct/incorrect answer. Comparing debate to direct question answering, the results depend on the type of task: in extractive QA tasks with information asymmetry debate outperforms direct question answering, but in other tasks without information asymmetry the results are mixed. Previous work assigned debaters/consultants an answer to argue for. When we allow them to instead choose which answer to argue for, we find judges are less frequently convinced by the wrong answer in debate than in consultancy. Further, we find that stronger debater models increase judge accuracy, though more modestly than in previous studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.10632v5">Beyond static AI evaluations: advancing human interaction evaluations for LLM harms and risks</a></div>
    <div class="paper-meta">
      📅 2024-07-12
      | 💬 revised figure
    </div>
    <details class="paper-abstract">
      Model evaluations are central to understanding the safety, risks, and societal impacts of AI systems. While most real-world AI applications involve human-AI interaction, most current evaluations (e.g., common benchmarks) of AI models do not. Instead, they incorporate human factors in limited ways, assessing the safety of models in isolation, thereby falling short of capturing the complexity of human-model interactions. In this paper, we discuss and operationalize a definition of an emerging category of evaluations -- "human interaction evaluations" (HIEs) -- which focus on the assessment of human-model interactions or the process and the outcomes of humans using models. First, we argue that HIEs can be used to increase the validity of safety evaluations, assess direct human impact and interaction-specific harms, and guide future assessments of models' societal impact. Second, we propose a safety-focused HIE design framework -- containing a human-LLM interaction taxonomy -- with three stages: (1) identifying the risk or harm area, (2) characterizing the use context, and (3) choosing the evaluation parameters. Third, we apply our framework to two potential evaluations for overreliance and persuasion risks. Finally, we conclude with tangible recommendations for addressing concerns over costs, replicability, and unrepresentativeness of HIEs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.01394v2">Gloss2Text: Sign Language Gloss translation using LLMs and Semantically Aware Label Smoothing</a></div>
    <div class="paper-meta">
      📅 2024-07-12
    </div>
    <details class="paper-abstract">
      Sign language translation from video to spoken text presents unique challenges owing to the distinct grammar, expression nuances, and high variation of visual appearance across different speakers and contexts. The intermediate gloss annotations of videos aim to guide the translation process. In our work, we focus on {\em Gloss2Text} translation stage and propose several advances by leveraging pre-trained large language models (LLMs), data augmentation, and novel label-smoothing loss function exploiting gloss translation ambiguities improving significantly the performance of state-of-the-art approaches. Through extensive experiments and ablation studies on the PHOENIX Weather 2014T dataset, our approach surpasses state-of-the-art performance in {\em Gloss2Text} translation, indicating its efficacy in addressing sign language translation and suggesting promising avenues for future research and development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.09290v1">Structuring Authenticity Assessments on Historical Documents using LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-12
    </div>
    <details class="paper-abstract">
      Given the wide use of forgery throughout history, scholars have and are continuously engaged in assessing the authenticity of historical documents. However, online catalogues merely offer descriptive metadata for these documents, relegating discussions about their authenticity to free-text formats, making it difficult to study these assessments at scale. This study explores the generation of structured data about documents' authenticity assessment from natural language texts. Our pipeline exploits Large Language Models (LLMs) to select, extract and classify relevant claims about the topic without the need for training, and Semantic Web technologies to structure and type-validate the LLM's results. The final output is a catalogue of documents whose authenticity has been debated, along with scholars' opinions on their authenticity. This process can serve as a valuable resource for integration into catalogues, allowing room for more intricate queries and analyses on the evolution of these debates over centuries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.19570v2">Synthetic Cancer -- Augmenting Worms with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-12
      | 💬 Won first place at the Swiss AI Safety Prize. Some technical details omitted, contact authors for more information
    </div>
    <details class="paper-abstract">
      With increasingly sophisticated large language models (LLMs), the potential for abuse rises drastically. As a submission to the Swiss AI Safety Prize, we present a novel type of metamorphic malware leveraging LLMs for two key processes. First, LLMs are used for automatic code rewriting to evade signature-based detection by antimalware programs. The malware then spreads its copies via email by utilizing an LLM to socially engineer email replies to encourage recipients to execute the attached malware. Our submission includes a functional minimal prototype, highlighting the risks that LLMs pose for cybersecurity and underscoring the need for further research into intelligent malware.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.06649v2">ProLLM: Protein Chain-of-Thoughts Enhanced LLM for Protein-Protein Interaction Prediction</a></div>
    <div class="paper-meta">
      📅 2024-07-12
      | 💬 Accepted by COLM 2024
    </div>
    <details class="paper-abstract">
      The prediction of protein-protein interactions (PPIs) is crucial for understanding biological functions and diseases. Previous machine learning approaches to PPI prediction mainly focus on direct physical interactions, ignoring the broader context of nonphysical connections through intermediate proteins, thus limiting their effectiveness. The emergence of Large Language Models (LLMs) provides a new opportunity for addressing this complex biological challenge. By transforming structured data into natural language prompts, we can map the relationships between proteins into texts. This approach allows LLMs to identify indirect connections between proteins, tracing the path from upstream to downstream. Therefore, we propose a novel framework ProLLM that employs an LLM tailored for PPI for the first time. Specifically, we propose Protein Chain of Thought (ProCoT), which replicates the biological mechanism of signaling pathways as natural language prompts. ProCoT considers a signaling pathway as a protein reasoning process, which starts from upstream proteins and passes through several intermediate proteins to transmit biological signals to downstream proteins. Thus, we can use ProCoT to predict the interaction between upstream proteins and downstream proteins. The training of ProLLM employs the ProCoT format, which enhances the model's understanding of complex biological problems. In addition to ProCoT, this paper also contributes to the exploration of embedding replacement of protein sites in natural language prompts, and instruction fine-tuning in protein knowledge datasets. We demonstrate the efficacy of ProLLM through rigorous validation against benchmark datasets, showing significant improvement over existing methods in terms of prediction accuracy and generalizability. The code is available at: https://github.com/MingyuJ666/ProLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.09152v1">The Two Sides of the Coin: Hallucination Generation and Detection with LLMs as Evaluators for LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-12
      | 💬 Paper accepted at ELOQUENT@CLEF'24
    </div>
    <details class="paper-abstract">
      Hallucination detection in Large Language Models (LLMs) is crucial for ensuring their reliability. This work presents our participation in the CLEF ELOQUENT HalluciGen shared task, where the goal is to develop evaluators for both generating and detecting hallucinated content. We explored the capabilities of four LLMs: Llama 3, Gemma, GPT-3.5 Turbo, and GPT-4, for this purpose. We also employed ensemble majority voting to incorporate all four models for the detection task. The results provide valuable insights into the strengths and weaknesses of these LLMs in handling hallucination generation and detection tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.09121v1">Refuse Whenever You Feel Unsafe: Improving Safety in LLMs via Decoupled Refusal Training</a></div>
    <div class="paper-meta">
      📅 2024-07-12
    </div>
    <details class="paper-abstract">
      This study addresses a critical gap in safety tuning practices for Large Language Models (LLMs) by identifying and tackling a refusal position bias within safety tuning data, which compromises the models' ability to appropriately refuse generating unsafe content. We introduce a novel approach, Decoupled Refusal Training (DeRTa), designed to empower LLMs to refuse compliance to harmful prompts at any response position, significantly enhancing their safety capabilities. DeRTa incorporates two novel components: (1) Maximum Likelihood Estimation (MLE) with Harmful Response Prefix, which trains models to recognize and avoid unsafe content by appending a segment of harmful response to the beginning of a safe response, and (2) Reinforced Transition Optimization (RTO), which equips models with the ability to transition from potential harm to safety refusal consistently throughout the harmful response sequence. Our empirical evaluation, conducted using LLaMA3 and Mistral model families across six attack scenarios, demonstrates that our method not only improves model safety without compromising performance but also surpasses well-known models such as GPT-4 in defending against attacks. Importantly, our approach successfully defends recent advanced attack methods (e.g., CodeAttack) that have jailbroken GPT-4 and LLaMA3-70B-Instruct. Our code and data can be found at https://github.com/RobustNLP/DeRTa.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.09044v1">Sensorimotor Attention and Language-based Regressions in Shared Latent Variables for Integrating Robot Motion Learning and LLM</a></div>
    <div class="paper-meta">
      📅 2024-07-12
      | 💬 7 pages, 8 figures, accepted at IROS 2024
    </div>
    <details class="paper-abstract">
      In recent years, studies have been actively conducted on combining large language models (LLM) and robotics; however, most have not considered end-to-end feedback in the robot-motion generation phase. The prediction of deep neural networks must contain errors, it is required to update the trained model to correspond to the real environment to generate robot motion adaptively. This study proposes an integration method that connects the robot-motion learning model and LLM using shared latent variables. When generating robot motion, the proposed method updates shared parameters based on prediction errors from both sensorimotor attention points and task language instructions given to the robot. This allows the model to search for latent parameters appropriate for the robot task efficiently. Through simulator experiments on multiple robot tasks, we demonstrated the effectiveness of our proposed method from two perspectives: position generalization and language instruction generalization abilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.08273v2">RB-SQL: A Retrieval-based LLM Framework for Text-to-SQL</a></div>
    <div class="paper-meta">
      📅 2024-07-12
      | 💬 Further improvement and modification are needed.
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) with in-context learning have significantly improved the performance of text-to-SQL task. Previous works generally focus on using exclusive SQL generation prompt to improve the LLMs' reasoning ability. However, they are mostly hard to handle large databases with numerous tables and columns, and usually ignore the significance of pre-processing database and extracting valuable information for more efficient prompt engineering. Based on above analysis, we propose RB-SQL, a novel retrieval-based LLM framework for in-context prompt engineering, which consists of three modules that retrieve concise tables and columns as schema, and targeted examples for in-context learning. Experiment results demonstrate that our model achieves better performance than several competitive baselines on public datasets BIRD and Spider.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.08995v1">Self-Prompt Tuning: Enable Autonomous Role-Playing in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-12
    </div>
    <details class="paper-abstract">
      Recent advancements in LLMs have showcased their remarkable role-playing capabilities, able to accurately simulate the dialogue styles and cognitive processes of various roles based on different instructions and contexts. Studies indicate that assigning LLMs the roles of experts, a strategy known as role-play prompting, can enhance their performance in the corresponding domains. However, the prompt needs to be manually designed for the given problem, requiring certain expertise and iterative modifications. To this end, we propose self-prompt tuning, making LLMs themselves generate role-play prompts through fine-tuning. Leveraging the LIMA dataset as our foundational corpus, we employ GPT-4 to annotate role-play prompts for each data points, resulting in the creation of the LIMA-Role dataset. We then fine-tune LLMs like Llama-2-7B and Mistral-7B on LIMA-Role. Consequently, the self-prompt tuned LLMs can automatically generate expert role prompts for any given question. We extensively evaluate self-prompt tuned LLMs on widely used NLP benchmarks and open-ended question test. Our empirical results illustrate that self-prompt tuned LLMs outperform standard instruction tuned baselines across most datasets. This highlights the great potential of utilizing fine-tuning to enable LLMs to self-prompt, thereby automating complex prompting strategies. We release the dataset, models, and code at this \href{https://anonymous.4open.science/r/Self-Prompt-Tuning-739E/}{url}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.08989v1">Robustness of LLMs to Perturbations in Text</a></div>
    <div class="paper-meta">
      📅 2024-07-12
      | 💬 8 pages, 1 figure, 6 tables, updated with results also from GPT-4, LLaMa-3
    </div>
    <details class="paper-abstract">
      Having a clean dataset has been the foundational assumption of most natural language processing (NLP) systems. However, properly written text is rarely found in real-world scenarios and hence, oftentimes invalidates the aforementioned foundational assumption. Recently, Large language models (LLMs) have shown impressive performance, but can they handle the inevitable noise in real-world data? This work tackles this critical question by investigating LLMs' resilience against morphological variations in text. To that end, we artificially introduce varying levels of noise into a diverse set of datasets and systematically evaluate LLMs' robustness against the corrupt variations of the original text. Our findings show that contrary to popular beliefs, generative LLMs are quiet robust to noisy perturbations in text. This is a departure from pre-trained models like BERT or RoBERTa whose performance has been shown to be sensitive to deteriorating noisy text. Additionally, we test LLMs' resilience on multiple real-world benchmarks that closely mimic commonly found errors in the wild. With minimal prompting, LLMs achieve a new state-of-the-art on the benchmark tasks of Grammar Error Correction (GEC) and Lexical Semantic Change (LSC). To empower future research, we also release a dataset annotated by humans stating their preference for LLM vs. human-corrected outputs along with the code to reproduce our results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.08983v1">Towards More Trustworthy and Interpretable LLMs for Code through Syntax-Grounded Explanations</a></div>
    <div class="paper-meta">
      📅 2024-07-12
      | 💬 Under Review to appear in ACM Transactions on Software Engineering and Methodology (TOSEM)
    </div>
    <details class="paper-abstract">
      Trustworthiness and interpretability are inextricably linked concepts for LLMs. The more interpretable an LLM is, the more trustworthy it becomes. However, current techniques for interpreting LLMs when applied to code-related tasks largely focus on accuracy measurements, measures of how models react to change, or individual task performance instead of the fine-grained explanations needed at prediction time for greater interpretability, and hence trust. To improve upon this status quo, this paper introduces ASTrust, an interpretability method for LLMs of code that generates explanations grounded in the relationship between model confidence and syntactic structures of programming languages. ASTrust explains generated code in the context of syntax categories based on Abstract Syntax Trees and aids practitioners in understanding model predictions at both local (individual code snippets) and global (larger datasets of code) levels. By distributing and assigning model confidence scores to well-known syntactic structures that exist within ASTs, our approach moves beyond prior techniques that perform token-level confidence mapping by offering a view of model confidence that directly aligns with programming language concepts with which developers are familiar. To put ASTrust into practice, we developed an automated visualization that illustrates the aggregated model confidence scores superimposed on sequence, heat-map, and graph-based visuals of syntactic structures from ASTs. We examine both the practical benefit that ASTrust can provide through a data science study on 12 popular LLMs on a curated set of GitHub repos and the usefulness of ASTrust through a human study.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.07387v3">BISCUIT: Scaffolding LLM-Generated Code with Ephemeral UIs in Computational Notebooks</a></div>
    <div class="paper-meta">
      📅 2024-07-12
    </div>
    <details class="paper-abstract">
      Programmers frequently engage with machine learning tutorials in computational notebooks and have been adopting code generation technologies based on large language models (LLMs). However, they encounter difficulties in understanding and working with code produced by LLMs. To mitigate these challenges, we introduce a novel workflow into computational notebooks that augments LLM-based code generation with an additional ephemeral UI step, offering users UI scaffolds as an intermediate stage between user prompts and code generation. We present this workflow in BISCUIT, an extension for JupyterLab that provides users with ephemeral UIs generated by LLMs based on the context of their code and intentions, scaffolding users to understand, guide, and explore with LLM-generated code. Through a user study where 10 novices used BISCUIT for machine learning tutorials, we found that BISCUIT offers users representations of code to aid their understanding, reduces the complexity of prompt engineering, and creates a playground for users to explore different variables and iterate on their ideas.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.08952v1">Detect, Investigate, Judge and Determine: A Novel LLM-based Framework for Few-shot Fake News Detection</a></div>
    <div class="paper-meta">
      📅 2024-07-12
    </div>
    <details class="paper-abstract">
      Few-Shot Fake News Detection (FS-FND) aims to distinguish inaccurate news from real ones in extremely low-resource scenarios. This task has garnered increased attention due to the widespread dissemination and harmful impact of fake news on social media. Large Language Models (LLMs) have demonstrated competitive performance with the help of their rich prior knowledge and excellent in-context learning abilities. However, existing methods face significant limitations, such as the Understanding Ambiguity and Information Scarcity, which significantly undermine the potential of LLMs. To address these shortcomings, we propose a Dual-perspective Augmented Fake News Detection (DAFND) model, designed to enhance LLMs from both inside and outside perspectives. Specifically, DAFND first identifies the keywords of each news article through a Detection Module. Subsequently, DAFND creatively designs an Investigation Module to retrieve inside and outside valuable information concerning to the current news, followed by another Judge Module to derive its respective two prediction results. Finally, a Determination Module further integrates these two predictions and derives the final result. Extensive experiments on two publicly available datasets show the efficacy of our proposed method, particularly in low-resource settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.08931v1">Global-Local Collaborative Inference with LLM for Lidar-Based Open-Vocabulary Detection</a></div>
    <div class="paper-meta">
      📅 2024-07-12
      | 💬 accepted by ECCV 2024
    </div>
    <details class="paper-abstract">
      Open-Vocabulary Detection (OVD) is the task of detecting all interesting objects in a given scene without predefined object classes. Extensive work has been done to deal with the OVD for 2D RGB images, but the exploration of 3D OVD is still limited. Intuitively, lidar point clouds provide 3D information, both object level and scene level, to generate trustful detection results. However, previous lidar-based OVD methods only focus on the usage of object-level features, ignoring the essence of scene-level information. In this paper, we propose a Global-Local Collaborative Scheme (GLIS) for the lidar-based OVD task, which contains a local branch to generate object-level detection result and a global branch to obtain scene-level global feature. With the global-local information, a Large Language Model (LLM) is applied for chain-of-thought inference, and the detection result can be refined accordingly. We further propose Reflected Pseudo Labels Generation (RPLG) to generate high-quality pseudo labels for supervision and Background-Aware Object Localization (BAOL) to select precise object proposals. Extensive experiments on ScanNetV2 and SUN RGB-D demonstrate the superiority of our methods. Code is released at https://github.com/GradiusTwinbee/GLIS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.08924v1">Disassembling Obfuscated Executables with LLM</a></div>
    <div class="paper-meta">
      📅 2024-07-12
    </div>
    <details class="paper-abstract">
      Disassembly is a challenging task, particularly for obfuscated executables containing junk bytes, which is designed to induce disassembly errors. Existing solutions rely on heuristics or leverage machine learning techniques, but only achieve limited successes. Fundamentally, such obfuscation cannot be defeated without in-depth understanding of the binary executable's semantics, which is made possible by the emergence of large language models (LLMs). In this paper, we present DisasLLM, a novel LLM-driven dissembler to overcome the challenge in analyzing obfuscated executables. DisasLLM consists of two components: an LLM-based classifier that determines whether an instruction in an assembly code snippet is correctly decoded, and a disassembly strategy that leverages this model to disassemble obfuscated executables end-to-end. We evaluated DisasLLM on a set of heavily obfuscated executables, which is shown to significantly outperform other state-of-the-art disassembly solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.09577v1">Flash normalization: fast RMSNorm for LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-12
      | 💬 7 pages, 8 figures
    </div>
    <details class="paper-abstract">
      RMSNorm is used by many LLMs such as Llama, Mistral, and OpenELM. This paper details FlashNorm, which is an exact but faster implementation of RMSNorm followed by linear layers. See https://huggingface.co/open-machine/FlashNorm for code and more transformer tricks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.08770v1">Model Surgery: Modulating LLM's Behavior Via Simple Parameter Editing</a></div>
    <div class="paper-meta">
      📅 2024-07-11
      | 💬 23 pages, 14 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated great potential as generalist assistants, showcasing powerful task understanding and problem-solving capabilities. To deploy LLMs as AI assistants, it is crucial that these models exhibit desirable behavioral traits, such as non-toxicity and resilience against jailbreak attempts. Current methods for detoxification or preventing jailbreaking usually involve Supervised Fine-Tuning (SFT) or Reinforcement Learning from Human Feedback (RLHF), which requires finetuning billions of parameters through gradient descent with substantial computation cost. Furthermore, models modified through SFT and RLHF may deviate from the pretrained models, potentially leading to a degradation in foundational LLM capabilities. In this paper, we observe that surprisingly, directly editing a small subset of parameters can effectively modulate specific behaviors of LLMs, such as detoxification and resistance to jailbreaking. Specifically, for a behavior that we aim to avoid, we employ a linear classifier, which we term the behavior probe, to classify binary behavior labels within the hidden state space of the LLM. Using this probe, we introduce an algorithm to identify a critical subset of LLM parameters that significantly influence this targeted behavior. Then we directly edit these selected parameters by shifting them towards the behavior probe. Such a direct parameter editing method necessitates only inference-level computational resources. Experiments demonstrate that in the representative detoxification task, our approach achieves reductions of up to 90.0\% in toxicity on the RealToxicityPrompts dataset and 49.2\% on ToxiGen, while maintaining the LLM's general capabilities in areas such as common sense, question answering, and mathematics. Our code is available at https://github.com/lucywang720/model-surgery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.08269v1">LLMs' morphological analyses of complex FST-generated Finnish words</a></div>
    <div class="paper-meta">
      📅 2024-07-11
      | 💬 To appear at the CMCL Workshop at ACL 2024
    </div>
    <details class="paper-abstract">
      Rule-based language processing systems have been overshadowed by neural systems in terms of utility, but it remains unclear whether neural NLP systems, in practice, learn the grammar rules that humans use. This work aims to shed light on the issue by evaluating state-of-the-art LLMs in a task of morphological analysis of complex Finnish noun forms. We generate the forms using an FST tool, and they are unlikely to have occurred in the training sets of the LLMs, therefore requiring morphological generalisation capacity. We find that GPT-4-turbo has some difficulties in the task while GPT-3.5-turbo struggles and smaller models Llama2-70B and Poro-34B fail nearly completely.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.08249v1">GeNet: A Multimodal LLM-Based Co-Pilot for Network Topology and Configuration</a></div>
    <div class="paper-meta">
      📅 2024-07-11
    </div>
    <details class="paper-abstract">
      Communication network engineering in enterprise environments is traditionally a complex, time-consuming, and error-prone manual process. Most research on network engineering automation has concentrated on configuration synthesis, often overlooking changes in the physical network topology. This paper introduces GeNet, a multimodal co-pilot for enterprise network engineers. GeNet is a novel framework that leverages a large language model (LLM) to streamline network design workflows. It uses visual and textual modalities to interpret and update network topologies and device configurations based on user intents. GeNet was evaluated on enterprise network scenarios adapted from Cisco certification exercises. Our results demonstrate GeNet's ability to interpret network topology images accurately, potentially reducing network engineers' efforts and accelerating network design processes in enterprise environments. Furthermore, we show the importance of precise topology understanding when handling intents that require modifications to the network's topology.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.08240v1">Leveraging LLMs to Predict Affective States via Smartphone Sensor Features</a></div>
    <div class="paper-meta">
      📅 2024-07-11
    </div>
    <details class="paper-abstract">
      As mental health issues for young adults present a pressing public health concern, daily digital mood monitoring for early detection has become an important prospect. An active research area, digital phenotyping, involves collecting and analysing data from personal digital devices such as smartphones (usage and sensors) and wearables to infer behaviours and mental health. Whilst this data is standardly analysed using statistical and machine learning approaches, the emergence of large language models (LLMs) offers a new approach to make sense of smartphone sensing data. Despite their effectiveness across various domains, LLMs remain relatively unexplored in digital mental health, particularly in integrating mobile sensor data. Our study aims to bridge this gap by employing LLMs to predict affect outcomes based on smartphone sensing data from university students. We demonstrate the efficacy of zero-shot and few-shot embedding LLMs in inferring general wellbeing. Our findings reveal that LLMs can make promising predictions of affect measures using solely smartphone sensing data. This research sheds light on the potential of LLMs for affective state prediction, emphasizing the intricate link between smartphone behavioral patterns and affective states. To our knowledge, this is the first work to leverage LLMs for affective state prediction and digital phenotyping tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.09971v2">Constructing Benchmarks and Interventions for Combating Hallucinations in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-11
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are prone to hallucinations, which sparked a widespread effort to detect and prevent them. Recent work attempts to mitigate hallucinations by intervening in the model's generation, typically computing representative vectors of hallucinations vs. grounded generations, for steering the model's hidden states away from a hallucinatory state. However, common studies employ different setups and do not properly separate different possible causes of hallucinations, making interventions misguided. In this work, we introduce a method for categorizing examples based on the model's prior knowledge, named WACK. We construct WACK benchmarks that support interventions in two settings: open-book and closed-book question answering. Using the benchmarks, we perform an extensive investigation of the effect of different choices for intervention, such as the intervened components, and how often and how strongly to intervene. We find that intervention success varies depending on the component, with the attention blocks performing well and the residual stream proving detrimental to language modeling capabilities. We also show that interventions can benefit from representative vectors collected before, rather than after, a hallucination occurs. Finally, we introduce a new dynamic intervention, which intervenes only if needed, and thus is more robust than standard static interventions. The code is available at https://github.com/technion-cs-nlp/hallucination-mitigation .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.01325v3">LLM Maybe LongLM: Self-Extend LLM Context Window Without Tuning</a></div>
    <div class="paper-meta">
      📅 2024-07-11
      | 💬 ICML2024 Spotlight
    </div>
    <details class="paper-abstract">
      It is well known that LLMs cannot generalize well to long contexts whose lengths are larger than the training sequence length. This poses challenges when employing LLMs for processing long input sequences during inference. In this work, we argue that LLMs themselves have inherent capabilities to handle long contexts without fine-tuning. To achieve this goal, we propose SelfExtend to extend the context window of LLMs by constructing bi-level attention information: the grouped attention and the neighbor attention. The grouped attention captures the dependencies among tokens that are far apart, while neighbor attention captures dependencies among adjacent tokens within a specified range. The two-level attentions are computed based on the original model's self-attention mechanism during inference. With minor code modification, our SelfExtend can effortlessly extend existing LLMs' context window without any fine-tuning. We conduct comprehensive experiments on multiple benchmarks and the results show that our SelfExtend can effectively extend existing LLMs' context window length. The code can be found at \url{https://github.com/datamllab/LongLM}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.07376v2">LLMs in Biomedicine: A study on clinical Named Entity Recognition</a></div>
    <div class="paper-meta">
      📅 2024-07-11
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate remarkable versatility in various NLP tasks but encounter distinct challenges in biomedical due to the complexities of language and data scarcity. This paper investigates LLMs application in the biomedical domain by exploring strategies to enhance their performance for the NER task. Our study reveals the importance of meticulously designed prompts in the biomedical. Strategic selection of in-context examples yields a marked improvement, offering ~15-20\% increase in F1 score across all benchmark datasets for biomedical few-shot NER. Additionally, our results indicate that integrating external biomedical knowledge via prompting strategies can enhance the proficiency of general-purpose LLMs to meet the specialized needs of biomedical NER. Leveraging a medical knowledge base, our proposed method, DiRAG, inspired by Retrieval-Augmented Generation (RAG), can boost the zero-shot F1 score of LLMs for biomedical NER. Code is released at \url{https://github.com/masoud-monajati/LLM_Bio_NER}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.00284v2">A Closer Look at Logical Reasoning with LLMs: The Choice of Tool Matters</a></div>
    <div class="paper-meta">
      📅 2024-07-11
      | 💬 Code and data are publicly available at: https://github.com/Mattylam/Logic_Symbolic_Solvers_Experiment
    </div>
    <details class="paper-abstract">
      The emergence of Large Language Models (LLMs) has demonstrated promising progress in solving logical reasoning tasks effectively. Several recent approaches have proposed to change the role of the LLM from the reasoner into a translator between natural language statements and symbolic representations which are then sent to external symbolic solvers to resolve. This paradigm has established the current state-of-the-art result in logical reasoning (i.e., deductive reasoning). However, it remains unclear whether the variance in performance of these approaches stems from the methodologies employed or the specific symbolic solvers utilized. There is a lack of consistent comparison between symbolic solvers and how they influence the overall reported performance. This is important, as each symbolic solver also has its own input symbolic language, presenting varying degrees of challenge in the translation process. To address this gap, we perform experiments on 3 deductive reasoning benchmarks with LLMs augmented with widely used symbolic solvers: Z3, Pyke, and Prover9. The tool-executable rates of symbolic translation generated by different LLMs exhibit a near 50% performance variation. This highlights a significant difference in performance rooted in very basic choices of tools. The almost linear correlation between the executable rate of translations and the accuracy of the outcomes from Prover9 highlight a strong alignment between LLMs ability to translate into Prover9 symbolic language, and the correctness of those translations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.07796v2">Evaluating Large Language Models with Grid-Based Game Competitions: An Extensible LLM Benchmark and Leaderboard</a></div>
    <div class="paper-meta">
      📅 2024-07-11
    </div>
    <details class="paper-abstract">
      We introduce a novel and extensible benchmark for large language models (LLMs) through grid-based games such as Tic-Tac-Toe, Connect Four, and Gomoku. The open-source game simulation code, available on GitHub, allows LLMs to compete and generates detailed data files in JSON, CSV, TXT, and PNG formats for leaderboard rankings and further analysis. We present the results of games among leading LLMs, including Claude 3.5 Sonnet and Claude 3 Sonnet by Anthropic, Gemini 1.5 Pro and Gemini 1.5 Flash by Google, GPT-4 Turbo and GPT-4o by OpenAI, and Llama3-70B by Meta. We also encourage submissions of results from other LLMs. In total, we simulated 2,310 matches (5 sessions for each pair among 7 LLMs and a random player) across three types of games, using three distinct prompt types: list, illustration, and image. The results revealed significant variations in LLM performance across different games and prompt types, with analysis covering win and disqualification rates, missed opportunity analysis, and invalid move analysis. The details of the leaderboard and result matrix data are available as open-access data on GitHub. This study enhances our understanding of LLMs' capabilities in playing games they were not specifically trained for, helping to assess their rule comprehension and strategic thinking. On the path to Artificial General Intelligence (AGI), this study lays the groundwork for future exploration into their utility in complex decision-making scenarios, illuminating their strategic thinking abilities and offering directions for further inquiry into the limits of LLMs within game-based frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.05797v4">In-Context Explainers: Harnessing LLMs for Explaining Black Box Models</a></div>
    <div class="paper-meta">
      📅 2024-07-11
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have demonstrated exceptional capabilities in complex tasks like machine translation, commonsense reasoning, and language understanding. One of the primary reasons for the adaptability of LLMs in such diverse tasks is their in-context learning (ICL) capability, which allows them to perform well on new tasks by simply using a few task samples in the prompt. Despite their effectiveness in enhancing the performance of LLMs on diverse language and tabular tasks, these methods have not been thoroughly explored for their potential to generate post hoc explanations. In this work, we carry out one of the first explorations to analyze the effectiveness of LLMs in explaining other complex predictive models using ICL. To this end, we propose a novel framework, In-Context Explainers, comprising of three novel approaches that exploit the ICL capabilities of LLMs to explain the predictions made by other predictive models. We conduct extensive analysis with these approaches on real-world tabular and text datasets and demonstrate that LLMs are capable of explaining other predictive models similar to state-of-the-art post hoc explainers, opening up promising avenues for future research into LLM-based post hoc explanations of complex predictive models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.06645v3">Entropy Law: The Story Behind Data Compression and LLM Performance</a></div>
    <div class="paper-meta">
      📅 2024-07-11
    </div>
    <details class="paper-abstract">
      Data is the cornerstone of large language models (LLMs), but not all data is useful for model learning. Carefully selected data can better elicit the capabilities of LLMs with much less computational overhead. Most methods concentrate on evaluating the quality of individual samples in data selection, while the combinatorial effects among samples are neglected. Even if each sample is of perfect quality, their combinations may be suboptimal in teaching LLMs due to their intrinsic homogeneity or contradiction. In this paper, we aim to uncover the underlying relationships between LLM performance and data selection. Inspired by the information compression nature of LLMs, we uncover an ``entropy law'' that connects LLM performance with data compression ratio and first-epoch training loss, which reflect the information redundancy of a dataset and the mastery of inherent knowledge encoded in this dataset, respectively. Through both theoretical deduction and empirical evaluation, we find that model performance is negatively correlated to the compression ratio of training data, which usually yields a lower training loss. Based on the findings of the entropy law, we propose a quite efficient and universal data selection method named \textbf{ZIP} for training LLMs, which aim to prioritize data subsets exhibiting a low compression ratio. Based on a multi-stage algorithm that selects diverse data in a greedy manner, we can obtain a good data subset with satisfactory diversity. Extensive experiments have been conducted to validate the entropy law and the superiority of ZIP across different LLM backbones and alignment stages. We also present an interesting application of entropy law that can detect potential performance risks at the beginning of model training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.04051v3">FunAudioLLM: Voice Understanding and Generation Foundation Models for Natural Interaction Between Humans and LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-11
      | 💬 Work in progress. Authors are listed in alphabetical order by family name
    </div>
    <details class="paper-abstract">
      This report introduces FunAudioLLM, a model family designed to enhance natural voice interactions between humans and large language models (LLMs). At its core are two innovative models: SenseVoice, which handles multilingual speech recognition, emotion recognition, and audio event detection; and CosyVoice, which facilitates natural speech generation with control over multiple languages, timbre, speaking style, and speaker identity. SenseVoice-Small delivers exceptionally low-latency ASR for 5 languages, and SenseVoice-Large supports high-precision ASR for over 50 languages, while CosyVoice excels in multi-lingual voice generation, zero-shot in-context learning, cross-lingual voice cloning, and instruction-following capabilities. The models related to SenseVoice and CosyVoice have been open-sourced on Modelscope and Huggingface, along with the corresponding training, inference, and fine-tuning codes released on GitHub. By integrating these models with LLMs, FunAudioLLM enables applications such as speech-to-speech translation, emotional voice chat, interactive podcasts, and expressive audiobook narration, thereby pushing the boundaries of voice interaction technology. Demos are available at https://fun-audio-llm.github.io, and the code can be accessed at https://github.com/FunAudioLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.08095v1">Virtual Agents for Alcohol Use Counseling: Exploring LLM-Powered Motivational Interviewing</a></div>
    <div class="paper-meta">
      📅 2024-07-10
    </div>
    <details class="paper-abstract">
      We introduce a novel application of large language models (LLMs) in developing a virtual counselor capable of conducting motivational interviewing (MI) for alcohol use counseling. Access to effective counseling remains limited, particularly for substance abuse, and virtual agents offer a promising solution by leveraging LLM capabilities to simulate nuanced communication techniques inherent in MI. Our approach combines prompt engineering and integration into a user-friendly virtual platform to facilitate realistic, empathetic interactions. We evaluate the effectiveness of our virtual agent through a series of studies focusing on replicating MI techniques and human counselor dialog. Initial findings suggest that our LLM-powered virtual agent matches human counselors' empathetic and adaptive conversational skills, presenting a significant step forward in virtual health counseling and providing insights into the design and implementation of LLM-based therapeutic interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.08067v1">On LLM Wizards: Identifying Large Language Models' Behaviors for Wizard of Oz Experiments</a></div>
    <div class="paper-meta">
      📅 2024-07-10
      | 💬 To be published in ACM IVA 2024
    </div>
    <details class="paper-abstract">
      The Wizard of Oz (WoZ) method is a widely adopted research approach where a human Wizard ``role-plays'' a not readily available technology and interacts with participants to elicit user behaviors and probe the design space. With the growing ability for modern large language models (LLMs) to role-play, one can apply LLMs as Wizards in WoZ experiments with better scalability and lower cost than the traditional approach. However, methodological guidance on responsibly applying LLMs in WoZ experiments and a systematic evaluation of LLMs' role-playing ability are lacking. Through two LLM-powered WoZ studies, we take the first step towards identifying an experiment lifecycle for researchers to safely integrate LLMs into WoZ experiments and interpret data generated from settings that involve Wizards role-played by LLMs. We also contribute a heuristic-based evaluation framework that allows the estimation of LLMs' role-playing ability in WoZ experiments and reveals LLMs' behavior patterns at scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.08460v2">Is Your LLM Outdated? Evaluating LLMs at Temporal Generalization</a></div>
    <div class="paper-meta">
      📅 2024-07-10
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) highlights the urgent need for evolving evaluation methodologies that keep pace with improvements in language comprehension and information processing. However, traditional benchmarks, which are often static, fail to capture the continually changing information landscape, leading to a disparity between the perceived and actual effectiveness of LLMs in ever-changing real-world scenarios. Our study examines temporal generalization, which includes the ability to understand, predict, and generate text relevant to past, present, and future contexts, revealing significant temporal biases in LLMs. We propose an evaluation framework, for dynamically generating benchmarks from recent real-world predictions. Experiments demonstrate that LLMs struggle with temporal generalization, showing performance decline over time. These findings highlight the necessity for improved training and updating processes to enhance adaptability and reduce biases. Our code, dataset and benchmark are available at https://github.com/FreedomIntelligence/FreshBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.02502v2">Trial and Error: Exploration-Based Trajectory Optimization for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2024-07-10
      | 💬 Accepted to ACL 2024 Main Conference; Camera Ready
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become integral components in various autonomous agent systems. In this study, we present an exploration-based trajectory optimization approach, referred to as ETO. This learning method is designed to enhance the performance of open LLM agents. Contrary to previous studies that exclusively train on successful expert trajectories, our method allows agents to learn from their exploration failures. This leads to improved performance through an iterative optimization framework. During the exploration phase, the agent interacts with the environment while completing given tasks, gathering failure trajectories to create contrastive trajectory pairs. In the subsequent training phase, the agent utilizes these trajectory preference pairs to update its policy using contrastive learning methods like DPO. This iterative cycle of exploration and training fosters continued improvement in the agents. Our experiments on three complex tasks demonstrate that ETO consistently surpasses baseline performance by a large margin. Furthermore, an examination of task-solving efficiency and potential in scenarios lacking expert trajectory underscores the effectiveness of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.01616v3">Transforming LLMs into Cross-modal and Cross-lingual Retrieval Systems</a></div>
    <div class="paper-meta">
      📅 2024-07-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are trained on text-only data that go far beyond the languages with paired speech and text data. At the same time, Dual Encoder (DE) based retrieval systems project queries and documents into the same embedding space and have demonstrated their success in retrieval and bi-text mining. To match speech and text in many languages, we propose using LLMs to initialize multi-modal DE retrieval systems. Unlike traditional methods, our system doesn't require speech data during LLM pre-training and can exploit LLM's multilingual text understanding capabilities to match speech and text in languages unseen during retrieval training. Our multi-modal LLM-based retrieval system is capable of matching speech and text in 102 languages despite only training on 21 languages. Our system outperforms previous systems trained explicitly on all 102 languages. We achieve a 10% absolute improvement in Recall@1 averaged across these languages. Additionally, our model demonstrates cross-lingual speech and text matching, which is further enhanced by readily available machine translation data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.08658v1">Evaluating Voice Command Pipelines for Drone Control: From STT and LLM to Direct Classification and Siamese Networks</a></div>
    <div class="paper-meta">
      📅 2024-07-10
    </div>
    <details class="paper-abstract">
      This paper presents the development and comparative evaluation of three voice command pipelines for controlling a Tello drone, using speech recognition and deep learning techniques. The aim is to enhance human-machine interaction by enabling intuitive voice control of drone actions. The pipelines developed include: (1) a traditional Speech-to-Text (STT) followed by a Large Language Model (LLM) approach, (2) a direct voice-to-function mapping model, and (3) a Siamese neural network-based system. Each pipeline was evaluated based on inference time, accuracy, efficiency, and flexibility. Detailed methodologies, dataset preparation, and evaluation metrics are provided, offering a comprehensive analysis of each pipeline's strengths and applicability across different scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.12273v2">The Ethics of Interaction: Mitigating Security Threats in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-10
    </div>
    <details class="paper-abstract">
      This paper comprehensively explores the ethical challenges arising from security threats to Large Language Models (LLMs). These intricate digital repositories are increasingly integrated into our daily lives, making them prime targets for attacks that can compromise their training data and the confidentiality of their data sources. The paper delves into the nuanced ethical repercussions of such security threats on society and individual privacy. We scrutinize five major threats--prompt injection, jailbreaking, Personal Identifiable Information (PII) exposure, sexually explicit content, and hate-based content--going beyond mere identification to assess their critical ethical consequences and the urgency they create for robust defensive strategies. The escalating reliance on LLMs underscores the crucial need for ensuring these systems operate within the bounds of ethical norms, particularly as their misuse can lead to significant societal and individual harm. We propose conceptualizing and developing an evaluative tool tailored for LLMs, which would serve a dual purpose: guiding developers and designers in preemptive fortification of backend systems and scrutinizing the ethical dimensions of LLM chatbot responses during the testing phase. By comparing LLM responses with those expected from humans in a moral context, we aim to discern the degree to which AI behaviors align with the ethical values held by a broader society. Ultimately, this paper not only underscores the ethical troubles presented by LLMs; it also highlights a path toward cultivating trust in these systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.11202v2">Data is all you need: Finetuning LLMs for Chip Design via an Automated design-data augmentation framework</a></div>
    <div class="paper-meta">
      📅 2024-07-10
      | 💬 DAC 2024
    </div>
    <details class="paper-abstract">
      Recent advances in large language models have demonstrated their potential for automated generation of hardware description language (HDL) code from high-level prompts. Researchers have utilized fine-tuning to enhance the ability of these large language models (LLMs) in the field of Chip Design. However, the lack of Verilog data hinders further improvement in the quality of Verilog generation by LLMs. Additionally, the absence of a Verilog and Electronic Design Automation (EDA) script data augmentation framework significantly increases the time required to prepare the training dataset for LLM trainers. This paper proposes an automated design-data augmentation framework, which generates high-volume and high-quality natural language aligned with Verilog and EDA scripts. For Verilog generation, it translates Verilog files to an abstract syntax tree and then maps nodes to natural language with a predefined template. For Verilog repair, it uses predefined rules to generate the wrong verilog file and then pairs EDA Tool feedback with the right and wrong verilog file. For EDA Script generation, it uses existing LLM(GPT-3.5) to obtain the description of the Script. To evaluate the effectiveness of our data augmentation method, we finetune Llama2-13B and Llama2-7B models using the dataset generated by our augmentation framework. The results demonstrate a significant improvement in the Verilog generation tasks with LLMs. Moreover, the accuracy of Verilog generation surpasses that of the current state-of-the-art open-source Verilog generation model, increasing from 58.8% to 70.6% with the same benchmark. Our 13B model (ChipGPT-FT) has a pass rate improvement compared with GPT-3.5 in Verilog generation and outperforms in EDA script (i.e., SiliconCompiler) generation with only 200 EDA script data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.04675v2">Seed-ASR: Understanding Diverse Speech and Contexts with LLM-based Speech Recognition</a></div>
    <div class="paper-meta">
      📅 2024-07-10
    </div>
    <details class="paper-abstract">
      Modern automatic speech recognition (ASR) model is required to accurately transcribe diverse speech signals (from different domains, languages, accents, etc) given the specific contextual information in various application scenarios. Classic end-to-end models fused with extra language models perform well, but mainly in data matching scenarios and are gradually approaching a bottleneck. In this work, we introduce Seed-ASR, a large language model (LLM) based speech recognition model. Seed-ASR is developed based on the framework of audio conditioned LLM (AcLLM), leveraging the capabilities of LLMs by inputting continuous speech representations together with contextual information into the LLM. Through stage-wise large-scale training and the elicitation of context-aware capabilities in LLM, Seed-ASR demonstrates significant improvement over end-to-end models on comprehensive evaluation sets, including multiple domains, accents/dialects and languages. Additionally, Seed-ASR can be further deployed to support specific needs in various scenarios without requiring extra language models. Compared to recently released large ASR models, Seed-ASR achieves 10%-40% reduction in word (or character, for Chinese) error rates on Chinese and English public test sets, further demonstrating its powerful performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.07472v1">Rectifier: Code Translation with Corrector via LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-10
      | 💬 arXiv admin note: text overlap with arXiv:2308.03109, arXiv:2302.03908 by other authors
    </div>
    <details class="paper-abstract">
      Software migration is garnering increasing attention with the evolution of software and society. Early studies mainly relied on handcrafted translation rules to translate between two languages, the translation process is error-prone and time-consuming. In recent years, researchers have begun to explore the use of pre-trained large language models (LLMs) in code translation. However, code translation is a complex task that LLMs would generate mistakes during code translation, they all produce certain types of errors when performing code translation tasks, which include (1) compilation error, (2) runtime error, (3) functional error, and (4) non-terminating execution. We found that the root causes of these errors are very similar (e.g. failure to import packages, errors in loop boundaries, operator errors, and more). In this paper, we propose a general corrector, namely Rectifier, which is a micro and universal model for repairing translation errors. It learns from errors generated by existing LLMs and can be widely applied to correct errors generated by any LLM. The experimental results on translation tasks between C++, Java, and Python show that our model has effective repair ability, and cross experiments also demonstrate the robustness of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.12860v1">STAGE: Simplified Text-Attributed Graph Embeddings Using Pre-trained LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-10
    </div>
    <details class="paper-abstract">
      We present Simplified Text-Attributed Graph Embeddings (STAGE), a straightforward yet effective method for enhancing node features in Graph Neural Network (GNN) models that encode Text-Attributed Graphs (TAGs). Our approach leverages Large-Language Models (LLMs) to generate embeddings for textual attributes. STAGE achieves competitive results on various node classification benchmarks while also maintaining a simplicity in implementation relative to current state-of-the-art (SoTA) techniques. We show that utilizing pre-trained LLMs as embedding generators provides robust features for ensemble GNN training, enabling pipelines that are simpler than current SoTA approaches which require multiple expensive training and prompting stages. We also implement diffusion-pattern GNNs in an effort to make this pipeline scalable to graphs beyond academic benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.07342v1">Multilingual Blending: LLM Safety Alignment Evaluation with Language Mixture</a></div>
    <div class="paper-meta">
      📅 2024-07-10
    </div>
    <details class="paper-abstract">
      As safety remains a crucial concern throughout the development lifecycle of Large Language Models (LLMs), researchers and industrial practitioners have increasingly focused on safeguarding and aligning LLM behaviors with human preferences and ethical standards. LLMs, trained on extensive multilingual corpora, exhibit powerful generalization abilities across diverse languages and domains. However, current safety alignment practices predominantly focus on single-language scenarios, which leaves their effectiveness in complex multilingual contexts, especially for those complex mixed-language formats, largely unexplored. In this study, we introduce Multilingual Blending, a mixed-language query-response scheme designed to evaluate the safety alignment of various state-of-the-art LLMs (e.g., GPT-4o, GPT-3.5, Llama3) under sophisticated, multilingual conditions. We further investigate language patterns such as language availability, morphology, and language family that could impact the effectiveness of Multilingual Blending in compromising the safeguards of LLMs. Our experimental results show that, without meticulously crafted prompt templates, Multilingual Blending significantly amplifies the detriment of malicious queries, leading to dramatically increased bypass rates in LLM safety alignment (67.23% on GPT-3.5 and 40.34% on GPT-4o), far exceeding those of single-language baselines. Moreover, the performance of Multilingual Blending varies notably based on intrinsic linguistic properties, with languages of different morphology and from diverse families being more prone to evading safety alignments. These findings underscore the necessity of evaluating LLMs and developing corresponding safety alignment strategies in a complex, multilingual context to align with their superior cross-language generalization capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.16713v2">Navigating Complexity: Orchestrated Problem Solving with Multi-Agent LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities in solving various tasks, yet they often struggle with comprehensively addressing complex and vague problems. Existing approaches, including multi-agent LLM systems, offer solutions to certain challenges but still require manual setup and lack scalability. To address this gap, we propose a novel approach leveraging decomposition to enable LLMs to tackle vague problems effectively. Our approach involves an orchestrating LLM that interacts with users to understand the problem and then decomposes it into tangible sub-problems. Instead of expecting the LLM to solve the entire problem in one go, we train it to ask follow-up questions to gain a deeper understanding of the user's requirements. Once the problem is adequately understood, the orchestrating LLM divides it into smaller, manageable sub-problems. Each sub-problem is then assigned to specialized LLM agents or non-LLM functions for resolution. These agents work in parallel to solve their respective sub-problems, with the orchestrating LLM overseeing the process and compiling the solutions into a comprehensive answer for the user. By adopting this decomposition approach, we alleviate the constraints imposed by token limitations on LLM outputs and empower them to provide nuanced solutions to complex and ambiguous problems. Through our approach, we aim to enable LLMs to think and operate more like humans, breaking down complex problems into manageable parts and collaboratively solving them. This not only enhances the problem-solving capabilities of LLMs but also offers a scalable and efficient method for addressing a wide range of real-world challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.12426v2">Can LLMs Augment Low-Resource Reading Comprehension Datasets? Opportunities and Challenges</a></div>
    <div class="paper-meta">
      📅 2024-07-10
      | 💬 ACL 2024 SRW
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive zero shot performance on a wide range of NLP tasks, demonstrating the ability to reason and apply commonsense. A relevant application is to use them for creating high quality synthetic datasets for downstream tasks. In this work, we probe whether GPT-4 can be used to augment existing extractive reading comprehension datasets. Automating data annotation processes has the potential to save large amounts of time, money and effort that goes into manually labelling datasets. In this paper, we evaluate the performance of GPT-4 as a replacement for human annotators for low resource reading comprehension tasks, by comparing performance after fine tuning, and the cost associated with annotation. This work serves to be the first analysis of LLMs as synthetic data augmenters for QA systems, highlighting the unique opportunities and challenges. Additionally, we release augmented versions of low resource datasets, that will allow the research community to create further benchmarks for evaluation of generated datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.14864v3">Just CHOP: Embarrassingly Simple LLM Compression</a></div>
    <div class="paper-meta">
      📅 2024-07-09
      | 💬 13 pages, 6 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) enable unparalleled few- and zero-shot reasoning capabilities but at a high computational footprint. A growing assortment of methods for compression promises to reduce the computational burden of LLMs in deployment, but so far, only quantization approaches have been demonstrated to be effective for LLM compression while maintaining zero-shot performance. A critical step in the compression process, the pretrain-then-finetune paradigm, has largely been overlooked when adapting existing pruning strategies to LLMs or proposing new ones. In this work, we show that embarrassingly simple layer pruning coupled with an extended language model pretraining as the finetuning phase produces state-of-the-art results against structured and even semi-structured compression of models at a 7B scale while being more inference efficient. We call this method LayerChop, where we deterministically remove layers from a model followed by task-agnostic finetuning of the remaining weights by continued self-supervised pretraining. At this scale, we also show how distillation, which has been super effective in task-agnostic compression of smaller BERT-style models, becomes inefficient against our simple pruning technique.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.07093v1">FBI-LLM: Scaling Up Fully Binarized LLMs from Scratch via Autoregressive Distillation</a></div>
    <div class="paper-meta">
      📅 2024-07-09
      | 💬 Github at https://github.com/LiqunMa/FBI-LLM
    </div>
    <details class="paper-abstract">
      This work presents a Fully BInarized Large Language Model (FBI-LLM), demonstrating for the first time how to train a large-scale binary language model from scratch (not the partial binary or ternary LLM like BitNet b1.58) to match the performance of its full-precision counterparts (e.g., FP16 or BF16) in transformer-based LLMs. It achieves this by employing an autoregressive distillation (AD) loss with maintaining equivalent model dimensions (130M, 1.3B, 7B) and training data volume as regular LLM pretraining, while delivering competitive results in terms of perplexity and task-specific effectiveness. Intriguingly, by analyzing the training trajectory, we find that the pretrained weight is not necessary for training binarized LLMs from scratch. This research encourages a new computational framework and may facilitate the future design of specialized hardware tailored for fully 1-bit LLMs. We make all models, code, and training dataset fully accessible and transparent to support further research (Code: https://github.com/LiqunMa/FBI-LLM. Model: https://huggingface.co/LiqunMa/).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.07080v1">Adapting LLMs to Hebrew: Unveiling DictaLM 2.0 with Enhanced Vocabulary and Instruction Capabilities</a></div>
    <div class="paper-meta">
      📅 2024-07-09
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) in low-resource languages such as Hebrew poses unique challenges. In this paper, we introduce DictaLM2.0 and DictaLM2.0-Instruct, two LLMs derived from the Mistral model, trained on a substantial corpus of approximately 200 billion tokens in both Hebrew and English. Adapting a pre-trained model to a new language involves specialized techniques that differ significantly from training a model from scratch or further training existing models on well-resourced languages such as English. We outline these novel training methodologies, which facilitate effective learning and adaptation to the linguistic properties of Hebrew. Additionally, we fine-tuned DictaLM2.0-Instruct on a comprehensive instruct dataset to enhance its performance on task-specific instructions. To rigorously evaluate our models, we introduce a new benchmark suite for Hebrew LLM evaluation, covering a diverse set of tasks including Question Answering, Sentiment Analysis, Winograd Schema Challenge, Translation, and Summarization. Our work not only addresses the intricacies of training LLMs in low-resource languages but also proposes a framework that can be leveraged for adapting other LLMs to various non-English languages, contributing to the broader field of multilingual NLP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.06129v2">Evaluating the Semantic Profiling Abilities of LLMs for Natural Language Utterances in Data Visualization</a></div>
    <div class="paper-meta">
      📅 2024-07-09
      | 💬 5 pages, 4 figures, IEEE VIS short papers
    </div>
    <details class="paper-abstract">
      Automatically generating data visualizations in response to human utterances on datasets necessitates a deep semantic understanding of the data utterance, including implicit and explicit references to data attributes, visualization tasks, and necessary data preparation steps. Natural Language Interfaces (NLIs) for data visualization have explored ways to infer such information, yet challenges persist due to inherent uncertainty in human speech. Recent advances in Large Language Models (LLMs) provide an avenue to address these challenges, but their ability to extract the relevant semantic information remains unexplored. In this study, we evaluate four publicly available LLMs (GPT-4, Gemini-Pro, Llama3, and Mixtral), investigating their ability to comprehend utterances even in the presence of uncertainty and identify the relevant data context and visual tasks. Our findings reveal that LLMs are sensitive to uncertainties in utterances. Despite this sensitivity, they are able to extract the relevant data context. However, LLMs struggle with inferring visualization tasks. Based on these results, we highlight future research directions on using LLMs for visualization generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.02147v2">GemmAr: Enhancing LLMs Through Arabic Instruction-Tuning</a></div>
    <div class="paper-meta">
      📅 2024-07-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have greatly impacted the natural language processing (NLP) field, particularly for the English language. These models have demonstrated capabilities in understanding and generating human-like text. The success of language models largely depends on the availability of high-quality instruction datasets, which consist of detailed task descriptions and corresponding responses that are essential for training the models to address a variety of prompts accurately. However, the availability and quality of these resources vary by language. While models perform well in English, they often need help with languages like Arabic, due to the lack of datasets for fine-tuning Arabic-specific tasks. To address this issue, we introduce InstAr-500k, a new Arabic instruction dataset created by generating and collecting content that covers several domains and instruction types. We assess this dataset by fine-tuning an open-source Gemma-7B model on several downstream tasks to improve its functionality. Based on multiple evaluations, our fine-tuned model achieves excellent performance on several Arabic NLP benchmarks. These outcomes emphasize the effectiveness of our dataset in elevating the capabilities of language models for Arabic. Our instruction dataset bridges the performance gap between English and Arabic language models by providing resources that amplify Arabic NLP development. Building on this foundation, we developed a model, GemmAr-7B-V1, specifically tuned to excel at a wide range of Arabic NLP tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.15723v2">A hybrid LLM workflow can help identify user privilege related variables in programs of any size</a></div>
    <div class="paper-meta">
      📅 2024-07-09
    </div>
    <details class="paper-abstract">
      Many programs involves operations and logic manipulating user privileges, which is essential for the security of an organization. Therefore, one common malicious goal of attackers is to obtain or escalate the privileges, causing privilege leakage. To protect the program and the organization against privilege leakage attacks, it is important to eliminate the vulnerabilities which can be exploited to achieve such attacks. Unfortunately, while memory vulnerabilities are less challenging to find, logic vulnerabilities are much more imminent, harmful and difficult to identify. Accordingly, many analysts choose to find user privilege related (UPR) variables first as start points to investigate the code where the UPR variables may be used to see if there exists any vulnerabilities, especially the logic ones. In this paper, we introduce a large language model (LLM) workflow that can assist analysts in identifying such UPR variables, which is considered to be a very time-consuming task. Specifically, our tool will audit all the variables in a program and output a UPR score, which is the degree of relationship (closeness) between the variable and user privileges, for each variable. The proposed approach avoids the drawbacks introduced by directly prompting a LLM to find UPR variables by focusing on leverage the LLM at statement level instead of supplying LLM with very long code snippets. Those variables with high UPR scores are essentially potential UPR variables, which should be manually investigated. Our experiments show that using a typical UPR score threshold (i.e., UPR score >0.8), the false positive rate (FPR) is only 13.49%, while UPR variable found is significantly more than that of the heuristic based method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.02181v3">Not All Layers of LLMs Are Necessary During Inference</a></div>
    <div class="paper-meta">
      📅 2024-07-09
    </div>
    <details class="paper-abstract">
      Due to the large number of parameters, the inference phase of Large Language Models (LLMs) is resource-intensive. However, not all requests posed to LLMs are equally difficult to handle. Through analysis, we show that for some tasks, LLMs can achieve results comparable to the final output at some intermediate layers. That is, not all layers of LLMs are necessary during inference. If we can predict at which layer the inferred results match the final results (produced by evaluating all layers), we could significantly reduce the inference cost. To this end, we propose a simple yet effective algorithm named AdaInfer to adaptively terminate the inference process for an input instance. AdaInfer relies on easily obtainable statistical features and classic classifiers like SVM. Experiments on well-known LLMs like the Llama2 series and OPT, show that AdaInfer can achieve an average of 17.8% pruning ratio, and up to 43% on sentiment tasks, with nearly no performance drop (<1%). Because AdaInfer does not alter LLM parameters, the LLMs incorporated with AdaInfer maintain generalizability across tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.06146v2">Using Grammar Masking to Ensure Syntactic Validity in LLM-based Modeling Tasks</a></div>
    <div class="paper-meta">
      📅 2024-07-09
      | 💬 Preprint to be published in the MODELS Workshop "MDE Intelligence"
    </div>
    <details class="paper-abstract">
      We present and evaluate a method called grammar masking, which is used to guide large language models (LLMs) toward producing syntactically correct models for a given context-free grammar. Prompt engineering methods such as few-shot learning or priming can be used to improve the chances of an LLM producing correct syntax, but the more complex the grammar, the more time-consuming and less promising these methods become. Previous work is focused primarily on the usage of either language model training or prompt engineering. In this work, a method is presented that restricts the output to a given grammar using constrained decoding to ensure the output adheres to a valid syntax. We use several DSLs built with MontiCore and task multiple LLMs to produce models with and without constrained decoding. A corresponding parser is used to confirm the syntactic correctness of each model. We show that grammar masking can dramatically improve the modeling capabilities of several LLMs, reducing the need for well-refined prompting while increasing the chance of producing correct models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.06573v1">LLM for Mobile: An Initial Roadmap</a></div>
    <div class="paper-meta">
      📅 2024-07-09
    </div>
    <details class="paper-abstract">
      When mobile meets LLMs, mobile app users deserve to have more intelligent usage experiences. For this to happen, we argue that there is a strong need to appl LLMs for the mobile ecosystem. We therefore provide a research roadmap for guiding our fellow researchers to achieve that as a whole. In this roadmap, we sum up six directions that we believe are urgently required for research to enable native intelligence in mobile devices. In each direction, we further summarize the current research progress and the gaps that still need to be filled by our fellow researchers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.00079v3">Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving</a></div>
    <div class="paper-meta">
      📅 2024-07-09
      | 💬 23 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Mooncake is the serving platform for Kimi, a leading LLM service provided by Moonshot AI. It features a KVCache-centric disaggregated architecture that separates the prefill and decoding clusters. It also leverages the underutilized CPU, DRAM, and SSD resources of the GPU cluster to implement a disaggregated cache of KVCache. The core of Mooncake is its KVCache-centric scheduler, which balances maximizing overall effective throughput while meeting latency-related Service Level Objectives (SLOs). Unlike traditional studies that assume all requests will be processed, Mooncake faces challenges due to highly overloaded scenarios. To mitigate these, we developed a prediction-based early rejection policy. Experiments show that Mooncake excels in long-context scenarios. Compared to the baseline method, Mooncake can achieve up to a 525% increase in throughput in certain simulated scenarios while adhering to SLOs. Under real workloads, Mooncake's innovative architecture enables Kimi to handle 75% more requests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.06443v1">Exposing Privacy Gaps: Membership Inference Attack on Preference Data for LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2024-07-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have seen widespread adoption due to their remarkable natural language capabilities. However, when deploying them in real-world settings, it is important to align LLMs to generate texts according to acceptable human standards. Methods such as Proximal Policy Optimization (PPO) and Direct Preference Optimization (DPO) have made significant progress in refining LLMs using human preference data. However, the privacy concerns inherent in utilizing such preference data have yet to be adequately studied. In this paper, we investigate the vulnerability of LLMs aligned using human preference datasets to membership inference attacks (MIAs), highlighting the shortcomings of previous MIA approaches with respect to preference data. Our study has two main contributions: first, we introduce a novel reference-based attack framework specifically for analyzing preference data called PREMIA (\uline{Pre}ference data \uline{MIA}); second, we provide empirical evidence that DPO models are more vulnerable to MIA compared to PPO models. Our findings highlight gaps in current privacy-preserving practices for LLM alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.04798v2">From LLMs to Actions: Latent Codes as Bridges in Hierarchical Robot Control</a></div>
    <div class="paper-meta">
      📅 2024-07-08
    </div>
    <details class="paper-abstract">
      Hierarchical control for robotics has long been plagued by the need to have a well defined interface layer to communicate between high-level task planners and low-level policies. With the advent of LLMs, language has been emerging as a prospective interface layer. However, this has several limitations. Not all tasks can be decomposed into steps that are easily expressible in natural language (e.g. performing a dance routine). Further, it makes end-to-end finetuning on embodied data challenging due to domain shift and catastrophic forgetting. We introduce our method -- Learnable Latent Codes as Bridges (LCB) -- as an alternate architecture to overcome these limitations. \method~uses a learnable latent code to act as a bridge between LLMs and low-level policies. This enables LLMs to flexibly communicate goals in the task plan without being entirely constrained by language limitations. Additionally, it enables end-to-end finetuning without destroying the embedding space of word tokens learned during pre-training. Through experiments on Language Table and Calvin, two common language based benchmarks for embodied agents, we find that \method~outperforms baselines (including those w/ GPT-4V) that leverage pure language as the interface layer on tasks that require reasoning and multi-step behaviors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.04383v2">Exploring the Latest LLMs for Leaderboard Extraction</a></div>
    <div class="paper-meta">
      📅 2024-07-08
    </div>
    <details class="paper-abstract">
      The rapid advancements in Large Language Models (LLMs) have opened new avenues for automating complex tasks in AI research. This paper investigates the efficacy of different LLMs-Mistral 7B, Llama-2, GPT-4-Turbo and GPT-4.o in extracting leaderboard information from empirical AI research articles. We explore three types of contextual inputs to the models: DocTAET (Document Title, Abstract, Experimental Setup, and Tabular Information), DocREC (Results, Experiments, and Conclusions), and DocFULL (entire document). Our comprehensive study evaluates the performance of these models in generating (Task, Dataset, Metric, Score) quadruples from research papers. The findings reveal significant insights into the strengths and limitations of each model and context type, providing valuable guidance for future AI research automation efforts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.11508v2">Towards LLM-based Autograding for Short Textual Answers</a></div>
    <div class="paper-meta">
      📅 2024-07-08
      | 💬 Proceedings of the 16th International Conference on Computer Supported Education (CSEDU 2024)
    </div>
    <details class="paper-abstract">
      Grading exams is an important, labor-intensive, subjective, repetitive, and frequently challenging task. The feasibility of autograding textual responses has greatly increased thanks to the availability of large language models (LLMs) such as ChatGPT and the substantial influx of data brought about by digitalization. However, entrusting AI models with decision-making roles raises ethical considerations, mainly stemming from potential biases and issues related to generating false information. Thus, in this manuscript, we provide an evaluation of a large language model for the purpose of autograding, while also highlighting how LLMs can support educators in validating their grading procedures. Our evaluation is targeted towards automatic short textual answers grading (ASAG), spanning various languages and examinations from two distinct courses. Our findings suggest that while "out-of-the-box" LLMs provide a valuable tool to provide a complementary perspective, their readiness for independent automated grading remains a work in progress, necessitating human oversight.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.05977v1">Exploring Human-LLM Conversations: Mental Models and the Originator of Toxicity</a></div>
    <div class="paper-meta">
      📅 2024-07-08
    </div>
    <details class="paper-abstract">
      This study explores real-world human interactions with large language models (LLMs) in diverse, unconstrained settings in contrast to most prior research focusing on ethically trimmed models like ChatGPT for specific tasks. We aim to understand the originator of toxicity. Our findings show that although LLMs are rightfully accused of providing toxic content, it is mostly demanded or at least provoked by humans who actively seek such content. Our manual analysis of hundreds of conversations judged as toxic by APIs commercial vendors, also raises questions with respect to current practices of what user requests are refused to answer. Furthermore, we conjecture based on multiple empirical indicators that humans exhibit a change of their mental model, switching from the mindset of interacting with a machine more towards interacting with a human.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.05925v1">Towards Optimizing and Evaluating a Retrieval Augmented QA Chatbot using LLMs with Human in the Loop</a></div>
    <div class="paper-meta">
      📅 2024-07-08
    </div>
    <details class="paper-abstract">
      Large Language Models have found application in various mundane and repetitive tasks including Human Resource (HR) support. We worked with the domain experts of SAP SE to develop an HR support chatbot as an efficient and effective tool for addressing employee inquiries. We inserted a human-in-the-loop in various parts of the development cycles such as dataset collection, prompt optimization, and evaluation of generated output. By enhancing the LLM-driven chatbot's response quality and exploring alternative retrieval methods, we have created an efficient, scalable, and flexible tool for HR professionals to address employee inquiries effectively. Our experiments and evaluation conclude that GPT-4 outperforms other models and can overcome inconsistencies in data through internal reasoning capabilities. Additionally, through expert analysis, we infer that reference-free evaluation metrics such as G-Eval and Prometheus demonstrate reliability closely aligned with that of human evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.05887v1">Generation and De-Identification of Indian Clinical Discharge Summaries using LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-08
      | 💬 Accepted at BioNLP Workshop at ACL 2024; 21 pages (9 pages main content)
    </div>
    <details class="paper-abstract">
      The consequences of a healthcare data breach can be devastating for the patients, providers, and payers. The average financial impact of a data breach in recent months has been estimated to be close to USD 10 million. This is especially significant for healthcare organizations in India that are managing rapid digitization while still establishing data governance procedures that align with the letter and spirit of the law. Computer-based systems for de-identification of personal information are vulnerable to data drift, often rendering them ineffective in cross-institution settings. Therefore, a rigorous assessment of existing de-identification against local health datasets is imperative to support the safe adoption of digital health initiatives in India. Using a small set of de-identified patient discharge summaries provided by an Indian healthcare institution, in this paper, we report the nominal performance of de-identification algorithms (based on language models) trained on publicly available non-Indian datasets, pointing towards a lack of cross-institutional generalization. Similarly, experimentation with off-the-shelf de-identification systems reveals potential risks associated with the approach. To overcome data scarcity, we explore generating synthetic clinical reports (using publicly available and Indian summaries) by performing in-context learning over Large Language Models (LLMs). Our experiments demonstrate the use of generated reports as an effective strategy for creating high-performing de-identification systems with good generalization capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.13972v2">CREF: An LLM-based Conversational Software Repair Framework for Programming Tutors</a></div>
    <div class="paper-meta">
      📅 2024-07-08
    </div>
    <details class="paper-abstract">
      Program repair techniques offer cost-saving benefits for debugging within software development and programming education scenarios. With the proven effectiveness of Large Language Models (LLMs) in code-related tasks, researchers have explored their potential for program repair. However, it is crucial to recognize that existing repair benchmarks may have influenced LLM training data, potentially causing data leakage. To evaluate LLMs' realistic repair capabilities, (1) we introduce an extensive, non-crawled benchmark, referred to as TutorCode, comprising 1,239 C++ defect codes and associated information such as tutor guidance, solution description, failing test cases, and the corrected code. Our work assesses the repair performance of 12 LLMs on TutorCode, measuring repair correctness (TOP-5 and AVG-5) and patch precision (RPSR). (2) We then provide a comprehensive investigation into which types of extra information can help LLMs improve their performance in repairing defects. Among these types, tutor guidance was found to be the most effective information in enhancing LLM repair capabilities. To fully harness LLMs' conversational capabilities and the benefits of augmented information, (3) we introduce a novel conversational semi-automatic repair framework CREF assisting human tutor. It demonstrates a remarkable AVG-5 improvement of 17.2%-24.6% compared to the baseline, achieving an impressive AVG-5 of 76.6% when utilizing GPT-4. These results highlight the potential for enhancing LLMs' repair capabilities through interactions with tutors and historical conversations involving incorrect responses. The successful application of CREF in a real-world educational setting demonstrates its effectiveness in reducing tutors' workload and improving students' learning experience, while also showcasing its promise for facilitating other software engineering tasks, such as code review.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.03805v2">Cognitive Modeling with Scaffolded LLMs: A Case Study of Referential Expression Generation</a></div>
    <div class="paper-meta">
      📅 2024-07-08
      | 💬 11 pages, 3 figures, 2 algorithms, to appear at the ICML 2024 workshop on Large Language Models and Cognition
    </div>
    <details class="paper-abstract">
      To what extent can LLMs be used as part of a cognitive model of language generation? In this paper, we approach this question by exploring a neuro-symbolic implementation of an algorithmic cognitive model of referential expression generation by Dale & Reiter (1995). The symbolic task analysis implements the generation as an iterative procedure that scaffolds symbolic and gpt-3.5-turbo-based modules. We compare this implementation to an ablated model and a one-shot LLM-only baseline on the A3DS dataset (Tsvilodub & Franke, 2023). We find that our hybrid approach is cognitively plausible and performs well in complex contexts, while allowing for more open-ended modeling of language generation in a larger domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.06407v2">Can LLMs' Tuning Methods Work in Medical Multimodal Domain?</a></div>
    <div class="paper-meta">
      📅 2024-07-08
      | 💬 Accepted by MICCAI 2024
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) excel in world knowledge understanding, adapting them to specific subfields requires precise adjustments. Due to the model's vast scale, traditional global fine-tuning methods for large models can be computationally expensive and impact generalization. To address this challenge, a range of innovative Parameters-Efficient Fine-Tuning (PEFT) methods have emerged and achieved remarkable success in both LLMs and Large Vision-Language Models (LVLMs). In the medical domain, fine-tuning a medical Vision-Language Pretrained (VLP) model is essential for adapting it to specific tasks. Can the fine-tuning methods for large models be transferred to the medical field to enhance transfer learning efficiency? In this paper, we delve into the fine-tuning methods of LLMs and conduct extensive experiments to investigate the impact of fine-tuning methods for large models on the existing multimodal model in the medical domain from the training data level and the model structure level. We show the different impacts of fine-tuning methods for large models on medical VLMs and develop the most efficient ways to fine-tune medical VLP models. We hope this research can guide medical domain researchers in optimizing VLMs' training costs, fostering the broader application of VLMs in healthcare fields. The code and dataset have been released at https://github.com/TIMMY-CHAN/MILE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.00994v2">LLM Uncertainty Quantification through Directional Entailment Graph and Claim Level Response Augmentation</a></div>
    <div class="paper-meta">
      📅 2024-07-08
      | 💬 11 pages main content, 5 pages appendix
    </div>
    <details class="paper-abstract">
      The Large language models (LLMs) have showcased superior capabilities in sophisticated tasks across various domains, stemming from basic question-answer (QA), they are nowadays used as decision assistants or explainers for unfamiliar content. However, they are not always correct due to the data sparsity in specific domain corpus, or the model's hallucination problems. Given this, how much should we trust the responses from LLMs? This paper presents a novel way to evaluate the uncertainty that captures the directional instability, by constructing a directional graph from entailment probabilities, and we innovatively conduct Random Walk Laplacian given the asymmetric property of a constructed directed graph, then the uncertainty is aggregated by the derived eigenvalues from the Laplacian process. We also provide a way to incorporate the existing work's semantics uncertainty with our proposed layer. Besides, this paper identifies the vagueness issues in the raw response set and proposes an augmentation approach to mitigate such a problem, we conducted extensive empirical experiments and demonstrated the superiority of our proposed solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.01991v2">Fill in the Blank: Exploring and Enhancing LLM Capabilities for Backward Reasoning in Math Word Problems</a></div>
    <div class="paper-meta">
      📅 2024-07-08
      | 💬 10 pages, 4 figures
    </div>
    <details class="paper-abstract">
      While forward reasoning (i.e., find the answer given the question) has been explored extensively in recent literature, backward reasoning is relatively unexplored. We examine the backward reasoning capabilities of LLMs on Math Word Problems (MWPs): given a mathematical question and its answer, with some details omitted from the question, can LLMs effectively retrieve the missing information? On modifying three benchmark datasets for this task, to evaluate this task: GSM8k, SVAMP, and MultiArith, we find a significant drop in the accuracy of models on this task compared to forward reasoning across SOTA LLMs (GPT4, GPT3.5, PaLM-2, and LLaMa). Motivated by the fact backward reasoning can be seen as the ''inverse'' of forward reasoning, we propose variations of three different forward reasoning strategies to improve performance. Rephrase reformulates the given problem into a forward reasoning problem, PAL-Tools combines the idea of Program-Aided LLMs to produce a set of equations that can be solved by an external solver, and Check your Work exploits the availability of natural verifier of high accuracy in the forward direction, interleaving solving and verification steps. Finally, realizing that each of our base methods correctly solves a different set of problems, we propose a novel Bayesian formulation for creating an ensemble over the base methods to further boost the accuracy. Extensive experimentation demonstrates successive improvement in the performance of LLMs on the backward reasoning task, using our strategies, with our ensemble-based method resulting in significant performance gains compared to the SOTA forward reasoning strategies we adapt.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.05557v1">$R^2$-Guard: Robust Reasoning Enabled LLM Guardrail via Knowledge-Enhanced Logical Reasoning</a></div>
    <div class="paper-meta">
      📅 2024-07-08
    </div>
    <details class="paper-abstract">
      As LLMs become increasingly prevalent across various applications, it is critical to establish safety guardrails to moderate input/output content of LLMs. Existing guardrail models treat various safety categories independently and fail to explicitly capture the intercorrelations among them. This has led to limitations such as ineffectiveness due to inadequate training on long-tail data from correlated safety categories, susceptibility to jailbreaking attacks, and inflexibility regarding new safety categories. To address these limitations, we propose $R^2$-Guard, a robust reasoning enabled LLM guardrail via knowledge-enhanced logical reasoning. Specifically, $R^2$-Guard comprises two parts: data-driven category-specific learning and reasoning components. The data-driven guardrail models provide unsafety probabilities of moderated content on different safety categories. We then encode safety knowledge among different categories as first-order logical rules and embed them into a probabilistic graphic model (PGM) based reasoning component. The unsafety probabilities of different categories from data-driven guardrail models are sent to the reasoning component for final inference. We employ two types of PGMs: Markov logic networks (MLNs) and probabilistic circuits (PCs), and optimize PCs to achieve precision-efficiency balance via improved graph structure. To further perform stress tests for guardrail models, we employ a pairwise construction method to construct a new safety benchmark TwinSafety, which features principled categories. We demonstrate the effectiveness of $R^2$-Guard by comparisons with eight strong guardrail models on six safety benchmarks, and demonstrate the robustness of $R^2$-Guard against four SOTA jailbreaking attacks. $R^2$-Guard significantly surpasses SOTA method LlamaGuard by 30.2% on ToxicChat and by 59.5% against jailbreaking attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.15232v2">How Beginning Programmers and Code LLMs (Mis)read Each Other</a></div>
    <div class="paper-meta">
      📅 2024-07-07
      | 💬 Published in CHI 2024
    </div>
    <details class="paper-abstract">
      Generative AI models, specifically large language models (LLMs), have made strides towards the long-standing goal of text-to-code generation. This progress has invited numerous studies of user interaction. However, less is known about the struggles and strategies of non-experts, for whom each step of the text-to-code problem presents challenges: describing their intent in natural language, evaluating the correctness of generated code, and editing prompts when the generated code is incorrect. This paper presents a large-scale controlled study of how 120 beginning coders across three academic institutions approach writing and editing prompts. A novel experimental design allows us to target specific steps in the text-to-code process and reveals that beginners struggle with writing and editing prompts, even for problems at their skill level and when correctness is automatically determined. Our mixed-methods evaluation provides insight into student processes and perceptions with key implications for non-expert Code LLM use within and outside of education.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.05798v2">$\textbf{S}^2$IP-LLM: Semantic Space Informed Prompt Learning with LLM for Time Series Forecasting</a></div>
    <div class="paper-meta">
      📅 2024-07-07
    </div>
    <details class="paper-abstract">
      Recently, there has been a growing interest in leveraging pre-trained large language models (LLMs) for various time series applications. However, the semantic space of LLMs, established through the pre-training, is still underexplored and may help yield more distinctive and informative representations to facilitate time series forecasting. To this end, we propose Semantic Space Informed Prompt learning with LLM ($S^2$IP-LLM) to align the pre-trained semantic space with time series embeddings space and perform time series forecasting based on learned prompts from the joint space. We first design a tokenization module tailored for cross-modality alignment, which explicitly concatenates patches of decomposed time series components to create embeddings that effectively encode the temporal dynamics. Next, we leverage the pre-trained word token embeddings to derive semantic anchors and align selected anchors with time series embeddings by maximizing the cosine similarity in the joint space. This way, $S^2$IP-LLM can retrieve relevant semantic anchors as prompts to provide strong indicators (context) for time series that exhibit different temporal dynamics. With thorough empirical studies on multiple benchmark datasets, we demonstrate that the proposed $S^2$IP-LLM can achieve superior forecasting performance over state-of-the-art baselines. Furthermore, our ablation studies and visualizations verify the necessity of prompt learning informed by semantic space.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.05437v1">Enhancing Computer Programming Education with LLMs: A Study on Effective Prompt Engineering for Python Code Generation</a></div>
    <div class="paper-meta">
      📅 2024-07-07
      | 💬 18 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) and prompt engineering hold significant potential for advancing computer programming education through personalized instruction. This paper explores this potential by investigating three critical research questions: the systematic categorization of prompt engineering strategies tailored to diverse educational needs, the empowerment of LLMs to solve complex problems beyond their inherent capabilities, and the establishment of a robust framework for evaluating and implementing these strategies. Our methodology involves categorizing programming questions based on educational requirements, applying various prompt engineering strategies, and assessing the effectiveness of LLM-generated responses. Experiments with GPT-4, GPT-4o, Llama3-8b, and Mixtral-8x7b models on datasets such as LeetCode and USACO reveal that GPT-4o consistently outperforms others, particularly with the "multi-step" prompt strategy. The results show that tailored prompt strategies significantly enhance LLM performance, with specific strategies recommended for foundational learning, competition preparation, and advanced problem-solving. This study underscores the crucial role of prompt engineering in maximizing the educational benefits of LLMs. By systematically categorizing and testing these strategies, we provide a comprehensive framework for both educators and students to optimize LLM-based learning experiences. Future research should focus on refining these strategies and addressing current LLM limitations to further enhance educational outcomes in computer programming instruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.05347v1">A Queueing Theoretic Perspective on Low-Latency LLM Inference with Variable Token Length</a></div>
    <div class="paper-meta">
      📅 2024-07-07
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) propel the prosperity of interactive AI applications showcased by ChatGPT that demand timely response of inference services. However, LLM inference is computation intensive and memory intensive, and improper parameter configuration at LLM platforms may exacerbate the inference time. In this paper, we analyze the impact of LLM output token distribution on the inference queueing delay, where the max-token clipping and the batched inference are considered. By formulating an M/G/1 model, we observe that enforcing a maximum output token limit on a very small fraction of inference requests can significantly reduce the queueing delay, and our model facilitates the selection of the optimal limit. For the batch inference, we model the service process as a bulk queue in which the batch processing time is affected by the batch size and the maximum token size inside this batch jointly. The queueing delays of the batching of all buffered requests (dynamic batching), the batching of constant number of requests (fixed batching), and the batching without intra-batch waiting (elastic batching) are derived. Experimental results show that our mathematical models coincide with the event-driven simulations well.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.05271v1">Beyond Binary Gender Labels: Revealing Gender Biases in LLMs through Gender-Neutral Name Predictions</a></div>
    <div class="paper-meta">
      📅 2024-07-07
      | 💬 Accepted at ACL 2024, GeBNLP Workshop
    </div>
    <details class="paper-abstract">
      Name-based gender prediction has traditionally categorized individuals as either female or male based on their names, using a binary classification system. That binary approach can be problematic in the cases of gender-neutral names that do not align with any one gender, among other reasons. Relying solely on binary gender categories without recognizing gender-neutral names can reduce the inclusiveness of gender prediction tasks. We introduce an additional gender category, i.e., "neutral", to study and address potential gender biases in Large Language Models (LLMs). We evaluate the performance of several foundational and large language models in predicting gender based on first names only. Additionally, we investigate the impact of adding birth years to enhance the accuracy of gender prediction, accounting for shifting associations between names and genders over time. Our findings indicate that most LLMs identify male and female names with high accuracy (over 80%) but struggle with gender-neutral names (under 40%), and the accuracy of gender prediction is higher for English-based first names than non-English names. The experimental results show that incorporating the birth year does not improve the overall accuracy of gender prediction, especially for names with evolving gender associations. We recommend using caution when applying LLMs for gender identification in downstream tasks, particularly when dealing with non-binary gender labels.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.05202v1">Harnessing the Power of LLMs: Automating Unit Test Generation for High-Performance Computing</a></div>
    <div class="paper-meta">
      📅 2024-07-06
    </div>
    <details class="paper-abstract">
      Unit testing is crucial in software engineering for ensuring quality. However, it's not widely used in parallel and high-performance computing software, particularly scientific applications, due to their smaller, diverse user base and complex logic. These factors make unit testing challenging and expensive, as it requires specialized knowledge and existing automated tools are often ineffective. To address this, we propose an automated method for generating unit tests for such software, considering their unique features like complex logic and parallel processing. Recently, large language models (LLMs) have shown promise in coding and testing. We explored the capabilities of Davinci (text-davinci-002) and ChatGPT (gpt-3.5-turbo) in creating unit tests for C++ parallel programs. Our results show that LLMs can generate mostly correct and comprehensive unit tests, although they have some limitations, such as repetitive assertions and blank test cases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.05194v1">LLMCloudHunter: Harnessing LLMs for Automated Extraction of Detection Rules from Cloud-Based CTI</a></div>
    <div class="paper-meta">
      📅 2024-07-06
    </div>
    <details class="paper-abstract">
      As the number and sophistication of cyber attacks have increased, threat hunting has become a critical aspect of active security, enabling proactive detection and mitigation of threats before they cause significant harm. Open-source cyber threat intelligence (OS-CTI) is a valuable resource for threat hunters, however, it often comes in unstructured formats that require further manual analysis. Previous studies aimed at automating OSCTI analysis are limited since (1) they failed to provide actionable outputs, (2) they did not take advantage of images present in OSCTI sources, and (3) they focused on on-premises environments, overlooking the growing importance of cloud environments. To address these gaps, we propose LLMCloudHunter, a novel framework that leverages large language models (LLMs) to automatically generate generic-signature detection rule candidates from textual and visual OSCTI data. We evaluated the quality of the rules generated by the proposed framework using 12 annotated real-world cloud threat reports. The results show that our framework achieved a precision of 92% and recall of 98% for the task of accurately extracting API calls made by the threat actor and a precision of 99% with a recall of 98% for IoCs. Additionally, 99.18% of the generated detection rule candidates were successfully compiled and converted into Splunk queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.02056v3">Multitask-based Evaluation of Open-Source LLM on Software Vulnerability</a></div>
    <div class="paper-meta">
      📅 2024-07-06
    </div>
    <details class="paper-abstract">
      This paper proposes a pipeline for quantitatively evaluating interactive Large Language Models (LLMs) using publicly available datasets. We carry out an extensive technical evaluation of LLMs using Big-Vul covering four different common software vulnerability tasks. This evaluation assesses the multi-tasking capabilities of LLMs based on this dataset. We find that the existing state-of-the-art approaches and pre-trained Language Models (LMs) are generally superior to LLMs in software vulnerability detection. However, in software vulnerability assessment and location, certain LLMs (e.g., CodeLlama and WizardCoder) have demonstrated superior performance compared to pre-trained LMs, and providing more contextual information can enhance the vulnerability assessment capabilities of LLMs. Moreover, LLMs exhibit strong vulnerability description capabilities, but their tendency to produce excessive output significantly weakens their performance compared to pre-trained LMs. Overall, though LLMs perform well in some aspects, they still need improvement in understanding the subtle differences in code vulnerabilities and the ability to describe vulnerabilities to fully realize their potential. Our evaluation pipeline provides valuable insights into the capabilities of LLMs in handling software vulnerabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.05088v1">Leveraging Task-Specific Knowledge from LLM for Semi-Supervised 3D Medical Image Segmentation</a></div>
    <div class="paper-meta">
      📅 2024-07-06
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Traditional supervised 3D medical image segmentation models need voxel-level annotations, which require huge human effort, time, and cost. Semi-supervised learning (SSL) addresses this limitation of supervised learning by facilitating learning with a limited annotated and larger amount of unannotated training samples. However, state-of-the-art SSL models still struggle to fully exploit the potential of learning from unannotated samples. To facilitate effective learning from unannotated data, we introduce LLM-SegNet, which exploits a large language model (LLM) to integrate task-specific knowledge into our co-training framework. This knowledge aids the model in comprehensively understanding the features of the region of interest (ROI), ultimately leading to more efficient segmentation. Additionally, to further reduce erroneous segmentation, we propose a Unified Segmentation loss function. This loss function reduces erroneous segmentation by not only prioritizing regions where the model is confident in predicting between foreground or background pixels but also effectively addressing areas where the model lacks high confidence in predictions. Experiments on publicly available Left Atrium, Pancreas-CT, and Brats-19 datasets demonstrate the superior performance of LLM-SegNet compared to the state-of-the-art. Furthermore, we conducted several ablation studies to demonstrate the effectiveness of various modules and loss functions leveraged by LLM-SegNet.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.05040v1">Code Less, Align More: Efficient LLM Fine-tuning for Code Generation with Data Pruning</a></div>
    <div class="paper-meta">
      📅 2024-07-06
    </div>
    <details class="paper-abstract">
      Recent work targeting large language models (LLMs) for code generation demonstrated that increasing the amount of training data through synthetic code generation often leads to exceptional performance. In this paper we explore data pruning methods aimed at enhancing the efficiency of model training specifically for code LLMs. We present techniques that integrate various clustering and pruning metrics to selectively reduce training data without compromising the accuracy and functionality of the generated code. We observe significant redundancies in synthetic training data generation, where our experiments demonstrate that benchmark performance can be largely preserved by training on only 10% of the data. Moreover, we observe consistent improvements in benchmark results through moderate pruning of the training data. Our experiments show that these pruning strategies not only reduce the computational resources needed but also enhance the overall quality code generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.04997v1">Achieving Tool Calling Functionality in LLMs Using Only Prompt Engineering Without Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2024-07-06
      | 💬 5 pages, 2 figures,review comments welcome
    </div>
    <details class="paper-abstract">
      Currently, the vast majority of locally deployed open-source large language models (LLMs) and some commercial model interfaces do not support stable tool calling functionality. The existing solution involves fine-tuning LLMs, which results in significant time and computational resource consumption. This paper proposes a method that enables LLMs to achieve stable tool calling capabilities using only prompt engineering and some ingenious code design. We conducted experiments on multiple LLMs that lack tool calling capabilities across various tool calling tasks, achieving a success rate of 100%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.04981v1">TRACE: TRansformer-based Attribution using Contrastive Embeddings in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-06
    </div>
    <details class="paper-abstract">
      The rapid evolution of large language models (LLMs) represents a substantial leap forward in natural language understanding and generation. However, alongside these advancements come significant challenges related to the accountability and transparency of LLM responses. Reliable source attribution is essential to adhering to stringent legal and regulatory standards, including those set forth by the General Data Protection Regulation. Despite the well-established methods in source attribution within the computer vision domain, the application of robust attribution frameworks to natural language processing remains underexplored. To bridge this gap, we propose a novel and versatile TRansformer-based Attribution framework using Contrastive Embeddings called TRACE that, in particular, exploits contrastive learning for source attribution. We perform an extensive empirical evaluation to demonstrate the performance and efficiency of TRACE in various settings and show that TRACE significantly improves the ability to attribute sources accurately, making it a valuable tool for enhancing the reliability and trustworthiness of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.04885v1">Automating Venture Capital: Founder assessment using LLM-powered segmentation, feature engineering and automated labeling techniques</a></div>
    <div class="paper-meta">
      📅 2024-07-05
      | 💬 For the relevant code, see https://github.com/velapartners/moneyball-LLM-based-founder-features.git
    </div>
    <details class="paper-abstract">
      This study explores the application of large language models (LLMs) in venture capital (VC) decision-making, focusing on predicting startup success based on founder characteristics. We utilize LLM prompting techniques, like chain-of-thought, to generate features from limited data, then extract insights through statistics and machine learning. Our results reveal potential relationships between certain founder characteristics and success, as well as demonstrate the effectiveness of these characteristics in prediction. This framework for integrating ML techniques and LLMs has vast potential for improving startup success prediction, with important implications for VC firms seeking to optimize their investment strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.04855v1">Towards Enhancing Coherence in Extractive Summarization: Dataset and Experiments with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-05
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Extractive summarization plays a pivotal role in natural language processing due to its wide-range applications in summarizing diverse content efficiently, while also being faithful to the original content. Despite significant advancement achieved in extractive summarization by Large Language Models (LLMs), these summaries frequently exhibit incoherence. An important aspect of the coherent summary is its readability for intended users. Although there have been many datasets and benchmarks proposed for creating coherent extractive summaries, none of them currently incorporate user intent to improve coherence in extractive summarization. Motivated by this, we propose a systematically created human-annotated dataset consisting of coherent summaries for five publicly available datasets and natural language user feedback, offering valuable insights into how to improve coherence in extractive summaries. We utilize this dataset for aligning LLMs through supervised fine-tuning with natural language human feedback to enhance the coherence of their generated summaries. Preliminary experiments with Falcon-40B and Llama-2-13B show significant performance improvements (~10% Rouge-L) in terms of producing coherent summaries. We further utilize human feedback to benchmark results over instruction-tuned models such as FLAN-T5 which resulted in several interesting findings. Data and source code are available at https://github.com/Mihir3009/Extract-AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.04694v1">Me, Myself, and AI: The Situational Awareness Dataset (SAD) for LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-05
      | 💬 11 page main body, 98 page appendix, 58 figures
    </div>
    <details class="paper-abstract">
      AI assistants such as ChatGPT are trained to respond to users by saying, "I am a large language model". This raises questions. Do such models know that they are LLMs and reliably act on this knowledge? Are they aware of their current circumstances, such as being deployed to the public? We refer to a model's knowledge of itself and its circumstances as situational awareness. To quantify situational awareness in LLMs, we introduce a range of behavioral tests, based on question answering and instruction following. These tests form the $\textbf{Situational Awareness Dataset (SAD)}$, a benchmark comprising 7 task categories and over 13,000 questions. The benchmark tests numerous abilities, including the capacity of LLMs to (i) recognize their own generated text, (ii) predict their own behavior, (iii) determine whether a prompt is from internal evaluation or real-world deployment, and (iv) follow instructions that depend on self-knowledge. We evaluate 16 LLMs on SAD, including both base (pretrained) and chat models. While all models perform better than chance, even the highest-scoring model (Claude 3 Opus) is far from a human baseline on certain tasks. We also observe that performance on SAD is only partially predicted by metrics of general knowledge (e.g. MMLU). Chat models, which are finetuned to serve as AI assistants, outperform their corresponding base models on SAD but not on general knowledge tasks. The purpose of SAD is to facilitate scientific understanding of situational awareness in LLMs by breaking it down into quantitative abilities. Situational awareness is important because it enhances a model's capacity for autonomous planning and action. While this has potential benefits for automation, it also introduces novel risks related to AI safety and control. Code and latest results available at https://situational-awareness-dataset.org .
    </details>
</div>
