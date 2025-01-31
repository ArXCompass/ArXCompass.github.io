# llm - 2025_01

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- Part 2
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14866v3">PAPILLON: Efficient and Stealthy Fuzz Testing-Powered Jailbreaks for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-26
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have excelled in various tasks but are still vulnerable to jailbreaking attacks, where attackers create jailbreak prompts to mislead the model to produce harmful or offensive content. Current jailbreak methods either rely heavily on manually crafted templates, which pose challenges in scalability and adaptability, or struggle to generate semantically coherent prompts, making them easy to detect. Additionally, most existing approaches involve lengthy prompts, leading to higher query costs.In this paper, to remedy these challenges, we introduce a novel jailbreaking attack framework called PAPILLON, which is an automated, black-box jailbreaking attack framework that adapts the black-box fuzz testing approach with a series of customized designs. Instead of relying on manually crafted templates,PAPILLON starts with an empty seed pool, removing the need to search for any related jailbreaking templates. We also develop three novel question-dependent mutation strategies using an LLM helper to generate prompts that maintain semantic coherence while significantly reducing their length. Additionally, we implement a two-level judge module to accurately detect genuine successful jailbreaks. We evaluated PAPILLON on 7 representative LLMs and compared it with 5 state-of-the-art jailbreaking attack strategies. For proprietary LLM APIs, such as GPT-3.5 turbo, GPT-4, and Gemini-Pro, PAPILLONs achieves attack success rates of over 90%, 80%, and 74%, respectively, exceeding existing baselines by more than 60\%. Additionally, PAPILLON can maintain high semantic coherence while significantly reducing the length of jailbreak prompts. When targeting GPT-4, PAPILLON can achieve over 78% attack success rate even with 100 tokens. Moreover, PAPILLON demonstrates transferability and is robust to state-of-the-art defenses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.15599v1">Evaluating an LLM-Powered Chatbot for Cognitive Restructuring: Insights from Mental Health Professionals</a></div>
    <div class="paper-meta">
      📅 2025-01-26
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) promise to expand mental health interventions by emulating therapeutic techniques, potentially easing barriers to care. Yet there is a lack of real-world empirical evidence evaluating the strengths and limitations of LLM-enabled psychotherapy interventions. In this work, we evaluate an LLM-powered chatbot, designed via prompt engineering to deliver cognitive restructuring (CR), with 19 users. Mental health professionals then examined the resulting conversation logs to uncover potential benefits and pitfalls. Our findings indicate that an LLM-based CR approach has the capability to adhere to core CR protocols, prompt Socratic questioning, and provide empathetic validation. However, issues of power imbalances, advice-giving, misunderstood cues, and excessive positivity reveal deeper challenges, including the potential to erode therapeutic rapport and ethical concerns. We also discuss design implications for leveraging LLMs in psychotherapy and underscore the importance of expert oversight to mitigate these concerns, which are critical steps toward safer, more effective AI-assisted interventions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.11380v3">Evaluating LLMs for Quotation Attribution in Literary Texts: A Case Study of LLaMa3</a></div>
    <div class="paper-meta">
      📅 2025-01-26
      | 💬 NAACL 2025 Main Conference -- short paper
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown promising results in a variety of literary tasks, often using complex memorized details of narration and fictional characters. In this work, we evaluate the ability of Llama-3 at attributing utterances of direct-speech to their speaker in novels. The LLM shows impressive results on a corpus of 28 novels, surpassing published results with ChatGPT and encoder-based baselines by a large margin. We then validate these results by assessing the impact of book memorization and annotation contamination. We found that these types of memorization do not explain the large performance gain, making Llama-3 the new state-of-the-art for quotation attribution in English literature. We release publicly our code and data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16153v3">Pangea: A Fully Open Multilingual Multimodal LLM for 39 Languages</a></div>
    <div class="paper-meta">
      📅 2025-01-26
      | 💬 54 pages, 27 figures
    </div>
    <details class="paper-abstract">
      Despite recent advances in multimodal large language models (MLLMs), their development has predominantly focused on English- and western-centric datasets and tasks, leaving most of the world's languages and diverse cultural contexts underrepresented. This paper introduces Pangea, a multilingual multimodal LLM trained on PangeaIns, a diverse 6M instruction dataset spanning 39 languages. PangeaIns features: 1) high-quality English instructions, 2) carefully machine-translated instructions, and 3) culturally relevant multimodal tasks to ensure cross-cultural coverage. To rigorously assess models' capabilities, we introduce PangeaBench, a holistic evaluation suite encompassing 14 datasets covering 47 languages. Results show that Pangea significantly outperforms existing open-source models in multilingual settings and diverse cultural contexts. Ablation studies further reveal the importance of English data proportions, language popularity, and the number of multimodal training samples on overall performance. We fully open-source our data, code, and trained checkpoints, to facilitate the development of inclusive and robust multilingual MLLMs, promoting equity and accessibility across a broader linguistic and cultural spectrum.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.03101v2">KnowAgent: Knowledge-Augmented Planning for LLM-Based Agents</a></div>
    <div class="paper-meta">
      📅 2025-01-26
      | 💬 NAACL 2025 Findings. Project page: https://zjunlp.github.io/project/KnowAgent/ Code: https://github.com/zjunlp/KnowAgent
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated great potential in complex reasoning tasks, yet they fall short when tackling more sophisticated challenges, especially when interacting with environments through generating executable actions. This inadequacy primarily stems from the lack of built-in action knowledge in language agents, which fails to effectively guide the planning trajectories during task solving and results in planning hallucination. To address this issue, we introduce KnowAgent, a novel approach designed to enhance the planning capabilities of LLMs by incorporating explicit action knowledge. Specifically, KnowAgent employs an action knowledge base and a knowledgeable self-learning strategy to constrain the action path during planning, enabling more reasonable trajectory synthesis, and thereby enhancing the planning performance of language agents. Experimental results on HotpotQA and ALFWorld based on various backbone models demonstrate that KnowAgent can achieve comparable or superior performance to existing baselines. Further analysis indicates the effectiveness of KnowAgent in terms of planning hallucinations mitigation. Code is available in https://github.com/zjunlp/KnowAgent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.09165v2">Breaking Barriers or Building Dependency? Exploring Team-LLM Collaboration in AI-infused Classroom Debate</a></div>
    <div class="paper-meta">
      📅 2025-01-26
    </div>
    <details class="paper-abstract">
      Classroom debates are a unique form of collaborative learning characterized by fast-paced, high-intensity interactions that foster critical thinking and teamwork. Despite the recognized importance of debates, the role of AI tools, particularly LLM-based systems, in supporting this dynamic learning environment has been under-explored in HCI. This study addresses this opportunity by investigating the integration of LLM-based AI into real-time classroom debates. Over four weeks, 22 students in a Design History course participated in three rounds of debates with support from ChatGPT. The findings reveal how learners prompted the AI to offer insights, collaboratively processed its outputs, and divided labor in team-AI interactions. The study also surfaces key advantages of AI usage, reducing social anxiety, breaking communication barriers, and providing scaffolding for novices, alongside risks, such as information overload and cognitive dependency, which could limit learners' autonomy. We thereby discuss a set of nuanced implications for future HCI exploration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.15463v1">Mind the Value-Action Gap: Do LLMs Act in Alignment with Their Values?</a></div>
    <div class="paper-meta">
      📅 2025-01-26
    </div>
    <details class="paper-abstract">
      Existing research primarily evaluates the values of LLMs by examining their stated inclinations towards specific values. However, the "Value-Action Gap," a phenomenon rooted in environmental and social psychology, reveals discrepancies between individuals' stated values and their actions in real-world contexts. To what extent do LLMs exhibit a similar gap between their stated values and their actions informed by those values? This study introduces ValueActionLens, an evaluation framework to assess the alignment between LLMs' stated values and their value-informed actions. The framework encompasses the generation of a dataset comprising 14.8k value-informed actions across twelve cultures and eleven social topics, and two tasks to evaluate how well LLMs' stated value inclinations and value-informed actions align across three different alignment measures. Extensive experiments reveal that the alignment between LLMs' stated values and actions is sub-optimal, varying significantly across scenarios and models. Analysis of misaligned results identifies potential harms from certain value-action gaps. To predict the value-action gaps, we also uncover that leveraging reasoned explanations improves performance. These findings underscore the risks of relying solely on the LLMs' stated values to predict their behaviors and emphasize the importance of context-aware evaluations of LLM values and value-action gaps.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.15427v1">OpenCharacter: Training Customizable Role-Playing LLMs with Large-Scale Synthetic Personas</a></div>
    <div class="paper-meta">
      📅 2025-01-26
    </div>
    <details class="paper-abstract">
      Customizable role-playing in large language models (LLMs), also known as character generalization, is gaining increasing attention for its versatility and cost-efficiency in developing and deploying role-playing dialogue agents. This study explores a large-scale data synthesis approach to equip LLMs with character generalization capabilities. We begin by synthesizing large-scale character profiles using personas from Persona Hub and then explore two strategies: response rewriting and response generation, to create character-aligned instructional responses. To validate the effectiveness of our synthetic instruction tuning data for character generalization, we perform supervised fine-tuning (SFT) using the LLaMA-3 8B model. Our best-performing model strengthens the original LLaMA-3 8B Instruct model and achieves performance comparable to GPT-4o models on role-playing dialogue. We release our synthetic characters and instruction-tuning dialogues to support public research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.00326v7">Agent-OM: Leveraging LLM Agents for Ontology Matching</a></div>
    <div class="paper-meta">
      📅 2025-01-26
      | 💬 19 pages, 12 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Ontology matching (OM) enables semantic interoperability between different ontologies and resolves their conceptual heterogeneity by aligning related entities. OM systems currently have two prevailing design paradigms: conventional knowledge-based expert systems and newer machine learning-based predictive systems. While large language models (LLMs) and LLM agents have revolutionised data engineering and have been applied creatively in many domains, their potential for OM remains underexplored. This study introduces a novel agent-powered LLM-based design paradigm for OM systems. With consideration of several specific challenges in leveraging LLM agents for OM, we propose a generic framework, namely Agent-OM (Agent for Ontology Matching), consisting of two Siamese agents for retrieval and matching, with a set of OM tools. Our framework is implemented in a proof-of-concept system. Evaluations of three Ontology Alignment Evaluation Initiative (OAEI) tracks over state-of-the-art OM systems show that our system can achieve results very close to the long-standing best performance on simple OM tasks and can significantly improve the performance on complex and few-shot OM tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.10711v2">How Should I Build A Benchmark? Revisiting Code-Related Benchmarks For LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-26
      | 💬 42 pages
    </div>
    <details class="paper-abstract">
      Various benchmarks have been proposed to assess the performance of large language models (LLMs) in different coding scenarios. We refer to them as code-related benchmarks. However, there are no systematic guidelines by which such a benchmark should be developed to ensure its quality, reliability, and reproducibility. We propose How2Bench, which is comprised of a 55- 55-criteria checklist as a set of guidelines to govern the development of code-related benchmarks comprehensively. Using HOW2BENCH, we profiled 274 benchmarks released within the past decade and found concerning issues. Nearly 70% of the benchmarks did not take measures for data quality assurance; over 10% did not even open source or only partially open source. Many highly cited benchmarks have loopholes, including duplicated samples, incorrect reference codes/tests/prompts, and unremoved sensitive/confidential information. Finally, we conducted a human study involving 49 participants, which revealed significant gaps in awareness of the importance of data quality, reproducibility, and transparency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.15351v1">Fairness in LLM-Generated Surveys</a></div>
    <div class="paper-meta">
      📅 2025-01-25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel in text generation and understanding, especially in simulating socio-political and economic patterns, serving as an alternative to traditional surveys. However, their global applicability remains questionable due to unexplored biases across socio-demographic and geographic contexts. This study examines how LLMs perform across diverse populations by analyzing public surveys from Chile and the United States, focusing on predictive accuracy and fairness metrics. The results show performance disparities, with LLM consistently outperforming on U.S. datasets. This bias originates from the U.S.-centric training data, remaining evident after accounting for socio-demographic differences. In the U.S., political identity and race significantly influence prediction accuracy, while in Chile, gender, education, and religious affiliation play more pronounced roles. Our study presents a novel framework for measuring socio-demographic biases in LLMs, offering a path toward ensuring fairer and more equitable model performance across diverse socio-cultural contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.16325v3">SLoPe: Double-Pruned Sparse Plus Lazy Low-Rank Adapter Pretraining of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-25
      | 💬 Published at ICLR2025
    </div>
    <details class="paper-abstract">
      We propose SLoPe, a Double-Pruned Sparse Plus Lazy Low-rank Adapter Pretraining method for LLMs that improves the accuracy of sparse LLMs while accelerating their pretraining and inference and reducing their memory footprint. Sparse pretraining of LLMs reduces the accuracy of the model, to overcome this, prior work uses dense models during fine-tuning. SLoPe improves the accuracy of sparsely pretrained models by adding low-rank adapters in the final 1% iterations of pretraining without adding significant overheads to the model pretraining and inference. In addition, SLoPe uses a double-pruned backward pass formulation that prunes the transposed weight matrix using N:M sparsity structures to enable an accelerated sparse backward pass. SLoPe accelerates the training and inference of models with billions of parameters up to $1.25\times$ and $1.54\times$ respectively (OPT-33B and OPT-66B) while reducing their memory usage by up to $0.63\times$ and $0.61\times$ for training and inference respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.15310v1">The Multicultural Medical Assistant: Can LLMs Improve Medical ASR Errors Across Borders?</a></div>
    <div class="paper-meta">
      📅 2025-01-25
      | 💬 15 pages, 8 figures
    </div>
    <details class="paper-abstract">
      The global adoption of Large Language Models (LLMs) in healthcare shows promise to enhance clinical workflows and improve patient outcomes. However, Automatic Speech Recognition (ASR) errors in critical medical terms remain a significant challenge. These errors can compromise patient care and safety if not detected. This study investigates the prevalence and impact of ASR errors in medical transcription in Nigeria, the United Kingdom, and the United States. By evaluating raw and LLM-corrected transcriptions of accented English in these regions, we assess the potential and limitations of LLMs to address challenges related to accents and medical terminology in ASR. Our findings highlight significant disparities in ASR accuracy across regions and identify specific conditions under which LLM corrections are most effective.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13116v2">Learning to Summarize from LLM-generated Feedback</a></div>
    <div class="paper-meta">
      📅 2025-01-25
      | 💬 Accepted at NAACL 2025 (main, long)
    </div>
    <details class="paper-abstract">
      Developing effective text summarizers remains a challenge due to issues like hallucinations, key information omissions, and verbosity in LLM-generated summaries. This work explores using LLM-generated feedback to improve summary quality by aligning the summaries with human preferences for faithfulness, completeness, and conciseness. We introduce FeedSum, a large-scale dataset containing multi-dimensional LLM feedback on summaries of varying quality across diverse domains. Our experiments show how feedback quality, dimensionality, and granularity influence preference learning, revealing that high-quality, multi-dimensional, fine-grained feedback significantly improves summary generation. We also compare two methods for using this feedback: supervised fine-tuning and direct preference optimization. Finally, we introduce SummLlama3-8b, a model that outperforms the nearly 10x larger Llama3-70b-instruct in generating human-preferred summaries, demonstrating that smaller models can achieve superior performance with appropriate training. The full dataset and SummLlama3-8B model are available at https://huggingface.co/datasets/DISLab/FeedSum and https://huggingface.co/DISLab/SummLlama3-8B.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.10282v2">Computational Protein Science in the Era of Large Language Models (LLMs)</a></div>
    <div class="paper-meta">
      📅 2025-01-25
    </div>
    <details class="paper-abstract">
      Considering the significance of proteins, computational protein science has always been a critical scientific field, dedicated to revealing knowledge and developing applications within the protein sequence-structure-function paradigm. In the last few decades, Artificial Intelligence (AI) has made significant impacts in computational protein science, leading to notable successes in specific protein modeling tasks. However, those previous AI models still meet limitations, such as the difficulty in comprehending the semantics of protein sequences, and the inability to generalize across a wide range of protein modeling tasks. Recently, LLMs have emerged as a milestone in AI due to their unprecedented language processing & generalization capability. They can promote comprehensive progress in fields rather than solving individual tasks. As a result, researchers have actively introduced LLM techniques in computational protein science, developing protein Language Models (pLMs) that skillfully grasp the foundational knowledge of proteins and can be effectively generalized to solve a diversity of sequence-structure-function reasoning problems. While witnessing prosperous developments, it's necessary to present a systematic overview of computational protein science empowered by LLM techniques. First, we summarize existing pLMs into categories based on their mastered protein knowledge, i.e., underlying sequence patterns, explicit structural and functional information, and external scientific languages. Second, we introduce the utilization and adaptation of pLMs, highlighting their remarkable achievements in promoting protein structure prediction, protein function prediction, and protein design studies. Then, we describe the practical application of pLMs in antibody design, enzyme design, and drug discovery. Finally, we specifically discuss the promising future directions in this fast-growing field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.17183v1">LLM Evaluation Based on Aerospace Manufacturing Expertise: Automated Generation and Multi-Model Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-01-25
      | 💬 conference paper
    </div>
    <details class="paper-abstract">
      Aerospace manufacturing demands exceptionally high precision in technical parameters. The remarkable performance of Large Language Models (LLMs), such as GPT-4 and QWen, in Natural Language Processing has sparked industry interest in their application to tasks including process design, material selection, and tool information retrieval. However, LLMs are prone to generating "hallucinations" in specialized domains, producing inaccurate or false information that poses significant risks to the quality of aerospace products and flight safety. This paper introduces a set of evaluation metrics tailored for LLMs in aerospace manufacturing, aiming to assess their accuracy by analyzing their performance in answering questions grounded in professional knowledge. Firstly, key information is extracted through in-depth textual analysis of classic aerospace manufacturing textbooks and guidelines. Subsequently, utilizing LLM generation techniques, we meticulously construct multiple-choice questions with multiple correct answers of varying difficulty. Following this, different LLM models are employed to answer these questions, and their accuracy is recorded. Experimental results demonstrate that the capabilities of LLMs in aerospace professional knowledge are in urgent need of improvement. This study provides a theoretical foundation and practical guidance for the application of LLMs in aerospace manufacturing, addressing a critical gap in the field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.15134v1">BitsAI-CR: Automated Code Review via LLM in Practice</a></div>
    <div class="paper-meta">
      📅 2025-01-25
    </div>
    <details class="paper-abstract">
      Code review remains a critical yet resource-intensive process in software development, particularly challenging in large-scale industrial environments. While Large Language Models (LLMs) show promise for automating code review, existing solutions face significant limitations in precision and practicality. This paper presents BitsAI-CR, an innovative framework that enhances code review through a two-stage approach combining RuleChecker for initial issue detection and ReviewFilter for precision verification. The system is built upon a comprehensive taxonomy of review rules and implements a data flywheel mechanism that enables continuous performance improvement through structured feedback and evaluation metrics. Our approach introduces an Outdated Rate metric that can reflect developers' actual adoption of review comments, enabling automated evaluation and systematic optimization at scale. Empirical evaluation demonstrates BitsAI-CR's effectiveness, achieving 75.0% precision in review comment generation. For the Go language which has predominant usage at ByteDance, we maintain an Outdated Rate of 26.7%. The system has been successfully deployed at ByteDance, serving over 12,000 Weekly Active Users (WAU). Our work provides valuable insights into the practical application of automated code review and offers a blueprint for organizations seeking to implement automated code reviews at scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.15108v1">Knowledge Hierarchy Guided Biological-Medical Dataset Distillation for Domain LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-01-25
      | 💬 16 pages, accepted by DASFAA 2025
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) in biological-medical applications has highlighted a gap between their potential and the limited scale and often low quality of available open-source annotated textual datasets. In addition, the inherent complexity of the biomedical knowledge hierarchy significantly hampers efforts to bridge this gap.Can LLMs themselves play a pivotal role in overcoming this limitation? Motivated by this question, we investigate this challenge in the present study.We propose a framework that automates the distillation of high-quality textual training data from the extensive scientific literature. Our approach self-evaluates and generates questions that are more closely aligned with the biomedical domain, guided by the biomedical knowledge hierarchy through medical subject headings (MeSH). This comprehensive framework establishes an automated workflow, thereby eliminating the need for manual intervention. Furthermore, we conducted comprehensive experiments to evaluate the impact of our framework-generated data on downstream language models of varying sizes. Our approach substantially improves question-answering tasks compared to pre-trained models from the life sciences domain and powerful close-source models represented by GPT-4. Notably, the generated AI-Ready dataset enabled the Llama3-70B base model to outperform GPT-4 using MedPrompt with multiple times the number of parameters. Detailed case studies and ablation experiments underscore the significance of each component within our framework
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.15087v1">PatchRec: Multi-Grained Patching for Efficient LLM-based Sequential Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-01-25
    </div>
    <details class="paper-abstract">
      Large Language Models for sequential recommendation (LLM4SR), which transform user-item interactions into language modeling, have shown promising results. However, due to the limitations of context window size and the computational costs associated with Large Language Models (LLMs), current approaches primarily truncate user history by only considering the textual information of items from the most recent interactions in the input prompt. This truncation fails to fully capture the long-term behavioral patterns of users. To address this, we propose a multi-grained patching framework -- PatchRec. It compresses the textual tokens of an item title into a compact item patch, and further compresses multiple item patches into a denser session patch, with earlier interactions being compressed to a greater degree. The framework consists of two stages: (1) Patch Pre-training, which familiarizes LLMs with item-level compression patterns, and (2) Patch Fine-tuning, which teaches LLMs to model sequences at multiple granularities. Through this simple yet effective approach, empirical results demonstrate that PatchRec outperforms existing methods, achieving significant performance gains with fewer tokens fed to the LLM. Specifically, PatchRec shows up to a 32% improvement in HR@20 on the Goodreads dataset over uncompressed baseline, while using only 7% of the tokens. This multi-grained sequence modeling paradigm, with an adjustable compression ratio, enables LLMs to be efficiently deployed in real-world recommendation systems that handle extremely long user behavior sequences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.15067v1">CG-RAG: Research Question Answering by Citation Graph Retrieval-Augmented LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-25
      | 💬 10 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Research question answering requires accurate retrieval and contextual understanding of scientific literature. However, current Retrieval-Augmented Generation (RAG) methods often struggle to balance complex document relationships with precise information retrieval. In this paper, we introduce Contextualized Graph Retrieval-Augmented Generation (CG-RAG), a novel framework that integrates sparse and dense retrieval signals within graph structures to enhance retrieval efficiency and subsequently improve generation quality for research question answering. First, we propose a contextual graph representation for citation graphs, effectively capturing both explicit and implicit connections within and across documents. Next, we introduce Lexical-Semantic Graph Retrieval (LeSeGR), which seamlessly integrates sparse and dense retrieval signals with graph encoding. It bridges the gap between lexical precision and semantic understanding in citation graph retrieval, demonstrating generalizability to existing graph retrieval and hybrid retrieval methods. Finally, we present a context-aware generation strategy that utilizes the retrieved graph-structured information to generate precise and contextually enriched responses using large language models (LLMs). Extensive experiments on research question answering benchmarks across multiple domains demonstrate that our CG-RAG framework significantly outperforms RAG methods combined with various state-of-the-art retrieval approaches, delivering superior retrieval accuracy and generation quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.05002v4">An Empirical Study on Challenges for LLM Application Developers</a></div>
    <div class="paper-meta">
      📅 2025-01-25
      | 💬 37 pages, 19 figures
    </div>
    <details class="paper-abstract">
      In recent years, large language models (LLMs) have seen rapid advancements, significantly impacting various fields such as computer vision, natural language processing, and software engineering. These LLMs, exemplified by OpenAI's ChatGPT, have revolutionized the way we approach language understanding and generation tasks. However, in contrast to traditional software development practices, LLM development introduces new challenges for AI developers in design, implementation, and deployment. These challenges span different areas (such as prompts, APIs, and plugins), requiring developers to navigate unique methodologies and considerations specific to LLM application development. Despite the profound influence of LLMs, to the best of our knowledge, these challenges have not been thoroughly investigated in previous empirical studies. To fill this gap, we present the first comprehensive study on understanding the challenges faced by LLM developers. Specifically, we crawl and analyze 29,057 relevant questions from a popular OpenAI developer forum. We first examine their popularity and difficulty. After manually analyzing 2,364 sampled questions, we construct a taxonomy of challenges faced by LLM developers. Based on this taxonomy, we summarize a set of findings and actionable implications for LLM-related stakeholders, including developers and providers (especially the OpenAI organization).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16383v1">RotateKV: Accurate and Robust 2-Bit KV Cache Quantization for LLMs via Outlier-Aware Adaptive Rotations</a></div>
    <div class="paper-meta">
      📅 2025-01-25
    </div>
    <details class="paper-abstract">
      Key-Value (KV) cache facilitates efficient large language models (LLMs) inference by avoiding recomputation of past KVs. As the batch size and context length increase, the oversized KV caches become a significant memory bottleneck, highlighting the need for efficient compression. Existing KV quantization rely on fine-grained quantization or the retention of a significant portion of high bit-widths caches, both of which compromise compression ratio and often fail to maintain robustness at extremely low average bit-widths. In this work, we explore the potential of rotation technique for 2-bit KV quantization and propose RotateKV, which achieves accurate and robust performance through the following innovations: (i) Outlier-Aware Rotation, which utilizes channel-reordering to adapt the rotations to varying channel-wise outlier distributions without sacrificing the computational efficiency of the fast Walsh-Hadamard transform (FWHT); (ii) Pre-RoPE Grouped-Head Rotation, which mitigates the impact of rotary position embedding (RoPE) on proposed outlier-aware rotation and further smooths outliers across heads; (iii) Attention-Sink-Aware Quantization, which leverages the massive activations to precisely identify and protect attention sinks. RotateKV achieves less than 0.3 perplexity (PPL) degradation with 2-bit quantization on WikiText-2 using LLaMA-2-13B, maintains strong CoT reasoning and long-context capabilities, with less than 1.7\% degradation on GSM8K, outperforming existing methods even at lower average bit-widths. RotateKV also showcases a 3.97x reduction in peak memory usage, supports 5.75x larger batch sizes, and achieves a 2.32x speedup in decoding stage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.15007v1">Controllable Protein Sequence Generation with LLM Preference Optimization</a></div>
    <div class="paper-meta">
      📅 2025-01-25
      | 💬 Accepted in the 39th Annual AAAI Conference on Artificial Intelligence (AAAI 2025)
    </div>
    <details class="paper-abstract">
      Designing proteins with specific attributes offers an important solution to address biomedical challenges. Pre-trained protein large language models (LLMs) have shown promising results on protein sequence generation. However, to control sequence generation for specific attributes, existing work still exhibits poor functionality and structural stability. In this paper, we propose a novel controllable protein design method called CtrlProt. We finetune a protein LLM with a new multi-listwise preference optimization strategy to improve generation quality and support multi-attribute controllable generation. Experiments demonstrate that CtrlProt can meet functionality and structural stability requirements effectively, achieving state-of-the-art performance in both single-attribute and multi-attribute protein sequence generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14983v1">Code Change Intention, Development Artifact and History Vulnerability: Putting Them Together for Vulnerability Fix Detection by LLM</a></div>
    <div class="paper-meta">
      📅 2025-01-24
      | 💬 Accepted in FSE 2025
    </div>
    <details class="paper-abstract">
      Detecting vulnerability fix commits in open-source software is crucial for maintaining software security. To help OSS identify vulnerability fix commits, several automated approaches are developed. However, existing approaches like VulFixMiner and CoLeFunDa, focus solely on code changes, neglecting essential context from development artifacts. Tools like Vulcurator, which integrates issue reports, fail to leverage semantic associations between different development artifacts (e.g., pull requests and history vulnerability fixes). Moreover, they miss vulnerability fixes in tangled commits and lack explanations, limiting practical use. Hence to address those limitations, we propose LLM4VFD, a novel framework that leverages Large Language Models (LLMs) enhanced with Chain-of-Thought reasoning and In-Context Learning to improve the accuracy of vulnerability fix detection. LLM4VFD comprises three components: (1) Code Change Intention, which analyzes commit summaries, purposes, and implications using Chain-of-Thought reasoning; (2) Development Artifact, which incorporates context from related issue reports and pull requests; (3) Historical Vulnerability, which retrieves similar past vulnerability fixes to enrich context. More importantly, on top of the prediction, LLM4VFD also provides a detailed analysis and explanation to help security experts understand the rationale behind the decision. We evaluated LLM4VFD against state-of-the-art techniques, including Pre-trained Language Model-based approaches and vanilla LLMs, using a newly collected dataset, BigVulFixes. Experimental results demonstrate that LLM4VFD significantly outperforms the best-performed existing approach by 68.1%--145.4%. Furthermore, We conducted a user study with security experts, showing that the analysis generated by LLM4VFD improves the efficiency of vulnerability fix identification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02825v3">Randomly Sampled Language Reasoning Problems Reveal Limits of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-24
      | 💬 8 pages, 3 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Can LLMs pick up language structure from examples? Evidence in prior work seems to indicate yes, as pretrained models repeatedly demonstrate the ability to adapt to new language structures and vocabularies. However, this line of research typically considers languages that are present within common pretraining datasets, or otherwise share notable similarities with these seen languages. In contrast, in this work we attempt to measure models' language understanding capacity while circumventing the risk of dataset recall. We parameterize large families of language tasks recognized by deterministic finite automata (DFAs), and can thus sample novel language reasoning problems to fairly evaulate LLMs regardless of training data. We find that, even in the strikingly simple setting of 3-state DFAs, LLMs underperform unparameterized ngram models on both language recognition and synthesis tasks. These results suggest that LLMs struggle to match the ability of basic language models in recognizing and reasoning over languages that are sufficiently distinct from the ones they see at training time, underscoring the distinction between learning individual languages and possessing a general theory of language.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.09775v2">Multiple Choice Questions: Reasoning Makes Large Language Models (LLMs) More Self-Confident Even When They Are Wrong</a></div>
    <div class="paper-meta">
      📅 2025-01-24
    </div>
    <details class="paper-abstract">
      One of the most widely used methods to evaluate LLMs are Multiple Choice Question (MCQ) tests. MCQ benchmarks enable the testing of LLM knowledge on almost any topic at scale as the results can be processed automatically. To help the LLM answer, a few examples called few shots can be included in the prompt. Moreover, the LLM can be asked to answer the question directly with the selected option or to first provide the reasoning and then the selected answer, which is known as chain of thought. In addition to checking whether the selected answer is correct, the evaluation can look at the LLM-estimated probability of its response as an indication of the confidence of the LLM in the response. In this paper, we study how the LLM confidence in its answer depends on whether the model has been asked to answer directly or to provide the reasoning before answering. The results of the evaluation of questions on a wide range of topics in seven different models show that LLMs are more confident in their answers when they provide reasoning before the answer. This occurs regardless of whether the selected answer is correct. Our hypothesis is that this behavior is due to the reasoning that modifies the probability of the selected answer, as the LLM predicts the answer based on the input question and the reasoning that supports the selection made. Therefore, LLM estimated probabilities seem to have intrinsic limitations that should be understood in order to use them in evaluation procedures. Interestingly, the same behavior has been observed in humans, for whom explaining an answer increases confidence in its correctness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14932v1">Explaining Categorical Feature Interactions Using Graph Covariance and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-24
      | 💬 18 pages main + 6 pages appendix
    </div>
    <details class="paper-abstract">
      Modern datasets often consist of numerous samples with abundant features and associated timestamps. Analyzing such datasets to uncover underlying events typically requires complex statistical methods and substantial domain expertise. A notable example, and the primary data focus of this paper, is the global synthetic dataset from the Counter Trafficking Data Collaborative (CTDC) -- a global hub of human trafficking data containing over 200,000 anonymized records spanning from 2002 to 2022, with numerous categorical features for each record. In this paper, we propose a fast and scalable method for analyzing and extracting significant categorical feature interactions, and querying large language models (LLMs) to generate data-driven insights that explain these interactions. Our approach begins with a binarization step for categorical features using one-hot encoding, followed by the computation of graph covariance at each time. This graph covariance quantifies temporal changes in dependence structures within categorical data and is established as a consistent dependence measure under the Bernoulli distribution. We use this measure to identify significant feature pairs, such as those with the most frequent trends over time or those exhibiting sudden spikes in dependence at specific moments. These extracted feature pairs, along with their timestamps, are subsequently passed to an LLM tasked with generating potential explanations of the underlying events driving these dependence changes. The effectiveness of our method is demonstrated through extensive simulations, and its application to the CTDC dataset reveals meaningful feature pairs and potential data stories underlying the observed feature interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14892v1">Causal Graphs Meet Thoughts: Enhancing Complex Reasoning in Graph-Augmented LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-24
      | 💬 18 pages, 3 figures, 3 tables
    </div>
    <details class="paper-abstract">
      In knowledge-intensive tasks, especially in high-stakes domains like medicine and law, it is critical not only to retrieve relevant information but also to provide causal reasoning and explainability. Large language models (LLMs) have achieved remarkable performance in natural language understanding and generation tasks. However, they often suffer from limitations such as difficulty in incorporating new knowledge, generating hallucinations, and explaining their reasoning process. To address these challenges, integrating knowledge graphs with Graph Retrieval-Augmented Generation (Graph RAG) has emerged as an effective solution. Traditional Graph RAG methods often rely on simple graph traversal or semantic similarity, which do not capture causal relationships or align well with the model's internal reasoning steps. This paper proposes a novel pipeline that filters large knowledge graphs to emphasize cause-effect edges, aligns the retrieval process with the model's chain-of-thought (CoT), and enhances reasoning through multi-stage path improvements. Experiments on medical question-answering tasks show consistent gains, with up to a 10\% absolute improvement across multiple large language models (LLMs). This approach demonstrates the value of combining causal reasoning with stepwise retrieval, leading to more interpretable and logically grounded solutions for complex queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12689v2">EchoLM: Accelerating LLM Serving with Real-time Knowledge Distillation</a></div>
    <div class="paper-meta">
      📅 2025-01-24
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have excelled in various applications, yet serving them at scale is challenging due to their substantial resource demands and high latency. Our real-world studies reveal that over 60% of user requests to LLMs have semantically similar counterparts, suggesting the potential for knowledge sharing among requests. However, naively caching and reusing past responses leads to large quality degradation. In this paper, we introduce EchoLM, an in-context caching system that leverages historical requests as examples to guide response generation, enabling selective offloading of requests to more efficient LLMs. However, enabling this real-time knowledge transfer leads to intricate tradeoffs between response quality, latency, and system throughput at scale. For a new request, EchoLM identifies similar, high-utility examples and efficiently prepends them to the input for better response. At scale, EchoLM adaptively routes requests to LLMs of varying capabilities, accounting for response quality and serving loads. EchoLM employs a cost-aware cache replay mechanism to improve example quality and coverage offline, maximizing cache utility and runtime efficiency. Evaluations on millions of open-source requests demonstrate that EchoLM has a throughput improvement of 1.4-5.9x while reducing latency by 28-71% without hurting response quality on average.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14719v1">Do LLMs Provide Consistent Answers to Health-Related Questions across Languages?</a></div>
    <div class="paper-meta">
      📅 2025-01-24
      | 💬 9 pages. Short paper appeared at 47th European Conference on Information Retrieval (ECIR 2025)
    </div>
    <details class="paper-abstract">
      Equitable access to reliable health information is vital for public health, but the quality of online health resources varies by language, raising concerns about inconsistencies in Large Language Models (LLMs) for healthcare. In this study, we examine the consistency of responses provided by LLMs to health-related questions across English, German, Turkish, and Chinese. We largely expand the HealthFC dataset by categorizing health-related questions by disease type and broadening its multilingual scope with Turkish and Chinese translations. We reveal significant inconsistencies in responses that could spread healthcare misinformation. Our main contributions are 1) a multilingual health-related inquiry dataset with meta-information on disease categories, and 2) a novel prompt-based evaluation workflow that enables sub-dimensional comparisons between two languages through parsing. Our findings highlight key challenges in deploying LLM-based tools in multilingual contexts and emphasize the need for improved cross-lingual alignment to ensure accurate and equitable healthcare information.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14683v1">An Empirical Study on LLM-based Classification of Requirements-related Provisions in Food-safety Regulations</a></div>
    <div class="paper-meta">
      📅 2025-01-24
    </div>
    <details class="paper-abstract">
      As Industry 4.0 transforms the food industry, the role of software in achieving compliance with food-safety regulations is becoming increasingly critical. Food-safety regulations, like those in many legal domains, have largely been articulated in a technology-independent manner to ensure their longevity and broad applicability. However, this approach leaves a gap between the regulations and the modern systems and software increasingly used to implement them. In this article, we pursue two main goals. First, we conduct a Grounded Theory study of food-safety regulations and develop a conceptual characterization of food-safety concepts that closely relate to systems and software requirements. Second, we examine the effectiveness of two families of large language models (LLMs) -- BERT and GPT -- in automatically classifying legal provisions based on requirements-related food-safety concepts. Our results show that: (a) when fine-tuned, the accuracy differences between the best-performing models in the BERT and GPT families are relatively small. Nevertheless, the most powerful model in our experiments, GPT-4o, still achieves the highest accuracy, with an average Precision of 89% and an average Recall of 87%; (b) few-shot learning with GPT-4o increases Recall to 97% but decreases Precision to 65%, suggesting a trade-off between fine-tuning and few-shot learning; (c) despite our training examples being drawn exclusively from Canadian regulations, LLM-based classification performs consistently well on test provisions from the US, indicating a degree of generalizability across regulatory jurisdictions; and (d) for our classification task, LLMs significantly outperform simpler baselines constructed using long short-term memory (LSTM) networks and automatic keyword extraction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14654v1">MedAgentBench: Dataset for Benchmarking LLMs as Agents in Medical Applications</a></div>
    <div class="paper-meta">
      📅 2025-01-24
    </div>
    <details class="paper-abstract">
      Recent large language models (LLMs) have demonstrated significant advancements, particularly in their ability to serve as agents thereby surpassing their traditional role as chatbots. These agents can leverage their planning and tool utilization capabilities to address tasks specified at a high level. However, a standardized dataset to benchmark the agent capabilities of LLMs in medical applications is currently lacking, making the evaluation of LLMs on complex tasks in interactive healthcare environments challenging. To address this gap, we introduce MedAgentBench, a broad evaluation suite designed to assess the agent capabilities of large language models within medical records contexts. MedAgentBench encompasses 100 patient-specific clinically-derived tasks from 10 categories written by human physicians, realistic profiles of 100 patients with over 700,000 data elements, a FHIR-compliant interactive environment, and an accompanying codebase. The environment uses the standard APIs and communication infrastructure used in modern EMR systems, so it can be easily migrated into live EMR systems. MedAgentBench presents an unsaturated agent-oriented benchmark that current state-of-the-art LLMs exhibit some ability to succeed at. The best model (GPT-4o) achieves a success rate of 72%. However, there is still substantial space for improvement to give the community a next direction to optimize. Furthermore, there is significant variation in performance across task categories. MedAgentBench establishes this and is publicly available at https://github.com/stanfordmlgroup/MedAgentBench , offering a valuable framework for model developers to track progress and drive continuous improvements in the agent capabilities of large language models within the medical domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.17178v1">Tuning LLM Judges Hyperparameters</a></div>
    <div class="paper-meta">
      📅 2025-01-24
    </div>
    <details class="paper-abstract">
      Evaluating Large Language Models (LLMs) often requires costly human annotations. To address this, LLM-based judges have been proposed, which compare the outputs of two LLMs enabling the ranking of models without human intervention. While several approaches have been proposed, many confounding factors are present between different papers. For instance the model, the prompt and other hyperparameters are typically changed at the same time making apple-to-apple comparisons challenging. In this paper, we propose to systematically analyze and tune hyperparameter of LLM judges. To alleviate the high cost of evaluating a judge, we propose to leverage multi-objective multi-fidelity which allows to find judges that trades accuracy for cost and also reduce significantly the cost of the search. Our method identifies judges that not only outperform existing benchmarks in accuracy and cost-efficiency but also utilize open-weight models, ensuring greater accessibility and reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14630v1">Extracting Problem Structure with LLMs for Optimized SAT Local Search</a></div>
    <div class="paper-meta">
      📅 2025-01-24
    </div>
    <details class="paper-abstract">
      Local search preprocessing makes Conflict-Driven Clause Learning (CDCL) solvers faster by providing high-quality starting points and modern SAT solvers have incorporated this technique into their preprocessing steps. However, these tools rely on basic strategies that miss the structural patterns in problems. We present a method that applies Large Language Models (LLMs) to analyze Python-based encoding code. This reveals hidden structural patterns in how problems convert into SAT. Our method automatically generates specialized local search algorithms that find these patterns and use them to create strong initial assignments. This works for any problem instance from the same encoding type. Our tests show encouraging results, achieving faster solving times compared to baseline preprocessing systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.15809v4">LaMSUM: Amplifying Voices Against Harassment through LLM Guided Extractive Summarization of User Incident Reports</a></div>
    <div class="paper-meta">
      📅 2025-01-24
    </div>
    <details class="paper-abstract">
      Citizen reporting platforms like Safe City in India help the public and authorities stay informed about sexual harassment incidents. However, the high volume of data shared on these platforms makes reviewing each individual case challenging. Therefore, a summarization algorithm capable of processing and understanding various Indian code-mixed languages is essential. In recent years, Large Language Models (LLMs) have shown exceptional performance in NLP tasks, including summarization. LLMs inherently produce abstractive summaries by paraphrasing the original text, while the generation of extractive summaries - selecting specific subsets from the original text - through LLMs remains largely unexplored. Moreover, LLMs have a limited context window size, restricting the amount of data that can be processed at once. We tackle these challenge by introducing LaMSUM, a novel multi-level framework designed to generate extractive summaries for large collections of Safe City posts using LLMs. LaMSUM integrates summarization with different voting methods to achieve robust summaries. Extensive evaluation using three popular LLMs (Llama, Mistral and GPT-4o) demonstrates that LaMSUM outperforms state-of-the-art extractive summarization methods for Safe City posts. Overall, this work represents one of the first attempts to achieve extractive summarization through LLMs, and is likely to support stakeholders by offering a comprehensive overview and enabling them to develop effective policies to minimize incidents of unwarranted harassment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14625v1">Accelerated Preference Elicitation with LLM-Based Proxies</a></div>
    <div class="paper-meta">
      📅 2025-01-24
    </div>
    <details class="paper-abstract">
      Bidders in combinatorial auctions face significant challenges when describing their preferences to an auctioneer. Classical work on preference elicitation focuses on query-based techniques inspired from proper learning--often via proxies that interface between bidders and an auction mechanism--to incrementally learn bidder preferences as needed to compute efficient allocations. Although such elicitation mechanisms enjoy theoretical query efficiency, the amount of communication required may still be too cognitively taxing in practice. We propose a family of efficient LLM-based proxy designs for eliciting preferences from bidders using natural language. Our proposed mechanism combines LLM pipelines and DNF-proper-learning techniques to quickly approximate preferences when communication is limited. To validate our approach, we create a testing sandbox for elicitation mechanisms that communicate in natural language. In our experiments, our most promising LLM proxy design reaches approximately efficient outcomes with five times fewer queries than classical proper learning based elicitation mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.10321v2">Towards Human-Guided, Data-Centric LLM Co-Pilots</a></div>
    <div class="paper-meta">
      📅 2025-01-24
      | 💬 Saveliev, Liu & Seedat contributed equally
    </div>
    <details class="paper-abstract">
      Machine learning (ML) has the potential to revolutionize various domains, but its adoption is often hindered by the disconnect between the needs of domain experts and translating these needs into robust and valid ML tools. Despite recent advances in LLM-based co-pilots to democratize ML for non-technical domain experts, these systems remain predominantly focused on model-centric aspects while overlooking critical data-centric challenges. This limitation is problematic in complex real-world settings where raw data often contains complex issues, such as missing values, label noise, and domain-specific nuances requiring tailored handling. To address this we introduce CliMB-DC, a human-guided, data-centric framework for LLM co-pilots that combines advanced data-centric tools with LLM-driven reasoning to enable robust, context-aware data processing. At its core, CliMB-DC introduces a novel, multi-agent reasoning system that combines a strategic coordinator for dynamic planning and adaptation with a specialized worker agent for precise execution. Domain expertise is then systematically incorporated to guide the reasoning process using a human-in-the-loop approach. To guide development, we formalize a taxonomy of key data-centric challenges that co-pilots must address. Thereafter, to address the dimensions of the taxonomy, we integrate state-of-the-art data-centric tools into an extensible, open-source architecture, facilitating the addition of new tools from the research community. Empirically, using real-world healthcare datasets we demonstrate CliMB-DC's ability to transform uncurated datasets into ML-ready formats, significantly outperforming existing co-pilot baselines for handling data-centric challenges. CliMB-DC promises to empower domain experts from diverse domains -- healthcare, finance, social sciences and more -- to actively participate in driving real-world impact using ML.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.14117v3">An Investigation of Prompt Variations for Zero-shot LLM-based Rankers</a></div>
    <div class="paper-meta">
      📅 2025-01-24
    </div>
    <details class="paper-abstract">
      We provide a systematic understanding of the impact of specific components and wordings used in prompts on the effectiveness of rankers based on zero-shot Large Language Models (LLMs). Several zero-shot ranking methods based on LLMs have recently been proposed. Among many aspects, methods differ across (1) the ranking algorithm they implement, e.g., pointwise vs. listwise, (2) the backbone LLMs used, e.g., GPT3.5 vs. FLAN-T5, (3) the components and wording used in prompts, e.g., the use or not of role-definition (role-playing) and the actual words used to express this. It is currently unclear whether performance differences are due to the underlying ranking algorithm, or because of spurious factors such as better choice of words used in prompts. This confusion risks to undermine future research. Through our large-scale experimentation and analysis, we find that ranking algorithms do contribute to differences between methods for zero-shot LLM ranking. However, so do the LLM backbones -- but even more importantly, the choice of prompt components and wordings affect the ranking. In fact, in our experiments, we find that, at times, these latter elements have more impact on the ranker's effectiveness than the actual ranking algorithms, and that differences among ranking methods become more blurred when prompt variations are considered.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.10642v3">Self-playing Adversarial Language Game Enhances LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-01-24
      | 💬 Accepted by NeurIPS 2024
    </div>
    <details class="paper-abstract">
      We explore the potential of self-play training for large language models (LLMs) in a two-player adversarial language game called Adversarial Taboo. In this game, an attacker and a defender communicate around a target word only visible to the attacker. The attacker aims to induce the defender to speak the target word unconsciously, while the defender tries to infer the target word from the attacker's utterances. To win the game, both players must have sufficient knowledge about the target word and high-level reasoning ability to infer and express in this information-reserved conversation. Hence, we are curious about whether LLMs' reasoning ability can be further enhanced by Self-Playing this Adversarial language Game (SPAG). With this goal, we select several open-source LLMs and let each act as the attacker and play with a copy of itself as the defender on an extensive range of target words. Through reinforcement learning on the game outcomes, we observe that the LLMs' performances uniformly improve on a broad range of reasoning benchmarks. Furthermore, iteratively adopting this self-play process can continuously promote LLMs' reasoning abilities. The code is available at https://github.com/Linear95/SPAG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14540v1">VERUS-LM: a Versatile Framework for Combining LLMs with Symbolic Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-01-24
    </div>
    <details class="paper-abstract">
      A recent approach to neurosymbolic reasoning is to explicitly combine the strengths of large language models (LLMs) and symbolic solvers to tackle complex reasoning tasks. However, current approaches face significant limitations, including poor generalizability due to task-specific prompts, inefficiencies caused by the lack of separation between knowledge and queries, and restricted inferential capabilities. These shortcomings hinder their scalability and applicability across diverse domains. In this paper, we introduce VERUS-LM, a novel framework designed to address these challenges. VERUS-LM employs a generic prompting mechanism, clearly separates domain knowledge from queries, and supports a wide range of different logical reasoning tasks. This framework enhances adaptability, reduces computational cost, and allows for richer forms of reasoning, such as optimization and constraint satisfaction. We show that our approach succeeds in diverse reasoning on a novel dataset, markedly outperforming LLMs. Additionally, our system achieves competitive results on common reasoning benchmarks when compared to other state-of-the-art approaches, and significantly surpasses them on the difficult AR-LSAT dataset. By pushing the boundaries of hybrid reasoning, VERUS-LM represents a significant step towards more versatile neurosymbolic AI systems
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06605v3">RoboHorizon: An LLM-Assisted Multi-View World Model for Long-Horizon Robotic Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-01-24
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Efficient control in long-horizon robotic manipulation is challenging due to complex representation and policy learning requirements. Model-based visual reinforcement learning (RL) has shown great potential in addressing these challenges but still faces notable limitations, particularly in handling sparse rewards and complex visual features in long-horizon environments. To address these limitations, we propose the Recognize-Sense-Plan-Act (RSPA) pipeline for long-horizon tasks and further introduce RoboHorizon, an LLM-assisted multi-view world model tailored for long-horizon robotic manipulation. In RoboHorizon, pre-trained LLMs generate dense reward structures for multi-stage sub-tasks based on task language instructions, enabling robots to better recognize long-horizon tasks. Keyframe discovery is then integrated into the multi-view masked autoencoder (MAE) architecture to enhance the robot's ability to sense critical task sequences, strengthening its multi-stage perception of long-horizon processes. Leveraging these dense rewards and multi-view representations, a robotic world model is constructed to efficiently plan long-horizon tasks, enabling the robot to reliably act through RL algorithms. Experiments on two representative benchmarks, RLBench and FurnitureBench, show that RoboHorizon outperforms state-of-the-art visual model-based RL methods, achieving a 23.35% improvement in task success rates on RLBench's 4 short-horizon tasks and a 29.23% improvement on 6 long-horizon tasks from RLBench and 3 furniture assembly tasks from FurnitureBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14465v1">Boundary Value Test Input Generation Using Prompt Engineering with LLMs: Fault Detection and Coverage Analysis</a></div>
    <div class="paper-meta">
      📅 2025-01-24
    </div>
    <details class="paper-abstract">
      As software systems grow more complex, automated testing has become essential to ensuring reliability and performance. Traditional methods for boundary value test input generation can be time-consuming and may struggle to address all potential error cases effectively, especially in systems with intricate or highly variable boundaries. This paper presents a framework for assessing the effectiveness of large language models (LLMs) in generating boundary value test inputs for white-box software testing by examining their potential through prompt engineering. Specifically, we evaluate the effectiveness of LLM-based test input generation by analyzing fault detection rates and test coverage, comparing these LLM-generated test sets with those produced using traditional boundary value analysis methods. Our analysis shows the strengths and limitations of LLMs in boundary value generation, particularly in detecting common boundary-related issues. However, they still face challenges in certain areas, especially when handling complex or less common test inputs. This research provides insights into the role of LLMs in boundary value testing, underscoring both their potential and areas for improvement in automated testing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14457v1">Understanding and Mitigating Gender Bias in LLMs via Interpretable Neuron Editing</a></div>
    <div class="paper-meta">
      📅 2025-01-24
      | 💬 preprint
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often exhibit gender bias, posing challenges for their safe deployment. Existing methods to mitigate bias lack a comprehensive understanding of its mechanisms or compromise the model's core capabilities. To address these issues, we propose the CommonWords dataset, to systematically evaluate gender bias in LLMs. Our analysis reveals pervasive bias across models and identifies specific neuron circuits, including gender neurons and general neurons, responsible for this behavior. Notably, editing even a small number of general neurons can disrupt the model's overall capabilities due to hierarchical neuron interactions. Based on these insights, we propose an interpretable neuron editing method that combines logit-based and causal-based strategies to selectively target biased neurons. Experiments on five LLMs demonstrate that our method effectively reduces gender bias while preserving the model's original capabilities, outperforming existing fine-tuning and editing approaches. Our findings contribute a novel dataset, a detailed analysis of bias mechanisms, and a practical solution for mitigating gender bias in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14431v1">Domaino1s: Guiding LLM Reasoning for Explainable Answers in High-Stakes Domains</a></div>
    <div class="paper-meta">
      📅 2025-01-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely applied to downstream domains. However, current LLMs for high-stakes domain tasks, such as financial investment and legal QA, typically generate brief answers without reasoning processes and explanations. This limits users' confidence in making decisions based on their responses. While original CoT shows promise, it lacks self-correction mechanisms during reasoning. This work introduces Domain$o1$s, which enhances LLMs' reasoning capabilities on domain tasks through supervised fine-tuning and tree search. We construct CoT-stock-2k and CoT-legal-2k datasets for fine-tuning models that activate domain-specific reasoning steps based on their judgment. Additionally, we propose Selective Tree Exploration to spontaneously explore solution spaces and sample optimal reasoning paths to improve performance. We also introduce PROOF-Score, a new metric for evaluating domain models' explainability, complementing traditional accuracy metrics with richer assessment dimensions. Extensive experiments on stock investment recommendation and legal reasoning QA tasks demonstrate Domaino1s's leading performance and explainability. Our code is available at https://anonymous.4open.science/r/Domaino1s-006F/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14427v1">GraphBC: Improving LLMs for Better Graph Data Processing</a></div>
    <div class="paper-meta">
      📅 2025-01-24
    </div>
    <details class="paper-abstract">
      The success of Large Language Models (LLMs) in various domains has led researchers to apply them to graph-related problems by converting graph data into natural language text. However, unlike graph data, natural language inherently has sequential order. We observe that when the order of nodes or edges in the natural language description of a graph is shuffled, despite describing the same graph, model performance fluctuates between high performance and random guessing. Additionally, due to the limited input context length of LLMs, current methods typically randomly sample neighbors of target nodes as representatives of their neighborhood, which may not always be effective for accurate reasoning. To address these gaps, we introduce GraphBC. This novel model framework features an Order Selector Module to ensure proper serialization order of the graph and a Subgraph Sampling Module to sample subgraphs with better structure for better reasoning. Furthermore, we propose Graph CoT obtained through distillation, and enhance LLM's reasoning and zero-shot learning capabilities for graph tasks through instruction tuning. Experiments on multiple datasets for node classification and graph question-answering demonstrate that GraphBC improves LLMs' performance and generalization ability on graph tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.08027v2">iLLuMinaTE: An LLM-XAI Framework Leveraging Social Science Explanation Theories Towards Actionable Student Performance Feedback</a></div>
    <div class="paper-meta">
      📅 2025-01-24
      | 💬 Accepted at AAAI 2025
    </div>
    <details class="paper-abstract">
      Recent advances in eXplainable AI (XAI) for education have highlighted a critical challenge: ensuring that explanations for state-of-the-art AI models are understandable for non-technical users such as educators and students. In response, we introduce iLLuMinaTE, a zero-shot, chain-of-prompts LLM-XAI pipeline inspired by Miller's cognitive model of explanation. iLLuMinaTE is designed to deliver theory-driven, actionable feedback to students in online courses. iLLuMinaTE navigates three main stages - causal connection, explanation selection, and explanation presentation - with variations drawing from eight social science theories (e.g. Abnormal Conditions, Pearl's Model of Explanation, Necessity and Robustness Selection, Contrastive Explanation). We extensively evaluate 21,915 natural language explanations of iLLuMinaTE extracted from three LLMs (GPT-4o, Gemma2-9B, Llama3-70B), with three different underlying XAI methods (LIME, Counterfactuals, MC-LIME), across students from three diverse online courses. Our evaluation involves analyses of explanation alignment to the social science theory, understandability of the explanation, and a real-world user preference study with 114 university students containing a novel actionability simulation. We find that students prefer iLLuMinaTE explanations over traditional explainers 89.52% of the time. Our work provides a robust, ready-to-use framework for effectively communicating hybrid XAI-driven insights in education, with significant generalization potential for other human-centric fields.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14371v1">DRESSing Up LLM: Efficient Stylized Question-Answering via Style Subspace Editing</a></div>
    <div class="paper-meta">
      📅 2025-01-24
      | 💬 ICLR 2025 Accepted
    </div>
    <details class="paper-abstract">
      We introduce DRESS, a novel approach for generating stylized large language model (LLM) responses through representation editing. Existing methods like prompting and fine-tuning are either insufficient for complex style adaptation or computationally expensive, particularly in tasks like NPC creation or character role-playing. Our approach leverages the over-parameterized nature of LLMs to disentangle a style-relevant subspace within the model's representation space to conduct representation editing, ensuring a minimal impact on the original semantics. By applying adaptive editing strengths, we dynamically adjust the steering vectors in the style subspace to maintain both stylistic fidelity and semantic integrity. We develop two stylized QA benchmark datasets to validate the effectiveness of DRESS, and the results demonstrate significant improvements compared to baseline methods such as prompting and ITI. In short, DRESS is a lightweight, train-free solution for enhancing LLMs with flexible and effective style control, making it particularly useful for developing stylized conversational agents. Codes and benchmark datasets are available at https://github.com/ArthurLeoM/DRESS-LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14350v1">FireRedASR: Open-Source Industrial-Grade Mandarin Speech Recognition Models from Encoder-Decoder to LLM Integration</a></div>
    <div class="paper-meta">
      📅 2025-01-24
    </div>
    <details class="paper-abstract">
      We present FireRedASR, a family of large-scale automatic speech recognition (ASR) models for Mandarin, designed to meet diverse requirements in superior performance and optimal efficiency across various applications. FireRedASR comprises two variants: FireRedASR-LLM: Designed to achieve state-of-the-art (SOTA) performance and to enable seamless end-to-end speech interaction. It adopts an Encoder-Adapter-LLM framework leveraging large language model (LLM) capabilities. On public Mandarin benchmarks, FireRedASR-LLM (8.3B parameters) achieves an average Character Error Rate (CER) of 3.05%, surpassing the latest SOTA of 3.33% with an 8.4% relative CER reduction (CERR). It demonstrates superior generalization capability over industrial-grade baselines, achieving 24%-40% CERR in multi-source Mandarin ASR scenarios such as video, live, and intelligent assistant. FireRedASR-AED: Designed to balance high performance and computational efficiency and to serve as an effective speech representation module in LLM-based speech models. It utilizes an Attention-based Encoder-Decoder (AED) architecture. On public Mandarin benchmarks, FireRedASR-AED (1.1B parameters) achieves an average CER of 3.18%, slightly worse than FireRedASR-LLM but still outperforming the latest SOTA model with over 12B parameters. It offers a more compact size, making it suitable for resource-constrained applications. Moreover, both models exhibit competitive results on Chinese dialects and English speech benchmarks and excel in singing lyrics recognition. To advance research in speech processing, we release our models and inference code at https://github.com/FireRedTeam/FireRedASR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.01639v3">Moral Alignment for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-01-24
      | 💬 To appear at the 13th International Conference on Learning Representations (ICLR'25), Singapore, Apr 2025
    </div>
    <details class="paper-abstract">
      Decision-making agents based on pre-trained Large Language Models (LLMs) are increasingly being deployed across various domains of human activity. While their applications are currently rather specialized, several research efforts are under way to develop more generalist agents. As LLM-based systems become more agentic, their influence on human activity will grow and the transparency of this will decrease. Consequently, developing effective methods for aligning them to human values is vital. The prevailing practice in alignment often relies on human preference data (e.g., in RLHF or DPO), in which values are implicit and are essentially deduced from relative preferences over different model outputs. In this work, instead of relying on human feedback, we introduce the design of reward functions that explicitly encode core human values for Reinforcement Learning-based fine-tuning of foundation agent models. Specifically, we use intrinsic rewards for the moral alignment of LLM agents. We evaluate our approach using the traditional philosophical frameworks of Deontological Ethics and Utilitarianism, quantifying moral rewards for agents in terms of actions and consequences on the Iterated Prisoner's Dilemma (IPD) environment. We also show how moral fine-tuning can be deployed to enable an agent to unlearn a previously developed selfish strategy. Finally, we find that certain moral strategies learned on the IPD game generalize to several other matrix game environments. In summary, we demonstrate that fine-tuning with intrinsic rewards is a promising general solution for aligning LLM agents to human values, and it might represent a more transparent and cost-effective alternative to currently predominant alignment techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.17094v2">Analysis on LLMs Performance for Code Summarization</a></div>
    <div class="paper-meta">
      📅 2025-01-24
    </div>
    <details class="paper-abstract">
      Code summarization aims to generate concise natural language descriptions for source code. Deep learning has been used more and more recently in software engineering, particularly for tasks like code creation and summarization. Specifically, it appears that the most current Large Language Models with coding perform well on these tasks. Large Language Models (LLMs) have significantly advanced the field of code summarization, providing sophisticated methods for generating concise and accurate summaries of source code. This study aims to perform a comparative analysis of several open-source LLMs, namely LLaMA-3, Phi-3, Mistral, and Gemma. These models' performance is assessed using important metrics such as BLEU\textsubscript{3.1} and ROUGE\textsubscript{3.2}. Through this analysis, we seek to identify the strengths and weaknesses of each model, offering insights into their applicability and effectiveness in code summarization tasks. Our findings contribute to the ongoing development and refinement of LLMs, supporting their integration into tools that enhance software development and maintenance processes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.11081v2">The Promises and Pitfalls of LLM Annotations in Dataset Labeling: a Case Study on Media Bias Detection</a></div>
    <div class="paper-meta">
      📅 2025-01-24
    </div>
    <details class="paper-abstract">
      High annotation costs from hiring or crowdsourcing complicate the creation of large, high-quality datasets needed for training reliable text classifiers. Recent research suggests using Large Language Models (LLMs) to automate the annotation process, reducing these costs while maintaining data quality. LLMs have shown promising results in annotating downstream tasks like hate speech detection and political framing. Building on the success in these areas, this study investigates whether LLMs are viable for annotating the complex task of media bias detection and whether a downstream media bias classifier can be trained on such data. We create annolexical, the first large-scale dataset for media bias classification with over 48000 synthetically annotated examples. Our classifier, fine-tuned on this dataset, surpasses all of the annotator LLMs by 5-9 percent in Matthews Correlation Coefficient (MCC) and performs close to or outperforms the model trained on human-labeled data when evaluated on two media bias benchmark datasets (BABE and BASIL). This study demonstrates how our approach significantly reduces the cost of dataset creation in the media bias domain and, by extension, the development of classifiers, while our subsequent behavioral stress-testing reveals some of its current limitations and trade-offs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14315v1">Clear Minds Think Alike: What Makes LLM Fine-tuning Robust? A Study of Token Perplexity</a></div>
    <div class="paper-meta">
      📅 2025-01-24
    </div>
    <details class="paper-abstract">
      Maintaining consistent model performance across domains is a fundamental challenge in machine learning. While recent work has explored using LLM-generated data for fine-tuning, its impact on cross-domain generalization remains poorly understood. In this paper, we present a systematic analysis revealing that fine-tuning with LLM-generated data not only improves target task performance but also reduces out-of-domain (OOD) degradation compared to fine-tuning with ground truth data. Through analyzing the data sequence in tasks of various domains, we demonstrate that this enhanced OOD robustness stems from a reduced prevalence of high perplexity tokens in LLM-generated sequences. Following this hypothesis we showed that masking high perplexity tokens in ground truth training data also achieves similar OOD preservation comparable to using LLM-generated data. Extensive experiments across diverse model architectures and scales, including Gemma2-2B, Mistral-7B and Llama3-8B, corroborate the consistency of our findings. To the best of our knowledge, this work provides the first mechanistic explanation for the superior OOD robustness conferred by LLM-generated training data, offering valuable insights for developing more robust fine-tuning strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.05006v2">COAST: Enhancing the Code Debugging Ability of LLMs through Communicative Agent Based Data Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-01-24
    </div>
    <details class="paper-abstract">
      Code debugging is a vital stage of software development, essential for ensuring the reliability and performance of Large Language Models (LLMs) in code generation task. Human debugging typically follows a multi-stage process, which includes Bug Localization, Bug Identification, Code Repair, and Code Recognition. However, existing code debugging benchmarks predominantly focus on the Code Repair stage, which offers only a limited perspective on evaluating the debugging capabilities of LLMs. In this paper, we introduce DEBUGEVAL, a comprehensive benchmark for evaluating the debugging abilities of LLMs by emulating the multi-stage human debugging process. Through evaluating on DEBUGEVAL, we observe that 7B-scale models consistently underperform compared to their larger counterparts, highlighting their limitations in comprehending code semantics. In this case, we propose the COmmunicative Agent-based data SynThesis (COAST) framework, which employs a multi-agent system to generate high-quality training data for supervised fine-tuning (SFT). Experimental results demonstrate that COAST-generated data outperform human-curated and GPT-4-generated data, enabling 7B-scale LLMs to achieve debugging performance comparable to GPT-3.5.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14312v1">Locality-aware Fair Scheduling in LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-01-24
    </div>
    <details class="paper-abstract">
      Large language model (LLM) inference workload dominates a wide variety of modern AI applications, ranging from multi-turn conversation to document analysis. Balancing fairness and efficiency is critical for managing diverse client workloads with varying prefix patterns. Unfortunately, existing fair scheduling algorithms for LLM serving, such as Virtual Token Counter (VTC), fail to take prefix locality into consideration and thus suffer from poor performance. On the other hand, locality-aware scheduling algorithms in existing LLM serving frameworks tend to maximize the prefix cache hit rate without considering fair sharing among clients. This paper introduces the first locality-aware fair scheduling algorithm, Deficit Longest Prefix Match (DLPM), which can maintain a high degree of prefix locality with a fairness guarantee. We also introduce a novel algorithm, Double Deficit LPM (D$^2$LPM), extending DLPM for the distributed setup that can find a balance point among fairness, locality, and load-balancing. Our extensive evaluation demonstrates the superior performance of DLPM and D$^2$LPM in ensuring fairness while maintaining high throughput (up to 2.87$\times$ higher than VTC) and low per-client (up to 7.18$\times$ lower than state-of-the-art distributed LLM serving system) latency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14305v1">A Zero-Shot LLM Framework for Automatic Assignment Grading in Higher Education</a></div>
    <div class="paper-meta">
      📅 2025-01-24
    </div>
    <details class="paper-abstract">
      Automated grading has become an essential tool in education technology due to its ability to efficiently assess large volumes of student work, provide consistent and unbiased evaluations, and deliver immediate feedback to enhance learning. However, current systems face significant limitations, including the need for large datasets in few-shot learning methods, a lack of personalized and actionable feedback, and an overemphasis on benchmark performance rather than student experience. To address these challenges, we propose a Zero-Shot Large Language Model (LLM)-Based Automated Assignment Grading (AAG) system. This framework leverages prompt engineering to evaluate both computational and explanatory student responses without requiring additional training or fine-tuning. The AAG system delivers tailored feedback that highlights individual strengths and areas for improvement, thereby enhancing student learning outcomes. Our study demonstrates the system's effectiveness through comprehensive evaluations, including survey responses from higher education students that indicate significant improvements in motivation, understanding, and preparedness compared to traditional grading methods. The results validate the AAG system's potential to transform educational assessment by prioritizing learning experiences and providing scalable, high-quality feedback.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14304v1">MASTER: A Multi-Agent System with LLM Specialized MCTS</a></div>
    <div class="paper-meta">
      📅 2025-01-24
      | 💬 Accepted by main NAACL 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLM) are increasingly being explored for problem-solving tasks. However, their strategic planning capability is often viewed with skepticism. Recent studies have incorporated the Monte Carlo Tree Search (MCTS) algorithm to augment the planning capacity of LLM. Despite its potential, MCTS relies on extensive sampling simulations to approximate the true reward distribution, leading to two primary issues. Firstly, MCTS is effective for tasks like the Game of Go, where simulation results can yield objective rewards (e.g., 1 for a win and 0 for a loss). However, for tasks such as question answering, the result of a simulation is the answer to the question, which cannot obtain an objective reward without the ground truth. Secondly, obtaining statistically significant reward estimations typically requires a sample size exceeding 30 simulations, resulting in excessive token usage and time consumption. To address these challenges, we present Multi-Agent System with Tactical Execution and Reasoning using LLM Specialized MCTS (MASTER), a novel framework that coordinates agent recruitment and communication using LLM specialized MCTS. This system autonomously adjusts the number of agents based on task complexity and ensures focused communication among them. Comprehensive experiments across various tasks demonstrate the effectiveness of our proposed framework. It achieves 76% accuracy on HotpotQA and 80% on WebShop, setting new state-of-the-art performance on these datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.12334v3">What Did I Do Wrong? Quantifying LLMs' Sensitivity and Consistency to Prompt Engineering</a></div>
    <div class="paper-meta">
      📅 2025-01-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) changed the way we design and interact with software systems. Their ability to process and extract information from text has drastically improved productivity in a number of routine tasks. Developers that want to include these models in their software stack, however, face a dreadful challenge: debugging LLMs' inconsistent behavior across minor variations of the prompt. We therefore introduce two metrics for classification tasks, namely sensitivity and consistency, which are complementary to task performance. First, sensitivity measures changes of predictions across rephrasings of the prompt, and does not require access to ground truth labels. Instead, consistency measures how predictions vary across rephrasings for elements of the same class. We perform an empirical comparison of these metrics on text classification tasks, using them as guideline for understanding failure modes of the LLM. Our hope is that sensitivity and consistency will be helpful to guide prompt engineering and obtain LLMs that balance robustness with performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14291v1">Advances in Temporal Point Processes: Bayesian, Deep, and LLM Approaches</a></div>
    <div class="paper-meta">
      📅 2025-01-24
    </div>
    <details class="paper-abstract">
      Temporal point processes (TPPs) are stochastic process models used to characterize event sequences occurring in continuous time. Traditional statistical TPPs have a long-standing history, with numerous models proposed and successfully applied across diverse domains. In recent years, advances in deep learning have spurred the development of neural TPPs, enabling greater flexibility and expressiveness in capturing complex temporal dynamics. The emergence of large language models (LLMs) has further sparked excitement, offering new possibilities for modeling and analyzing event sequences by leveraging their rich contextual understanding. This survey presents a comprehensive review of recent research on TPPs from three perspectives: Bayesian, deep learning, and LLM approaches. We begin with a review of the fundamental concepts of TPPs, followed by an in-depth discussion of model design and parameter estimation techniques in these three frameworks. We also revisit classic application areas of TPPs to highlight their practical relevance. Finally, we outline challenges and promising directions for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14275v1">Leveraging Online Olympiad-Level Math Problems for LLMs Training and Contamination-Resistant Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-01-24
    </div>
    <details class="paper-abstract">
      Advances in Large Language Models (LLMs) have sparked interest in their ability to solve Olympiad-level math problems. However, the training and evaluation of these models are constrained by the limited size and quality of available datasets, as creating large-scale data for such advanced problems requires extensive effort from human experts. In addition, current benchmarks are prone to contamination, leading to unreliable evaluations. In this paper, we present an automated pipeline that leverages the rich resources of the Art of Problem Solving (AoPS) forum, which predominantly features Olympiad-level problems and community-driven solutions. Using open-source LLMs, we develop a method to extract question-answer pairs from the forum, resulting in AoPS-Instruct, a dataset of more than 600,000 high-quality QA pairs. Our experiments demonstrate that fine-tuning LLMs on AoPS-Instruct improves their reasoning abilities across various benchmarks. Moreover, we build an automatic pipeline that introduces LiveAoPSBench, an evolving evaluation set with timestamps, derived from the latest forum data, providing a contamination-resistant benchmark for assessing LLM performance. Notably, we observe a significant decline in LLM performance over time, suggesting their success on older examples may stem from pre-training exposure rather than true reasoning ability. Our work presents a scalable approach to creating and maintaining large-scale, high-quality datasets for advanced math reasoning, offering valuable insights into the capabilities and limitations of LLMs in this domain. Our benchmark and code is available at https://github.com/DSL-Lab/aops
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12612v2">Auto-Cypher: Improving LLMs on Cypher generation via LLM-supervised generation-verification framework</a></div>
    <div class="paper-meta">
      📅 2025-01-24
      | 💬 Accepted at NAACL 2025 main conference
    </div>
    <details class="paper-abstract">
      Graph databases like Neo4j are gaining popularity for handling complex, interconnected data, over traditional relational databases in modeling and querying relationships. While translating natural language into SQL queries is well-researched, generating Cypher queries for Neo4j remains relatively underexplored. In this work, we present an automated, LLM-Supervised, pipeline to generate high-quality synthetic data for Text2Cypher. Our Cypher data generation pipeline introduces LLM-As-Database-Filler, a novel strategy for ensuring Cypher query correctness, thus resulting in high quality generations. Using our pipeline, we generate high quality Text2Cypher data - SynthCypher containing 29.8k instances across various domains and queries with varying complexities. Training open-source LLMs like LLaMa-3.1-8B, Mistral-7B, and QWEN-7B on SynthCypher results in performance gains of up to 40% on the Text2Cypher test split and 30% on the SPIDER benchmark, adapted for graph databases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.12692v2">SparseInfer: Training-free Prediction of Activation Sparsity for Fast LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-01-24
      | 💬 7 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Leveraging sparsity is crucial for optimizing large language model inference. however, modern LLMs employing SiLU as their activation function exhibit minimal activation sparsity. Recent research has proposed replacing SiLU with ReLU to induce significant activation sparsity and showed no downstream task accuracy degradation through fine tuning. However, taking full advantage of it required training a predictor to estimate this sparsity. In this paper, we introduce SparseInfer, a simple, light weight, and training free predictor for activation sparsity of ReLU field LLMs, in which activation sparsity is predicted by comparing only the sign bits of inputs and weights. To compensate for possible prediction inaccuracy, an adaptive tuning of the predictor's conservativeness is enabled, which can also serve as a control knob for optimizing LLM inference. The proposed method achieves approximately faster inference speed over the state of the art, with negligible accuracy loss of within 1%p.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14205v1">Serving Long-Context LLMs at the Mobile Edge: Test-Time Reinforcement Learning-based Model Caching and Inference Offloading</a></div>
    <div class="paper-meta">
      📅 2025-01-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can perform zero-shot learning on unseen tasks and few-shot learning on complex reasoning tasks. However, resource-limited mobile edge networks struggle to support long-context LLM serving for LLM agents during multi-round interactions with users. Unlike stateless computation offloading and static service offloading in edge computing, optimizing LLM serving at edge servers is challenging because LLMs continuously learn from context which raises accuracy, latency, and resource consumption dynamics. In this paper, we propose a joint model caching and inference offloading framework that utilizes test-time deep reinforcement learning (T2DRL) to optimize deployment and execution strategies for long-context LLM serving. In this framework, we analyze the performance convergence and design an optimization problem considering the utilization of context windows in LLMs. Furthermore, the T2DRL algorithm can learn in both the training phase and the testing phase to proactively manage cached models and service requests and adapt to context changes and usage patterns during execution. To further enhance resource allocation efficiency, we propose a double Dutch auction (DDA) mechanism, which dynamically matches supply and demand while maximizing social welfare. Finally, experimental results demonstrate that the T2DRL algorithm can reduce system costs by at least 30% compared to baselines while guaranteeing the performance of LLM agents in real-world perception and reasoning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16376v1">HWPQ: Hessian-free Weight Pruning-Quantization For LLM Compression And Acceleration</a></div>
    <div class="paper-meta">
      📅 2025-01-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable success across numerous domains. However, the high time complexity of existing pruning and quantization methods significantly hinders their effective deployment on resource-constrained consumer or edge devices. In this study, we propose a novel Hessian-free Weight Pruning-Quantization (HWPQ) method. HWPQ eliminates the need for computationally intensive Hessian matrix calculations by introducing a contribution-based weight metric, which evaluates the importance of weights without relying on second-order derivatives. Additionally, we employ the Exponentially Weighted Moving Average (EWMA) technique to bypass weight sorting, enabling the selection of weights that contribute most to LLM accuracy and further reducing time complexity. Our approach is extended to support 2:4 structured sparsity pruning, facilitating efficient execution on modern hardware accelerators. Experimental results demonstrate that HWPQ significantly enhances the compression performance of LLaMA2. Compared to state-of-the-art quantization and pruning frameworks, HWPQ achieves average speedups of 5.97x (up to 20.75x) in quantization time and 12.29x (up to 56.02x) in pruning time, while largely preserving model accuracy. Furthermore, we observe a 1.50x inference speedup compared to the baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06252v3">Transformer-Squared: Self-adaptive LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-24
      | 💬 To appear at the 13th International Conference on Learning Representations (ICLR 2025)
    </div>
    <details class="paper-abstract">
      Self-adaptive large language models (LLMs) aim to solve the challenges posed by traditional fine-tuning methods, which are often computationally intensive and static in their ability to handle diverse tasks. We introduce Transformer-Squared, a novel self-adaptation framework that adapts LLMs for unseen tasks in real-time by selectively adjusting only the singular components of their weight matrices. During inference, Transformer-Squared employs a two-pass mechanism: first, a dispatch system identifies the task properties, and then task-specific 'expert' vectors, trained using reinforcement learning, are dynamically mixed to obtain targeted behavior for the incoming prompt. Our method consistently outperforms ubiquitous approaches such as LoRA, with fewer parameters and greater efficiency. Furthermore, Transformer-Squared demonstrates versatility across different LLM architectures and modalities, including vision-language tasks. Transformer-Squared represents a significant leap forward, offering a scalable, efficient solution for enhancing the adaptability and task-specific performance of LLMs, paving the way for truly dynamic, self-organizing AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10989v3">Liger Kernel: Efficient Triton Kernels for LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-01-24
      | 💬 17 pages, 12 figures
    </div>
    <details class="paper-abstract">
      Training Large Language Models (LLMs) efficiently at scale presents a formidable challenge, driven by their ever-increasing computational demands and the need for enhanced performance. In this work, we introduce Liger-Kernel, an open-sourced set of Triton kernels developed specifically for LLM training. With kernel optimization techniques like kernel operation fusing and input chunking, our kernels achieve on average a 20% increase in training throughput and a 60% reduction in GPU memory usage for popular LLMs compared to HuggingFace implementations. In addition, Liger-Kernel is designed with modularity, accessibility, and adaptability in mind, catering to both casual and expert users. Comprehensive benchmarks and integration tests are built in to ensure compatibility, performance, correctness, and convergence across diverse computing environments and model architectures. The source code is available under a permissive license at: github.com/linkedin/Liger-Kernel.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06394v4">GameArena: Evaluating LLM Reasoning through Live Computer Games</a></div>
    <div class="paper-meta">
      📅 2025-01-23
    </div>
    <details class="paper-abstract">
      Evaluating the reasoning abilities of large language models (LLMs) is challenging. Existing benchmarks often depend on static datasets, which are vulnerable to data contamination and may get saturated over time, or on binary live human feedback that conflates reasoning with other abilities. As the most prominent dynamic benchmark, Chatbot Arena evaluates open-ended questions in real-world settings, but lacks the granularity in assessing specific reasoning capabilities. We introduce GameArena, a dynamic benchmark designed to evaluate LLM reasoning capabilities through interactive gameplay with humans. GameArena consists of three games designed to test specific reasoning capabilities (e.g., deductive and inductive reasoning), while keeping participants entertained and engaged. We analyze the gaming data retrospectively to uncover the underlying reasoning processes of LLMs and measure their fine-grained reasoning capabilities. We collect over 2000 game sessions and provide detailed assessments of various reasoning capabilities for five state-of-the-art LLMs. Our user study with 100 participants suggests that GameArena improves user engagement compared to Chatbot Arena. For the first time, GameArena enables the collection of step-by-step LLM reasoning data in the wild.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.07341v2">A Guide To Effectively Leveraging LLMs for Low-Resource Text Summarization: Data Augmentation and Semi-supervised Approaches</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 Accepted to NAACL 2025 (Findings)
    </div>
    <details class="paper-abstract">
      Existing approaches for low-resource text summarization primarily employ large language models (LLMs) like GPT-3 or GPT-4 at inference time to generate summaries directly; however, such approaches often suffer from inconsistent LLM outputs and are difficult to adapt to domain-specific data in low-resource scenarios. In this work, we propose two novel methods to effectively utilize LLMs for low-resource text summarization: 1) MixSumm, an LLM-based data augmentation regime that synthesizes high-quality documents (short and long) for few-shot text summarization, and 2) PPSL, a prompt-based pseudolabeling strategy for sample-efficient semi-supervised text summarization. Specifically, MixSumm leverages the open-source LLaMA-3-70b-Instruct model to generate new documents by mixing topical information derived from a small seed set, and PPSL leverages the LLaMA-3-70b-Instruct model to generate high-quality pseudo-labels in a semi-supervised learning setup. We evaluate our methods on the TweetSumm, WikiHow, and ArXiv/PubMed datasets and use L-Eval, a LLaMA-3-based evaluation metric, and ROUGE scores to measure the quality of generated summaries. Our experiments on extractive and abstractive summarization show that MixSumm and PPSL achieve competitive ROUGE scores as a fully supervised method with 5% of the labeled data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.11747v2">Optimizing Pretraining Data Mixtures with LLM-Estimated Utility</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 10 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Large Language Models improve with increasing amounts of high-quality training data. However, leveraging larger datasets requires balancing quality, quantity, and diversity across sources. After evaluating nine baseline methods under both compute- and data-constrained scenarios, we find token-count heuristics outperform manual and learned mixes, indicating that simple approaches accounting for dataset size and diversity are surprisingly effective. Building on this insight, we propose two complementary approaches: UtiliMax, which extends token-based heuristics by incorporating utility estimates from reduced-scale ablations, achieving up to a 10.6x speedup over manual baselines; and Model Estimated Data Utility (MEDU), which leverages LLMs to estimate data utility from small samples, matching ablation-based performance while reducing computational requirements by $\sim$200x. Together, these approaches establish a new framework for automated, compute-efficient data mixing that is robust across training regimes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14073v1">LLMs are Vulnerable to Malicious Prompts Disguised as Scientific Language</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) have been deployed in various real-world settings, concerns about the harm they may propagate have grown. Various jailbreaking techniques have been developed to expose the vulnerabilities of these models and improve their safety. This work reveals that many state-of-the-art proprietary and open-source LLMs are vulnerable to malicious requests hidden behind scientific language. Specifically, our experiments with GPT4o, GPT4o-mini, GPT-4, LLama3-405B-Instruct, Llama3-70B-Instruct, Cohere, Gemini models on the StereoSet data demonstrate that, the models' biases and toxicity substantially increase when prompted with requests that deliberately misinterpret social science and psychological studies as evidence supporting the benefits of stereotypical biases. Alarmingly, these models can also be manipulated to generate fabricated scientific arguments claiming that biases are beneficial, which can be used by ill-intended actors to systematically jailbreak even the strongest models like GPT. Our analysis studies various factors that contribute to the models' vulnerabilities to malicious requests in academic language. Mentioning author names and venues enhances the persuasiveness of some models, and the bias scores can increase as dialogues progress. Our findings call for a more careful investigation on the use of scientific data in the training of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13387v3">CLEAR: Towards Contextual LLM-Empowered Privacy Policy Analysis and Risk Generation for Large Language Model Applications</a></div>
    <div class="paper-meta">
      📅 2025-01-23
    </div>
    <details class="paper-abstract">
      The rise of end-user applications powered by large language models (LLMs), including both conversational interfaces and add-ons to existing graphical user interfaces (GUIs), introduces new privacy challenges. However, many users remain unaware of the risks. This paper explores methods to increase user awareness of privacy risks associated with LLMs in end-user applications. We conducted five co-design workshops to uncover user privacy concerns and their demand for contextual privacy information within LLMs. Based on these insights, we developed CLEAR (Contextual LLM-Empowered Privacy Policy Analysis and Risk Generation), a just-in-time contextual assistant designed to help users identify sensitive information, summarize relevant privacy policies, and highlight potential risks when sharing information with LLMs. We evaluated the usability and usefulness of CLEAR across two example domains: ChatGPT and the Gemini plugin in Gmail. Our findings demonstrated that CLEAR is easy to use and improves users' understanding of data practices and privacy risks. We also discussed LLM's duality in posing and mitigating privacy risks, offering design and policy implications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14046v1">LLM-guided Instance-level Image Manipulation with Diffusion U-Net Cross-Attention Maps</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 Presented at BMVC 2024
    </div>
    <details class="paper-abstract">
      The advancement of text-to-image synthesis has introduced powerful generative models capable of creating realistic images from textual prompts. However, precise control over image attributes remains challenging, especially at the instance level. While existing methods offer some control through fine-tuning or auxiliary information, they often face limitations in flexibility and accuracy. To address these challenges, we propose a pipeline leveraging Large Language Models (LLMs), open-vocabulary detectors, cross-attention maps and intermediate activations of diffusion U-Net for instance-level image manipulation. Our method detects objects mentioned in the prompt and present in the generated image, enabling precise manipulation without extensive training or input masks. By incorporating cross-attention maps, our approach ensures coherence in manipulated images while controlling object positions. Our method enables precise manipulations at the instance level without fine-tuning or auxiliary information such as masks or bounding boxes. Code is available at https://github.com/Palandr123/DiffusionU-NetLLM
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13912v1">Analysis of Indic Language Capabilities in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 17 pages, 2 figures, 5 tables
    </div>
    <details class="paper-abstract">
      This report evaluates the performance of text-in text-out Large Language Models (LLMs) to understand and generate Indic languages. This evaluation is used to identify and prioritize Indic languages suited for inclusion in safety benchmarks. We conduct this study by reviewing existing evaluation studies and datasets; and a set of twenty-eight LLMs that support Indic languages. We analyze the LLMs on the basis of the training data, license for model and data, type of access and model developers. We also compare Indic language performance across evaluation datasets and find that significant performance disparities in performance across Indic languages. Hindi is the most widely represented language in models. While model performance roughly correlates with number of speakers for the top five languages, the assessment after that varies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.03797v2">NESTFUL: A Benchmark for Evaluating LLMs on Nested Sequences of API Calls</a></div>
    <div class="paper-meta">
      📅 2025-01-23
    </div>
    <details class="paper-abstract">
      The resurgence of autonomous agents built using large language models (LLMs) to solve complex real-world tasks has brought increased focus on LLMs' fundamental ability of tool or function calling. At the core of these agents, an LLM must plan, execute, and respond using external tools, APIs, and custom functions. Research on tool calling has gathered momentum, but evaluation benchmarks and datasets representing the complexity of the tasks have lagged behind. In this work, we focus on one such complexity, nested sequencing, with the goal of extending existing benchmarks and evaluation. Specifically, we present NESTFUL, a benchmark to evaluate LLMs on nested sequences of API calls, i.e., sequences where the output of one API call is passed as input to a subsequent call. NESTFUL contains 1800+ nested sequences where all the function calls are executable. Experimental results on multiple models and settings show that the best-performing model on the dataset has a full sequence match accuracy of 25% and win-rate of 34% necessitating a large scope for improvement in the nested sequencing aspect of function calling. Our analysis of these results provides possible future research directions for the community, in addition to a benchmark to track progress. We have released the NESTFUL dataset under the Apache 2.0 license at https://github.com/IBM/NESTFUL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13884v1">Exploring Finetuned Audio-LLM on Heart Murmur Features</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 5 pages, 1 figure, and 3 tables. Submitted to IEEE/ACM Conference on Connected Health: Applications, Systems , and Engineering Technologies
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) for audio have excelled in recognizing and analyzing human speech, music, and environmental sounds. However, their potential for understanding other types of sounds, particularly biomedical sounds, remains largely underexplored despite significant scientific interest. In this study, we focus on diagnosing cardiovascular diseases using phonocardiograms, i.e., heart sounds. Most existing deep neural network (DNN) paradigms are restricted to heart murmur classification (healthy vs unhealthy) and do not predict other acoustic features of the murmur such as timing, grading, harshness, pitch, and quality, which are important in helping physicians diagnose the underlying heart conditions. We propose to finetune an audio LLM, Qwen2-Audio, on the PhysioNet CirCor DigiScope phonocardiogram (PCG) dataset and evaluate its performance in classifying 11 expert-labeled murmur features. Additionally, we aim to achieve more noise-robust and generalizable system by exploring a preprocessing segmentation algorithm using an audio representation model, SSAMBA. Our results indicate that the LLM-based model outperforms state-of-the-art methods in 8 of the 11 features and performs comparably in the remaining 3. Moreover, the LLM successfully classifies long-tail murmur features with limited training data, a task that all previous methods have failed to classify. These findings underscore the potential of audio LLMs as assistants to human cardiologists in enhancing heart disease diagnosis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.19599v3">Take Caution in Using LLMs as Human Surrogates: Scylla Ex Machina</a></div>
    <div class="paper-meta">
      📅 2025-01-23
    </div>
    <details class="paper-abstract">
      Recent studies suggest large language models (LLMs) can exhibit human-like reasoning, aligning with human behavior in economic experiments, surveys, and political discourse. This has led many to propose that LLMs can be used as surrogates or simulations for humans in social science research. However, LLMs differ fundamentally from humans, relying on probabilistic patterns, absent the embodied experiences or survival objectives that shape human cognition. We assess the reasoning depth of LLMs using the 11-20 money request game. Nearly all advanced approaches fail to replicate human behavior distributions across many models. Causes of failure are diverse and unpredictable, relating to input language, roles, and safeguarding. These results advise caution when using LLMs to study human behavior or as surrogates or simulations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13802v1">Enhancing LLMs for Governance with Human Oversight: Evaluating and Aligning LLMs on Expert Classification of Climate Misinformation for Detecting False or Misleading Claims about Climate Change</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 Accepted to the AI Governance Workshop at AAAI 2025
    </div>
    <details class="paper-abstract">
      Climate misinformation is a problem that has the potential to be substantially aggravated by the development of Large Language Models (LLMs). In this study we evaluate the potential for LLMs to be part of the solution for mitigating online dis/misinformation rather than the problem. Employing a public expert annotated dataset and a curated sample of social media content we evaluate the performance of proprietary vs. open source LLMs on climate misinformation classification task, comparing them to existing climate-focused computer-assisted tools and expert assessments. Results show (1) state-of-the-art (SOTA) open-source models substantially under-perform in classifying climate misinformation compared to proprietary models, (2) existing climate-focused computer-assisted tools leveraging expert-annotated datasets continues to outperform many of proprietary models, including GPT-4o, and (3) demonstrate the efficacy and generalizability of fine-tuning GPT-3.5-turbo on expert annotated dataset in classifying claims about climate change at the equivalency of climate change experts with over 20 years of experience in climate communication. These findings highlight 1) the importance of incorporating human-oversight, such as incorporating expert-annotated datasets in training LLMs, for governance tasks that require subject-matter expertise like classifying climate misinformation, and 2) the potential for LLMs in facilitating civil society organizations to engage in various governance tasks such as classifying false or misleading claims in domains beyond climate change such as politics and health science.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13778v1">Explainable XR: Understanding User Behaviors of XR Environments using LLM-assisted Analytics Framework</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 11 pages, 8 figures. This is the author's version of the article that has been accepted for publication in IEEE Transactions on Visualization and Computer Graphics
    </div>
    <details class="paper-abstract">
      We present Explainable XR, an end-to-end framework for analyzing user behavior in diverse eXtended Reality (XR) environments by leveraging Large Language Models (LLMs) for data interpretation assistance. Existing XR user analytics frameworks face challenges in handling cross-virtuality - AR, VR, MR - transitions, multi-user collaborative application scenarios, and the complexity of multimodal data. Explainable XR addresses these challenges by providing a virtuality-agnostic solution for the collection, analysis, and visualization of immersive sessions. We propose three main components in our framework: (1) A novel user data recording schema, called User Action Descriptor (UAD), that can capture the users' multimodal actions, along with their intents and the contexts; (2) a platform-agnostic XR session recorder, and (3) a visual analytics interface that offers LLM-assisted insights tailored to the analysts' perspectives, facilitating the exploration and analysis of the recorded XR session data. We demonstrate the versatility of Explainable XR by demonstrating five use-case scenarios, in both individual and collaborative XR applications across virtualities. Our technical evaluation and user studies show that Explainable XR provides a highly usable analytics solution for understanding user actions and delivering multifaceted, actionable insights into user behaviors in immersive environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.14804v2">Can LLMs Solve longer Math Word Problems Better?</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 Accepted to ICLR 2025
    </div>
    <details class="paper-abstract">
      Math Word Problems (MWPs) play a vital role in assessing the capabilities of Large Language Models (LLMs), yet current research primarily focuses on questions with concise contexts. The impact of longer contexts on mathematical reasoning remains under-explored. This study pioneers the investigation of Context Length Generalizability (CoLeG), which refers to the ability of LLMs to solve MWPs with extended narratives. We introduce Extended Grade-School Math (E-GSM), a collection of MWPs featuring lengthy narratives, and propose two novel metrics to evaluate the efficacy and resilience of LLMs in tackling these problems. Our analysis of existing zero-shot prompting techniques with proprietary LLMs along with open-source LLMs reveals a general deficiency in CoLeG. To alleviate these issues, we propose tailored approaches for different categories of LLMs. For proprietary LLMs, we introduce a new instructional prompt designed to mitigate the impact of long contexts. For open-source LLMs, we develop a novel auxiliary task for fine-tuning to enhance CoLeG. Our comprehensive results demonstrate the effectiveness of our proposed methods, showing improved performance on E-GSM. Additionally, we conduct an in-depth analysis to differentiate the effects of semantic understanding and reasoning efficacy, showing that our methods improves the latter. We also establish the generalizability of our methods across several other MWP benchmarks. Our findings highlight the limitations of current LLMs and offer practical solutions correspondingly, paving the way for further exploration of model generalizability and training methodologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13746v1">EICopilot: Search and Explore Enterprise Information over Large-scale Knowledge Graphs with LLM-driven Agents</a></div>
    <div class="paper-meta">
      📅 2025-01-23
    </div>
    <details class="paper-abstract">
      The paper introduces EICopilot, an novel agent-based solution enhancing search and exploration of enterprise registration data within extensive online knowledge graphs like those detailing legal entities, registered capital, and major shareholders. Traditional methods necessitate text-based queries and manual subgraph explorations, often resulting in time-consuming processes. EICopilot, deployed as a chatbot via Baidu Enterprise Search, improves this landscape by utilizing Large Language Models (LLMs) to interpret natural language queries. This solution automatically generates and executes Gremlin scripts, providing efficient summaries of complex enterprise relationships. Distinct feature a data pre-processing pipeline that compiles and annotates representative queries into a vector database of examples for In-context learning (ICL), a comprehensive reasoning pipeline combining Chain-of-Thought with ICL to enhance Gremlin script generation for knowledge graph search and exploration, and a novel query masking strategy that improves intent recognition for heightened script accuracy. Empirical evaluations demonstrate the superior performance of EICopilot, including speed and accuracy, over baseline methods, with the \emph{Full Mask} variant achieving a syntax error rate reduction to as low as 10.00% and an execution correctness of up to 82.14%. These components collectively contribute to superior querying capabilities and summarization of intricate datasets, positioning EICopilot as a groundbreaking tool in the exploration and exploitation of large-scale knowledge graphs for enterprise information search.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13731v1">Pseudocode-Injection Magic: Enabling LLMs to Tackle Graph Computational Tasks</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 24 pages
    </div>
    <details class="paper-abstract">
      Graph computational tasks are inherently challenging and often demand the development of advanced algorithms for effective solutions. With the emergence of large language models (LLMs), researchers have begun investigating their potential to address these tasks. However, existing approaches are constrained by LLMs' limited capability to comprehend complex graph structures and their high inference costs, rendering them impractical for handling large-scale graphs. Inspired by human approaches to graph problems, we introduce a novel framework, PIE (Pseudocode-Injection-Enhanced LLM Reasoning for Graph Computational Tasks), which consists of three key steps: problem understanding, prompt design, and code generation. In this framework, LLMs are tasked with understanding the problem and extracting relevant information to generate correct code. The responsibility for analyzing the graph structure and executing the code is delegated to the interpreter. We inject task-related pseudocodes into the prompts to further assist the LLMs in generating efficient code. We also employ cost-effective trial-and-error techniques to ensure that the LLM-generated code executes correctly. Unlike other methods that require invoking LLMs for each individual test case, PIE only calls the LLM during the code generation phase, allowing the generated code to be reused and significantly reducing inference costs. Extensive experiments demonstrate that PIE outperforms existing baselines in terms of both accuracy and computational efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13687v1">Question Answering on Patient Medical Records with Private Fine-Tuned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-23
    </div>
    <details class="paper-abstract">
      Healthcare systems continuously generate vast amounts of electronic health records (EHRs), commonly stored in the Fast Healthcare Interoperability Resources (FHIR) standard. Despite the wealth of information in these records, their complexity and volume make it difficult for users to retrieve and interpret crucial health insights. Recent advances in Large Language Models (LLMs) offer a solution, enabling semantic question answering (QA) over medical data, allowing users to interact with their health records more effectively. However, ensuring privacy and compliance requires edge and private deployments of LLMs. This paper proposes a novel approach to semantic QA over EHRs by first identifying the most relevant FHIR resources for a user query (Task1) and subsequently answering the query based on these resources (Task2). We explore the performance of privately hosted, fine-tuned LLMs, evaluating them against benchmark models such as GPT-4 and GPT-4o. Our results demonstrate that fine-tuned LLMs, while 250x smaller in size, outperform GPT-4 family models by 0.55% in F1 score on Task1 and 42% on Meteor Task in Task2. Additionally, we examine advanced aspects of LLM usage, including sequential fine-tuning, model self-evaluation (narcissistic evaluation), and the impact of training data size on performance. The models and datasets are available here: https://huggingface.co/genloop
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13677v1">HumorReject: Decoupling LLM Safety from Refusal Prefix via A Little Humor</a></div>
    <div class="paper-meta">
      📅 2025-01-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) commonly rely on explicit refusal prefixes for safety, making them vulnerable to prefix injection attacks. We introduce HumorReject, a novel data-driven approach that fundamentally reimagines LLM safety by decoupling it from refusal prefixes through the use of humor as an indirect refusal strategy. Rather than explicitly rejecting harmful instructions, HumorReject responds with contextually appropriate humor that naturally defuses potentially dangerous requests while maintaining engaging interactions. Our approach effectively addresses the common "over-defense" issues in existing safety mechanisms, demonstrating superior robustness against various attack vectors while preserving natural and high-quality interactions on legitimate tasks. Our findings suggest that innovations at the data level are even more fundamental than the alignment algorithm itself in achieving effective LLM safety, opening new directions for developing more resilient and user-friendly AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13669v1">How to Complete Domain Tuning while Keeping General Ability in LLM: Adaptive Layer-wise and Element-wise Regularization</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit strong general-purpose language capabilities. However, fine-tuning these models on domain-specific tasks often leads to catastrophic forgetting, where the model overwrites or loses essential knowledge acquired during pretraining. This phenomenon significantly limits the broader applicability of LLMs. To address this challenge, we propose a novel approach to compute the element-wise importance of model parameters crucial for preserving general knowledge during fine-tuning. Our method utilizes a dual-objective optimization strategy: (1) regularization loss to retain the parameter crucial for general knowledge; (2) cross-entropy loss to adapt to domain-specific tasks. Additionally, we introduce layer-wise coefficients to account for the varying contributions of different layers, dynamically balancing the dual-objective optimization. Extensive experiments on scientific, medical, and physical tasks using GPT-J and LLaMA-3 demonstrate that our approach mitigates catastrophic forgetting while enhancing model adaptability. Compared to previous methods, our solution is approximately 20 times faster and requires only 10%-15% of the storage, highlighting the practical efficiency. The code will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13545v1">LLMs Can Plan Only If We Tell Them</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 ICLR 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated significant capabilities in natural language processing and reasoning, yet their effectiveness in autonomous planning has been under debate. While existing studies have utilized LLMs with external feedback mechanisms or in controlled environments for planning, these approaches often involve substantial computational and development resources due to the requirement for careful design and iterative backprompting. Moreover, even the most advanced LLMs like GPT-4 struggle to match human performance on standard planning benchmarks, such as the Blocksworld, without additional support. This paper investigates whether LLMs can independently generate long-horizon plans that rival human baselines. Our novel enhancements to Algorithm-of-Thoughts (AoT), which we dub AoT+, help achieve state-of-the-art results in planning benchmarks out-competing prior methods and human baselines all autonomously.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.11433v2">One Does Not Simply Meme Alone: Evaluating Co-Creativity Between LLMs and Humans in the Generation of Humor</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 to appear in: 30th International Conference on Intelligent User Interfaces IUI 25 March 2427 2025 Cagliari Italy
    </div>
    <details class="paper-abstract">
      Collaboration has been shown to enhance creativity, leading to more innovative and effective outcomes. While previous research has explored the abilities of Large Language Models (LLMs) to serve as co-creative partners in tasks like writing poetry or creating narratives, the collaborative potential of LLMs in humor-rich and culturally nuanced domains remains an open question. To address this gap, we conducted a user study to explore the potential of LLMs in co-creating memes - a humor-driven and culturally specific form of creative expression. We conducted a user study with three groups of 50 participants each: a human-only group creating memes without AI assistance, a human-AI collaboration group interacting with a state-of-the-art LLM model, and an AI-only group where the LLM autonomously generated memes. We assessed the quality of the generated memes through crowdsourcing, with each meme rated on creativity, humor, and shareability. Our results showed that LLM assistance increased the number of ideas generated and reduced the effort participants felt. However, it did not improve the quality of the memes when humans collaborated with LLM. Interestingly, memes created entirely by AI performed better than both human-only and human-AI collaborative memes in all areas on average. However, when looking at the top-performing memes, human-created ones were better in humor, while human-AI collaborations stood out in creativity and shareability. These findings highlight the complexities of human-AI collaboration in creative tasks. While AI can boost productivity and create content that appeals to a broad audience, human creativity remains crucial for content that connects on a deeper level.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.04454v2">Inner-Probe: Discovering Copyright-related Data Generation in LLM Architecture</a></div>
    <div class="paper-meta">
      📅 2025-01-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) utilize extensive knowledge databases and show powerful text generation ability. However, their reliance on high-quality copyrighted datasets raises concerns about copyright infringements in generated texts. Current research often employs prompt engineering or semantic classifiers to identify copyrighted content, but these approaches have two significant limitations: (1) Challenging to identify which specific sub-dataset (e.g., works from particular authors) influences an LLM's output. (2) Treating the entire training database as copyrighted, hence overlooking the inclusion of non-copyrighted training data. We propose InnerProbe, a lightweight framework designed to evaluate the influence of copyrighted sub-datasets on LLM-generated texts. Unlike traditional methods relying solely on text, we discover that the results of multi-head attention (MHA) during LLM output generation provide more effective information. Thus, InnerProbe performs sub-dataset contribution analysis using a lightweight LSTM-based network trained on MHA results in a supervised manner. Harnessing such a prior, InnerProbe enables non-copyrighted text detection through a concatenated global projector trained with unsupervised contrastive learning. InnerProbe demonstrates 3x improved efficiency compared to semantic model training in sub-dataset contribution analysis on Books3, achieves 15.04%-58.7% higher accuracy over baselines on the Pile, and delivers a 0.104 increase in AUC for non-copyrighted data filtering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13480v1">Adaptive Testing for LLM-Based Applications: A Diversity-based Approach</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      The recent surge of building software systems powered by Large Language Models (LLMs) has led to the development of various testing frameworks, primarily focused on treating prompt templates as the unit of testing. Despite the significant costs associated with test input execution and output assessment, the curation of optimized test suites is yet overlooked in these tools, which calls for tailored test selection or prioritization strategies. In this paper, we show that diversity-based testing techniques, such as Adaptive Random Testing (ART) with appropriate string distance metrics, can be effectively applied to the testing of prompt templates. Our proposed adaptive testing approach adjusts the conventional ART process to this context by selecting new test inputs based on scores derived from existing test suite and their labelling results. Our results, obtained using various implementations that explore several string-based distances, confirm that our approach enables the discovery of failures with reduced testing budgets and promotes the generation of more varied outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.19825v2">Concise Thoughts: Impact of Output Length on LLM Reasoning and Cost</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 Preprint version, under review
    </div>
    <details class="paper-abstract">
      Today's large language models (LLMs) can solve challenging question-answering tasks, and prompt engineering techniques, such as chain-of-thought (CoT), have gained attention for enhancing the explanation and correctness of outputs. However, many models and techniques tend to produce excessively verbose and lengthy answers, leading to issues with both conciseness and generation time. To address this, this paper analyzes the impact of output lengths on LLM inference pipelines by introducing and proposing novel metrics to evaluate the \textit{correct conciseness} of a model and related prompting techniques. Then, we examine the impact of controlling output length through a refined prompt engineering strategy, Constrained-CoT (CCoT), which encourages the model to produce more concise outputs. To better understand the effects of such a prompt, we also introduce two additional scores for analyzing the conciseness, measured in terms of redundancy and information flow in generated answers. Experiments on pretrained LLMs and multiple datasets demonstrate the benefits of the proposed metrics and the effectiveness of CCoT across different models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.14162v4">DIRAS: Efficient LLM Annotation of Document Relevance in Retrieval Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 NAACL 2025 Long
    </div>
    <details class="paper-abstract">
      Retrieval Augmented Generation (RAG) is widely employed to ground responses to queries on domain-specific documents. But do RAG implementations leave out important information when answering queries that need an integrated analysis of information (e.g., Tell me good news in the stock market today.)? To address these concerns, RAG developers need to annotate information retrieval (IR) data for their domain of interest, which is challenging because (1) domain-specific queries usually need nuanced definitions of relevance beyond shallow semantic relevance; and (2) human or GPT-4 annotation is costly and cannot cover all (query, document) pairs (i.e., annotation selection bias), thus harming the effectiveness in evaluating IR recall. To address these challenges, we propose DIRAS (Domain-specific Information Retrieval Annotation with Scalability), a manual-annotation-free schema that fine-tunes open-sourced LLMs to consider nuanced relevance definition and annotate (partial) relevance labels with calibrated relevance scores. Extensive evaluation shows that DIRAS enables smaller (8B) LLMs to achieve GPT-4-level performance on annotating and ranking unseen (query, document) pairs, and is helpful for real-world RAG development. All code, LLM generations, and human annotations can be found in \url{https://github.com/EdisonNi-hku/DIRAS}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.11885v3">Med-R$^2$: Crafting Trustworthy LLM Physicians through Retrieval and Reasoning of Evidence-Based Medicine</a></div>
    <div class="paper-meta">
      📅 2025-01-23
    </div>
    <details class="paper-abstract">
      In recent years, Large Language Models (LLMs) have exhibited remarkable capabilities in clinical scenarios. However, despite their potential, existing works face challenges when applying LLMs to medical settings. Strategies relying on training with medical datasets are highly cost-intensive and may suffer from outdated training data. Leveraging external knowledge bases is a suitable alternative, yet it faces obstacles such as limited retrieval precision and poor effectiveness in answer extraction. These issues collectively prevent LLMs from demonstrating the expected level of proficiency in mastering medical expertise. To address these challenges, we introduce Med-R^2, a novel LLM physician framework that adheres to the Evidence-Based Medicine (EBM) process, efficiently integrating retrieval mechanisms as well as the selection and reasoning processes of evidence, thereby enhancing the problem-solving capabilities of LLMs in healthcare scenarios and fostering a trustworthy LLM physician. Our comprehensive experiments indicate that Med-R^2 achieves a 14.87\% improvement over vanilla RAG methods and even a 3.59\% enhancement compared to fine-tuning strategies, without incurring additional training costs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13984v1">Comprehensive Modeling and Question Answering of Cancer Clinical Practice Guidelines using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-23
    </div>
    <details class="paper-abstract">
      The updated recommendations on diagnostic procedures and treatment pathways for a medical condition are documented as graphical flows in Clinical Practice Guidelines (CPGs). For effective use of the CPGs in helping medical professionals in the treatment decision process, it is necessary to fully capture the guideline knowledge, particularly the contexts and their relationships in the graph. While several existing works have utilized these guidelines to create rule bases for Clinical Decision Support Systems, limited work has been done toward directly capturing the full medical knowledge contained in CPGs. This work proposes an approach to create a contextually enriched, faithful digital representation of National Comprehensive Cancer Network (NCCN) Cancer CPGs in the form of graphs using automated extraction and node & relationship classification. We also implement semantic enrichment of the model by using Large Language Models (LLMs) for node classification, achieving an accuracy of 80.86% and 88.47% with zero-shot learning and few-shot learning, respectively. Additionally, we introduce a methodology for answering natural language questions with constraints to guideline text by leveraging LLMs to extract the relevant subgraph from the guideline knowledge base. By generating natural language answers based on subgraph paths and semantic information, we mitigate the risk of incorrect answers and hallucination associated with LLMs, ensuring factual accuracy in medical domain Question Answering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16454v2">Catastrophic Failure of LLM Unlearning via Quantization</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 25 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable proficiency in generating text, benefiting from extensive training on vast textual corpora. However, LLMs may also acquire unwanted behaviors from the diverse and sensitive nature of their training data, which can include copyrighted and private content. Machine unlearning has been introduced as a viable solution to remove the influence of such problematic content without the need for costly and time-consuming retraining. This process aims to erase specific knowledge from LLMs while preserving as much model utility as possible. Despite the effectiveness of current unlearning methods, little attention has been given to whether existing unlearning methods for LLMs truly achieve forgetting or merely hide the knowledge, which current unlearning benchmarks fail to detect. This paper reveals that applying quantization to models that have undergone unlearning can restore the "forgotten" information. To thoroughly evaluate this phenomenon, we conduct comprehensive experiments using various quantization techniques across multiple precision levels. We find that for unlearning methods with utility constraints, the unlearned model retains an average of 21\% of the intended forgotten knowledge in full precision, which significantly increases to 83\% after 4-bit quantization. ... Our code is available at: \href{https://github.com/zzwjames/FailureLLMUnlearning}{https://github.com/zzwjames/FailureLLMUnlearning}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.11478v2">Each Graph is a New Language: Graph Learning with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-23
    </div>
    <details class="paper-abstract">
      Recent efforts leverage Large Language Models (LLMs) for modeling text-attributed graph structures in node classification tasks. These approaches describe graph structures for LLMs to understand or aggregate LLM-generated textual attribute embeddings through graph structure. However, these approaches face two main limitations in modeling graph structures with LLMs. (i) Graph descriptions become verbose in describing high-order graph structure. (ii) Textual attributes alone do not contain adequate graph structure information. It is challenging to model graph structure concisely and adequately with LLMs. LLMs lack built-in mechanisms to model graph structures directly. They also struggle with complex long-range dependencies between high-order nodes and target nodes. Inspired by the observation that LLMs pre-trained on one language can achieve exceptional performance on another with minimal additional training, we propose \textbf{G}raph-\textbf{D}efined \textbf{L}anguage for \textbf{L}arge \textbf{L}anguage \textbf{M}odel (GDL4LLM). This novel framework enables LLMs to transfer their powerful language understanding capabilities to graph-structured data. GDL4LLM translates graphs into a graph language corpus instead of graph descriptions and pre-trains LLMs on this corpus to adequately understand graph structures. During fine-tuning, this corpus describes the structural information of target nodes concisely with only a few tokens. By treating graphs as a new language, GDL4LLM enables LLMs to model graph structures adequately and concisely for node classification tasks. Extensive experiments on three real-world datasets demonstrate that GDL4LLM outperforms description-based and textual attribute embeddings-based baselines by efficiently modeling different orders of graph structure with LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.03518v4">Improving LLM Abilities in Idiomatic Translation</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 Preprint for LoResLM Workshop at COLING 2025
    </div>
    <details class="paper-abstract">
      For large language models (LLMs) like NLLB and GPT, translating idioms remains a challenge. Our goal is to enhance translation fidelity by improving LLM processing of idiomatic language while preserving the original linguistic style. This has a significant social impact, as it preserves cultural nuances and ensures translated texts retain their intent and emotional resonance, fostering better cross-cultural communication. Previous work has utilized knowledge bases like IdiomKB by providing the LLM with the meaning of an idiom to use in translation. Although this method yielded better results than a direct translation, it is still limited in its ability to preserve idiomatic writing style across languages. In this research, we expand upon the knowledge base to find corresponding idioms in the target language. Our research performs translations using two methods: The first method employs the SentenceTransformers model to semantically generate cosine similarity scores between the meanings of the original and target language idioms, selecting the best idiom (Cosine Similarity method). The second method uses an LLM to find a corresponding idiom in the target language for use in the translation (LLM-generated idiom method). As a baseline, we performed a direct translation without providing additional information. Human evaluations on the English -> Chinese, and Chinese -> English show the Cosine Similarity Lookup method out-performed others in all GPT4o translations. To further build upon IdiomKB, we developed a low-resource Urdu dataset containing Urdu idioms and their translations. Despite dataset limitations, the Cosine Similarity Lookup method shows promise, potentially overcoming language barriers and enabling the exploration of diverse literary works in Chinese and Urdu.(LoResLM @ COLING Preprint)
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13331v1">Qrazor: Reliable and effortless 4-bit llm quantization by significant data razoring</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 19 pages
    </div>
    <details class="paper-abstract">
      Large-scale language models (LLMs) have demonstrated outstanding performance in language processing tasks, yet their deployment is often hindered by high memory demands and computational complexity. Although low-bit quantization techniques, such as 4-bit quantization, present a potential solution, they frequently lead to significant accuracy degradation or require substantial effort for such aggressive quantization approaches. To overcome these challenges, we introduce QRazor, a reliable and effortless quantization scheme designed to enable 4-bit quantization for weights, activations, and KV cache in transformer-based LLMs. The scheme involves two main stages: quantization and compression. During the quantization stage, weights, activations, and KV cache values are quantized with wider 8 or 16-bit integers as a basis to achieve nearly identical accuracy to the original full-precision LLM models, using the absolute max scaling. Subsequently, all data are compressed to 4-bit using our proposed significant data razoring (SDR) technique, which retains only the four most salient bits while discarding the others. Furthermore, we present an integer-based arithmetic unit dedicated to QRazor, enabling direct low-precision arithmetic operations without decompressing the SDR data. Despite the reduced quantization effort, QRazor achieves LLM accuracies better or comparable to state-of-the-art 4-bit methods. By also validating the hardware efficiency, our decompression-free arithmetic unit achieves 61.2% and 57.8% reduction in area and power consumption, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16372v1">Low-Rank Adapters Meet Neural Architecture Search for LLM Compression</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 AAAI-25 Workshop on Connecting Low-rank Representations in AI
    </div>
    <details class="paper-abstract">
      The rapid expansion of Large Language Models (LLMs) has posed significant challenges regarding the computational resources required for fine-tuning and deployment. Recent advancements in low-rank adapters have demonstrated their efficacy in parameter-efficient fine-tuning (PEFT) of these models. This retrospective paper comprehensively discusses innovative approaches that synergize low-rank representations with Neural Architecture Search (NAS) techniques, particularly weight-sharing super-networks. Robust solutions for compressing and fine-tuning large pre-trained models are developed by integrating these methodologies. Our analysis highlights the potential of these combined strategies to democratize the use of LLMs, making them more accessible for deployment in resource-constrained environments. The resulting models exhibit reduced memory footprints and faster inference times, paving the way for more practical and scalable applications of LLMs. Models and code are available at https://github.com/IntelLabs/Hardware-Aware-Automated-Machine-Learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.10245v3">From Text to Emoji: How PEFT-Driven Personality Manipulation Unleashes the Emoji Potential in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 Findings paper of NAACL 2025 and NeurIPS 2024 Workshop on Behavioral Machine Learning
    </div>
    <details class="paper-abstract">
      The manipulation of the personality traits of large language models (LLMs) has emerged as a key area of research. Methods like prompt-based In-Context Knowledge Editing (IKE) and gradient-based Model Editor Networks (MEND) have been explored but show irregularity and variability; IKE depends on the prompt, leading to variability and sensitivity, while MEND yields inconsistent and gibberish outputs. To address this, we employed Opinion QA Based Parameter-Efficient Fine-Tuning (PEFT), specifically Quantized Low-Rank Adaptation (QLoRA), to manipulate the Big Five personality traits: Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism. After PEFT, models such as Mistral-7B-Instruct and LLaMA-2-7B-chat began generating emojis, even though no emojis were present in the PEFT data. For instance, LLaMA-2-7B-chat generated emojis in 99.5% of extraversion-related test instances, while Mistral-7B-Instruct did so in 92.5% of openness-related test instances. ICL Explainability analysis indicated that the LLMs used emojis intentionally to express these traits. Mechanistic Interpretability analysis showed that this latent behaviour of LLMs could be traced to specific neurons that became activated or amplified after PEFT. This paper provides a number of novel contributions. First, introducing an Opinion QA dataset for PEFT-driven personality manipulation; second, developing metric models to benchmark LLM personality traits; third, demonstrating PEFT's superiority over IKE in personality manipulation; and finally, analysing and validating emoji usage through explainability methods such as Mechanistic Interpretability and In-context learning Explainability methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.14876v2">Precise and Robust Sidewalk Detection: Leveraging Ensemble Learning to Surpass LLM Limitations in Urban Environments</a></div>
    <div class="paper-meta">
      📅 2025-01-23
    </div>
    <details class="paper-abstract">
      This study aims to compare the effectiveness of a robust ensemble model with the state-of-the-art ONE-PEACE Large Language Model (LLM) for accurate detection of sidewalks. Accurate sidewalk detection is crucial in improving road safety and urban planning. The study evaluated the model's performance on Cityscapes, Ade20k, and the Boston Dataset. The results showed that the ensemble model performed better than the individual models, achieving mean Intersection Over Union (mIOU) scores of 93.1\%, 90.3\%, and 90.6\% on these datasets under ideal conditions. Additionally, the ensemble model maintained a consistent level of performance even in challenging conditions such as Salt-and-Pepper and Speckle noise, with only a gradual decrease in efficiency observed. On the other hand, the ONE-PEACE LLM performed slightly better than the ensemble model in ideal scenarios but experienced a significant decline in performance under noisy conditions. These findings demonstrate the robustness and reliability of the ensemble model, making it a valuable asset for improving urban infrastructure related to road safety and curb space management. This study contributes positively to the broader context of urban health and mobility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13299v1">Hypothesis Generation for Materials Discovery and Design Using Goal-Driven and Constraint-Guided LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 Accepted in NAACL 2025
    </div>
    <details class="paper-abstract">
      Materials discovery and design are essential for advancing technology across various industries by enabling the development of application-specific materials. Recent research has leveraged Large Language Models (LLMs) to accelerate this process. We explore the potential of LLMs to generate viable hypotheses that, once validated, can expedite materials discovery. Collaborating with materials science experts, we curated a novel dataset from recent journal publications, featuring real-world goals, constraints, and methods for designing real-world applications. Using this dataset, we test LLM-based agents that generate hypotheses for achieving given goals under specific constraints. To assess the relevance and quality of these hypotheses, we propose a novel scalable evaluation metric that emulates the process a materials scientist would use to evaluate a hypothesis critically. Our curated dataset, proposed method, and evaluation framework aim to advance future research in accelerating materials discovery and design with LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.02076v7">LongGenBench: Benchmarking Long-Form Generation in Long Context LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 ICLR 2025; Github: https://github.com/mozhu621/LongGenBench/
    </div>
    <details class="paper-abstract">
      Current benchmarks like Needle-in-a-Haystack (NIAH), Ruler, and Needlebench focus on models' ability to understand long-context input sequences but fail to capture a critical dimension: the generation of high-quality long-form text. Applications such as design proposals, technical documentation, and creative writing rely on coherent, instruction-following outputs over extended sequences - a challenge that existing benchmarks do not adequately address. To fill this gap, we introduce LongGenBench, a novel benchmark designed to rigorously evaluate large language models' (LLMs) ability to generate long text while adhering to complex instructions. Through tasks requiring specific events or constraints within generated text, LongGenBench evaluates model performance across four distinct scenarios, three instruction types, and two generation-lengths (16K and 32K tokens). Our evaluation of ten state-of-the-art LLMs reveals that, despite strong results on Ruler, all models struggled with long text generation on LongGenBench, particularly as text length increased. This suggests that current LLMs are not yet equipped to meet the demands of real-world, long-form text generation.
    </details>
</div>
