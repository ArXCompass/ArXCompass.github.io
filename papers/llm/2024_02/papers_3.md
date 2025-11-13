# llm - 2024_02

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.17072v2">SemScore: Automated Evaluation of Instruction-Tuned LLMs based on Semantic Textual Similarity</a></div>
    <div class="paper-meta">
      📅 2024-02-05
    </div>
    <details class="paper-abstract">
      Instruction-tuned Large Language Models (LLMs) have recently showcased remarkable advancements in their ability to generate fitting responses to natural language instructions. However, many current works rely on manual evaluation to judge the quality of generated responses. Since such manual evaluation is time-consuming, it does not easily scale to the evaluation of multiple models and model variants. In this short paper, we propose a straightforward but remarkably effective evaluation metric called SemScore, in which we directly compare model outputs to gold target responses using semantic textual similarity (STS). We conduct a comparative evaluation of the model outputs of 12 prominent instruction-tuned LLMs using 8 widely-used evaluation metrics for text generation. We find that our proposed SemScore metric outperforms all other, in many cases more complex, evaluation metrics in terms of correlation to human evaluation. These findings indicate the utility of our proposed metric for the evaluation of instruction-tuned LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.02716v1">Understanding the planning of LLM agents: A survey</a></div>
    <div class="paper-meta">
      📅 2024-02-05
      | 💬 9 pages, 2 tables, 2 figures
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) have shown significant intelligence, the progress to leverage LLMs as planning modules of autonomous agents has attracted more attention. This survey provides the first systematic view of LLM-based agents planning, covering recent works aiming to improve planning ability. We provide a taxonomy of existing works on LLM-Agent planning, which can be categorized into Task Decomposition, Plan Selection, External Module, Reflection and Memory. Comprehensive analyses are conducted for each direction, and further challenges for the field of research are discussed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.09675v1">Open-Universe Indoor Scene Generation using LLM Program Synthesis and Uncurated Object Databases</a></div>
    <div class="paper-meta">
      📅 2024-02-05
      | 💬 See ancillary files for link to supplemental material
    </div>
    <details class="paper-abstract">
      We present a system for generating indoor scenes in response to text prompts. The prompts are not limited to a fixed vocabulary of scene descriptions, and the objects in generated scenes are not restricted to a fixed set of object categories -- we call this setting indoor scene generation. Unlike most prior work on indoor scene generation, our system does not require a large training dataset of existing 3D scenes. Instead, it leverages the world knowledge encoded in pre-trained large language models (LLMs) to synthesize programs in a domain-specific layout language that describe objects and spatial relations between them. Executing such a program produces a specification of a constraint satisfaction problem, which the system solves using a gradient-based optimization scheme to produce object positions and orientations. To produce object geometry, the system retrieves 3D meshes from a database. Unlike prior work which uses databases of category-annotated, mutually-aligned meshes, we develop a pipeline using vision-language models (VLMs) to retrieve meshes from massive databases of un-annotated, inconsistently-aligned meshes. Experimental evaluations show that our system outperforms generative models trained on 3D data for traditional, closed-universe scene generation tasks; it also outperforms a recent LLM-based layout generation method on open-universe scene generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.02656v1">RACER: An LLM-powered Methodology for Scalable Analysis of Semi-structured Mental Health Interviews</a></div>
    <div class="paper-meta">
      📅 2024-02-05
    </div>
    <details class="paper-abstract">
      Semi-structured interviews (SSIs) are a commonly employed data-collection method in healthcare research, offering in-depth qualitative insights into subject experiences. Despite their value, the manual analysis of SSIs is notoriously time-consuming and labor-intensive, in part due to the difficulty of extracting and categorizing emotional responses, and challenges in scaling human evaluation for large populations. In this study, we develop RACER, a Large Language Model (LLM) based expert-guided automated pipeline that efficiently converts raw interview transcripts into insightful domain-relevant themes and sub-themes. We used RACER to analyze SSIs conducted with 93 healthcare professionals and trainees to assess the broad personal and professional mental health impacts of the COVID-19 crisis. RACER achieves moderately high agreement with two human evaluators (72%), which approaches the human inter-rater agreement (77%). Interestingly, LLMs and humans struggle with similar content involving nuanced emotional, ambivalent/dialectical, and psychological statements. Our study highlights the opportunities and challenges in using LLMs to improve research efficiency and opens new avenues for scalable analysis of SSIs in healthcare research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.02643v1">LLM-Enhanced Data Management</a></div>
    <div class="paper-meta">
      📅 2024-02-04
    </div>
    <details class="paper-abstract">
      Machine learning (ML) techniques for optimizing data management problems have been extensively studied and widely deployed in recent five years. However traditional ML methods have limitations on generalizability (adapting to different scenarios) and inference ability (understanding the context). Fortunately, large language models (LLMs) have shown high generalizability and human-competitive abilities in understanding context, which are promising for data management tasks (e.g., database diagnosis, database tuning). However, existing LLMs have several limitations: hallucination, high cost, and low accuracy for complicated tasks. To address these challenges, we design LLMDB, an LLM-enhanced data management paradigm which has generalizability and high inference ability while avoiding hallucination, reducing LLM cost, and achieving high accuracy. LLMDB embeds domain-specific knowledge to avoid hallucination by LLM fine-tuning and prompt engineering. LLMDB reduces the high cost of LLMs by vector databases which provide semantic search and caching abilities. LLMDB improves the task accuracy by LLM agent which provides multiple-round inference and pipeline executions. We showcase three real-world scenarios that LLMDB can well support, including query rewrite, database diagnosis and data analytics. We also summarize the open research challenges of LLMDB.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.07924v1">IllusionX: An LLM-powered mixed reality personal companion</a></div>
    <div class="paper-meta">
      📅 2024-02-04
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      Mixed Reality (MR) and Artificial Intelligence (AI) are increasingly becoming integral parts of our daily lives. Their applications range in fields from healthcare to education to entertainment. MR has opened a new frontier for such fields as well as new methods of enhancing user engagement. In this paper, We propose a new system one that combines the power of Large Language Models (LLMs) and mixed reality (MR) to provide a personalized companion for educational purposes. We present an overview of its structure and components as well tests to measure its performance. We found that our system is better in generating coherent information, however it's rather limited by the documents provided to it. This interdisciplinary approach aims to provide a better user experience and enhance user engagement. The user can interact with the system through a custom-design smart watch, smart glasses and a mobile app.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.05123v1">A Survey on Data Selection for LLM Instruction Tuning</a></div>
    <div class="paper-meta">
      📅 2024-02-04
    </div>
    <details class="paper-abstract">
      Instruction tuning is a vital step of training large language models (LLM), so how to enhance the effect of instruction tuning has received increased attention. Existing works indicate that the quality of the dataset is more crucial than the quantity during instruction tuning of LLM. Therefore, recently a lot of studies focus on exploring the methods of selecting high-quality subset from instruction datasets, aiming to reduce training costs and enhance the instruction-following capabilities of LLMs. This paper presents a comprehensive survey on data selection for LLM instruction tuning. Firstly, we introduce the wildly used instruction datasets. Then, we propose a new taxonomy of the data selection methods and provide a detailed introduction of recent advances,and the evaluation strategies and results of data selection methods are also elaborated in detail. Finally, we emphasize the open challenges and present new frontiers of this task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.02243v1">Language Writ Large: LLMs, ChatGPT, Grounding, Meaning and Understanding</a></div>
    <div class="paper-meta">
      📅 2024-02-03
      | 💬 48 pages, 25 references
    </div>
    <details class="paper-abstract">
      Apart from what (little) OpenAI may be concealing from us, we all know (roughly) how ChatGPT works (its huge text database, its statistics, its vector representations, and their huge number of parameters, its next-word training, and so on). But none of us can say (hand on heart) that we are not surprised by what ChatGPT has proved to be able to do with these resources. This has even driven some of us to conclude that ChatGPT actually understands. It is not true that it understands. But it is also not true that we understand how it can do what it can do. I will suggest some hunches about benign biases: convergent constraints that emerge at LLM scale that may be helping ChatGPT do so much better than we would have expected. These biases are inherent in the nature of language itself, at LLM scale, and they are closely linked to what it is that ChatGPT lacks, which is direct sensorimotor grounding to connect its words to their referents and its propositions to their meanings. These convergent biases are related to (1) the parasitism of indirect verbal grounding on direct sensorimotor grounding, (2) the circularity of verbal definition, (3) the mirroring of language production and comprehension, (4) iconicity in propositions at LLM scale, (5) computational counterparts of human categorical perception in category learning by neural nets, and perhaps also (6) a conjecture by Chomsky about the laws of thought. The exposition will be in the form of a dialogue with ChatGPT-4.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.02167v1">Vi(E)va LLM! A Conceptual Stack for Evaluating and Interpreting Generative AI-based Visualizations</a></div>
    <div class="paper-meta">
      📅 2024-02-03
    </div>
    <details class="paper-abstract">
      The automatic generation of visualizations is an old task that, through the years, has shown more and more interest from the research and practitioner communities. Recently, large language models (LLM) have become an interesting option for supporting generative tasks related to visualization, demonstrating initial promising results. At the same time, several pitfalls, like the multiple ways of instructing an LLM to generate the desired result, the different perspectives leading the generation (code-based, image-based, grammar-based), and the presence of hallucinations even for the visualization generation task, make their usage less affordable than expected. Following similar initiatives for benchmarking LLMs, this paper copes with the problem of modeling the evaluation of a generated visualization through an LLM. We propose a theoretical evaluation stack, EvaLLM, that decomposes the evaluation effort in its atomic components, characterizes their nature, and provides an overview of how to implement and interpret them. We also designed and implemented an evaluation platform that provides a benchmarking resource for the visualization generation task. The platform supports automatic and manual scoring conducted by multiple assessors to support a fine-grained and semantic evaluation based on the EvaLLM stack. Two case studies on GPT3.5-turbo with Code Interpreter and Llama2-70-b models show the benefits of EvaLLM and illustrate interesting results on the current state-of-the-art LLM-generated visualizations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.02135v1">Do Moral Judgment and Reasoning Capability of LLMs Change with Language? A Study using the Multilingual Defining Issues Test</a></div>
    <div class="paper-meta">
      📅 2024-02-03
      | 💬 Accepted to EACL 2024 (main)
    </div>
    <details class="paper-abstract">
      This paper explores the moral judgment and moral reasoning abilities exhibited by Large Language Models (LLMs) across languages through the Defining Issues Test. It is a well known fact that moral judgment depends on the language in which the question is asked. We extend the work of beyond English, to 5 new languages (Chinese, Hindi, Russian, Spanish and Swahili), and probe three LLMs -- ChatGPT, GPT-4 and Llama2Chat-70B -- that shows substantial multilingual text processing and generation abilities. Our study shows that the moral reasoning ability for all models, as indicated by the post-conventional score, is substantially inferior for Hindi and Swahili, compared to Spanish, Russian, Chinese and English, while there is no clear trend for the performance of the latter four languages. The moral judgments too vary considerably by the language.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.02057v1">Break the Sequential Dependency of LLM Inference Using Lookahead Decoding</a></div>
    <div class="paper-meta">
      📅 2024-02-03
    </div>
    <details class="paper-abstract">
      Autoregressive decoding of large language models (LLMs) is memory bandwidth bounded, resulting in high latency and significant wastes of the parallel processing power of modern accelerators. Existing methods for accelerating LLM decoding often require a draft model (e.g., speculative decoding), which is nontrivial to obtain and unable to generalize. In this paper, we introduce Lookahead decoding, an exact, parallel decoding algorithm that accelerates LLM decoding without needing auxiliary models or data stores. It allows trading per-step log(FLOPs) to reduce the number of total decoding steps, is more parallelizable on single or multiple modern accelerators, and is compatible with concurrent memory-efficient attention (e.g., FlashAttention). Our implementation of Lookahead decoding can speed up autoregressive decoding by up to 1.8x on MT-bench and 4x with strong scaling on multiple GPUs in code completion tasks. Our code is avialable at https://github.com/hao-ai-lab/LookaheadDecoding
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.02008v1">How well do LLMs cite relevant medical references? An evaluation framework and analyses</a></div>
    <div class="paper-meta">
      📅 2024-02-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are currently being used to answer medical questions across a variety of clinical domains. Recent top-performing commercial LLMs, in particular, are also capable of citing sources to support their responses. In this paper, we ask: do the sources that LLMs generate actually support the claims that they make? To answer this, we propose three contributions. First, as expert medical annotations are an expensive and time-consuming bottleneck for scalable evaluation, we demonstrate that GPT-4 is highly accurate in validating source relevance, agreeing 88% of the time with a panel of medical doctors. Second, we develop an end-to-end, automated pipeline called \textit{SourceCheckup} and use it to evaluate five top-performing LLMs on a dataset of 1200 generated questions, totaling over 40K pairs of statements and sources. Interestingly, we find that between ~50% to 90% of LLM responses are not fully supported by the sources they provide. We also evaluate GPT-4 with retrieval augmented generation (RAG) and find that, even still, around 30\% of individual statements are unsupported, while nearly half of its responses are not fully supported. Third, we open-source our curated dataset of medical questions and expert annotations for future evaluations. Given the rapid pace of LLM development and the potential harms of incorrect or outdated medical information, it is crucial to also understand and quantify their capability to produce relevant, trustworthy medical references.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.00715v2">Intent Assurance using LLMs guided by Intent Drift</a></div>
    <div class="paper-meta">
      📅 2024-02-02
    </div>
    <details class="paper-abstract">
      Intent-Based Networking (IBN) presents a paradigm shift for network management, by promising to align intents and business objectives with network operations--in an automated manner. However, its practical realization is challenging: 1) processing intents, i.e., translate, decompose and identify the logic to fulfill the intent, and 2) intent conformance, that is, considering dynamic networks, the logic should be adequately adapted to assure intents. To address the latter, intent assurance is tasked with continuous verification and validation, including taking the necessary actions to align the operational and target states. In this paper, we define an assurance framework that allows us to detect and act when intent drift occurs. To do so, we leverage AI-driven policies, generated by Large Language Models (LLMs) which can quickly learn the necessary in-context requirements, and assist with the fulfillment and assurance of intents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.09582v2">Fabricator: An Open Source Toolkit for Generating Labeled Training Data with Teacher LLMs</a></div>
    <div class="paper-meta">
      📅 2024-02-02
      | 💬 3 Figures and 2 Tables
    </div>
    <details class="paper-abstract">
      Most NLP tasks are modeled as supervised learning and thus require labeled training data to train effective models. However, manually producing such data at sufficient quality and quantity is known to be costly and time-intensive. Current research addresses this bottleneck by exploring a novel paradigm called zero-shot learning via dataset generation. Here, a powerful LLM is prompted with a task description to generate labeled data that can be used to train a downstream NLP model. For instance, an LLM might be prompted to "generate 500 movie reviews with positive overall sentiment, and another 500 with negative sentiment." The generated data could then be used to train a binary sentiment classifier, effectively leveraging an LLM as a teacher to a smaller student model. With this demo, we introduce Fabricator, an open-source Python toolkit for dataset generation. Fabricator implements common dataset generation workflows, supports a wide range of downstream NLP tasks (such as text classification, question answering, and entity recognition), and is integrated with well-known libraries to facilitate quick experimentation. With Fabricator, we aim to support researchers in conducting reproducible dataset generation experiments using LLMs and help practitioners apply this approach to train models for downstream tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.01874v1">The RL/LLM Taxonomy Tree: Reviewing Synergies Between Reinforcement Learning and Large Language Models</a></div>
    <div class="paper-meta">
      📅 2024-02-02
      | 💬 30 pages (including bibliography), 1 figure, 7 tables
    </div>
    <details class="paper-abstract">
      In this work, we review research studies that combine Reinforcement Learning (RL) and Large Language Models (LLMs), two areas that owe their momentum to the development of deep neural networks. We propose a novel taxonomy of three main classes based on the way that the two model types interact with each other. The first class, RL4LLM, includes studies where RL is leveraged to improve the performance of LLMs on tasks related to Natural Language Processing. L4LLM is divided into two sub-categories depending on whether RL is used to directly fine-tune an existing LLM or to improve the prompt of the LLM. In the second class, LLM4RL, an LLM assists the training of an RL model that performs a task that is not inherently related to natural language. We further break down LLM4RL based on the component of the RL training framework that the LLM assists or replaces, namely reward shaping, goal generation, and policy function. Finally, in the third class, RL+LLM, an LLM and an RL agent are embedded in a common planning framework without either of them contributing to training or fine-tuning of the other. We further branch this class to distinguish between studies with and without natural language feedback. We use this taxonomy to explore the motivations behind the synergy of LLMs and RL and explain the reasons for its success, while pinpointing potential shortcomings and areas where further research is needed, as well as alternative methodologies that serve the same goal.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.14279v4">Two Failures of Self-Consistency in the Multi-Step Reasoning of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-02-02
      | 💬 Accepted to TMLR: https://openreview.net/forum?id=5nBqY1y96B
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved widespread success on a variety of in-context few-shot tasks, but this success is typically evaluated via correctness rather than consistency. We argue that self-consistency is an important criteria for valid multi-step reasoning in tasks where the solution is composed of the answers to multiple sub-steps. We propose two types of self-consistency that are particularly important for multi-step reasoning -- hypothetical consistency (a model's ability to predict what its output would be in a hypothetical other context) and compositional consistency (consistency of a model's final outputs when intermediate sub-steps are replaced with the model's outputs for those steps). We demonstrate that multiple variants of the GPT-3/-4 models exhibit poor consistency rates across both types of consistency on a variety of tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.07920v1">Exploring patient trust in clinical advice from AI-driven LLMs like ChatGPT for self-diagnosis</a></div>
    <div class="paper-meta">
      📅 2024-02-02
    </div>
    <details class="paper-abstract">
      Trustworthy clinical advice is crucial but burdensome when seeking health support from professionals. Inaccessibility and financial burdens present obstacles to obtaining professional clinical advice, even when healthcare is available. Consequently, individuals often resort to self-diagnosis, utilizing medical materials to validate the health conditions of their families and friends. However, the convenient method of self-diagnosis requires a commitment to learning and is often not effective, presenting risks when individuals seek self-care approaches or treatment strategies without professional guidance. Artificial Intelligence (AI), supported by Large Language Models (LLM), may become a powerful yet risky self-diagnosis tool for clinical advice due to the hallucination of LLM, where it produces inaccurate yet deceiving information. Thus, can we trust the clinical advice from AI-driven LLMs like ChatGPT like ChatGPT4 for self-diagnosis? We examined this issue through a think-aloud observation: a patient uses GPT4 for self-diagnosis and clinical advice while a doctor assesses ChatGPT responses with their own expertise. After that, we conducted a semi-structured interview with the patient to understand their trust in AI-driven LLMs for clinical advice. we have concluded that the confounding factors influencing a patient's trust revolve around their competency-evaluation. Essentially, trust is equated with efficacy, which is determined by whether decisions made based on the AI agent's clinical advice and suggestion will effectively achieve the patient health goals. Patients tend to trust doctors more than AI agents due to this strategy, believing that educated, authorized doctors can provide effective medical guidance. This competency-based trust also explains why patients often perceive more experienced doctors as more trustworthy compared to less experienced ones.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.01812v1">Distilling LLMs' Decomposition Abilities into Compact Language Models</a></div>
    <div class="paper-meta">
      📅 2024-02-02
      | 💬 https://github.com/DT6A/GSM8K-AI-SubQ
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated proficiency in their reasoning abilities, yet their large size presents scalability challenges and limits any further customization. In contrast, compact models offer customized training but often fall short in solving complex reasoning tasks. This study focuses on distilling the LLMs' decomposition skills into compact models using offline reinforcement learning. We leverage the advancements in the LLM`s capabilities to provide feedback and generate a specialized task-specific dataset for training compact models. The development of an AI-generated dataset and the establishment of baselines constitute the primary contributions of our work, underscoring the potential of compact models in replicating complex problem-solving skills.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.11855v2">Evil Geniuses: Delving into the Safety of LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2024-02-02
      | 💬 11 pages
    </div>
    <details class="paper-abstract">
      Rapid advancements in large language models (LLMs) have revitalized in LLM-based agents, exhibiting impressive human-like behaviors and cooperative capabilities in various scenarios. However, these agents also bring some exclusive risks, stemming from the complexity of interaction environments and the usability of tools. This paper delves into the safety of LLM-based agents from three perspectives: agent quantity, role definition, and attack level. Specifically, we initially propose to employ a template-based attack strategy on LLM-based agents to find the influence of agent quantity. In addition, to address interaction environment and role specificity issues, we introduce Evil Geniuses (EG), an effective attack method that autonomously generates prompts related to the original role to examine the impact across various role definitions and attack levels. EG leverages Red-Blue exercises, significantly improving the generated prompt aggressiveness and similarity to original roles. Our evaluations on CAMEL, Metagpt and ChatDev based on GPT-3.5 and GPT-4, demonstrate high success rates. Extensive evaluation and discussion reveal that these agents are less robust, prone to more harmful behaviors, and capable of generating stealthier content than LLMs, highlighting significant safety challenges and guiding future research. Our code is available at https://github.com/T1aNS1R/Evil-Geniuses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.01158v1">LLM-Detector: Improving AI-Generated Chinese Text Detection with Open-Source LLM Instruction Tuning</a></div>
    <div class="paper-meta">
      📅 2024-02-02
      | 💬 17 pages, 13 tables, 7 figures
    </div>
    <details class="paper-abstract">
      ChatGPT and other general large language models (LLMs) have achieved remarkable success, but they have also raised concerns about the misuse of AI-generated texts. Existing AI-generated text detection models, such as based on BERT and RoBERTa, are prone to in-domain over-fitting, leading to poor out-of-domain (OOD) detection performance. In this paper, we first collected Chinese text responses generated by human experts and 9 types of LLMs, for which to multiple domains questions, and further created a dataset that mixed human-written sentences and sentences polished by LLMs. We then proposed LLM-Detector, a novel method for both document-level and sentence-level text detection through Instruction Tuning of LLMs. Our method leverages the wealth of knowledge LLMs acquire during pre-training, enabling them to detect the text they generate. Instruction tuning aligns the model's responses with the user's expected text detection tasks. Experimental results show that previous methods struggle with sentence-level AI-generated text detection and OOD detection. In contrast, our proposed method not only significantly outperforms baseline methods in both sentence-level and document-level text detection but also demonstrates strong generalization capabilities. Furthermore, since LLM-Detector is trained based on open-source LLMs, it is easy to customize for deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.05941v1">Character-based Outfit Generation with Vision-augmented Style Extraction via LLMs</a></div>
    <div class="paper-meta">
      📅 2024-02-02
      | 💬 7 pages, 4 figures, IEEE Big Data 2023 3rd Workshop on Multimodal AI (MMAI 2023), IEEE BigData 2023
    </div>
    <details class="paper-abstract">
      The outfit generation problem involves recommending a complete outfit to a user based on their interests. Existing approaches focus on recommending items based on anchor items or specific query styles but do not consider customer interests in famous characters from movie, social media, etc. In this paper, we define a new Character-based Outfit Generation (COG) problem, designed to accurately interpret character information and generate complete outfit sets according to customer specifications such as age and gender. To tackle this problem, we propose a novel framework LVA-COG that leverages Large Language Models (LLMs) to extract insights from customer interests (e.g., character information) and employ prompt engineering techniques for accurate understanding of customer preferences. Additionally, we incorporate text-to-image models to enhance the visual understanding and generation (factual or counterfactual) of cohesive outfits. Our framework integrates LLMs with text-to-image models and improves the customer's approach to fashion by generating personalized recommendations. With experiments and case studies, we demonstrate the effectiveness of our solution from multiple dimensions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.01018v1">HR-MultiWOZ: A Task Oriented Dialogue (TOD) Dataset for HR LLM Agent</a></div>
    <div class="paper-meta">
      📅 2024-02-01
      | 💬 13 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have been reshaping Natural Language Processing (NLP) task in several domains. Their use in the field of Human Resources (HR) has still room for expansions and could be beneficial for several time consuming tasks. Examples such as time-off submissions, medical claims filing, and access requests are noteworthy, but they are by no means the sole instances. However, the aforementioned developments must grapple with the pivotal challenge of constructing a high-quality training dataset. On one hand, most conversation datasets are solving problems for customers not employees. On the other hand, gathering conversations with HR could raise privacy concerns. To solve it, we introduce HR-Multiwoz, a fully-labeled dataset of 550 conversations spanning 10 HR domains to evaluate LLM Agent. Our work has the following contributions: (1) It is the first labeled open-sourced conversation dataset in the HR domain for NLP research. (2) It provides a detailed recipe for the data generation procedure along with data analysis and human evaluations. The data generation pipeline is transferable and can be easily adapted for labeled conversation data generation in other domains. (3) The proposed data-collection pipeline is mostly based on LLMs with minimal human involvement for annotation, which is time and cost-efficient.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.15378v4">A RAG-based Question Answering System Proposal for Understanding Islam: MufassirQAS LLM</a></div>
    <div class="paper-meta">
      📅 2024-02-01
    </div>
    <details class="paper-abstract">
      Challenges exist in learning and understanding religions, such as the complexity and depth of religious doctrines and teachings. Chatbots as question-answering systems can help in solving these challenges. LLM chatbots use NLP techniques to establish connections between topics and accurately respond to complex questions. These capabilities make it perfect for enlightenment on religion as a question-answering chatbot. However, LLMs also tend to generate false information, known as hallucination. Also, the chatbots' responses can include content that insults personal religious beliefs, interfaith conflicts, and controversial or sensitive topics. It must avoid such cases without promoting hate speech or offending certain groups of people or their beliefs. This study uses a vector database-based Retrieval Augmented Generation (RAG) approach to enhance the accuracy and transparency of LLMs. Our question-answering system is called "MufassirQAS". We created a database consisting of several open-access books that include Turkish context. These books contain Turkish translations and interpretations of Islam. This database is utilized to answer religion-related questions and ensure our answers are trustworthy. The relevant part of the dataset, which LLM also uses, is presented along with the answer. We have put careful effort into creating system prompts that give instructions to prevent harmful, offensive, or disrespectful responses to respect people's values and provide reliable results. The system answers and shares additional information, such as the page number from the respective book and the articles referenced for obtaining the information. MufassirQAS and ChatGPT are also tested with sensitive questions. We got better performance with our system. Study and enhancements are still in progress. Results and future works are given.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.00620v1">Actor Identification in Discourse: A Challenge for LLMs?</a></div>
    <div class="paper-meta">
      📅 2024-02-01
      | 💬 Proceedings of the EACL 2024 workshop on Computational Models of Discourse (St. Julian's, Malta)
    </div>
    <details class="paper-abstract">
      The identification of political actors who put forward claims in public debate is a crucial step in the construction of discourse networks, which are helpful to analyze societal debates. Actor identification is, however, rather challenging: Often, the locally mentioned speaker of a claim is only a pronoun ("He proposed that [claim]"), so recovering the canonical actor name requires discourse understanding. We compare a traditional pipeline of dedicated NLP components (similar to those applied to the related task of coreference) with a LLM, which appears a good match for this generation task. Evaluating on a corpus of German actors in newspaper reports, we find surprisingly that the LLM performs worse. Further analysis reveals that the LLM is very good at identifying the right reference, but struggles to generate the correct canonical form. This points to an underlying issue in LLMs with controlling generated output. Indeed, a hybrid model combining the LLM with a classifier to normalize its output substantially outperforms both initial models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.01769v1">Redefining "Hallucination" in LLMs: Towards a psychology-informed framework for mitigating misinformation</a></div>
    <div class="paper-meta">
      📅 2024-02-01
    </div>
    <details class="paper-abstract">
      In recent years, large language models (LLMs) have become incredibly popular, with ChatGPT for example being used by over a billion users. While these models exhibit remarkable language understanding and logical prowess, a notable challenge surfaces in the form of "hallucinations." This phenomenon results in LLMs outputting misinformation in a confident manner, which can lead to devastating consequences with such a large user base. However, we question the appropriateness of the term "hallucination" in LLMs, proposing a psychological taxonomy based on cognitive biases and other psychological phenomena. Our approach offers a more fine-grained understanding of this phenomenon, allowing for targeted solutions. By leveraging insights from how humans internally resolve similar challenges, we aim to develop strategies to mitigate LLM hallucinations. This interdisciplinary approach seeks to move beyond conventional terminology, providing a nuanced understanding and actionable pathways for improvement in LLM reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.00292v1">Effective Bug Detection in Graph Database Engines: An LLM-based Approach</a></div>
    <div class="paper-meta">
      📅 2024-02-01
    </div>
    <details class="paper-abstract">
      Graph database engines play a pivotal role in efficiently storing and managing graph data across various domains, including bioinformatics, knowledge graphs, and recommender systems. Ensuring data accuracy within graph database engines is paramount, as inaccuracies can yield unreliable analytical outcomes. Current bug-detection approaches are confined to specific graph query languages, limiting their applicabilities when handling graph database engines that use various graph query languages across various domains. Moreover, they require extensive prior knowledge to generate queries for detecting bugs. To address these challenges, we introduces DGDB, a novel paradigm harnessing large language models(LLM), such as ChatGPT, for comprehensive bug detection in graph database engines. DGDB leverages ChatGPT to generate high-quality queries for different graph query languages. It subsequently employs differential testing to identify bugs in graph database engines. We applied this paradigm to graph database engines using the Gremlin query language and those using the Cypher query language, generating approximately 4,000 queries each. In the latest versions of Neo4j, Agensgraph, and JanusGraph databases, we detected 2, 5, and 3 wrong-result bugs, respectively.
    </details>
</div>
