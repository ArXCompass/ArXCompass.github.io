# llm - 2024_04

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.00176v5">ChipNeMo: Domain-Adapted LLMs for Chip Design</a></div>
    <div class="paper-meta">
      📅 2024-04-04
      | 💬 Updated results for ChipNeMo-70B model
    </div>
    <details class="paper-abstract">
      ChipNeMo aims to explore the applications of large language models (LLMs) for industrial chip design. Instead of directly deploying off-the-shelf commercial or open-source LLMs, we instead adopt the following domain adaptation techniques: domain-adaptive tokenization, domain-adaptive continued pretraining, model alignment with domain-specific instructions, and domain-adapted retrieval models. We evaluate these methods on three selected LLM applications for chip design: an engineering assistant chatbot, EDA script generation, and bug summarization and analysis. Our evaluations demonstrate that domain-adaptive pretraining of language models, can lead to superior performance in domain related downstream tasks compared to their base LLaMA2 counterparts, without degradations in generic capabilities. In particular, our largest model, ChipNeMo-70B, outperforms the highly capable GPT-4 on two of our use cases, namely engineering assistant chatbot and EDA scripts generation, while exhibiting competitive performance on bug summarization and analysis. These results underscore the potential of domain-specific customization for enhancing the effectiveness of large language models in specialized applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.03732v1">SHROOM-INDElab at SemEval-2024 Task 6: Zero- and Few-Shot LLM-Based Classification for Hallucination Detection</a></div>
    <div class="paper-meta">
      📅 2024-04-04
      | 💬 6 pages, 6 figures, 4 tables, camera-ready copy, accepted to the 18th International Workshop on Semantic Evaluation (SemEval-2024), for associated code and data see https://github.com/bradleypallen/shroom
    </div>
    <details class="paper-abstract">
      We describe the University of Amsterdam Intelligent Data Engineering Lab team's entry for the SemEval-2024 Task 6 competition. The SHROOM-INDElab system builds on previous work on using prompt programming and in-context learning with large language models (LLMs) to build classifiers for hallucination detection, and extends that work through the incorporation of context-specific definition of task, role, and target concept, and automated generation of examples for use in a few-shot prompting approach. The resulting system achieved fourth-best and sixth-best performance in the model-agnostic track and model-aware tracks for Task 6, respectively, and evaluation using the validation sets showed that the system's classification decisions were consistent with those of the crowd-sourced human labellers. We further found that a zero-shot approach provided better accuracy than a few-shot approach using automatically generated examples. Code for the system described in this paper is available on Github.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.03301v2">Ziya2: Data-centric Learning is All LLMs Need</a></div>
    <div class="paper-meta">
      📅 2024-04-04
    </div>
    <details class="paper-abstract">
      Various large language models (LLMs) have been proposed in recent years, including closed- and open-source ones, continually setting new records on multiple benchmarks. However, the development of LLMs still faces several issues, such as high cost of training models from scratch, and continual pre-training leading to catastrophic forgetting, etc. Although many such issues are addressed along the line of research on LLMs, an important yet practical limitation is that many studies overly pursue enlarging model sizes without comprehensively analyzing and optimizing the use of pre-training data in their learning process, as well as appropriate organization and leveraging of such data in training LLMs under cost-effective settings. In this work, we propose Ziya2, a model with 13 billion parameters adopting LLaMA2 as the foundation model, and further pre-trained on 700 billion tokens, where we focus on pre-training techniques and use data-centric optimization to enhance the learning process of Ziya2 on different stages. We define three data attributes and firstly establish data-centric scaling laws to illustrate how different data impacts LLMs. Experiments show that Ziya2 significantly outperforms other models in multiple benchmarks especially with promising results compared to representative open-source ones. Ziya2 (Base) is released at https://huggingface.co/IDEA-CCNL/Ziya2-13B-Base and https://modelscope.cn/models/Fengshenbang/Ziya2-13B-Base/summary.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.16045v1">Elicitron: An LLM Agent-Based Simulation Framework for Design Requirements Elicitation</a></div>
    <div class="paper-meta">
      📅 2024-04-04
    </div>
    <details class="paper-abstract">
      Requirements elicitation, a critical, yet time-consuming and challenging step in product development, often fails to capture the full spectrum of user needs. This may lead to products that fall short of expectations. This paper introduces a novel framework that leverages Large Language Models (LLMs) to automate and enhance the requirements elicitation process. LLMs are used to generate a vast array of simulated users (LLM agents), enabling the exploration of a much broader range of user needs and unforeseen use cases. These agents engage in product experience scenarios, through explaining their actions, observations, and challenges. Subsequent agent interviews and analysis uncover valuable user needs, including latent ones. We validate our framework with three experiments. First, we explore different methodologies for diverse agent generation, discussing their advantages and shortcomings. We measure the diversity of identified user needs and demonstrate that context-aware agent generation leads to greater diversity. Second, we show how our framework effectively mimics empathic lead user interviews, identifying a greater number of latent needs than conventional human interviews. Third, we showcase that LLMs can be used to analyze interviews, capture needs, and classify them as latent or not. Our work highlights the potential of using LLM agents to accelerate early-stage product development, reduce costs, and increase innovation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.00492v3">From Language Modeling to Instruction Following: Understanding the Behavior Shift in LLMs after Instruction Tuning</a></div>
    <div class="paper-meta">
      📅 2024-04-04
      | 💬 Accepted by NAACL 2024
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable success, where instruction tuning is the critical step in aligning LLMs with user intentions. In this work, we investigate how the instruction tuning adjusts pre-trained models with a focus on intrinsic changes. Specifically, we first develop several local and global explanation methods, including a gradient-based method for input-output attribution, and techniques for interpreting patterns and concepts in self-attention and feed-forward layers. The impact of instruction tuning is then studied by comparing the explanations derived from the pre-trained and instruction-tuned models. This approach provides an internal perspective of the model shifts on a human-comprehensible level. Our findings reveal three significant impacts of instruction tuning: 1) It empowers LLMs to recognize the instruction parts of user prompts, and promotes the response generation constantly conditioned on the instructions. 2) It encourages the self-attention heads to capture more word-word relationships about instruction verbs. 3) It encourages the feed-forward networks to rotate their pre-trained knowledge toward user-oriented tasks. These insights contribute to a more comprehensive understanding of instruction tuning and lay the groundwork for future work that aims at explaining and optimizing LLMs for various applications. Our code and data are publicly available at https://github.com/JacksonWuxs/Interpret_Instruction_Tuning_LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.11696v3">LLM-based Medical Assistant Personalization with Short- and Long-Term Memory Coordination</a></div>
    <div class="paper-meta">
      📅 2024-04-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), such as GPT3.5, have exhibited remarkable proficiency in comprehending and generating natural language. On the other hand, medical assistants hold the potential to offer substantial benefits for individuals. However, the exploration of LLM-based personalized medical assistant remains relatively scarce. Typically, patients converse differently based on their background and preferences which necessitates the task of enhancing user-oriented medical assistant. While one can fully train an LLM for this objective, the resource consumption is unaffordable. Prior research has explored memory-based methods to enhance the response with aware of previous mistakes for new queries during a dialogue session. We contend that a mere memory module is inadequate and fully training an LLM can be excessively costly. In this study, we propose a novel computational bionic memory mechanism, equipped with a parameter-efficient fine-tuning (PEFT) schema, to personalize medical assistants.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.03413v1">MiniGPT4-Video: Advancing Multimodal LLMs for Video Understanding with Interleaved Visual-Textual Tokens</a></div>
    <div class="paper-meta">
      📅 2024-04-04
      | 💬 6 pages,8 figures
    </div>
    <details class="paper-abstract">
      This paper introduces MiniGPT4-Video, a multimodal Large Language Model (LLM) designed specifically for video understanding. The model is capable of processing both temporal visual and textual data, making it adept at understanding the complexities of videos. Building upon the success of MiniGPT-v2, which excelled in translating visual features into the LLM space for single images and achieved impressive results on various image-text benchmarks, this paper extends the model's capabilities to process a sequence of frames, enabling it to comprehend videos. MiniGPT4-video does not only consider visual content but also incorporates textual conversations, allowing the model to effectively answer queries involving both visual and text components. The proposed model outperforms existing state-of-the-art methods, registering gains of 4.22%, 1.13%, 20.82%, and 13.1% on the MSVD, MSRVTT, TGIF, and TVQA benchmarks respectively. Our models and code have been made publicly available here https://vision-cair.github.io/MiniGPT4-video/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.01563v1">Mitigating LLM Hallucinations via Conformal Abstention</a></div>
    <div class="paper-meta">
      📅 2024-04-04
    </div>
    <details class="paper-abstract">
      We develop a principled procedure for determining when a large language model (LLM) should abstain from responding (e.g., by saying "I don't know") in a general domain, instead of resorting to possibly "hallucinating" a non-sensical or incorrect answer. Building on earlier approaches that use self-consistency as a more reliable measure of model confidence, we propose using the LLM itself to self-evaluate the similarity between each of its sampled responses for a given query. We then further leverage conformal prediction techniques to develop an abstention procedure that benefits from rigorous theoretical guarantees on the hallucination rate (error rate). Experimentally, our resulting conformal abstention method reliably bounds the hallucination rate on various closed-book, open-domain generative question answering datasets, while also maintaining a significantly less conservative abstention rate on a dataset with long responses (Temporal Sequences) compared to baselines using log-probability scores to quantify uncertainty, while achieveing comparable performance on a dataset with short answers (TriviaQA). To evaluate the experiments automatically, one needs to determine if two responses are equivalent given a question. Following standard practice, we use a thresholded similarity function to determine if two responses match, but also provide a method for calibrating the threshold based on conformal prediction, with theoretical guarantees on the accuracy of the match prediction, which might be of independent interest.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.06395v2">ModaVerse: Efficiently Transforming Modalities with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-04-04
      | 💬 CVPR2024
    </div>
    <details class="paper-abstract">
      Humans possess the capability to comprehend diverse modalities and seamlessly transfer information between them. In this work, we introduce ModaVerse, a Multi-modal Large Language Model (MLLM) capable of comprehending and transforming content across various modalities including images, videos, and audio. Predominant MLLM frameworks have largely relied on the alignment of latent spaces of textual and non-textual features. This alignment process, which synchronizes a language model trained on textual data with encoders and decoders trained on multi-modal data, often necessitates extensive training of several projection layers in multiple stages. Inspired by LLM-as-agent methodologies, we propose a novel Input/Output (I/O) alignment mechanism that operates directly at the level of natural language. It aligns the LLM's output with the input of generative models, avoiding the complexities associated with latent feature alignments, and simplifying the multiple training stages of existing MLLMs into a single, efficient process. This conceptual advancement leads to significant reductions in both data and computational costs. By conducting experiments on several benchmarks, we demonstrate that our approach attains comparable performance with the state of the art while achieving considerable efficiencies in data usage and training duration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.03122v1">Towards Standards-Compliant Assistive Technology Product Specifications via LLMs</a></div>
    <div class="paper-meta">
      📅 2024-04-04
    </div>
    <details class="paper-abstract">
      In the rapidly evolving field of assistive technology (AT), ensuring that products meet national and international standards is essential for user safety, efficacy, and accessibility. In this vision paper, we introduce CompliAT, a pioneering framework designed to streamline the compliance process of AT product specifications with these standards through the innovative use of Large Language Models (LLMs). CompliAT addresses three critical tasks: checking terminology consistency, classifying products according to standards, and tracing key product specifications to standard requirements. We tackle the challenge of terminology consistency to ensure that the language used in product specifications aligns with relevant standards, reducing misunderstandings and non-compliance risks. We propose a novel approach for product classification, leveraging a retrieval-augmented generation model to accurately categorize AT products aligning to international standards, despite the sparse availability of training data. Finally, CompliAT implements a traceability and compliance mechanism from key product specifications to standard requirements, ensuring all aspects of an AT product are thoroughly vetted against the corresponding standards. By semi-automating these processes, CompliAT aims to significantly reduce the time and effort required for AT product standards compliance and uphold quality and safety standards. We outline our planned implementation and evaluation plan for CompliAT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2402.01687v2">"Which LLM should I use?": Evaluating LLMs for tasks performed by Undergraduate Computer Science Students</a></div>
    <div class="paper-meta">
      📅 2024-04-04
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      This study evaluates the effectiveness of various large language models (LLMs) in performing tasks common among undergraduate computer science students. Although a number of research studies in the computing education community have explored the possibility of using LLMs for a variety of tasks, there is a lack of comprehensive research comparing different LLMs and evaluating which LLMs are most effective for different tasks. Our research systematically assesses some of the publicly available LLMs such as Google Bard, ChatGPT(3.5), GitHub Copilot Chat, and Microsoft Copilot across diverse tasks commonly encountered by undergraduate computer science students in India. These tasks include code explanation and documentation, solving class assignments, technical interview preparation, learning new concepts and frameworks, and email writing. Evaluation for these tasks was carried out by pre-final year and final year undergraduate computer science students and provides insights into the models' strengths and limitations. This study aims to guide students as well as instructors in selecting suitable LLMs for any specific task and offers valuable insights on how LLMs can be used constructively by students and instructors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.03044v1">The Artificial Intelligence Ontology: LLM-assisted construction of AI concept hierarchies</a></div>
    <div class="paper-meta">
      📅 2024-04-03
    </div>
    <details class="paper-abstract">
      The Artificial Intelligence Ontology (AIO) is a systematization of artificial intelligence (AI) concepts, methodologies, and their interrelations. Developed via manual curation, with the additional assistance of large language models (LLMs), AIO aims to address the rapidly evolving landscape of AI by providing a comprehensive framework that encompasses both technical and ethical aspects of AI technologies. The primary audience for AIO includes AI researchers, developers, and educators seeking standardized terminology and concepts within the AI domain. The ontology is structured around six top-level branches: Networks, Layers, Functions, LLMs, Preprocessing, and Bias, each designed to support the modular composition of AI methods and facilitate a deeper understanding of deep learning architectures and ethical considerations in AI. AIO's development utilized the Ontology Development Kit (ODK) for its creation and maintenance, with its content being dynamically updated through AI-driven curation support. This approach not only ensures the ontology's relevance amidst the fast-paced advancements in AI but also significantly enhances its utility for researchers, developers, and educators by simplifying the integration of new AI concepts and methodologies. The ontology's utility is demonstrated through the annotation of AI methods data in a catalog of AI research publications and the integration into the BioPortal ontology resource, highlighting its potential for cross-disciplinary research. The AIO ontology is open source and is available on GitHub (https://github.com/berkeleybop/artificial-intelligence-ontology) and BioPortal (https://bioportal.bioontology.org/ontologies/AIO).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.09057v3">A Continued Pretrained LLM Approach for Automatic Medical Note Generation</a></div>
    <div class="paper-meta">
      📅 2024-04-03
      | 💬 Accepted to NAACL 2024
    </div>
    <details class="paper-abstract">
      LLMs are revolutionizing NLP tasks. However, the use of the most advanced LLMs, such as GPT-4, is often prohibitively expensive for most specialized fields. We introduce HEAL, the first continuously trained 13B LLaMA2-based LLM that is purpose-built for medical conversations and measured on automated scribing. Our results demonstrate that HEAL outperforms GPT-4 and PMC-LLaMA in PubMedQA, with an accuracy of 78.4\%. It also achieves parity with GPT-4 in generating medical notes. Remarkably, HEAL surpasses GPT-4 and Med-PaLM 2 in identifying more correct medical concepts and exceeds the performance of human scribes and other comparable models in correctness and completeness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.00267v2">Secret Keepers: The Impact of LLMs on Linguistic Markers of Personal Traits</a></div>
    <div class="paper-meta">
      📅 2024-04-03
    </div>
    <details class="paper-abstract">
      Prior research has established associations between individuals' language usage and their personal traits; our linguistic patterns reveal information about our personalities, emotional states, and beliefs. However, with the increasing adoption of Large Language Models (LLMs) as writing assistants in everyday writing, a critical question emerges: are authors' linguistic patterns still predictive of their personal traits when LLMs are involved in the writing process? We investigate the impact of LLMs on the linguistic markers of demographic and psychological traits, specifically examining three LLMs - GPT3.5, Llama 2, and Gemini - across six different traits: gender, age, political affiliation, personality, empathy, and morality. Our findings indicate that although the use of LLMs slightly reduces the predictive power of linguistic patterns over authors' personal traits, the significant changes are infrequent, and the use of LLMs does not fully diminish the predictive power of authors' linguistic patterns over their personal traits. We also note that some theoretically established lexical-based linguistic markers lose their reliability as predictors when LLMs are used in the writing process. Our findings have important implications for the study of linguistic markers of personal traits in the age of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.01687v2">"Which LLM should I use?": Evaluating LLMs for tasks performed by Undergraduate Computer Science Students</a></div>
    <div class="paper-meta">
      📅 2024-04-03
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      This study evaluates the effectiveness of various large language models (LLMs) in performing tasks common among undergraduate computer science students. Although a number of research studies in the computing education community have explored the possibility of using LLMs for a variety of tasks, there is a lack of comprehensive research comparing different LLMs and evaluating which LLMs are most effective for different tasks. Our research systematically assesses some of the publicly available LLMs such as Google Bard, ChatGPT(3.5), GitHub Copilot Chat, and Microsoft Copilot across diverse tasks commonly encountered by undergraduate computer science students in India. These tasks include code explanation and documentation, solving class assignments, technical interview preparation, learning new concepts and frameworks, and email writing. Evaluation for these tasks was carried out by pre-final year and final year undergraduate computer science students and provides insights into the models' strengths and limitations. This study aims to guide students as well as instructors in selecting suitable LLMs for any specific task and offers valuable insights on how LLMs can be used constructively by students and instructors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.02706v1">Unblind Text Inputs: Predicting Hint-text of Text Input in Mobile Apps via LLM</a></div>
    <div class="paper-meta">
      📅 2024-04-03
      | 💬 Accepted by the 2024 CHI Conference on Human Factors in Computing Systems
    </div>
    <details class="paper-abstract">
      Mobile apps have become indispensable for accessing and participating in various environments, especially for low-vision users. Users with visual impairments can use screen readers to read the content of each screen and understand the content that needs to be operated. Screen readers need to read the hint-text attribute in the text input component to remind visually impaired users what to fill in. Unfortunately, based on our analysis of 4,501 Android apps with text inputs, over 0.76 of them are missing hint-text. These issues are mostly caused by developers' lack of awareness when considering visually impaired individuals. To overcome these challenges, we developed an LLM-based hint-text generation model called HintDroid, which analyzes the GUI information of input components and uses in-context learning to generate the hint-text. To ensure the quality of hint-text generation, we further designed a feedback-based inspection mechanism to further adjust hint-text. The automated experiments demonstrate the high BLEU and a user study further confirms its usefulness. HintDroid can not only help visually impaired individuals, but also help ordinary people understand the requirements of input components. HintDroid demo video: https://youtu.be/FWgfcctRbfI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.01544v3">Divergent Token Metrics: Measuring degradation to prune away LLM components -- and optimize quantization</a></div>
    <div class="paper-meta">
      📅 2024-04-03
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have reshaped natural language processing with their impressive capabilities. However, their ever-increasing size has raised concerns about their effective deployment and the need for LLM compression. This study introduces the Divergent Token Metrics (DTMs), a novel approach to assessing compressed LLMs, addressing the limitations of traditional perplexity or accuracy measures that fail to accurately reflect text generation quality. DTMs measure token divergences that allow deeper insights into the subtleties of model compression, in particular, when evaluating components' impacts individually. Utilizing the First Divergent Token Metric (FDTM) in model sparsification reveals that 25% of all attention components can be pruned beyond 90% on the Llama-2 model family, still keeping SOTA performance. For quantization, FDTM suggests that more than 80% of parameters can be naively transformed to int8 without special outlier management. These evaluations indicate the necessity of choosing appropriate compressions for parameters individually -- and that FDTM can identify those -- while standard metrics result in deteriorated outcomes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.15317v1">Concept-Guided LLM Agents for Human-AI Safety Codesign</a></div>
    <div class="paper-meta">
      📅 2024-04-03
      | 💬 5 pages
    </div>
    <details class="paper-abstract">
      Generative AI is increasingly important in software engineering, including safety engineering, where its use ensures that software does not cause harm to people. This also leads to high quality requirements for generative AI. Therefore, the simplistic use of Large Language Models (LLMs) alone will not meet these quality demands. It is crucial to develop more advanced and sophisticated approaches that can effectively address the complexities and safety concerns of software systems. Ultimately, humans must understand and take responsibility for the suggestions provided by generative AI to ensure system safety. To this end, we present an efficient, hybrid strategy to leverage LLMs for safety analysis and Human-AI codesign. In particular, we develop a customized LLM agent that uses elements of prompt engineering, heuristic reasoning, and retrieval-augmented generation to solve tasks associated with predefined safety concepts, in interaction with a system model graph. The reasoning is guided by a cascade of micro-decisions that help preserve structured information. We further suggest a graph verbalization which acts as an intermediate representation of the system model to facilitate LLM-graph interactions. Selected pairs of prompts and responses relevant for safety analytics illustrate our method for the use case of a simplified automated driving system.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.02616v1">Improving Topic Relevance Model by Mix-structured Summarization and LLM-based Data Augmentation</a></div>
    <div class="paper-meta">
      📅 2024-04-03
    </div>
    <details class="paper-abstract">
      Topic relevance between query and document is a very important part of social search, which can evaluate the degree of matching between document and user's requirement. In most social search scenarios such as Dianping, modeling search relevance always faces two challenges. One is that many documents in social search are very long and have much redundant information. The other is that the training data for search relevance model is difficult to get, especially for multi-classification relevance model. To tackle above two problems, we first take query concatenated with the query-based summary and the document summary without query as the input of topic relevance model, which can help model learn the relevance degree between query and the core topic of document. Then, we utilize the language understanding and generation abilities of large language model (LLM) to rewrite and generate query from queries and documents in existing training data, which can construct new query-document pairs as training data. Extensive offline experiments and online A/B tests show that the proposed approaches effectively improve the performance of relevance modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.02474v1">uTeBC-NLP at SemEval-2024 Task 9: Can LLMs be Lateral Thinkers?</a></div>
    <div class="paper-meta">
      📅 2024-04-03
      | 💬 12 pages, 5 figures, 6 tables, Proceedings of the 18th International Workshop on Semantic Evaluation (SemEval-2024) @ NAACL 2024
    </div>
    <details class="paper-abstract">
      Inspired by human cognition, Jiang et al.(2023c) create a benchmark for assessing LLMs' lateral thinking-thinking outside the box. Building upon this benchmark, we investigate how different prompting methods enhance LLMs' performance on this task to reveal their inherent power for outside-the-box thinking ability. Through participating in SemEval-2024, task 9, Sentence Puzzle sub-task, we explore prompt engineering methods: chain of thoughts (CoT) and direct prompting, enhancing with informative descriptions, and employing contextualizing prompts using a retrieval augmented generation (RAG) pipeline. Our experiments involve three LLMs including GPT-3.5, GPT-4, and Zephyr-7B-beta. We generate a dataset of thinking paths between riddles and options using GPT-4, validated by humans for quality. Findings indicate that compressed informative prompts enhance performance. Dynamic in-context learning enhances model performance significantly. Furthermore, fine-tuning Zephyr on our dataset enhances performance across other commonsense datasets, underscoring the value of innovative thinking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.15992v3">Towards Codable Watermarking for Injecting Multi-bits Information to LLMs</a></div>
    <div class="paper-meta">
      📅 2024-04-03
      | 💬 ICLR 2024 poster
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) generate texts with increasing fluency and realism, there is a growing need to identify the source of texts to prevent the abuse of LLMs. Text watermarking techniques have proven reliable in distinguishing whether a text is generated by LLMs by injecting hidden patterns. However, we argue that existing LLM watermarking methods are encoding-inefficient and cannot flexibly meet the diverse information encoding needs (such as encoding model version, generation time, user id, etc.). In this work, we conduct the first systematic study on the topic of Codable Text Watermarking for LLMs (CTWL) that allows text watermarks to carry multi-bit customizable information. First of all, we study the taxonomy of LLM watermarking technologies and give a mathematical formulation for CTWL. Additionally, we provide a comprehensive evaluation system for CTWL: (1) watermarking success rate, (2) robustness against various corruptions, (3) coding rate of payload information, (4) encoding and decoding efficiency, (5) impacts on the quality of the generated text. To meet the requirements of these non-Pareto-improving metrics, we follow the most prominent vocabulary partition-based watermarking direction, and devise an advanced CTWL method named Balance-Marking. The core idea of our method is to use a proxy language model to split the vocabulary into probability-balanced parts, thereby effectively maintaining the quality of the watermarked text. Our code is available at https://github.com/lancopku/codable-watermarking-for-llm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.02422v1">Enhancing Low-Resource LLMs Classification with PEFT and Synthetic Data</a></div>
    <div class="paper-meta">
      📅 2024-04-03
      | 💬 Accepted at LREC-COLING 2024
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) operating in 0-shot or few-shot settings achieve competitive results in Text Classification tasks. In-Context Learning (ICL) typically achieves better accuracy than the 0-shot setting, but it pays in terms of efficiency, due to the longer input prompt. In this paper, we propose a strategy to make LLMs as efficient as 0-shot text classifiers, while getting comparable or better accuracy than ICL. Our solution targets the low resource setting, i.e., when only 4 examples per class are available. Using a single LLM and few-shot real data we perform a sequence of generation, filtering and Parameter-Efficient Fine-Tuning steps to create a robust and efficient classifier. Experimental results show that our approach leads to competitive results on multiple text classification datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.10168v2">Head-to-Tail: How Knowledgeable are Large Language Models (LLMs)? A.K.A. Will LLMs Replace Knowledge Graphs?</a></div>
    <div class="paper-meta">
      📅 2024-04-03
      | 💬 To appear in NAACL 2024
    </div>
    <details class="paper-abstract">
      Since the recent prosperity of Large Language Models (LLMs), there have been interleaved discussions regarding how to reduce hallucinations from LLM responses, how to increase the factuality of LLMs, and whether Knowledge Graphs (KGs), which store the world knowledge in a symbolic form, will be replaced with LLMs. In this paper, we try to answer these questions from a new angle: How knowledgeable are LLMs? To answer this question, we constructed Head-to-Tail, a benchmark that consists of 18K question-answer (QA) pairs regarding head, torso, and tail facts in terms of popularity. We designed an automated evaluation method and a set of metrics that closely approximate the knowledge an LLM confidently internalizes. Through a comprehensive evaluation of 16 publicly available LLMs, we show that existing LLMs are still far from being perfect in terms of their grasp of factual knowledge, especially for facts of torso-to-tail entities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2404.01361v1">LLM Attributor: Interactive Visual Attribution for LLM Generation</a></div>
    <div class="paper-meta">
      📅 2024-04-03
      | 💬 8 pages, 3 figures, For a video demo, see https://youtu.be/mIG2MDQKQxM
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have shown remarkable capability to generate convincing text across diverse domains, concerns around its potential risks have highlighted the importance of understanding the rationale behind text generation. We present LLM Attributor, a Python library that provides interactive visualizations for training data attribution of an LLM's text generation. Our library offers a new way to quickly attribute an LLM's text generation to training data points to inspect model behaviors, enhance its trustworthiness, and compare model-generated text with user-provided text. We describe the visual and interactive design of our tool and highlight usage scenarios for LLaMA2 models fine-tuned with two different datasets: online articles about recent disasters and finance-related question-answer pairs. Thanks to LLM Attributor's broad support for computational notebooks, users can easily integrate it into their workflow to interactively visualize attributions of their models. For easier access and extensibility, we open-source LLM Attributor at https://github.com/poloclub/ LLM-Attribution. The video demo is available at https://youtu.be/mIG2MDQKQxM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2402.01687v2">"Which LLM should I use?": Evaluating LLMs for tasks performed by Undergraduate Computer Science Students</a></div>
    <div class="paper-meta">
      📅 2024-04-03
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      This study evaluates the effectiveness of various large language models (LLMs) in performing tasks common among undergraduate computer science students. Although a number of research studies in the computing education community have explored the possibility of using LLMs for a variety of tasks, there is a lack of comprehensive research comparing different LLMs and evaluating which LLMs are most effective for different tasks. Our research systematically assesses some of the publicly available LLMs such as Google Bard, ChatGPT(3.5), GitHub Copilot Chat, and Microsoft Copilot across diverse tasks commonly encountered by undergraduate computer science students in India. These tasks include code explanation and documentation, solving class assignments, technical interview preparation, learning new concepts and frameworks, and email writing. Evaluation for these tasks was carried out by pre-final year and final year undergraduate computer science students and provides insights into the models' strengths and limitations. This study aims to guide students as well as instructors in selecting suitable LLMs for any specific task and offers valuable insights on how LLMs can be used constructively by students and instructors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.14607v2">Confronting LLMs with Traditional ML: Rethinking the Fairness of Large Language Models in Tabular Classifications</a></div>
    <div class="paper-meta">
      📅 2024-04-02
      | 💬 NAACL 2024 (Main Conference)
    </div>
    <details class="paper-abstract">
      Recent literature has suggested the potential of using large language models (LLMs) to make classifications for tabular tasks. However, LLMs have been shown to exhibit harmful social biases that reflect the stereotypes and inequalities present in society. To this end, as well as the widespread use of tabular data in many high-stake applications, it is important to explore the following questions: what sources of information do LLMs draw upon when making classifications for tabular tasks; whether and to what extent are LLM classifications for tabular data influenced by social biases and stereotypes; and what are the consequential implications for fairness? Through a series of experiments, we delve into these questions and show that LLMs tend to inherit social biases from their training data which significantly impact their fairness in tabular classification tasks. Furthermore, our investigations show that in the context of bias mitigation, though in-context learning and finetuning have a moderate effect, the fairness metric gap between different subgroups is still larger than that in traditional machine learning models, such as Random Forest and shallow Neural Networks. This observation emphasizes that the social biases are inherent within the LLMs themselves and inherited from their pretraining corpus, not only from the downstream task datasets. Besides, we demonstrate that label-flipping of in-context examples can significantly reduce biases, further highlighting the presence of inherent bias within LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.00097v2">Code-Aware Prompting: A study of Coverage Guided Test Generation in Regression Setting using LLM</a></div>
    <div class="paper-meta">
      📅 2024-04-02
    </div>
    <details class="paper-abstract">
      Testing plays a pivotal role in ensuring software quality, yet conventional Search Based Software Testing (SBST) methods often struggle with complex software units, achieving suboptimal test coverage. Recent works using large language models (LLMs) for test generation have focused on improving generation quality through optimizing the test generation context and correcting errors in model outputs, but use fixed prompting strategies that prompt the model to generate tests without additional guidance. As a result LLM-generated testsuites still suffer from low coverage. In this paper, we present SymPrompt, a code-aware prompting strategy for LLMs in test generation. SymPrompt's approach is based on recent work that demonstrates LLMs can solve more complex logical problems when prompted to reason about the problem in a multi-step fashion. We apply this methodology to test generation by deconstructing the testsuite generation process into a multi-stage sequence, each of which is driven by a specific prompt aligned with the execution paths of the method under test, and exposing relevant type and dependency focal context to the model. Our approach enables pretrained LLMs to generate more complete test cases without any additional training. We implement SymPrompt using the TreeSitter parsing framework and evaluate on a benchmark challenging methods from open source Python projects. SymPrompt enhances correct test generations by a factor of 5 and bolsters relative coverage by 26% for CodeGen2. Notably, when applied to GPT-4, SymPrompt improves coverage by over 2x compared to baseline prompting strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.05190v2">DCR: Divide-and-Conquer Reasoning for Multi-choice Question Answering with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-04-02
      | 💬 Technique Report
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown impressive performance in reasoning benchmarks with the emergence of Chain-of-Thought (CoT), particularly in multi-choice question (MCQ). However, current works equally resolve questions regardless of the problem-solving difficulty, leading to an excessive focus on simple items while insufficient attention on intricate ones. To address this challenge, we propose a simple yet effective strategy, Divide and Conquer Reasoning (DCR), to enhance the reasoning capability of LLMs for MCQs, as inspired by human beings using heuristics to first categorize tasks and then handle them separately. In particular, we first categorize questions into two subsets based on confidence score ($\mathcal{CS}$), which is estimated by statistical frequency of generated answers. Subsequently, we propose Filter Choices based Reasoning (FCR) to improve model performance on MCQs with low ($\mathcal{CS}$). Our experiments demonstrate that the proposed strategy only costs 85% of SOTA, while still achieves average accuracy improvement of 1.56% across nine datasets including arithmetic, commonsense, and logic reasoning tasks. The code is at \url{https://github.com/AiMijie/Divide-and-Conquer}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.02294v1">Constrained Robotic Navigation on Preferred Terrains Using LLMs and Speech Instruction: Exploiting the Power of Adverbs</a></div>
    <div class="paper-meta">
      📅 2024-04-02
      | 💬 Presented at ISER 2023
    </div>
    <details class="paper-abstract">
      This paper explores leveraging large language models for map-free off-road navigation using generative AI, reducing the need for traditional data collection and annotation. We propose a method where a robot receives verbal instructions, converted to text through Whisper, and a large language model (LLM) model extracts landmarks, preferred terrains, and crucial adverbs translated into speed settings for constrained navigation. A language-driven semantic segmentation model generates text-based masks for identifying landmarks and terrain types in images. By translating 2D image points to the vehicle's motion plane using camera parameters, an MPC controller can guides the vehicle towards the desired terrain. This approach enhances adaptation to diverse environments and facilitates the use of high-level instructions for navigating complex and challenging terrains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.09782v2">More Samples or More Prompts? Exploring Effective In-Context Sampling for LLM Few-Shot Prompt Engineering</a></div>
    <div class="paper-meta">
      📅 2024-04-02
      | 💬 Accepted at NAACL 2024 Findings
    </div>
    <details class="paper-abstract">
      While most existing works on LLM prompting techniques focus only on how to select a better set of data samples inside one single prompt input (In-Context Learning or ICL), why can not we design and leverage multiple prompts together to further improve the LLM's performance? In this work, we propose In-Context Sampling (ICS), a low-resource LLM prompting technique to produce confident predictions by optimizing the construction of multiple ICL prompt inputs. Extensive experiments with three open-source LLMs (FlanT5-XL, Mistral-7B, and Mixtral-8x7B) on four NLI datasets (e-SNLI, Multi-NLI, ANLI, and Contract-NLI) and one QA dataset (CommonsenseQA) illustrate that ICS can consistently enhance LLMs' performance. An in-depth evaluation with three data similarity-based ICS strategies suggests that these strategies can further elevate LLM's performance, which sheds light on a new yet promising future research direction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.02078v1">Advancing LLM Reasoning Generalists with Preference Trees</a></div>
    <div class="paper-meta">
      📅 2024-04-02
      | 💬 Models and data are available at https://github.com/OpenBMB/Eurus
    </div>
    <details class="paper-abstract">
      We introduce Eurus, a suite of large language models (LLMs) optimized for reasoning. Finetuned from Mistral-7B and CodeLlama-70B, Eurus models achieve state-of-the-art results among open-source models on a diverse set of benchmarks covering mathematics, code generation, and logical reasoning problems. Notably, Eurus-70B beats GPT-3.5 Turbo in reasoning through a comprehensive benchmarking across 12 tests covering five tasks, and achieves a 33.3% pass@1 accuracy on LeetCode and 32.6% on TheoremQA, two challenging benchmarks, substantially outperforming existing open-source models by margins more than 13.3%. The strong performance of Eurus can be primarily attributed to UltraInteract, our newly-curated large-scale, high-quality alignment dataset specifically designed for complex reasoning tasks. UltraInteract can be used in both supervised fine-tuning and preference learning. For each instruction, it includes a preference tree consisting of (1) reasoning chains with diverse planning strategies in a unified format, (2) multi-turn interaction trajectories with the environment and the critique, and (3) pairwise data to facilitate preference learning. UltraInteract allows us to conduct an in-depth exploration of preference learning for reasoning tasks. Our investigation reveals that some well-established preference learning algorithms may be less suitable for reasoning tasks compared to their effectiveness in general conversations. Inspired by this, we derive a novel reward modeling objective which, together with UltraInteract, leads to a strong reward model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.09447v2">How Trustworthy are Open-Source LLMs? An Assessment under Malicious Demonstrations Shows their Vulnerabilities</a></div>
    <div class="paper-meta">
      📅 2024-04-02
      | 💬 NAACL 2024
    </div>
    <details class="paper-abstract">
      The rapid progress in open-source Large Language Models (LLMs) is significantly driving AI development forward. However, there is still a limited understanding of their trustworthiness. Deploying these models at scale without sufficient trustworthiness can pose significant risks, highlighting the need to uncover these issues promptly. In this work, we conduct an adversarial assessment of open-source LLMs on trustworthiness, scrutinizing them across eight different aspects including toxicity, stereotypes, ethics, hallucination, fairness, sycophancy, privacy, and robustness against adversarial demonstrations. We propose advCoU, an extended Chain of Utterances-based (CoU) prompting strategy by incorporating carefully crafted malicious demonstrations for trustworthiness attack. Our extensive experiments encompass recent and representative series of open-source LLMs, including Vicuna, MPT, Falcon, Mistral, and Llama 2. The empirical outcomes underscore the efficacy of our attack strategy across diverse aspects. More interestingly, our result analysis reveals that models with superior performance in general NLP tasks do not always have greater trustworthiness; in fact, larger models can be more vulnerable to attacks. Additionally, models that have undergone instruction tuning, focusing on instruction following, tend to be more susceptible, although fine-tuning LLMs for safety alignment proves effective in mitigating adversarial trustworthiness attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.02183v1">Self-Organized Agents: A LLM Multi-Agent Framework toward Ultra Large-Scale Code Generation and Optimization</a></div>
    <div class="paper-meta">
      📅 2024-04-02
    </div>
    <details class="paper-abstract">
      Recent advancements in automatic code generation using large language model (LLM) agent have brought us closer to the future of automated software development. However, existing single-agent approaches face limitations in generating and improving large-scale, complex codebases due to constraints in context length. To tackle this challenge, we propose Self-Organized multi-Agent framework (SoA), a novel multi-agent framework that enables the scalable and efficient generation and optimization of large-scale code. In SoA, self-organized agents operate independently to generate and modify code components while seamlessly collaborating to construct the overall codebase. A key feature of our framework is the automatic multiplication of agents based on problem complexity, allowing for dynamic scalability. This enables the overall code volume to be increased indefinitely according to the number of agents, while the amount of code managed by each agent remains constant. We evaluate SoA on the HumanEval benchmark and demonstrate that, compared to a single-agent system, each agent in SoA handles significantly less code, yet the overall generated code is substantially greater. Moreover, SoA surpasses the powerful single-agent baseline by 5% in terms of Pass@1 accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.01940v1">Towards Better Understanding of Cybercrime: The Role of Fine-Tuned LLMs in Translation</a></div>
    <div class="paper-meta">
      📅 2024-04-02
      | 💬 9 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Understanding cybercrime communications is paramount for cybersecurity defence. This often involves translating communications into English for processing, interpreting, and generating timely intelligence. The problem is that translation is hard. Human translation is slow, expensive, and scarce. Machine translation is inaccurate and biased. We propose using fine-tuned Large Language Models (LLM) to generate translations that can accurately capture the nuances of cybercrime language. We apply our technique to public chats from the NoName057(16) Russian-speaking hacktivist group. Our results show that our fine-tuned LLM model is better, faster, more accurate, and able to capture nuances of the language. Our method shows it is possible to achieve high-fidelity translations and significantly reduce costs by a factor ranging from 430 to 23,000 compared to a human translator.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.00640v2">Face It Yourselves: An LLM-Based Two-Stage Strategy to Localize Configuration Errors via Logs</a></div>
    <div class="paper-meta">
      📅 2024-04-02
      | 💬 13 pages, accepted by ISSTA 2024 (The 33rd ACM SIGSOFT International Symposium on Software Testing and Analysis)
    </div>
    <details class="paper-abstract">
      Configurable software systems are prone to configuration errors, resulting in significant losses to companies. However, diagnosing these errors is challenging due to the vast and complex configuration space. These errors pose significant challenges for both experienced maintainers and new end-users, particularly those without access to the source code of the software systems. Given that logs are easily accessible to most end-users, we conduct a preliminary study to outline the challenges and opportunities of utilizing logs in localizing configuration errors. Based on the insights gained from the preliminary study, we propose an LLM-based two-stage strategy for end-users to localize the root-cause configuration properties based on logs. We further implement a tool, LogConfigLocalizer, aligned with the design of the aforementioned strategy, hoping to assist end-users in coping with configuration errors through log analysis. To the best of our knowledge, this is the first work to localize the root-cause configuration properties for end-users based on Large Language Models~(LLMs) and logs. We evaluate the proposed strategy on Hadoop by LogConfigLocalizer and prove its efficiency with an average accuracy as high as 99.91%. Additionally, we also demonstrate the effectiveness and necessity of different phases of the methodology by comparing it with two other variants and a baseline tool. Moreover, we validate the proposed methodology through a practical case study to demonstrate its effectiveness and feasibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.01041v2">Can LLMs get help from other LLMs without revealing private information?</a></div>
    <div class="paper-meta">
      📅 2024-04-02
    </div>
    <details class="paper-abstract">
      Cascades are a common type of machine learning systems in which a large, remote model can be queried if a local model is not able to accurately label a user's data by itself. Serving stacks for large language models (LLMs) increasingly use cascades due to their ability to preserve task performance while dramatically reducing inference costs. However, applying cascade systems in situations where the local model has access to sensitive data constitutes a significant privacy risk for users since such data could be forwarded to the remote model. In this work, we show the feasibility of applying cascade systems in such setups by equipping the local model with privacy-preserving techniques that reduce the risk of leaking private information when querying the remote model. To quantify information leakage in such setups, we introduce two privacy measures. We then propose a system that leverages the recently introduced social learning paradigm in which LLMs collaboratively learn from each other by exchanging natural language. Using this paradigm, we demonstrate on several datasets that our methods minimize the privacy loss while at the same time improving task performance compared to a non-cascade baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.09060v2">Do Localization Methods Actually Localize Memorized Data in LLMs? A Tale of Two Benchmarks</a></div>
    <div class="paper-meta">
      📅 2024-04-02
      | 💬 accepted by NAACL 2024
    </div>
    <details class="paper-abstract">
      The concept of localization in LLMs is often mentioned in prior work; however, methods for localization have never been systematically and directly evaluated. We propose two complementary benchmarks that evaluate the ability of localization methods to pinpoint LLM components responsible for memorized data. In our INJ benchmark, we actively inject a piece of new information into a small subset of LLM weights, enabling us to directly evaluate whether localization methods can identify these "ground truth" weights. In our DEL benchmark, we evaluate localization by measuring how much dropping out identified neurons deletes a memorized pretrained sequence. Despite their different perspectives, our two benchmarks yield consistent rankings of five localization methods. Methods adapted from network pruning perform well on both benchmarks, and all evaluated methods show promising localization ability. On the other hand, even successful methods identify neurons that are not specific to a single memorized sequence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.00600v2">AI Act and Large Language Models (LLMs): When critical issues and privacy impact require human and ethical oversight</a></div>
    <div class="paper-meta">
      📅 2024-04-02
    </div>
    <details class="paper-abstract">
      The imposing evolution of artificial intelligence systems and, specifically, of Large Language Models (LLM) makes it necessary to carry out assessments of their level of risk and the impact they may have in the area of privacy, personal data protection and at an ethical level, especially on the weakest and most vulnerable. This contribution addresses human oversight, ethical oversight, and privacy impact assessment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.11851v1">GUARD-D-LLM: An LLM-Based Risk Assessment Engine for the Downstream uses of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-04-02
    </div>
    <details class="paper-abstract">
      Amidst escalating concerns about the detriments inflicted by AI systems, risk management assumes paramount importance, notably for high-risk applications as demanded by the European Union AI Act. Guidelines provided by ISO and NIST aim to govern AI risk management; however, practical implementations remain scarce in scholarly works. Addressing this void, our research explores risks emanating from downstream uses of large language models (LLMs), synthesizing a taxonomy grounded in earlier research. Building upon this foundation, we introduce a novel LLM-based risk assessment engine (GUARD-D-LLM: Guided Understanding and Assessment for Risk Detection for Downstream use of LLMs) designed to pinpoint and rank threats relevant to specific use cases derived from text-based user inputs. Integrating thirty intelligent agents, this innovative approach identifies bespoke risks, gauges their severity, offers targeted suggestions for mitigation, and facilitates risk-aware development. The paper also documents the limitations of such an approach along with way forward suggestions to augment experts in such risk assessment thereby leveraging GUARD-D-LLM in identifying risks early on and enabling early mitigations. This paper and its associated code serve as a valuable resource for developers seeking to mitigate risks associated with LLM-based applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.11487v3">Can LLMs Generate Human-Like Wayfinding Instructions? Towards Platform-Agnostic Embodied Instruction Synthesis</a></div>
    <div class="paper-meta">
      📅 2024-04-02
      | 💬 14 Pages
    </div>
    <details class="paper-abstract">
      We present a novel approach to automatically synthesize "wayfinding instructions" for an embodied robot agent. In contrast to prior approaches that are heavily reliant on human-annotated datasets designed exclusively for specific simulation platforms, our algorithm uses in-context learning to condition an LLM to generate instructions using just a few references. Using an LLM-based Visual Question Answering strategy, we gather detailed information about the environment which is used by the LLM for instruction synthesis. We implement our approach on multiple simulation platforms including Matterport3D, AI Habitat and ThreeDWorld, thereby demonstrating its platform-agnostic nature. We subjectively evaluate our approach via a user study and observe that 83.3% of users find the synthesized instructions accurately capture the details of the environment and show characteristics similar to those of human-generated instructions. Further, we conduct zero-shot navigation with multiple approaches on the REVERIE dataset using the generated instructions, and observe very close correlation with the baseline on standard success metrics (< 1% change in SR), quantifying the viability of generated instructions in replacing human-annotated data. We finally discuss the applicability of our approach in enabling a generalizable evaluation of embodied navigation policies. To the best of our knowledge, ours is the first LLM-driven approach capable of generating "human-like" instructions in a platform-agnostic manner, without training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.14122v3">Beyond Yes and No: Improving Zero-Shot LLM Rankers via Scoring Fine-Grained Relevance Labels</a></div>
    <div class="paper-meta">
      📅 2024-04-02
      | 💬 NAACL 2024; 13 pages
    </div>
    <details class="paper-abstract">
      Zero-shot text rankers powered by recent LLMs achieve remarkable ranking performance by simply prompting. Existing prompts for pointwise LLM rankers mostly ask the model to choose from binary relevance labels like "Yes" and "No". However, the lack of intermediate relevance label options may cause the LLM to provide noisy or biased answers for documents that are partially relevant to the query. We propose to incorporate fine-grained relevance labels into the prompt for LLM rankers, enabling them to better differentiate among documents with different levels of relevance to the query and thus derive a more accurate ranking. We study two variants of the prompt template, coupled with different numbers of relevance levels. Our experiments on 8 BEIR data sets show that adding fine-grained relevance labels significantly improves the performance of LLM rankers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.11653v2">"It's a Fair Game", or Is It? Examining How Users Navigate Disclosure Risks and Benefits When Using LLM-Based Conversational Agents</a></div>
    <div class="paper-meta">
      📅 2024-04-02
      | 💬 26 pages, 5 figures
    </div>
    <details class="paper-abstract">
      The widespread use of Large Language Model (LLM)-based conversational agents (CAs), especially in high-stakes domains, raises many privacy concerns. Building ethical LLM-based CAs that respect user privacy requires an in-depth understanding of the privacy risks that concern users the most. However, existing research, primarily model-centered, does not provide insight into users' perspectives. To bridge this gap, we analyzed sensitive disclosures in real-world ChatGPT conversations and conducted semi-structured interviews with 19 LLM-based CA users. We found that users are constantly faced with trade-offs between privacy, utility, and convenience when using LLM-based CAs. However, users' erroneous mental models and the dark patterns in system design limited their awareness and comprehension of the privacy risks. Additionally, the human-like interactions encouraged more sensitive disclosures, which complicated users' ability to navigate the trade-offs. We discuss practical design guidelines and the needs for paradigm shifts to protect the privacy of LLM-based CA users.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2406.11851v1">GUARD-D-LLM: An LLM-Based Risk Assessment Engine for the Downstream uses of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-04-02
    </div>
    <details class="paper-abstract">
      Amidst escalating concerns about the detriments inflicted by AI systems, risk management assumes paramount importance, notably for high-risk applications as demanded by the European Union AI Act. Guidelines provided by ISO and NIST aim to govern AI risk management; however, practical implementations remain scarce in scholarly works. Addressing this void, our research explores risks emanating from downstream uses of large language models (LLMs), synthesizing a taxonomy grounded in earlier research. Building upon this foundation, we introduce a novel LLM-based risk assessment engine (GUARD-D-LLM: Guided Understanding and Assessment for Risk Detection for Downstream use of LLMs) designed to pinpoint and rank threats relevant to specific use cases derived from text-based user inputs. Integrating thirty intelligent agents, this innovative approach identifies bespoke risks, gauges their severity, offers targeted suggestions for mitigation, and facilitates risk-aware development. The paper also documents the limitations of such an approach along with way forward suggestions to augment experts in such risk assessment thereby leveraging GUARD-D-LLM in identifying risks early on and enabling early mitigations. This paper and its associated code serve as a valuable resource for developers seeking to mitigate risks associated with LLM-based applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.01535v1">Syntactic Robustness for LLM-based Code Generation</a></div>
    <div class="paper-meta">
      📅 2024-04-01
      | 💬 12 pages, 12 figures
    </div>
    <details class="paper-abstract">
      Rapid advances in the field of Large Language Models (LLMs) have made LLM-based code generation an important area for investigation. An LLM-based code generator takes a prompt as input and produces code that implements the requirements specified in the prompt. Many software requirements include mathematical formulas that specify the expected behavior of the code to be generated. Given a code generation prompt that includes a mathematical formula, a reasonable expectation is that, if the formula is syntactically modified without changing its semantics, the generated code for the modified prompt should be semantically equivalent. We formalize this concept as syntactic robustness and investigate the syntactic robustness of GPT-3.5-Turbo and GPT-4 as code generators. To test syntactic robustness, we generate syntactically different but semantically equivalent versions of prompts using a set of mutators that only modify mathematical formulas in prompts. In this paper, we focus on prompts that ask for code that generates solutions to variables in an equation, when given coefficients of the equation as input. Our experimental evaluation demonstrates that GPT-3.5-Turbo and GPT-4 are not syntactically robust for this type of prompts. To improve syntactic robustness, we define a set of reductions that transform the formulas to a simplified form and use these reductions as a pre-processing step. Our experimental results indicate that the syntactic robustness of LLM-based code generation can be improved using our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.01453v1">Unveiling Divergent Inductive Biases of LLMs on Temporal Data</a></div>
    <div class="paper-meta">
      📅 2024-04-01
    </div>
    <details class="paper-abstract">
      Unraveling the intricate details of events in natural language necessitates a subtle understanding of temporal dynamics. Despite the adeptness of Large Language Models (LLMs) in discerning patterns and relationships from data, their inherent comprehension of temporal dynamics remains a formidable challenge. This research meticulously explores these intrinsic challenges within LLMs, with a specific emphasis on evaluating the performance of GPT-3.5 and GPT-4 models in the analysis of temporal data. Employing two distinct prompt types, namely Question Answering (QA) format and Textual Entailment (TE) format, our analysis probes into both implicit and explicit events. The findings underscore noteworthy trends, revealing disparities in the performance of GPT-3.5 and GPT-4. Notably, biases toward specific temporal relationships come to light, with GPT-3.5 demonstrating a preference for "AFTER'' in the QA format for both implicit and explicit events, while GPT-4 leans towards "BEFORE''. Furthermore, a consistent pattern surfaces wherein GPT-3.5 tends towards "TRUE'', and GPT-4 exhibits a preference for "FALSE'' in the TE format for both implicit and explicit events. This persistent discrepancy between GPT-3.5 and GPT-4 in handling temporal data highlights the intricate nature of inductive bias in LLMs, suggesting that the evolution of these models may not merely mitigate bias but may introduce new layers of complexity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.01430v1">Position-Aware Parameter Efficient Fine-Tuning Approach for Reducing Positional Bias in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-04-01
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have enhanced their ability to process long input contexts. This development is particularly crucial for tasks that involve retrieving knowledge from an external datastore, which can result in long inputs. However, recent studies show a positional bias in LLMs, demonstrating varying performance depending on the location of useful information within the input sequence. In this study, we conduct extensive experiments to investigate the root causes of positional bias. Our findings indicate that the primary contributor to LLM positional bias stems from the inherent positional preferences of different models. We demonstrate that merely employing prompt-based solutions is inadequate for overcoming the positional preferences. To address this positional bias issue of a pre-trained LLM, we developed a Position-Aware Parameter Efficient Fine-Tuning (PAPEFT) approach which is composed of a data augmentation technique and a parameter efficient adapter, enhancing a uniform attention distribution across the input context. Our experiments demonstrate that the proposed approach effectively reduces positional bias, improving LLMs' effectiveness in handling long context sequences for various tasks that require externally retrieved knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.01425v1">A Preliminary Roadmap for LLMs as Assistants in Exploring, Analyzing, and Visualizing Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2024-04-01
    </div>
    <details class="paper-abstract">
      We present a mixed-methods study to explore how large language models (LLMs) can assist users in the visual exploration and analysis of knowledge graphs (KGs). We surveyed and interviewed 20 professionals from industry, government laboratories, and academia who regularly work with KGs and LLMs, either collaboratively or concurrently. Our findings show that participants overwhelmingly want an LLM to facilitate data retrieval from KGs through joint query construction, to identify interesting relationships in the KG through multi-turn conversation, and to create on-demand visualizations from the KG that enhance their trust in the LLM's outputs. To interact with an LLM, participants strongly prefer a chat-based 'widget,' built on top of their regular analysis workflows, with the ability to guide the LLM using their interactions with a visualization. When viewing an LLM's outputs, participants similarly prefer a combination of annotated visuals (e.g., subgraphs or tables extracted from the KG) alongside summarizing text. However, participants also expressed concerns about an LLM's ability to maintain semantic intent when translating natural language questions into KG queries, the risk of an LLM 'hallucinating' false data from the KG, and the difficulties of engineering a 'perfect prompt.' From the analysis of our interviews, we contribute a preliminary roadmap for the design of LLM-driven knowledge graph exploration systems and outline future opportunities in this emergent design space.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.01268v1">Mapping the Increasing Use of LLMs in Scientific Papers</a></div>
    <div class="paper-meta">
      📅 2024-04-01
    </div>
    <details class="paper-abstract">
      Scientific publishing lays the foundation of science by disseminating research findings, fostering collaboration, encouraging reproducibility, and ensuring that scientific knowledge is accessible, verifiable, and built upon over time. Recently, there has been immense speculation about how many people are using large language models (LLMs) like ChatGPT in their academic writing, and to what extent this tool might have an effect on global scientific practices. However, we lack a precise measure of the proportion of academic writing substantially modified or produced by LLMs. To address this gap, we conduct the first systematic, large-scale analysis across 950,965 papers published between January 2020 and February 2024 on the arXiv, bioRxiv, and Nature portfolio journals, using a population-level statistical framework to measure the prevalence of LLM-modified content over time. Our statistical estimation operates on the corpus level and is more robust than inference on individual instances. Our findings reveal a steady increase in LLM usage, with the largest and fastest growth observed in Computer Science papers (up to 17.5%). In comparison, Mathematics papers and the Nature portfolio showed the least LLM modification (up to 6.3%). Moreover, at an aggregate level, our analysis reveals that higher levels of LLM-modification are associated with papers whose first authors post preprints more frequently, papers in more crowded research areas, and papers of shorter lengths. Our findings suggests that LLMs are being broadly used in scientific writings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.01230v1">LLM as a Mastermind: A Survey of Strategic Reasoning with Large Language Models</a></div>
    <div class="paper-meta">
      📅 2024-04-01
      | 💬 9 pages, 5 figures
    </div>
    <details class="paper-abstract">
      This paper presents a comprehensive survey of the current status and opportunities for Large Language Models (LLMs) in strategic reasoning, a sophisticated form of reasoning that necessitates understanding and predicting adversary actions in multi-agent settings while adjusting strategies accordingly. Strategic reasoning is distinguished by its focus on the dynamic and uncertain nature of interactions among multi-agents, where comprehending the environment and anticipating the behavior of others is crucial. We explore the scopes, applications, methodologies, and evaluation metrics related to strategic reasoning with LLMs, highlighting the burgeoning development in this area and the interdisciplinary approaches enhancing their decision-making performance. It aims to systematize and clarify the scattered literature on this subject, providing a systematic review that underscores the importance of strategic reasoning as a critical cognitive capability and offers insights into future research directions and potential improvements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.01147v1">Do LLMs Find Human Answers To Fact-Driven Questions Perplexing? A Case Study on Reddit</a></div>
    <div class="paper-meta">
      📅 2024-04-01
      | 💬 4 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been shown to be proficient in correctly answering questions in the context of online discourse. However, the study of using LLMs to model human-like answers to fact-driven social media questions is still under-explored. In this work, we investigate how LLMs model the wide variety of human answers to fact-driven questions posed on several topic-specific Reddit communities, or subreddits. We collect and release a dataset of 409 fact-driven questions and 7,534 diverse, human-rated answers from 15 r/Ask{Topic} communities across 3 categories: profession, social identity, and geographic location. We find that LLMs are considerably better at modeling highly-rated human answers to such questions, as opposed to poorly-rated human answers. We present several directions for future research based on our initial findings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.01402v2">Evaluating the Decency and Consistency of Data Validation Tests Generated by LLMs</a></div>
    <div class="paper-meta">
      📅 2024-04-01
      | 💬 36 pages, 18 figures
    </div>
    <details class="paper-abstract">
      We investigated whether large language models (LLMs) can develop data validation tests. We considered 96 conditions each for both GPT-3.5 and GPT-4, examining different prompt scenarios, learning modes, temperature settings, and roles. The prompt scenarios were: 1) Asking for expectations, 2) Asking for expectations with a given context, 3) Asking for expectations after requesting a data simulation, and 4) Asking for expectations with a provided data sample. The learning modes were: 1) zero-shot, 2) one-shot, and 3) few-shot learning. We also tested four temperature settings: 0, 0.4, 0.6, and 1. And the two distinct roles were: 1) helpful assistant, 2) expert data scientist. To gauge consistency, every setup was tested five times. The LLM-generated responses were benchmarked against a gold standard data validation suite, created by an experienced data scientist knowledgeable about the data in question. We find there are considerable returns to the use of few-shot learning, and that the more explicit the data setting can be the better, to a point. The best LLM configurations complement, rather than substitute, the gold standard results. This study underscores the value LLMs can bring to the data cleaning and preparation stages of the data science workflow, but highlights that they need considerable evaluation by experienced analysts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.11369v2">What Makes Math Word Problems Challenging for LLMs?</a></div>
    <div class="paper-meta">
      📅 2024-04-01
      | 💬 Accepted to NAACL Findings 2024
    </div>
    <details class="paper-abstract">
      This paper investigates the question of what makes math word problems (MWPs) in English challenging for large language models (LLMs). We conduct an in-depth analysis of the key linguistic and mathematical characteristics of MWPs. In addition, we train feature-based classifiers to better understand the impact of each feature on the overall difficulty of MWPs for prominent LLMs and investigate whether this helps predict how well LLMs fare against specific categories of MWPs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.01361v1">LLM Attributor: Interactive Visual Attribution for LLM Generation</a></div>
    <div class="paper-meta">
      📅 2024-04-01
      | 💬 8 pages, 3 figures, For a video demo, see https://youtu.be/mIG2MDQKQxM
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have shown remarkable capability to generate convincing text across diverse domains, concerns around its potential risks have highlighted the importance of understanding the rationale behind text generation. We present LLM Attributor, a Python library that provides interactive visualizations for training data attribution of an LLM's text generation. Our library offers a new way to quickly attribute an LLM's text generation to training data points to inspect model behaviors, enhance its trustworthiness, and compare model-generated text with user-provided text. We describe the visual and interactive design of our tool and highlight usage scenarios for LLaMA2 models fine-tuned with two different datasets: online articles about recent disasters and finance-related question-answer pairs. Thanks to LLM Attributor's broad support for computational notebooks, users can easily integrate it into their workflow to interactively visualize attributions of their models. For easier access and extensibility, we open-source LLM Attributor at https://github.com/poloclub/ LLM-Attribution. The video demo is available at https://youtu.be/mIG2MDQKQxM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.01096v1">Enabling Memory Safety of C Programs using LLMs</a></div>
    <div class="paper-meta">
      📅 2024-04-01
    </div>
    <details class="paper-abstract">
      Memory safety violations in low-level code, written in languages like C, continues to remain one of the major sources of software vulnerabilities. One method of removing such violations by construction is to port C code to a safe C dialect. Such dialects rely on programmer-supplied annotations to guarantee safety with minimal runtime overhead. This porting, however, is a manual process that imposes significant burden on the programmer and, hence, there has been limited adoption of this technique. The task of porting not only requires inferring annotations, but may also need refactoring/rewriting of the code to make it amenable to such annotations. In this paper, we use Large Language Models (LLMs) towards addressing both these concerns. We show how to harness LLM capabilities to do complex code reasoning as well as rewriting of large codebases. We also present a novel framework for whole-program transformations that leverages lightweight static analysis to break the transformation into smaller steps that can be carried out effectively by an LLM. We implement our ideas in a tool called MSA that targets the CheckedC dialect. We evaluate MSA on several micro-benchmarks, as well as real-world code ranging up to 20K lines of code. We showcase superior performance compared to a vanilla LLM baseline, as well as demonstrate improvement over a state-of-the-art symbolic (non-LLM) technique.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.05950v2">Exploring the Effectiveness of LLMs in Automated Logging Generation: An Empirical Study</a></div>
    <div class="paper-meta">
      📅 2024-04-01
    </div>
    <details class="paper-abstract">
      Automated logging statement generation supports developers in documenting critical software runtime behavior. Given the great success in natural language generation and programming language comprehension, large language models (LLMs) might help developers generate logging statements, but this has not yet been investigated. To fill the gap, this paper performs the first study on exploring LLMs for logging statement generation.We first build a logging statement generation dataset, LogBench, with two parts: (1) LogBench-O: logging statements collected from GitHub repositories, and (2) LogBench-T: the transformed unseen code from LogBench-O. Then, we leverage LogBench to evaluate the effectiveness and generalization capabilities (using LogBench-T) of eleven top-performing LLMs. In addition, we examine the performance of these LLMs against classical retrieval-based and machine learning-based logging methods from the era preceding LLMs. We further evaluate LLM's logging generalization capabilities using unseen data (LogBench-T) derived from code transformation techniques. While existing LLMs deliver decent predictions on logging levels and logging variables, our study indicates that they only achieve a maximum BLEU score of 0.249, thus calling for improvements. The paper also highlights the importance of prompt constructions and external factors (e.g., programming contexts and code comments) for LLMs' logging performance. Based on these findings, we identify five implications and provide practical advice for future logging research. Our empirical analysis discloses the limitations of current logging approaches while showcasing the potential of LLM-based logging tools, and provides actionable guidance for building more practical models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.18647v2">SDSAT: Accelerating LLM Inference through Speculative Decoding with Semantic Adaptive Tokens</a></div>
    <div class="paper-meta">
      📅 2024-04-01
      | 💬 12 pages, 7 figures
    </div>
    <details class="paper-abstract">
      We propose an acceleration scheme for large language models (LLMs) through Speculative Decoding with Semantic Adaptive Tokens (SDSAT). The primary objective of this design is to enhance the LLM model's ability to generate draft tokens more accurately without compromising the model's accuracy. The core strategies involve: 1) Fine-tune the model by incorporating semantic adaptive tokens that possess flexible decoding capabilities without changing its structure, allowing them to generate high-quality draft tokens. 2) By employing a training method that does not affect the standard tokens, the model can acquire parallel decoding abilities atop its original framework with minimal training overhead. 3) We have designed the "two-step-draft-then-verify" generation strategies using both greedy search and nucleus sampling. Experiments conducted on the CodeLlama-13B and 7B models have yielded speed increases of over 3.5X and 3.0X, respectively. Please refer to https://github.com/hasuoshenyun/SDSAT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.01353v1">Efficiently Distilling LLMs for Edge Applications</a></div>
    <div class="paper-meta">
      📅 2024-04-01
      | 💬 This paper has been accepted for publication in NAACL 2024 (Industry Track)
    </div>
    <details class="paper-abstract">
      Supernet training of LLMs is of great interest in industrial applications as it confers the ability to produce a palette of smaller models at constant cost, regardless of the number of models (of different size / latency) produced. We propose a new method called Multistage Low-rank Fine-tuning of Super-transformers (MLFS) for parameter-efficient supernet training. We show that it is possible to obtain high-quality encoder models that are suitable for commercial edge applications, and that while decoder-only models are resistant to a comparable degree of compression, decoders can be effectively sliced for a significant reduction in training time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.19318v2">TableLLM: Enabling Tabular Data Manipulation by LLMs in Real Office Usage Scenarios</a></div>
    <div class="paper-meta">
      📅 2024-04-01
      | 💬 https://tablellm.github.io/
    </div>
    <details class="paper-abstract">
      We introduce TableLLM, a robust large language model (LLM) with 13 billion parameters, purpose-built for proficiently handling tabular data manipulation tasks, whether they are embedded within documents or spreadsheets, catering to real-world office scenarios. We propose a distant supervision method for training, which comprises a reasoning process extension strategy, aiding in training LLMs to understand reasoning patterns more effectively as well as a cross-way validation strategy, ensuring the quality of the automatically generated data. To evaluate the performance of TableLLM, we have crafted a benchmark tailored to address both document and spreadsheet formats as well as constructed a well-organized evaluation pipeline capable of handling both scenarios. Thorough evaluations underscore the advantages of TableLLM when compared to various existing general-purpose and tabular data-focused LLMs. We have publicly released the model checkpoint, source code, benchmarks, and a web application for user interaction.Our codes and data are publicly available at https://github.com/TableLLM/TableLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.00925v1">LLMs are Good Sign Language Translators</a></div>
    <div class="paper-meta">
      📅 2024-04-01
      | 💬 Accepted to CVPR 2024
    </div>
    <details class="paper-abstract">
      Sign Language Translation (SLT) is a challenging task that aims to translate sign videos into spoken language. Inspired by the strong translation capabilities of large language models (LLMs) that are trained on extensive multilingual text corpora, we aim to harness off-the-shelf LLMs to handle SLT. In this paper, we regularize the sign videos to embody linguistic characteristics of spoken language, and propose a novel SignLLM framework to transform sign videos into a language-like representation for improved readability by off-the-shelf LLMs. SignLLM comprises two key modules: (1) The Vector-Quantized Visual Sign module converts sign videos into a sequence of discrete character-level sign tokens, and (2) the Codebook Reconstruction and Alignment module converts these character-level tokens into word-level sign representations using an optimal transport formulation. A sign-text alignment loss further bridges the gap between sign and text tokens, enhancing semantic compatibility. We achieve state-of-the-art gloss-free results on two widely-used SLT benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2304.08103v3">Low-code LLM: Graphical User Interface over Large Language Models</a></div>
    <div class="paper-meta">
      📅 2024-04-01
      | 💬 Accepted as a Demo Track paper at NAACL 2024
    </div>
    <details class="paper-abstract">
      Utilizing Large Language Models (LLMs) for complex tasks is challenging, often involving a time-consuming and uncontrollable prompt engineering process. This paper introduces a novel human-LLM interaction framework, Low-code LLM. It incorporates six types of simple low-code visual programming interactions to achieve more controllable and stable responses. Through visual interaction with a graphical user interface, users can incorporate their ideas into the process without writing trivial prompts. The proposed Low-code LLM framework consists of a Planning LLM that designs a structured planning workflow for complex tasks, which can be correspondingly edited and confirmed by users through low-code visual programming operations, and an Executing LLM that generates responses following the user-confirmed workflow. We highlight three advantages of the low-code LLM: user-friendly interaction, controllable generation, and wide applicability. We demonstrate its benefits using four typical applications. By introducing this framework, we aim to bridge the gap between humans and LLMs, enabling more effective and efficient utilization of LLMs for complex tasks. The code, prompts, and experimental details are available at https://github.com/moymix/TaskMatrix/tree/main/LowCodeLLM. A system demonstration video can be found at https://www.youtube.com/watch?v=jb2C1vaeO3E.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.00899v1">TM-TREK at SemEval-2024 Task 8: Towards LLM-Based Automatic Boundary Detection for Human-Machine Mixed Text</a></div>
    <div class="paper-meta">
      📅 2024-04-01
      | 💬 1st place at SemEval-2024 Task 8, Subtask C, to appear in SemEval-2024 proceedings
    </div>
    <details class="paper-abstract">
      With the increasing prevalence of text generated by large language models (LLMs), there is a growing concern about distinguishing between LLM-generated and human-written texts in order to prevent the misuse of LLMs, such as the dissemination of misleading information and academic dishonesty. Previous research has primarily focused on classifying text as either entirely human-written or LLM-generated, neglecting the detection of mixed texts that contain both types of content. This paper explores LLMs' ability to identify boundaries in human-written and machine-generated mixed texts. We approach this task by transforming it into a token classification problem and regard the label turning point as the boundary. Notably, our ensemble model of LLMs achieved first place in the 'Human-Machine Mixed Text Detection' sub-task of the SemEval'24 Competition Task 8. Additionally, we investigate factors that influence the capability of LLMs in detecting boundaries within mixed texts, including the incorporation of extra layers on top of LLMs, combination of segmentation loss, and the impact of pretraining. Our findings aim to provide valuable insights for future research in this area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.05915v3">Fake Alignment: Are LLMs Really Aligned Well?</a></div>
    <div class="paper-meta">
      📅 2024-04-01
      | 💬 Accepted to the NAACL 2024
    </div>
    <details class="paper-abstract">
      The growing awareness of safety concerns in large language models (LLMs) has sparked considerable interest in the evaluation of safety. This study investigates an under-explored issue about the evaluation of LLMs, namely the substantial discrepancy in performance between multiple-choice questions and open-ended questions. Inspired by research on jailbreak attack patterns, we argue this is caused by mismatched generalization. That is, LLM only remembers the answer style for open-ended safety questions, which makes it unable to solve other forms of safety tests. We refer to this phenomenon as fake alignment and construct a comparative benchmark to empirically verify its existence in LLMs. We introduce a Fake alIgNment Evaluation (FINE) framework and two novel metrics--Consistency Score (CS) and Consistent Safety Score (CSS), which jointly assess two complementary forms of evaluation to quantify fake alignment and obtain corrected performance estimation. Applying FINE to 14 widely-used LLMs reveals several models with purported safety are poorly aligned in practice. Subsequently, we found that multiple-choice format data can also be used as high-quality contrast distillation-based fine-tuning data, which can strongly improve the alignment consistency of LLMs with minimal fine-tuning overhead. For data and code, see https://github.com/AIFlames/Fake-Alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.06742v2">Honeybee: Locality-enhanced Projector for Multimodal LLM</a></div>
    <div class="paper-meta">
      📅 2024-04-01
      | 💬 CVPR 2024 camera-ready
    </div>
    <details class="paper-abstract">
      In Multimodal Large Language Models (MLLMs), a visual projector plays a crucial role in bridging pre-trained vision encoders with LLMs, enabling profound visual understanding while harnessing the LLMs' robust capabilities. Despite the importance of the visual projector, it has been relatively less explored. In this study, we first identify two essential projector properties: (i) flexibility in managing the number of visual tokens, crucial for MLLMs' overall efficiency, and (ii) preservation of local context from visual features, vital for spatial understanding. Based on these findings, we propose a novel projector design that is both flexible and locality-enhanced, effectively satisfying the two desirable properties. Additionally, we present comprehensive strategies to effectively utilize multiple and multifaceted instruction datasets. Through extensive experiments, we examine the impact of individual design choices. Finally, our proposed MLLM, Honeybee, remarkably outperforms previous state-of-the-art methods across various benchmarks, including MME, MMBench, SEED-Bench, and LLaVA-Bench, achieving significantly higher efficiency. Code and models are available at https://github.com/kakaobrain/honeybee.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.09618v4">Simulating Opinion Dynamics with Networks of LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2024-04-01
    </div>
    <details class="paper-abstract">
      Accurately simulating human opinion dynamics is crucial for understanding a variety of societal phenomena, including polarization and the spread of misinformation. However, the agent-based models (ABMs) commonly used for such simulations often over-simplify human behavior. We propose a new approach to simulating opinion dynamics based on populations of Large Language Models (LLMs). Our findings reveal a strong inherent bias in LLM agents towards producing accurate information, leading simulated agents to consensus in line with scientific reality. This bias limits their utility for understanding resistance to consensus views on issues like climate change. After inducing confirmation bias through prompt engineering, however, we observed opinion fragmentation in line with existing agent-based modeling and opinion dynamics research. These insights highlight the promise and limitations of LLM agents in this domain and suggest a path forward: refining LLMs with real-world discourse to better simulate the evolution of human beliefs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2404.01361v1">LLM Attributor: Interactive Visual Attribution for LLM Generation</a></div>
    <div class="paper-meta">
      📅 2024-04-01
      | 💬 8 pages, 3 figures, For a video demo, see https://youtu.be/mIG2MDQKQxM
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have shown remarkable capability to generate convincing text across diverse domains, concerns around its potential risks have highlighted the importance of understanding the rationale behind text generation. We present LLM Attributor, a Python library that provides interactive visualizations for training data attribution of an LLM's text generation. Our library offers a new way to quickly attribute an LLM's text generation to training data points to inspect model behaviors, enhance its trustworthiness, and compare model-generated text with user-provided text. We describe the visual and interactive design of our tool and highlight usage scenarios for LLaMA2 models fine-tuned with two different datasets: online articles about recent disasters and finance-related question-answer pairs. Thanks to LLM Attributor's broad support for computational notebooks, users can easily integrate it into their workflow to interactively visualize attributions of their models. For easier access and extensibility, we open-source LLM Attributor at https://github.com/poloclub/ LLM-Attribution. The video demo is available at https://youtu.be/mIG2MDQKQxM.
    </details>
</div>
