# llm - 2025_07

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- Part 4
- [Part 5](papers_5.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.00225v3">Large-scale, Independent and Comprehensive study of the power of LLMs for test case generation</a></div>
    <div class="paper-meta">
      📅 2025-07-05
    </div>
    <details class="paper-abstract">
      Unit testing is essential for software reliability, yet manual test creation is time-consuming and often neglected. Although search-based software testing improves efficiency, it produces tests with poor readability and maintainability. Although LLMs show promise for test generation, existing research lacks comprehensive evaluation across execution-driven assessment, reasoning-based prompting, and real-world testing scenarios. This study presents the first large-scale empirical evaluation of LLM-generated unit tests at the class level, systematically analyzing four state-of-the-art models - GPT-3.5, GPT-4, Mistral 7B, and Mixtral 8x7B - against EvoSuite across 216,300 test cases from Defects4J, SF110, and CMD (a dataset mitigating LLM training data leakage). We evaluate five prompting techniques - Zero-Shot Learning (ZSL), Few-Shot Learning (FSL), Chain-of-Thought (CoT), Tree-of-Thought (ToT), and Guided Tree-of-Thought (GToT) - assessing syntactic correctness, compilability, hallucination-driven failures, readability, code coverage metrics, and fault detection capabilities. Our findings challenge prior claims that in-context learning is ineffective for test generation in code-specialized LLMs. Reasoning-based prompting - particularly GToT - significantly enhances test reliability, compilability, and structural adherence in general-purpose LLMs. However, hallucination-driven failures remain a persistent challenge, manifesting as non-existent symbol references, incorrect API calls, and fabricated dependencies, resulting in high compilation failure rates (up to 86%). Execution-based classification and mutation testing reveal that many failing tests stem from hallucinated dependencies, limiting effective fault detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04034v1">Lyria: A General LLM-Driven Genetic Algorithm Framework for Problem Solving</a></div>
    <div class="paper-meta">
      📅 2025-07-05
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have demonstrated impressive abilities across various domains, they still struggle with complex problems characterized by multi-objective optimization, precise constraint satisfaction, immense solution spaces, etc. To address the limitation, drawing on the superior semantic understanding ability of LLMs and also the outstanding global search and optimization capability of genetic algorithms, we propose to capitalize on their respective strengths and introduce Lyria, a general LLM-driven genetic algorithm framework, comprising 7 essential components. Through conducting extensive experiments with 4 LLMs across 3 types of problems, we demonstrated the efficacy of Lyria. Additionally, with 7 additional ablation experiments, we further systematically analyzed and elucidated the factors that affect its performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04009v1">Easy Dataset: A Unified and Extensible Framework for Synthesizing LLM Fine-Tuning Data from Unstructured Documents</a></div>
    <div class="paper-meta">
      📅 2025-07-05
      | 💬 preprint
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown impressive performance on general-purpose tasks, yet adapting them to specific domains remains challenging due to the scarcity of high-quality domain data. Existing data synthesis tools often struggle to extract reliable fine-tuning data from heterogeneous documents effectively. To address this limitation, we propose Easy Dataset, a unified framework for synthesizing fine-tuning data from unstructured documents via an intuitive graphical user interface (GUI). Specifically, Easy Dataset allows users to easily configure text extraction models and chunking strategies to transform raw documents into coherent text chunks. It then leverages a persona-driven prompting approach to generate diverse question-answer pairs using public-available LLMs. Throughout the pipeline, a human-in-the-loop visual interface facilitates the review and refinement of intermediate outputs to ensure data quality. Experiments on a financial question-answering task show that fine-tuning LLMs on the synthesized dataset significantly improves domain-specific performance while preserving general knowledge. The source code and installable package are available at https://github.com/ConardLi/easy-dataset and have garnered over 9,000 GitHub stars.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14363v2">Improving RL Exploration for LLM Reasoning through Retrospective Replay</a></div>
    <div class="paper-meta">
      📅 2025-07-05
      | 💬 13 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has increasingly become a pivotal technique in the post-training of large language models (LLMs). The effective exploration of the output space is essential for the success of RL. We observe that for complex problems, during the early stages of training, the model exhibits strong exploratory capabilities and can identify promising solution ideas. However, its limited capability at this stage prevents it from successfully solving these problems. The early suppression of these potentially valuable solution ideas by the policy gradient hinders the model's ability to revisit and re-explore these ideas later. Consequently, although the LLM's capabilities improve in the later stages of training, it still struggles to effectively address these complex problems. To address this exploration issue, we propose a novel algorithm named Retrospective Replay-based Reinforcement Learning (RRL), which introduces a dynamic replay mechanism throughout the training process. RRL enables the model to revisit promising states identified in the early stages, thereby improving its efficiency and effectiveness in exploration. To evaluate the effectiveness of RRL, we conduct extensive experiments on complex reasoning tasks, including mathematical reasoning and code generation, and general dialogue tasks. The results indicate that RRL maintains high exploration efficiency throughout the training period, significantly enhancing the effectiveness of RL in optimizing LLMs for complicated reasoning tasks. Moreover, it also improves the performance of RLHF, making the model both safer and more helpful.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03958v1">A Comparative Study of Specialized LLMs as Dense Retrievers</a></div>
    <div class="paper-meta">
      📅 2025-07-05
      | 💬 Accepted by CCIR25 and published by Springer LNCS or LNAI
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) are increasingly deployed as dense retrievers, the impact of their domain-specific specialization on retrieval effectiveness remains underexplored. This investigation systematically examines how task-specific adaptations in LLMs influence their retrieval capabilities, an essential step toward developing unified retrievers capable of handling text, code, images, and multimodal content. We conduct extensive experiments with eight Qwen2.5 7B LLMs, including base, instruction-tuned, code/math-specialized, long reasoning, and vision-language models across zero-shot retrieval settings and the supervised setting. For the zero-shot retrieval settings, we consider text retrieval from the BEIR benchmark and code retrieval from the CoIR benchmark. Further, to evaluate supervised performance, all LLMs are fine-tuned on the MS MARCO dataset. We find that mathematical specialization and the long reasoning capability cause consistent degradation in three settings, indicating conflicts between mathematical reasoning and semantic matching. The vision-language model and code-specialized LLMs demonstrate superior zero-shot performance compared to other LLMs, even surpassing BM25 on the code retrieval task, and maintain comparable performance to base LLMs in supervised settings. These findings suggest promising directions for the unified retrieval task leveraging cross-domain and cross-modal fusion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03945v1">Function-based Labels for Complementary Recommendation: Definition, Annotation, and LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-07-05
    </div>
    <details class="paper-abstract">
      Complementary recommendations enhance the user experience by suggesting items that are frequently purchased together while serving different functions from the query item. Inferring or evaluating whether two items have a complementary relationship requires complementary relationship labels; however, defining these labels is challenging because of the inherent ambiguity of such relationships. Complementary labels based on user historical behavior logs attempt to capture these relationships, but often produce inconsistent and unreliable results. Recent efforts have introduced large language models (LLMs) to infer these relationships. However, these approaches provide a binary classification without a nuanced understanding of complementary relationships. In this study, we address these challenges by introducing Function-Based Labels (FBLs), a novel definition of complementary relationships independent of user purchase logs and the opaque decision processes of LLMs. We constructed a human-annotated FBLs dataset comprising 2,759 item pairs and demonstrated that it covered possible item relationships and minimized ambiguity. We then evaluated whether some machine learning (ML) methods using annotated FBLs could accurately infer labels for unseen item pairs, and whether LLM-generated complementary labels align with human perception. Our results demonstrate that even with limited data, ML models, such as logistic regression and SVM achieve high macro-F1 scores (approximately 0.82). Furthermore, LLMs, such as gpt-4o-mini, demonstrated high consistency (0.989) and classification accuracy (0.849) under the detailed definition of FBLs, indicating their potential as effective annotators that mimic human judgment. Overall, our study presents FBLs as a clear definition of complementary relationships, enabling more accurate inferences and automated labeling of complementary recommendations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03908v1">Bridging Vision and Language: Optimal Transport-Driven Radiology Report Generation via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-05
    </div>
    <details class="paper-abstract">
      Radiology report generation represents a significant application within medical AI, and has achieved impressive results. Concurrently, large language models (LLMs) have demonstrated remarkable performance across various domains. However, empirical validation indicates that general LLMs tend to focus more on linguistic fluency rather than clinical effectiveness, and lack the ability to effectively capture the relationship between X-ray images and their corresponding texts, thus resulting in poor clinical practicability. To address these challenges, we propose Optimal Transport-Driven Radiology Report Generation (OTDRG), a novel framework that leverages Optimal Transport (OT) to align image features with disease labels extracted from reports, effectively bridging the cross-modal gap. The core component of OTDRG is Alignment \& Fine-Tuning, where OT utilizes results from the encoding of label features and image visual features to minimize cross-modal distances, then integrating image and text features for LLMs fine-tuning. Additionally, we design a novel disease prediction module to predict disease labels contained in X-ray images during validation and testing. Evaluated on the MIMIC-CXR and IU X-Ray datasets, OTDRG achieves state-of-the-art performance in both natural language generation (NLG) and clinical efficacy (CE) metrics, delivering reports that are not only linguistically coherent but also clinically accurate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03834v1">Economic Evaluation of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-04
      | 💬 14 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Practitioners often navigate LLM performance trade-offs by plotting Pareto frontiers of optimal accuracy-cost trade-offs. However, this approach offers no way to compare between LLMs with distinct strengths and weaknesses: for example, a cheap, error-prone model vs a pricey but accurate one. To address this gap, we propose economic evaluation of LLMs. Our framework quantifies the performance trade-off of an LLM as a single number based on the economic constraints of a concrete use case, all expressed in dollars: the cost of making a mistake, the cost of incremental latency, and the cost of abstaining from a query. We apply our economic evaluation framework to compare the performance of reasoning and non-reasoning models on difficult questions from the MATH benchmark, discovering that reasoning models offer better accuracy-cost tradeoffs as soon as the economic cost of a mistake exceeds \$0.01. In addition, we find that single large LLMs often outperform cascades when the cost of making a mistake is as low as \$0.1. Overall, our findings suggest that when automating meaningful human tasks with AI models, practitioners should typically use the most powerful available model, rather than attempt to minimize AI deployment costs, since deployment costs are likely dwarfed by the economic impact of AI errors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03829v1">RELRaE: LLM-Based Relationship Extraction, Labelling, Refinement, and Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-07-04
      | 💬 18 Pages, 8 Tables, Under-review at ISWC 2025
    </div>
    <details class="paper-abstract">
      A large volume of XML data is produced in experiments carried out by robots in laboratories. In order to support the interoperability of data between labs, there is a motivation to translate the XML data into a knowledge graph. A key stage of this process is the enrichment of the XML schema to lay the foundation of an ontology schema. To achieve this, we present the RELRaE framework, a framework that employs large language models in different stages to extract and accurately label the relationships implicitly present in the XML schema. We investigate the capability of LLMs to accurately generate these labels and then evaluate them. Our work demonstrates that LLMs can be effectively used to support the generation of relationship labels in the context of lab automation, and that they can play a valuable role within semi-automatic ontology generation frameworks more generally.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03722v1">Roadmap for using large language models (LLMs) to accelerate cross-disciplinary research with an example from computational biology</a></div>
    <div class="paper-meta">
      📅 2025-07-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are powerful artificial intelligence (AI) tools transforming how research is conducted. However, their use in research has been met with skepticism, due to concerns about hallucinations, biases and potential harms to research. These emphasize the importance of clearly understanding the strengths and weaknesses of LLMs to ensure their effective and responsible use. Here, we present a roadmap for integrating LLMs into cross-disciplinary research, where effective communication, knowledge transfer and collaboration across diverse fields are essential but often challenging. We examine the capabilities and limitations of LLMs and provide a detailed computational biology case study (on modeling HIV rebound dynamics) demonstrating how iterative interactions with an LLM (ChatGPT) can facilitate interdisciplinary collaboration and research. We argue that LLMs are best used as augmentative tools within a human-in-the-loop framework. Looking forward, we envisage that the responsible use of LLMs will enhance innovative cross-disciplinary research and substantially accelerate scientific discoveries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03711v1">Can LLMs Play Ô Ăn Quan Game? A Study of Multi-Step Planning and Decision Making</a></div>
    <div class="paper-meta">
      📅 2025-07-04
      | 💬 Accepted paper at MAPR 2025
    </div>
    <details class="paper-abstract">
      In this paper, we explore the ability of large language models (LLMs) to plan and make decisions through the lens of the traditional Vietnamese board game, \^O \u{A}n Quan. This game, which involves a series of strategic token movements and captures, offers a unique environment for evaluating the decision-making and strategic capabilities of LLMs. Specifically, we develop various agent personas, ranging from aggressive to defensive, and employ the \^O \u{A}n Quan game as a testbed for assessing LLM performance across different strategies. Through experimentation with models like Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct, and Llama-3.3-70B-Instruct, we aim to understand how these models execute strategic decision-making, plan moves, and manage dynamic game states. The results will offer insights into the strengths and weaknesses of LLMs in terms of reasoning and strategy, contributing to a deeper understanding of their general capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03662v1">Re-Emergent Misalignment: How Narrow Fine-Tuning Erodes Safety Alignment in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-04
    </div>
    <details class="paper-abstract">
      Recent work has shown that fine-tuning large language models (LLMs) on code with security vulnerabilities can result in misaligned and unsafe behaviors across broad domains. These results prompted concerns about the emergence of harmful behaviors from narrow domain fine-tuning. In this paper, we contextualize these findings by analyzing how such narrow adaptation impacts the internal mechanisms and behavioral manifestations of LLMs. Through a series of experiments covering output probability distributions, loss and gradient vector geometry, layer-wise activation dynamics, and activation space dimensions, we find that behaviors attributed to "emergent misalignment" may be better interpreted as an erosion of prior alignment. We show that fine tuning on insecure code induces internal changes that oppose alignment. Further, we identify a shared latent dimension in the model's activation space that governs alignment behavior. We show that this space is activated by insecure code and by misaligned responses more generally, revealing how narrow fine-tuning can degrade general safety behavior by interfering with shared internal mechanisms. Our findings offer a mechanistic interpretation for previously observed misalignment phenomena, and highlights the fragility of alignment in LLMs. The results underscore the need for more robust fine-tuning strategies that preserve intended behavior across domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03659v1">Specification-Guided Repair of Arithmetic Errors in Dafny Programs using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-04
    </div>
    <details class="paper-abstract">
      Formal verification offers strong assurances of software correctness. However, debugging and repairing the underlying faults can be complex and time-consuming when verification fails. Automated Program Repair (APR) aims to ease this by automatically identifying and fixing faults. Traditional APR techniques often depend on test suites for validation, but these may fail to capture all scenarios. In contrast, formal specifications provide stronger correctness criteria for effective repairs. We present an innovative APR tool for Dafny, a verification-aware programming language that uses formal specifications - including pre-conditions, post-conditions, and invariants - as oracles for fault localization and repair. Assuming the correctness of the specifications and focusing on arithmetic bugs, we localize faults through a series of steps, which include using Hoare Logic to determine the state of each statement within the program and state-of-the-art Large Language Models (LLMs) to synthesize candidate fixes. The chosen models were GPT-4o mini, Llama 3, Mistral 7B, and Llemma 7B. We evaluate our approach using DafnyBench, a benchmark of real-world Dafny programs. Our tool achieves 89.6% accuracy in fault localization, with GPT-4o mini yielding the highest repair success rate (74.18%). These results highlight the potential of combining formal reasoning with LLM-driven program synthesis for automated program repair.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.08276v2">Exploring Robustness of LLMs to Paraphrasing Based on Sociodemographic Factors</a></div>
    <div class="paper-meta">
      📅 2025-07-04
    </div>
    <details class="paper-abstract">
      Despite their linguistic prowess, LLMs have been shown to be vulnerable to small input perturbations. While robustness to local adversarial changes has been studied, robustness to global modifications such as different linguistic styles remains underexplored. Therefore, we take a broader approach to explore a wider range of variations across sociodemographic dimensions. We extend the SocialIQA dataset to create diverse paraphrased sets conditioned on sociodemographic factors (age and gender). The assessment aims to provide a deeper understanding of LLMs in (a) their capability of generating demographic paraphrases with engineered prompts and (b) their capabilities in interpreting real-world, complex language scenarios. We also perform a reliability analysis of the generated paraphrases looking into linguistic diversity and perplexity as well as manual evaluation. We find that demographic-based paraphrasing significantly impacts the performance of language models, indicating that the subtleties of linguistic variation remain a significant challenge. We will make the code and dataset available for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.08203v2">ArithmAttack: Evaluating Robustness of LLMs to Noisy Context in Math Problem Solving</a></div>
    <div class="paper-meta">
      📅 2025-07-04
      | 💬 Accepted to LLMSEC Workshop at ACL 2025
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have shown impressive capabilities in math problem-solving tasks, their robustness to noisy inputs is not well-studied. We propose ArithmAttack to examine how robust the LLMs are when they encounter noisy prompts that contain extra noise in the form of punctuation marks. While being easy to implement, ArithmAttack does not cause any information loss since words are not added or deleted from the context. We evaluate the robustness of eight LLMs, including LLama3, Mistral, Mathstral, and DeepSeek on noisy GSM8K and MultiArith datasets. Our experiments suggest that all the studied models show vulnerability to such noise, with more noise leading to poorer performances.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09066v2">Probing Latent Subspaces in LLM for AI Security: Identifying and Manipulating Adversarial States</a></div>
    <div class="paper-meta">
      📅 2025-07-04
      | 💬 4 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities across various tasks, yet they remain vulnerable to adversarial manipulations such as jailbreaking via prompt injection attacks. These attacks bypass safety mechanisms to generate restricted or harmful content. In this study, we investigated the underlying latent subspaces of safe and jailbroken states by extracting hidden activations from a LLM. Inspired by attractor dynamics in neuroscience, we hypothesized that LLM activations settle into semi stable states that can be identified and perturbed to induce state transitions. Using dimensionality reduction techniques, we projected activations from safe and jailbroken responses to reveal latent subspaces in lower dimensional spaces. We then derived a perturbation vector that when applied to safe representations, shifted the model towards a jailbreak state. Our results demonstrate that this causal intervention results in statistically significant jailbreak responses in a subset of prompts. Next, we probed how these perturbations propagate through the model's layers, testing whether the induced state change remains localized or cascades throughout the network. Our findings indicate that targeted perturbations induced distinct shifts in activations and model responses. Our approach paves the way for potential proactive defenses, shifting from traditional guardrail based methods to preemptive, model agnostic techniques that neutralize adversarial states at the representation level.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03619v1">Blackbox Dataset Inference for LLM</a></div>
    <div class="paper-meta">
      📅 2025-07-04
    </div>
    <details class="paper-abstract">
      Today, the training of large language models (LLMs) can involve personally identifiable information and copyrighted material, incurring dataset misuse. To mitigate the problem of dataset misuse, this paper explores \textit{dataset inference}, which aims to detect if a suspect model $\mathcal{M}$ used a victim dataset $\mathcal{D}$ in training. Previous research tackles dataset inference by aggregating results of membership inference attacks (MIAs) -- methods to determine whether individual samples are a part of the training dataset. However, restricted by the low accuracy of MIAs, previous research mandates grey-box access to $\mathcal{M}$ to get intermediate outputs (probabilities, loss, perplexity, etc.) for obtaining satisfactory results. This leads to reduced practicality, as LLMs, especially those deployed for profits, have limited incentives to return the intermediate outputs. In this paper, we propose a new method of dataset inference with only black-box access to the target model (i.e., assuming only the text-based responses of the target model are available). Our method is enabled by two sets of locally built reference models, one set involving $\mathcal{D}$ in training and the other not. By measuring which set of reference model $\mathcal{M}$ is closer to, we determine if $\mathcal{M}$ used $\mathcal{D}$ for training. Evaluations of real-world LLMs in the wild show that our method offers high accuracy in all settings and presents robustness against bypassing attempts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19937v2">ALAS: Measuring Latent Speech-Text Alignment For Spoken Language Understanding In Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used in Spoken Language Understanding (SLU), where effective multimodal learning depends on the alignment between audio and text. Despite various fusion methods, no standard metric exists to assess this alignment. This work introduces ALAS (Automatic Latent Alignment Score), a metric that evaluates alignment by measuring correlations between audio and text representations across transformer layers. Experiments on Spoken Question Answering and Emotion Recognition show that ALAS captures meaningful patterns across tasks and layers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03605v1">Behaviour Space Analysis of LLM-driven Meta-heuristic Discovery</a></div>
    <div class="paper-meta">
      📅 2025-07-04
    </div>
    <details class="paper-abstract">
      We investigate the behaviour space of meta-heuristic optimisation algorithms automatically generated by Large Language Model driven algorithm discovery methods. Using the Large Language Evolutionary Algorithm (LLaMEA) framework with a GPT o4-mini LLM, we iteratively evolve black-box optimisation heuristics, evaluated on 10 functions from the BBOB benchmark suite. Six LLaMEA variants, featuring different mutation prompt strategies, are compared and analysed. We log dynamic behavioural metrics including exploration, exploitation, convergence and stagnation measures, for each run, and analyse these via visual projections and network-based representations. Our analysis combines behaviour-based projections, Code Evolution Graphs built from static code features, performance convergence curves, and behaviour-based Search Trajectory Networks. The results reveal clear differences in search dynamics and algorithm structures across LLaMEA configurations. Notably, the variant that employs both a code simplification prompt and a random perturbation prompt in a 1+1 elitist evolution strategy, achieved the best performance, with the highest Area Over the Convergence Curve. Behaviour-space visualisations show that higher-performing algorithms exhibit more intensive exploitation behaviour and faster convergence with less stagnation. Our findings demonstrate how behaviour-space analysis can explain why certain LLM-designed heuristics outperform others and how LLM-driven algorithm discovery navigates the open-ended and complex search space of algorithms. These findings provide insights to guide the future design of adaptive LLM-driven algorithm generators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03378v1">Making Sense of Korean Sentences: A Comprehensive Evaluation of LLMs through KoSEnd Dataset</a></div>
    <div class="paper-meta">
      📅 2025-07-04
      | 💬 ACL 2025 student research workshop
    </div>
    <details class="paper-abstract">
      Although LLMs have made significant progress in various languages, there are still concerns about their effectiveness with low-resource agglutinative languages compared to languages such as English. In this study, we focused on Korean, a language known for its complex sentence endings, and evaluated LLMs on this challenging aspect. We introduce the Korean Sentence Endings (KoSEnd) dataset, which includes 3,000 sentences, each annotated for the naturalness of 15 sentence ending forms. These were collected from diverse sources to cover a range of contexts. We evaluated 11 LLMs to assess their understanding of Korean sentence endings, analyzing them based on parameter count and prediction consistency. Notably, we found that informing models about the possibility of missing sentence endings improved performance, highlighting the impact of explicitly considering certain linguistic features.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03336v1">Disambiguation-Centric Finetuning Makes Enterprise Tool-Calling LLMs More Realistic and Less Risky</a></div>
    <div class="paper-meta">
      📅 2025-07-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly tasked with invoking enterprise APIs, yet they routinely falter when near-duplicate tools vie for the same user intent or when required arguments are left underspecified. We introduce DiaFORGE (Dialogue Framework for Organic Response Generation & Evaluation), a disambiguation-centric, three-stage pipeline that (i) synthesizes persona-driven, multi-turn dialogues in which the assistant must distinguish among highly similar tools, (ii) performs supervised fine-tuning of open-source models with reasoning traces across 3B - 70B parameters, and (iii) evaluates real-world readiness via a dynamic suite that redeploys each model in a live agentic loop and reports end-to-end goal completion alongside conventional static metrics. On our dynamic benchmark DiaBENCH, models trained with DiaFORGE raise tool-invocation success by 27 pp over GPT-4o and by 49 pp over Claude-3.5-Sonnet, both under optimized prompting. To spur further research, we release an open corpus of 5000 production-grade enterprise API specifications paired with rigorously validated, disambiguation-focused dialogues, offering a practical blueprint for building reliable, enterprise-ready tool-calling agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03327v1">Read Quietly, Think Aloud: Decoupling Comprehension and Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-04
      | 💬 Under submission
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable proficiency in understanding text and generating high-quality responses. However, a critical distinction from human cognition is their typical lack of a distinct internal `reading' or deliberation phase before `speaking' (i.e., generating text). Humans often engage in silent reading to comprehend context and formulate thoughts prior to articulation. This paper investigates methods to imbue LLMs with a similar capacity for internal processing. We introduce and evaluate techniques that encourage LLMs to `read silently.' Our findings indicate that even a straightforward approach, such as providing the model with an initial contextual prompt or `reading space' before it begins predicting subsequent tokens for the final output, can yield significant performance improvements. We further enhance this concept by developing a `reading buddy' architecture, where an auxiliary component silently processes the input and provides refined contextual insights to the primary generation model. These approaches aim to foster deeper understanding from LLMs so that they can produce better reasoned responses, moving them one step closer to more human-like text processing. Our results indicate that these simple techniques can provide surprisingly strong impact on accuracy with multiple point accuracy boost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03326v1">Mirror in the Model: Ad Banner Image Generation via Reflective Multi-LLM and Multi-modal Agents</a></div>
    <div class="paper-meta">
      📅 2025-07-04
    </div>
    <details class="paper-abstract">
      Recent generative models such as GPT-4o have shown strong capabilities in producing high-quality images with accurate text rendering. However, commercial design tasks like advertising banners demand more than visual fidelity -- they require structured layouts, precise typography, consistent branding, and more. In this paper, we introduce MIMO (Mirror In-the-Model), an agentic refinement framework for automatic ad banner generation. MIMO combines a hierarchical multi-modal agent system (MIMO-Core) with a coordination loop (MIMO-Loop) that explores multiple stylistic directions and iteratively improves design quality. Requiring only a simple natural language based prompt and logo image as input, MIMO automatically detects and corrects multiple types of errors during generation. Experiments show that MIMO significantly outperforms existing diffusion and LLM-based baselines in real-world banner design scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02881v3">Rewriting Pre-Training Data Boosts LLM Performance in Math and Code</a></div>
    <div class="paper-meta">
      📅 2025-07-04
    </div>
    <details class="paper-abstract">
      The performance of large language models (LLMs) in program synthesis and mathematical reasoning is fundamentally limited by the quality of their pre-training corpora. We introduce two openly licensed datasets, released under the Llama 3.3 Community License, that significantly enhance LLM performance by systematically rewriting public data. SwallowCode (approximately 16.1 billion tokens) refines Python snippets from The-Stack-v2 through a novel four-stage pipeline: syntax validation, pylint-based style filtering, and a two-stage LLM rewriting process that enforces style conformity and transforms snippets into self-contained, algorithmically efficient examples. Unlike prior methods that rely on exclusionary filtering or limited transformations, our transform-and-retain approach upgrades low-quality code, maximizing data utility. SwallowMath (approximately 2.3 billion tokens) enhances Finemath-4+ by removing boilerplate, restoring context, and reformatting solutions into concise, step-by-step explanations. Within a fixed 50 billion token training budget, continual pre-training of Llama-3.1-8B with SwallowCode boosts pass@1 by +17.0 on HumanEval and +17.7 on HumanEval+ compared to Stack-Edu, surpassing the baseline model's code generation capabilities. Similarly, substituting SwallowMath yields +12.4 accuracy on GSM8K and +7.6 on MATH. Ablation studies confirm that each pipeline stage contributes incrementally, with rewriting delivering the largest gains. All datasets, prompts, and checkpoints are publicly available, enabling reproducible research and advancing LLM pre-training for specialized domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03305v1">Analysis and Optimized CXL-Attached Memory Allocation for Long-Context LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-07-04
      | 💬 9 pages, 10 figures, 2 tables
    </div>
    <details class="paper-abstract">
      The growing prevalence of Large Language Models (LLMs) and their substantial memory requirements have prompted renewed interest in CPU offloading as a method to compensate for limited GPU memory. In particular, when CPU memory is leveraged to temporarily store intermediate states of LLMs, CPU memory becomes a new bottleneck and soon reaches the capacity limitation of commodity CPUs. In this work, we investigate the effectiveness of Compute Express Link (CXL) add-in card (AIC) memory as an extension to CPU memory, enabling larger model sizes and longer context lengths during fine-tuning. Through extensive benchmarking, this study quantifies the performance overhead introduced by transferring data between CXL memory, CPU, and GPUs, focusing on how concurrency and data volume influence bandwidth utilization and latency. This study also compares CPUbased optimizer steps when model parameters, gradients, and optimizer states reside in local memory versus CXL memory, revealing that naive adoption of CXL often degrades performance during the optimizer phase. To overcome these challenges, this study proposes a CXL-aware allocation to strategically partition CPU offloading workloads across both local and CXL memory. This study further demonstrates that employing multiple AICs significantly reduces bandwidth contention, thus improving scalability. Experimental results show that these optimizations enable efficient long-context LLM fine-tuning, underscoring CXL as a promising avenue for unlocking the full potential of CPU offloading in long-context LLM fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03294v1">MGAA: Multi-Granular Adaptive Allocation fof Low-Rank Compression of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-04
      | 💬 13 pages, 8 figures
    </div>
    <details class="paper-abstract">
      The enormous parameter scale of large language models (LLMs) has made model compression a research hotspot, which aims to alleviate computational resource demands during deployment and inference. As a promising direction, low-rank approximation technique has made remarkable achievements. Nevertheless, unfortunately, the vast majority of studies to low-rank approximation compression generally apply uniform compression ratios across all weight matrices, while disregarding their inherently differentiated impacts on the model's performance. Although a few recent work attempts to employ heuristic search strategies to achieve the optimal parameter allocation, such strategies are computationally inefficient and lose the generalization ability in the era of LLMs. In this study, we propose a novel parameter Multi-Granular Adaptive Allocation (MGAA) method, which can adaptively allocate parameters between and within sublayers without task-specific evaluations in the compression process. MGAA consists of two components: 1) Among different sublayers, it assigns compression ratios based on their cosine similarity between inputs and outputs, allowing for a more tailored compression in sublayers with varying degrees of importance, and 2) Within each sublayer, it allocates different compression ratios to weight matrices based on their energy distribution characteristics, ensuring a consistent energy retention ratio while optimizing compression efficiency. Comprehensive evaluations of MGAA across multiple LLMs backbone models and benchmark datasets demonstrate its superior performance. Additionally, we apply our MGAA to multimodal model LLaVA, exhibiting remarkable performance improvements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03293v1">LTLCrit: A Temporal Logic-based LLM Critic for Safe and Efficient Embodied Agents</a></div>
    <div class="paper-meta">
      📅 2025-07-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated promise in reasoning tasks and general decision-making in static environments. In long-term planning tasks, however, errors tend to accumulate, often leading to unsafe or inefficient behavior, limiting their use in general-purpose settings. We propose a modular actor-critic architecture in which an LLM actor is guided by LTLCrit, a trajectory-level LLM critic that communicates via linear temporal logic (LTL). Our setup combines the reasoning strengths of language models with the guarantees of formal logic. The actor selects high-level actions from natural language observations, while the critic analyzes full trajectories and proposes new LTL constraints that shield the actor from future unsafe or inefficient behavior. The architecture supports both fixed, hand-specified safety constraints and adaptive, learned soft constraints that promote long-term efficiency. Our architecture is model-agnostic: any LLM-based planner can serve as the actor, and LTLCrit serves as a logic-generating wrapper. We formalize planning as graph traversal under symbolic constraints, allowing LTLCrit to analyze failed or suboptimal trajectories and generate new temporal logic rules that improve future behavior. We evaluate our system on the Minecraft diamond-mining benchmark, achieving 100% completion rates and improving efficiency compared to baseline LLM planners. Our results suggest that enabling LLMs to supervise each other through logic is a powerful and flexible paradigm for safe, generalizable decision making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.04428v2">MoralBench: Moral Evaluation of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-04
      | 💬 Accepted to ACM SIGKDD Explorations Volume 27 Issue 1
    </div>
    <details class="paper-abstract">
      In the rapidly evolving field of artificial intelligence, large language models (LLMs) have emerged as powerful tools for a myriad of applications, from natural language processing to decision-making support systems. However, as these models become increasingly integrated into societal frameworks, the imperative to ensure they operate within ethical and moral boundaries has never been more critical. This paper introduces a novel benchmark designed to measure and compare the moral reasoning capabilities of LLMs. We present the first comprehensive dataset specifically curated to probe the moral dimensions of LLM outputs, addressing a wide range of ethical dilemmas and scenarios reflective of real-world complexities. The main contribution of this work lies in the development of benchmark datasets and metrics for assessing the moral identity of LLMs, which accounts for nuance, contextual sensitivity, and alignment with human ethical standards. Our methodology involves a multi-faceted approach, combining quantitative analysis with qualitative insights from ethics scholars to ensure a thorough evaluation of model performance. By applying our benchmark across several leading LLMs, we uncover significant variations in moral reasoning capabilities of different models. These findings highlight the importance of considering moral reasoning in the development and evaluation of LLMs, as well as the need for ongoing research to address the biases and limitations uncovered in our study. We publicly release the benchmark at https://drive.google.com/drive/u/0/folders/1k93YZJserYc2CkqP8d4B3M3sgd3kA8W7 and also open-source the code of the project at https://github.com/agiresearch/MoralBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03254v1">CodeAgents: A Token-Efficient Framework for Codified Multi-Agent Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-04
    </div>
    <details class="paper-abstract">
      Effective prompt design is essential for improving the planning capabilities of large language model (LLM)-driven agents. However, existing structured prompting strategies are typically limited to single-agent, plan-only settings, and often evaluate performance solely based on task accuracy - overlooking critical factors such as token efficiency, modularity, and scalability in multi-agent environments. To address these limitations, we introduce CodeAgents, a prompting framework that codifies multi-agent reasoning and enables structured, token-efficient planning in multi-agent systems. In CodeAgents, all components of agent interaction - Task, Plan, Feedback, system roles, and external tool invocations - are codified into modular pseudocode enriched with control structures (e.g., loops, conditionals), boolean logic, and typed variables. This design transforms loosely connected agent plans into cohesive, interpretable, and verifiable multi-agent reasoning programs. We evaluate the proposed framework across three diverse benchmarks - GAIA, HotpotQA, and VirtualHome - using a range of representative LLMs. Results show consistent improvements in planning performance, with absolute gains of 3-36 percentage points over natural language prompting baselines. On VirtualHome, our method achieves a new state-of-the-art success rate of 56%. In addition, our approach reduces input and output token usage by 55-87% and 41-70%, respectively, underscoring the importance of token-aware evaluation metrics in the development of scalable multi-agent LLM systems. The code and resources are available at: https://anonymous.4open.science/r/CodifyingAgent-5A86
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10401v2">HPCTransCompile: An AI Compiler Generated Dataset for High-Performance CUDA Transpilation and LLM Preliminary Exploration</a></div>
    <div class="paper-meta">
      📅 2025-07-04
    </div>
    <details class="paper-abstract">
      The rapid growth of deep learning has driven exponential increases in model parameters and computational demands. NVIDIA GPUs and their CUDA-based software ecosystem provide robust support for parallel computing, significantly alleviating computational bottlenecks. Meanwhile, due to the cultivation of user programming habits and the high performance of GPUs, the CUDA ecosystem has established a dominant position in the field of parallel software. This dominance requires other hardware platforms to support CUDA-based software with performance portability. However, translating CUDA code to other platforms poses significant challenges due to differences in parallel programming paradigms and hardware architectures. Existing approaches rely on language extensions, domain-specific languages (DSLs), or compilers but face limitations in workload coverage and generalizability. Moreover, these methods often incur substantial development costs. Recently, LLMs have demonstrated extraordinary potential in various vertical domains, especially in code-related tasks. However, the performance of existing LLMs in CUDA transpilation, particularly for high-performance code, remains suboptimal. To address these challenges, we propose a novel framework for generating high-performance CUDA and corresponding platform code pairs, leveraging AI compiler and automatic optimization technology. We further enhance the framework with a graph-based data augmentation method and introduce HPCTransEval, a benchmark for evaluating LLM performance on CUDA transpilation. We conduct experiments using CUDA-to-CPU transpilation as a case study on leading LLMs. The speedup ratio of the CPU operators has an average improvemnet of 43.8\%, highlighting the potential of LLMs to address compatibility challenges within the CUDA ecosystem. Our code is available at https://github.com/PJLAB-CHIP/HPCTransCompile.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06845v6">7B Fully Open Source Moxin-LLM/VLM -- From Pretraining to GRPO-based Reinforcement Learning Enhancement</a></div>
    <div class="paper-meta">
      📅 2025-07-04
    </div>
    <details class="paper-abstract">
      Recently, Large Language Models (LLMs) have undergone a significant transformation, marked by a rapid rise in both their popularity and capabilities. Leading this evolution are proprietary LLMs like GPT-4 and GPT-o1, which have captured widespread attention in the AI community due to their remarkable performance and versatility. Simultaneously, open-source LLMs, such as LLaMA, have made great contributions to the ever-increasing popularity of LLMs due to the ease to customize and deploy the models across diverse applications. Although open-source LLMs present unprecedented opportunities for innovation and research, the commercialization of LLMs has raised concerns about transparency, reproducibility, and safety. Many open-source LLMs fail to meet fundamental transparency requirements by withholding essential components like training code and data, which may hinder further innovations on LLMs. To mitigate this issue, we introduce Moxin 7B, a fully open-source LLM developed, adhering to principles of open science, open source, open data, and open access. We release the pre-training code and configurations, training and fine-tuning datasets, and intermediate and final checkpoints, aiming to make continuous commitments to fully open-source LLMs. After pre-training the base model, we finetune the Moxin Base model with SOTA post-training framework and instruction data to obtain Moxin Instruct model. To improve the reasoning capability, we further finetune our Instruct model with chain-of-thought data distilled from DeepSeek R1, and then use Group Relative Policy Optimization (GRPO) following DeepSeek R1 to finetune our model, leading to the Moxin Reasoning model. Moreover, we develop our vision language model based on our Moxin model. Experiments show that our models achieve superior performance in various evaluations such as zero-shot evaluation, few-shot evaluation, and CoT evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05282v1">Exploring LLM Capabilities in Extracting DCAT-Compatible Metadata for Data Cataloging</a></div>
    <div class="paper-meta">
      📅 2025-07-04
    </div>
    <details class="paper-abstract">
      Efficient data exploration is crucial as data becomes increasingly important for accelerating processes, improving forecasts and developing new business models. Data consumers often spend 25-98 % of their time searching for suitable data due to the exponential growth, heterogeneity and distribution of data. Data catalogs can support and accelerate data exploration by using metadata to answer user queries. However, as metadata creation and maintenance is often a manual process, it is time-consuming and requires expertise. This study investigates whether LLMs can automate metadata maintenance of text-based data and generate high-quality DCAT-compatible metadata. We tested zero-shot and few-shot prompting strategies with LLMs from different vendors for generating metadata such as titles and keywords, along with a fine-tuned model for classification. Our results show that LLMs can generate metadata comparable to human-created content, particularly on tasks that require advanced semantic understanding. Larger models outperformed smaller ones, and fine-tuning significantly improves classification accuracy, while few-shot prompting yields better results in most cases. Although LLMs offer a faster and reliable way to create metadata, a successful application requires careful consideration of task-specific criteria and domain context.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05279v1">ReservoirChat: Interactive Documentation Enhanced with LLM and Knowledge Graph for ReservoirPy</a></div>
    <div class="paper-meta">
      📅 2025-07-04
    </div>
    <details class="paper-abstract">
      We introduce a tool designed to improve the capabilities of Large Language Models (LLMs) in assisting with code development using the ReservoirPy library, as well as in answering complex questions in the field of Reservoir Computing. By incorporating external knowledge through Retrieval-Augmented Generation (RAG) and knowledge graphs, our approach aims to reduce hallucinations and increase the factual accuracy of generated responses. The system provides an interactive experience similar to ChatGPT, tailored specifically for ReservoirPy, enabling users to write, debug, and understand Python code while accessing reliable domain-specific insights. In our evaluation, while proprietary models such as ChatGPT-4o and NotebookLM performed slightly better on general knowledge questions, our model outperformed them on coding tasks and showed a significant improvement over its base model, Codestral-22B.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02735v1">Meta SecAlign: A Secure Foundation LLM Against Prompt Injection Attacks</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      Prompt injection attacks pose a significant security threat to LLM-integrated applications. Model-level defenses have shown strong effectiveness, but are currently deployed into commercial-grade models in a closed-source manner. We believe open-source models are needed by the AI security community, where co-development of attacks and defenses through open research drives scientific progress in mitigation against prompt injection attacks. To this end, we develop Meta SecAlign, the first open-source and open-weight LLM with built-in model-level defense that achieves commercial-grade model performance. We provide complete details of our training recipe, which utilizes an improved version of the SOTA SecAlign defense. Evaluations on 9 utility benchmarks and 7 security benchmarks show that Meta SecAlign, despite being trained on a generic instruction-tuning dataset, confers security in unseen downstream tasks, including tool-calling and agentic web navigation, in addition general instruction-following. Our best model -- Meta-SecAlign-70B -- achieves state-of-the-art robustness against prompt injection attacks and comparable utility to closed-source commercial LLM with model-level defense.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01631v2">Gradient-Based Model Fingerprinting for LLM Similarity Detection and Family Classification</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) become integral software components in modern applications, unauthorized model derivations through fine-tuning, merging, and redistribution have emerged as critical software engineering challenges. Unlike traditional software where clone detection and license compliance are well-established, the LLM ecosystem lacks effective mechanisms to detect model lineage and enforce licensing agreements. This gap is particularly problematic when open-source model creators, such as Meta's LLaMA, require derivative works to maintain naming conventions for attribution, yet no technical means exist to verify compliance. To fill this gap, treating LLMs as software artifacts requiring provenance tracking, we present TensorGuard, a gradient-based fingerprinting framework for LLM similarity detection and family classification. Our approach extracts model-intrinsic behavioral signatures by analyzing gradient responses to random input perturbations across tensor layers, operating independently of training data, watermarks, or specific model formats. TensorGuard supports the widely-adopted safetensors format and constructs high-dimensional fingerprints through statistical analysis of gradient features. These fingerprints enable two complementary capabilities: direct pairwise similarity assessment between arbitrary models through distance computation, and systematic family classification of unknown models via the K-Means clustering algorithm with domain-informed centroid initialization using known base models. Experimental evaluation on 58 models comprising 8 base models and 50 derivatives across five model families (Llama, Qwen, Gemma, Phi, Mistral) demonstrates 94% classification accuracy under our centroid-initialized K-Means clustering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.00863v2">Next-Token Prediction Task Assumes Optimal Data Ordering for LLM Training in Proof Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      In the field of large language model (LLM)-based proof generation, despite extensive training on large datasets such as ArXiv, LLMs still exhibit only modest performance on proving tasks of moderate difficulty. We believe that this is partly due to the widespread presence of suboptimal ordering within the data for each proof used in training. For example, published proofs often follow a purely logical order, where each step logically proceeds from the previous steps based on the deductive rules. This order is designed to facilitate the verification of the proof's soundness, rather than to help people and models learn the discovery process of the proof. In proof generation, we argue that the optimal order for one training data sample occurs when the relevant intermediate supervision for a particular proof step in the proof is always positioned to the left of that proof step. We call such order the intuitively sequential order. We validate our claims using two tasks: intuitionistic propositional logic theorem-proving and digit multiplication. Our experiments verify the order effect and provide support for our explanations. We demonstrate that training is most effective when the proof is in the intuitively sequential order. Moreover, the order effect and the performance gap between models trained on different data orders can be substantial -- with an 11 percent improvement in proof success rate observed in the propositional logic theorem-proving task, between models trained on the optimal order compared to the worst order. Lastly, we define a common type of order issue in advanced math proofs and find that 17.3 percent of theorems with nontrivial proofs in the first two chapters of a widely used graduate-level mathematics textbook suffer from this issue. A detailed list of those proofs is provided in the appendix.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02699v1">Control at Stake: Evaluating the Security Landscape of LLM-Driven Email Agents</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      The increasing capabilities of LLMs have led to the rapid proliferation of LLM agent apps, where developers enhance LLMs with access to external resources to support complex task execution. Among these, LLM email agent apps represent one of the widely used categories, as email remains a critical communication medium for users. LLM email agents are capable of managing and responding to email using LLM-driven reasoning and autonomously executing user instructions via external email APIs (e.g., send email). However, despite their growing deployment and utility, the security mechanism of LLM email agent apps remains underexplored. Currently, there is no comprehensive study into the potential security risk within these agent apps and their broader implications. In this paper, we conduct the first in-depth and systematic security study of LLM email agents. We propose the Email Agent Hijacking (EAH) attack, which overrides the original prompts of the email agent via external email resources, allowing attackers to gain control of the email agent remotely and further perform specific attack scenarios without user awareness. To facilitate the large-scale evaluation, we propose EAHawk, a pipeline to evaluate the EAH attack of LLM email agent apps. By EAHawk, we performed an empirical study spanning 14 representative LLM agent frameworks, 63 agent apps, 12 LLMs, and 20 email services, which led to the generation of 1,404 real-world email agent instances for evaluation. Experimental results indicate that all 1,404 instances were successfully hijacked; on average, only 2.03 attack attempts are required to control an email agent instance. Even worse, for some LLMs, the average number of attempts needed to achieve full agent control drops to as few as 1.23.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05272v1">FuzzFeed: An Automatic Approach to Weakest Precondition Generation using LLMs and Fuzzing</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      The weakest precondition (WP) of a program describes the largest set of initial states from which all terminating executions of the program satisfy a given postcondition. The generation of WPs is an important task with practical applications in areas ranging from verification to run-time error checking. This paper proposes the combination of Large Language Models (LLMs) and fuzz testing for generating WPs. In pursuit of this goal, we introduce Fuzzing Guidance (FG); FG acts as a means of directing LLMs towards correct WPs using program execution feedback. FG utilises fuzz testing for approximately checking the validity and weakness of candidate WPs, this information is then fed back to the LLM as a means of context refinement. We demonstrate the effectiveness of our approach on a comprehensive benchmark set of deterministic array programs in Java. Our experiments indicate that LLMs are capable of producing viable candidate WPs, and that this ability can be practically enhanced through FG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05269v1">CORE: Benchmarking LLMs Code Reasoning Capabilities through Static Analysis Tasks</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been widely adopted across diverse software engineering domains, such as code generation, program repair, and vulnerability detection. These applications require understanding beyond surface-level code patterns: value propagation, control flow, and interdependence between program elements. However, existing benchmarks primarily evaluate end-to-end outcomes, such as whether code is correctly repaired or generated, leaving the models ability for program semantic reasoning underexplored. This work presents CoRe, a high-quality, human-verified benchmark designed to evaluate LLMs on fundamental static analysis tasks. CoRe includes 12,553 task instances spanning data dependency, control dependency, and information flow across programs written in C/C++, Java, and Python. To ensure semantic diversity and reasoning complexity, we propose a semantics-aware diverse sampling strategy that selects targets and task instances based on structural coverage and dependency depth. We evaluate 10 mainstream LLMs and show that, while they perform well at identifying dependencies, models still struggle with tasks that require deeper semantic understanding and multi-step reasoning. We further conduct qualitative analyses to uncover key challenges, such as complex control structures and backward dependency patterns, offering insights into improving LLMs code reasoning capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02694v1">Can LLMs Identify Critical Limitations within Scientific Research? A Systematic Evaluation on AI Research Papers</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      Peer review is fundamental to scientific research, but the growing volume of publications has intensified the challenges of this expertise-intensive process. While LLMs show promise in various scientific tasks, their potential to assist with peer review, particularly in identifying paper limitations, remains understudied. We first present a comprehensive taxonomy of limitation types in scientific research, with a focus on AI. Guided by this taxonomy, for studying limitations, we present LimitGen, the first comprehensive benchmark for evaluating LLMs' capability to support early-stage feedback and complement human peer review. Our benchmark consists of two subsets: LimitGen-Syn, a synthetic dataset carefully created through controlled perturbations of high-quality papers, and LimitGen-Human, a collection of real human-written limitations. To improve the ability of LLM systems to identify limitations, we augment them with literature retrieval, which is essential for grounding identifying limitations in prior scientific findings. Our approach enhances the capabilities of LLM systems to generate limitations in research papers, enabling them to provide more concrete and constructive feedback.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02620v1">FlowSpec: Continuous Pipelined Speculative Decoding for Efficient Distributed LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 16 pages, and the last 3 are appendix
    </div>
    <details class="paper-abstract">
      Distributed inference serves as a promising approach to enabling the inference of large language models (LLMs) at the network edge. It distributes the inference process to multiple devices to ensure that the LLMs can fit into the device memory. Recent pipeline-based approaches have the potential to parallelize communication and computation, which helps reduce inference latency. However, the benefit diminishes when the inference request at the network edge is sparse, where pipeline is typically at low utilization. To enable efficient distributed LLM inference at the edge, we propose \textbf{FlowSpec}, a pipeline-parallel tree-based speculative decoding framework. FlowSpec incorporates three key mechanisms to improve decoding efficiency: 1) score-based step-wise verification prioritizes more important draft tokens to bring earlier accpeted tokens; 2) efficient draft management to prune invalid tokens while maintaining correct causal relationship during verification; 3) dynamic draft expansion strategies to supply high-quality speculative inputs. These techniques work in concert to enhance both pipeline utilization and speculative efficiency. We evaluate FlowSpec on a real-world testbed with other baselines. Experimental results demonstrate that our proposed framework significantly improves inference speed across diverse models and configurations, achieving speedup ratios 1.36$\times$-1.77$\times$ compared to baselines. Our code is publicly available at \href{https://github.com/Leosang-lx/FlowSpec#}{https://github.com/Leosang-lx/FlowSpec\#}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01334v2">Symbolic or Numerical? Understanding Physics Problem Solving in Reasoning LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      Navigating the complexities of physics reasoning has long been a difficult task for Large Language Models (LLMs), requiring a synthesis of profound conceptual understanding and adept problem-solving techniques. In this study, we investigate the application of advanced instruction-tuned reasoning models, such as Deepseek-R1, to address a diverse spectrum of physics problems curated from the challenging SciBench benchmark. Our comprehensive experimental evaluation reveals the remarkable capabilities of reasoning models. Not only do they achieve state-of-the-art accuracy in answering intricate physics questions, but they also generate distinctive reasoning patterns that emphasize on symbolic derivation. Furthermore, our findings indicate that even for these highly sophisticated reasoning models, the strategic incorporation of few-shot prompting can still yield measurable improvements in overall accuracy, highlighting the potential for continued performance gains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12532v3">MedAide: Information Fusion and Anatomy of Medical Intents via LLM-based Agent Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 LLM-based Multi-Agent Collaboration for Medical Applications
    </div>
    <details class="paper-abstract">
      In healthcare intelligence, the ability to fuse heterogeneous, multi-intent information from diverse clinical sources is fundamental to building reliable decision-making systems. Large Language Model (LLM)-driven information interaction systems currently showing potential promise in the healthcare domain. Nevertheless, they often suffer from information redundancy and coupling when dealing with complex medical intents, leading to severe hallucinations and performance bottlenecks. To this end, we propose MedAide, an LLM-based medical multi-agent collaboration framework designed to enable intent-aware information fusion and coordinated reasoning across specialized healthcare domains. Specifically, we introduce a regularization-guided module that combines syntactic constraints with retrieval augmented generation to decompose complex queries into structured representations, facilitating fine-grained clinical information fusion and intent resolution. Additionally, a dynamic intent prototype matching module is proposed to utilize dynamic prototype representation with a semantic similarity matching mechanism to achieve adaptive recognition and updating of the agent's intent in multi-round healthcare dialogues. Ultimately, we design a rotation agent collaboration mechanism that introduces dynamic role rotation and decision-level information fusion across specialized medical agents. Extensive experiments are conducted on four medical benchmarks with composite intents. Experimental results from automated metrics and expert doctor evaluations show that MedAide outperforms current LLMs and improves their medical proficiency and strategic reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08500v2">Aerial Vision-and-Language Navigation via Semantic-Topo-Metric Representation Guided LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      Aerial Vision-and-Language Navigation (VLN) is a novel task enabling Unmanned Aerial Vehicles (UAVs) to navigate in outdoor environments through natural language instructions and visual cues. It remains challenging due to the complex spatial relationships in outdoor aerial scenes. In this paper, we propose an end-to-end zero-shot framework for aerial VLN tasks, where the large language model (LLM) is introduced as our agent for action prediction. Specifically, we develop a novel Semantic-Topo-Metric Representation (STMR) to enhance the spatial reasoning ability of LLMs. This is achieved by extracting and projecting instruction-related semantic masks of landmarks into a top-down map that contains the location information of surrounding landmarks. Further, this map is transformed into a matrix representation with distance metrics as the text prompt to the LLM, for action prediction according to the instruction. Experiments conducted in real and simulation environments have successfully proved the effectiveness and robustness of our method, achieving 15.9% and 12.5% improvements (absolute) in Oracle Success Rate (OSR) on AerialVLN-S dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02503v1">Continual Gradient Low-Rank Projection Fine-Tuning for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 15 pages, 6 figures, accepted by ACL 2025 main
    </div>
    <details class="paper-abstract">
      Continual fine-tuning of Large Language Models (LLMs) is hampered by the trade-off between efficiency and expressiveness. Low-Rank Adaptation (LoRA) offers efficiency but constrains the model's ability to learn new tasks and transfer knowledge due to its low-rank nature and reliance on explicit parameter constraints. We propose GORP (Gradient LOw Rank Projection) for Continual Learning, a novel training strategy that overcomes these limitations by synergistically combining full and low-rank parameters and jointly updating within a unified low-rank gradient subspace. GORP expands the optimization space while preserving efficiency and mitigating catastrophic forgetting. Extensive experiments on continual learning benchmarks demonstrate GORP's superior performance compared to existing state-of-the-art approaches. Code is available at https://github.com/Wcxwcxw/GORP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.07016v5">Delving into LLM-assisted writing in biomedical publications through excess vocabulary</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 v5: Reverting to v3
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) like ChatGPT can generate and revise text with human-level performance. These models come with clear limitations: they can produce inaccurate information, reinforce existing biases, and be easily misused. Yet, many scientists use them for their scholarly writing. But how wide-spread is such LLM usage in the academic literature? To answer this question for the field of biomedical research, we present an unbiased, large-scale approach: we study vocabulary changes in over 15 million biomedical abstracts from 2010--2024 indexed by PubMed, and show how the appearance of LLMs led to an abrupt increase in the frequency of certain style words. This excess word analysis suggests that at least 13.5% of 2024 abstracts were processed with LLMs. This lower bound differed across disciplines, countries, and journals, reaching 40% for some subcorpora. We show that LLMs have had an unprecedented impact on scientific writing in biomedical research, surpassing the effect of major world events such as the Covid pandemic.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07967v2">Code Digital Twin: Empowering LLMs with Tacit Knowledge for Complex Software Maintenance</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 A vision paper that will be continuously updated
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have demonstrated promise in software engineering tasks like code completion and generation, their support for the maintenance of complex software systems remains limited. These models often struggle with understanding the tacit knowledge embedded in systems, such as responsibility allocation and collaboration across different modules. To address this gap, we introduce the concept and framework of \textbf{Code Digital Twin}, a conceptual representation of tacit knowledge that captures the concepts, functionalities, and design rationales behind code elements, co-evolving with the software. A code digital twin is constructed using a methodology that combines knowledge extraction from both structured and unstructured sources--such as source code, documentation, and change histories--leveraging LLMs, static analysis tools, and human expertise. This framework can empower LLMs for software maintenance tasks such as issue localization and repository-level code generation by providing tacit knowledge as contexts. Based on the proposed methodology, we explore the key challenges and opportunities involved in the continuous construction and refinement of code digital twin.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02380v1">JoyTTS: LLM-based Spoken Chatbot With Voice Cloning</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      JoyTTS is an end-to-end spoken chatbot that combines large language models (LLM) with text-to-speech (TTS) technology, featuring voice cloning capabilities. This project is built upon the open-source MiniCPM-o and CosyVoice2 models and trained on 2000 hours of conversational data. We have also provided the complete training code to facilitate further development and optimization by the community. On the testing machine seed-tts-zh, it achieves a SS (speaker similarity) score of 0.73 and a WER (Word Error Rate) of 5.09. The code and models, along with training and inference scripts, are available at https://github.com/jdh-algo/JoyTTS.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02378v1">Efficient Code LLM Training via Distribution-Consistent and Diversity-Aware Data Selection</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have significantly improved code generation and program comprehension, accelerating the evolution of software engineering. Current methods primarily enhance model performance by leveraging vast amounts of data, focusing on data quantity while often overlooking data quality, thereby reducing training efficiency. To address this, we introduce an approach that utilizes a parametric model for code data selection, aimed at improving both training efficiency and model performance. Our method optimizes the parametric model to ensure distribution consistency and diversity within the selected subset, guaranteeing high-quality data. Experimental results demonstrate that using only 10K samples, our method achieves gains of 2.4% (HumanEval) and 2.3% (MBPP) over 92K full-sampled baseline, outperforming other sampling approaches in both performance and efficiency. This underscores that our method effectively boosts model performance while significantly reducing computational costs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21880v2">Incorporating LLMs for Large-Scale Urban Complex Mobility Simulation</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 8 pages, 8 figures. This paper is reviewed and accepted by the CUPUM (Computational Urban Planning and Urban Management) Conference held by University College London (UCL) in 2025
    </div>
    <details class="paper-abstract">
      This study presents an innovative approach to urban mobility simulation by integrating a Large Language Model (LLM) with Agent-Based Modeling (ABM). Unlike traditional rule-based ABM, the proposed framework leverages LLM to enhance agent diversity and realism by generating synthetic population profiles, allocating routine and occasional locations, and simulating personalized routes. Using real-world data, the simulation models individual behaviors and large-scale mobility patterns in Taipei City. Key insights, such as route heat maps and mode-specific indicators, provide urban planners with actionable information for policy-making. Future work focuses on establishing robust validation frameworks to ensure accuracy and reliability in urban planning applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02353v1">OMS: On-the-fly, Multi-Objective, Self-Reflective Ad Keyword Generation via LLM Agent</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      Keyword decision in Sponsored Search Advertising is critical to the success of ad campaigns. While LLM-based methods offer automated keyword generation, they face three major limitations: reliance on large-scale query-keyword pair data, lack of online multi-objective performance monitoring and optimization, and weak quality control in keyword selection. These issues hinder the agentic use of LLMs in fully automating keyword decisions by monitoring and reasoning over key performance indicators such as impressions, clicks, conversions, and CTA effectiveness. To overcome these challenges, we propose OMS, a keyword generation framework that is On-the-fly (requires no training data, monitors online performance, and adapts accordingly), Multi-objective (employs agentic reasoning to optimize keywords based on multiple performance metrics), and Self-reflective (agentically evaluates keyword quality). Experiments on benchmarks and real-world ad campaigns show that OMS outperforms existing methods; ablation and human evaluations confirm the effectiveness of each component and the quality of generated keywords.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.11556v2">Token Prepending: A Training-Free Approach for Eliciting Better Sentence Embeddings from LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 Accept to ACL 2025 (Oral). Code are available on https://github.com/fuyuchenIfyw/token_prepending.git
    </div>
    <details class="paper-abstract">
      Extracting sentence embeddings from large language models (LLMs) is a promising direction, as LLMs have demonstrated stronger semantic understanding capabilities. Previous studies typically focus on prompt engineering to elicit sentence embeddings from LLMs by prompting the model to encode sentence information into the embedding of the last token. However, LLMs are mostly decoder-only models with causal attention and the earlier tokens in the sentence cannot attend to the latter tokens, resulting in biased encoding of sentence information and cascading effects on the final decoded token. To this end, we propose a novel Token Prepending (TP) technique that prepends each layer's decoded sentence embedding to the beginning of the sentence in the next layer's input, allowing earlier tokens to attend to the complete sentence information under the causal attention mechanism. The proposed TP technique is a plug-and-play and training-free technique, which means it can be seamlessly integrated with various prompt-based sentence embedding methods and autoregressive LLMs. Extensive experiments on various Semantic Textual Similarity (STS) tasks and downstream classification tasks demonstrate that our proposed TP technique can significantly improve the performance of existing prompt-based sentence embedding methods across different LLMs, while incurring negligible additional inference cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02332v1">PII Jailbreaking in LLMs via Activation Steering Reveals Personal Information Leakage</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      This paper investigates privacy jailbreaking in LLMs via steering, focusing on whether manipulating activations can bypass LLM alignment and alter response behaviors to privacy related queries (e.g., a certain public figure's sexual orientation). We begin by identifying attention heads predictive of refusal behavior for private attributes (e.g., sexual orientation) using lightweight linear probes trained with privacy evaluator labels. Next, we steer the activations of a small subset of these attention heads guided by the trained probes to induce the model to generate non-refusal responses. Our experiments show that these steered responses often disclose sensitive attribute details, along with other private information about data subjects such as life events, relationships, and personal histories that the models would typically refuse to produce. Evaluations across four LLMs reveal jailbreaking disclosure rates of at least 95%, with more than 50% on average of these responses revealing true personal information. Our controlled study demonstrates that private information memorized in LLMs can be extracted through targeted manipulation of internal activations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17828v2">Aligning Frozen LLMs by Reinforcement Learning: An Iterative Reweight-then-Optimize Approach</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      Aligning large language models (LLMs) with human preferences usually requires fine-tuning methods such as RLHF and DPO. These methods directly optimize the model parameters, so they cannot be used in test-time to improve model performance, nor are they applicable when the model weights are not accessible. In contrast, test-time methods sidestep weight updates by leveraging reward functions to guide and improve output quality. However, they incur high inference costs, and their one-shot guidance is often based on imperfect reward or value functions, leading to suboptimal outputs. In this work, we present a method named Iterative Reweight-then-Optimize (IRO), a reinforcement learning (RL) framework that performs RL-style alignment of the (frozen) base model without touching its parameters. During training, each iteration (i) samples candidates from the base model, (ii) resamples using current value functions, and (iii) trains a new lightweight value function that guides the next decoding pass. At test time, the value functions are used to guide the base model generation via a search-based optimization process. Notably, users can apply IRO to align a model on their own dataset, similar to OpenAI's reinforcement fine-tuning (RFT), but without requiring access to the model weights.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22618v3">Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      Diffusion-based large language models (Diffusion LLMs) have shown promise for non-autoregressive text generation with parallel decoding capabilities. However, the practical inference speed of open-sourced Diffusion LLMs often lags behind autoregressive models due to the lack of Key-Value (KV) Cache and quality degradation when decoding multiple tokens simultaneously. To bridge this gap, we introduce a novel block-wise approximate KV Cache mechanism tailored for bidirectional diffusion models, enabling cache reuse with negligible performance drop. Additionally, we identify the root cause of generation quality degradation in parallel decoding as the disruption of token dependencies under the conditional independence assumption. To address this, we propose a confidence-aware parallel decoding strategy that selectively decodes tokens exceeding a confidence threshold, mitigating dependency violations and maintaining generation quality. Experimental results on LLaDA and Dream models across multiple LLM benchmarks demonstrate up to \textbf{27.6$\times$ throughput} improvement with minimal accuracy loss, closing the performance gap with autoregressive models and paving the way for practical deployment of Diffusion LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02253v1">Scaling LLM Planning: NL2FLOW for Parametric Problem Generation and Rigorous Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 20 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Progress in enhancing large language model (LLM) planning and reasoning capabilities is significantly hampered by the bottleneck of scalable, reliable data generation and evaluation. To overcome this, I introduce NL2FLOW, a fully automated system for parametrically generating planning problems - expressed in natural language, a structured intermediate representation, and formal PDDL - and rigorously evaluating the quality of generated plans. I demonstrate NL2FLOW's capabilities by generating a dataset of 2296 problems in the automated workflow generation domain and evaluating multiple open-sourced, instruct-tuned LLMs. My results reveal that the highest performing models achieved 86% success in generating valid plans and 69% in generating optimal plans, specifically for problems with feasible solutions. Regression analysis shows that the influence of problem characteristics on plan generation is contingent on both model and prompt design. Notably, I observed that the highest success rate for translating natural language into a JSON representation of a plan was lower than the highest rate of generating a valid plan directly. This suggests that unnecessarily decomposing the reasoning task - introducing intermediate translation steps - may actually degrade performance, implying a benefit to models capable of reasoning directly from natural language to action. As I scale LLM reasoning to increasingly complex problems, the bottlenecks and sources of error within these systems will inevitably shift. Therefore, a dynamic understanding of these limitations - and the tools to systematically reveal them - will be crucial for unlocking the full potential of LLMs as intelligent problem solvers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02226v1">DecoRTL: A Run-time Decoding Framework for RTL Code Generation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 Accepted to the International Conference on Computer-Aided Design (ICCAD 2025)
    </div>
    <details class="paper-abstract">
      As one of their many applications, large language models (LLMs) have recently shown promise in automating register transfer level (RTL) code generation. However, conventional LLM decoding strategies, originally designed for natural language, often fail to meet the structural and semantic demands of RTL, leading to hallucinated, repetitive, or invalid code outputs. In this paper, we first investigate the root causes of these decoding failures through an empirical analysis of token-level entropy during RTL generation. Our findings reveal that LLMs exhibit low confidence in regions of structural ambiguity or semantic complexity, showing that standard decoding strategies fail to differentiate between regions requiring determinism (syntax-critical regions) and those that benefit from creative exploratory variability (design-critical regions). Then, to overcome this, we introduce DecoRTL, a novel run-time decoding strategy, that is both syntax-aware and contrastive for RTL code generation. DecoRTL integrates two complementary components: (i) self-consistency sampling, which generates multiple candidates and re-ranks them based on token-level agreement to promote correctness while maintaining diversity; and (ii) syntax-aware temperature adaptation, which classifies tokens by their syntactical and functional roles and adjusts the sampling temperature accordingly, enforcing low temperature for syntax-critical tokens and higher temperature for exploratory ones. Our approach operates entirely at inference time without requiring any additional model fine-tuning. Through evaluations on multiple open-source LLMs using the VerilogEval benchmark, we demonstrate significant improvements in syntactic validity, functional correctness, and output diversity, while the execution overhead (performance overhead) is imperceptible.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21551v2">Where to find Grokking in LLM Pretraining? Monitor Memorization-to-Generalization without Test</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 10 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Grokking, i.e., test performance keeps improving long after training loss converged, has been recently witnessed in neural network training, making the mechanism of generalization and other emerging capabilities such as reasoning mysterious. While prior studies usually train small models on a few toy or highly-specific tasks for thousands of epochs, we conduct the first study of grokking on checkpoints during one-pass pretraining of a 7B large language model (LLM), i.e., OLMoE. We compute the training loss and evaluate generalization on diverse benchmark tasks, including math reasoning, code generation, and commonsense/domain-specific knowledge retrieval tasks. Our study, for the first time, verifies that grokking still happens in the pretraining of large-scale foundation models, though different data may enter grokking stages asynchronously. We further demystify grokking's "emergence of generalization" by investigating LLM internal dynamics. Specifically, we find that training samples' pathways (i.e., expert choices across layers) evolve from random, instance-specific to more structured and shareable between samples. Also, the complexity of a sample's pathway reduces despite the converged loss. These indicate a memorization-to-generalization "knowledge digestion", providing a mechanistic explanation of delayed generalization. In the study, we develop two novel metrics to quantify pathway distance and the complexity of a single pathway. We show their ability to predict the generalization improvement on diverse downstream tasks. They are efficient, simple to compute and solely dependent on training data. Hence, they have practical value for pretraining, enabling us to monitor the generalization performance without finetuning and test. Theoretically, we show that more structured pathways reduce model complexity and improve the generalization bound.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.22694v2">VOCABTRIM: Vocabulary Pruning for Efficient Speculative Decoding in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 8 pages, 4 figures, 5 tables, accepted at ICML 2025 workshop on Efficient Systems for Foundational Models
    </div>
    <details class="paper-abstract">
      In this paper, we introduce a simple training-free technique to improve the performance of drafter-based speculative decoding (SpD) methods that incorporates language modeling head (LM head) during drafting process. A drafter-based speculative decoding leverages one or more smaller language models, a.k.a. drafters or draft models, to sample a draft sequence or tree consisting of multiple tokens, followed by verification by a base LLM, a target model, accepting a subset as its valid generation. As it is usually considered that the speculative decoding requires one-to-one mapping between vocabularies of the target model and the draft model, it has been natural to share the vocabulary between them, or even share the LM head as in EAGLE or Medusa. We first identify that this draft token sampling scheme inherently contains an unnecessary inference overhead in drafting, especially for some target LLMs with very large vocabularies. Then, we propose a simple technique, VocabTrim, to mitigate the drafting overhead to improve the generation speed in memory-bound environment. VocabTrim reconstructs the drafter LM head to contain only a limited set of tokens, selected by the most frequently sampled from the vocabulary of the target model. While limiting the vocabulary in drafting slightly degrades the acceptance rate, it significantly reduces the drafting latency in memory-bound process which is often the case on edge devices, resulting in higher memory-bound speed up (MBSU). We show that our method can boost the memory-bound speed-up for Llama-3 models on Spec-Bench, specifically by 16% for Llama-3.2-3B-Instruct.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03211v1">DistZO2: High-Throughput and Memory-Efficient Zeroth-Order Fine-tuning LLMs with Distributed Parallel Computing</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) remains resource-intensive due to their sheer scale. While zeroth-order (ZO) optimization provides a memory-efficient alternative by eliminating backward passes, its application to multi-hundred-billion-parameter models is constrained by GPU memory and compute throughput. The ZO2 framework addresses the memory bottleneck by offloading model parameters to CPU memory and overlapping transformer block transfer with dual forward computation on a single GPU. However, ZO2 remains limited by its single-device execution and achieves modest throughput. In this work, we present DistZO2, a high-throughput, memory-efficient framework for distributed zeroth-order fine-tuning of LLMs. DistZO2 introduces three parallel strategies: (1) Perturbation Parallelism (PertP), which parallelizes the two perturbed forward passes across devices; (2) Distributed Data Parallelism (DDP), adapted to the scalar-gradient nature of ZO training; and (3) a unified 2D Parallelism design that combines PertP and DDP. To further mitigate communication bottlenecks introduced by parameter offloading, we propose a hardware-aware communication strategy that slices parameter blocks and redistributes them across GPUs via high-speed interconnects such as NVLink. DistZO2 scales zeroth-order fine-tuning to modern multi-GPU systems, preserving ZO2's memory efficiency while substantially improving training throughput. In our experiments on OPT-175B, DistZO2 achieves a 3x speedup over ZO2 with distributed computing. DistZO2's code has been open-sourced in https://github.com/liangyuwang/zo2.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03194v1">How Much Content Do LLMs Generate That Induces Cognitive Bias in Users?</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 17 pages, 2 figures. to be submitted to AACL 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly integrated into applications ranging from review summarization to medical diagnosis support, where they affect human decisions. Even though LLMs perform well in many tasks, they may also inherit societal or cognitive biases, which can inadvertently transfer to humans. We investigate when and how LLMs expose users to biased content and quantify its severity. Specifically, we assess three LLM families in summarization and news fact-checking tasks, evaluating how much LLMs stay consistent with their context and/or hallucinate. Our findings show that LLMs expose users to content that changes the sentiment of the context in 21.86% of the cases, hallucinates on post-knowledge-cutoff data questions in 57.33% of the cases, and primacy bias in 5.94% of the cases. We evaluate 18 distinct mitigation methods across three LLM families and find that targeted interventions can be effective. Given the prevalent use of LLMs in high-stakes domains, such as healthcare or legal analysis, our results highlight the need for robust technical safeguards and for developing user-centered interventions that address LLM limitations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18382v2">One Demo Is All It Takes: Planning Domain Derivation with LLMs from A Single Demonstration</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 31 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Pre-trained Large Language Models (LLMs) have shown promise in solving planning problems but often struggle to ensure plan correctness, especially for long-horizon tasks. Meanwhile, traditional robotic task and motion planning (TAMP) frameworks address these challenges more reliably by combining high-level symbolic search with low-level motion planning. At the core of TAMP is the planning domain, an abstract world representation defined through symbolic predicates and actions. However, creating these domains typically involves substantial manual effort and domain expertise, limiting generalizability. We introduce Planning Domain Derivation with LLMs (PDDLLM), a novel approach that combines simulated physical interaction with LLM reasoning to improve planning performance. The method reduces reliance on humans by inferring planning domains from a single annotated task-execution demonstration. Unlike prior domain-inference methods that rely on partially predefined or language descriptions of planning domains, PDDLLM constructs domains entirely from scratch and automatically integrates them with low-level motion planning skills, enabling fully automated long-horizon planning. PDDLLM is evaluated on over 1,200 diverse tasks spanning nine environments and benchmarked against six LLM-based planning baselines, demonstrating superior long-horizon planning performance, lower token costs, and successful deployment on multiple physical robot platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03162v1">MateInfoUB: A Real-World Benchmark for Testing LLMs in Competitive, Multilingual, and Multimodal Educational Tasks</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 14 pages (9 paper, 2 references, 3 annexes). Accepted for BEA 2025!
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) has transformed various domains, particularly computer science (CS) education. These models exhibit remarkable capabilities in code-related tasks and problem-solving, raising questions about their potential and limitations in advanced CS contexts. This study presents a novel bilingual (English-Romanian) multimodal (text and image) dataset of multiple-choice questions derived from a high-level computer science competition. A particularity of our dataset is that the problems are conceived such that some of them are easier solved using reasoning on paper, while for others writing code is more efficient. We systematically evaluate State of The Art LLMs on this dataset, analyzing their performance on theoretical programming tasks. Our findings reveal the strengths and limitations of current LLMs, including the influence of language choice (English vs. Romanian), providing insights into their applicability in CS education and competition settings. We also address critical ethical considerations surrounding educational integrity and the fairness of assessments in the context of LLM usage. These discussions aim to inform future educational practices and policies. To support further research, our dataset will be made publicly available in both English and Romanian. Additionally, we release an educational application tailored for Romanian students, enabling them to self-assess using the dataset in an interactive and practice-oriented environment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03156v1">The Impact of LLM-Assistants on Software Developer Productivity: A Systematic Literature Review</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 37 pages
    </div>
    <details class="paper-abstract">
      Large language model assistants (LLM-assistants) present new opportunities to transform software development. Developers are increasingly adopting these tools across tasks, including coding, testing, debugging, documentation, and design. Yet, despite growing interest, there is no synthesis of how LLM-assistants affect software developer productivity. In this paper, we present a systematic literature review of 37 peer-reviewed studies published between January 2014 and December 2024 that examine this impact. Our analysis reveals that LLM-assistants offer both considerable benefits and critical risks. Commonly reported gains include minimized code search, accelerated development, and the automation of trivial and repetitive tasks. However, studies also highlight concerns around cognitive offloading, reduced team collaboration, and inconsistent effects on code quality. While the majority of studies (92%) adopt a multi-dimensional perspective by examining at least two SPACE dimensions, reflecting increased awareness of the complexity of developer productivity, only 14% extend beyond three dimensions, indicating substantial room for more integrated evaluations. Satisfaction, Performance, and Efficiency are the most frequently investigated dimensions, whereas Communication and Activity remain underexplored. Most studies are exploratory (64%) and methodologically diverse, but lack longitudinal and team-based evaluations. This review surfaces key research gaps and provides recommendations for future research and practice. All artifacts associated with this study are publicly available at https://zenodo.org/records/15788502.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03153v1">HGCA: Hybrid GPU-CPU Attention for Long Context LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      Scaling inference for large language models (LLMs) is increasingly constrained by limited GPU memory, especially due to growing key-value (KV) caches required for long-context generation. While existing approaches offload KV caches to CPU memory or apply sparse attention to reduce GPU load, they often underutilize CPU compute resources and compromise accuracy. We present HGCA, a hybrid CPU-GPU attention mechanism that enables scalable, high-throughput LLM inference with near-full attention quality. HGCA performs dense attention on recently generated KV entries retained in GPU memory and parallel sparse attention on selected, salient KV entries in CPU memory. The attention outputs are efficiently merged using log-sum-exp fusion, minimizing PCIe transfer overhead. HGCA also introduces a finegrained, per-head sparsification strategy optimized for CPU execution, preserving contextual relevance while reducing computation. Our implementation seamlessly integrates into existing LLM frameworks without requiring model retraining. Experiments across diverse models and workloads show that HGCA achieves superior scalability, supports longer sequences and larger batch sizes, and outperforms existing sparse attention baselines in both performance and accuracy -- all on commodity GPU hardware.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.05232v3">LIAR: Leveraging Inference Time Alignment (Best-of-N) to Jailbreak LLMs in Seconds</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      Jailbreak attacks expose vulnerabilities in safety-aligned LLMs by eliciting harmful outputs through carefully crafted prompts. Existing methods rely on discrete optimization or trained adversarial generators, but are slow, compute-intensive, and often impractical. We argue that these inefficiencies stem from a mischaracterization of the problem. Instead, we frame jailbreaks as inference-time misalignment and introduce LIAR (Leveraging Inference-time misAlignment to jailbReak), a fast, black-box, best-of-$N$ sampling attack requiring no training. LIAR matches state-of-the-art success rates while reducing perplexity by $10\times$ and Time-to-Attack from hours to seconds. We also introduce a theoretical "safety net against jailbreaks" metric to quantify safety alignment strength and derive suboptimality bounds. Our work offers a simple yet effective tool for evaluating LLM robustness and advancing alignment research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03064v1">LLM-Driven Auto Configuration for Transient IoT Device Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      Today's Internet of Things (IoT) has evolved from simple sensing and actuation devices to those with embedded processing and intelligent services, enabling rich collaborations between users and their devices. However, enabling such collaboration becomes challenging when transient devices need to interact with host devices in temporarily visited environments. In such cases, fine-grained access control policies are necessary to ensure secure interactions; however, manually implementing them is often impractical for non-expert users. Moreover, at run-time, the system must automatically configure the devices and enforce such fine-grained access control rules. Additionally, the system must address the heterogeneity of devices. In this paper, we present CollabIoT, a system that enables secure and seamless device collaboration in transient IoT environments. CollabIoT employs a Large language Model (LLM)-driven approach to convert users' high-level intents to fine-grained access control policies. To support secure and seamless device collaboration, CollabIoT adopts capability-based access control for authorization and uses lightweight proxies for policy enforcement, providing hardware-independent abstractions. We implement a prototype of CollabIoT's policy generation and auto configuration pipelines and evaluate its efficacy on an IoT testbed and in large-scale emulated environments. We show that our LLM-based policy generation pipeline is able to generate functional and correct policies with 100% accuracy. At runtime, our evaluation shows that our system configures new devices in ~150 ms, and our proxy-based data plane incurs network overheads of up to 2 ms and access control overheads up to 0.3 ms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03052v1">From 2:4 to 8:16 sparsity patterns in LLMs for Outliers and Weights with Variance Correction</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) grow in size, efficient compression techniques like quantization and sparsification are critical. While quantization maintains performance with reduced precision, structured sparsity methods, such as N:M sparsification, often fall short due to limited flexibility, and sensitivity to outlier weights. We explore 8:16 semi-structured sparsity, demonstrating its ability to surpass the Performance Threshold-where a compressed model matches the accuracy of its uncompressed or smaller counterpart under equivalent memory constraints. Compared to 2:4 sparsity, 8:16 offers greater flexibility with minimal storage overhead (0.875 vs. 0.75 bits/element). We also apply sparse structured patterns for salient weights, showing that structured sparsity for outliers is competitive with unstructured approaches leading to equivalent or better results. Finally, we demonstrate that simple techniques such as variance correction and SmoothQuant like weight equalization improve sparse models performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03051v1">Improving LLM Reasoning for Vulnerability Detection via Group Relative Policy Optimization</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Improving and understanding the training dynamics and reasoning of Large Language Models (LLMs) has become essential for their deployment in AI-based security tools, such as software vulnerability detection. In this work, we present an extensive study aimed at advancing recent RL-based finetuning techniques for LLMs in the context of vulnerability detection. We start by highlighting key limitations of commonly adopted LLMs, such as their tendency to over-predict certain types of vulnerabilities while failing to detect others. To address this challenge, we explore the use of Group Relative Policy Optimization (GRPO), a recent policy-gradient method, for guiding LLM behavior through structured, rule-based rewards. We enable its application to the vulnerability detection task by redefining its advantage functions and reward signals using annotations from widely used datasets in the field, including BigVul, DiverseVul, and CleanVul. The proposed methodology enables an extensive set of experiments, addressing multiple research questions regarding the impact of GRPO on generalization, reasoning capabilities, and performance improvements over standard supervised finetuning (SFT). Our findings offer valuable insights into the potential of RL-based training to enhance both the performance and reasoning abilities of LLMs in the context of software vulnerability detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03042v1">Dynamic Long Short-Term Memory Based Memory Storage For Long Horizon LLM Interaction</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 7 pages, 4 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Memory storage for Large Language models (LLMs) is becoming an increasingly active area of research, particularly for enabling personalization across long conversations. We propose Pref-LSTM, a dynamic and lightweight framework that combines a BERT-based classifier with a LSTM memory module that generates memory embedding which then is soft-prompt injected into a frozen LLM. We synthetically curate a dataset of preference and non-preference conversation turns to train our BERT-based classifier. Although our LSTM-based memory encoder did not yield strong results, we find that the BERT-based classifier performs reliably in identifying explicit and implicit user preferences. Our research demonstrates the viability of using preference filtering with LSTM gating principals as an efficient path towards scalable user preference modeling, without extensive overhead and fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02851v1">MOTIF: Modular Thinking via Reinforcement Fine-tuning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      Recent advancements in the reasoning capabilities of large language models (LLMs) show that employing group relative policy optimization (GRPO) algorithm for reinforcement learning (RL) training allows the models to use more thinking/reasoning tokens for generating better responses. However, LLMs can generate only a finite amount of tokens while maintaining attention to the previously generated tokens. This limit, also known as the context size of an LLM, is a bottleneck in LLM reasoning with arbitrarily large number of tokens. To think beyond the limit of context size, an LLM must employ a modular thinking strategy to reason over multiple rounds. In this work, we propose $\textbf{MOTIF: Modular Thinking via Reinforcement Finetuning}$ -- an RL training method for generating thinking tokens in multiple rounds, effectively allowing the model to think with additional context size. We trained the open-source model Qwen2.5-3B-Instruct on GSM8K dataset via parameter efficient fine-tuning and tested its accuracy on MATH500 and AIME2024 benchmarks. Our experiments show 3.8\% and 3.3\% improvements over vanilla GRPO based training in the respective benchmarks. Furthermore, this improvement was achieved with only 15\% of samples, thus demonstrating sample efficiency of MOTIF. Our code and models are available at https://github.com/purbeshmitra/MOTIF and https://huggingface.co/purbeshmitra/MOTIF, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02778v1">Self-Correction Bench: Revealing and Addressing the Self-Correction Blind Spot in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 31 pages, 18 figures
    </div>
    <details class="paper-abstract">
      Although large language models (LLMs) have become transformative, they still make mistakes and can explore unproductive reasoning paths. Self-correction is an important capability for a trustworthy LLM, particularly an autoregressive LLM. While LLMs can identify error in user input, they exhibit a systematic 'Self-Correction Blind Spot' - failing to correct identical error in their own outputs. To systematically study this phenomenon, we introduce Self-Correction Bench, a systematic framework to measure this phenomenon through controlled error injection at three complexity levels. Testing 14 models, we find an average 64.5% blind spot rate. We find multiple evidences that this limitation relates to training data composition: human training demonstrations predominantly show error-free responses rather than error-correction sequences, unlike RL-trained models that learn error correction through outcome feedback. Remarkably, simply appending "Wait" reduces blind spots by 89.3%, suggesting that the capability exists but requires activation. Our work highlights a critical limitation in current LLMs and offers potential avenues for improving their reliability and trustworthiness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02773v1">KERAP: A Knowledge-Enhanced Reasoning Approach for Accurate Zero-shot Diagnosis Prediction Using Multi-agent LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      Medical diagnosis prediction plays a critical role in disease detection and personalized healthcare. While machine learning (ML) models have been widely adopted for this task, their reliance on supervised training limits their ability to generalize to unseen cases, particularly given the high cost of acquiring large, labeled datasets. Large language models (LLMs) have shown promise in leveraging language abilities and biomedical knowledge for diagnosis prediction. However, they often suffer from hallucinations, lack structured medical reasoning, and produce useless outputs. To address these challenges, we propose KERAP, a knowledge graph (KG)-enhanced reasoning approach that improves LLM-based diagnosis prediction through a multi-agent architecture. Our framework consists of a linkage agent for attribute mapping, a retrieval agent for structured knowledge extraction, and a prediction agent that iteratively refines diagnosis predictions. Experimental results demonstrate that KERAP enhances diagnostic reliability efficiently, offering a scalable and interpretable solution for zero-shot medical diagnosis prediction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02745v1">Who's Sorry Now: User Preferences Among Rote, Empathic, and Explanatory Apologies from LLM Chatbots</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      As chatbots driven by large language models (LLMs) are increasingly deployed in everyday contexts, their ability to recover from errors through effective apologies is critical to maintaining user trust and satisfaction. In a preregistered study with Prolific workers (N=162), we examine user preferences for three types of apologies (rote, explanatory, and empathic) issued in response to three categories of common LLM mistakes (bias, unfounded fabrication, and factual errors). We designed a pairwise experiment in which participants evaluated chatbot responses consisting of an initial error, a subsequent apology, and a resolution. Explanatory apologies were generally preferred, but this varied by context and user. In the bias scenario, empathic apologies were favored for acknowledging emotional impact, while hallucinations, though seen as serious, elicited no clear preference, reflecting user uncertainty. Our findings show the complexity of effective apology in AI systems. We discuss key insights such as personalization and calibration that future systems must navigate to meaningfully repair trust.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02737v1">Early Signs of Steganographic Capabilities in Frontier LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      Monitoring Large Language Model (LLM) outputs is crucial for mitigating risks from misuse and misalignment. However, LLMs could evade monitoring through steganography: Encoding hidden information within seemingly benign generations. In this paper, we evaluate the steganography capabilities in frontier LLMs to better understand the risk they pose. We focus on two types of steganography: passing encoded messages and performing encoded reasoning. We find that current models are unable to encode short messages in their outputs without a monitor noticing under standard affordances. They can succeed, however, if given additional affordances such as using an unmonitored scratchpad and coordinating on what encoding scheme to use. We additionally find early signs that models can perform basic encoded reasoning in a simple state-tracking problem. This includes some ability to reason with their own and pre-defined schemes, including encoding schemes such as Hexadecimal. Despite this, they can rarely hide reasoning subtly within a cover task to fool a monitor. Overall, our results indicate that current LLMs exhibit nascent steganographic capabilities. While these capabilities are likely insufficient to bypass well-designed monitors at present, this could change in the future.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00032v4">KatFishNet: Detecting LLM-Generated Korean Text through Linguistic Feature Analysis</a></div>
    <div class="paper-meta">
      📅 2025-07-02
      | 💬 Accepted to ACL 2025 main conference
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) increases the difficulty of distinguishing between human-written and LLM-generated text. Detecting LLM-generated text is crucial for upholding academic integrity, preventing plagiarism, protecting copyrights, and ensuring ethical research practices. Most prior studies on detecting LLM-generated text focus primarily on English text. However, languages with distinct morphological and syntactic characteristics require specialized detection approaches. Their unique structures and usage patterns can hinder the direct application of methods primarily designed for English. Among such languages, we focus on Korean, which has relatively flexible spacing rules, a rich morphological system, and less frequent comma usage compared to English. We introduce KatFish, the first benchmark dataset for detecting LLM-generated Korean text. The dataset consists of text written by humans and generated by four LLMs across three genres. By examining spacing patterns, part-of-speech diversity, and comma usage, we illuminate the linguistic differences between human-written and LLM-generated Korean text. Building on these observations, we propose KatFishNet, a detection method specifically designed for the Korean language. KatFishNet achieves an average of 19.78% higher AUROC compared to the best-performing existing detection method. Our code and data are available at https://github.com/Shinwoo-Park/detecting_llm_generated_korean_text_through_linguistic_analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01334v1">Symbolic or Numerical? Understanding Physics Problem Solving in Reasoning LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-02
    </div>
    <details class="paper-abstract">
      Navigating the complexities of physics reasoning has long been a difficult task for Large Language Models (LLMs), requiring a synthesis of profound conceptual understanding and adept problem-solving techniques. In this study, we investigate the application of advanced instruction-tuned reasoning models, such as Deepseek-R1, to address a diverse spectrum of physics problems curated from the challenging SciBench benchmark. Our comprehensive experimental evaluation reveals the remarkable capabilities of reasoning models. Not only do they achieve state-of-the-art accuracy in answering intricate physics questions, but they also generate distinctive reasoning patterns that emphasize on symbolic derivation. Furthermore, our findings indicate that even for these highly sophisticated reasoning models, the strategic incorporation of few-shot prompting can still yield measurable improvements in overall accuracy, highlighting the potential for continued performance gains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01315v1">Context-Aware Code Wiring Recommendation with LLM-based Agent</a></div>
    <div class="paper-meta">
      📅 2025-07-02
    </div>
    <details class="paper-abstract">
      Copy-paste-modify is a widespread and pragmatic practice in software development, where developers adapt reused code snippets, sourced from platforms such as Stack Overflow, GitHub, or LLM outputs, into their local codebase. A critical yet underexplored aspect of this adaptation is code wiring, which involves substituting unresolved variables in the pasted code with suitable ones from the surrounding context. Existing solutions either rely on heuristic rules or historical templates, often failing to effectively utilize contextual information, despite studies showing that over half of adaptation cases are context-dependent. In this paper, we introduce WIRL, an LLM-based agent for code wiring framed as a Retrieval-Augmented Generation (RAG) infilling task. WIRL combines an LLM, a customized toolkit, and an orchestration module to identify unresolved variables, retrieve context, and perform context-aware substitutions. To balance efficiency and autonomy, the agent adopts a mixed strategy: deterministic rule-based steps for common patterns, and a state-machine-guided decision process for intelligent exploration. We evaluate WIRL on a carefully curated, high-quality dataset consisting of real-world code adaptation scenarios. Our approach achieves an exact match precision of 91.7% and a recall of 90.0%, outperforming advanced LLMs by 22.6 and 13.7 percentage points in precision and recall, respectively, and surpassing IntelliJ IDEA by 54.3 and 49.9 percentage points. These results underscore its practical utility, particularly in contexts with complex variable dependencies or multiple unresolved variables. We believe WIRL paves the way for more intelligent and context-aware developer assistance in modern IDEs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01299v1">La RoSA: Enhancing LLM Efficiency via Layerwise Rotated Sparse Activation</a></div>
    <div class="paper-meta">
      📅 2025-07-02
      | 💬 ICML 2025 Acceptance
    </div>
    <details class="paper-abstract">
      Activation sparsity can reduce the computational overhead and memory transfers during the forward pass of Large Language Model (LLM) inference. Existing methods face limitations, either demanding time-consuming recovery training that hinders real-world adoption, or relying on empirical magnitude-based pruning, which causes fluctuating sparsity and unstable inference speed-up. This paper introduces LaRoSA (Layerwise Rotated Sparse Activation), a novel method for activation sparsification designed to improve LLM efficiency without requiring additional training or magnitude-based pruning. We leverage layerwise orthogonal rotations to transform input activations into rotated forms that are more suitable for sparsification. By employing a Top-K selection approach within the rotated activations, we achieve consistent model-level sparsity and reliable wall-clock time speed-up. LaRoSA is effective across various sizes and types of LLMs, demonstrating minimal performance degradation and robust inference acceleration. Specifically, for LLaMA2-7B at 40% sparsity, LaRoSA achieves a mere 0.17 perplexity gap with a consistent 1.30x wall-clock time speed-up, and reduces the accuracy gap in zero-shot tasks compared to the dense model to just 0.54%, while surpassing TEAL by 1.77% and CATS by 17.14%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12123v3">Large Language Models, and LLM-Based Agents, Should Be Used to Enhance the Digital Public Sphere</a></div>
    <div class="paper-meta">
      📅 2025-07-02
    </div>
    <details class="paper-abstract">
      This paper argues that large language model-based recommenders can displace today's attention-allocation machinery. LLM-based recommenders would ingest open-web content, infer a user's natural-language goals, and present information that matches their reflective preferences. Properly designed, they could deliver personalization without industrial-scale data hoarding, return control to individuals, optimize for genuine ends rather than click-through proxies, and support autonomous attention management. Synthesizing evidence of current systems' harms with recent work on LLM-driven pipelines, we identify four key research hurdles: generating candidates without centralized data, maintaining computational efficiency, modeling preferences robustly, and defending against prompt-injection. None looks prohibitive; surmounting them would steer the digital public sphere toward democratic, human-centered values.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01264v1">LLM-based Realistic Safety-Critical Driving Video Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-02
    </div>
    <details class="paper-abstract">
      Designing diverse and safety-critical driving scenarios is essential for evaluating autonomous driving systems. In this paper, we propose a novel framework that leverages Large Language Models (LLMs) for few-shot code generation to automatically synthesize driving scenarios within the CARLA simulator, which has flexibility in scenario scripting, efficient code-based control of traffic participants, and enforcement of realistic physical dynamics. Given a few example prompts and code samples, the LLM generates safety-critical scenario scripts that specify the behavior and placement of traffic participants, with a particular focus on collision events. To bridge the gap between simulation and real-world appearance, we integrate a video generation pipeline using Cosmos-Transfer1 with ControlNet, which converts rendered scenes into realistic driving videos. Our approach enables controllable scenario generation and facilitates the creation of rare but critical edge cases, such as pedestrian crossings under occlusion or sudden vehicle cut-ins. Experimental results demonstrate the effectiveness of our method in generating a wide range of realistic, diverse, and safety-critical scenarios, offering a promising tool for simulation-based testing of autonomous vehicles.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.17495v2">Human Mobility Modeling with Household Coordination Activities under Limited Information via Retrieval-Augmented LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-02
    </div>
    <details class="paper-abstract">
      Understanding human mobility patterns has long been a challenging task in transportation modeling. Due to the difficulties in obtaining high-quality training datasets across diverse locations, conventional activity-based models and learning-based human mobility modeling algorithms are particularly limited by the availability and quality of datasets. Current approaches primarily focus on spatial-temporal patterns while neglecting semantic relationships such as logical connections or dependencies between activities and household coordination activities like joint shopping trips or family meal times, both crucial for realistic mobility modeling. We propose a retrieval-augmented large language model (LLM) framework that generates activity chains with household coordination using only public accessible statistical and socio-demographic information, reducing the need for sophisticated mobility data. The retrieval-augmentation mechanism enables household coordination and maintains statistical consistency across generated patterns, addressing a key gap in existing methods. Our validation with NHTS and SCAG-ABM datasets demonstrates effective mobility synthesis and strong adaptability for regions with limited mobility data availability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08320v2">How Good LLM-Generated Password Policies Are?</a></div>
    <div class="paper-meta">
      📅 2025-07-02
      | 💬 11 pages, 2 Tables, 9 figures, 3 Algorithms
    </div>
    <details class="paper-abstract">
      Generative AI technologies, particularly Large Language Models (LLMs), are rapidly being adopted across industry, academia, and government sectors, owing to their remarkable capabilities in natural language processing. However, despite their strengths, the inconsistency and unpredictability of LLM outputs present substantial challenges, especially in security-critical domains such as access control. One critical issue that emerges prominently is the consistency of LLM-generated responses, which is paramount for ensuring secure and reliable operations. In this paper, we study the application of LLMs within the context of Cybersecurity Access Control Systems. Specifically, we investigate the consistency and accuracy of LLM-generated password policies, translating natural language prompts into executable pwquality.conf configuration files. Our experimental methodology adopts two distinct approaches: firstly, we utilize pre-trained LLMs to generate configuration files purely from natural language prompts without additional guidance. Secondly, we provide these models with official pwquality.conf documentation to serve as an informative baseline. We systematically assess the soundness, accuracy, and consistency of these AI-generated configurations. Our findings underscore significant challenges in the current generation of LLMs and contribute valuable insights into refining the deployment of LLMs in Access Control Systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02197v1">Do Role-Playing Agents Practice What They Preach? Belief-Behavior Consistency in LLM-Based Simulations of Human Trust</a></div>
    <div class="paper-meta">
      📅 2025-07-02
    </div>
    <details class="paper-abstract">
      As LLMs are increasingly studied as role-playing agents to generate synthetic data for human behavioral research, ensuring that their outputs remain coherent with their assigned roles has become a critical concern. In this paper, we investigate how consistently LLM-based role-playing agents' stated beliefs about the behavior of the people they are asked to role-play ("what they say") correspond to their actual behavior during role-play ("how they act"). Specifically, we establish an evaluation framework to rigorously measure how well beliefs obtained by prompting the model can predict simulation outcomes in advance. Using an augmented version of the GenAgents persona bank and the Trust Game (a standard economic game used to quantify players' trust and reciprocity), we introduce a belief-behavior consistency metric to systematically investigate how it is affected by factors such as: (1) the types of beliefs we elicit from LLMs, like expected outcomes of simulations versus task-relevant attributes of individual characters LLMs are asked to simulate; (2) when and how we present LLMs with relevant information about Trust Game; and (3) how far into the future we ask the model to forecast its actions. We also explore how feasible it is to impose a researcher's own theoretical priors in the event that the originally elicited beliefs are misaligned with research objectives. Our results reveal systematic inconsistencies between LLMs' stated (or imposed) beliefs and the outcomes of their role-playing simulation, at both an individual- and population-level. Specifically, we find that, even when models appear to encode plausible beliefs, they may fail to apply them in a consistent way. These findings highlight the need to identify how and when LLMs' stated beliefs align with their simulated behavior, allowing researchers to use LLM-based agents appropriately in behavioral studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02186v1">EvalAssist: A Human-Centered Tool for LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-07-02
    </div>
    <details class="paper-abstract">
      With the broad availability of large language models and their ability to generate vast outputs using varied prompts and configurations, determining the best output for a given task requires an intensive evaluation process, one where machine learning practitioners must decide how to assess the outputs and then carefully carry out the evaluation. This process is both time-consuming and costly. As practitioners work with an increasing number of models, they must now evaluate outputs to determine which model and prompt performs best for a given task. LLMs are increasingly used as evaluators to filter training data, evaluate model performance, assess harms and risks, or assist human evaluators with detailed assessments. We present EvalAssist, a framework that simplifies the LLM-as-a-judge workflow. The system provides an online criteria development environment, where users can interactively build, test, and share custom evaluation criteria in a structured and portable format. We support a set of LLM-based evaluation pipelines that leverage off-the-shelf LLMs and use a prompt-chaining approach we developed and contributed to the UNITXT open-source library. Additionally, our system also includes specially trained evaluators to detect harms and risks in LLM outputs. We have deployed the system internally in our organization with several hundreds of users.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02173v1">Data Diversification Methods In Alignment Enhance Math Performance In LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-02
    </div>
    <details class="paper-abstract">
      While recent advances in preference learning have enhanced alignment in human feedback, mathematical reasoning remains a persistent challenge. We investigate how data diversification strategies in preference optimization can improve the mathematical reasoning abilities of large language models (LLMs). We evaluate three common data generation methods: temperature sampling, Chain-of-Thought prompting, and Monte Carlo Tree Search (MCTS), and introduce Diversified-ThinkSolve (DTS), a novel structured approach that systematically decomposes problems into diverse reasoning paths. Our results show that with strategically diversified preference data, models can substantially improve mathematical reasoning performance, with the best approach yielding gains of 7.1% on GSM8K and 4.2% on MATH over the base model. Despite its strong performance, DTS incurs only a marginal computational overhead (1.03x) compared to the baseline, while MCTS is nearly five times more costly with lower returns. These findings demonstrate that structured exploration of diverse problem-solving methods creates more effective preference data for mathematical alignment than traditional approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15830v3">Rethinking LLM Training through Information Geometry and Quantum Metrics</a></div>
    <div class="paper-meta">
      📅 2025-07-02
      | 💬 9 pages, 1 figure(s)
    </div>
    <details class="paper-abstract">
      Optimization in large language models (LLMs) unfolds over high-dimensional parameter spaces with non-Euclidean structure. Information geometry frames this landscape using the Fisher information metric, enabling more principled learning via natural gradient descent. Though often impractical, this geometric lens clarifies phenomena such as sharp minima, generalization, and observed scaling laws. We argue that curvature-aware approaches deepen our understanding of LLM training. Finally, we speculate on quantum analogies based on the Fubini-Study metric and Quantum Fisher Information, hinting at efficient optimization in quantum-enhanced systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02145v1">Reasoning or Not? A Comprehensive Evaluation of Reasoning LLMs for Dialogue Summarization</a></div>
    <div class="paper-meta">
      📅 2025-07-02
    </div>
    <details class="paper-abstract">
      Dialogue summarization is a challenging task with significant practical value in customer service, meeting analysis, and conversational AI. Although large language models (LLMs) have achieved substantial progress in summarization tasks, the performance of step-by-step reasoning architectures-specifically Long Chain-of-Thought (CoT) implementations such as OpenAI-o1 and DeepSeek-R1-remains unexplored for dialogue scenarios requiring concurrent abstraction and conciseness. In this work, we present the first comprehensive and systematic evaluation of state-of-the-art reasoning LLMs and non-reasoning LLMs across three major paradigms-generic, role-oriented, and query-oriented dialogue summarization. Our study spans diverse languages, domains, and summary lengths, leveraging strong benchmarks (SAMSum, DialogSum, CSDS, and QMSum) and advanced evaluation protocols that include both LLM-based automatic metrics and human-inspired criteria. Contrary to trends in other reasoning-intensive tasks, our findings show that explicit stepwise reasoning does not consistently improve dialogue summarization quality. Instead, reasoning LLMs are often prone to verbosity, factual inconsistencies, and less concise summaries compared to their non-reasoning counterparts. Through scenario-specific analyses and detailed case studies, we further identify when and why explicit reasoning may fail to benefit-or even hinder-summarization in complex dialogue contexts. Our work provides new insights into the limitations of current reasoning LLMs and highlights the need for targeted modeling and evaluation strategies for real-world dialogue summarization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02139v1">When LLMs Disagree: Diagnosing Relevance Filtering Bias and Retrieval Divergence in SDG Search</a></div>
    <div class="paper-meta">
      📅 2025-07-02
      | 💬 Presented at LLM4Eval Workshop, SIGIR 2025 Padova, Italy, July 17, 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to assign document relevance labels in information retrieval pipelines, especially in domains lacking human-labeled data. However, different models often disagree on borderline cases, raising concerns about how such disagreement affects downstream retrieval. This study examines labeling disagreement between two open-weight LLMs, LLaMA and Qwen, on a corpus of scholarly abstracts related to Sustainable Development Goals (SDGs) 1, 3, and 7. We isolate disagreement subsets and examine their lexical properties, rank-order behavior, and classification predictability. Our results show that model disagreement is systematic, not random: disagreement cases exhibit consistent lexical patterns, produce divergent top-ranked outputs under shared scoring functions, and are distinguishable with AUCs above 0.74 using simple classifiers. These findings suggest that LLM-based filtering introduces structured variability in document retrieval, even under controlled prompting and shared ranking logic. We propose using classification disagreement as an object of analysis in retrieval evaluation, particularly in policy-relevant or thematic search tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02135v1">Dissecting the Impact of Mobile DVFS Governors on LLM Inference Performance and Energy Efficiency</a></div>
    <div class="paper-meta">
      📅 2025-07-02
      | 💬 equal contribution between Zhang and Dash
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly being integrated into various applications and services running on billions of mobile devices. However, deploying LLMs on resource-limited mobile devices faces a significant challenge due to their high demand for computation, memory, and ultimately energy. While current LLM frameworks for mobile use three power-hungry components-CPU, GPU, and Memory-even when running primarily-GPU LLM models, optimized DVFS governors for CPU, GPU, and memory featured in modern mobile devices operate independently and are oblivious of each other. Motivated by the above observation, in this work, we first measure the energy-efficiency of a SOTA LLM framework consisting of various LLM models on mobile phones which showed the triplet mobile governors result in up to 40.4% longer prefilling and decoding latency compared to optimal combinations of CPU, GPU, and memory frequencies with the same energy consumption for sampled prefill and decode lengths. Second, we conduct an in-depth measurement study to uncover how the intricate interplay (or lack of) among the mobile governors cause the above inefficiency in LLM inference. Finally, based on these insights, we design FUSE - a unified energy-aware governor for optimizing the energy efficiency of LLM inference on mobile devices. Our evaluation using a ShareGPT dataset shows FUSE reduces the time-to-first-token and time-per-output-token latencies by 7.0%-16.9% and 25.4%-36.8% on average with the same energy-per-token for various mobile LLM models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02128v1">CROP: Circuit Retrieval and Optimization with Parameter Guidance using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-02
      | 💬 Accepted by ICCAD 2025
    </div>
    <details class="paper-abstract">
      Modern very large-scale integration (VLSI) design requires the implementation of integrated circuits using electronic design automation (EDA) tools. Due to the complexity of EDA algorithms, the vast parameter space poses a huge challenge to chip design optimization, as the combination of even moderate numbers of parameters creates an enormous solution space to explore. Manual parameter selection remains industrial practice despite being excessively laborious and limited by expert experience. To address this issue, we present CROP, the first large language model (LLM)-powered automatic VLSI design flow tuning framework. Our approach includes: (1) a scalable methodology for transforming RTL source code into dense vector representations, (2) an embedding-based retrieval system for matching designs with semantically similar circuits, and (3) a retrieval-augmented generation (RAG)-enhanced LLM-guided parameter search system that constrains the search process with prior knowledge from similar designs. Experiment results demonstrate CROP's ability to achieve superior quality-of-results (QoR) with fewer iterations than existing approaches on industrial designs, including a 9.9% reduction in power consumption.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14557v2">Enhancing LLM-based Quantum Code Generation with Multi-Agent Optimization and Quantum Error Correction</a></div>
    <div class="paper-meta">
      📅 2025-07-02
      | 💬 Paper accepted by DAC'25
    </div>
    <details class="paper-abstract">
      Multi-agent frameworks with Large Language Models (LLMs) have become promising tools for generating general-purpose programming languages using test-driven development, allowing developers to create more accurate and robust code. However, their potential has not been fully unleashed for domain-specific programming languages, where specific domain exhibits unique optimization opportunities for customized improvement. In this paper, we take the first step in exploring multi-agent code generation for quantum programs. By identifying the unique optimizations in quantum designs such as quantum error correction, we introduce a novel multi-agent framework tailored to generating accurate, fault-tolerant quantum code. Each agent in the framework focuses on distinct optimizations, iteratively refining the code using a semantic analyzer with multi-pass inference, alongside an error correction code decoder. We also examine the effectiveness of inference-time techniques, like Chain-of-Thought (CoT) and Retrieval-Augmented Generation (RAG) in the context of quantum programming, uncovering observations that are different from general-purpose code generation. To evaluate our approach, we develop a test suite to measure the impact each optimization has on the accuracy of the generated code. Our findings indicate that techniques such as structured CoT significantly improve the generation of quantum algorithms by up to 50%. In contrast, we have also found that certain techniques such as RAG show limited improvement, yielding an accuracy increase of only 4%. Moreover, we showcase examples of AI-assisted quantum error prediction and correction, demonstrating the effectiveness of our multi-agent framework in reducing the quantum errors of generated quantum programs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02087v1">Evaluating the Promise and Pitfalls of LLMs in Hiring Decisions</a></div>
    <div class="paper-meta">
      📅 2025-07-02
      | 💬 10 pages, 2 figures, 2 tables. Submitted to NeurIPS 2025
    </div>
    <details class="paper-abstract">
      The use of large language models (LLMs) in hiring promises to streamline candidate screening, but it also raises serious concerns regarding accuracy and algorithmic bias where sufficient safeguards are not in place. In this work, we benchmark several state-of-the-art foundational LLMs - including models from OpenAI, Anthropic, Google, Meta, and Deepseek, and compare them with our proprietary domain-specific hiring model (Match Score) for job candidate matching. We evaluate each model's predictive accuracy (ROC AUC, Precision-Recall AUC, F1-score) and fairness (impact ratio of cut-off analysis across declared gender, race, and intersectional subgroups). Our experiments on a dataset of roughly 10,000 real-world recent candidate-job pairs show that Match Score outperforms the general-purpose LLMs on accuracy (ROC AUC 0.85 vs 0.77) and achieves significantly more equitable outcomes across demographic groups. Notably, Match Score attains a minimum race-wise impact ratio of 0.957 (near-parity), versus 0.809 or lower for the best LLMs, (0.906 vs 0.773 for the intersectionals, respectively). We discuss why pretraining biases may cause LLMs with insufficient safeguards to propagate societal biases in hiring scenarios, whereas a bespoke supervised model can more effectively mitigate these biases. Our findings highlight the importance of domain-specific modeling and bias auditing when deploying AI in high-stakes domains such as hiring, and caution against relying on off-the-shelf LLMs for such tasks without extensive fairness safeguards. Furthermore, we show with empirical evidence that there shouldn't be a dichotomy between choosing accuracy and fairness in hiring: a well-designed algorithm can achieve both accuracy in hiring and fairness in outcomes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02076v1">Reasoning on a Budget: A Survey of Adaptive and Controllable Test-Time Compute in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have rapidly progressed into general-purpose agents capable of solving a broad spectrum of tasks. However, current models remain inefficient at reasoning: they apply fixed inference-time compute regardless of task complexity, often overthinking simple problems while underthinking hard ones. This survey presents a comprehensive review of efficient test-time compute (TTC) strategies, which aim to improve the computational efficiency of LLM reasoning. We introduce a two-tiered taxonomy that distinguishes between L1-controllability, methods that operate under fixed compute budgets, and L2-adaptiveness, methods that dynamically scale inference based on input difficulty or model confidence. We benchmark leading proprietary LLMs across diverse datasets, highlighting critical trade-offs between reasoning performance and token usage. Compared to prior surveys on efficient reasoning, our review emphasizes the practical control, adaptability, and scalability of TTC methods. Finally, we discuss emerging trends such as hybrid thinking models and identify key challenges for future work towards making LLMs more computationally efficient, robust, and responsive to user constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02057v1">MGC: A Compiler Framework Exploiting Compositional Blindness in Aligned LLMs for Malware Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have democratized software development, reducing the expertise barrier for programming complex applications. This accessibility extends to malicious software development, raising significant security concerns. While LLM providers have implemented alignment mechanisms to prevent direct generation of overtly malicious code, these safeguards predominantly evaluate individual prompts in isolation, overlooking a critical vulnerability: malicious operations can be systematically decomposed into benign-appearing sub-tasks. In this paper, we introduce the Malware Generation Compiler (MGC), a novel framework that leverages this vulnerability through modular decomposition and alignment-evasive generation. MGC employs a specialized Malware Description Intermediate Representation (MDIR) to bridge high-level malicious intents and benign-appearing code snippets. Extensive evaluation demonstrates that our attack reliably generates functional malware across diverse task specifications and categories, outperforming jailbreaking methods by +365.79% and underground services by +78.07% in correctness on three benchmark datasets. Case studies further show that MGC can reproduce and even enhance 16 real-world malware samples. This work provides critical insights for security researchers by exposing the risks of compositional attacks against aligned AI systems. Demonstrations are available at https://sites.google.com/view/malware-generation-compiler.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03020v1">AI Literacy and LLM Engagement in Higher Education: A Cross-National Quantitative Study</a></div>
    <div class="paper-meta">
      📅 2025-07-02
      | 💬 26 pages, 8 figures, 3 tables. Submitted for consideration in a forthcoming issue of the International Journal of Educational Technology in Higher Education
    </div>
    <details class="paper-abstract">
      This study presents a cross-national quantitative analysis of how university students in the United States and Bangladesh interact with Large Language Models (LLMs). Based on an online survey of 318 students, results show that LLMs enhance access to information, improve writing, and boost academic performance. However, concerns about overreliance, ethical risks, and critical thinking persist. Guided by the AI Literacy Framework, Expectancy-Value Theory, and Biggs' 3P Model, the study finds that motivational beliefs and technical competencies shape LLM engagement. Significant correlations were found between LLM use and perceived literacy benefits (r = .59, p < .001) and optimism (r = .41, p < .001). ANOVA results showed more frequent use among U.S. students (F = 7.92, p = .005) and STEM majors (F = 18.11, p < .001). Findings support the development of ethical, inclusive, and pedagogically sound frameworks for integrating LLMs in higher education.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03014v1">Intrinsic Fingerprint of LLMs: Continue Training is NOT All You Need to Steal A Model!</a></div>
    <div class="paper-meta">
      📅 2025-07-02
      | 💬 This paper flags a potential case of model plagiarism, copyright violation, and information fabrication in arXiv:2505.21411
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) face significant copyright and intellectual property challenges as the cost of training increases and model reuse becomes prevalent. While watermarking techniques have been proposed to protect model ownership, they may not be robust to continue training and development, posing serious threats to model attribution and copyright protection. This work introduces a simple yet effective approach for robust LLM fingerprinting based on intrinsic model characteristics. We discover that the standard deviation distributions of attention parameter matrices across different layers exhibit distinctive patterns that remain stable even after extensive continued training. These parameter distribution signatures serve as robust fingerprints that can reliably identify model lineage and detect potential copyright infringement. Our experimental validation across multiple model families demonstrates the effectiveness of our method for model authentication. Notably, our investigation uncovers evidence that a recently Pangu Pro MoE model released by Huawei is derived from Qwen-2.5 14B model through upcycling techniques rather than training from scratch, highlighting potential cases of model plagiarism, copyright violation, and information fabrication. These findings underscore the critical importance of developing robust fingerprinting methods for protecting intellectual property in large-scale model development and emphasize that deliberate continued training alone is insufficient to completely obscure model origins.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03004v1">CLUES: Collaborative High-Quality Data Selection for LLMs via Training Dynamics</a></div>
    <div class="paper-meta">
      📅 2025-07-02
      | 💬 NeurIPS 2024
    </div>
    <details class="paper-abstract">
      Recent research has highlighted the importance of data quality in scaling large language models (LLMs). However, automated data quality control faces unique challenges in collaborative settings where sharing is not allowed directly between data silos. To tackle this issue, this paper proposes a novel data quality control technique based on the notion of data influence on the training dynamics of LLMs, that high quality data are more likely to have similar training dynamics to the anchor dataset. We then leverage the influence of the training dynamics to select high-quality data from different private domains, with centralized model updates on the server side in a collaborative training fashion by either model merging or federated learning. As for the data quality indicator, we compute the per-sample gradients with respect to the private data and the anchor dataset, and use the trace of the accumulated inner products as a measurement of data quality. In addition, we develop a quality control evaluation tailored for collaborative settings with heterogeneous domain data. Experiments show that training on the high-quality data selected by our method can often outperform other data selection methods for collaborative fine-tuning of LLMs, across diverse private domain datasets, in medical, multilingual and financial settings. Our code is released at github.com/Ryan0v0/CLUES.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03001v1">Evaluating Hierarchical Clinical Document Classification Using Reasoning-Based LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-02
    </div>
    <details class="paper-abstract">
      This study evaluates how well large language models (LLMs) can classify ICD-10 codes from hospital discharge summaries, a critical but error-prone task in healthcare. Using 1,500 summaries from the MIMIC-IV dataset and focusing on the 10 most frequent ICD-10 codes, the study tested 11 LLMs, including models with and without structured reasoning capabilities. Medical terms were extracted using a clinical NLP tool (cTAKES), and models were prompted in a consistent, coder-like format. None of the models achieved an F1 score above 57%, with performance dropping as code specificity increased. Reasoning-based models generally outperformed non-reasoning ones, with Gemini 2.5 Pro performing best overall. Some codes, such as those related to chronic heart disease, were classified more accurately than others. The findings suggest that while LLMs can assist human coders, they are not yet reliable enough for full automation. Future work should explore hybrid methods, domain-specific model training, and the use of structured clinical data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01936v1">The Thin Line Between Comprehension and Persuasion in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are excellent at maintaining high-level, convincing dialogues. They are being fast deployed as chatbots and evaluators in sensitive areas, such as peer review and mental health applications. This, along with the disparate accounts on their reasoning capabilities, calls for a closer examination of LLMs and their comprehension of dialogue. In this work we begin by evaluating LLMs' ability to maintain a debate--one of the purest yet most complex forms of human communication. Then we measure how this capability relates to their understanding of what is being talked about, namely, their comprehension of dialogical structures and the pragmatic context. We find that LLMs are capable of maintaining coherent, persuasive debates, often swaying the beliefs of participants and audiences alike. We also note that awareness or suspicion of AI involvement encourage people to be more critical of the arguments made. When polling LLMs on their comprehension of deeper structures of dialogue, however, they cannot demonstrate said understanding. Our findings tie the shortcomings of LLMs-as-evaluators to their (in)ability to understand the context. More broadly, for the field of argumentation theory we posit that, if an agent can convincingly maintain a dialogue, it is not necessary for it to know what it is talking about. Hence, the modelling of pragmatic context and coherence are secondary to effectiveness.
    </details>
</div>
