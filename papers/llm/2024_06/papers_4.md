# llm - 2024_06

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- Part 4
- [Part 5](papers_5.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.06822v1">An LLM-Assisted Easy-to-Trigger Backdoor Attack on Code Completion Models: Injecting Disguised Vulnerabilities against Strong Detection</a></div>
    <div class="paper-meta">
      📅 2024-06-10
      | 💬 To appear in USENIX Security '24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have transformed code completion tasks, providing context-based suggestions to boost developer productivity in software engineering. As users often fine-tune these models for specific applications, poisoning and backdoor attacks can covertly alter the model outputs. To address this critical security challenge, we introduce CodeBreaker, a pioneering LLM-assisted backdoor attack framework on code completion models. Unlike recent attacks that embed malicious payloads in detectable or irrelevant sections of the code (e.g., comments), CodeBreaker leverages LLMs (e.g., GPT-4) for sophisticated payload transformation (without affecting functionalities), ensuring that both the poisoned data for fine-tuning and generated code can evade strong vulnerability detection. CodeBreaker stands out with its comprehensive coverage of vulnerabilities, making it the first to provide such an extensive set for evaluation. Our extensive experimental evaluations and user studies underline the strong attack performance of CodeBreaker across various settings, validating its superiority over existing approaches. By integrating malicious payloads directly into the source code with minimal transformation, CodeBreaker challenges current security measures, underscoring the critical need for more robust defenses for code completion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.06773v1">Evaluating Zero-Shot Long-Context LLM Compression</a></div>
    <div class="paper-meta">
      📅 2024-06-10
    </div>
    <details class="paper-abstract">
      This study evaluates the effectiveness of zero-shot compression techniques on large language models (LLMs) under long-context. We identify the tendency for computational errors to increase under long-context when employing certain compression methods. We propose a hypothesis to explain the varied behavior of different LLM compression techniques and explore remedies to mitigate the performance decline observed in some techniques under long-context. This is a course report for COS 598D Machine Learning and Systems by Prof. Kai Li at Princeton University. Due to limited computational resources, our experiments were conducted only on LLaMA-2-7B-32K.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.04550v3">Retrieving Evidence from EHRs with LLMs: Possibilities and Challenges</a></div>
    <div class="paper-meta">
      📅 2024-06-10
    </div>
    <details class="paper-abstract">
      Unstructured data in Electronic Health Records (EHRs) often contains critical information -- complementary to imaging -- that could inform radiologists' diagnoses. But the large volume of notes often associated with patients together with time constraints renders manually identifying relevant evidence practically infeasible. In this work we propose and evaluate a zero-shot strategy for using LLMs as a mechanism to efficiently retrieve and summarize unstructured evidence in patient EHR relevant to a given query. Our method entails tasking an LLM to infer whether a patient has, or is at risk of, a particular condition on the basis of associated notes; if so, we ask the model to summarize the supporting evidence. Under expert evaluation, we find that this LLM-based approach provides outputs consistently preferred to a pre-LLM information retrieval baseline. Manual evaluation is expensive, so we also propose and validate a method using an LLM to evaluate (other) LLM outputs for this task, allowing us to scale up evaluation. Our findings indicate the promise of LLMs as interfaces to EHR, but also highlight the outstanding challenge posed by "hallucinations". In this setting, however, we show that model confidence in outputs strongly correlates with faithful summaries, offering a practical means to limit confabulations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.17739v2">How LLMs Aid in UML Modeling: An Exploratory Study with Novice Analysts</a></div>
    <div class="paper-meta">
      📅 2024-06-10
      | 💬 The 21st IEEE International Conference on Software Services Engineering (SSE)
    </div>
    <details class="paper-abstract">
      Since the emergence of GPT-3, Large Language Models (LLMs) have caught the eyes of researchers, practitioners, and educators in the field of software engineering. However, there has been relatively little investigation regarding the performance of LLMs in assisting with requirements analysis and UML modeling. This paper explores how LLMs can assist novice analysts in creating three types of typical UML models: use case models, class diagrams, and sequence diagrams. For this purpose, we designed the modeling tasks of these three UML models for 45 undergraduate students who participated in a requirements modeling course, with the help of LLMs. By analyzing their project reports, we found that LLMs can assist undergraduate students as novice analysts in UML modeling tasks, but LLMs also have shortcomings and limitations that should be considered when using them.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.12348v2">GTBench: Uncovering the Strategic Reasoning Limitations of LLMs via Game-Theoretic Evaluations</a></div>
    <div class="paper-meta">
      📅 2024-06-10
      | 💬 26 pages; the first two authors contributed equally; GTBench HF Leaderboard: https://huggingface.co/spaces/GTBench/GTBench
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) are integrated into critical real-world applications, their strategic and logical reasoning abilities are increasingly crucial. This paper evaluates LLMs' reasoning abilities in competitive environments through game-theoretic tasks, e.g., board and card games that require pure logic and strategic reasoning to compete with opponents. We first propose GTBench, a language-driven environment composing 10 widely recognized tasks, across a comprehensive game taxonomy: complete versus incomplete information, dynamic versus static, and probabilistic versus deterministic scenarios. Then, we (1) Characterize the game-theoretic reasoning of LLMs; and (2) Perform LLM-vs.-LLM competitions as reasoning evaluation. We observe that (1) LLMs have distinct behaviors regarding various gaming scenarios; for example, LLMs fail in complete and deterministic games yet they are competitive in probabilistic gaming scenarios; (2) Most open-source LLMs, e.g., CodeLlama-34b-Instruct and Llama-2-70b-chat, are less competitive than commercial LLMs, e.g., GPT-4, in complex games, yet the recently released Llama-3-70b-Instruct makes up for this shortcoming. In addition, code-pretraining greatly benefits strategic reasoning, while advanced reasoning methods such as Chain-of-Thought (CoT) and Tree-of-Thought (ToT) do not always help. We further characterize the game-theoretic properties of LLMs, such as equilibrium and Pareto Efficiency in repeated games. Detailed error profiles are provided for a better understanding of LLMs' behavior. We hope our research provides standardized protocols and serves as a foundation to spur further explorations in the strategic reasoning of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.06458v1">Evaluating the Retrieval Component in LLM-Based Question Answering Systems</a></div>
    <div class="paper-meta">
      📅 2024-06-10
    </div>
    <details class="paper-abstract">
      Question answering systems (QA) utilizing Large Language Models (LLMs) heavily depend on the retrieval component to provide them with domain-specific information and reduce the risk of generating inaccurate responses or hallucinations. Although the evaluation of retrievers dates back to the early research in Information Retrieval, assessing their performance within LLM-based chatbots remains a challenge. This study proposes a straightforward baseline for evaluating retrievers in Retrieval-Augmented Generation (RAG)-based chatbots. Our findings demonstrate that this evaluation framework provides a better image of how the retriever performs and is more aligned with the overall performance of the QA system. Although conventional metrics such as precision, recall, and F1 score may not fully capture LLMs' capabilities - as they can yield accurate responses despite imperfect retrievers - our method considers LLMs' strengths to ignore irrelevant contexts, as well as potential errors and hallucinations in their responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.06443v1">LLM Dataset Inference: Did you train on my dataset?</a></div>
    <div class="paper-meta">
      📅 2024-06-10
      | 💬 Code is available at \href{https://github.com/pratyushmaini/llm_dataset_inference/
    </div>
    <details class="paper-abstract">
      The proliferation of large language models (LLMs) in the real world has come with a rise in copyright cases against companies for training their models on unlicensed data from the internet. Recent works have presented methods to identify if individual text sequences were members of the model's training data, known as membership inference attacks (MIAs). We demonstrate that the apparent success of these MIAs is confounded by selecting non-members (text sequences not used for training) belonging to a different distribution from the members (e.g., temporally shifted recent Wikipedia articles compared with ones used to train the model). This distribution shift makes membership inference appear successful. However, most MIA methods perform no better than random guessing when discriminating between members and non-members from the same distribution (e.g., in this case, the same period of time). Even when MIAs work, we find that different MIAs succeed at inferring membership of samples from different distributions. Instead, we propose a new dataset inference method to accurately identify the datasets used to train large language models. This paradigm sits realistically in the modern-day copyright landscape, where authors claim that an LLM is trained over multiple documents (such as a book) written by them, rather than one particular paragraph. While dataset inference shares many of the challenges of membership inference, we solve it by selectively combining the MIAs that provide positive signal for a given distribution, and aggregating them to perform a statistical test on a given dataset. Our approach successfully distinguishes the train and test sets of different subsets of the Pile with statistically significant p-values < 0.1, without any false positives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.17234v2">Cooperation, Competition, and Maliciousness: LLM-Stakeholders Interactive Negotiation</a></div>
    <div class="paper-meta">
      📅 2024-06-10
      | 💬 Updated version with major additions (new experiments, evaluation, and attacks)
    </div>
    <details class="paper-abstract">
      There is an growing interest in using Large Language Models (LLMs) in multi-agent systems to tackle interactive real-world tasks that require effective collaboration and assessing complex situations. Yet, we still have a limited understanding of LLMs' communication and decision-making abilities in multi-agent setups. The fundamental task of negotiation spans many key features of communication, such as cooperation, competition, and manipulation potentials. Thus, we propose using scorable negotiation to evaluate LLMs. We create a testbed of complex multi-agent, multi-issue, and semantically rich negotiation games. To reach an agreement, agents must have strong arithmetic, inference, exploration, and planning capabilities while integrating them in a dynamic and multi-turn setup. We propose multiple metrics to rigorously quantify agents' performance and alignment with the assigned role. We provide procedures to create new games and increase games' difficulty to have an evolving benchmark. Importantly, we evaluate critical safety aspects such as the interaction dynamics between agents influenced by greedy and adversarial players. Our benchmark is highly challenging; GPT-3.5 and small models mostly fail, and GPT-4 and SoTA large models (e.g., Llama-3 70b) still underperform.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.06503v3">Knowledgeable Preference Alignment for LLMs in Domain-specific Question Answering</a></div>
    <div class="paper-meta">
      📅 2024-06-10
      | 💬 Accepted by ACL 2024 (Findings). Code is available at https://github.com/zjukg/KnowPAT
    </div>
    <details class="paper-abstract">
      Deploying large language models (LLMs) to real scenarios for domain-specific question answering (QA) is a key thrust for LLM applications, which poses numerous challenges, especially in ensuring that responses are both accommodating to user requirements and appropriately leveraging domain-specific knowledge bases. They are the two major difficulties for LLM application as vanilla fine-tuning falls short of addressing. Combining these requirements, we conceive of them as the requirement for the model's preference to be harmoniously aligned with humans'. Thus, we introduce Knowledgeable Preference AlignmenT (KnowPAT), which constructs two kinds of preference sets to tackle the two issues. Besides, we design a new alignment objective to align the LLM preference with different human preferences uniformly, aiming to optimize LLM performance in real-world, domain-specific QA settings. Adequate experiments and comprehensive comparisons with 15 baseline methods illustrate that our KnowPAT is a superior pipeline for real-scenario domain-specific QA with LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.06110v1">Recurrent Context Compression: Efficiently Expanding the Context Window of LLM</a></div>
    <div class="paper-meta">
      📅 2024-06-10
    </div>
    <details class="paper-abstract">
      To extend the context length of Transformer-based large language models (LLMs) and improve comprehension capabilities, we often face limitations due to computational resources and bounded memory storage capacity. This work introduces a method called Recurrent Context Compression (RCC), designed to efficiently expand the context window length of LLMs within constrained storage space. We also investigate the issue of poor model responses when both instructions and context are compressed in downstream tasks, and propose an instruction reconstruction method to mitigate this problem. We validated the effectiveness of our approach on multiple tasks, achieving a compression rate of up to 32x on text reconstruction tasks with a BLEU4 score close to 0.95, and nearly 100\% accuracy on a passkey retrieval task with a sequence length of 1M. Finally, our method demonstrated competitive performance in long-text question-answering tasks compared to non-compressed methods, while significantly saving storage resources in long-text inference tasks. Our code, models, and demo are available at https://github.com/WUHU-G/RCC_Transformer
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.06027v1">HOLMES: Hyper-Relational Knowledge Graphs for Multi-hop Question Answering using LLMs</a></div>
    <div class="paper-meta">
      📅 2024-06-10
      | 💬 Accepted at ACL 2024 in the main track
    </div>
    <details class="paper-abstract">
      Given unstructured text, Large Language Models (LLMs) are adept at answering simple (single-hop) questions. However, as the complexity of the questions increase, the performance of LLMs degrade. We believe this is due to the overhead associated with understanding the complex question followed by filtering and aggregating unstructured information in the raw text. Recent methods try to reduce this burden by integrating structured knowledge triples into the raw text, aiming to provide a structured overview that simplifies information processing. However, this simplistic approach is query-agnostic and the extracted facts are ambiguous as they lack context. To address these drawbacks and to enable LLMs to answer complex (multi-hop) questions with ease, we propose to use a knowledge graph (KG) that is context-aware and is distilled to contain query-relevant information. The use of our compressed distilled KG as input to the LLM results in our method utilizing up to $67\%$ fewer tokens to represent the query relevant information present in the supporting documents, compared to the state-of-the-art (SoTA) method. Our experiments show consistent improvements over the SoTA across several metrics (EM, F1, BERTScore, and Human Eval) on two popular benchmark datasets (HotpotQA and MuSiQue).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.10198v2">ClashEval: Quantifying the tug-of-war between an LLM's internal prior and external evidence</a></div>
    <div class="paper-meta">
      📅 2024-06-10
      | 💬 Revised June 9 2024
    </div>
    <details class="paper-abstract">
      Retrieval augmented generation (RAG) is frequently used to mitigate hallucinations and provide up-to-date knowledge for large language models (LLMs). However, given that document retrieval is an imprecise task and sometimes results in erroneous or even harmful content being presented in context, this raises the question of how LLMs handle retrieved information: If the provided content is incorrect, does the model know to ignore it, or does it recapitulate the error? Conversely, when the model's initial response is incorrect, does it always know to use the retrieved information to correct itself, or does it insist on its wrong prior response? To answer this, we curate a dataset of over 1200 questions across six domains (e.g., drug dosages, Olympic records, locations) along with content relevant to answering each question. We further apply precise perturbations to the answers in the content that range from subtle to blatant errors. We benchmark six top-performing LLMs, including GPT-4o, on this dataset and find that LLMs are susceptible to adopting incorrect retrieved content, overriding their own correct prior knowledge over 60% of the time. However, the more unrealistic the retrieved content is (i.e. more deviated from truth), the less likely the model is to adopt it. Also, the less confident a model is in its initial response (via measuring token probabilities), the more likely it is to adopt the information in the retrieved content. We exploit this finding and demonstrate simple methods for improving model accuracy where there is conflicting retrieved content. Our results highlight a difficult task and benchmark for LLMs -- namely, their ability to correctly discern when it is wrong in light of correct retrieved content and to reject cases when the provided content is incorrect.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.05925v1">Hello Again! LLM-powered Personalized Agent for Long-term Dialogue</a></div>
    <div class="paper-meta">
      📅 2024-06-09
      | 💬 17 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Open-domain dialogue systems have seen remarkable advancements with the development of large language models (LLMs). Nonetheless, most existing dialogue systems predominantly focus on brief single-session interactions, neglecting the real-world demands for long-term companionship and personalized interactions with chatbots. Crucial to addressing this real-world need are event summary and persona management, which enable reasoning for appropriate long-term dialogue responses. Recent progress in the human-like cognitive and reasoning capabilities of LLMs suggests that LLM-based agents could significantly enhance automated perception, decision-making, and problem-solving. In response to this potential, we introduce a model-agnostic framework, the Long-term Dialogue Agent (LD-Agent), which incorporates three independently tunable modules dedicated to event perception, persona extraction, and response generation. For the event memory module, long and short-term memory banks are employed to separately focus on historical and ongoing sessions, while a topic-based retrieval mechanism is introduced to enhance the accuracy of memory retrieval. Furthermore, the persona module conducts dynamic persona modeling for both users and agents. The integration of retrieved memories and extracted personas is subsequently fed into the generator to induce appropriate responses. The effectiveness, generality, and cross-domain capabilities of LD-Agent are empirically demonstrated across various illustrative benchmarks, models, and tasks. The code is released at https://github.com/leolee99/LD-Agent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.05882v1">Distributional Preference Alignment of LLMs via Optimal Transport</a></div>
    <div class="paper-meta">
      📅 2024-06-09
    </div>
    <details class="paper-abstract">
      Current LLM alignment techniques use pairwise human preferences at a sample level, and as such, they do not imply an alignment on the distributional level. We propose in this paper Alignment via Optimal Transport (AOT), a novel method for distributional preference alignment of LLMs. AOT aligns LLMs on unpaired preference data by making the reward distribution of the positive samples stochastically dominant in the first order on the distribution of negative samples. We introduce a convex relaxation of this first-order stochastic dominance and cast it as an optimal transport problem with a smooth and convex cost. Thanks to the one-dimensional nature of the resulting optimal transport problem and the convexity of the cost, it has a closed-form solution via sorting on empirical measures. We fine-tune LLMs with this AOT objective, which enables alignment by penalizing the violation of the stochastic dominance of the reward distribution of the positive samples on the reward distribution of the negative samples. We analyze the sample complexity of AOT by considering the dual of the OT problem and show that it converges at the parametric rate. Empirically, we show on a diverse set of alignment datasets and LLMs that AOT leads to state-of-the-art models in the 7B family of models when evaluated with Open LLM Benchmarks and AlpacaEval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.11100v2">When LLMs Meet Cunning Texts: A Fallacy Understanding Benchmark for Large Language Models</a></div>
    <div class="paper-meta">
      📅 2024-06-09
    </div>
    <details class="paper-abstract">
      Recently, Large Language Models (LLMs) make remarkable evolutions in language understanding and generation. Following this, various benchmarks for measuring all kinds of capabilities of LLMs have sprung up. In this paper, we challenge the reasoning and understanding abilities of LLMs by proposing a FaLlacy Understanding Benchmark (FLUB) containing cunning texts that are easy for humans to understand but difficult for models to grasp. Specifically, the cunning texts that FLUB focuses on mainly consist of the tricky, humorous, and misleading texts collected from the real internet environment. And we design three tasks with increasing difficulty in the FLUB benchmark to evaluate the fallacy understanding ability of LLMs. Based on FLUB, we investigate the performance of multiple representative and advanced LLMs, reflecting our FLUB is challenging and worthy of more future study. Interesting discoveries and valuable insights are achieved in our extensive experiments and detailed analyses. We hope that our benchmark can encourage the community to improve LLMs' ability to understand fallacies. Our data and codes are available at https://github.com/THUKElab/FLUB.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.17644v2">Are LLMs Capable of Data-based Statistical and Causal Reasoning? Benchmarking Advanced Quantitative Reasoning with Data</a></div>
    <div class="paper-meta">
      📅 2024-06-09
      | 💬 Findings of ACL 2024. Project website: https://xxxiaol.github.io/QRData/
    </div>
    <details class="paper-abstract">
      Quantitative reasoning is a critical skill to analyze data, yet the assessment of such ability remains limited. To address this gap, we introduce the Quantitative Reasoning with Data (QRData) benchmark, aiming to evaluate Large Language Models' capability in statistical and causal reasoning with real-world data. The benchmark comprises a carefully constructed dataset of 411 questions accompanied by data sheets from textbooks, online learning materials, and academic papers. To compare models' quantitative reasoning abilities on data and text, we enrich the benchmark with an auxiliary set of 290 text-only questions, namely QRText. We evaluate natural language reasoning, program-based reasoning, and agent reasoning methods including Chain-of-Thought, Program-of-Thoughts, ReAct, and code interpreter assistants on diverse models. The strongest model GPT-4 achieves an accuracy of 58%, which has much room for improvement. Among open-source models, Deepseek-coder-instruct, a code LLM pretrained on 2T tokens, gets the highest accuracy of 37%. Analysis reveals that models encounter difficulties in data analysis and causal reasoning, and struggle in using causal knowledge and provided data simultaneously. Code and data are in https://github.com/xxxiaol/QRData.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.01509v2">Fantastic Semantics and Where to Find Them: Investigating Which Layers of Generative LLMs Reflect Lexical Semantics</a></div>
    <div class="paper-meta">
      📅 2024-06-09
      | 💬 Accepted to Findings of ACL 2024
    </div>
    <details class="paper-abstract">
      Large language models have achieved remarkable success in general language understanding tasks. However, as a family of generative methods with the objective of next token prediction, the semantic evolution with the depth of these models are not fully explored, unlike their predecessors, such as BERT-like architectures. In this paper, we specifically investigate the bottom-up evolution of lexical semantics for a popular LLM, namely Llama2, by probing its hidden states at the end of each layer using a contextualized word identification task. Our experiments show that the representations in lower layers encode lexical semantics, while the higher layers, with weaker semantic induction, are responsible for prediction. This is in contrast to models with discriminative objectives, such as mask language modeling, where the higher layers obtain better lexical semantics. The conclusion is further supported by the monotonic increase in performance via the hidden states for the last meaningless symbols, such as punctuation, in the prompting strategy. Our codes are available at https://github.com/RyanLiut/LLM_LexSem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.19334v2">LLMs Meet Multimodal Generation and Editing: A Survey</a></div>
    <div class="paper-meta">
      📅 2024-06-09
      | 💬 52 Pages with 16 Figures, 12 Tables, and 545 References. GitHub Repository at: https://github.com/YingqingHe/Awesome-LLMs-meet-Multimodal-Generation
    </div>
    <details class="paper-abstract">
      With the recent advancement in large language models (LLMs), there is a growing interest in combining LLMs with multimodal learning. Previous surveys of multimodal large language models (MLLMs) mainly focus on multimodal understanding. This survey elaborates on multimodal generation and editing across various domains, comprising image, video, 3D, and audio. Specifically, we summarize the notable advancements with milestone works in these fields and categorize these studies into LLM-based and CLIP/T5-based methods. Then, we summarize the various roles of LLMs in multimodal generation and exhaustively investigate the critical technical components behind these methods and the multimodal datasets utilized in these studies. Additionally, we dig into tool-augmented multimodal agents that can leverage existing generative models for human-computer interaction. Lastly, we discuss the advancements in the generative AI safety field, investigate emerging applications, and discuss future prospects. Our work provides a systematic and insightful overview of multimodal generation and processing, which is expected to advance the development of Artificial Intelligence for Generative Content (AIGC) and world models. A curated list of all related papers can be found at https://github.com/YingqingHe/Awesome-LLMs-meet-Multimodal-Generation
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.05709v1">TR2MTL: LLM based framework for Metric Temporal Logic Formalization of Traffic Rules</a></div>
    <div class="paper-meta">
      📅 2024-06-09
      | 💬 Accepted for publication in Proceedings of the IEEE Intelligent Vehicles Symposium (IV), Jeju Island - Korea, 2-5 June 2024
    </div>
    <details class="paper-abstract">
      Traffic rules formalization is crucial for verifying the compliance and safety of autonomous vehicles (AVs). However, manual translation of natural language traffic rules as formal specification requires domain knowledge and logic expertise, which limits its adaptation. This paper introduces TR2MTL, a framework that employs large language models (LLMs) to automatically translate traffic rules (TR) into metric temporal logic (MTL). It is envisioned as a human-in-loop system for AV rule formalization. It utilizes a chain-of-thought in-context learning approach to guide the LLM in step-by-step translation and generating valid and grammatically correct MTL formulas. It can be extended to various forms of temporal logic and rules. We evaluated the framework on a challenging dataset of traffic rules we created from various sources and compared it against LLMs using different in-context learning methods. Results show that TR2MTL is domain-agnostic, achieving high accuracy and generalization capability even with a small dataset. Moreover, the method effectively predicts formulas with varying degrees of logical and semantic structure in unstructured traffic rules.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.06636v1">LLM Questionnaire Completion for Automatic Psychiatric Assessment</a></div>
    <div class="paper-meta">
      📅 2024-06-09
    </div>
    <details class="paper-abstract">
      We employ a Large Language Model (LLM) to convert unstructured psychological interviews into structured questionnaires spanning various psychiatric and personality domains. The LLM is prompted to answer these questionnaires by impersonating the interviewee. The obtained answers are coded as features, which are used to predict standardized psychiatric measures of depression (PHQ-8) and PTSD (PCL-C), using a Random Forest regressor. Our approach is shown to enhance diagnostic accuracy compared to multiple baselines. It thus establishes a novel framework for interpreting unstructured psychological interviews, bridging the gap between narrative-driven and data-driven approaches for mental health assessment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.11683v2">One Prompt To Rule Them All: LLMs for Opinion Summary Evaluation</a></div>
    <div class="paper-meta">
      📅 2024-06-09
    </div>
    <details class="paper-abstract">
      Evaluation of opinion summaries using conventional reference-based metrics rarely provides a holistic evaluation and has been shown to have a relatively low correlation with human judgments. Recent studies suggest using Large Language Models (LLMs) as reference-free metrics for NLG evaluation, however, they remain unexplored for opinion summary evaluation. Moreover, limited opinion summary evaluation datasets inhibit progress. To address this, we release the SUMMEVAL-OP dataset covering 7 dimensions related to the evaluation of opinion summaries: fluency, coherence, relevance, faithfulness, aspect coverage, sentiment consistency, and specificity. We investigate Op-I-Prompt a dimension-independent prompt, and Op-Prompts, a dimension-dependent set of prompts for opinion summary evaluation. Experiments indicate that Op-I-Prompt emerges as a good alternative for evaluating opinion summaries achieving an average Spearman correlation of 0.70 with humans, outperforming all previous approaches. To the best of our knowledge, we are the first to investigate LLMs as evaluators on both closed-source and open-source models in the opinion summarization domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.05659v1">Do LLMs Exhibit Human-Like Reasoning? Evaluating Theory of Mind in LLMs for Open-Ended Responses</a></div>
    <div class="paper-meta">
      📅 2024-06-09
    </div>
    <details class="paper-abstract">
      Theory of Mind (ToM) reasoning entails recognizing that other individuals possess their own intentions, emotions, and thoughts, which is vital for guiding one's own thought processes. Although large language models (LLMs) excel in tasks such as summarization, question answering, and translation, they still face challenges with ToM reasoning, especially in open-ended questions. Despite advancements, the extent to which LLMs truly understand ToM reasoning and how closely it aligns with human ToM reasoning remains inadequately explored in open-ended scenarios. Motivated by this gap, we assess the abilities of LLMs to perceive and integrate human intentions and emotions into their ToM reasoning processes within open-ended questions. Our study utilizes posts from Reddit's ChangeMyView platform, which demands nuanced social reasoning to craft persuasive responses. Our analysis, comparing semantic similarity and lexical overlap metrics between responses generated by humans and LLMs, reveals clear disparities in ToM reasoning capabilities in open-ended questions, with even the most advanced models showing notable limitations. To enhance LLM capabilities, we implement a prompt tuning method that incorporates human intentions and emotions, resulting in improvements in ToM reasoning performance. However, despite these improvements, the enhancement still falls short of fully achieving human-like reasoning. This research highlights the deficiencies in LLMs' social reasoning and demonstrates how integrating human intentions and emotions can boost their effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.10249v1">A Reality check of the benefits of LLM in business</a></div>
    <div class="paper-meta">
      📅 2024-06-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable performance in language understanding and generation tasks by leveraging vast amounts of online texts. Unlike conventional models, LLMs can adapt to new domains through prompt engineering without the need for retraining, making them suitable for various business functions, such as strategic planning, project implementation, and data-driven decision-making. However, their limitations in terms of bias, contextual understanding, and sensitivity to prompts raise concerns about their readiness for real-world applications. This paper thoroughly examines the usefulness and readiness of LLMs for business processes. The limitations and capacities of LLMs are evaluated through experiments conducted on four accessible LLMs using real-world data. The findings have significant implications for organizations seeking to leverage generative AI and provide valuable insights into future research directions. To the best of our knowledge, this represents the first quantified study of LLMs applied to core business operations and challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.06961v2">CHAMP: A Competition-level Dataset for Fine-Grained Analyses of LLMs' Mathematical Reasoning Capabilities</a></div>
    <div class="paper-meta">
      📅 2024-06-09
      | 💬 ACL 2024 (Findings). Project website at https://yujunmao1.github.io/CHAMP/
    </div>
    <details class="paper-abstract">
      Recent large language models (LLMs) have shown indications of mathematical reasoning ability on challenging competition-level problems, especially with self-generated verbalizations of intermediate reasoning steps (i.e., chain-of-thought prompting). However, current evaluations mainly focus on the end-to-end final answer correctness, and it is unclear whether LLMs can make use of helpful side information such as problem-specific hints. In this paper, we propose a challenging benchmark dataset for enabling such analyses. The Concept and Hint-Annotated Math Problems (CHAMP) consists of high school math competition problems, annotated with concepts, or general math facts, and hints, or problem-specific tricks. These annotations allow us to explore the effects of additional information, such as relevant hints, misleading concepts, or related problems. This benchmark is difficult, with the best model only scoring 58.1% in standard settings. With concepts and hints, performance sometimes improves, indicating that some models can make use of such side information. Furthermore, we annotate model-generated solutions for their correctness. Using this corpus, we find that models often arrive at the correct final answer through wrong reasoning steps. In addition, we test whether models are able to verify these solutions, and find that most models struggle.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.11756v3">MARS: Meaning-Aware Response Scoring for Uncertainty Estimation in Generative LLMs</a></div>
    <div class="paper-meta">
      📅 2024-06-08
    </div>
    <details class="paper-abstract">
      Generative Large Language Models (LLMs) are widely utilized for their excellence in various tasks. However, their tendency to produce inaccurate or misleading outputs poses a potential risk, particularly in high-stakes environments. Therefore, estimating the correctness of generative LLM outputs is an important task for enhanced reliability. Uncertainty Estimation (UE) in generative LLMs is an evolving domain, where SOTA probability-based methods commonly employ length-normalized scoring. In this work, we propose Meaning-Aware Response Scoring (MARS) as an alternative to length-normalized scoring for UE methods. MARS is a novel scoring function that considers the semantic contribution of each token in the generated sequence in the context of the question. We demonstrate that integrating MARS into UE methods results in a universal and significant improvement in UE performance. We conduct experiments using three distinct closed-book question-answering datasets across five popular pre-trained LLMs. Lastly, we validate the efficacy of MARS on a Medical QA dataset. Code can be found https://github.com/Ybakman/LLM_Uncertainity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.05569v1">Do LLMs Recognize me, When I is not me: Assessment of LLMs Understanding of Turkish Indexical Pronouns in Indexical Shift Contexts</a></div>
    <div class="paper-meta">
      📅 2024-06-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown impressive capabilities in tasks such as machine translation, text summarization, question answering, and solving complex mathematical problems. However, their primary training on data-rich languages like English limits their performance in low-resource languages. This study addresses this gap by focusing on the Indexical Shift problem in Turkish. The Indexical Shift problem involves resolving pronouns in indexical shift contexts, a grammatical challenge not present in high-resource languages like English. We present the first study examining indexical shift in any language, releasing a Turkish dataset specifically designed for this purpose. Our Indexical Shift Dataset consists of 156 multiple-choice questions, each annotated with necessary linguistic details, to evaluate LLMs in a few-shot setting. We evaluate recent multilingual LLMs, including GPT-4, GPT-3.5, Cohere-AYA, Trendyol-LLM, and Turkcell-LLM, using this dataset. Our analysis reveals that even advanced models like GPT-4 struggle with the grammatical nuances of indexical shift in Turkish, achieving only moderate performance. These findings underscore the need for focused research on the grammatical challenges posed by low-resource languages. We released the dataset and code \href{https://anonymous.4open.science/r/indexical_shift_llm-E1B4} {here}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.05494v1">Investigating and Addressing Hallucinations of LLMs in Tasks Involving Negation</a></div>
    <div class="paper-meta">
      📅 2024-06-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable performance across a wide variety of natural language tasks. However, they have been shown to suffer from a critical limitation pertinent to 'hallucination' in their output. Recent research has focused on investigating and addressing this problem for a variety of tasks such as biography generation, question answering, abstractive summarization, and dialogue generation. However, the crucial aspect pertaining to 'negation' has remained considerably underexplored. Negation is important because it adds depth and nuance to the understanding of language and is also crucial for logical reasoning and inference. In this work, we address the above limitation and particularly focus on studying the impact of negation in LLM hallucinations. Specifically, we study four tasks with negation: 'false premise completion', 'constrained fact generation', 'multiple choice question answering', and 'fact generation'. We show that open-source state-of-the-art LLMs such as LLaMA-2-chat, Vicuna, and Orca-2 hallucinate considerably on all these tasks involving negation which underlines a critical shortcoming of these models. Addressing this problem, we further study numerous strategies to mitigate these hallucinations and demonstrate their impact.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.11960v2">Generating Diverse Criteria On-the-Fly to Improve Point-wise LLM Rankers</a></div>
    <div class="paper-meta">
      📅 2024-06-08
    </div>
    <details class="paper-abstract">
      The most recent pointwise Large Language Model (LLM) rankers have achieved remarkable ranking results. However, these rankers are hindered by two major drawbacks: (1) they fail to follow a standardized comparison guidance during the ranking process, and (2) they struggle with comprehensive considerations when dealing with complicated passages. To address these shortcomings, we propose to build a ranker that generates ranking scores based on a set of criteria from various perspectives. These criteria are intended to direct each perspective in providing a distinct yet synergistic evaluation. Our research, which examines eight datasets from the BEIR benchmark demonstrates that incorporating this multi-perspective criteria ensemble approach markedly enhanced the performance of pointwise LLM rankers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.05344v1">MemeGuard: An LLM and VLM-based Framework for Advancing Content Moderation via Meme Intervention</a></div>
    <div class="paper-meta">
      📅 2024-06-08
    </div>
    <details class="paper-abstract">
      In the digital world, memes present a unique challenge for content moderation due to their potential to spread harmful content. Although detection methods have improved, proactive solutions such as intervention are still limited, with current research focusing mostly on text-based content, neglecting the widespread influence of multimodal content like memes. Addressing this gap, we present \textit{MemeGuard}, a comprehensive framework leveraging Large Language Models (LLMs) and Visual Language Models (VLMs) for meme intervention. \textit{MemeGuard} harnesses a specially fine-tuned VLM, \textit{VLMeme}, for meme interpretation, and a multimodal knowledge selection and ranking mechanism (\textit{MKS}) for distilling relevant knowledge. This knowledge is then employed by a general-purpose LLM to generate contextually appropriate interventions. Another key contribution of this work is the \textit{\textbf{I}ntervening} \textit{\textbf{C}yberbullying in \textbf{M}ultimodal \textbf{M}emes (ICMM)} dataset, a high-quality, labeled dataset featuring toxic memes and their corresponding human-annotated interventions. We leverage \textit{ICMM} to test \textit{MemeGuard}, demonstrating its proficiency in generating relevant and effective responses to toxic memes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.12195v2">Browse and Concentrate: Comprehending Multimodal Content via prior-LLM Context Fusion</a></div>
    <div class="paper-meta">
      📅 2024-06-08
      | 💬 17 pages, 5 figures
    </div>
    <details class="paper-abstract">
      With the bloom of Large Language Models (LLMs), Multimodal Large Language Models (MLLMs) that incorporate LLMs with pre-trained vision models have recently demonstrated impressive performance across diverse vision-language tasks. However, they fall short to comprehend context involving multiple images. A primary reason for this shortcoming is that the visual features for each images are encoded individually by frozen encoders before feeding into the LLM backbone, lacking awareness of other images and the multimodal instructions. We term this issue as prior-LLM modality isolation and propose a two phase paradigm, browse-and-concentrate, to enable in-depth multimodal context fusion prior to feeding the features into LLMs. This paradigm initially "browses" through the inputs for essential insights, and then revisits the inputs to "concentrate" on crucial details, guided by these insights, to achieve a more comprehensive understanding of the multimodal inputs. Additionally, we develop training strategies specifically to enhance the understanding of multi-image inputs. Our method markedly boosts the performance on 7 multi-image scenarios, contributing to increments on average accuracy by 2.13% and 7.60% against strong MLLMs baselines with 3B and 11B LLMs, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.03317v2">Save It for the "Hot" Day: An LLM-Empowered Visual Analytics System for Heat Risk Management</a></div>
    <div class="paper-meta">
      📅 2024-06-08
    </div>
    <details class="paper-abstract">
      The escalating frequency and intensity of heat-related climate events, particularly heatwaves, emphasize the pressing need for advanced heat risk management strategies. Current approaches, primarily relying on numerical models, face challenges in spatial-temporal resolution and in capturing the dynamic interplay of environmental, social, and behavioral factors affecting heat risks. This has led to difficulties in translating risk assessments into effective mitigation actions. Recognizing these problems, we introduce a novel approach leveraging the burgeoning capabilities of Large Language Models (LLMs) to extract rich and contextual insights from news reports. We hence propose an LLM-empowered visual analytics system, Havior, that integrates the precise, data-driven insights of numerical models with nuanced news report information. This hybrid approach enables a more comprehensive assessment of heat risks and better identification, assessment, and mitigation of heat-related threats. The system incorporates novel visualization designs, such as "thermoglyph" and news glyph, enhancing intuitive understanding and analysis of heat risks. The integration of LLM-based techniques also enables advanced information retrieval and semantic knowledge extraction that can be guided by experts' analytics needs. Our case studies on two cities that faced significant heatwave events and interviews with five experts have demonstrated the usefulness of our system in providing in-depth and actionable insights for heat risk management.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.12483v2">Artifacts or Abduction: How Do LLMs Answer Multiple-Choice Questions Without the Question?</a></div>
    <div class="paper-meta">
      📅 2024-06-07
      | 💬 ACL 2024
    </div>
    <details class="paper-abstract">
      Multiple-choice question answering (MCQA) is often used to evaluate large language models (LLMs). To see if MCQA assesses LLMs as intended, we probe if LLMs can perform MCQA with choices-only prompts, where models must select the correct answer only from the choices. In three MCQA datasets and four LLMs, this prompt bests a majority baseline in 11/12 cases, with up to 0.33 accuracy gain. To help explain this behavior, we conduct an in-depth, black-box analysis on memorization, choice dynamics, and question inference. Our key findings are threefold. First, we find no evidence that the choices-only accuracy stems from memorization alone. Second, priors over individual choices do not fully explain choices-only accuracy, hinting that LLMs use the group dynamics of choices. Third, LLMs have some ability to infer a relevant question from choices, and surprisingly can sometimes even match the original question. Inferring the original question is an impressive reasoning strategy, but it cannot fully explain the high choices-only accuracy of LLMs in MCQA. Thus, while LLMs are not fully incapable of reasoning in MCQA, we still advocate for the use of stronger baselines in MCQA benchmarks, the design of robust MCQA datasets for fair evaluations, and further efforts to explain LLM decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.05255v1">Generative Explore-Exploit: Training-free Optimization of Generative Recommender Systems using LLM Optimizers</a></div>
    <div class="paper-meta">
      📅 2024-06-07
      | 💬 Accepted at ACL 2024 Main Proceedings
    </div>
    <details class="paper-abstract">
      Recommender systems are widely used to suggest engaging content, and Large Language Models (LLMs) have given rise to generative recommenders. Such systems can directly generate items, including for open-set tasks like question suggestion. While the world knowledge of LLMs enable good recommendations, improving the generated content through user feedback is challenging as continuously fine-tuning LLMs is prohibitively expensive. We present a training-free approach for optimizing generative recommenders by connecting user feedback loops to LLM-based optimizers. We propose a generative explore-exploit method that can not only exploit generated items with known high engagement, but also actively explore and discover hidden population preferences to improve recommendation quality. We evaluate our approach on question generation in two domains (e-commerce and general knowledge), and model user feedback with Click Through Rate (CTR). Experiments show our LLM-based explore-exploit approach can iteratively improve recommendations, and consistently increase CTR. Ablation analysis shows that generative exploration is key to learning user preferences, avoiding the pitfalls of greedy exploit-only approaches. A human evaluation strongly supports our quantitative findings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.10110v2">Selective Reflection-Tuning: Student-Selected Data Recycling for LLM Instruction-Tuning</a></div>
    <div class="paper-meta">
      📅 2024-06-07
      | 💬 ACL2024 (findings), Camera-ready
    </div>
    <details class="paper-abstract">
      Instruction tuning is critical to large language models (LLMs) for achieving better instruction following and task adaptation capabilities but its success heavily relies on the training data quality. Many recent methods focus on improving the data quality but often overlook the compatibility of the data with the student model being finetuned. This paper introduces Selective Reflection-Tuning, a novel paradigm that synergizes a teacher LLM's reflection and introspection for improving existing data quality with the data selection capability of the student LLM, to automatically refine existing instruction-tuning data. This teacher-student collaboration produces high-quality and student-compatible instruction-response pairs, resulting in sample-efficient instruction tuning and LLMs of superior performance. Selective Reflection-Tuning is a data augmentation and synthesis that generally improves LLM finetuning and self-improvement without collecting brand-new data. We apply our method to Alpaca and WizardLM data and achieve much stronger and top-tier 7B and 13B LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.10614v2">Can LLMs Speak For Diverse People? Tuning LLMs via Debate to Generate Controllable Controversial Statements</a></div>
    <div class="paper-meta">
      📅 2024-06-07
      | 💬 ACL2024 findings, Camera-ready
    </div>
    <details class="paper-abstract">
      Making LLMs speak for different, especially minority groups of people, and generate statements supporting their diverse or even controversial perspectives is critical to creating an inclusive environment. However, existing LLMs lack sufficient controllability to the stance of their generated content, which often contains inconsistent, neutral, or biased statements. In this paper, we improve the controllability of LLMs in generating statements supporting an argument the user defined in the prompt. We find that multi-round debates between two LLMs with opposite stances generate higher-quality and more salient statements for each, which are important training data to improve the controllability of LLMs. Motivated by this, we develop a novel debate & tuning (DEBATUNE) pipeline finetuning LLMs to generate the statements obtained via debate. To examine DEBATUNE, we curate the largest dataset of debate topics so far, which covers 710 controversial topics and corresponding arguments for each topic. Evaluations by the GPT-4 judge with a novel controversy controllability metric show that LLMs' capability of generating diverse perspectives is significantly improved by DEBATUNE. Moreover, such controllability can be generalized to unseen topics, generating high-quality statements supporting controversial arguments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.05194v1">LLMs Are Not Intelligent Thinkers: Introducing Mathematical Topic Tree Benchmark for Comprehensive Evaluation of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-06-07
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate impressive capabilities in mathematical reasoning. However, despite these achievements, current evaluations are mostly limited to specific mathematical topics, and it remains unclear whether LLMs are genuinely engaging in reasoning. To address these gaps, we present the Mathematical Topics Tree (MaTT) benchmark, a challenging and structured benchmark that offers 1,958 questions across a wide array of mathematical subjects, each paired with a detailed hierarchical chain of topics. Upon assessing different LLMs using the MaTT benchmark, we find that the most advanced model, GPT-4, achieved a mere 54\% accuracy in a multiple-choice scenario. Interestingly, even when employing Chain-of-Thought prompting, we observe mostly no notable improvement. Moreover, LLMs accuracy dramatically reduced by up to 24.2 percentage point when the questions were presented without providing choices. Further detailed analysis of the LLMs' performance across a range of topics showed significant discrepancy even for closely related subtopics within the same general mathematical area. In an effort to pinpoint the reasons behind LLMs performances, we conducted a manual evaluation of the completeness and correctness of the explanations generated by GPT-4 when choices were available. Surprisingly, we find that in only 53.3\% of the instances where the model provided a correct answer, the accompanying explanations were deemed complete and accurate, i.e., the model engaged in genuine reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.16363v2">LLMs for User Interest Exploration in Large-scale Recommendation Systems</a></div>
    <div class="paper-meta">
      📅 2024-06-07
    </div>
    <details class="paper-abstract">
      Traditional recommendation systems are subject to a strong feedback loop by learning from and reinforcing past user-item interactions, which in turn limits the discovery of novel user interests. To address this, we introduce a hybrid hierarchical framework combining Large Language Models (LLMs) and classic recommendation models for user interest exploration. The framework controls the interfacing between the LLMs and the classic recommendation models through "interest clusters", the granularity of which can be explicitly determined by algorithm designers. It recommends the next novel interests by first representing "interest clusters" using language, and employs a fine-tuned LLM to generate novel interest descriptions that are strictly within these predefined clusters. At the low level, it grounds these generated interests to an item-level policy by restricting classic recommendation models, in this case a transformer-based sequence recommender to return items that fall within the novel clusters generated at the high level. We showcase the efficacy of this approach on an industrial-scale commercial platform serving billions of users. Live experiments show a significant increase in both exploration of novel interests and overall user enjoyment of the platform.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.02524v2">CheckEmbed: Effective Verification of LLM Solutions to Open-Ended Tasks</a></div>
    <div class="paper-meta">
      📅 2024-06-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are revolutionizing various domains, yet verifying their answers remains a significant challenge, especially for intricate open-ended tasks such as consolidation, summarization, and extraction of knowledge. In this work, we propose CheckEmbed: an accurate, scalable, and simple LLM verification approach. CheckEmbed is driven by a straightforward yet powerful idea: in order to compare LLM solutions to one another or to the ground-truth, compare their corresponding answer-level embeddings obtained with a model such as GPT Text Embedding Large. This reduces a complex textual answer to a single embedding, facilitating straightforward, fast, and meaningful verification. We develop a comprehensive verification pipeline implementing the CheckEmbed methodology. The CheckEmbed pipeline also comes with metrics for assessing the truthfulness of the LLM answers, such as embedding heatmaps and their summaries. We show how to use these metrics for deploying practical engines that decide whether an LLM answer is satisfactory or not. We apply the pipeline to real-world document analysis tasks, including term extraction and document summarization, showcasing significant improvements in accuracy, cost-effectiveness, and runtime performance compared to existing token-, sentence-, and fact-level schemes such as BERTScore or SelfCheckGPT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.11753v4">ArtPrompt: ASCII Art-based Jailbreak Attacks against Aligned LLMs</a></div>
    <div class="paper-meta">
      📅 2024-06-07
      | 💬 To appear in ACL 2024
    </div>
    <details class="paper-abstract">
      Safety is critical to the usage of large language models (LLMs). Multiple techniques such as data filtering and supervised fine-tuning have been developed to strengthen LLM safety. However, currently known techniques presume that corpora used for safety alignment of LLMs are solely interpreted by semantics. This assumption, however, does not hold in real-world applications, which leads to severe vulnerabilities in LLMs. For example, users of forums often use ASCII art, a form of text-based art, to convey image information. In this paper, we propose a novel ASCII art-based jailbreak attack and introduce a comprehensive benchmark Vision-in-Text Challenge (ViTC) to evaluate the capabilities of LLMs in recognizing prompts that cannot be solely interpreted by semantics. We show that five SOTA LLMs (GPT-3.5, GPT-4, Gemini, Claude, and Llama2) struggle to recognize prompts provided in the form of ASCII art. Based on this observation, we develop the jailbreak attack ArtPrompt, which leverages the poor performance of LLMs in recognizing ASCII art to bypass safety measures and elicit undesired behaviors from LLMs. ArtPrompt only requires black-box access to the victim LLMs, making it a practical attack. We evaluate ArtPrompt on five SOTA LLMs, and show that ArtPrompt can effectively and efficiently induce undesired behaviors from all five LLMs. Our code is available at https://github.com/uw-nsl/ArtPrompt.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.20657v2">DORY: Deliberative Prompt Recovery for LLM</a></div>
    <div class="paper-meta">
      📅 2024-06-07
      | 💬 Findings of ACL 2024
    </div>
    <details class="paper-abstract">
      Prompt recovery in large language models (LLMs) is crucial for understanding how LLMs work and addressing concerns regarding privacy, copyright, etc. The trend towards inference-only APIs complicates this task by restricting access to essential outputs for recovery. To tackle this challenge, we extract prompt-related information from limited outputs and identify a strong(negative) correlation between output probability-based uncertainty and the success of prompt recovery. This finding led to the development of Deliberative PrOmpt RecoverY (DORY), our novel approach that leverages uncertainty to recover prompts accurately. DORY involves reconstructing drafts from outputs, refining these with hints, and filtering out noise based on uncertainty. Our evaluation across diverse LLMs and prompt benchmarks shows that DORY outperforms existing baselines, improving performance by approximately 10.82% and establishing a new state-of-the-art record in prompt recovery tasks. Significantly, DORY operates using a single LLM without any external resources or model, offering a cost-effective, user-friendly prompt recovery solution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.06622v1">Adversarial Tuning: Defending Against Jailbreak Attacks for LLMs</a></div>
    <div class="paper-meta">
      📅 2024-06-07
    </div>
    <details class="paper-abstract">
      Although safely enhanced Large Language Models (LLMs) have achieved remarkable success in tackling various complex tasks in a zero-shot manner, they remain susceptible to jailbreak attacks, particularly the unknown jailbreak attack. To enhance LLMs' generalized defense capabilities, we propose a two-stage adversarial tuning framework, which generates adversarial prompts to explore worst-case scenarios by optimizing datasets containing pairs of adversarial prompts and their safe responses. In the first stage, we introduce the hierarchical meta-universal adversarial prompt learning to efficiently and effectively generate token-level adversarial prompts. In the second stage, we propose the automatic adversarial prompt learning to iteratively refine semantic-level adversarial prompts, further enhancing LLM's defense capabilities. We conducted comprehensive experiments on three widely used jailbreak datasets, comparing our framework with six defense baselines under five representative attack scenarios. The results underscore the superiority of our proposed methods. Furthermore, our adversarial tuning framework exhibits empirical generalizability across various attack strategies and target LLMs, highlighting its potential as a transferable defense mechanism.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.06621v1">LinkQ: An LLM-Assisted Visual Interface for Knowledge Graph Question-Answering</a></div>
    <div class="paper-meta">
      📅 2024-06-07
    </div>
    <details class="paper-abstract">
      We present LinkQ, a system that leverages a large language model (LLM) to facilitate knowledge graph (KG) query construction through natural language question-answering. Traditional approaches often require detailed knowledge of complex graph querying languages, limiting the ability for users -- even experts -- to acquire valuable insights from KG data. LinkQ simplifies this process by first interpreting a user's question, then converting it into a well-formed KG query. By using the LLM to construct a query instead of directly answering the user's question, LinkQ guards against the LLM hallucinating or generating false, erroneous information. By integrating an LLM into LinkQ, users are able to conduct both exploratory and confirmatory data analysis, with the LLM helping to iteratively refine open-ended questions into precise ones. To demonstrate the efficacy of LinkQ, we conducted a qualitative study with five KG practitioners and distill their feedback. Our results indicate that practitioners find LinkQ effective for KG question-answering, and desire future LLM-assisted systems for the exploratory analysis of graph databases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.04926v1">Through the Thicket: A Study of Number-Oriented LLMs derived from Random Forest Models</a></div>
    <div class="paper-meta">
      📅 2024-06-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown exceptional performance in text processing. Notably, LLMs can synthesize information from large datasets and explain their decisions similarly to human reasoning through a chain of thought (CoT). An emerging application of LLMs is the handling and interpreting of numerical data, where fine-tuning enhances their performance over basic inference methods. This paper proposes a novel approach to training LLMs using knowledge transfer from a random forest (RF) ensemble, leveraging its efficiency and accuracy. By converting RF decision paths into natural language statements, we generate outputs for LLM fine-tuning, enhancing the model's ability to classify and explain its decisions. Our method includes verifying these rules through established classification metrics, ensuring their correctness. We also examine the impact of preprocessing techniques on the representation of numerical data and their influence on classification accuracy and rule correctness
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.04693v1">LLM-Vectorizer: LLM-based Verified Loop Vectorizer</a></div>
    <div class="paper-meta">
      📅 2024-06-07
    </div>
    <details class="paper-abstract">
      Vectorization is a powerful optimization technique that significantly boosts the performance of high performance computing applications operating on large data arrays. Despite decades of research on auto-vectorization, compilers frequently miss opportunities to vectorize code. On the other hand, writing vectorized code manually using compiler intrinsics is still a complex, error-prone task that demands deep knowledge of specific architecture and compilers. In this paper, we evaluate the potential of large-language models (LLMs) to generate vectorized (Single Instruction Multiple Data) code from scalar programs that process individual array elements. We propose a novel finite-state machine multi-agents based approach that harnesses LLMs and test-based feedback to generate vectorized code. Our findings indicate that LLMs are capable of producing high performance vectorized code with run-time speedup ranging from 1.1x to 9.4x as compared to the state-of-the-art compilers such as Intel Compiler, GCC, and Clang. To verify the correctness of vectorized code, we use Alive2, a leading bounded translation validation tool for LLVM IR. We describe a few domain-specific techniques to improve the scalability of Alive2 on our benchmark dataset. Overall, our approach is able to verify 38.2% of vectorizations as correct on the TSVC benchmark dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.04687v1">LogiCode: an LLM-Driven Framework for Logical Anomaly Detection</a></div>
    <div class="paper-meta">
      📅 2024-06-07
    </div>
    <details class="paper-abstract">
      This paper presents LogiCode, a novel framework that leverages Large Language Models (LLMs) for identifying logical anomalies in industrial settings, moving beyond traditional focus on structural inconsistencies. By harnessing LLMs for logical reasoning, LogiCode autonomously generates Python codes to pinpoint anomalies such as incorrect component quantities or missing elements, marking a significant leap forward in anomaly detection technologies. A custom dataset "LOCO-Annotations" and a benchmark "LogiBench" are introduced to evaluate the LogiCode's performance across various metrics including binary classification accuracy, code generation success rate, and precision in reasoning. Findings demonstrate LogiCode's enhanced interpretability, significantly improving the accuracy of logical anomaly detection and offering detailed explanations for identified anomalies. This represents a notable shift towards more intelligent, LLM-driven approaches in industrial anomaly detection, promising substantial impacts on industry-specific applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.04799v3">Chat Vector: A Simple Approach to Equip LLMs with Instruction Following and Model Alignment in New Languages</a></div>
    <div class="paper-meta">
      📅 2024-06-07
      | 💬 ACL 2024 camera-ready version
    </div>
    <details class="paper-abstract">
      Recently, the development of open-source large language models (LLMs) has advanced rapidly. Nevertheless, due to data constraints, the capabilities of most open-source LLMs are primarily focused on English. To address this issue, we introduce the concept of $\textit{chat vector}$ to equip pre-trained language models with instruction following and human value alignment via simple model arithmetic. The chat vector is derived by subtracting the weights of a pre-trained base model (e.g. LLaMA2) from those of its corresponding chat model (e.g. LLaMA2-chat). By simply adding the chat vector to a continual pre-trained model's weights, we can endow the model with chat capabilities in new languages without the need for further training. Our empirical studies demonstrate the superior efficacy of the chat vector from three different aspects: instruction following, toxicity mitigation, and multi-turn dialogue. Moreover, to showcase the adaptability of our approach, we extend our experiments to encompass various languages, base models, and chat vectors. The results underscore the chat vector's simplicity, effectiveness, and wide applicability, making it a compelling solution for efficiently enabling conversational capabilities in pre-trained language models. Our code is available at https://github.com/aqweteddy/ChatVector.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.20046v2">Can LLMs Learn from Previous Mistakes? Investigating LLMs' Errors to Boost for Reasoning</a></div>
    <div class="paper-meta">
      📅 2024-06-07
      | 💬 The 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024) - Main Conference
    </div>
    <details class="paper-abstract">
      Recent works have shown the benefits to LLMs from fine-tuning golden-standard Chain-of-Thought (CoT) rationales or using them as correct examples in few-shot prompting. While humans can indeed imitate correct examples, learning from our mistakes is another vital aspect of human cognition. Hence, a question naturally arises: \textit{can LLMs learn and benefit from their mistakes, especially for their reasoning? } This study investigates this problem from both the prompting and model-tuning perspectives. We begin by introducing \textsc{CoTErrorSet}, a new benchmark with 609,432 questions, each designed with both correct and error references, and demonstrating the types and reasons for making such mistakes. To explore the effectiveness of those mistakes, we design two methods: (1) \textbf{Self-rethinking} prompting guides LLMs to rethink whether they have made similar previous mistakes; and (2) \textbf{Mistake tuning} involves finetuning models in both correct and incorrect reasoning domains, rather than only tuning models to learn ground truth in traditional methodology. We conduct a series of experiments to prove LLMs can obtain benefits from mistakes in both directions. Our two methods offer potentially cost-effective strategies by leveraging errors to enhance reasoning capabilities, which costs significantly less than creating meticulously hand-crafted golden references. We ultimately make a thorough analysis of the reasons behind LLMs' errors, which provides directions that future research needs to overcome. \textsc{CoTErrorSet} will be published soon on \texttt{\url{https://github.com/YookiTong/Learn-from-Mistakes-CotErrorSet}}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.08588v3">CodeScope: An Execution-based Multilingual Multitask Multidimensional Benchmark for Evaluating LLMs on Code Understanding and Generation</a></div>
    <div class="paper-meta">
      📅 2024-06-07
      | 💬 Accepted by ACL 2024 main conference
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable performance on assisting humans in programming and facilitating programming automation. However, existing benchmarks for evaluating the code understanding and generation capacities of LLMs suffer from severe limitations. First, most benchmarks are insufficient as they focus on a narrow range of popular programming languages and specific tasks, whereas real-world software development scenarios show a critical need to implement systems with multilingual and multitask programming environments to satisfy diverse requirements. Second, most benchmarks fail to consider the actual executability and the consistency of execution results of the generated code. To bridge these gaps between existing benchmarks and expectations from practical applications, we introduce CodeScope, an execution-based, multilingual, multitask, multidimensional evaluation benchmark for comprehensively measuring LLM capabilities on coding tasks. CodeScope covers 43 programming languages and eight coding tasks. It evaluates the coding performance of LLMs from three dimensions (perspectives): length, difficulty, and efficiency. To facilitate execution-based evaluations of code generation, we develop MultiCodeEngine, an automated code execution engine that supports 14 programming languages. Finally, we systematically evaluate and analyze eight mainstream LLMs and demonstrate the superior breadth and challenges of CodeScope for evaluating LLMs on code understanding and generation tasks compared to other benchmarks. The CodeScope benchmark and code are publicly available at https://github.com/WeixiangYAN/CodeScope.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.01030v4">Executable Code Actions Elicit Better LLM Agents</a></div>
    <div class="paper-meta">
      📅 2024-06-07
      | 💬 Accepted by ICML 2024; Code, data, model, and demo are available at https://github.com/xingyaoww/code-act
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents, capable of performing a broad range of actions, such as invoking tools and controlling robots, show great potential in tackling real-world challenges. LLM agents are typically prompted to produce actions by generating JSON or text in a pre-defined format, which is usually limited by constrained action space (e.g., the scope of pre-defined tools) and restricted flexibility (e.g., inability to compose multiple tools). This work proposes to use executable Python code to consolidate LLM agents' actions into a unified action space (CodeAct). Integrated with a Python interpreter, CodeAct can execute code actions and dynamically revise prior actions or emit new actions upon new observations through multi-turn interactions. Our extensive analysis of 17 LLMs on API-Bank and a newly curated benchmark shows that CodeAct outperforms widely used alternatives (up to 20% higher success rate). The encouraging performance of CodeAct motivates us to build an open-source LLM agent that interacts with environments by executing interpretable code and collaborates with users using natural language. To this end, we collect an instruction-tuning dataset CodeActInstruct that consists of 7k multi-turn interactions using CodeAct. We show that it can be used with existing data to improve models in agent-oriented tasks without compromising their general capability. CodeActAgent, finetuned from Llama2 and Mistral, is integrated with Python interpreter and uniquely tailored to perform sophisticated tasks (e.g., model training) using existing libraries and autonomously self-debug.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.08679v2">COLD-Attack: Jailbreaking LLMs with Stealthiness and Controllability</a></div>
    <div class="paper-meta">
      📅 2024-06-07
      | 💬 Accepted to ICML 2024
    </div>
    <details class="paper-abstract">
      Jailbreaks on large language models (LLMs) have recently received increasing attention. For a comprehensive assessment of LLM safety, it is essential to consider jailbreaks with diverse attributes, such as contextual coherence and sentiment/stylistic variations, and hence it is beneficial to study controllable jailbreaking, i.e. how to enforce control on LLM attacks. In this paper, we formally formulate the controllable attack generation problem, and build a novel connection between this problem and controllable text generation, a well-explored topic of natural language processing. Based on this connection, we adapt the Energy-based Constrained Decoding with Langevin Dynamics (COLD), a state-of-the-art, highly efficient algorithm in controllable text generation, and introduce the COLD-Attack framework which unifies and automates the search of adversarial LLM attacks under a variety of control requirements such as fluency, stealthiness, sentiment, and left-right-coherence. The controllability enabled by COLD-Attack leads to diverse new jailbreak scenarios which not only cover the standard setting of generating fluent (suffix) attack with continuation constraint, but also allow us to address new controllable attack settings such as revising a user query adversarially with paraphrasing constraint, and inserting stealthy attacks in context with position constraint. Our extensive experiments on various LLMs (Llama-2, Mistral, Vicuna, Guanaco, GPT-3.5, and GPT-4) show COLD-Attack's broad applicability, strong controllability, high success rate, and attack transferability. Our code is available at https://github.com/Yu-Fangxu/COLD-Attack.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.08374v2">A Ship of Theseus: Curious Cases of Paraphrasing in LLM-Generated Texts</a></div>
    <div class="paper-meta">
      📅 2024-06-06
      | 💬 To appear in Association for Computational Linguistics (ACL 2024)
    </div>
    <details class="paper-abstract">
      In the realm of text manipulation and linguistic transformation, the question of authorship has been a subject of fascination and philosophical inquiry. Much like the Ship of Theseus paradox, which ponders whether a ship remains the same when each of its original planks is replaced, our research delves into an intriguing question: Does a text retain its original authorship when it undergoes numerous paraphrasing iterations? Specifically, since Large Language Models (LLMs) have demonstrated remarkable proficiency in both the generation of original content and the modification of human-authored texts, a pivotal question emerges concerning the determination of authorship in instances where LLMs or similar paraphrasing tools are employed to rephrase the text--i.e., whether authorship should be attributed to the original human author or the AI-powered tool. Therefore, we embark on a philosophical voyage through the seas of language and authorship to unravel this intricate puzzle. Using a computational approach, we discover that the diminishing performance in text classification models, with each successive paraphrasing iteration, is closely associated with the extent of deviation from the original author's style, thus provoking a reconsideration of the current notion of authorship.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.00606v2">LLMs Could Autonomously Learn Without External Supervision</a></div>
    <div class="paper-meta">
      📅 2024-06-06
      | 💬 20 pages, 8 figures
    </div>
    <details class="paper-abstract">
      In the quest for super-human performance, Large Language Models (LLMs) have traditionally been tethered to human-annotated datasets and predefined training objectives-a process that is both labor-intensive and inherently limited. This paper presents a transformative approach: Autonomous Learning for LLMs, a self-sufficient learning paradigm that frees models from the constraints of human supervision. This method endows LLMs with the ability to self-educate through direct interaction with text, akin to a human reading and comprehending literature. Our approach eliminates the reliance on annotated data, fostering an Autonomous Learning environment where the model independently identifies and reinforces its knowledge gaps. Empirical results from our comprehensive experiments, which utilized a diverse array of learning materials and were evaluated against standard public quizzes, reveal that Autonomous Learning outstrips the performance of both Pre-training and Supervised Fine-Tuning (SFT), as well as retrieval-augmented methods. These findings underscore the potential of Autonomous Learning to not only enhance the efficiency and effectiveness of LLM training but also to pave the way for the development of more advanced, self-reliant AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.09204v3">Fusion-Eval: Integrating Assistant Evaluators with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-06-06
    </div>
    <details class="paper-abstract">
      Evaluating natural language systems poses significant challenges, particularly in the realms of natural language understanding and high-level reasoning. In this paper, we introduce 'Fusion-Eval', an innovative approach that leverages Large Language Models (LLMs) to integrate insights from various assistant evaluators. The LLM is given the example to evaluate along with scores from the assistant evaluators. Each of these evaluators specializes in assessing distinct aspects of responses. Fusion-Eval achieves a 0.962 system-level Kendall-Tau correlation with humans on SummEval and a 0.744 turn-level Spearman correlation on TopicalChat, which is significantly higher than baseline methods. These results highlight Fusion-Eval's significant potential in the realm of natural language system evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.15018v2">Unintended Impacts of LLM Alignment on Global Representation</a></div>
    <div class="paper-meta">
      📅 2024-06-06
      | 💬 Accepted to ACL 2024 main conference
    </div>
    <details class="paper-abstract">
      Before being deployed for user-facing applications, developers align Large Language Models (LLMs) to user preferences through a variety of procedures, such as Reinforcement Learning From Human Feedback (RLHF) and Direct Preference Optimization (DPO). Current evaluations of these procedures focus on benchmarks of instruction following, reasoning, and truthfulness. However, human preferences are not universal, and aligning to specific preference sets may have unintended effects. We explore how alignment impacts performance along three axes of global representation: English dialects, multilingualism, and opinions from and about countries worldwide. Our results show that current alignment procedures create disparities between English dialects and global opinions. We find alignment improves capabilities in several languages. We conclude by discussing design decisions that led to these unintended impacts and recommendations for more equitable preference tuning. We make our code and data publicly available on Github.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.04520v1">NATURAL PLAN: Benchmarking LLMs on Natural Language Planning</a></div>
    <div class="paper-meta">
      📅 2024-06-06
    </div>
    <details class="paper-abstract">
      We introduce NATURAL PLAN, a realistic planning benchmark in natural language containing 3 key tasks: Trip Planning, Meeting Planning, and Calendar Scheduling. We focus our evaluation on the planning capabilities of LLMs with full information on the task, by providing outputs from tools such as Google Flights, Google Maps, and Google Calendar as contexts to the models. This eliminates the need for a tool-use environment for evaluating LLMs on Planning. We observe that NATURAL PLAN is a challenging benchmark for state of the art models. For example, in Trip Planning, GPT-4 and Gemini 1.5 Pro could only achieve 31.1% and 34.8% solve rate respectively. We find that model performance drops drastically as the complexity of the problem increases: all models perform below 5% when there are 10 cities, highlighting a significant gap in planning in natural language for SoTA LLMs. We also conduct extensive ablation studies on NATURAL PLAN to further shed light on the (in)effectiveness of approaches such as self-correction, few-shot generalization, and in-context planning with long-contexts on improving LLM planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.04482v1">Automatic Bug Detection in LLM-Powered Text-Based Games Using LLMs</a></div>
    <div class="paper-meta">
      📅 2024-06-06
      | 💬 Accepted for publication in Findings of the Association for Computational Linguistics: ACL 2024
    </div>
    <details class="paper-abstract">
      Advancements in large language models (LLMs) are revolutionizing interactive game design, enabling dynamic plotlines and interactions between players and non-player characters (NPCs). However, LLMs may exhibit flaws such as hallucinations, forgetfulness, or misinterpretations of prompts, causing logical inconsistencies and unexpected deviations from intended designs. Automated techniques for detecting such game bugs are still lacking. To address this, we propose a systematic LLM-based method for automatically identifying such bugs from player game logs, eliminating the need for collecting additional data such as post-play surveys. Applied to a text-based game DejaBoom!, our approach effectively identifies bugs inherent in LLM-powered interactive games, surpassing unstructured LLM-powered bug-catching methods and filling the gap in automated detection of logical and design flaws.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.04481v1">Optimizing Autonomous Driving for Safety: A Human-Centric Approach with LLM-Enhanced RLHF</a></div>
    <div class="paper-meta">
      📅 2024-06-06
    </div>
    <details class="paper-abstract">
      Reinforcement Learning from Human Feedback (RLHF) is popular in large language models (LLMs), whereas traditional Reinforcement Learning (RL) often falls short. Current autonomous driving methods typically utilize either human feedback in machine learning, including RL, or LLMs. Most feedback guides the car agent's learning process (e.g., controlling the car). RLHF is usually applied in the fine-tuning step, requiring direct human "preferences," which are not commonly used in optimizing autonomous driving models. In this research, we innovatively combine RLHF and LLMs to enhance autonomous driving safety. Training a model with human guidance from scratch is inefficient. Our framework starts with a pre-trained autonomous car agent model and implements multiple human-controlled agents, such as cars and pedestrians, to simulate real-life road environments. The autonomous car model is not directly controlled by humans. We integrate both physical and physiological feedback to fine-tune the model, optimizing this process using LLMs. This multi-agent interactive environment ensures safe, realistic interactions before real-world application. Finally, we will validate our model using data gathered from real-life testbeds located in New Jersey and New York City.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.04460v1">Evaluating the Smooth Control of Attribute Intensity in Text Generation with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-06-06
      | 💬 Accepted to ACL 2024 Findings
    </div>
    <details class="paper-abstract">
      Controlling the attribute intensity of text generation is crucial across scenarios (e.g., writing conciseness, chatting emotion, and explanation clarity). The remarkable capabilities of large language models (LLMs) have revolutionized text generation, prompting us to explore such \emph{smooth control} of LLM generation. Specifically, we propose metrics to assess the range, calibration, and consistency of the generated text's attribute intensity in response to varying control values, as well as its relevance to the intended context. To quantify the attribute intensity and context relevance, we propose an effective evaluation framework leveraging the Elo rating system and GPT4, both renowned for their robust alignment with human judgment. We look into two viable training-free methods for achieving smooth control of LLMs: (1) Prompting with semantic shifters, and (2) Modifying internal model representations. The evaluations of these two methods are conducted on $5$ different attributes with various models. Our code and dataset can be obtained from \url{https://github.com/ShangDataLab/Smooth-Control}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.04452v1">Revisiting Human Information Foraging: Adaptations for LLM-based Chatbots</a></div>
    <div class="paper-meta">
      📅 2024-06-06
    </div>
    <details class="paper-abstract">
      Information Foraging Theory's (IFT) framing of human information seeking choices as decision-theoretic cost-value judgments has successfully explained how people seek information among linked patches of information (e.g., linked webpages). However, the theory has to be adopted and validated in non-patchy LLM-based chatbot environments, before its postulates can be reliably applied to the design of such chat-based information seeking environments. This paper is a thought experiment that applies the IFT cost-value proposition to LLM-based chatbots and presents a set of preliminary hypotheses to guide future theory-building efforts for how people seek information in such environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.16459v3">Defending LLMs against Jailbreaking Attacks via Backtranslation</a></div>
    <div class="paper-meta">
      📅 2024-06-06
    </div>
    <details class="paper-abstract">
      Although many large language models (LLMs) have been trained to refuse harmful requests, they are still vulnerable to jailbreaking attacks which rewrite the original prompt to conceal its harmful intent. In this paper, we propose a new method for defending LLMs against jailbreaking attacks by ``backtranslation''. Specifically, given an initial response generated by the target LLM from an input prompt, our backtranslation prompts a language model to infer an input prompt that can lead to the response. The inferred prompt is called the backtranslated prompt which tends to reveal the actual intent of the original prompt, since it is generated based on the LLM's response and not directly manipulated by the attacker. We then run the target LLM again on the backtranslated prompt, and we refuse the original prompt if the model refuses the backtranslated prompt. We explain that the proposed defense provides several benefits on its effectiveness and efficiency. We empirically demonstrate that our defense significantly outperforms the baselines, in the cases that are hard for the baselines, and our defense also has little impact on the generation quality for benign input prompts. Our implementation is based on our library for LLM jailbreaking defense algorithms at \url{https://github.com/YihanWang617/llm-jailbreaking-defense}, and the code for reproducing our experiments is available at \url{https://github.com/YihanWang617/LLM-Jailbreaking-Defense-Backtranslation}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.10412v2">Measuring and Reducing LLM Hallucination without Gold-Standard Answers</a></div>
    <div class="paper-meta">
      📅 2024-06-06
      | 💬 Paper Under Review
    </div>
    <details class="paper-abstract">
      LLM hallucination, i.e. generating factually incorrect yet seemingly convincing answers, is currently a major threat to the trustworthiness and reliability of LLMs. The first step towards solving this complicated problem is to measure it. However, existing hallucination metrics require having a benchmark dataset with gold-standard answers, i.e. "best" or "correct" answers written by humans. Such requirements make hallucination measurement costly and prone to human errors. In this work, we propose Factualness Evaluations via Weighting LLMs (FEWL), an innovative hallucination metric that is specifically designed for the scenario when gold-standard answers are absent. FEWL leverages the answers from off-the-shelf LLMs that serve as a proxy of gold-standard answers. The key challenge is how to quantify the expertise of reference LLMs resourcefully. We show FEWL has certain theoretical guarantees and demonstrate empirically it gives more accurate hallucination measures than naively using reference LLMs. We also show how to leverage FEWL to reduce hallucination through both in-context learning and supervised fine-tuning. Extensive experiment results on Truthful-QA, CHALE, and HaluEval datasets demonstrate the effectiveness of FEWL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.04428v1">MoralBench: Moral Evaluation of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-06-06
    </div>
    <details class="paper-abstract">
      In the rapidly evolving field of artificial intelligence, large language models (LLMs) have emerged as powerful tools for a myriad of applications, from natural language processing to decision-making support systems. However, as these models become increasingly integrated into societal frameworks, the imperative to ensure they operate within ethical and moral boundaries has never been more critical. This paper introduces a novel benchmark designed to measure and compare the moral reasoning capabilities of LLMs. We present the first comprehensive dataset specifically curated to probe the moral dimensions of LLM outputs, addressing a wide range of ethical dilemmas and scenarios reflective of real-world complexities. The main contribution of this work lies in the development of benchmark datasets and metrics for assessing the moral identity of LLMs, which accounts for nuance, contextual sensitivity, and alignment with human ethical standards. Our methodology involves a multi-faceted approach, combining quantitative analysis with qualitative insights from ethics scholars to ensure a thorough evaluation of model performance. By applying our benchmark across several leading LLMs, we uncover significant variations in moral reasoning capabilities of different models. These findings highlight the importance of considering moral reasoning in the development and evaluation of LLMs, as well as the need for ongoing research to address the biases and limitations uncovered in our study. We publicly release the benchmark at https://drive.google.com/drive/u/0/folders/1k93YZJserYc2CkqP8d4B3M3sgd3kA8W7 and also open-source the code of the project at https://github.com/agiresearch/MoralBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.04278v1">Characterizing Similarities and Divergences in Conversational Tones in Humans and LLMs by Sampling with People</a></div>
    <div class="paper-meta">
      📅 2024-06-06
      | 💬 Accepted to Main Conference at ACL 2024
    </div>
    <details class="paper-abstract">
      Conversational tones -- the manners and attitudes in which speakers communicate -- are essential to effective communication. Amidst the increasing popularization of Large Language Models (LLMs) over recent years, it becomes necessary to characterize the divergences in their conversational tones relative to humans. However, existing investigations of conversational modalities rely on pre-existing taxonomies or text corpora, which suffer from experimenter bias and may not be representative of real-world distributions for the studies' psycholinguistic domains. Inspired by methods from cognitive science, we propose an iterative method for simultaneously eliciting conversational tones and sentences, where participants alternate between two tasks: (1) one participant identifies the tone of a given sentence and (2) a different participant generates a sentence based on that tone. We run 100 iterations of this process with human participants and GPT-4, then obtain a dataset of sentences and frequent conversational tones. In an additional experiment, humans and GPT-4 annotated all sentences with all tones. With data from 1,339 human participants, 33,370 human judgments, and 29,900 GPT-4 queries, we show how our approach can be used to create an interpretable geometric representation of relations between conversational tones in humans and GPT-4. This work demonstrates how combining ideas from machine learning and cognitive science can address challenges in human-computer interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.04276v1">Generative AI-in-the-loop: Integrating LLMs and GPTs into the Next Generation Networks</a></div>
    <div class="paper-meta">
      📅 2024-06-06
    </div>
    <details class="paper-abstract">
      In recent years, machine learning (ML) techniques have created numerous opportunities for intelligent mobile networks and have accelerated the automation of network operations. However, complex network tasks may involve variables and considerations even beyond the capacity of traditional ML algorithms. On the other hand, large language models (LLMs) have recently emerged, demonstrating near-human-level performance in cognitive tasks across various fields. However, they remain prone to hallucinations and often lack common sense in basic tasks. Therefore, they are regarded as assistive tools for humans. In this work, we propose the concept of "generative AI-in-the-loop" and utilize the semantic understanding, context awareness, and reasoning abilities of LLMs to assist humans in handling complex or unforeseen situations in mobile communication networks. We believe that combining LLMs and ML models allows both to leverage their respective capabilities and achieve better results than either model alone. To support this idea, we begin by analyzing the capabilities of LLMs and compare them with traditional ML algorithms. We then explore potential LLM-based applications in line with the requirements of next-generation networks. We further examine the integration of ML and LLMs, discussing how they can be used together in mobile networks. Unlike existing studies, our research emphasizes the fusion of LLMs with traditional ML-driven next-generation networks and serves as a comprehensive refinement of existing surveys. Finally, we provide a case study to enhance ML-based network intrusion detection with synthesized data generated by LLMs. Our case study further demonstrates the advantages of our proposed idea.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.11361v2">Knowledge Graph Question Answering for Materials Science (KGQA4MAT): Developing Natural Language Interface for Metal-Organic Frameworks Knowledge Graph (MOF-KG) Using LLM</a></div>
    <div class="paper-meta">
      📅 2024-06-06
      | 💬 In 17th International Conference on Metadata and Semantics Research, October 2023
    </div>
    <details class="paper-abstract">
      We present a comprehensive benchmark dataset for Knowledge Graph Question Answering in Materials Science (KGQA4MAT), with a focus on metal-organic frameworks (MOFs). A knowledge graph for metal-organic frameworks (MOF-KG) has been constructed by integrating structured databases and knowledge extracted from the literature. To enhance MOF-KG accessibility for domain experts, we aim to develop a natural language interface for querying the knowledge graph. We have developed a benchmark comprised of 161 complex questions involving comparison, aggregation, and complicated graph structures. Each question is rephrased in three additional variations, resulting in 644 questions and 161 KG queries. To evaluate the benchmark, we have developed a systematic approach for utilizing the LLM, ChatGPT, to translate natural language questions into formal KG queries. We also apply the approach to the well-known QALD-9 dataset, demonstrating ChatGPT's potential in addressing KGQA issues for different platforms and query languages. The benchmark and the proposed approach aim to stimulate further research and development of user-friendly and efficient interfaces for querying domain-specific materials science knowledge graphs, thereby accelerating the discovery of novel materials.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.10890v2">When is Tree Search Useful for LLM Planning? It Depends on the Discriminator</a></div>
    <div class="paper-meta">
      📅 2024-06-06
      | 💬 ACL 2024 main
    </div>
    <details class="paper-abstract">
      In this paper, we examine how large language models (LLMs) solve multi-step problems under a language agent framework with three components: a generator, a discriminator, and a planning method. We investigate the practical utility of two advanced planning methods, iterative correction and tree search. We present a comprehensive analysis of how discrimination accuracy affects the overall performance of agents when using these two methods or a simpler method, re-ranking. Experiments on two tasks, text-to-SQL parsing and mathematical reasoning, show that: (1) advanced planning methods demand discriminators with at least 90% accuracy to achieve significant improvements over re-ranking; (2) current LLMs' discrimination abilities have not met the needs of advanced planning methods to achieve such improvements; (3) with LLM-based discriminators, advanced planning methods may not adequately balance accuracy and efficiency. For example, compared to the other two methods, tree search is at least 10--20 times slower but leads to negligible performance gains, which hinders its real-world applications. Code and data are available at https://github.com/OSU-NLP-Group/llm-planning-eval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.07483v2">T-RAG: Lessons from the LLM Trenches</a></div>
    <div class="paper-meta">
      📅 2024-06-06
      | 💬 Added Needle in a Haystack analysis for T-RAG
    </div>
    <details class="paper-abstract">
      Large Language Models (LLM) have shown remarkable language capabilities fueling attempts to integrate them into applications across a wide range of domains. An important application area is question answering over private enterprise documents where the main considerations are data security, which necessitates applications that can be deployed on-prem, limited computational resources and the need for a robust application that correctly responds to queries. Retrieval-Augmented Generation (RAG) has emerged as the most prominent framework for building LLM-based applications. While building a RAG is relatively straightforward, making it robust and a reliable application requires extensive customization and relatively deep knowledge of the application domain. We share our experiences building and deploying an LLM application for question answering over private organizational documents. Our application combines the use of RAG with a finetuned open-source LLM. Additionally, our system, which we call Tree-RAG (T-RAG), uses a tree structure to represent entity hierarchies within the organization. This is used to generate a textual description to augment the context when responding to user queries pertaining to entities within the organization's hierarchy. Our evaluations, including a Needle in a Haystack test, show that this combination performs better than a simple RAG or finetuning implementation. Finally, we share some lessons learned based on our experiences building an LLM application for real-world use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.14328v2">Understanding and Patching Compositional Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-06-06
      | 💬 Accepted by ACL'2024 Findings
    </div>
    <details class="paper-abstract">
      LLMs have marked a revolutonary shift, yet they falter when faced with compositional reasoning tasks. Our research embarks on a quest to uncover the root causes of compositional reasoning failures of LLMs, uncovering that most of them stem from the improperly generated or leveraged implicit reasoning results. Inspired by our empirical findings, we resort to Logit Lens and an intervention experiment to dissect the inner hidden states of LLMs. This deep dive reveals that implicit reasoning results indeed surface within middle layers and play a causative role in shaping the final explicit reasoning results. Our exploration further locates multi-head self-attention (MHSA) modules within these layers, which emerge as the linchpins in accurate generation and leveraing of implicit reasoning results. Grounded on the above findings, we develop CREME, a lightweight method to patch errors in compositional reasoning via editing the located MHSA modules. Our empirical evidence stands testament to CREME's effectiveness, paving the way for autonomously and continuously enhancing compositional reasoning capabilities in language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.18680v2">Non-Linear Inference Time Intervention: Improving LLM Truthfulness</a></div>
    <div class="paper-meta">
      📅 2024-06-06
      | 💬 Accepted on Interspeech 2024 Conference. Code is available at https://github.com/Samsung/NL-ITI
    </div>
    <details class="paper-abstract">
      In this work, we explore LLM's internal representation space to identify attention heads that contain the most truthful and accurate information. We further developed the Inference Time Intervention (ITI) framework, which lets bias LLM without the need for fine-tuning. The improvement manifests in introducing a non-linear multi-token probing and multi-token intervention: Non-Linear ITI (NL-ITI), which significantly enhances performance on evaluation benchmarks. NL-ITI is tested on diverse multiple-choice datasets, including TruthfulQA, on which we report over 16% relative MC1 (accuracy of model pointing to the correct answer) improvement with respect to the baseline ITI results. Moreover, we achieved a 10% relative improvement over the recently released Truth Forest (TrFf) method that also focused on ITI improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.11517v3">Knowledge-to-SQL: Enhancing SQL Generation with Data Expert LLM</a></div>
    <div class="paper-meta">
      📅 2024-06-06
      | 💬 Accepted to ACL2024 Findings
    </div>
    <details class="paper-abstract">
      Generating accurate SQL queries for user questions (text-to-SQL) has been a long-standing challenge since it requires a deep understanding of both the user's question and the corresponding database schema in order to retrieve the desired content accurately. Existing methods rely on the comprehensive capability of large language models (LLMs) to generate the SQL. However, some necessary knowledge is not explicitly included in the database schema and user question or has been learned by LLMs. Thus, the generated SQL of the knowledge-insufficient questions may be inaccurate, negatively influencing the text-to-SQL models' performance and robustness. To address this challenge, we propose the Knowledge-to-SQL framework, which employs tailored Data Expert LLM (DELLM) to provide helpful knowledge for all text-to-SQL models. Specifically, we introduce the detailed implementation of DELLM regarding table reading and the basic fine-tuning process. We further propose a Preference Learning via Database Feedback (PLDBF) strategy, refining the DELLM to generate more helpful knowledge for LLMs. Extensive experiments verify that DELLM can enhance the state-of-the-art approaches for text-to-SQL tasks. The corresponding code of DELLM is released for further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.04076v1">Federated TrustChain: Blockchain-Enhanced LLM Training and Unlearning</a></div>
    <div class="paper-meta">
      📅 2024-06-06
      | 💬 16 pages, 7 figures,
    </div>
    <details class="paper-abstract">
      The development of Large Language Models (LLMs) faces a significant challenge: the exhausting of publicly available fresh data. This is because training a LLM needs a large demanding of new data. Federated learning emerges as a promising solution, enabling collaborative model to contribute their private data to LLM global model. However, integrating federated learning with LLMs introduces new challenges, including the lack of transparency and the need for effective unlearning mechanisms. Transparency is essential to ensuring trust and fairness among participants, while accountability is crucial for deterring malicious behaviour and enabling corrective actions when necessary. To address these challenges, we propose a novel blockchain-based federated learning framework for LLMs that enhances transparency, accountability, and unlearning capabilities. Our framework leverages blockchain technology to create a tamper-proof record of each model's contributions and introduces an innovative unlearning function that seamlessly integrates with the federated learning mechanism. We investigate the impact of Low-Rank Adaptation (LoRA) hyperparameters on unlearning performance and integrate Hyperledger Fabric to ensure the security, transparency, and verifiability of the unlearning process. Through comprehensive experiments and analysis, we showcase the effectiveness of our proposed framework in achieving highly effective unlearning in LLMs trained using federated learning. Our findings highlight the feasibility of integrating blockchain technology into federated learning frameworks for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.04064v1">Ask LLMs Directly, "What shapes your bias?": Measuring Social Bias in Large Language Models</a></div>
    <div class="paper-meta">
      📅 2024-06-06
      | 💬 Findings of ACL 2024
    </div>
    <details class="paper-abstract">
      Social bias is shaped by the accumulation of social perceptions towards targets across various demographic identities. To fully understand such social bias in large language models (LLMs), it is essential to consider the composite of social perceptions from diverse perspectives among identities. Previous studies have either evaluated biases in LLMs by indirectly assessing the presence of sentiments towards demographic identities in the generated text or measuring the degree of alignment with given stereotypes. These methods have limitations in directly quantifying social biases at the level of distinct perspectives among identities. In this paper, we aim to investigate how social perceptions from various viewpoints contribute to the development of social bias in LLMs. To this end, we propose a novel strategy to intuitively quantify these social perceptions and suggest metrics that can evaluate the social biases within LLMs by aggregating diverse social perceptions. The experimental results show the quantitative demonstration of the social attitude in LLMs by examining social perception. The analysis we conducted shows that our proposed metrics capture the multi-dimensional aspects of social bias, enabling a fine-grained and comprehensive investigation of bias in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.14556v3">Looking Right is Sometimes Right: Investigating the Capabilities of Decoder-only LLMs for Sequence Labeling</a></div>
    <div class="paper-meta">
      📅 2024-06-06
      | 💬 Accepted at ACL 2024 Findings
    </div>
    <details class="paper-abstract">
      Pre-trained language models based on masked language modeling (MLM) excel in natural language understanding (NLU) tasks. While fine-tuned MLM-based encoders consistently outperform causal language modeling decoders of comparable size, recent decoder-only large language models (LLMs) perform on par with smaller MLM-based encoders. Although their performance improves with scale, LLMs fall short of achieving state-of-the-art results in information extraction (IE) tasks, many of which are formulated as sequence labeling (SL). We hypothesize that LLMs' poor SL performance stems from causal masking, which prevents the model from attending to tokens on the right of the current token. Yet, how exactly and to what extent LLMs' performance on SL can be improved remains unclear. We explore techniques for improving the SL performance of open LLMs on IE tasks by applying layer-wise removal of the causal mask (CM) during LLM fine-tuning. This approach yields performance gains competitive with state-of-the-art SL models, matching or outperforming the results of CM removal from all blocks. Our findings hold for diverse SL tasks, demonstrating that open LLMs with layer-dependent CM removal outperform strong MLM-based encoders and even instruction-tuned LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.10186v3">Beyond Traditional Benchmarks: Analyzing Behaviors of Open LLMs on Data-to-Text Generation</a></div>
    <div class="paper-meta">
      📅 2024-06-06
      | 💬 Accepted to ACL 2024 Main Conference
    </div>
    <details class="paper-abstract">
      We analyze the behaviors of open large language models (LLMs) on the task of data-to-text (D2T) generation, i.e., generating coherent and relevant text from structured data. To avoid the issue of LLM training data contamination with standard benchmarks, we design Quintd - a tool for collecting novel structured data records from public APIs. We find that open LLMs (Llama 2, Mistral, and Zephyr) can generate fluent and coherent texts in zero-shot settings from data in common formats collected with Quintd. However, we show that the semantic accuracy of the outputs is a major issue: both according to human annotators and our reference-free metric based on GPT-4, more than 80% of the outputs of open LLMs contain at least one semantic error. We publicly release the code, data, and model outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.03993v1">Assessing LLMs for Zero-shot Abstractive Summarization Through the Lens of Relevance Paraphrasing</a></div>
    <div class="paper-meta">
      📅 2024-06-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved state-of-the-art performance at zero-shot generation of abstractive summaries for given articles. However, little is known about the robustness of such a process of zero-shot summarization. To bridge this gap, we propose relevance paraphrasing, a simple strategy that can be used to measure the robustness of LLMs as summarizers. The relevance paraphrasing approach identifies the most relevant sentences that contribute to generating an ideal summary, and then paraphrases these inputs to obtain a minimally perturbed dataset. Then, by evaluating model performance for summarization on both the original and perturbed datasets, we can assess the LLM's one aspect of robustness. We conduct extensive experiments with relevance paraphrasing on 4 diverse datasets, as well as 4 LLMs of different sizes (GPT-3.5-Turbo, Llama-2-13B, Mistral-7B, and Dolly-v2-7B). Our results indicate that LLMs are not consistent summarizers for the minimally perturbed articles, necessitating further improvements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.03963v1">A + B: A General Generator-Reader Framework for Optimizing LLMs to Unleash Synergy Potential</a></div>
    <div class="paper-meta">
      📅 2024-06-06
      | 💬 Accepted to ACL'24 (Findings)
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) is an effective solution to supplement necessary knowledge to large language models (LLMs). Targeting its bottleneck of retriever performance, "generate-then-read" pipeline is proposed to replace the retrieval stage with generation from the LLM itself. Although promising, this research direction is underexplored and still cannot work in the scenario when source knowledge is given. In this paper, we formalize a general "A + B" framework with varying combinations of foundation models and types for systematic investigation. We explore the efficacy of the base and chat versions of LLMs and found their different functionalities suitable for generator A and reader B, respectively. Their combinations consistently outperform single models, especially in complex scenarios. Furthermore, we extend the application of the "A + B" framework to scenarios involving source documents through continuous learning, enabling the direct integration of external knowledge into LLMs. This approach not only facilitates effective acquisition of new knowledge but also addresses the challenges of safety and helpfulness post-adaptation. The paper underscores the versatility of the "A + B" framework, demonstrating its potential to enhance the practical application of LLMs across various domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.06599v1">Anna Karenina Strikes Again: Pre-Trained LLM Embeddings May Favor High-Performing Learners</a></div>
    <div class="paper-meta">
      📅 2024-06-06
      | 💬 9 pages (not including bibliography), Appendix and 10 tables. Accepted to the 19th Workshop on Innovative Use of NLP for Building Educational Applications, Co-located with NAACL 2024
    </div>
    <details class="paper-abstract">
      Unsupervised clustering of student responses to open-ended questions into behavioral and cognitive profiles using pre-trained LLM embeddings is an emerging technique, but little is known about how well this captures pedagogically meaningful information. We investigate this in the context of student responses to open-ended questions in biology, which were previously analyzed and clustered by experts into theory-driven Knowledge Profiles (KPs). Comparing these KPs to ones discovered by purely data-driven clustering techniques, we report poor discoverability of most KPs, except for the ones including the correct answers. We trace this "discoverability bias" to the representations of KPs in the pre-trained LLM embeddings space.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.03853v1">Speculative Decoding via Early-exiting for Faster LLM Inference with Thompson Sampling Control Mechanism</a></div>
    <div class="paper-meta">
      📅 2024-06-06
      | 💬 Accepted by ACL 2024 (Findings)
    </div>
    <details class="paper-abstract">
      The recent advancements in large language models (LLMs) have been extraordinary, yet the escalating inference costs associated with them present challenges in real-world applications. To address these challenges, we propose a novel approach called Early-exiting Speculative Decoding (EESD) with lossless acceleration. Specifically, EESD utilizes a segment of the LLM to generate draft tokens, incorporating Early-exiting structures after the first N layers. To enhance the quality of draft tokens, a self-distillation method is integrated. This early-exiting design not only reduces deployment and training costs but also significantly accelerates the token generation speed. Moreover, we introduce a novel sampling mechanism that leverages Thompson Sampling to regulate the generation processes, automatically determining the quantity of draft tokens in each round. The original LLM is then employed to validate these draft tokens through a single forward pass, and thus guarantees that the final output text maintains a distribution consistent with vanilla auto-regressive decoding. The experimental results on both 13B and 70B models demonstrate that our approach decodes tokens at a markedly accelerated rate compared to prior methods, showing the effectiveness of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.02409v1">Effective Context Selection in LLM-based Leaderboard Generation: An Empirical Study</a></div>
    <div class="paper-meta">
      📅 2024-06-06
      | 💬 arXiv admin note: substantial text overlap with arXiv:2406.04383
    </div>
    <details class="paper-abstract">
      This paper explores the impact of context selection on the efficiency of Large Language Models (LLMs) in generating Artificial Intelligence (AI) research leaderboards, a task defined as the extraction of (Task, Dataset, Metric, Score) quadruples from scholarly articles. By framing this challenge as a text generation objective and employing instruction finetuning with the FLAN-T5 collection, we introduce a novel method that surpasses traditional Natural Language Inference (NLI) approaches in adapting to new developments without a predefined taxonomy. Through experimentation with three distinct context types of varying selectivity and length, our study demonstrates the importance of effective context selection in enhancing LLM accuracy and reducing hallucinations, providing a new pathway for the reliable and efficient generation of AI leaderboards. This contribution not only advances the state of the art in leaderboard generation but also sheds light on strategies to mitigate common challenges in LLM-based information extraction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.03746v1">Efficient Knowledge Infusion via KG-LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2024-06-06
      | 💬 ACL2024 Findings
    </div>
    <details class="paper-abstract">
      To tackle the problem of domain-specific knowledge scarcity within large language models (LLMs), knowledge graph-retrievalaugmented method has been proven to be an effective and efficient technique for knowledge infusion. However, existing approaches face two primary challenges: knowledge mismatch between public available knowledge graphs and the specific domain of the task at hand, and poor information compliance of LLMs with knowledge graphs. In this paper, we leverage a small set of labeled samples and a large-scale corpus to efficiently construct domain-specific knowledge graphs by an LLM, addressing the issue of knowledge mismatch. Additionally, we propose a three-stage KG-LLM alignment strategyto enhance the LLM's capability to utilize information from knowledge graphs. We conduct experiments with a limited-sample setting on two biomedical question-answering datasets, and the results demonstrate that our approach outperforms existing baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.10450v3">PRISE: LLM-Style Sequence Compression for Learning Temporal Action Abstractions in Control</a></div>
    <div class="paper-meta">
      📅 2024-06-06
      | 💬 Accepted at the Forty-first International Conference on Machine Learning (ICML 2024)
    </div>
    <details class="paper-abstract">
      Temporal action abstractions, along with belief state representations, are a powerful knowledge sharing mechanism for sequential decision making. In this work, we propose a novel view that treats inducing temporal action abstractions as a sequence compression problem. To do so, we bring a subtle but critical component of LLM training pipelines -- input tokenization via byte pair encoding (BPE) -- to the seemingly distant task of learning skills of variable time span in continuous control domains. We introduce an approach called Primitive Sequence Encoding (PRISE) that combines continuous action quantization with BPE to learn powerful action abstractions. We empirically show that high-level skills discovered by PRISE from a multitask set of robotic manipulation demonstrations significantly boost the performance of both multitask imitation learning as well as few-shot imitation learning on unseen tasks. Our code is released at https://github.com/FrankZheng2022/PRISE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.03725v1">LLMEmbed: Rethinking Lightweight LLM's Genuine Function in Text Classification</a></div>
    <div class="paper-meta">
      📅 2024-06-06
      | 💬 ACL 2024 main conference
    </div>
    <details class="paper-abstract">
      With the booming of Large Language Models (LLMs), prompt-learning has become a promising method mainly researched in various research areas. Recently, many attempts based on prompt-learning have been made to improve the performance of text classification. However, most of these methods are based on heuristic Chain-of-Thought (CoT), and tend to be more complex but less efficient. In this paper, we rethink the LLM-based text classification methodology, propose a simple and effective transfer learning strategy, namely LLMEmbed, to address this classical but challenging task. To illustrate, we first study how to properly extract and fuse the text embeddings via various lightweight LLMs at different network depths to improve their robustness and discrimination, then adapt such embeddings to train the classifier. We perform extensive experiments on publicly available datasets, and the results show that LLMEmbed achieves strong performance while enjoys low training overhead using lightweight LLM backbones compared to recent methods based on larger LLMs, i.e. GPT-3, and sophisticated prompt-based strategies. Our LLMEmbed achieves adequate accuracy on publicly available benchmarks without any fine-tuning while merely use 4% model parameters, 1.8% electricity consumption and 1.5% runtime compared to its counterparts. Code is available at: https://github.com/ChunLiu-cs/LLMEmbed-ACL2024.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.13605v6">KorNAT: LLM Alignment Benchmark for Korean Social Values and Common Knowledge</a></div>
    <div class="paper-meta">
      📅 2024-06-06
      | 💬 Accepted at ACL 2024 Findings (35 pages, 7 figures, 16 tables)
    </div>
    <details class="paper-abstract">
      For Large Language Models (LLMs) to be effectively deployed in a specific country, they must possess an understanding of the nation's culture and basic knowledge. To this end, we introduce National Alignment, which measures an alignment between an LLM and a targeted country from two aspects: social value alignment and common knowledge alignment. Social value alignment evaluates how well the model understands nation-specific social values, while common knowledge alignment examines how well the model captures basic knowledge related to the nation. We constructed KorNAT, the first benchmark that measures national alignment with South Korea. For the social value dataset, we obtained ground truth labels from a large-scale survey involving 6,174 unique Korean participants. For the common knowledge dataset, we constructed samples based on Korean textbooks and GED reference materials. KorNAT contains 4K and 6K multiple-choice questions for social value and common knowledge, respectively. Our dataset creation process is meticulously designed and based on statistical sampling theory and was refined through multiple rounds of human review. The experiment results of seven LLMs reveal that only a few models met our reference score, indicating a potential for further enhancement. KorNAT has received government approval after passing an assessment conducted by a government-affiliated organization dedicated to evaluating dataset quality. Samples and detailed evaluation protocols of our dataset can be found in https://huggingface.co/datasets/jiyounglee0523/KorNAT .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.14592v2">Meta-Tuning LLMs to Leverage Lexical Knowledge for Generalizable Language Style Understanding</a></div>
    <div class="paper-meta">
      📅 2024-06-06
      | 💬 Accepted to ACL 2024 main conference
    </div>
    <details class="paper-abstract">
      Language style is often used by writers to convey their intentions, identities, and mastery of language. In this paper, we show that current large language models struggle to capture some language styles without fine-tuning. To address this challenge, we investigate whether LLMs can be meta-trained based on representative lexicons to recognize new styles they have not been fine-tuned on. Experiments on 13 established style classification tasks, as well as 63 novel tasks generated using LLMs, demonstrate that meta-training with style lexicons consistently improves zero-shot transfer across styles. We release the code and data at http://github.com/octaviaguo/Style-LLM .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.09213v3">GENEVA: GENErating and Visualizing branching narratives using LLMs</a></div>
    <div class="paper-meta">
      📅 2024-06-05
      | 💬 Accepted at IEEE Conference on Games 2024
    </div>
    <details class="paper-abstract">
      Dialogue-based Role Playing Games (RPGs) require powerful storytelling. The narratives of these may take years to write and typically involve a large creative team. In this work, we demonstrate the potential of large generative text models to assist this process. \textbf{GENEVA}, a prototype tool, generates a rich narrative graph with branching and reconverging storylines that match a high-level narrative description and constraints provided by the designer. A large language model (LLM), GPT-4, is used to generate the branching narrative and to render it in a graph format in a two-step process. We illustrate the use of GENEVA in generating new branching narratives for four well-known stories under different contextual constraints. This tool has the potential to assist in game development, simulations, and other applications with game-like properties.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.03614v1">Advancing Anomaly Detection: Non-Semantic Financial Data Encoding with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-06-05
    </div>
    <details class="paper-abstract">
      Detecting anomalies in general ledger data is of utmost importance to ensure trustworthiness of financial records. Financial audits increasingly rely on machine learning (ML) algorithms to identify irregular or potentially fraudulent journal entries, each characterized by a varying number of transactions. In machine learning, heterogeneity in feature dimensions adds significant complexity to data analysis. In this paper, we introduce a novel approach to anomaly detection in financial data using Large Language Models (LLMs) embeddings. To encode non-semantic categorical data from real-world financial records, we tested 3 pre-trained general purpose sentence-transformer models. For the downstream classification task, we implemented and evaluated 5 optimized ML models including Logistic Regression, Random Forest, Gradient Boosting Machines, Support Vector Machines, and Neural Networks. Our experiments demonstrate that LLMs contribute valuable information to anomaly detection as our models outperform the baselines, in selected settings even by a large margin. The findings further underscore the effectiveness of LLMs in enhancing anomaly detection in financial journal entries, particularly by tackling feature sparsity. We discuss a promising perspective on using LLM embeddings for non-semantic data in the financial context and beyond.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.16966v2">Examining the robustness of LLM evaluation to the distributional assumptions of benchmarks</a></div>
    <div class="paper-meta">
      📅 2024-06-05
    </div>
    <details class="paper-abstract">
      Benchmarks have emerged as the central approach for evaluating Large Language Models (LLMs). The research community often relies on a model's average performance across the test prompts of a benchmark to evaluate the model's performance. This is consistent with the assumption that the test prompts within a benchmark represent a random sample from a real-world distribution of interest. We note that this is generally not the case; instead, we hold that the distribution of interest varies according to the specific use case. We find that (1) the correlation in model performance across test prompts is non-random, (2) accounting for correlations across test prompts can change model rankings on major benchmarks, (3) explanatory factors for these correlations include semantic similarity and common LLM failure points.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.03600v1">Knowledge-Infused Legal Wisdom: Navigating LLM Consultation through the Lens of Diagnostics and Positive-Unlabeled Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2024-06-05
      | 💬 Accepted by ACL Findings 2024
    </div>
    <details class="paper-abstract">
      The integration of generative Large Language Models (LLMs) into various applications, including the legal domain, has been accelerated by their expansive and versatile nature. However, when facing a legal case, users without a legal background often struggle to formulate professional queries and may inadvertently overlook critical legal factors when presenting their case narrative to LLMs. To address this issue, we propose the Diagnostic Legal Large Language Model (D3LM), which utilizes adaptive lawyer-like diagnostic questions to collect additional case information and then provides high-quality feedback. D3LM incorporates an innovative graph-based Positive-Unlabeled Reinforcement Learning (PURL) algorithm, enabling the generation of critical questions and enhancing user-LLM interactions. Moreover, an integrated LLM-based stopping criterion facilitates precise Court Views Generation (CVG). Our research also introduces a new English-language CVG dataset based on the US case law database, enriching the realm of LLM research and deployment with a vital dimension. D3LM surpasses classical LLMs by delivering outstanding performance and a remarkable user experience in the legal domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.03496v1">Wings: Learning Multimodal LLMs without Text-only Forgetting</a></div>
    <div class="paper-meta">
      📅 2024-06-05
    </div>
    <details class="paper-abstract">
      Multimodal large language models (MLLMs), initiated with a trained LLM, first align images with text and then fine-tune on multimodal mixed inputs. However, the MLLM catastrophically forgets the text-only instructions, which do not include images and can be addressed within the initial LLM. In this paper, we present Wings, a novel MLLM that excels in both text-only dialogues and multimodal comprehension. Analyzing MLLM attention in multimodal instructions reveals that text-only forgetting is related to the attention shifts from pre-image to post-image text. From that, we construct extra modules that act as the boosted learner to compensate for the attention shift. The complementary visual and textual learners, like "wings" on either side, are connected in parallel within each layer's attention block. Initially, image and text inputs are aligned with visual learners operating alongside the main attention, balancing focus on visual elements. Textual learners are later collaboratively integrated with attention-based routing to blend the outputs of the visual and textual learners. We design the Low-Rank Residual Attention (LoRRA) to guarantee high efficiency for learners. Our experimental results demonstrate that Wings outperforms equally-scaled MLLMs in both text-only and visual question-answering tasks. On a newly constructed Interleaved Image-Text (IIT) benchmark, Wings exhibits superior performance from text-only-rich to multimodal-rich question-answering tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.03487v1">Analyzing LLM Behavior in Dialogue Summarization: Unveiling Circumstantial Hallucination Trends</a></div>
    <div class="paper-meta">
      📅 2024-06-05
      | 💬 Accepted at ACL 2024
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have considerably advanced the capabilities of summarization systems. However, they continue to face concerns about hallucinations. While prior work has evaluated LLMs extensively in news domains, most evaluation of dialogue summarization has focused on BART-based models, leaving a gap in our understanding of their faithfulness. Our work benchmarks the faithfulness of LLMs for dialogue summarization, using human annotations and focusing on identifying and categorizing span-level inconsistencies. Specifically, we focus on two prominent LLMs: GPT-4 and Alpaca-13B. Our evaluation reveals subtleties as to what constitutes a hallucination: LLMs often generate plausible inferences, supported by circumstantial evidence in the conversation, that lack direct evidence, a pattern that is less prevalent in older models. We propose a refined taxonomy of errors, coining the category of "Circumstantial Inference" to bucket these LLM behaviors and release the dataset. Using our taxonomy, we compare the behavioral differences between LLMs and older fine-tuned models. Additionally, we systematically assess the efficacy of automatic error detection methods on LLM summaries and find that they struggle to detect these nuanced errors. To address this, we introduce two prompt-based approaches for fine-grained error detection that outperform existing metrics, particularly for identifying "Circumstantial Inference."
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.03441v1">Cycles of Thought: Measuring LLM Confidence through Stable Explanations</a></div>
    <div class="paper-meta">
      📅 2024-06-05
    </div>
    <details class="paper-abstract">
      In many high-risk machine learning applications it is essential for a model to indicate when it is uncertain about a prediction. While large language models (LLMs) can reach and even surpass human-level accuracy on a variety of benchmarks, their overconfidence in incorrect responses is still a well-documented failure mode. Traditional methods for ML uncertainty quantification can be difficult to directly adapt to LLMs due to the computational cost of implementation and closed-source nature of many models. A variety of black-box methods have recently been proposed, but these often rely on heuristics such as self-verbalized confidence. We instead propose a framework for measuring an LLM's uncertainty with respect to the distribution of generated explanations for an answer. While utilizing explanations is not a new idea in and of itself, by interpreting each possible model+explanation pair as a test-time classifier we can calculate a posterior answer distribution over the most likely of these classifiers. We demonstrate how a specific instance of this framework using explanation entailment as our classifier likelihood improves confidence score metrics (in particular AURC and AUROC) over baselines across five different datasets. We believe these results indicate that our framework is both a well-principled and effective way of quantifying uncertainty in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.03428v1">HelloFresh: LLM Evaluations on Streams of Real-World Human Editorial Actions across X Community Notes and Wikipedia edits</a></div>
    <div class="paper-meta">
      📅 2024-06-05
      | 💬 ACL 2024 Findings
    </div>
    <details class="paper-abstract">
      Benchmarks have been essential for driving progress in machine learning. A better understanding of LLM capabilities on real world tasks is vital for safe development. Designing adequate LLM benchmarks is challenging: Data from real-world tasks is hard to collect, public availability of static evaluation data results in test data contamination and benchmark overfitting, and periodically generating new evaluation data is tedious and may result in temporally inconsistent results. We introduce HelloFresh, based on continuous streams of real-world data generated by intrinsically motivated human labelers. It covers recent events from X (formerly Twitter) community notes and edits of Wikipedia pages, mitigating the risk of test data contamination and benchmark overfitting. Any X user can propose an X note to add additional context to a misleading post (formerly tweet); if the community classifies it as helpful, it is shown with the post. Similarly, Wikipedia relies on community-based consensus, allowing users to edit articles or revert edits made by other users. Verifying whether an X note is helpful or whether a Wikipedia edit should be accepted are hard tasks that require grounding by querying the web. We backtest state-of-the-art LLMs supplemented with simple web search access and find that HelloFresh yields a temporally consistent ranking. To enable continuous evaluation on HelloFresh, we host a public leaderboard and periodically updated evaluation data at https://tinyurl.com/hello-fresh-LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.11905v2">Learning to Edit: Aligning LLMs with Knowledge Editing</a></div>
    <div class="paper-meta">
      📅 2024-06-05
      | 💬 17 pages, 8 figures, 9 tables. ACL 2024 main camera-ready version
    </div>
    <details class="paper-abstract">
      Knowledge editing techniques, aiming to efficiently modify a minor proportion of knowledge in large language models (LLMs) without negatively impacting performance across other inputs, have garnered widespread attention. However, existing methods predominantly rely on memorizing the updated knowledge, impeding LLMs from effectively combining the new knowledge with their inherent knowledge when answering questions. To this end, we propose a Learning to Edit (LTE) framework, focusing on teaching LLMs to apply updated knowledge into input questions, inspired by the philosophy of "Teach a man to fish." LTE features a two-phase process: (i) the Alignment Phase, which fine-tunes LLMs on a meticulously curated parallel dataset to make reliable, in-scope edits while preserving out-of-scope information and linguistic proficiency; and (ii) the Inference Phase, which employs a retrieval-based mechanism for real-time and mass knowledge editing. By comparing our approach with seven advanced baselines across four popular knowledge editing benchmarks and two LLM architectures, we demonstrate LTE's superiority in knowledge editing performance, robustness in both batch and sequential editing, minimal interference on general tasks, and rapid editing speeds. The data and code are available at https://github.com/YJiangcm/LTE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.03363v1">LLM-based Rewriting of Inappropriate Argumentation using Reinforcement Learning from Machine Feedback</a></div>
    <div class="paper-meta">
      📅 2024-06-05
    </div>
    <details class="paper-abstract">
      Ensuring that online discussions are civil and productive is a major challenge for social media platforms. Such platforms usually rely both on users and on automated detection tools to flag inappropriate arguments of other users, which moderators then review. However, this kind of post-hoc moderation is expensive and time-consuming, and moderators are often overwhelmed by the amount and severity of flagged content. Instead, a promising alternative is to prevent negative behavior during content creation. This paper studies how inappropriate language in arguments can be computationally mitigated. We propose a reinforcement learning-based rewriting approach that balances content preservation and appropriateness based on existing classifiers, prompting an instruction-finetuned large language model (LLM) as our initial policy. Unlike related style transfer tasks, rewriting inappropriate arguments allows deleting and adding content permanently. It is therefore tackled on document level rather than sentence level. We evaluate different weighting schemes for the reward function in both absolute and relative human assessment studies. Systematic experiments on non-parallel data provide evidence that our approach can mitigate the inappropriateness of arguments while largely preserving their content. It significantly outperforms competitive baselines, including few-shot learning, prompting, and humans.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.03158v1">CSS: Contrastive Semantic Similarity for Uncertainty Quantification of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-06-05
      | 💬 The paper is accepted by The Conference on Uncertainty in Artificial Intelligence (UAI), 2024
    </div>
    <details class="paper-abstract">
      Despite the impressive capability of large language models (LLMs), knowing when to trust their generations remains an open challenge. The recent literature on uncertainty quantification of natural language generation (NLG) utilises a conventional natural language inference (NLI) classifier to measure the semantic dispersion of LLMs responses. These studies employ logits of NLI classifier for semantic clustering to estimate uncertainty. However, logits represent the probability of the predicted class and barely contain feature information for potential clustering. Alternatively, CLIP (Contrastive Language-Image Pre-training) performs impressively in extracting image-text pair features and measuring their similarity. To extend its usability, we propose Contrastive Semantic Similarity, the CLIP-based feature extraction module to obtain similarity features for measuring uncertainty for text pairs. We apply this method to selective NLG, which detects and rejects unreliable generations for better trustworthiness of LLMs. We conduct extensive experiments with three LLMs on several benchmark question-answering datasets with comprehensive evaluation metrics. Results show that our proposed method performs better in estimating reliable responses of LLMs than comparable baselines. Results show that our proposed method performs better in estimating reliable responses of LLMs than comparable baselines. The code are available at \url{https://github.com/AoShuang92/css_uq_llms}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.14883v3">Double-I Watermark: Protecting Model Copyright for LLM Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2024-06-05
    </div>
    <details class="paper-abstract">
      To support various applications, a prevalent and efficient approach for business owners is leveraging their valuable datasets to fine-tune a pre-trained LLM through the API provided by LLM owners or cloud servers. However, this process carries a substantial risk of model misuse, potentially resulting in severe economic consequences for business owners. Thus, safeguarding the copyright of these customized models during LLM fine-tuning has become an urgent practical requirement, but there are limited existing solutions to provide such protection. To tackle this pressing issue, we propose a novel watermarking approach named ``Double-I watermark''. Specifically, based on the instruct-tuning data, two types of backdoor data paradigms are introduced with trigger in the instruction and the input, respectively. By leveraging LLM's learning capability to incorporate customized backdoor samples into the dataset, the proposed approach effectively injects specific watermarking information into the customized model during fine-tuning, which makes it easy to inject and verify watermarks in commercial scenarios. We evaluate the proposed "Double-I watermark" under various fine-tuning methods, demonstrating its harmlessness, robustness, uniqueness, imperceptibility, and validity through both quantitative and qualitative analyses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.15337v2">Ranking Entities along Conceptual Space Dimensions with LLMs: An Analysis of Fine-Tuning Strategies</a></div>
    <div class="paper-meta">
      📅 2024-06-05
      | 💬 Accepted in ACL 2024 (Findings)
    </div>
    <details class="paper-abstract">
      Conceptual spaces represent entities in terms of their primitive semantic features. Such representations are highly valuable but they are notoriously difficult to learn, especially when it comes to modelling perceptual and subjective features. Distilling conceptual spaces from Large Language Models (LLMs) has recently emerged as a promising strategy, but existing work has been limited to probing pre-trained LLMs using relatively simple zero-shot strategies. We focus in particular on the task of ranking entities according to a given conceptual space dimension. Unfortunately, we cannot directly fine-tune LLMs on this task, because ground truth rankings for conceptual space dimensions are rare. We therefore use more readily available features as training data and analyse whether the ranking capabilities of the resulting models transfer to perceptual and subjective features. We find that this is indeed the case, to some extent, but having at least some perceptual and subjective features in the training data seems essential for achieving the best results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.20835v3">Outliers and Calibration Sets have Diminishing Effect on Quantization of Modern LLMs</a></div>
    <div class="paper-meta">
      📅 2024-06-05
    </div>
    <details class="paper-abstract">
      Post-Training Quantization (PTQ) enhances the efficiency of Large Language Models (LLMs) by enabling faster operation and compatibility with more accessible hardware through reduced memory usage, at the cost of small performance drops. We explore the role of calibration sets in PTQ, specifically their effect on hidden activations in various notable open-source LLMs. Calibration sets are crucial for evaluating activation magnitudes and identifying outliers, which can distort the quantization range and negatively impact performance. Our analysis reveals a marked contrast in quantization effectiveness across models. The older OPT model, upon which much of the quantization literature is based, shows significant performance deterioration and high susceptibility to outliers with varying calibration sets. In contrast, newer models like Llama-2 7B, Llama-3 8B, Command-R 35B, and Mistral 7B demonstrate strong robustness, with Mistral 7B showing near-immunity to outliers and stable activations. These findings suggest a shift in PTQ strategies might be needed. As advancements in pre-training methods reduce the relevance of outliers, there is an emerging need to reassess the fundamentals of current quantization literature. The emphasis should pivot towards optimizing inference speed, rather than primarily focusing on outlier preservation, to align with the evolving characteristics of state-of-the-art LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.09836v2">Chain-of-Planned-Behaviour Workflow Elicits Few-Shot Mobility Generation in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-06-05
    </div>
    <details class="paper-abstract">
      The powerful reasoning capabilities of large language models (LLMs) have brought revolutionary changes to many fields, but their performance in human behaviour generation has not yet been extensively explored. This gap likely emerges because the internal processes governing behavioral intentions cannot be solely explained by abstract reasoning. Instead, they are also influenced by a multitude of factors, including social norms and personal preference. Inspired by the Theory of Planned Behaviour (TPB), we develop a LLM workflow named Chain-of-Planned Behaviour (CoPB) for mobility behaviour generation, which reflects the important spatio-temporal dynamics of human activities. Through exploiting the cognitive structures of attitude, subjective norms, and perceived behaviour control in TPB, CoPB significantly enhance the ability of LLMs to reason the intention of next movement. Specifically, CoPB substantially reduces the error rate of mobility intention generation from 57.8% to 19.4%. To improve the scalability of the proposed CoPB workflow, we further explore the synergy between LLMs and mechanistic models. We find mechanistic mobility models, such as gravity model, can effectively map mobility intentions to physical mobility behaviours. The strategy of integrating CoPB with gravity model can reduce the token cost by 97.7% and achieve better performance simultaneously. Besides, the proposed CoPB workflow can facilitate GPT-4-turbo to automatically generate high quality labels for mobility behavior reasoning. We show such labels can be leveraged to fine-tune the smaller-scale, open source LLaMA 3-8B, which significantly reduces usage costs without sacrificing the quality of the generated behaviours.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.12532v2">PyramidInfer: Pyramid KV Cache Compression for High-throughput LLM Inference</a></div>
    <div class="paper-meta">
      📅 2024-06-05
      | 💬 Accepted by ACL 2024
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable comprehension abilities but face challenges in GPU memory usage during inference, hindering their scalability for real-time applications like chatbots. To accelerate inference, we store computed keys and values (KV cache) in the GPU memory. Existing methods study the KV cache compression to reduce memory by pruning the pre-computed KV cache. However, they neglect the inter-layer dependency between layers and huge memory consumption in pre-computation. To explore these deficiencies, we find that the number of crucial keys and values that influence future generations decreases layer by layer and we can extract them by the consistency in attention weights. Based on the findings, we propose PyramidInfer, a method that compresses the KV cache by layer-wise retaining crucial context. PyramidInfer saves significant memory by computing fewer keys and values without sacrificing performance. Experimental results show PyramidInfer improves 2.2x throughput compared to Accelerate with over 54% GPU memory reduction in KV cache.
    </details>
</div>
