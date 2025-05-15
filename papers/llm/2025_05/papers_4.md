# llm - 2025_05

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- Part 4

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02099v1">MemEngine: A Unified and Modular Library for Developing Advanced Memory of LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-04
      | 💬 Just accepted by TheWebConf'25 Resource Track
    </div>
    <details class="paper-abstract">
      Recently, large language model based (LLM-based) agents have been widely applied across various fields. As a critical part, their memory capabilities have captured significant interest from both industrial and academic communities. Despite the proposal of many advanced memory models in recent research, however, there remains a lack of unified implementations under a general framework. To address this issue, we develop a unified and modular library for developing advanced memory models of LLM-based agents, called MemEngine. Based on our framework, we implement abundant memory models from recent research works. Additionally, our library facilitates convenient and extensible memory development, and offers user-friendly and pluggable memory usage. For benefiting our community, we have made our project publicly available at https://github.com/nuster1128/MemEngine.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02091v1">LLM-OptiRA: LLM-Driven Optimization of Resource Allocation for Non-Convex Problems in Wireless Communications</a></div>
    <div class="paper-meta">
      📅 2025-05-04
      | 💬 6 pages,4 figures
    </div>
    <details class="paper-abstract">
      Solving non-convex resource allocation problems poses significant challenges in wireless communication systems, often beyond the capability of traditional optimization techniques. To address this issue, we propose LLM-OptiRA, the first framework that leverages large language models (LLMs) to automatically detect and transform non-convex components into solvable forms, enabling fully automated resolution of non-convex resource allocation problems in wireless communication systems. LLM-OptiRA not only simplifies problem-solving by reducing reliance on expert knowledge, but also integrates error correction and feasibility validation mechanisms to ensure robustness. Experimental results show that LLM-OptiRA achieves an execution rate of 96% and a success rate of 80% on GPT-4, significantly outperforming baseline approaches in complex optimization tasks across diverse scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02076v1">Leveraging LLM Agents and Digital Twins for Fault Handling in Process Plants</a></div>
    <div class="paper-meta">
      📅 2025-05-04
    </div>
    <details class="paper-abstract">
      Advances in Automation and Artificial Intelligence continue to enhance the autonomy of process plants in handling various operational scenarios. However, certain tasks, such as fault handling, remain challenging, as they rely heavily on human expertise. This highlights the need for systematic, knowledge-based methods. To address this gap, we propose a methodological framework that integrates Large Language Model (LLM) agents with a Digital Twin environment. The LLM agents continuously interpret system states and initiate control actions, including responses to unexpected faults, with the goal of returning the system to normal operation. In this context, the Digital Twin acts both as a structured repository of plant-specific engineering knowledge for agent prompting and as a simulation platform for the systematic validation and verification of the generated corrective control actions. The evaluation using a mixing module of a process plant demonstrates that the proposed framework is capable not only of autonomously controlling the mixing module, but also of generating effective corrective actions to mitigate a pipe clogging with only a few reprompts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02009v1">Towards Safer Pretraining: Analyzing and Filtering Harmful Content in Webscale datasets for Responsible LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have become integral to various real-world applications, leveraging massive, web-sourced datasets like Common Crawl, C4, and FineWeb for pretraining. While these datasets provide linguistic data essential for high-quality natural language generation, they often contain harmful content, such as hate speech, misinformation, and biased narratives. Training LLMs on such unfiltered data risks perpetuating toxic behaviors, spreading misinformation, and amplifying societal biases which can undermine trust in LLM-driven applications and raise ethical concerns about their use. This paper presents a large-scale analysis of inappropriate content across these datasets, offering a comprehensive taxonomy that categorizes harmful webpages into Topical and Toxic based on their intent. We also introduce a prompt evaluation dataset, a high-accuracy Topical and Toxic Prompt (TTP), and a transformer-based model (HarmFormer) for content filtering. Additionally, we create a new multi-harm open-ended toxicity benchmark (HAVOC) and provide crucial insights into how models respond to adversarial toxic inputs. Upon publishing, we will also opensource our model signal on the entire C4 dataset. Our work offers insights into ensuring safer LLM pretraining and serves as a resource for Responsible AI (RAI) compliance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09061v3">CRANE: Reasoning with constrained LLM generation</a></div>
    <div class="paper-meta">
      📅 2025-05-04
      | 💬 Accepted at ICML 2025
    </div>
    <details class="paper-abstract">
      Code generation, symbolic math reasoning, and other tasks require LLMs to produce outputs that are both syntactically and semantically correct. Constrained LLM generation is a promising direction to enforce adherence to formal grammar, but prior works have empirically observed that strict enforcement of formal constraints often diminishes the reasoning capabilities of LLMs. In this work, we first provide a theoretical explanation for why constraining LLM outputs to very restrictive grammars that only allow syntactically valid final answers reduces the reasoning capabilities of the model. Second, we demonstrate that by augmenting the output grammar with carefully designed additional rules, it is always possible to preserve the reasoning capabilities of the LLM while ensuring syntactic and semantic correctness in its outputs. Building on these theoretical insights, we propose a reasoning-augmented constrained decoding algorithm, CRANE, which effectively balances the correctness of constrained generation with the flexibility of unconstrained generation. Experiments on multiple open-source LLMs and benchmarks show that CRANE significantly outperforms both state-of-the-art constrained decoding strategies and standard unconstrained decoding, showing up to 10% points accuracy improvement over baselines on challenging symbolic reasoning benchmarks GSM-symbolic and FOLIO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14567v4">ELOQ: Resources for Enhancing LLM Detection of Out-of-Scope Questions</a></div>
    <div class="paper-meta">
      📅 2025-05-04
      | 💬 Accepted by SIGIR'25
    </div>
    <details class="paper-abstract">
      Retrieval-augmented generation (RAG) has become integral to large language models (LLMs), particularly for conversational AI systems where user questions may reference knowledge beyond the LLMs' training cutoff. However, many natural user questions lack well-defined answers, either due to limited domain knowledge or because the retrieval system returns documents that are relevant in appearance but uninformative in content. In such cases, LLMs often produce hallucinated answers without flagging them. While recent work has largely focused on questions with false premises, we study out-of-scope questions, where the retrieved document appears semantically similar to the question but lacks the necessary information to answer it. In this paper, we propose a guided hallucination-based approach ELOQ to automatically generate a diverse set of out-of-scope questions from post-cutoff documents, followed by human verification to ensure quality. We use this dataset to evaluate several LLMs on their ability to detect out-of-scope questions and generate appropriate responses. Finally, we introduce an improved detection method that enhances the reliability of LLM-based question-answering systems in handling out-of-scope questions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.01980v1">LLM-based Text Simplification and its Effect on User Comprehension and Cognitive Load</a></div>
    <div class="paper-meta">
      📅 2025-05-04
    </div>
    <details class="paper-abstract">
      Information on the web, such as scientific publications and Wikipedia, often surpasses users' reading level. To help address this, we used a self-refinement approach to develop a LLM capability for minimally lossy text simplification. To validate our approach, we conducted a randomized study involving 4563 participants and 31 texts spanning 6 broad subject areas: PubMed (biomedical scientific articles), biology, law, finance, literature/philosophy, and aerospace/computer science. Participants were randomized to viewing original or simplified texts in a subject area, and answered multiple-choice questions (MCQs) that tested their comprehension of the text. The participants were also asked to provide qualitative feedback such as task difficulty. Our results indicate that participants who read the simplified text answered more MCQs correctly than their counterparts who read the original text (3.9% absolute increase, p<0.05). This gain was most striking with PubMed (14.6%), while more moderate gains were observed for finance (5.5%), aerospace/computer science (3.8%) domains, and legal (3.5%). Notably, the results were robust to whether participants could refer back to the text while answering MCQs. The absolute accuracy decreased by up to ~9% for both original and simplified setups where participants could not refer back to the text, but the ~4% overall improvement persisted. Finally, participants' self-reported perceived ease based on a simplified NASA Task Load Index was greater for those who read the simplified text (absolute change on a 5-point scale 0.33, p<0.05). This randomized study, involving an order of magnitude more participants than prior works, demonstrates the potential of LLMs to make complex information easier to understand. Our work aims to enable a broader audience to better learn and make use of expert knowledge available on the web, improving information accessibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.01900v1">CAMOUFLAGE: Exploiting Misinformation Detection Systems Through LLM-driven Adversarial Claim Transformation</a></div>
    <div class="paper-meta">
      📅 2025-05-03
    </div>
    <details class="paper-abstract">
      Automated evidence-based misinformation detection systems, which evaluate the veracity of short claims against evidence, lack comprehensive analysis of their adversarial vulnerabilities. Existing black-box text-based adversarial attacks are ill-suited for evidence-based misinformation detection systems, as these attacks primarily focus on token-level substitutions involving gradient or logit-based optimization strategies, which are incapable of fooling the multi-component nature of these detection systems. These systems incorporate both retrieval and claim-evidence comparison modules, which requires attacks to break the retrieval of evidence and/or the comparison module so that it draws incorrect inferences. We present CAMOUFLAGE, an iterative, LLM-driven approach that employs a two-agent system, a Prompt Optimization Agent and an Attacker Agent, to create adversarial claim rewritings that manipulate evidence retrieval and mislead claim-evidence comparison, effectively bypassing the system without altering the meaning of the claim. The Attacker Agent produces semantically equivalent rewrites that attempt to mislead detectors, while the Prompt Optimization Agent analyzes failed attack attempts and refines the prompt of the Attacker to guide subsequent rewrites. This enables larger structural and stylistic transformations of the text rather than token-level substitutions, adapting the magnitude of changes based on previous outcomes. Unlike existing approaches, CAMOUFLAGE optimizes its attack solely based on binary model decisions to guide its rewriting process, eliminating the need for classifier logits or extensive querying. We evaluate CAMOUFLAGE on four systems, including two recent academic systems and two real-world APIs, with an average attack success rate of 46.92\% while preserving textual coherence and semantic equivalence to the original claims.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20117v2">ResearchCodeAgent: An LLM Multi-Agent System for Automated Codification of Research Methodologies</a></div>
    <div class="paper-meta">
      📅 2025-05-03
    </div>
    <details class="paper-abstract">
      In this paper we introduce ResearchCodeAgent, a novel multi-agent system leveraging large language models (LLMs) agents to automate the codification of research methodologies described in machine learning literature. The system bridges the gap between high-level research concepts and their practical implementation, allowing researchers auto-generating code of existing research papers for benchmarking or building on top-of existing methods specified in the literature with availability of partial or complete starter code. ResearchCodeAgent employs a flexible agent architecture with a comprehensive action suite, enabling context-aware interactions with the research environment. The system incorporates a dynamic planning mechanism, utilizing both short and long-term memory to adapt its approach iteratively. We evaluate ResearchCodeAgent on three distinct machine learning tasks with distinct task complexity and representing different parts of the ML pipeline: data augmentation, optimization, and data batching. Our results demonstrate the system's effectiveness and generalizability, with 46.9% of generated code being high-quality and error-free, and 25% showing performance improvements over baseline implementations. Empirical analysis shows an average reduction of 57.9% in coding time compared to manual implementation. We observe higher gains for more complex tasks. ResearchCodeAgent represents a significant step towards automating the research implementation process, potentially accelerating the pace of machine learning research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.01841v1">Harnessing the Power of LLMs, Informers and Decision Transformers for Intent-driven RAN Management in 6G</a></div>
    <div class="paper-meta">
      📅 2025-05-03
      | 💬 Currently under review
    </div>
    <details class="paper-abstract">
      Intent-driven network management is critical for managing the complexity of 5G and 6G networks. It enables adaptive, on-demand management of the network based on the objectives of the network operators. In this paper, we propose an innovative three-step framework for intent-driven network management based on Generative AI (GenAI) algorithms. First, we fine-tune a Large Language Model (LLM) on a custom dataset using a Quantized Low-Rank Adapter (QLoRA) to enable memory-efficient intent processing within limited computational resources. A Retrieval Augmented Generation (RAG) module is included to support dynamic decision-making. Second, we utilize a transformer architecture for time series forecasting to predict key parameters, such as power consumption, traffic load, and packet drop rate, to facilitate intent validation proactively. Lastly, we introduce a Hierarchical Decision Transformer with Goal Awareness (HDTGA) to optimize the selection and orchestration of network applications and hence, optimize the network. Our intent guidance and processing approach improves BERTScore by 6% and the semantic similarity score by 9% compared to the base LLM model. Again, the proposed predictive intent validation approach can successfully rule out the performance-degrading intents with an average of 88% accuracy. Finally, compared to the baselines, the proposed HDTGA algorithm increases throughput at least by 19.3%, reduces delay by 48.5%, and boosts energy efficiency by 54.9%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.01834v1">Model Context Protocol-based Internet of Experts For Wireless Environment-aware LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-03
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit strong general-purpose reasoning abilities but lack access to wireless environment information due to the absence of native sensory input and domain-specific priors. Previous attempts to apply LLMs in wireless systems either depend on retraining with network-specific data, which compromises language generalization, or rely on manually scripted interfaces, which hinder scalability. To overcome these limitations, we propose a Model Context Protocol (MCP)-based Internet of Experts (IoX) framework that equips LLMs with wireless environment-aware reasoning capabilities. The framework incorporates a set of lightweight expert models, each trained to solve a specific deterministic task in wireless communications, such as detecting a specific wireless attribute, e.g., line-of-sight propagation, Doppler effects, or fading conditions. Through MCP, the LLM can selectively query and interpret expert outputs at inference time, without modifying its own parameters. This architecture enables modular, extensible, and interpretable reasoning over wireless contexts. Evaluated across multiple mainstream LLMs, the proposed wireless environment-aware LLM agents achieve 40%-50% improvements in classification tasks over LLM-only baselines. More broadly, the MCP-based design offers a viable paradigm for future LLMs to inherit structured wireless network management capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.01743v1">An LLM-Empowered Low-Resolution Vision System for On-Device Human Behavior Understanding</a></div>
    <div class="paper-meta">
      📅 2025-05-03
    </div>
    <details class="paper-abstract">
      The rapid advancements in Large Vision Language Models (LVLMs) offer the potential to surpass conventional labeling by generating richer, more detailed descriptions of on-device human behavior understanding (HBU) in low-resolution vision systems, such as depth, thermal, and infrared. However, existing large vision language model (LVLM) approaches are unable to understand low-resolution data well as they are primarily designed for high-resolution data, such as RGB images. A quick fixing approach is to caption a large amount of low-resolution data, but it requires a significant amount of labor-intensive annotation efforts. In this paper, we propose a novel, labor-saving system, Llambda, designed to support low-resolution HBU. The core idea is to leverage limited labeled data and a large amount of unlabeled data to guide LLMs in generating informative captions, which can be combined with raw data to effectively fine-tune LVLM models for understanding low-resolution videos in HBU. First, we propose a Contrastive-Oriented Data Labeler, which can capture behavior-relevant information from long, low-resolution videos and generate high-quality pseudo labels for unlabeled data via contrastive learning. Second, we propose a Physical-Knowledge Guided Captioner, which utilizes spatial and temporal consistency checks to mitigate errors in pseudo labels. Therefore, it can improve LLMs' understanding of sequential data and then generate high-quality video captions. Finally, to ensure on-device deployability, we employ LoRA-based efficient fine-tuning to adapt LVLMs for low-resolution data. We evaluate Llambda using a region-scale real-world testbed and three distinct low-resolution datasets, and the experiments show that Llambda outperforms several state-of-the-art LVLM systems up to $40.03\%$ on average Bert-Score.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.01698v1">Amplifying Your Social Media Presence: Personalized Influential Content Generation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-03
    </div>
    <details class="paper-abstract">
      The remarkable advancements in Large Language Models (LLMs) have revolutionized the content generation process in social media, offering significant convenience in writing tasks. However, existing applications, such as sentence completion and fluency enhancement, do not fully address the complex challenges in real-world social media contexts. A prevalent goal among social media users is to increase the visibility and influence of their posts. This paper, therefore, delves into the compelling question: Can LLMs generate personalized influential content to amplify a user's presence on social media? We begin by examining prevalent techniques in content generation to assess their impact on post influence. Acknowledging the critical impact of underlying network structures in social media, which are instrumental in initiating content cascades and highly related to the influence/popularity of a post, we then inject network information into prompt for content generation to boost the post's influence. We design multiple content-centric and structure-aware prompts. The empirical experiments across LLMs validate their ability in improving the influence and draw insights on which strategies are more effective. Our code is available at https://github.com/YuyingZhao/LLM-influence-amplifier.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.01636v1">Structured Prompting and Feedback-Guided Reasoning with LLMs for Data Interpretation</a></div>
    <div class="paper-meta">
      📅 2025-05-03
      | 💬 21 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable capabilities in natural language understanding and task generalization. However, their application to structured data analysis remains fragile due to inconsistencies in schema interpretation, misalignment between user intent and model output, and limited mechanisms for self-correction when failures occur. This paper introduces the STROT Framework (Structured Task Reasoning and Output Transformation), a method for structured prompting and feedback-driven transformation logic generation aimed at improving the reliability and semantic alignment of LLM-based analytical workflows. STROT begins with lightweight schema introspection and sample-based field classification, enabling dynamic context construction that captures both the structure and statistical profile of the input data. This contextual information is embedded in structured prompts that guide the model toward generating task-specific, interpretable outputs. To address common failure modes in complex queries, STROT incorporates a refinement mechanism in which the model iteratively revises its outputs based on execution feedback and validation signals. Unlike conventional approaches that rely on static prompts or single-shot inference, STROT treats the LLM as a reasoning agent embedded within a controlled analysis loop -- capable of adjusting its output trajectory through planning and correction. The result is a robust and reproducible framework for reasoning over structured data with LLMs, applicable to diverse data exploration and analysis tasks where interpretability, stability, and correctness are essential.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02862v1">Cannot See the Forest for the Trees: Invoking Heuristics and Biases to Elicit Irrational Choices of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-03
    </div>
    <details class="paper-abstract">
      Despite the remarkable performance of Large Language Models (LLMs), they remain vulnerable to jailbreak attacks, which can compromise their safety mechanisms. Existing studies often rely on brute-force optimization or manual design, failing to uncover potential risks in real-world scenarios. To address this, we propose a novel jailbreak attack framework, ICRT, inspired by heuristics and biases in human cognition. Leveraging the simplicity effect, we employ cognitive decomposition to reduce the complexity of malicious prompts. Simultaneously, relevance bias is utilized to reorganize prompts, enhancing semantic alignment and inducing harmful outputs effectively. Furthermore, we introduce a ranking-based harmfulness evaluation metric that surpasses the traditional binary success-or-failure paradigm by employing ranking aggregation methods such as Elo, HodgeRank, and Rank Centrality to comprehensively quantify the harmfulness of generated content. Experimental results show that our approach consistently bypasses mainstream LLMs' safety mechanisms and generates high-risk content, providing insights into jailbreak attack risks and contributing to stronger defense strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03824v1">Memory Assisted LLM for Personalized Recommendation System</a></div>
    <div class="paper-meta">
      📅 2025-05-03
      | 💬 8 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated significant potential in solving recommendation tasks. With proven capabilities in understanding user preferences, LLM personalization has emerged as a critical area for providing tailored responses to individuals. Current studies explore personalization through prompt design and fine-tuning, paving the way for further research in personalized LLMs. However, existing approaches are either costly and inefficient in capturing diverse user preferences or fail to account for timely updates to user history. To address these gaps, we propose the Memory-Assisted Personalized LLM (MAP). Through user interactions, we first create a history profile for each user, capturing their preferences, such as ratings for historical items. During recommendation, we extract relevant memory based on similarity, which is then incorporated into the prompts to enhance personalized recommendations. In our experiments, we evaluate MAP using a sequential rating prediction task under two scenarios: single domain, where memory and tasks are from the same category (e.g., movies), and cross-domain (e.g., memory from movies and recommendation tasks in books). The results show that MAP outperforms regular LLM-based recommenders that integrate user history directly through prompt design. Moreover, as user history grows, MAP's advantage increases in both scenarios, making it more suitable for addressing successive personalized user requests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02851v1">30DayGen: Leveraging LLMs to Create a Content Corpus for Habit Formation</a></div>
    <div class="paper-meta">
      📅 2025-05-02
      | 💬 8 pages (main content), 4 figures. Submitted to ACL BEA2025
    </div>
    <details class="paper-abstract">
      In this paper, we present 30 Day Me, a habit formation application that leverages Large Language Models (LLMs) to help users break down their goals into manageable, actionable steps and track their progress. Central to the app is the 30DAYGEN system, which generates 3,531 unique 30-day challenges sourced from over 15K webpages, and enables runtime search of challenge ideas aligned with user-defined goals. We showcase how LLMs can be harnessed to rapidly construct domain specific content corpora for behavioral and educational purposes, and propose a practical pipeline that incorporates effective LLM enhanced approaches for content generation and semantic deduplication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03814v1">Cer-Eval: Certifiable and Cost-Efficient Evaluation Framework for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-02
    </div>
    <details class="paper-abstract">
      As foundation models continue to scale, the size of trained models grows exponentially, presenting significant challenges for their evaluation. Current evaluation practices involve curating increasingly large datasets to assess the performance of large language models (LLMs). However, there is a lack of systematic analysis and guidance on determining the sufficiency of test data or selecting informative samples for evaluation. This paper introduces a certifiable and cost-efficient evaluation framework for LLMs. Our framework adapts to different evaluation objectives and outputs confidence intervals that contain true values with high probability. We use ``test sample complexity'' to quantify the number of test points needed for a certifiable evaluation and derive tight bounds on test sample complexity. Based on the developed theory, we develop a partition-based algorithm, named Cer-Eval, that adaptively selects test points to minimize the cost of LLM evaluation. Real-world experiments demonstrate that Cer-Eval can save 20% to 40% test points across various benchmarks, while maintaining an estimation error level comparable to the current evaluation process and providing a 95% confidence guarantee.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00234v2">Self-Generated In-Context Examples Improve LLM Agents for Sequential Decision-Making Tasks</a></div>
    <div class="paper-meta">
      📅 2025-05-02
    </div>
    <details class="paper-abstract">
      Many methods for improving Large Language Model (LLM) agents for sequential decision-making tasks depend on task-specific knowledge engineering--such as prompt tuning, curated in-context examples, or customized observation and action spaces. Using these approaches, agent performance improves with the quality or amount of knowledge engineering invested. Instead, we investigate how LLM agents can automatically improve their performance by learning in-context from their own successful experiences on similar tasks. Rather than relying on task-specific knowledge engineering, we focus on constructing and refining a database of self-generated examples. We demonstrate that even a naive accumulation of successful trajectories across training tasks boosts test performance on three benchmarks: ALFWorld (73% to 89%), Wordcraft (55% to 64%), and InterCode-SQL (75% to 79%)--matching the performance the initial agent achieves if allowed two to three attempts per task. We then introduce two extensions: (1) database-level selection through population-based training to identify high-performing example collections, and (2) exemplar-level selection that retains individual trajectories based on their empirical utility as in-context examples. These extensions further enhance performance, achieving 91% on ALFWorld--matching more complex approaches that employ task-specific components and prompts. Our results demonstrate that automatic trajectory database construction offers a compelling alternative to labor-intensive knowledge engineering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.01263v1">FlowDubber: Movie Dubbing with LLM-based Semantic-aware Learning and Flow Matching based Voice Enhancing</a></div>
    <div class="paper-meta">
      📅 2025-05-02
    </div>
    <details class="paper-abstract">
      Movie Dubbing aims to convert scripts into speeches that align with the given movie clip in both temporal and emotional aspects while preserving the vocal timbre of a given brief reference audio. Existing methods focus primarily on reducing the word error rate while ignoring the importance of lip-sync and acoustic quality. To address these issues, we propose a large language model (LLM) based flow matching architecture for dubbing, named FlowDubber, which achieves high-quality audio-visual sync and pronunciation by incorporating a large speech language model and dual contrastive aligning while achieving better acoustic quality via the proposed voice-enhanced flow matching than previous works. First, we introduce Qwen2.5 as the backbone of LLM to learn the in-context sequence from movie scripts and reference audio. Then, the proposed semantic-aware learning focuses on capturing LLM semantic knowledge at the phoneme level. Next, dual contrastive aligning (DCA) boosts mutual alignment with lip movement, reducing ambiguities where similar phonemes might be confused. Finally, the proposed Flow-based Voice Enhancing (FVE) improves acoustic quality in two aspects, which introduces an LLM-based acoustics flow matching guidance to strengthen clarity and uses affine style prior to enhance identity when recovering noise into mel-spectrograms via gradient vector field prediction. Extensive experiments demonstrate that our method outperforms several state-of-the-art methods on two primary benchmarks. The demos are available at {\href{https://galaxycong.github.io/LLM-Flow-Dubber/}{\textcolor{red}{https://galaxycong.github.io/LLM-Flow-Dubber/}}}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.09830v4">Automating the Generation of Prompts for LLM-based Action Choice in PDDL Planning</a></div>
    <div class="paper-meta">
      📅 2025-05-02
      | 💬 Extended version of the paper from the ICAPS'25 proceedings (same main part + additional appendix)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have revolutionized a large variety of NLP tasks. An active debate is to what extent they can do reasoning and planning. Prior work has assessed the latter in the specific context of PDDL planning, based on manually converting three PDDL domains into natural language (NL) prompts. Here we automate this conversion step, showing how to leverage an LLM to automatically generate NL prompts from PDDL input. Our automatically generated NL prompts result in similar LLM-planning performance as the previous manually generated ones. Beyond this, the automation enables us to run much larger experiments, providing for the first time a broad evaluation of LLM planning performance in PDDL. Our NL prompts yield better performance than PDDL prompts and simple template-based NL prompts. Compared to symbolic planners, LLM planning lags far behind; but in some domains, our best LLM configuration scales up further than A$^\star$ using LM-cut.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.01177v1">LLM Security: Vulnerabilities, Attacks, Defenses, and Countermeasures</a></div>
    <div class="paper-meta">
      📅 2025-05-02
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) continue to evolve, it is critical to assess the security threats and vulnerabilities that may arise both during their training phase and after models have been deployed. This survey seeks to define and categorize the various attacks targeting LLMs, distinguishing between those that occur during the training phase and those that affect already trained models. A thorough analysis of these attacks is presented, alongside an exploration of defense mechanisms designed to mitigate such threats. Defenses are classified into two primary categories: prevention-based and detection-based defenses. Furthermore, our survey summarizes possible attacks and their corresponding defense strategies. It also provides an evaluation of the effectiveness of the known defense mechanisms for the different security threats. Our survey aims to offer a structured framework for securing LLMs, while also identifying areas that require further research to improve and strengthen defenses against emerging security challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.01077v1">Zero-Shot Document-Level Biomedical Relation Extraction via Scenario-based Prompt Design in Two-Stage with LLM</a></div>
    <div class="paper-meta">
      📅 2025-05-02
    </div>
    <details class="paper-abstract">
      With the advent of artificial intelligence (AI), many researchers are attempting to extract structured information from document-level biomedical literature by fine-tuning large language models (LLMs). However, they face significant challenges such as the need for expensive hardware, like high-performance GPUs and the high labor costs associated with annotating training datasets, especially in biomedical realm. Recent research on LLMs, such as GPT-4 and Llama3, has shown promising performance in zero-shot settings, inspiring us to explore a novel approach to achieve the same results from unannotated full documents using general LLMs with lower hardware and labor costs. Our approach combines two major stages: named entity recognition (NER) and relation extraction (RE). NER identifies chemical, disease and gene entities from the document with synonym and hypernym extraction using an LLM with a crafted prompt. RE extracts relations between entities based on predefined relation schemas and prompts. To enhance the effectiveness of prompt, we propose a five-part template structure and a scenario-based prompt design principles, along with evaluation method to systematically assess the prompts. Finally, we evaluated our approach against fine-tuning and pre-trained models on two biomedical datasets: ChemDisGene and CDR. The experimental results indicate that our proposed method can achieve comparable accuracy levels to fine-tuning and pre-trained models but with reduced human and hardware expenses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.01015v1">Value Portrait: Understanding Values of LLMs with Human-aligned Benchmark</a></div>
    <div class="paper-meta">
      📅 2025-05-02
      | 💬 32 pages, 7 figures
    </div>
    <details class="paper-abstract">
      The importance of benchmarks for assessing the values of language models has been pronounced due to the growing need of more authentic, human-aligned responses. However, existing benchmarks rely on human or machine annotations that are vulnerable to value-related biases. Furthermore, the tested scenarios often diverge from real-world contexts in which models are commonly used to generate text and express values. To address these issues, we propose the Value Portrait benchmark, a reliable framework for evaluating LLMs' value orientations with two key characteristics. First, the benchmark consists of items that capture real-life user-LLM interactions, enhancing the relevance of assessment results to real-world LLM usage and thus ecological validity. Second, each item is rated by human subjects based on its similarity to their own thoughts, and correlations between these ratings and the subjects' actual value scores are derived. This psychometrically validated approach ensures that items strongly correlated with specific values serve as reliable items for assessing those values. Through evaluating 27 LLMs with our benchmark, we find that these models prioritize Benevolence, Security, and Self-Direction values while placing less emphasis on Tradition, Power, and Achievement values. Also, our analysis reveals biases in how LLMs perceive various demographic groups, deviating from real human data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00989v1">VTS-LLM: Domain-Adaptive LLM Agent for Enhancing Awareness in Vessel Traffic Services through Natural Language</a></div>
    <div class="paper-meta">
      📅 2025-05-02
      | 💬 8 pages, 5 figures, 7 tablels, submitted to ITSC2025
    </div>
    <details class="paper-abstract">
      Vessel Traffic Services (VTS) are essential for maritime safety and regulatory compliance through real-time traffic management. However, with increasing traffic complexity and the prevalence of heterogeneous, multimodal data, existing VTS systems face limitations in spatiotemporal reasoning and intuitive human interaction. In this work, we propose VTS-LLM Agent, the first domain-adaptive large LLM agent tailored for interactive decision support in VTS operations. We formalize risk-prone vessel identification as a knowledge-augmented Text-to-SQL task, combining structured vessel databases with external maritime knowledge. To support this, we construct a curated benchmark dataset consisting of a custom schema, domain-specific corpus, and a query-SQL test set in multiple linguistic styles. Our framework incorporates NER-based relational reasoning, agent-based domain knowledge injection, semantic algebra intermediate representation, and query rethink mechanisms to enhance domain grounding and context-aware understanding. Experimental results show that VTS-LLM outperforms both general-purpose and SQL-focused baselines under command-style, operational-style, and formal natural language queries, respectively. Moreover, our analysis provides the first empirical evidence that linguistic style variation introduces systematic performance challenges in Text-to-SQL modeling. This work lays the foundation for natural language interfaces in vessel traffic services and opens new opportunities for proactive, LLM-driven maritime real-time traffic management.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00985v1">Position: Enough of Scaling LLMs! Lets Focus on Downscaling</a></div>
    <div class="paper-meta">
      📅 2025-05-02
    </div>
    <details class="paper-abstract">
      We challenge the dominant focus on neural scaling laws and advocate for a paradigm shift toward downscaling in the development of large language models (LLMs). While scaling laws have provided critical insights into performance improvements through increasing model and dataset size, we emphasize the significant limitations of this approach, particularly in terms of computational inefficiency, environmental impact, and deployment constraints. To address these challenges, we propose a holistic framework for downscaling LLMs that seeks to maintain performance while drastically reducing resource demands. This paper outlines practical strategies for transitioning away from traditional scaling paradigms, advocating for a more sustainable, efficient, and accessible approach to LLM development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00951v1">Preserving Privacy and Utility in LLM-Based Product Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-05-02
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based recommendation systems leverage powerful language models to generate personalized suggestions by processing user interactions and preferences. Unlike traditional recommendation systems that rely on structured data and collaborative filtering, LLM-based models process textual and contextual information, often using cloud-based infrastructure. This raises privacy concerns, as user data is transmitted to remote servers, increasing the risk of exposure and reducing control over personal information. To address this, we propose a hybrid privacy-preserving recommendation framework which separates sensitive from nonsensitive data and only shares the latter with the cloud to harness LLM-powered recommendations. To restore lost recommendations related to obfuscated sensitive data, we design a de-obfuscation module that reconstructs sensitive recommendations locally. Experiments on real-world e-commerce datasets show that our framework achieves almost the same recommendation utility with a system which shares all data with an LLM, while preserving privacy to a large extend. Compared to obfuscation-only techniques, our approach improves HR@10 scores and category distribution alignment, offering a better balance between privacy and recommendation quality. Furthermore, our method runs efficiently on consumer-grade hardware, making privacy-aware LLM-based recommendation systems practical for real-world use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00945v1">SSRLBot: Designing and Developing an LLM-based Agent using Socially Shared Regulated Learning</a></div>
    <div class="paper-meta">
      📅 2025-05-02
      | 💬 8 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based agents are increasingly used to support human experts by streamlining complex tasks and offering actionable insights. However, their application in multi-professional decision-making, particularly in teamwork contexts, remains underexplored. This design-based study addresses that gap by developing LLM functions to enhance collaboration, grounded in the Socially Shared Regulation of Learning (SSRL) framework and applied to medical diagnostic teamwork. SSRL emphasizes metacognitive, cognitive, motivational, and emotional processes in shared learning, focusing on how teams manage these processes to improve decision-making. This paper introduces SSRLBot, a prototype chatbot designed to help team members reflect on both their diagnostic performance and key SSRL skills. Its core functions include summarizing dialogues, analyzing SSRL behaviors, evaluating diagnostic outcomes, annotating SSRL markers in conversation, assessing their impact on performance, and identifying interpersonal regulatory dynamics. We compare SSRLBot's capabilities with those of Gemini-1.5, GPT-3.5, and Deepseek-R1 in a case study. SSRLBot demonstrates stronger alignment with SSRL theory, offering detailed evaluations that link behaviors to regulatory dimensions and suggesting improvements for collaboration. By integrating SSRL theory with LLM capabilities, SSRLBot contributes a novel tool for enhancing team-based decision-making and collaborative learning in high-stakes environments, such as medical education.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08067v5">Reward-Augmented Data Enhances Direct Preference Alignment of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-02
      | 💬 Published at ICML 2025
    </div>
    <details class="paper-abstract">
      Preference alignment in Large Language Models (LLMs) has significantly improved their ability to adhere to human instructions and intentions. However, existing direct alignment algorithms primarily focus on relative preferences and often overlook the qualitative aspects of responses, despite having access to preference data that includes reward scores from judge models during AI feedback. Striving to maximize the implicit reward gap between the chosen and the slightly inferior rejected responses can cause overfitting and unnecessary unlearning of the high-quality rejected responses. The unawareness of the reward scores also drives the LLM to indiscriminately favor the low-quality chosen responses and fail to generalize to optimal responses that are sparse in data. To overcome these shortcomings, our study introduces reward-conditioned LLM policies that discern and learn from the entire spectrum of response quality within the dataset, helping extrapolate to more optimal regions. We propose an effective yet simple data relabeling method that conditions the preference pairs on quality scores to construct a reward-augmented dataset. The experiments across various benchmarks and diverse models demonstrate that our approach consistently boosts DPO by a considerable margin. Through comprehensive ablation studies, we demonstrate that our method not only maximizes the utility of preference data but also mitigates the issue of unlearning, demonstrating its broad effectiveness beyond mere data expansion. Our code is available at https://github.com/shenao-zhang/reward-augmented-preference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.01572v1">PipeSpec: Breaking Stage Dependencies in Hierarchical LLM Decoding</a></div>
    <div class="paper-meta">
      📅 2025-05-02
      | 💬 10 pages, 5 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Speculative decoding accelerates large language model inference by using smaller draft models to generate candidate tokens for parallel verification. However, current approaches are limited by sequential stage dependencies that prevent full hardware utilization. We present PipeSpec, a framework that generalizes speculative decoding to $k$ models arranged in a hierarchical pipeline, enabling asynchronous execution with lightweight coordination for prediction verification and rollback. Our analytical model characterizes token generation rates across pipeline stages and proves guaranteed throughput improvements over traditional decoding for any non-zero acceptance rate. We further derive closed-form expressions for steady-state verification probabilities that explain the empirical benefits of pipeline depth. Experimental results show that PipeSpec achieves up to 2.54$\times$ speedup while outperforming state-of-the-art methods. We validate PipeSpec across text summarization and code generation tasks using LLaMA 2 and 3 models, demonstrating that pipeline efficiency increases with model depth, providing a scalable approach to accelerating LLM inference on multi-device systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.01484v1">LLM Watermarking Using Mixtures and Statistical-to-Computational Gaps</a></div>
    <div class="paper-meta">
      📅 2025-05-02
    </div>
    <details class="paper-abstract">
      Given a text, can we determine whether it was generated by a large language model (LLM) or by a human? A widely studied approach to this problem is watermarking. We propose an undetectable and elementary watermarking scheme in the closed setting. Also, in the harder open setting, where the adversary has access to most of the model, we propose an unremovable watermarking scheme.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.01482v1">Understanding LLM Scientific Reasoning through Promptings and Model's Explanation on the Answers</a></div>
    <div class="paper-meta">
      📅 2025-05-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable capabilities in natural language understanding, reasoning, and problem-solving across various domains. However, their ability to perform complex, multi-step reasoning task-essential for applications in science, medicine, and law-remains an area of active investigation. This paper examines the reasoning capabilities of contemporary LLMs, analyzing their strengths, limitations, and potential for improvement. The study uses prompt engineering techniques on the Graduate-Level GoogleProof Q&A (GPQA) dataset to assess the scientific reasoning of GPT-4o. Five popular prompt engineering techniques and two tailored promptings were tested: baseline direct answer (zero-shot), chain-of-thought (CoT), zero-shot CoT, self-ask, self-consistency, decomposition, and multipath promptings. Our findings indicate that while LLMs exhibit emergent reasoning abilities, they often rely on pattern recognition rather than true logical inference, leading to inconsistencies in complex problem-solving. The results indicated that self-consistency outperformed the other prompt engineering technique with an accuracy of 52.99%, followed by direct answer (52.23%). Zero-shot CoT (50%) outperformed multipath (48.44%), decomposition (47.77%), self-ask (46.88%), and CoT (43.75%). Self-consistency performed the second worst in explaining the answers. Simple techniques such as direct answer, CoT, and zero-shot CoT have the best scientific reasoning. We propose a research agenda aimed at bridging these gaps by integrating structured reasoning frameworks, hybrid AI approaches, and human-in-the-loop methodologies. By critically evaluating the reasoning mechanisms of LLMs, this paper contributes to the ongoing discourse on the future of artificial general intelligence and the development of more robust, trustworthy AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00651v1">Open-Source LLM-Driven Federated Transformer for Predictive IoV Management</a></div>
    <div class="paper-meta">
      📅 2025-05-01
      | 💬 Preprint version; submitted for academic peer review
    </div>
    <details class="paper-abstract">
      The proliferation of connected vehicles within the Internet of Vehicles (IoV) ecosystem presents critical challenges in ensuring scalable, real-time, and privacy-preserving traffic management. Existing centralized IoV solutions often suffer from high latency, limited scalability, and reliance on proprietary Artificial Intelligence (AI) models, creating significant barriers to widespread deployment, particularly in dynamic and privacy-sensitive environments. Meanwhile, integrating Large Language Models (LLMs) in vehicular systems remains underexplored, especially concerning prompt optimization and effective utilization in federated contexts. To address these challenges, we propose the Federated Prompt-Optimized Traffic Transformer (FPoTT), a novel framework that leverages open-source LLMs for predictive IoV management. FPoTT introduces a dynamic prompt optimization mechanism that iteratively refines textual prompts to enhance trajectory prediction. The architecture employs a dual-layer federated learning paradigm, combining lightweight edge models for real-time inference with cloud-based LLMs to retain global intelligence. A Transformer-driven synthetic data generator is incorporated to augment training with diverse, high-fidelity traffic scenarios in the Next Generation Simulation (NGSIM) format. Extensive evaluations demonstrate that FPoTT, utilizing EleutherAI Pythia-1B, achieves 99.86% prediction accuracy on real-world data while maintaining high performance on synthetic datasets. These results underscore the potential of open-source LLMs in enabling secure, adaptive, and scalable IoV management, offering a promising alternative to proprietary solutions in smart mobility ecosystems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08067v4">Reward-Augmented Data Enhances Direct Preference Alignment of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-01
      | 💬 Published at ICML 2025
    </div>
    <details class="paper-abstract">
      Preference alignment in Large Language Models (LLMs) has significantly improved their ability to adhere to human instructions and intentions. However, existing direct alignment algorithms primarily focus on relative preferences and often overlook the qualitative aspects of responses, despite having access to preference data that includes reward scores from judge models during AI feedback. Striving to maximize the implicit reward gap between the chosen and the slightly inferior rejected responses can cause overfitting and unnecessary unlearning of the high-quality rejected responses. The unawareness of the reward scores also drives the LLM to indiscriminately favor the low-quality chosen responses and fail to generalize to optimal responses that are sparse in data. To overcome these shortcomings, our study introduces reward-conditioned LLM policies that discern and learn from the entire spectrum of response quality within the dataset, helping extrapolate to more optimal regions. We propose an effective yet simple data relabeling method that conditions the preference pairs on quality scores to construct a reward-augmented dataset. The experiments across various benchmarks and diverse models demonstrate that our approach consistently boosts DPO by a considerable margin. Through comprehensive ablation studies, we demonstrate that our method not only maximizes the utility of preference data but also mitigates the issue of unlearning, demonstrating its broad effectiveness beyond mere data expansion. Our code is available at https://github.com/shenao-zhang/reward-augmented-preference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00626v1">The Illusion of Role Separation: Hidden Shortcuts in LLM Role Learning (and How to Fix Them)</a></div>
    <div class="paper-meta">
      📅 2025-05-01
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) that integrate multiple input roles (e.g., system instructions, user queries, external tool outputs) are increasingly prevalent in practice. Ensuring that the model accurately distinguishes messages from each role -- a concept we call \emph{role separation} -- is crucial for consistent multi-role behavior. Although recent work often targets state-of-the-art prompt injection defenses, it remains unclear whether such methods truly teach LLMs to differentiate roles or merely memorize known triggers. In this paper, we examine \emph{role-separation learning}: the process of teaching LLMs to robustly distinguish system and user tokens. Through a \emph{simple, controlled experimental framework}, we find that fine-tuned models often rely on two proxies for role identification: (1) task type exploitation, and (2) proximity to begin-of-text. Although data augmentation can partially mitigate these shortcuts, it generally leads to iterative patching rather than a deeper fix. To address this, we propose reinforcing \emph{invariant signals} that mark role boundaries by adjusting token-wise cues in the model's input encoding. In particular, manipulating position IDs helps the model learn clearer distinctions and reduces reliance on superficial proxies. By focusing on this mechanism-centered perspective, our work illuminates how LLMs can more reliably maintain consistent multi-role behavior without merely memorizing known prompts or triggers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00610v1">Combining LLMs with Logic-Based Framework to Explain MCTS</a></div>
    <div class="paper-meta">
      📅 2025-05-01
      | 💬 Accepted by AAMAS-25 as an extended abstract
    </div>
    <details class="paper-abstract">
      In response to the lack of trust in Artificial Intelligence (AI) for sequential planning, we design a Computational Tree Logic-guided large language model (LLM)-based natural language explanation framework designed for the Monte Carlo Tree Search (MCTS) algorithm. MCTS is often considered challenging to interpret due to the complexity of its search trees, but our framework is flexible enough to handle a wide range of free-form post-hoc queries and knowledge-based inquiries centered around MCTS and the Markov Decision Process (MDP) of the application domain. By transforming user queries into logic and variable statements, our framework ensures that the evidence obtained from the search tree remains factually consistent with the underlying environmental dynamics and any constraints in the actual stochastic control process. We evaluate the framework rigorously through quantitative assessments, where it demonstrates strong performance in terms of accuracy and factual consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00603v1">Can LLMs Help Improve Analogical Reasoning For Strategic Decisions? Experimental Evidence from Humans and GPT-4</a></div>
    <div class="paper-meta">
      📅 2025-05-01
    </div>
    <details class="paper-abstract">
      This study investigates whether large language models, specifically GPT4, can match human capabilities in analogical reasoning within strategic decision making contexts. Using a novel experimental design involving source to target matching, we find that GPT4 achieves high recall by retrieving all plausible analogies but suffers from low precision, frequently applying incorrect analogies based on superficial similarities. In contrast, human participants exhibit high precision but low recall, selecting fewer analogies yet with stronger causal alignment. These findings advance theory by identifying matching, the evaluative phase of analogical reasoning, as a distinct step that requires accurate causal mapping beyond simple retrieval. While current LLMs are proficient in generating candidate analogies, humans maintain a comparative advantage in recognizing deep structural similarities across domains. Error analysis reveals that AI errors arise from surface level matching, whereas human errors stem from misinterpretations of causal structure. Taken together, the results suggest a productive division of labor in AI assisted organizational decision making where LLMs may serve as broad analogy generators, while humans act as critical evaluators, applying the most contextually appropriate analogies to strategic problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20984v3">UoR-NCL at SemEval-2025 Task 1: Using Generative LLMs and CLIP Models for Multilingual Multimodal Idiomaticity Representation</a></div>
    <div class="paper-meta">
      📅 2025-05-01
    </div>
    <details class="paper-abstract">
      SemEval-2025 Task 1 focuses on ranking images based on their alignment with a given nominal compound that may carry idiomatic meaning in both English and Brazilian Portuguese. To address this challenge, this work uses generative large language models (LLMs) and multilingual CLIP models to enhance idiomatic compound representations. LLMs generate idiomatic meanings for potentially idiomatic compounds, enriching their semantic interpretation. These meanings are then encoded using multilingual CLIP models, serving as representations for image ranking. Contrastive learning and data augmentation techniques are applied to fine-tune these embeddings for improved performance. Experimental results show that multimodal representations extracted through this method outperformed those based solely on the original nominal compounds. The fine-tuning approach shows promising outcomes but is less effective than using embeddings without fine-tuning. The source code used in this paper is available at https://github.com/tongwu17/SemEval-2025-Task1-UoR-NCL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00557v1">Triggering Hallucinations in LLMs: A Quantitative Study of Prompt-Induced Hallucination in Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-05-01
    </div>
    <details class="paper-abstract">
      Hallucinations in large language models (LLMs) present a growing challenge across real-world applications, from healthcare to law, where factual reliability is essential. Despite advances in alignment and instruction tuning, LLMs can still generate outputs that are fluent yet fundamentally untrue. Understanding the cognitive dynamics that underlie these hallucinations remains an open problem. In this study, we propose a prompt-based framework to systematically trigger and quantify hallucination: a Hallucination-Inducing Prompt (HIP), which synthetically fuses semantically distant concepts (e.g., periodic table of elements and tarot divination) in a misleading way, and a Hallucination Quantifying Prompt (HQP), which scores the plausibility, confidence, and coherence of the output. Controlled experiments across multiple LLMs revealed that HIPs consistently produced less coherent and more hallucinated responses than their null-fusion controls. These effects varied across models, with reasoning-oriented LLMs showing distinct profiles from general-purpose ones. Our framework provides a reproducible testbed for studying hallucination vulnerability, and opens the door to developing safer, more introspective LLMs that can detect and self-regulate the onset of conceptual instability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20119v2">Can LLMs Be Trusted for Evaluating RAG Systems? A Survey of Methods and Datasets</a></div>
    <div class="paper-meta">
      📅 2025-05-01
      | 💬 8 Pages. This paper has been accepted for presentation at the IEEE Swiss Conference on Data Science (SDS25)
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) has advanced significantly in recent years. The complexity of RAG systems, which involve multiple components-such as indexing, retrieval, and generation-along with numerous other parameters, poses substantial challenges for systematic evaluation and quality enhancement. Previous research highlights that evaluating RAG systems is essential for documenting advancements, comparing configurations, and identifying effective approaches for domain-specific applications. This study systematically reviews 63 academic articles to provide a comprehensive overview of state-of-the-art RAG evaluation methodologies, focusing on four key areas: datasets, retrievers, indexing and databases, and the generator component. We observe the feasibility of an automated evaluation approach for each component of a RAG system, leveraging an LLM capable of both generating evaluation datasets and conducting evaluations. In addition, we found that further practical research is essential to provide companies with clear guidance on the do's and don'ts of implementing and evaluating RAG systems. By synthesizing evaluation approaches for key RAG components and emphasizing the creation and adaptation of domain-specific datasets for benchmarking, we contribute to the advancement of systematic evaluation methods and the improvement of evaluation rigor for RAG systems. Furthermore, by examining the interplay between automated approaches leveraging LLMs and human judgment, we contribute to the ongoing discourse on balancing automation and human input, clarifying their respective contributions, limitations, and challenges in achieving robust and reliable evaluations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.08532v3">EvoPrompt: Connecting LLMs with Evolutionary Algorithms Yields Powerful Prompt Optimizers</a></div>
    <div class="paper-meta">
      📅 2025-05-01
      | 💬 International Conference on Learning Representations (ICLR) 2024
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel in various tasks, but they rely on carefully crafted prompts that often demand substantial human effort. To automate this process, in this paper, we propose a novel framework for discrete prompt optimization, called EvoPrompt, which borrows the idea of evolutionary algorithms (EAs) as they exhibit good performance and fast convergence. To enable EAs to work on discrete prompts, which are natural language expressions that need to be coherent and human-readable, we connect LLMs with EAs. This approach allows us to simultaneously leverage the powerful language processing capabilities of LLMs and the efficient optimization performance of EAs. Specifically, abstaining from any gradients or parameters, EvoPrompt starts from a population of prompts and iteratively generates new prompts with LLMs based on the evolutionary operators, improving the population based on the development set. We optimize prompts for both closed- and open-source LLMs including GPT-3.5 and Alpaca, on 31 datasets covering language understanding, generation tasks, as well as BIG-Bench Hard (BBH) tasks. EvoPrompt significantly outperforms human-engineered prompts and existing methods for automatic prompt generation (e.g., up to 25% on BBH). Furthermore, EvoPrompt demonstrates that connecting LLMs with EAs creates synergies, which could inspire further research on the combination of LLMs and conventional algorithms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.21036v2">Can Differentially Private Fine-tuning LLMs Protect Against Privacy Attacks?</a></div>
    <div class="paper-meta">
      📅 2025-05-01
      | 💬 accepted by DBSec25
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) has become an essential strategy for adapting them to specialized tasks; however, this process introduces significant privacy challenges, as sensitive training data may be inadvertently memorized and exposed. Although differential privacy (DP) offers strong theoretical guarantees against such leakage, its empirical privacy effectiveness on LLMs remains unclear, especially under different fine-tuning methods. In this paper, we systematically investigate the impact of DP across fine-tuning methods and privacy budgets, using both data extraction and membership inference attacks to assess empirical privacy risks. Our main findings are as follows: (1) Differential privacy reduces model utility, but its impact varies significantly across different fine-tuning methods. (2) Without DP, the privacy risks of models fine-tuned with different approaches differ considerably. (3) When DP is applied, even a relatively high privacy budget can substantially lower privacy risk. (4) The privacy-utility trade-off under DP training differs greatly among fine-tuning methods, with some methods being unsuitable for DP due to severe utility degradation. Our results provide practical guidance for privacy-conscious deployment of LLMs and pave the way for future research on optimizing the privacy-utility trade-off in fine-tuning methodologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03621v3">Network-aided Efficient LLM Services With Denoising-inspired Prompt Compression</a></div>
    <div class="paper-meta">
      📅 2025-05-01
      | 💬 arXiv admin note: substantial text overlap with arXiv:2411.18010
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities in various tasks, leading to their increasing adoption in diverse services delivered through wireless networks. There is a growing trend toward longer prompts to better leverage LLMs' capabilities and address difficult tasks. However, longer prompts not only increase data transmission costs but also require more computing resources and processing time, which impacts overall system efficiency and user experience. To address this challenge, we propose Joint Power and Prompt Optimization (JPPO), a framework that combines Small Language Model (SLM)-based prompt compression with wireless power allocation optimization. By deploying SLM at edge devices for prompt compression and employing Deep Reinforcement Learning (DRL) for joint optimization of compression ratio and transmission power, JPPO effectively balances service quality with resource efficiency. Furthermore, inspired by denoising diffusion models, we design a denoising-inspired prompt compression approach that iteratively compresses prompts by gradually removing non-critical information, further enhancing the framework's performance. Experimental results with long prompt tokens demonstrate that our framework achieves high service fidelity while optimizing power usage in wireless LLM services, significantly reducing the total service response time. With our DRL-based JPPO, the framework maintains fidelity comparable to the no-compression baseline while still achieving a 17% service time reduction through adaptive compression.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00368v1">Urban Air Mobility as a System of Systems: An LLM-Enhanced Holonic Approach</a></div>
    <div class="paper-meta">
      📅 2025-05-01
    </div>
    <details class="paper-abstract">
      Urban Air Mobility (UAM) is an emerging System of System (SoS) that faces challenges in system architecture, planning, task management, and execution. Traditional architectural approaches struggle with scalability, adaptability, and seamless resource integration within dynamic and complex environments. This paper presents an intelligent holonic architecture that incorporates Large Language Model (LLM) to manage the complexities of UAM. Holons function semi autonomously, allowing for real time coordination among air taxis, ground transport, and vertiports. LLMs process natural language inputs, generate adaptive plans, and manage disruptions such as weather changes or airspace closures.Through a case study of multimodal transportation with electric scooters and air taxis, we demonstrate how this architecture enables dynamic resource allocation, real time replanning, and autonomous adaptation without centralized control, creating more resilient and efficient urban transportation networks. By advancing decentralized control and AI driven adaptability, this work lays the groundwork for resilient, human centric UAM ecosystems, with future efforts targeting hybrid AI integration and real world validation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00342v1">LLMPrism: Black-box Performance Diagnosis for Production LLM Training Platforms</a></div>
    <div class="paper-meta">
      📅 2025-05-01
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have brought about revolutionary changes in diverse fields, rendering LLM training of utmost importance for modern enterprises. To meet this demand, multi-tenant large-scale LLM training platforms have been built to offer LLM training services. Nevertheless, due to the complexity and synchronous nature of LLM training process, performance issues occur frequently and can result in substantial resource wastage. The limited visibility from the perspective of platform providers impedes existing profiling methods and poses challenges to the monitoring and diagnosis of the performance of LLM training jobs. For the first time, this paper proposes the utilization of underlying network flow data to reconstruct the training timelines of jobs based on the distinct characteristics in the LLM training procedure. We design LLMPrism, the first black-box performance diagnosis system for LLM training platforms. By progressively recognizing LLM training jobs, identifying their parallelism strategies, and reconstructing the training timelines, LLMPrism achieves non-intrusive, lightweight, and continuous monitoring of LLM training systems. Leveraging this monitoring capability, it further effectively diagnoses potential performance issues. Since Oct. 2024, LLMPrism has been deployed on our large-scale production Platform-X, in which the evaluations and deployment experiences demonstrate that LLMPrism can achieve accurate timeline reconstruction with an error within 0.3% and effectively diagnose various performance issues.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15546v2">A Framework for Testing and Adapting REST APIs as LLM Tools</a></div>
    <div class="paper-meta">
      📅 2025-05-01
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are enabling autonomous agents to perform complex workflows using external tools or functions, often provided via REST APIs in enterprise systems. However, directly utilizing these APIs as tools poses challenges due to their complex input schemas, elaborate responses, and often ambiguous documentation. Current benchmarks for tool testing do not adequately address these complexities, leading to a critical gap in evaluating API readiness for agent-driven automation. In this work, we present a novel testing framework aimed at evaluating and enhancing the readiness of REST APIs to function as tools for LLM-based agents. Our framework transforms apis as tools, generates comprehensive test cases for the APIs, translates tests cases into natural language instructions suitable for agents, enriches tool definitions and evaluates the agent's ability t correctly invoke the API and process its inputs and responses. To provide actionable insights, we analyze the outcomes of 750 test cases, presenting a detailed taxonomy of errors, including input misinterpretation, output handling inconsistencies, and schema mismatches. Additionally, we classify these test cases to streamline debugging and refinement of tool integrations. This work offers a foundational step toward enabling enterprise APIs as tools, improving their usability in agent-based applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16473v2">TerEffic: Highly Efficient Ternary LLM Inference on FPGA</a></div>
    <div class="paper-meta">
      📅 2025-05-01
    </div>
    <details class="paper-abstract">
      Deploying Large Language Models (LLMs) efficiently on edge devices is often constrained by limited memory capacity and high power consumption. Low-bit quantization methods, particularly ternary quantization, have demonstrated significant potential in preserving model accuracy while substantially decreasing memory footprint and computational costs. However, existing general-purpose architectures and accelerators have not fully exploited the advantages of low-bit quantization due to insufficient specialized hardware support. We introduce TerEffic, an FPGA-based architecture tailored for ternary-quantized LLM inference. The proposed system offers flexibility through reconfigurable hardware to meet various system requirements. We evaluated two representative configurations: a fully on-chip design that stores all weights within on-chip memories, scaling out using multiple FPGAs, and an HBM-assisted design capable of accommodating larger models on a single FPGA board. Experimental results demonstrate significant performance and energy efficiency improvements. For single-batch inference on a 370 M-parameter model, our fully on-chip architecture achieves 16,300 tokens/second, delivering a throughput 192 times higher than NVIDIA Jetson Orin Nano with a power efficiency of 455 tokens/second/W, marking a 19-fold improvement. The HBM-assisted architecture processes 727 tokens/second for a larger 2.7B-parameter model, which is 3 times of the throughput of NVIDIA A100, while consuming only 46W, resulting in a power efficiency of 16 tokens/second/W, an 8-fold improvement over the A100.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.04532v3">QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-05-01
      | 💬 The first three authors contribute equally to this project and are listed in the alphabetical order. Yujun Lin leads the quantization algorithm, Haotian Tang and Shang Yang lead the GPU kernels and the serving system. Code is available at https://github.com/mit-han-lab/omniserve
    </div>
    <details class="paper-abstract">
      Quantization can accelerate large language model (LLM) inference. Going beyond INT8 quantization, the research community is actively exploring even lower precision, such as INT4. Nonetheless, state-of-the-art INT4 quantization techniques only accelerate low-batch, edge LLM inference, failing to deliver performance gains in large-batch, cloud-based LLM serving. We uncover a critical issue: existing INT4 quantization methods suffer from significant runtime overhead (20-90%) when dequantizing either weights or partial sums on GPUs. To address this challenge, we introduce QoQ, a W4A8KV4 quantization algorithm with 4-bit weight, 8-bit activation, and 4-bit KV cache. QoQ stands for quattuor-octo-quattuor, which represents 4-8-4 in Latin. QoQ is implemented by the QServe inference library that achieves measured speedup. The key insight driving QServe is that the efficiency of LLM serving on GPUs is critically influenced by operations on low-throughput CUDA cores. Building upon this insight, in QoQ algorithm, we introduce progressive quantization that can allow low dequantization overhead in W4A8 GEMM. Additionally, we develop SmoothAttention to effectively mitigate the accuracy degradation incurred by 4-bit KV quantization. In the QServe system, we perform compute-aware weight reordering and take advantage of register-level parallelism to reduce dequantization latency. We also make fused attention memory-bound, harnessing the performance gain brought by KV4 quantization. As a result, QServe improves the maximum achievable serving throughput of Llama-3-8B by 1.2x on A100, 1.4x on L40S; and Qwen1.5-72B by 2.4x on A100, 3.5x on L40S, compared to TensorRT-LLM. Remarkably, QServe on L40S GPU can achieve even higher throughput than TensorRT-LLM on A100. Thus, QServe effectively reduces the dollar cost of LLM serving by 3x. Code is available at https://github.com/mit-han-lab/omniserve.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00240v1">LLM-Based Threat Detection and Prevention Framework for IoT Ecosystems</a></div>
    <div class="paper-meta">
      📅 2025-05-01
      | 💬 Preprint version; submitted for academic peer review
    </div>
    <details class="paper-abstract">
      The increasing complexity and scale of the Internet of Things (IoT) have made security a critical concern. This paper presents a novel Large Language Model (LLM)-based framework for comprehensive threat detection and prevention in IoT environments. The system integrates lightweight LLMs fine-tuned on IoT-specific datasets (IoT-23, TON_IoT) for real-time anomaly detection and automated, context-aware mitigation strategies optimized for resource-constrained devices. A modular Docker-based deployment enables scalable and reproducible evaluation across diverse network conditions. Experimental results in simulated IoT environments demonstrate significant improvements in detection accuracy, response latency, and resource efficiency over traditional security methods. The proposed framework highlights the potential of LLM-driven, autonomous security solutions for future IoT ecosystems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.08498v5">"Reasoning" with Rhetoric: On the Style-Evidence Tradeoff in LLM-Generated Counter-Arguments</a></div>
    <div class="paper-meta">
      📅 2025-05-01
      | 💬 22 pages, 9 figures, 13 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) play a key role in generating evidence-based and stylistic counter-arguments, yet their effectiveness in real-world applications has been underexplored. Previous research often neglects the balance between evidentiality and style, which are crucial for persuasive arguments. To address this, we evaluated the effectiveness of stylized evidence-based counter-argument generation in Counterfire, a new dataset of 38,000 counter-arguments generated by revising counter-arguments to Reddit's ChangeMyView community to follow different discursive styles. We evaluated generic and stylized counter-arguments from basic and fine-tuned models such as GPT-3.5, PaLM-2, and Koala-13B, as well as newer models (GPT-4o, Claude Haiku, LLaMA-3.1) focusing on rhetorical quality and persuasiveness. Our findings reveal that humans prefer stylized counter-arguments over the original outputs, with GPT-3.5 Turbo performing well, though still not reaching human standards of rhetorical quality nor persuasiveness. Additionally, our work created a novel argument triplets dataset for studying style control, with human preference labels that provide insights into the tradeoffs between evidence integration and argument quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00234v1">Self-Generated In-Context Examples Improve LLM Agents for Sequential Decision-Making Tasks</a></div>
    <div class="paper-meta">
      📅 2025-05-01
    </div>
    <details class="paper-abstract">
      Many methods for improving Large Language Model (LLM) agents for sequential decision-making tasks depend on task-specific knowledge engineering--such as prompt tuning, curated in-context examples, or customized observation and action spaces. Using these approaches, agent performance improves with the quality or amount of knowledge engineering invested. Instead, we investigate how LLM agents can automatically improve their performance by learning in-context from their own successful experiences on similar tasks. Rather than relying on task-specific knowledge engineering, we focus on constructing and refining a database of self-generated examples. We demonstrate that even a naive accumulation of successful trajectories across training tasks boosts test performance on three benchmarks: ALFWorld (73% to 89%), Wordcraft (55% to 64%), and InterCode-SQL (75% to 79%)--matching the performance the initial agent achieves if allowed two to three attempts per task. We then introduce two extensions: (1) database-level selection through population-based training to identify high-performing example collections, and (2) exemplar-level selection that retains individual trajectories based on their empirical utility as in-context examples. These extensions further enhance performance, achieving 91% on ALFWorld--matching more complex approaches that employ task-specific components and prompts. Our results demonstrate that automatic trajectory database construction offers a compelling alternative to labor-intensive knowledge engineering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07963v2">Caught in the Web of Words: Do LLMs Fall for Spin in Medical Literature?</a></div>
    <div class="paper-meta">
      📅 2025-05-01
      | 💬 22 pages, 12 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Medical research faces well-documented challenges in translating novel treatments into clinical practice. Publishing incentives encourage researchers to present "positive" findings, even when empirical results are equivocal. Consequently, it is well-documented that authors often spin study results, especially in article abstracts. Such spin can influence clinician interpretation of evidence and may affect patient care decisions. In this study, we ask whether the interpretation of trial results offered by Large Language Models (LLMs) is similarly affected by spin. This is important since LLMs are increasingly being used to trawl through and synthesize published medical evidence. We evaluated 22 LLMs and found that they are across the board more susceptible to spin than humans. They might also propagate spin into their outputs: We find evidence, e.g., that LLMs implicitly incorporate spin into plain language summaries that they generate. We also find, however, that LLMs are generally capable of recognizing spin, and can be prompted in a way to mitigate spin's impact on LLM outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08954v2">Should you use LLMs to simulate opinions? Quality checks for early-stage deliberation</a></div>
    <div class="paper-meta">
      📅 2025-05-01
    </div>
    <details class="paper-abstract">
      The emergent capabilities of large language models (LLMs) have sparked interest in assessing their ability to simulate human opinions in a variety of contexts, potentially serving as surrogates for human subjects in opinion surveys. However, previous evaluations of this capability have depended heavily on costly, domain-specific human survey data, and mixed empirical results about LLM effectiveness create uncertainty for managers about whether investing in this technology is justified in early-stage research. To address these challenges, we introduce a series of quality checks to support early-stage deliberation about the viability of using LLMs for simulating human opinions. These checks emphasize logical constraints, model stability, and alignment with stakeholder expectations of model outputs, thereby reducing dependence on human-generated data in the initial stages of evaluation. We demonstrate the usefulness of the proposed quality control tests in the context of AI-assisted content moderation, an application that both advocates and critics of LLMs' capabilities to simulate human opinion see as a desirable potential use case. None of the tested models passed all quality control checks, revealing several failure modes. We conclude by discussing implications of these failure modes and recommend how organizations can utilize our proposed tests for prompt engineering and in their risk management practices when considering the use of LLMs for opinion simulation. We make our crowdsourced dataset of claims with human and LLM annotations publicly available for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00903v1">NeMo-Inspector: A Visualization Tool for LLM Generation Analysis</a></div>
    <div class="paper-meta">
      📅 2025-05-01
      | 💬 Presented at the NAACL 2025 conference
    </div>
    <details class="paper-abstract">
      Adapting Large Language Models (LLMs) to novel tasks and enhancing their overall capabilities often requires large, high-quality training datasets. Synthetic data, generated at scale, serves a valuable alternative when real-world data is scarce or difficult to obtain. However, ensuring the quality of synthetic datasets is challenging, as developers must manually inspect and refine numerous samples to identify errors and areas for improvement. This process is time-consuming and requires specialized tools. We introduce NeMo-Inspector, an open-source tool designed to simplify the analysis of synthetic datasets with integrated inference capabilities. We demonstrate its effectiveness through two real-world cases. Analysis and cleaning of the synthetically generated GSM-Plus dataset with NeMo-Inspector led to a significant decrease in low-quality samples from 46.99% to 19.51%. The tool also helped identify and correct generation errors in OpenMath models, improving accuracy by 1.92% on the MATH dataset and by 4.17% on the GSM8K dataset for a Meta-Llama-3-8B model fine-tuned on synthetic data generated from Nemotron-4-340B.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00886v1">Towards Explainable Temporal User Profiling with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-01
    </div>
    <details class="paper-abstract">
      Accurately modeling user preferences is vital not only for improving recommendation performance but also for enhancing transparency in recommender systems. Conventional user profiling methods, such as averaging item embeddings, often overlook the evolving, nuanced nature of user interests, particularly the interplay between short-term and long-term preferences. In this work, we leverage large language models (LLMs) to generate natural language summaries of users' interaction histories, distinguishing recent behaviors from more persistent tendencies. Our framework not only models temporal user preferences but also produces natural language profiles that can be used to explain recommendations in an interpretable manner. These textual profiles are encoded via a pre-trained model, and an attention mechanism dynamically fuses the short-term and long-term embeddings into a comprehensive user representation. Beyond boosting recommendation accuracy over multiple baselines, our approach naturally supports explainability: the interpretable text summaries and attention weights can be exposed to end users, offering insights into why specific items are suggested. Experiments on real-world datasets underscore both the performance gains and the promise of generating clearer, more transparent justifications for content-based recommendations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00875v1">Thoughts without Thinking: Reconsidering the Explanatory Value of Chain-of-Thought Reasoning in LLMs through Agentic Pipelines</a></div>
    <div class="paper-meta">
      📅 2025-05-01
    </div>
    <details class="paper-abstract">
      Agentic pipelines present novel challenges and opportunities for human-centered explainability. The HCXAI community is still grappling with how best to make the inner workings of LLMs transparent in actionable ways. Agentic pipelines consist of multiple LLMs working in cooperation with minimal human control. In this research paper, we present early findings from an agentic pipeline implementation of a perceptive task guidance system. Through quantitative and qualitative analysis, we analyze how Chain-of-Thought (CoT) reasoning, a common vehicle for explainability in LLMs, operates within agentic pipelines. We demonstrate that CoT reasoning alone does not lead to better outputs, nor does it offer explainability, as it tends to produce explanations without explainability, in that they do not improve the ability of end users to better understand systems or achieve their goals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00853v1">LLM Ethics Benchmark: A Three-Dimensional Assessment System for Evaluating Moral Reasoning in Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-05-01
    </div>
    <details class="paper-abstract">
      This study establishes a novel framework for systematically evaluating the moral reasoning capabilities of large language models (LLMs) as they increasingly integrate into critical societal domains. Current assessment methodologies lack the precision needed to evaluate nuanced ethical decision-making in AI systems, creating significant accountability gaps. Our framework addresses this challenge by quantifying alignment with human ethical standards through three dimensions: foundational moral principles, reasoning robustness, and value consistency across diverse scenarios. This approach enables precise identification of ethical strengths and weaknesses in LLMs, facilitating targeted improvements and stronger alignment with societal values. To promote transparency and collaborative advancement in ethical AI development, we are publicly releasing both our benchmark datasets and evaluation codebase at https://github.com/ The-Responsible-AI-Initiative/LLM_Ethics_Benchmark.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00850v1">ICQuant: Index Coding enables Low-bit LLM Quantization</a></div>
    <div class="paper-meta">
      📅 2025-05-01
    </div>
    <details class="paper-abstract">
      The rapid deployment of Large Language Models (LLMs) highlights the need for efficient low-bit post-training quantization (PTQ), due to their high memory costs. A key challenge in weight quantization is the presence of outliers, which inflate quantization ranges and lead to large errors. While a number of outlier suppression techniques have been proposed, they either: fail to effectively shrink the quantization range, or incur (relatively) high bit overhead. In this paper, we present ICQuant, a novel framework that leverages outlier statistics to design an efficient index coding scheme for outlier-aware weight-only quantization. Compared to existing outlier suppression techniques requiring $\approx 1$ bit overhead to halve the quantization range, ICQuant requires only $\approx 0.3$ bits; a significant saving in extreme compression regimes (e.g., 2-3 bits per weight). ICQuant can be used on top of any existing quantizers to eliminate outliers, improving the quantization quality. Using just 2.3 bits per weight and simple scalar quantizers, ICQuant improves the zero-shot accuracy of the 2-bit Llama3-70B model by up to 130% and 150% relative to QTIP and QuIP#; and it achieves comparable performance to the best-known fine-tuned quantizer (PV-tuning) without fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00831v1">SmallPlan: Leverage Small Language Models for Sequential Path Planning with Simulation-Powered, LLM-Guided Distillation</a></div>
    <div class="paper-meta">
      📅 2025-05-01
      | 💬 Paper is under review
    </div>
    <details class="paper-abstract">
      Efficient path planning in robotics, particularly within large-scale, dynamic environments, remains a significant hurdle. While Large Language Models (LLMs) offer strong reasoning capabilities, their high computational cost and limited adaptability in dynamic scenarios hinder real-time deployment on edge devices. We present SmallPlan -- a novel framework leveraging LLMs as teacher models to train lightweight Small Language Models (SLMs) for high-level path planning tasks. In SmallPlan, the SLMs provide optimal action sequences to navigate across scene graphs that compactly represent full-scaled 3D scenes. The SLMs are trained in a simulation-powered, interleaved manner with LLM-guided supervised fine-tuning (SFT) and reinforcement learning (RL). This strategy not only enables SLMs to successfully complete navigation tasks but also makes them aware of important factors like travel distance and number of trials. Through experiments, we demonstrate that the fine-tuned SLMs perform competitively with larger models like GPT-4o on sequential path planning, without suffering from hallucination and overfitting. SmallPlan is resource-efficient, making it well-suited for edge-device deployment and advancing practical autonomous robotics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.01456v1">Unlearning Sensitive Information in Multimodal LLMs: Benchmark and Attack-Defense Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-05-01
      | 💬 The dataset and code are publicly available at https://github.com/Vaidehi99/UnLOK-VQA
    </div>
    <details class="paper-abstract">
      LLMs trained on massive datasets may inadvertently acquire sensitive information such as personal details and potentially harmful content. This risk is further heightened in multimodal LLMs as they integrate information from multiple modalities (image and text). Adversaries can exploit this knowledge through multimodal prompts to extract sensitive details. Evaluating how effectively MLLMs can forget such information (targeted unlearning) necessitates the creation of high-quality, well-annotated image-text pairs. While prior work on unlearning has focused on text, multimodal unlearning remains underexplored. To address this gap, we first introduce a multimodal unlearning benchmark, UnLOK-VQA (Unlearning Outside Knowledge VQA), as well as an attack-and-defense framework to evaluate methods for deleting specific multimodal knowledge from MLLMs. We extend a visual question-answering dataset using an automated pipeline that generates varying-proximity samples for testing generalization and specificity, followed by manual filtering for maintaining high quality. We then evaluate six defense objectives against seven attacks (four whitebox, three blackbox), including a novel whitebox method leveraging interpretability of hidden states. Our results show multimodal attacks outperform text- or image-only ones, and that the most effective defense removes answer information from internal model states. Additionally, larger models exhibit greater post-editing robustness, suggesting that scale enhances safety. UnLOK-VQA provides a rigorous benchmark for advancing unlearning in MLLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03793v1">LENSLLM: Unveiling Fine-Tuning Dynamics for LLM Selection</a></div>
    <div class="paper-meta">
      📅 2025-05-01
      | 💬 It is accepted by ICML'2025, and the code is open-sourcing on https://github.com/Susan571/LENSLLM.git
    </div>
    <details class="paper-abstract">
      The proliferation of open-sourced Large Language Models (LLMs) and diverse downstream tasks necessitates efficient model selection, given the impracticality of fine-tuning all candidates due to computational constraints. Despite the recent advances in LLM selection, a fundamental research question largely remains nascent: how can we model the dynamic behaviors of LLMs during fine-tuning, thereby enhancing our understanding of their generalization performance across diverse downstream tasks? In this work, we propose a novel theoretical framework that provides a proper lens to assess the generalization capabilities of LLMs, thereby enabling accurate and efficient LLM selection for downstream applications. In particular, we first derive a Hessian-based PAC-Bayes generalization bound that unveils fine-tuning dynamics of LLMs and then introduce LENSLLM, a Neural Tangent Kernel(NTK)-based Rectified Scaling Model that enables accurate performance predictions across diverse tasks while maintaining computational efficiency. Extensive empirical results on 3 large-scale benchmarks demonstrate that our model achieves up to 91.1% accuracy and reduces up to 88.5% computational cost in LLM selection, outperforming 5 state-of-the-art methods. We open-source our proposed LENSLLM model and corresponding results at the Github link: https://github.com/Susan571/LENSLLM.git.
    </details>
</div>
