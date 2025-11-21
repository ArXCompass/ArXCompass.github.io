# llm - 2025_11

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- Part 2
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14450v1">Hyperion: Hierarchical Scheduling for Parallel LLM Acceleration in Multi-tier Networks</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly executed across edge, fog, and cloud tiers where limited GPU memory, heterogeneous compute, and variable inter-tier bandwidth jointly constrain deployment and motivate model partitioning and request scheduling. In this setting, achieving low end-to-end latency is governed not only by where a model is deployed (inter-tier model partitioning) but also by how incoming requests are scheduled (intra-tier task scheduling) across heterogeneous nodes. These two problems are tightly coupled, as a suboptimal scheduler can negate the benefits of a good partition, and vice versa. In this paper, we propose Hyperion, a hierarchical two-stage framework that jointly optimizes partitioning and scheduling to minimize end-to-end latency for pipelined LLM inference in multi-tier networks, balancing compute and memory across tiers while introducing negligible runtime overhead and requiring no model retraining. Motivated by the observation that partition choices evolve on slower timescales than request arrivals, Stage 1 performs offline, inter-tier partitioning via a Binary Search with Dynamic Programming (BSDP) procedure to produce balanced stage times under tier capacity and memory constraints; to adapt to time-varying load, Stage 2 performs online, intra-tier scheduling with a lightweight Adaptive Real-time Task Scheduling (ARTS) algorithm that maps each request to the best available node using real-time estimates of queue length and effective capacity. Experimental results on multi-tier inference tasks demonstrate that Hyperion significantly reduces end-to-end latency by up to 52.1\% and 31.2\%, with the Phi-3-medium model, compared to the GPipe and HEFT baselines, respectively. Furthermore, Hyperion shows superior scalability in long-sequence generation, maintaining a 44.5\% lower latency than GPipe and achieving higher GPU utilization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.03628v3">LLMDistill4Ads: Using Cross-Encoders to Distill from LLM Signals for Advertiser Keyphrase Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      E-commerce sellers are advised to bid on keyphrases to boost their advertising campaigns. These keyphrases must be relevant to prevent irrelevant items from cluttering search systems and to maintain positive seller perception. It is vital that keyphrase suggestions align with seller, search and buyer judgments. Given the challenges in collecting negative feedback in these systems, LLMs have been used as a scalable proxy to human judgments. This paper presents an empirical study on a major ecommerce platform of a distillation framework involving an LLM teacher, a cross-encoder assistant and a bi-encoder Embedding Based Retrieval (EBR) student model, aimed at mitigating click-induced biases in keyphrase recommendations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14445v1">Tell Me: An LLM-powered Mental Well-being Assistant with RAG, Synthetic Dialogue Generation, and Agentic Planning</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 8 pages, 2 figures, 1 Table. Submitted to the Computation and Language (cs.CL) category. Uses the ACL-style template. Code and demo will be released at: https://github.com/trystine/Tell_Me_Mental_Wellbeing_System
    </div>
    <details class="paper-abstract">
      We present Tell Me, a mental well-being system that leverages advances in large language models to provide accessible, context-aware support for users and researchers. The system integrates three components: (i) a retrieval-augmented generation (RAG) assistant for personalized, knowledge-grounded dialogue; (ii) a synthetic client-therapist dialogue generator conditioned on client profiles to facilitate research on therapeutic language and data augmentation; and (iii) a Well-being AI crew, implemented with CrewAI, that produces weekly self-care plans and guided meditation audio. The system is designed as a reflective space for emotional processing rather than a substitute for professional therapy. It illustrates how conversational assistants can lower barriers to support, complement existing care, and broaden access to mental health resources. To address the shortage of confidential therapeutic data, we introduce synthetic client-therapist dialogue generation conditioned on client profiles. Finally, the planner demonstrates an innovative agentic workflow for dynamically adaptive, personalized self-care, bridging the limitations of static well-being tools. We describe the architecture, demonstrate its functionalities, and report evaluation of the RAG assistant in curated well-being scenarios using both automatic LLM-based judgments and a human-user study. This work highlights opportunities for interdisciplinary collaboration between NLP researchers and mental health professionals to advance responsible innovation in human-AI interaction for well-being.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14391v1">Enhancing LLM-based Autonomous Driving with Modular Traffic Light and Sign Recognition</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used for decision-making and planning in autonomous driving, showing promising reasoning capabilities and potential to generalize across diverse traffic situations. However, current LLM-based driving agents lack explicit mechanisms to enforce traffic rules and often struggle to reliably detect small, safety-critical objects such as traffic lights and signs. To address this limitation, we introduce TLS-Assist, a modular redundancy layer that augments LLM-based autonomous driving agents with explicit traffic light and sign recognition. TLS-Assist converts detections into structured natural language messages that are injected into the LLM input, enforcing explicit attention to safety-critical cues. The framework is plug-and-play, model-agnostic, and supports both single-view and multi-view camera setups. We evaluate TLS-Assist in a closed-loop setup on the LangAuto benchmark in CARLA. The results demonstrate relative driving performance improvements of up to 14% over LMDrive and 7% over BEVDriver, while consistently reducing traffic light and sign infractions. We publicly release the code and models on https://github.com/iis-esslingen/TLS-Assist.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14359v1">Towards LLM-Based Usability Analysis for Recommender User Interfaces</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 The paper was presented at IntRS'25: Joint Workshop on Interfaces and Human Decision Making for Recommender Systems, September 22, 2025, Prague, Czech Republic and is published in the workshop proceedings: https://ceur-ws.org/Vol-4027/
    </div>
    <details class="paper-abstract">
      Usability is a key factor in the effectiveness of recommender systems. However, the analysis of user interfaces is a time-consuming process that requires expertise. Recent advances in multimodal large language models (LLMs) offer promising opportunities to automate such evaluations. In this work, we explore the potential of multimodal LLMs to assess the usability of recommender system interfaces by considering a variety of publicly available systems as examples. We take user interface screenshots from multiple of these recommender platforms to cover both preference elicitation and recommendation presentation scenarios. An LLM is instructed to analyze these interfaces with regard to different usability criteria and provide explanatory feedback. Our evaluation demonstrates how LLMs can support heuristic-style usability assessments at scale to support the improvement of user experience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2406.08426v7">Next-Generation Database Interfaces: A Survey of LLM-based Text-to-SQL</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 Accepted to IEEE TKDE2025
    </div>
    <details class="paper-abstract">
      Generating accurate SQL from users' natural language questions (text-to-SQL) remains a long-standing challenge due to the complexities involved in user question understanding, database schema comprehension, and SQL generation. Traditional text-to-SQL systems, which combine human engineering and deep neural networks, have made significant progress. Subsequently, pre-trained language models (PLMs) have been developed for text-to-SQL tasks, achieving promising results. However, as modern databases and user questions grow more complex, PLMs with a limited parameter size often produce incorrect SQL. This necessitates more sophisticated and tailored optimization methods, which restricts the application of PLM-based systems. Recently, large language models (LLMs) have shown significant capabilities in natural language understanding as model scale increases. Thus, integrating LLM-based solutions can bring unique opportunities, improvements, and solutions to text-to-SQL research. In this survey, we provide a comprehensive review of existing LLM-based text-to-SQL studies. Specifically, we offer a brief overview of the technical challenges and evolutionary process of text-to-SQL. Next, we introduce the datasets and metrics designed to evaluate text-to-SQL systems. Subsequently, we present a systematic analysis of recent advances in LLM-based text-to-SQL. Finally, we make a summarization and discuss the remaining challenges in this field and suggest expectations for future research directions. All the related resources of LLM-based, including research papers, benchmarks, and open-source projects, are collected for the community in our repository: https://github.com/DEEP-PolyU/Awesome-LLM-based-Text2SQL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14334v1">When Words Change the Model: Sensitivity of LLMs for Constraint Programming Modelling</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      One of the long-standing goals in optimisation and constraint programming is to describe a problem in natural language and automatically obtain an executable, efficient model. Large language models appear to bring this vision closer, showing impressive results in automatically generating models for classical benchmarks. However, much of this apparent success may derive from data contamination rather than genuine reasoning: many standard CP problems are likely included in the training data of these models. To examine this hypothesis, we systematically rephrased and perturbed a set of well-known CSPLib problems to preserve their structure while modifying their context and introducing misleading elements. We then compared the models produced by three representative LLMs across original and modified descriptions. Our qualitative analysis shows that while LLMs can produce syntactically valid and semantically plausible models, their performance drops sharply under contextual and linguistic variation, revealing shallow understanding and sensitivity to wording.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.11858v3">OpeNLGauge: An Explainable Metric for NLG Evaluation with Open-Weights LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 INLG 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated great potential as evaluators of NLG systems, allowing for high-quality, reference-free, and multi-aspect assessments. However, existing LLM-based metrics suffer from two major drawbacks: reliance on proprietary models to generate training data or perform evaluations, and a lack of fine-grained, explanatory feedback. In this paper, we introduce OpeNLGauge, a fully open-source, reference-free NLG evaluation metric that provides accurate explanations based on error spans. OpeNLGauge is available as a two-stage ensemble of larger open-weight LLMs, or as a small fine-tuned evaluation model, with confirmed generalizability to unseen tasks, domains and aspects. Our extensive meta-evaluation shows that OpeNLGauge achieves competitive correlation with human judgments, outperforming state-of-the-art models on certain tasks while maintaining full reproducibility and providing explanations more than twice as accurate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14275v1">Don't Miss the Forest for the Trees: In-Depth Confidence Estimation for LLMs via Reasoning over the Answer Space</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      Knowing the reliability of a model's response is essential in application. With the strong generation capabilities of LLMs, research has focused on generating verbalized confidence. This is further enhanced by combining chain-of-thought reasoning, which provides logical and transparent estimation. However, how reasoning strategies affect the estimated confidence is still under-explored. In this work, we demonstrate that predicting a verbalized probability distribution can effectively encourage in-depth reasoning for confidence estimation. Intuitively, it requires an LLM to consider all candidates within the answer space instead of basing on a single guess, and to carefully assign confidence scores to meet the requirements of a distribution. This method shows an advantage across different models and various tasks, regardless of whether the answer space is known. Its advantage is maintained even after reinforcement learning, and further analysis shows its reasoning patterns are aligned with human expectations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14248v1">Enhancing Regional Airbnb Trend Forecasting Using LLM-Based Embeddings of Accessibility and Human Mobility</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 Accepted at ASONAM 2025
    </div>
    <details class="paper-abstract">
      The expansion of short-term rental platforms, such as Airbnb, has significantly disrupted local housing markets, often leading to increased rental prices and housing affordability issues. Accurately forecasting regional Airbnb market trends can thus offer critical insights for policymakers and urban planners aiming to mitigate these impacts. This study proposes a novel time-series forecasting framework to predict three key Airbnb indicators -- Revenue, Reservation Days, and Number of Reservations -- at the regional level. Using a sliding-window approach, the model forecasts trends 1 to 3 months ahead. Unlike prior studies that focus on individual listings at fixed time points, our approach constructs regional representations by integrating listing features with external contextual factors such as urban accessibility and human mobility. We convert structured tabular data into prompt-based inputs for a Large Language Model (LLM), producing comprehensive regional embeddings. These embeddings are then fed into advanced time-series models (RNN, LSTM, Transformer) to better capture complex spatio-temporal dynamics. Experiments on Seoul's Airbnb dataset show that our method reduces both average RMSE and MAE by approximately 48% compared to conventional baselines, including traditional statistical and machine learning models. Our framework not only improves forecasting accuracy but also offers practical insights for detecting oversupplied regions and supporting data-driven urban policy decisions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14233v1">Visionary Co-Driver: Enhancing Driver Perception of Potential Risks with LLM and HUD</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 Accepted for publication in IEEE Transactions on Intelligent Transportation Systems (T-ITS)
    </div>
    <details class="paper-abstract">
      Drivers' perception of risky situations has always been a challenge in driving. Existing risk-detection methods excel at identifying collisions but face challenges in assessing the behavior of road users in non-collision situations. This paper introduces Visionary Co-Driver, a system that leverages large language models to identify non-collision roadside risks and alert drivers based on their eye movements. Specifically, the system combines video processing algorithms and LLMs to identify potentially risky road users. These risks are dynamically indicated on an adaptive heads-up display interface to enhance drivers' attention. A user study with 41 drivers confirms that Visionary Co-Driver improves drivers' risk perception and supports their recognition of roadside risks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.03758v3">Leveraging LLM-based agents for social science research: insights from citation network simulations</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 accepted by HSSCOMMS'25
    </div>
    <details class="paper-abstract">
      The emergence of Large Language Models (LLMs) demonstrates their potential to encapsulate the logic and patterns inherent in human behavior simulation by leveraging extensive web data pre-training. However, the boundaries of LLM capabilities in social simulation remain unclear. To further explore the social attributes of LLMs, we introduce the CiteAgent framework, designed to generate citation networks based on human-behavior simulation with LLM-based agents. CiteAgent successfully captures predominant phenomena in real-world citation networks, including power-law distribution, citational distortion, and shrinking diameter. Building on this realistic simulation, we establish two LLM-based research paradigms in social science: LLM-SE (LLM-based Survey Experiment) and LLM-LE (LLM-based Laboratory Experiment). These paradigms facilitate rigorous analyses of citation network phenomena, allowing us to validate and challenge existing theories. Additionally, we extend the research scope of traditional science of science studies through idealized social experiments, with the simulation experiment results providing valuable insights for real-world academic environments. Our work demonstrates the potential of LLMs for advancing science of science research in social science.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14224v1">KTester: Leveraging Domain and Testing Knowledge for More Effective LLM-based Test Generation</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 13 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Automated unit test generation using large language models (LLMs) holds great promise but often struggles with generating tests that are both correct and maintainable in real-world projects. This paper presents KTester, a novel framework that integrates project-specific knowledge and testing domain knowledge to enhance LLM-based test generation. Our approach first extracts project structure and usage knowledge through static analysis, which provides rich context for the model. It then employs a testing-domain-knowledge-guided separation of test case design and test method generation, combined with a multi-perspective prompting strategy that guides the LLM to consider diverse testing heuristics. The generated tests follow structured templates, improving clarity and maintainability. We evaluate KTester on multiple open-source projects, comparing it against state-of-the-art LLM-based baselines using automatic correctness and coverage metrics, as well as a human study assessing readability and maintainability. Results demonstrate that KTester significantly outperforms existing methods across six key metrics, improving execution pass rate by 5.69% and line coverage by 8.83% over the strongest baseline, while requiring less time and generating fewer test cases. Human evaluators also rate the tests produced by KTester significantly higher in terms of correctness, readability, and maintainability, confirming the practical advantages of our knowledge-driven framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14221v1">LLM-Aligned Geographic Item Tokenization for Local-Life Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have enhanced text-based recommendation by enriching traditional ID-based methods with semantic generalization capabilities. Text-based methods typically encode item textual information via prompt design and generate discrete semantic IDs through item tokenization. However, in domain-specific tasks such as local-life services, simply injecting location information into prompts fails to capture fine-grained spatial characteristics and real-world distance awareness among items. To address this, we propose LGSID, an LLM-Aligned Geographic Item Tokenization Framework for Local-life Recommendation. This framework consists of two key components: (1) RL-based Geographic LLM Alignment, and (2) Hierarchical Geographic Item Tokenization. In the RL-based alignment module, we initially train a list-wise reward model to capture real-world spatial relationships among items. We then introduce a novel G-DPO algorithm that uses pre-trained reward model to inject generalized spatial knowledge and collaborative signals into LLMs while preserving their semantic understanding. Furthermore, we propose a hierarchical geographic item tokenization strategy, where primary tokens are derived from discrete spatial and content attributes, and residual tokens are refined using the aligned LLM's geographic representation vectors. Extensive experiments on real-world Kuaishou industry datasets show that LGSID consistently outperforms state-of-the-art discriminative and generative recommendation models. Ablation studies, visualizations, and case studies further validate its effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14214v1">Do Large Language Models (LLMs) Understand Chronology?</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 47 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used in finance and economics, where prompt-based attempts against look-ahead bias implicitly assume that models understand chronology. We test this fundamental question with a series of chronological ordering tasks with increasing complexities over facts the model already knows from pre-training. Our tasks cover (1) chronological ordering, (2) conditional sorting (filter, then order), and (3) anachronism detection. We evaluate GPT-4.1, Claude-3.7 Sonnet, with and without Extended Thinking (ET), and GPT-5 across multiple reasoning-effort settings. Across models, Exact match rate drops sharply as sequences lengthen even while rank correlations stay high as LLMs largely preserve local order but struggle to maintain a single globally consistent timeline. In conditional sorting, most failures stem from the filtering step rather than the ordering step, but GPT-5 and Claude-3.7 Sonnet with Extended Thinking outshine normal models significantly. Lastly, anachronism detection is found to be the easiest task for the LLMs but performance still declines with increasingly overlapping timelines or entities. Overall, our main contribution is showing that allocating explicit reasoning budget helps with chronological ordering with GPT-5 at medium/high reasoning effort achieving flawless ordering at all lengths and perfect conditional sorting (both self-filtered and given-subset), whereas low/minimal effort degrades with longer lists, mirroring earlier models. Our findings delineate limits of current LLMs on chronological tasks, providing insights into task complexity, and demonstrate scenarios in which reasoning helps. These patterns are important for the real-time application of LLMs in finance. We release all code and evaluation templates to support full reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.23535v2">Comparative Analysis of the Code Generated by Popular Large Language Models (LLMs) for MISRA C++ Compliance</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      Safety-critical systems are engineered systems whose failure or malfunction could result in catastrophic consequences. The software development for safety-critical systems necessitates rigorous engineering practices and adherence to certification standards like DO-178C for avionics. DO-178C is a guidance document which requires compliance to well-defined software coding standards like MISRA C++ to enforce coding guidelines that prevent the use of ambiguous, unsafe, or undefined constructs. Large Language Models (LLMs) have demonstrated significant capabilities in automatic code generation across a wide range of programming languages, including C++. Despite their impressive performance, code generated by LLMs in safety-critical domains must be carefully analyzed for conformance to MISRA C++ coding standards. In this paper, I have conducted a comparative analysis of the C++ code generated by popular LLMs including: OpenAI ChatGPT, Google Gemini, DeepSeek, Meta AI, and Microsoft Copilot for compliance with MISRA C++. The study revealed that none of the evaluated LLMs generated MISRA-compliant code despite clear prompts, with DeepSeek showing the fewest violations and Meta AI the most. While all models could correct individual violations when explicitly instructed, only ChatGPT consistently identified and resolved all targeted rule violations across complete code snippets, whereas others achieved partial success. Overall, LLMs show promise as aids for initial code generation, but they are not yet dependable for producing fully MISRA-compliant code required in safety-critical domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.23947v2">The Social Gaze of LLMs: A Literature Review of Multimodal Approaches to Human Behavior Understanding</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      LLM-powered multimodal systems are increasingly used to interpret human behavior, yet how researchers apply the models' 'social competence' remains poorly understood. This paper presents a systematic literature review of 176 publications across different application domains (e.g., healthcare, education, and entertainment). Using a four-dimensional coding framework (application, technical, evaluative, and ethical), we find (1) frequent use of pattern recognition and information extraction from multimodal sources, but limited support for adaptive, interactive reasoning; (2) a dominant 'modality-to-text' pipeline that privileges language over rich audiovisual cues, striping away nuanced social cues; (3) evaluation practices reliant on static benchmarks, with socially grounded, human-centered assessments rare; and (4) Ethical discussions focused mainly on legal and rights-related risks (e.g., privacy), leaving societal risks (e.g., deception) overlooked--or at best acknowledged but left unaddressed. We outline a research agenda for evaluating socially competent, ethically informed, and interaction-aware multi-modal systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09148v2">LoopTool: Closing the Data-Training Loop for Robust LLM Tool Calls</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 The code is accessible at https://github.com/Rednote-DeepExperience/LoopTool. The LoopTool-8B is accessible at https://huggingface.co/zhuiguang-ning/LoopTool-8B
    </div>
    <details class="paper-abstract">
      Augmenting Large Language Models (LLMs) with external tools enables them to execute complex, multi-step tasks. However, tool learning is hampered by the static synthetic data pipelines where data generation and model training are executed as two separate, non-interactive processes. This approach fails to adaptively focus on a model's specific weaknesses and allows noisy labels to persist, degrading training efficiency. We introduce LoopTool, a fully automated, model-aware data evolution framework that closes this loop by tightly integrating data synthesis and model training. LoopTool iteratively refines both the data and the model through three synergistic modules: (1) Greedy Capability Probing (GCP) diagnoses the model's mastered and failed capabilities; (2) Judgement-Guided Label Verification (JGLV) uses an open-source judge model to find and correct annotation errors, progressively purifying the dataset; and (3) Error-Driven Data Expansion (EDDE) generates new, challenging samples based on identified failures. This closed-loop process operates within a cost-effective, open-source ecosystem, eliminating dependence on expensive closed-source APIs. Experiments show that our 8B model trained with LoopTool significantly surpasses its 32B data generator and achieves new state-of-the-art results on the BFCL-v3 and ACEBench benchmarks for its scale. Our work demonstrates that closed-loop, self-refining data pipelines can dramatically enhance the tool-use capabilities of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14195v1">N-GLARE: An Non-Generative Latent Representation-Efficient LLM Safety Evaluator</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      Evaluating the safety robustness of LLMs is critical for their deployment. However, mainstream Red Teaming methods rely on online generation and black-box output analysis. These approaches are not only costly but also suffer from feedback latency, making them unsuitable for agile diagnostics after training a new model. To address this, we propose N-GLARE (A Non-Generative, Latent Representation-Efficient LLM Safety Evaluator). N-GLARE operates entirely on the model's latent representations, bypassing the need for full text generation. It characterizes hidden layer dynamics by analyzing the APT (Angular-Probabilistic Trajectory) of latent representations and introducing the JSS (Jensen-Shannon Separability) metric. Experiments on over 40 models and 20 red teaming strategies demonstrate that the JSS metric exhibits high consistency with the safety rankings derived from Red Teaming. N-GLARE reproduces the discriminative trends of large-scale red-teaming tests at less than 1\% of the token cost and the runtime cost, providing an efficient output-free evaluation proxy for real-time diagnostics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14182v1">WebRec: Enhancing LLM-based Recommendations with Attention-guided RAG from Web</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      Recommender systems play a vital role in alleviating information overload and enriching users' online experience. In the era of large language models (LLMs), LLM-based recommender systems have emerged as a prevalent paradigm for advancing personalized recommendations. Recently, retrieval-augmented generation (RAG) has drawn growing interest to facilitate the recommendation capability of LLMs, incorporating useful information retrieved from external knowledge bases. However, as a rich source of up-to-date information, the web remains under-explored by existing RAG-based recommendations. In particular, unique challenges are posed from two perspectives: one is to generate effective queries for web retrieval, considering the inherent knowledge gap between web search and recommendations; another challenge lies in harnessing online websites that contain substantial noisy content. To tackle these limitations, we propose WebRec, a novel web-based RAG framework, which takes advantage of the reasoning capability of LLMs to interpret recommendation tasks into queries of user preferences that cater to web retrieval. Moreover, given noisy web-retrieved information, where relevant pieces of evidence are scattered far apart, an insightful MP-Head is designed to enhance LLM attentions between distant tokens of relevant information via message passing. Extensive experiments have been conducted to demonstrate the effectiveness of our proposed web-based RAG methods in recommendation scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14181v1">Harnessing Deep LLM Participation for Robust Entity Linking</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      Entity Linking (EL), the task of mapping textual entity mentions to their corresponding entries in knowledge bases, constitutes a fundamental component of natural language understanding. Recent advancements in Large Language Models (LLMs) have demonstrated remarkable potential for enhancing EL performance. Prior research has leveraged LLMs to improve entity disambiguation and input representation, yielding significant gains in accuracy and robustness. However, these approaches typically apply LLMs to isolated stages of the EL task, failing to fully integrate their capabilities throughout the entire process. In this work, we introduce DeepEL, a comprehensive framework that incorporates LLMs into every stage of the entity linking task. Furthermore, we identify that disambiguating entities in isolation is insufficient for optimal performance. To address this limitation, we propose a novel self-validation mechanism that utilizes global contextual information, enabling LLMs to rectify their own predictions and better recognize cohesive relationships among entities within the same sentence. Extensive empirical evaluation across ten benchmark datasets demonstrates that DeepEL substantially outperforms existing state-of-the-art methods, achieving an average improvement of 2.6\% in overall F1 score and a remarkable 4% gain on out-of-domain datasets. These results underscore the efficacy of deep LLM integration in advancing the state-of-the-art in entity linking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.19670v3">CoSense-LLM: Semantics at the Edge with Cost- and Uncertainty-Aware Cloud-Edge Cooperation</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 19 pages,8 figures
    </div>
    <details class="paper-abstract">
      We present CoSense-LLM, an edge-first framework that turns continuous multimodal sensor streams (for example Wi-Fi CSI, IMU, audio, RFID, and lightweight vision) into compact, verifiable semantic tokens and coordinates with large language models under explicit latency, energy, bandwidth, and privacy constraints. CoSense-LLM has four parts: (i) SenseFusion, a lightweight encoder that aligns sensor embeddings with language and compresses them into short discrete code sequences; (ii) Edge-RAG, a local hybrid retrieval layer that grounds generation in site specific policies and notes; (iii) PromptRouter, a cost and uncertainty aware policy that selects edge only generation, edge plus retrieval, or compact cloud escalation; and (iv) Secure Execution, an auditable redaction path that enforces data minimization so raw waveforms never leave the device. The system works with modern serving optimizations, including paged or streaming KV caches, FlashAttention style kernels, speculative decoding, and quantized LoRA adapters, and supports on device personalization and federated updates under non IID drift. Across home, office, and clinic deployments, CoSense-LLM delivers grounded explanations while meeting tight service level objectives: it sustains sub second (p95) end to end latency on edge dominant paths, reduces inter tier token and bandwidth costs by preferring local retrieval grounded responses, and preserves privacy by transmitting only discrete codes and redacted metadata. Ablations show that Edge-RAG improves factual consistency and reduces contradictions, calibrated uncertainty enables selective abstention and controlled escalations, and KV plus decoding accelerators lower energy per decision. The results support an edge first design that treats semantics, privacy, and predictable latency as co equal goals for large model deployments in interference prone environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.18970v2">LLM-based Agents Suffer from Hallucinations: A Survey of Taxonomy, Methods, and Directions</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      Driven by the rapid advancements of Large Language Models (LLMs), LLM-based agents have emerged as powerful intelligent systems capable of human-like cognition, reasoning, and interaction. These agents are increasingly being deployed across diverse real-world applications, including student education, scientific research, and financial analysis. However, despite their remarkable potential, LLM-based agents remain vulnerable to hallucination issues, which can result in erroneous task execution and undermine the reliability of the overall system design. Addressing this critical challenge requires a deep understanding and a systematic consolidation of recent advances on LLM-based agents. To this end, we present the first comprehensive survey of hallucinations in LLM-based agents. By carefully analyzing the complete workflow of agents, we propose a new taxonomy that identifies different types of agent hallucinations occurring at different stages. Furthermore, we conduct an in-depth examination of eighteen triggering causes underlying the emergence of agent hallucinations. Through a detailed review of a large number of existing studies, we summarize approaches for hallucination mitigation and detection, and highlight promising directions for future research. We hope this survey will inspire further efforts toward addressing hallucinations in LLM-based agents, ultimately contributing to the development of more robust and reliable agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14140v1">Beyond Fixed and Dynamic Prompts: Embedded Jailbreak Templates for Advancing LLM Security</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      As the use of large language models (LLMs) continues to expand, ensuring their safety and robustness has become a critical challenge. In particular, jailbreak attacks that bypass built-in safety mechanisms are increasingly recognized as a tangible threat across industries, driving the need for diverse templates to support red-teaming efforts and strengthen defensive techniques. However, current approaches predominantly rely on two limited strategies: (i) substituting harmful queries into fixed templates, and (ii) having the LLM generate entire templates, which often compromises intent clarity and reproductibility. To address this gap, this paper introduces the Embedded Jailbreak Template, which preserves the structure of existing templates while naturally embedding harmful queries within their context. We further propose a progressive prompt-engineering methodology to ensure template quality and consistency, alongside standardized protocols for generation and evaluation. Together, these contributions provide a benchmark that more accurately reflects real-world usage scenarios and harmful intent, facilitating its application in red-teaming and policy regression testing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07991v2">VSPO: Validating Semantic Pitfalls in Ontology via LLM-Based CQ Generation</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 Accepted at AAAI 2026 oral
    </div>
    <details class="paper-abstract">
      Competency Questions (CQs) play a crucial role in validating ontology design. While manually crafting CQs can be highly time-consuming and costly for ontology engineers, recent studies have explored the use of large language models (LLMs) to automate this process. However, prior approaches have largely evaluated generated CQs based on their similarity to existing datasets, which often fail to verify semantic pitfalls such as "Misusing allValuesFrom". Since such pitfalls cannot be reliably detected through rule-based methods, we propose a novel dataset and model of Validating Semantic Pitfalls in Ontology (VSPO) for CQ generation specifically designed to verify the semantic pitfalls. To simulate missing and misused axioms, we use LLMs to generate natural language definitions of classes and properties and introduce misalignments between the definitions and the ontology by removing axioms or altering logical operators (e.g., substituting union with intersection). We then fine-tune LLaMA-3.1-8B-Instruct to generate CQs that validate these semantic discrepancies between the provided definitions and the corresponding axioms. The resulting CQs can detect a broader range of modeling errors compared to existing public datasets. Our fine-tuned model demonstrates superior performance over baselines, showing 26% higher precision and 28.2% higher recall than GPT-4.1 in generating CQs for pitfall validation. This research enables automatic generation of TBox-validating CQs using LLMs, significantly reducing manual effort while improving semantic alignment between ontologies and expert knowledge. To the best of our knowledge, this is the first study to target semantic pitfall validation in CQ generation using LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.01558v3">Predicting the Performance of Black-box LLMs through Self-Queries</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly relied on in AI systems, predicting when they make mistakes is crucial. While a great deal of work in the field uses internal representations to interpret model behavior, these representations are inaccessible when given solely black-box access through an API. In this paper, we extract features of LLMs in a black-box manner by using follow-up prompts and taking the probabilities of different responses as representations to train reliable predictors of model behavior. We demonstrate that training a linear model on these low-dimensional representations produces reliable and generalizable predictors of model performance at the instance level (e.g., if a particular generation correctly answers a question). Remarkably, these can often outperform white-box linear predictors that operate over a model's hidden state or the full distribution over its vocabulary. In addition, we demonstrate that these extracted features can be used to evaluate more nuanced aspects of a language model's state. For instance, they can be used to distinguish between a clean version of GPT-4o-mini and a version that has been influenced via an adversarial system prompt that answers question-answering tasks incorrectly or introduces bugs into generated code. Furthermore, they can reliably distinguish between different model architectures and sizes, enabling the detection of misrepresented models provided through an API (e.g., identifying if GPT-3.5 is supplied instead of GPT-4o-mini).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.15902v3">IPAD: Inverse Prompt for AI Detection - A Robust and Interpretable LLM-Generated Text Detector</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have attained human-level fluency in text generation, which complicates the distinguishing between human-written and LLM-generated texts. This increases the risk of misuse and highlights the need for reliable detectors. Yet, existing detectors exhibit poor robustness on out-of-distribution (OOD) data and attacked data, which is critical for real-world scenarios. Also, they struggle to provide interpretable evidence to support their decisions, thus undermining the reliability. In light of these challenges, we propose IPAD (Inverse Prompt for AI Detection), a novel framework consisting of a Prompt Inverter that identifies predicted prompts that could have generated the input text, and two Distinguishers that examine the probability that the input texts align with the predicted prompts. Empirical evaluations demonstrate that IPAD outperforms the strongest baselines by 9.05% (Average Recall) on in-distribution data, 12.93% (AUROC) on out-of-distribution data, and 5.48% (AUROC) on attacked data. IPAD also performs robustly on structured datasets. Furthermore, an interpretability assessment is conducted to illustrate that IPAD enhances the AI detection trustworthiness by allowing users to directly examine the decision-making evidence, which provides interpretable support for its state-of-the-art detection results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14129v1">MalRAG: A Retrieval-Augmented LLM Framework for Open-set Malicious Traffic Identification</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 13 pages, 13 figures. Intended for submission to IEEE Transactions on Information Forensics and Security (TIFS)
    </div>
    <details class="paper-abstract">
      Fine-grained identification of IDS-flagged suspicious traffic is crucial in cybersecurity. In practice, cyber threats evolve continuously, making the discovery of novel malicious traffic a critical necessity as well as the identification of known classes. Recent studies have advanced this goal with deep models, but they often rely on task-specific architectures that limit transferability and require per-dataset tuning. In this paper we introduce MalRAG, the first LLM driven retrieval-augmented framework for open-set malicious traffic identification. MalRAG freezes the LLM and operates via comprehensive traffic knowledge construction, adaptive retrieval, and prompt engineering. Concretely, we construct a multi-view traffic database by mining prior malicious traffic from content, structural, and temporal perspectives. Furthermore, we introduce a Coverage-Enhanced Retrieval Algorithm that queries across these views to assemble the most probable candidates, thereby improving the inclusion of correct evidence. We then employ Traffic-Aware Adaptive Pruning to select a variable subset of these candidates based on traffic-aware similarity scores, suppressing incorrect matches and yielding reliable retrieved evidence. Moreover, we develop a suite of guidance prompts where task instruction, evidence referencing, and decision guidance are integrated with the retrieved evidence to improve LLM performance. Across diverse real-world datasets and settings, MalRAG delivers state-of-the-art results in both fine-grained identification of known classes and novel malicious traffic discovery. Ablation and deep-dive analyses further show that MalRAG effective leverages LLM capabilities yet achieves open-set malicious traffic identification without relying on a specific LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14124v1">10Cache: Heterogeneous Resource-Aware Tensor Caching and Migration for LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 This paper accepted for presentation to the 16th ACM Symposium on Cloud Computing (SOCC'25)
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) in the cloud faces growing memory bottlenecks due to the limited capacity and high cost of GPUs. While GPU memory offloading to CPU and NVMe has made large-scale training more feasible, existing approaches suffer from high tensor migration latency and suboptimal device memory utilization, ultimately increasing training time and cloud costs. To address these challenges, we present 10Cache, a resource-aware tensor caching and migration system that accelerates LLM training by intelligently coordinating memory usage across GPU, CPU, and NVMe tiers. 10Cache profiles tensor execution order to construct prefetch policies, allocates memory buffers in pinned memory based on tensor size distributions, and reuses memory buffers to minimize allocation overhead. Designed for cloud-scale deployments, 10Cache improves memory efficiency and reduces reliance on high-end GPUs. Across diverse LLM workloads, it achieves up to 2x speedup in training time, improves GPU cache hit rate by up to 86.6x, and increases CPU/GPU memory utilization by up to 2.15x and 1.33x, respectively, compared to state-of-the-art offloading methods. These results demonstrate that 10Cache is a practical and scalable solution for optimizing LLM training throughput and resource efficiency in cloud environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12977v2">ArtiWorld: LLM-Driven Articulation of 3D Objects in Scenes</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      Building interactive simulators and scalable robot-learning environments requires a large number of articulated assets. However, most existing 3D assets in simulation are rigid, and manually converting them into articulated objects is extremely labor- and cost-intensive. This raises a natural question: can we automatically identify articulable objects in a scene and convert them into articulated assets directly? In this paper, we present ArtiWorld, a scene-aware pipeline that localizes candidate articulable objects from textual scene descriptions and reconstructs executable URDF models that preserve the original geometry. At the core of this pipeline is Arti4URDF, which leverages 3D point cloud, prior knowledge of a large language model (LLM), and a URDF-oriented prompt design to rapidly convert rigid objects into interactive URDF-based articulated objects while maintaining their 3D shape. We evaluate ArtiWorld at three levels: 3D simulated objects, full 3D simulated scenes, and real-world scan scenes. Across all three settings, our method consistently outperforms existing approaches and achieves state-of-the-art performance, while preserving object geometry and correctly capturing object interactivity to produce usable URDF-based articulated models. This provides a practical path toward building interactive, robot-ready simulation environments directly from existing 3D assets. Code and data will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14098v1">Collaborative QA using Interacting LLMs. Impact of Network Structure, Node Capability and Distributed Data</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      In this paper, we model and analyze how a network of interacting LLMs performs collaborative question-answering (CQA) in order to estimate a ground truth given a distributed set of documents. This problem is interesting because LLMs often hallucinate when direct evidence to answer a question is lacking, and these effects become more pronounced in a network of interacting LLMs. The hallucination spreads, causing previously accurate LLMs to hallucinate. We study interacting LLMs and their hallucination by combining novel ideas of mean-field dynamics (MFD) from network science and the randomized utility model from economics to construct a useful generative model. We model the LLM with a latent state that indicates if it is truthful or not with respect to the ground truth, and extend a tractable analytical model considering an MFD to model the diffusion of information in a directed network of LLMs. To specify the probabilities that govern the dynamics of the MFD, we propose a randomized utility model. For a network of LLMs, where each LLM has two possible latent states, we posit sufficient conditions for the existence and uniqueness of a fixed point and analyze the behavior of the fixed point in terms of the incentive (e.g., test-time compute) given to individual LLMs. We experimentally study and analyze the behavior of a network of $100$ open-source LLMs with respect to data heterogeneity, node capability, network structure, and sensitivity to framing on multiple semi-synthetic datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.12215v2">Xiangqi-R1: Enhancing Spatial Strategic Reasoning in LLMs for Chinese Chess via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 10 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Game playing has long served as a fundamental benchmark for evaluating Artificial General Intelligence. While Large Language Models (LLMs) have demonstrated impressive capabilities in general reasoning, their effectiveness in spatial strategic reasoning, which is critical for complex and fully observable board games, remains insufficiently explored. In this work, we adopt Chinese Chess (Xiangqi) as a challenging and rich testbed due to its intricate rules and spatial complexity. To advance LLMs' strategic competence in such environments, we propose a training framework tailored to Xiangqi, built upon a large-scale dataset of five million board-move pairs enhanced with expert annotations and engine evaluations. Building on this foundation, we introduce Xiangqi-R1, a 7B-parameter model trained in multi-stage manner. Our Experimental results indicate that, despite their size and power, general-purpose LLMs struggle to achieve satisfactory performance in these tasks. Compared to general-purpose LLMs, Xiangqi-R1 greatly advances with an 18% rise in move legality and a 22% boost in analysis accuracy. Our results point to a promising path for creating general strategic intelligence in complex areas.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.11056v2">From Reasoning LLMs to BERT: A Two-Stage Distillation Framework for Search Relevance</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      Query-service relevance prediction in e-commerce search systems faces strict latency requirements that prevent the direct application of Large Language Models (LLMs). To bridge this gap, we propose a two-stage reasoning distillation framework to transfer reasoning capabilities from a powerful teacher LLM to a lightweight, deployment-friendly student model. In the first stage, we address the limitations of general-purpose LLMs by constructing a domain-adapted teacher model. This is achieved through a three-step process: domain-adaptive pre-training to inject platform knowledge, supervised fine-tuning to elicit reasoning skills, and preference optimization with a multi-dimensional reward model to ensure the generation of reliable and preference-aligned reasoning paths. This teacher can then automatically annotate massive query-service pairs from search logs with both relevance labels and reasoning chains. In the second stage, to address the challenges of architectural heterogeneity in standard distillation, we introduce Contrastive Reasoning Self-Distillation (CRSD). By modeling the behavior of the same student model under "standard" and "reasoning-augmented" inputs as a teacher-student relationship, CRSD enables the lightweight model to internalize the teacher's complex decision-making mechanisms without needing the explicit reasoning path at inference. Offline evaluations and online A/B testing in the Meituan search advertising system demonstrate that our framework achieves significant improvements across multiple metrics, validating its effectiveness and practical value.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12376v2">BitSnap: Checkpoint Sparsification and Quantization in LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 12 pages, numerous figures
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) continue to grow in size and complexity, efficient checkpoint saving\&loading has become crucial for managing storage, memory usage, and fault tolerance in LLM training. The current works do not comprehensively take into account the optimization of these several aspects. This paper proposes a novel checkpoint sparsification and quantization method that adapts dynamically to different training stages and model architectures. We present a comprehensive analysis of existing lossy and lossless compression techniques, identify current limitations, and introduce our adaptive approach that balances compression ratio, speed, and precision impact throughout the training process. Experiments on different sizes of LLMs demonstrate that our bitmask-based sparsification method achieves 16x compression ratio without compromising model accuracy. Additionally, the cluster-based quantization method achieves 2x compression ratio with little precision loss.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09087v2">Tele-LLM-Hub: Building Context-Aware Multi-Agent LLM Systems for Telecom Networks</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      This paper introduces Tele-LLM-Hub, a user friendly low-code solution for rapid prototyping and deployment of context aware multi-agent (MA) Large Language Model (LLM) systems tailored for 5G and beyond. As telecom wireless networks become increasingly complex, intelligent LLM applications must share a domainspecific understanding of network state. We propose TeleMCP, the Telecom Model Context Protocol, to enable structured and context-rich communication between agents in telecom environments. Tele-LLM-Hub actualizes TeleMCP through a low-code interface that supports agent creation, workflow composition, and interaction with software stacks such as srsRAN. Key components include a direct chat interface, a repository of pre-built systems, an Agent Maker leveraging finetuning with our RANSTRUCT framework, and an MA-Maker for composing MA workflows. The goal of Tele-LLM-Hub is to democratize the design of contextaware MA systems and accelerate innovation in next-generation wireless networks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.18646v2">Beyond Benchmark: LLMs Evaluation with an Anthropomorphic and Value-oriented Roadmap</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 Preprint. Under Review
    </div>
    <details class="paper-abstract">
      For Large Language Models (LLMs), a disconnect persists between benchmark performance and real-world utility. Current evaluation frameworks remain fragmented, prioritizing technical metrics while neglecting holistic assessment for deployment. This survey introduces an anthropomorphic evaluation paradigm through the lens of human intelligence, proposing a novel three-dimensional taxonomy: Intelligence Quotient (IQ)-General Intelligence for foundational capacity, Emotional Quotient (EQ)-Alignment Ability for value-based interactions, and Professional Quotient (PQ)-Professional Expertise for specialized proficiency. For practical value, we pioneer a Value-oriented Evaluation (VQ) framework assessing economic viability, social impact, ethical alignment, and environmental sustainability. Our modular architecture integrates six components with an implementation roadmap. Through analysis of 200+ benchmarks, we identify key challenges including dynamic assessment needs and interpretability gaps. It provides actionable guidance for developing LLMs that are technically proficient, contextually relevant, and ethically sound. We maintain a curated repository of open-source evaluation resources at: https://github.com/onejune2018/Awesome-LLM-Eval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14023v1">Syn-STARTS: Synthesized START Triage Scenario Generation Framework for Scalable LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 Introducing an open dataset
    </div>
    <details class="paper-abstract">
      Triage is a critically important decision-making process in mass casualty incidents (MCIs) to maximize victim survival rates. While the role of AI in such situations is gaining attention for making optimal decisions within limited resources and time, its development and performance evaluation require benchmark datasets of sufficient quantity and quality. However, MCIs occur infrequently, and sufficient records are difficult to accumulate at the scene, making it challenging to collect large-scale realworld data for research use. Therefore, we developed Syn-STARTS, a framework that uses LLMs to generate triage cases, and verified its effectiveness. The results showed that the triage cases generated by Syn-STARTS were qualitatively indistinguishable from the TRIAGE open dataset generated by manual curation from training materials. Furthermore, when evaluating the LLM accuracy using hundreds of cases each from the green, yellow, red, and black categories defined by the standard triage method START, the results were found to be highly stable. This strongly indicates the possibility of synthetic data in developing high-performance AI models for severe and critical medical situations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14977v1">SVBRD-LLM: Self-Verifying Behavioral Rule Discovery for Autonomous Vehicle Identification</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      As more autonomous vehicles operate on public roads, understanding real-world behavior of autonomous vehicles is critical to analyzing traffic safety, making policies, and public acceptance. This paper proposes SVBRD-LLM, a framework that automatically discovers, verifies, and applies interpretable behavioral rules from real traffic videos through zero-shot prompt engineering. The framework extracts vehicle trajectories using YOLOv8 and ByteTrack, computes kinematic features, and employs GPT-5 zero-shot prompting to compare autonomous and human-driven vehicles, generating 35 structured behavioral rule hypotheses. These rules are tested on a validation set, iteratively refined based on failure cases to filter spurious correlations, and compiled into a high-confidence rule library. The framework is evaluated on an independent test set for speed change prediction, lane change prediction, and autonomous vehicle identification tasks. Experiments on over 1500 hours of real traffic videos show that the framework achieves 90.0% accuracy and 93.3% F1-score in autonomous vehicle identification. The discovered rules clearly reveal distinctive characteristics of autonomous vehicles in speed control smoothness, lane change conservativeness, and acceleration stability, with each rule accompanied by semantic description, applicable context, and validation confidence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14967v1">MermaidSeqBench: An Evaluation Benchmark for LLM-to-Mermaid Sequence Diagram Generation</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated excellent capabilities in generating structured diagrams from natural language descriptions. In particular, they have shown great promise in generating sequence diagrams for software engineering, typically represented in a text-based syntax such as Mermaid. However, systematic evaluations in this space remain underdeveloped as there is a lack of existing benchmarks to assess the LLM's correctness in this task. To address this shortcoming, we introduce MermaidSeqBench, a human-verified and LLM-synthetically-extended benchmark for assessing an LLM's capabilities in generating Mermaid sequence diagrams from textual prompts. The benchmark consists of a core set of 132 samples, starting from a small set of manually crafted and verified flows. These were expanded via a hybrid methodology combining human annotation, in-context LLM prompting, and rule-based variation generation. Our benchmark uses an LLM-as-a-judge model to assess Mermaid sequence diagram generation across fine-grained metrics, including syntax correctness, activation handling, error handling, and practical usability. We perform initial evaluations on numerous state-of-the-art LLMs and utilize multiple LLM judge models to demonstrate the effectiveness and flexibility of our benchmark. Our results reveal significant capability gaps across models and evaluation modes. Our proposed benchmark provides a foundation for advancing research in structured diagram generation and for developing more rigorous, fine-grained evaluation methodologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14937v1">CIMemories: A Compositional Benchmark for Contextual Integrity of Persistent Memory in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) increasingly use persistent memory from past interactions to enhance personalization and task performance. However, this memory introduces critical risks when sensitive information is revealed in inappropriate contexts. We present CIMemories, a benchmark for evaluating whether LLMs appropriately control information flow from memory based on task context. CIMemories uses synthetic user profiles with over 100 attributes per user, paired with diverse task contexts in which each attribute may be essential for some tasks but inappropriate for others. Our evaluation reveals that frontier models exhibit up to 69% attribute-level violations (leaking information inappropriately), with lower violation rates often coming at the cost of task utility. Violations accumulate across both tasks and runs: as usage increases from 1 to 40 tasks, GPT-5's violations rise from 0.1% to 9.6%, reaching 25.1% when the same prompt is executed 5 times, revealing arbitrary and unstable behavior in which models leak different attributes for identical prompts. Privacy-conscious prompting does not solve this - models overgeneralize, sharing everything or nothing rather than making nuanced, context-dependent decisions. These findings reveal fundamental limitations that require contextually aware reasoning capabilities, not just better prompting or scaling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14903v1">It's LIT! Reliability-Optimized LLMs with Inspectable Tools</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 Accepted to the 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop on Multi-Turn Interactions in Large Language Models
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have exhibited remarkable capabilities across various domains. The ability to call external tools further expands their capability to handle real-world tasks. However, LLMs often follow an opaque reasoning process, which limits their usefulness in high-stakes domains where solutions need to be trustworthy to end users. LLMs can choose solutions that are unreliable and difficult to troubleshoot, even if better options are available. We address this issue by forcing LLMs to use external -- more reliable -- tools to solve problems when possible. We present a framework built on the tool-calling capabilities of existing LLMs to enable them to select the most reliable and easy-to-troubleshoot solution path, which may involve multiple sequential tool calls. We refer to this framework as LIT (LLMs with Inspectable Tools). In order to support LIT, we introduce a new and challenging benchmark dataset of 1,300 questions and a customizable set of reliability cost functions associated with a collection of specialized tools. These cost functions summarize how reliable each tool is and how easy it is to troubleshoot. For instance, a calculator is reliable across domains, whereas a linear prediction model is not reliable if there is distribution shift, but it is easy to troubleshoot. A tool that constructs a random forest is neither reliable nor easy to troubleshoot. These tools interact with the Harvard USPTO Patent Dataset and a new dataset of NeurIPS 2023 papers to solve mathematical, coding, and modeling problems of varying difficulty levels. We demonstrate that LLMs can achieve more reliable and informed problem-solving while maintaining task performance using our framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.25939v2">SoK: Honeypots & LLMs, More Than the Sum of Their Parts?</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 Systemization of Knowledge
    </div>
    <details class="paper-abstract">
      The advent of Large Language Models (LLMs) promised to resolve the long-standing paradox in honeypot design: achieving high-fidelity deception with low operational risk. However, despite a flurry of research since late 2022, progress has been incremental, and the field lacks a cohesive understanding of the emerging architectural patterns, core challenges, and evaluation paradigms. To fill this gap, this Systematization of Knowledge (SoK) paper provides the first comprehensive overview of this new domain. We survey and systematize three critical, intersecting research areas: first, we provide a taxonomy of honeypot detection vectors, structuring the core problems that LLM-based realism must solve; second, we synthesize the emerging literature on LLM-honeypots, identifying a canonical architecture and key evaluation trends; and third, we chart the evolutionary path of honeypot log analysis, from simple data reduction to automated intelligence generation. We synthesize these findings into a forward-looking research roadmap, arguing that the true potential of this technology lies in creating autonomous, self-improving deception systems to counter the emerging threat of intelligent, automated attackers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15745v1">Structured Extraction of Vulnerabilities in OpenVAS and Tenable WAS Reports Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 5 pages, 4 tables, 3 figures, submitted to ERRC/WRSeg 2025
    </div>
    <details class="paper-abstract">
      This paper proposes an automated LLM-based method to extract and structure vulnerabilities from OpenVAS and Tenable WAS scanner reports, converting unstructured data into a standardized format for risk management. In an evaluation using a report with 34 vulnerabilities, GPT-4.1 and DeepSeek achieved the highest similarity to the baseline (ROUGE-L greater than 0.7). The method demonstrates feasibility in transforming complex reports into usable datasets, enabling effective prioritization and future anonymization of sensitive data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.21359v3">Can Machines Think Like Humans? A Behavioral Evaluation of LLM Agents in Dictator Games</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      As Large Language Model (LLM)-based agents increasingly engage with human society, how well do we understand their prosocial behaviors? We (1) investigate how LLM agents' prosocial behaviors can be induced by different personas and benchmarked against human behaviors; and (2) introduce a social science approach to evaluate LLM agents' decision-making. We explored how different personas and experimental framings affect these AI agents' altruistic behavior in dictator games and compared their behaviors within the same LLM family, across various families, and with human behaviors. The findings reveal that merely assigning a human-like identity to LLMs does not produce human-like behaviors. These findings suggest that LLM agents' reasoning does not consistently exhibit textual markers of human decision-making in dictator games and that their alignment with human behavior varies substantially across model architectures and prompt formulations; even worse, such dependence does not follow a clear pattern. As society increasingly integrates machine intelligence, "Prosocial AI" emerges as a promising and urgent research direction in philanthropic studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13900v1">What Works for 'Lost-in-the-Middle' in LLMs? A Study on GM-Extract and Mitigations</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 To be submitted for publication
    </div>
    <details class="paper-abstract">
      The diminishing ability of large language models (LLMs) to effectively utilize long-range context-the "lost-in-the-middle" phenomenon-poses a significant challenge in retrieval-based LLM applications. To study the impact of this phenomenon in a real-world application setting, we introduce GM-Extract, a novel benchmark dataset meticulously designed to evaluate LLM performance on retrieval of control variables. To accurately diagnose failure modes, we propose a simple yet elegant evaluation system using two distinct metrics: one for spatial retrieval capability (Document Metric) and the other for semantic retrieval capability (Variable Extraction Metric). We conduct a systematic evaluation of 7-8B parameter models on two multi-document tasks (key-value extraction and question-answering), demonstrating a significant change in retrieval performance simply by altering how the data is represented in the context window. While a distinct U-shaped curve was not consistently observed, our analysis reveals a clear pattern of performance across models, which we further correlate with perplexity scores. Furthermore, we perform a literature survey of mitigation methods, which we categorize into two distinct approaches: black-box and white-box methods. We then apply these techniques to our benchmark, finding that their efficacy is highly nuanced. Our evaluation highlights scenarios where these strategies successfully improve performance, as well as surprising cases where they lead to a negative impact, providing a comprehensive understanding of their utility in a practical context.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13876v1">QwenCLIP: Boosting Medical Vision-Language Pretraining via LLM Embeddings and Prompt tuning</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 This work has been submitted to the IEEE ISBI for possible publication
    </div>
    <details class="paper-abstract">
      Contrastive Language-Image Pretraining (CLIP) has demonstrated strong generalization for vision-language tasks in computer vision and medical domains, yet its text encoder accepts only up to 77 tokens, which limits its ability to represent long and information-rich radiology reports. Recent adaptations using domain-specific encoders, such as PubMedBERT or ClinicalBERT, mitigate this issue by leveraging medical corpora, but remain constrained by their limited input length (typically 512 tokens) and relatively shallow semantic understanding. To address these limitations, we propose QwenCLIP, a vision-language framework that replaces CLIP's text encoder with a large language model (LLM)-based embedding module (e.g., Qwen3-Embedding) and introduces learnable prompts to enhance cross-modal alignment. By leveraging the extended context window and richer representations of LLMs, QwenCLIP captures comprehensive medical semantics from long-form clinical text, substantially improving medical image-text alignment and downstream performance on radiology benchmarks. Our code is publicly available at https://github.com/Wxy-24/QwenCLIP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10459v2">LocalBench: Benchmarking LLMs on County-Level Local Knowledge and Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been widely evaluated on macro-scale geographic tasks, such as global factual recall, event summarization, and regional reasoning. Yet, their ability to handle hyper-local knowledge remains poorly understood. This gap is increasingly consequential as real-world applications, from civic platforms to community journalism, demand AI systems that can reason about neighborhood-specific dynamics, cultural narratives, and local governance. Existing benchmarks fall short in capturing this complexity, often relying on coarse-grained data or isolated references. We present LocalBench, the first benchmark designed to systematically evaluate LLMs on county-level local knowledge across the United States. Grounded in the Localness Conceptual Framework, LocalBench includes 14,782 validated question-answer pairs across 526 U.S. counties in 49 states, integrating diverse sources such as Census statistics, local subreddit discourse, and regional news. It spans physical, cognitive, and relational dimensions of locality. Using LocalBench, we evaluate 13 state-of-the-art LLMs under both closed-book and web-augmented settings. Our findings reveal critical limitations: even the best-performing models reach only 56.8% accuracy on narrative-style questions and perform below 15.5% on numerical reasoning. Moreover, larger model size and web augmentation do not guarantee better performance, for example, search improves Gemini's accuracy by +13.6%, but reduces GPT-series performance by -11.4%. These results underscore the urgent need for language models that can support equitable, place-aware AI systems: capable of engaging with the diverse, fine-grained realities of local communities across geographic and cultural contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10687v2">Who Gets the Reward, Who Gets the Blame? Evaluation-Aligned Training Signals for Multi-LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 Withdrawing temporarily to coordinate revisions with co-authors. A revised version will be resubmitted
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) in multi-agent systems (MAS) have shown promise for complex tasks, yet current training methods lack principled ways to connect system-level evaluation with agent-level and message-level learning. We propose a theoretical framework that unifies cooperative game-theoretic attribution with process reward modeling to transform system evaluation into agent credit and then into response-level signals. Unlike prior approaches that rely only on attribution (e.g., Shapley) or step-level labels (e.g., PRM), our method produces local, signed, and credit-conserving signals. In success cases, Shapley-based credit assignment fairly allocates outcomes across agents and is refined into per-message rewards that promote cooperation while discouraging redundancy or sabotage. In failure cases, first-error localization yields repair-aware preferences that penalize harmful steps while rewarding corrective attempts. The resulting signals are bounded, cooperative, and directly compatible with reinforcement-based or preference-based post-training, providing a unified and auditable pathway from global evaluation to local supervision in LLM multi-agent training. Our contribution is conceptual: we present a theoretical foundation and training signals, leaving empirical validation for future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13717v1">TZ-LLM: Protecting On-Device Large Language Models with Arm TrustZone</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) deployed on mobile devices offer benefits like user privacy and reduced network latency, but introduce a significant security risk: the leakage of proprietary models to end users. To mitigate this risk, we propose a system design for protecting on-device LLMs using Arm Trusted Execution Environment (TEE), TrustZone. Our system addresses two primary challenges: (1) The dilemma between memory efficiency and fast inference (caching model parameters within TEE memory). (2) The lack of efficient and secure Neural Processing Unit (NPU) time-sharing between Rich Execution Environment (REE) and TEE. Our approach incorporates two key innovations. First, we employ pipelined restoration, leveraging the deterministic memory access patterns of LLM inference to prefetch parameters on demand, hiding memory allocation, I/O and decryption latency under computation time. Second, we introduce a co-driver design, creating a minimal data plane NPU driver in the TEE that collaborates with the full-fledged REE driver. This reduces the TEE TCB size and eliminates control plane reinitialization overhead during NPU world switches. We implemented our system on the emerging OpenHarmony OS and the llama.cpp inference framework, and evaluated it with various LLMs on an Arm Rockchip device. Compared to a strawman TEE baseline lacking our optimizations, our system reduces TTFT by up to 90.9% and increases decoding speed by up to 23.2%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13676v1">T-SAR: A Full-Stack Co-design for CPU-Only Ternary LLM Inference via In-Place SIMD ALU Reorganization</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 Accepted to DATE 2026
    </div>
    <details class="paper-abstract">
      Recent advances in LLMs have outpaced the computational and memory capacities of edge platforms that primarily employ CPUs, thereby challenging efficient and scalable deployment. While ternary quantization enables significant resource savings, existing CPU solutions rely heavily on memory-based lookup tables (LUTs) which limit scalability, and FPGA or GPU accelerators remain impractical for edge use. This paper presents T-SAR, the first framework to achieve scalable ternary LLM inference on CPUs by repurposing the SIMD register file for dynamic, in-register LUT generation with minimal hardware modifications. T-SAR eliminates memory bottlenecks and maximizes data-level parallelism, delivering 5.6-24.5x and 1.1-86.2x improvements in GEMM latency and GEMV throughput, respectively, with only 3.2% power and 1.4% area overheads in SIMD units. T-SAR achieves up to 2.5-4.9x the energy efficiency of an NVIDIA Jetson AGX Orin, establishing a practical approach for efficient LLM inference on edge platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13658v1">Why is "Chicago" Predictive of Deceptive Reviews? Using LLMs to Discover Language Phenomena from Lexical Cues</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Deceptive reviews mislead consumers, harm businesses, and undermine trust in online marketplaces. Machine learning classifiers can learn from large amounts of training examples to effectively distinguish deceptive reviews from genuine ones. However, the distinguishing features learned by these classifiers are often subtle, fragmented, and difficult for humans to interpret. In this work, we explore using large language models (LLMs) to translate machine-learned lexical cues into human-understandable language phenomena that can differentiate deceptive reviews from genuine ones. We show that language phenomena obtained in this manner are empirically grounded in data, generalizable across similar domains, and more predictive than phenomena either in LLMs' prior knowledge or obtained through in-context learning. These language phenomena have the potential to aid people in critically assessing the credibility of online reviews in environments where deception detection classifiers are unavailable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13640v1">Data Value in the Age of Scaling: Understanding LLM Scaling Dynamics Under Real-Synthetic Data Mixtures</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      The rapid progress of large language models (LLMs) is fueled by the growing reliance on datasets that blend real and synthetic data. While synthetic data offers scalability and cost-efficiency, it often introduces systematic distributional discrepancies, particularly underrepresenting long-tail knowledge due to truncation effects from data generation mechanisms like top-p sampling, temperature scaling, and finite sampling. These discrepancies pose fundamental challenges in characterizing and evaluating the utility of mixed real-synthetic datasets. In this paper, we identify a three-phase scaling behavior characterized by two breakpoints that reflect transitions in model behavior across learning head and tail knowledge. We further derive an LLM generalization bound designed for real and synthetic mixtures, revealing several key factors that govern their generalization performance. Building on our theoretical findings, we propose an effective yet efficient data valuation method that scales to large-scale datasets. Comprehensive experiments across four tasks, including image classification, sentiment classification, instruction following, and complex reasoning, demonstrate that our method surpasses state-of-the-art baselines in data valuation with significantly low computational cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.21323v2">LLM-driven Provenance Forensics for Threat Investigation and Detection</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      We introduce PROVSEEK, an LLM-powered agentic framework for automated provenance-driven forensic analysis and threat intelligence extraction. PROVSEEK employs specialized toolchains to dynamically retrieve relevant context by generating precise, context-aware queries that fuse knowledge from threat reports with evidence from system provenance data. The framework resolves provenance queries, orchestrates multiple role-specific agents, and synthesizes structured, ground-truth verifiable forensic summaries. By combining agent orchestration with Retrieval-Augmented Generation (RAG) and chain-of-thought (CoT) reasoning, data-guided filtration using a behavioral model, PROVSEEK enables adaptive multi-step analysis that iteratively refines hypotheses, verifies supporting evidence, and produces scalable, interpretable forensic explanations of attack behaviors. PROVSEEK is designed for automated threat investigation without task-specific training data, enabling forensic-style investigation even when no prior knowledge of the environment. We conduct a comprehensive evaluation on publicly available DARPA datasets, demonstrating that PROVSEEK outperforms retrieval-based methods for the intelligence extraction task, achieving a 34% improvement in contextual precision/recall; and for threat detection task, PROVSEEK achieves 22%/29% higher precision/recall compared to both a baseline agent approach and State-Of-The-Art (SOTA) Provenance-based Intrusion Detection System (PIDS). In our scalability study, we show that PROVSEEK increases token usage by 1.42x and latency by 1.63x as the database size increases 50x, making it optimal for large-scale deployment. We also conducted an ablation and error analysis study to show how different components of PROVSEEK affect the detection performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.19662v3">HALO: Hardware-aware quantization with low critical-path-delay weights for LLM acceleration</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Quantization is critical for efficiently deploying large language models (LLMs). Yet conventional methods remain hardware-agnostic, limited to bit-width constraints, and do not account for intrinsic circuit characteristics such as the timing behaviors and energy profiles of Multiply-Accumulate (MAC) units. This disconnect from circuit-level behavior limits the ability to exploit available timing margins and energy-saving opportunities, reducing the overall efficiency of deployment on modern accelerators. To address these limitations, we propose HALO, a versatile framework for Hardware-Aware Post-Training Quantization (PTQ). Unlike traditional methods, HALO explicitly incorporates detailed hardware characteristics, including critical-path timing and power consumption, into its quantization approach. HALO strategically selects weights with low critical-path-delays enabling higher operational frequencies and dynamic frequency scaling without disrupting the architecture's dataflow. Remarkably, HALO achieves these improvements with only a few dynamic voltage and frequency scaling (DVFS) adjustments, ensuring simplicity and practicality in deployment. Additionally, by reducing switching activity within the MAC units, HALO effectively lowers energy consumption. Evaluations on accelerators such as Tensor Processing Units (TPUs) and Graphics Processing Units (GPUs) demonstrate that HALO significantly enhances inference efficiency, achieving average performance improvements of 270% and energy savings of 51% over baseline quantization methods, all with minimal impact on accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06390v2">Ghost in the Transformer: Tracing LLM Lineage with SVD-Fingerprint</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 Accepted at AAAI 2026 (Oral)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have rapidly advanced and are widely adopted across diverse fields. Due to the substantial computational cost and data requirements of training from scratch, many developers choose to fine-tune or modify existing open-source models. While most adhere to open-source licenses, some falsely claim original training despite clear derivation from public models. This raises pressing concerns about intellectual property protection and highlights the need for reliable methods to verify model provenance. In this paper, we propose GhostSpec, a lightweight yet effective method for verifying LLM lineage without access to training data or modification of model behavior. Our approach constructs compact and robust fingerprints by applying singular value decomposition (SVD) to invariant products of internal attention weight matrices, effectively capturing the structural identity of a model. Unlike watermarking or output-based methods, GhostSpec is fully data-free, non-invasive, and computationally efficient. It demonstrates strong robustness to sequential fine-tuning, pruning, block expansion, and even adversarial transformations. Extensive experiments show that GhostSpec can reliably trace the lineage of transformed models with minimal overhead. By offering a practical solution for model verification and reuse tracking, our method contributes to the protection of intellectual property and fosters a transparent, trustworthy ecosystem for large-scale language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2409.19100v2">Personalizing Prostate Cancer Education for Patients Using an EHR-Integrated LLM Agent</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Cancer patients often lack timely education and personalized support due to clinician workload. This quality improvement study develops and evaluates a Large Language Model (LLM) agent, MedEduChat, which is integrated with the clinic's electronic health records (EHR) and designed to enhance prostate cancer patient education. Fifteen non-metastatic prostate cancer patients and three clinicians recruited from the Mayo Clinic interacted with the agent between May 2024 and April 2025. Findings showed that MedEduChat has a high usability score (UMUX 83.7 out of 100) and improves patients' health confidence (Health Confidence Score rose from 9.9 to 13.9). Clinicians evaluated the patient-chat interaction history and rated MedEduChat as highly correct (2.9 out of 3), complete (2.7 out of 3), and safe (2.7 out of 3), with moderate personalization (2.3 out of 3). This study highlights the potential of LLM agents to improve patient engagement and health education.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.19838v3">LLM-Powered GUI Agents in Phone Automation: Surveying Progress and Prospects</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 Paper accepted to TMLR 2025, Project Homepage: https://github.com/PhoneLLM/Awesome-LLM-Powered-Phone-GUI-Agents
    </div>
    <details class="paper-abstract">
      With the rapid rise of large language models (LLMs), phone automation has undergone transformative changes. This paper systematically reviews LLM-driven phone GUI agents, highlighting their evolution from script-based automation to intelligent, adaptive systems. We first contextualize key challenges, (i) limited generality, (ii) high maintenance overhead, and (iii) weak intent comprehension, and show how LLMs address these issues through advanced language understanding, multimodal perception, and robust decision-making. We then propose a taxonomy covering fundamental agent frameworks (single-agent, multi-agent, plan-then-act), modeling approaches (prompt engineering, training-based), and essential datasets and benchmarks. Furthermore, we detail task-specific architectures, supervised fine-tuning, and reinforcement learning strategies that bridge user intent and GUI operations. Finally, we discuss open challenges such as dataset diversity, on-device deployment efficiency, user-centric adaptation, and security concerns, offering forward-looking insights into this rapidly evolving field. By providing a structured overview and identifying pressing research gaps, this paper serves as a definitive reference for researchers and practitioners seeking to harness LLMs in designing scalable, user-friendly phone GUI agents. The collection of papers reviewed in this survey will be hosted and regularly updated on the GitHub repository: https://github.com/PhoneLLM/Awesome-LLM-Powered-Phone-GUI-Agents
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.04486v2">EDIT-Bench: Evaluating LLM Abilities to Perform Real-World Instructed Code Edits</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Instructed code editing, where LLMs directly modify a developer's existing code based on a user instruction, is becoming a widely used interaction mode in AI coding assistants. However, few benchmarks directly evaluate this capability and current datasets often rely on artificial sources. We introduce EDIT-Bench, a benchmark for evaluating LLM code editing capabilities grounded in real-world usage, i.e., user instructions and code contexts collected in the wild. EDIT-Bench comprises of 540 problems, multiple natural and programming languages, and a diverse set of real-world use cases, ranging from resolving errors to adding features. EDIT-Bench introduces context-dependent problems that require the model to understand code context, highlighted code, and cursor position in addition to the user instruction. We evaluate 40 diverse LLMs and observe that EDIT-Bench is a challenging set of problems where only 1 model scores over 60%. We find that model performance varies across different categories of user instructions. Further, we find that varying levels of contextual information greatly affect task success rate, with performance varying up to 11%, indicating the importance of evaluating with realistic context.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.16124v3">Benchmarking LLM Privacy Recognition for Social Robot Decision Making</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 18 pages, 7 figures. Dakota Sullivan and Shirley Zhang contributed equally to this work
    </div>
    <details class="paper-abstract">
      While robots have previously utilized rule-based systems or probabilistic models for user interaction, the rapid evolution of large language models (LLMs) presents new opportunities to develop LLM-powered robots for enhanced human-robot interaction (HRI). To fully realize these capabilities, however, robots need to collect data such as audio, fine-grained images, video, and locations. As a result, LLMs often process sensitive personal information, particularly within private environments, such as homes. Given the tension between utility and privacy risks, evaluating how current LLMs manage sensitive data is critical. Specifically, we aim to explore the extent to which out-of-the-box LLMs are privacy-aware in the context of household robots. In this work, we present a set of privacy-relevant scenarios developed using the Contextual Integrity (CI) framework. We first surveyed users' privacy preferences regarding in-home robot behaviors and then examined how their privacy orientations affected their choices of these behaviors (N = 450). We then provided the same set of scenarios and questions to state-of-the-art LLMs (N = 10) and found that the agreement between humans and LLMs was generally low. To further investigate the capabilities of LLMs as potential privacy controllers, we implemented four additional prompting strategies and compared their results. We discuss the performance of the evaluated models as well as the implications and potential of AI privacy awareness in human-robot interaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.10950v2">Unveiling Challenges for LLMs in Enterprise Data Engineering</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) promise to automate data engineering on tabular data, offering enterprises a valuable opportunity to cut the high costs of manual data handling. But the enterprise domain comes with unique challenges that existing LLM-based approaches for data engineering often overlook, such as large table sizes, more complex tasks, and the need for internal knowledge. To bridge these gaps, we identify key enterprise-specific challenges related to data, tasks, and background knowledge and extensively evaluate how they affect data engineering with LLMs. Our analysis reveals that LLMs face substantial limitations in real-world enterprise scenarios, with accuracy declining sharply. Our findings contribute to a systematic understanding of LLMs for enterprise data engineering to support their adoption in industry.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13373v1">A Novel Hierarchical Integration Method for Efficient Model Merging in Medical LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) face significant challenges in distributed healthcare, including consolidating specialized domain knowledge across institutions while maintaining privacy, reducing computational overhead, and preventing catastrophic forgetting during model updates.This paper presents a systematic evaluation of six parameter-space merging techniques applied to two architecturally compatible medical LLMs derived from the Mistral-7B base model. We introduce a novel hierarchical method that combines selective Optimal Transport (OT) alignment for attention layers with cosine similarity-weighted interpolation, designed to address permutation variance while minimizing computational overhead for edge deployment scenarios. Our study evaluates Task Arithmetic, Linear Averaging, DARE-TIES, DELLA, Breadcrumbs, and our Hierarchical approach across five medical benchmarks. Results demonstrate that architecturally compatible models benefit significantly from simple averaging methods, with Task Arithmetic achieving 45.80% accuracy on MedQA, outperforming complex pruning-based approaches. These findings offer critical insights for the deployment of distributed medical AI in resource-constrained IoT environments, where computational efficiency and model compatibility are paramount. Our work establishes that for architecturally compatible models, simple averaging provides a robust and computationally efficient baseline for knowledge consolidation, offering a pragmatic path forward for scalable medical AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.04108v2">Can Linear Probes Measure LLM Uncertainty?</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Effective Uncertainty Quantification (UQ) represents a key aspect for reliable deployment of Large Language Models (LLMs) in automated decision-making and beyond. Yet, for LLM generation with multiple choice structure, the state-of-the-art in UQ is still dominated by the naive baseline given by the maximum softmax score. To address this shortcoming, we demonstrate that taking a principled approach via Bayesian statistics leads to improved performance despite leveraging the simplest possible model, namely linear regression. More precisely, we propose to train multiple Bayesian linear models, each predicting the output of a layer given the output of the previous one. Based on the obtained layer-level posterior distributions, we infer the global uncertainty level of the LLM by identifying a sparse combination of distributional features, leading to an efficient UQ scheme. Numerical experiments on various LLMs show consistent improvement over state-of-the-art baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13341v1">An LLM-based Quantitative Framework for Evaluating High-Stealthy Backdoor Risks in OSS Supply Chains</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 7 figures, 4 tables, conference
    </div>
    <details class="paper-abstract">
      In modern software development workflows, the open-source software supply chain contributes significantly to efficient and convenient engineering practices. With increasing system complexity, using open-source software as third-party dependencies has become a common practice. However, the lack of maintenance for underlying dependencies and insufficient community auditing create challenges in ensuring source code security and the legitimacy of repository maintainers, especially under high-stealthy backdoor attacks exemplified by the XZ-Util incident. To address these problems, we propose a fine-grained project evaluation framework for backdoor risk assessment in open-source software. The framework models stealthy backdoor attacks from the viewpoint of the attacker and defines targeted metrics for each attack stage. In addition, to overcome the limitations of static analysis in assessing the reliability of repository maintenance activities such as irregular committer privilege escalation and limited participation in reviews, the framework uses large language models (LLMs) to conduct semantic evaluation of code repositories without relying on manually crafted patterns. The framework is evaluated on sixty six high-priority packages in the Debian ecosystem. The experimental results indicate that the current open-source software supply chain is exposed to various security risks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.11864v2">NeuroStrike: Neuron-Level Attacks on Aligned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Safety alignment is critical for the ethical deployment of large language models (LLMs), guiding them to avoid generating harmful or unethical content. Current alignment techniques, such as supervised fine-tuning and reinforcement learning from human feedback, remain fragile and can be bypassed by carefully crafted adversarial prompts. Unfortunately, such attacks rely on trial and error, lack generalizability across models, and are constrained by scalability and reliability. This paper presents NeuroStrike, a novel and generalizable attack framework that exploits a fundamental vulnerability introduced by alignment techniques: the reliance on sparse, specialized safety neurons responsible for detecting and suppressing harmful inputs. We apply NeuroStrike to both white-box and black-box settings: In the white-box setting, NeuroStrike identifies safety neurons through feedforward activation analysis and prunes them during inference to disable safety mechanisms. In the black-box setting, we propose the first LLM profiling attack, which leverages safety neuron transferability by training adversarial prompt generators on open-weight surrogate models and then deploying them against black-box and proprietary targets. We evaluate NeuroStrike on over 20 open-weight LLMs from major LLM developers. By removing less than 0.6% of neurons in targeted layers, NeuroStrike achieves an average attack success rate (ASR) of 76.9% using only vanilla malicious prompts. Moreover, Neurostrike generalizes to four multimodal LLMs with 100% ASR on unsafe image inputs. Safety neurons transfer effectively across architectures, raising ASR to 78.5% on 11 fine-tuned models and 77.7% on five distilled models. The black-box LLM profiling attack achieves an average ASR of 63.7% across five black-box models, including the Google Gemini family.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13319v1">Whistledown: Combining User-Level Privacy with Conversational Coherence in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Users increasingly rely on large language models (LLMs) for personal, emotionally charged, and socially sensitive conversations. However, prompts sent to cloud-hosted models can contain personally identifiable information (PII) that users do not want logged, retained, or leaked. We observe this to be especially acute when users discuss friends, coworkers, or adversaries, i.e., when they spill the tea. Enterprises face the same challenge when they want to use LLMs for internal communication and decision-making. In this whitepaper, we present Whistledown, a best-effort privacy layer that modifies prompts before they are sent to the LLM. Whistledown combines pseudonymization and $ε$-local differential privacy ($ε$-LDP) with transformation caching to provide best-effort privacy protection without sacrificing conversational utility. Whistledown is designed to have low compute and memory overhead, allowing it to be deployed directly on a client's device in the case of individual users. For enterprise users, Whistledown is deployed centrally within a zero-trust gateway that runs on an enterprise's trusted infrastructure. Whistledown requires no changes to the existing APIs of popular LLM providers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13305v1">SAINT: Service-level Integration Test Generation with Program Analysis and LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 Accepted at ICSE'26
    </div>
    <details class="paper-abstract">
      Enterprise applications are typically tested at multiple levels, with service-level testing playing an important role in validating application functionality. Existing service-level testing tools, especially for RESTful APIs, often employ fuzzing and/or depend on OpenAPI specifications which are not readily available in real-world enterprise codebases. Moreover, these tools are limited in their ability to generate functional tests that effectively exercise meaningful scenarios. In this work, we present SAINT, a novel white-box testing approach for service-level testing of enterprise Java applications. SAINT combines static analysis, large language models (LLMs), and LLM-based agents to automatically generate endpoint and scenario-based tests. The approach builds two key models: an endpoint model, capturing syntactic and semantic information about service endpoints, and an operation dependency graph, capturing inter-endpoint ordering constraints. SAINT then employs LLM-based agents to generate tests. Endpoint-focused tests aim to maximize code and database interaction coverage. Scenario-based tests are synthesized by extracting application use cases from code and refining them into executable tests via planning, action, and reflection phases of the agentic loop. We evaluated SAINT on eight Java applications, including a proprietary enterprise application. Our results illustrate the effectiveness of SAINT in coverage, fault detection, and scenario generation. Moreover, a developer survey provides strong endorsement of the scenario-based tests generated by SAINT. Overall, our work shows that combining static analysis with agentic LLM workflows enables more effective, functional, and developer-aligned service-level test generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.02962v5">RAG-R1: Incentivizing the Search and Reasoning Capabilities of LLMs through Multi-query Parallelism</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), despite their remarkable capabilities, are prone to generating hallucinated or outdated content due to their static internal knowledge. While Retrieval-Augmented Generation (RAG) integrated with Reinforcement Learning (RL) offers a solution, these methods are fundamentally constrained by a single-query mode, leading to prohibitive latency and inherent brittleness. To overcome these limitations, we introduce RAG-R1, a novel two-stage training framework centered around multi-query parallelism. Our framework enables LLMs to adaptively leverage internal and external knowledge during the reasoning process while transitioning from the single-query mode to multi-query parallelism. This architectural shift bolsters reasoning robustness while significantly reducing inference latency. Extensive experiments on seven question-answering benchmarks confirm the superiority of our method, which outperforms the strongest baseline by up to 13.7% and decreases inference time by 11.1%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13290v1">Dropouts in Confidence: Moral Uncertainty in Human-LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 Accepted to AAAI 2026
    </div>
    <details class="paper-abstract">
      Humans display significant uncertainty when confronted with moral dilemmas, yet the extent of such uncertainty in machines and AI agents remains underexplored. Recent studies have confirmed the overly confident tendencies of machine-generated responses, particularly in large language models (LLMs). As these systems are increasingly embedded in ethical decision-making scenarios, it is important to understand their moral reasoning and the inherent uncertainties in building reliable AI systems. This work examines how uncertainty influences moral decisions in the classical trolley problem, analyzing responses from 32 open-source models and 9 distinct moral dimensions. We first find that variance in model confidence is greater across models than within moral dimensions, suggesting that moral uncertainty is predominantly shaped by model architecture and training method. To quantify uncertainty, we measure binary entropy as a linear combination of total entropy, conditional entropy, and mutual information. To examine its effects, we introduce stochasticity into models via "dropout" at inference time. Our findings show that our mechanism increases total entropy, mainly through a rise in mutual information, while conditional entropy remains largely unchanged. Moreover, this mechanism significantly improves human-LLM moral alignment, with correlations in mutual information and alignment score shifts. Our results highlight the potential to better align model-generated decisions and human preferences by deliberately modulating uncertainty and reducing LLMs' confidence in morally complex scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.25506v3">Reflections on the Reproducibility of Commercial LLM Performance in Empirical Software Engineering Studies</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Large Language Models have gained remarkable interest in industry and academia. The increasing interest in LLMs in academia is also reflected in the number of publications on this topic over the last years. For instance, alone 78 of the around 425 publications at ICSE 2024 performed experiments with LLMs. Conducting empirical studies with LLMs remains challenging and raises questions on how to achieve reproducible results, for both researchers and practitioners. One important step towards excelling in empirical research on LLM and their application is to first understand to what extent current research results are eventually reproducible and what factors may impede reproducibility. This investigation is within the scope of our work. We contribute an analysis of the reproducibility of LLM-centric studies, provide insights into the factors impeding reproducibility, and discuss suggestions on how to improve the current state. In particular, we studied the 85 articles describing LLM-centric studies, published at ICSE 2024 and ASE 2024. Of the 85 articles, 18 provided research artefacts and used OpenAI models. We attempted to replicate those 18 studies. Of the 18 studies, only five were sufficiently complete and executable. For none of the five studies, we were able to fully reproduce the results. Two studies seemed to be partially reproducible, and three studies did not seem to be reproducible. Our results highlight not only the need for stricter research artefact evaluations but also for more robust study designs to ensure the reproducible value of future publications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.22041v3">An LLM-based Simulation Framework for Embodied Conversational Agents in Psychological Counseling</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 Accepted to AAAI 2026
    </div>
    <details class="paper-abstract">
      Due to privacy concerns, open dialogue datasets for mental health are primarily generated through human or AI synthesis methods. However, the inherent implicit nature of psychological processes, particularly those of clients, poses challenges to the authenticity and diversity of synthetic data. In this paper, we propose ECAs (short for Embodied Conversational Agents), a framework for embodied agent simulation based on Large Language Models (LLMs) that incorporates multiple psychological theoretical principles.Using simulation, we expand real counseling case data into a nuanced embodied cognitive memory space and generate dialogue data based on high-frequency counseling questions.We validated our framework using the D4 dataset. First, we created a public ECAs dataset through batch simulations based on D4. Licensed counselors evaluated our method, demonstrating that it significantly outperforms baselines in simulation authenticity and necessity. Additionally, two LLM-based automated evaluation methods were employed to confirm the higher quality of the generated dialogues compared to the baselines. The source code and dataset are available at https://github.com/AIR-DISCOVER/ECAs-Dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.01223v2">Jailbreaking LLMs via Semantically Relevant Nested Scenarios with Targeted Toxic Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities in various tasks. However, they remain exposed to jailbreak attacks, eliciting harmful responses. The nested scenario strategy has been increasingly adopted across various methods, demonstrating immense potential. Nevertheless, these methods are easily detectable due to their prominent malicious intentions. In this work, we are the first to find and systematically verify that LLMs' alignment defenses are not sensitive to nested scenarios, where these scenarios are highly semantically relevant to the queries and incorporate targeted toxic knowledge. This is a crucial yet insufficiently explored direction. Based on this, we propose RTS-Attack (Semantically Relevant Nested Scenarios with Targeted Toxic Knowledge), an adaptive and automated framework to examine LLMs' alignment. By building scenarios highly relevant to the queries and integrating targeted toxic knowledge, RTS-Attack bypasses the alignment defenses of LLMs. Moreover, the jailbreak prompts generated by RTS-Attack are free from harmful queries, leading to outstanding concealment. Extensive experiments demonstrate that RTS-Attack exhibits superior performance in both efficiency and universality compared to the baselines across diverse advanced LLMs, including GPT-4o, Llama3-70b, and Gemini-pro. Our complete code is available at https://github.com/nercode/Work. WARNING: THIS PAPER CONTAINS POTENTIALLY HARMFUL CONTENT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13254v1">Souper-Model: How Simple Arithmetic Unlocks State-of-the-Art LLM Performance</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities across diverse domains, but their training remains resource- and time-intensive, requiring massive compute power and careful orchestration of training procedures. Model souping-the practice of averaging weights from multiple models of the same architecture-has emerged as a promising pre- and post-training technique that can enhance performance without expensive retraining. In this paper, we introduce Soup Of Category Experts (SoCE), a principled approach for model souping that utilizes benchmark composition to identify optimal model candidates and applies non-uniform weighted averaging to maximize performance. Contrary to previous uniform-averaging approaches, our method leverages the observation that benchmark categories often exhibit low inter-correlations in model performance. SoCE identifies "expert" models for each weakly-correlated category cluster and combines them using optimized weighted averaging rather than uniform weights. We demonstrate that the proposed method improves performance and robustness across multiple domains, including multilingual capabilities, tool calling, and math and achieves state-of-the-art results on the Berkeley Function Calling Leaderboard.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.06261v4">Hogwild! Inference: Parallel LLM Generation via Concurrent Attention</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 39th Conference on Neural Information Processing Systems (NeurIPS 2025)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated the ability to tackle increasingly complex tasks through advanced reasoning, long-form content generation, and tool use. Solving these tasks often involves long inference-time computations. In human problem solving, a common strategy to expedite work is collaboration: by dividing the problem into sub-tasks, exploring different strategies concurrently, etc. Recent research has shown that LLMs can also operate in parallel by implementing explicit cooperation frameworks, such as voting mechanisms or the explicit creation of independent sub-tasks that can be executed in parallel. However, each of these frameworks may not be suitable for all types of tasks, which can hinder their applicability. In this work, we propose a different design approach: we run LLM "workers" in parallel , allowing them to synchronize via a concurrently-updated attention cache and prompt these workers to decide how best to collaborate. Our approach allows the LLM instances to come up with their own collaboration strategy for the problem at hand, all the while "seeing" each other's memory in the concurrent KV cache. We implement this approach via Hogwild! Inference: a parallel LLM inference engine where multiple instances of the same LLM run in parallel with the same attention cache, with "instant" access to each other's memory. Hogwild! Inference takes advantage of Rotary Position Embeddings (RoPE) to avoid recomputation while improving parallel hardware utilization. We find that modern reasoning-capable LLMs can perform inference with shared Key-Value cache out of the box, without additional fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13233v1">LLM-based Multi-Agent System for Simulating Strategic and Goal-Oriented Data Marketplaces</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 10 pages, 12 figures
    </div>
    <details class="paper-abstract">
      Data marketplaces, which mediate the purchase and exchange of data from third parties, have attracted growing attention for reducing the cost and effort of data collection while enabling the trading of diverse datasets. However, a systematic understanding of the interactions between market participants, data, and regulations remains limited. To address this gap, we propose a Large Language Model-based Multi-Agent System (LLM-MAS) for data marketplaces. In our framework, buyer and seller agents powered by LLMs operate with explicit objectives and autonomously perform strategic actions, such as planning, searching, purchasing, pricing, and updating data. These agents can reason about market dynamics, forecast future demand, and adjust strategies accordingly. Unlike conventional model-based simulations, which are typically constrained to predefined rules, LLM-MAS supports broader and more adaptive behavior selection through natural language reasoning. We evaluated the framework via simulation experiments using three distribution-based metrics: (1) the number of purchases per dataset, (2) the number of purchases per buyer, and (3) the number of repeated purchases of the same dataset. The results demonstrate that LLM-MAS more faithfully reproduces trading patterns observed in real data marketplaces compared to traditional approaches, and further captures the emergence and evolution of market trends.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2408.14398v4">On the Limitations of Language Targeted Pruning: Investigating the Calibration Language Impact in Multilingual LLM Pruning</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 Accepted for publication in TACL
    </div>
    <details class="paper-abstract">
      Recent advances in large language model (LLM) pruning have shown state-of-the-art (SotA) compression results in post-training and retraining-free settings while maintaining high predictive performance. However, previous research mainly considered calibrating based on English text, despite the multilingual nature of modern LLMs and their frequent use in non-English languages. This analysis paper conducts an in-depth investigation of the performance and internal representation changes associated with pruning multilingual language models for monolingual applications. We present the first comprehensive empirical study, comparing different calibration languages for pruning multilingual models across diverse languages, tasks, models, and SotA pruning techniques. We further analyze the latent subspaces, pruning masks, and individual neurons within pruned models. Our results reveal that while calibration on the target language effectively retains perplexity and yields high signal-to-noise ratios, it does not consistently improve downstream task performance. Further analysis of internal representations at three different levels highlights broader limitations of current pruning approaches: While they effectively preserve dominant information like language-specific features, this is insufficient to counteract the loss of nuanced, language-agnostic features that are crucial for knowledge retention and reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13223v1">TokenSqueeze: Performance-Preserving Compression for Reasoning LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 Accepted to NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Emerging reasoning LLMs such as OpenAI-o1 and DeepSeek-R1 have achieved strong performance on complex reasoning tasks by generating long chain-of-thought (CoT) traces. However, these long CoTs result in increased token usage, leading to higher inference latency and memory consumption. As a result, balancing accuracy and reasoning efficiency has become essential for deploying reasoning LLMs in practical applications. Existing long-to-short (Long2Short) methods aim to reduce inference length but often sacrifice accuracy, revealing a need for an approach that maintains performance while lowering token costs. To address this efficiency-accuracy tradeoff, we propose TokenSqueeze, a novel Long2Short method that condenses reasoning paths while preserving performance and relying exclusively on self-generated data. First, to prevent performance degradation caused by excessive compression of reasoning depth, we propose to select self-generated samples whose reasoning depth is adaptively matched to the complexity of the problem. To further optimize the linguistic expression without altering the underlying reasoning paths, we introduce a distribution-aligned linguistic refinement method that enhances the clarity and conciseness of the reasoning path while preserving its logical integrity. Comprehensive experimental results demonstrate the effectiveness of TokenSqueeze in reducing token usage while maintaining accuracy. Notably, DeepSeek-R1-Distill-Qwen-7B fine-tuned using our proposed method achieved a 50\% average token reduction while preserving accuracy on the MATH500 benchmark. TokenSqueeze exclusively utilizes the model's self-generated data, enabling efficient and high-fidelity reasoning without relying on manually curated short-answer datasets across diverse applications. Our code is available at https://github.com/zhangyx1122/TokenSqueeze.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.00034v2">Is Our Chatbot Telling Lies? Assessing Correctness of an LLM-based Dutch Support Chatbot</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 10 pages + 2 pages references, 4 figures
    </div>
    <details class="paper-abstract">
      Companies support their customers using live chats and chatbots to gain their loyalty. AFAS is a Dutch company aiming to leverage the opportunity large language models (LLMs) offer to answer customer queries with minimal to no input from its customer support team. Adding to its complexity, it is unclear what makes a response correct, and that too in Dutch. Further, with minimal data available for training, the challenge is to identify whether an answer generated by a large language model is correct and do it on the fly. This study is the first to define the correctness of a response based on how the support team at AFAS makes decisions. It leverages literature on natural language generation and automated answer grading systems to automate the decision-making of the customer support team. We investigated questions requiring a binary response (e.g., Would it be possible to adjust tax rates manually?) or instructions (e.g., How would I adjust tax rate manually?) to test how close our automated approach reaches support rating. Our approach can identify wrong messages in 55\% of the cases. This work demonstrates the potential for automatically assessing when our chatbot may provide incorrect or misleading answers. Specifically, we contribute (1) a definition and metrics for assessing correctness, and (2) suggestions to improve correctness with respect to regional language and question type.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.16407v2">CREME: Robustness Enhancement of Code LLMs via Layer-Aware Model Editing</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive capabilities in code generation, where the natural language prompt plays a crucial role in conveying user intent to the model. However, prior studies have shown that LLMs are highly sensitive to prompt perturbations. Minor modifications in wording, syntax, or formatting can significantly reduce the functional correctness of generated code. As perturbations frequently occur in real-world scenarios, improving the robustness of LLMs to prompt perturbations is essential for ensuring reliable performance in practical code generation. In this paper, we introduce CREME (Code Robustness Enhancement via Model Editing), a novel approach that enhances LLM robustness through targeted parameter updates. CREME first identifies robustness-sensitive layers by comparing hidden states between an original prompt and its perturbed variant. Then, it performs lightweight parameter editing at the identified layer to reduce performance degradation. We evaluate CREME on two widely used code generation benchmarks (HumanEval and MBPP) along with their perturbed counterparts. Experimental results show that CREME improves Pass@1 accuracy by 63% on perturbed prompts while maintaining stable performance on clean inputs, with accuracy deviations within 1%. Further analysis reveals that robustness-sensitive layers are primarily concentrated in the middle and deeper layers of the network, and their locations vary across different model architectures. These insights provide a valuable foundation for developing future robustness-oriented editing strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13169v1">TCM-5CEval: Extended Deep Evaluation Benchmark for LLM's Comprehensive Clinical Research Competence in Traditional Chinese Medicine</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 17 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated exceptional capabilities in general domains, yet their application in highly specialized and culturally-rich fields like Traditional Chinese Medicine (TCM) requires rigorous and nuanced evaluation. Building upon prior foundational work such as TCM-3CEval, which highlighted systemic knowledge gaps and the importance of cultural-contextual alignment, we introduce TCM-5CEval, a more granular and comprehensive benchmark. TCM-5CEval is designed to assess LLMs across five critical dimensions: (1) Core Knowledge (TCM-Exam), (2) Classical Literacy (TCM-LitQA), (3) Clinical Decision-making (TCM-MRCD), (4) Chinese Materia Medica (TCM-CMM), and (5) Clinical Non-pharmacological Therapy (TCM-ClinNPT). We conducted a thorough evaluation of fifteen prominent LLMs, revealing significant performance disparities and identifying top-performing models like deepseek\_r1 and gemini\_2\_5\_pro. Our findings show that while models exhibit proficiency in recalling foundational knowledge, they struggle with the interpretative complexities of classical texts. Critically, permutation-based consistency testing reveals widespread fragilities in model inference. All evaluated models, including the highest-scoring ones, displayed a substantial performance degradation when faced with varied question option ordering, indicating a pervasive sensitivity to positional bias and a lack of robust understanding. TCM-5CEval not only provides a more detailed diagnostic tool for LLM capabilities in TCM but aldso exposes fundamental weaknesses in their reasoning stability. To promote further research and standardized comparison, TCM-5CEval has been uploaded to the Medbench platform, joining its predecessor in the "In-depth Challenge for Comprehensive TCM Abilities" special track.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.22564v2">Exploiting Synergistic Cognitive Biases to Bypass Safety in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate impressive capabilities across a wide range of tasks, yet their safety mechanisms remain susceptible to adversarial attacks that exploit cognitive biases -- systematic deviations from rational judgment. Unlike prior jailbreaking approaches focused on prompt engineering or algorithmic manipulation, this work highlights the overlooked power of multi-bias interactions in undermining LLM safeguards. We propose CognitiveAttack, a novel red-teaming framework that systematically leverages both individual and combined cognitive biases. By integrating supervised fine-tuning and reinforcement learning, CognitiveAttack generates prompts that embed optimized bias combinations, effectively bypassing safety protocols while maintaining high attack success rates. Experimental results reveal significant vulnerabilities across 30 diverse LLMs, particularly in open-source models. CognitiveAttack achieves a substantially higher attack success rate compared to the SOTA black-box method PAP (60.1% vs. 31.6%), exposing critical limitations in current defense mechanisms. These findings highlight multi-bias interactions as a powerful yet underexplored attack vector. This work introduces a novel interdisciplinary perspective by bridging cognitive science and LLM safety, paving the way for more robust and human-aligned AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13147v1">OTARo: Once Tuning for All Precisions toward Robust On-Device LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) fine-tuning techniques not only improve the adaptability to diverse downstream tasks, but also mitigate adverse effects of model quantization. Despite this, conventional quantization suffers from its structural limitation that hinders flexibility during the fine-tuning and deployment stages. Practical on-device tasks demand different quantization precisions (i.e. different bit-widths), e.g., understanding tasks tend to exhibit higher tolerance to reduced precision compared to generation tasks. Conventional quantization, typically relying on scaling factors that are incompatible across bit-widths, fails to support the on-device switching of precisions when confronted with complex real-world scenarios. To overcome the dilemma, we propose OTARo, a novel method that enables on-device LLMs to flexibly switch quantization precisions while maintaining performance robustness through once fine-tuning. OTARo introduces Shared Exponent Floating Point (SEFP), a distinct quantization mechanism, to produce different bit-widths through simple mantissa truncations of a single model. Moreover, to achieve bit-width robustness in downstream applications, OTARo performs a learning process toward losses induced by different bit-widths. The method involves two critical strategies: (1) Exploitation-Exploration Bit-Width Path Search (BPS), which iteratively updates the search path via a designed scoring mechanism; (2) Low-Precision Asynchronous Accumulation (LAA), which performs asynchronous gradient accumulations and delayed updates under low bit-widths. Experiments on popular LLMs, e.g., LLaMA3.2-1B, LLaMA3-8B, demonstrate that OTARo achieves consistently strong and robust performance for all precisions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.22963v3">CompressionAttack: Exploiting Prompt Compression as a New Attack Surface in LLM-Powered Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      LLM-powered agents often use prompt compression to reduce inference costs, but this introduces a new security risk. Compression modules, which are optimized for efficiency rather than safety, can be manipulated by adversarial inputs, causing semantic drift and altering LLM behavior. This work identifies prompt compression as a novel attack surface and presents CompressionAttack, the first framework to exploit it. CompressionAttack includes two strategies: HardCom, which uses discrete adversarial edits for hard compression, and SoftCom, which performs latent-space perturbations for soft compression. Experiments on multiple LLMs show up to an average ASR of 83% and 87% in two tasks, while remaining highly stealthy and transferable. Case studies in three practical scenarios confirm real-world impact, and current defenses prove ineffective, highlighting the need for stronger protections.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.01891v2">Multi-Personality Generation of LLMs at Decoding-time</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 Accepted by WSDM 2026
    </div>
    <details class="paper-abstract">
      Multi-personality generation for LLMs, enabling simultaneous embodiment of multiple personalization attributes, is a fundamental challenge. Existing retraining-based approaches are costly and poorly scalable, while decoding-time methods often rely on external models or heuristics, limiting flexibility and robustness. In this paper, we propose a novel Multi-Personality Generation (MPG) framework under the decoding-time combination paradigm. It flexibly controls multi-personality without relying on scarce multi-dimensional models or extra training, leveraging implicit density ratios in single-dimensional models as a "free lunch" to reformulate the task as sampling from a target strategy aggregating these ratios. To implement MPG efficiently, we design Speculative Chunk-level based Rejection sampling (SCR), which generates responses in chunks and parallelly validates them via estimated thresholds within a sliding window. This significantly reduces computational overhead while maintaining high-quality generation. Experiments on MBTI personality and Role-Playing demonstrate the effectiveness of MPG, showing improvements up to 16%-18%. Code and data are available at https://github.com/Libra117/MPG .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.00829v2">Exposing the Cracks: Vulnerabilities of Retrieval-Augmented LLM-based Machine Translation</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 Accepted by AAAI 2026
    </div>
    <details class="paper-abstract">
      \textbf{RE}trieval-\textbf{A}ugmented \textbf{L}LM-based \textbf{M}achine \textbf{T}ranslation (REAL-MT) shows promise for knowledge-intensive tasks like idiomatic translation, but its reliability under noisy retrieval contexts remains poorly understood despite this being a common challenge in real-world deployment. To address this gap, we propose a noise synthesis framework and new metrics to evaluate the robustness of REAL-MT systematically. Using this framework, we instantiate REAL-MT with Qwen-series models, including standard LLMs and large reasoning models (LRMs) with enhanced reasoning, and evaluate their performance on idiomatic translation across high-, medium-, and low-resource language pairs under synthesized noise. Our results show that low-resource language pairs, which rely more heavily on retrieved context, degrade more severely under noise than high-resource ones and often produce nonsensical translations. Although LRMs possess enhanced reasoning capabilities, they show no improvement in error correction and are even more susceptible to noise, tending to rationalize incorrect contexts. We find that this stems from an attention shift away from the source idiom to noisy content, while confidence increases despite declining accuracy, indicating poor calibration. To mitigate these issues, we investigate training-free and fine-tuning strategies, which improve robustness at the cost of performance in clean contexts, revealing a fundamental trade-off. Our findings highlight the limitations of current approaches, underscoring the need for self-verifying integration mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.13768v3">Evaluation-Driven Development and Operations of LLM Agents: A Process Model and Reference Architecture</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 Revised based on review comments. Submission under review
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have enabled the emergence of LLM agents, systems capable of pursuing under-specified goals and adapting after deployment. Evaluating such agents is challenging because their behavior is open ended, probabilistic, and shaped by system-level interactions over time. Traditional evaluation methods, built around fixed benchmarks and static test suites, fail to capture emergent behaviors or support continuous adaptation across the lifecycle. To ground a more systematic approach, we conduct a multivocal literature review (MLR) synthesizing academic and industrial evaluation practices. The findings directly inform two empirically derived artifacts: a process model and a reference architecture that embed evaluation as a continuous, governing function rather than a terminal checkpoint. Together they constitute the evaluation-driven development and operations (EDDOps) approach, which unifies offline (development-time) and online (runtime) evaluation within a closed feedback loop. By making evaluation evidence drive both runtime adaptation and governed redevelopment, EDDOps supports safer, more traceable evolution of LLM agents aligned with changing objectives, user needs, and governance constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13007v1">GEM: Generative Entropy-Guided Preference Modeling for Few-shot Alignment of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 This paper has been accepted by AAAI 2026-AIA and designated as an oral presentation paper
    </div>
    <details class="paper-abstract">
      Alignment of large language models (LLMs) with human preferences typically relies on supervised reward models or external judges that demand abundant annotations. However, in fields that rely on professional knowledge, such as medicine and law, such large-scale preference labels are often unachievable. In this paper, we propose a generative entropy-guided preference modeling approach named GEM for LLMs aligment at low-resource and domain-specific scenarios. Instead of training a discriminative reward model on preference data, we directly train the LLM to internalize a closed-loop optimization architecture that can extract and exploit the multi-dimensional, fine-grained cognitive signals implicit in human preferences. Specifically, our Cognitive Filtering module, based on entropy theory in decision making, first leverages Chain-of-Thought (CoT) prompting to generate diverse candidate reasoning chains (CoTs) from preference data. Subsequently, it introduces a token scoring mechanism to rank and weight the sampled CoTs, boosting the importance of high-confidence answers and strategically high-entropy tokens. Building on these filtered preferences, we fine-tune the LLM using a novel self-evaluated group advantage algorithm, SEGA, which effectively aggregates group-level cognitive signals and transforms the entropy-based scores into implicit rewards for policy optimization. In these ways, GEM empowers the LLM to rely on its own judgments and establishes an entropy-guided closed-loop cognitive optimization framework, enabling highly efficient few-shot alignment of LLMs. Experiments on general benchmarks and domain-specific tasks (such as mathematical reasoning and medical dialogues) demonstrate that our GEM achieves significant improvements with few-shot preference data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06852v3">Differentiated Directional Intervention A Framework for Evading LLM Safety Alignment</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 AAAI-26-AIA
    </div>
    <details class="paper-abstract">
      Safety alignment instills in Large Language Models (LLMs) a critical capacity to refuse malicious requests. Prior works have modeled this refusal mechanism as a single linear direction in the activation space. We posit that this is an oversimplification that conflates two functionally distinct neural processes: the detection of harm and the execution of a refusal. In this work, we deconstruct this single representation into a Harm Detection Direction and a Refusal Execution Direction. Leveraging this fine-grained model, we introduce Differentiated Bi-Directional Intervention (DBDI), a new white-box framework that precisely neutralizes the safety alignment at critical layer. DBDI applies adaptive projection nullification to the refusal execution direction while suppressing the harm detection direction via direct steering. Extensive experiments demonstrate that DBDI outperforms prominent jailbreaking methods, achieving up to a 97.88\% attack success rate on models such as Llama-2. By providing a more granular and mechanistic framework, our work offers a new direction for the in-depth understanding of LLM safety alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12991v1">Fine-Tuned LLMs Know They Don't Know: A Parameter-Efficient Approach to Recovering Honesty</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 Accepted by AAAI 2026 Main Track
    </div>
    <details class="paper-abstract">
      The honesty of Large Language Models (LLMs) is increasingly important for safe deployment in high-stakes domains. However, this crucial trait is severely undermined by supervised fine-tuning (SFT), a common technique for model specialization. Existing recovery methods rely on data-intensive global parameter adjustments, implicitly assuming that SFT deeply corrupts the models' ability to recognize their knowledge boundaries. However, we observe that fine-tuned LLMs still preserve this ability; what is damaged is their capacity to faithfully express that awareness. Building on this, we propose Honesty-Critical Neurons Restoration (HCNR) to surgically repair this suppressed capacity. HCNR identifies and restores key expression-governing neurons to their pre-trained state while harmonizing them with task-oriented neurons via Hessian-guided compensation. Experiments on four QA tasks and five LLM families demonstrate that HCNR effectively recovers 33.25% of the compromised honesty while achieving at least 2.23x speedup with over 10x less data compared to baseline methods, offering a practical solution for trustworthy LLM deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12977v1">ArtiWorld: LLM-Driven Articulation of 3D Objects in Scenes</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Building interactive simulators and scalable robot-learning environments requires a large number of articulated assets. However, most existing 3D assets in simulation are rigid, and manually converting them into articulated objects is extremely labor- and cost-intensive. This raises a natural question: can we automatically identify articulable objects in a scene and convert them into articulated assets directly? In this paper, we present ArtiWorld, a scene-aware pipeline that localizes candidate articulable objects from textual scene descriptions and reconstructs executable URDF models that preserve the original geometry. At the core of this pipeline is Arti4URDF, which leverages 3D point cloud, prior knowledge of a large language model (LLM), and a URDF-oriented prompt design to rapidly convert rigid objects into interactive URDF-based articulated objects while maintaining their 3D shape. We evaluate ArtiWorld at three levels: 3D simulated objects, full 3D simulated scenes, and real-world scan scenes. Across all three settings, our method consistently outperforms existing approaches and achieves state-of-the-art performance, while preserving object geometry and correctly capturing object interactivity to produce usable URDF-based articulated models. This provides a practical path toward building interactive, robot-ready simulation environments directly from existing 3D assets. Code and data will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.23188v2">Diagnose, Localize, Align: A Full-Stack Framework for Reliable LLM Multi-Agent Systems under Instruction Conflicts</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 Upon further review, we realized that the version submitted to arXiv was not the final draft and omits crucial results and discussion. To avoid confusion and ensure the integrity of the record, we request withdrawal and will resubmit once the complete work is ready
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-powered multi-agent systems (MAS) have rapidly advanced collaborative reasoning, tool use, and role-specialized coordination in complex tasks. However, reliability-critical deployment remains hindered by a systemic failure mode: hierarchical compliance under instruction conflicts (system-user, peer-peer), where agents misprioritize system-level rules in the presence of competing demands. Moreover, widely used macro-level metrics (e.g., pass@k) obscure these micro-level violations and offer little actionable guidance for remedy. In this work, we present a full-stack, three-stage framework: (1) Diagnose - Contextualized Role Adherence Score (CRAS), a query-wise, context-aware scoring metric that decomposes role adherence into four measurable dimensions; (2) Localize - attention drift analysis revealing that instruction conflicts are resolved by attention heads that are largely concentrated in middle layers; (3) Align - Surgical Alignment of Instruction Layers (SAIL), which installs LoRA only on the localized focal layers and optimizes a token-weighted DPO-style preference objective that credits tokens by their focal attentional contribution. Across standard benchmarks and MAS frameworks, our surgical approach improves instruction hierarchy compliance (e.g., +5.60% with AutoGen on MedQA) without full-model finetuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12922v1">Tokenize Once, Recommend Anywhere: Unified Item Tokenization for Multi-domain LLM-based Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 20 pages, 8 figures, 9 tables; Annual AAAI Conference on Artificial Intelligence (AAAI-26) (to appear) (Please cite our conference version.)
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based recommender systems have achieved high-quality performance by bridging the discrepancy between the item space and the language space through item tokenization. However, existing item tokenization methods typically require training separate models for each item domain, limiting generalization. Moreover, the diverse distributions and semantics across item domains make it difficult to construct a unified tokenization that preserves domain-specific information. To address these challenges, we propose UniTok, a Unified item Tokenization framework that integrates our own mixture-of-experts (MoE) architecture with a series of codebooks to convert items into discrete tokens, enabling scalable tokenization while preserving semantic information across multiple item domains. Specifically, items from different domains are first projected into a unified latent space through a shared encoder. They are then routed to domain-specific experts to capture the unique semantics, while a shared expert, which is always active, encodes common knowledge transferable across domains. Additionally, to mitigate semantic imbalance across domains, we present a mutual information calibration mechanism, which guides the model towards retaining similar levels of semantic information for each domain. Comprehensive experiments on wide-ranging real-world datasets demonstrate that the proposed UniTok framework is (a) highly effective: achieving up to 51.89% improvements over strong benchmarks, (b) theoretically sound: showing the analytical validity of our architectural design and optimization; and (c) highly generalizable: demonstrating robust performance across diverse domains without requiring per-domain retraining, a capability not supported by existing baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11000v2">DialogGraph-LLM: Graph-Informed LLMs for End-to-End Audio Dialogue Intent Recognition</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 8 pages, 2 figures. To appear in: Proceedings of the 28th European Conference on Artificial Intelligence (ECAI 2025), Frontiers in Artificial Intelligence and Applications, Vol. 413. DOI: 10.3233/FAIA251182
    </div>
    <details class="paper-abstract">
      Recognizing speaker intent in long audio dialogues among speakers has a wide range of applications, but is a non-trivial AI task due to complex inter-dependencies in speaker utterances and scarce annotated data. To address these challenges, an end-to-end framework, namely DialogGraph-LLM, is proposed in the current work. DialogGraph-LLM combines a novel Multi-Relational Dialogue Attention Network (MR-DAN) architecture with multimodal foundation models (e.g., Qwen2.5-Omni-7B) for direct acoustic-to-intent inference. An adaptive semi-supervised learning strategy is designed using LLM with a confidence-aware pseudo-label generation mechanism based on dual-threshold filtering using both global and class confidences, and an entropy-based sample selection process that prioritizes high-information unlabeled instances. Extensive evaluations on the proprietary MarketCalls corpus and the publicly available MIntRec 2.0 benchmark demonstrate DialogGraph-LLM's superiority over strong audio and text-driven baselines. The framework demonstrates strong performance and efficiency in intent recognition in real world scenario audio dialogues, proving its practical value for audio-rich domains with limited supervision. Our code is available at https://github.com/david188888/DialogGraph-LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12901v1">Online Learning of HTN Methods for integrated LLM-HTN Planning</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 The Twelfth Annual Conference on Advances in Cognitive Systems (ACS-2025)
    </div>
    <details class="paper-abstract">
      We present online learning of Hierarchical Task Network (HTN) methods in the context of integrated HTN planning and LLM-based chatbots. Methods indicate when and how to decompose tasks into subtasks. Our method learner is built on top of the ChatHTN planner. ChatHTN queries ChatGPT to generate a decomposition of a task into primitive tasks when no applicable method for the task is available. In this work, we extend ChatHTN. Namely, when ChatGPT generates a task decomposition, ChatHTN learns from it, akin to memoization. However, unlike memoization, it learns a generalized method that applies not only to the specific instance encountered, but to other instances of the same task. We conduct experiments on two domains and demonstrate that our online learning procedure reduces the number of calls to ChatGPT while solving at least as many problems, and in some cases, even more.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12869v1">On the Fundamental Limits of LLMs at Scale</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 Submitted to TMLR 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have benefited enormously from scaling, yet these gains are bounded by five fundamental limitations: (1) hallucination, (2) context compression, (3) reasoning degradation, (4) retrieval fragility, and (5) multimodal misalignment. While existing surveys describe these phenomena empirically, they lack a rigorous theoretical synthesis connecting them to the foundational limits of computation, information, and learning. This work closes that gap by presenting a unified, proof-informed framework that formalizes the innate theoretical ceilings of LLM scaling. First, computability and uncomputability imply an irreducible residue of error: for any computably enumerable model family, diagonalization guarantees inputs on which some model must fail, and undecidable queries (e.g., halting-style tasks) induce infinite failure sets for all computable predictors. Second, information-theoretic and statistical constraints bound attainable accuracy even on decidable tasks, finite description length enforces compression error, and long-tail factual knowledge requires prohibitive sample complexity. Third, geometric and computational effects compress long contexts far below their nominal size due to positional under-training, encoding attenuation, and softmax crowding. We further show how likelihood-based training favors pattern completion over inference, how retrieval under token limits suffers from semantic drift and coupling noise, and how multimodal scaling inherits shallow cross-modal alignment. Across sections, we pair theorems and empirical evidence to outline where scaling helps, where it saturates, and where it cannot progress, providing both theoretical foundations and practical mitigation paths like bounded-oracle retrieval, positional curricula, and sparse or hierarchical attention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12867v1">Bootstrapping LLMs via Preference-Based Policy Optimization</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Bootstrapping large language models (LLMs) through preference-based policy optimization offers a promising direction for aligning model behavior with human preferences without relying on extensive manual annotations. In this work, we propose a novel preference-based policy optimization (PbPO) framework that formulates the learning process as a min-max game between the main policy and a reward model (RM). The RM is constrained within a confidence set derived from preference data to ensure reliable exploitation. Our iterative online algorithm actively collects preference data through guided exploration of the evolving policy, enabling continual self-improvement of both the policy and the RM. We provide theoretical guarantees for our method, establishing high-probability regret bounds for both settings with sequence-level RM and token-level RM, demonstrating its effectiveness in bootstrapping LLMs. Extensive experiments on five benchmarks show that our approach consistently outperforms existing state-of-the-art preference optimization techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12860v1">Dissecting and Re-architecting 3D NAND Flash PIM Arrays for Efficient Single-Batch Token Generation in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 This paper is accepted in the 43rd IEEE International Conference on Computer Design (ICCD), 2025
    </div>
    <details class="paper-abstract">
      The advancement of large language models has led to models with billions of parameters, significantly increasing memory and compute demands. Serving such models on conventional hardware is challenging due to limited DRAM capacity and high GPU costs. Thus, in this work, we propose offloading the single-batch token generation to a 3D NAND flash processing-in-memory (PIM) device, leveraging its high storage density to overcome the DRAM capacity wall. We explore 3D NAND flash configurations and present a re-architected PIM array with an H-tree network for optimal latency and cell density. Along with the well-chosen PIM array size, we develop operation tiling and mapping methods for LLM layers, achieving a 2.4x speedup over four RTX4090 with vLLM and comparable performance to four A100 with only 4.9% latency overhead. Our detailed area analysis reveals that the proposed 3D NAND flash PIM architecture can be integrated within a 4.98mm2 die area under the memory array, without extra area overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14803v1">Scalable and Efficient Large-Scale Log Analysis with LLMs: An IT Software Support Case Study</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      IT environments typically have logging mechanisms to monitor system health and detect issues. However, the huge volume of generated logs makes manual inspection impractical, highlighting the importance of automated log analysis in IT Software Support. In this paper, we propose a log analytics tool that leverages Large Language Models (LLMs) for log data processing and issue diagnosis, enabling the generation of automated insights and summaries. We further present a novel approach for efficiently running LLMs on CPUs to process massive log volumes in minimal time without compromising output quality. We share the insights and lessons learned from deployment of the tool - in production since March 2024 - scaled across 70 software products, processing over 2000 tickets for issue diagnosis, achieving a time savings of 300+ man hours and an estimated $15,444 per month in manpower costs compared to the traditional log analysis practices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13998v1">LoCoBench-Agent: An Interactive Benchmark for LLM Agents in Long-Context Software Engineering</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 54-pages
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) evolve into sophisticated autonomous agents capable of complex software development tasks, evaluating their real-world capabilities becomes critical. While existing benchmarks like LoCoBench~\cite{qiu2025locobench} assess long-context code understanding, they focus on single-turn evaluation and cannot capture the multi-turn interactive nature, tool usage patterns, and adaptive reasoning required by real-world coding agents. We introduce \textbf{LoCoBench-Agent}, a comprehensive evaluation framework specifically designed to assess LLM agents in realistic, long-context software engineering workflows. Our framework extends LoCoBench's 8,000 scenarios into interactive agent environments, enabling systematic evaluation of multi-turn conversations, tool usage efficiency, error recovery, and architectural consistency across extended development sessions. We also introduce an evaluation methodology with 9 metrics across comprehension and efficiency dimensions. Our framework provides agents with 8 specialized tools (file operations, search, code analysis) and evaluates them across context lengths ranging from 10K to 1M tokens, enabling precise assessment of long-context performance. Through systematic evaluation of state-of-the-art models, we reveal several key findings: (1) agents exhibit remarkable long-context robustness; (2) comprehension-efficiency trade-off exists with negative correlation, where thorough exploration increases comprehension but reduces efficiency; and (3) conversation efficiency varies dramatically across models, with strategic tool usage patterns differentiating high-performing agents. As the first long-context LLM agent benchmark for software engineering, LoCoBench-Agent establishes a rigorous foundation for measuring agent capabilities, identifying performance gaps, and advancing autonomous software development at scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13994v1">Hint-Augmented Re-ranking: Efficient Product Search using LLM-Based Query Decomposition</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 AACL 2025
    </div>
    <details class="paper-abstract">
      Search queries with superlatives (e.g., best, most popular) require comparing candidates across multiple dimensions, demanding linguistic understanding and domain knowledge. We show that LLMs can uncover latent intent behind these expressions in e-commerce queries through a framework that extracts structured interpretations or hints. Our approach decomposes queries into attribute-value hints generated concurrently with retrieval, enabling efficient integration into the ranking pipeline. Our method improves search performanc eby 10.9 points in MAP and ranking by 5.9 points in MRR over baselines. Since direct LLM-based reranking faces prohibitive latency, we develop an efficient approach transferring superlative interpretations to lightweight models. Our findings provide insights into how superlative semantics can be represented and transferred between models, advancing linguistic interpretation in retrieval systems while addressing practical deployment constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13984v1">Node-Level Uncertainty Estimation in LLM-Generated SQL</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      We present a practical framework for detecting errors in LLM-generated SQL by estimating uncertainty at the level of individual nodes in the query's abstract syntax tree (AST). Our approach proceeds in two stages. First, we introduce a semantically aware labeling algorithm that, given a generated SQL and a gold reference, assigns node-level correctness without over-penalizing structural containers or alias variation. Second, we represent each node with a rich set of schema-aware and lexical features - capturing identifier validity, alias resolution, type compatibility, ambiguity in scope, and typo signals - and train a supervised classifier to predict per-node error probabilities. We interpret these probabilities as calibrated uncertainty, enabling fine-grained diagnostics that pinpoint exactly where a query is likely to be wrong. Across multiple databases and datasets, our method substantially outperforms token log-probabilities: average AUC improves by +27.44% while maintaining robustness under cross-database evaluation. Beyond serving as an accuracy signal, node-level uncertainty supports targeted repair, human-in-the-loop review, and downstream selective execution. Together, these results establish node-centric, semantically grounded uncertainty estimation as a strong and interpretable alternative to aggregate sequence level confidence measures.
    </details>
</div>
