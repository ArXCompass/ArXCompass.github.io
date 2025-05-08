# llm - 2025_04

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- Part 8
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07199v2">SemEval-2025 Task 5: LLMs4Subjects -- LLM-based Automated Subject Tagging for a National Technical Library's Open-Access Catalog</a></div>
    <div class="paper-meta">
      📅 2025-04-11
      | 💬 10 pages, 4 figures, Accepted as SemEval 2025 Task 5 description paper
    </div>
    <details class="paper-abstract">
      We present SemEval-2025 Task 5: LLMs4Subjects, a shared task on automated subject tagging for scientific and technical records in English and German using the GND taxonomy. Participants developed LLM-based systems to recommend top-k subjects, evaluated through quantitative metrics (precision, recall, F1-score) and qualitative assessments by subject specialists. Results highlight the effectiveness of LLM ensembles, synthetic data generation, and multilingual processing, offering insights into applying LLMs for digital library classification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07557v2">Using LLMs for Analyzing AIS Data</a></div>
    <div class="paper-meta">
      📅 2025-04-11
    </div>
    <details class="paper-abstract">
      Recent research in Large Language Models (LLMs), has had a profound impact across various fields, including mobility data science. This paper explores the and experiment with different approaches to using LLMs for analyzing AIS data. We propose a set of carefully designed queries to assess the reasoning capabilities of LLMs in this kind of tasks. Further, we experiment with four different methods: (1) using LLMs as a natural language interface to a spatial database, (2) reasoning on raw data, (3) reasoning on compressed trajectories, and (4) reasoning on semantic trajectories. We investigate the strengths and weaknesses for the four methods, and discuss the findings. The goal is to provide valuable insights for both researchers and practitioners on selecting the most appropriate LLM-based method depending on their specific data analysis objectives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08378v1">Scaling Up On-Device LLMs via Active-Weight Swapping Between DRAM and Flash</a></div>
    <div class="paper-meta">
      📅 2025-04-11
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly being deployed on mobile devices, but the limited DRAM capacity constrains the deployable model size. This paper introduces ActiveFlow, the first LLM inference framework that can achieve adaptive DRAM usage for modern LLMs (not ReLU-based), enabling the scaling up of deployable model sizes. The framework is based on the novel concept of active weight DRAM-flash swapping and incorporates three novel techniques: (1) Cross-layer active weights preloading. It uses the activations from the current layer to predict the active weights of several subsequent layers, enabling computation and data loading to overlap, as well as facilitating large I/O transfers. (2) Sparsity-aware self-distillation. It adjusts the active weights to align with the dense-model output distribution, compensating for approximations introduced by contextual sparsity. (3) Active weight DRAM-flash swapping pipeline. It orchestrates the DRAM space allocation among the hot weight cache, preloaded active weights, and computation-involved weights based on available memory. Results show ActiveFlow achieves the performance-cost Pareto frontier compared to existing efficiency optimization methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17365v2">How Effective Is Constitutional AI in Small LLMs? A Study on DeepSeek-R1 and Its Peers</a></div>
    <div class="paper-meta">
      📅 2025-04-11
    </div>
    <details class="paper-abstract">
      Recent incidents highlight safety risks in Large Language Models (LLMs), motivating research into alignment methods like Constitutional AI (CAI). This paper explores CAI's self-critique mechanism on small, uncensored 7-9B parameter models: DeepSeek-R1-8B, Gemma-2-9B, Llama 3.1-8B, and Qwen2.5-7B. We show that while Llama-based models exhibited significant harm reduction through self-critique, other architectures demonstrated less improvement in harm detection after abliteration. These results suggest CAI's effectiveness may vary depending on model architecture and reasoning capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07583v2">Do LLMs Understand Your Translations? Evaluating Paragraph-level MT with Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-04-11
    </div>
    <details class="paper-abstract">
      Despite the steady progress in machine translation evaluation, existing automatic metrics struggle to capture how well meaning is preserved beyond sentence boundaries. We posit that reliance on a single intrinsic quality score, trained to mimic human judgments, might be insufficient for evaluating translations of long, complex passages, and a more ``pragmatic'' approach that assesses how accurately key information is conveyed by a translation in context is needed. We introduce TREQA (Translation Evaluation via Question-Answering), a framework that extrinsically evaluates translation quality by assessing how accurately candidate translations answer reading comprehension questions that target key information in the original source or reference texts. In challenging domains that require long-range understanding, such as literary texts, we show that TREQA is competitive with and, in some cases, outperforms state-of-the-art neural and LLM-based metrics in ranking alternative paragraph-level translations, despite never being explicitly optimized to correlate with human judgments. Furthermore, the generated questions and answers offer interpretability: empirical analysis shows that they effectively target translation errors identified by experts in evaluated datasets. Our code is available at https://github.com/deep-spin/treqa
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02623v2">Multi-Mission Tool Bench: Assessing the Robustness of LLM based Agents through Related and Dynamic Missions</a></div>
    <div class="paper-meta">
      📅 2025-04-11
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate strong potential as agents for tool invocation due to their advanced comprehension and planning capabilities. Users increasingly rely on LLM-based agents to solve complex missions through iterative interactions. However, existing benchmarks predominantly access agents in single-mission scenarios, failing to capture real-world complexity. To bridge this gap, we propose the Multi-Mission Tool Bench. In the benchmark, each test case comprises multiple interrelated missions. This design requires agents to dynamically adapt to evolving demands. Moreover, the proposed benchmark explores all possible mission-switching patterns within a fixed mission number. Specifically, we propose a multi-agent data generation framework to construct the benchmark. We also propose a novel method to evaluate the accuracy and efficiency of agent decisions with dynamic decision trees. Experiments on diverse open-source and closed-source LLMs reveal critical factors influencing agent robustness and provide actionable insights to the tool invocation society.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08312v1">SortBench: Benchmarking LLMs based on their ability to sort lists</a></div>
    <div class="paper-meta">
      📅 2025-04-11
    </div>
    <details class="paper-abstract">
      Sorting is a tedious but simple task for human intelligence and can be solved fairly easily algorithmically. However, for Large Language Models (LLMs) this task is surprisingly hard, as some properties of sorting are among known weaknesses of LLMs: being faithful to the input data, logical comparisons between values, and strictly differentiating between syntax (used for sorting) and semantics (typically learned by embeddings). Within this paper, we describe the new SortBench benchmark for LLMs that comes with different difficulties and that can be easily scaled in terms of difficulty. We apply this benchmark to seven state-of-the-art LLMs, including current test-time reasoning models. Our results show that while the o3-mini model is very capable at sorting in general, even this can be fooled if strings are defined to mix syntactical and semantical aspects, e.g., by asking to sort numbers written-out as word. Furthermore, all models have problems with the faithfulness to the input of long lists, i.e., they drop items and add new ones. Our results also show that test-time reasoning has a tendency to overthink problems which leads to performance degradation. Finally, models without test-time reasoning like GPT-4o are not much worse than reasoning models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06943v2">Review of Case-Based Reasoning for LLM Agents: Theoretical Foundations, Architectural Components, and Cognitive Integration</a></div>
    <div class="paper-meta">
      📅 2025-04-11
    </div>
    <details class="paper-abstract">
      Agents powered by Large Language Models (LLMs) have recently demonstrated impressive capabilities in various tasks. Still, they face limitations in tasks requiring specific, structured knowledge, flexibility, or accountable decision-making. While agents are capable of perceiving their environments, forming inferences, planning, and executing actions towards goals, they often face issues such as hallucinations and lack of contextual memory across interactions. This paper explores how Case-Based Reasoning (CBR), a strategy that solves new problems by referencing past experiences, can be integrated into LLM agent frameworks. This integration allows LLMs to leverage explicit knowledge, enhancing their effectiveness. We systematically review the theoretical foundations of these enhanced agents, identify critical framework components, and formulate a mathematical model for the CBR processes of case retrieval, adaptation, and learning. We also evaluate CBR-enhanced agents against other methods like Chain-of-Thought reasoning and standard Retrieval-Augmented Generation, analyzing their relative strengths. Moreover, we explore how leveraging CBR's cognitive dimensions (including self-reflection, introspection, and curiosity) via goal-driven autonomy mechanisms can further enhance the LLM agent capabilities. Contributing to the ongoing research on neuro-symbolic hybrid systems, this work posits CBR as a viable technique for enhancing the reasoning skills and cognitive aspects of autonomous LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08260v1">Evaluating the Bias in LLMs for Surveying Opinion and Decision Making in Healthcare</a></div>
    <div class="paper-meta">
      📅 2025-04-11
    </div>
    <details class="paper-abstract">
      Generative agents have been increasingly used to simulate human behaviour in silico, driven by large language models (LLMs). These simulacra serve as sandboxes for studying human behaviour without compromising privacy or safety. However, it remains unclear whether such agents can truly represent real individuals. This work compares survey data from the Understanding America Study (UAS) on healthcare decision-making with simulated responses from generative agents. Using demographic-based prompt engineering, we create digital twins of survey respondents and analyse how well different LLMs reproduce real-world behaviours. Our findings show that some LLMs fail to reflect realistic decision-making, such as predicting universal vaccine acceptance. However, Llama 3 captures variations across race and Income more accurately but also introduces biases not present in the UAS data. This study highlights the potential of generative agents for behavioural research while underscoring the risks of bias from both LLMs and prompting strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08242v1">Jupiter: Fast and Resource-Efficient Collaborative Inference of Generative LLMs on Edge Devices</a></div>
    <div class="paper-meta">
      📅 2025-04-11
      | 💬 Accepted by IEEE International Conference on Computer Communications 2025
    </div>
    <details class="paper-abstract">
      Generative large language models (LLMs) have garnered significant attention due to their exceptional capabilities in various AI tasks. Traditionally deployed in cloud datacenters, LLMs are now increasingly moving towards more accessible edge platforms to protect sensitive user data and ensure privacy preservation. The limited computational resources of individual edge devices, however, can result in excessively prolonged inference latency and overwhelmed memory usage. While existing research has explored collaborative edge computing to break the resource wall of individual devices, these solutions yet suffer from massive communication overhead and under-utilization of edge resources. Furthermore, they focus exclusively on optimizing the prefill phase, neglecting the crucial autoregressive decoding phase for generative LLMs. To address that, we propose Jupiter, a fast, scalable, and resource-efficient collaborative edge AI system for generative LLM inference. Jupiter introduces a flexible pipelined architecture as a principle and differentiates its system design according to the differentiated characteristics of the prefill and decoding phases. For prefill phase, Jupiter submits a novel intra-sequence pipeline parallelism and develops a meticulous parallelism planning strategy to maximize resource efficiency; For decoding, Jupiter devises an effective outline-based pipeline parallel decoding mechanism combined with speculative decoding, which further magnifies inference acceleration. Extensive evaluation based on realistic implementation demonstrates that Jupiter remarkably outperforms state-of-the-art approaches under various edge environment setups, achieving up to 26.1x end-to-end latency reduction while rendering on-par generation quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08211v1">LLM for Comparative Narrative Analysis</a></div>
    <div class="paper-meta">
      📅 2025-04-11
      | 💬 5 pages, 4 figures, Appendix included
    </div>
    <details class="paper-abstract">
      In this paper, we conducted a Multi-Perspective Comparative Narrative Analysis (CNA) on three prominent LLMs: GPT-3.5, PaLM2, and Llama2. We applied identical prompts and evaluated their outputs on specific tasks, ensuring an equitable and unbiased comparison between various LLMs. Our study revealed that the three LLMs generated divergent responses to the same prompt, indicating notable discrepancies in their ability to comprehend and analyze the given task. Human evaluation was used as the gold standard, evaluating four perspectives to analyze differences in LLM performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08207v1">DRAFT-ing Architectural Design Decisions using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-11
    </div>
    <details class="paper-abstract">
      Architectural Knowledge Management (AKM) is crucial for software development but remains challenging due to the lack of standardization and high manual effort. Architecture Decision Records (ADRs) provide a structured approach to capture Architecture Design Decisions (ADDs), but their adoption is limited due to the manual effort involved and insufficient tool support. Our previous work has shown that Large Language Models (LLMs) can assist in generating ADDs. However, simply prompting the LLM does not produce quality ADDs. Moreover, using third-party LLMs raises privacy concerns, while self-hosting them poses resource challenges. To this end, we experimented with different approaches like few-shot, retrieval-augmented generation (RAG) and fine-tuning to enhance LLM's ability to generate ADDs. Our results show that both techniques improve effectiveness. Building on this, we propose Domain Specific Retreival Augumented Few Shot Fine Tuninng, DRAFT, which combines the strengths of all these three approaches for more effective ADD generation. DRAFT operates in two phases: an offline phase that fine-tunes an LLM on generating ADDs augmented with retrieved examples and an online phase that generates ADDs by leveraging retrieved ADRs and the fine-tuned model. We evaluated DRAFT against existing approaches on a dataset of 4,911 ADRs and various LLMs and analyzed them using automated metrics and human evaluations. Results show DRAFT outperforms all other approaches in effectiveness while maintaining efficiency. Our findings indicate that DRAFT can aid architects in drafting ADDs while addressing privacy and resource constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07596v2">Boosting Universal LLM Reward Design through Heuristic Reward Observation Space Evolution</a></div>
    <div class="paper-meta">
      📅 2025-04-11
      | 💬 7 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are emerging as promising tools for automated reinforcement learning (RL) reward design, owing to their robust capabilities in commonsense reasoning and code generation. By engaging in dialogues with RL agents, LLMs construct a Reward Observation Space (ROS) by selecting relevant environment states and defining their internal operations. However, existing frameworks have not effectively leveraged historical exploration data or manual task descriptions to iteratively evolve this space. In this paper, we propose a novel heuristic framework that enhances LLM-driven reward design by evolving the ROS through a table-based exploration caching mechanism and a text-code reconciliation strategy. Our framework introduces a state execution table, which tracks the historical usage and success rates of environment states, overcoming the Markovian constraint typically found in LLM dialogues and facilitating more effective exploration. Furthermore, we reconcile user-provided task descriptions with expert-defined success criteria using structured prompts, ensuring alignment in reward design objectives. Comprehensive evaluations on benchmark RL tasks demonstrate the effectiveness and stability of the proposed framework. Code and video demos are available at jingjjjjjie.github.io/LLM2Reward.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08192v1">SAEs $\textit{Can}$ Improve Unlearning: Dynamic Sparse Autoencoder Guardrails for Precision Unlearning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-11
    </div>
    <details class="paper-abstract">
      Machine unlearning is a promising approach to improve LLM safety by removing unwanted knowledge from the model. However, prevailing gradient-based unlearning methods suffer from issues such as high computational costs, hyperparameter instability, poor sequential unlearning capability, vulnerability to relearning attacks, low data efficiency, and lack of interpretability. While Sparse Autoencoders are well-suited to improve these aspects by enabling targeted activation-based unlearning, prior approaches underperform gradient-based methods. This work demonstrates that, contrary to these earlier findings, SAEs can significantly improve unlearning when employed dynamically. We introduce $\textbf{Dynamic DAE Guardrails}$ (DSG), a novel method for precision unlearning that leverages principled feature selection and a dynamic classifier. Our experiments show DSG substantially outperforms leading unlearning methods, achieving superior forget-utility trade-offs. DSG addresses key drawbacks of gradient-based approaches for unlearning -- offering enhanced computational efficiency and stability, robust performance in sequential unlearning, stronger resistance to relearning attacks, better data efficiency including zero-shot settings, and more interpretable unlearning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.16557v2">Patched RTC: evaluating LLMs for diverse software development tasks</a></div>
    <div class="paper-meta">
      📅 2025-04-11
    </div>
    <details class="paper-abstract">
      This paper introduces Patched Round-Trip Correctness (Patched RTC), a novel evaluation technique for Large Language Models (LLMs) applied to diverse software development tasks, particularly focusing on "outer loop" activities such as bug fixing, code review, and documentation updates. Patched RTC extends the original Round-Trip Correctness method to work with any LLM and downstream task, offering a self-evaluating framework that measures consistency and robustness of model responses without human intervention. The study demonstrates a correlation between Patched RTC scores and task-specific accuracy metrics, presenting it as an alternative to the LLM-as-Judge paradigm for open-domain task evaluation. We implement Patched RTC in an open-source framework called patchwork, allowing for transparent evaluation during inference across various patchflows. Experiments comparing GPT-3.5 and GPT-4 models across different software development tasks reveal that Patched RTC effectively distinguishes model performance and task difficulty. The paper also explores the impact of consistency prompts on improving model accuracy, suggesting that Patched RTC can guide prompt refinement and model selection for complex software development workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08999v1">MCP Bridge: A Lightweight, LLM-Agnostic RESTful Proxy for Model Context Protocol Servers</a></div>
    <div class="paper-meta">
      📅 2025-04-11
      | 💬 13 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly augmented with external tools through standardized interfaces like the Model Context Protocol (MCP). However, current MCP implementations face critical limitations: they typically require local process execution through STDIO transports, making them impractical for resource-constrained environments like mobile devices, web browsers, and edge computing. We present MCP Bridge, a lightweight RESTful proxy that connects to multiple MCP servers and exposes their capabilities through a unified API. Unlike existing solutions, MCP Bridge is fully LLM-agnostic, supporting any backend regardless of vendor. The system implements a risk-based execution model with three security levels standard execution, confirmation workflow, and Docker isolation while maintaining backward compatibility with standard MCP clients. Complementing this server-side infrastructure is a Python based MCP Gemini Agent that facilitates natural language interaction with MCP tools. The evaluation demonstrates that MCP Bridge successfully addresses the constraints of direct MCP connections while providing enhanced security controls and cross-platform compatibility, enabling sophisticated LLM-powered applications in previously inaccessible environments
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05522v2">User Feedback Alignment for LLM-powered Exploration in Large-scale Recommendation Systems</a></div>
    <div class="paper-meta">
      📅 2025-04-11
    </div>
    <details class="paper-abstract">
      Exploration, the act of broadening user experiences beyond their established preferences, is challenging in large-scale recommendation systems due to feedback loops and limited signals on user exploration patterns. Large Language Models (LLMs) offer potential by leveraging their world knowledge to recommend novel content outside these loops. A key challenge is aligning LLMs with user preferences while preserving their knowledge and reasoning. While using LLMs to plan for the next novel user interest, this paper introduces a novel approach combining hierarchical planning with LLM inference-time scaling to improve recommendation relevancy without compromising novelty. We decouple novelty and user-alignment, training separate LLMs for each objective. We then scale up the novelty-focused LLM's inference and select the best-of-n predictions using the user-aligned LLM. Live experiments demonstrate efficacy, showing significant gains in both user satisfaction (measured by watch activity and active user counts) and exploration diversity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08985v1">Learning from Elders: Making an LLM-powered Chatbot for Retirement Communities more Accessible through User-centered Design</a></div>
    <div class="paper-meta">
      📅 2025-04-11
      | 💬 Accepted as Research talk for Considering Cultural and Linguistic Diversity in AI Applications workshop at CALD-AI@ASIS&T 2025
    </div>
    <details class="paper-abstract">
      Low technology and eHealth literacy among older adults in retirement communities hinder engagement with digital tools. To address this, we designed an LLM-powered chatbot prototype using a human-centered approach for a local retirement community. Through interviews and persona development, we prioritized accessibility and dual functionality: simplifying internal information retrieval and improving technology and eHealth literacy. A pilot trial with residents demonstrated high satisfaction and ease of use, but also identified areas for further improvement. Based on the feedback, we refined the chatbot using GPT-3.5 Turbo and Streamlit. The chatbot employs tailored prompt engineering to deliver concise responses. Accessible features like adjustable font size, interface theme and personalized follow-up responses were implemented. Future steps include enabling voice-to-text function and longitudinal intervention studies. Together, our results highlight the potential of LLM-driven chatbots to empower older adults through accessible, personalized interactions, bridging literacy gaps in retirement communities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07415v1">Leveraging LLMs for Multimodal Retrieval-Augmented Radiology Report Generation via Key Phrase Extraction</a></div>
    <div class="paper-meta">
      📅 2025-04-10
    </div>
    <details class="paper-abstract">
      Automated radiology report generation (RRG) holds potential to reduce radiologists' workload, especially as recent advancements in large language models (LLMs) enable the development of multimodal models for chest X-ray (CXR) report generation. However, multimodal LLMs (MLLMs) are resource-intensive, requiring vast datasets and substantial computational cost for training. To address these challenges, we propose a retrieval-augmented generation approach that leverages multimodal retrieval and LLMs to generate radiology reports while mitigating hallucinations and reducing computational demands. Our method uses LLMs to extract key phrases from radiology reports, effectively focusing on essential diagnostic information. Through exploring effective training strategies, including image encoder structure search, adding noise to text embeddings, and additional training objectives, we combine complementary pre-trained image encoders and adopt contrastive learning between text and semantic image embeddings. We evaluate our approach on MIMIC-CXR dataset, achieving state-of-the-art results on CheXbert metrics and competitive RadGraph F1 metric alongside MLLMs, without requiring LLM fine-tuning. Our method demonstrates robust generalization for multi-view RRG, making it suitable for comprehensive clinical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09647v5">Leveraging LLMS for Top-Down Sector Allocation In Automated Trading</a></div>
    <div class="paper-meta">
      📅 2025-04-10
    </div>
    <details class="paper-abstract">
      This paper introduces a methodology leveraging Large Language Models (LLMs) for sector-level portfolio allocation through systematic analysis of macroeconomic conditions and market sentiment. Our framework emphasizes top-down sector allocation by processing multiple data streams simultaneously, including policy documents, economic indicators, and sentiment patterns. Empirical results demonstrate superior risk-adjusted returns compared to traditional cross momentum strategies, achieving a Sharpe ratio of 2.51 and portfolio return of 8.79% versus -0.61 and -1.39% respectively. These results suggest that LLM-based systematic macro analysis presents a viable approach for enhancing automated portfolio allocation decisions at the sector level.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.09722v4">Optimized Multi-Token Joint Decoding with Auxiliary Model for LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-04-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable success across diverse tasks, yet their inference processes are hindered by substantial time and energy demands due to single-token generation at each decoding step. While previous methods such as speculative decoding mitigate these inefficiencies by producing multiple tokens per step, each token is still generated by its single-token distribution, thereby enhancing speed without improving effectiveness. In contrast, our work simultaneously enhances inference speed and improves the output effectiveness. We consider multi-token joint decoding (MTJD), which generates multiple tokens from their joint distribution at each iteration, theoretically reducing perplexity and enhancing task performance. However, MTJD suffers from the high cost of sampling from the joint distribution of multiple tokens. Inspired by speculative decoding, we introduce multi-token assisted decoding (MTAD), a novel framework designed to accelerate MTJD. MTAD leverages a smaller auxiliary model to approximate the joint distribution of a larger model, incorporating a verification mechanism that not only ensures the accuracy of this approximation, but also improves the decoding efficiency over conventional speculative decoding. Theoretically, we demonstrate that MTAD closely approximates exact MTJD with bounded error. Empirical evaluations using Llama-2 and OPT models ranging from 13B to 70B parameters across various tasks reveal that MTAD reduces perplexity by 21.2% and improves downstream performance compared to standard single-token sampling. Furthermore, MTAD achieves a 1.42x speed-up and consumes 1.54x less energy than conventional speculative decoding methods. These results highlight MTAD's ability to make multi-token joint decoding both effective and efficient, promoting more sustainable and high-performance deployment of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05500v2">Prism: Dynamic and Flexible Benchmarking of LLMs Code Generation with Monte Carlo Tree Search</a></div>
    <div class="paper-meta">
      📅 2025-04-10
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) has outpaced traditional evaluation methods. Static benchmarks fail to capture the depth and breadth of LLM capabilities and eventually become obsolete, while most dynamic approaches either rely too heavily on LLM-based evaluation or remain constrained by predefined test sets. We introduce Prism, a flexible, dynamic benchmarking framework designed for comprehensive LLM assessment. Prism builds on three key components: (1) a tree-based state representation that models evaluation as a Markov Decision Process, (2) a Monte Carlo Tree Search algorithm adapted to uncover challenging evaluation scenarios, and (3) a multi-agent evaluation pipeline that enables simultaneous assessment of diverse capabilities. To ensure robust evaluation, Prism integrates structural measurements of tree exploration patterns with performance metrics across difficulty levels, providing detailed diagnostics of error patterns, test coverage, and solution approaches. Through extensive experiments on five state-of-the-art LLMs, we analyze how model architecture and scale influence code generation performance across varying task difficulties. Our results demonstrate Prism's effectiveness as a dynamic benchmark that evolves with model advancements while offering deeper insights into their limitations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07360v1">Enhancing Time Series Forecasting via Multi-Level Text Alignment with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-10
    </div>
    <details class="paper-abstract">
      The adaptation of large language models (LLMs) to time series forecasting poses unique challenges, as time series data is continuous in nature, while LLMs operate on discrete tokens. Despite the success of LLMs in natural language processing (NLP) and other structured domains, aligning time series data with language-based representations while maintaining both predictive accuracy and interpretability remains a significant hurdle. Existing methods have attempted to reprogram time series data into text-based forms, but these often fall short in delivering meaningful, interpretable results. In this paper, we propose a multi-level text alignment framework for time series forecasting using LLMs that not only improves prediction accuracy but also enhances the interpretability of time series representations. Our method decomposes time series into trend, seasonal, and residual components, which are then reprogrammed into component-specific text representations. We introduce a multi-level alignment mechanism, where component-specific embeddings are aligned with pre-trained word tokens, enabling more interpretable forecasts. Experiments on multiple datasets demonstrate that our method outperforms state-of-the-art models in accuracy while providing good interpretability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.22318v2">Online Detecting LLM-Generated Texts via Sequential Hypothesis Testing by Betting</a></div>
    <div class="paper-meta">
      📅 2025-04-10
    </div>
    <details class="paper-abstract">
      Developing algorithms to differentiate between machine-generated texts and human-written texts has garnered substantial attention in recent years. Existing methods in this direction typically concern an offline setting where a dataset containing a mix of real and machine-generated texts is given upfront, and the task is to determine whether each sample in the dataset is from a large language model (LLM) or a human. However, in many practical scenarios, sources such as news websites, social media accounts, or on other forums publish content in a streaming fashion. Therefore, in this online scenario, how to quickly and accurately determine whether the source is an LLM with strong statistical guarantees is crucial for these media or platforms to function effectively and prevent the spread of misinformation and other potential misuse of LLMs. To tackle the problem of online detection, we develop an algorithm based on the techniques of sequential hypothesis testing by betting that not only builds upon and complements existing offline detection techniques but also enjoys statistical guarantees, which include a controlled false positive rate and the expected time to correctly identify a source as an LLM. Experiments were conducted to demonstrate the effectiveness of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07347v1">Throughput-Optimal Scheduling Algorithms for LLM Inference and AI Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-10
    </div>
    <details class="paper-abstract">
      As demand for Large Language Models (LLMs) and AI agents rapidly grows, optimizing systems for efficient LLM inference becomes critical. While significant efforts have targeted system-level engineering, little is explored through a mathematical modeling and queuing perspective. In this paper, we aim to develop the queuing fundamentals for LLM inference, bridging the gap between queuing and LLM system communities. In particular, we study the throughput aspect in LLM inference systems. We prove that a large class of 'work-conserving' scheduling algorithms can achieve maximum throughput for both individual requests and AI agent workloads, highlighting 'work-conserving' as a key design principle in practice. Evaluations of real-world systems show that Orca and Sarathi-serve are throughput-optimal, reassuring practitioners, while FastTransformer and vanilla vLLM are not maximally stable and should be used with caution. Our results highlight the substantial benefits queuing community can offer in improving LLM inference systems and call for more interdisciplinary developments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01995v2">Brains vs. Bytes: Evaluating LLM Proficiency in Olympiad Mathematics</a></div>
    <div class="paper-meta">
      📅 2025-04-10
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have shown impressive progress in mathematical reasoning tasks. However, current evaluation benchmarks predominantly focus on the accuracy of final answers, often overlooking the crucial logical rigor for mathematical problem solving. The claim that state-of-the-art LLMs can solve Math Olympiad-level problems requires closer examination. To explore this, we conducted both qualitative and quantitative human evaluations of proofs generated by LLMs, and developed a schema for automatically assessing their reasoning capabilities. Our study reveals that current LLMs fall significantly short of solving challenging Olympiad-level problems and frequently fail to distinguish correct mathematical reasoning from clearly flawed solutions. Our analyses demonstrate that the occasional correct final answers provided by LLMs often result from pattern recognition or heuristic shortcuts rather than genuine mathematical reasoning. These findings underscore the substantial gap between LLM performance and human expertise in advanced mathematical reasoning and highlight the importance of developing benchmarks that prioritize the soundness of the reasoning used to arrive at an answer rather than the mere correctness of the final answers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08120v1">DeepSeek vs. o3-mini: How Well can Reasoning LLMs Evaluate MT and Summarization?</a></div>
    <div class="paper-meta">
      📅 2025-04-10
    </div>
    <details class="paper-abstract">
      Reasoning-enabled large language models (LLMs) have recently demonstrated impressive performance in complex logical and mathematical tasks, yet their effectiveness in evaluating natural language generation remains unexplored. This study systematically compares reasoning-based LLMs (DeepSeek-R1 and OpenAI o3) with their non-reasoning counterparts across machine translation (MT) and text summarization (TS) evaluation tasks. We evaluate eight models across three architectural categories, including state-of-the-art reasoning models, their distilled variants (ranging from 8B to 70B parameters), and equivalent conventional, non-reasoning LLMs. Our experiments on WMT23 and SummEval benchmarks reveal that the benefits of reasoning capabilities are highly model and task-dependent: while OpenAI o3-mini models show consistent performance improvements with increased reasoning intensity, DeepSeek-R1 underperforms compared to its non-reasoning variant, with exception to certain aspects of TS evaluation. Correlation analysis demonstrates that increased reasoning token usage positively correlates with evaluation quality in o3-mini models. Furthermore, our results show that distillation of reasoning capabilities maintains reasonable performance in medium-sized models (32B) but degrades substantially in smaller variants (8B). This work provides the first comprehensive assessment of reasoning LLMs for NLG evaluation and offers insights into their practical use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08113v1">Test Amplification for REST APIs via Single and Multi-Agent LLM Systems</a></div>
    <div class="paper-meta">
      📅 2025-04-10
    </div>
    <details class="paper-abstract">
      REST APIs (Representational State Transfer Application Programming Interfaces) are essential to modern cloud-native applications. Strong and automated test cases are crucial to expose lurking bugs in the API. However, creating automated tests for REST APIs is difficult, and it requires test cases that explore the protocol's boundary conditions. In this paper, we investigate how single-agent and multi-agent LLM (Large Language Model) systems can amplify a REST API test suite. Our evaluation demonstrates increased API coverage, identification of numerous bugs in the API under test, and insights into the computational cost and energy consumption of both approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08104v1">Geneshift: Impact of different scenario shift on Jailbreaking LLM</a></div>
    <div class="paper-meta">
      📅 2025-04-10
    </div>
    <details class="paper-abstract">
      Jailbreak attacks, which aim to cause LLMs to perform unrestricted behaviors, have become a critical and challenging direction in AI safety. Despite achieving the promising attack success rate using dictionary-based evaluation, existing jailbreak attack methods fail to output detailed contents to satisfy the harmful request, leading to poor performance on GPT-based evaluation. To this end, we propose a black-box jailbreak attack termed GeneShift, by using a genetic algorithm to optimize the scenario shifts. Firstly, we observe that the malicious queries perform optimally under different scenario shifts. Based on it, we develop a genetic algorithm to evolve and select the hybrid of scenario shifts. It guides our method to elicit detailed and actionable harmful responses while keeping the seemingly benign facade, improving stealthiness. Extensive experiments demonstrate the superiority of GeneShift. Notably, GeneShift increases the jailbreak success rate from 0% to 60% when direct prompting alone would fail.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17604v3">OmniScience: A Domain-Specialized LLM for Scientific Reasoning and Discovery</a></div>
    <div class="paper-meta">
      📅 2025-04-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable potential in advancing scientific knowledge and addressing complex challenges. In this work, we introduce OmniScience, a specialized large reasoning model for general science, developed through three key components: (1) domain adaptive pretraining on a carefully curated corpus of scientific literature, (2) instruction tuning on a specialized dataset to guide the model in following domain-specific tasks, and (3) reasoning-based knowledge distillation through fine-tuning to significantly enhance its ability to generate contextually relevant and logically sound responses. We demonstrate the versatility of OmniScience by developing a battery agent that efficiently ranks molecules as potential electrolyte solvents or additives. Comprehensive evaluations reveal that OmniScience is competitive with state-of-the-art large reasoning models on the GPQA Diamond and domain-specific battery benchmarks, while outperforming all public reasoning and non-reasoning models with similar parameter counts. We further demonstrate via ablation experiments that domain adaptive pretraining and reasoning-based knowledge distillation are critical to attain our performance levels, across benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18791v2">Can LLMs Help Uncover Insights about LLMs? A Large-Scale, Evolving Literature Analysis of Frontier LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-10
      | 💬 22 pages, 9 figures
    </div>
    <details class="paper-abstract">
      The surge of LLM studies makes synthesizing their findings challenging. Analysis of experimental results from literature can uncover important trends across studies, but the time-consuming nature of manual data extraction limits its use. Our study presents a semi-automated approach for literature analysis that accelerates data extraction using LLMs. It automatically identifies relevant arXiv papers, extracts experimental results and related attributes, and organizes them into a structured dataset, LLMEvalDB. We then conduct an automated literature analysis of frontier LLMs, reducing the effort of paper surveying and data extraction by more than 93% compared to manual approaches. We validate LLMEvalDB by showing that it reproduces key findings from a recent manual analysis of Chain-of-Thought (CoT) reasoning and also uncovers new insights that go beyond it, showing, for example, that in-context examples benefit coding and multimodal tasks but offer limited gains in math reasoning tasks compared to zero-shot CoT. Our automatically updatable dataset enables continuous tracking of target models by extracting evaluation studies as new data becomes available. Through LLMEvalDB and empirical analysis, we provide insights into LLMs while facilitating ongoing literature analyses of their behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06845v3">7B Fully Open Source Moxin-LLM -- From Pretraining to GRPO-based Reinforcement Learning Enhancement</a></div>
    <div class="paper-meta">
      📅 2025-04-10
    </div>
    <details class="paper-abstract">
      Recently, Large Language Models (LLMs) have undergone a significant transformation, marked by a rapid rise in both their popularity and capabilities. Leading this evolution are proprietary LLMs like GPT-4 and GPT-o1, which have captured widespread attention in the AI community due to their remarkable performance and versatility. Simultaneously, open-source LLMs, such as LLaMA, have made great contributions to the ever-increasing popularity of LLMs due to the ease to customize and deploy the models across diverse applications. Although open-source LLMs present unprecedented opportunities for innovation and research, the commercialization of LLMs has raised concerns about transparency, reproducibility, and safety. Many open-source LLMs fail to meet fundamental transparency requirements by withholding essential components like training code and data, which may hinder further innovations on LLMs. To mitigate this issue, we introduce Moxin 7B, a fully open-source LLM developed, adhering to principles of open science, open source, open data, and open access. We release the pre-training code and configurations, training and fine-tuning datasets, and intermediate and final checkpoints, aiming to make continuous commitments to fully open-source LLMs. After pre-training and obtaining the base model, we finetune the Moxin Base model with SOTA post-training framework and instruction data to obtain Moxin Instruct model. To improve the reasoning capability, we further finetune our Instruct model with chain-of-thought data distilled from DeepSeek R1, and then use Group Relative Policy Optimization (GRPO), an efficient and effective reinforcement learning algorithm following DeepSeek R1, to finetune our model, leading to the Moxin Reasoning model. Experiments show that our models achieve superior performance in various evaluations such as zero-shot evaluation, few-shot evaluation, and CoT evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08040v1">Can Reasoning LLMs Enhance Clinical Document Classification?</a></div>
    <div class="paper-meta">
      📅 2025-04-10
      | 💬 28 pages, 13 tables, 12 figures
    </div>
    <details class="paper-abstract">
      Clinical document classification is essential for converting unstructured medical texts into standardised ICD-10 diagnoses, yet it faces challenges due to complex medical language, privacy constraints, and limited annotated datasets. Large Language Models (LLMs) offer promising improvements in accuracy and efficiency for this task. This study evaluates the performance and consistency of eight LLMs; four reasoning (Qwen QWQ, Deepseek Reasoner, GPT o3 Mini, Gemini 2.0 Flash Thinking) and four non-reasoning (Llama 3.3, GPT 4o Mini, Gemini 2.0 Flash, Deepseek Chat); in classifying clinical discharge summaries using the MIMIC-IV dataset. Using cTAKES to structure clinical narratives, models were assessed across three experimental runs, with majority voting determining final predictions. Results showed that reasoning models outperformed non-reasoning models in accuracy (71% vs 68%) and F1 score (67% vs 60%), with Gemini 2.0 Flash Thinking achieving the highest accuracy (75%) and F1 score (76%). However, non-reasoning models demonstrated greater stability (91% vs 84% consistency). Performance varied across ICD-10 codes, with reasoning models excelling in complex cases but struggling with abstract categories. Findings indicate a trade-off between accuracy and consistency, suggesting that a hybrid approach could optimise clinical coding. Future research should explore multi-label classification, domain-specific fine-tuning, and ensemble methods to enhance model reliability in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07907v1">Porting an LLM based Application from ChatGPT to an On-Premise Environment</a></div>
    <div class="paper-meta">
      📅 2025-04-10
      | 💬 Actual article is a part of the proceedings of the International Conference on Software Reuse (ICSR) 2025
    </div>
    <details class="paper-abstract">
      Given the data-intensive nature of Machine Learning (ML) systems in general, and Large Language Models (LLM) in particular, using them in cloud based environments can become a challenge due to legislation related to privacy and security of data. Taking such aspects into consideration implies porting the LLMs to an on-premise environment, where privacy and security can be controlled. In this paper, we study this porting process of a real-life application using ChatGPT, which runs in a public cloud, to an on-premise environment. The application being ported is AIPA, a system that leverages Large Language Models (LLMs) and sophisticated data analytics to enhance the assessment of procurement call bids. The main considerations in the porting process include transparency of open source models and cost of hardware, which are central design choices of the on-premise environment. In addition to presenting the porting process, we evaluate downsides and benefits associated with porting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07887v1">Benchmarking Adversarial Robustness to Bias Elicitation in Large Language Models: Scalable Automated Assessment with LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-04-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized artificial intelligence, driving advancements in machine translation, summarization, and conversational agents. However, their increasing integration into critical societal domains has raised concerns about embedded biases, which can perpetuate stereotypes and compromise fairness. These biases stem from various sources, including historical inequalities in training data, linguistic imbalances, and adversarial manipulation. Despite mitigation efforts, recent studies indicate that LLMs remain vulnerable to adversarial attacks designed to elicit biased responses. This work proposes a scalable benchmarking framework to evaluate LLM robustness against adversarial bias elicitation. Our methodology involves (i) systematically probing models with a multi-task approach targeting biases across various sociocultural dimensions, (ii) quantifying robustness through safety scores using an LLM-as-a-Judge approach for automated assessment of model responses, and (iii) employing jailbreak techniques to investigate vulnerabilities in safety mechanisms. Our analysis examines prevalent biases in both small and large state-of-the-art models and their impact on model safety. Additionally, we assess the safety of domain-specific models fine-tuned for critical fields, such as medicine. Finally, we release a curated dataset of bias-related prompts, CLEAR-Bias, to facilitate systematic vulnerability benchmarking. Our findings reveal critical trade-offs between model size and safety, aiding the development of fairer and more robust future language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07863v1">Robust Hallucination Detection in LLMs via Adaptive Token Selection</a></div>
    <div class="paper-meta">
      📅 2025-04-10
    </div>
    <details class="paper-abstract">
      Hallucinations in large language models (LLMs) pose significant safety concerns that impede their broader deployment. Recent research in hallucination detection has demonstrated that LLMs' internal representations contain truthfulness hints, which can be harnessed for detector training. However, the performance of these detectors is heavily dependent on the internal representations of predetermined tokens, fluctuating considerably when working on free-form generations with varying lengths and sparse distributions of hallucinated entities. To address this, we propose HaMI, a novel approach that enables robust detection of hallucinations through adaptive selection and learning of critical tokens that are most indicative of hallucinations. We achieve this robustness by an innovative formulation of the Hallucination detection task as Multiple Instance (HaMI) learning over token-level representations within a sequence, thereby facilitating a joint optimisation of token selection and hallucination detection on generation sequences of diverse forms. Comprehensive experimental results on four hallucination benchmarks show that HaMI significantly outperforms existing state-of-the-art approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07840v1">Understanding Learner-LLM Chatbot Interactions and the Impact of Prompting Guidelines</a></div>
    <div class="paper-meta">
      📅 2025-04-10
      | 💬 Accepted for AIED 2025, the 26th International Conference on Artificial Intelligence in Education, July 22 - 26, 2025, Palermo, Italy
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have transformed human-computer interaction by enabling natural language-based communication with AI-powered chatbots. These models are designed to be intuitive and user-friendly, allowing users to articulate requests with minimal effort. However, despite their accessibility, studies reveal that users often struggle with effective prompting, resulting in inefficient responses. Existing research has highlighted both the limitations of LLMs in interpreting vague or poorly structured prompts and the difficulties users face in crafting precise queries. This study investigates learner-AI interactions through an educational experiment in which participants receive structured guidance on effective prompting. We introduce and compare three types of prompting guidelines: a task-specific framework developed through a structured methodology and two baseline approaches. To assess user behavior and prompting efficacy, we analyze a dataset of 642 interactions from 107 users. Using Von NeuMidas, an extended pragmatic annotation schema for LLM interaction analysis, we categorize common prompting errors and identify recurring behavioral patterns. We then evaluate the impact of different guidelines by examining changes in user behavior, adherence to prompting strategies, and the overall quality of AI-generated responses. Our findings provide a deeper understanding of how users engage with LLMs and the role of structured prompting guidance in enhancing AI-assisted communication. By comparing different instructional frameworks, we offer insights into more effective approaches for improving user competency in AI interactions, with implications for AI literacy, chatbot usability, and the design of more responsive AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12349v3">SPIN-Bench: How Well Do LLMs Plan Strategically and Reason Socially?</a></div>
    <div class="paper-meta">
      📅 2025-04-10
      | 💬 42 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Reasoning and strategic behavior in social interactions is a hallmark of intelligence. This form of reasoning is significantly more sophisticated than isolated planning or reasoning tasks in static settings (e.g., math problem solving). In this paper, we present Strategic Planning, Interaction, and Negotiation (SPIN-Bench), a new multi-domain evaluation designed to measure the intelligence of strategic planning and social reasoning. While many existing benchmarks focus on narrow planning or single-agent reasoning, SPIN-Bench combines classical PDDL tasks, competitive board games, cooperative card games, and multi-agent negotiation scenarios in one unified framework. The framework includes both a benchmark as well as an arena to simulate and evaluate the variety of social settings to test reasoning and strategic behavior of AI agents. We formulate the benchmark SPIN-Bench by systematically varying action spaces, state complexity, and the number of interacting agents to simulate a variety of social settings where success depends on not only methodical and step-wise decision making, but also conceptual inference of other (adversarial or cooperative) participants. Our experiments reveal that while contemporary LLMs handle basic fact retrieval and short-range planning reasonably well, they encounter significant performance bottlenecks in tasks requiring deep multi-hop reasoning over large state spaces and socially adept coordination under uncertainty. We envision SPIN-Bench as a catalyst for future research on robust multi-agent planning, social reasoning, and human--AI teaming. Project Website: https://spinbench.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07801v1">FairEval: Evaluating Fairness in LLM-Based Recommendations with Personality Awareness</a></div>
    <div class="paper-meta">
      📅 2025-04-10
      | 💬 11 pages, 5 figures, under review at a top-tier ACM conference in recommender systems
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have enabled their application to recommender systems (RecLLMs), yet concerns remain regarding fairness across demographic and psychological user dimensions. We introduce FairEval, a novel evaluation framework to systematically assess fairness in LLM-based recommendations. FairEval integrates personality traits with eight sensitive demographic attributes,including gender, race, and age, enabling a comprehensive assessment of user-level bias. We evaluate models, including ChatGPT 4o and Gemini 1.5 Flash, on music and movie recommendations. FairEval's fairness metric, PAFS, achieves scores up to 0.9969 for ChatGPT 4o and 0.9997 for Gemini 1.5 Flash, with disparities reaching 34.79 percent. These results highlight the importance of robustness in prompt sensitivity and support more inclusive recommendation systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07745v1">SF2T: Self-supervised Fragment Finetuning of Video-LLMs for Fine-Grained Understanding</a></div>
    <div class="paper-meta">
      📅 2025-04-10
      | 💬 Accepted to CVPR2025
    </div>
    <details class="paper-abstract">
      Video-based Large Language Models (Video-LLMs) have witnessed substantial advancements in recent years, propelled by the advancement in multi-modal LLMs. Although these models have demonstrated proficiency in providing the overall description of videos, they struggle with fine-grained understanding, particularly in aspects such as visual dynamics and video details inquiries. To tackle these shortcomings, we find that fine-tuning Video-LLMs on self-supervised fragment tasks, greatly improve their fine-grained video understanding abilities. Hence we propose two key contributions:(1) Self-Supervised Fragment Fine-Tuning (SF$^2$T), a novel effortless fine-tuning method, employs the rich inherent characteristics of videos for training, while unlocking more fine-grained understanding ability of Video-LLMs. Moreover, it relieves researchers from labor-intensive annotations and smartly circumvents the limitations of natural language, which often fails to capture the complex spatiotemporal variations in videos; (2) A novel benchmark dataset, namely FineVidBench, for rigorously assessing Video-LLMs' performance at both the scene and fragment levels, offering a comprehensive evaluation of their capabilities. We assessed multiple models and validated the effectiveness of SF$^2$T on them. Experimental results reveal that our approach improves their ability to capture and interpret spatiotemporal details.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07733v1">DeepGreen: Effective LLM-Driven Green-washing Monitoring System Designed for Empirical Testing -- Evidence from China</a></div>
    <div class="paper-meta">
      📅 2025-04-10
    </div>
    <details class="paper-abstract">
      This paper proposes DeepGreen, an Large Language Model Driven (LLM-Driven) system for detecting corporate green-washing behaviour. Utilizing dual-layer LLM analysis, DeepGreen preliminarily identifies potential green keywords in financial statements and then assesses their implementation degree via iterative semantic analysis of LLM. A core variable GreenImplement is derived from the ratio from the two layers' output. We extract 204 financial statements of 68 companies from A-share market over three years, comprising 89,893 words, and analyse them through DeepGreen. Our analysis, supported by violin plots and K-means clustering, reveals insights and validates the variable against the Huazheng ESG rating. It offers a novel perspective for regulatory agencies and investors, serving as a proactive monitoring tool that complements traditional methods.Empirical tests show that green implementation can significantly boost the asset return rate of companies, but there is heterogeneity in scale. Small and medium-sized companies have limited contribution to asset return via green implementation, so there is a stronger motivation for green-washing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07680v1">Synthetic Fluency: Hallucinations, Confabulations, and the Creation of Irish Words in LLM-Generated Translations</a></div>
    <div class="paper-meta">
      📅 2025-04-10
    </div>
    <details class="paper-abstract">
      This study examines hallucinations in Large Language Model (LLM) translations into Irish, specifically focusing on instances where the models generate novel, non-existent words. We classify these hallucinations within verb and noun categories, identifying six distinct patterns among the latter. Additionally, we analyse whether these hallucinations adhere to Irish morphological rules and what linguistic tendencies they exhibit. Our findings show that while both GPT-4.o and GPT-4.o Mini produce similar types of hallucinations, the Mini model generates them at a significantly higher frequency. Beyond classification, the discussion raises speculative questions about the implications of these hallucinations for the Irish language. Rather than seeking definitive answers, we offer food for thought regarding the increasing use of LLMs and their potential role in shaping Irish vocabulary and linguistic evolution. We aim to prompt discussion on how such technologies might influence language over time, particularly in the context of low-resource, morphologically rich languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07596v1">Boosting Universal LLM Reward Design through the Heuristic Reward Observation Space Evolution</a></div>
    <div class="paper-meta">
      📅 2025-04-10
      | 💬 7 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are emerging as promising tools for automated reinforcement learning (RL) reward design, owing to their robust capabilities in commonsense reasoning and code generation. By engaging in dialogues with RL agents, LLMs construct a Reward Observation Space (ROS) by selecting relevant environment states and defining their internal operations. However, existing frameworks have not effectively leveraged historical exploration data or manual task descriptions to iteratively evolve this space. In this paper, we propose a novel heuristic framework that enhances LLM-driven reward design by evolving the ROS through a table-based exploration caching mechanism and a text-code reconciliation strategy. Our framework introduces a state execution table, which tracks the historical usage and success rates of environment states, overcoming the Markovian constraint typically found in LLM dialogues and facilitating more effective exploration. Furthermore, we reconcile user-provided task descriptions with expert-defined success criteria using structured prompts, ensuring alignment in reward design objectives. Comprehensive evaluations on benchmark RL tasks demonstrate the effectiveness and stability of the proposed framework. Code and video demos are available at jingjjjjjie.github.io/LLM2Reward.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07583v1">Do LLMs Understand Your Translations? Evaluating Paragraph-level MT with Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-04-10
    </div>
    <details class="paper-abstract">
      Despite the steady progress in machine translation evaluation, existing automatic metrics struggle to capture how well meaning is preserved beyond sentence boundaries. We posit that reliance on a single intrinsic quality score, trained to mimic human judgments, might be insufficient for evaluating translations of long, complex passages, and a more ``pragmatic'' approach that assesses how accurately key information is conveyed by a translation in context is needed. We introduce TREQA (Translation Evaluation via Question-Answering), a framework that extrinsically evaluates translation quality by assessing how accurately candidate translations answer reading comprehension questions that target key information in the original source or reference texts. In challenging domains that require long-range understanding, such as literary texts, we show that TREQA is competitive with and, in some cases, outperforms state-of-the-art neural and LLM-based metrics in ranking alternative paragraph-level translations, despite never being explicitly optimized to correlate with human judgments. Furthermore, the generated questions and answers offer interpretability: empirical analysis shows that they effectively target translation errors identified by experts in evaluated datasets. Our code is available at https://github.com/deep-spin/treqa
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12586v3">How to Make LLMs Forget: On Reversing In-Context Knowledge Edits</a></div>
    <div class="paper-meta">
      📅 2025-04-10
      | 💬 Accepted at NAACL Main 2025
    </div>
    <details class="paper-abstract">
      In-context knowledge editing (IKE) enables efficient modification of large language model (LLM) outputs without parameter changes and at zero-cost. However, it can be misused to manipulate responses opaquely, e.g., insert misinformation or offensive content. Such malicious interventions could be incorporated into high-level wrapped APIs where the final input prompt is not shown to end-users. To address this issue, we investigate the detection and reversal of IKE-edits. First, we demonstrate that IKE-edits can be detected with high accuracy (F1 > 80\%) using only the top-10 output probabilities of the next token, even in a black-box setting, e.g. proprietary LLMs with limited output information. Further, we introduce the novel task of reversing IKE-edits using specially tuned reversal tokens. We explore using both continuous and discrete reversal tokens, achieving over 80\% accuracy in recovering original, unedited outputs across multiple LLMs. Our continuous reversal tokens prove particularly effective, with minimal impact on unedited prompts. Through analysis of output distributions, attention patterns, and token rankings, we provide insights into IKE's effects on LLMs and how reversal tokens mitigate them. This work represents a significant step towards enhancing LLM resilience against potential misuse of in-context editing, improving their transparency and trustworthiness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07557v1">Using LLMs for Analyzing AIS Data</a></div>
    <div class="paper-meta">
      📅 2025-04-10
    </div>
    <details class="paper-abstract">
      Recent research in Large Language Models (LLMs), has had a profound impact across various fields, including mobility data science. This paper explores the and experiment with different approaches to using LLMs for analyzing AIS data. We propose a set of carefully designed queries to assess the reasoning capabilities of LLMs in this kind of tasks. Further, we experiment with four different methods: (1) using LLMs as a natural language interface to a spatial database, (2) reasoning on raw data, (3) reasoning on compressed trajectories, and (4) reasoning on semantic trajectories. We investigate the strengths and weaknesses for the four methods, and discuss the findings. The goal is to provide valuable insights for both researchers and practitioners on selecting the most appropriate LLM-based method depending on their specific data analysis objectives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.06193v2">Can LLMs Replace Human Evaluators? An Empirical Study of LLM-as-a-Judge in Software Engineering</a></div>
    <div class="paper-meta">
      📅 2025-04-10
      | 💬 Accepted by ISSTA 2025: https://conf.researchr.org/details/issta-2025/issta-2025-papers/85/Can-LLMs-replace-Human-Evaluators-An-Empirical-Study-of-LLM-as-a-Judge-in-Software-E
    </div>
    <details class="paper-abstract">
      Recently, large language models (LLMs) have been deployed to tackle various software engineering (SE) tasks like code generation, significantly advancing the automation of SE tasks. However, assessing the quality of these LLM-generated code and text remains challenging. The commonly used Pass@k metric necessitates extensive unit tests and configured environments, demands a high labor cost, and is not suitable for evaluating LLM-generated text. Conventional metrics like BLEU, which measure only lexical rather than semantic similarity, have also come under scrutiny. In response, a new trend has emerged to employ LLMs for automated evaluation, known as LLM-as-a-judge. These LLM-as-a-judge methods are claimed to better mimic human assessment than conventional metrics without relying on high-quality reference answers. Nevertheless, their exact human alignment in SE tasks remains unexplored. In this paper, we empirically explore LLM-as-a-judge methods for evaluating SE tasks, focusing on their alignment with human judgments. We select seven LLM-as-a-judge methods that utilize general-purpose LLMs, alongside two LLMs specifically fine-tuned for evaluation. After generating and manually scoring LLM responses on three recent SE datasets of code translation, code generation, and code summarization, we then prompt these methods to evaluate each response. Finally, we compare the scores generated by these methods with human evaluation. The results indicate that output-based methods reach the highest Pearson correlation of 81.32 and 68.51 with human scores in code translation and generation, achieving near-human evaluation, noticeably outperforming ChrF++, one of the best conventional metrics, at 34.23 and 64.92. Such output-based methods prompt LLMs to output judgments directly, and exhibit more balanced score distributions that resemble human score patterns. Finally, we provide...
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07494v1">Apt-Serve: Adaptive Request Scheduling on Hybrid Cache for Scalable LLM Inference Serving</a></div>
    <div class="paper-meta">
      📅 2025-04-10
    </div>
    <details class="paper-abstract">
      Large language model (LLM) inference serving systems are essential to various LLM-based applications. As demand for LLM services continues to grow, scaling these systems to handle high request rates while meeting latency Service-Level Objectives (SLOs), referred to as effective throughput, becomes critical. However, existing systems often struggle to improve effective throughput, primarily due to a significant decline in Time To First Token (TTFT) SLO attainment. We identify two major causes of this bottleneck: (1) memory-intensive KV cache that limits batch size expansion under GPU memory constraints, and (2) rigid batch composition enforced by the default First-Come-First-Serve scheduling policy. In this paper, we introduce Apt-Serve, a scalable framework designed to enhance effective throughput in LLM inference serving. Apt-Serve features a new hybrid cache scheme that combines KV cache with a memory-efficient hidden cache for reusable input hidden state vectors, allowing large batch sizes and improving request concurrency. Based on the hybrid cache, Apt-Serve employs an adaptive runtime scheduling mechanism that dynamically optimizes batch composition. We formally define the adaptive scheduling optimization problem and propose an efficient algorithm with theoretical guarantees. Extensive evaluations on three real-world datasets and LLMs ranging from 13B to 66B parameters demonstrate that Apt-Serve achieves up to 8.8x improvement in effective throughput compared to the state-of-the-art inference serving systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07479v1">UniCAIM: A Unified CAM/CIM Architecture with Static-Dynamic KV Cache Pruning for Efficient Long-Context LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-04-10
    </div>
    <details class="paper-abstract">
      Transformer-based large language models (LLMs) have achieved impressive performance in various natural language processing (NLP) applications. However, the high memory and computation cost induced by the KV cache limits the inference efficiency, especially for long input sequences. Compute-in-memory (CIM)-based accelerators have been proposed for LLM acceleration with KV cache pruning. However, as existing accelerators only support static pruning with a fixed pattern or dynamic pruning with primitive implementations, they suffer from either high accuracy degradation or low efficiency. In this paper, we propose a ferroelectric FET (FeFET)-based unified content addressable memory (CAM) and CIM architecture, dubbed as UniCAIM. UniCAIM features simultaneous support for static and dynamic pruning with 3 computation modes: 1) in the CAM mode, UniCAIM enables approximate similarity measurement in O(1) time for dynamic KV cache pruning with high energy efficiency; 2) in the charge-domain CIM mode, static pruning can be supported based on accumulative similarity score, which is much more flexible compared to fixed patterns; 3) in the current-domain mode, exact attention computation can be conducted with a subset of selected KV cache. We further propose a novel CAM/CIM cell design that leverages the multi-level characteristics of FeFETs for signed multibit storage of the KV cache and in-place attention computation. With extensive experimental results, we demonstrate UniCAIM can reduce the area-energy-delay product (AEDP) by 8.2-831x over the state-ofthe-art CIM-based LLM accelerators at the circuit level, along with high accuracy comparable with dense attention at the application level, showing its great potential for efficient long-context LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07459v1">Beyond LLMs: A Linguistic Approach to Causal Graph Generation from Narrative Texts</a></div>
    <div class="paper-meta">
      📅 2025-04-10
      | 💬 published at the 7th Workshop on Narrative Understanding, NAACL 2025
    </div>
    <details class="paper-abstract">
      We propose a novel framework for generating causal graphs from narrative texts, bridging high-level causality and detailed event-specific relationships. Our method first extracts concise, agent-centered vertices using large language model (LLM)-based summarization. We introduce an "Expert Index," comprising seven linguistically informed features, integrated into a Situation-Task-Action-Consequence (STAC) classification model. This hybrid system, combining RoBERTa embeddings with the Expert Index, achieves superior precision in causal link identification compared to pure LLM-based approaches. Finally, a structured five-iteration prompting process refines and constructs connected causal graphs. Experiments on 100 narrative chapters and short stories demonstrate that our approach consistently outperforms GPT-4o and Claude 3.5 in causal graph quality, while maintaining readability. The open-source tool provides an interpretable, efficient solution for capturing nuanced causal chains in narratives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19379v3">Marconi: Prefix Caching for the Era of Hybrid LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-10
      | 💬 MLSys 2025 camera-ready version
    </div>
    <details class="paper-abstract">
      Hybrid models that combine the language modeling capabilities of Attention layers with the efficiency of Recurrent layers (e.g., State Space Models) have gained traction in practically supporting long contexts in Large Language Model serving. Yet, the unique properties of these models complicate the usage of complementary efficiency optimizations such as prefix caching that skip redundant computations across requests. Most notably, their use of in-place state updates for recurrent layers precludes rolling back cache entries for partial sequence overlaps, and instead mandates only exact-match cache hits; the effect is a deluge of (large) cache entries per sequence, most of which yield minimal reuse opportunities. We present Marconi, the first system that supports efficient prefix caching with Hybrid LLMs. Key to Marconi are its novel admission and eviction policies that more judiciously assess potential cache entries based not only on recency, but also on (1) forecasts of their reuse likelihood across a taxonomy of different hit scenarios, and (2) the compute savings that hits deliver relative to memory footprints. Across diverse workloads and Hybrid models, Marconi achieves up to 34.4$\times$ higher token hit rates (71.1% or 617 ms lower TTFT) compared to state-of-the-art prefix caching systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07440v1">Revisiting LLM Evaluation through Mechanism Interpretability: a New Metric and Model Utility Law</a></div>
    <div class="paper-meta">
      📅 2025-04-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become indispensable across academia, industry, and daily applications, yet current evaluation methods struggle to keep pace with their rapid development. In this paper, we analyze the core limitations of traditional evaluation pipelines and propose a novel metric, the Model Utilization Index (MUI), which introduces mechanism interpretability techniques to complement traditional performance metrics. MUI quantifies the extent to which a model leverages its capabilities to complete tasks. The core idea is that to assess an LLM's overall ability, we must evaluate not only its task performance but also the effort expended to achieve the outcome. Our extensive experiments reveal an inverse relationship between MUI and performance, from which we deduce a common trend observed in popular LLMs, which we term the Utility Law. Based on this, we derive four corollaries that address key challenges, including training judgement, the issue of data contamination, fairness in model comparison, and data diversity. We hope that our survey, novel metric, and utility law will foster mutual advancement in both evaluation and mechanism interpretability. Our code can be found at https://github.com/ALEX-nlp/MUI-Eva.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07431v1">LLM-Enabled Data Transmission in End-to-End Semantic Communication</a></div>
    <div class="paper-meta">
      📅 2025-04-10
    </div>
    <details class="paper-abstract">
      Emerging services such as augmented reality (AR) and virtual reality (VR) have increased the volume of data transmitted in wireless communication systems, revealing the limitations of traditional Shannon theory. To address these limitations, semantic communication has been proposed as a solution that prioritizes the meaning of messages over the exact transmission of bits. This paper explores semantic communication for text data transmission in end-to-end (E2E) systems through a novel approach called KG-LLM semantic communication, which integrates knowledge graph (KG) extraction and large language model (LLM) coding. In this method, the transmitter first utilizes a KG to extract key entities and relationships from sentences. The extracted information is then encoded using an LLM to obtain the semantic meaning. On the receiver side, messages are decoded using another LLM, while a bidirectional encoder representations from transformers (i.e., BERT) model further refines the reconstructed sentences for improved semantic similarity. The KG-LLM semantic communication method reduces the transmitted text data volume by 30% through KG-based compression and achieves 84\% semantic similarity between the original and received messages. This demonstrates the KG-LLM methods efficiency and robustness in semantic communication systems, outperforming the deep learning-based semantic communication model (DeepSC), which achieves only 63%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06575v2">Defending LLM Watermarking Against Spoofing Attacks with Contrastive Representation Learning</a></div>
    <div class="paper-meta">
      📅 2025-04-10
    </div>
    <details class="paper-abstract">
      Watermarking has emerged as a promising technique for detecting texts generated by LLMs. Current research has primarily focused on three design criteria: high quality of the watermarked text, high detectability, and robustness against removal attack. However, the security against spoofing attacks remains relatively understudied. For example, a piggyback attack can maliciously alter the meaning of watermarked text-transforming it into hate speech-while preserving the original watermark, thereby damaging the reputation of the LLM provider. We identify two core challenges that make defending against spoofing difficult: (1) the need for watermarks to be both sensitive to semantic-distorting changes and insensitive to semantic-preserving edits, and (2) the contradiction between the need to detect global semantic shifts and the local, auto-regressive nature of most watermarking schemes. To address these challenges, we propose a semantic-aware watermarking algorithm that post-hoc embeds watermarks into a given target text while preserving its original meaning. Our method introduces a semantic mapping model, which guides the generation of a green-red token list, contrastively trained to be sensitive to semantic-distorting changes and insensitive to semantic-preserving changes. Experiments on two standard benchmarks demonstrate strong robustness against removal attacks and security against spoofing attacks, including sentiment reversal and toxic content insertion, while maintaining high watermark detectability. Our approach offers a significant step toward more secure and semantically aware watermarking for LLMs. Our code is available at https://github.com/UCSB-NLP-Chang/contrastive-watermark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06261v2">Hogwild! Inference: Parallel LLM Generation via Concurrent Attention</a></div>
    <div class="paper-meta">
      📅 2025-04-09
      | 💬 Preprint, work in progress
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated the ability to tackle increasingly complex tasks through advanced reasoning, long-form content generation, and tool use. Solving these tasks often involves long inference-time computations. In human problem solving, a common strategy to expedite work is collaboration: by dividing the problem into sub-tasks, exploring different strategies concurrently, etc. Recent research has shown that LLMs can also operate in parallel by implementing explicit cooperation frameworks, such as voting mechanisms or the explicit creation of independent sub-tasks that can be executed in parallel. However, each of these frameworks may not be suitable for all types of tasks, which can hinder their applicability. In this work, we propose a different design approach: we run LLM "workers" in parallel , allowing them to synchronize via a concurrently-updated attention cache and prompt these workers to decide how best to collaborate. Our approach allows the instances to come up with their own collaboration strategy for the problem at hand, all the while "seeing" each other's partial progress in the concurrent cache. We implement this approach via Hogwild! Inference: a parallel LLM inference engine where multiple instances of the same LLM run in parallel with the same attention cache, with "instant" access to each other's generated tokens. Hogwild! inference takes advantage of Rotary Position Embeddings (RoPE) to avoid recomputation while improving parallel hardware utilization. We find that modern reasoning-capable LLMs can perform inference with shared Key-Value cache out of the box, without additional fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18389v2">Monte Carlo Temperature: a robust sampling strategy for LLM's uncertainty quantification methods</a></div>
    <div class="paper-meta">
      📅 2025-04-09
    </div>
    <details class="paper-abstract">
      Uncertainty quantification (UQ) in Large Language Models (LLMs) is essential for their safe and reliable deployment, particularly in critical applications where incorrect outputs can have serious consequences. Current UQ methods typically rely on querying the model multiple times using non-zero temperature sampling to generate diverse outputs for uncertainty estimation. However, the impact of selecting a given temperature parameter is understudied, and our analysis reveals that temperature plays a fundamental role in the quality of uncertainty estimates. The conventional approach of identifying optimal temperature values requires expensive hyperparameter optimization (HPO) that must be repeated for each new model-dataset combination. We propose Monte Carlo Temperature (MCT), a robust sampling strategy that eliminates the need for temperature calibration. Our analysis reveals that: 1) MCT provides more robust uncertainty estimates across a wide range of temperatures, 2) MCT improves the performance of UQ methods by replacing fixed-temperature strategies that do not rely on HPO, and 3) MCT achieves statistical parity with oracle temperatures, which represent the ideal outcome of a well-tuned but computationally expensive HPO process. These findings demonstrate that effective UQ can be achieved without the computational burden of temperature parameter calibration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07015v1">LLM-IFT: LLM-Powered Information Flow Tracking for Secure Hardware</a></div>
    <div class="paper-meta">
      📅 2025-04-09
      | 💬 This paper is presented at IEEE VLSI Test Symposium (VTS) 2025
    </div>
    <details class="paper-abstract">
      As modern hardware designs grow in complexity and size, ensuring security across the confidentiality, integrity, and availability (CIA) triad becomes increasingly challenging. Information flow tracking (IFT) is a widely-used approach to tracing data propagation, identifying unauthorized activities that may compromise confidentiality or/and integrity in hardware. However, traditional IFT methods struggle with scalability and adaptability, particularly in high-density and interconnected architectures, leading to tracing bottlenecks that limit applicability in large-scale hardware. To address these limitations and show the potential of transformer-based models in integrated circuit (IC) design, this paper introduces LLM-IFT that integrates large language models (LLM) for the realization of the IFT process in hardware. LLM-IFT exploits LLM-driven structured reasoning to perform hierarchical dependency analysis, systematically breaking down even the most complex designs. Through a multi-step LLM invocation, the framework analyzes both intra-module and inter-module dependencies, enabling comprehensive IFT assessment. By focusing on a set of Trust-Hub vulnerability test cases at both the IP level and the SoC level, our experiments demonstrate a 100\% success rate in accurate IFT analysis for confidentiality and integrity checks in hardware.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11283v2">AdvBDGen: Adversarially Fortified Prompt-Specific Fuzzy Backdoor Generator Against LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-04-09
      | 💬 Published at the Neurips Safe Generative AI Workshop 2024
    </div>
    <details class="paper-abstract">
      With the growing adoption of reinforcement learning with human feedback (RLHF) for aligning large language models (LLMs), the risk of backdoor installation during alignment has increased, leading to unintended and harmful behaviors. Existing backdoor triggers are typically limited to fixed word patterns, making them detectable during data cleaning and easily removable post-poisoning. In this work, we explore the use of prompt-specific paraphrases as backdoor triggers, enhancing their stealth and resistance to removal during LLM alignment. We propose AdvBDGen, an adversarially fortified generative fine-tuning framework that automatically generates prompt-specific backdoors that are effective, stealthy, and transferable across models. AdvBDGen employs a generator-discriminator pair, fortified by an adversary, to ensure the installability and stealthiness of backdoors. It enables the crafting and successful installation of complex triggers using as little as 3% of the fine-tuning data. Once installed, these backdoors can jailbreak LLMs during inference, demonstrate improved stability against perturbations compared to traditional constant triggers, and are more challenging to remove. These findings underscore an urgent need for the research community to develop more robust defenses against adversarial backdoor threats in LLM alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06969v1">Towards LLMs Robustness to Changes in Prompt Format Styles</a></div>
    <div class="paper-meta">
      📅 2025-04-09
      | 💬 NAACL Student Research Workshop (SRW) 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have gained popularity in recent years for their utility in various applications. However, they are sensitive to non-semantic changes in prompt formats, where small changes in the prompt format can lead to significant performance fluctuations. In the literature, this problem is commonly referred to as prompt brittleness. Previous research on prompt engineering has focused mainly on developing techniques for identifying the optimal prompt for specific tasks. Some studies have also explored the issue of prompt brittleness and proposed methods to quantify performance variations; however, no simple solution has been found to address this challenge. We propose Mixture of Formats (MOF), a simple and efficient technique for addressing prompt brittleness in LLMs by diversifying the styles used in the prompt few-shot examples. MOF was inspired by computer vision techniques that utilize diverse style datasets to prevent models from associating specific styles with the target variable. Empirical results show that our proposed technique reduces style-induced prompt brittleness in various LLMs while also enhancing overall performance across prompt variations and different datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02916v3">LLM Safeguard is a Double-Edged Sword: Exploiting False Positives for Denial-of-Service Attacks</a></div>
    <div class="paper-meta">
      📅 2025-04-09
    </div>
    <details class="paper-abstract">
      Safety is a paramount concern for large language models (LLMs) in open deployment, motivating the development of safeguard methods that enforce ethical and responsible use through safety alignment or guardrail mechanisms. Jailbreak attacks that exploit the \emph{false negatives} of safeguard methods have emerged as a prominent research focus in the field of LLM security. However, we found that the malicious attackers could also exploit false positives of safeguards, i.e., fooling the safeguard model to block safe content mistakenly, leading to a denial-of-service (DoS) affecting LLM users. To bridge the knowledge gap of this overlooked threat, we explore multiple attack methods that include inserting a short adversarial prompt into user prompt templates and corrupting the LLM on the server by poisoned fine-tuning. In both ways, the attack triggers safeguard rejections of user requests from the client. Our evaluation demonstrates the severity of this threat across multiple scenarios. For instance, in the scenario of white-box adversarial prompt injection, the attacker can use our optimization process to automatically generate seemingly safe adversarial prompts, approximately only 30 characters long, that universally block over 97% of user requests on Llama Guard 3. These findings reveal a new dimension in LLM safeguard evaluation -- adversarial robustness to false positives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.07991v5">Human and LLM Biases in Hate Speech Annotations: A Socio-Demographic Analysis of Annotators and Targets</a></div>
    <div class="paper-meta">
      📅 2025-04-09
    </div>
    <details class="paper-abstract">
      The rise of online platforms exacerbated the spread of hate speech, demanding scalable and effective detection. However, the accuracy of hate speech detection systems heavily relies on human-labeled data, which is inherently susceptible to biases. While previous work has examined the issue, the interplay between the characteristics of the annotator and those of the target of the hate are still unexplored. We fill this gap by leveraging an extensive dataset with rich socio-demographic information of both annotators and targets, uncovering how human biases manifest in relation to the target's attributes. Our analysis surfaces the presence of widespread biases, which we quantitatively describe and characterize based on their intensity and prevalence, revealing marked differences. Furthermore, we compare human biases with those exhibited by persona-based LLMs. Our findings indicate that while persona-based LLMs do exhibit biases, these differ significantly from those of human annotators. Overall, our work offers new and nuanced results on human biases in hate speech annotations, as well as fresh insights into the design of AI-driven hate speech detection systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06943v1">Review of Case-Based Reasoning for LLM Agents: Theoretical Foundations, Architectural Components, and Cognitive Integration</a></div>
    <div class="paper-meta">
      📅 2025-04-09
    </div>
    <details class="paper-abstract">
      Agents powered by Large Language Models (LLMs) have recently demonstrated impressive capabilities in various tasks. Still, they face limitations in tasks requiring specific, structured knowledge, flexibility, or accountable decision-making. While agents are capable of perceiving their environments, forming inferences, planning, and executing actions towards goals, they often face issues such as hallucinations and lack of contextual memory across interactions. This paper explores how Case-Based Reasoning (CBR), a strategy that solves new problems by referencing past experiences, can be integrated into LLM agent frameworks. This integration allows LLMs to leverage explicit knowledge, enhancing their effectiveness. We systematically review the theoretical foundations of these enhanced agents, identify critical framework components, and formulate a mathematical model for the CBR processes of case retrieval, adaptation, and learning. We also evaluate CBR-enhanced agents against other methods like Chain-of-Thought reasoning and standard Retrieval-Augmented Generation, analyzing their relative strengths. Moreover, we explore how leveraging CBR's cognitive dimensions (including self-reflection, introspection, and curiosity) via goal-driven autonomy mechanisms can further enhance the LLM agent capabilities. Contributing to the ongoing research on neuro-symbolic hybrid systems, this work posits CBR as a viable technique for enhancing the reasoning skills and cognitive aspects of autonomous LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15050v2">Studying and Understanding the Effectiveness and Failures of Conversational LLM-Based Repair</a></div>
    <div class="paper-meta">
      📅 2025-04-09
    </div>
    <details class="paper-abstract">
      Automated program repair (APR) is designed to automate the process of bug-fixing. In recent years, thanks to the rapid development of large language models (LLMs), automated repair has achieved remarkable progress. Advanced APR techniques powered by conversational LLMs, most notably ChatGPT, have exhibited impressive repair abilities and gained increasing popularity due to the capabilities of the underlying LLMs in providing repair feedback and performing iterative patch improvement. Despite the superiority, conversational APR techniques still fail to repair a large number of bugs. For example, a state-of-the-art conversational technique ChatRepair does not correctly repair over half of the single-function bugs in the Defects4J dataset. To understand the effectiveness and failures of conversational LLM-based repair and provide possible directions for improvement, we studied the exemplary ChatRepair with a focus on comparing the effectiveness of its cloze-style and full function repair strategies, assessing its key iterative component for patch improvement, and analyzing the repair failures. Our study has led to a series of findings, which we believe provide key implications for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06823v1">Open Problems and a Hypothetical Path Forward in LLM Knowledge Paradigms</a></div>
    <div class="paper-meta">
      📅 2025-04-09
      | 💬 Blog post preprint, work in progress
    </div>
    <details class="paper-abstract">
      Knowledge is fundamental to the overall capabilities of Large Language Models (LLMs). The knowledge paradigm of a model, which dictates how it encodes and utilizes knowledge, significantly affects its performance. Despite the continuous development of LLMs under existing knowledge paradigms, issues within these frameworks continue to constrain model potential. This blog post highlight three critical open problems limiting model capabilities: (1) challenges in knowledge updating for LLMs, (2) the failure of reverse knowledge generalization (the reversal curse), and (3) conflicts in internal knowledge. We review recent progress made in addressing these issues and discuss potential general solutions. Based on observations in these areas, we propose a hypothetical paradigm based on Contextual Knowledge Scaling, and further outline implementation pathways that remain feasible within contemporary techniques. Evidence suggests this approach holds potential to address current shortcomings, serving as our vision for future model paradigms. This blog post aims to provide researchers with a brief overview of progress in LLM knowledge systems, while provide inspiration for the development of next-generation model architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.05821v2">Optimizing LLM Queries in Relational Data Analytics Workloads</a></div>
    <div class="paper-meta">
      📅 2025-04-09
    </div>
    <details class="paper-abstract">
      Batch data analytics is a growing application for Large Language Models (LLMs). LLMs enable users to perform a wide range of natural language tasks, such as classification, entity extraction, and translation, over large datasets. However, LLM inference is highly costly and slow: for example, an NVIDIA L4 GPU running Llama3-8B can only process 6 KB of text per second, taking about a day to handle 15 GB of data; processing a similar amount of data costs around $10K on OpenAI's GPT-4o. In this paper, we propose novel techniques that can significantly reduce the cost of LLM calls for relational data analytics workloads. Our key contribution is developing efficient algorithms for reordering the rows and the fields within each row of an input table to maximize key-value (KV) cache reuse when performing LLM serving. As such, our approach can be easily applied to existing analytics systems and serving platforms. Our evaluation shows that our solution can yield up to 3.4x improvement in job completion time on a benchmark of diverse LLM-based queries using Llama 3 models. Our solution also achieves a 32% cost savings under OpenAI and Anthropic pricing models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06650v1">ThoughtProbe: Classifier-Guided Thought Space Exploration Leveraging LLM Intrinsic Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-04-09
    </div>
    <details class="paper-abstract">
      Pre-trained large language models (LLMs) have been demonstrated to possess intrinsic reasoning capabilities that can emerge naturally when expanding the response space. However, the neural representation mechanisms underlying these intrinsic capabilities and approaches for their optimal utilization remain inadequately understood. In this work, we make the key discovery that a simple linear classifier can effectively detect intrinsic reasoning capabilities in LLMs' activation space, particularly within specific representation types and network layers. Based on this finding, we propose a classifier-guided search framework that strategically explore a tree-structured response space. In each node expansion, the classifier serves as a scoring and ranking mechanism that efficiently allocates computational resources by identifying and prioritizing more thoughtful reasoning directions for continuation. After completing the tree expansion, we collect answers from all branches to form a candidate answer pool. We propose a branch-aggregation selection method that marginalizes over all supporting branches by aggregating their thoughtfulness scores, thereby identifying the optimal answer from the pool. Experimental results show that our framework's comprehensive exploration not only covers valid reasoning chains but also effectively identifies them, achieving significant improvements across multiple arithmetic reasoning benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06614v1">AgentFM: Role-Aware Failure Management for Distributed Databases with LLM-Driven Multi-Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-09
      | 💬 accepted by FSE-IVR'25
    </div>
    <details class="paper-abstract">
      Distributed databases are critical infrastructures for today's large-scale software systems, making effective failure management essential to ensure software availability. However, existing approaches often overlook the role distinctions within distributed databases and rely on small-scale models with limited generalization capabilities. In this paper, we conduct a preliminary empirical study to emphasize the unique significance of different roles. Building on this insight, we propose AgentFM, a role-aware failure management framework for distributed databases powered by LLM-driven multi-agents. AgentFM addresses failure management by considering system roles, data roles, and task roles, with a meta-agent orchestrating these components. Preliminary evaluations using Apache IoTDB demonstrate the effectiveness of AgentFM and open new directions for further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06600v1">Automated Business Process Analysis: An LLM-Based Approach to Value Assessment</a></div>
    <div class="paper-meta">
      📅 2025-04-09
    </div>
    <details class="paper-abstract">
      Business processes are fundamental to organizational operations, yet their optimization remains challenging due to the timeconsuming nature of manual process analysis. Our paper harnesses Large Language Models (LLMs) to automate value-added analysis, a qualitative process analysis technique that aims to identify steps in the process that do not deliver value. To date, this technique is predominantly manual, time-consuming, and subjective. Our method offers a more principled approach which operates in two phases: first, decomposing high-level activities into detailed steps to enable granular analysis, and second, performing a value-added analysis to classify each step according to Lean principles. This approach enables systematic identification of waste while maintaining the semantic understanding necessary for qualitative analysis. We develop our approach using 50 business process models, for which we collect and publish manual ground-truth labels. Our evaluation, comparing zero-shot baselines with more structured prompts reveals (a) a consistent benefit of structured prompting and (b) promising performance for both tasks. We discuss the potential for LLMs to augment human expertise in qualitative process analysis while reducing the time and subjectivity inherent in manual approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06581v1">Right Prediction, Wrong Reasoning: Uncovering LLM Misalignment in RA Disease Diagnosis</a></div>
    <div class="paper-meta">
      📅 2025-04-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) offer a promising pre-screening tool, improving early disease detection and providing enhanced healthcare access for underprivileged communities. The early diagnosis of various diseases continues to be a significant challenge in healthcare, primarily due to the nonspecific nature of early symptoms, the shortage of expert medical practitioners, and the need for prolonged clinical evaluations, all of which can delay treatment and adversely affect patient outcomes. With impressive accuracy in prediction across a range of diseases, LLMs have the potential to revolutionize clinical pre-screening and decision-making for various medical conditions. In this work, we study the diagnostic capability of LLMs for Rheumatoid Arthritis (RA) with real world patients data. Patient data was collected alongside diagnoses from medical experts, and the performance of LLMs was evaluated in comparison to expert diagnoses for RA disease prediction. We notice an interesting pattern in disease diagnosis and find an unexpected \textit{misalignment between prediction and explanation}. We conduct a series of multi-round analyses using different LLM agents. The best-performing model accurately predicts rheumatoid arthritis (RA) diseases approximately 95\% of the time. However, when medical experts evaluated the reasoning generated by the model, they found that nearly 68\% of the reasoning was incorrect. This study highlights a clear misalignment between LLMs high prediction accuracy and its flawed reasoning, raising important questions about relying on LLM explanations in clinical settings. \textbf{LLMs provide incorrect reasoning to arrive at the correct answer for RA disease diagnosis.}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06577v1">Bypassing Safety Guardrails in LLMs Using Humor</a></div>
    <div class="paper-meta">
      📅 2025-04-09
    </div>
    <details class="paper-abstract">
      In this paper, we show it is possible to bypass the safety guardrails of large language models (LLMs) through a humorous prompt including the unsafe request. In particular, our method does not edit the unsafe request and follows a fixed template -- it is simple to implement and does not need additional LLMs to craft prompts. Extensive experiments show the effectiveness of our method across different LLMs. We also show that both removing and adding more humor to our method can reduce its effectiveness -- excessive humor possibly distracts the LLM from fulfilling its unsafe request. Thus, we argue that LLM jailbreaking occurs when there is a proper balance between focus on the unsafe request and presence of humor.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06575v1">Defending LLM Watermarking Against Spoofing Attacks with Contrastive Representation Learning</a></div>
    <div class="paper-meta">
      📅 2025-04-09
    </div>
    <details class="paper-abstract">
      Watermarking has emerged as a promising technique for detecting texts generated by LLMs. Current research has primarily focused on three design criteria: high quality of the watermarked text, high detectability, and robustness against removal attack. However, the security against spoofing attacks remains relatively understudied. For example, a piggyback attack can maliciously alter the meaning of watermarked text-transforming it into hate speech-while preserving the original watermark, thereby damaging the reputation of the LLM provider. We identify two core challenges that make defending against spoofing difficult: (1) the need for watermarks to be both sensitive to semantic-distorting changes and insensitive to semantic-preserving edits, and (2) the contradiction between the need to detect global semantic shifts and the local, auto-regressive nature of most watermarking schemes. To address these challenges, we propose a semantic-aware watermarking algorithm that post-hoc embeds watermarks into a given target text while preserving its original meaning. Our method introduces a semantic mapping model, which guides the generation of a green-red token list, contrastively trained to be sensitive to semantic-distorting changes and insensitive to semantic-preserving changes. Experiments on two standard benchmarks demonstrate strong robustness against removal attacks and security against spoofing attacks, including sentiment reversal and toxic content insertion, while maintaining high watermark detectability. Our approach offers a significant step toward more secure and semantically aware watermarking for LLMs. Our code is available at https://github.com/UCSB-NLP-Chang/contrastive-watermark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06160v2">Navigating the Rabbit Hole: Emergent Biases in LLM-Generated Attack Narratives Targeting Mental Health Groups</a></div>
    <div class="paper-meta">
      📅 2025-04-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have been shown to demonstrate imbalanced biases against certain groups. However, the study of unprovoked targeted attacks by LLMs towards at-risk populations remains underexplored. Our paper presents three novel contributions: (1) the explicit evaluation of LLM-generated attacks on highly vulnerable mental health groups; (2) a network-based framework to study the propagation of relative biases; and (3) an assessment of the relative degree of stigmatization that emerges from these attacks. Our analysis of a recently released large-scale bias audit dataset reveals that mental health entities occupy central positions within attack narrative networks, as revealed by a significantly higher mean centrality of closeness (p-value = 4.06e-10) and dense clustering (Gini coefficient = 0.7). Drawing from sociological foundations of stigmatization theory, our stigmatization analysis indicates increased labeling components for mental health disorder-related targets relative to initial targets in generation chains. Taken together, these insights shed light on the structural predilections of large language models to heighten harmful discourse and highlight the need for suitable approaches for mitigation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17620v2">A Case Study of Scalable Content Annotation Using Multi-LLM Consensus and Human Review</a></div>
    <div class="paper-meta">
      📅 2025-04-09
      | 💬 4 pages, GenAICHI 2025 accepted
    </div>
    <details class="paper-abstract">
      Content annotation at scale remains challenging, requiring substantial human expertise and effort. This paper presents a case study in code documentation analysis, where we explore the balance between automation efficiency and annotation accuracy. We present MCHR (Multi-LLM Consensus with Human Review), a novel semi-automated framework that enhances annotation scalability through the systematic integration of multiple LLMs and targeted human review. Our framework introduces a structured consensus-building mechanism among LLMs and an adaptive review protocol that strategically engages human expertise. Through our case study, we demonstrate that MCHR reduces annotation time by 32% to 100% compared to manual annotation while maintaining high accuracy (85.5% to 98%) across different difficulty levels, from basic binary classification to challenging open-set scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06511v1">GTS-LUM: Reshaping User Behavior Modeling with LLMs in Telecommunications Industry</a></div>
    <div class="paper-meta">
      📅 2025-04-09
    </div>
    <details class="paper-abstract">
      As telecommunication service providers shifting their focus to analyzing user behavior for package design and marketing interventions, a critical challenge lies in developing a unified, end-to-end framework capable of modeling long-term and periodic user behavior sequences with diverse time granularities, multi-modal data inputs, and heterogeneous labels. This paper introduces GTS-LUM, a novel user behavior model that redefines modeling paradigms in telecommunication settings. GTS-LUM adopts a (multi-modal) encoder-adapter-LLM decoder architecture, enhanced with several telecom-specific innovations. Specifically, the model incorporates an advanced timestamp processing method to handle varying time granularities. It also supports multi-modal data inputs -- including structured tables and behavior co-occurrence graphs -- and aligns these with semantic information extracted by a tokenizer using a Q-former structure. Additionally, GTS-LUM integrates a front-placed target-aware mechanism to highlight historical behaviors most relevant to the target. Extensive experiments on industrial dataset validate the effectiveness of this end-to-end framework and also demonstrate that GTS-LUM outperforms LLM4Rec approaches which are popular in recommendation systems, offering an effective and generalizing solution for user behavior modeling in telecommunications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.11871v4">Generative AI Voting: Fair Collective Choice is Resilient to LLM Biases and Inconsistencies</a></div>
    <div class="paper-meta">
      📅 2025-04-09
      | 💬 23 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Scaling up deliberative and voting participation is a longstanding endeavor -- a cornerstone for direct democracy and legitimate collective choice. Recent breakthroughs in generative artificial intelligence (AI) and large language models (LLMs) unravel new capabilities for AI personal assistants to overcome cognitive bandwidth limitations of humans, providing decision support or even direct representation of human voters at large scale. However, the quality of this representation and what underlying biases manifest when delegating collective decision-making to LLMs is an alarming and timely challenge to tackle. By rigorously emulating with high realism more than >50K LLM voting personas in 306 real-world voting elections, we disentangle the nature of different biases in LLMS (GPT 3, GPT 3.5, and Llama2). Complex preferential ballot formats exhibit significant inconsistencies compared to simpler majoritarian elections that show higher consistency. Strikingly though, by demonstrating for the first time in real-world a proportional representation of voters in direct democracy, we are also able to show that fair ballot aggregation methods, such as equal shares, prove to be a win-win: fairer voting outcomes for humans with fairer AI representation, especially for voters who are likely to abstain. This novel underlying relationship proves paramount for democratic resilience in progressives scenarios with low voters turnout and voter fatigue supported by AI representatives: abstained voters are mitigated by recovering highly representative voting outcomes that are fairer. These interdisciplinary insights provide remarkable foundations for science, policymakers, and citizens to develop safeguards and resilience for AI risks in democratic innovations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06265v2">GOLLuM: Gaussian Process Optimized LLMs -- Reframing LLM Finetuning through Bayesian Optimization</a></div>
    <div class="paper-meta">
      📅 2025-04-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can encode complex relationships in their latent spaces, yet harnessing them for optimization under uncertainty remains challenging. We address this gap with a novel architecture that reframes LLM finetuning as Gaussian process (GP) marginal likelihood optimization via deep kernel methods. We introduce LLM-based deep kernels, jointly optimized with GPs to preserve the benefits of both - LLMs to provide a rich and flexible input space for Bayesian optimization and - GPs to model this space with predictive uncertainty for more efficient sampling. Applied to Buchwald-Hartwig reaction optimization, our method nearly doubles the discovery rate of high-performing reactions compared to static LLM embeddings (from 24% to 43% coverage of the top 5% reactions in just 50 optimization iterations). We also observe a 14% improvement over domain-specific representations without requiring specialized features. Extensive empirical evaluation across 19 benchmarks - ranging from general chemistry to reaction and molecular property optimization - demonstrates our method's robustness, generality, and consistent improvements across: (1) tasks, (2) LLM architectures (encoder, decoder, encoder-decoder), (3) pretraining domains (chemistry-related or general-purpose) and (4) hyperparameter settings (tuned once on a single dataset). Finally, we explain these improvements: joint LLM-GP optimization through marginal likelihood implicitly performs contrastive learning, aligning representations to produce (1) better-structured embedding spaces, (2) improved uncertainty calibration, and (3) more efficient sampling - without requiring any external loss. This work provides both practical advances in sample-efficient optimization and insights into what makes effective Bayesian optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07336v1">Zeus: Zero-shot LLM Instruction for Union Segmentation in Multimodal Medical Imaging</a></div>
    <div class="paper-meta">
      📅 2025-04-09
      | 💬 21 pages, 4 figures, In Press by a journal
    </div>
    <details class="paper-abstract">
      Medical image segmentation has achieved remarkable success through the continuous advancement of UNet-based and Transformer-based foundation backbones. However, clinical diagnosis in the real world often requires integrating domain knowledge, especially textual information. Conducting multimodal learning involves visual and text modalities shown as a solution, but collecting paired vision-language datasets is expensive and time-consuming, posing significant challenges. Inspired by the superior ability in numerous cross-modal tasks for Large Language Models (LLMs), we proposed a novel Vision-LLM union framework to address the issues. Specifically, we introduce frozen LLMs for zero-shot instruction generation based on corresponding medical images, imitating the radiology scanning and report generation process. {To better approximate real-world diagnostic processes}, we generate more precise text instruction from multimodal radiology images (e.g., T1-w or T2-w MRI and CT). Based on the impressive ability of semantic understanding and rich knowledge of LLMs. This process emphasizes extracting special features from different modalities and reunion the information for the ultimate clinical diagnostic. With generated text instruction, our proposed union segmentation framework can handle multimodal segmentation without prior collected vision-language datasets. To evaluate our proposed method, we conduct comprehensive experiments with influential baselines, the statistical results and the visualized case study demonstrate the superiority of our novel method.}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07303v1">Modeling Response Consistency in Multi-Agent LLM Systems: A Comparative Analysis of Shared and Separate Context Approaches</a></div>
    <div class="paper-meta">
      📅 2025-04-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly utilized in multi-agent systems (MAS) to enhance collaborative problem-solving and interactive reasoning. Recent advancements have enabled LLMs to function as autonomous agents capable of understanding complex interactions across multiple topics. However, deploying LLMs in MAS introduces challenges related to context management, response consistency, and scalability, especially when agents must operate under memory limitations and handle noisy inputs. While prior research has explored optimizing context sharing and response latency in LLM-driven MAS, these efforts often focus on either fully centralized or decentralized configurations, each with distinct trade-offs. In this paper, we develop a probabilistic framework to analyze the impact of shared versus separate context configurations on response consistency and response times in LLM-based MAS. We introduce the Response Consistency Index (RCI) as a metric to evaluate the effects of context limitations, noise, and inter-agent dependencies on system performance. Our approach differs from existing research by focusing on the interplay between memory constraints and noise management, providing insights into optimizing scalability and response times in environments with interdependent topics. Through this analysis, we offer a comprehensive understanding of how different configurations impact the efficiency of LLM-driven multi-agent systems, thereby guiding the design of more robust architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.21934v3">Proof or Bluff? Evaluating LLMs on 2025 USA Math Olympiad</a></div>
    <div class="paper-meta">
      📅 2025-04-09
    </div>
    <details class="paper-abstract">
      Recent math benchmarks for large language models (LLMs) such as MathArena indicate that state-of-the-art reasoning models achieve impressive performance on mathematical competitions like AIME, with the leading model, Gemini-2.5-Pro, achieving scores comparable to top human competitors. However, these benchmarks evaluate models solely based on final numerical answers, neglecting rigorous reasoning and proof generation which are essential for real-world mathematical tasks. To address this, we introduce the first comprehensive evaluation of full-solution reasoning for challenging mathematical problems. Using expert human annotators, we evaluated several state-of-the-art reasoning models on the six problems from the 2025 USAMO within hours of their release. Our results reveal that all tested models struggled significantly: only Gemini-2.5-Pro achieves a non-trivial score of 25%, while all other models achieve less than 5%. Through detailed analysis of reasoning traces, we identify the most common failure modes and find several unwanted artifacts arising from the optimization strategies employed during model training. Overall, our results suggest that current LLMs are inadequate for rigorous mathematical reasoning tasks, highlighting the need for substantial improvements in reasoning and proof generation capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07278v1">A Multi-Phase Analysis of Blood Culture Stewardship: Machine Learning Prediction, Expert Recommendation Assessment, and LLM Automation</a></div>
    <div class="paper-meta">
      📅 2025-04-09
      | 💬 10 pages, 2 figures, 2 tables, conference
    </div>
    <details class="paper-abstract">
      Blood cultures are often over ordered without clear justification, straining healthcare resources and contributing to inappropriate antibiotic use pressures worsened by the global shortage. In study of 135483 emergency department (ED) blood culture orders, we developed machine learning (ML) models to predict the risk of bacteremia using structured electronic health record (EHR) data and provider notes via a large language model (LLM). The structured models AUC improved from 0.76 to 0.79 with note embeddings and reached 0.81 with added diagnosis codes. Compared to an expert recommendation framework applied by human reviewers and an LLM-based pipeline, our ML approach offered higher specificity without compromising sensitivity. The recommendation framework achieved sensitivity 86%, specificity 57%, while the LLM maintained high sensitivity (96%) but over classified negatives, reducing specificity (16%). These findings demonstrate that ML models integrating structured and unstructured data can outperform consensus recommendations, enhancing diagnostic stewardship beyond existing standards of care.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07199v1">SemEval-2025 Task 5: LLMs4Subjects -- LLM-based Automated Subject Tagging for a National Technical Library's Open-Access Catalog</a></div>
    <div class="paper-meta">
      📅 2025-04-09
      | 💬 10 pages, 4 figures, Accepted as SemEval 2025 Task 5 description paper
    </div>
    <details class="paper-abstract">
      We present SemEval-2025 Task 5: LLMs4Subjects, a shared task on automated subject tagging for scientific and technical records in English and German using the GND taxonomy. Participants developed LLM-based systems to recommend top-k subjects, evaluated through quantitative metrics (precision, recall, F1-score) and qualitative assessments by subject specialists. Results highlight the effectiveness of LLM ensembles, synthetic data generation, and multilingual processing, offering insights into applying LLMs for digital library classification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13453v3">Adaptive Augmentation Policy Optimization with LLM Feedback</a></div>
    <div class="paper-meta">
      📅 2025-04-09
      | 💬 15 pages, 4 tables, 3 figures submitted for consideration to 2025 Medical Image Understanding and Analysis Conference (MIUA)
    </div>
    <details class="paper-abstract">
      Data augmentation is a critical component of deep learning pipelines, enhancing model generalization by increasing dataset diversity. Traditional augmentation strategies rely on manually designed transformations, stochastic sampling, or automated search-based approaches. Although automated methods improve performance, they often require extensive computational resources and are tailored to specific datasets. In this work, we propose a Large Language Model (LLM)-guided augmentation optimization strategy that refines augmentation policies based on model performance feedback. We introduce two approaches: (1) LLM-Guided Augmentation Policy Optimization, where augmentation policies are selected by an LLM prior to training and iteratively refined across multiple training cycles, and (2) Adaptive LLM-Guided Augmentation Policy Optimization, where policies adapt in real-time based on performance metrics. This in-training approach eliminates the need for full model retraining before receiving LLM feedback, thereby reducing computational costs while improving performance. Our methodology employs an LLM to dynamically select augmentation transformations based on dataset characteristics, model architecture, and prior training outcomes. Unlike traditional search-based methods, our approach leverages the contextual knowledge of LLMs, particularly in specialized domains like medical imaging, to recommend augmentation strategies tailored to domain-specific data. We evaluate our approach on multiple domain-specific image classification datasets where augmentation is key to model robustness. Results show that LLM-guided augmentation optimization outperforms traditional methods, improving model accuracy. These findings highlight the potential of LLMs in automating and adapting deep learning training workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08002v1">More diverse more adaptive: Comprehensive Multi-task Learning for Improved LLM Domain Adaptation in E-commerce</a></div>
    <div class="paper-meta">
      📅 2025-04-09
      | 💬 Accepted by KDD workshop 2024
    </div>
    <details class="paper-abstract">
      In recent years, Large Language Models (LLMs) have been widely applied across various domains due to their powerful domain adaptation capabilities. Previous studies have suggested that diverse, multi-modal data can enhance LLMs' domain adaptation performance. However, this hypothesis remains insufficiently validated in the e-commerce sector. To address this gap, we propose a comprehensive e-commerce multi-task framework and design empirical experiments to examine the impact of diverse data and tasks on LLMs from two perspectives: "capability comprehensiveness" and "task comprehensiveness." Specifically, we observe significant improvements in LLM performance by progressively introducing tasks related to new major capability areas and by continuously adding subtasks within different major capability domains. Furthermore, we observe that increasing model capacity amplifies the benefits of diversity, suggesting a synergistic relationship between model capacity and data diversity. Finally, we validate the best-performing model from our empirical experiments in the KDD Cup 2024, achieving a rank 5 in Task 1. This outcome demonstrates the significance of our research for advancing LLMs in the e-commerce domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08820v1">CAReDiO: Cultural Alignment of LLM via Representativeness and Distinctiveness Guided Data Optimization</a></div>
    <div class="paper-meta">
      📅 2025-04-09
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) more deeply integrate into human life across various regions, aligning them with pluralistic cultures is crucial for improving user experience and mitigating cultural conflicts. Existing approaches develop culturally aligned LLMs primarily through fine-tuning with massive carefully curated culture-specific corpora. Nevertheless, inspired by culture theories, we identify two key challenges faced by these datasets: (1) Representativeness: These corpora fail to fully capture the target culture's core characteristics with redundancy, causing computation waste; (2) Distinctiveness: They struggle to distinguish the unique nuances of a given culture from shared patterns across other relevant ones, hindering precise cultural modeling. To handle these challenges, we introduce CAReDiO, a novel cultural data construction framework. Specifically, CAReDiO utilizes powerful LLMs to automatically generate cultural conversation data, where both the queries and responses are further optimized by maximizing representativeness and distinctiveness. Using CAReDiO, we construct a small yet effective dataset, covering five cultures, and compare it with several recent cultural corpora. Extensive experiments demonstrate that our method generates more effective data and enables cultural alignment with as few as 100 training samples, enhancing both performance and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08808v1">Exploring the Effectiveness and Interpretability of Texts in LLM-based Time Series Models</a></div>
    <div class="paper-meta">
      📅 2025-04-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have been applied to time series forecasting tasks, leveraging pre-trained language models as the backbone and incorporating textual data to purportedly enhance the comprehensive capabilities of LLMs for time series. However, are these texts really helpful for interpretation? This study seeks to investigate the actual efficacy and interpretability of such textual incorporations. Through a series of empirical experiments on textual prompts and textual prototypes, our findings reveal that the misalignment between two modalities exists, and the textual information does not significantly improve time series forecasting performance in many cases. Furthermore, visualization analysis indicates that the textual representations learned by existing frameworks lack sufficient interpretability when applied to time series data. We further propose a novel metric named Semantic Matching Index (SMI) to better evaluate the matching degree between time series and texts during our post hoc interpretability investigation. Our analysis reveals the misalignment and limited interpretability of texts in current time-series LLMs, and we hope this study can raise awareness of the interpretability of texts for time series. The code is available at https://github.com/zachysun/TS-Lang-Exp.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10509v1">Beyond Reproducibility: Advancing Zero-shot LLM Reranking Efficiency with Setwise Insertion</a></div>
    <div class="paper-meta">
      📅 2025-04-09
    </div>
    <details class="paper-abstract">
      This study presents a comprehensive reproducibility and extension analysis of the Setwise prompting methodology for zero-shot ranking with Large Language Models (LLMs), as proposed by Zhuang et al. We evaluate its effectiveness and efficiency compared to traditional Pointwise, Pairwise, and Listwise approaches in document ranking tasks. Our reproduction confirms the findings of Zhuang et al., highlighting the trade-offs between computational efficiency and ranking effectiveness in Setwise methods. Building on these insights, we introduce Setwise Insertion, a novel approach that leverages the initial document ranking as prior knowledge, reducing unnecessary comparisons and uncertainty by focusing on candidates more likely to improve the ranking results. Experimental results across multiple LLM architectures (Flan-T5, Vicuna, and Llama) show that Setwise Insertion yields a 31% reduction in query time, a 23% reduction in model inferences, and a slight improvement in reranking effectiveness compared to the original Setwise method. These findings highlight the practical advantage of incorporating prior ranking knowledge into Setwise prompting for efficient and accurate zero-shot document reranking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07097v1">Sculpting Subspaces: Constrained Full Fine-Tuning in LLMs for Continual Learning</a></div>
    <div class="paper-meta">
      📅 2025-04-09
      | 💬 25 pages, 13 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Continual learning in large language models (LLMs) is prone to catastrophic forgetting, where adapting to new tasks significantly degrades performance on previously learned ones. Existing methods typically rely on low-rank, parameter-efficient updates that limit the model's expressivity and introduce additional parameters per task, leading to scalability issues. To address these limitations, we propose a novel continual full fine-tuning approach leveraging adaptive singular value decomposition (SVD). Our method dynamically identifies task-specific low-rank parameter subspaces and constrains updates to be orthogonal to critical directions associated with prior tasks, thus effectively minimizing interference without additional parameter overhead or storing previous task gradients. We evaluate our approach extensively on standard continual learning benchmarks using both encoder-decoder (T5-Large) and decoder-only (LLaMA-2 7B) models, spanning diverse tasks including classification, generation, and reasoning. Empirically, our method achieves state-of-the-art results, up to 7% higher average accuracy than recent baselines like O-LoRA, and notably maintains the model's general linguistic capabilities, instruction-following accuracy, and safety throughout the continual learning process by reducing forgetting to near-negligible levels. Our adaptive SVD framework effectively balances model plasticity and knowledge retention, providing a practical, theoretically grounded, and computationally scalable solution for continual learning scenarios in large language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07087v1">KG-LLM-Bench: A Scalable Benchmark for Evaluating LLM Reasoning on Textualized Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2025-04-09
      | 💬 To be presented at NAACL-HLT, KnowledgeNLP Workshop (2025)
    </div>
    <details class="paper-abstract">
      Knowledge graphs have emerged as a popular method for injecting up-to-date, factual knowledge into large language models (LLMs). This is typically achieved by converting the knowledge graph into text that the LLM can process in context. While multiple methods of encoding knowledge graphs have been proposed, the impact of this textualization process on LLM performance remains under-explored. We introduce KG-LLM-Bench, a comprehensive and extensible benchmark spanning five knowledge graph understanding tasks, and evaluate how different encoding strategies affect performance across various base models. Our extensive experiments with seven language models and five textualization strategies provide insights for optimizing LLM performance on KG reasoning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06426v1">S'MoRE: Structural Mixture of Residual Experts for LLM Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2025-04-08
    </div>
    <details class="paper-abstract">
      Fine-tuning pre-trained large language models (LLMs) presents a dual challenge of balancing parameter efficiency and model capacity. Existing methods like low-rank adaptations (LoRA) are efficient but lack flexibility, while Mixture-of-Experts (MoE) architectures enhance model capacity at the cost of more & under-utilized parameters. To address these limitations, we propose Structural Mixture of Residual Experts (S'MoRE), a novel framework that seamlessly integrates the efficiency of LoRA with the flexibility of MoE. Specifically, S'MoRE employs hierarchical low-rank decomposition of expert weights, yielding residuals of varying orders interconnected in a multi-layer structure. By routing input tokens through sub-trees of residuals, S'MoRE emulates the capacity of many experts by instantiating and assembling just a few low-rank matrices. We craft the inter-layer propagation of S'MoRE's residuals as a special type of Graph Neural Network (GNN), and prove that under similar parameter budget, S'MoRE improves "structural flexibility" of traditional MoE (or Mixture-of-LoRA) by exponential order. Comprehensive theoretical analysis and empirical results demonstrate that S'MoRE achieves superior fine-tuning performance, offering a transformative approach for efficient LLM adaptation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.03038v3">Towards Federated RLHF with Aggregated Client Preference for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-08
      | 💬 ICLR'25
    </div>
    <details class="paper-abstract">
      Reinforcement learning with human feedback (RLHF) fine-tunes a pretrained large language model (LLM) using user preference data, enabling it to generate content aligned with human preferences. However, due to privacy concerns, users may be reluctant to share sensitive preference data. To address this, we propose utilizing Federated Learning (FL) techniques, allowing large-scale preference collection from diverse real-world users without requiring them to transmit data to a central server. Our federated RLHF methods (i.e., FedBis and FedBiscuit) encode each client's preferences into binary selectors and aggregate them to capture common preferences. In particular, FedBiscuit overcomes key challenges, such as preference heterogeneity and reward hacking, through innovative solutions like grouping clients with similar preferences to reduce heterogeneity and using multiple binary selectors to enhance LLM output quality. To evaluate the performance of the proposed methods, we establish the first federated RLHF benchmark with a heterogeneous human preference dataset. Experimental results show that by integrating the LLM with aggregated client preferences, FedBis and FedBiscuit significantly enhance the professionalism and readability of the generated content.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06356v1">Query Understanding in LLM-based Conversational Information Seeking</a></div>
    <div class="paper-meta">
      📅 2025-04-08
      | 💬 WWW'25 Tutorial
    </div>
    <details class="paper-abstract">
      Query understanding in Conversational Information Seeking (CIS) involves accurately interpreting user intent through context-aware interactions. This includes resolving ambiguities, refining queries, and adapting to evolving information needs. Large Language Models (LLMs) enhance this process by interpreting nuanced language and adapting dynamically, improving the relevance and precision of search results in real-time. In this tutorial, we explore advanced techniques to enhance query understanding in LLM-based CIS systems. We delve into LLM-driven methods for developing robust evaluation metrics to assess query understanding quality in multi-turn interactions, strategies for building more interactive systems, and applications like proactive query management and query reformulation. We also discuss key challenges in integrating LLMs for query understanding in conversational search systems and outline future research directions. Our goal is to deepen the audience's understanding of LLM-based conversational query understanding and inspire discussions to drive ongoing advancements in this field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06324v1">From Stability to Inconsistency: A Study of Moral Preferences in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-08
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) increasingly integrate into our daily lives, it becomes crucial to understand their implicit biases and moral tendencies. To address this, we introduce a Moral Foundations LLM dataset (MFD-LLM) grounded in Moral Foundations Theory, which conceptualizes human morality through six core foundations. We propose a novel evaluation method that captures the full spectrum of LLMs' revealed moral preferences by answering a range of real-world moral dilemmas. Our findings reveal that state-of-the-art models have remarkably homogeneous value preferences, yet demonstrate a lack of consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06323v1">Mosaic: Composite Projection Pruning for Resource-efficient LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-08
    </div>
    <details class="paper-abstract">
      Extensive compute and memory requirements limit the deployment of large language models (LLMs) on any hardware. Compression methods, such as pruning, can reduce model size, which in turn reduces resource requirements. State-of-the-art pruning is based on coarse-grained methods. They are time-consuming and inherently remove critical model parameters, adversely impacting the quality of the pruned model. This paper introduces projection pruning, a novel fine-grained method for pruning LLMs. In addition, LLM projection pruning is enhanced by a new approach we refer to as composite projection pruning - the synergistic combination of unstructured pruning that retains accuracy and structured pruning that reduces model size. We develop Mosaic, a novel system to create and deploy pruned LLMs using composite projection pruning. Mosaic is evaluated using a range of performance and quality metrics on multiple hardware platforms, LLMs, and datasets. Mosaic is 7.19x faster in producing models than existing approaches. Mosaic models achieve up to 84.2% lower perplexity and 31.4% higher accuracy than models obtained from coarse-grained pruning. Up to 67% faster inference and 68% lower GPU memory use is noted for Mosaic models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06319v1">Accelerating LLM Inference Throughput via Asynchronous KV Cache Prefetching</a></div>
    <div class="paper-meta">
      📅 2025-04-08
      | 💬 8 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit pronounced memory-bound characteristics during inference due to High Bandwidth Memory (HBM) bandwidth constraints. In this paper, we propose an L2 Cache-oriented asynchronous KV Cache prefetching method to break through the memory bandwidth bottleneck in LLM inference through computation-load overlap. By strategically scheduling idle memory bandwidth during active computation windows, our method proactively prefetches required KV Cache into GPU L2 cache, enabling high-speed L2 cache hits for subsequent accesses and effectively hiding HBM access latency within computational cycles. Extensive experiments on NVIDIA H20 GPUs demonstrate that the proposed method achieves 2.15x improvement in attention kernel efficiency and up to 1.97x end-to-end throughput enhancement, surpassing state-of-the-art baseline FlashAttention-3. Notably, our solution maintains orthogonality to existing optimization techniques and can be integrated with current inference frameworks, providing a scalable latency-hiding solution for next-generation LLM inference engines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06265v1">GOLLuM: Gaussian Process Optimized LLMs -- Reframing LLM Finetuning through Bayesian Optimization</a></div>
    <div class="paper-meta">
      📅 2025-04-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can encode complex relationships in their latent spaces, yet harnessing them for optimization under uncertainty remains challenging. We address this gap with a novel architecture that reframes LLM finetuning as Gaussian process (GP) marginal likelihood optimization via deep kernel methods. We introduce LLM-based deep kernels, jointly optimized with GPs to preserve the benefits of both - LLMs to provide a rich and flexible input space for Bayesian optimization and - GPs to model this space with predictive uncertainty for more efficient sampling. Applied to Buchwald-Hartwig reaction optimization, our method nearly doubles the discovery rate of high-performing reactions compared to static LLM embeddings (from 24% to 43% coverage of the top 5% reactions in just 50 optimization iterations). We also observe a 14% improvement over domain-specific representations without requiring specialized features. Extensive empirical evaluation across 19 benchmarks - ranging from general chemistry to reaction and molecular property optimization - demonstrates our method's robustness, generality, and consistent improvements across: (1) tasks, (2) LLM architectures (encoder, decoder, encoder-decoder), (3) pretraining domains (chemistry-related or general-purpose) and (4) hyperparameter settings (tuned once on a single dataset). Finally, we explain these improvements: joint LLM-GP optimization through marginal likelihood implicitly performs contrastive learning, aligning representations to produce (1) better-structured embedding spaces, (2) improved uncertainty calibration, and (3) more efficient sampling - without requiring any external loss. This work provides both practical advances in sample-efficient optimization and insights into what makes effective Bayesian optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06261v1">Hogwild! Inference: Parallel LLM Generation via Concurrent Attention</a></div>
    <div class="paper-meta">
      📅 2025-04-08
      | 💬 Preprint, work in progress
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated the ability to tackle increasingly complex tasks through advanced reasoning, long-form content generation, and tool use. Solving these tasks often involves long inference-time computations. In human problem solving, a common strategy to expedite work is collaboration: by dividing the problem into sub-tasks, exploring different strategies concurrently, etc. Recent research has shown that LLMs can also operate in parallel by implementing explicit cooperation frameworks, such as voting mechanisms or the explicit creation of independent sub-tasks that can be executed in parallel. However, each of these frameworks may not be suitable for all types of tasks, which can hinder their applicability. In this work, we propose a different design approach: we run LLM "workers" in parallel , allowing them to synchronize via a concurrently-updated attention cache and prompt these workers to decide how best to collaborate. Our approach allows the instances to come up with their own collaboration strategy for the problem at hand, all the while "seeing" each other's partial progress in the concurrent cache. We implement this approach via Hogwild! Inference: a parallel LLM inference engine where multiple instances of the same LLM run in parallel with the same attention cache, with "instant" access to each other's generated tokens. Hogwild! inference takes advantage of Rotary Position Embeddings (RoPE) to avoid recomputation while improving parallel hardware utilization. We find that modern reasoning-capable LLMs can perform inference with shared Key-Value cache out of the box, without additional fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.22250v2">Modeling Challenging Patient Interactions: LLMs for Medical Communication Training</a></div>
    <div class="paper-meta">
      📅 2025-04-08
    </div>
    <details class="paper-abstract">
      Effective patient communication is pivotal in healthcare, yet traditional medical training often lacks exposure to diverse, challenging interpersonal dynamics. To bridge this gap, this study proposes the use of Large Language Models (LLMs) to simulate authentic patient communication styles, specifically the "accuser" and "rationalizer" personas derived from the Satir model, while also ensuring multilingual applicability to accommodate diverse cultural contexts and enhance accessibility for medical professionals. Leveraging advanced prompt engineering, including behavioral prompts, author's notes, and stubbornness mechanisms, we developed virtual patients (VPs) that embody nuanced emotional and conversational traits. Medical professionals evaluated these VPs, rating their authenticity (accuser: $3.8 \pm 1.0$; rationalizer: $3.7 \pm 0.8$ on a 5-point Likert scale (from one to five)) and correctly identifying their styles. Emotion analysis revealed distinct profiles: the accuser exhibited pain, anger, and distress, while the rationalizer displayed contemplation and calmness, aligning with predefined, detailed patient description including medical history. Sentiment scores (on a scale from zero to nine) further validated these differences in the communication styles, with the accuser adopting negative ($3.1 \pm 0.6$) and the rationalizer more neutral ($4.0 \pm 0.4$) tone. These results underscore LLMs' capability to replicate complex communication styles, offering transformative potential for medical education. This approach equips trainees to navigate challenging clinical scenarios by providing realistic, adaptable patient interactions, enhancing empathy and diagnostic acumen. Our findings advocate for AI-driven tools as scalable, cost-effective solutions to cultivate nuanced communication skills, setting a foundation for future innovations in healthcare training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02879v2">Position: LLM Unlearning Benchmarks are Weak Measures of Progress</a></div>
    <div class="paper-meta">
      📅 2025-04-08
      | 💬 Appears in IEEE Secure and Trustworthy Machine Learning (SaTML) '25
    </div>
    <details class="paper-abstract">
      Unlearning methods have the potential to improve the privacy and safety of large language models (LLMs) by removing sensitive or harmful information post hoc. The LLM unlearning research community has increasingly turned toward empirical benchmarks to assess the effectiveness of such methods. In this paper, we find that existing benchmarks provide an overly optimistic and potentially misleading view on the effectiveness of candidate unlearning methods. By introducing simple, benign modifications to a number of popular benchmarks, we expose instances where supposedly unlearned information remains accessible, or where the unlearning process has degraded the model's performance on retained information to a much greater extent than indicated by the original benchmark. We identify that existing benchmarks are particularly vulnerable to modifications that introduce even loose dependencies between the forget and retain information. Further, we show that ambiguity in unlearning targets in existing benchmarks can easily lead to the design of methods that overfit to the given test queries. Based on our findings, we urge the community to be cautious when interpreting benchmark results as reliable measures of progress, and we provide several recommendations to guide future LLM unlearning research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.15341v3">GenoTEX: An LLM Agent Benchmark for Automated Gene Expression Data Analysis</a></div>
    <div class="paper-meta">
      📅 2025-04-08
      | 💬 31 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Recent advancements in machine learning have significantly improved the identification of disease-associated genes from gene expression datasets. However, these processes often require extensive expertise and manual effort, limiting their scalability. Large Language Model (LLM)-based agents have shown promise in automating these tasks due to their increasing problem-solving abilities. To support the evaluation and development of such methods, we introduce GenoTEX, a benchmark dataset for the automated analysis of gene expression data. GenoTEX provides analysis code and results for solving a wide range of gene-trait association problems, encompassing dataset selection, preprocessing, and statistical analysis, in a pipeline that follows computational genomics standards. The benchmark includes expert-curated annotations from bioinformaticians to ensure accuracy and reliability. To provide baselines for these tasks, we present GenoAgent, a team of LLM-based agents that adopt a multi-step programming workflow with flexible self-correction, to collaboratively analyze gene expression datasets. Our experiments demonstrate the potential of LLM-based methods in analyzing genomic data, while error analysis highlights the challenges and areas for future improvement. We propose GenoTEX as a promising resource for benchmarking and enhancing automated methods for gene expression data analysis. The benchmark is available at https://github.com/Liu-Hy/GenoTEX.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06219v1">Can Performant LLMs Be Ethical? Quantifying the Impact of Web Crawling Opt-Outs</a></div>
    <div class="paper-meta">
      📅 2025-04-08
    </div>
    <details class="paper-abstract">
      The increasing adoption of web crawling opt-outs by copyright holders of online content raises critical questions about the impact of data compliance on large language model (LLM) performance. However, little is known about how these restrictions (and the resultant filtering of pretraining datasets) affect the capabilities of models trained using these corpora. In this work, we conceptualize this effect as the $\textit{data compliance gap}$ (DCG), which quantifies the performance difference between models trained on datasets that comply with web crawling opt-outs, and those that do not. We measure the data compliance gap in two settings: pretraining models from scratch and continual pretraining from existing compliant models (simulating a setting where copyrighted data could be integrated later in pretraining). Our experiments with 1.5B models show that, as of January 2025, compliance with web data opt-outs does not degrade general knowledge acquisition (close to 0\% DCG). However, in specialized domains such as biomedical research, excluding major publishers leads to performance declines. These findings suggest that while general-purpose LLMs can be trained to perform equally well using fully open data, performance in specialized domains may benefit from access to high-quality copyrighted sources later in training. Our study provides empirical insights into the long-debated trade-off between data compliance and downstream model performance, informing future discussions on AI training practices and policy decisions.
    </details>
</div>
