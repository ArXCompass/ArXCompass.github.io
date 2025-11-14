# llm - 2024_12

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- Part 2
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18174v1">INVESTORBENCH: A Benchmark for Financial Decision-Making Tasks with LLM-based Agent</a></div>
    <div class="paper-meta">
      📅 2024-12-24
    </div>
    <details class="paper-abstract">
      Recent advancements have underscored the potential of large language model (LLM)-based agents in financial decision-making. Despite this progress, the field currently encounters two main challenges: (1) the lack of a comprehensive LLM agent framework adaptable to a variety of financial tasks, and (2) the absence of standardized benchmarks and consistent datasets for assessing agent performance. To tackle these issues, we introduce \textsc{InvestorBench}, the first benchmark specifically designed for evaluating LLM-based agents in diverse financial decision-making contexts. InvestorBench enhances the versatility of LLM-enabled agents by providing a comprehensive suite of tasks applicable to different financial products, including single equities like stocks, cryptocurrencies and exchange-traded funds (ETFs). Additionally, we assess the reasoning and decision-making capabilities of our agent framework using thirteen different LLMs as backbone models, across various market environments and tasks. Furthermore, we have curated a diverse collection of open-source, multi-modal datasets and developed a comprehensive suite of environments for financial decision-making. This establishes a highly accessible platform for evaluating financial agents' performance across various scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.15268v2">Enhancing LLM-based Hatred and Toxicity Detection with Meta-Toxic Knowledge Graph</a></div>
    <div class="paper-meta">
      📅 2024-12-24
      | 💬 8 pages of content
    </div>
    <details class="paper-abstract">
      The rapid growth of social media platforms has raised significant concerns regarding online content toxicity. When Large Language Models (LLMs) are used for toxicity detection, two key challenges emerge: 1) the absence of domain-specific toxic knowledge leads to false negatives; 2) the excessive sensitivity of LLMs to toxic speech results in false positives, limiting freedom of speech. To address these issues, we propose a novel method called MetaTox, leveraging graph search on a meta-toxic knowledge graph to enhance hatred and toxicity detection. First, we construct a comprehensive meta-toxic knowledge graph by utilizing LLMs to extract toxic information through a three-step pipeline, with toxic benchmark datasets serving as corpora. Second, we query the graph via retrieval and ranking processes to supplement accurate, relevant toxic knowledge. Extensive experiments and in-depth case studies across multiple datasets demonstrate that our MetaTox significantly decreases the false positive rate while boosting overall toxicity detection performance. Our code will be available soon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.11934v2">Stepwise Reasoning Error Disruption Attack of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-24
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have made remarkable strides in complex reasoning tasks, but their safety and robustness in reasoning processes remain underexplored. Existing attacks on LLM reasoning are constrained by specific settings or lack of imperceptibility, limiting their feasibility and generalizability. To address these challenges, we propose the Stepwise rEasoning Error Disruption (SEED) attack, which subtly injects errors into prior reasoning steps to mislead the model into producing incorrect subsequent reasoning and final answers. Unlike previous methods, SEED is compatible with zero-shot and few-shot settings, maintains the natural reasoning flow, and ensures covert execution without modifying the instruction. Extensive experiments on four datasets across four different models demonstrate SEED's effectiveness, revealing the vulnerabilities of LLMs to disruptions in reasoning processes. These findings underscore the need for greater attention to the robustness of LLM reasoning to ensure safety in practical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.00852v2">EF-LLM: Energy Forecasting LLM with AI-assisted Automation, Enhanced Sparse Prediction, Hallucination Detection</a></div>
    <div class="paper-meta">
      📅 2024-12-24
    </div>
    <details class="paper-abstract">
      Accurate prediction helps to achieve supply-demand balance in energy systems, supporting decision-making and scheduling. Traditional models, lacking AI-assisted automation, rely on experts, incur high costs, and struggle with sparse data prediction. To address these challenges, we propose the Energy Forecasting Large Language Model (EF-LLM), which integrates domain knowledge and temporal data for time-series forecasting, supporting both pre-forecast operations and post-forecast decision-support. EF-LLM's human-AI interaction capabilities lower the entry barrier in forecasting tasks, reducing the need for extra expert involvement. To achieve this, we propose a continual learning approach with updatable LoRA and a multi-channel architecture for aligning heterogeneous multimodal data, enabling EF-LLM to continually learn heterogeneous multimodal knowledge. In addition, EF-LLM enables accurate predictions under sparse data conditions through its ability to process multimodal data. We propose Fusion Parameter-Efficient Fine-Tuning (F-PEFT) method to effectively leverage both time-series data and text for this purpose. EF-LLM is also the first energy-specific LLM to detect hallucinations and quantify their occurrence rate, achieved via multi-task learning, semantic similarity analysis, and ANOVA. We have achieved success in energy prediction scenarios for load, photovoltaic, and wind power forecast.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18106v1">Tackling the Dynamicity in a Production LLM Serving System with SOTA Optimizations via Hybrid Prefill/Decode/Verify Scheduling on Efficient Meta-kernels</a></div>
    <div class="paper-meta">
      📅 2024-12-24
    </div>
    <details class="paper-abstract">
      Meeting growing demands for low latency and cost efficiency in production-grade large language model (LLM) serving systems requires integrating advanced optimization techniques. However, dynamic and unpredictable input-output lengths of LLM, compounded by these optimizations, exacerbate the issues of workload variability, making it difficult to maintain high efficiency on AI accelerators, especially DSAs with tile-based programming models. To address this challenge, we introduce XY-Serve, a versatile, Ascend native, end-to-end production LLM-serving system. The core idea is an abstraction mechanism that smooths out the workload variability by decomposing computations into unified, hardware-friendly, fine-grained meta primitives. For attention, we propose a meta-kernel that computes the basic pattern of matmul-softmax-matmul with architectural-aware tile sizes. For GEMM, we introduce a virtual padding scheme that adapts to dynamic shape changes while using highly efficient GEMM primitives with assorted fixed tile sizes. XY-Serve sits harmoniously with vLLM. Experimental results show up to 89% end-to-end throughput improvement compared with current publicly available baselines on Ascend NPUs. Additionally, our approach outperforms existing GEMM (average 14.6% faster) and attention (average 21.5% faster) kernels relative to existing libraries. While the work is Ascend native, we believe the approach can be readily applicable to SIMT architectures as well.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18100v1">EvoPat: A Multi-LLM-based Patents Summarization and Analysis Agent</a></div>
    <div class="paper-meta">
      📅 2024-12-24
      | 💬 15 pages,2 figures,8 tables
    </div>
    <details class="paper-abstract">
      The rapid growth of scientific techniques and knowledge is reflected in the exponential increase in new patents filed annually. While these patents drive innovation, they also present significant burden for researchers and engineers, especially newcomers. To avoid the tedious work of navigating a vast and complex landscape to identify trends and breakthroughs, researchers urgently need efficient tools to summarize, evaluate, and contextualize patents, revealing their innovative contributions and underlying scientific principles.To address this need, we present EvoPat, a multi-LLM-based patent agent designed to assist users in analyzing patents through Retrieval-Augmented Generation (RAG) and advanced search strategies. EvoPat leverages multiple Large Language Models (LLMs), each performing specialized roles such as planning, identifying innovations, and conducting comparative evaluations. The system integrates data from local databases, including patents, literature, product catalogous, and company repositories, and online searches to provide up-to-date insights. The ability to collect information not included in original database automatically is also implemented. Through extensive testing in the natural language processing (NLP) domain, we demonstrate that EvoPat outperforms GPT-4 in tasks such as patent summarization, comparative analysis, and technical evaluation. EvoPat represents a significant step toward creating AI-powered tools that empower researchers and engineers to efficiently navigate the complexities of the patent landscape.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.16834v2">Online Learning from Strategic Human Feedback in LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2024-12-24
    </div>
    <details class="paper-abstract">
      Reinforcement learning from human feedback (RLHF) has become an essential step in fine-tuning large language models (LLMs) to align them with human preferences. However, human labelers are selfish and have diverse preferences. They may strategically misreport their online feedback to influence the system's aggregation towards their own preferences. Current practice simply averages labelers' feedback per time and fails to identify the most accurate human labeler, leading to linear regret $\mathcal{O}(T)$ for $T$ time slots. To our best knowledge, we are the first to study online learning mechanisms against strategic human labelers in the LLM fine-tuning process. We formulate a new dynamic Bayesian game and dynamically adjust human labelers' weights in the preference aggregation, ensuring their truthful feedback and sublinear regret $\mathcal{O}(T^{1/2})$. Simulation results demonstrate our mechanism's great advantages over the existing benchmark schemes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10179v2">Is Parameter Collision Hindering Continual Learning in LLMs?</a></div>
    <div class="paper-meta">
      📅 2024-12-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often suffer from catastrophic forgetting when learning multiple tasks sequentially, making continual learning (CL) essential for their dynamic deployment. Existing state-of-the-art (SOTA) methods, such as O-LoRA, typically focus on constructing orthogonality tasks to decouple parameter interdependence from various domains.In this paper, we reveal that building non-collision parameters is a more critical factor in addressing CL challenges. Our theoretical and experimental analyses demonstrate that non-collision parameters can provide better task orthogonality, which is a sufficient but unnecessary condition. Furthermore, knowledge from multiple domains will be preserved in non-collision parameter subspaces, making it more difficult to forget previously seen data. Leveraging this insight, we propose Non-collision Low-Rank Adaptation (N-LoRA), a simple yet effective approach leveraging low collision rates to enhance CL in LLMs. Experimental results on multiple CL benchmarks indicate that N-LoRA achieves superior performance (+2.9), higher task orthogonality (*4.1 times), and lower parameter collision (*58.1 times) than SOTA methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.06024v4">Thinking Assistants: LLM-Based Conversational Assistants that Help Users Think By Asking rather than Answering</a></div>
    <div class="paper-meta">
      📅 2024-12-24
    </div>
    <details class="paper-abstract">
      Many AI systems focus solely on providing solutions or explaining outcomes. However, complex tasks like research and strategic thinking often benefit from a more comprehensive approach to augmenting the thinking process rather than passively getting information. We introduce the concept of "Thinking Assistant", a new genre of assistants that help users improve decision-making with a combination of asking reflection questions based on expert knowledge. Through our lab study (N=80), these Large Language Model (LLM) based Thinking Assistants were better able to guide users to make important decisions, compared with conversational agents that only asked questions, provided advice, or neither. Based on the results, we develop a Thinking Assistant in academic career development, determining research trajectory or developing one's unique research identity, which requires deliberation, reflection and experts' advice accordingly. In a longitudinal deployment with 223 conversations, participants responded positively to approximately 65% of the responses. Our work proposes directions for developing more effective LLM agents. Rather than adhering to the prevailing authoritative approach of generating definitive answers, LLM agents aimed at assisting with cognitive enhancement should prioritize fostering reflection. They should initially provide responses designed to prompt thoughtful consideration through inquiring, followed by offering advice only after gaining a deeper understanding of the user's context and needs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18702v1">CypherBench: Towards Precise Retrieval over Full-scale Modern Knowledge Graphs in the LLM Era</a></div>
    <div class="paper-meta">
      📅 2024-12-24
    </div>
    <details class="paper-abstract">
      Retrieval from graph data is crucial for augmenting large language models (LLM) with both open-domain knowledge and private enterprise data, and it is also a key component in the recent GraphRAG system (edge et al., 2024). Despite decades of research on knowledge graphs and knowledge base question answering, leading LLM frameworks (e.g. Langchain and LlamaIndex) have only minimal support for retrieval from modern encyclopedic knowledge graphs like Wikidata. In this paper, we analyze the root cause and suggest that modern RDF knowledge graphs (e.g. Wikidata, Freebase) are less efficient for LLMs due to overly large schemas that far exceed the typical LLM context window, use of resource identifiers, overlapping relation types and lack of normalization. As a solution, we propose property graph views on top of the underlying RDF graph that can be efficiently queried by LLMs using Cypher. We instantiated this idea on Wikidata and introduced CypherBench, the first benchmark with 11 large-scale, multi-domain property graphs with 7.8 million entities and over 10,000 questions. To achieve this, we tackled several key challenges, including developing an RDF-to-property graph conversion engine, creating a systematic pipeline for text-to-Cypher task generation, and designing new evaluation metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.23884v2">Failure Modes of LLMs for Causal Reasoning on Narratives</a></div>
    <div class="paper-meta">
      📅 2024-12-24
    </div>
    <details class="paper-abstract">
      In this work, we investigate the causal reasoning abilities of large language models (LLMs) through the representative problem of inferring causal relationships from narratives. We find that even state-of-the-art language models rely on unreliable shortcuts, both in terms of the narrative presentation and their parametric knowledge. For example, LLMs tend to determine causal relationships based on the topological ordering of events (i.e., earlier events cause later ones), resulting in lower performance whenever events are not narrated in their exact causal order. Similarly, we demonstrate that LLMs struggle with long-term causal reasoning and often fail when the narratives are long and contain many events. Additionally, we show LLMs appear to rely heavily on their parametric knowledge at the expense of reasoning over the provided narrative. This degrades their abilities whenever the narrative opposes parametric knowledge. We extensively validate these failure modes through carefully controlled synthetic experiments, as well as evaluations on real-world narratives. Finally, we observe that explicitly generating a causal graph generally improves performance while naive chain-of-thought is ineffective. Collectively, our results distill precise failure modes of current state-of-the-art models and can pave the way for future techniques to enhance causal reasoning in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18695v1">TimelyLLM: Segmented LLM Serving System for Time-sensitive Robotic Applications</a></div>
    <div class="paper-meta">
      📅 2024-12-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) such as GPT-4 and Llama3 can already comprehend complex commands and process diverse tasks. This advancement facilitates their application in controlling drones and robots for various tasks. However, existing LLM serving systems typically employ a first-come, first-served (FCFS) batching mechanism, which fails to address the time-sensitive requirements of robotic applications. To address it, this paper proposes a new system named TimelyLLM serving multiple robotic agents with time-sensitive requests. TimelyLLM introduces novel mechanisms of segmented generation and scheduling that optimally leverage redundancy between robot plan generation and execution phases. We report an implementation of TimelyLLM on a widely-used LLM serving framework and evaluate it on a range of robotic applications. Our evaluation shows that TimelyLLM improves the time utility up to 1.97x, and reduces the overall waiting time by 84%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18690v1">AgreeMate: Teaching LLMs to Haggle</a></div>
    <div class="paper-meta">
      📅 2024-12-24
      | 💬 15 pages, 22 figures, 6 tables
    </div>
    <details class="paper-abstract">
      We introduce AgreeMate, a framework for training Large Language Models (LLMs) to perform strategic price negotiations through natural language. We apply recent advances to a negotiation setting where two agents (i.e. buyer or seller) use natural language to bargain on goods using coarse actions. Specifically, we present the performance of Large Language Models when used as agents within a decoupled (modular) bargaining architecture. We demonstrate that using prompt engineering, fine-tuning, and chain-of-thought prompting enhances model performance, as defined by novel metrics. We use attention probing to show model attention to semantic relationships between tokens during negotiations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18588v1">A Paragraph is All It Takes: Rich Robot Behaviors from Interacting, Trusted LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-24
      | 💬 10 pages, 1 figure
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are compact representations of all public knowledge of our physical environment and animal and human behaviors. The application of LLMs to robotics may offer a path to highly capable robots that perform well across most human tasks with limited or even zero tuning. Aside from increasingly sophisticated reasoning and task planning, networks of (suitably designed) LLMs offer ease of upgrading capabilities and allow humans to directly observe the robot's thinking. Here we explore the advantages, limitations, and particularities of using LLMs to control physical robots. The basic system consists of four LLMs communicating via a human language data bus implemented via web sockets and ROS2 message passing. Surprisingly, rich robot behaviors and good performance across different tasks could be achieved despite the robot's data fusion cycle running at only 1Hz and the central data bus running at the extremely limited rates of the human brain, of around 40 bits/s. The use of natural language for inter-LLM communication allowed the robot's reasoning and decision making to be directly observed by humans and made it trivial to bias the system's behavior with sets of rules written in plain English. These rules were immutably written into Ethereum, a global, public, and censorship resistant Turing-complete computer. We suggest that by using natural language as the data bus among interacting AIs, and immutable public ledgers to store behavior constraints, it is possible to build robots that combine unexpectedly rich performance, upgradability, and durable alignment with humans.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18573v1">How Well Do LLMs Generate Code for Different Application Domains? Benchmark and Evaluation</a></div>
    <div class="paper-meta">
      📅 2024-12-24
    </div>
    <details class="paper-abstract">
      Recently, an increasing number of AI-driven programming assistants powered by code LLMs have been integrated into various real-world software development environments, significantly boosting developer productivity. However, existing code generation benchmarks primarily focus on general-purpose scenarios, leaving the code generation performance of LLMs for specific application domains largely unknown. In this paper, we introduce a new benchmark, MultiCodeBench, to fill this gap. MultiCodeBench comprises 2,400 programming tasks, covering 12 popular software development domains and 15 programming languages. Specifically, we perform in-depth research to identify these 12 application domains. Given that each domain may involve multiple technical frameworks, and that different frameworks present distinct challenges in the coding process, we categorize the commonly used frameworks and platforms within each domain. We then sample programming problems from GitHub repositories related to these subdomains. To ensure the quality of the tasks and mitigate data leakage issues, we invite annotators to rewrite the docstrings for each task in MultiCodeBench. Additionally, we build a static analysis-based dependency parsing tool to extract the dependencies in the ground truth for each task, enabling deeper performance analysis. Through extensive experiments on MultiCodeBench with eleven representative mainstream LLMs, we reveal the code generation performance of the LLMs across different application domains, providing practical insights for developers in downstream fields when selecting LLMs. Furthermore, we analyze the reasons behind the models' failures in completing software application development tasks, offering guidance for model developers to enhance domain-specific code generation capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18497v1">Think or Remember? Detecting and Directing LLMs Towards Memorization or Generalization</a></div>
    <div class="paper-meta">
      📅 2024-12-24
    </div>
    <details class="paper-abstract">
      In this paper, we explore the foundational mechanisms of memorization and generalization in Large Language Models (LLMs), inspired by the functional specialization observed in the human brain. Our investigation serves as a case study leveraging specially designed datasets and experimental-scale LLMs to lay the groundwork for understanding these behaviors. Specifically, we aim to first enable LLMs to exhibit both memorization and generalization by training with the designed dataset, then (a) examine whether LLMs exhibit neuron-level spatial differentiation for memorization and generalization, (b) predict these behaviors using model internal representations, and (c) steer the behaviors through inference-time interventions. Our findings reveal that neuron-wise differentiation of memorization and generalization is observable in LLMs, and targeted interventions can successfully direct their behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18051v1">Factuality or Fiction? Benchmarking Modern LLMs on Ambiguous QA with Citations</a></div>
    <div class="paper-meta">
      📅 2024-12-23
    </div>
    <details class="paper-abstract">
      Benchmarking modern large language models (LLMs) on complex and realistic tasks is critical to advancing their development. In this work, we evaluate the factual accuracy and citation performance of state-of-the-art LLMs on the task of Question Answering (QA) in ambiguous settings with source citations. Using three recently published datasets-DisentQA-DupliCite, DisentQA-ParaCite, and AmbigQA-Cite-featuring a range of real-world ambiguities, we analyze the performance of two leading LLMs, GPT-4o-mini and Claude-3.5. Our results show that larger, recent models consistently predict at least one correct answer in ambiguous contexts but fail to handle cases with multiple valid answers. Additionally, all models perform equally poorly in citation generation, with citation accuracy consistently at 0. However, introducing conflict-aware prompting leads to large improvements, enabling models to better address multiple valid answers and improve citation accuracy, while maintaining their ability to predict correct answers. These findings highlight the challenges and opportunities in developing LLMs that can handle ambiguity and provide reliable source citations. Our benchmarking study provides critical insights and sets a foundation for future improvements in trustworthy and interpretable QA systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18022v1">Trustworthy and Efficient LLMs Meet Databases</a></div>
    <div class="paper-meta">
      📅 2024-12-23
    </div>
    <details class="paper-abstract">
      In the rapidly evolving AI era with large language models (LLMs) at the core, making LLMs more trustworthy and efficient, especially in output generation (inference), has gained significant attention. This is to reduce plausible but faulty LLM outputs (a.k.a hallucinations) and meet the highly increased inference demands. This tutorial explores such efforts and makes them transparent to the database community. Understanding these efforts is essential in harnessing LLMs in database tasks and adapting database techniques to LLMs. Furthermore, we delve into the synergy between LLMs and databases, highlighting new opportunities and challenges in their intersection. This tutorial aims to share with database researchers and practitioners essential concepts and strategies around LLMs, reduce the unfamiliarity of LLMs, and inspire joining in the intersection between LLMs and databases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18011v1">StructTest: Benchmarking LLMs' Reasoning through Compositional Structured Outputs</a></div>
    <div class="paper-meta">
      📅 2024-12-23
    </div>
    <details class="paper-abstract">
      The rapid development of large language models (LLMs) necessitates robust, unbiased, and scalable methods for evaluating their capabilities. However, human annotations are expensive to scale, model-based evaluations are prone to biases in answer style, while target-answer-based benchmarks are vulnerable to data contamination and cheating. To address these limitations, we propose StructTest, a novel benchmark that evaluates LLMs on their ability to produce compositionally specified structured outputs as an unbiased, cheap-to-run and difficult-to-cheat measure. The evaluation is done deterministically by a rule-based evaluator, which can be easily extended to new tasks. By testing structured outputs across diverse task domains -- including Summarization, Code, HTML and Math -- we demonstrate that StructTest serves as a good proxy for general reasoning abilities, as producing structured outputs often requires internal logical reasoning. We believe that StructTest offers a critical, complementary approach to objective and robust model evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.13178v2">AgentBoard: An Analytical Evaluation Board of Multi-turn LLM Agents</a></div>
    <div class="paper-meta">
      📅 2024-12-23
      | 💬 NeurIPS 2024 (Oral)
    </div>
    <details class="paper-abstract">
      Evaluating Large Language Models (LLMs) as general-purpose agents is essential for understanding their capabilities and facilitating their integration into practical applications. However, the evaluation process presents substantial challenges. A primary obstacle is the benchmarking of agent performance across diverse scenarios within a unified framework, especially in maintaining partially-observable environments and ensuring multi-round interactions. Moreover, current evaluation frameworks mostly focus on the final success rate, revealing few insights during the process and failing to provide a deep understanding of the model abilities. To address these challenges, we introduce AgentBoard, a pioneering comprehensive benchmark and accompanied open-source evaluation framework tailored to analytical evaluation of LLM agents. AgentBoard offers a fine-grained progress rate metric that captures incremental advancements as well as a comprehensive evaluation toolkit that features easy assessment of agents for multi-faceted analysis. This not only sheds light on the capabilities and limitations of LLM agents but also propels the interpretability of their performance to the forefront. Ultimately, AgentBoard serves as a step towards demystifying agent behaviors and accelerating the development of stronger LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.04108v3">Future Events as Backdoor Triggers: Investigating Temporal Vulnerabilities in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-23
    </div>
    <details class="paper-abstract">
      Backdoors are hidden behaviors that are only triggered once an AI system has been deployed. Bad actors looking to create successful backdoors must design them to avoid activation during training and evaluation. Since data used in these stages often only contains information about events that have already occurred, a component of a simple backdoor trigger could be a model recognizing data that is in the future relative to when it was trained. Through prompting experiments and by probing internal activations, we show that current large language models (LLMs) can distinguish past from future events, with probes on model activations achieving 90% accuracy. We train models with backdoors triggered by a temporal distributional shift; they activate when the model is exposed to news headlines beyond their training cut-off dates. Fine-tuning on helpful, harmless and honest (HHH) data does not work well for removing simpler backdoor triggers but is effective on our backdoored models, although this distinction is smaller for the larger-scale model we tested. We also find that an activation-steering vector representing a model's internal representation of the date influences the rate of backdoor activation. We take these results as initial evidence that, at least for models at the modest scale we test, standard safety measures are enough to remove these backdoors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.14962v2">LLM for Barcodes: Generating Diverse Synthetic Data for Identity Documents</a></div>
    <div class="paper-meta">
      📅 2024-12-23
      | 💬 5 pages, 1 figures
    </div>
    <details class="paper-abstract">
      Accurate barcode detection and decoding in Identity documents is crucial for applications like security, healthcare, and education, where reliable data extraction and verification are essential. However, building robust detection models is challenging due to the lack of diverse, realistic datasets an issue often tied to privacy concerns and the wide variety of document formats. Traditional tools like Faker rely on predefined templates, making them less effective for capturing the complexity of real-world identity documents. In this paper, we introduce a new approach to synthetic data generation that uses LLMs to create contextually rich and realistic data without relying on predefined field. Using the vast knowledge LLMs have about different documents and content, our method creates data that reflects the variety found in real identity documents. This data is then encoded into barcode and overlayed on templates for documents such as Driver's licenses, Insurance cards, Student IDs. Our approach simplifies the process of dataset creation, eliminating the need for extensive domain knowledge or predefined fields. Compared to traditional methods like Faker, data generated by LLM demonstrates greater diversity and contextual relevance, leading to improved performance in barcode detection models. This scalable, privacy-first solution is a big step forward in advancing machine learning for automated document processing and identity verification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.17892v1">LLM-Driven Feedback for Enhancing Conceptual Design Learning in Database Systems Courses</a></div>
    <div class="paper-meta">
      📅 2024-12-23
    </div>
    <details class="paper-abstract">
      The integration of LLM-generated feedback into educational settings has shown promise in enhancing student learning outcomes. This paper presents a novel LLM-driven system that provides targeted feedback for conceptual designs in a Database Systems course. The system converts student-created entity-relationship diagrams (ERDs) into JSON format, allows the student to prune the diagram by isolating a relationship, extracts relevant requirements for the selected relationship, and utilizes a large language model (LLM) to generate detailed feedback. Additionally, the system creates a tailored set of questions and answers to further aid student understanding. Our pilot implementation in a Database System course demonstrates effective feedback generation that helped the students improve their design skills.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.17637v1">SCBench: A Sports Commentary Benchmark for Video LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-23
    </div>
    <details class="paper-abstract">
      Recently, significant advances have been made in Video Large Language Models (Video LLMs) in both academia and industry. However, methods to evaluate and benchmark the performance of different Video LLMs, especially their fine-grained, temporal visual capabilities, remain very limited. On one hand, current benchmarks use relatively simple videos (e.g., subtitled movie clips) where the model can understand the entire video by processing just a few frames. On the other hand, their datasets lack diversity in task format, comprising only QA or multi-choice QA, which overlooks the models' capacity for generating in-depth and precise texts. Sports videos, which feature intricate visual information, sequential events, and emotionally charged commentary, present a critical challenge for Video LLMs, making sports commentary an ideal benchmarking task. Inspired by these challenges, we propose a novel task: sports video commentary generation, developed $\textbf{SCBench}$ for Video LLMs. To construct such a benchmark, we introduce (1) $\textbf{SCORES}$, a six-dimensional metric specifically designed for our task, upon which we propose a GPT-based evaluation method, and (2) $\textbf{CommentarySet}$, a dataset consisting of 5,775 annotated video clips and ground-truth labels tailored to our metric. Based on SCBench, we conduct comprehensive evaluations on multiple Video LLMs (e.g. VILA, Video-LLaVA, etc.) and chain-of-thought baseline methods. Our results found that InternVL-Chat-2 achieves the best performance with 5.44, surpassing the second-best by 1.04. Our work provides a fresh perspective for future research, aiming to enhance models' overall capabilities in complex visual understanding tasks. Our dataset will be released soon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.17626v1">Tracking the Feature Dynamics in LLM Training: A Mechanistic Study</a></div>
    <div class="paper-meta">
      📅 2024-12-23
    </div>
    <details class="paper-abstract">
      Understanding training dynamics and feature evolution is crucial for the mechanistic interpretability of large language models (LLMs). Although sparse autoencoders (SAEs) have been used to identify features within LLMs, a clear picture of how these features evolve during training remains elusive. In this study, we: (1) introduce SAE-Track, a method to efficiently obtain a continual series of SAEs; (2) formulate the process of feature formation and conduct a mechanistic analysis; and (3) analyze and visualize feature drift during training. Our work provides new insights into the dynamics of features in LLMs, enhancing our understanding of training mechanisms and feature evolution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.17593v1">Leveraging Memory Retrieval to Enhance LLM-based Generative Recommendation</a></div>
    <div class="paper-meta">
      📅 2024-12-23
    </div>
    <details class="paper-abstract">
      Leveraging Large Language Models (LLMs) to harness user-item interaction histories for item generation has emerged as a promising paradigm in generative recommendation. However, the limited context window of LLMs often restricts them to focusing on recent user interactions only, leading to the neglect of long-term interests involved in the longer histories. To address this challenge, we propose a novel Automatic Memory-Retrieval framework (AutoMR), which is capable of storing long-term interests in the memory and extracting relevant information from it for next-item generation within LLMs. Extensive experimental results on two real-world datasets demonstrate the effectiveness of our proposed AutoMR framework in utilizing long-term interests for generative recommendation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12394v1">The ELEVATE-AI LLMs Framework: An Evaluation Framework for Use of Large Language Models in HEOR: an ISPOR Working Group Report</a></div>
    <div class="paper-meta">
      📅 2024-12-23
      | 💬 4 Tables, 1 Figure, Supplemental Material
    </div>
    <details class="paper-abstract">
      Introduction. Generative Artificial Intelligence, particularly large language models (LLMs), offers transformative potential for Health Economics and Outcomes Research (HEOR). However, evaluating the quality, transparency, and rigor of LLM-assisted research lacks standardized guidance. This article introduces the ELEVATE AI LLMs framework and checklist, designed to support researchers and reviewers in assessing LLM use in HEOR. Methods. The ELEVATE AI LLMs framework was developed through a targeted review of existing guidelines and evaluation frameworks. The framework comprises ten evaluation domains, including model characteristics, accuracy, comprehensiveness, and fairness. The accompanying checklist operationalizes the framework. To validate the framework, we applied it to two published studies, demonstrating its usability across different HEOR tasks. Results. The ELEVATE AI LLMs framework provides a comprehensive structure for evaluating LLM-assisted research, while the checklist facilitates practical application. Validation of the framework and checklist on studies of systematic literature reviews and health economic modeling highlighted their ability to identify strengths and gaps in reporting. Limitations. While the ELEVATE AI LLMs framework provides robust guidance, its broader generalizability and applicability to diverse HEOR tasks require further empirical testing. Additionally, several metrics adapted from computer science need further validation in HEOR contexts. Conclusion. The ELEVATE AI LLMs framework and checklist fill a critical gap in HEOR by offering structured guidance for evaluating LLM-assisted research. By promoting transparency, accuracy, and reproducibility, they aim to standardize and improve the integration of LLMs into HEOR, ensuring their outputs meet the field's rigorous standards.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.03408v3">A Framework for Effective Invocation Methods of Various LLM Services</a></div>
    <div class="paper-meta">
      📅 2024-12-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown impressive abilities in solving various natural language processing tasks and are now widely offered as services. LLM services enable users to accomplish tasks without requiring specialized knowledge, simply by paying service providers. However, numerous providers offer various LLM services with variations in pricing, latency, and performance. These factors are also affected by different invocation methods, such as the choice of context and the use of cache, which lead to unpredictable and uncontrollable service cost and quality. Consequently, utilizing various LLM services invocation methods to construct an effective (cost-saving, low-latency and high-performance) invocation strategy that best meets task demands becomes a pressing challenge. This paper provides a comprehensive overview of methods help LLM services to be invoked efficiently. Technically, we define the problem of constructing an effective LLM services invocation strategy, and based on this, propose a unified LLM service invocation framework. The framework classifies existing methods into four categories: input abstraction, semantic cache, solution design, and output enhancement, which can be used separately or jointly during the invocation life cycle. We discuss the methods in each category and compare them to provide valuable guidance for researchers. Finally, we emphasize the open challenges in this domain and shed light on future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.21315v4">Beyond Silent Letters: Amplifying LLMs in Emotion Recognition with Vocal Nuances</a></div>
    <div class="paper-meta">
      📅 2024-12-23
    </div>
    <details class="paper-abstract">
      Emotion recognition in speech is a challenging multimodal task that requires understanding both verbal content and vocal nuances. This paper introduces a novel approach to emotion detection using Large Language Models (LLMs), which have demonstrated exceptional capabilities in natural language understanding. To overcome the inherent limitation of LLMs in processing audio inputs, we propose SpeechCueLLM, a method that translates speech characteristics into natural language descriptions, allowing LLMs to perform multimodal emotion analysis via text prompts without any architectural changes. Our method is minimal yet impactful, outperforming baseline models that require structural modifications. We evaluate SpeechCueLLM on two datasets: IEMOCAP and MELD, showing significant improvements in emotion recognition accuracy, particularly for high-quality audio data. We also explore the effectiveness of various feature representations and fine-tuning strategies for different LLMs. Our experiments demonstrate that incorporating speech descriptions yields a more than 2% increase in the average weighted F1 score on IEMOCAP (from 70.111% to 72.596%).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.14546v3">Connecting the Dots: LLMs can Infer and Verbalize Latent Structure from Disparate Training Data</a></div>
    <div class="paper-meta">
      📅 2024-12-23
      | 💬 Accepted at NeurIPS 2024. 10 pages, 8 figures
    </div>
    <details class="paper-abstract">
      One way to address safety risks from large language models (LLMs) is to censor dangerous knowledge from their training data. While this removes the explicit information, implicit information can remain scattered across various training documents. Could an LLM infer the censored knowledge by piecing together these implicit hints? As a step towards answering this question, we study inductive out-of-context reasoning (OOCR), a type of generalization in which LLMs infer latent information from evidence distributed across training documents and apply it to downstream tasks without in-context learning. Using a suite of five tasks, we demonstrate that frontier LLMs can perform inductive OOCR. In one experiment we finetune an LLM on a corpus consisting only of distances between an unknown city and other known cities. Remarkably, without in-context examples or Chain of Thought, the LLM can verbalize that the unknown city is Paris and use this fact to answer downstream questions. Further experiments show that LLMs trained only on individual coin flip outcomes can verbalize whether the coin is biased, and those trained only on pairs $(x,f(x))$ can articulate a definition of $f$ and compute inverses. While OOCR succeeds in a range of cases, we also show that it is unreliable, particularly for smaller LLMs learning complex structures. Overall, the ability of LLMs to "connect the dots" without explicit in-context learning poses a potential obstacle to monitoring and controlling the knowledge acquired by LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13148v2">SWAN: SGD with Normalization and Whitening Enables Stateless LLM Training</a></div>
    <div class="paper-meta">
      📅 2024-12-23
      | 💬 In v2 we have revised the related work, added more comprehensive citations, and clarified our key contributions
    </div>
    <details class="paper-abstract">
      Adaptive optimizers such as Adam (Kingma & Ba, 2015) have been central to the success of large language models. However, they often require to maintain optimizer states throughout training, which can result in memory requirements several times greater than the model footprint. This overhead imposes constraints on scalability and computational efficiency. Stochastic Gradient Descent (SGD), in contrast, is a stateless optimizer, as it does not track state variables during training. Consequently, it achieves optimal memory efficiency. However, its capability in LLM training is limited (Zhao et al., 2024b). In this work, we show that pre-processing SGD in a stateless manner can achieve the same performance as the Adam optimizer for LLM training, while drastically reducing the memory cost. Specifically, we propose to pre-process the instantaneous stochastic gradients using normalization and whitening. We show that normalization stabilizes gradient distributions, and whitening counteracts the local curvature of the loss landscape. This results in SWAN (SGD with Whitening And Normalization), a stochastic optimizer that eliminates the need to store any optimizer states. Empirically, SWAN has the same memory footprint as SGD, achieving $\approx 50\%$ reduction on total end-to-end memory compared to Adam. In language modeling tasks, SWAN demonstrates comparable or even better performance than Adam: when pre-training the LLaMA model with 350M and 1.3B parameters, SWAN achieves a 2x speedup by reaching the same evaluation perplexity using half as many tokens.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.15714v2">AutoLife: Automatic Life Journaling with Smartphones and LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-23
      | 💬 13 pages
    </div>
    <details class="paper-abstract">
      This paper introduces a novel mobile sensing application - life journaling - designed to generate semantic descriptions of users' daily lives. We present AutoLife, an automatic life journaling system based on commercial smartphones. AutoLife only inputs low-cost sensor data (without photos or audio) from smartphones and can automatically generate comprehensive life journals for users. To achieve this, we first derive time, motion, and location contexts from multimodal sensor data, and harness the zero-shot capabilities of Large Language Models (LLMs), enriched with commonsense knowledge about human lives, to interpret diverse contexts and generate life journals. To manage the task complexity and long sensing duration, a multilayer framework is proposed, which decomposes tasks and seamlessly integrates LLMs with other techniques for life journaling. This study establishes a real-life dataset as a benchmark and extensive experiment results demonstrate that AutoLife produces accurate and reliable life journals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.12498v2">Enhancing Reasoning Capabilities of LLMs via Principled Synthetic Logic Corpus</a></div>
    <div class="paper-meta">
      📅 2024-12-23
      | 💬 NeurIPS 2024
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are capable of solving a wide range of tasks, yet they have struggled with reasoning. To address this, we propose $\textbf{Additional Logic Training (ALT)}$, which aims to enhance LLMs' reasoning capabilities by program-generated logical reasoning samples. We first establish principles for designing high-quality samples by integrating symbolic logic theory and previous empirical insights. Then, based on these principles, we construct a synthetic corpus named $\textbf{Formal Logic Deduction Diverse}$ ($\textbf{FLD}$$_{\times 2}$), comprising numerous samples of multi-step deduction with unknown facts, diverse reasoning rules, diverse linguistic expressions, and challenging distractors. Finally, we empirically show that ALT on FLD$_{\times2}$ substantially enhances the reasoning capabilities of state-of-the-art LLMs, including LLaMA-3.1-70B. Improvements include gains of up to 30 points on logical reasoning benchmarks, up to 10 points on math and coding benchmarks, and 5 points on the benchmark suite BBH.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.17321v1">Assessing Human Editing Effort on LLM-Generated Texts via Compression-Based Edit Distance</a></div>
    <div class="paper-meta">
      📅 2024-12-23
    </div>
    <details class="paper-abstract">
      Assessing the extent of human edits on texts generated by Large Language Models (LLMs) is crucial to understanding the human-AI interactions and improving the quality of automated text generation systems. Existing edit distance metrics, such as Levenshtein, BLEU, ROUGE, and TER, often fail to accurately measure the effort required for post-editing, especially when edits involve substantial modifications, such as block operations. In this paper, we introduce a novel compression-based edit distance metric grounded in the Lempel-Ziv-77 algorithm, designed to quantify the amount of post-editing applied to LLM-generated texts. Our method leverages the properties of text compression to measure the informational difference between the original and edited texts. Through experiments on real-world human edits datasets, we demonstrate that our proposed metric is highly correlated with actual edit time and effort. We also show that LLMs exhibit an implicit understanding of editing speed, that aligns well with our metric. Furthermore, we compare our metric with existing ones, highlighting its advantages in capturing complex edits with linear computational efficiency. Our code and data are available at: https://github.com/NDV-tiime/CompressionDistance
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.02746v5">RATT: A Thought Structure for Coherent and Correct LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2024-12-23
      | 💬 Accepted by AAAI 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) gain substantial reasoning and decision-making capabilities from thought structures. However, existing methods such as Tree of Thought and Retrieval Augmented Thoughts often fall short in complex tasks due to the limitations of insufficient local retrieval of factual knowledge and inadequate global selection of strategies. These limitations make it challenging for these methods to balance factual accuracy and comprehensive logical optimization effectively. To address these limitations, we introduce the Retrieval Augmented Thought Tree (RATT), a novel thought structure that considers both overall logical soundness and factual correctness at each step of the thinking process. Specifically, at every point of a thought branch, RATT performs planning and lookahead to explore and evaluate multiple potential reasoning steps, and integrate the fact-checking ability of Retrieval-Augmented Generation (RAG) with LLM's ability to assess overall strategy. Through this combination of factual knowledge and strategic feasibility, the RATT adjusts and integrates the thought tree structure to search for the most promising branches within the search space. This thought structure significantly enhances the model's coherence in logical inference and efficiency in decision-making, and thus increases the limit of the capacity of LLM to generate reliable inferences and decisions based on thought structures. A broad range of experiments on different types of tasks showcases that the RATT structure significantly outperforms existing methods in factual correctness and logical coherence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.18580v2">FFT: Towards Harmlessness Evaluation and Analysis for LLMs with Factuality, Fairness, Toxicity</a></div>
    <div class="paper-meta">
      📅 2024-12-23
      | 💬 Accepted by KDD workshop on Evaluation and Trustworthiness of Generative AI Models
    </div>
    <details class="paper-abstract">
      The widespread of generative artificial intelligence has heightened concerns about the potential harms posed by AI-generated texts, primarily stemming from factoid, unfair, and toxic content. Previous researchers have invested much effort in assessing the harmlessness of generative language models. However, existing benchmarks are struggling in the era of large language models (LLMs), due to the stronger language generation and instruction following capabilities, as well as wider applications. In this paper, we propose FFT, a new benchmark with 2116 elaborated-designed instances, for LLM harmlessness evaluation with factuality, fairness, and toxicity. To investigate the potential harms of LLMs, we evaluate 9 representative LLMs covering various parameter scales, training stages, and creators. Experiments show that the harmlessness of LLMs is still under-satisfactory, and extensive analysis derives some insightful findings that could inspire future research for harmless LLM research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.17259v1">LegalAgentBench: Evaluating LLM Agents in Legal Domain</a></div>
    <div class="paper-meta">
      📅 2024-12-23
      | 💬 23 pages
    </div>
    <details class="paper-abstract">
      With the increasing intelligence and autonomy of LLM agents, their potential applications in the legal domain are becoming increasingly apparent. However, existing general-domain benchmarks cannot fully capture the complexity and subtle nuances of real-world judicial cognition and decision-making. Therefore, we propose LegalAgentBench, a comprehensive benchmark specifically designed to evaluate LLM Agents in the Chinese legal domain. LegalAgentBench includes 17 corpora from real-world legal scenarios and provides 37 tools for interacting with external knowledge. We designed a scalable task construction framework and carefully annotated 300 tasks. These tasks span various types, including multi-hop reasoning and writing, and range across different difficulty levels, effectively reflecting the complexity of real-world legal scenarios. Moreover, beyond evaluating final success, LegalAgentBench incorporates keyword analysis during intermediate processes to calculate progress rates, enabling more fine-grained evaluation. We evaluated eight popular LLMs, highlighting the strengths, limitations, and potential areas for improvement of existing models and methods. LegalAgentBench sets a new benchmark for the practical application of LLMs in the legal domain, with its code and data available at \url{https://github.com/CSHaitao/LegalAgentBench}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.17250v1">SyNeg: LLM-Driven Synthetic Hard-Negatives for Dense Retrieval</a></div>
    <div class="paper-meta">
      📅 2024-12-23
    </div>
    <details class="paper-abstract">
      The performance of Dense retrieval (DR) is significantly influenced by the quality of negative sampling. Traditional DR methods primarily depend on naive negative sampling techniques or on mining hard negatives through external retriever and meticulously crafted strategies. However, naive negative sampling often fails to adequately capture the accurate boundaries between positive and negative samples, whereas existing hard negative sampling methods are prone to false negatives, resulting in performance degradation and training instability. Recent advancements in large language models (LLMs) offer an innovative solution to these challenges by generating contextually rich and diverse negative samples. In this work, we present a framework that harnesses LLMs to synthesize high-quality hard negative samples. We first devise a \textit{multi-attribute self-reflection prompting strategy} to direct LLMs in hard negative sample generation. Then, we implement a \textit{hybrid sampling strategy} that integrates these synthetic negatives with traditionally retrieved negatives, thereby stabilizing the training process and improving retrieval performance. Extensive experiments on five benchmark datasets demonstrate the efficacy of our approach, and code is also publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.00032v1">Highly Optimized Kernels and Fine-Grained Codebooks for LLM Inference on Arm CPUs</a></div>
    <div class="paper-meta">
      📅 2024-12-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have transformed the way we think about language understanding and generation, enthralling both researchers and developers. However, deploying LLMs for inference has been a significant challenge due to their unprecedented size and resource requirements. While quantizing model weights to sub-byte precision has emerged as a promising solution to ease memory pressure, the group quantization formats commonly used for LLM quantization have significant compute overheads and a resource-intensive dequantization process. As a result, a higher proportion of compute instructions do not perform multiplies, i.e., real work, rendering them unsuitable for meeting the required latency requirements for LLMs deployed on commodity CPUs. In this work, we propose a set of highly optimized kernels to accelerate LLM inference and unleash the full potential of CPUs, particularly Arm CPUs. These kernels amortize the cost of loading the operands and the cost of weight unpacking across multiple output rows. This, along with the introduction of an optimized interleaved group data layout for weights and decompression path optimizations to reduce unnecessary operations and dequantization overhead while maximizing the use of vector and matrix multiply operations, significantly improves the efficiency of MAC operations. Furthermore, we present a groupwise non-uniform codebook-based quantization method for ultra-low-precision quantization of LLMs to better match non-uniform patterns in their weight distributions, demonstrating better throughput during token generation while ensuring better quality than the state-of-the-art. Applying these improvements to 4-bit LLMs results in a 3-3.2x improvement in prompt processing and a 2x improvement in autoregressive decoding on Arm CPUs, compared to LLaMA.cpp-based solution. The optimized kernels are available at https://github.com/ggerganov/llama.cpp.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.17156v1">LLM-based relevance assessment still can't replace human relevance assessment</a></div>
    <div class="paper-meta">
      📅 2024-12-22
    </div>
    <details class="paper-abstract">
      The use of large language models (LLMs) for relevance assessment in information retrieval has gained significant attention, with recent studies suggesting that LLM-based judgments provide comparable evaluations to human judgments. Notably, based on TREC 2024 data, Upadhyay et al. make a bold claim that LLM-based relevance assessments, such as those generated by the UMBRELA system, can fully replace traditional human relevance assessments in TREC-style evaluations. This paper critically examines this claim, highlighting practical and theoretical limitations that undermine the validity of this conclusion. First, we question whether the evidence provided by Upadhyay et al. really supports their claim, particularly if a test collection is used asa benchmark for future improvements. Second, through a submission deliberately intended to do so, we demonstrate the ease with which automatic evaluation metrics can be subverted, showing that systems designed to exploit these evaluations can achieve artificially high scores. Theoretical challenges -- such as the inherent narcissism of LLMs, the risk of overfitting to LLM-based metrics, and the potential degradation of future LLM performance -- must be addressed before LLM-based relevance assessments can be considered a viable replacement for human judgments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.17149v1">A Multi-AI Agent System for Autonomous Optimization of Agentic AI Solutions via Iterative Refinement and LLM-Driven Feedback Loops</a></div>
    <div class="paper-meta">
      📅 2024-12-22
    </div>
    <details class="paper-abstract">
      Agentic AI systems use specialized agents to handle tasks within complex workflows, enabling automation and efficiency. However, optimizing these systems often requires labor-intensive, manual adjustments to refine roles, tasks, and interactions. This paper introduces a framework for autonomously optimizing Agentic AI solutions across industries, such as NLP-driven enterprise applications. The system employs agents for Refinement, Execution, Evaluation, Modification, and Documentation, leveraging iterative feedback loops powered by an LLM (Llama 3.2-3B). The framework achieves optimal performance without human input by autonomously generating and testing hypotheses to improve system configurations. This approach enhances scalability and adaptability, offering a robust solution for real-world applications in dynamic environments. Case studies across diverse domains illustrate the transformative impact of this framework, showcasing significant improvements in output quality, relevance, and actionability. All data for these case studies, including original and evolved agent codes, along with their outputs, are here: https://anonymous.4open.science/r/evolver-1D11/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.17146v1">LLM Agent for Fire Dynamics Simulations</a></div>
    <div class="paper-meta">
      📅 2024-12-22
      | 💬 NeurIPS 2024 Foundation Models for Science Workshop (38th Conference on Neural Information Processing Systems). 12 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Significant advances have been achieved in leveraging foundation models, such as large language models (LLMs), to accelerate complex scientific workflows. In this work we introduce FoamPilot, a proof-of-concept LLM agent designed to enhance the usability of FireFOAM, a specialized solver for fire dynamics and fire suppression simulations built using OpenFOAM, a popular open-source toolbox for computational fluid dynamics (CFD). FoamPilot provides three core functionalities: code insight, case configuration and simulation evaluation. Code insight is an alternative to traditional keyword searching leveraging retrieval-augmented generation (RAG) and aims to enable efficient navigation and summarization of the FireFOAM source code for developers and experienced users. For case configuration, the agent interprets user requests in natural language and aims to modify existing simulation setups accordingly to support intermediate users. FoamPilot's job execution functionality seeks to manage the submission and execution of simulations in high-performance computing (HPC) environments and provide preliminary analysis of simulation results to support less experienced users. Promising results were achieved for each functionality, particularly for simple tasks, and opportunities were identified for significant further improvement for more complex tasks. The integration of these functionalities into a single LLM agent is a step aimed at accelerating the simulation workflow for engineers and scientists employing FireFOAM for complex simulations critical for improving fire safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.00401v2">VIoTGPT: Learning to Schedule Vision Tools in LLMs towards Intelligent Video Internet of Things</a></div>
    <div class="paper-meta">
      📅 2024-12-22
      | 💬 AAAI 2025, 12 pages
    </div>
    <details class="paper-abstract">
      Video Internet of Things (VIoT) has shown full potential in collecting an unprecedented volume of video data. How to schedule the domain-specific perceiving models and analyze the collected videos uniformly, efficiently, and especially intelligently to accomplish complicated tasks is challenging. To address the challenge, we build VIoTGPT, the framework based on LLMs to correctly interact with humans, query knowledge videos, and invoke vision models to analyze multimedia data collaboratively. To support VIoTGPT and related future works, we meticulously crafted the VIoT-Tool dataset, including the training dataset and the benchmark involving 11 representative vision models across three categories based on semi-automatic annotations. To guide LLM to act as the intelligent agent towards intelligent VIoT, we resort to the ReAct instruction tuning method based on VIoT-Tool to learn the tool capability. Quantitative and qualitative experiments and analyses demonstrate the effectiveness of VIoTGPT. We believe VIoTGPT contributes to improving human-centered experiences in VIoT applications. The project website is https://github.com/zhongyy/VIoTGPT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.17056v1">The HalluRAG Dataset: Detecting Closed-Domain Hallucinations in RAG Applications Using an LLM's Internal States</a></div>
    <div class="paper-meta">
      📅 2024-12-22
      | 💬 19 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Detecting hallucinations in large language models (LLMs) is critical for enhancing their reliability and trustworthiness. Most research focuses on hallucinations as deviations from information seen during training. However, the opaque nature of an LLM's parametric knowledge complicates the understanding of why generated texts appear ungrounded: The LLM might not have picked up the necessary knowledge from large and often inaccessible datasets, or the information might have been changed or contradicted during further training. Our focus is on hallucinations involving information not used in training, which we determine by using recency to ensure the information emerged after a cut-off date. This study investigates these hallucinations by detecting them at sentence level using different internal states of various LLMs. We present HalluRAG, a dataset designed to train classifiers on these hallucinations. Depending on the model and quantization, MLPs trained on HalluRAG detect hallucinations with test accuracies ranging up to 75 %, with Mistral-7B-Instruct-v0.1 achieving the highest test accuracies. Our results show that IAVs detect hallucinations as effectively as CEVs and reveal that answerable and unanswerable prompts are encoded differently as separate classifiers for these categories improved accuracy. However, HalluRAG showed some limited generalizability, advocating for more diversity in datasets on hallucinations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.16984v1">LLM-Powered User Simulator for Recommender System</a></div>
    <div class="paper-meta">
      📅 2024-12-22
    </div>
    <details class="paper-abstract">
      User simulators can rapidly generate a large volume of timely user behavior data, providing a testing platform for reinforcement learning-based recommender systems, thus accelerating their iteration and optimization. However, prevalent user simulators generally suffer from significant limitations, including the opacity of user preference modeling and the incapability of evaluating simulation accuracy. In this paper, we introduce an LLM-powered user simulator to simulate user engagement with items in an explicit manner, thereby enhancing the efficiency and effectiveness of reinforcement learning-based recommender systems training. Specifically, we identify the explicit logic of user preferences, leverage LLMs to analyze item characteristics and distill user sentiments, and design a logical model to imitate real human engagement. By integrating a statistical model, we further enhance the reliability of the simulation, proposing an ensemble model that synergizes logical and statistical insights for user interaction simulations. Capitalizing on the extensive knowledge and semantic generation capabilities of LLMs, our user simulator faithfully emulates user behaviors and preferences, yielding high-fidelity training data that enrich the training of recommendation algorithms. We establish quantifying and qualifying experiments on five datasets to validate the simulator's effectiveness and stability across various recommendation scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.16974v1">Cannot or Should Not? Automatic Analysis of Refusal Composition in IFT/RLHF Datasets and Refusal Behavior of Black-Box LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-22
      | 💬 NeurIPS 2024 Workshop SFLLM
    </div>
    <details class="paper-abstract">
      Refusals - instances where large language models (LLMs) decline or fail to fully execute user instructions - are crucial for both AI safety and AI capabilities and the reduction of hallucinations in particular. These behaviors are learned during post-training, especially in instruction fine-tuning (IFT) and reinforcement learning from human feedback (RLHF). However, existing taxonomies and evaluation datasets for refusals are inadequate, often focusing solely on should-not-related (instead of cannot-related) categories, and lacking tools for auditing refusal content in black-box LLM outputs. We present a comprehensive framework for classifying LLM refusals: (a) a taxonomy of 16 refusal categories, (b) a human-annotated dataset of over 8,600 instances from publicly available IFT and RLHF datasets, (c) a synthetic dataset with 8,000 examples for each refusal category, and (d) classifiers trained for refusal classification. Our work enables precise auditing of refusal behaviors in black-box LLMs and automatic analyses of refusal patterns in large IFT and RLHF datasets. This facilitates the strategic adjustment of LLM refusals, contributing to the development of more safe and reliable LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.17874v1">Evaluating LLM Reasoning in the Operations Research Domain with ORQA</a></div>
    <div class="paper-meta">
      📅 2024-12-22
      | 💬 12 pages, 10 figures. Accepted and to be published in AAAI25
    </div>
    <details class="paper-abstract">
      In this paper, we introduce and apply Operations Research Question Answering (ORQA), a new benchmark designed to assess the generalization capabilities of Large Language Models (LLMs) in the specialized technical domain of Operations Research (OR). This benchmark evaluates whether LLMs can emulate the knowledge and reasoning skills of OR experts when confronted with diverse and complex optimization problems. The dataset, developed by OR experts, features real-world optimization problems that demand multistep reasoning to construct their mathematical models. Our evaluations of various open source LLMs, such as LLaMA 3.1, DeepSeek, and Mixtral, reveal their modest performance, highlighting a gap in their ability to generalize to specialized technical domains. This work contributes to the ongoing discourse on LLMs generalization capabilities, offering valuable insights for future research in this area. The dataset and evaluation code are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.16922v1">Enhancing Supply Chain Transparency in Emerging Economies Using Online Contents and LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-22
      | 💬 6 pages
    </div>
    <details class="paper-abstract">
      In the current global economy, supply chain transparency plays a pivotal role in ensuring this security by enabling companies to monitor supplier performance and fostering accountability and responsibility. Despite the advancements in supply chain relationship datasets like Bloomberg and FactSet, supply chain transparency remains a significant challenge in emerging economies due to issues such as information asymmetry and institutional gaps in regulation. This study proposes a novel approach to enhance supply chain transparency in emerging economies by leveraging online content and large language models (LLMs). We develop a Supply Chain Knowledge Graph Mining System that integrates advanced LLMs with web crawler technology to automatically collect and analyze supply chain information. The system's effectiveness is validated through a case study focusing on the semiconductor supply chain, a domain that has recently gained significant attention due to supply chain risks. Our results demonstrate that the proposed system provides greater applicability for emerging economies, such as mainland China, complementing the data gaps in existing datasets. However, challenges including the accurate estimation of monetary and material flows, the handling of time series data, synonyms disambiguation, and mitigating biases from online contents still remains. Future research should focus on addressing these issues to further enhance the system's capabilities and broaden its application to other emerging economies and industries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.05868v2">KG-FPQ: Evaluating Factuality Hallucination in LLMs with Knowledge Graph-based False Premise Questions</a></div>
    <div class="paper-meta">
      📅 2024-12-22
      | 💬 COLING2025 main
    </div>
    <details class="paper-abstract">
      Recent studies have demonstrated that large language models (LLMs) are susceptible to being misled by false premise questions (FPQs), leading to errors in factual knowledge, know as factuality hallucination. Existing benchmarks that assess this vulnerability primarily rely on manual construction, resulting in limited scale and lack of scalability. In this work, we introduce an automated, scalable pipeline to create FPQs based on knowledge graphs (KGs). The first step is modifying true triplets extracted from KGs to create false premises. Subsequently, utilizing the state-of-the-art capabilities of GPTs, we generate semantically rich FPQs. Based on the proposed method, we present a comprehensive benchmark, the Knowledge Graph-based False Premise Questions (KG-FPQ), which contains approximately 178k FPQs across three knowledge domains, at six levels of confusability, and in two task formats. Using KG-FPQ, we conduct extensive evaluations on several representative LLMs and provide valuable insights. The KG-FPQ dataset and code are available at~https://github.com/yanxuzhu/KG-FPQ.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.07810v3">Transformer Block Coupling and its Correlation with Generalization in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-22
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have made significant strides in natural language processing, and a precise understanding of the internal mechanisms driving their success is essential. In this work, we trace the trajectories of individual tokens as they pass through transformer blocks, and linearize the system along these trajectories through their Jacobian matrices. By examining the relationships between these Jacobians, we uncover a $\textbf{transformer block coupling}$ phenomenon in a variety of LLMs, characterized by the coupling of their top singular vectors across tokens and depth. Our findings reveal that coupling $\textit{positively correlates}$ with model performance, and that this relationship is stronger than with other hyperparameters, namely parameter budget, model depth, and embedding dimension. We further investigate the emergence of these properties through training, noting the development of coupling, as well as an increase in linearity and layer-wise exponential growth in the token trajectories. These collective insights provide a novel perspective on the interactions between token embeddings, and prompt further approaches to study training and generalization in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.16871v1">Teaching LLMs to Refine with Tools</a></div>
    <div class="paper-meta">
      📅 2024-12-22
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can refine their responses based on feedback, enabling self-improvement through iterative training or test-time refinement. However, existing methods predominantly focus on refinement within the same reasoning format, which may lead to non-correcting behaviors. We propose CaP, a novel approach that uses external tools to refine chain-of-thought (CoT) responses generated by the same or other LLMs. CaP employs a two-stage training process: supervised fine-tuning followed by preference optimization with DPO variants. Our observations highlight the critical role of preference optimization in enabling effective refinement. Additionally, we compare several sampling strategies to leverage CoT and tools at inference time. Experimental results demonstrate CaP's potential for effective cross-reasoning refinement and efficient inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09429v2">From Intention To Implementation: Automating Biomedical Research via LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-22
    </div>
    <details class="paper-abstract">
      Conventional biomedical research is increasingly labor-intensive due to the exponential growth of scientific literature and datasets. Artificial intelligence (AI), particularly Large Language Models (LLMs), has the potential to revolutionize this process by automating various steps. Still, significant challenges remain, including the need for multidisciplinary expertise, logicality of experimental design, and performance measurements. This paper introduces BioResearcher, the first end-to-end automated system designed to streamline the entire biomedical research process involving dry lab experiments. BioResearcher employs a modular multi-agent architecture, integrating specialized agents for search, literature processing, experimental design, and programming. By decomposing complex tasks into logically related sub-tasks and utilizing a hierarchical learning approach, BioResearcher effectively addresses the challenges of multidisciplinary requirements and logical complexity. Furthermore, BioResearcher incorporates an LLM-based reviewer for in-process quality control and introduces novel evaluation metrics to assess the quality and automation of experimental protocols. BioResearcher successfully achieves an average execution success rate of 63.07% across eight previously unmet research objectives. The generated protocols averagely outperform typical agent systems by 22.0% on five quality metrics. The system demonstrates significant potential to reduce researchers' workloads and accelerate biomedical discoveries, paving the way for future innovations in automated research systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.16838v1">Ask-Before-Detection: Identifying and Mitigating Conformity Bias in LLM-Powered Error Detector for Math Word Problem Solutions</a></div>
    <div class="paper-meta">
      📅 2024-12-22
      | 💬 12 pages, 4 figures
    </div>
    <details class="paper-abstract">
      The rise of large language models (LLMs) offers new opportunities for automatic error detection in education, particularly for math word problems (MWPs). While prior studies demonstrate the promise of LLMs as error detectors, they overlook the presence of multiple valid solutions for a single MWP. Our preliminary analysis reveals a significant performance gap between conventional and alternative solutions in MWPs, a phenomenon we term conformity bias in this work. To mitigate this bias, we introduce the Ask-Before-Detect (AskBD) framework, which generates adaptive reference solutions using LLMs to enhance error detection. Experiments on 200 examples of GSM8K show that AskBD effectively mitigates bias and improves performance, especially when combined with reasoning-enhancing techniques like chain-of-thought prompting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.16783v1">SubData: A Python Library to Collect and Combine Datasets for Evaluating LLM Alignment on Downstream Tasks</a></div>
    <div class="paper-meta">
      📅 2024-12-21
      | 💬 4 pages, 1 figure
    </div>
    <details class="paper-abstract">
      With the release of ever more capable large language models (LLMs), researchers in NLP and related disciplines have started to explore the usability of LLMs for a wide variety of different annotation tasks. Very recently, a lot of this attention has shifted to tasks that are subjective in nature. Given that the latest generations of LLMs have digested and encoded extensive knowledge about different human subpopulations and individuals, the hope is that these models can be trained, tuned or prompted to align with a wide range of different human perspectives. While researchers already evaluate the success of this alignment via surveys and tests, there is a lack of resources to evaluate the alignment on what oftentimes matters the most in NLP; the actual downstream tasks. To fill this gap we present SubData, a Python library that offers researchers working on topics related to subjectivity in annotation tasks a convenient way of collecting, combining and using a range of suitable datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13233v2">Creating an LLM-based AI-agent: A high-level methodology towards enhancing LLMs with APIs</a></div>
    <div class="paper-meta">
      📅 2024-12-21
      | 💬 in Greek language. This Master's Thesis was supervised by Prof. Nikolaos Papaspyrou (National Technical University of Athens) and Dr. Aifen Sui (Huawei Munich Research Center). English version: pages 57-104. Original submission link: http://artemis.cslab.ece.ntua.gr:8080/jspui/handle/123456789/19180
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized various aspects of engineering and science. Their utility is often bottlenecked by the lack of interaction with the external digital environment. To overcome this limitation and achieve integration of LLMs and Artificial Intelligence (AI) into real-world applications, customized AI agents are being constructed. Based on the technological trends and techniques, we extract a high-level approach for constructing these AI agents, focusing on their underlying architecture. This thesis serves as a comprehensive guide that elucidates a multi-faceted approach for empowering LLMs with the capability to leverage Application Programming Interfaces (APIs). We present a 7-step methodology that begins with the selection of suitable LLMs and the task decomposition that is necessary for complex problem-solving. This methodology includes techniques for generating training data for API interactions and heuristics for selecting the appropriate API among a plethora of options. These steps eventually lead to the generation of API calls that are both syntactically and semantically aligned with the LLM's understanding of a given task. Moreover, we review existing frameworks and tools that facilitate these processes and highlight the gaps in current attempts. In this direction, we propose an on-device architecture that aims to exploit the functionality of carry-on devices by using small models from the Hugging Face community. We examine the effectiveness of these approaches on real-world applications of various domains, including the generation of a piano sheet. Through an extensive analysis of the literature and available technologies, this thesis aims to set a compass for researchers and practitioners to harness the full potential of LLMs augmented with external tool capabilities, thus paving the way for more autonomous, robust, and context-aware AI agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.06059v2">LLM-Based Intent Processing and Network Optimization Using Attention-Based Hierarchical Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2024-12-21
      | 💬 Accepted paper at WCNC 2025
    </div>
    <details class="paper-abstract">
      Intent-based network automation is a promising tool to enable easier network management however certain challenges need to be effectively addressed. These are: 1) processing intents, i.e., identification of logic and necessary parameters to fulfill an intent, 2) validating an intent to align it with current network status, and 3) satisfying intents via network optimizing functions like xApps and rApps in O-RAN. This paper addresses these points via a three-fold strategy to introduce intent-based automation for O-RAN. First, intents are processed via a lightweight Large Language Model (LLM). Secondly, once an intent is processed, it is validated against future incoming traffic volume profiles (high or low). Finally, a series of network optimization applications (rApps and xApps) have been developed. With their machine learning-based functionalities, they can improve certain key performance indicators such as throughput, delay, and energy efficiency. In this final stage, using an attention-based hierarchical reinforcement learning algorithm, these applications are optimally initiated to satisfy the intent of an operator. Our simulations show that the proposed method can achieve at least 12% increase in throughput, 17.1% increase in energy efficiency, and 26.5% decrease in network delay compared to the baseline algorithms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14234v2">Syzygy: Dual Code-Test C to (safe) Rust Translation using LLMs and Dynamic Analysis</a></div>
    <div class="paper-meta">
      📅 2024-12-21
      | 💬 Project webpage at https://syzygy-project.github.io/. Preliminary version accepted at LLM4Code 2025, 34 pages
    </div>
    <details class="paper-abstract">
      Despite extensive usage in high-performance, low-level systems programming applications, C is susceptible to vulnerabilities due to manual memory management and unsafe pointer operations. Rust, a modern systems programming language, offers a compelling alternative. Its unique ownership model and type system ensure memory safety without sacrificing performance. In this paper, we present Syzygy, an automated approach to translate C to safe Rust. Our technique uses a synergistic combination of LLM-driven code and test translation guided by dynamic-analysis-generated execution information. This paired translation runs incrementally in a loop over the program in dependency order of the code elements while maintaining per-step correctness. Our approach exposes novel insights on combining the strengths of LLMs and dynamic analysis in the context of scaling and combining code generation with testing. We apply our approach to successfully translate Zopfli, a high-performance compression library with ~3000 lines of code and 98 functions. We validate the translation by testing equivalence with the source C program on a set of inputs. To our knowledge, this is the largest automated and test-validated C to safe Rust code translation achieved so far.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.16691v1">From Correlation to Causation: Understanding Climate Change through Causal Analysis and LLM Interpretations</a></div>
    <div class="paper-meta">
      📅 2024-12-21
    </div>
    <details class="paper-abstract">
      This research presents a three-step causal inference framework that integrates correlation analysis, machine learning-based causality discovery, and LLM-driven interpretations to identify socioeconomic factors influencing carbon emissions and contributing to climate change. The approach begins with identifying correlations, progresses to causal analysis, and enhances decision making through LLM-generated inquiries about the context of climate change. The proposed framework offers adaptable solutions that support data-driven policy-making and strategic decision-making in climate-related contexts, uncovering causal relationships within the climate change domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.16682v1">The Task Shield: Enforcing Task Alignment to Defend Against Indirect Prompt Injection in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2024-12-21
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents are increasingly being deployed as conversational assistants capable of performing complex real-world tasks through tool integration. This enhanced ability to interact with external systems and process various data sources, while powerful, introduces significant security vulnerabilities. In particular, indirect prompt injection attacks pose a critical threat, where malicious instructions embedded within external data sources can manipulate agents to deviate from user intentions. While existing defenses based on rule constraints, source spotlighting, and authentication protocols show promise, they struggle to maintain robust security while preserving task functionality. We propose a novel and orthogonal perspective that reframes agent security from preventing harmful actions to ensuring task alignment, requiring every agent action to serve user objectives. Based on this insight, we develop Task Shield, a test-time defense mechanism that systematically verifies whether each instruction and tool call contributes to user-specified goals. Through experiments on the AgentDojo benchmark, we demonstrate that Task Shield reduces attack success rates (2.07\%) while maintaining high task utility (69.79\%) on GPT-4o.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.13945v3">How Multi-Modal LLMs Reshape Visual Deep Learning Testing? A Comprehensive Study Through the Lens of Image Mutation</a></div>
    <div class="paper-meta">
      📅 2024-12-21
    </div>
    <details class="paper-abstract">
      Visual deep learning (VDL) systems have shown significant success in real-world applications like image recognition, object detection, and autonomous driving. To evaluate the reliability of VDL, a mainstream approach is software testing, which requires diverse mutations over image semantics. The rapid development of multi-modal large language models (MLLMs) has introduced revolutionary image mutation potentials through instruction-driven methods. Users can now freely describe desired mutations and let MLLMs generate the mutated images. Hence, parallel to large language models' (LLMs) recent success in traditional software fuzzing, one may also expect MLLMs to be promising for VDL testing in terms of offering unified, diverse, and complex image mutations. However, the quality and applicability of MLLM-based mutations in VDL testing remain largely unexplored. We present the first study, aiming to assess MLLMs' adequacy from 1) the semantic validity of MLLM mutated images, 2) the alignment of MLLM mutated images with their text instructions (prompts), and 3) the faithfulness of how different mutations preserve semantics that are ought to remain unchanged. With large-scale human studies and quantitative evaluations, we identify MLLM's promising potentials in expanding the covered semantics of image mutations. Notably, while SoTA MLLMs (e.g., GPT-4V) fail to support or perform worse in editing existing semantics in images (as in traditional mutations like rotation), they generate high-quality test inputs using "semantic-replacement" mutations (e.g., "dress a dog with clothes"), which bring extra semantics to images; these were infeasible for past approaches. Hence, we view MLLM-based mutations as a vital complement to traditional mutations, and advocate future VDL testing tasks to combine MLLM-based methods and traditional image mutations for comprehensive and reliable testing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.16643v1">TimeRAG: BOOSTING LLM Time Series Forecasting via Retrieval-Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2024-12-21
    </div>
    <details class="paper-abstract">
      Although the rise of large language models (LLMs) has introduced new opportunities for time series forecasting, existing LLM-based solutions require excessive training and exhibit limited transferability. In view of these challenges, we propose TimeRAG, a framework that incorporates Retrieval-Augmented Generation (RAG) into time series forecasting LLMs, which constructs a time series knowledge base from historical sequences, retrieves reference sequences from the knowledge base that exhibit similar patterns to the query sequence measured by Dynamic Time Warping (DTW), and combines these reference sequences and the prediction query as a textual prompt to the time series forecasting LLM. Experiments on datasets from various domains show that the integration of RAG improved the prediction accuracy of the original model by 2.97% on average.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.17565v3">MemServe: Context Caching for Disaggregated LLM Serving with Elastic Memory Pool</a></div>
    <div class="paper-meta">
      📅 2024-12-21
    </div>
    <details class="paper-abstract">
      Large language model (LLM) serving has transformed from stateless to stateful systems, utilizing techniques like context caching and disaggregated inference. These optimizations extend the lifespan and domain of the KV cache, necessitating a new architectural approach. We present MemServe, a unified system that integrates both inter-request and intra-request optimizations. MemServe introduces MemPool, an elastic memory pool managing distributed memory and KV caches across serving instances. Using MemPool APIs, MemServe combines context caching with disaggregated inference for the first time, supported by a global scheduler that enhances cache reuse through a global prompt tree-based locality-aware policy. Tests show that MemServe significantly improves job completion time and time-to-first-time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.03095v5">TestART: Improving LLM-based Unit Testing via Co-evolution of Automated Generation and Repair Iteration</a></div>
    <div class="paper-meta">
      📅 2024-12-21
    </div>
    <details class="paper-abstract">
      Unit testing is crucial for detecting bugs in individual program units but consumes time and effort. Recently, large language models (LLMs) have demonstrated remarkable capabilities in generating unit test cases. However, several problems limit their ability to generate high-quality unit test cases: (1) compilation and runtime errors caused by the hallucination of LLMs; (2) lack of testing and coverage feedback information restricting the increase of code coverage;(3) the repetitive suppression problem causing invalid LLM-based repair and generation attempts. To address these limitations, we propose TestART, a novel unit test generation method. TestART improves LLM-based unit testing via co-evolution of automated generation and repair iteration, representing a significant advancement in automated unit test generation. TestART leverages the template-based repair strategy to effectively fix bugs in LLM-generated test cases for the first time. Meanwhile, TestART extracts coverage information from successful test cases and uses it as coverage-guided testing feedback. It also incorporates positive prompt injection to prevent repetition suppression, thereby enhancing the sufficiency of the final test case. This synergy between generation and repair elevates the correctness and sufficiency of the produced test cases significantly beyond previous methods. In comparative experiments, TestART demonstrates an 18% improvement in pass rate and a 20% enhancement in coverage across three types of datasets compared to baseline models. Additionally, it achieves better coverage rates than EvoSuite with only half the number of test cases. These results demonstrate TestART's superior ability to produce high-quality unit test cases by harnessing the power of LLMs while overcoming their inherent flaws.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.15356v2">Regulator-Manufacturer AI Agents Modeling: Mathematical Feedback-Driven Multi-Agent LLM Framework</a></div>
    <div class="paper-meta">
      📅 2024-12-21
    </div>
    <details class="paper-abstract">
      The increasing complexity of regulatory updates from global authorities presents significant challenges for medical device manufacturers, necessitating agile strategies to sustain compliance and maintain market access. Concurrently, regulatory bodies must effectively monitor manufacturers' responses and develop strategic surveillance plans. This study employs a multi-agent modeling approach, enhanced with Large Language Models (LLMs), to simulate regulatory dynamics and examine the adaptive behaviors of key actors, including regulatory bodies, manufacturers, and competitors. These agents operate within a simulated environment governed by regulatory flow theory, capturing the impacts of regulatory changes on compliance decisions, market adaptation, and innovation strategies. Our findings illuminate the influence of regulatory shifts on industry behaviour and identify strategic opportunities for improving regulatory practices, optimizing compliance, and fostering innovation. By leveraging the integration of multi-agent systems and LLMs, this research provides a novel perspective and offers actionable insights for stakeholders navigating the evolving regulatory landscape of the medical device industry.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.17867v1">Evaluating and Enhancing LLMs for Multi-turn Text-to-SQL with Multiple Question Types</a></div>
    <div class="paper-meta">
      📅 2024-12-21
      | 💬 20 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have significantly advanced text-to-SQL systems. However, most LLM-based methods often narrowly focus on SQL generation, neglecting the complexities of real-world conversational queries. This oversight can lead to unreliable responses, particularly for ambiguous questions that cannot be directly addressed with SQL. To bridge this gap, we propose MMSQL, a comprehensive test suite designed to evaluate the question classification and SQL generation capabilities of LLMs by simulating real-world scenarios with diverse question types and multi-turn Q\&A interactions. Using MMSQL, we assessed the performance of popular LLMs, including both open-source and closed-source models, and identified key factors impacting their performance in such scenarios. Moreover, we introduce an LLM-based multi-agent framework that employs specialized agents to identify question types and determine appropriate answering strategies. Our experiments demonstrate that this approach significantly enhances the model's ability to navigate the complexities of conversational dynamics, effectively handling the diverse and complex nature of user queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.01644v2">InsightLens: Augmenting LLM-Powered Data Analysis with Interactive Insight Management and Navigation</a></div>
    <div class="paper-meta">
      📅 2024-12-21
      | 💬 Accepted to IEEE TVCG (PacificVis 2025)
    </div>
    <details class="paper-abstract">
      The proliferation of large language models (LLMs) has revolutionized the capabilities of natural language interfaces (NLIs) for data analysis. LLMs can perform multi-step and complex reasoning to generate data insights based on users' analytic intents. However, these insights often entangle with an abundance of contexts in analytic conversations such as code, visualizations, and natural language explanations. This hinders efficient recording, organization, and navigation of insights within the current chat-based LLM interfaces. In this paper, we first conduct a formative study with eight data analysts to understand their general workflow and pain points of insight management during LLM-powered data analysis. Accordingly, we introduce InsightLens, an interactive system to overcome such challenges. Built upon an LLM-agent-based framework that automates insight recording and organization along with the analysis process, InsightLens visualizes the complex conversational contexts from multiple aspects to facilitate insight navigation. A user study with twelve data analysts demonstrates the effectiveness of InsightLens, showing that it significantly reduces users' manual and cognitive effort without disrupting their conversational data analysis workflow, leading to a more efficient analysis experience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18652v5">$C^2$: Scalable Auto-Feedback for LLM-based Chart Generation</a></div>
    <div class="paper-meta">
      📅 2024-12-21
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Generating high-quality charts with Large Language Models (LLMs) presents significant challenges due to limited data and the high cost of scaling through human curation. $\langle \text{instruction}, \text{data}, \text{code} \rangle$ triplets are scarce and expensive to manually curate as their creation demands technical expertise. To address this scalability challenge, we introduce a reference-free automatic feedback generator, which eliminates the need for costly human intervention. Our novel framework, C$^2$, consists of (1) an automatic feedback provider (ChartAF) and (2) a diverse, reference-free dataset (ChartUIE-8K). The results are compelling: in our first experiment, 74% of respondents strongly preferred, and 10% preferred, the results after feedback. The second post-feedback experiment demonstrates that ChartAF outperform nine baselines. Moreover, ChartUIE-8K significantly improves data diversity by increasing queries, datasets, and chart types by 5982%, 1936%, and 91%, respectively, over benchmarks. Finally, a study of LLM users revealed that 94% of participants preferred ChartUIE-8K's queries, with 93% deeming them aligned with real-world use cases. Core contributions are available as open-source at chartsquared.github.io, with ample qualitative examples.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.16464v1">Transducer-Llama: Integrating LLMs into Streamable Transducer-based Speech Recognition</a></div>
    <div class="paper-meta">
      📅 2024-12-21
      | 💬 Accepted by ICASSP 2025
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have been applied to automatic speech recognition (ASR), the task of making the model streamable remains a challenge. This paper proposes a novel model architecture, Transducer-Llama, that integrates LLMs into a Factorized Transducer (FT) model, naturally enabling streaming capabilities. Furthermore, given that the large vocabulary of LLMs can cause data sparsity issue and increased training costs for spoken language systems, this paper introduces an efficient vocabulary adaptation technique to align LLMs with speech system vocabularies. The results show that directly optimizing the FT model with a strong pre-trained LLM-based predictor using the RNN-T loss yields some but limited improvements over a smaller pre-trained LM predictor. Therefore, this paper proposes a weak-to-strong LM swap strategy, using a weak LM predictor during RNN-T loss training and then replacing it with a strong LLM. After LM replacement, the minimum word error rate (MWER) loss is employed to finetune the integration of the LLM predictor with the Transducer-Llama model. Experiments on the LibriSpeech and large-scale multi-lingual LibriSpeech corpora show that the proposed streaming Transducer-Llama approach gave a 17% relative WER reduction (WERR) over a strong FT baseline and a 32% WERR over an RNN-T baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.16443v1">Has LLM Reached the Scaling Ceiling Yet? Unified Insights into LLM Regularities and Constraints</a></div>
    <div class="paper-meta">
      📅 2024-12-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities, yet their scalability raises a critical question: Have we reached the scaling ceiling? This paper addresses this pivotal question by developing a unified theoretical framework that integrates mathematical and statistical insights to explain the scaling dynamics of LLMs. We present: 1. Central Limit Theorem (CLT) for Hidden Representations: We show that noise in hidden representations scales inversely with context size, explaining stabilization effects and the limits of context length improvements. 2. Bias-Variance Decomposition: We decompose next-token prediction loss into irreducible entropy, capacity-driven bias, and finite sample variance, revealing trade-offs where scaling yields diminishing returns. 3. Emergent SNR Thresholds: By defining signal-to-noise ratio (SNR), we quantify how capabilities emerge abruptly once SNR surpasses a threshold, offering insights into when scaling becomes less effective. Through this framework, we conclude that while LLMs have not reached an absolute scaling ceiling, practical constraints are increasingly prominent: diminishing returns, resource inefficiencies, and data limitations. Future progress will require a shift from brute-force scaling to innovations in architecture, data quality, and training paradigms. This work provides a roadmap for guiding the efficient development of next-generation LLMs and advancing the field beyond traditional scaling strategies. Keywords: Large Language Models; Scaling Ceiling; Central Limit Theorem; Bias-Variance Trade-Off; Signal-to-Noise Ratio; Emergent Capabilities
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.17169v3">REAL: Response Embedding-based Alignment for LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-21
    </div>
    <details class="paper-abstract">
      Aligning large language models (LLMs) to human preferences is a crucial step in building helpful and safe AI tools, which usually involve training on supervised datasets. Popular algorithms such as Direct Preference Optimization rely on pairs of AI-generated responses ranked according to human feedback. The response pair annotation process is the most labor-intensive and costly part of the alignment pipeline, and improving its efficiency and annotation quality would have a meaningful impact on AI development. We propose REAL: Response Embedding-based Alignment for LLMs, a strategy for constructing a high-quality training dataset that focuses on acquiring the most informative response pairs for labeling out of a set of response candidates. Our selection process is based on embedding responses independently of prompts. Experimental results on real-world dataset SHP2 and synthetic HH-RLHF benchmarks indicate that choosing dissimilar response pairs enhances the direct alignment of LLMs while reducing inherited labeling errors. The model aligned on dissimilar response pairs obtained a better margin and win rate on the dialogue task. Our findings suggest that focusing on distinct pairs can reduce the label error to improve the efficiency of LLM alignment, saving up to 65% of annotators' work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.16434v1">SYMPHONY: Improving Memory Management for LLM Inference Workloads</a></div>
    <div class="paper-meta">
      📅 2024-12-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly being deployed in applications such as chatbots, code editors, and conversational agents. A key feature of LLMs is their ability to engage in multi-turn interactions with humans or external tools, enabling a wide range of tasks. Each new request in a multi-turn interaction depends on the intermediate state, specifically the key-value (K,V) caches, from previous requests in the ongoing interaction. Existing serving engines either recompute the K,V caches or offload them to main memory. Profiling reveals that recomputation can result in over 99% of processed tokens being redundant. On the other hand, offloading K,V caches from GPU memory makes inference serving stateful, leading to load imbalances across the cluster. To address these challenges, we developed SYMPHONY. SYMPHONY leverages the observation that multi-turn work loads provide additional hints that allow K,V caches to be migrated off the critical serving path. By utilizing these hints, SYMPHONY dynamically migrates K,V caches to enable finegrained scheduling of inference requests. Our experiments demonstrate that SYMPHONY can handle over 8x the number of requests compared to state-of-the-art baselines, with a similar latency profile.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14140v2">GLIDER: Grading LLM Interactions and Decisions using Explainable Ranking</a></div>
    <div class="paper-meta">
      📅 2024-12-20
    </div>
    <details class="paper-abstract">
      The LLM-as-judge paradigm is increasingly being adopted for automated evaluation of model outputs. While LLM judges have shown promise on constrained evaluation tasks, closed source LLMs display critical shortcomings when deployed in real world applications due to challenges of fine grained metrics and explainability, while task specific evaluation models lack cross-domain generalization. We introduce GLIDER, a powerful 3B evaluator LLM that can score any text input and associated context on arbitrary user defined criteria. GLIDER shows higher Pearson's correlation than GPT-4o on FLASK and greatly outperforms prior evaluation models, achieving comparable performance to LLMs 17x its size. GLIDER supports fine-grained scoring, multilingual reasoning, span highlighting and was trained on 685 domains and 183 criteria. Extensive qualitative analysis shows that GLIDER scores are highly correlated with human judgments, with 91.3% human agreement. We have open-sourced GLIDER to facilitate future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.16359v1">Human-Readable Adversarial Prompts: An Investigation into LLM Vulnerabilities Using Situational Context</a></div>
    <div class="paper-meta">
      📅 2024-12-20
    </div>
    <details class="paper-abstract">
      Previous research on LLM vulnerabilities often relied on nonsensical adversarial prompts, which were easily detectable by automated methods. We address this gap by focusing on human-readable adversarial prompts, a more realistic and potent threat. Our key contributions are situation-driven attacks leveraging movie scripts to create contextually relevant, human-readable prompts that successfully deceive LLMs, adversarial suffix conversion to transform nonsensical adversarial suffixes into meaningful text, and AdvPrompter with p-nucleus sampling, a method to generate diverse, human-readable adversarial suffixes, improving attack efficacy in models like GPT-3.5 and Gemma 7B. Our findings demonstrate that LLMs can be tricked by sophisticated adversaries into producing harmful responses with human-readable adversarial prompts and that there exists a scope for improvement when it comes to robust LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.17875v2">Understanding Layer Significance in LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2024-12-20
    </div>
    <details class="paper-abstract">
      Aligning large language models (LLMs) through fine-tuning is essential for tailoring them to specific applications. Therefore, understanding what LLMs learn during the alignment process is crucial. Recent studies suggest that alignment primarily adjusts a model's presentation style rather than its foundational knowledge, indicating that only certain components of the model are significantly impacted. To delve deeper into LLM alignment, we propose to identify which layers within LLMs are most critical to the alignment process, thereby uncovering how alignment influences model behavior at a granular level. We propose a novel approach to identify the important layers for LLM alignment (ILA). It involves learning a binary mask for each incremental weight matrix in the LoRA algorithm, indicating the significance of each layer. ILA consistently identifies important layers across various alignment datasets, with nearly 90% overlap even with substantial dataset differences, highlighting fundamental patterns in LLM alignment. Experimental results indicate that freezing non-essential layers improves overall model performance, while selectively tuning the most critical layers significantly enhances fine-tuning efficiency with minimal performance loss.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.16291v1">Benchmarking LLMs and SLMs for patient reported outcomes</a></div>
    <div class="paper-meta">
      📅 2024-12-20
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      LLMs have transformed the execution of numerous tasks, including those in the medical domain. Among these, summarizing patient-reported outcomes (PROs) into concise natural language reports is of particular interest to clinicians, as it enables them to focus on critical patient concerns and spend more time in meaningful discussions. While existing work with LLMs like GPT-4 has shown impressive results, real breakthroughs could arise from leveraging SLMs as they offer the advantage of being deployable locally, ensuring patient data privacy and compliance with healthcare regulations. This study benchmarks several SLMs against LLMs for summarizing patient-reported Q\&A forms in the context of radiotherapy. Using various metrics, we evaluate their precision and reliability. The findings highlight both the promise and limitations of SLMs for high-stakes medical tasks, fostering more efficient and privacy-preserving AI-driven healthcare solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.06857v4">What is the Role of Small Models in the LLM Era: A Survey</a></div>
    <div class="paper-meta">
      📅 2024-12-20
      | 💬 a survey paper of small models
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have made significant progress in advancing artificial general intelligence (AGI), leading to the development of increasingly large models such as GPT-4 and LLaMA-405B. However, scaling up model sizes results in exponentially higher computational costs and energy consumption, making these models impractical for academic researchers and businesses with limited resources. At the same time, Small Models (SMs) are frequently used in practical settings, although their significance is currently underestimated. This raises important questions about the role of small models in the era of LLMs, a topic that has received limited attention in prior research. In this work, we systematically examine the relationship between LLMs and SMs from two key perspectives: Collaboration and Competition. We hope this survey provides valuable insights for practitioners, fostering a deeper understanding of the contribution of small models and promoting more efficient use of computational resources. The code is available at https://github.com/tigerchen52/role_of_small_models
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.16120v1">PromptOptMe: Error-Aware Prompt Compression for LLM-based MT Evaluation Metrics</a></div>
    <div class="paper-meta">
      📅 2024-12-20
    </div>
    <details class="paper-abstract">
      Evaluating the quality of machine-generated natural language content is a challenging task in Natural Language Processing (NLP). Recently, large language models (LLMs) like GPT-4 have been employed for this purpose, but they are computationally expensive due to the extensive token usage required by complex evaluation prompts. In this paper, we propose a prompt optimization approach that uses a smaller, fine-tuned language model to compress input data for evaluation prompt, thus reducing token usage and computational cost when using larger LLMs for downstream evaluation. Our method involves a two-stage fine-tuning process: supervised fine-tuning followed by preference optimization to refine the model's outputs based on human preferences. We focus on Machine Translation (MT) evaluation and utilize the GEMBA-MQM metric as a starting point. Our results show a $2.37\times$ reduction in token usage without any loss in evaluation quality. This work makes state-of-the-art LLM-based metrics like GEMBA-MQM more cost-effective and efficient, enhancing their accessibility for broader use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.16119v1">Deciphering the Underserved: Benchmarking LLM OCR for Low-Resource Scripts</a></div>
    <div class="paper-meta">
      📅 2024-12-20
    </div>
    <details class="paper-abstract">
      This study investigates the potential of Large Language Models (LLMs), particularly GPT-4o, for Optical Character Recognition (OCR) in low-resource scripts such as Urdu, Albanian, and Tajik, with English serving as a benchmark. Using a meticulously curated dataset of 2,520 images incorporating controlled variations in text length, font size, background color, and blur, the research simulates diverse real-world challenges. Results emphasize the limitations of zero-shot LLM-based OCR, particularly for linguistically complex scripts, highlighting the need for annotated datasets and fine-tuned models. This work underscores the urgency of addressing accessibility gaps in text digitization, paving the way for inclusive and robust OCR solutions for underserved languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.16089v1">The Evolution of LLM Adoption in Industry Data Curation Practices</a></div>
    <div class="paper-meta">
      📅 2024-12-20
      | 💬 19 pages, 4 tables, 3 figures
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) grow increasingly adept at processing unstructured text data, they offer new opportunities to enhance data curation workflows. This paper explores the evolution of LLM adoption among practitioners at a large technology company, evaluating the impact of LLMs in data curation tasks through participants' perceptions, integration strategies, and reported usage scenarios. Through a series of surveys, interviews, and user studies, we provide a timely snapshot of how organizations are navigating a pivotal moment in LLM evolution. In Q2 2023, we conducted a survey to assess LLM adoption in industry for development tasks (N=84), and facilitated expert interviews to assess evolving data needs (N=10) in Q3 2023. In Q2 2024, we explored practitioners' current and anticipated LLM usage through a user study involving two LLM-based prototypes (N=12). While each study addressed distinct research goals, they revealed a broader narrative about evolving LLM usage in aggregate. We discovered an emerging shift in data understanding from heuristic-first, bottom-up approaches to insights-first, top-down workflows supported by LLMs. Furthermore, to respond to a more complex data landscape, data practitioners now supplement traditional subject-expert-created 'golden datasets' with LLM-generated 'silver' datasets and rigorously validated 'super golden' datasets curated by diverse experts. This research sheds light on the transformative role of LLMs in large-scale analysis of unstructured data and highlights opportunities for further tool development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14426v2">All-in-One Tuning and Structural Pruning for Domain-Specific LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-20
      | 💬 Updated a typo in the author list;
    </div>
    <details class="paper-abstract">
      Existing pruning techniques for large language models (LLMs) targeting domain-specific applications typically follow a two-stage process: pruning the pretrained general-purpose LLMs and then fine-tuning the pruned LLMs on specific domains. However, the pruning decisions, derived from the pretrained weights, remain unchanged during fine-tuning, even if the weights have been updated. Therefore, such a combination of the pruning decisions and the finetuned weights may be suboptimal, leading to non-negligible performance degradation. To address these limitations, we propose ATP: All-in-One Tuning and Structural Pruning, a unified one-stage structural pruning and fine-tuning approach that dynamically identifies the current optimal substructure throughout the fine-tuning phase via a trainable pruning decision generator. Moreover, given the limited available data for domain-specific applications, Low-Rank Adaptation (LoRA) becomes a common technique to fine-tune the LLMs. In ATP, we introduce LoRA-aware forward and sparsity regularization to ensure that the substructures corresponding to the learned pruning decisions can be directly removed after the ATP process. ATP outperforms the state-of-the-art two-stage pruning methods on tasks in the legal and healthcare domains. More specifically, ATP recovers up to 88% and 91% performance of the dense model when pruning 40% parameters of LLaMA2-7B and LLaMA3-8B models, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.15993v1">Fearful Falcons and Angry Llamas: Emotion Category Annotations of Arguments by Humans and LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-20
    </div>
    <details class="paper-abstract">
      Arguments evoke emotions, influencing the effect of the argument itself. Not only the emotional intensity but also the category influence the argument's effects, for instance, the willingness to adapt stances. While binary emotionality has been studied in arguments, there is no work on discrete emotion categories (e.g., "Anger") in such data. To fill this gap, we crowdsource subjective annotations of emotion categories in a German argument corpus and evaluate automatic LLM-based labeling methods. Specifically, we compare three prompting strategies (zero-shot, one-shot, chain-of-thought) on three large instruction-tuned language models (Falcon-7b-instruct, Llama-3.1-8B-instruct, GPT-4o-mini). We further vary the definition of the output space to be binary (is there emotionality in the argument?), closed-domain (which emotion from a given label set is in the argument?), or open-domain (which emotion is in the argument?). We find that emotion categories enhance the prediction of emotionality in arguments, emphasizing the need for discrete emotion annotations in arguments. Across all prompt settings and models, automatic predictions show a high recall but low precision for predicting anger and fear, indicating a strong bias toward negative emotions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00535v5">FullStack Bench: Evaluating LLMs as Full Stack Coders</a></div>
    <div class="paper-meta">
      📅 2024-12-20
      | 💬 26 pages
    </div>
    <details class="paper-abstract">
      As the capabilities of code large language models (LLMs) continue to expand, their applications across diverse code intelligence domains are rapidly increasing. However, most existing datasets only evaluate limited application domains. To address this gap, we have developed a comprehensive code evaluation dataset FullStack Bench focusing on full-stack programming, which encompasses a wide range of application domains (e.g., basic programming, data analysis, software engineering, mathematics, and machine learning). Besides, to assess multilingual programming capabilities, in FullStack Bench, we design real-world instructions and corresponding unit test cases from 16 widely-used programming languages to reflect real-world usage scenarios rather than simple translations. Moreover, we also release an effective code sandbox execution tool (i.e., SandboxFusion) supporting various programming languages and packages to evaluate the performance of our FullStack Bench efficiently. Comprehensive experimental results on our FullStack Bench demonstrate the necessity and effectiveness of our FullStack Bench and SandboxFusion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.15902v1">On the Suitability of pre-trained foundational LLMs for Analysis in German Legal Education</a></div>
    <div class="paper-meta">
      📅 2024-12-20
      | 💬 11 pages
    </div>
    <details class="paper-abstract">
      We show that current open-source foundational LLMs possess instruction capability and German legal background knowledge that is sufficient for some legal analysis in an educational context. However, model capability breaks down in very specific tasks, such as the classification of "Gutachtenstil" appraisal style components, or with complex contexts, such as complete legal opinions. Even with extended context and effective prompting strategies, they cannot match the Bag-of-Words baseline. To combat this, we introduce a Retrieval Augmented Generation based prompt example selection method that substantially improves predictions in high data availability scenarios. We further evaluate the performance of pre-trained LLMs on two standard tasks for argument mining and automated essay scoring and find it to be more adequate. Throughout, pre-trained LLMs improve upon the baseline in scenarios with little or no labeled data with Chain-of-Thought prompting further helping in the zero-shot case.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.01866v3">Efficient Solutions For An Intriguing Failure of LLMs: Long Context Window Does Not Mean LLMs Can Analyze Long Sequences Flawlessly</a></div>
    <div class="paper-meta">
      📅 2024-12-20
      | 💬 12 pages, 5 figures, 7 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities in comprehending and analyzing lengthy sequential inputs, owing to their extensive context windows that allow processing millions of tokens in a single forward pass. However, this paper uncovers a surprising limitation: LLMs fall short when handling long input sequences. We investigate this issue using three datasets and two tasks (sentiment analysis and news categorization) across various LLMs, including Claude 3, Gemini Pro, GPT 3.5 Turbo, Llama 3 Instruct, and Mistral Instruct models. To address this limitation, we propose and evaluate ad-hoc solutions that substantially enhance LLMs' performance on long input sequences by up to 50%, while reducing API cost and latency by up to 93% and 50%, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.13975v3">MR-Ben: A Meta-Reasoning Benchmark for Evaluating System-2 Thinking in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown increasing capability in problem-solving and decision-making, largely based on the step-by-step chain-of-thought reasoning processes. However, evaluating these reasoning abilities has become increasingly challenging. Existing outcome-based benchmarks are beginning to saturate, becoming less effective in tracking meaningful progress. To address this, we present a process-based benchmark MR-Ben that demands a meta-reasoning skill, where LMs are asked to locate and analyse potential errors in automatically generated reasoning steps. Our meta-reasoning paradigm is especially suited for system-2 slow thinking, mirroring the human cognitive process of carefully examining assumptions, conditions, calculations, and logic to identify mistakes.MR-Ben comprises 5,975 questions curated by human experts across a wide range of subjects, including physics, chemistry, logic, coding, and more. Through our designed metrics for assessing meta-reasoning on this benchmark, we identify interesting limitations and weaknesses of current LLMs (open-source and closed-source models). For example, with models like the o1 series from OpenAI demonstrating strong performance by effectively scrutinizing the solution space, many other state-of-the-art models fall significantly behind on MR-Ben, exposing potential shortcomings in their training strategies and inference methodologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09569v2">Are You Human? An Adversarial Benchmark to Expose LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated an alarming ability to impersonate humans in conversation, raising concerns about their potential misuse in scams and deception. Humans have a right to know if they are conversing to an LLM. We evaluate text-based prompts designed as challenges to expose LLM imposters in real-time. To this end we compile and release an open-source benchmark dataset that includes 'implicit challenges' that exploit an LLM's instruction-following mechanism to cause role deviation, and 'exlicit challenges' that test an LLM's ability to perform simple tasks typically easy for humans but difficult for LLMs. Our evaluation of 9 leading models from the LMSYS leaderboard revealed that explicit challenges successfully detected LLMs in 78.4% of cases, while implicit challenges were effective in 22.9% of instances. User studies validate the real-world applicability of our methods, with humans outperforming LLMs on explicit challenges (78% vs 22% success rate). Our framework unexpectedly revealed that many study participants were using LLMs to complete tasks, demonstrating its effectiveness in detecting both AI impostors and human misuse of AI tools. This work addresses the critical need for reliable, real-time LLM detection methods in high-stakes conversations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14050v2">Cross-Lingual Transfer of Debiasing and Detoxification in Multilingual LLMs: An Extensive Investigation</a></div>
    <div class="paper-meta">
      📅 2024-12-20
    </div>
    <details class="paper-abstract">
      Recent generative large language models (LLMs) show remarkable performance in non-English languages, but when prompted in those languages they tend to express higher harmful social biases and toxicity levels. Prior work has shown that finetuning on specialized datasets can mitigate this behavior, and doing so in English can transfer to other languages. In this work, we investigate the impact of different finetuning methods on the model's bias and toxicity, but also on its ability to produce fluent and diverse text. Our results show that finetuning on curated non-harmful text is more effective for mitigating bias, and finetuning on direct preference optimization (DPO) datasets is more effective for mitigating toxicity. The mitigation caused by applying these methods in English also transfers to non-English languages. We find evidence that the extent to which transfer takes place can be predicted by the amount of data in a given language present in the model's pretraining data. However, this transfer of bias and toxicity mitigation often comes at the expense of decreased language generation ability in non-English languages, highlighting the importance of developing language-specific bias and toxicity mitigation methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.11532v3">CKGFuzzer: LLM-Based Fuzz Driver Generation Enhanced By Code Knowledge Graph</a></div>
    <div class="paper-meta">
      📅 2024-12-20
      | 💬 12 pages, 3 figures
    </div>
    <details class="paper-abstract">
      In recent years, the programming capabilities of large language models (LLMs) have garnered significant attention. Fuzz testing, a highly effective technique, plays a key role in enhancing software reliability and detecting vulnerabilities. However, traditional fuzz testing tools rely on manually crafted fuzz drivers, which can limit both testing efficiency and effectiveness. To address this challenge, we propose an automated fuzz testing method driven by a code knowledge graph and powered by an LLM-based intelligent agent system, referred to as CKGFuzzer. We approach fuzz driver creation as a code generation task, leveraging the knowledge graph of the code repository to automate the generation process within the fuzzing loop, while continuously refining both the fuzz driver and input seeds. The code knowledge graph is constructed through interprocedural program analysis, where each node in the graph represents a code entity, such as a function or a file. The knowledge graph-enhanced CKGFuzzer not only effectively resolves compilation errors in fuzz drivers and generates input seeds tailored to specific API usage scenarios, but also analyzes fuzz driver crash reports, assisting developers in improving code quality. By querying the knowledge graph of the code repository and learning from API usage scenarios, we can better identify testing targets and understand the specific purpose of each fuzz driver. We evaluated our approach using eight open-source software projects. The experimental results indicate that CKGFuzzer achieved an average improvement of 8.73% in code coverage compared to state-of-the-art techniques. Additionally, CKGFuzzer reduced the manual review workload in crash case analysis by 84.4% and successfully detected 11 real bugs (including nine previously unreported bugs) across the tested libraries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.15803v1">WebLLM: A High-Performance In-Browser LLM Inference Engine</a></div>
    <div class="paper-meta">
      📅 2024-12-20
    </div>
    <details class="paper-abstract">
      Advancements in large language models (LLMs) have unlocked remarkable capabilities. While deploying these models typically requires server-grade GPUs and cloud-based inference, the recent emergence of smaller open-source models and increasingly powerful consumer devices have made on-device deployment practical. The web browser as a platform for on-device deployment is universally accessible, provides a natural agentic environment, and conveniently abstracts out the different backends from diverse device vendors. To address this opportunity, we introduce WebLLM, an open-source JavaScript framework that enables high-performance LLM inference entirely within web browsers. WebLLM provides an OpenAI-style API for seamless integration into web applications, and leverages WebGPU for efficient local GPU acceleration and WebAssembly for performant CPU computation. With machine learning compilers MLC-LLM and Apache TVM, WebLLM leverages optimized WebGPU kernels, overcoming the absence of performant WebGPU kernel libraries. Evaluations show that WebLLM can retain up to 80% native performance on the same device, with room to further close the gap. WebLLM paves the way for universally accessible, privacy-preserving, personalized, and locally powered LLM applications in web browsers. The code is available at: https://github.com/mlc-ai/web-llm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.15676v1">Code Review Automation Via Multi-task Federated LLM -- An Empirical Study</a></div>
    <div class="paper-meta">
      📅 2024-12-20
    </div>
    <details class="paper-abstract">
      Code review is a crucial process before deploying code to production, as it validates the code, provides suggestions for improvements, and identifies errors such as missed edge cases. In projects with regular production releases, the effort required for peer code-reviews remains high. Consequently, there has been significant interest from software engineering (SE) researchers in automating the code review process. Previous research on code review automation has typically approached the task as three independent sub-tasks: review necessity prediction, review comment generation, and code refinement. Our study attempts to (i) leverage the relationships between the sub-tasks of code review automation, by developing a multi-task model that addresses all tasks in an integrated manner, and (ii) increase model robustness on unseen data via collaborative large language model (LLM) modeling, while retaining the proprietary nature of code, by using federated learning (FL). The study explores five simple techniques for multi-task training, including two sequential methods, one parallel method, and two cumulative methods. The results indicate that sequentially training a federated LLM (FedLLM) for our code review multi-task use case is less efficient in terms of time, computation, and performance metrics, compared to training separate models for each task. Because sequential training demonstrates catastrophic forgetting, alternatively cumulative fine-tuning for multi-task training performs better than training models for individual tasks. This study highlights the need for research focused on effective fine-tuning of multi-task FedLLMs for SE tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.16260v1">Inference Scaling vs Reasoning: An Empirical Analysis of Compute-Optimal LLM Problem-Solving</a></div>
    <div class="paper-meta">
      📅 2024-12-20
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have predominantly focused on maximizing accuracy and reasoning capabilities, often overlooking crucial computational efficiency considerations. While this approach has yielded impressive accuracy improvements, it has led to methods that may be impractical for real-world deployment due to computational overhead and latency constraints. This paper investigates the potential synergy between reasoning enhancement and computational efficiency by analyzing the integration of two contrasting approaches: Quiet-STaR (Self-Taught Reasoner) and REBASE (REward BAlanced SEarch). Through comprehensive empirical analysis using the Mistral-7B model on the GSM8K dataset, we demonstrate that while each method excels in its primary objective-Quiet-STaR achieving superior accuracy (32.03%) despite high computational cost (554.66s runtime, 12.73T FLOPs), and REBASE providing exceptional efficiency (8.47s runtime, 2.35T FLOPs) while maintaining baseline-comparable accuracy (10.94%)-their integration reveals fundamental challenges in reconciling reasoning depth with computational efficiency. The combined approach unexpectedly results in degraded performance (9.38% accuracy, 143.66s runtime), highlighting critical insights about the complex interplay between reasoning enhancement and efficiency optimization in LLMs. Our findings illuminate the need for novel architectures and algorithms specifically designed to bridge the gap between these competing objectives, while providing concrete directions for future research in compute-efficient reasoning methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.15623v1">JailPO: A Novel Black-box Jailbreak Framework via Preference Optimization against Aligned LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-20
      | 💬 Accepted by AAAI 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) aligned with human feedback have recently garnered significant attention. However, it remains vulnerable to jailbreak attacks, where adversaries manipulate prompts to induce harmful outputs. Exploring jailbreak attacks enables us to investigate the vulnerabilities of LLMs and further guides us in enhancing their security. Unfortunately, existing techniques mainly rely on handcrafted templates or generated-based optimization, posing challenges in scalability, efficiency and universality. To address these issues, we present JailPO, a novel black-box jailbreak framework to examine LLM alignment. For scalability and universality, JailPO meticulously trains attack models to automatically generate covert jailbreak prompts. Furthermore, we introduce a preference optimization-based attack method to enhance the jailbreak effectiveness, thereby improving efficiency. To analyze model vulnerabilities, we provide three flexible jailbreak patterns. Extensive experiments demonstrate that JailPO not only automates the attack process while maintaining effectiveness but also exhibits superior performance in efficiency, universality, and robustness against defenses compared to baselines. Additionally, our analysis of the three JailPO patterns reveals that attacks based on complex templates exhibit higher attack strength, whereas covert question transformations elicit riskier responses and are more likely to bypass defense mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.15594v1">Template-Driven LLM-Paraphrased Framework for Tabular Math Word Problem Generation</a></div>
    <div class="paper-meta">
      📅 2024-12-20
      | 💬 Accepted at AAAI 2025, extended version with appendix
    </div>
    <details class="paper-abstract">
      Solving tabular math word problems (TMWPs) has become a critical role in evaluating the mathematical reasoning ability of large language models (LLMs), where large-scale TMWP samples are commonly required for LLM fine-tuning. Since the collection of high-quality TMWP datasets is costly and time-consuming, recent research has concentrated on automatic TMWP generation. However, current generated samples usually suffer from issues of either correctness or diversity. In this paper, we propose a Template-driven LLM-paraphrased (TeLL) framework for generating high-quality TMWP samples with diverse backgrounds and accurate tables, questions, answers, and solutions. To this end, we first extract templates from existing real samples to generate initial problems, ensuring correctness. Then, we adopt an LLM to extend templates and paraphrase problems, obtaining diverse TMWP samples. Furthermore, we find the reasoning annotation is important for solving TMWPs. Therefore, we propose to enrich each solution with illustrative reasoning steps. Through the proposed framework, we construct a high-quality dataset TabMWP-TeLL by adhering to the question types in the TabMWP dataset, and we conduct extensive experiments on a variety of LLMs to demonstrate the effectiveness of TabMWP-TeLL in improving TMWP solving performance. The code and data of this paper are available at: https://github.com/Jason8Kang/TELL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.10144v3">ExpeL: LLM Agents Are Experiential Learners</a></div>
    <div class="paper-meta">
      📅 2024-12-20
      | 💬 Accepted by the 38th Annual AAAI Conference on Artificial Intelligence (AAAI-24)
    </div>
    <details class="paper-abstract">
      The recent surge in research interest in applying large language models (LLMs) to decision-making tasks has flourished by leveraging the extensive world knowledge embedded in LLMs. While there is a growing demand to tailor LLMs for custom decision-making tasks, finetuning them for specific tasks is resource-intensive and may diminish the model's generalization capabilities. Moreover, state-of-the-art language models like GPT-4 and Claude are primarily accessible through API calls, with their parametric weights remaining proprietary and unavailable to the public. This scenario emphasizes the growing need for new methodologies that allow learning from agent experiences without requiring parametric updates. To address these problems, we introduce the Experiential Learning (ExpeL) agent. Our agent autonomously gathers experiences and extracts knowledge using natural language from a collection of training tasks. At inference, the agent recalls its extracted insights and past experiences to make informed decisions. Our empirical results highlight the robust learning efficacy of the ExpeL agent, indicating a consistent enhancement in its performance as it accumulates experiences. We further explore the emerging capabilities and transfer learning potential of the ExpeL agent through qualitative observations and additional experiments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.04834v3">LLM-Based Multi-Agent Systems for Software Engineering: Literature Review, Vision and the Road Ahead</a></div>
    <div class="paper-meta">
      📅 2024-12-20
      | 💬 TOSEM 2030 Special Issue
    </div>
    <details class="paper-abstract">
      Integrating Large Language Models (LLMs) into autonomous agents marks a significant shift in the research landscape by offering cognitive abilities that are competitive with human planning and reasoning. This paper explores the transformative potential of integrating Large Language Models into Multi-Agent (LMA) systems for addressing complex challenges in software engineering (SE). By leveraging the collaborative and specialized abilities of multiple agents, LMA systems enable autonomous problem-solving, improve robustness, and provide scalable solutions for managing the complexity of real-world software projects. In this paper, we conduct a systematic review of recent primary studies to map the current landscape of LMA applications across various stages of the software development lifecycle (SDLC). To illustrate current capabilities and limitations, we perform two case studies to demonstrate the effectiveness of state-of-the-art LMA frameworks. Additionally, we identify critical research gaps and propose a comprehensive research agenda focused on enhancing individual agent capabilities and optimizing agent synergy. Our work outlines a forward-looking vision for developing fully autonomous, scalable, and trustworthy LMA systems, laying the foundation for the evolution of Software Engineering 2.0.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.20695v2">Combining Domain-Specific Models and LLMs for Automated Disease Phenotyping from Survey Data</a></div>
    <div class="paper-meta">
      📅 2024-12-20
    </div>
    <details class="paper-abstract">
      This exploratory pilot study investigated the potential of combining a domain-specific model, BERN2, with large language models (LLMs) to enhance automated disease phenotyping from research survey data. Motivated by the need for efficient and accurate methods to harmonize the growing volume of survey data with standardized disease ontologies, we employed BERN2, a biomedical named entity recognition and normalization model, to extract disease information from the ORIGINS birth cohort survey data. After rigorously evaluating BERN2's performance against a manually curated ground truth dataset, we integrated various LLMs using prompt engineering, Retrieval-Augmented Generation (RAG), and Instructional Fine-Tuning (IFT) to refine the model's outputs. BERN2 demonstrated high performance in extracting and normalizing disease mentions, and the integration of LLMs, particularly with Few Shot Inference and RAG orchestration, further improved accuracy. This approach, especially when incorporating structured examples, logical reasoning prompts, and detailed context, offers a promising avenue for developing tools to enable efficient cohort profiling and data harmonization across large, heterogeneous research datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.15574v1">J-EDI QA: Benchmark for deep-sea organism-specific multimodal LLM</a></div>
    <div class="paper-meta">
      📅 2024-12-20
    </div>
    <details class="paper-abstract">
      Japan Agency for Marine-Earth Science and Technology (JAMSTEC) has made available the JAMSTEC Earth Deep-sea Image (J-EDI), a deep-sea video and image archive (https://www.godac.jamstec.go.jp/jedi/e/index.html). This archive serves as a valuable resource for researchers and scholars interested in deep-sea imagery. The dataset comprises images and videos of deep-sea phenomena, predominantly of marine organisms, but also of the seafloor and physical processes. In this study, we propose J-EDI QA, a benchmark for understanding images of deep-sea organisms using a multimodal large language model (LLM). The benchmark is comprised of 100 images, accompanied by questions and answers with four options by JAMSTEC researchers for each image. The QA pairs are provided in Japanese, and the benchmark assesses the ability to understand deep-sea species in Japanese. In the evaluation presented in this paper, OpenAI o1 achieved a 50% correct response rate. This result indicates that even with the capabilities of state-of-the-art models as of December 2024, deep-sea species comprehension is not yet at an expert level. Further advances in deep-sea species-specific LLMs are therefore required.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.13331v2">Improving Zero-shot LLM Re-Ranker with Risk Minimization</a></div>
    <div class="paper-meta">
      📅 2024-12-20
      | 💬 EMNLP 2024
    </div>
    <details class="paper-abstract">
      In the Retrieval-Augmented Generation (RAG) system, advanced Large Language Models (LLMs) have emerged as effective Query Likelihood Models (QLMs) in an unsupervised way, which re-rank documents based on the probability of generating the query given the content of a document. However, directly prompting LLMs to approximate QLMs inherently is biased, where the estimated distribution might diverge from the actual document-specific distribution. In this study, we introduce a novel framework, $\mathrm{UR^3}$, which leverages Bayesian decision theory to both quantify and mitigate this estimation bias. Specifically, $\mathrm{UR^3}$ reformulates the problem as maximizing the probability of document generation, thereby harmonizing the optimization of query and document generation probabilities under a unified risk minimization objective. Our empirical results indicate that $\mathrm{UR^3}$ significantly enhances re-ranking, particularly in improving the Top-1 accuracy. It benefits the QA tasks by achieving higher accuracy with fewer input documents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.15557v1">MORTAR: Metamorphic Multi-turn Testing for LLM-based Dialogue Systems</a></div>
    <div class="paper-meta">
      📅 2024-12-20
    </div>
    <details class="paper-abstract">
      With the widespread application of LLM-based dialogue systems in daily life, quality assurance has become more important than ever. Recent research has successfully introduced methods to identify unexpected behaviour in single-turn scenarios. However, multi-turn dialogue testing remains underexplored, with the Oracle problem in multi-turn testing posing a persistent challenge for dialogue system developers and researchers. In this paper, we propose MORTAR, a MetamORphic multi-TuRn diAlogue testing appRoach, which mitigates the test oracle problem in the assessment of LLM-based dialogue systems. MORTAR automates the generation of follow-up question-answer (QA) dialogue test cases with multiple dialogue-level perturbations and metamorphic relations. MORTAR employs a novel knowledge graph-based dialogue information model which effectively generates perturbed dialogue test datasets and detects bugs of multi-turn dialogue systems in a low-cost manner. The proposed approach does not require an LLM as a judge, eliminating potential of any biases in the evaluation step. According to the experiment results on multiple LLM-based dialogue systems and comparisons with single-turn metamorphic testing approaches, MORTAR explores more unique bugs in LLM-based dialogue systems, especially for severe bugs that MORTAR detects up to four times more unique bugs than the most effective existing metamorphic testing approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.15487v1">Multi-LLM Text Summarization</a></div>
    <div class="paper-meta">
      📅 2024-12-20
    </div>
    <details class="paper-abstract">
      In this work, we propose a Multi-LLM summarization framework, and investigate two different multi-LLM strategies including centralized and decentralized. Our multi-LLM summarization framework has two fundamentally important steps at each round of conversation: generation and evaluation. These steps are different depending on whether our multi-LLM decentralized summarization is used or centralized. In both our multi-LLM decentralized and centralized strategies, we have k different LLMs that generate diverse summaries of the text. However, during evaluation, our multi-LLM centralized summarization approach leverages a single LLM to evaluate the summaries and select the best one whereas k LLMs are used for decentralized multi-LLM summarization. Overall, we find that our multi-LLM summarization approaches significantly outperform the baselines that leverage only a single LLM by up to 3x. These results indicate the effectiveness of multi-LLM approaches for summarization.
    </details>
</div>
