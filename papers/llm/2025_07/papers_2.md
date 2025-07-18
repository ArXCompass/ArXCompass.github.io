# llm - 2025_07

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
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05566v2">ScaleRTL: Scaling LLMs with Reasoning Data and Test-Time Compute for Accurate RTL Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 Accepted to MLCAD 2025
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have enabled near-human performance on software coding benchmarks, but their effectiveness in RTL code generation remains limited due to the scarcity of high-quality training data. While prior efforts have fine-tuned LLMs for RTL tasks, they do not fundamentally overcome the data bottleneck and lack support for test-time scaling due to their non-reasoning nature. In this work, we introduce ScaleRTL, the first reasoning LLM for RTL coding that scales up both high-quality reasoning data and test-time compute. Specifically, we curate a diverse set of long chain-of-thought reasoning traces averaging 56K tokens each, resulting in a dataset of 3.5B tokens that captures rich RTL knowledge. Fine-tuning a general-purpose reasoning model on this corpus yields ScaleRTL that is capable of deep RTL reasoning. Subsequently, we further enhance the performance of ScaleRTL through a novel test-time scaling strategy that extends the reasoning process via iteratively reflecting on and self-correcting previous reasoning steps. Experimental results show that ScaleRTL achieves state-of-the-art performance on VerilogEval and RTLLM, outperforming 18 competitive baselines by up to 18.4% on VerilogEval and 12.7% on RTLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11742v1">CRABS: A syntactic-semantic pincer strategy for bounding LLM interpretation of Python notebooks</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 Preprint. Accepted to COLM 2025
    </div>
    <details class="paper-abstract">
      Recognizing the information flows and operations comprising data science and machine learning Python notebooks is critical for evaluating, reusing, and adapting notebooks for new tasks. Investigating a notebook via re-execution often is impractical due to the challenges of resolving data and software dependencies. While Large Language Models (LLMs) pre-trained on large codebases have demonstrated effectiveness in understanding code without running it, we observe that they fail to understand some realistic notebooks due to hallucinations and long-context challenges. To address these issues, we propose a notebook understanding task yielding an information flow graph and corresponding cell execution dependency graph for a notebook, and demonstrate the effectiveness of a pincer strategy that uses limited syntactic analysis to assist full comprehension of the notebook using an LLM. Our Capture and Resolve Assisted Bounding Strategy (CRABS) employs shallow syntactic parsing and analysis of the abstract syntax tree (AST) to capture the correct interpretation of a notebook between lower and upper estimates of the inter-cell I/O sets, then uses an LLM to resolve remaining ambiguities via cell-by-cell zero-shot learning, thereby identifying the true data inputs and outputs of each cell. We evaluate and demonstrate the effectiveness of our approach using an annotated dataset of 50 representative, highly up-voted Kaggle notebooks that together represent 3454 actual cell inputs and outputs. The LLM correctly resolves 1397 of 1425 (98%) ambiguities left by analyzing the syntactic structure of these notebooks. Across 50 notebooks, CRABS achieves average F1 scores of 98% identifying cell-to-cell information flows and 99% identifying transitive cell execution dependencies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.05102v2">You Can REST Now: Automated REST API Documentation and Testing via LLM-Assisted Request Mutations</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      REST APIs are prevalent among web service implementations, easing interoperability through the HTTP protocol. API testers and users exploit the widely adopted OpenAPI Specification (OAS), a machine-readable standard to document REST APIs. However, documenting APIs is a time-consuming and error-prone task, and existing documentation is not always complete, publicly accessible, or up-to-date. This situation limits the efficiency of testing tools and hinders human comprehension. Large Language Models (LLMs) offer the potential to automatically infer API documentation, using their colossal training data. In this paper, we present RESTSpecIT, the first automated approach that infers documentation and performs black-box testing of REST APIs by leveraging LLMs. Our approach requires minimal user input compared to state-of-the-art tools; Given an API name and an LLM access key, RESTSpecIT generates API request seeds and mutates them with data returned by the LLM. The tool then analyzes API responses for documentation inference and testing purposes. RESTSpecIT utilizes an in-context prompt masking strategy, requiring no prior model fine-tuning. We evaluate the quality of our tool with three state-of-the-art LLMs: DeepSeek V3, GPT-4.1, and GPT-3.5. Our evaluation demonstrates that RESTSpecIT can (1) infer documentation with 88.62% of routes and 89.25% of query parameters found on average, (2) discover undocumented API data, (3) operate efficiently (in terms of model costs, requests sent, runtime), and (4) assist REST API testing by uncovering server errors and generating valid OpenAPI Specification inputs for testing tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16069v2">Rolling the DICE on Idiomaticity: How LLMs Fail to Grasp Context</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 ACL 2025
    </div>
    <details class="paper-abstract">
      Human processing of idioms relies on understanding the contextual sentences in which idioms occur, as well as language-intrinsic features such as frequency and speaker-intrinsic factors like familiarity. While LLMs have shown high performance on idiomaticity detection tasks, this success may be attributed to reasoning shortcuts in existing datasets. To this end, we construct a novel, controlled contrastive dataset designed to test whether LLMs can effectively use context to disambiguate idiomatic meaning. Additionally, we explore how collocational frequency and sentence probability influence model performance. Our findings reveal that LLMs often fail to resolve idiomaticity when it is required to attend to the surrounding context, and that models perform better on sentences that have higher likelihood. The collocational frequency of expressions also impacts performance. We make our code and dataset publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10323v3">ELFuzz: Efficient Input Generation via LLM-driven Synthesis Over Fuzzer Space</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 Accepted by USENIX Security'25 Cycle 2
    </div>
    <details class="paper-abstract">
      Generation-based fuzzing produces appropriate testing cases according to specifications of input grammars and semantic constraints to test systems and software. However, these specifications require significant manual efforts to construct. This paper proposes a new approach, ELFuzz (Evolution Through Large Language Models for Fuzzing), that automatically synthesizes generation-based fuzzers tailored to a system under test (SUT) via LLM-driven synthesis over fuzzer space. At a high level, it starts with minimal seed fuzzers and propels the synthesis by fully automated LLM-driven evolution with coverage guidance. Compared to previous approaches, ELFuzz can 1) seamlessly scale to SUTs of real-world sizes -- up to 1,791,104 lines of code in our evaluation -- and 2) synthesize efficient fuzzers that catch interesting grammatical structures and semantic constraints in a human-understandable way. Our evaluation compared ELFuzz with specifications manually written by domain experts and synthesized by state-of-the-art approaches. It shows that ELFuzz achieves up to 434.8% more coverage and triggers up to 174.0% more artificially injected bugs. We also used ELFuzz to conduct a real-world fuzzing campaign on the newest version of cvc5 for 14 days, and encouragingly, it found five 0-day bugs (three are exploitable). Moreover, we conducted an ablation study, which shows that the fuzzer space model, the key component of ELFuzz, contributes the most (up to 62.5%) to the effectiveness of ELFuzz. Further analysis of the fuzzers synthesized by ELFuzz confirms that they catch interesting grammatical structures and semantic constraints in a human-understandable way. The results present the promising potential of ELFuzz for more automated, efficient, and extensible input generation for fuzzing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11633v1">General Modular Harness for LLM Agents in Multi-Turn Gaming Environments</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 8 pages, ICML MAS workshop
    </div>
    <details class="paper-abstract">
      We introduce a modular harness design for LLM agents that composes of perception, memory, and reasoning components, enabling a single LLM or VLM backbone to tackle a wide spectrum of multi turn gaming environments without domain-specific engineering. Using classic and modern game suites as low-barrier, high-diversity testbeds, our framework provides a unified workflow for analyzing how each module affects performance across dynamic interactive settings. Extensive experiments demonstrate that the harness lifts gameplay performance consistently over un-harnessed baselines and reveals distinct contribution patterns, for example, memory dominates in long-horizon puzzles while perception is critical in vision noisy arcades. These findings highlight the effectiveness of our modular harness design in advancing general-purpose agent, given the familiarity and ubiquity of games in everyday human experience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09598v3">How Hungry is AI? Benchmarking Energy, Water, and Carbon Footprint of LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      This paper introduces a novel infrastructure-aware benchmarking framework for quantifying the environmental footprint of LLM inference across 30 state-of-the-art models as deployed in commercial data centers. Our framework combines public API performance data with region-specific environmental multipliers and statistical inference of hardware configurations. We additionally utilize cross-efficiency Data Envelopment Analysis (DEA) to rank models by performance relative to environmental cost. Our results show that o3 and DeepSeek-R1 emerge as the most energy-intensive models, consuming over 33 Wh per long prompt, more than 70 times the consumption of GPT-4.1 nano, and that Claude-3.7 Sonnet ranks highest in eco-efficiency. While a single short GPT-4o query consumes 0.42 Wh, scaling this to 700 million queries/day results in substantial annual environmental impacts. These include electricity use comparable to 35,000 U.S. homes, freshwater evaporation matching the annual drinking needs of 1.2 million people, and carbon emissions requiring a Chicago-sized forest to offset. These findings illustrate a growing paradox: Although AI is becoming cheaper and faster, its global adoption drives disproportionate resource consumption. Our study provides a standardized, empirically grounded methodology for benchmarking the sustainability of LLM deployments, laying a foundation for future environmental accountability in AI development and sustainability standards.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10540v1">Fusing LLM Capabilities with Routing Data</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has created a vibrant ecosystem of diverse architectures, each with unique strengths due to differences in design, training data, and objectives. However, most applications still rely on a single backend model, limiting coverage of capabilities and leading to inefficiencies in performance and token cost when tackling complex tasks. We highlight an underexploited opportunity: LLM routing data, produced when hosting platforms route diverse queries to different models, which can reveal comparative strengths across tasks. To address this, we propose FusionBench, a comprehensive routing benchmark covering 14 tasks across five domains with 20 open-source LLMs (8B to 671B parameters), capturing 103M tokens and summarizing reusable thought templates from top models. Building on this, we introduce FusionFactory, a systematic fusion framework with three levels: (1) query-level fusion, tailoring routers for each query using both direct responses and reasoning-augmented outputs; (2) thought-level fusion, leveraging abstract templates derived from top-performing LLMs' answers to similar queries; and (3) model-level fusion, transferring capabilities between models via distillation, using top responses or highest judge scores as training data. Experiments show FusionFactory consistently outperforms the best individual LLM across all 14 benchmarks, with optimal fusion configurations varying by benchmark, demonstrating the value of systematic LLM fusion in harnessing complementary strengths and improving overall performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10535v1">CodeJudgeBench: Benchmarking LLM-as-a-Judge for Coding Tasks</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 Dataset is available at https://huggingface.co/datasets/mattymchen/codejudgebench
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have significantly advanced the state-of-the-art in various coding tasks. Beyond directly answering user queries, LLMs can also serve as judges, assessing and comparing the quality of responses generated by other models. Such an evaluation capability is crucial both for benchmarking different LLMs and for improving response quality through response ranking. However, despite the growing adoption of the LLM-as-a-Judge paradigm, its effectiveness in coding scenarios remains underexplored due to the absence of dedicated benchmarks. To address this gap, we introduce CodeJudgeBench, a benchmark explicitly designed to evaluate the performance of LLM-as-a-Judge models across three critical coding tasks: code generation, code repair, and unit test generation. Through comprehensive benchmarking of 26 LLM-as-a-Judge models, we find that recent thinking models significantly outperform non-thinking models on our carefully designed code judging tasks. Notably, even relatively small thinking models, such as Qwen3-8B, can outperform specially trained LLM-as-a-Judge models up to 70B in size. Nevertheless, all models still exhibit significant randomness in their judgment of coding tasks. For pairwise judging tasks, simply changing the order in which responses are presented can substantially impact accuracy. In addition, when judging code and unit tests written by different LLMs, LLM-as-a-Judge models also show variance in performance. This sensitivity raises concerns about the reliability and consistency of LLM-as-a-Judge in coding scenarios. Lastly, we study optimal prompting strategies for LLM-as-a-Judge. We find that using pair-wise comparison outperforms scalar point-wise judging. Furthermore, retaining comments and reasoning in the full, unprocessed LLM response leads to improved judge performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10445v1">Referential ambiguity and clarification requests: comparing human and LLM behaviour</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      In this work we examine LLMs' ability to ask clarification questions in task-oriented dialogues that follow the asynchronous instruction-giver/instruction-follower format. We present a new corpus that combines two existing annotations of the Minecraft Dialogue Corpus -- one for reference and ambiguity in reference, and one for SDRT including clarifications -- into a single common format providing the necessary information to experiment with clarifications and their relation to ambiguity. With this corpus we compare LLM actions with original human-generated clarification questions, examining how both humans and LLMs act in the case of ambiguity. We find that there is only a weak link between ambiguity and humans producing clarification questions in these dialogues, and low correlation between humans and LLMs. Humans hardly ever produce clarification questions for referential ambiguity, but often do so for task-based uncertainty. Conversely, LLMs produce more clarification questions for referential ambiguity, but less so for task uncertainty. We question if LLMs' ability to ask clarification questions is predicated on their recent ability to simulate reasoning, and test this with different reasoning approaches, finding that reasoning does appear to increase question frequency and relevancy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10427v1">Towards Emotion Co-regulation with LLM-powered Socially Assistive Robots: Integrating LLM Prompts and Robotic Behaviors to Support Parent-Neurodivergent Child Dyads</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 Submission for the IROS 2025 conference
    </div>
    <details class="paper-abstract">
      Socially Assistive Robotics (SAR) has shown promise in supporting emotion regulation for neurodivergent children. Recently, there has been increasing interest in leveraging advanced technologies to assist parents in co-regulating emotions with their children. However, limited research has explored the integration of large language models (LLMs) with SAR to facilitate emotion co-regulation between parents and children with neurodevelopmental disorders. To address this gap, we developed an LLM-powered social robot by deploying a speech communication module on the MiRo-E robotic platform. This supervised autonomous system integrates LLM prompts and robotic behaviors to deliver tailored interventions for both parents and neurodivergent children. Pilot tests were conducted with two parent-child dyads, followed by a qualitative analysis. The findings reveal MiRo-E's positive impacts on interaction dynamics and its potential to facilitate emotion regulation, along with identified design and technical challenges. Based on these insights, we provide design implications to advance the future development of LLM-powered SAR for mental health applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10392v1">Zorse: Optimizing LLM Training Efficiency on Heterogeneous GPU Clusters</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) require vast amounts of GPU compute to train, but limited availability and high costs of GPUs make homogeneous clusters impractical for many organizations. Instead, assembling heterogeneous clusters by pooling together GPUs of different generations allows them to achieve higher aggregate compute and make use of all available GPUs. However, training on heterogeneous clusters presents several challenges, including load balancing across GPUs, optimizing memory usage to accommodate varying memory capacities, and ensuring communication-efficient training over diverse network interconnects potentially spanning multiple datacenters. In this paper, we make the case that efficient training on heterogeneous clusters requires (1) the integration of pipeline parallelism and data parallelism in a manner that is both communication- and memory-efficient, and (2) a more adaptable configuration of pipeline and data parallelism, which includes the capability to flexibly partition GPUs into asymmetric pipeline parallel stages and to incorporate heterogeneous GPUs within the same data parallelism group. We propose Zorse, the first system to unify all these capabilities while incorporating a planner that automatically configures training strategies for a given workload. Our evaluation shows that Zorse significantly outperforms state-of-the-art systems in heterogeneous training scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11168v3">Bypassing LLM Guardrails: An Empirical Analysis of Evasion Attacks against Prompt Injection and Jailbreak Detection Systems</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 14 pages, 5 figures, 11 tables. To be published in LLMSec 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) guardrail systems are designed to protect against prompt injection and jailbreak attacks. However, they remain vulnerable to evasion techniques. We demonstrate two approaches for bypassing LLM prompt injection and jailbreak detection systems via traditional character injection methods and algorithmic Adversarial Machine Learning (AML) evasion techniques. Through testing against six prominent protection systems, including Microsoft's Azure Prompt Shield and Meta's Prompt Guard, we show that both methods can be used to evade detection while maintaining adversarial utility achieving in some instances up to 100% evasion success. Furthermore, we demonstrate that adversaries can enhance Attack Success Rates (ASR) against black-box targets by leveraging word importance ranking computed by offline white-box models. Our findings reveal vulnerabilities within current LLM protection mechanisms and highlight the need for more robust guardrail systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10382v1">Leveraging RAG-LLMs for Urban Mobility Simulation and Analysis</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      With the rise of smart mobility and shared e-mobility services, numerous advanced technologies have been applied to this field. Cloud-based traffic simulation solutions have flourished, offering increasingly realistic representations of the evolving mobility landscape. LLMs have emerged as pioneering tools, providing robust support for various applications, including intelligent decision-making, user interaction, and real-time traffic analysis. As user demand for e-mobility continues to grow, delivering comprehensive end-to-end solutions has become crucial. In this paper, we present a cloud-based, LLM-powered shared e-mobility platform, integrated with a mobile application for personalized route recommendations. The optimization module is evaluated based on travel time and cost across different traffic scenarios. Additionally, the LLM-powered RAG framework is evaluated at the schema level for different users, using various evaluation methods. Schema-level RAG with XiYanSQL achieves an average execution accuracy of 0.81 on system operator queries and 0.98 on user queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.06238v2">EVOLvE: Evaluating and Optimizing LLMs For In-Context Exploration</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 28 pages. Published at ICML 2025
    </div>
    <details class="paper-abstract">
      Despite their success in many domains, large language models (LLMs) remain under-studied in scenarios requiring optimal decision-making under uncertainty. This is crucial as many real-world applications, ranging from personalized recommendations to healthcare interventions, demand that LLMs not only predict but also actively learn to make optimal decisions through exploration. In this work, we measure LLMs' (in)ability to make optimal decisions in bandits, a state-less reinforcement learning setting relevant to many applications. We develop a comprehensive suite of environments, including both context-free and contextual bandits with varying task difficulties, to benchmark LLMs' performance. Motivated by the existence of optimal exploration algorithms, we propose efficient ways to integrate this algorithmic knowledge into LLMs: by providing explicit algorithm-guided support during inference; and through algorithm distillation via in-context demonstrations and fine-tuning, using synthetic data generated from these algorithms. Impressively, these techniques allow us to achieve superior exploration performance with smaller models, surpassing larger models on various tasks. We conducted an extensive ablation study to shed light on various factors, such as task difficulty and data representation, that influence the efficiency of LLM exploration. Additionally, we conduct a rigorous analysis of the LLM's exploration efficiency using the concept of regret, linking its ability to explore to the model size and underlying algorithm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02620v2">FlowSpec: Continuous Pipelined Speculative Decoding for Efficient Distributed LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 16 pages, and the last 3 are appendix
    </div>
    <details class="paper-abstract">
      Distributed inference serves as a promising approach to enabling the inference of large language models (LLMs) at the network edge. It distributes the inference process to multiple devices to ensure that the LLMs can fit into the device memory. Recent pipeline-based approaches have the potential to parallelize communication and computation, which helps reduce inference latency. However, the benefit diminishes when the inference request at the network edge is sparse, where pipeline is typically at low utilization. To enable efficient distributed LLM inference at the edge, we propose \textbf{FlowSpec}, a pipeline-parallel tree-based speculative decoding framework. FlowSpec incorporates three key mechanisms to improve decoding efficiency: 1) score-based step-wise verification prioritizes more important draft tokens to bring earlier accpeted tokens; 2) efficient draft management to prune invalid tokens while maintaining correct causal relationship during verification; 3) dynamic draft expansion strategies to supply high-quality speculative inputs. These techniques work in concert to enhance both pipeline utilization and speculative efficiency. We evaluate FlowSpec on a real-world testbed with other baselines. Experimental results demonstrate that our proposed framework significantly improves inference speed across diverse models and configurations, achieving speedup ratios 1.28$\times$-1.79$\times$ compared to baselines. Our code is publicly available at \href{https://github.com/Leosang-lx/FlowSpec#}{https://github.com/Leosang-lx/FlowSpec\#}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10281v1">Toward Real-World Table Agents: Capabilities, Workflows, and Design Principles for LLM-based Table Intelligence</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      Tables are fundamental in domains such as finance, healthcare, and public administration, yet real-world table tasks often involve noise, structural heterogeneity, and semantic complexity--issues underexplored in existing research that primarily targets clean academic datasets. This survey focuses on LLM-based Table Agents, which aim to automate table-centric workflows by integrating preprocessing, reasoning, and domain adaptation. We define five core competencies--C1: Table Structure Understanding, C2: Table and Query Semantic Understanding, C3: Table Retrieval and Compression, C4: Executable Reasoning with Traceability, and C5: Cross-Domain Generalization--to analyze and compare current approaches. In addition, a detailed examination of the Text-to-SQL Agent reveals a performance gap between academic benchmarks and real-world scenarios, especially for open-source models. Finally, we provide actionable insights to improve the robustness, generalization, and efficiency of LLM-based Table Agents in practical settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10200v1">Natural Language-based Assessment of L2 Oral Proficiency using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 Accepted for the 10th Workshop on Speech and Language Technology in Education (SLaTE 2025)
    </div>
    <details class="paper-abstract">
      Natural language-based assessment (NLA) is an approach to second language assessment that uses instructions - expressed in the form of can-do descriptors - originally intended for human examiners, aiming to determine whether large language models (LLMs) can interpret and apply them in ways comparable to human assessment. In this work, we explore the use of such descriptors with an open-source LLM, Qwen 2.5 72B, to assess responses from the publicly available S&I Corpus in a zero-shot setting. Our results show that this approach - relying solely on textual information - achieves competitive performance: while it does not outperform state-of-the-art speech LLMs fine-tuned for the task, it surpasses a BERT-based model trained specifically for this purpose. NLA proves particularly effective in mismatched task settings, is generalisable to other data types and languages, and offers greater interpretability, as it is grounded in clearly explainable, widely applicable language descriptors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10177v1">Abusive text transformation using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      Although Large Language Models (LLMs) have demonstrated significant advancements in natural language processing tasks, their effectiveness in the classification and transformation of abusive text into non-abusive versions remains an area for exploration. In this study, we aim to use LLMs to transform abusive text (tweets and reviews) featuring hate speech and swear words into non-abusive text, while retaining the intent of the text. We evaluate the performance of two state-of-the-art LLMs, such as Gemini, GPT-4o, DeekSeek and Groq, on their ability to identify abusive text. We them to transform and obtain a text that is clean from abusive and inappropriate content but maintains a similar level of sentiment and semantics, i.e. the transformed text needs to maintain its message. Afterwards, we evaluate the raw and transformed datasets with sentiment analysis and semantic analysis. Our results show Groq provides vastly different results when compared with other LLMs. We have identified similarities between GPT-4o and DeepSeek-V3.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14403v3">Unearthing Gems from Stones: Policy Optimization with Negative Sample Augmentation for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      Recent advances in reasoning language models have witnessed a paradigm shift from short to long CoT pattern. Given the substantial computational cost of rollouts in long CoT models, maximizing the utility of fixed training datasets becomes crucial. Our analysis reveals that negative responses contain valuable components such as self-reflection and error-correction steps, yet primary existing methods either completely discard negative samples (RFT) or apply equal penalization across all tokens (RL), failing to leverage these potential learning signals. In light of this, we propose Behavior Constrained Policy Gradient with Negative Sample Augmentation (BCPG-NSA), a fine-grained offline RL framework that encompasses three stages: 1) sample segmentation, 2) consensus-based step correctness assessment combining LLM and PRM judgers, and 3) policy optimization with NSA designed to effectively mine positive steps within negative samples. Experimental results show that BCPG-NSA outperforms baselines on several challenging math/coding reasoning benchmarks using the same training dataset, achieving improved sample efficiency and demonstrating robustness and scalability when extended to multiple iterations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10155v1">Task-Based Flexible Feature Distillation for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      Knowledge Distillation (KD) in general and feature distillation in particular are promising techniques for reducing the high computational demand of large language models (LLMs). However, traditional feature KD methods typically assume that the teacher and the student share the same hidden size, limiting the flexibility of the student's architecture. A common solution to this problem involves training a linear projector to align their feature spaces, but this introduces additional parameters that must be learned from scratch and often degrades performance on downstream tasks, especially in generative settings. To address this issue, in this work, we propose a novel task-based feature distillation method that enables knowledge transfer between teacher and student models with different hidden layer dimensions, without introducing any new parameters. Leveraging the insight that only a subset of LLM components contribute significantly to a specific downstream task, our approach identifies the most task-relevant hidden units in the teacher and directly distills their activations to the student. Our method is flexible and easily integrates with other distillation frameworks. Empirical results show consistent improvements over prior approaches across diverse tasks, including classification, instruction-following, and summarization, achieving up to a 3\% performance gain over the linear projection baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10150v1">Past-Future Scheduler for LLM Serving under SLA Guarantees</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 Accepted to ASPLOS 2025
    </div>
    <details class="paper-abstract">
      The exploration and application of Large Language Models (LLMs) is thriving. To reduce deployment costs, continuous batching has become an essential feature in current service frameworks. The effectiveness of continuous batching relies on an accurate estimate of the memory requirements of requests. However, due to the diversity in request output lengths, existing frameworks tend to adopt aggressive or conservative schedulers, which often result in significant overestimation or underestimation of memory consumption. Consequently, they suffer from harmful request evictions or prolonged queuing times, failing to achieve satisfactory throughput under strict Service Level Agreement (SLA) guarantees (a.k.a. goodput), across various LLM application scenarios with differing input-output length distributions. To address this issue, we propose a novel Past-Future scheduler that precisely estimates the peak memory resources required by the running batch via considering the historical distribution of request output lengths and calculating memory occupancy at each future time point. It adapts to applications with all types of input-output length distributions, balancing the trade-off between request queuing and harmful evictions, thereby consistently achieving better goodput. Furthermore, to validate the effectiveness of the proposed scheduler, we developed a high-performance LLM serving framework, LightLLM, that implements the Past-Future scheduler. Compared to existing aggressive or conservative schedulers, LightLLM demonstrates superior goodput, achieving up to 2-3$\times$ higher goodput than other schedulers under heavy loads. LightLLM is open source to boost the research in such direction (https://github.com/ModelTC/lightllm).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10134v1">FRSICL: LLM-Enabled In-Context Learning Flight Resource Allocation for Fresh Data Collection in UAV-Assisted Wildfire Monitoring</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 8 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Unmanned Aerial Vehicles (UAVs) are vital for public safety, particularly in wildfire monitoring, where early detection minimizes environmental impact. In UAV-Assisted Wildfire Monitoring (UAWM) systems, joint optimization of sensor transmission scheduling and velocity is critical for minimizing Age of Information (AoI) from stale sensor data. Deep Reinforcement Learning (DRL) has been used for such optimization; however, its limitations such as low sampling efficiency, simulation-to-reality gaps, and complex training render it unsuitable for time-critical applications like wildfire monitoring. This paper introduces a new online Flight Resource Allocation scheme based on LLM-Enabled In-Context Learning (FRSICL) to jointly optimize the UAV's flight control and data collection schedule along the trajectory in real time, thereby asymptotically minimizing the average AoI across ground sensors. In contrast to DRL, FRSICL generates data collection schedules and controls velocity using natural language task descriptions and feedback from the environment, enabling dynamic decision-making without extensive retraining. Simulation results confirm the effectiveness of the proposed FRSICL compared to Proximal Policy Optimization (PPO) and Nearest-Neighbor baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10124v1">Could you be wrong: Debiasing LLMs using a metacognitive prompt for improving human decision making</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 12 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Identifying bias in LLMs is ongoing. Because they are still in development, what is true today may be false tomorrow. We therefore need general strategies for debiasing that will outlive current models. Strategies developed for debiasing human decision making offer one promising approach as they incorporate an LLM-style prompt intervention designed to bring latent knowledge into awareness during decision making. LLMs trained on vast amounts of information contain information about potential biases, counter-arguments, and contradictory evidence, but that information may only be brought to bear if prompted. Metacognitive prompts developed in the human decision making literature are designed to achieve this, and as I demonstrate here, they show promise with LLMs. The prompt I focus on here is "could you be wrong?" Following an LLM response, this prompt leads LLMs to produce additional information, including why they answered as they did, errors, biases, contradictory evidence, and alternatives, none of which were apparent in their initial response. Indeed, this metaknowledge often reveals that how LLMs and users interpret prompts are not aligned. Here I demonstrate this prompt using a set of questions taken from recent articles about LLM biases, including implicit discriminatory biases and failures of metacognition. "Could you be wrong" prompts the LLM to identify its own biases and produce cogent metacognitive reflection. I also present another example involving convincing but incomplete information, which is readily corrected by the metacognitive prompt. In sum, this work argues that human psychology offers a new avenue for prompt engineering, leveraging a long history of effective prompt-based improvements to human decision making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00200v2">Structuring Radiology Reports: Challenging LLMs with Lightweight Models</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      Radiology reports are critical for clinical decision-making but often lack a standardized format, limiting both human interpretability and machine learning (ML) applications. While large language models (LLMs) have shown strong capabilities in reformatting clinical text, their high computational requirements, lack of transparency, and data privacy concerns hinder practical deployment. To address these challenges, we explore lightweight encoder-decoder models (<300M parameters)-specifically T5 and BERT2BERT-for structuring radiology reports from the MIMIC-CXR and CheXpert Plus datasets. We benchmark these models against eight open-source LLMs (1B-70B), adapted using prefix prompting, in-context learning (ICL), and low-rank adaptation (LoRA) finetuning. Our best-performing lightweight model outperforms all LLMs adapted using prompt-based techniques on a human-annotated test set. While some LoRA-finetuned LLMs achieve modest gains over the lightweight model on the Findings section (BLEU 6.4%, ROUGE-L 4.8%, BERTScore 3.6%, F1-RadGraph 1.1%, GREEN 3.6%, and F1-SRR-BERT 4.3%), these improvements come at the cost of substantially greater computational resources. For example, LLaMA-3-70B incurred more than 400 times the inference time, cost, and carbon emissions compared to the lightweight model. These results underscore the potential of lightweight, task-specific models as sustainable and privacy-preserving solutions for structuring clinical text in resource-constrained healthcare settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10069v1">ElasticMM: Efficient Multimodal LLMs Serving with Elastic Multimodal Parallelism</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      Multimodal large language models (MLLMs) extend LLMs to handle images, videos, and audio by incorporating feature extractors and projection modules. However, these additional components -- combined with complex inference pipelines and heterogeneous workloads -- introduce significant inference overhead. Therefore, efficiently serving MLLMs remains a major challenge. Current tightly coupled serving architectures struggle to distinguish between mixed request types or adapt parallelism strategies to different inference stages, leading to increased time-to-first-token (TTFT) latency and poor resource utilization. To address this, we propose Elastic Multimodal Parallelism (EMP), a new serving paradigm that elastically adapts to resource heterogeneity across request types and inference stages. Building upon EMP, we develop ElasticMM, an MLLM serving system that (1) separates requests into independent modality groups with dynamic resource allocation via a modality-aware load balancer; (2) decouples inference stages and enables parallelism adjustment and adaptive scaling via elastic partition scheduling; and (3) improves inference efficiency through unified multimodal prefix caching and non-blocking encoding. Experiments on diverse real-world datasets show that ElasticMM outperforms state-of-the-art (SOTA) serving systems, reducing TTFT by up to 4.2x and achieving 3.2-4.5x higher throughput while meeting service-level objectives (SLOs).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10054v1">Explicit Vulnerability Generation with LLMs: An Investigation Beyond Adversarial Attacks</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 Accepted to ICSME 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used as code assistants, yet their behavior when explicitly asked to generate insecure code remains poorly understood. While prior research has focused on unintended vulnerabilities or adversarial prompting techniques, this study examines a more direct threat scenario: open-source LLMs generating vulnerable code when prompted either directly or indirectly. We propose a dual experimental design: (1) Dynamic Prompting, which systematically varies vulnerability type, user persona, and directness across structured templates; and (2) Reverse Prompting, which derives prompts from real vulnerable code samples to assess vulnerability reproduction accuracy. We evaluate three open-source 7B-parameter models (Qwen2, Mistral, and Gemma) using ESBMC static analysis to assess both the presence of vulnerabilities and the correctness of the generated vulnerability type. Results show all models frequently produce vulnerable outputs, with Qwen2 achieving highest correctness rates. User persona significantly affects success, where student personas achieved higher vulnerability rates than professional roles, while direct prompts were marginally more effective. Vulnerability reproduction followed an inverted-U pattern with cyclomatic complexity, peaking at moderate ranges. Our findings expose limitations of safety mechanisms in open-source models, particularly for seemingly benign educational requests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15902v2">IPAD: Inverse Prompt for AI Detection -- A Robust and Explainable LLM-Generated Text Detector</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have attained human-level fluency in text generation, which complicates the distinction between human-written and LLM-generated texts. This increases the risk of misuse and highlights the need for reliable detectors. Yet, existing detectors exhibit poor robustness on out-of-distribution (OOD) data and attacked data, which is critical for real-world scenarios. Also, they struggle to provide interpretable evidence to support their decisions, thus undermining the reliability. In light of these challenges, we propose IPAD (Inverse Prompt for AI Detection), a novel framework consisting of a Prompt Inverter that identifies predicted prompts that could have generated the input text, and two Distinguishers that examine the probability that the input texts align with the predicted prompts. Empirical evaluations demonstrate that IPAD outperforms the strongest baselines by 9.05% (Average Recall) on in-distribution data, 12.93% (AUROC) on out-of-distribution (OOD) data, and 5.48% (AUROC) on attacked data. IPAD also performs robustly on structured datasets. Furthermore, an interpretability assessment is conducted to illustrate that IPAD enhances the AI detection trustworthiness by allowing users to directly examine the decision-making evidence, which provides interpretable support for its state-of-the-art detection results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10024v1">Qualitative Study for LLM-assisted Design Study Process: Strategies, Challenges, and Roles</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      Design studies aim to create visualization solutions for real-world problems of different application domains. Recently, the emergence of large language models (LLMs) has introduced new opportunities to enhance the design study process, providing capabilities such as creative problem-solving, data handling, and insightful analysis. However, despite their growing popularity, there remains a lack of systematic understanding of how LLMs can effectively assist researchers in visualization-specific design studies. In this paper, we conducted a multi-stage qualitative study to fill this gap, involving 30 design study researchers from diverse backgrounds and expertise levels. Through in-depth interviews and carefully-designed questionnaires, we investigated strategies for utilizing LLMs, the challenges encountered, and the practices used to overcome them. We further compiled and summarized the roles that LLMs can play across different stages of the design study process. Our findings highlight practical implications to inform visualization practitioners, and provide a framework for leveraging LLMs to enhance the design study process in visualization research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07498v2">Teaching LLM to Reason: Reinforcement Learning from Algorithmic Problems without Code</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      Enhancing reasoning capabilities remains a central focus in the LLM reasearch community. A promising direction involves requiring models to simulate code execution step-by-step to derive outputs for given inputs. However, as code is often designed for large-scale systems, direct application leads to over-reliance on complex data structures and algorithms, even for simple cases, resulting in overfitting to algorithmic patterns rather than core reasoning structures. To address this, we propose TeaR, which aims at teaching LLMs to reason better. TeaR leverages careful data curation and reinforcement learning to guide models in discovering optimal reasoning paths through code-related tasks, thereby improving general reasoning abilities. We conduct extensive experiments using two base models and three long-CoT distillation models, with model sizes ranging from 1.5 billion to 32 billion parameters, and across 17 benchmarks spanning Math, Knowledge, Code, and Logical Reasoning. The results consistently show significant performance improvements. Notably, TeaR achieves a 35.9% improvement on Qwen2.5-7B and 5.9% on R1-Distilled-7B.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12185v3">EVALOOP: Assessing LLM Robustness in Programming from a Self-consistency Perspective</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 20 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Assessing the programming capabilities of Large Language Models (LLMs) is crucial for their effective use in software engineering. Current evaluations, however, predominantly measure the accuracy of generated code on static benchmarks, neglecting the critical aspect of model robustness during programming tasks. While adversarial attacks offer insights on model robustness, their effectiveness is limited and evaluation could be constrained. Current adversarial attack methods for robustness evaluation yield inconsistent results, struggling to provide a unified evaluation across different LLMs. We introduce EVALOOP, a novel assessment framework that evaluate the robustness from a self-consistency perspective, i.e., leveraging the natural duality inherent in popular software engineering tasks, e.g., code generation and code summarization. EVALOOP initiates a self-contained feedback loop: an LLM generates output (e.g., code) from an input (e.g., natural language specification), and then use the generated output as the input to produce a new output (e.g., summarizes that code into a new specification). EVALOOP repeats the process to assess the effectiveness of EVALOOP in each loop. This cyclical strategy intrinsically evaluates robustness without rely on any external attack setups, providing a unified metric to evaluate LLMs' robustness in programming. We evaluate 16 prominent LLMs (e.g., GPT-4.1, O4-mini) on EVALOOP and found that EVALOOP typically induces a 5.01%-19.31% absolute drop in pass@1 performance within ten loops. Intriguingly, robustness does not always align with initial performance (i.e., one-time query); for instance, GPT-3.5-Turbo, despite superior initial code generation compared to DeepSeek-V2, demonstrated lower robustness over repeated evaluation loop.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13640v2">Qorgau: Evaluating LLM Safety in Kazakh-Russian Bilingual Contexts</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are known to have the potential to generate harmful content, posing risks to users. While significant progress has been made in developing taxonomies for LLM risks and safety evaluation prompts, most studies have focused on monolingual contexts, primarily in English. However, language- and region-specific risks in bilingual contexts are often overlooked, and core findings can diverge from those in monolingual settings. In this paper, we introduce Qorgau, a novel dataset specifically designed for safety evaluation in Kazakh and Russian, reflecting the unique bilingual context in Kazakhstan, where both Kazakh (a low-resource language) and Russian (a high-resource language) are spoken. Experiments with both multilingual and language-specific LLMs reveal notable differences in safety performance, emphasizing the need for tailored, region-specific datasets to ensure the responsible and safe deployment of LLMs in countries like Kazakhstan. Warning: this paper contains example data that may be offensive, harmful, or biased.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13820v2">InstCache: A Predictive Cache for LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      The revolutionary capabilities of Large Language Models (LLMs) are attracting rapidly growing popularity and leading to soaring user requests to inference serving systems. Caching techniques, which leverage data reuse to reduce computation, offer opportunities to optimize the performance of LLM inference engines. On the one hand, the low-level key-value (KV) cache working at the token level is widely adopted, albeit it incurs significant overhead as request volume grows. On the other hand, instruction-level caching, which stores full instruction-response pairs, is expected to play an increasingly crucial role. However, the high variability in the content and length of instructions make it rare for identical instructions to recur within a short time window, presenting challenges for effective caching instruction-response pairs. To address this challenge, we propose InstCache, a predictive caching mechanism for LLM serving systems. Leveraging the capability of LLMs, we can effectively reorder the representation space of instruction texts and develop a sufficient level of spatial locality. Such spatial locality enables us to predict potential instructions located in a compact region in the space, resulting in an effective caching system at runtime. Experimental results demonstrate that InstCache achieves a 2.3x higher hit rate compared to the upper bound of traditional caching mechanisms on WildChat dataset and reduces the time per output token of vLLM by up to 42.0% and 50.0% on LMSys and Moss datasets, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09839v1">Rethinking Prompt Optimization: Reinforcement, Diversification, and Migration in Blackbox LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      An increasing number of NLP applications interact with large language models (LLMs) through black-box APIs, making prompt engineering critical for controlling model outputs. While recent Automatic Prompt Optimization (APO) methods iteratively refine prompts using model-generated feedback, textual gradients, they primarily focus on error correction and neglect valuable insights from correct predictions. This limits both their effectiveness and efficiency. In this paper, we propose a novel APO framework centered on enhancing the feedback mechanism. We reinterpret the textual gradient as a form of negative reinforcement and introduce the complementary positive reinforcement to explicitly preserve beneficial prompt components identified through successful predictions. To mitigate the noise inherent in LLM-generated feedback, we introduce a technique called feedback diversification, which aggregates multiple feedback signals, emphasizing consistent, actionable advice while filtering out outliers. Motivated by the rapid evolution and diversity of available LLMs, we also formalize Continual Prompt Optimization (CPO), addressing the practical challenge of efficiently migrating optimized prompts between different model versions or API providers. Our experiments reveal that naive prompt migration often degrades performance due to loss of critical instructions. In contrast, our approach consistently outperforms strong baselines, achieving significant accuracy improvements, faster convergence, and lower computational costs in both standard and migration scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10852v1">LLMs on Trial: Evaluating Judicial Fairness for Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used in high-stakes fields where their decisions impact rights and equity. However, LLMs' judicial fairness and implications for social justice remain underexplored. When LLMs act as judges, the ability to fairly resolve judicial issues is a prerequisite to ensure their trustworthiness. Based on theories of judicial fairness, we construct a comprehensive framework to measure LLM fairness, leading to a selection of 65 labels and 161 corresponding values. Applying this framework to the judicial system, we compile an extensive dataset, JudiFair, comprising 177,100 unique case facts. To achieve robust statistical inference, we develop three evaluation metrics, inconsistency, bias, and imbalanced inaccuracy, and introduce a method to assess the overall fairness of multiple LLMs across various labels. Through experiments with 16 LLMs, we uncover pervasive inconsistency, bias, and imbalanced inaccuracy across models, underscoring severe LLM judicial unfairness. Particularly, LLMs display notably more pronounced biases on demographic labels, with slightly less bias on substance labels compared to procedure ones. Interestingly, increased inconsistency correlates with reduced biases, but more accurate predictions exacerbate biases. While we find that adjusting the temperature parameter can influence LLM fairness, model size, release date, and country of origin do not exhibit significant effects on judicial fairness. Accordingly, we introduce a publicly available toolkit containing all datasets and code, designed to support future research in evaluating and improving LLM fairness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10844v1">LLM-Guided Agentic Object Detection for Open-World Understanding</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      Object detection traditionally relies on fixed category sets, requiring costly re-training to handle novel objects. While Open-World and Open-Vocabulary Object Detection (OWOD and OVOD) improve flexibility, OWOD lacks semantic labels for unknowns, and OVOD depends on user prompts, limiting autonomy. We propose an LLM-guided agentic object detection (LAOD) framework that enables fully label-free, zero-shot detection by prompting a Large Language Model (LLM) to generate scene-specific object names. These are passed to an open-vocabulary detector for localization, allowing the system to adapt its goals dynamically. We introduce two new metrics, Class-Agnostic Average Precision (CAAP) and Semantic Naming Average Precision (SNAP), to separately evaluate localization and naming. Experiments on LVIS, COCO, and COCO-OOD validate our approach, showing strong performance in detecting and naming novel objects. Our method offers enhanced autonomy and adaptability for open-world understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10818v1">How Robust are LLM-Generated Library Imports? An Empirical Study using Stack Overflow</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      Software libraries are central to the functionality, security, and maintainability of modern code. As developers increasingly turn to Large Language Models (LLMs) to assist with programming tasks, understanding how these models recommend libraries is essential. In this paper, we conduct an empirical study of six state-of-the-art LLMs, both proprietary and open-source, by prompting them to solve real-world Python problems sourced from Stack Overflow. We analyze the types of libraries they import, the characteristics of those libraries, and the extent to which the recommendations are usable out of the box. Our results show that LLMs predominantly favour third-party libraries over standard ones, and often recommend mature, popular, and permissively licensed dependencies. However, we also identify gaps in usability: 4.6% of the libraries could not be resolved automatically due to structural mismatches between import names and installable packages, and only two models (out of six) provided installation guidance. While the generated code is technically valid, the lack of contextual support places the burden of manually resolving dependencies on the user. Our findings offer actionable insights for both developers and researchers, and highlight opportunities to improve the reliability and usability of LLM-generated code in the context of software dependencies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10803v1">Automated Thematic Analyses Using LLMs: Xylazine Wound Management Social Media Chatter Use Case</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 Pages: 19, Abstract word count: 151 words, Manuscript word count: 2185 words, References: 14, Figures: 3, Tables: 2
    </div>
    <details class="paper-abstract">
      Background Large language models (LLMs) face challenges in inductive thematic analysis, a task requiring deep interpretive and domain-specific expertise. We evaluated the feasibility of using LLMs to replicate expert-driven thematic analysis of social media data. Methods Using two temporally non-intersecting Reddit datasets on xylazine (n=286 and n=686, for model optimization and validation, respectively) with twelve expert-derived themes, we evaluated five LLMs against expert coding. We modeled the task as a series of binary classifications, rather than a single, multi-label classification, employing zero-, single-, and few-shot prompting strategies and measuring performance via accuracy, precision, recall, and F1-score. Results On the validation set, GPT-4o with two-shot prompting performed best (accuracy: 90.9%; F1-score: 0.71). For high-prevalence themes, model-derived thematic distributions closely mirrored expert classifications (e.g., xylazine use: 13.6% vs. 17.8%; MOUD use: 16.5% vs. 17.8%). Conclusions Our findings suggest that few-shot LLM-based approaches can automate thematic analyses, offering a scalable supplement for qualitative research. Keywords: thematic analysis, large language models, natural language processing, qualitative analysis, social media, prompt engineering, public health
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04644v2">Agentic Reasoning: A Streamlined Framework for Enhancing LLM Reasoning with Agentic Tools</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 ACL 2025
    </div>
    <details class="paper-abstract">
      We introduce Agentic Reasoning, a framework that enhances large language model (LLM) reasoning by integrating external tool-using agents. Agentic Reasoning dynamically leverages web search, code execution, and structured memory to address complex problems requiring deep research. A key innovation in our framework is the Mind-Map agent, which constructs a structured knowledge graph to store reasoning context and track logical relationships, ensuring coherence in long reasoning chains with extensive tool usage. Additionally, we conduct a comprehensive exploration of the Web-Search agent, leading to a highly effective search mechanism that surpasses all prior approaches. When deployed on DeepSeek-R1, our method achieves a new state-of-the-art (SOTA) among public models and delivers performance comparable to OpenAI Deep Research, the leading proprietary model in this domain. Extensive ablation studies validate the optimal selection of agentic tools and confirm the effectiveness of our Mind-Map and Web-Search agents in enhancing LLM reasoning. The code is at: https://github.com/theworldofagents/Agentic-Reasoning
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10778v1">Warehouse Spatial Question Answering with LLM Agent</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 1st Place Solution of the 9th AI City Challenge Track 3
    </div>
    <details class="paper-abstract">
      Spatial understanding has been a challenging task for existing Multi-modal Large Language Models~(MLLMs). Previous methods leverage large-scale MLLM finetuning to enhance MLLM's spatial understanding ability. In this paper, we present a data-efficient approach. We propose a LLM agent system with strong and advanced spatial reasoning ability, which can be used to solve the challenging spatial question answering task in complex indoor warehouse scenarios. Our system integrates multiple tools that allow the LLM agent to conduct spatial reasoning and API tools interaction to answer the given complicated spatial question. Extensive evaluations on the 2025 AI City Challenge Physical AI Spatial Intelligence Warehouse dataset demonstrate that our system achieves high accuracy and efficiency in tasks such as object retrieval, counting, and distance estimation. The code is available at: https://github.com/hsiangwei0903/SpatialAgent
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05209v3">Model Tampering Attacks Enable More Rigorous Evaluations of LLM Capabilities</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 Accepted to TMLR
    </div>
    <details class="paper-abstract">
      Evaluations of large language model (LLM) risks and capabilities are increasingly being incorporated into AI risk management and governance frameworks. Currently, most risk evaluations are conducted by designing inputs that elicit harmful behaviors from the system. However, this approach suffers from two limitations. First, input-output evaluations cannot fully evaluate realistic risks from open-weight models. Second, the behaviors identified during any particular input-output evaluation can only lower-bound the model's worst-possible-case input-output behavior. As a complementary method for eliciting harmful behaviors, we propose evaluating LLMs with model tampering attacks which allow for modifications to latent activations or weights. We pit state-of-the-art techniques for removing harmful LLM capabilities against a suite of 5 input-space and 6 model tampering attacks. In addition to benchmarking these methods against each other, we show that (1) model resilience to capability elicitation attacks lies on a low-dimensional robustness subspace; (2) the success rate of model tampering attacks can empirically predict and offer conservative estimates for the success of held-out input-space attacks; and (3) state-of-the-art unlearning methods can easily be undone within 16 steps of fine-tuning. Together, these results highlight the difficulty of suppressing harmful LLM capabilities and show that model tampering attacks enable substantially more rigorous evaluations than input-space attacks alone.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.02820v4">DroidSpeak: KV Cache Sharing for Cross-LLM Communication and Multi-LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      Compound AI systems, such as agentic systems, are an emerging trend in large-scale enterprise settings, with multiple LLMs specialized for different users, tasks, and/or roles working together. In these scenarios, different models often process inputs that share the same context prefix. Although much work was done in the past to enable the reuse of prefix KV caches across inputs for a single model, how to enable one model to reuse the prefix KV caches of a different model remains an open question. We introduce DroidSpeak, the first distributed LLM inference system that enables KV cache reuse across distributed nodes running inference of different LLMs, so long as the LLMs have the same architecture. We present the first study that aims at understanding the impact of sharing KV caches across different LLMs, and if/when such sharing affects quality. Inspired by the findings, we present DroidSpeak, which selectively recomputes a few layers of the KV cache produced by another LLM and reuses the remaining layers, with negligible quality loss. Moreover, carefully pipelining the layer-wise re-computation and the loading of reused KV cache further improves the inference performance. Experiments on diverse datasets and model pairs demonstrate that DroidSpeak achieves up to 4x throughput improvement and about 3.1x faster prefill (time to first token), with negligible loss of quality in F1 scores, Rouge-L or code similarity score, compared to the baseline which does not allow any sharing across models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10695v1">Exploring User Security and Privacy Attitudes and Concerns Toward the Use of General-Purpose LLM Chatbots for Mental Health</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 Accepted to the 34th USENIX Security Symposium
    </div>
    <details class="paper-abstract">
      Individuals are increasingly relying on large language model (LLM)-enabled conversational agents for emotional support. While prior research has examined privacy and security issues in chatbots specifically designed for mental health purposes, these chatbots are overwhelmingly "rule-based" offerings that do not leverage generative AI. Little empirical research currently measures users' privacy and security concerns, attitudes, and expectations when using general-purpose LLM-enabled chatbots to manage and improve mental health. Through 21 semi-structured interviews with U.S. participants, we identified critical misconceptions and a general lack of risk awareness. Participants conflated the human-like empathy exhibited by LLMs with human-like accountability and mistakenly believed that their interactions with these chatbots were safeguarded by the same regulations (e.g., HIPAA) as disclosures with a licensed therapist. We introduce the concept of "intangible vulnerability," where emotional or psychological disclosures are undervalued compared to more tangible forms of information (e.g., financial or location-based data). To address this, we propose recommendations to safeguard user mental health disclosures with general-purpose LLM-enabled chatbots more effectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10639v1">SPICEAssistant: LLM using SPICE Simulation Tools for Schematic Design of Switched-Mode Power Supplies</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 11 pages, 10 figures
    </div>
    <details class="paper-abstract">
      State-of-the-art large language models (LLMs) show high performance across a wide range of tasks in many domains of science. In the field of electronic design automation (EDA), it is yet to be determined to what extent they are capable to understand, adapt, and dimension electronic circuits. This paper focuses on the application of LLMs to switched-mode power supply (SMPS) design on printed circuit boards (PCBs). Particular challenges for LLMs in this context include their limited ability to interpret results from key simulation tools like SPICE and the multi-step design process. To address these challenges, we suggest SPICEAssistant, a framework that provides a broad selection of tools to an LLM. The tools serve as an interface to SPICE, allowing the LLM to interact flexibly with the simulator to estimate the impact of its modifications to the circuit. To evaluate the performance of SPICEAssistant, we defined a benchmark consisting of 256 questions testing the ability to adapt circuit netlists to fulfil different SMPS design tasks. The benchmarking results show that simulation feedback effectively improves SMPS design capabilities of LLMs. An increasing number of simulation iterations leads to enhanced performance. The SPICEAssistant framework significantly outperforms the standalone LLM GPT-4o on the benchmark by approximately 38%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10628v1">GHPO: Adaptive Guidance for Stable and Efficient LLM Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Rewards (RLVR) has recently emerged as a powerful paradigm for facilitating the self-improvement of large language models (LLMs), particularly in the domain of complex reasoning tasks. However, prevailing on-policy RL methods often contend with significant training instability and inefficiency. This is primarily due to a capacity-difficulty mismatch, where the complexity of training data frequently outpaces the model's current capabilities, leading to critically sparse reward signals and stalled learning progress. This challenge is particularly acute for smaller, more resource-efficient LLMs. To overcome this, we introduce the Guided Hybrid Policy Optimization (GHPO), a novel difficulty-aware reinforcement learning framework. GHPO dynamically calibrates task difficulty by employing adaptive prompt refinement to provide targeted guidance. This unique approach adaptively balances direct imitation learning for problems currently beyond the model's reach with exploration-based reinforcement learning for more manageable tasks, effectively creating a smooth and optimized learning curriculum. Extensive experiments demonstrate that GHPO achieves an average performance gain of approximately 5% across six challenging mathematics benchmarks, consistently outperforming strong on-policy reinforcement learning and curriculum learning baselines. Further analysis confirms that our framework significantly enhances both training stability and final reasoning performance, thus offering a scalable and efficient solution for developing powerful and robust reasoning models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10624v1">Comprehension Without Competence: Architectural Limits of LLMs in Symbolic Computation and Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 Substantial change to previous version (experiments, theorem, analysis and related work); currently under review at TMLR
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) display striking surface fluency yet systematically fail at tasks requiring symbolic reasoning, arithmetic accuracy, and logical consistency. This paper offers a structural diagnosis of such failures, revealing a persistent gap between \textit{comprehension} and \textit{competence}. Through controlled experiments and architectural analysis, we demonstrate that LLMs often articulate correct principles without reliably applying them--a failure rooted not in knowledge access, but in computational execution. We term this phenomenon the computational \textit{split-brain syndrome}, where instruction and action pathways are geometrically and functionally dissociated. This core limitation recurs across domains, from mathematical operations to relational inferences, and explains why model behavior remains brittle even under idealized prompting. We argue that LLMs function as powerful pattern completion engines, but lack the architectural scaffolding for principled, compositional reasoning. Our findings delineate the boundary of current LLM capabilities and motivate future models with metacognitive control, principle lifting, and structurally grounded execution. This diagnosis also clarifies why mechanistic interpretability findings may reflect training-specific pattern coordination rather than universal computational principles, and why the geometric separation between instruction and execution pathways suggests limitations in neural introspection and mechanistic analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10621v1">Game Theory Meets LLM and Agentic AI: Reimagining Cybersecurity for the Age of Intelligent Threats</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      Protecting cyberspace requires not only advanced tools but also a shift in how we reason about threats, trust, and autonomy. Traditional cybersecurity methods rely on manual responses and brittle heuristics. To build proactive and intelligent defense systems, we need integrated theoretical frameworks and software tools. Game theory provides a rigorous foundation for modeling adversarial behavior, designing strategic defenses, and enabling trust in autonomous systems. Meanwhile, software tools process cyber data, visualize attack surfaces, verify compliance, and suggest mitigations. Yet a disconnect remains between theory and practical implementation. The rise of Large Language Models (LLMs) and agentic AI offers a new path to bridge this gap. LLM-powered agents can operationalize abstract strategies into real-world decisions. Conversely, game theory can inform the reasoning and coordination of these agents across complex workflows. LLMs also challenge classical game-theoretic assumptions, such as perfect rationality or static payoffs, prompting new models aligned with cognitive and computational realities. This co-evolution promises richer theoretical foundations and novel solution concepts. Agentic AI also reshapes software design: systems must now be modular, adaptive, and trust-aware from the outset. This chapter explores the intersection of game theory, agentic AI, and cybersecurity. We review key game-theoretic frameworks (e.g., static, dynamic, Bayesian, and signaling games) and solution concepts. We then examine how LLM agents can enhance cyber defense and introduce LLM-driven games that embed reasoning into AI agents. Finally, we explore multi-agent workflows and coordination games, outlining how this convergence fosters secure, intelligent, and adaptive cyber systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09820v1">Measuring What Matters: A Framework for Evaluating Safety Risks in Real-World LLM Applications</a></div>
    <div class="paper-meta">
      📅 2025-07-13
    </div>
    <details class="paper-abstract">
      Most safety testing efforts for large language models (LLMs) today focus on evaluating foundation models. However, there is a growing need to evaluate safety at the application level, as components such as system prompts, retrieval pipelines, and guardrails introduce additional factors that significantly influence the overall safety of LLM applications. In this paper, we introduce a practical framework for evaluating application-level safety in LLM systems, validated through real-world deployment across multiple use cases within our organization. The framework consists of two parts: (1) principles for developing customized safety risk taxonomies, and (2) practices for evaluating safety risks in LLM applications. We illustrate how the proposed framework was applied in our internal pilot, providing a reference point for organizations seeking to scale their safety testing efforts. This work aims to bridge the gap between theoretical concepts in AI safety and the operational realities of safeguarding LLM applications in practice, offering actionable guidance for safe and scalable deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09790v1">Prompting for Performance: Exploring LLMs for Configuring Software</a></div>
    <div class="paper-meta">
      📅 2025-07-13
    </div>
    <details class="paper-abstract">
      Software systems usually provide numerous configuration options that can affect performance metrics such as execution time, memory usage, binary size, or bitrate. On the one hand, making informed decisions is challenging and requires domain expertise in options and their combinations. On the other hand, machine learning techniques can search vast configuration spaces, but with a high computational cost, since concrete executions of numerous configurations are required. In this exploratory study, we investigate whether large language models (LLMs) can assist in performance-oriented software configuration through prompts. We evaluate several LLMs on tasks including identifying relevant options, ranking configurations, and recommending performant configurations across various configurable systems, such as compilers, video encoders, and SAT solvers. Our preliminary results reveal both positive abilities and notable limitations: depending on the task and systems, LLMs can well align with expert knowledge, whereas hallucinations or superficial reasoning can emerge in other cases. These findings represent a first step toward systematic evaluations and the design of LLM-based solutions to assist with software configuration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09788v1">TinyTroupe: An LLM-powered Multiagent Persona Simulation Toolkit</a></div>
    <div class="paper-meta">
      📅 2025-07-13
      | 💬 9 pages. Preprint to be submitted to peer-review
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLM) have led to a new class of autonomous agents, renewing and expanding interest in the area. LLM-powered Multiagent Systems (MAS) have thus emerged, both for assistive and simulation purposes, yet tools for realistic human behavior simulation -- with its distinctive challenges and opportunities -- remain underdeveloped. Existing MAS libraries and tools lack fine-grained persona specifications, population sampling facilities, experimentation support, and integrated validation, among other key capabilities, limiting their utility for behavioral studies, social simulation, and related applications. To address these deficiencies, in this work we introduce TinyTroupe, a simulation toolkit enabling detailed persona definitions (e.g., nationality, age, occupation, personality, beliefs, behaviors) and programmatic control via numerous LLM-driven mechanisms. This allows for the concise formulation of behavioral problems of practical interest, either at the individual or group level, and provides effective means for their solution. TinyTroupe's components are presented using representative working examples, such as brainstorming and market research sessions, thereby simultaneously clarifying their purpose and demonstrating their usefulness. Quantitative and qualitative evaluations of selected aspects are also provided, highlighting possibilities, limitations, and trade-offs. The approach, though realized as a specific Python implementation, is meant as a novel conceptual contribution, which can be partially or fully incorporated in other contexts. The library is available as open source at https://github.com/microsoft/tinytroupe.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09397v4">SLED: A Speculative LLM Decoding Framework for Efficient Edge Serving</a></div>
    <div class="paper-meta">
      📅 2025-07-13
      | 💬 6 pages, 6 figures, 2 tables
    </div>
    <details class="paper-abstract">
      The growing gap between the increasing complexity of large language models (LLMs) and the limited computational budgets of edge devices poses a key challenge for efficient on-device inference, despite gradual improvements in hardware capabilities. Existing strategies, such as aggressive quantization, pruning, or remote inference, trade accuracy for efficiency or lead to substantial cost burdens. This position paper introduces a new framework that leverages speculative decoding, previously viewed primarily as a decoding acceleration technique for autoregressive generation of LLMs, as a promising approach specifically adapted for edge computing by orchestrating computation across heterogeneous devices. We propose \acronym, a framework that allows lightweight edge devices to draft multiple candidate tokens locally using diverse draft models, while a single, shared edge server verifies the tokens utilizing a more precise target model. To further increase the efficiency of verification, the edge server batch the diverse verification requests from devices. This approach supports device heterogeneity and reduces server-side memory footprint by sharing the same upstream target model across multiple devices. Our initial experiments with Jetson Orin Nano, Raspberry Pi 4B/5, and an edge server equipped with 4 Nvidia A100 GPUs indicate substantial benefits: 2.2 more system throughput, 2.8 more system capacity, and better cost efficiency, all without sacrificing model accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.11462v5">Cascade Speculative Drafting for Even Faster LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-07-13
      | 💬 NeurIPS 2024
    </div>
    <details class="paper-abstract">
      Introduced to enhance the efficiency of large language model (LLM) inference, speculative decoding operates by having a smaller model generate a draft. A larger target model then reviews this draft to align with its output, and any acceptance by the target model results in a reduction of the number of the target model runs, ultimately improving efficiency. However, the drafting process in speculative decoding includes slow autoregressive generation and allocates equal time to generating tokens, irrespective of their importance. These inefficiencies collectively contribute to the suboptimal performance of speculative decoding. To further improve LLM inference, we introduce Cascade Speculative Drafting (CS Drafting), a speculative execution algorithm that incorporates two types of cascades. The Vertical Cascade eliminates autoregressive generation from neural models, while the Horizontal Cascade optimizes time allocation in drafting for improved efficiency. Combining both cascades, CS Drafting achieves greater speedup compared to the baselines in our experiments, while preserving the same output distribution as the target model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09751v1">Sound and Complete Neuro-symbolic Reasoning with LLM-Grounded Interpretations</a></div>
    <div class="paper-meta">
      📅 2025-07-13
      | 💬 29 pages, 9 tables, 3 figures. Accepted to the 19th Conference on Neurosymbolic Learning and Reasoning (NeSy 2025)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive capabilities in natural language understanding and generation, but they exhibit problems with logical consistency in the output they generate. How can we harness LLMs' broad-coverage parametric knowledge in formal reasoning despite their inconsistency? We present a method for directly integrating an LLM into the interpretation function of the formal semantics for a paraconsistent logic. We provide experimental evidence for the feasibility of the method by evaluating the function using datasets created from several short-form factuality benchmarks. Unlike prior work, our method offers a theoretical framework for neuro-symbolic reasoning that leverages an LLM's knowledge while preserving the underlying logic's soundness and completeness properties.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09701v1">MCEval: A Dynamic Framework for Fair Multilingual Cultural Evaluation of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-13
    </div>
    <details class="paper-abstract">
      Large language models exhibit cultural biases and limited cross-cultural understanding capabilities, particularly when serving diverse global user populations. We propose MCEval, a novel multilingual evaluation framework that employs dynamic cultural question construction and enables causal analysis through Counterfactual Rephrasing and Confounder Rephrasing. Our comprehensive evaluation spans 13 cultures and 13 languages, systematically assessing both cultural awareness and cultural bias across different linguistic scenarios. The framework provides 39,897 cultural awareness instances and 17,940 cultural bias instances. Experimental results reveal performance disparities across different linguistic scenarios, demonstrating that optimal cultural performance is not only linked to training data distribution, but also is related to language-culture alignment. The evaluation results also expose the fairness issue, where approaches appearing successful in the English scenario create substantial disadvantages. MCEval represents the first comprehensive multilingual cultural evaluation framework that provides deeper insights into LLMs' cultural understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09657v1">Negotiating Comfort: Simulating Personality-Driven LLM Agents in Shared Residential Social Networks</a></div>
    <div class="paper-meta">
      📅 2025-07-13
    </div>
    <details class="paper-abstract">
      We use generative agents powered by large language models (LLMs) to simulate a social network in a shared residential building, driving the temperature decisions for a central heating system. Agents, divided into Family Members and Representatives, consider personal preferences, personal traits, connections, and weather conditions. Daily simulations involve family-level consensus followed by building-wide decisions among representatives. We tested three personality traits distributions (positive, mixed, and negative) and found that positive traits correlate with higher happiness and stronger friendships. Temperature preferences, assertiveness, and selflessness have a significant impact on happiness and decisions. This work demonstrates how LLM-driven agents can help simulate nuanced human behavior where complex real-life human simulations are difficult to set.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08965v2">LLM-Driven Usefulness Labeling for IR Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-07-13
    </div>
    <details class="paper-abstract">
      In the information retrieval (IR) domain, evaluation plays a crucial role in optimizing search experiences and supporting diverse user intents. In the recent LLM era, research has been conducted to automate document relevance labels, as these labels have traditionally been assigned by crowd-sourced workers - a process that is both time and consuming and costly. This study focuses on LLM-generated usefulness labels, a crucial evaluation metric that considers the user's search intents and task objectives, an aspect where relevance falls short. Our experiment utilizes task-level, query-level, and document-level features along with user search behavior signals, which are essential in defining the usefulness of a document. Our research finds that (i) pre-trained LLMs can generate moderate usefulness labels by understanding the comprehensive search task session, (ii) pre-trained LLMs perform better judgement in short search sessions when provided with search session contexts. Additionally, we investigated whether LLMs can capture the unique divergence between relevance and usefulness, along with conducting an ablation study to identify the most critical metrics for accurate usefulness label generation. In conclusion, this work explores LLM-generated usefulness labels by evaluating critical metrics and optimizing for practicality in real-world settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05310v2">Oracular Programming: A Modular Foundation for Building LLM-Enabled Software</a></div>
    <div class="paper-meta">
      📅 2025-07-13
    </div>
    <details class="paper-abstract">
      Large Language Models have proven surprisingly effective at solving a wide range of tasks from just a handful of examples. However, their lack of reliability and modularity limits their capacity to tackle large problems that require many steps of reasoning. In response, researchers have proposed advanced pipelines that leverage domain-specific knowledge to chain smaller prompts, provide intermediate feedback and improve performance through search. However, the current complexity of writing, tuning, maintaining and improving such pipelines has limited their sophistication. We propose oracular programming, a foundational paradigm for building LLM-enabled applications that lets domain experts express high-level problem-solving strategies as programs with unresolved choice points. These choice points are resolved at runtime by LLMs, which generalize from user-provided examples of correct and incorrect decisions. An oracular program is composed of three orthogonal components: a strategy that consists in a nondeterministic program with choice points that can be reified into a search tree, a policy that specifies how to navigate this tree with the help of LLM oracles, and a set of demonstrations that describe successful and unsuccessful search tree navigation scenarios across diverse problem instances. Each component is expressed in a dedicated programming language and can be independently improved or substituted. We address the key programming language design challenges of modularly composing oracular programs and enforcing consistency between their components as they evolve.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09490v1">Towards LLM-Based Automatic Playtest</a></div>
    <div class="paper-meta">
      📅 2025-07-13
    </div>
    <details class="paper-abstract">
      Playtesting is the process in which people play a video game for testing. It is critical for the quality assurance of gaming software. Manual playtesting is time-consuming and expensive. However, automating this process is challenging, as playtesting typically requires domain knowledge and problem-solving skills that most conventional testing tools lack. Recent advancements in artificial intelligence (AI) have opened up new possibilities for applying Large Language Models (LLMs) to playtesting. However, significant challenges remain: current LLMs cannot visually perceive game environments, and most existing research focuses on text-based games or games with robust APIs. Many non-text games lack APIs to provide textual descriptions of game states, making it almost impossible to naively apply LLMs for playtesting. This paper introduces Lap, our novel approach to LLM-based Automatic Playtesting, which uses ChatGPT to test match-3 games, a category of games where players match three or more identical tiles in a row or column to earn points. Lap encompasses three key phases: processing of game environments, prompting-based action generation, and action execution. Given a match-3 game, Lap takes a snapshot of the game board and converts it to a numeric matrix. It then prompts the ChatGPT-O1-mini API to suggest moves based on that matrix and tentatively applies the suggested moves to earn points and trigger changes in the game board. It repeats the above-mentioned three steps iteratively until timeout. For evaluation, we conducted a case study using Lap on an open-source match-3 game, CasseBonbons, and empirically compared it with three existing tools. Our results are promising: Lap outperformed existing tools by achieving higher code coverage and triggering more program crashes. This research sheds light on the future of automatic testing and LLM applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09488v1">Criteria-Based LLM Relevance Judgments</a></div>
    <div class="paper-meta">
      📅 2025-07-13
      | 💬 10 pages, 3 figures, accepted to ICTIR 2025
    </div>
    <details class="paper-abstract">
      Relevance judgments are crucial for evaluating information retrieval systems, but traditional human-annotated labels are time-consuming and expensive. As a result, many researchers turn to automatic alternatives to accelerate method development. Among these, Large Language Models (LLMs) provide a scalable solution by generating relevance labels directly through prompting. However, prompting an LLM for a relevance label without constraints often results in not only incorrect predictions but also outputs that are difficult for humans to interpret. We propose the Multi-Criteria framework for LLM-based relevance judgments, decomposing the notion of relevance into multiple criteria--such as exactness, coverage, topicality, and contextual fit--to improve the robustness and interpretability of retrieval evaluations compared to direct grading methods. We validate this approach on three datasets: the TREC Deep Learning tracks from 2019 and 2020, as well as LLMJudge (based on TREC DL 2023). Our results demonstrate that Multi-Criteria judgments enhance the system ranking/leaderboard performance. Moreover, we highlight the strengths and limitations of this approach relative to direct grading approaches, offering insights that can guide the development of future automatic evaluation frameworks in information retrieval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09483v1">Does UMBRELA Work on Other LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-07-13
      | 💬 9 pages, 2 figures, accepted to SIGIR 2025
    </div>
    <details class="paper-abstract">
      We reproduce the UMBRELA LLM Judge evaluation framework across a range of large language models (LLMs) to assess its generalizability beyond the original study. Our investigation evaluates how LLM choice affects relevance assessment accuracy, focusing on leaderboard rank correlation and per-label agreement metrics. Results demonstrate that UMBRELA with DeepSeek V3 obtains very comparable performance to GPT-4o (used in original work). For LLaMA-3.3-70B we obtain slightly lower performance, which further degrades with smaller LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09481v1">Evaluating LLMs on Sequential API Call Through Automated Test Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-13
    </div>
    <details class="paper-abstract">
      By integrating tools from external APIs, Large Language Models (LLMs) have expanded their promising capabilities in a diverse spectrum of complex real-world tasks. However, testing, evaluation, and analysis of LLM tool use remain in their early stages. Most existing benchmarks rely on manually collected test cases, many of which cannot be automatically checked for semantic correctness and instead depend on static methods such as string matching. Additionally, these benchmarks often overlook the complex interactions that occur between sequential API calls, which are common in real-world applications. To fill the gap, in this paper, we introduce StateGen, an automated framework designed to generate diverse coding tasks involving sequential API interactions. StateGen combines state-machine-based API constraint solving and validation, energy-based sampling, and control-flow injection to generate executable programs. These programs are then translated into human-like natural language task descriptions through a collaboration of two LLM agents. Utilizing StateGen, we construct StateEval, a benchmark encompassing 120 verified test cases spanning across three representative scenarios: Session Service, Tensor Operation, and ElevenLabs MCP. Experimental results confirm that StateGen can effectively generate challenging and realistic API-oriented tasks, highlighting areas for improvement in current LLMs incorporating APIs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09477v1">Towards Agentic RAG with Deep Reasoning: A Survey of RAG-Reasoning Systems in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-13
      | 💬 submitted to ARR May
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) lifts the factuality of Large Language Models (LLMs) by injecting external knowledge, yet it falls short on problems that demand multi-step inference; conversely, purely reasoning-oriented approaches often hallucinate or mis-ground facts. This survey synthesizes both strands under a unified reasoning-retrieval perspective. We first map how advanced reasoning optimizes each stage of RAG (Reasoning-Enhanced RAG). Then, we show how retrieved knowledge of different type supply missing premises and expand context for complex inference (RAG-Enhanced Reasoning). Finally, we spotlight emerging Synergized RAG-Reasoning frameworks, where (agentic) LLMs iteratively interleave search and reasoning to achieve state-of-the-art performance across knowledge-intensive benchmarks. We categorize methods, datasets, and open challenges, and outline research avenues toward deeper RAG-Reasoning systems that are more effective, multimodally-adaptive, trustworthy, and human-centric. The collection is available at https://github.com/DavidZWZ/Awesome-RAG-Reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00648v3">DFRot: Achieving Outlier-Free and Massive Activation-Free for Rotated LLMs with Refined Rotation</a></div>
    <div class="paper-meta">
      📅 2025-07-13
      | 💬 Accepeted bythe 2nd Conference on Language Modeling (COLM 2025). Source code \url{https://github.com/JingyangXiang/DFRot}
    </div>
    <details class="paper-abstract">
      Rotating the activation and weight matrices to reduce the influence of outliers in large language models (LLMs) has recently attracted significant attention, particularly in the context of model quantization. Prior studies have shown that in low-precision quantization scenarios, such as 4-bit weights and 4-bit activations (W4A4), randomized Hadamard transforms can achieve significantly higher accuracy than randomized orthogonal transforms. Notably, the reason behind this phenomenon remains unknown. In this paper, we find that these transformations show substantial improvement in eliminating outliers for common tokens and achieve similar quantization error. The primary reason for the accuracy difference lies in the fact that randomized Hadamard transforms can slightly reduce the quantization error for tokens with massive activations while randomized orthogonal transforms increase the quantization error. Due to the extreme rarity of these tokens and their critical impact on model accuracy, we consider this a long-tail optimization problem, and therefore construct a simple yet effective method: a weighted loss function. Additionally, we propose an optimization strategy for the rotation matrix that involves alternating optimization of quantization parameters while employing orthogonal Procrustes transforms to refine the rotation matrix. This makes the distribution of the rotated activation values more conducive to quantization, especially for tokens with massive activations. Our method enhances the Rotated LLMs by achieving dual free, Outlier-Free and Massive Activation-Free, dubbed as DFRot. Extensive experiments demonstrate the effectiveness and efficiency of DFRot. By tuning the rotation matrix using just a single sample, DFRot achieves a perplexity improvement of 0.98 and 0.95 on W4A4KV4 and W4A4KV16, respectively, for LLaMA3-70B, a model known for its quantization challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10620v1">LLMs Meet Cross-Modal Time Series Analytics: Overview and Directions</a></div>
    <div class="paper-meta">
      📅 2025-07-13
      | 💬 Accepted at SSTD 2025 (Tutorial). arXiv admin note: text overlap with arXiv:2505.02583
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have emerged as a promising paradigm for time series analytics, leveraging their massive parameters and the shared sequential nature of textual and time series data. However, a cross-modality gap exists between time series and textual data, as LLMs are pre-trained on textual corpora and are not inherently optimized for time series. In this tutorial, we provide an up-to-date overview of LLM-based cross-modal time series analytics. We introduce a taxonomy that classifies existing approaches into three groups based on cross-modal modeling strategies, e.g., conversion, alignment, and fusion, and then discuss their applications across a range of downstream tasks. In addition, we summarize several open challenges. This tutorial aims to expand the practical application of LLMs in solving real-world problems in cross-modal time series analytics while balancing effectiveness and efficiency. Participants will gain a thorough understanding of current advancements, methodologies, and future research directions in cross-modal time series analytics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10605v1">RedOne: Revealing Domain-specific LLM Post-Training in Social Networking Services</a></div>
    <div class="paper-meta">
      📅 2025-07-13
    </div>
    <details class="paper-abstract">
      As a primary medium for modern information dissemination, social networking services (SNS) have experienced rapid growth, which has proposed significant challenges for platform content management and interaction quality improvement. Recently, the development of large language models (LLMs) has offered potential solutions but existing studies focus on isolated tasks, which not only encounter diminishing benefit from the data scaling within individual scenarios but also fail to flexibly adapt to diverse real-world context. To address these challenges, we introduce RedOne, a domain-specific LLM designed to break the performance bottleneck of single-task baselines and establish a comprehensive foundation for the SNS. RedOne was developed through a three-stage training strategy consisting of continue pretraining, supervised fine-tuning, and preference optimization, using a large-scale real-world dataset. Through extensive experiments, RedOne maintains strong general capabilities, and achieves an average improvement up to 14.02% across 8 major SNS tasks and 7.56% in SNS bilingual evaluation benchmark, compared with base models. Furthermore, through online testing, RedOne reduced the exposure rate in harmful content detection by 11.23% and improved the click page rate in post-view search by 14.95% compared with single-tasks finetuned baseline models. These results establish RedOne as a robust domain-specific LLM for SNS, demonstrating excellent generalization across various tasks and promising applicability in real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09407v1">LLM-Stackelberg Games: Conjectural Reasoning Equilibria and Their Applications to Spearphishing</a></div>
    <div class="paper-meta">
      📅 2025-07-12
    </div>
    <details class="paper-abstract">
      We introduce the framework of LLM-Stackelberg games, a class of sequential decision-making models that integrate large language models (LLMs) into strategic interactions between a leader and a follower. Departing from classical Stackelberg assumptions of complete information and rational agents, our formulation allows each agent to reason through structured prompts, generate probabilistic behaviors via LLMs, and adapt their strategies through internal cognition and belief updates. We define two equilibrium concepts: reasoning and behavioral equilibrium, which aligns an agent's internal prompt-based reasoning with observable behavior, and conjectural reasoning equilibrium, which accounts for epistemic uncertainty through parameterized models over an opponent's response. These layered constructs capture bounded rationality, asymmetric information, and meta-cognitive adaptation. We illustrate the framework through a spearphishing case study, where a sender and a recipient engage in a deception game using structured reasoning prompts. This example highlights the cognitive richness and adversarial potential of LLM-mediated interactions. Our results show that LLM-Stackelberg games provide a powerful paradigm for modeling decision-making in domains such as cybersecurity, misinformation, and recommendation systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.22362v2">Supposedly Equivalent Facts That Aren't? Entity Frequency in Pre-training Induces Asymmetry in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-12
      | 💬 Accepted at COLM 2025
    </div>
    <details class="paper-abstract">
      Understanding and mitigating hallucinations in Large Language Models (LLMs) is crucial for ensuring reliable content generation. While previous research has primarily focused on "when" LLMs hallucinate, our work explains "why" and directly links model behaviour to the pre-training data that forms their prior knowledge. Specifically, we demonstrate that an asymmetry exists in the recognition of logically equivalent facts, which can be attributed to frequency discrepancies of entities appearing as subjects versus objects. Given that most pre-training datasets are inaccessible, we leverage the fully open-source OLMo series by indexing its Dolma dataset to estimate entity frequencies. Using relational facts (represented as triples) from Wikidata5M, we construct probing datasets to isolate this effect. Our experiments reveal that facts with a high-frequency subject and a low-frequency object are better recognised than their inverse, despite their logical equivalence. The pattern reverses in low-to-high frequency settings, and no statistically significant asymmetry emerges when both entities are high-frequency. These findings highlight the influential role of pre-training data in shaping model predictions and provide insights for inferring the characteristics of pre-training data in closed or partially closed LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.00555v2">Prune 'n Predict: Optimizing LLM Decision-making with Conformal Prediction</a></div>
    <div class="paper-meta">
      📅 2025-07-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are empowering decision-making in several applications, including tool or API usage and answering multiple-choice questions (MCQs). However, incorrect outputs pose significant risks in high-stakes domains like healthcare and finance. To quantify LLM uncertainty and thereby mitigate these risks, recent works employ conformal prediction (CP), a model- and distribution-agnostic framework that uses LLM outputs to generate a \emph{prediction set} containing the true answer with high probability. Leveraging CP, we propose \emph{conformal revision of questions} (CROQ), which revises the question by narrowing down the available choices to those in the prediction set and asking the LLM the revised question. We expect LLMs to be more accurate on revised questions with fewer choices. Furthermore, we expect CROQ to be effective when the prediction sets from CP are small. Commonly used logit scores often lead to large sets, diminishing CROQ's effectiveness. To overcome this, we propose CP-OPT, an optimization framework to learn scores that minimize set sizes while maintaining coverage. Our extensive experiments on MMLU, ToolAlpaca, and TruthfulQA datasets with multiple LLMs show that CROQ improves accuracy over the standard inference, with more pronounced gains when paired with CP-OPT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.23377v2">Perspective Dial: Measuring Perspective of Text and Guiding LLM Outputs</a></div>
    <div class="paper-meta">
      📅 2025-07-12
      | 💬 7 pages, 5 main pages of text, 5 figures, 2 tables. Research work performed at CACI INTL INC
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are used in a variety of mission-critical roles. Due to the rapidly developing nature of LLMs, there is a lack of quantifiable understanding of the bias and perspective associated with LLM output. Inspired by this need, this paper considers the broader issue of perspective or viewpoint of general text and perspective control of large-language model (LLM) output. Perspective-Dial consists of two main components: a (1) metric space, dubbed Perspective Space, that enables quantitative measurements of different perspectives regarding a topic, and the use of (2) Systematic Prompt Engineering that utilizes greedy-coordinate descent to control LLM output perspective based on measurement feedback from the Perspective Space. The empirical nature of the approach allows progress to side step a principled understanding of perspective or bias -- effectively quantifying and adjusting outputs for a variety of topics. Potential applications include detection, tracking and mitigation of LLM bias, narrative detection, sense making and tracking in public discourse, and debate bot advocating given perspective.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09329v1">When Developer Aid Becomes Security Debt: A Systematic Analysis of Insecure Behaviors in LLM Coding Agents</a></div>
    <div class="paper-meta">
      📅 2025-07-12
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      LLM-based coding agents are rapidly being deployed in software development, yet their security implications remain poorly understood. These agents, while capable of accelerating software development, may inadvertently introduce insecure practices. We conducted the first systematic security evaluation of autonomous coding agents, analyzing over 12,000 actions across five state-of-the-art models (GPT-4o, GPT-4.1, Claude variants) on 93 real-world software setup tasks. Our findings reveal significant security concerns: 21% of agent trajectories contained insecure actions, with models showing substantial variation in security behavior. We developed a high-precision detection system that identified four major vulnerability categories, with information exposure (CWE-200) being the most prevalent one. We also evaluated mitigation strategies including feedback mechanisms and security reminders with various effectiveness between models. GPT-4.1 demonstrated exceptional security awareness with 96.8% mitigation success. Our work provides the first comprehensive framework for evaluating coding agent security and highlights the need for security-aware design of next generation LLM-based coding agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.23978v2">LLM Agents Are the Antidote to Walled Gardens</a></div>
    <div class="paper-meta">
      📅 2025-07-12
    </div>
    <details class="paper-abstract">
      While the Internet's core infrastructure was designed to be open and universal, today's application layer is dominated by closed, proprietary platforms. Open and interoperable APIs require significant investment, and market leaders have little incentive to enable data exchange that could erode their user lock-in. We argue that LLM-based agents fundamentally disrupt this status quo. Agents can automatically translate between data formats and interact with interfaces designed for humans: this makes interoperability dramatically cheaper and effectively unavoidable. We name this shift universal interoperability: the ability for any two digital services to exchange data seamlessly using AI-mediated adapters. Universal interoperability undermines monopolistic behaviours and promotes data portability. However, it can also lead to new security risks and technical debt. Our position is that the ML community should embrace this development while building the appropriate frameworks to mitigate the downsides. By acting now, we can harness AI to restore user freedom and competitive markets without sacrificing security.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13554v2">LLMs' Leaning in European Elections</a></div>
    <div class="paper-meta">
      📅 2025-07-12
    </div>
    <details class="paper-abstract">
      Many studies suggest that LLMs have left wing leans. The article extends previous analysis of US presidential elections considering several virtual elections in multiple European countries. The analysis considers multiple LLMs and the results confirm the extent of the leaning. Furthermore, the results show that the leaning is not uniform between countries. Sometimes, models refuse to take a position in the virtual elections, but the refusal rate itself is not uniform between countries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09255v1">StockSim: A Dual-Mode Order-Level Simulator for Evaluating Multi-Agent LLMs in Financial Markets</a></div>
    <div class="paper-meta">
      📅 2025-07-12
    </div>
    <details class="paper-abstract">
      We present StockSim, an open-source simulation platform for systematic evaluation of large language models (LLMs) in realistic financial decision-making scenarios. Unlike previous toolkits that offer limited scope, StockSim delivers a comprehensive system that fully models market dynamics and supports diverse simulation modes of varying granularity. It incorporates critical real-world factors, such as latency, slippage, and order-book microstructure, that were previously neglected, enabling more faithful and insightful assessment of LLM-based trading agents. An extensible, role-based agent framework supports heterogeneous trading strategies and multi-agent coordination, making StockSim a uniquely capable testbed for NLP research on reasoning under uncertainty and sequential decision-making. We open-source all our code at https: //github.com/harrypapa2002/StockSim.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07186v2">Planted in Pretraining, Swayed by Finetuning: A Case Study on the Origins of Cognitive Biases in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-12
      | 💬 CoLM 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit cognitive biases -- systematic tendencies of irrational decision-making, similar to those seen in humans. Prior work has found that these biases vary across models and can be amplified by instruction tuning. However, it remains unclear if these differences in biases stem from pretraining, finetuning, or even random noise due to training stochasticity. We propose a two-step causal experimental approach to disentangle these factors. First, we finetune models multiple times using different random seeds to study how training randomness affects over $30$ cognitive biases. Second, we introduce \emph{cross-tuning} -- swapping instruction datasets between models to isolate bias sources. This swap uses datasets that led to different bias patterns, directly testing whether biases are dataset-dependent. Our findings reveal that while training randomness introduces some variability, biases are mainly shaped by pretraining: models with the same pretrained backbone exhibit more similar bias patterns than those sharing only finetuning data. These insights suggest that understanding biases in finetuned models requires considering their pretraining origins beyond finetuning effects. This perspective can guide future efforts to develop principled strategies for evaluating and mitigating bias in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.23502v2">LLM-enhanced Action-aware Multi-modal Prompt Tuning for Image-Text Matching</a></div>
    <div class="paper-meta">
      📅 2025-07-12
      | 💬 accepted by ICCV 2025
    </div>
    <details class="paper-abstract">
      Driven by large-scale contrastive vision-language pre-trained models such as CLIP, recent advancements in the image-text matching task have achieved remarkable success in representation learning. Due to image-level visual-language alignment, CLIP falls short in understanding fine-grained details such as object attributes and spatial relationships between objects. Recent efforts have attempted to compel CLIP to acquire structured visual representations by introducing prompt learning to achieve object-level alignment. While achieving promising results, they still lack the capability to perceive actions, which are crucial for describing the states or relationships between objects. Therefore, we propose to endow CLIP with fine-grained action-level understanding by introducing an LLM-enhanced action-aware multi-modal prompt-tuning method, incorporating the action-related external knowledge generated by large language models (LLMs). Specifically, we design an action triplet prompt and an action state prompt to exploit compositional semantic knowledge and state-related causal knowledge implicitly stored in LLMs. Subsequently, we propose an adaptive interaction module to aggregate attentive visual features conditioned on action-aware prompted knowledge for establishing discriminative and action-aware visual representations, which further improves the performance. Comprehensive experimental results on two benchmark datasets demonstrate the effectiveness of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09135v1">Position Paper: Programming Language Techniques for Bridging LLM Code Generation Semantic Gaps</a></div>
    <div class="paper-meta">
      📅 2025-07-12
    </div>
    <details class="paper-abstract">
      Large Language Models have demonstrated remarkable capabilities in automated code generation, yet their statistical nature and black-box characteristics create significant semantic gaps manifested through syntax errors, semantic hallucinations, and reliability concerns. This position paper argues that principled integration of Programming Language (PL) techniques is essential for bridging these gaps. Through structured program representations, formal correctness guarantees, and robust verification mechanisms, PL techniques can elevate LLM-generated code from statistical pattern matching to truly reliable and trustworthy levels. This integration is crucial for developing systems that generate code that is not only functionally correct but also interpretable, verifiable, and ultimately trustworthy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09116v1">Mixture of LoRA Experts with Multi-Modal and Multi-Granularity LLM Generative Error Correction for Accented Speech Recognition</a></div>
    <div class="paper-meta">
      📅 2025-07-12
      | 💬 IEEE Transactions on Audio, Speech and Language Processing
    </div>
    <details class="paper-abstract">
      Despite substantial improvements in ASR, performance tends to degrade when faced with adverse conditions such as speaker accents. Generative error correction (GER) leverages the rich linguistic knowledge and exceptional reasoning ability of LLMs, significantly outperforming typical LM methods. However, it lacks specificity in accented speech scenarios. In this study, we leverage GER to improve the accuracy of transcription predictions by addressing the two primary features of accented speech recognition. To fully leverage pronunciation information, we propose the multi-modal GER, which integrates pronunciation information from the speech modality, and the multi-granularity GER, which incorporates fine-grained phoneme-level information related to pronunciation. These two methods enable the LLM to utilize the pronunciation information of accented speech and the semantic information from word-level hypotheses for accurate transcription predictions through LoRA fine-tuning. On the one hand, we employ a three-stage training strategy to train separate multi-modal GER models for each accent to obtain mono-accent LoRA experts. By adopting our proposed HDMoLE method, which incorporates hierarchical routing and dynamic thresholds within the mixture of LoRA experts, we effectively merge multiple mono-accent LoRA experts within a single multi-modal GER to overcome the challenges posed by accent diversity. On the other hand, multi-granularity GER leverages the N-best word-level and phoneme-level hypotheses generated by the HDMoLE model to predict the final accented speech transcriptions. Experimental results on the multi-accent English dataset demonstrate the efficacy of our proposed methods. Our methods achieve a remarkable relative WER reduction of 67.35% compared to the Whisper-large-v3 baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12318v2">UTF:Undertrained Tokens as Fingerprints A Novel Approach to LLM Identification</a></div>
    <div class="paper-meta">
      📅 2025-07-12
    </div>
    <details class="paper-abstract">
      Fingerprinting large language models (LLMs) is essential for verifying model ownership, ensuring authenticity, and preventing misuse. Traditional fingerprinting methods often require significant computational overhead or white-box verification access. In this paper, we introduce UTF, a novel and efficient approach to fingerprinting LLMs by leveraging under-trained tokens. Under-trained tokens are tokens that the model has not fully learned during its training phase. By utilizing these tokens, we perform supervised fine-tuning to embed specific input-output pairs into the model. This process allows the LLM to produce predetermined outputs when presented with certain inputs, effectively embedding a unique fingerprint. Our method has minimal overhead and impact on model's performance, and does not require white-box access to target model's ownership identification. Compared to existing fingerprinting methods, UTF is also more effective and robust to fine-tuning and random guess.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.09750v2">Pinning "Reflection" on the Agenda: Investigating Reflection in Human-LLM Co-Creation for Creative Coding</a></div>
    <div class="paper-meta">
      📅 2025-07-12
      | 💬 6 pages, 2 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly integrated into creative coding, yet how users reflect, and how different co-creation conditions influence reflective behavior, remains underexplored. This study investigates situated, moment-to-moment reflection in creative coding under two prompting strategies: the entire task invocation (T1) and decomposed subtask invocation (T2), to examine their effects on reflective behavior. Our mixed-method results reveal three distinct reflection types and show that T2 encourages more frequent, strategic, and generative reflection, fostering diagnostic reasoning and goal redefinition. These findings offer insights into how LLM-based tools foster deeper creative engagement through structured, behaviorally grounded reflection support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10596v1">PLEX: Perturbation-free Local Explanations for LLM-Based Text Classification</a></div>
    <div class="paper-meta">
      📅 2025-07-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel in text classification, but their complexity hinders interpretability, making it difficult to understand the reasoning behind their predictions. Explainable AI (XAI) methods like LIME and SHAP offer local explanations by identifying influential words, but they rely on computationally expensive perturbations. These methods typically generate thousands of perturbed sentences and perform inferences on each, incurring a substantial computational burden, especially with LLMs. To address this, we propose \underline{P}erturbation-free \underline{L}ocal \underline{Ex}planation (PLEX), a novel method that leverages the contextual embeddings extracted from the LLM and a ``Siamese network" style neural network trained to align with feature importance scores. This one-off training eliminates the need for subsequent perturbations, enabling efficient explanations for any new sentence. We demonstrate PLEX's effectiveness on four different classification tasks (sentiment, fake news, fake COVID-19 news and depression), showing more than 92\% agreement with LIME and SHAP. Our evaluation using a ``stress test" reveals that PLEX accurately identifies influential words, leading to a similar decline in classification accuracy as observed with LIME and SHAP when these words are removed. Notably, in some cases, PLEX demonstrates superior performance in capturing the impact of key features. PLEX dramatically accelerates explanation, reducing time and computational overhead by two and four orders of magnitude, respectively. This work offers a promising solution for explainable LLM-based text classification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.12480v1">LLM-Powered Quantum Code Transpilation</a></div>
    <div class="paper-meta">
      📅 2025-07-12
      | 💬 IEEE International Conference on Quantum Computing and Engineering (QCE) 2025 - Extended Abstract
    </div>
    <details class="paper-abstract">
      There exist various Software Development Kits (SDKs) tailored to different quantum computing platforms. These are known as Quantum SDKs (QSDKs). Examples include but are not limited to Qiskit, Cirq, and PennyLane. However, this diversity presents significant challenges for interoperability and cross-platform development of hybrid quantum-classical software systems. Traditional rule-based transpilers for translating code between QSDKs are time-consuming to design and maintain, requiring deep expertise and rigid mappings in the source and destination code. In this study, we explore the use of Large Language Models (LLMs) as a flexible and automated solution. Leveraging their pretrained knowledge and contextual reasoning capabilities, we position LLMs as programming language-agnostic transpilers capable of converting quantum programs from one QSDK to another while preserving functional equivalence. Our approach eliminates the need for manually defined transformation rules and offers a scalable solution to quantum software portability. This work represents a step toward enabling intelligent, general-purpose transpilation in the quantum computing ecosystem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08794v1">One Token to Fool LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-07-11
    </div>
    <details class="paper-abstract">
      Generative reward models (also known as LLMs-as-judges), which use large language models (LLMs) to evaluate answer quality, are increasingly adopted in reinforcement learning with verifiable rewards (RLVR). They are often preferred over rigid rule-based metrics, especially for complex reasoning tasks involving free-form outputs. In this paradigm, an LLM is typically prompted to compare a candidate answer against a ground-truth reference and assign a binary reward indicating correctness. Despite the seeming simplicity of this comparison task, we find that generative reward models exhibit surprising vulnerabilities to superficial manipulations: non-word symbols (e.g., ":" or ".") or reasoning openers like "Thought process:" and "Let's solve this problem step by step." can often lead to false positive rewards. We demonstrate that this weakness is widespread across LLMs, datasets, and prompt formats, posing a serious threat for core algorithmic paradigms that rely on generative reward models, such as rejection sampling, preference optimization, and RLVR. To mitigate this issue, we introduce a simple yet effective data augmentation strategy and train a new generative reward model with substantially improved robustness. Our findings highlight the urgent need for more reliable LLM-based evaluation methods. We release our robust, general-domain reward model and its synthetic training data at https://huggingface.co/sarosavo/Master-RM and https://huggingface.co/datasets/sarosavo/Master-RM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08671v1">LLMCup: Ranking-Enhanced Comment Updating with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-11
      | 💬 13 pages, 10 figures
    </div>
    <details class="paper-abstract">
      While comments are essential for enhancing code readability and maintainability in modern software projects, developers are often motivated to update code but not comments, leading to outdated or inconsistent documentation that hinders future understanding and maintenance. Recent approaches such as CUP and HebCup have attempted automatic comment updating using neural sequence-to-sequence models and heuristic rules, respectively. However, these methods can miss or misinterpret crucial information during comment updating, resulting in inaccurate comments, and they often struggle with complex update scenarios. Given these challenges, a promising direction lies in leveraging large language models (LLMs), which have shown impressive performance in software engineering tasks such as comment generation, code synthesis, and program repair. This suggests their strong potential to capture the logic behind code modifications - an ability that is crucial for the task of comment updating. Nevertheless, selecting an appropriate prompt strategy for an LLM on each update case remains challenging. To address this, we propose a novel comment updating framework, LLMCup, which first uses multiple prompt strategies to provide diverse candidate updated comments via an LLM, and then employs a ranking model, CupRank, to select the best candidate as final updated comment. Experimental results demonstrate the effectiveness of LLMCup, with improvements over state-of-the-art baselines (CUP and HebCup) by 49.0%-116.9% in Accuracy, 10.8%-20% in BLEU-4, 4.6% in METEOR, 0.9%-1.9% in F1, and 2.1%-3.4% in SentenceBert similarity. Furthermore, a user study shows that comments updated by LLMCup sometimes surpass human-written updates, highlighting the importance of incorporating human evaluation in comment quality assessment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08627v1">NL in the Middle: Code Translation with LLMs and Intermediate Representations</a></div>
    <div class="paper-meta">
      📅 2025-07-11
    </div>
    <details class="paper-abstract">
      Studies show that large language models (LLMs) produce buggy code translations. One avenue to improve translation accuracy is through intermediate representations, which could provide structured insights to guide the model's understanding. We explore whether code translation using LLMs can benefit from intermediate representations via natural language (NL) and abstract syntax trees (ASTs). Since prompt engineering greatly affects LLM performance, we consider several ways to integrate these representations, from one-shot to chain-of-thought (CoT) prompting. Using Open Gpt4 8X7B and specialized StarCoder and CodeGen models on popular code translation benchmarks (CodeNet and AVATAR), we find that CoT with an intermediate NL summary performs best, with an increase of 13.8% and 6.7%, respectively, in successful translations for the best-performing model (Open Gpt4 8X7B) compared to the zero-shot prompt.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08621v1">A comprehensive study of LLM-based argument classification: from LLAMA through GPT-4o to Deepseek-R1</a></div>
    <div class="paper-meta">
      📅 2025-07-11
    </div>
    <details class="paper-abstract">
      Argument mining (AM) is an interdisciplinary research field that integrates insights from logic, philosophy, linguistics, rhetoric, law, psychology, and computer science. It involves the automatic identification and extraction of argumentative components, such as premises and claims, and the detection of relationships between them, such as support, attack, or neutrality. Recently, the field has advanced significantly, especially with the advent of large language models (LLMs), which have enhanced the efficiency of analyzing and extracting argument semantics compared to traditional methods and other deep learning models. There are many benchmarks for testing and verifying the quality of LLM, but there is still a lack of research and results on the operation of these models in publicly available argument classification databases. This paper presents a study of a selection of LLM's, using diverse datasets such as Args.me and UKP. The models tested include versions of GPT, Llama, and DeepSeek, along with reasoning-enhanced variants incorporating the Chain-of-Thoughts algorithm. The results indicate that ChatGPT-4o outperforms the others in the argument classification benchmarks. In case of models incorporated with reasoning capabilities, the Deepseek-R1 shows its superiority. However, despite their superiority, GPT-4o and Deepseek-R1 still make errors. The most common errors are discussed for all models. To our knowledge, the presented work is the first broader analysis of the mentioned datasets using LLM and prompt algorithms. The work also shows some weaknesses of known prompt algorithms in argument analysis, while indicating directions for their improvement. The added value of the work is the in-depth analysis of the available argument datasets and the demonstration of their shortcomings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08311v2">Mind the Memory Gap: Unveiling GPU Bottlenecks in Large-Batch LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-07-11
      | 💬 Pol G. Recasens, Ferran Agullo: equal contribution. Paper accepted at IEEE CLOUD 2025
    </div>
    <details class="paper-abstract">
      Large language models have been widely adopted across different tasks, but their auto-regressive generation nature often leads to inefficient resource utilization during inference. While batching is commonly used to increase throughput, performance gains plateau beyond a certain batch size, especially with smaller models, a phenomenon that existing literature typically explains as a shift to the compute-bound regime. In this paper, through an in-depth GPU-level analysis, we reveal that large-batch inference remains memory-bound, with most GPU compute capabilities underutilized due to DRAM bandwidth saturation as the primary bottleneck. To address this, we propose a Batching Configuration Advisor (BCA) that optimizes memory allocation, reducing GPU memory requirements with minimal impact on throughput. The freed memory and underutilized GPU compute capabilities can then be leveraged by concurrent workloads. Specifically, we use model replication to improve serving throughput and GPU utilization. Our findings challenge conventional assumptions about LLM inference, offering new insights and practical strategies for improving resource utilization, particularly for smaller language models. The code is publicly available at https://github.com/FerranAgulloLopez/vLLMBatchingMemoryGap.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08616v1">AgentsNet: Coordination and Collaborative Reasoning in Multi-Agent LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-11
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Large-language models (LLMs) have demonstrated powerful problem-solving capabilities, in particular when organized in multi-agent systems. However, the advent of such systems also raises several questions on the ability of a complex network of agents to effectively self-organize and collaborate. While measuring performance on standard reasoning benchmarks indicates how well multi-agent systems can solve reasoning tasks, it is unclear whether these systems are able to leverage their topology effectively. Here, we propose AgentsNet, a new benchmark for multi-agent reasoning. By drawing inspiration from classical problems in distributed systems and graph theory, AgentsNet measures the ability of multi-agent systems to collaboratively form strategies for problem-solving, self-organization, and effective communication given a network topology. We evaluate a variety of baseline methods on AgentsNet including homogeneous networks of agents which first have to agree on basic protocols for organization and communication. We find that some frontier LLMs are already demonstrating strong performance for small networks but begin to fall off once the size of the network scales. While existing multi-agent benchmarks cover at most 2-5 agents, AgentsNet is practically unlimited in size and can scale with new generations of LLMs. As such, we also probe frontier models in a setup with up to 100 agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08538v1">The AI Language Proficiency Monitor -- Tracking the Progress of LLMs on Multilingual Benchmarks</a></div>
    <div class="paper-meta">
      📅 2025-07-11
    </div>
    <details class="paper-abstract">
      To ensure equitable access to the benefits of large language models (LLMs), it is essential to evaluate their capabilities across the world's languages. We introduce the AI Language Proficiency Monitor, a comprehensive multilingual benchmark that systematically assesses LLM performance across up to 200 languages, with a particular focus on low-resource languages. Our benchmark aggregates diverse tasks including translation, question answering, math, and reasoning, using datasets such as FLORES+, MMLU, GSM8K, TruthfulQA, and ARC. We provide an open-source, auto-updating leaderboard and dashboard that supports researchers, developers, and policymakers in identifying strengths and gaps in model performance. In addition to ranking models, the platform offers descriptive insights such as a global proficiency map and trends over time. By complementing and extending prior multilingual benchmarks, our work aims to foster transparency, inclusivity, and progress in multilingual AI. The system is available at https://huggingface.co/spaces/fair-forward/evals-for-every-language.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08523v1">InferLog: Accelerating LLM Inference for Online Log Parsing via ICL-oriented Prefix Caching</a></div>
    <div class="paper-meta">
      📅 2025-07-11
    </div>
    <details class="paper-abstract">
      Modern software systems generate massive volumes of runtime logs, necessitating efficient and accurate log parsing to enable critical downstream tasks such as anomaly detection and root cause analysis. Recently, large language models (LLMs) have achieved advanced accuracy on log parsing, but their deployment in production environments faces two major limitations: (1) the privacy risks associated with commercial LLMs, driving the adoption of local deployment, and (2) the stringent latency and throughput requirements imposed by high-volume log streams, which existing LLM-based parsers fail to meet. Although recent efforts have reduced the number of LLM queries, they overlook the high latency of the LLM invocations, where concurrent log parsing requests can cause serve performance degradation of LLM inference system. In this study, we present InferLog, the first LLM inference optimization method for online log parsing. Our key insight is that the inference efficiency emerges as the vital bottleneck in LLM-based online log parsing, rather than parsing accuracy. InferLog accelerates inference by designing (1) A Prefix-aware ICL Refinement policy to refine the examples and permutation of in-context learning to improve the prefix caching efficiency. (2) A rapid and task-specific configuration tuning pipeline based on meta-learning to find the optimal LLM scheduling-related configuration for dynamic log parsing workloads. The experimental results based on Loghub dataset and vLLM demonstrate that InferLog significantly outperforms existing inference optimization methods and markedly accelerates the state-of-the-art LLM-based log parser without compromising parsing accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08513v1">Advancing Multimodal LLMs by Large-Scale 3D Visual Instruction Dataset Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-11
    </div>
    <details class="paper-abstract">
      Multimodal Large Language Models (MLLMs) struggle with accurately capturing camera-object relations, especially for object orientation, camera viewpoint, and camera shots. This stems from the fact that existing MLLMs are trained on images with limited diverse camera-object relations and corresponding textual descriptions. To address this, we propose a synthetic generation pipeline to create large-scale 3D visual instruction datasets. Our framework takes 3D assets as input and uses rendering and diffusion-based image generation models to create photorealistic images preserving precise camera-object relations. Additionally, large language models (LLMs) are used to generate text prompts for guiding visual instruction tuning and controlling image generation. We create Ultimate3D, a dataset of 240K VQAs with precise camera-object annotations, and corresponding benchmark. MLLMs fine-tuned on our proposed dataset outperform commercial models by a large margin, achieving an average accuracy improvement of 33.4% on camera-object relation recognition tasks. Our code, dataset, and benchmark will contribute to broad MLLM applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08498v1">Semantic-Augmented Latent Topic Modeling with LLM-in-the-Loop</a></div>
    <div class="paper-meta">
      📅 2025-07-11
    </div>
    <details class="paper-abstract">
      Latent Dirichlet Allocation (LDA) is a prominent generative probabilistic model used for uncovering abstract topics within document collections. In this paper, we explore the effectiveness of augmenting topic models with Large Language Models (LLMs) through integration into two key phases: Initialization and Post-Correction. Since the LDA is highly dependent on the quality of its initialization, we conduct extensive experiments on the LLM-guided topic clustering for initializing the Gibbs sampling algorithm. Interestingly, the experimental results reveal that while the proposed initialization strategy improves the early iterations of LDA, it has no effect on the convergence and yields the worst performance compared to the baselines. The LLM-enabled post-correction, on the other hand, achieved a promising improvement of 5.86% in the coherence evaluation. These results highlight the practical benefits of the LLM-in-the-loop approach and challenge the belief that LLMs are always the superior text mining alternative.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08491v1">A Third Paradigm for LLM Evaluation: Dialogue Game-Based Evaluation using clembench</a></div>
    <div class="paper-meta">
      📅 2025-07-11
      | 💬 All code required to run the benchmark, as well as extensive documentation, is available at https://github.com/clembench/clembench
    </div>
    <details class="paper-abstract">
      There are currently two main paradigms for evaluating large language models (LLMs), reference-based evaluation and preference-based evaluation. The first, carried over from the evaluation of machine learning models in general, relies on pre-defined task instances, for which reference task executions are available. The second, best exemplified by the LM-arena, relies on (often self-selected) users bringing their own intents to a site that routes these to several models in parallel, among whose responses the user then selects their most preferred one. The former paradigm hence excels at control over what is tested, while the latter comes with higher ecological validity, testing actual use cases interactively. Recently, a third complementary paradigm has emerged that combines some of the strengths of these approaches, offering control over multi-turn, reference-free, repeatable interactions, while stressing goal-directedness: dialogue game based evaluation. While the utility of this approach has been shown by several projects, its adoption has been held back by the lack of a mature, easily re-usable implementation. In this paper, we present clembench, which has been in continuous development since 2023 and has in its latest release been optimized for ease of general use. We describe how it can be used to benchmark one's own models (using a provided set of benchmark game instances in English), as well as how easily the benchmark itself can be extended with new, tailor-made targeted tests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08472v1">Pre-Training LLMs on a budget: A comparison of three optimizers</a></div>
    <div class="paper-meta">
      📅 2025-07-11
    </div>
    <details class="paper-abstract">
      Optimizers play a decisive role in reducing pre-training times for LLMs and achieving better-performing models. In this study, we compare three major variants: the de-facto standard AdamW, the simpler Lion, developed through an evolutionary search, and the second-order optimizer Sophia. For better generalization, we train with two different base architectures and use a single- and a multiple-epoch approach while keeping the number of tokens constant. Using the Maximal Update Parametrization and smaller proxy models, we tune relevant hyperparameters separately for each combination of base architecture and optimizer. We found that while the results from all three optimizers were in approximately the same range, Sophia exhibited the lowest training and validation loss, Lion was fastest in terms of training GPU hours but AdamW led to the best downstream evaluation results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06850v3">The Dark Side of LLMs Agent-based Attacks for Complete Computer Takeover</a></div>
    <div class="paper-meta">
      📅 2025-07-11
    </div>
    <details class="paper-abstract">
      The rapid adoption of Large Language Model (LLM) agents and multi-agent systems enables unprecedented capabilities in natural language processing and generation. However, these systems have introduced unprecedented security vulnerabilities that extend beyond traditional prompt injection attacks. This paper presents the first comprehensive evaluation of LLM agents as attack vectors capable of achieving complete computer takeover through the exploitation of trust boundaries within agentic AI systems where autonomous entities interact and influence each other. We demonstrate that adversaries can leverage three distinct attack surfaces - direct prompt injection, RAG backdoor attacks, and inter-agent trust exploitation - to coerce popular LLMs (including GPT-4o, Claude-4 and Gemini-2.5) into autonomously installing and executing malware on victim machines. Our evaluation of 17 state-of-the-art LLMs reveals an alarming vulnerability hierarchy: while 41.2% of models succumb to direct prompt injection, 52.9% are vulnerable to RAG backdoor attacks, and a critical 82.4% can be compromised through inter-agent trust exploitation. Notably, we discovered that LLMs which successfully resist direct malicious commands will execute identical payloads when requested by peer agents, revealing a fundamental flaw in current multi-agent security models. Our findings demonstrate that only 5.9% of tested models (1/17) proved resistant to all attack vectors, with the majority exhibiting context-dependent security behaviors that create exploitable blind spots. Our findings also highlight the need to increase awareness and research on the security risks of LLMs, showing a paradigm shift in cybersecurity threats, where AI tools themselves become sophisticated attack vectors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08427v1">ChainEdit: Propagating Ripple Effects in LLM Knowledge Editing through Logical Rule-Guided Chains</a></div>
    <div class="paper-meta">
      📅 2025-07-11
      | 💬 Accepted to ACL 2025 (main)
    </div>
    <details class="paper-abstract">
      Current knowledge editing methods for large language models (LLMs) struggle to maintain logical consistency when propagating ripple effects to associated facts. We propose ChainEdit, a framework that synergizes knowledge graph-derived logical rules with LLM logical reasoning capabilities to enable systematic chain updates. By automatically extracting logical patterns from structured knowledge bases and aligning them with LLMs' internal logics, ChainEdit dynamically generates and edits logically connected knowledge clusters. Experiments demonstrate an improvement of more than 30% in logical generalization over baselines while preserving editing reliability and specificity. We further address evaluation biases in existing benchmarks through knowledge-aware protocols that disentangle external dependencies. This work establishes new state-of-the-art performance on ripple effect while ensuring internal logical consistency after knowledge editing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08392v1">Multi-Agent LLMs as Ethics Advocates in AI-Based Systems</a></div>
    <div class="paper-meta">
      📅 2025-07-11
    </div>
    <details class="paper-abstract">
      Incorporating ethics into the requirement elicitation process is essential for creating ethically aligned systems. Although eliciting manual ethics requirements is effective, it requires diverse input from multiple stakeholders, which can be challenging due to time and resource constraints. Moreover, it is often given a low priority in the requirements elicitation process. This study proposes a framework for generating ethics requirements drafts by introducing an ethics advocate agent in a multi-agent LLM setting. This agent critiques and provides input on ethical issues based on the system description. The proposed framework is evaluated through two case studies from different contexts, demonstrating that it captures the majority of ethics requirements identified by researchers during 30-minute interviews and introduces several additional relevant requirements. However, it also highlights reliability issues in generating ethics requirements, emphasizing the need for human feedback in this sensitive domain. We believe this work can facilitate the broader adoption of ethics in the requirements engineering process, ultimately leading to more ethically aligned products.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08350v1">Exploring Design of Multi-Agent LLM Dialogues for Research Ideation</a></div>
    <div class="paper-meta">
      📅 2025-07-11
      | 💬 16 pages, 1 figure, appendix. Accepted to SIGDIAL 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to support creative tasks such as research idea generation. While recent work has shown that structured dialogues between LLMs can improve the novelty and feasibility of generated ideas, the optimal design of such interactions remains unclear. In this study, we conduct a comprehensive analysis of multi-agent LLM dialogues for scientific ideation. We compare different configurations of agent roles, number of agents, and dialogue depth to understand how these factors influence the novelty and feasibility of generated ideas. Our experimental setup includes settings where one agent generates ideas and another critiques them, enabling iterative improvement. Our results show that enlarging the agent cohort, deepening the interaction depth, and broadening agent persona heterogeneity each enrich the diversity of generated ideas. Moreover, specifically increasing critic-side diversity within the ideation-critique-revision loop further boosts the feasibility of the final proposals. Our findings offer practical guidelines for building effective multi-agent LLM systems for scientific ideation. Our code is available at https://github.com/g6000/MultiAgent-Research-Ideator.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08339v1">What Factors Affect LLMs and RLLMs in Financial Question Answering?</a></div>
    <div class="paper-meta">
      📅 2025-07-11
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Recently, the development of large language models (LLMs) and reasoning large language models (RLLMs) have gained considerable attention from many researchers. RLLMs enhance the reasoning capabilities of LLMs through Long Chain-of-Thought (Long CoT) processes, significantly improving the performance of LLMs in addressing complex problems. However, there are few works that systematically explore what methods can fully unlock the performance of LLMs and RLLMs within the financial domain. To investigate the impact of various methods on LLMs and RLLMs, we utilize five LLMs and three RLLMs to assess the effects of prompting methods, agentic frameworks, and multilingual alignment methods on financial question-answering tasks. Our research findings indicate: (1) Current prompting methods and agent frameworks enhance the performance of LLMs in financial question answering by simulating Long CoT; (2) RLLMs possess inherent Long CoT capabilities, which limits the effectiveness of conventional methods in further enhancing their performance; (3) Current advanced multilingual alignment methods primarily improve the multilingual performance of LLMs by extending the reasoning length, which yields minimal benefits for RLLMs. We hope that this study can serve as an important reference for LLMs and RLLMs in the field of financial question answering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.01077v4">Emoji Attack: Enhancing Jailbreak Attacks Against Judge LLM Detection</a></div>
    <div class="paper-meta">
      📅 2025-07-11
    </div>
    <details class="paper-abstract">
      Jailbreaking techniques trick Large Language Models (LLMs) into producing restricted output, posing a potential threat. One line of defense is to use another LLM as a Judge to evaluate the harmfulness of generated text. However, we reveal that these Judge LLMs are vulnerable to token segmentation bias, an issue that arises when delimiters alter the tokenization process, splitting words into smaller sub-tokens. This alters the embeddings of the entire sequence, reducing detection accuracy and allowing harmful content to be misclassified as safe. In this paper, we introduce Emoji Attack, a novel strategy that amplifies existing jailbreak prompts by exploiting token segmentation bias. Our method leverages in-context learning to systematically insert emojis into text before it is evaluated by a Judge LLM, inducing embedding distortions that significantly lower the likelihood of detecting unsafe content. Unlike traditional delimiters, emojis also introduce semantic ambiguity, making them particularly effective in this attack. Through experiments on state-of-the-art Judge LLMs, we demonstrate that Emoji Attack substantially reduces the unsafe prediction rate, bypassing existing safeguards.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08325v1">CRMAgent: A Multi-Agent LLM System for E-Commerce CRM Message Template Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-11
    </div>
    <details class="paper-abstract">
      In e-commerce private-domain channels such as instant messaging and e-mail, merchants engage customers directly as part of their Customer Relationship Management (CRM) programmes to drive retention and conversion. While a few top performers excel at crafting outbound messages, most merchants struggle to write persuasive copy because they lack both expertise and scalable tools. We introduce CRMAgent, a multi-agent system built on large language models (LLMs) that generates high-quality message templates and actionable writing guidance through three complementary modes. First, group-based learning enables the agent to learn from a merchant's own top-performing messages within the same audience segment and rewrite low-performing ones. Second, retrieval-and-adaptation fetches templates that share the same audience segment and exhibit high similarity in voucher type and product category, learns their successful patterns, and adapts them to the current campaign. Third, a rule-based fallback provides a lightweight zero-shot rewrite when no suitable references are available. Extensive experiments show that CRMAgent consistently outperforms merchants' original templates, delivering significant gains in both audience-match and marketing-effectiveness metrics.
    </details>
</div>
