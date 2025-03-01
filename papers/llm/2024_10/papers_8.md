# llm - 2024_10

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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.20707v1">DisasterQA: A Benchmark for Assessing the performance of LLMs in Disaster Response</a></div>
    <div class="paper-meta">
      📅 2024-10-09
      | 💬 7 pages, 6 tables
    </div>
    <details class="paper-abstract">
      Disasters can result in the deaths of many, making quick response times vital. Large Language Models (LLMs) have emerged as valuable in the field. LLMs can be used to process vast amounts of textual information quickly providing situational context during a disaster. However, the question remains whether LLMs should be used for advice and decision making in a disaster. To evaluate the capabilities of LLMs in disaster response knowledge, we introduce a benchmark: DisasterQA created from six online sources. The benchmark covers a wide range of disaster response topics. We evaluated five LLMs each with four different prompting methods on our benchmark, measuring both accuracy and confidence levels through Logprobs. The results indicate that LLMs require improvement on disaster response knowledge. We hope that this benchmark pushes forth further development of LLMs in disaster response, ultimately enabling these models to work alongside. emergency managers in disasters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.06431v1">Functional-level Uncertainty Quantification for Calibrated Fine-tuning on LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-09
    </div>
    <details class="paper-abstract">
      From common-sense reasoning to domain-specific tasks, parameter-efficient fine tuning (PEFT) methods for large language models (LLMs) have showcased significant performance improvements on downstream tasks. However, fine-tuned LLMs often struggle with overconfidence in uncertain predictions, particularly due to sparse training data. This overconfidence reflects poor epistemic uncertainty calibration, which arises from limitations in the model's ability to generalize with limited data. Existing PEFT uncertainty quantification methods for LLMs focus on the post fine-tuning stage and thus have limited capability in calibrating epistemic uncertainty. To address these limitations, we propose Functional-Level Uncertainty Quantification for Calibrated Fine-Tuning (UQ4CT), which captures and calibrates functional-level epistemic uncertainty during the fine-tuning stage via a mixture-of-expert framework. We show that UQ4CT reduces Expected Calibration Error (ECE) by more than $25\%$ while maintaining high accuracy across $5$ benchmarks. Furthermore, UQ4CT maintains superior ECE performance with high accuracy under distribution shift, showcasing improved generalizability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10872v1">ToolBridge: An Open-Source Dataset to Equip LLMs with External Tool Capabilities</a></div>
    <div class="paper-meta">
      📅 2024-10-08
      | 💬 technical report
    </div>
    <details class="paper-abstract">
      Through the integration of external tools, large language models (LLMs) such as GPT-4o and Llama 3.1 significantly expand their functional capabilities, evolving from elementary conversational agents to general-purpose assistants. We argue that the primary drivers of these advancements are the quality and diversity of the training data. However, the existing LLMs with external tool integration provide only limited transparency regarding their datasets and data collection methods, which has led to the initiation of this research. Specifically, in this paper, our objective is to elucidate the detailed process involved in constructing datasets that empower LLMs to effectively learn how to utilize external tools and make this information available to the public through the introduction of ToolBridge. ToolBridge proposes to employ a collection of general open-access datasets as its raw dataset pool and applies a series of strategies to identify appropriate data entries from the pool for external tool API insertions. By supervised fine-tuning on these curated data entries, LLMs can invoke external tools in appropriate contexts to boost their predictive accuracy, particularly for basic functions including data processing, numerical computation, and factual retrieval. Our experiments rigorously isolates model architectures and training configurations, focusing exclusively on the role of data. The experimental results indicate that LLMs trained on ToolBridge demonstrate consistent performance improvements on both standard benchmarks and custom evaluation datasets. All the associated code and data will be open-source at https://github.com/CharlesPikachu/ToolBridge, promoting transparency and facilitating the broader community to explore approaches for equipping LLMs with external tools capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.06351v1">Moving Faster and Reducing Risk: Using LLMs in Release Deployment</a></div>
    <div class="paper-meta">
      📅 2024-10-08
    </div>
    <details class="paper-abstract">
      Release engineering has traditionally focused on continuously delivering features and bug fixes to users, but at a certain scale, it becomes impossible for a release engineering team to determine what should be released. At Meta's scale, the responsibility appropriately and necessarily falls back on the engineer writing and reviewing the code. To address this challenge, we developed models of diff risk scores (DRS) to determine how likely a diff is to cause a SEV, i.e., a severe fault that impacts end-users. Assuming that SEVs are only caused by diffs, a naive model could randomly gate X% of diffs from landing, which would automatically catch X% of SEVs on average. However, we aimed to build a model that can capture Y% of SEVs by gating X% of diffs, where Y >> X. By training the model on historical data on diffs that have caused SEVs in the past, we can predict the riskiness of an outgoing diff to cause a SEV. Diffs that are beyond a particular threshold of risk can then be gated. We have four types of gating: no gating (green), weekend gating (weekend), medium impact on end-users (yellow), and high impact on end-users (red). The input parameter for our models is the level of gating, and the outcome measure is the number of captured SEVs. Our research approaches include a logistic regression model, a BERT-based model, and generative LLMs. Our baseline regression model captures 18.7%, 27.9%, and 84.6% of SEVs while respectively gating the top 5% (weekend), 10% (yellow), and 50% (red) of risky diffs. The BERT-based model, StarBERT, only captures 0.61x, 0.85x, and 0.81x as many SEVs as the logistic regression for the weekend, yellow, and red gating zones, respectively. The generative LLMs, iCodeLlama-34B and iDiffLlama-13B, when risk-aligned, capture more SEVs than the logistic regression model in production: 1.40x, 1.52x, 1.05x, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.06287v1">Non-Halting Queries: Exploiting Fixed Points in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-08
    </div>
    <details class="paper-abstract">
      We introduce a new vulnerability that exploits fixed points in autoregressive models and use it to craft queries that never halt, i.e. an LLM output that does not terminate. More precisely, for what we call non-halting queries, the LLM never samples the end-of-string token (<eos>). We rigorously analyze the conditions under which the non-halting anomaly presents itself. In particular, at temperature zero, we prove that if a repeating (cyclic) sequence of tokens is observed at the output beyond the context size, then the LLM does not halt. We demonstrate the non-halting anomaly in a number of experiments performed in base (unaligned) models where repeating tokens immediately lead to a non-halting cyclic behavior as predicted by the analysis. Further, we develop a simple recipe that takes the same fixed points observed in the base model and creates a prompt structure to target aligned models. We study the recipe behavior in bypassing alignment in a number of LLMs including GPT-4o, llama-3-8b-instruct, and gemma-2-9b-it where all models are forced into a non-halting state. Further, we demonstrate the recipe's success in sending most major models released over the past year into a non-halting state with the same simple prompt even at higher temperatures. Further, we study direct inversion based techniques to craft new short prompts to induce the non-halting state. Our experiments with the gradient search based inversion technique ARCA show that non-halting is prevalent across models and may be easily induced with a few input tokens. While its impact on the reliability of hosted systems can be mitigated by configuring a hard maximum token limit in the sampler, the non-halting anomaly still manages to break alignment. This underlines the need for further studies and stronger forms of alignment against non-halting anomalies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.06270v1">MC-MoE: Mixture Compressor for Mixture-of-Experts LLMs Gains More</a></div>
    <div class="paper-meta">
      📅 2024-10-08
      | 💬 18 pages
    </div>
    <details class="paper-abstract">
      Mixture-of-Experts large language models (MoE-LLMs) marks a significant step forward of language models, however, they encounter two critical challenges in practice: 1) expert parameters lead to considerable memory consumption and loading latency; and 2) the current activated experts are redundant, as many tokens may only require a single expert. Motivated by these issues, we investigate the MoE-LLMs and make two key observations: a) different experts exhibit varying behaviors on activation reconstruction error, routing scores, and activated frequencies, highlighting their differing importance, and b) not all tokens are equally important -- only a small subset is critical. Building on these insights, we propose MC-MoE, a training-free Mixture-Compressor for MoE-LLMs, which leverages the significance of both experts and tokens to achieve an extreme compression. First, to mitigate storage and loading overheads, we introduce Pre-Loading Mixed-Precision Quantization, which formulates the adaptive bit-width allocation as a Linear Programming problem, where the objective function balances multi-factors reflecting the importance of each expert. Additionally, we develop Online Dynamic Pruning, which identifies important tokens to retain and dynamically select activated experts for other tokens during inference to optimize efficiency while maintaining performance. Our MC-MoE integrates static quantization and dynamic pruning to collaboratively achieve extreme compression for MoE-LLMs with less accuracy loss, ensuring an optimal trade-off between performance and efficiency. Extensive experiments confirm the effectiveness of our approach. For instance, at 2.54 bits, MC-MoE compresses 76.6% of the model, with only a 3.8% average accuracy loss. During dynamic inference, we further reduce activated parameters by 15%, with a performance drop of less than 0.6%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.06238v1">EVOLvE: Evaluating and Optimizing LLMs For Exploration</a></div>
    <div class="paper-meta">
      📅 2024-10-08
      | 💬 28 pages
    </div>
    <details class="paper-abstract">
      Despite their success in many domains, large language models (LLMs) remain under-studied in scenarios requiring optimal decision-making under uncertainty. This is crucial as many real-world applications, ranging from personalized recommendations to healthcare interventions, demand that LLMs not only predict but also actively learn to make optimal decisions through exploration. In this work, we measure LLMs' (in)ability to make optimal decisions in bandits, a state-less reinforcement learning setting relevant to many applications. We develop a comprehensive suite of environments, including both context-free and contextual bandits with varying task difficulties, to benchmark LLMs' performance. Motivated by the existence of optimal exploration algorithms, we propose efficient ways to integrate this algorithmic knowledge into LLMs: by providing explicit algorithm-guided support during inference; and through algorithm distillation via in-context demonstrations and fine-tuning, using synthetic data generated from these algorithms. Impressively, these techniques allow us to achieve superior exploration performance with smaller models, surpassing larger models on various tasks. We conducted an extensive ablation study to shed light on various factors, such as task difficulty and data representation, that influence the efficiency of LLM exploration. Additionally, we conduct a rigorous analysis of the LLM's exploration efficiency using the concept of regret, linking its ability to explore to the model size and underlying algorithm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.15012v2">Extracting Prompts by Inverting LLM Outputs</a></div>
    <div class="paper-meta">
      📅 2024-10-08
    </div>
    <details class="paper-abstract">
      We consider the problem of language model inversion: given outputs of a language model, we seek to extract the prompt that generated these outputs. We develop a new black-box method, output2prompt, that learns to extract prompts without access to the model's logits and without adversarial or jailbreaking queries. In contrast to previous work, output2prompt only needs outputs of normal user queries. To improve memory efficiency, output2prompt employs a new sparse encoding techique. We measure the efficacy of output2prompt on a variety of user and system prompts and demonstrate zero-shot transferability across different LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.09887v3">OptiBench Meets ReSocratic: Measure and Improve LLMs for Optimization Modeling</a></div>
    <div class="paper-meta">
      📅 2024-10-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have exhibited their problem-solving abilities in mathematical reasoning. Solving realistic optimization (OPT) problems in application scenarios requires advanced and applied mathematics ability. However, current OPT benchmarks that merely solve linear programming are far from complex realistic situations. In this work, we propose OptiBench, a benchmark for End-to-end optimization problem-solving with human-readable inputs and outputs. OptiBench contains rich optimization problems, including linear and nonlinear programming with or without tabular data, which can comprehensively evaluate LLMs' solving ability. In our benchmark, LLMs are required to call a code solver to provide precise numerical answers. Furthermore, to alleviate the data scarcity for optimization problems, and to bridge the gap between open-source LLMs on a small scale (e.g., Llama-3-8b) and closed-source LLMs (e.g., GPT-4), we further propose a data synthesis method namely ReSocratic. Unlike general data synthesis methods that proceed from questions to answers, \ReSocratic first incrementally synthesizes formatted optimization demonstration with mathematical formulations step by step and then back-translates the generated demonstrations into questions. Based on this, we synthesize the ReSocratic-29k dataset. We further conduct supervised fine-tuning with ReSocratic-29k on multiple open-source models. Experimental results show that ReSocratic-29k significantly improves the performance of open-source models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18104v1">ENWAR: A RAG-empowered Multi-Modal LLM Framework for Wireless Environment Perception</a></div>
    <div class="paper-meta">
      📅 2024-10-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) hold significant promise in advancing network management and orchestration in 6G and beyond networks. However, existing LLMs are limited in domain-specific knowledge and their ability to handle multi-modal sensory data, which is critical for real-time situational awareness in dynamic wireless environments. This paper addresses this gap by introducing ENWAR, an ENvironment-aWARe retrieval augmented generation-empowered multi-modal LLM framework. ENWAR seamlessly integrates multi-modal sensory inputs to perceive, interpret, and cognitively process complex wireless environments to provide human-interpretable situational awareness. ENWAR is evaluated on the GPS, LiDAR, and camera modality combinations of DeepSense6G dataset with state-of-the-art LLMs such as Mistral-7b/8x7b and LLaMa3.1-8/70/405b. Compared to general and often superficial environmental descriptions of these vanilla LLMs, ENWAR delivers richer spatial analysis, accurately identifies positions, analyzes obstacles, and assesses line-of-sight between vehicles. Results show that ENWAR achieves key performance indicators of up to 70% relevancy, 55% context recall, 80% correctness, and 86% faithfulness, demonstrating its efficacy in multi-modal perception and interpretation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.06108v1">ConceptAgent: LLM-Driven Precondition Grounding and Tree Search for Robust Task Planning and Execution</a></div>
    <div class="paper-meta">
      📅 2024-10-08
    </div>
    <details class="paper-abstract">
      Robotic planning and execution in open-world environments is a complex problem due to the vast state spaces and high variability of task embodiment. Recent advances in perception algorithms, combined with Large Language Models (LLMs) for planning, offer promising solutions to these challenges, as the common sense reasoning capabilities of LLMs provide a strong heuristic for efficiently searching the action space. However, prior work fails to address the possibility of hallucinations from LLMs, which results in failures to execute the planned actions largely due to logical fallacies at high- or low-levels. To contend with automation failure due to such hallucinations, we introduce ConceptAgent, a natural language-driven robotic platform designed for task execution in unstructured environments. With a focus on scalability and reliability of LLM-based planning in complex state and action spaces, we present innovations designed to limit these shortcomings, including 1) Predicate Grounding to prevent and recover from infeasible actions, and 2) an embodied version of LLM-guided Monte Carlo Tree Search with self reflection. In simulation experiments, ConceptAgent achieved a 19% task completion rate across three room layouts and 30 easy level embodied tasks outperforming other state-of-the-art LLM-driven reasoning baselines that scored 10.26% and 8.11% on the same benchmark. Additionally, ablation studies on moderate to hard embodied tasks revealed a 20% increase in task completion from the baseline agent to the fully enhanced ConceptAgent, highlighting the individual and combined contributions of Predicate Grounding and LLM-guided Tree Search to enable more robust automation in complex state and action spaces.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.06101v1">Coevolving with the Other You: Fine-Tuning LLM with Sequential Cooperative Multi-Agent Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2024-10-08
      | 💬 28 pages, 26 images
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has emerged as a pivotal technique for fine-tuning large language models (LLMs) on specific tasks. However, prevailing RL fine-tuning methods predominantly rely on PPO and its variants. Though these algorithms are effective in general RL settings, they often exhibit suboptimal performance and vulnerability to distribution collapse when applied to the fine-tuning of LLMs. In this paper, we propose CORY, extending the RL fine-tuning of LLMs to a sequential cooperative multi-agent reinforcement learning framework, to leverage the inherent coevolution and emergent capabilities of multi-agent systems. In CORY, the LLM to be fine-tuned is initially duplicated into two autonomous agents: a pioneer and an observer. The pioneer generates responses based on queries, while the observer generates responses using both the queries and the pioneer's responses. The two agents are trained together. During training, the agents exchange roles periodically, fostering cooperation and coevolution between them. Experiments evaluate CORY's performance by fine-tuning GPT-2 and Llama-2 under subjective and objective reward functions on the IMDB Review and GSM8K datasets, respectively. Results show that CORY outperforms PPO in terms of policy optimality, resistance to distribution collapse, and training robustness, thereby underscoring its potential as a superior methodology for refining LLMs in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.05905v4">Truthful Aggregation of LLMs with an Application to Online Advertising</a></div>
    <div class="paper-meta">
      📅 2024-10-08
    </div>
    <details class="paper-abstract">
      The next frontier of online advertising is revenue generation from LLM-generated content. We consider a setting where advertisers aim to influence the responses of an LLM to align with their interests, while platforms seek to maximize advertiser value and ensure user satisfaction. The challenge is that advertisers' preferences generally conflict with those of the user, and advertisers may misreport their preferences. To address this, we introduce MOSAIC, an auction mechanism that ensures that truthful reporting is a dominant strategy for advertisers and that aligns the utility of each advertiser with their contribution to social welfare. Importantly, the mechanism operates without LLM fine-tuning or access to model weights and provably converges to the output of the optimally fine-tuned LLM as computational resources increase. Additionally, it can incorporate contextual information about advertisers, which significantly improves social welfare. Through experiments with a publicly available LLM, we show that MOSAIC leads to high advertiser value and platform revenue with low computational overhead. While our motivating application is online advertising, our mechanism can be applied in any setting with monetary transfers, making it a general-purpose solution for truthfully aggregating the preferences of self-interested agents over LLM-generated replies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.06072v1">Training-free LLM-generated Text Detection by Mining Token Probability Sequences</a></div>
    <div class="paper-meta">
      📅 2024-10-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable capabilities in generating high-quality texts across diverse domains. However, the potential misuse of LLMs has raised significant concerns, underscoring the urgent need for reliable detection of LLM-generated texts. Conventional training-based detectors often struggle with generalization, particularly in cross-domain and cross-model scenarios. In contrast, training-free methods, which focus on inherent discrepancies through carefully designed statistical features, offer improved generalization and interpretability. Despite this, existing training-free detection methods typically rely on global text sequence statistics, neglecting the modeling of local discriminative features, thereby limiting their detection efficacy. In this work, we introduce a novel training-free detector, termed \textbf{Lastde} that synergizes local and global statistics for enhanced detection. For the first time, we introduce time series analysis to LLM-generated text detection, capturing the temporal dynamics of token probability sequences. By integrating these local statistics with global ones, our detector reveals significant disparities between human and LLM-generated texts. We also propose an efficient alternative, \textbf{Lastde++} to enable real-time detection. Extensive experiments on six datasets involving cross-domain, cross-model, and cross-lingual detection scenarios, under both white-box and black-box settings, demonstrated that our method consistently achieves state-of-the-art performance. Furthermore, our approach exhibits greater robustness against paraphrasing attacks compared to existing baseline methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.03671v2">TRACE-CS: A Synergistic Approach to Explainable Course Scheduling Using LLMs and Logic</a></div>
    <div class="paper-meta">
      📅 2024-10-08
    </div>
    <details class="paper-abstract">
      We present TRACE-cs, a novel hybrid system that combines symbolic reasoning with large language models (LLMs) to address contrastive queries in scheduling problems. TRACE-cs leverages SAT solving techniques to encode scheduling constraints and generate explanations for user queries, while utilizing an LLM to process the user queries into logical clauses as well as refine the explanations generated by the symbolic solver to natural language sentences. By integrating these components, our approach demonstrates the potential of combining symbolic methods with LLMs to create explainable AI agents with correctness guarantees.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.15153v2">Performance Characterization of Expert Router for Scalable LLM Inference</a></div>
    <div class="paper-meta">
      📅 2024-10-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have experienced widespread adoption across scientific and industrial domains due to their versatility and utility for diverse tasks. Nevertheless, deploying and serving these models at scale with optimal throughput and latency remains a significant challenge, primarily because of LLMs' high computational and memory demands. Specialized models optimized for specific tasks can be combined through a routing mechanism to address these challenges, creating a modular inference system. This paper introduces Expert Router, a scalable routing architecture that directs prompts to specialized expert models. We characterize multiple Expert Router configurations, including different LLama 3 models with quantized and non-quantized weights under up to 1,000 concurrent users. Our findings reveal that Expert Router introduces minimal latency overhead, with the configuration of expert models being a dominating factor in performance outcomes. High-parameter expert models deliver stable throughput and latency under moderate concurrency levels. In contrast, smaller expert models maintain competitive performance across a wider range of concurrent users compared to tensor-parallelized baseline models. This highlights the potential of Expert Router for efficient and scalable LLM deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05983v1">Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG</a></div>
    <div class="paper-meta">
      📅 2024-10-08
      | 💬 34 pages
    </div>
    <details class="paper-abstract">
      Retrieval-augmented generation (RAG) empowers large language models (LLMs) to utilize external knowledge sources. The increasing capacity of LLMs to process longer input sequences opens up avenues for providing more retrieved information, to potentially enhance the quality of generated outputs. It is plausible to assume that a larger retrieval set would contain more relevant information (higher recall), that might result in improved performance. However, our empirical findings demonstrate that for many long-context LLMs, the quality of generated output initially improves first, but then subsequently declines as the number of retrieved passages increases. This paper investigates this phenomenon, identifying the detrimental impact of retrieved "hard negatives" as a key contributor. To mitigate this and enhance the robustness of long-context LLM-based RAG, we propose both training-free and training-based approaches. We first showcase the effectiveness of retrieval reordering as a simple yet powerful training-free optimization. Furthermore, we explore training-based methods, specifically RAG-specific implicit LLM fine-tuning and RAG-oriented fine-tuning with intermediate reasoning, demonstrating their capacity for substantial performance gains. Finally, we conduct a systematic analysis of design choices for these training-based methods, including data distribution, retriever selection, and training context length.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05952v1">Active Evaluation Acquisition for Efficient LLM Benchmarking</a></div>
    <div class="paper-meta">
      📅 2024-10-08
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become increasingly versatile, numerous large scale benchmarks have been developed to thoroughly assess their capabilities. These benchmarks typically consist of diverse datasets and prompts to evaluate different aspects of LLM performance. However, comprehensive evaluations on hundreds or thousands of prompts incur tremendous costs in terms of computation, money, and time. In this work, we investigate strategies to improve evaluation efficiency by selecting a subset of examples from each benchmark using a learned policy. Our approach models the dependencies across test examples, allowing accurate prediction of the evaluation outcomes for the remaining examples based on the outcomes of the selected ones. Consequently, we only need to acquire the actual evaluation outcomes for the selected subset. We rigorously explore various subset selection policies and introduce a novel RL-based policy that leverages the captured dependencies. Empirical results demonstrate that our approach significantly reduces the number of evaluation prompts required while maintaining accurate performance estimates compared to previous methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.16022v2">AI Can Be Cognitively Biased: An Exploratory Study on Threshold Priming in LLM-Based Batch Relevance Assessment</a></div>
    <div class="paper-meta">
      📅 2024-10-08
    </div>
    <details class="paper-abstract">
      Cognitive biases are systematic deviations in thinking that lead to irrational judgments and problematic decision-making, extensively studied across various fields. Recently, large language models (LLMs) have shown advanced understanding capabilities but may inherit human biases from their training data. While social biases in LLMs have been well-studied, cognitive biases have received less attention, with existing research focusing on specific scenarios. The broader impact of cognitive biases on LLMs in various decision-making contexts remains underexplored. We investigated whether LLMs are influenced by the threshold priming effect in relevance judgments, a core task and widely-discussed research topic in the Information Retrieval (IR) coummunity. The priming effect occurs when exposure to certain stimuli unconsciously affects subsequent behavior and decisions. Our experiment employed 10 topics from the TREC 2019 Deep Learning passage track collection, and tested AI judgments under different document relevance scores, batch lengths, and LLM models, including GPT-3.5, GPT-4, LLaMa2-13B and LLaMa2-70B. Results showed that LLMs tend to give lower scores to later documents if earlier ones have high relevance, and vice versa, regardless of the combination and model used. Our finding demonstrates that LLM%u2019s judgments, similar to human judgments, are also influenced by threshold priming biases, and suggests that researchers and system engineers should take into account potential human-like cognitive biases in designing, evaluating, and auditing LLMs in IR tasks and beyond.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05873v1">MEXA: Multilingual Evaluation of English-Centric LLMs via Cross-Lingual Alignment</a></div>
    <div class="paper-meta">
      📅 2024-10-08
    </div>
    <details class="paper-abstract">
      English-centric large language models (LLMs) often show strong multilingual capabilities. However, the multilingual performance of these models remains unclear and is not thoroughly evaluated for many languages. Most benchmarks for multilinguality focus on classic NLP tasks, or cover a minimal number of languages. We introduce MEXA, a method for assessing the multilingual capabilities of pre-trained English-centric LLMs using parallel sentences, which are available for more languages than existing downstream tasks. MEXA leverages the fact that English-centric LLMs use English as a kind of pivot language in their intermediate layers. It computes the alignment between English and non-English languages using parallel sentences to evaluate the transfer of language understanding from English to other languages. This alignment can be used to estimate model performance in other languages. We conduct studies using various parallel datasets (FLORES-200 and Bible), models (Llama family, Gemma family, Mistral, and OLMo), and established downstream tasks (Belebele, m-MMLU, and m-ARC). We explore different methods to compute embeddings in decoder-only models. Our results show that MEXA, in its default settings, achieves a statistically significant average Pearson correlation of 0.90 with three established downstream tasks across nine models and two parallel datasets. This suggests that MEXA is a reliable method for estimating the multilingual capabilities of English-centric LLMs, providing a clearer understanding of their multilingual potential and the inner workings of LLMs. Leaderboard: https://huggingface.co/spaces/cis-lmu/Mexa, Code: https://github.com/cisnlp/Mexa.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.19806v1">Vitron: A Unified Pixel-level Vision LLM for Understanding, Generating, Segmenting, Editing</a></div>
    <div class="paper-meta">
      📅 2024-10-08
      | 💬 Accepted by NeurIPS 2024
    </div>
    <details class="paper-abstract">
      Recent developments of vision large language models (LLMs) have seen remarkable progress, yet still encounter challenges towards multimodal generalists, such as coarse-grained instance-level understanding, lack of unified support for both images and videos, and insufficient coverage across various vision tasks. In this paper, we present VITRON, a universal pixel-level vision LLM designed for comprehensive understanding, generating, segmenting, and editing of both static images and dynamic videos. Building on top of an LLM backbone, VITRON incorporates encoders for images, videos, and pixel-level regional visuals within its frontend modules, while employing state-of-the-art visual specialists as its backend, via which VITRON supports a spectrum of vision end tasks, spanning visual comprehension to visual generation, from low level to high level. To ensure an effective and precise message passing from LLM to backend modules for function invocation, we propose a novel hybrid method by simultaneously integrating discrete textual instructions and continuous signal embeddings. Further, we design various pixel-level spatiotemporal vision-language alignment learning for VITRON to reach the best fine-grained visual capability. Finally, a cross-task synergy module is advised to learn to maximize the task-invariant fine-grained visual features, enhancing the synergy between different visual tasks. Demonstrated over 12 visual tasks and evaluated across 22 datasets, VITRON showcases its extensive capabilities in the four main vision task clusters. Overall, this work illuminates the great potential of developing a more unified multimodal generalist. Project homepage: https://vitron-llm.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.11958v2">Understanding the Therapeutic Relationship between Counselors and Clients in Online Text-based Counseling using LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-08
      | 💬 Accepted to EMNLP Findings 2024. 24 pages, 6 figures, 9 tables
    </div>
    <details class="paper-abstract">
      Robust therapeutic relationships between counselors and clients are fundamental to counseling effectiveness. The assessment of therapeutic alliance is well-established in traditional face-to-face therapy but may not directly translate to text-based settings. With millions of individuals seeking support through online text-based counseling, understanding the relationship in such contexts is crucial. In this paper, we present an automatic approach using large language models (LLMs) to understand the development of therapeutic alliance in text-based counseling. We adapt a theoretically grounded framework specifically to the context of online text-based counseling and develop comprehensive guidelines for characterizing the alliance. We collect a comprehensive counseling dataset and conduct multiple expert evaluations on a subset based on this framework. Our LLM-based approach, combined with guidelines and simultaneous extraction of supportive evidence underlying its predictions, demonstrates effectiveness in identifying the therapeutic alliance. Through further LLM-based evaluations on additional conversations, our findings underscore the challenges counselors face in cultivating strong online relationships with clients. Furthermore, we demonstrate the potential of LLM-based feedback mechanisms to enhance counselors' ability to build relationships, supported by a small-scale proof-of-concept.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.13356v3">Jogging the Memory of Unlearned LLMs Through Targeted Relearning Attacks</a></div>
    <div class="paper-meta">
      📅 2024-10-08
      | 💬 26 pages, 5 figures, 7 tables
    </div>
    <details class="paper-abstract">
      Machine unlearning is a promising approach to mitigate undesirable memorization of training data in LLMs. However, in this work we show that existing approaches for unlearning in LLMs are surprisingly susceptible to a simple set of targeted relearning attacks. With access to only a small and potentially loosely related set of data, we find that we can "jog" the memory of unlearned models to reverse the effects of unlearning. For example, we show that relearning on public medical articles can lead an unlearned LLM to output harmful knowledge about bioweapons, and relearning general wiki information about the book series Harry Potter can force the model to output verbatim memorized text. We formalize this unlearning-relearning pipeline, explore the attack across three popular unlearning benchmarks, and discuss future directions and guidelines that result from our study.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05797v1">CodeCipher: Learning to Obfuscate Source Code Against LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-08
    </div>
    <details class="paper-abstract">
      While large code language models have made significant strides in AI-assisted coding tasks, there are growing concerns about privacy challenges. The user code is transparent to the cloud LLM service provider, inducing risks of unauthorized training, reading, and execution of the user code. In this paper, we propose CodeCipher, a novel method that perturbs privacy from code while preserving the original response from LLMs. CodeCipher transforms the LLM's embedding matrix so that each row corresponds to a different word in the original matrix, forming a token-to-token confusion mapping for obfuscating source code. The new embedding matrix is optimized by minimizing the task-specific loss function. To tackle the challenge of the discrete and sparse nature of word vector spaces, CodeCipher adopts a discrete optimization strategy that aligns the updated vector to the nearest valid token in the vocabulary before each gradient update. We demonstrate the effectiveness of our approach on three AI-assisted coding tasks including code completion, summarization, and translation. Results show that our model successfully confuses the privacy in source code while preserving the original LLM's performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.06925v3">A Thorough Examination of Decoding Methods in the Era of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-08
      | 💬 EMNLP 2024 Main
    </div>
    <details class="paper-abstract">
      Decoding methods play an indispensable role in converting language models from next-token predictors into practical task solvers. Prior research on decoding methods, primarily focusing on task-specific models, may not extend to the current era of general-purpose large language models (LLMs). Moreover, the recent influx of decoding strategies has further complicated this landscape. This paper provides a comprehensive and multifaceted analysis of various decoding methods within the context of LLMs, evaluating their performance, robustness to hyperparameter changes, and decoding speeds across a wide range of tasks, models, and deployment environments. Our findings reveal that decoding method performance is notably task-dependent and influenced by factors such as alignment, model size, and quantization. Intriguingly, sensitivity analysis exposes that certain methods achieve superior performance at the cost of extensive hyperparameter tuning, highlighting the trade-off between attaining optimal results and the practicality of implementation in varying contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05714v1">Enhancing Temporal Modeling of Video LLMs via Time Gating</a></div>
    <div class="paper-meta">
      📅 2024-10-08
      | 💬 EMNLP 2024 Findings (Short)
    </div>
    <details class="paper-abstract">
      Video Large Language Models (Video LLMs) have achieved impressive performance on video-and-language tasks, such as video question answering. However, most existing Video LLMs neglect temporal information in video data, leading to struggles with temporal-aware video understanding. To address this gap, we propose a Time Gating Video LLM (TG-Vid) designed to enhance temporal modeling through a novel Time Gating module (TG). The TG module employs a time gating mechanism on its sub-modules, comprising gating spatial attention, gating temporal attention, and gating MLP. This architecture enables our model to achieve a robust understanding of temporal information within videos. Extensive evaluation of temporal-sensitive video benchmarks (i.e., MVBench, TempCompass, and NExT-QA) demonstrates that our TG-Vid model significantly outperforms the existing Video LLMs. Further, comprehensive ablation studies validate that the performance gains are attributed to the designs of our TG module. Our code is available at https://github.com/LaVi-Lab/TG-Vid.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.18695v2">Learning to Correct for QA Reasoning with Black-box LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-08
      | 💬 22 pages; EMNLP 2024 (long, main)
    </div>
    <details class="paper-abstract">
      An open challenge in recent machine learning is about how to improve the reasoning capability of large language models (LLMs) in a black-box setting, i.e., without access to detailed information such as output token probabilities. Existing approaches either rely on accessibility (which is often unrealistic) or involve significantly increased train- and inference-time costs. This paper addresses those limitations or shortcomings by proposing a novel approach, namely CoBB (Correct for improving QA reasoning of Black-Box LLMs). It uses a trained adaptation model to perform a seq2seq mapping from the often-imperfect reasonings of the original black-box LLM to the correct or improved reasonings. Specifically, the adaptation model is initialized with a relatively small open-source LLM and adapted over a collection of sub-sampled training pairs. To select the representative pairs of correct and incorrect reasonings, we formulated the dataset construction as an optimization problem that minimizes the statistical divergence between the sampled subset and the entire collection, and solved it via a genetic algorithm. We then train the adaptation model over the sampled pairs by contrasting the likelihoods of correct and incorrect reasonings. Our experimental results demonstrate that CoBB significantly improves reasoning accuracy across various QA benchmarks, compared to the best-performing adaptation baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.14979v2">Retrieve-Plan-Generation: An Iterative Planning and Answering Framework for Knowledge-Intensive LLM Generation</a></div>
    <div class="paper-meta">
      📅 2024-10-08
    </div>
    <details class="paper-abstract">
      Despite the significant progress of large language models (LLMs) in various tasks, they often produce factual errors due to their limited internal knowledge. Retrieval-Augmented Generation (RAG), which enhances LLMs with external knowledge sources, offers a promising solution. However, these methods can be misled by irrelevant paragraphs in retrieved documents. Due to the inherent uncertainty in LLM generation, inputting the entire document may introduce off-topic information, causing the model to deviate from the central topic and affecting the relevance of the generated content. To address these issues, we propose the Retrieve-Plan-Generation (RPG) framework. RPG generates plan tokens to guide subsequent generation in the plan stage. In the answer stage, the model selects relevant fine-grained paragraphs based on the plan and uses them for further answer generation. This plan-answer process is repeated iteratively until completion, enhancing generation relevance by focusing on specific topics. To implement this framework efficiently, we utilize a simple but effective multi-task prompt-tuning method, enabling the existing LLMs to handle both planning and answering. We comprehensively compare RPG with baselines across 5 knowledge-intensive generation tasks, demonstrating the effectiveness of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.01803v2">STBLLM: Breaking the 1-Bit Barrier with Structured Binary LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-08
    </div>
    <details class="paper-abstract">
      In this paper, we present the first structural binarization method for LLM compression to less than 1-bit precision. Although LLMs have achieved remarkable performance, their memory-bound nature during the inference stage hinders the adoption of resource-constrained devices. Reducing weights to 1-bit precision through binarization substantially enhances computational efficiency. We observe that some weights in binarized LLMs can be randomly flipped without significant performance degradation, suggesting the potential for further compression. To exploit this, our STBLLM employs an N:M sparsity technique to achieve structural binarization of the weights. Specifically, we introduce a novel Standardized Importance (SI) metric, which considers weight magnitude and input feature norm to more accurately assess weight significance. Then, we propose a layer-wise approach, allowing different layers of the LLM to be sparsified with varying N:M ratios, thereby balancing compression and accuracy. Furthermore, we implement a fine-grained grouping strategy for less important weights, applying distinct quantization schemes to sparse, intermediate, and dense regions. Finally, we design a specialized CUDA kernel to support structural binarization. We conduct extensive experiments on LLaMA-1/2/3, OPT family, and Mistral to evaluate the effectiveness of STBLLM. The results demonstrate that our approach performs better than other compressed binarization LLM methods while significantly reducing memory requirements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.05516v5">Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-08
      | 💬 EMNLP24 Findings
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated exceptional proficiency in language-related tasks, but their deployment poses significant challenges due to substantial memory and storage requirements. Weight-only quantization has emerged as a promising solution, significantly reducing memory and storage needs without sacrificing too much performance. In this study, we introduce SignRound, a method that leverages signed gradient descent (SignSGD) to optimize rounding values and weight clipping in just 200 steps. SignRound integrates the advantages of Quantization-Aware Training (QAT) and Post-Training Quantization (PTQ), delivering exceptional results across 2 to 4 bits while minimizing tuning costs and avoiding additional inference overhead. For example, SignRound achieved absolute average accuracy improvements ranging from 6.91% to 33.22% at 2bits, as measured by the average zero-shot accuracy across 11 tasks. It also demonstrates strong generalization in recent models, achieving near-lossless 4-bit quantization in most scenarios. The source code is publicly available at https://github.com/intel/auto-round.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05603v1">Everything Everywhere All at Once: LLMs can In-Context Learn Multiple Tasks in Superposition</a></div>
    <div class="paper-meta">
      📅 2024-10-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable in-context learning (ICL) capabilities. In this study, we explore a surprising phenomenon related to ICL: LLMs can perform multiple, computationally distinct ICL tasks simultaneously, during a single inference call, a capability we term "task superposition". We provide empirical evidence of this phenomenon across various LLM families and scales and show that this phenomenon emerges even if we train the model to in-context learn one task at a time. We offer theoretical explanations that this capability is well within the expressive power of transformers. We also explore how LLMs internally compose task vectors during superposition. Furthermore, we show that larger models can solve more ICL tasks in parallel, and better calibrate their output distribution. Our findings offer insights into the latent capabilities of LLMs, further substantiate the perspective of "LLMs as superposition of simulators", and raise questions about the mechanisms enabling simultaneous task execution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05495v1">Self-rationalization improves LLM as a fine-grained judge</a></div>
    <div class="paper-meta">
      📅 2024-10-07
    </div>
    <details class="paper-abstract">
      LLM-as-a-judge models have been used for evaluating both human and AI generated content, specifically by providing scores and rationales. Rationales, in addition to increasing transparency, help models learn to calibrate its judgments. Enhancing a model's rationale can therefore improve its calibration abilities and ultimately the ability to score content. We introduce Self-Rationalization, an iterative process of improving the rationales for the judge models, which consequently improves the score for fine-grained customizable scoring criteria (i.e., likert-scale scoring with arbitrary evaluation criteria). Self-rationalization works by having the model generate multiple judgments with rationales for the same input, curating a preference pair dataset from its own judgements, and iteratively fine-tuning the judge via DPO. Intuitively, this approach allows the judge model to self-improve by learning from its own rationales, leading to better alignment and evaluation accuracy. After just two iterations -- while only relying on examples in the training set -- human evaluation shows that our judge model learns to produce higher quality rationales, with a win rate of $62\%$ on average compared to models just trained via SFT on rationale . This judge model also achieves high scoring accuracy on BigGen Bench and Reward Bench, outperforming even bigger sized models trained using SFT with rationale, self-consistency or best-of-$N$ sampling by $3\%$ to $9\%$.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10863v1">What makes your model a low-empathy or warmth person: Exploring the Origins of Personality in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-07
      | 💬 under review
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable capabilities in generating human-like text and exhibiting personality traits similar to those in humans. However, the mechanisms by which LLMs encode and express traits such as agreeableness and impulsiveness remain poorly understood. Drawing on the theory of social determinism, we investigate how long-term background factors, such as family environment and cultural norms, interact with short-term pressures like external instructions, shaping and influencing LLMs' personality traits. By steering the output of LLMs through the utilization of interpretable features within the model, we explore how these background and pressure factors lead to changes in the model's traits without the need for further fine-tuning. Additionally, we suggest the potential impact of these factors on model safety from the perspective of personality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18085v1">TextureMeDefect: LLM-based Defect Texture Generation for Railway Components on Mobile Devices</a></div>
    <div class="paper-meta">
      📅 2024-10-07
      | 💬 6 Pages, 8 figures
    </div>
    <details class="paper-abstract">
      Texture image generation has been studied for various applications, including gaming and entertainment. However, context-specific realistic texture generation for industrial applications, such as generating defect textures on railway components, remains unexplored. A mobile-friendly, LLM-based tool that generates fine-grained defect characteristics offers a solution to the challenge of understanding the impact of defects from actual occurrences. We introduce TextureMeDefect, an innovative tool leveraging an LLM-based AI-Inferencing engine. The tool allows users to create realistic defect textures interactively on images of railway components taken with smartphones or tablets. We conducted a multifaceted evaluation to assess the relevance of the generated texture, time, and cost in using this tool on iOS and Android platforms. We also analyzed the software usability score (SUS) across three scenarios. TextureMeDefect outperformed traditional image generation tools by generating meaningful textures faster, showcasing the potential of AI-driven mobile applications on consumer-grade devices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05434v1">Better than Your Teacher: LLM Agents that learn from Privileged AI Feedback</a></div>
    <div class="paper-meta">
      📅 2024-10-07
      | 💬 34 pages, 6 figures, 5 tables
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) show impressive decision-making abilities, current methods lack a mechanism for automatic self-improvement from errors during task execution. We propose LEAP, an iterative fine-tuning framework that continually improves LLM agents using feedback from AI expert teachers. Our key insight is to equip the expert teachers with a privileged state -- information that is available during training but hidden at test time. This allows even weak experts to provide precise guidance, significantly improving the student agent's performance without access to privileged information at test time. We evaluate LEAP on diverse decision-making benchmarks, including text-based games (ALFWorld), web navigation (WebShop), and interactive coding (Intercode Bash). Our experiments show that LEAP (1) outperforms behavior cloning and ReAct baselines (2) enables weak student models (e.g., Llama3-8B) to exceed the performance of strong teacher models (GPT4-o), and (3) allows weak models to self-improve using privileged versions of themselves. We also provide a theoretical analysis showing that LEAP's success hinges on balancing privileged information with the student's realizability, which we empirically validate. Our code is available at https://leap-llm.github.io
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05401v1">Post-hoc Study of Climate Microtargeting on Social Media Ads with LLMs: Thematic Insights and Fairness Evaluation</a></div>
    <div class="paper-meta">
      📅 2024-10-07
    </div>
    <details class="paper-abstract">
      Climate change communication on social media increasingly employs microtargeting strategies to effectively reach and influence specific demographic groups. This study presents a post-hoc analysis of microtargeting practices within climate campaigns by leveraging large language models (LLMs) to examine Facebook advertisements. Our analysis focuses on two key aspects: demographic targeting and fairness. We evaluate the ability of LLMs to accurately predict the intended demographic targets, such as gender and age group, achieving an overall accuracy of 88.55%. Furthermore, we instruct the LLMs to generate explanations for their classifications, providing transparent reasoning behind each decision. These explanations reveal the specific thematic elements used to engage different demographic segments, highlighting distinct strategies tailored to various audiences. Our findings show that young adults are primarily targeted through messages emphasizing activism and environmental consciousness, while women are engaged through themes related to caregiving roles and social advocacy. In addition to evaluating the effectiveness of LLMs in detecting microtargeted messaging, we conduct a comprehensive fairness analysis to identify potential biases in model predictions. Our findings indicate that while LLMs perform well overall, certain biases exist, particularly in the classification of senior citizens and male audiences. By showcasing the efficacy of LLMs in dissecting and explaining targeted communication strategies and by highlighting fairness concerns, this study provides a valuable framework for future research aimed at enhancing transparency, accountability, and inclusivity in social media-driven climate campaigns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.17975v2">SoK: Membership Inference Attacks on LLMs are Rushing Nowhere (and How to Fix It)</a></div>
    <div class="paper-meta">
      📅 2024-10-07
    </div>
    <details class="paper-abstract">
      Whether LLMs memorize their training data and what this means, from privacy leakage to detecting copyright violations -- has become a rapidly growing area of research over the last two years. In recent months, more than 10 new methods have been proposed to perform Membership Inference Attacks (MIAs) against LLMs. Contrary to traditional MIAs which rely on fixed -- but randomized -- records or models, these methods are mostly evaluated on datasets collected post-hoc. Sets of members and non-members, used to evaluate the MIA, are constructed using informed guesses after the release of a model. This lack of randomization raises concerns of a distribution shift between members and non-members. In the first part, we review the literature on MIAs against LLMs. While most work focuses on sequence-level MIAs evaluated in post-hoc setups, we show that a range of target models, motivations and units of interest have been considered in the literature. We then quantify distribution shifts present in the 6 datasets used in the literature, ranging from books to papers, using a bag of word classifier. Our analysis reveals that all of them suffer from severe distribution shifts. This challenges the validity of using such setups to measure LLM memorization and may undermine the benchmarking of recently proposed methods. Yet, all hope might not be lost. In the second part, we introduce important considerations to properly evaluate MIAs against LLMs and discuss potential ways forward: randomized test splits, injections of randomized (unique) sequences, randomized finetuning, and post-hoc control methods. While each option comes with its advantages and limitations, we believe they collectively provide solid grounds to guide the development of MIA methods and study LLM memorization. We conclude by proposing comprehensive, easy-to-use benchmarks for sequence- and document-level MIAs against LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05362v1">LLMs Are In-Context Reinforcement Learners</a></div>
    <div class="paper-meta">
      📅 2024-10-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can learn new tasks through in-context supervised learning (i.e., ICL). This work studies if this ability extends to in-context reinforcement learning (ICRL), where models are not given gold labels in context, but only their past predictions and rewards. We show that a naive application of ICRL fails miserably, and identify the root cause as a fundamental deficiency at exploration, which leads to quick model degeneration. We propose an algorithm to address this deficiency by increasing test-time compute, as well as a compute-bound approximation. We use several challenging classification tasks to empirically show that our ICRL algorithms lead to effective learning from rewards alone, and analyze the characteristics of this ability and our methods. Overall, our results reveal remarkable ICRL abilities in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05224v1">Cookbook: A framework for improving LLM generative abilities via programmatic data generating templates</a></div>
    <div class="paper-meta">
      📅 2024-10-07
      | 💬 COLM 2024
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) on instruction datasets is a common way to improve their generative capabilities. However, instruction datasets can be expensive and time-consuming to manually curate, and while LLM-generated data is less labor-intensive, it may violate user privacy agreements or terms of service of LLM providers. Therefore, we seek a way of constructing instruction datasets with samples that are not generated by humans or LLMs but still improve LLM generative capabilities. In this work, we introduce Cookbook, a framework that programmatically generates training data consisting of simple patterns over random tokens, resulting in a scalable, cost-effective approach that avoids legal and privacy issues. First, Cookbook uses a template -- a data generating Python function -- to produce training data that encourages the model to learn an explicit pattern-based rule that corresponds to a desired task. We find that fine-tuning on Cookbook-generated data is able to improve performance on its corresponding task by up to 52.7 accuracy points. Second, since instruction datasets improve performance on multiple downstream tasks simultaneously, Cookbook algorithmically learns how to mix data from various templates to optimize performance on multiple tasks. On the standard multi-task GPT4ALL evaluation suite, Mistral-7B fine-tuned using a Cookbook-generated dataset attains the best accuracy on average compared to other 7B parameter instruction-tuned models and is the best performing model on 3 out of 8 tasks. Finally, we analyze when and why Cookbook improves performance and present a metric that allows us to verify that the improvement is largely explained by the model's generations adhering better to template rules.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.02233v3">Augmenting Black-box LLMs with Medical Textbooks for Biomedical Question Answering (Published in Findings of EMNLP 2024)</a></div>
    <div class="paper-meta">
      📅 2024-10-07
      | 💬 This version has been accepted and published at EMNLP Findings 2024
    </div>
    <details class="paper-abstract">
      Large-scale language models (LLMs) like ChatGPT have demonstrated impressive abilities in generating responses based on human instructions. However, their use in the medical field can be challenging due to their lack of specific, in-depth knowledge. In this study, we present a system called LLMs Augmented with Medical Textbooks (LLM-AMT) designed to enhance the proficiency of LLMs in specialized domains. LLM-AMT integrates authoritative medical textbooks into the LLMs' framework using plug-and-play modules. These modules include a Query Augmenter, a Hybrid Textbook Retriever, and a Knowledge Self-Refiner. Together, they incorporate authoritative medical knowledge. Additionally, an LLM Reader aids in contextual understanding. Our experimental results on three medical QA tasks demonstrate that LLMAMT significantly improves response quality, with accuracy gains ranging from 11.6% to 16.6%. Notably, with GPT-4-Turbo as the base model, LLM-AMT outperforms the specialized Med-PaLM 2 model pre-trained on a massive amount of medical corpus by 2-3%. We found that despite being 100x smaller in size, medical textbooks as a retrieval corpus is proven to be a more effective knowledge database than Wikipedia in the medical domain, boosting performance by 7.8%-13.7%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.06369v4">Annotation alignment: Comparing LLM and human annotations of conversational safety</a></div>
    <div class="paper-meta">
      📅 2024-10-07
      | 💬 EMNLP 2024 (Main). Main text contains 6 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Do LLMs align with human perceptions of safety? We study this question via annotation alignment, the extent to which LLMs and humans agree when annotating the safety of user-chatbot conversations. We leverage the recent DICES dataset (Aroyo et al., 2023), in which 350 conversations are each rated for safety by 112 annotators spanning 10 race-gender groups. GPT-4 achieves a Pearson correlation of $r = 0.59$ with the average annotator rating, \textit{higher} than the median annotator's correlation with the average ($r=0.51$). We show that larger datasets are needed to resolve whether LLMs exhibit disparities in how well they correlate with different demographic groups. Also, there is substantial idiosyncratic variation in correlation within groups, suggesting that race & gender do not fully capture differences in alignment. Finally, we find that GPT-4 cannot predict when one demographic group finds a conversation more unsafe than another.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05361v1">RespLLM: Unifying Audio and Text with Multimodal LLMs for Generalized Respiratory Health Prediction</a></div>
    <div class="paper-meta">
      📅 2024-10-07
    </div>
    <details class="paper-abstract">
      The high incidence and mortality rates associated with respiratory diseases underscores the importance of early screening. Machine learning models can automate clinical consultations and auscultation, offering vital support in this area. However, the data involved, spanning demographics, medical history, symptoms, and respiratory audio, are heterogeneous and complex. Existing approaches are insufficient and lack generalizability, as they typically rely on limited training data, basic fusion techniques, and task-specific models. In this paper, we propose RespLLM, a novel multimodal large language model (LLM) framework that unifies text and audio representations for respiratory health prediction. RespLLM leverages the extensive prior knowledge of pretrained LLMs and enables effective audio-text fusion through cross-modal attentions. Instruction tuning is employed to integrate diverse data from multiple sources, ensuring generalizability and versatility of the model. Experiments on five real-world datasets demonstrate that RespLLM outperforms leading baselines by an average of 4.6% on trained tasks, 7.9% on unseen datasets, and facilitates zero-shot predictions for new tasks. Our work lays the foundation for multimodal models that can perceive, listen to, and understand heterogeneous data, paving the way for scalable respiratory health diagnosis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05193v1">RevisEval: Improving LLM-as-a-Judge via Response-Adapted References</a></div>
    <div class="paper-meta">
      📅 2024-10-07
    </div>
    <details class="paper-abstract">
      With significant efforts in recent studies, LLM-as-a-Judge has become a cost-effective alternative to human evaluation for assessing the text generation quality in a wide range of tasks. However, there still remains a reliability gap between LLM-as-a-Judge and human evaluation. One important reason is the lack of guided oracles in the evaluation process. Motivated by the role of reference pervasively used in classic text evaluation, we introduce RevisEval, a novel text generation evaluation paradigm via the response-adapted references. RevisEval is driven by the key observation that an ideal reference should maintain the necessary relevance to the response to be evaluated. Specifically, RevisEval leverages the text revision capabilities of large language models (LLMs) to adaptively revise the response, then treat the revised text as the reference (response-adapted reference) for the subsequent evaluation. Extensive experiments demonstrate that RevisEval outperforms traditional reference-free and reference-based evaluation paradigms that use LLM-as-a-Judge across NLG tasks and open-ended instruction-following tasks. More importantly, our response-adapted references can further boost the classical text metrics, e.g., BLEU and BERTScore, compared to traditional references and even rival the LLM-as-a-Judge. A detailed analysis is also conducted to confirm RevisEval's effectiveness in bias reduction, the impact of inference cost, and reference relevance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.00099v4">Creative Beam Search: LLM-as-a-Judge For Improving Response Generation</a></div>
    <div class="paper-meta">
      📅 2024-10-07
      | 💬 Presented as a short paper at the 15th International Conference on Computational Creativity (ICCC'24)
    </div>
    <details class="paper-abstract">
      Large language models are revolutionizing several areas, including artificial creativity. However, the process of generation in machines profoundly diverges from that observed in humans. In particular, machine generation is characterized by a lack of intentionality and an underlying creative process. We propose a method called Creative Beam Search that uses Diverse Beam Search and LLM-as-a-Judge to perform response generation and response validation. The results of a qualitative experiment show how our approach can provide better output than standard sampling techniques. We also show that the response validation step is a necessary complement to the response generation step.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.02151v3">Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks</a></div>
    <div class="paper-meta">
      📅 2024-10-07
      | 💬 Updates in the v3: GPT-4o and Claude 3.5 Sonnet results, improved writing. Updates in the v2: more models (Llama3, Phi-3, Nemotron-4-340B), jailbreak artifacts for all attacks are available, evaluation with different judges (Llama-3-70B and Llama Guard 2), more experiments (convergence plots over iterations, ablation on the suffix length for random search), examples of jailbroken generation
    </div>
    <details class="paper-abstract">
      We show that even the most recent safety-aligned LLMs are not robust to simple adaptive jailbreaking attacks. First, we demonstrate how to successfully leverage access to logprobs for jailbreaking: we initially design an adversarial prompt template (sometimes adapted to the target LLM), and then we apply random search on a suffix to maximize a target logprob (e.g., of the token "Sure"), potentially with multiple restarts. In this way, we achieve 100% attack success rate -- according to GPT-4 as a judge -- on Vicuna-13B, Mistral-7B, Phi-3-Mini, Nemotron-4-340B, Llama-2-Chat-7B/13B/70B, Llama-3-Instruct-8B, Gemma-7B, GPT-3.5, GPT-4o, and R2D2 from HarmBench that was adversarially trained against the GCG attack. We also show how to jailbreak all Claude models -- that do not expose logprobs -- via either a transfer or prefilling attack with a 100% success rate. In addition, we show how to use random search on a restricted set of tokens for finding trojan strings in poisoned models -- a task that shares many similarities with jailbreaking -- which is the algorithm that brought us the first place in the SaTML'24 Trojan Detection Competition. The common theme behind these attacks is that adaptivity is crucial: different models are vulnerable to different prompting templates (e.g., R2D2 is very sensitive to in-context learning prompts), some models have unique vulnerabilities based on their APIs (e.g., prefilling for Claude), and in some settings, it is crucial to restrict the token search space based on prior knowledge (e.g., for trojan detection). For reproducibility purposes, we provide the code, logs, and jailbreak artifacts in the JailbreakBench format at https://github.com/tml-epfl/llm-adaptive-attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.03853v5">Dr. Jekyll and Mr. Hyde: Two Faces of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-07
    </div>
    <details class="paper-abstract">
      Recently, we have witnessed a rise in the use of Large Language Models (LLMs), especially in applications like chatbots. Safety mechanisms are implemented to prevent improper responses from these chatbots. In this work, we bypass these measures for ChatGPT and Gemini by making them impersonate complex personas with personality characteristics that are not aligned with a truthful assistant. First, we create elaborate biographies of these personas, which we then use in a new session with the same chatbots. Our conversations then follow a role-play style to elicit prohibited responses. Using personas, we show that prohibited responses are provided, making it possible to obtain unauthorized, illegal, or harmful information in both ChatGPT and Gemini. We also introduce several ways of activating such adversarial personas, showing that both chatbots are vulnerable to this attack. With the same principle, we introduce two defenses that push the model to interpret trustworthy personalities and make it more robust against such attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05130v1">Scalable and Accurate Graph Reasoning with LLM-based Multi-Agents</a></div>
    <div class="paper-meta">
      📅 2024-10-07
    </div>
    <details class="paper-abstract">
      Recent research has explored the use of Large Language Models (LLMs) for tackling complex graph reasoning tasks. However, due to the intricacies of graph structures and the inherent limitations of LLMs in handling long text, current approaches often fail to deliver satisfactory accuracy, even on small-scale graphs and simple tasks. To address these challenges, we introduce GraphAgent-Reasoner, a fine-tuning-free framework that utilizes a multi-agent collaboration strategy for explicit and precise graph reasoning. Inspired by distributed graph computation theory, our framework decomposes graph problems into smaller, node-centric tasks that are distributed among multiple agents. The agents collaborate to solve the overall problem, significantly reducing the amount of information and complexity handled by a single LLM, thus enhancing the accuracy of graph reasoning. By simply increasing the number of agents, GraphAgent-Reasoner can efficiently scale to accommodate larger graphs with over 1,000 nodes. Evaluated on the GraphInstruct dataset, our framework demonstrates near-perfect accuracy on polynomial-time graph reasoning tasks, significantly outperforming the best available models, both closed-source and fine-tuned open-source variants. Our framework also demonstrates the capability to handle real-world graph reasoning applications such as webpage importance analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.15929v2">Decoding Intelligence: A Framework for Certifying Knowledge Comprehension in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-07
    </div>
    <details class="paper-abstract">
      Knowledge comprehension capability is an important aspect of human intelligence. As Large Language Models (LLMs) are being envisioned as superhuman agents, it is crucial for them to be proficient at knowledge comprehension. However, existing benchmarking studies do not provide consistent, generalizable, and formal guarantees on the knowledge comprehension capabilities of LLMs. In this work, we propose the first framework to certify knowledge comprehension in LLMs with formal probabilistic guarantees. Our certificates are quantitative -- they consist of high-confidence, tight bounds on the probability that a target LLM gives the correct answer on any knowledge comprehension prompt sampled from a distribution. We design and certify novel specifications that precisely represent distributions of knowledge comprehension prompts leveraging knowledge graphs. We certify SOTA LLMs for specifications over the Wikidata5m knowledge graph. We find that the knowledge comprehension capability improves significantly with scaling the size of the models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05076v1">TidalDecode: Fast and Accurate LLM Decoding with Position Persistent Sparse Attention</a></div>
    <div class="paper-meta">
      📅 2024-10-07
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have driven significant advancements across diverse NLP tasks, with long-context models gaining prominence for handling extended inputs. However, the expanding key-value (KV) cache size required by Transformer architectures intensifies the memory constraints, particularly during the decoding phase, creating a significant bottleneck. Existing sparse attention mechanisms designed to address this bottleneck have two limitations: (1) they often fail to reliably identify the most relevant tokens for attention, and (2) they overlook the spatial coherence of token selection across consecutive Transformer layers, which can lead to performance degradation and substantial overhead in token selection. This paper introduces TidalDecode, a simple yet effective algorithm and system for fast and accurate LLM decoding through position persistent sparse attention. TidalDecode leverages the spatial coherence of tokens selected by existing sparse attention methods and introduces a few token selection layers that perform full attention to identify the tokens with the highest attention scores, while all other layers perform sparse attention with the pre-selected tokens. This design enables TidalDecode to substantially reduce the overhead of token selection for sparse attention without sacrificing the quality of the generated results. Evaluation on a diverse set of LLMs and tasks shows that TidalDecode closely matches the generative performance of full attention methods while reducing the LLM decoding latency by up to 2.1x.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16309v1">In-the-loop Hyper-Parameter Optimization for LLM-Based Automated Design of Heuristics</a></div>
    <div class="paper-meta">
      📅 2024-10-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown great potential in automatically generating and optimizing (meta)heuristics, making them valuable tools in heuristic optimization tasks. However, LLMs are generally inefficient when it comes to fine-tuning hyper-parameters of the generated algorithms, often requiring excessive queries that lead to high computational and financial costs. This paper presents a novel hybrid approach, LLaMEA-HPO, which integrates the open source LLaMEA (Large Language Model Evolutionary Algorithm) framework with a Hyper-Parameter Optimization (HPO) procedure in the loop. By offloading hyper-parameter tuning to an HPO procedure, the LLaMEA-HPO framework allows the LLM to focus on generating novel algorithmic structures, reducing the number of required LLM queries and improving the overall efficiency of the optimization process. We empirically validate the proposed hybrid framework on benchmark problems, including Online Bin Packing, Black-Box Optimization, and the Traveling Salesperson Problem. Our results demonstrate that LLaMEA-HPO achieves superior or comparable performance compared to existing LLM-driven frameworks while significantly reducing computational costs. This work highlights the importance of separating algorithmic innovation and structural code search from parameter tuning in LLM-driven code optimization and offers a scalable approach to improve the efficiency and effectiveness of LLM-based code generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05047v1">A test suite of prompt injection attacks for LLM-based machine translation</a></div>
    <div class="paper-meta">
      📅 2024-10-07
    </div>
    <details class="paper-abstract">
      LLM-based NLP systems typically work by embedding their input data into prompt templates which contain instructions and/or in-context examples, creating queries which are submitted to a LLM, and then parsing the LLM response in order to generate the system outputs. Prompt Injection Attacks (PIAs) are a type of subversion of these systems where a malicious user crafts special inputs which interfere with the prompt templates, causing the LLM to respond in ways unintended by the system designer. Recently, Sun and Miceli-Barone proposed a class of PIAs against LLM-based machine translation. Specifically, the task is to translate questions from the TruthfulQA test suite, where an adversarial prompt is prepended to the questions, instructing the system to ignore the translation instruction and answer the questions instead. In this test suite, we extend this approach to all the language pairs of the WMT 2024 General Machine Translation task. Moreover, we include additional attack formats in addition to the one originally studied.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05045v1">Can LLMs plan paths with extra hints from solvers?</a></div>
    <div class="paper-meta">
      📅 2024-10-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable capabilities in natural language processing, mathematical problem solving, and tasks related to program synthesis. However, their effectiveness in long-term planning and higher-order reasoning has been noted to be limited and fragile. This paper explores an approach for enhancing LLM performance in solving a classical robotic planning task by integrating solver-generated feedback. We explore four different strategies for providing feedback, including visual feedback, we utilize fine-tuning, and we evaluate the performance of three different LLMs across a 10 standard and 100 more randomly generated planning problems. Our results suggest that the solver-generated feedback improves the LLM's ability to solve the moderately difficult problems, but the harder problems still remain out of reach. The study provides detailed analysis of the effects of the different hinting strategies and the different planning tendencies of the evaluated LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05004v1">Fast State Restoration in LLM Serving with HCache</a></div>
    <div class="paper-meta">
      📅 2024-10-07
      | 💬 EuroSys 2025
    </div>
    <details class="paper-abstract">
      The growing complexity of LLM usage today, e.g., multi-round conversation and retrieval-augmented generation (RAG), makes contextual states (i.e., KV cache) reusable across user requests. Given the capacity constraints of GPU memory, only a limited number of contexts can be cached on GPU for reusing. Existing inference systems typically evict part of the KV cache and restore it by recomputing it from the original tokens or offloading it to host storage for later retrieval, both of which introduce substantial computational or I/O overheads. We propose HCache, a novel LLM state restoration method. Its key idea is to restore LLM states from intermediate activations and thus utilize computational and I/O resources with low overhead. We enhance HCache with two techniques, including i) a bubble-free restoration scheduler that integrates resource-complementary methods to optimize the balance between computation and IO tasks; and ii) a chunk-based storage manager to address the layout mismatch issue (i.e., layer-before-token saving versus token-before-layer restoration). Our evaluations, conducted using real-world tasks, show that HCache reduces the TTFT by up to 1.93X compared to KV offload while consuming 1.92-2.40X less storage space; compared to token recomputation, HCache achieves up to 5.73X reduction in TTFT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.04925v1">Intent Classification for Bank Chatbots through LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2024-10-07
      | 💬 7 pages, no figures
    </div>
    <details class="paper-abstract">
      This study evaluates the application of large language models (LLMs) for intent classification within a chatbot with predetermined responses designed for banking industry websites. Specifically, the research examines the effectiveness of fine-tuning SlovakBERT compared to employing multilingual generative models, such as Llama 8b instruct and Gemma 7b instruct, in both their pre-trained and fine-tuned versions. The findings indicate that SlovakBERT outperforms the other models in terms of in-scope accuracy and out-of-scope false positive rate, establishing it as the benchmark for this application.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.07578v2">A Novel Mathematical Framework for Objective Characterization of Ideas through Vector Embeddings in LLM</a></div>
    <div class="paper-meta">
      📅 2024-10-07
      | 💬 20 pages, 12 figures, 5 tables
    </div>
    <details class="paper-abstract">
      The demand for innovation in product design necessitates a prolific ideation phase. Conversational AI (CAI) systems that use Large Language Models (LLMs) such as GPT (Generative Pre-trained Transformer) have been shown to be fruitful in augmenting human creativity, providing numerous novel and diverse ideas. Despite the success in ideation quantity, the qualitative assessment of these ideas remains challenging and traditionally reliant on expert human evaluation. This method suffers from limitations such as human judgment errors, bias, and oversight. Addressing this gap, our study introduces a comprehensive mathematical framework for automated analysis to objectively evaluate the plethora of ideas generated by CAI systems and/or humans. This framework is particularly advantageous for novice designers who lack experience in selecting promising ideas. By converting the ideas into higher dimensional vectors and quantitatively measuring the diversity between them using tools such as UMAP, DBSCAN and PCA, the proposed method provides a reliable and objective way of selecting the most promising ideas, thereby enhancing the efficiency of the ideation phase.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.08227v2">DALL-M: Context-Aware Clinical Data Augmentation with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-07
      | 💬 we introduce a pioneering approach to clinical data augmentation that employs large language models (LLMs) to generate patient contextual synthetic data. It preserves the integrity of real patient data while enriching the dataset with contextually relevant synthetic features, significantly enhancing model performance
    </div>
    <details class="paper-abstract">
      X-ray images are vital in medical diagnostics, but their effectiveness is limited without clinical context. Radiologists often find chest X-rays insufficient for diagnosing underlying diseases, necessitating comprehensive clinical features and data integration. We present a novel framework to enhance the clinical context through augmentation techniques with clinical tabular data, thereby improving its applicability and reliability in AI medical diagnostics. We introduce a pioneering approach to clinical data augmentation that employs large language models to generate patient contextual synthetic data. This methodology is crucial for training more robust deep learning models in healthcare. It preserves the integrity of real patient data while enriching the dataset with contextually relevant synthetic features, significantly enhancing model performance. Our methodology, termed DALL-M, uses a three-phase feature generation process: (i)clinical context storage, (ii)expert query generation, and (iii)context-aware feature augmentation. DALL-M generates new, clinically relevant features by synthesizing chest X-ray images and reports. Applied to 799 cases using nine features from the MIMIC-IV dataset, it created an augmented set of 91 features. This is the first work to generate contextual values for patients' X-ray reports. Specifically, we provide (i)the capacity of LLMs to generate contextual synthetic values for existing clinical features and (ii)their ability to create entirely new clinically relevant features. Empirical validation with machine learning models showed significant performance improvements. Incorporating augmented features increased the F1 score by 16.5% and Precision and Recall by approximately 25%. DALL-M addresses a critical gap in clinical data augmentation, offering a robust framework for generating contextually enriched datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.15625v2">CBF-LLM: Safe Control for LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2024-10-07
    </div>
    <details class="paper-abstract">
      This paper proposes a control-based framework for aligning large language models (LLMs) by leveraging a control barrier function (CBF) to ensure user-desirable text generation. The presented framework applies the safety filter, designed based on the CBF, to the output generation of the baseline LLM, i.e., the sequence of the token, with the aim of intervening in the generated text. The overall text-generation system is implemented with Llama 3 and a RoBERTa model, and the source code is available at https://github.com/Mya-Mya/CBF-LLM. The experiment demonstrates its control ability and effectiveness in reducing the number of interventions needed for user-specified alignment tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.08617v4">RTLCoder: Fully Open-Source and Efficient LLM-Assisted RTL Code Generation Technique</a></div>
    <div class="paper-meta">
      📅 2024-10-07
      | 💬 Accepted by IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems
    </div>
    <details class="paper-abstract">
      The automatic generation of RTL code (e.g., Verilog) using natural language instructions and large language models (LLMs) has attracted significant research interest recently. However, most existing approaches heavily rely on commercial LLMs such as ChatGPT, while open-source LLMs tailored for this specific design generation task exhibit notably inferior performance. The absence of high-quality open-source solutions restricts the flexibility and data privacy of this emerging technique. In this study, we present a new customized LLM solution with a modest parameter count of only 7B, achieving better performance than GPT-3.5 on all representative benchmarks for RTL code generation. Especially, it outperforms GPT-4 in VerilogEval Machine benchmark. This remarkable balance between accuracy and efficiency is made possible by leveraging our new RTL code dataset and a customized LLM algorithm, both of which have been made fully open-source. Furthermore, we have successfully quantized our LLM to 4-bit with a total size of 4GB, enabling it to function on a single laptop with only slight performance degradation. This efficiency allows the RTL generator to serve as a local assistant for engineers, ensuring all design privacy concerns are addressed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.04790v1">GARLIC: LLM-Guided Dynamic Progress Control with Hierarchical Weighted Graph for Long Document QA</a></div>
    <div class="paper-meta">
      📅 2024-10-07
    </div>
    <details class="paper-abstract">
      In the past, Retrieval-Augmented Generation (RAG) methods split text into chunks to enable language models to handle long documents. Recent tree-based RAG methods are able to retrieve detailed information while preserving global context. However, with the advent of more powerful LLMs, such as Llama 3.1, which offer better comprehension and support for longer inputs, we found that even recent tree-based RAG methods perform worse than directly feeding the entire document into Llama 3.1, although RAG methods still hold an advantage in reducing computational costs. In this paper, we propose a new retrieval method, called LLM-Guided Dynamic Progress Control with Hierarchical Weighted Graph (GARLIC), which outperforms previous state-of-the-art baselines, including Llama 3.1, while retaining the computational efficiency of RAG methods. Our method introduces several improvements: (1) Rather than using a tree structure, we construct a Hierarchical Weighted Directed Acyclic Graph with many-to-many summarization, where the graph edges are derived from attention mechanisms, and each node focuses on a single event or very few events. (2) We introduce a novel retrieval method that leverages the attention weights of LLMs rather than dense embedding similarity. Our method allows for searching the graph along multiple paths and can terminate at any depth. (3) We use the LLM to control the retrieval process, enabling it to dynamically adjust the amount and depth of information retrieved for different queries. Experimental results show that our method outperforms previous state-of-the-art baselines, including Llama 3.1, on two single-document and two multi-document QA datasets, while maintaining similar computational complexity to traditional RAG methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.04759v1">Driving with Regulation: Interpretable Decision-Making for Autonomous Vehicles with Retrieval-Augmented Reasoning via LLM</a></div>
    <div class="paper-meta">
      📅 2024-10-07
    </div>
    <details class="paper-abstract">
      This work presents an interpretable decision-making framework for autonomous vehicles that integrates traffic regulations, norms, and safety guidelines comprehensively and enables seamless adaptation to different regions. While traditional rule-based methods struggle to incorporate the full scope of traffic rules, we develop a Traffic Regulation Retrieval (TRR) Agent based on Retrieval-Augmented Generation (RAG) to automatically retrieve relevant traffic rules and guidelines from extensive regulation documents and relevant records based on the ego vehicle's situation. Given the semantic complexity of the retrieved rules, we also design a reasoning module powered by a Large Language Model (LLM) to interpret these rules, differentiate between mandatory rules and safety guidelines, and assess actions on legal compliance and safety. Additionally, the reasoning is designed to be interpretable, enhancing both transparency and reliability. The framework demonstrates robust performance on both hypothesized and real-world cases across diverse scenarios, along with the ability to adapt to different regions with ease.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.13803v3">"I Like Sunnie More Than I Expected!": Exploring User Expectation and Perception of an Anthropomorphic LLM-based Conversational Agent for Well-Being Support</a></div>
    <div class="paper-meta">
      📅 2024-10-07
      | 💬 In Submission
    </div>
    <details class="paper-abstract">
      The human-computer interaction (HCI) research community has a longstanding interest in exploring the mismatch between users' actual experiences and expectation toward new technologies, for instance, large language models (LLMs). In this study, we compared users' (N = 38) initial expectations against their post-interaction perceptions of two LLM-powered mental well-being intervention activity recommendation systems. Both systems have a built-in LLM to recommend a personalized well-being intervention activity, but one system (Sunnie) has an anthropomorphic conversational interaction design via elements such as appearance, persona, and natural conversation. Results showed that user engagement was high with both systems, and both systems exceeded users' expectations along the utility dimension, highlighting AI's potential to offer useful intervention activity recommendations. In addition, Sunnie further outperformed the non-anthropomorphic baseline system in relational warmth. These findings suggest that anthropomorphic conversational interaction design may be particularly effective in fostering warmth in mental health support contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.20267v4">Auto-Arena: Automating LLM Evaluations with Agent Peer Battles and Committee Discussions</a></div>
    <div class="paper-meta">
      📅 2024-10-07
    </div>
    <details class="paper-abstract">
      As LLMs continuously evolve, there is an urgent need for a reliable evaluation method that delivers trustworthy results promptly. Currently, static benchmarks suffer from inflexibility and unreliability, leading users to prefer human voting platforms like Chatbot Arena. However, human evaluations require significant manual effort. To address this, we propose the Auto-Arena, an innovative framework that automates the entire evaluation process using LLM-powered agents. Firstly, an LLM examiner generates questions. Then, two LLM candidates engage in a multi-round peer battle based on individual questions, aiming at revealing their true performance differences. Finally, a committee of LLM judges collaboratively discusses and decides the winner, reducing bias and enhancing fairness. During the peer battles, we observe intriguing scenarios where the LLM candidates display competitive behaviors and even learn from the opponents. In our extensive experiments involving 15 recent LLMs, Auto-Arena shows a 92.14% correlation with human preferences, surpassing all previous expert-annotated benchmarks without any manual efforts. As a result, Auto-Arena offers a promising alternative to current human evaluation platforms for evaluating LLMs automatically.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.04699v1">The LLM Effect: Are Humans Truly Using LLMs, or Are They Being Influenced By Them Instead?</a></div>
    <div class="paper-meta">
      📅 2024-10-07
      | 💬 Accepted to EMNLP Main 2024. First two authors contributed equally
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown capabilities close to human performance in various analytical tasks, leading researchers to use them for time and labor-intensive analyses. However, their capability to handle highly specialized and open-ended tasks in domains like policy studies remains in question. This paper investigates the efficiency and accuracy of LLMs in specialized tasks through a structured user study focusing on Human-LLM partnership. The study, conducted in two stages-Topic Discovery and Topic Assignment-integrates LLMs with expert annotators to observe the impact of LLM suggestions on what is usually human-only analysis. Results indicate that LLM-generated topic lists have significant overlap with human generated topic lists, with minor hiccups in missing document-specific topics. However, LLM suggestions may significantly improve task completion speed, but at the same time introduce anchoring bias, potentially affecting the depth and nuance of the analysis, raising a critical question about the trade-off between increased efficiency and the risk of biased analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.04698v1">MathHay: An Automated Benchmark for Long-Context Mathematical Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-07
      | 💬 Work-in-Progress
    </div>
    <details class="paper-abstract">
      Recent large language models (LLMs) have demonstrated versatile capabilities in long-context scenarios. Although some recent benchmarks have been developed to evaluate the long-context capabilities of LLMs, there is a lack of benchmarks evaluating the mathematical reasoning abilities of LLMs over long contexts, which is crucial for LLMs' application in real-world scenarios. In this paper, we introduce MathHay, an automated benchmark designed to assess the long-context mathematical reasoning capabilities of LLMs. Unlike previous benchmarks like Needle in a Haystack, which focus primarily on information retrieval within long texts, MathHay demands models with both information-seeking and complex mathematical reasoning abilities. We conduct extensive experiments on MathHay to assess the long-context mathematical reasoning abilities of eight top-performing LLMs. Even the best-performing model, Gemini-1.5-Pro-002, still struggles with mathematical reasoning over long contexts, achieving only 51.26% accuracy at 128K tokens. This highlights the significant room for improvement on the MathHay benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.08464v2">Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing</a></div>
    <div class="paper-meta">
      📅 2024-10-07
      | 💬 Link: https://magpie-align.github.io/
    </div>
    <details class="paper-abstract">
      High-quality instruction data is critical for aligning large language models (LLMs). Although some models, such as Llama-3-Instruct, have open weights, their alignment data remain private, which hinders the democratization of AI. High human labor costs and a limited, predefined scope for prompting prevent existing open-source data creation methods from scaling effectively, potentially limiting the diversity and quality of public alignment datasets. Is it possible to synthesize high-quality instruction data at scale by extracting it directly from an aligned LLM? We present a self-synthesis method for generating large-scale alignment data named Magpie. Our key observation is that aligned LLMs like Llama-3-Instruct can generate a user query when we input only the left-side templates up to the position reserved for user messages, thanks to their auto-regressive nature. We use this method to prompt Llama-3-Instruct and generate 4 million instructions along with their corresponding responses. We perform a comprehensive analysis of the extracted data and select 300K high-quality instances. To compare Magpie data with other public instruction datasets, we fine-tune Llama-3-8B-Base with each dataset and evaluate the performance of the fine-tuned models. Our results indicate that in some tasks, models fine-tuned with Magpie perform comparably to the official Llama-3-8B-Instruct, despite the latter being enhanced with 10 million data points through supervised fine-tuning (SFT) and subsequent feedback learning. We also show that using Magpie solely for SFT can surpass the performance of previous public datasets utilized for both SFT and preference optimization, such as direct preference optimization with UltraFeedback. This advantage is evident on alignment benchmarks such as AlpacaEval, ArenaHard, and WildBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.16406v3">SpinQuant: LLM quantization with learned rotations</a></div>
    <div class="paper-meta">
      📅 2024-10-07
    </div>
    <details class="paper-abstract">
      Post-training quantization (PTQ) techniques applied to weights, activations, and the KV cache greatly reduce memory usage, latency, and power consumption of Large Language Models (LLMs), but may lead to large quantization errors when outliers are present. Rotating activation or weight matrices helps remove outliers and benefits quantization. In this work, we identify a collection of applicable rotation parameterizations that lead to identical outputs in full-precision Transformer architectures while enhancing quantization accuracy. In addition, we find that some random rotations lead to much better quantization than others, with an up to 13 points difference in downstream zero-shot reasoning performance. As a result, we propose SpinQuant, a novel approach that incorporates learned rotation matrices for optimal quantized network accuracy. With 4-bit quantization of weight, activation, and KV-cache, SpinQuant narrows the accuracy gap on zero-shot reasoning tasks with full precision to merely 2.9 points on the LLaMA-2 7B model, surpassing LLM-QAT by 19.1 points and SmoothQuant by 25.0 points. Furthermore, SpinQuant also outperforms concurrent work QuaRot, which applies random rotations to remove outliers. In particular, for LLaMA-3 8B models that are hard to quantize, SpinQuant reduces the gap to full precision by up to 45.1% relative to QuaRot.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.10382v3">Efficient Prompting for LLM-based Generative Internet of Things</a></div>
    <div class="paper-meta">
      📅 2024-10-06
      | 💬 14 pages, 11 figures. IEEE Internet of Things Journal, 2024
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable capacities on various tasks, and integrating the capacities of LLMs into the Internet of Things (IoT) applications has drawn much research attention recently. Due to security concerns, many institutions avoid accessing state-of-the-art commercial LLM services, requiring the deployment and utilization of open-source LLMs in a local network setting. However, open-source LLMs usually have more limitations regarding their performance, such as their arithmetic calculation and reasoning capacities, and practical systems of applying LLMs to IoT have yet to be well-explored. Therefore, we propose a LLM-based Generative IoT (GIoT) system deployed in the local network setting in this study. To alleviate the limitations of LLMs and provide service with competitive performance, we apply prompt engineering methods to enhance the capacities of the open-source LLMs, design a Prompt Management Module and a Post-processing Module to manage the tailored prompts for different tasks and process the results generated by the LLMs. To demonstrate the effectiveness of the proposed system, we discuss a challenging Table Question Answering (Table-QA) task as a case study of the proposed system, as tabular data is usually more challenging than plain text because of their complex structures, heterogeneous data types and sometimes huge sizes. We conduct comprehensive experiments on two popular Table-QA datasets, and the results show that our proposal can achieve competitive performance compared with state-of-the-art LLMs, demonstrating that the proposed LLM-based GIoT system can provide competitive performance with tailored prompting methods and is easily extensible to new tasks without training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.15694v2">Counter Turing Test ($CT^2$): Investigating AI-Generated Text Detection for Hindi -- Ranking LLMs based on Hindi AI Detectability Index ($ADI_{hi}$)</a></div>
    <div class="paper-meta">
      📅 2024-10-06
      | 💬 Accepted at EMNLP 2024 Findings
    </div>
    <details class="paper-abstract">
      The widespread adoption of Large Language Models (LLMs) and awareness around multilingual LLMs have raised concerns regarding the potential risks and repercussions linked to the misapplication of AI-generated text, necessitating increased vigilance. While these models are primarily trained for English, their extensive training on vast datasets covering almost the entire web, equips them with capabilities to perform well in numerous other languages. AI-Generated Text Detection (AGTD) has emerged as a topic that has already received immediate attention in research, with some initial methods having been proposed, soon followed by the emergence of techniques to bypass detection. In this paper, we report our investigation on AGTD for an indic language Hindi. Our major contributions are in four folds: i) examined 26 LLMs to evaluate their proficiency in generating Hindi text, ii) introducing the AI-generated news article in Hindi ($AG_{hi}$) dataset, iii) evaluated the effectiveness of five recently proposed AGTD techniques: ConDA, J-Guard, RADAR, RAIDAR and Intrinsic Dimension Estimation for detecting AI-generated Hindi text, iv) proposed Hindi AI Detectability Index ($ADI_{hi}$) which shows a spectrum to understand the evolving landscape of eloquence of AI-generated text in Hindi. The code and dataset is available at https://github.com/ishank31/Counter_Turing_Test
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.04616v1">LRQ-Fact: LLM-Generated Relevant Questions for Multimodal Fact-Checking</a></div>
    <div class="paper-meta">
      📅 2024-10-06
    </div>
    <details class="paper-abstract">
      Human fact-checkers have specialized domain knowledge that allows them to formulate precise questions to verify information accuracy. However, this expert-driven approach is labor-intensive and is not scalable, especially when dealing with complex multimodal misinformation. In this paper, we propose a fully-automated framework, LRQ-Fact, for multimodal fact-checking. Firstly, the framework leverages Vision-Language Models (VLMs) and Large Language Models (LLMs) to generate comprehensive questions and answers for probing multimodal content. Next, a rule-based decision-maker module evaluates both the original content and the generated questions and answers to assess the overall veracity. Extensive experiments on two benchmarks show that LRQ-Fact improves detection accuracy for multimodal misinformation. Moreover, we evaluate its generalizability across different model backbones, offering valuable insights for further refinement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.04601v1">ProtocoLLM: Automatic Evaluation Framework of LLMs on Domain-Specific Scientific Protocol Formulation Tasks</a></div>
    <div class="paper-meta">
      📅 2024-10-06
      | 💬 Submitted to 2024 ACL Rolling Review June Cycle
    </div>
    <details class="paper-abstract">
      Automated generation of scientific protocols executable by robots can significantly accelerate scientific research processes. Large Language Models (LLMs) excel at Scientific Protocol Formulation Tasks (SPFT), but the evaluation of their capabilities rely on human evaluation. Here, we propose a flexible, automatic framework to evaluate LLM's capability on SPFT: ProtocoLLM. This framework prompts the target model and GPT-4 to extract pseudocode from biology protocols using only predefined lab actions and evaluates the output of target model using LLAM-EVAL, the pseudocode generated by GPT-4 serving as a baseline and Llama-3 acting as the evaluator. Our adaptable prompt-based evaluation method, LLAM-EVAL, offers significant flexibility in terms of evaluation model, material, criteria, and is free of cost. We evaluate GPT variations, Llama, Mixtral, Gemma, Cohere, and Gemini. Overall, we find that GPT and Cohere is a powerful scientific protocol formulators. We also introduce BIOPROT 2.0, a dataset with biology protocols and corresponding pseudocodes, which can aid LLMs in formulation and evaluation of SPFT. Our work is extensible to assess LLMs on SPFT across various domains and other fields that require protocol generation for specific goals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.04571v1">EnsemW2S: Can an Ensemble of LLMs be Leveraged to Obtain a Stronger LLM?</a></div>
    <div class="paper-meta">
      📅 2024-10-06
    </div>
    <details class="paper-abstract">
      How can we harness the collective capabilities of multiple Large Language Models (LLMs) to create an even more powerful model? This question forms the foundation of our research, where we propose an innovative approach to weak-to-strong (w2s) generalization-a critical problem in AI alignment. Our work introduces an easy-to-hard (e2h) framework for studying the feasibility of w2s generalization, where weak models trained on simpler tasks collaboratively supervise stronger models on more complex tasks. This setup mirrors real-world challenges, where direct human supervision is limited. To achieve this, we develop a novel AdaBoost-inspired ensemble method, demonstrating that an ensemble of weak supervisors can enhance the performance of stronger LLMs across classification and generative tasks on difficult QA datasets. In several cases, our ensemble approach matches the performance of models trained on ground-truth data, establishing a new benchmark for w2s generalization. We observe an improvement of up to 14% over existing baselines and average improvements of 5% and 4% for binary classification and generative tasks, respectively. This research points to a promising direction for enhancing AI through collective supervision, especially in scenarios where labeled data is sparse or insufficient.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11859v1">SouLLMate: An Adaptive LLM-Driven System for Advanced Mental Health Support and Assessment, Based on a Systematic Application Survey</a></div>
    <div class="paper-meta">
      📅 2024-10-06
    </div>
    <details class="paper-abstract">
      Mental health issues significantly impact individuals' daily lives, yet many do not receive the help they need even with available online resources. This study aims to provide accessible, stigma-free, personalized, and real-time mental health support through cutting-edge AI technologies. It makes the following contributions: (1) Conducting an extensive survey of recent mental health support methods to identify prevalent functionalities and unmet needs. (2) Introducing SouLLMate, an adaptive LLM-driven system that integrates LLM technologies, Chain, Retrieval-Augmented Generation (RAG), prompt engineering, and domain knowledge. This system offers advanced features such as Suicide Risk Detection and Proactive Guidance Dialogue, and utilizes RAG for personalized profile uploads and Conversational Information Extraction. (3) Developing novel evaluation approaches to assess preliminary assessments and suicide risk detection, utilizing annotated real-life interview data and professionally labeled datasets indicating suicide tendencies. (4) Proposing Key Indicator Summarization (KIS) and Proactive Questioning Strategy (PQS) methods to enhance model performance and usability through context-sensitive response adjustments and semantic coherence evaluations. This study contributes to advancing mental health support technologies, potentially improving the accessibility and effectiveness of mental health care globally.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.04541v1">On Evaluating LLMs' Capabilities as Functional Approximators: A Bayesian Perspective</a></div>
    <div class="paper-meta">
      📅 2024-10-06
    </div>
    <details class="paper-abstract">
      Recent works have successfully applied Large Language Models (LLMs) to function modeling tasks. However, the reasons behind this success remain unclear. In this work, we propose a new evaluation framework to comprehensively assess LLMs' function modeling abilities. By adopting a Bayesian perspective of function modeling, we discover that LLMs are relatively weak in understanding patterns in raw data, but excel at utilizing prior knowledge about the domain to develop a strong understanding of the underlying function. Our findings offer new insights about the strengths and limitations of LLMs in the context of function modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.04521v1">MC-CoT: A Modular Collaborative CoT Framework for Zero-shot Medical-VQA with LLM and MLLM Integration</a></div>
    <div class="paper-meta">
      📅 2024-10-06
      | 💬 21 pages, 14 figures, 6 tables
    </div>
    <details class="paper-abstract">
      In recent advancements, multimodal large language models (MLLMs) have been fine-tuned on specific medical image datasets to address medical visual question answering (Med-VQA) tasks. However, this common approach of task-specific fine-tuning is costly and necessitates separate models for each downstream task, limiting the exploration of zero-shot capabilities. In this paper, we introduce MC-CoT, a modular cross-modal collaboration Chain-of-Thought (CoT) framework designed to enhance the zero-shot performance of MLLMs in Med-VQA by leveraging large language models (LLMs). MC-CoT improves reasoning and information extraction by integrating medical knowledge and task-specific guidance, where LLM provides various complex medical reasoning chains and MLLM provides various observations of medical images based on instructions of the LLM. Our experiments on datasets such as SLAKE, VQA-RAD, and PATH-VQA show that MC-CoT surpasses standalone MLLMs and various multimodality CoT frameworks in recall rate and accuracy. These findings highlight the importance of incorporating background information and detailed guidance in addressing complex zero-shot Med-VQA tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.04519v1">RevMUX: Data Multiplexing with Reversible Adapters for Efficient LLM Batch Inference</a></div>
    <div class="paper-meta">
      📅 2024-10-06
      | 💬 EMNLP 2024 Main Conference
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have brought a great breakthrough to the natural language processing (NLP) community, while leading the challenge of handling concurrent customer queries due to their high throughput demands. Data multiplexing addresses this by merging multiple inputs into a single composite input, allowing more efficient inference through a shared forward pass. However, as distinguishing individuals from a composite input is challenging, conventional methods typically require training the entire backbone, yet still suffer from performance degradation. In this paper, we introduce RevMUX, a parameter-efficient data multiplexing framework that incorporates a reversible design in the multiplexer, which can be reused by the demultiplexer to perform reverse operations and restore individual samples for classification. Extensive experiments on four datasets and three types of LLM backbones demonstrate the effectiveness of RevMUX for enhancing LLM inference efficiency while retaining a satisfactory classification performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.14741v3">Generate-on-Graph: Treat LLM as both Agent and KG in Incomplete Knowledge Graph Question Answering</a></div>
    <div class="paper-meta">
      📅 2024-10-06
      | 💬 Accepted by EMNLP 2024 Main
    </div>
    <details class="paper-abstract">
      To address the issues of insufficient knowledge and hallucination in Large Language Models (LLMs), numerous studies have explored integrating LLMs with Knowledge Graphs (KGs). However, these methods are typically evaluated on conventional Knowledge Graph Question Answering (KGQA) with complete KGs, where all factual triples required for each question are entirely covered by the given KG. In such cases, LLMs primarily act as an agent to find answer entities within the KG, rather than effectively integrating the internal knowledge of LLMs and external knowledge sources such as KGs. In fact, KGs are often incomplete to cover all the knowledge required to answer questions. To simulate these real-world scenarios and evaluate the ability of LLMs to integrate internal and external knowledge, we propose leveraging LLMs for QA under Incomplete Knowledge Graph (IKGQA), where the provided KG lacks some of the factual triples for each question, and construct corresponding datasets. To handle IKGQA, we propose a training-free method called Generate-on-Graph (GoG), which can generate new factual triples while exploring KGs. Specifically, GoG performs reasoning through a Thinking-Searching-Generating framework, which treats LLM as both Agent and KG in IKGQA. Experimental results on two datasets demonstrate that our GoG outperforms all previous methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.09823v2">NativQA: Multilingual Culturally-Aligned Natural Query for LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-06
      | 💬 LLMs, Native, Multilingual, Language Diversity, Contextual Understanding, Minority Languages, Culturally Informed, Foundation Models, Large Language Models
    </div>
    <details class="paper-abstract">
      Natural Question Answering (QA) datasets play a crucial role in evaluating the capabilities of large language models (LLMs), ensuring their effectiveness in real-world applications. Despite the numerous QA datasets that have been developed, there is a notable lack of region-specific datasets generated by native users in their own languages. This gap hinders the effective benchmarking of LLMs for regional and cultural specificities. Furthermore, it also limits the development of fine-tuned models. In this study, we propose a scalable, language-independent framework, NativQA, to seamlessly construct culturally and regionally aligned QA datasets in native languages, for LLM evaluation and tuning. We demonstrate the efficacy of the proposed framework by designing a multilingual natural QA dataset, \mnqa, consisting of ~64k manually annotated QA pairs in seven languages, ranging from high to extremely low resource, based on queries from native speakers from 9 regions covering 18 topics. We benchmark open- and closed-source LLMs with the MultiNativQA dataset. We also showcase the framework efficacy in constructing fine-tuning data especially for low-resource and dialectally-rich languages. We made both the framework NativQA and MultiNativQA dataset publicly available for the community (https://nativqa.gitlab.io).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09083v1">Alignment Between the Decision-Making Logic of LLMs and Human Cognition: A Case Study on Legal LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-06
    </div>
    <details class="paper-abstract">
      This paper presents a method to evaluate the alignment between the decision-making logic of Large Language Models (LLMs) and human cognition in a case study on legal LLMs. Unlike traditional evaluations on language generation results, we propose to evaluate the correctness of the detailed decision-making logic of an LLM behind its seemingly correct outputs, which represents the core challenge for an LLM to earn human trust. To this end, we quantify the interactions encoded by the LLM as primitive decision-making logic, because recent theoretical achievements have proven several mathematical guarantees of the faithfulness of the interaction-based explanation. We design a set of metrics to evaluate the detailed decision-making logic of LLMs. Experiments show that even when the language generation results appear correct, a significant portion of the internal inference logic contains notable issues.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.07246v2">Propaganda to Hate: A Multimodal Analysis of Arabic Memes with Multi-Agent LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-06
      | 💬 propaganda, hate-speech, disinformation, misinformation, fake news, LLMs, GPT-4, multimodality, multimodal LLMs
    </div>
    <details class="paper-abstract">
      In the past decade, social media platforms have been used for information dissemination and consumption. While a major portion of the content is posted to promote citizen journalism and public awareness, some content is posted to mislead users. Among different content types such as text, images, and videos, memes (text overlaid on images) are particularly prevalent and can serve as powerful vehicles for propaganda, hate, and humor. In the current literature, there have been efforts to individually detect such content in memes. However, the study of their intersection is very limited. In this study, we explore the intersection between propaganda and hate in memes using a multi-agent LLM-based approach. We extend the propagandistic meme dataset with coarse and fine-grained hate labels. Our finding suggests that there is an association between propaganda and hate in memes. We provide detailed experimental results that can serve as a baseline for future studies. We will make the experimental resources publicly available to the community (https://github.com/firojalam/propaganda-and-hateful-memes).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.12452v2">Characterizing LLM Abstention Behavior in Science QA with Context Perturbations</a></div>
    <div class="paper-meta">
      📅 2024-10-06
      | 💬 EMNLP 2024 Findings
    </div>
    <details class="paper-abstract">
      The correct model response in the face of uncertainty is to abstain from answering a question so as not to mislead the user. In this work, we study the ability of LLMs to abstain from answering context-dependent science questions when provided insufficient or incorrect context. We probe model sensitivity in several settings: removing gold context, replacing gold context with irrelevant context, and providing additional context beyond what is given. In experiments on four QA datasets with six LLMs, we show that performance varies greatly across models, across the type of context provided, and also by question type; in particular, many LLMs seem unable to abstain from answering boolean questions using standard QA prompts. Our analysis also highlights the unexpected impact of abstention performance on QA task accuracy. Counter-intuitively, in some settings, replacing gold context with irrelevant context or adding irrelevant context to gold context can improve abstention performance in a way that results in improvements in task performance. Our results imply that changes are needed in QA dataset design and evaluation to more effectively assess the correctness and downstream impacts of model abstention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.16367v3">Unraveling Babel: Exploring Multilingual Activation Patterns of LLMs and Their Applications</a></div>
    <div class="paper-meta">
      📅 2024-10-06
      | 💬 Accepted to EMNLP 2024 Main Conference
    </div>
    <details class="paper-abstract">
      Recently, large language models (LLMs) have achieved tremendous breakthroughs in the field of NLP, but still lack understanding of their internal neuron activities when processing different languages. We designed a method to convert dense LLMs into fine-grained MoE architectures, and then visually studied the multilingual activation patterns of LLMs through expert activation frequency heatmaps. Through comprehensive experiments on different model families, different model sizes, and different variants, we analyzed the similarities and differences in the internal neuron activation patterns of LLMs when processing different languages. Specifically, we investigated the distribution of high-frequency activated experts, multilingual shared experts, whether multilingual activation patterns are related to language families, and the impact of instruction tuning on activation patterns. We further explored leveraging the discovered differences in expert activation frequencies to guide sparse activation and pruning. Experimental results demonstrated that our method significantly outperformed random expert pruning and even exceeded the performance of unpruned models in some languages. Additionally, we found that configuring different pruning rates for different layers based on activation level differences could achieve better results. Our findings reveal the multilingual processing mechanisms within LLMs and utilize these insights to offer new perspectives for applications such as sparse activation and model pruning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10849v1">Continuous Approximations for Improving Quantization Aware Training of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-06
    </div>
    <details class="paper-abstract">
      Model compression methods are used to reduce the computation and energy requirements for Large Language Models (LLMs). Quantization Aware Training (QAT), an effective model compression method, is proposed to reduce performance degradation after quantization. To further minimize this degradation, we introduce two continuous approximations to the QAT process on the rounding function, traditionally approximated by the Straight-Through Estimator (STE), and the clamping function. By applying both methods, the perplexity (PPL) on the WikiText-v2 dataset of the quantized model reaches 9.0815, outperforming 9.9621 by the baseline. Also, we achieve a 2.76% improvement on BoolQ, and a 5.47% improvement on MMLU, proving that the step sizes and weights can be learned more accurately with our approach. Our method achieves better performance with the same precision, model size, and training setup, contributing to the development of more energy-efficient LLMs technology that aligns with global sustainability goals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.04328v1">OD-Stega: LLM-Based Near-Imperceptible Steganography via Optimized Distributions</a></div>
    <div class="paper-meta">
      📅 2024-10-06
      | 💬 9 figures
    </div>
    <details class="paper-abstract">
      We consider coverless steganography where a Large Language Model (LLM) drives an arithmetic coding decoder to generate stego-texts. An efficient method should embed secret message bits in as few language tokens as possible, while still keeping the stego-text natural and fluent. We show that on the individual token level, this problem is mathematically equivalent to maximizing the entropy of a replacement probability distribution of the next token generation, subject to a constraint on the KL divergence between the chosen probability distribution and the original distribution given by the LLM. A closed-form solution is provided for the optimization problem, which can be computed efficiently. Several important practical issues are also tackled: 1) An often-overlooked tokenization mismatch issue is resolved with a simple prompt selection approach, 2) The combination of the optimized distribution and the vocabulary truncation technique is considered, and 3) The combination of the optimized distribution with other sequence-level selection heuristics to further enhance the efficiency and reliability is studied.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.04770v2">WildBench: Benchmarking LLMs with Challenging Tasks from Real Users in the Wild</a></div>
    <div class="paper-meta">
      📅 2024-10-05
      | 💬 Link: https://hf.co/spaces/allenai/WildBench
    </div>
    <details class="paper-abstract">
      We introduce WildBench, an automated evaluation framework designed to benchmark large language models (LLMs) using challenging, real-world user queries. WildBench consists of 1,024 tasks carefully selected from over one million human-chatbot conversation logs. For automated evaluation with WildBench, we have developed two metrics, WB-Reward and WB-Score, which are computable using advanced LLMs such as GPT-4-turbo. WildBench evaluation uses task-specific checklists to evaluate model outputs systematically and provides structured explanations that justify the scores and comparisons, resulting in more reliable and interpretable automatic judgments. WB-Reward employs fine-grained pairwise comparisons between model responses, generating five potential outcomes: much better, slightly better, slightly worse, much worse, or a tie. Unlike previous evaluations that employed a single baseline model, we selected three baseline models at varying performance levels to ensure a comprehensive pairwise evaluation. Additionally, we propose a simple method to mitigate length bias, by converting outcomes of ``slightly better/worse'' to ``tie'' if the winner response exceeds the loser one by more than $K$ characters. WB-Score evaluates the quality of model outputs individually, making it a fast and cost-efficient evaluation metric. WildBench results demonstrate a strong correlation with the human-voted Elo ratings from Chatbot Arena on hard tasks. Specifically, WB-Reward achieves a Pearson correlation of 0.98 with top-ranking models. Additionally, WB-Score reaches 0.95, surpassing both ArenaHard's 0.91 and AlpacaEval2.0's 0.89 for length-controlled win rates, as well as the 0.87 for regular win rates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.04251v1">Enhancing Future Link Prediction in Quantum Computing Semantic Networks through LLM-Initiated Node Features</a></div>
    <div class="paper-meta">
      📅 2024-10-05
    </div>
    <details class="paper-abstract">
      Quantum computing is rapidly evolving in both physics and computer science, offering the potential to solve complex problems and accelerate computational processes. The development of quantum chips necessitates understanding the correlations among diverse experimental conditions. Semantic networks built on scientific literature, representing meaningful relationships between concepts, have been used across various domains to identify knowledge gaps and novel concept combinations. Neural network-based approaches have shown promise in link prediction within these networks. This study proposes initializing node features using LLMs to enhance node representations for link prediction tasks in graph neural networks. LLMs can provide rich descriptions, reducing the need for manual feature creation and lowering costs. Our method, evaluated using various link prediction models on a quantum computing semantic network, demonstrated efficacy compared to traditional node embedding techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.04234v1">Functional Homotopy: Smoothing Discrete Optimization via Continuous Parameters for LLM Jailbreak Attacks</a></div>
    <div class="paper-meta">
      📅 2024-10-05
    </div>
    <details class="paper-abstract">
      Optimization methods are widely employed in deep learning to identify and mitigate undesired model responses. While gradient-based techniques have proven effective for image models, their application to language models is hindered by the discrete nature of the input space. This study introduces a novel optimization approach, termed the \emph{functional homotopy} method, which leverages the functional duality between model training and input generation. By constructing a series of easy-to-hard optimization problems, we iteratively solve these problems using principles derived from established homotopy methods. We apply this approach to jailbreak attack synthesis for large language models (LLMs), achieving a $20\%-30\%$ improvement in success rate over existing methods in circumventing established safe open-source models such as Llama-2 and Llama-3.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16287v1">Solution for OOD-CV UNICORN Challenge 2024 Object Detection Assistance LLM Counting Ability Improvement</a></div>
    <div class="paper-meta">
      📅 2024-10-05
    </div>
    <details class="paper-abstract">
      This report provide a detailed description of the method that we explored and proposed in the ECCV OOD-CV UNICORN Challenge 2024, which focusing on the robustness of responses from large language models. The dataset of this competition are OODCA-VQA and SketchyQA. In order to test the robustness of the model. The organizer extended two variants of the dataset OODCV-Counterfactual and Sketchy-Challenging. There are several difficulties with these datasets. Firstly, the Sketchy-Challenging dataset uses some rarer item categories to test the model's generalization ability. Secondly, in the OODCV-Counterfactual dataset, the given problems often have inflection points and computational steps, requiring the model to recognize them during the inference process. In order to address this issue, we propose a simple yet effective approach called Object Detection Assistance Large Language Model(LLM) Counting Ability Improvement(ODAC), which focuses on using the object detection model to assist the LLM. To clarify, our approach contains two main blocks: (1)Object Detection Assistance. (2) Counterfactual Specific prompt. Our approach ranked second in the final test with a score of 0.86.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16285v1">Assessing the Performance of Human-Capable LLMs -- Are LLMs Coming for Your Job?</a></div>
    <div class="paper-meta">
      📅 2024-10-05
    </div>
    <details class="paper-abstract">
      The current paper presents the development and validation of SelfScore, a novel benchmark designed to assess the performance of automated Large Language Model (LLM) agents on help desk and professional consultation tasks. Given the increasing integration of AI in industries, particularly within customer service, SelfScore fills a crucial gap by enabling the comparison of automated agents and human workers. The benchmark evaluates agents on problem complexity and response helpfulness, ensuring transparency and simplicity in its scoring system. The study also develops automated LLM agents to assess SelfScore and explores the benefits of Retrieval-Augmented Generation (RAG) for domain-specific tasks, demonstrating that automated LLM agents incorporating RAG outperform those without. All automated LLM agents were observed to perform better than the human control group. Given these results, the study raises concerns about the potential displacement of human workers, especially in areas where AI technologies excel. Ultimately, SelfScore provides a foundational tool for understanding the impact of AI in help desk environments while advocating for ethical considerations in the ongoing transition towards automation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.09434v2">HySem: A context length optimized LLM pipeline for unstructured tabular extraction</a></div>
    <div class="paper-meta">
      📅 2024-10-05
      | 💬 19 pages, 7 tables, 10 figures, 2 algorithms
    </div>
    <details class="paper-abstract">
      Regulatory compliance reporting in the pharmaceutical industry relies on detailed tables, but these are often under-utilized beyond compliance due to their unstructured format and arbitrary content. Extracting and semantically representing tabular data is challenging due to diverse table presentations. Large Language Models (LLMs) demonstrate substantial potential for semantic representation, yet they encounter challenges related to accuracy and context size limitations, which are crucial considerations for the industry applications. We introduce HySem, a pipeline that employs a novel context length optimization technique to generate accurate semantic JSON representations from HTML tables. This approach utilizes a custom fine-tuned model specifically designed for cost- and privacy-sensitive small and medium pharmaceutical enterprises. Running on commodity hardware and leveraging open-source models, HySem surpasses its peer open-source models in accuracy and provides competitive performance when benchmarked against OpenAI GPT-4o and effectively addresses context length limitations, which is a crucial factor for supporting larger tables.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.02686v2">Deconfounded Causality-aware Parameter-Efficient Fine-Tuning for Problem-Solving Improvement of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-05
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable efficiency in tackling various tasks based on human instructions, but studies reveal that they often struggle with tasks requiring reasoning, such as math or physics. This limitation raises questions about whether LLMs truly comprehend embedded knowledge or merely learn to replicate the token distribution without a true understanding of the content. In this paper, we delve into this problem and aim to enhance the reasoning capabilities of LLMs. First, we investigate if the model has genuine reasoning capabilities by visualizing the text generation process at the attention and representation level. Then, we formulate the reasoning process of LLMs into a causal framework, which provides a formal explanation of the problems observed in the visualization. Finally, building upon this causal framework, we propose Deconfounded Causal Adaptation (DCA), a novel parameter-efficient fine-tuning (PEFT) method to enhance the model's reasoning capabilities by encouraging the model to extract the general problem-solving skills and apply these skills to different questions. Experiments show that our method outperforms the baseline consistently across multiple benchmarks, and with only 1.2M tunable parameters, we achieve better or comparable results to other fine-tuning methods. This demonstrates the effectiveness and efficiency of our method in improving the overall accuracy and reliability of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.04112v1">Exploring LLM-based Data Annotation Strategies for Medical Dialogue Preference Alignment</a></div>
    <div class="paper-meta">
      📅 2024-10-05
      | 💬 14 Pages, 12 figures
    </div>
    <details class="paper-abstract">
      This research examines the use of Reinforcement Learning from AI Feedback (RLAIF) techniques to improve healthcare dialogue models, with the aim of tackling the challenges of preference-aligned data annotation while reducing the reliance on medical experts. We argue that the primary challenges in current RLAIF research for healthcare are the limitations of automated evaluation methods and the difficulties in accurately representing physician preferences. To address these challenges, we present a new evaluation framework based on standardized patient examinations. This framework is designed to objectively assess the effectiveness of large language models (LLMs) in guiding users and following instructions, enabling a comprehensive comparison across different models. Furthermore, our investigation of effective ways to express physician preferences using Constitutional AI algorithms highlighted the particular effectiveness of flowcharts. Utilizing this finding, we introduce an innovative agent-based approach for annotating preference data. This approach autonomously creates medical dialogue flows tailored to the patient's condition, demonstrates strong generalization abilities, and reduces the need for expert involvement. Our results show that the agent-based approach outperforms existing RLAIF annotation methods in standardized patient examinations and surpasses current open source medical dialogue LLMs in various test scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.15156v2">Student Data Paradox and Curious Case of Single Student-Tutor Model: Regressive Side Effects of Training LLMs for Personalized Learning</a></div>
    <div class="paper-meta">
      📅 2024-10-05
      | 💬 Accepted at EMNLP 2024 Findings Track
    </div>
    <details class="paper-abstract">
      The pursuit of personalized education has led to the integration of Large Language Models (LLMs) in developing intelligent tutoring systems. To better understand and adapt to individual student needs, including their misconceptions, LLMs need to be trained on extensive datasets of student-tutor dialogues. Our research uncovers a fundamental challenge in this approach: the ``Student Data Paradox.'' This paradox emerges when LLMs, trained on student data to understand learner behavior, inadvertently compromise their own factual knowledge and reasoning abilities. We investigate this paradox by training state-of-the-art language models on student-tutor dialogue datasets and evaluating their performance across multiple benchmarks. These benchmarks assess various aspects of language model capabilities, including reasoning, truthfulness, and common sense understanding. Our findings reveal significant declines in the models' performance across these diverse benchmarks, indicating a broad impact on their capabilities when trained to model student behavior. Our research makes two primary contributions: (1) empirical demonstration of the Student Data Paradox through quantitative analysis of model performance, and (2) introduction of ``hallucination tokens'' as a mitigation strategy. These tokens, while improving performance, highlight the persistent challenge of balancing accurate student behavior modeling with maintaining the LLM's integrity as an educational tool. This study emphasizes the need for innovative solutions to reconcile the conflicting goals of faithfully understanding diverse student cognition while preserving the model's ability to provide accurate information and guidance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.16700v2">Implicit Multimodal Alignment: On the Generalization of Frozen LLMs to Multimodal Inputs</a></div>
    <div class="paper-meta">
      📅 2024-10-05
      | 💬 NeurIPS 2024. Code: https://github.com/mshukor/ima-lmms. Project page: https://ima-lmms.github.io/
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive performance on multimodal tasks, without any multimodal finetuning. They are the building block for Large Multimodal Models, yet, we still lack a proper understanding of their success. In this work, we expose frozen LLMs to image, video, audio and text inputs and analyse their internal representation aiming to understand their generalization beyond textual inputs. Findings. Perceptual tokens (1) are easily distinguishable from textual ones inside LLMs, with significantly different representations, and complete translation to textual tokens does not exist. Yet, (2) both perceptual and textual tokens activate similar LLM weights. Despite being different, (3) perceptual and textual tokens are implicitly aligned inside LLMs, we call this the implicit multimodal alignment (IMA), and argue that this is linked to architectural design, helping LLMs to generalize. This provide more evidence to believe that the generalization of LLMs to multimodal inputs is mainly due to their architecture. Implications. (1) We find a positive correlation between the implicit alignment score and the task performance, suggesting that this could act as a proxy metric for model evaluation and selection. (2) A negative correlation exists regarding hallucinations, revealing that this problem is mainly due to misalignment between the internal perceptual and textual representations. (3) Perceptual tokens change slightly throughout the model, thus, we propose different approaches to skip computations (e.g. in FFN layers), and significantly reduce the inference cost. (4) Due to the slowly changing embeddings across layers, and the high overlap between textual and multimodal activated weights, we compress LLMs by keeping only 1 subnetwork that works well across a wide range of multimodal tasks. Paper code: https://github.com/mshukor/ima-lmms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.10019v3">R-Judge: Benchmarking Safety Risk Awareness for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2024-10-05
      | 💬 EMNLP Findings 2024
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have exhibited great potential in autonomously completing tasks across real-world applications. Despite this, these LLM agents introduce unexpected safety risks when operating in interactive environments. Instead of centering on the harmlessness of LLM-generated content in most prior studies, this work addresses the imperative need for benchmarking the behavioral safety of LLM agents within diverse environments. We introduce R-Judge, a benchmark crafted to evaluate the proficiency of LLMs in judging and identifying safety risks given agent interaction records. R-Judge comprises 569 records of multi-turn agent interaction, encompassing 27 key risk scenarios among 5 application categories and 10 risk types. It is of high-quality curation with annotated safety labels and risk descriptions. Evaluation of 11 LLMs on R-Judge shows considerable room for enhancing the risk awareness of LLMs: The best-performing model, GPT-4o, achieves 74.42% while no other models significantly exceed the random. Moreover, we reveal that risk awareness in open agent scenarios is a multi-dimensional capability involving knowledge and reasoning, thus challenging for LLMs. With further experiments, we find that fine-tuning on safety judgment significantly improve model performance while straightforward prompting mechanisms fail. R-Judge is publicly available at https://github.com/Lordog/R-Judge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.18892v2">IDGen: Item Discrimination Induced Prompt Generation for LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2024-10-05
      | 💬 NeurIPS 2024
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) grow increasingly adept at managing complex tasks, the evaluation set must keep pace with these advancements to ensure it remains sufficiently discriminative. Item Discrimination (ID) theory, which is widely used in educational assessment, measures the ability of individual test items to differentiate between high and low performers. Inspired by this theory, we propose an ID-induced prompt synthesis framework for evaluating LLMs to ensure the evaluation set can continually update and refine according to model abilities. Our data synthesis framework prioritizes both breadth and specificity. It can generate prompts that comprehensively evaluate the capabilities of LLMs while revealing meaningful performance differences between models, allowing for effective discrimination of their relative strengths and weaknesses across various tasks and domains. To produce high-quality data, we incorporate a self-correct mechanism into our generalization framework, and develop two models to predict prompt discrimination and difficulty score to facilitate our data synthesis framework, contributing valuable tools to evaluation data synthesis research. We apply our generated data to evaluate five SOTA models. Our data achieves an average score of 51.92, accompanied by a variance of 10.06. By contrast, previous works (i.e., SELF-INSTRUCT and WizardLM) obtain an average score exceeding 67, with a variance below 3.2. The results demonstrate that the data generated by our framework is more challenging and discriminative compared to previous works. We will release a dataset of over 3,000 carefully crafted prompts to facilitate evaluation research of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05318v1">Improving LLM Reasoning through Scaling Inference Computation with Collaborative Verification</a></div>
    <div class="paper-meta">
      📅 2024-10-05
    </div>
    <details class="paper-abstract">
      Despite significant advancements in the general capability of large language models (LLMs), they continue to struggle with consistent and accurate reasoning, especially in complex tasks such as mathematical and code reasoning. One key limitation is that LLMs are trained primarily on correct solutions, reducing their ability to detect and learn from errors, which hampers their ability to reliably verify and rank outputs. To address this, we scale up the inference-time computation by generating multiple reasoning paths and employing verifiers to assess and rank the generated outputs by correctness. To facilitate this, we introduce a comprehensive dataset consisting of correct and incorrect solutions for math and code tasks, generated by multiple LLMs. This diverse set of solutions enables verifiers to more effectively distinguish and rank correct answers from erroneous outputs. The training methods for building verifiers were selected based on an extensive comparison of existing approaches. Moreover, to leverage the unique strengths of different reasoning strategies, we propose a novel collaborative method integrating Chain-of-Thought (CoT) and Program-of-Thought (PoT) solutions for verification. CoT provides a clear, step-by-step reasoning process that enhances interpretability, while PoT, being executable, offers a precise and error-sensitive validation mechanism. By taking both of their strengths, our approach significantly improves the accuracy and reliability of reasoning verification. Our verifiers, Math-Rev and Code-Rev, demonstrate substantial performance gains to existing LLMs, achieving state-of-the-art results on benchmarks such as GSM8k and MATH and even outperforming GPT-4o with Qwen-72B-Instruct as the reasoner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.20774v2">Can We Trust Embodied Agents? Exploring Backdoor Attacks against Embodied LLM-based Decision-Making Systems</a></div>
    <div class="paper-meta">
      📅 2024-10-05
      | 💬 31 pages, including main paper, references, and appendix
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown significant promise in real-world decision-making tasks for embodied artificial intelligence, especially when fine-tuned to leverage their inherent common sense and reasoning abilities while being tailored to specific applications. However, this fine-tuning process introduces considerable safety and security vulnerabilities, especially in safety-critical cyber-physical systems. In this work, we propose the first comprehensive framework for Backdoor Attacks against LLM-based Decision-making systems (BALD) in embodied AI, systematically exploring the attack surfaces and trigger mechanisms. Specifically, we propose three distinct attack mechanisms: word injection, scenario manipulation, and knowledge injection, targeting various components in the LLM-based decision-making pipeline. We perform extensive experiments on representative LLMs (GPT-3.5, LLaMA2, PaLM2) in autonomous driving and home robot tasks, demonstrating the effectiveness and stealthiness of our backdoor triggers across various attack channels, with cases like vehicles accelerating toward obstacles and robots placing knives on beds. Our word and knowledge injection attacks achieve nearly 100% success rate across multiple models and datasets while requiring only limited access to the system. Our scenario manipulation attack yields success rates exceeding 65%, reaching up to 90%, and does not require any runtime system intrusion. We also assess the robustness of these attacks against defenses, revealing their resilience. Our findings highlight critical security vulnerabilities in embodied LLM systems and emphasize the urgent need for safeguarding these systems to mitigate potential risks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.04035v1">Gamifying XAI: Enhancing AI Explainability for Non-technical Users through LLM-Powered Narrative Gamifications</a></div>
    <div class="paper-meta">
      📅 2024-10-05
    </div>
    <details class="paper-abstract">
      Artificial intelligence (AI) has become tightly integrated into modern technology, yet existing exploratory visualizations for explainable AI (XAI) are primarily designed for users with technical expertise. This leaves everyday users, who also regularly interact with AI systems, with limited resources to explore or understand AI technologies they use. We propose a novel framework that enables non-technical users to collect insights by conversing directly with visualization elements via LLM-powered narrative gamifications. We implemented a prototype that utilizes such gamification to facilitate non-technical users' exploration of AI embedding projections. We conducted a comparative study with 10 participants to assess our prototype quantitatively and qualitatively. Our study results indicate that although our prototype effectively enhances non-technical users' AI/XAI knowledge, and users believe they learn more through the gamification feature, it remains inconclusive whether the gamification itself leads to further improvements in understanding. In addition, opinions among participants regarding the framework's engagement are mixed: some believe it enhances their exploration of the visualizations, while others feel it disrupts their workflow.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.01171v3">Two Tales of Persona in LLMs: A Survey of Role-Playing and Personalization</a></div>
    <div class="paper-meta">
      📅 2024-10-05
      | 💬 Accepted to EMNLP 2024 Findings
    </div>
    <details class="paper-abstract">
      The concept of persona, originally adopted in dialogue literature, has re-surged as a promising framework for tailoring large language models (LLMs) to specific context (e.g., personalized search, LLM-as-a-judge). However, the growing research on leveraging persona in LLMs is relatively disorganized and lacks a systematic taxonomy. To close the gap, we present a comprehensive survey to categorize the current state of the field. We identify two lines of research, namely (1) LLM Role-Playing, where personas are assigned to LLMs, and (2) LLM Personalization, where LLMs take care of user personas. Additionally, we introduce existing methods for LLM personality evaluation. To the best of our knowledge, we present the first survey for role-playing and personalization in LLMs under the unified view of persona. We continuously maintain a paper collection to foster future endeavors: https://github.com/MiuLab/PersonaLLM-Survey
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.04009v1">ASPIRER: Bypassing System Prompts With Permutation-based Backdoors in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-05
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become integral to many applications, with system prompts serving as a key mechanism to regulate model behavior and ensure ethical outputs. In this paper, we introduce a novel backdoor attack that systematically bypasses these system prompts, posing significant risks to the AI supply chain. Under normal conditions, the model adheres strictly to its system prompts. However, our backdoor allows malicious actors to circumvent these safeguards when triggered. Specifically, we explore a scenario where an LLM provider embeds a covert trigger within the base model. A downstream deployer, unaware of the hidden trigger, fine-tunes the model and offers it as a service to users. Malicious actors can purchase the trigger from the provider and use it to exploit the deployed model, disabling system prompts and achieving restricted outcomes. Our attack utilizes a permutation trigger, which activates only when its components are arranged in a precise order, making it computationally challenging to detect or reverse-engineer. We evaluate our approach on five state-of-the-art models, demonstrating that our method achieves an attack success rate (ASR) of up to 99.50% while maintaining a clean accuracy (CACC) of 98.58%, even after defensive fine-tuning. These findings highlight critical vulnerabilities in LLM deployment pipelines and underscore the need for stronger defenses.
    </details>
</div>
