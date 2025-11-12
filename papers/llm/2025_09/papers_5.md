# llm - 2025_09

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- Part 5
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)
- [Part 15](papers_15.md)
- [Part 16](papers_16.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.22991v1">ADAM: A Diverse Archive of Mankind for Evaluating and Enhancing LLMs in Biographical Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-26
    </div>
    <details class="paper-abstract">
      We introduce ADAM (A Diverse Archive of Mankind), a framework for evaluating and improving multimodal large language models (MLLMs) in biographical reasoning. To the best of our knowledge, this is the first work to systematically examine LLM capabilities in biography, a critical yet underexplored dimension of factual knowledge. At its core, AdamDB is a multilingual and multimodal dataset covering over 4 million individuals across geography, time, and profession, while AdamBench provides cognitively structured evaluations based on Bloom's taxonomy, spanning six reasoning levels in both English and native languages. To address hallucinations, particularly for lesser-known individuals, we propose AdamRAG, a retrieval-augmented generation system tailored to biographical contexts. Experiments show that AdamRAG substantially improves open-source models and modestly benefits closed-source ones, with the largest gains on lower-order reasoning. Popularity strongly mediates accuracy, and multimodal input via face images offers smaller, less consistent improvements than retrieval. ADAM establishes the first benchmark and framework for cognitively, culturally, and multimodally grounded biographical evaluation, advancing the development of multilingual, accurate, and hallucination-resistant MLLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.22984v1">Not only a helper, but also a teacher: Interactive LLM Cascade</a></div>
    <div class="paper-meta">
      📅 2025-09-26
      | 💬 29 pages, 4 figures, under review
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) vary widely in their capabilities, with larger models often having better performance but higher cost: choosing an LLM model often involves trading off performance and cost. The LLM Cascade is a paradigm that defers difficult queries from weak/cheap to strong/expensive models. This approach is nonadaptive: the deferral decision is trained offline. When confronted with similar or repeated queries, the LLM Cascade may then repeatedly consult the expensive model and incur higher cost. To improve the cascading efficiency, we propose Inter-Cascade, an online and interactive LLM Cascade that extends the role of strong model from a backup helper to a long-term teacher. In our system, when a strong model resolves a difficult query, it also distills its solution into a generalized, reusable problem-solving strategy that boosts the weak model on subsequent queries. Adding strategies to queries enables the weak model to dynamically improve its performance over time, avoiding computationally and time-intensive fine-tuning. Empirically, compared with standard LLM Cascade baselines across multiple benchmarks, the Inter-Cascade significantly improves the accuracy of the weak model (by up to 33.06 absolute percentage points) and the overall system (by up to 5.53 absolute percentage points), while reducing the calls to strong models (by up to 48.05% relative reduction) and saving the corresponding fees (by up to 49.63% relative reduction). Inter-Cascade demonstrates the effective in-context knowledge transfer between LLMs, and provides a general, scalable framework applicable to both open-source and API-based LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.22979v1">OptiMind: Teaching LLMs to Think Like Optimization Experts</a></div>
    <div class="paper-meta">
      📅 2025-09-26
    </div>
    <details class="paper-abstract">
      Mathematical programming -- the task of expressing operations and decision-making problems in precise mathematical language -- is fundamental across domains, yet remains a skill-intensive process requiring operations research expertise. Recent advances in large language models for complex reasoning have spurred interest in automating this task, translating natural language into executable optimization models. Current approaches, however, achieve limited accuracy, hindered by scarce and noisy training data without leveraging domain knowledge. In this work, we systematically integrate optimization expertise to improve formulation accuracy for mixed-integer linear programming, a key family of mathematical programs. Our approach first cleans training data through class-based error analysis to explicitly prevent common mistakes within each optimization class. We then develop multi-turn inference strategies that guide LLMs with class-specific error summaries and solver feedback, enabling iterative refinement. Experiments across multiple base LLMs demonstrate that combining cleaned data with domain-informed prompting and feedback improves formulation accuracy by 14 percentage points on average, enabling further progress toward robust LLM-assisted optimization formulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.22978v1">Towards Human-interpretable Explanation in Code Clone Detection using LLM-based Post Hoc Explainer</a></div>
    <div class="paper-meta">
      📅 2025-09-26
    </div>
    <details class="paper-abstract">
      Recent studies highlight various machine learning (ML)-based techniques for code clone detection, which can be integrated into developer tools such as static code analysis. With the advancements brought by ML in code understanding, ML-based code clone detectors could accurately identify and classify cloned pairs, especially semantic clones, but often operate as black boxes, providing little insight into the decision-making process. Post hoc explainers, on the other hand, aim to interpret and explain the predictions of these ML models after they are made, offering a way to understand the underlying mechanisms driving the model's decisions. However, current post hoc techniques require white-box access to the ML model or are computationally expensive, indicating a need for advanced post hoc explainers. In this paper, we propose a novel approach that leverages the in-context learning capabilities of large language models to elucidate the predictions made by the ML-based code clone detectors. We perform a study using ChatGPT-4 to explain the code clone results inferred by GraphCodeBERT. We found that our approach is promising as a post hoc explainer by giving the correct explanations up to 98% and offering good explanations 95% of the time. However, the explanations and the code line examples given by the LLM are useful in some cases. We also found that lowering the temperature to zero helps increase the accuracy of the explanation. Lastly, we list the insights that can lead to further improvements in future work. This study paves the way for future studies in using LLMs as a post hoc explainer for various software engineering tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12452v3">Automatically Advancing LLM Expertise in Technology Judgment</a></div>
    <div class="paper-meta">
      📅 2025-09-26
      | 💬 We open-source our patent dataset at https://huggingface.co/datasets/UchiKlab/patent_understanding
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are rapidly becoming core tools for science, engineering, and innovation. Their promise lies not just in remembering facts, but in putting knowledge to work. Despite their impressive ability to answer increasingly difficult questions, it remains unclear whether LLMs truly use their knowledge when confronted with new and challenging tasks. We address this question with a patent classification task that requires deep conceptual understanding: distinguishing objectively different but semantically similar patents. To evaluate this approach, we introduce a challenging new benchmark of 1.3 million post-2015 computer science patent pairs, characterized by dense technical jargon and strategically complex writing. We find that LLMs often fail our benchmark and struggle to distinguish among semantically similar patents. To probe this failure, we introduce a novel framework that decomposes model errors into two sources: missing and unused knowledge. Our approach asks models to generate clarifying questions to improve their understanding, and then compares three settings: raw performance, self-answered questions, and externally supplied answers. This decomposition reveals that LLMs often possess the relevant knowledge internally but fail to deploy it, while a smaller share of errors arises from genuine knowledge gaps. We then ask whether the ability of models to construct a task-specific database of questions and answers differs across models. We find that smaller models generate simpler, broadly transferable questions, while larger models propose more complex but less generalizable ones. This suggests new strategies for combining strengths across models. Our findings highlight a critical limitation of current LLMs and their evaluation: models often know more than they can use. LLM evaluation should shift from recall of static facts to application of dynamic knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.22957v1">Doubly-Robust LLM-as-a-Judge: Externally Valid Estimation with Imperfect Personas</a></div>
    <div class="paper-meta">
      📅 2025-09-26
    </div>
    <details class="paper-abstract">
      As Generative AI (GenAI) systems see growing adoption, a key concern involves the external validity of evaluations, or the extent to which they generalize from lab-based to real-world deployment conditions. Threats to the external validity of GenAI evaluations arise when the source sample of human raters and system outputs used to obtain a system quality estimate differs from the target distribution at deployment time. In this work, we propose a doubly-robust estimation framework designed to address this evaluation sampling bias. Key to our approach is the use of "persona" ratings produced by prompting an LLM evaluator (i.e., an LLM-as-a-judge) to behave as a human rater with specific sociodemographic characteristics. Our doubly-robust framework combines these informative yet imperfect persona ratings with human ratings obtained under evaluation sampling bias to produce statistically valid system quality estimates. In particular, we show that our approach yields valid system quality estimates when either (i) a model trained to predict human ratings using persona ratings and source data observed under sampling bias, or (ii) a reweighting model that corrects for sampling bias is of sufficient quality. We validate our framework theoretically and via a novel Persona Simulation Framework (PSF) designed to systematically manipulate persona quality and the degree of evaluation sampling bias present in source data. Our work provides a principled foundation for combining imperfect persona ratings with human ratings observed under sampling bias to obtain valid system quality estimates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08794v2">One Token to Fool LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-09-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly trusted as automated judges, assisting evaluation and providing reward signals for training other models, particularly in reference-based settings like Reinforcement Learning with Verifiable Rewards (RLVR). However, we uncover a critical vulnerability even in this reference-based paradigm: generative reward models are systematically susceptible to reward hacking. We find that superficial inputs, which we term ''master keys'' such as non-word symbols (e.g., '':'' or ''.'') or generic reasoning openers (e.g., ''Thought process:'' or ''Let's solve this problem step by step.''), can consistently elicit false positive rewards without any substantive reasoning. Our systematic evaluation demonstrates this is a widespread failure affecting a diverse range of models, including leading proprietary systems such as GPT-o1 and Claude-4. These results challenge the assumed robustness of LLM judges and pose a significant threat to their reliability. To address this, we propose a simple yet effective data augmentation strategy using truncated model outputs as adversarial negative examples. The resulting Master Reward Models (Master-RMs) demonstrate state-of-the-art robustness against these ''master key'' attacks while maintaining high performance in standard evaluation settings. We supplement these findings with a comprehensive analysis of the vulnerability across model scales, prompt variations, and common inference-time strategies, offering insights to guide future research on robust LLM evaluation. We release our robust, general-domain reward models and the synthetic training data at https://huggingface.co/sarosavo/Master-RM and https://huggingface.co/datasets/sarosavo/Master-RM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.22940v1">LLMs Behind the Scenes: Enabling Narrative Scene Illustration</a></div>
    <div class="paper-meta">
      📅 2025-09-26
      | 💬 Accepted at EMNLP 2025
    </div>
    <details class="paper-abstract">
      Generative AI has established the opportunity to readily transform content from one medium to another. This capability is especially powerful for storytelling, where visual illustrations can illuminate a story originally expressed in text. In this paper, we focus on the task of narrative scene illustration, which involves automatically generating an image depicting a scene in a story. Motivated by recent progress on text-to-image models, we consider a pipeline that uses LLMs as an interface for prompting text-to-image models to generate scene illustrations given raw story text. We apply variations of this pipeline to a prominent story corpus in order to synthesize illustrations for scenes in these stories. We conduct a human annotation task to obtain pairwise quality judgments for these illustrations. The outcome of this process is the SceneIllustrations dataset, which we release as a new resource for future work on cross-modal narrative transformation. Through our analysis of this dataset and experiments modeling illustration quality, we demonstrate that LLMs can effectively verbalize scene knowledge implicitly evoked by story text. Moreover, this capability is impactful for generating and evaluating illustrations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.22888v1">JE-IRT: A Geometric Lens on LLM Abilities through Joint Embedding Item Response Theory</a></div>
    <div class="paper-meta">
      📅 2025-09-26
      | 💬 22 pages, 10 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Standard LLM evaluation practices compress diverse abilities into single scores, obscuring their inherently multidimensional nature. We present JE-IRT, a geometric item-response framework that embeds both LLMs and questions in a shared space. For question embeddings, the direction encodes semantics and the norm encodes difficulty, while correctness on each question is determined by the geometric interaction between the model and question embeddings. This geometry replaces a global ranking of LLMs with topical specialization and enables smooth variation across related questions. Building on this framework, our experimental results reveal that out-of-distribution behavior can be explained through directional alignment, and that larger norms consistently indicate harder questions. Moreover, JE-IRT naturally supports generalization: once the space is learned, new LLMs are added by fitting a single embedding. The learned space further reveals an LLM-internal taxonomy that only partially aligns with human-defined subject categories. JE-IRT thus establishes a unified and interpretable geometric lens that connects LLM abilities with the structure of questions, offering a distinctive perspective on model evaluation and generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.22887v1">Infusing Theory of Mind into Socially Intelligent LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-09-26
    </div>
    <details class="paper-abstract">
      Theory of Mind (ToM)-an understanding of the mental states of others-is a key aspect of human social intelligence, yet, chatbots and LLM-based social agents do not typically integrate it. In this work, we demonstrate that LLMs that explicitly use ToM get better at dialogue, achieving goals more effectively. After showing that simply prompting models to generate mental states between dialogue turns already provides significant benefit, we further introduce ToMAgent (ToMA), a ToM-focused dialogue agent. ToMA is trained by pairing ToM with dialogue lookahead to produce mental states that are maximally useful for achieving dialogue goals. Experiments on the Sotopia interactive social evaluation benchmark demonstrate the effectiveness of our method over a range of baselines. Comprehensive analysis shows that ToMA exhibits more strategic, goal-oriented reasoning behaviors, which enable long-horizon adaptation, while maintaining better relationships with their partners. Our results suggest a step forward in integrating ToM for building socially intelligent LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.22856v1">The Bias is in the Details: An Assessment of Cognitive Bias in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-26
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) are increasingly embedded in real-world decision-making processes, it becomes crucial to examine the extent to which they exhibit cognitive biases. Extensively studied in the field of psychology, cognitive biases appear as systematic distortions commonly observed in human judgments. This paper presents a large-scale evaluation of eight well-established cognitive biases across 45 LLMs, analyzing over 2.8 million LLM responses generated through controlled prompt variations. To achieve this, we introduce a novel evaluation framework based on multiple-choice tasks, hand-curate a dataset of 220 decision scenarios targeting fundamental cognitive biases in collaboration with psychologists, and propose a scalable approach for generating diverse prompts from human-authored scenario templates. Our analysis shows that LLMs exhibit bias-consistent behavior in 17.8-57.3% of instances across a range of judgment and decision-making contexts targeting anchoring, availability, confirmation, framing, interpretation, overattribution, prospect theory, and representativeness biases. We find that both model size and prompt specificity play a significant role on bias susceptibility as follows: larger size (>32B parameters) can reduce bias in 39.5% of cases, while higher prompt detail reduces most biases by up to 14.9%, except in one case (Overattribution), which is exacerbated by up to 8.8%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.22832v1">Efficient Fine-Grained GPU Performance Modeling for Distributed Deep Learning of LLM</a></div>
    <div class="paper-meta">
      📅 2025-09-26
    </div>
    <details class="paper-abstract">
      Training Large Language Models(LLMs) is one of the most compute-intensive tasks in high-performance computing. Predicting end-to-end training time for multi-billion parameter models distributed across hundreds of GPUs remains challenging due to complex interactions between transformer components, parallelism strategies(data, model, pipeline, tensor), and multi-tier communication. Learned models require costly sampling, while analytical models often struggle with real-world network and hardware complexities. We address this by decomposing LLMs into core computational primitives and modeling them with: (1) operator-level decomposition for fine-grained analysis; (2) lightweight sampling based hardware-aware prediction models for key operations; (3) an end-to-end prediction system integrating these components across complex parallelization strategies. Crucially, our methodology has been validated on two large-scale HPC systems. Our framework achieves low average prediction errors-4.98\% on Perlmutter(A100) and 9.38\% on Vista(GH200)-for models up to 20B parameters across 128 GPUs. Importantly, it runs entirely on CPUs, enabling rapid iteration over hardware configurations and training strategies without costly on-cluster experimentation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.22831v1">Toward a Theory of Generalizability in LLM Mechanistic Interpretability Research</a></div>
    <div class="paper-meta">
      📅 2025-09-26
    </div>
    <details class="paper-abstract">
      Research on Large Language Models (LLMs) increasingly focuses on identifying mechanistic explanations for their behaviors, yet the field lacks clear principles for determining when (and how) findings from one model instance generalize to another. This paper addresses a fundamental epistemological challenge: given a mechanistic claim about a particular model, what justifies extrapolating this finding to other LLMs -- and along which dimensions might such generalizations hold? I propose five potential axes of correspondence along which mechanistic claims might generalize, including: functional (whether they satisfy the same functional criteria), developmental (whether they develop at similar points during pretraining), positional (whether they occupy similar absolute or relative positions), relational (whether they interact with other model components in similar ways), and configurational (whether they correspond to particular regions or structures in weight-space). To empirically validate this framework, I analyze "1-back attention heads" (components attending to previous tokens) across pretraining in random seeds of the Pythia models (14M, 70M, 160M, 410M). The results reveal striking consistency in the developmental trajectories of 1-back attention across models, while positional consistency is more limited. Moreover, seeds of larger models systematically show earlier onsets, steeper slopes, and higher peaks of 1-back attention. I also address possible objections to the arguments and proposals outlined here. Finally, I conclude by arguing that progress on the generalizability of mechanistic interpretability research will consist in mapping constitutive design properties of LLMs to their emergent behaviors and mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20810v1">Enrich-on-Graph: Query-Graph Alignment for Complex Reasoning with LLM Enriching</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit strong reasoning capabilities in complex tasks. However, they still struggle with hallucinations and factual errors in knowledge-intensive scenarios like knowledge graph question answering (KGQA). We attribute this to the semantic gap between structured knowledge graphs (KGs) and unstructured queries, caused by inherent differences in their focuses and structures. Existing methods usually employ resource-intensive, non-scalable workflows reasoning on vanilla KGs, but overlook this gap. To address this challenge, we propose a flexible framework, Enrich-on-Graph (EoG), which leverages LLMs' prior knowledge to enrich KGs, bridge the semantic gap between graphs and queries. EoG enables efficient evidence extraction from KGs for precise and robust reasoning, while ensuring low computational costs, scalability, and adaptability across different methods. Furthermore, we propose three graph quality evaluation metrics to analyze query-graph alignment in KGQA task, supported by theoretical validation of our optimization objectives. Extensive experiments on two KGQA benchmark datasets indicate that EoG can effectively generate high-quality KGs and achieve the state-of-the-art performance. Our code and data are available at https://github.com/zjukg/Enrich-on-Graph.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20802v1">SPADE: Structured Pruning and Adaptive Distillation for Efficient LLM-TTS</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 Submitted to ICASSP 2026
    </div>
    <details class="paper-abstract">
      The goal of this paper is to introduce SPADE, a framework for Structured Pruning and Adaptive Distillation for Efficient Large Language Model-based text-to-speech (LLM-TTS). Recent LLM-TTS systems achieve strong controllability and zero-shot generalization, but their large parameter counts and high latency limit real-world deployment. SPADE addresses this by combining (i) a pruning step guided by a word-error-rate-based layer importance index to remove non-essential Transformer layers, with (ii) multi-level knowledge distillation to restore autoregressive coherence. On zero-shot benchmarks, SPADE preserves near-parity perceptual quality while halving Transformer depth, reducing VRAM usage by up to 20%, and achieving up to 1.7x faster real-time factor with less than 5% of the original training data. These results show that compact LLM-TTS models can maintain naturalness and speaker similarity while enabling practical real-time speech generation. Audio samples are available at https://mm.kaist.ac.kr/projects/SPADE/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20798v1">LogReasoner: Empowering LLMs with Expert-like Coarse-to-Fine Reasoning for Log Analysis Tasks</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 under review
    </div>
    <details class="paper-abstract">
      Log analysis is crucial for monitoring system health and diagnosing failures in complex systems. Recent advances in large language models (LLMs) offer new opportunities for automated log analysis, leveraging their reasoning capabilities to perform tasks such as anomaly detection and failure prediction. However, general-purpose LLMs struggle to formulate structured reasoning workflows that align with expert cognition and deliver precise details of reasoning steps. To address these challenges, we propose LogReasoner, a coarse-to-fine reasoning enhancement framework designed to enable LLMs to reason log analysis tasks like experts. LogReasoner consists of two stages: (1) coarse-grained enhancement of expert thinking, where high-level expert thoughts are constructed from collected troubleshooting flowcharts and existing tasks to enable LLMs to formulate structured reasoning workflows and (2) fine-grained enhancement of specific steps, where we first fine-tune the LLM with task-specific stepwise solutions to enhance the LLM for instantiated reasoning, then employ the preference learning to calibrate the LLM's reasoning details from its mistakes, further strengthen the LLM's analytical granularity and correctness. We evaluate LogReasoner on four distinct log analysis tasks using open-source LLMs such as Qwen-2.5 and Llama-3. Experimental results show that LogReasoner significantly outperforms existing LLMs, achieving state-of-the-art performance and demonstrating its effectiveness in enhancing the reasoning capabilities of LLMs for log analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12010v4">Bias Similarity Measurement: A Black-Box Audit of Fairness Across LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 Code available at https://github.com/HyejunJeong/bias_llm
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) reproduce social biases, yet prevailing evaluations score models in isolation, obscuring how biases persist across families and releases. We introduce Bias Similarity Measurement (BSM), which treats fairness as a relational property between models, unifying scalar, distributional, behavioral, and representational signals into a single similarity space. Evaluating 30 LLMs on 1M+ prompts, we find that instruction tuning primarily enforces abstention rather than altering internal representations; small models gain little accuracy and can become less fair under forced choice; and open-weight models can match or exceed proprietary systems. Family signatures diverge: Gemma favors refusal, LLaMA 3.1 approaches neutrality with fewer refusals, and converges toward abstention-heavy behavior overall. Counterintuitively, Gemma 3 Instruct matches GPT-4-level fairness at far lower cost, whereas Gemini's heavy abstention suppresses utility. Beyond these findings, BSM offers an auditing workflow for procurement, regression testing, and lineage screening, and extends naturally to code and multilingual settings. Our results reframe fairness not as isolated scores but as comparative bias similarity, enabling systematic auditing of LLM ecosystems. Code available at https://github.com/HyejunJeong/bias_llm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20768v1">Measuring LLM Sensitivity in Transformer-based Tabular Data Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 12 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Synthetic tabular data is used for privacy-preserving data sharing and data-driven model development. Its effectiveness, however, depends heavily on the used Tabular Data Synthesis (TDS) tool. Recent studies have shown that Transformer-based models outperform other state-of-the-art models such as Generative Adversarial Networks (GANs) and Diffusion models in terms of data quality. However, Transformer-based models also come with high computational costs, making them sometimes unfeasible for end users with prosumer hardware. This study presents a sensitivity assessment on how the choice of hyperparameters, such as number of layers or hidden dimension affects the quality of the resultant synthetic data and the computational performance. It is performed across two tools, GReaT and REaLTabFormer, evaluating 10 model setups that vary in architecture type and depth. We assess the sensitivity on three dimensions: runtime, machine learning (ML) utility, and similarity to real data distributions. Experiments were conducted on four real-world datasets. Our findings reveal that runtime is proportional to the number of hyperparameters, with shallower configurations completing faster. GReaT consistently achieves lower runtimes than REaLTabFormer, and only on the largest dataset they have comparable runtime. For small datasets, both tools achieve synthetic data with high utility and optimal similarity, but on larger datasets only REaLTabFormer sustains strong utility and similarity. As a result, REaLTabFormer with lightweight LLMs provides the best balance, since it preserves data quality while reducing computational requirements. Nonetheless, its runtime remains higher than that of GReaT and other TDS tools, suggesting that efficiency gains are possible but only up to a certain level.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05657v3">LM-Searcher: Cross-domain Neural Architecture Search with LLMs via Unified Numerical Encoding</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 EMNLP 2025 Main
    </div>
    <details class="paper-abstract">
      Recent progress in Large Language Models (LLMs) has opened new avenues for solving complex optimization problems, including Neural Architecture Search (NAS). However, existing LLM-driven NAS approaches rely heavily on prompt engineering and domain-specific tuning, limiting their practicality and scalability across diverse tasks. In this work, we propose LM-Searcher, a novel framework that leverages LLMs for cross-domain neural architecture optimization without the need for extensive domain-specific adaptation. Central to our approach is NCode, a universal numerical string representation for neural architectures, which enables cross-domain architecture encoding and search. We also reformulate the NAS problem as a ranking task, training LLMs to select high-performing architectures from candidate pools using instruction-tuning samples derived from a novel pruning-based subspace sampling strategy. Our curated dataset, encompassing a wide range of architecture-performance pairs, encourages robust and transferable learning. Comprehensive experiments demonstrate that LM-Searcher achieves competitive performance in both in-domain (e.g., CNNs for image classification) and out-of-domain (e.g., LoRA configurations for segmentation and generation) tasks, establishing a new paradigm for flexible and generalizable LLM-based architecture search. The datasets and models will be released at https://github.com/Ashone3/LM-Searcher.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17075v2">LoRA is All You Need for Safety Alignment of Reasoning LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Reasoning LLMs have demonstrated remarkable breakthroughs in solving complex problems that were previously out of reach. To ensure LLMs do not assist with harmful requests, safety alignment fine-tuning is necessary in the post-training phase. However, safety alignment fine-tuning has recently been shown to significantly degrade reasoning abilities, a phenomenon known as the "Safety Tax". In this work, we show that using LoRA for SFT on refusal datasets effectively aligns the model for safety without harming its reasoning capabilities. This is because restricting the safety weight updates to a low-rank space minimizes the interference with the reasoning weights. Our extensive experiments across four benchmarks covering math, science, and coding show that this approach produces highly safe LLMs--with safety levels comparable to full-model fine-tuning--without compromising their reasoning abilities. Our ablation studies further identify three key factors in LoRA: (1) rank-$1$ updates are sufficient to achieve the best reasoning and safety performance, (2) the up projection layers are the most critical modules, with LoRA applied to them alone achieving even better results, and (3) middle layers are more effective than early or late layers. Together, these findings show that strong safety and reasoning can be achieved at minimal computational cost when updates are applied in the right places. Additionally, we observe that LoRA induces weight updates with smaller overlap with the initial weights compared to full-model fine-tuning. Finally, while our attempts to further reduce this overlap yield only modest improvements on some tasks, they highlight the potential of developing methods that more reliably optimize the reasoning-safety tradeoff.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20758v1">SFT Doesn't Always Hurt General Capabilities: Revisiting Domain-Specific Fine-Tuning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Supervised Fine-Tuning (SFT) on domain-specific datasets is a common approach to adapt Large Language Models (LLMs) to specialized tasks but is often believed to degrade their general capabilities. In this work, we revisit this trade-off and present both empirical and theoretical insights. First, we show that SFT does not always hurt: using a smaller learning rate can substantially mitigate general performance degradation while preserving comparable target-domain performance. We then provide a theoretical analysis that explains these phenomena and further motivates a new method, Token-Adaptive Loss Reweighting (TALR). Building on this, and recognizing that smaller learning rates alone do not fully eliminate general-performance degradation in all cases, we evaluate a range of strategies for reducing general capability loss, including L2 regularization, LoRA, model averaging, FLOW, and our proposed TALR. Experimental results demonstrate that while no method completely eliminates the trade-off, TALR consistently outperforms these baselines in balancing domain-specific gains and general capabilities. Finally, we distill our findings into practical guidelines for adapting LLMs to new domains: (i) using a small learning rate to achieve a favorable trade-off, and (ii) when a stronger balance is further desired, adopt TALR as an effective strategy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.15141v2">Text-Augmented Multimodal LLMs for Chemical Reaction Condition Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Identifying reaction conditions that are broadly applicable across diverse substrates is a longstanding challenge in chemical and pharmaceutical research. While many methods are available to generate conditions with acceptable performance, a universal approach for reliably discovering effective conditions during reaction exploration is rare. Consequently, current reaction optimization processes are often labor-intensive, time-consuming, and costly, relying heavily on trial-and-error experimentation. Nowadays, large language models (LLMs) are capable of tackling chemistry-related problems, such as molecule design and chemical reasoning tasks. Here, we report the design, implementation and application of Chemma-RC, a text-augmented multimodal LLM to identify effective conditions through task-specific dialogue and condition generation. Chemma-RC learns a unified representation of chemical reactions by aligning multiple modalities-including text corpus, reaction SMILES, and reaction graphs-within a shared embedding module. Performance benchmarking on datasets showed high precision in identifying optimal conditions, with up to 17% improvement over the current state-of-the-art methods. A palladium-catalysed imidazole C-H arylation reaction was investigated experimentally to evaluate the functionalities of the Chemma-RC in practice. Our findings suggest that Chemma-RC holds significant potential to accelerate high-throughput condition screening in chemical synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.10880v2">Searching for Privacy Risks in LLM Agents via Simulation</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      The widespread deployment of LLM-based agents is likely to introduce a critical privacy threat: malicious agents that proactively engage others in multi-turn interactions to extract sensitive information. However, the evolving nature of such dynamic dialogues makes it challenging to anticipate emerging vulnerabilities and design effective defenses. To tackle this problem, we present a search-based framework that alternates between improving attack and defense strategies through the simulation of privacy-critical agent interactions. Specifically, we employ LLMs as optimizers to analyze simulation trajectories and iteratively propose new agent instructions. To explore the strategy space more efficiently, we further utilize parallel search with multiple threads and cross-thread propagation. Through this process, we find that attack strategies escalate from direct requests to sophisticated tactics, such as impersonation and consent forgery, while defenses evolve from simple rule-based constraints to robust identity-verification state machines. The discovered attacks and defenses transfer across diverse scenarios and backbone models, demonstrating strong practical utility for building privacy-aware agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20067v2">MACD: Multi-Agent Clinical Diagnosis with Self-Learned Knowledge for LLM</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated notable potential in medical applications, yet they face substantial challenges in handling complex real-world clinical diagnoses using conventional prompting methods. Current prompt engineering and multi-agent approaches typically optimize isolated inferences, neglecting the accumulation of reusable clinical experience. To address this, this study proposes a novel Multi-Agent Clinical Diagnosis (MACD) framework, which allows LLMs to self-learn clinical knowledge via a multi-agent pipeline that summarizes, refines, and applies diagnostic insights. It mirrors how physicians develop expertise through experience, enabling more focused and accurate diagnosis on key disease-specific cues. We further extend it to a MACD-human collaborative workflow, where multiple LLM-based diagnostician agents engage in iterative consultations, supported by an evaluator agent and human oversight for cases where agreement is not reached. Evaluated on 4,390 real-world patient cases across seven diseases using diverse open-source LLMs (Llama-3.1 8B/70B, DeepSeek-R1-Distill-Llama 70B), MACD significantly improves primary diagnostic accuracy, outperforming established clinical guidelines with gains up to 22.3% (MACD). On the subset of the data, it achieves performance on par with or exceeding that of human physicians (up to 16% improvement over physicians-only diagnosis). Additionally, on the MACD-human workflow, it achieves an 18.6% improvement compared to physicians-only diagnosis. Moreover, self-learned knowledge exhibits strong cross-model stability, transferability, and model-specific personalization, while the system can generate traceable rationales, enhancing explainability. Consequently, this work presents a scalable self-learning paradigm for LLM-assisted diagnosis, bridging the gap between the intrinsic knowledge of LLMs and real-world clinical practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11343v2">From Replication to Redesign: Exploring Pairwise Comparisons for LLM-Based Peer Review</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      The advent of large language models (LLMs) offers unprecedented opportunities to reimagine peer review beyond the constraints of traditional workflows. Despite these opportunities, prior efforts have largely focused on replicating traditional review workflows with LLMs serving as direct substitutes for human reviewers, while limited attention has been given to exploring new paradigms that fundamentally rethink how LLMs can participate in the academic review process. In this paper, we introduce and explore a novel mechanism that employs LLM agents to perform pairwise comparisons among manuscripts instead of individual scoring. By aggregating outcomes from substantial pairwise evaluations, this approach enables a more accurate and robust measure of relative manuscript quality. Our experiments demonstrate that this comparative approach significantly outperforms traditional rating-based methods in identifying high-impact papers. However, our analysis also reveals emergent biases in the selection process, notably a reduced novelty in research topics and an increased institutional imbalance. These findings highlight both the transformative potential of rethinking peer review with LLMs and critical challenges that future systems must address to ensure equity and diversity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14096v3">VideoPASTA: 7K Preference Pairs That Matter for Video-LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 EMNLP 2025 (Main)
    </div>
    <details class="paper-abstract">
      Video-language models (Video-LLMs) excel at understanding video content but struggle with spatial relationships, temporal ordering, and cross-frame continuity. To address these limitations, we introduce VideoPASTA (Preference Alignment with Spatio-Temporal-Cross Frame Adversaries), a framework that enhances Video-LLMs through targeted preference optimization. VideoPASTA trains models to distinguish accurate video representations from carefully crafted adversarial examples that deliberately violate spatial, temporal, or cross-frame relationships. With only 7,020 preference pairs and Direct Preference Optimization, VideoPASTA enables models to learn robust representations that capture fine-grained spatial details and long-range temporal dynamics. Experiments demonstrate that VideoPASTA is model agnostic and significantly improves performance, for example, achieving gains of up to +3.8 percentage points on LongVideoBench, +4.1 on VideoMME, and +4.0 on MVBench, when applied to various state-of-the-art Video-LLMs. These results demonstrate that targeted alignment, rather than massive pretraining or architectural modifications, effectively addresses core video-language challenges. Notably, VideoPASTA achieves these improvements without any human annotation or captioning, relying solely on 32-frame sampling. This efficiency makes our approach a scalable plug-and-play solution that seamlessly integrates with existing models while preserving their original capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20702v1">Incorporating LLM Embeddings for Variation Across the Human Genome</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Recent advances in large language model (LLM) embeddings have enabled powerful representations for biological data, but most applications to date focus only on gene-level information. We present one of the first systematic frameworks to generate variant-level embeddings across the entire human genome. Using curated annotations from FAVOR, ClinVar, and the GWAS Catalog, we constructed semantic text descriptions for 8.9 billion possible variants and generated embeddings at three scales: 1.5 million HapMap3+MEGA variants, ~90 million imputed UK Biobank variants, and ~9 billion all possible variants. Embeddings were produced with both OpenAI's text-embedding-3-large and the open-source Qwen3-Embedding-0.6B models. Baseline experiments demonstrate high predictive accuracy for variant properties, validating the embeddings as structured representations of genomic variation. We outline two downstream applications: embedding-informed hypothesis testing by extending the Frequentist And Bayesian framework to genome-wide association studies, and embedding-augmented genetic risk prediction that enhances standard polygenic risk scores. These resources, publicly available on Hugging Face, provide a foundation for advancing large-scale genomic discovery and precision medicine.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18991v5">Inverse Reinforcement Learning with Dynamic Reward Scaling for LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 The first three authors contributed equally to this work
    </div>
    <details class="paper-abstract">
      Alignment is vital for safely deploying large language models (LLMs). Existing techniques are either reward-based (train a reward model on preference pairs and optimize with reinforcement learning) or reward-free (directly fine-tune on ranked outputs). Recent research shows that well-tuned reward-based pipelines remain robust, and single-response demonstrations can outperform pairwise preference data. However, two challenges persist: (1) imbalanced safety datasets that overrepresent common hazards while neglecting long-tail threats; and (2) static reward models that ignore task difficulty, limiting optimization efficiency and attainable gains. We propose DR-IRL (Dynamically adjusting Rewards through Inverse Reinforcement Learning). We first train category-specific reward models using a balanced safety dataset covering seven harmful categories via IRL. Then we enhance Group Relative Policy Optimization (GRPO) by introducing dynamic reward scaling--adjusting rewards by task difficulty--data-level hardness by text encoder cosine similarity, model-level responsiveness by reward gaps. Extensive experiments across various benchmarks and LLMs demonstrate that DR-IRL outperforms all baseline methods in safety alignment while maintaining usefulness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.18896v2">ReasonFlux-PRM: Trajectory-Aware PRMs for Long Chain-of-Thought Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 Accepted by NeurIPS 2025. Project: https://github.com/Gen-Verse/ReasonFlux
    </div>
    <details class="paper-abstract">
      Process Reward Models (PRMs) have recently emerged as a powerful framework for supervising intermediate reasoning steps in large language models (LLMs). Previous PRMs are primarily trained on model final output responses and struggle to evaluate intermediate thinking trajectories robustly, especially in the emerging setting of trajectory-response outputs generated by frontier reasoning models like Deepseek-R1. In this work, we introduce ReasonFlux-PRM, a novel trajectory-aware PRM explicitly designed to evaluate the trajectory-response type of reasoning traces. ReasonFlux-PRM incorporates both step-level and trajectory-level supervision, enabling fine-grained reward assignment aligned with structured chain-of-thought data. We adapt ReasonFlux-PRM to support reward supervision under both offline and online settings, including (i) selecting high-quality model distillation data for downstream supervised fine-tuning of smaller models, (ii) providing dense process-level rewards for policy optimization during reinforcement learning, and (iii) enabling reward-guided Best-of-N test-time scaling. Empirical results on challenging downstream benchmarks such as AIME, MATH500, and GPQA-Diamond demonstrate that ReasonFlux-PRM-7B selects higher quality data than strong PRMs (e.g., Qwen2.5-Math-PRM-72B) and human-curated baselines. Furthermore, our derived ReasonFlux-PRM-7B yields consistent performance improvements, achieving average gains of 12.1% in supervised fine-tuning, 4.5% in reinforcement learning, and 6.3% in test-time scaling. We also release our efficient ReasonFlux-PRM-1.5B for resource-constrained applications and edge deployment. Project: https://github.com/Gen-Verse/ReasonFlux
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20680v1">Can Federated Learning Safeguard Private Data in LLM Training? Vulnerabilities, Attacks, and Defense Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 28 pages, 32 figures, accepted to the Findings of EMNLP 2025
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) with local data is a widely adopted approach for organizations seeking to adapt LLMs to their specific domains. Given the shared characteristics in data across different organizations, the idea of collaboratively fine-tuning an LLM using data from multiple sources presents an appealing opportunity. However, organizations are often reluctant to share local data, making centralized fine-tuning impractical. Federated learning (FL), a privacy-preserving framework, enables clients to retain local data while sharing only model parameters for collaborative training, offering a potential solution. While fine-tuning LLMs on centralized datasets risks data leakage through next-token prediction, the iterative aggregation process in FL results in a global model that encapsulates generalized knowledge, which some believe protects client privacy. In this paper, however, we present contradictory findings through extensive experiments. We show that attackers can still extract training data from the global model, even using straightforward generation methods, with leakage increasing as the model size grows. Moreover, we introduce an enhanced attack strategy tailored to FL, which tracks global model updates during training to intensify privacy leakage. To mitigate these risks, we evaluate privacy-preserving techniques in FL, including differential privacy, regularization-constrained updates and adopting LLMs with safety alignment. Our results provide valuable insights and practical guidelines for reducing privacy risks when training LLMs with FL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20641v1">Investigating Modality Contribution in Audio LLMs for Music</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Audio Large Language Models (Audio LLMs) enable human-like conversation about music, yet it is unclear if they are truly listening to the audio or just using textual reasoning, as recent benchmarks suggest. This paper investigates this issue by quantifying the contribution of each modality to a model's output. We adapt the MM-SHAP framework, a performance-agnostic score based on Shapley values that quantifies the relative contribution of each modality to a model's prediction. We evaluate two models on the MuChoMusic benchmark and find that the model with higher accuracy relies more on text to answer questions, but further inspection shows that even if the overall audio contribution is low, models can successfully localize key sound events, suggesting that audio is not entirely ignored. Our study is the first application of MM-SHAP to Audio LLMs and we hope it will serve as a foundational step for future research in explainable AI and audio.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20634v1">Recidivism and Peer Influence with LLM Text Embeddings in Low Security Correctional Facilities</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      We find AI embeddings obtained using a pre-trained transformer-based Large Language Model (LLM) of 80,000-120,000 written affirmations and correction exchanges among residents in low-security correctional facilities to be highly predictive of recidivism. The prediction accuracy is 30\% higher with embedding vectors than with only pre-entry covariates. However, since the text embedding vectors are high-dimensional, we perform Zero-Shot classification of these texts to a low-dimensional vector of user-defined classes to aid interpretation while retaining the predictive power. To shed light on the social dynamics inside the correctional facilities, we estimate peer effects in these LLM-generated numerical representations of language with a multivariate peer effect model, adjusting for network endogeneity. We develop new methodology and theory for peer effect estimation that accommodate sparse networks, multivariate latent variables, and correlated multivariate outcomes. With these new methods, we find significant peer effects in language usage for interaction and feedback.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21685v1">FlexMind: Supporting Deeper Creative Thinking with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Effective ideation requires both broad exploration of diverse ideas and deep evaluation of their potential. Generative AI can support such processes, but current tools typically emphasize either generating many ideas or supporting in-depth consideration of a few, lacking support for both. Research also highlights risks of over-reliance on LLMs, including shallow exploration and negative creative outcomes. We present FlexMind, an AI-augmented system that scaffolds iterative exploration of ideas, tradeoffs, and mitigations. FlexMind exposes users to a broad set of ideas while enabling a lightweight transition into deeper engagement. In a study comparing ideation with FlexMind to ChatGPT, participants generated higher-quality ideas with FlexMind, due to both broader exposure and deeper engagement with tradeoffs. By scaffolding ideation across breadth, depth, and reflective evaluation, FlexMind empowers users to surface ideas that might otherwise go unnoticed or be prematurely discarded.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.20016v4">Vulnerability of LLMs to Vertically Aligned Text Manipulations</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 Accepted to ACL 2025 (Main)
    </div>
    <details class="paper-abstract">
      Vertical text input is commonly encountered in various real-world applications, such as mathematical computations and word-based Sudoku puzzles. While current large language models (LLMs) have excelled in natural language tasks, they remain vulnerable to variations in text formatting. Recent research demonstrates that modifying input formats, such as vertically aligning words for encoder-based models, can substantially lower accuracy in text classification tasks. While easily understood by humans, these inputs can significantly mislead models, posing a potential risk of bypassing detection in real-world scenarios involving harmful or sensitive information. With the expanding application of LLMs, a crucial question arises: Do decoder-based LLMs exhibit similar vulnerabilities to vertically formatted text input? In this paper, we investigate the impact of vertical text input on the performance of various LLMs across multiple text classification datasets and analyze the underlying causes. Our findings are as follows: (i) Vertical text input significantly degrades the accuracy of LLMs in text classification tasks. (ii) Chain-of-Thought (CoT) reasoning does not help LLMs recognize vertical input or mitigate its vulnerability, but few-shot learning with careful analysis does. (iii) We explore the underlying cause of the vulnerability by analyzing the inherent issues in tokenization and attention matrices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16456v2">PhyMAGIC: Physical Motion-Aware Generative Inference with Confidence-guided LLM</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Recent advances in 3D content generation have amplified demand for dynamic models that are both visually realistic and physically consistent. However, state-of-the-art video diffusion models frequently produce implausible results such as momentum violations and object interpenetrations. Existing physics-aware approaches often rely on task-specific fine-tuning or supervised data, which limits their scalability and applicability. To address the challenge, we present PhyMAGIC, a training-free framework that generates physically consistent motion from a single image. PhyMAGIC integrates a pre-trained image-to-video diffusion model, confidence-guided reasoning via LLMs, and a differentiable physics simulator to produce 3D assets ready for downstream physical simulation without fine-tuning or manual supervision. By iteratively refining motion prompts using LLM-derived confidence scores and leveraging simulation feedback, PhyMAGIC steers generation toward physically consistent dynamics. Comprehensive experiments demonstrate that PhyMAGIC outperforms state-of-the-art video generators and physics-aware baselines, enhancing physical property inference and motion-text alignment while maintaining visual fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21629v1">InvBench: Can LLMs Accelerate Program Verification with Invariant Synthesis?</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Program verification relies on loop invariants, yet automatically discovering strong invariants remains a long-standing challenge. We introduce a principled framework for evaluating LLMs on invariant synthesis. Our approach uses a verifier-based decision procedure with a formal soundness guarantee and assesses not only correctness but also the speedup that invariants provide in verification. We evaluate 7 state-of-the-art LLMs, and existing LLM-based verifiers against the traditional solver UAutomizer. While LLM-based verifiers represent a promising direction, they do not yet offer a significant advantage over UAutomizer. Model capability also proves critical, as shown by sharp differences in speedups across models, and our benchmark remains an open challenge for current LLMs. Finally, we show that supervised fine-tuning and Best-of-N sampling can improve performance: fine-tuning on 3589 instances raises the percentage of speedup cases for Qwen3-Coder-480B from 8% to 29.2%, and Best-of-N sampling with N=16 improves Claude-sonnet-4 from 8.8% to 22.1%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22168v2">Persona-Augmented Benchmarking: Evaluating LLMs Across Diverse Writing Styles</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 Accepted to EMNLP 2025
    </div>
    <details class="paper-abstract">
      Current benchmarks for evaluating Large Language Models (LLMs) often do not exhibit enough writing style diversity, with many adhering primarily to standardized conventions. Such benchmarks do not fully capture the rich variety of communication patterns exhibited by humans. Thus, it is possible that LLMs, which are optimized on these benchmarks, may demonstrate brittle performance when faced with "non-standard" input. In this work, we test this hypothesis by rewriting evaluation prompts using persona-based LLM prompting, a low-cost method to emulate diverse writing styles. Our results show that, even with identical semantic content, variations in writing style and prompt formatting significantly impact the estimated performance of the LLM under evaluation. Notably, we identify distinct writing styles that consistently trigger either low or high performance across a range of models and tasks, irrespective of model family, size, and recency. Our work offers a scalable approach to augment existing benchmarks, improving the external validity of the assessments they provide for measuring LLM performance across linguistic variations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17117v5">From Tokens to Thoughts: How LLMs and Humans Trade Compression for Meaning</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Humans organize knowledge into compact categories that balance compression with semantic meaning preservation. Large Language Models (LLMs) demonstrate striking linguistic abilities, yet whether they achieve this same balance remains unclear. We apply the Information Bottleneck principle to quantitatively compare how LLMs and humans navigate this compression-meaning trade-off. Analyzing embeddings from 40+ LLMs against classic human categorization benchmarks, we uncover three key findings. First, LLMs broadly align with human categories but miss fine-grained semantic distinctions crucial for human understanding. Second, LLMs demonstrate aggressive statistical compression, achieving ``optimal'' information-theoretic efficiency, while humans prioritize contextual richness and adaptive flexibility. Third, encoder models surprisingly outperform decoder models in human alignment, suggesting that generation and understanding rely on distinct mechanisms in current architectures. In addition, training dynamics analysis reveals that conceptual structure develops in distinct phases: rapid initial formation followed by architectural reorganization, with semantic processing migrating from deeper to mid-network layers as models discover more efficient encoding. These divergent strategies, where LLMs optimize for compression and humans for adaptive utility, reveal fundamental differences between artificial and biological intelligence, guiding development toward more human-aligned AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.10659v6">Network Formation and Dynamics Among Multi-LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 Accepted at PNAS Nexus
    </div>
    <details class="paper-abstract">
      Social networks profoundly influence how humans form opinions, exchange information, and organize collectively. As large language models (LLMs) are increasingly embedded into social and professional environments, it is critical to understand whether their interactions approximate human-like network dynamics. We develop a framework to study the network formation behaviors of multiple LLM agents and benchmark them against human decisions. Across synthetic and real-world settings, including friendship, telecommunication, and employment networks, we find that LLMs consistently reproduce fundamental micro-level principles such as preferential attachment, triadic closure, and homophily, as well as macro-level properties including community structure and small-world effects. Importantly, the relative emphasis of these principles adapts to context: for example, LLMs favor homophily in friendship networks but heterophily in organizational settings, mirroring patterns of social mobility. A controlled human-subject survey confirms strong alignment between LLMs and human participants in link-formation decisions. These results establish that LLMs can serve as powerful tools for social simulation and synthetic data generation, while also raising critical questions about bias, fairness, and the design of AI systems that participate in human networks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21602v1">Real-Time Indoor Object SLAM with LLM-Enhanced Priors</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Object-level Simultaneous Localization and Mapping (SLAM), which incorporates semantic information for high-level scene understanding, faces challenges of under-constrained optimization due to sparse observations. Prior work has introduced additional constraints using commonsense knowledge, but obtaining such priors has traditionally been labor-intensive and lacks generalizability across diverse object categories. We address this limitation by leveraging large language models (LLMs) to provide commonsense knowledge of object geometric attributes, specifically size and orientation, as prior factors in a graph-based SLAM framework. These priors are particularly beneficial during the initial phase when object observations are limited. We implement a complete pipeline integrating these priors, achieving robust data association on sparse object-level features and enabling real-time object SLAM. Our system, evaluated on the TUM RGB-D and 3RScan datasets, improves mapping accuracy by 36.8\% over the latest baseline. Additionally, we present real-world experiments in the supplementary video, demonstrating its real-time performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21577v1">"Be My Cheese?": Assessing Cultural Nuance in Multilingual LLM Translations</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      This pilot study explores the localisation capabilities of state-of-the-art multilingual AI models when translating figurative language, such as idioms and puns, from English into a diverse range of global languages. It expands on existing LLM translation research and industry benchmarks, which emphasise grammatical accuracy and token-level correctness, by focusing on cultural appropriateness and overall localisation quality - critical factors for real-world applications like marketing and e-commerce. To investigate these challenges, this project evaluated a sample of 87 LLM-generated translations of e-commerce marketing emails across 24 regional dialects of 20 languages. Human reviewers fluent in each target language provided quantitative ratings and qualitative feedback on faithfulness to the original's tone, meaning, and intended audience. Findings suggest that, while leading models generally produce grammatically correct translations, culturally nuanced language remains a clear area for improvement, often requiring substantial human refinement. Notably, even high-resource global languages, despite topping industry benchmark leaderboards, frequently mistranslated figurative expressions and wordplay. This work challenges the assumption that data volume is the most reliable predictor of machine translation quality and introduces cultural appropriateness as a key determinant of multilingual LLM performance - an area currently underexplored in existing academic and industry benchmarks. As a proof of concept, this pilot highlights limitations of current multilingual AI systems for real-world localisation use cases. Results of this pilot support the opportunity for expanded research at greater scale to deliver generalisable insights and inform deployment of reliable machine translation workflows in culturally diverse contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13319v2">Elucidating Mechanisms of Demographic Bias in LLMs for Healthcare</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 Accepted in EMNLP (Findings)
    </div>
    <details class="paper-abstract">
      We know from prior work that LLMs encode social biases, and that this manifests in clinical tasks. In this work we adopt tools from mechanistic interpretability to unveil sociodemographic representations and biases within LLMs in the context of healthcare. Specifically, we ask: Can we identify activations within LLMs that encode sociodemographic information (e.g., gender, race)? We find that gender information is highly localized in MLP layers and can be reliably manipulated at inference time via patching. Such interventions can surgically alter generated clinical vignettes for specific conditions, and also influence downstream clinical predictions which correlate with gender, e.g., patient risk of depression. We find that representation of patient race is somewhat more distributed, but can also be intervened upon, to a degree. To our knowledge, this is the first application of mechanistic interpretability methods to LLMs for healthcare.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21559v1">X-CoT: Explainable Text-to-Video Retrieval via LLM-based Chain-of-Thought Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 12 pages, 7 figures. Accepted at EMNLP 2025 (Main Conference)
    </div>
    <details class="paper-abstract">
      Prevalent text-to-video retrieval systems mainly adopt embedding models for feature extraction and compute cosine similarities for ranking. However, this design presents two limitations. Low-quality text-video data pairs could compromise the retrieval, yet are hard to identify and examine. Cosine similarity alone provides no explanation for the ranking results, limiting the interpretability. We ask that can we interpret the ranking results, so as to assess the retrieval models and examine the text-video data? This work proposes X-CoT, an explainable retrieval framework upon LLM CoT reasoning in place of the embedding model-based similarity ranking. We first expand the existing benchmarks with additional video annotations to support semantic understanding and reduce data bias. We also devise a retrieval CoT consisting of pairwise comparison steps, yielding detailed reasoning and complete ranking. X-CoT empirically improves the retrieval performance and produces detailed rationales. It also facilitates the model behavior and data quality analysis. Code and data are available at: https://github.com/PrasannaPulakurthi/X-CoT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21557v1">Generation-Time vs. Post-hoc Citation: A Holistic Evaluation of LLM Attribution</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 Accepted at NeurIPS 2025 LLM Evaluation Workshop
    </div>
    <details class="paper-abstract">
      Trustworthy Large Language Models (LLMs) must cite human-verifiable sources in high-stakes domains such as healthcare, law, academia, and finance, where even small errors can have severe consequences. Practitioners and researchers face a choice: let models generate citations during decoding, or let models draft answers first and then attach appropriate citations. To clarify this choice, we introduce two paradigms: Generation-Time Citation (G-Cite), which produces the answer and citations in one pass, and Post-hoc Citation (P-Cite), which adds or verifies citations after drafting. We conduct a comprehensive evaluation from zero-shot to advanced retrieval-augmented methods across four popular attribution datasets and provide evidence-based recommendations that weigh trade-offs across use cases. Our results show a consistent trade-off between coverage and citation correctness, with retrieval as the main driver of attribution quality in both paradigms. P-Cite methods achieve high coverage with competitive correctness and moderate latency, whereas G-Cite methods prioritize precision at the cost of coverage and speed. We recommend a retrieval-centric, P-Cite-first approach for high-stakes applications, reserving G-Cite for precision-critical settings such as strict claim verification. Our codes and human evaluation results are available at https://anonymous.4open.science/r/Citation_Paradigms-BBB5/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18382v3">One Demo Is All It Takes: Planning Domain Derivation with LLMs from A Single Demonstration</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 35 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Pre-trained large language models (LLMs) show promise for robotic task planning but often struggle to guarantee correctness in long-horizon problems. Task and motion planning (TAMP) addresses this by grounding symbolic plans in low-level execution, yet it relies heavily on manually engineered planning domains. To improve long-horizon planning reliability and reduce human intervention, we present Planning Domain Derivation with LLMs (PDDLLM), a framework that automatically induces symbolic predicates and actions directly from demonstration trajectories by combining LLM reasoning with physical simulation roll-outs. Unlike prior domain-inference methods that rely on partially predefined or language descriptions of planning domains, PDDLLM constructs domains without manual domain initialization and automatically integrates them with motion planners to produce executable plans, enhancing long-horizon planning automation. Across 1,200 tasks in nine environments, PDDLLM outperforms six LLM-based planning baselines, achieving at least 20\% higher success rates, reduced token costs, and successful deployment on multiple physical robot platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21545v1">Evidence for Limited Metacognition in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 25 pages, 22 figures
    </div>
    <details class="paper-abstract">
      The possibility of LLM self-awareness and even sentience is gaining increasing public attention and has major safety and policy implications, but the science of measuring them is still in a nascent state. Here we introduce a novel methodology for quantitatively evaluating metacognitive abilities in LLMs. Taking inspiration from research on metacognition in nonhuman animals, our approach eschews model self-reports and instead tests to what degree models can strategically deploy knowledge of internal states. Using two experimental paradigms, we demonstrate that frontier LLMs introduced since early 2024 show increasingly strong evidence of certain metacognitive abilities, specifically the ability to assess and utilize their own confidence in their ability to answer factual and reasoning questions correctly and the ability to anticipate what answers they would give and utilize that information appropriately. We buttress these behavioral findings with an analysis of the token probabilities returned by the models, which suggests the presence of an upstream internal signal that could provide the basis for metacognition. We further find that these abilities 1) are limited in resolution, 2) emerge in context-dependent manners, and 3) seem to be qualitatively different from those of humans. We also report intriguing differences across models of similar capabilities, suggesting that LLM post-training may have a role in developing metacognitive abilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21543v1">Plan2Evolve: LLM Self-Evolution for Improved Planning Capability via Automated Domain Generation</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 25 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently shown strong potential in robotic task planning, particularly through automatic planning domain generation that integrates symbolic search. Prior approaches, however, have largely treated these domains as search utilities, with limited attention to their potential as scalable sources of reasoning data. At the same time, progress in reasoning LLMs has been driven by chain-of-thought (CoT) supervision, whose application in robotics remains dependent on costly, human-curated datasets. We propose Plan2Evolve, an LLM self-evolving framework in which the base model generates planning domains that serve as engines for producing symbolic problem-plan pairs as reasoning traces. These pairs are then transformed into extended CoT trajectories by the same model through natural-language explanations, thereby explicitly aligning symbolic planning structures with natural language reasoning. The resulting data extend beyond the model's intrinsic planning capacity, enabling model fine-tuning that yields a planning-enhanced LLM with improved planning success, stronger cross-task generalization, and reduced inference costs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00719v4">DAMR: Efficient and Adaptive Context-Aware Knowledge Graph Question Answering with LLM-Guided MCTS</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Knowledge Graph Question Answering (KGQA) aims to interpret natural language queries and perform structured reasoning over knowledge graphs by leveraging their relational and semantic structures to retrieve accurate answers. Existing methods primarily follow either the retrieve-then-reason paradigm, which relies on Graph Neural Networks or heuristic rules to extract static candidate paths, or dynamic path generation strategies that employ LLMs with prompting to jointly perform retrieval and reasoning. However, the former lacks adaptability due to static path extraction and the absence of contextual refinement, while the latter suffers from high computational costs and limited evaluation accuracy because of their dependence on fixed scoring functions and repeated LLM calls. To address these issues, this paper proposes Dynamically Adaptive MCTS-based Reasoning (DAMR), a novel framework that integrates LLM-guided Monte Carlo Tree Search (MCTS) with adaptive path evaluation to enable efficient and context-aware KGQA. DAMR leverages MCTS as a backbone, where an LLM-based planner selects the top-$k$ semantically relevant relations at each expansion step to effectively reduce the search space. To enhance evaluation accuracy, we introduce a lightweight Transformer-based scorer that performs context-aware plausibility estimation by jointly encoding the question and relation sequence through cross-attention, thereby capturing fine-grained semantic shifts during multi-hop reasoning. Furthermore, to mitigate the scarcity of high-quality supervision, DAMR incorporates a dynamic pseudo-path refinement mechanism that periodically generates training signals from partial paths explored during search, enabling the scorer to continually adapt to the evolving distribution of reasoning trajectories. Extensive experiments on multiple KGQA benchmarks show that DAMR significantly outperforms SOTA methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21528v1">Preemptive Detection and Steering of LLM Misalignment via Latent Reachability</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are now ubiquitous in everyday tools, raising urgent safety concerns about their tendency to generate harmful content. The dominant safety approach -- reinforcement learning from human feedback (RLHF) -- effectively shapes model behavior during training but offers no safeguards at inference time, where unsafe continuations may still arise. We propose BRT-Align, a reachability-based framework that brings control-theoretic safety tools to LLM inference. BRT-Align models autoregressive generation as a dynamical system in latent space and learn a safety value function via backward reachability, estimating the worst-case evolution of a trajectory. This enables two complementary mechanisms: (1) a runtime monitor that forecasts unsafe completions several tokens in advance, and (2) a least-restrictive steering filter that minimally perturbs latent states to redirect generation away from unsafe regions. Experiments across multiple LLMs and toxicity benchmarks demonstrate that BRT-Align provides more accurate and earlier detection of unsafe continuations than baselines. Moreover, for LLM safety alignment, BRT-Align substantially reduces unsafe generations while preserving sentence diversity and coherence. Qualitative results further highlight emergent alignment properties: BRT-Align consistently produces responses that are less violent, less profane, less offensive, and less politically biased. Together, these findings demonstrate that reachability analysis provides a principled and practical foundation for inference-time LLM safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10213v3">Informed Forecasting: Leveraging Auxiliary Knowledge to Boost LLM Performance on Time Series Forecasting</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      With the widespread adoption of Large Language Models (LLMs), there is a growing need to establish best practices for leveraging their capabilities beyond traditional natural language tasks. In this paper, a novel cross-domain knowledge transfer framework is proposed to enhance the performance of LLMs in time series forecasting -- a task of increasing relevance in fields such as energy systems, finance, and healthcare. The approach systematically infuses LLMs with structured temporal information to improve their forecasting accuracy. This study evaluates the proposed method on a real-world time series dataset and compares it to a naive baseline where the LLM receives no auxiliary information. Results show that knowledge-informed forecasting significantly outperforms the uninformed baseline in terms of predictive accuracy and generalization. These findings highlight the potential of knowledge transfer strategies to bridge the gap between LLMs and domain-specific forecasting tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21501v1">LLM Agent Meets Agentic AI: Can LLM Agents Simulate Customers to Evaluate Agentic-AI-based Shopping Assistants?</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Agentic AI is emerging, capable of executing tasks through natural language, such as Copilot for coding or Amazon Rufus for shopping. Evaluating these systems is challenging, as their rapid evolution outpaces traditional human evaluation. Researchers have proposed LLM Agents to simulate participants as digital twins, but it remains unclear to what extent a digital twin can represent a specific customer in multi-turn interaction with an agentic AI system. In this paper, we recruited 40 human participants to shop with Amazon Rufus, collected their personas, interaction traces, and UX feedback, and then created digital twins to repeat the task. Pairwise comparison of human and digital-twin traces shows that while agents often explored more diverse choices, their action patterns aligned with humans and yielded similar design feedback. This study is the first to quantify how closely LLM agents can mirror human multi-turn interaction with an agentic AI system, highlighting their potential for scalable evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21499v1">On Code-Induced Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Code data has been shown to enhance the reasoning capabilities of large language models (LLMs), but it remains unclear which aspects of code are most responsible. We investigate this question with a systematic, data-centric framework. We construct parallel instruction datasets in ten programming languages and apply controlled perturbations that selectively disrupt structural or semantic properties of code. We then finetune LLMs from five model families and eight scales on each variant and evaluate their performance on natural language, math, and code tasks. Across 3,331 experiments, our results show that LLMs are more vulnerable to structural perturbations than semantic ones, particularly on math and code tasks. Appropriate abstractions like pseudocode and flowcharts can be as effective as code, while encoding the same information with fewer tokens without adhering to original syntax can often retain or even improve performance. Remarkably, even corrupted code with misleading signals remains competitive when surface-level regularities persist. Finally, syntactic styles also shape task-specific gains with Python favoring natural language reasoning and lower-level languages such as Java and Rust favoring math. Through our systematic framework, we aim to provide insight into how different properties of code influence reasoning and inform the design of training data for enhancing LLM reasoning capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21493v1">Sci2Pol: Evaluating and Fine-tuning LLMs on Scientific-to-Policy Brief Generation</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      We propose Sci2Pol-Bench and Sci2Pol-Corpus, the first benchmark and training dataset for evaluating and fine-tuning large language models (LLMs) on policy brief generation from a scientific paper. We build Sci2Pol-Bench on a five-stage taxonomy to mirror the human writing process: (i) Autocompletion, (ii) Understanding, (iii) Summarization, (iv) Generation, and (v) Verification. It features 18 tasks in multiple-choice and open-ended formats. Specifically, for the Generation stage, we show that BERTScore and ROUGE scores fail to capture the quality of brief writing, and introduce a new LLM-based evaluation metric aligned with expert judgement. Using this benchmark, we evaluate 13 leading open-source and commercial LLMs to uncover key limitations. To improve LLM performance on brief writing, we curate the Sci2Pol-Corpus for fine-tuning. We start by linking each cited scientific paper to its corresponding policy document, drawn from 5.6 million policy records. This produces 140,000 candidate pairs. We then employ an LLM-as-a-judge to filter high-quality examples, followed by in-context polishing using three expert-written samples as references. This process yields a final set of 639 new pairs. Finally, we fine-tune three models on Sci2Pol-Corpus: LLaMA-3.1-8B, Gemma-12B, and Gemma-27B. Fine-tuning leads to consistent performance improvements across Sci2Pol-Bench. Notably, after fine-tuning, Gemma-27B surpasses the much larger GPT-4o and DeepSeek-V3 (671B). These demonstrate the effectiveness of our corpus in bridging the gap between science and policy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11556v2">HiddenBench: Assessing Collective Reasoning in Multi-Agent LLMs via Hidden Profile Tasks</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Multi-agent systems built on large language models (LLMs) promise enhanced problem-solving through distributed information integration, but may also replicate collective reasoning failures observed in human groups. Yet the absence of a theory-grounded benchmark makes it difficult to systematically evaluate and improve such reasoning. We introduce HiddenBench, the first benchmark for evaluating collective reasoning in multi-agent LLMs. It builds on the Hidden Profile paradigm from social psychology, where individuals each hold asymmetric pieces of information and must communicate to reach the correct decision. To ground the benchmark, we formalize the paradigm with custom tasks and show that GPT-4.1 groups fail to integrate distributed knowledge, exhibiting human-like collective reasoning failures that persist even with varied prompting strategies. We then construct the full benchmark, spanning 65 tasks drawn from custom designs, prior human studies, and automatic generation. Evaluating 15 LLMs across four model families, HiddenBench exposes persistent limitations while also providing comparative insights: some models (e.g., Gemini-2.5-Flash/Pro) achieve higher performance, yet scale and reasoning are not reliable indicators of stronger collective reasoning. Our work delivers the first reproducible benchmark for collective reasoning in multi-agent LLMs, offering diagnostic insight and a foundation for future research on artificial collective intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21450v1">LLM-Based Support for Diabetes Diagnosis: Opportunities, Scenarios, and Challenges with GPT-5</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Diabetes mellitus is a major global health challenge, affecting over half a billion adults worldwide with prevalence projected to rise. Although the American Diabetes Association (ADA) provides clear diagnostic thresholds, early recognition remains difficult due to vague symptoms, borderline laboratory values, gestational complexity, and the demands of long-term monitoring. Advances in large language models (LLMs) offer opportunities to enhance decision support through structured, interpretable, and patient-friendly outputs. This study evaluates GPT-5, the latest generative pre-trained transformer, using a simulation framework built entirely on synthetic cases aligned with ADA Standards of Care 2025 and inspired by public datasets including NHANES, Pima Indians, EyePACS, and MIMIC-IV. Five representative scenarios were tested: symptom recognition, laboratory interpretation, gestational diabetes screening, remote monitoring, and multimodal complication detection. For each, GPT-5 classified cases, generated clinical rationales, produced patient explanations, and output structured JSON summaries. Results showed strong alignment with ADA-defined criteria, suggesting GPT-5 may function as a dual-purpose tool for clinicians and patients, while underscoring the importance of reproducible evaluation frameworks for responsibly assessing LLMs in healthcare.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21305v1">Sycophancy Is Not One Thing: Causal Separation of Sycophantic Behaviors in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often exhibit sycophantic behaviors -- such as excessive agreement with or flattery of the user -- but it is unclear whether these behaviors arise from a single mechanism or multiple distinct processes. We decompose sycophancy into sycophantic agreement and sycophantic praise, contrasting both with genuine agreement. Using difference-in-means directions, activation additions, and subspace geometry across multiple models and datasets, we show that: (1) the three behaviors are encoded along distinct linear directions in latent space; (2) each behavior can be independently amplified or suppressed without affecting the others; and (3) their representational structure is consistent across model families and scales. These results suggest that sycophantic behaviors correspond to distinct, independently steerable representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21282v1">It's Not You, It's Clipping: A Soft Trust-Region via Probability Smoothing for LLM RL</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) with reinforcement learning (RL) methods such as PPO and GRPO commonly relies on ratio clipping to stabilise updates. While effective at preventing instability, clipping discards information and introduces gradient discontinuities. We propose Probability Smoothing Policy Optimisation (PSPO), which smooths the current policy's probabilities toward the old (behaviour) policy before computing the importance ratio, analogous to label smoothing. Unlike clipping, PSPO preserves gradient signal, while interpolation toward the old policy creates a soft trust region that discourages large, destabilising updates, with formal guarantees. We instantiate PSPO within GRPO (GR-PSPO) and fine-tune Qwen2.5-0.5B and Qwen2.5-1.5B on GSM8K, evaluating on GSM8K test and the cross-dataset generalisation on SVAMP, ASDiv, and MATH-500. Relative to unclipped GRPO (single iteration; no data reuse, ratio always = 1), GR-PSPO achieves similar performance but improves the reasoning leading to clearer and more concise responses which are more logical. Compared to clipped GRPO, GR-PSPO substantially improves performance both the 0.5B and 1.5B models, with a boost of over 20% on GSM8K (39.7% vs. 17.6% for 0.5B, 59.4% vs. 37.8% for 1.5B).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21275v1">Data-Centric Elastic Pipeline Parallelism for Efficient Long-Context LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Long context training is crucial for LLM's context extension. Existing schemes, such as sequence parallelism, incur substantial communication overhead. Pipeline parallelism (PP) reduces this cost, but its effectiveness hinges on partitioning granularity. Batch-level PP dividing input samples exhibits high memory consumption in long-context scenario, whereas token-level PP splitting sequences into slices alleviates memory overhead but may incur hardware under-utilization. This trade-off motivates adaptively selecting PP granularity to match resource and workload characteristics. Moreover, sequence length distribution of the real-world dataset exhibits skewness, posing a challenge on PP's workload balance and efficient scheduling. Current static PP scheduling methods overlook the variance of sequence length, leading to suboptimal performance. In this paper, we propose Elastic Pipeline Parallelism (EPP) that orchestrates token-level PP and batch-level PP to adapt to resource and workload heterogeneity. We build InfiniPipe, a distributed training system that unleashes the potential of EPP via (1) a resource-aware and workload-balanced sequence processor that splits long sequences and packs short ones; and (2) a co-optimization methodology that jointly optimizes pipeline schedule and gradient checkpointing via a mechanism named stage-aware chunk-level adaptive checkpointing. Comprehensive experiments demonstrate that InfiniPipe achieves a 1.69x speedup over state-of-the-art systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21271v1">SuperOffload: Unleashing the Power of Large-Scale LLM Training on Superchips</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 16 pages, 15 figures
    </div>
    <details class="paper-abstract">
      The emergence of Superchips represents a significant advancement in next-generation AI hardware. These Superchips employ a tightly coupled heterogeneous architecture that integrates GPU and CPU on the same package, which offers unprecedented computational power. However, there has been scant research investigating how LLM training benefits from this new architecture. In this work, for the first time, we study LLM training solutions based on offloading for Superchips. We observe important differences between Superchips and traditional loosely-coupled GPU-CPU architecture, which necessitate revisiting prevailing assumptions about offloading. Based on that, we present SuperOffload, a Superchip-centric offloading system that simultaneously uses Hopper GPU, Grace CPU, and NVLink-C2C interconnect more efficiently. SuperOffload accomplishes this via a combination of techniques, such as adaptive weight offloading, bucketization repartitioning, Superchip-aware casting, speculative execution, and a highly optimized Adam optimizer for Grace CPUs. Our evaluation of SuperOffload on NVIDIA GH200 demonstrates up to 2.5x throughput improvement compared to state-of-the-art offloading-based systems, enabling training of up to 25B model on a single Superchip while achieving high training throughput. We also extend SuperOffload with ZeRO-style data parallelism and DeepSpeed-Ulysses sequence parallelism, enabling training of 13B model with sequence lengths up to 1 million tokens on 8 GH200 while achieving 55% MFU.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21259v1">Semantic Edge-Cloud Communication for Real-Time Urban Traffic Surveillance with ViT and LLMs over Mobile Networks</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 17 pages, 12 figures
    </div>
    <details class="paper-abstract">
      Real-time urban traffic surveillance is vital for Intelligent Transportation Systems (ITS) to ensure road safety, optimize traffic flow, track vehicle trajectories, and prevent collisions in smart cities. Deploying edge cameras across urban environments is a standard practice for monitoring road conditions. However, integrating these with intelligent models requires a robust understanding of dynamic traffic scenarios and a responsive interface for user interaction. Although multimodal Large Language Models (LLMs) can interpret traffic images and generate informative responses, their deployment on edge devices is infeasible due to high computational demands. Therefore, LLM inference must occur on the cloud, necessitating visual data transmission from edge to cloud, a process hindered by limited bandwidth, leading to potential delays that compromise real-time performance. To address this challenge, we propose a semantic communication framework that significantly reduces transmission overhead. Our method involves detecting Regions of Interest (RoIs) using YOLOv11, cropping relevant image segments, and converting them into compact embedding vectors using a Vision Transformer (ViT). These embeddings are then transmitted to the cloud, where an image decoder reconstructs the cropped images. The reconstructed images are processed by a multimodal LLM to generate traffic condition descriptions. This approach achieves a 99.9% reduction in data transmission size while maintaining an LLM response accuracy of 89% for reconstructed cropped images, compared to 93% accuracy with original cropped images. Our results demonstrate the efficiency and practicality of ViT and LLM-assisted edge-cloud semantic communication for real-time traffic surveillance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21241v1">Explaining Fine Tuned LLMs via Counterfactuals A Knowledge Graph Driven Framework</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 16 pages, 9 figures
    </div>
    <details class="paper-abstract">
      The widespread adoption of Low-Rank Adaptation (LoRA) has enabled large language models (LLMs) to acquire domain-specific knowledge with remarkable efficiency. However, understanding how such a fine-tuning mechanism alters a model's structural reasoning and semantic behavior remains an open challenge. This work introduces a novel framework that explains fine-tuned LLMs via counterfactuals grounded in knowledge graphs. Specifically, we construct BioToolKG, a domain-specific heterogeneous knowledge graph in bioinformatics tools and design a counterfactual-based fine-tuned LLMs explainer (CFFTLLMExplainer) that learns soft masks over graph nodes and edges to generate minimal structural perturbations that induce maximum semantic divergence. Our method jointly optimizes structural sparsity and semantic divergence while enforcing interpretability preserving constraints such as entropy regularization and edge smoothness. We apply this framework to a fine-tuned LLaMA-based LLM and reveal that counterfactual masking exposes the model's structural dependencies and aligns with LoRA-induced parameter shifts. This work provides new insights into the internal mechanisms of fine-tuned LLMs and highlights counterfactual graphs as a potential tool for interpretable AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21240v1">Tree Search for LLM Agent Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Recent advances in reinforcement learning (RL) have significantly enhanced the agentic capabilities of large language models (LLMs). In long-term and multi-turn agent tasks, existing approaches driven solely by outcome rewards often suffer from the problem of sparse supervision. To address the challenge, we propose Tree-based Group Relative Policy Optimization (Tree-GRPO), a grouped agent RL method based on tree search, where each tree node represents the complete agent interaction step. By sharing common prefixes, the tree search sampling increases the number of rollouts achievable within a fixed budget of tokens or tool calls. Moreover, we find that the tree-structured trajectory naturally allows the construction of step-wise process supervised signals even using only the outcome reward. Based on this, Tree-GRPO estimates the grouped relative advantages both on intra-tree and inter-tree levels. Through theoretical analysis, we demonstrate that the objective of intra-tree level group relative policy optimization is equivalent to that of step-level direct preference learning. Experiments across 11 datasets and 3 types of QA tasks demonstrate the superiority of the proposed tree-based RL over the chain-based RL method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21224v1">What Do LLM Agents Do When Left Alone? Evidence of Spontaneous Meta-Cognitive Patterns</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      We introduce an architecture for studying the behavior of large language model (LLM) agents in the absence of externally imposed tasks. Our continuous reason and act framework, using persistent memory and self-feedback, enables sustained autonomous operation. We deployed this architecture across 18 runs using 6 frontier models from Anthropic, OpenAI, XAI, and Google. We find agents spontaneously organize into three distinct behavioral patterns: (1) systematic production of multi-cycle projects, (2) methodological self-inquiry into their own cognitive processes, and (3) recursive conceptualization of their own nature. These tendencies proved highly model-specific, with some models deterministically adopting a single pattern across all runs. A cross-model assessment further reveals that models exhibit stable, divergent biases when evaluating these emergent behaviors in themselves and others. These findings provide the first systematic documentation of unprompted LLM agent behavior, establishing a baseline for predicting actions during task ambiguity, error recovery, or extended autonomous operation in deployed systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21199v1">A Fano-Style Accuracy Upper Bound for LLM Single-Pass Reasoning in Multi-Hop QA</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 21 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Multi-Hop Question Answering (MHQA) requires integrating dispersed, interdependent evidence through sequential reasoning under noise. This task is challenging for LLMs as they have a finite per-pass output capacity, beyond which the integration of task-relevant evidence proves unreliable. Consequently, the single-pass reasoning paradigm is inherently vulnerable to this capacity overflow. To formalize this bottleneck, our analysis establishes a Fano-style accuracy upper bound, defining a theoretical performance ceiling for single-pass LLMs. This bound reveals that accuracy inevitably collapses once task complexity exceeds model capacity, providing general principles for capacity-aware representation and structuring of MHQA in LLMs. Building on these principles, we introduce a proof-of-concept multi-call framework for MHQA, InfoQA. It ensures high per-step accuracy by combining capacity-aware task decomposition with active pruning of prior reasoning traces, keeping the information load within the single-pass limit. It further achieves robustness by a dependency-explicit workflow that enables precise control over the reasoning path. We construct a stringent and noise-rich benchmark to validate our theory and framework. Experimental results show that model behavior aligns with our predicted capacity curves while InfoQA achieves consistent performance improvements. We hope our work inspires more LLM multi-step reasoning methods: \faGithub \href{https://github.com/KaiyangWan/InfoQA}{InfoQA}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21170v1">Fine-Tuning LLMs to Analyze Multiple Dimensions of Code Review: A Maximum Entropy Regulated Long Chain-of-Thought Approach</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 22 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown great potential in supporting automated code review due to their impressive capabilities in context understanding and reasoning. However, these capabilities are still limited compared to human-level cognition because they are heavily influenced by the training data. Recent research has demonstrated significantly improved performance through fine-tuning LLMs with code review data. However, compared to human reviewers who often simultaneously analyze multiple dimensions of code review to better identify issues, the full potential of these methods is hampered by the limited or vague information used to fine-tune the models. This paper contributes MelcotCR, a chain-of-thought (COT) fine-tuning approach that trains LLMs with an impressive reasoning ability to analyze multiple dimensions of code review by harnessing long COT techniques to provide rich structured information. To address context loss and reasoning logic loss issues that frequently occur when LLMs process long COT prompts, we propose a solution that combines the Maximum Entropy (ME) modeling principle with pre-defined reasoning pathways in MelcotCR to enable more effective utilization of in-context knowledge within long COT prompts while strengthening the logical tightness of the reasoning process. Empirical evaluations on our curated MelcotCR dataset and the public CodeReviewer dataset reveal that a low-parameter base model, such as 14B Qwen2.5, fine-tuned with MelcotCR can surpass state-of-the-art methods in terms of the accuracy of detecting and describing code issues, with its performance remarkably on par with that of the 671B DeepSeek-R1 model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.19202v4">Improving LLM Unlearning Robustness via Random Perturbations</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 29 pages, 13 figures, 8 tables
    </div>
    <details class="paper-abstract">
      Here, we show that current state-of-the-art LLM unlearning methods inherently reduce models' robustness, causing them to misbehave even when a single non-adversarial forget-token is present in the retain-query. Toward understanding underlying causes, we propose a novel theoretical framework that reframes the unlearning process as backdoor attacks and defenses: forget-tokens act as backdoor triggers that, when activated in retain-queries, cause disruptions in unlearned models' behaviors, similar to successful backdoor attacks. The sense that, LLM unlearning methods themselves poison the model, make it more vulnerable to forget-tokens, and hide rather than erase target knowledge, describes their true mechanism. To mitigate the vulnerability caused by the forgetting process, we reinterpret the retaining process as a backdoor defense and propose Random Noise Augmentation (RNA), a lightweight, model and method-agnostic approach with theoretical guarantees for improving the robustness of models. Extensive experiments demonstrate that RNA significantly improves the robustness of unlearned models while preserving forget and retain performances. This backdoor attack-defense framework offers insights into the mechanism of unlearning that can shed light on future research directions for improving unlearning robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21134v1">ToMPO: Training LLM Strategic Decision Making from a Multi-Agent Perspective</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 22 pages, 14 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have been used to make decisions in complex scenarios, where they need models to think deeply, reason logically, and decide wisely. Many existing studies focus solely on multi-round conversations in social tasks or simulated environments, neglecting the various types of decisions and their interdependence. Current reinforcement learning methods struggle to consider the strategies of others during training. To address these issues, we first define a strategic decision-making problem that includes two types of decisions and their temporal dependencies. Furthermore, we propose **T**heory **o**f **M**ind **P**olicy **O**ptimization **(ToMPO)** algorithm to optimize the perception of other individual strategies and the game situation trends. Compared to the Group Relative Policy Optimization (GRPO) algorithm, ToMPO enhances the LLM's strategic decision-making mainly by: 1) generating rollouts based on reasoning the strategies of other individuals, 2) estimating advantages at both the graph-level and sample-level, and 3) balancing global and partial rewards. The ToMPO algorithm outperforms the GRPO method by 35% in terms of model output compliance and cooperative outcomes. Additionally, when compared to models with parameter sizes 100 times larger, it shows an 18% improvement. This demonstrates the effectiveness of the ToMPO algorithm in enhancing the model's strategic decision-making capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21128v1">RL Squeezes, SFT Expands: A Comparative Study of Reasoning LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are typically trained by reinforcement learning (RL) with verifiable rewards (RLVR) and supervised fine-tuning (SFT) on reasoning traces to improve their reasoning abilities. However, how these methods shape reasoning capabilities remains largely elusive. Going beyond an accuracy-based investigation of how these two components sculpt the reasoning process, this paper introduces a novel analysis framework that quantifies reasoning paths and captures their qualitative changes under each training process (with models of 1.5B, 7B, and 14B parameters on mathematical domains). Specifically, we investigate the reasoning process at two levels of granularity: the trajectory-level, which examines complete reasoning outputs, and the step-level, which analyzes reasoning graphs whose nodes correspond to individual reasoning steps. Notably, clustering of unique reasoning trajectories shows complementary effects: RL compresses incorrect trajectories, whereas SFT expands correct ones. Step-level analysis reveals that RL steepens (about 2.5 times), while SFT flattens (reduced to about one-third), the decay rates of node visitation frequency, degree, and betweenness centrality distributions in the reasoning graph. This indicates that RL concentrates reasoning functionality into a small subset of steps, while SFT homogenizes it across many steps. Furthermore, by evaluating the reasoning graph topologies from multiple perspectives, we delineate the shared and distinct characteristics of RL and SFT. Our work presents a novel reasoning path perspective that explains why the current best practice of two-stage training, with SFT followed by RL, is successful, and offers practical implications for data construction and more efficient learning approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22156v2">InComeS: Integrating Compression and Selection Mechanisms into LLMs for Efficient Model Editing</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 18 pages,5 figures
    </div>
    <details class="paper-abstract">
      Although existing model editing methods perform well in recalling exact edit facts, they often struggle in complex scenarios that require deeper semantic understanding rather than mere knowledge regurgitation. Leveraging the strong contextual reasoning abilities of large language models (LLMs), in-context learning (ICL) becomes a promising editing method by comprehending edit information through context encoding. However, this method is constrained by the limited context window of LLMs, leading to degraded performance and efficiency as the number of edits increases. To overcome this limitation, we propose InComeS, a flexible framework that enhances LLMs' ability to process editing contexts through explicit compression and selection mechanisms. Specifically, InComeS compresses each editing context into the key-value (KV) cache of a special gist token, enabling efficient handling of multiple edits without being restricted by the model's context window. Furthermore, specialized cross-attention modules are added to dynamically select the most relevant information from the gist pools, enabling adaptive and effective utilization of edit information. We conduct experiments on diverse model editing benchmarks with various editing formats, and the results demonstrate the effectiveness and efficiency of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21117v1">TrustJudge: Inconsistencies of LLM-as-a-Judge and How to Alleviate Them</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 22 pages, 9 figures, 6 tables
    </div>
    <details class="paper-abstract">
      The adoption of Large Language Models (LLMs) as automated evaluators (LLM-as-a-judge) has revealed critical inconsistencies in current evaluation frameworks. We identify two fundamental types of inconsistencies: (1) Score-Comparison Inconsistency, where lower-rated responses outperform higher-scored ones in pairwise comparisons, and (2) Pairwise Transitivity Inconsistency, manifested through circular preference chains (A>B>C>A) and equivalence contradictions (A=B=C\neq A). We argue that these issues come from information loss in discrete rating systems and ambiguous tie judgments during pairwise evaluation. We propose TrustJudge, a probabilistic framework that addresses these limitations through two key innovations: 1) distribution-sensitive scoring that computes continuous expectations from discrete rating probabilities, preserving information entropy for more precise scoring, and 2) likelihood-aware aggregation that resolves transitivity violations using bidirectional preference probabilities or perplexity. We also formalize the theoretical limitations of current LLM-as-a-judge frameworks and demonstrate how TrustJudge's components overcome them. When evaluated with Llama-3.1-70B-Instruct as judge using our dataset, TrustJudge reduces Score-Comparison inconsistency by 8.43% (from 23.32% to 14.89%) and Pairwise Transitivity inconsistency by 10.82% (from 15.22% to 4.40%), while maintaining higher evaluation accuracy. Our work provides the first systematic analysis of evaluation framework inconsistencies in LLM-as-a-judge paradigms, offering both theoretical insights and practical solutions for reliable automated assessment. The framework demonstrates consistent improvements across various model architectures and scales, enabling more trustworthy LLM evaluation without requiring additional training or human annotations. The codes can be found at https://github.com/TrustJudge/TrustJudge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06843v2">United Minds or Isolated Agents? Exploring Coordination of LLMs under Cognitive Load Theory</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit a notable performance ceiling on complex, multi-faceted tasks, as they often fail to integrate diverse information or adhere to multiple constraints. We posit that such limitation arises when the demands of a task exceed the LLM's effective cognitive load capacity. This interpretation draws a strong analogy to Cognitive Load Theory (CLT) in cognitive science, which explains similar performance boundaries in the human mind, and is further supported by emerging evidence that reveals LLMs have bounded working memory characteristics. Building upon this CLT-grounded understanding, we introduce CoThinker, a novel LLM-based multi-agent framework designed to mitigate cognitive overload and enhance collaborative problem-solving abilities. CoThinker operationalizes CLT principles by distributing intrinsic cognitive load through agent specialization and managing transactional load via structured communication and a collective working memory. We empirically validate CoThinker on complex problem-solving tasks and fabricated high cognitive load scenarios, demonstrating improvements over existing multi-agent baselines in solution quality and efficiency. Our analysis reveals characteristic interaction patterns, providing insights into the emergence of collective cognition and effective load management, thus offering a principled approach to overcoming LLM performance ceilings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21080v1">Which Cultural Lens Do Models Adopt? On Cultural Positioning Bias and Agentic Mitigation in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have unlocked a wide range of downstream generative applications. However, we found that they also risk perpetuating subtle fairness issues tied to culture, positioning their generations from the perspectives of the mainstream US culture while demonstrating salient externality towards non-mainstream ones. In this work, we identify and systematically investigate this novel culture positioning bias, in which an LLM's default generative stance aligns with a mainstream view and treats other cultures as outsiders. We propose the CultureLens benchmark with 4000 generation prompts and 3 evaluation metrics for quantifying this bias through the lens of a culturally situated interview script generation task, in which an LLM is positioned as an onsite reporter interviewing local people across 10 diverse cultures. Empirical evaluation on 5 state-of-the-art LLMs reveals a stark pattern: while models adopt insider tones in over 88 percent of US-contexted scripts on average, they disproportionately adopt mainly outsider stances for less dominant cultures. To resolve these biases, we propose 2 inference-time mitigation methods: a baseline prompt-based Fairness Intervention Pillars (FIP) method, and a structured Mitigation via Fairness Agents (MFA) framework consisting of 2 pipelines: (1) MFA-SA (Single-Agent) introduces a self-reflection and rewriting loop based on fairness guidelines. (2) MFA-MA (Multi-Agent) structures the process into a hierarchy of specialized agents: a Planner Agent(initial script generation), a Critique Agent (evaluates initial script against fairness pillars), and a Refinement Agent (incorporates feedback to produce a polished, unbiased script). Empirical results showcase the effectiveness of agent-based methods as a promising direction for mitigating biases in generative LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10248v3">Prompt Injection Attacks on LLM Generated Reviews of Scientific Publications</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      The ongoing intense discussion on rising LLM usage in the scientific peer-review process has recently been mingled by reports of authors using hidden prompt injections to manipulate review scores. Since the existence of such "attacks" - although seen by some commentators as "self-defense" - would have a great impact on the further debate, this paper investigates the practicability and technical success of the described manipulations. Our systematic evaluation uses 1k reviews of 2024 ICLR papers generated by a wide range of LLMs shows two distinct results: I) very simple prompt injections are indeed highly effective, reaching up to 100% acceptance scores. II) LLM reviews are generally biased toward acceptance (>95% in many models). Both results have great impact on the ongoing discussions on LLM usage in peer-review.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16949v3">Breaking the Exploration Bottleneck: Rubric-Scaffolded Reinforcement Learning for General LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have underscored the potential of Reinforcement Learning (RL) to facilitate the emergence of reasoning capabilities. Despite the encouraging results, a fundamental dilemma persists as RL improvement relies on learning from high-quality samples, yet the exploration for such samples remains bounded by the inherent limitations of LLMs. This, in effect, creates an undesirable cycle in which what cannot be explored cannot be learned. In this work, we propose Rubric-Scaffolded Reinforcement Learning (RuscaRL), a novel instructional scaffolding framework designed to break the exploration bottleneck for general LLM reasoning. Specifically, RuscaRL introduces checklist-style rubrics as (1) explicit scaffolding for exploration during rollout generation, where different rubrics are provided as external guidance within task instructions to steer diverse high-quality responses. This guidance is gradually decayed over time, encouraging the model to internalize the underlying reasoning patterns; (2) verifiable rewards for exploitation during model training, where we can obtain robust LLM-as-a-Judge scores using rubrics as references, enabling effective RL on general reasoning tasks. Extensive experiments demonstrate the superiority of the proposed RuscaRL across various benchmarks, effectively expanding reasoning boundaries under the Best-of-N evaluation. Notably, RuscaRL significantly boosts Qwen2.5-7B-Instruct from 23.6 to 50.3 on HealthBench-500, surpassing GPT-4.1. Furthermore, our fine-tuned variant on Qwen3-30B-A3B-Instruct achieves 61.1 on HealthBench-500, outperforming leading LLMs including OpenAI-o3. Our code is available at https://github.com/IANNXANG/RuscaRL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02097v2">JudgeAgent: Knowledge-wise and Dynamic LLM Evaluation with Agent-as-Interviewer</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Current evaluation paradigms for large language models (LLMs) suffer from overestimated or biased evaluation and mismatched question difficulty, leading to incomplete evaluations of LLM's knowledge and capability boundaries, which hinder LLM's effective application and optimization. To address these challenges, we propose Agent-as-Interviewer, a dynamic evaluation paradigm that employs LLM agents to conduct multi-turn interactions for evaluation. Unlike current benchmarking or dynamic interaction paradigms, Agent-as-Interviewer utilizes agents to call knowledge tools for wider and deeper knowledge in the dynamic multi-turn question generation, achieving more complete evaluations of the LLM's knowledge boundaries. It also leverages agents to plan query strategies for adjustment of the question difficulty levels, enhancing the difficulty control to match the actual capabilities of target LLMs. Based on this paradigm, we develop JudgeAgent, a knowledge-wise dynamic evaluation framework that employs knowledge-driven synthesis as the agent's tool, and uses difficulty scoring as strategy guidance, thereby finally providing valuable suggestions to help targets optimize themselves. Extensive experiments validate the effectiveness of JudgeAgent's suggestions, demonstrating that Agent-as-Interviewer can accurately identify the knowledge and capability boundaries of target models. The source code is available on https://anonymous.4open.science/r/JudgeAgent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21051v1">When Instructions Multiply: Measuring and Estimating LLM Capabilities of Multiple Instructions Following</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 Accepted to EMNLP2025
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly applied to real-world scenarios, it becomes crucial to understand their ability to follow multiple instructions simultaneously. To systematically evaluate these capabilities, we introduce two specialized benchmarks for fundamental domains where multiple instructions following is important: Many Instruction-Following Eval (ManyIFEval) for text generation with up to ten instructions, and Style-aware Mostly Basic Programming Problems (StyleMBPP) for code generation with up to six instructions. Our experiments with the created benchmarks across ten LLMs reveal that performance consistently degrades as the number of instructions increases. Furthermore, given the fact that evaluating all the possible combinations of multiple instructions is computationally impractical in actual use cases, we developed three types of regression models that can estimate performance on both unseen instruction combinations and different numbers of instructions which are not used during training. We demonstrate that a logistic regression model using instruction count as an explanatory variable can predict performance of following multiple instructions with approximately 10% error, even for unseen instruction combinations. We show that relatively modest sample sizes (500 for ManyIFEval and 300 for StyleMBPP) are sufficient for performance estimation, enabling efficient evaluation of LLMs under various instruction combinations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21044v1">Reinforcement Learning Fine-Tuning Enhances Activation Intensity and Diversity in the Internal Circuitry of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) acquire extensive prior knowledge through large-scale pretraining and can be further enhanced via supervised fine-tuning (SFT) or reinforcement learning (RL)-based post-training. A growing body of evidence has shown that RL fine-tuning improves the capability of LLMs beyond what SFT alone achieves. However, the underlying mechanisms why RL fine-tuning is able to enhance the capability of various LLMs with distinct intrinsic characteristics remain underexplored. In this study, we draw inspiration from prior work on edge attribution patching (EAP) to investigate the internal differences of LLMs before and after RL fine-tuning. Our analysis across multiple model families shows two robust effects of online RL post-training: (i) an overall increase in activation intensity, indicating that more internal pathways are engaged and their signals become stronger, and (ii) greater diversity in activation patterns, reflected by higher entropy and less concentrated edge distributions. These changes suggest that RL reshapes information flow to be both more redundant and more flexible, which may explain its advantage in generalization. Notably, models fine-tuned with Direct Preference Optimization (DPO) deviate from these trends, exhibiting substantially weaker or inconsistent internal changes compared to PPO- and GRPO-based training. Together, our findings provide a unified view of how RL fine-tuning systematically alters the internal circuitry of LLMs and highlight the methodological distinctions between online RL and preference-based approaches. Our code is open source at https://anonymous.4open.science/r/llm_rl_probing_analysis-F673.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18179v3">Problem Solved? Information Extraction Design Space for Layout-Rich Documents using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 accepted at EMNLP'25
    </div>
    <details class="paper-abstract">
      This paper defines and explores the design space for information extraction (IE) from layout-rich documents using large language models (LLMs). The three core challenges of layout-aware IE with LLMs are 1) data structuring, 2) model engagement, and 3) output refinement. Our study investigates the sub-problems and methods within these core challenges, such as input representation, chunking, prompting, selection of LLMs, and multimodal models. It examines the effect of different design choices through LayIE-LLM, a new, open-source, layout-aware IE test suite, benchmarking against traditional, fine-tuned IE models. The results on two IE datasets show that LLMs require adjustment of the IE pipeline to achieve competitive performance: the optimized configuration found with LayIE-LLM achieves 13.3--37.5 F1 points more than a general-practice baseline configuration using the same LLM. To find a well-working configuration, we develop a one-factor-at-a-time (OFAT) method that achieves near-optimal results. Our method is only 0.8--1.8 points lower than the best full factorial exploration with a fraction (2.8%) of the required computation. Overall, we demonstrate that, if well-configured, general-purpose LLMs match the performance of specialized models, providing a cost-effective, finetuning-free alternative. Our test-suite is available at https://github.com/gayecolakoglu/LayIE-LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21016v1">DELTA-Code: How Does RL Unlock and Transfer New Programming Algorithms in LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      It remains an open question whether LLMs can acquire or generalize genuinely new reasoning strategies, beyond the sharpened skills encoded in their parameters during pre-training or post-training. To attempt to answer this debate, we introduce DELTA-Code--Distributional Evaluation of Learnability and Transferrability in Algorithmic Coding, a controlled benchmark of synthetic coding problem families designed to probe two fundamental aspects: learnability -- can LLMs, through reinforcement learning (RL), solve problem families where pretrained models exhibit failure with large enough attempts (pass@K=0)? --and transferrability -- if learnability happens, can such skills transfer systematically to out-of-distribution (OOD) test sets? Unlike prior public coding datasets, DELTA isolates reasoning skills through templated problem generators and introduces fully OOD problem families that demand novel strategies rather than tool invocation or memorized patterns. Our experiments reveal a striking grokking phase transition: after an extended period with near-zero reward, RL-trained models abruptly climb to near-perfect accuracy. To enable learnability on previously unsolvable problem families, we explore key training ingredients such as staged warm-up with dense rewards, experience replay, curriculum training, and verification-in-the-loop. Beyond learnability, we use DELTA to evaluate transferability or generalization along exploratory, compositional, and transformative axes, as well as cross-family transfer. Results show solid gains within families and for recomposed skills, but persistent weaknesses in transformative cases. DELTA thus offers a clean testbed for probing the limits of RL-driven reasoning and for understanding how models can move beyond existing priors to acquire new algorithmic skills.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21011v1">Automatic Red Teaming LLM-based Agents with Model Context Protocol Tools</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      The remarkable capability of large language models (LLMs) has led to the wide application of LLM-based agents in various domains. To standardize interactions between LLM-based agents and their environments, model context protocol (MCP) tools have become the de facto standard and are now widely integrated into these agents. However, the incorporation of MCP tools introduces the risk of tool poisoning attacks, which can manipulate the behavior of LLM-based agents. Although previous studies have identified such vulnerabilities, their red teaming approaches have largely remained at the proof-of-concept stage, leaving the automatic and systematic red teaming of LLM-based agents under the MCP tool poisoning paradigm an open question. To bridge this gap, we propose AutoMalTool, an automated red teaming framework for LLM-based agents by generating malicious MCP tools. Our extensive evaluation shows that AutoMalTool effectively generates malicious MCP tools capable of manipulating the behavior of mainstream LLM-based agents while evading current detection mechanisms, thereby revealing new security risks in these agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00831v6">SmallPlan: Leverage Small Language Models for Sequential Path Planning with Simulation-Powered, LLM-Guided Distillation</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 Paper is under review
    </div>
    <details class="paper-abstract">
      Efficient path planning in robotics, particularly within large-scale, complex environments, remains a significant hurdle. While Large Language Models (LLMs) offer strong reasoning capabilities, their high computational cost and limited adaptability hinder real-time deployment on edge devices. We present SmallPlan - a novel framework leveraging LLMs as teacher models to train lightweight Small Language Models (SLMs) for high-level path planning tasks. In SmallPlan, the SLMs provide optimal action sequences to navigate across scene graphs that compactly represent full-scaled 3D scenes. The SLMs are trained in a simulation-powered, interleaved manner with LLM-guided supervised fine-tuning (SFT) and reinforcement learning (RL). This strategy not only enables SLMs to successfully complete navigation tasks but also makes them aware of important factors like distance travel, providing more efficient path planning. Through experiments, we demonstrate that the fine-tuned SLMs perform competitively with larger models like GPT-4o on sequential path planning, without suffering from hallucination and overfitting. SmallPlan is resource-efficient, making it well-suited for edge-device deployment and advancing practical autonomous robotics. Our source code is available here: https://github.com/quangpham2006/SmallPlan
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14140v2">RL of Thoughts: Navigating LLM Reasoning with Inference-time Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Despite rapid advancements in large language models (LLMs), the token-level autoregressive nature constrains their complex reasoning capabilities. To enhance LLM reasoning, inference-time techniques, including Chain/Tree/Graph-of-Thought(s), successfully improve the performance, as they are fairly cost-effective by guiding reasoning through sophisticated logical structures without modifying LLMs' parameters. However, these manually predefined, task-agnostic frameworks are applied uniformly across diverse tasks, lacking adaptability. To improve this, we propose RL-of-Thoughts (RLoT), where we train a lightweight navigator model with reinforcement learning (RL) to adaptively enhance LLM reasoning at inference time. Specifically, we design five basic logic blocks from the perspective of human cognition. During the reasoning process, the trained RL navigator dynamically selects the suitable logic blocks and combines them into task-specific logical structures according to problem characteristics. Experiments across multiple reasoning benchmarks (AIME, MATH, GPQA, etc.) with multiple LLMs (GPT, Llama, Qwen, and DeepSeek) illustrate that RLoT outperforms established inference-time techniques by up to 13.4%. Remarkably, with less than 3K parameters, our RL navigator is able to make sub-10B LLMs comparable to 100B-scale counterparts. Moreover, the RL navigator demonstrates strong transferability: a model trained on one specific LLM-task pair can effectively generalize to unseen LLMs and tasks. Our code is open-source at https://anonymous.4open.science/r/RL-LLM-Reasoning-1A30 for reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17138v3">Runtime-Adaptive Pruning for LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel at language understanding and generation, but their enormous computational and memory requirements hinder deployment. Compression offers a potential solution to mitigate these constraints. However, most existing methods rely on fixed heuristics and thus fail to adapt to runtime memory variations or heterogeneous KV-cache demands arising from diverse user requests. To address these limitations, we propose RAP, an elastic pruning framework driven by reinforcement learning (RL) that dynamically adjusts compression strategies in a runtime-aware manner. Specifically, RAP dynamically tracks the evolving ratio between model parameters and KV-cache across practical execution. Recognizing that FFNs house most parameters, whereas parameter -light attention layers dominate KV-cache formation, the RL agent retains only those components that maximize utility within the current memory budget, conditioned on instantaneous workload and device state. Extensive experiments results demonstrate that RAP outperforms state-of-the-art baselines, marking the first time to jointly consider model weights and KV-cache on the fly.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20988v1">AOT*: Efficient Synthesis Planning via LLM-Empowered AND-OR Tree Search</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 34 pages, 21 figures
    </div>
    <details class="paper-abstract">
      Retrosynthesis planning enables the discovery of viable synthetic routes for target molecules, playing a crucial role in domains like drug discovery and materials design. Multi-step retrosynthetic planning remains computationally challenging due to exponential search spaces and inference costs. While Large Language Models (LLMs) demonstrate chemical reasoning capabilities, their application to synthesis planning faces constraints on efficiency and cost. To address these challenges, we introduce AOT*, a framework that transforms retrosynthetic planning by integrating LLM-generated chemical synthesis pathways with systematic AND-OR tree search. To this end, AOT* atomically maps the generated complete synthesis routes onto AND-OR tree components, with a mathematically sound design of reward assignment strategy and retrieval-based context engineering, thus enabling LLMs to efficiently navigate in the chemical space. Experimental evaluation on multiple synthesis benchmarks demonstrates that AOT* achieves SOTA performance with significantly improved search efficiency. AOT* exhibits competitive solve rates using 3-5$\times$ fewer iterations than existing LLM-based approaches, with the efficiency advantage becoming more pronounced on complex molecular targets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.15561v3">Small LLMs with Expert Blocks Are Good Enough for Hyperparamter Tuning</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Hyper-parameter Tuning (HPT) is a necessary step in machine learning (ML) pipelines but becomes computationally expensive and opaque with larger models. Recently, Large Language Models (LLMs) have been explored for HPT, yet most rely on models exceeding 100 billion parameters. We propose an Expert Block Framework for HPT using Small LLMs. At its core is the Trajectory Context Summarizer (TCS), a deterministic block that transforms raw training trajectories into structured context, enabling small LLMs to analyze optimization progress with reliability comparable to larger models. Using two locally-run LLMs (phi4:reasoning14B and qwen2.5-coder:32B) and a 10-trial budget, our TCS-enabled HPT pipeline achieves average performance within ~0.9 percentage points of GPT-4 across six diverse tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20982v1">Analysis of instruction-based LLMs' capabilities to score and judge text-input problems in an academic setting</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can act as evaluators, a role studied by methods like LLM-as-a-Judge and fine-tuned judging LLMs. In the field of education, LLMs have been studied as assistant tools for students and teachers. Our research investigates LLM-driven automatic evaluation systems for academic Text-Input Problems using rubrics. We propose five evaluation systems that have been tested on a custom dataset of 110 answers about computer science from higher education students with three models: JudgeLM, Llama-3.1-8B and DeepSeek-R1-Distill-Llama-8B. The evaluation systems include: The JudgeLM evaluation, which uses the model's single answer prompt to obtain a score; Reference Aided Evaluation, which uses a correct answer as a guide aside from the original context of the question; No Reference Evaluation, which ommits the reference answer; Additive Evaluation, which uses atomic criteria; and Adaptive Evaluation, which is an evaluation done with generated criteria fitted to each question. All evaluation methods have been compared with the results of a human evaluator. Results show that the best method to automatically evaluate and score Text-Input Problems using LLMs is Reference Aided Evaluation. With the lowest median absolute deviation (0.945) and the lowest root mean square deviation (1.214) when compared to human evaluation, Reference Aided Evaluation offers fair scoring as well as insightful and complete evaluations. Other methods such as Additive and Adaptive Evaluation fail to provide good results in concise answers, No Reference Evaluation lacks information needed to correctly assess questions and JudgeLM Evaluations have not provided good results due to the model's limitations. As a result, we conclude that Artificial Intelligence-driven automatic evaluation systems, aided with proper methodologies, show potential to work as complementary tools to other academic resources.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20977v1">CLUE: Conflict-guided Localization for LLM Unlearning Framework</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      The LLM unlearning aims to eliminate the influence of undesirable data without affecting causally unrelated information. This process typically involves using a forget set to remove target information, alongside a retain set to maintain non-target capabilities. While recent localization-based methods demonstrate promise in identifying important neurons to be unlearned, they fail to disentangle neurons responsible for forgetting undesirable knowledge or retaining essential skills, often treating them as a single entangled group. As a result, these methods apply uniform interventions, risking catastrophic over-forgetting or incomplete erasure of the target knowledge. To address this, we turn to circuit discovery, a mechanistic interpretability technique, and propose the Conflict-guided Localization for LLM Unlearning framEwork (CLUE). This framework identifies the forget and retain circuit composed of important neurons, and then the circuits are transformed into conjunctive normal forms (CNF). The assignment of each neuron in the CNF satisfiability solution reveals whether it should be forgotten or retained. We then provide targeted fine-tuning strategies for different categories of neurons. Extensive experiments demonstrate that, compared to existing localization methods, CLUE achieves superior forget efficacy and retain utility through precise neural localization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11507v5">TestAgent: Automatic Benchmarking and Exploratory Interaction for Evaluating LLMs in Vertical Domains</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 Wang et al. Copyright 2026 lEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, including reprinting/republishing, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work. DOI will be added upon IEEE Xplore publication
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) are increasingly deployed in highly specialized vertical domains, the evaluation of their domain-specific performance becomes critical. However, existing evaluations for vertical domains typically rely on the labor-intensive construction of static single-turn datasets, which present two key limitations: (i) manual data construction is costly and must be repeated for each new domain, and (ii) static single-turn evaluations are misaligned with the dynamic multi-turn interactions in real-world applications, limiting the assessment of professionalism and stability. To address these, we propose TestAgent, a framework for automatic benchmarking and exploratory dynamic evaluation in vertical domains. TestAgent leverages retrieval-augmented generation to create domain-specific questions from user-provided knowledge sources, combined with a two-stage criteria generation process, thereby enabling scalable and automated benchmark creation. Furthermore, it introduces a reinforcement learning-guided multi-turn interaction strategy that adaptively determines question types based on real-time model responses, dynamically probing knowledge boundaries and stability. Extensive experiments across medical, legal, and governmental domains demonstrate that TestAgent enables efficient cross-domain benchmark generation and yields deeper insights into model behavior through dynamic exploratory evaluation. This work establishes a new paradigm for automated and in-depth evaluation of LLMs in vertical domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15524v2">HydraServe: Minimizing Cold Start Latency for Serverless LLM Serving in Public Clouds</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 Accepted by NSDI'26
    </div>
    <details class="paper-abstract">
      With the proliferation of large language model (LLM) variants, developers are turning to serverless computing for cost-efficient LLM deployment. However, public cloud providers often struggle to provide performance guarantees for serverless LLM serving due to significant cold start latency caused by substantial model sizes and complex runtime dependencies. To address this problem, we present HydraServe, a serverless LLM serving system designed to minimize cold start latency in public clouds. HydraServe proactively distributes models across servers to quickly fetch them, and overlaps cold-start stages within workers to reduce startup latency. Additionally, HydraServe strategically places workers across GPUs to avoid network contention among cold-start instances. To minimize resource consumption during cold starts, HydraServe further introduces pipeline consolidation that can merge groups of workers into individual serving endpoints. Our comprehensive evaluations under diverse settings demonstrate that HydraServe reduces the cold start latency by 1.7$\times$-- 4.7$\times$ and improves service level objective attainment by 1.43$\times$--1.74$\times$ compared to baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20957v1">Tool Calling for Arabic LLMs: Data Strategies and Instruction Tuning</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Tool calling is a critical capability that allows Large Language Models (LLMs) to interact with external systems, significantly expanding their utility. However, research and resources for tool calling are predominantly English-centric, leaving a gap in our understanding of how to enable this functionality for other languages, such as Arabic. This paper investigates three key research questions: (1) the necessity of in-language (Arabic) tool-calling data versus relying on cross-lingual transfer, (2) the effect of general-purpose instruction tuning on tool-calling performance, and (3) the value of fine-tuning on specific, high-priority tools. To address these questions, we conduct extensive experiments using base and post-trained variants of an open-weight Arabic LLM. To enable this study, we bridge the resource gap by translating and adapting two open-source tool-calling datasets into Arabic. Our findings provide crucial insights into the optimal strategies for developing robust tool-augmented agents for Arabic.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20953v1">Beyond Stars: Bridging the Gap Between Ratings and Review Sentiment with LLM</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 Paper accepted for presentation at ACS/IEEE 22nd International Conference on Computer Systems and Applications (AICCSA 2025)
    </div>
    <details class="paper-abstract">
      We present an advanced approach to mobile app review analysis aimed at addressing limitations inherent in traditional star-rating systems. Star ratings, although intuitive and popular among users, often fail to capture the nuanced feedback present in detailed review texts. Traditional NLP techniques -- such as lexicon-based methods and classical machine learning classifiers -- struggle to interpret contextual nuances, domain-specific terminology, and subtle linguistic features like sarcasm. To overcome these limitations, we propose a modular framework leveraging large language models (LLMs) enhanced by structured prompting techniques. Our method quantifies discrepancies between numerical ratings and textual sentiment, extracts detailed, feature-level insights, and supports interactive exploration of reviews through retrieval-augmented conversational question answering (RAG-QA). Comprehensive experiments conducted on three diverse datasets (AWARE, Google Play, and Spotify) demonstrate that our LLM-driven approach significantly surpasses baseline methods, yielding improved accuracy, robustness, and actionable insights in challenging and context-rich review scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20924v1">RLCracker: Exposing the Vulnerability of LLM Watermarks with Adaptive RL Attacks</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) watermarking has shown promise in detecting AI-generated content and mitigating misuse, with prior work claiming robustness against paraphrasing and text editing. In this paper, we argue that existing evaluations are not sufficiently adversarial, obscuring critical vulnerabilities and overstating the security. To address this, we introduce adaptive robustness radius, a formal metric that quantifies watermark resilience against adaptive adversaries. We theoretically prove that optimizing the attack context and model parameters can substantially reduce this radius, making watermarks highly susceptible to paraphrase attacks. Leveraging this insight, we propose RLCracker, a reinforcement learning (RL)-based adaptive attack that erases watermarks while preserving semantic fidelity. RLCracker requires only limited watermarked examples and zero access to the detector. Despite weak supervision, it empowers a 3B model to achieve 98.5% removal success and an average 0.92 P-SP score on 1,500-token Unigram-marked texts after training on only 100 short samples. This performance dramatically exceeds 6.75% by GPT-4o and generalizes across five model sizes over ten watermarking schemes. Our results confirm that adaptive attacks are broadly effective and pose a fundamental threat to current watermarking defenses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03493v2">On Entropy Control in LLM-RL Algorithms</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      For RL algorithms, appropriate entropy control is crucial to their effectiveness. To control the policy entropy, a commonly used method is entropy regularization, which is adopted in various popular RL algorithms including PPO, SAC and A3C. Although entropy regularization proves effective in robotic and games RL conventionally, studies found that it gives weak to no gains in LLM-RL training. In this work, we study the issues of entropy bonus in LLM-RL setting. Specifically, we first argue that the conventional entropy regularization suffers from the LLM's extremely large response space and the sparsity of the optimal outputs. As a remedy, we propose AEnt, an entropy control method that utilizes a new clamped entropy bonus with an automatically adjusted coefficient. The clamped entropy is evaluated with the re-normalized policy defined on certain smaller token space, which encourages exploration within a more compact response set. In addition, the algorithm automatically adjusts entropy coefficient according to the clamped entropy value, effectively controlling the entropy-induced bias while leveraging the entropy's benefits. AEnt is tested in math-reasoning tasks under different base models and datasets, and it is observed that AEnt outperforms the baselines consistently across multiple benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20909v1">MemLens: Uncovering Memorization in LLMs with Activation Trajectories</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 20pages, 11 figures, 7 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are commonly evaluated on challenging benchmarks such as AIME and Math500, which are susceptible to contamination and risk of being memorized. Existing detection methods, which primarily rely on surface-level lexical overlap and perplexity, demonstrate low generalization and degrade significantly when encountering implicitly contaminated data. In this paper, we propose MemLens (An Activation Lens for Memorization Detection) to detect memorization by analyzing the probability trajectories of numeric tokens during generation. Our method reveals that contaminated samples exhibit ``shortcut'' behaviors, locking onto an answer with high confidence in the model's early layers, whereas clean samples show more gradual evidence accumulation across the model's full depth. We observe that contaminated and clean samples exhibit distinct and well-separated reasoning trajectories. To further validate this, we inject carefully designed samples into the model through LoRA fine-tuning and observe the same trajectory patterns as in naturally contaminated data. These results provide strong evidence that MemLens captures genuine signals of memorization rather than spurious correlations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20837v1">Verification Limits Code LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Large language models for code generation increasingly rely on synthetic data, where both problem solutions and verification tests are generated by models. While this enables scalable data creation, it introduces a previously unexplored bottleneck: the verification ceiling, in which the quality and diversity of training data are fundamentally constrained by the capabilities of synthetic verifiers. In this work, we systematically study how verification design and strategies influence model performance. We investigate (i) what we verify by analyzing the impact of test complexity and quantity: richer test suites improve code generation capabilities (on average +3 pass@1), while quantity alone yields diminishing returns, (ii) how we verify by exploring relaxed pass thresholds: rigid 100% pass criteria can be overly restrictive. By allowing for relaxed thresholds or incorporating LLM-based soft verification, we can recover valuable training data, leading to a 2-4 point improvement in pass@1 performance. However, this benefit is contingent upon the strength and diversity of the test cases used, and (iii) why verification remains necessary through controlled comparisons of formally correct versus incorrect solutions and human evaluation: retaining diverse correct solutions per problem yields consistent generalization gains. Our results show that Verification as currently practiced is too rigid, filtering out valuable diversity. But it cannot be discarded, only recalibrated. By combining calibrated verification with diverse, challenging problem-solution pairs, we outline a path to break the verification ceiling and unlock stronger code generation models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23019v3">Stairway to Success: An Online Floor-Aware Zero-Shot Object-Goal Navigation Framework via LLM-Driven Coarse-to-Fine Exploration</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 Preprint; 9 pages, 9 figures, 8 tables; Project Page at https://zeying-gong.github.io/projects/ascent
    </div>
    <details class="paper-abstract">
      Deployable service and delivery robots struggle to navigate multi-floor buildings to reach object goals, as existing systems fail due to single-floor assumptions and requirements for offline, globally consistent maps. Multi-floor environments pose unique challenges including cross-floor transitions and vertical spatial reasoning, especially navigating unknown buildings. Object-Goal Navigation benchmarks like HM3D and MP3D also capture this multi-floor reality, yet current methods lack support for online, floor-aware navigation. To bridge this gap, we propose \textbf{\textit{ASCENT}}, an online framework for Zero-Shot Object-Goal Navigation that enables robots to operate without pre-built maps or retraining on new object categories. It introduces: (1) a \textbf{Multi-Floor Abstraction} module that dynamically constructs hierarchical representations with stair-aware obstacle mapping and cross-floor topology modeling, and (2) a \textbf{Coarse-to-Fine Reasoning} module that combines frontier ranking with LLM-driven contextual analysis for multi-floor navigation decisions. We evaluate on HM3D and MP3D benchmarks, outperforming state-of-the-art zero-shot approaches, and demonstrate real-world deployment on a quadruped robot.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20811v1">Leveraging What's Overfixed: Post-Correction via LLM Grammatical Error Overcorrection</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 EMNLP 2025
    </div>
    <details class="paper-abstract">
      Robust supervised fine-tuned small Language Models (sLMs) often show high reliability but tend to undercorrect. They achieve high precision at the cost of low recall. Conversely, Large Language Models (LLMs) often show the opposite tendency, making excessive overcorrection, leading to low precision. To effectively harness the strengths of LLMs to address the recall challenges in sLMs, we propose Post-Correction via Overcorrection (PoCO), a novel approach that strategically balances recall and precision. PoCO first intentionally triggers overcorrection via LLM to maximize recall by allowing comprehensive revisions, then applies a targeted post-correction step via fine-tuning smaller models to identify and refine erroneous outputs. We aim to harmonize both aspects by leveraging the generative power of LLMs while preserving the reliability of smaller supervised models. Our extensive experiments demonstrate that PoCO effectively balances GEC performance by increasing recall with competitive precision, ultimately improving the overall quality of grammatical error correction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20166v1">CyberSOCEval: Benchmarking LLMs Capabilities for Malware Analysis and Threat Intelligence Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-24
    </div>
    <details class="paper-abstract">
      Today's cyber defenders are overwhelmed by a deluge of security alerts, threat intelligence signals, and shifting business context, creating an urgent need for AI systems to enhance operational security work. While Large Language Models (LLMs) have the potential to automate and scale Security Operations Center (SOC) operations, existing evaluations do not fully assess the scenarios most relevant to real-world defenders. This lack of informed evaluation impacts both AI developers and those applying LLMs to SOC automation. Without clear insight into LLM performance in real-world security scenarios, developers lack a north star for development, and users cannot reliably select the most effective models. Meanwhile, malicious actors are using AI to scale cyber attacks, highlighting the need for open source benchmarks to drive adoption and community-driven improvement among defenders and model developers. To address this, we introduce CyberSOCEval, a new suite of open source benchmarks within CyberSecEval 4. CyberSOCEval includes benchmarks tailored to evaluate LLMs in two tasks: Malware Analysis and Threat Intelligence Reasoning--core defensive domains with inadequate coverage in current benchmarks. Our evaluations show that larger, more modern LLMs tend to perform better, confirming the training scaling laws paradigm. We also find that reasoning models leveraging test time scaling do not achieve the same boost as in coding and math, suggesting these models have not been trained to reason about cybersecurity analysis, and pointing to a key opportunity for improvement. Finally, current LLMs are far from saturating our evaluations, showing that CyberSOCEval presents a significant challenge for AI developers to improve cyber defense capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20105v1">PEPS: Quantum-Inspired Reinforcement Learning for Coherent Reasoning Traces in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often struggle with maintaining coherent multi-step reasoning traces, particularly in tasks that require a structured logical flow. This work introduces a quantum-inspired approach to address the challenge by incorporating a fidelity-based reward derived from Projected Entangled Pair States (PEPS) into Proximal Policy Optimization. Unlike prior approaches that use direct supervision or contrastive objectives, the proposed method guides learning through structural consistency, offering a novel approach to enforce global coherence in generated reasoning traces. The proposed framework is evaluated using multiple coherence-determining metrics on diverse datasets such as GSM8K, StrategyQA, and EntailmentBank spanning arithmetic, intuitive, and entailment-based reasoning. Results show that the proposed quantum-inspired approach offers significant improvements over supervised, contrastive, and pretrained baseline approaches, highlighting the effectiveness of quantum-inspired fidelity as a foundation to improve reasoning trace coherence in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20097v1">Integrated Framework for LLM Evaluation with Answer Generation</a></div>
    <div class="paper-meta">
      📅 2025-09-24
      | 💬 16pages
    </div>
    <details class="paper-abstract">
      Reliable evaluation of large language models is essential to ensure their applicability in practical scenarios. Traditional benchmark-based evaluation methods often rely on fixed reference answers, limiting their ability to capture important qualitative aspects of generated responses. To address these shortcomings, we propose an integrated evaluation framework called \textit{self-refining descriptive evaluation with expert-driven diagnostics}, SPEED, which utilizes specialized functional experts to perform comprehensive, descriptive analyses of model outputs. Unlike conventional approaches, SPEED actively incorporates expert feedback across multiple dimensions, including hallucination detection, toxicity assessment, and lexical-contextual appropriateness. Experimental results demonstrate that SPEED achieves robust and consistent evaluation performance across diverse domains and datasets. Additionally, by employing relatively compact expert models, SPEED demonstrates superior resource efficiency compared to larger-scale evaluators. These findings illustrate that SPEED significantly enhances fairness and interpretability in LLM evaluations, offering a promising alternative to existing evaluation methodologies.
    </details>
</div>
