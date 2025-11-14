# llm - 2025_09

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
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
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.06233v2">Confidence Improves Self-Consistency in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      Self-consistency decoding enhances LLMs' performance on reasoning tasks by sampling diverse reasoning paths and selecting the most frequent answer. However, it is computationally expensive, as sampling many of these (lengthy) paths is required to increase the chances that the correct answer emerges as the most frequent one. To address this, we introduce Confidence-Informed Self-Consistency (CISC). CISC performs a weighted majority vote based on confidence scores obtained directly from the model. By prioritizing high-confidence paths, it can identify the correct answer with a significantly smaller sample size. When tested on nine models and four datasets, CISC outperforms self-consistency in nearly all configurations, reducing the required number of reasoning paths by over 40% on average. In addition, we introduce the notion of within-question confidence evaluation, after showing that standard evaluation methods are poor predictors of success in distinguishing correct and incorrect answers to the same question. In fact, the most calibrated confidence method proved to be the least effective for CISC. Lastly, beyond these practical implications, our results and analyses show that LLMs can effectively judge the correctness of their own outputs, contributing to the ongoing debate on this topic.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24807v1">Active Authentication via Korean Keystrokes Under Varying LLM Assistance and Cognitive Contexts</a></div>
    <div class="paper-meta">
      📅 2025-09-29
      | 💬 Accepted for publication at IEEE-ICMLA 2025. Contains nine pages, six figures, and two tables
    </div>
    <details class="paper-abstract">
      Keystroke dynamics is a promising modality for active user authentication, but its effectiveness under varying LLM-assisted typing and cognitive conditions remains understudied. Using data from 50 users and cognitive labels from Bloom's Taxonomy, we evaluate keystroke-based authentication in Korean across three realistic typing scenarios: bona fide composition, LLM content paraphrasing, and transcription. Our pipeline incorporates continuity-aware segmentation, feature extraction, and classification via SVM, MLP, and XGB. Results show that the system maintains reliable performance across varying LLM usages and cognitive contexts, with Equal Error Rates ranging from 5.1% to 10.4%. These findings demonstrate the feasibility of behavioral authentication under modern writing conditions and offer insights into designing more context-resilient models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24765v1">From Ambiguity to Verdict: A Semiotic-Grounded Multi-Perspective Agent for LLM Logical Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      Logical reasoning is a fundamental capability of large language models (LLMs). However, existing studies largely overlook the interplay between logical complexity and semantic complexity, resulting in methods that struggle to address challenging scenarios involving abstract propositions, ambiguous contexts, and conflicting stances, which are central to human reasoning. For this gap, we propose LogicAgent, a semiotic-square-guided framework designed to jointly address logical complexity and semantic complexity. LogicAgent explicitly performs multi-perspective deduction in first-order logic (FOL), while mitigating vacuous reasoning through existential import checks that incorporate a three-valued decision scheme (True, False, Uncertain) to handle boundary cases more faithfully. Furthermore, to overcome the semantic simplicity and low logical complexity of existing datasets, we introduce RepublicQA, a benchmark that reaches college-level difficulty (FKGL = 11.94) and exhibits substantially greater lexical and structural diversity than prior benchmarks. RepublicQA is grounded in philosophical concepts, featuring abstract propositions and systematically organized contrary and contradictory relations, making it the most semantically rich resource for evaluating logical reasoning. Experiments demonstrate that LogicAgent achieves state-of-the-art performance on RepublicQA, with a 6.25% average gain over strong baselines, and generalizes effectively to mainstream logical reasoning benchmarks including ProntoQA, ProofWriter, FOLIO, and ProverQA, achieving an additional 7.05% average gain. These results highlight the strong effectiveness of our semiotic-grounded multi-perspective reasoning in boosting LLMs' logical performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02890v2">Grocery to General Merchandise: A Cross-Pollination Recommender using LLMs and Real-Time Cart Context</a></div>
    <div class="paper-meta">
      📅 2025-09-29
      | 💬 Accepted at RecSys 2025 EARL Workshop on Evaluating and Applying Recommender Systems with Large Language Models
    </div>
    <details class="paper-abstract">
      Modern e-commerce platforms strive to enhance customer experience by providing timely and contextually relevant recommendations. However, recommending general merchandise to customers focused on grocery shopping -- such as pairing milk with a milk frother -- remains a critical yet under-explored challenge. This paper introduces a cross-pollination (XP) framework, a novel approach that bridges grocery and general merchandise cross-category recommendations by leveraging multi-source product associations and real-time cart context. Our solution employs a two-stage framework: (1) A candidate generation mechanism that uses co-purchase market basket analysis and LLM-based approach to identify novel item-item associations; and (2) a transformer-based ranker that leverages the real-time sequential cart context and optimizes for engagement signals such as add-to-carts. Offline analysis and online A/B tests show an increase of 36\% add-to-cart rate with LLM-based retrieval on the item page, and 15\% lift in add-to-cart using cart context-based ranker on the cart page. Our work contributes practical techniques for cross-category recommendations and broader insights for e-commerce systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24830v2">Improving Reliability and Explainability of Medical Question Answering through Atomic Fact Checking in Retrieval-Augmented LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-29
      | 💬 15 pages, 5 figures and tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit extensive medical knowledge but are prone to hallucinations and inaccurate citations, which pose a challenge to their clinical adoption and regulatory compliance. Current methods, such as Retrieval Augmented Generation, partially address these issues by grounding answers in source documents, but hallucinations and low fact-level explainability persist. In this work, we introduce a novel atomic fact-checking framework designed to enhance the reliability and explainability of LLMs used in medical long-form question answering. This method decomposes LLM-generated responses into discrete, verifiable units called atomic facts, each of which is independently verified against an authoritative knowledge base of medical guidelines. This approach enables targeted correction of errors and direct tracing to source literature, thereby improving the factual accuracy and explainability of medical Q&A. Extensive evaluation using multi-reader assessments by medical experts and an automated open Q&A benchmark demonstrated significant improvements in factual accuracy and explainability. Our framework achieved up to a 40% overall answer improvement and a 50% hallucination detection rate. The ability to trace each atomic fact back to the most relevant chunks from the database provides a granular, transparent explanation of the generated responses, addressing a major gap in current medical AI applications. This work represents a crucial step towards more trustworthy and reliable clinical applications of LLMs, addressing key prerequisites for clinical application and fostering greater confidence in AI-assisted healthcare.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24730v1">Diamonds in the rough: Transforming SPARCs of imagination into a game concept by leveraging medium sized LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-29
      | 💬 Appears in Proceedings of AI4HGI '25, the First Workshop on Artificial Intelligence for Human-Game Interaction at the 28th European Conference on Artificial Intelligence (ECAI '25), Bologna, October 25-30, 2025
    </div>
    <details class="paper-abstract">
      Recent research has demonstrated that large language models (LLMs) can support experts across various domains, including game design. In this study, we examine the utility of medium-sized LLMs, models that operate on consumer-grade hardware typically available in small studios or home environments. We began by identifying ten key aspects that contribute to a strong game concept and used ChatGPT to generate thirty sample game ideas. Three medium-sized LLMs, LLaMA 3.1, Qwen 2.5, and DeepSeek-R1, were then prompted to evaluate these ideas according to the previously identified aspects. A qualitative assessment by two researchers compared the models' outputs, revealing that DeepSeek-R1 produced the most consistently useful feedback, despite some variability in quality. To explore real-world applicability, we ran a pilot study with ten students enrolled in a storytelling course for game development. At the early stages of their own projects, students used our prompt and DeepSeek-R1 to refine their game concepts. The results indicate a positive reception: most participants rated the output as high quality and expressed interest in using such tools in their workflows. These findings suggest that current medium-sized LLMs can provide valuable feedback in early game design, though further refinement of prompting methods could improve consistency and overall effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24706v1">LLM-Handover:Exploiting LLMs for Task-Oriented Robot-Human Handovers</a></div>
    <div class="paper-meta">
      📅 2025-09-29
      | 💬 Accepted to IEEE Robotics and Automation Letters (RA-L)
    </div>
    <details class="paper-abstract">
      Effective human-robot collaboration depends on task-oriented handovers, where robots present objects in ways that support the partners intended use. However, many existing approaches neglect the humans post-handover action, relying on assumptions that limit generalizability. To address this gap, we propose LLM-Handover, a novel framework that integrates large language model (LLM)-based reasoning with part segmentation to enable context-aware grasp selection and execution. Given an RGB-D image and a task description, our system infers relevant object parts and selects grasps that optimize post-handover usability. To support evaluation, we introduce a new dataset of 60 household objects spanning 12 categories, each annotated with detailed part labels. We first demonstrate that our approach improves the performance of the used state-of-the-art part segmentation method, in the context of robot-human handovers. Next, we show that LLM-Handover achieves higher grasp success rates and adapts better to post-handover task constraints. During hardware experiments, we achieve a success rate of 83% in a zero-shot setting over conventional and unconventional post-handover tasks. Finally, our user study underlines that our method enables more intuitive, context-aware handovers, with participants preferring it in 86% of cases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24678v1">Reference-Free Rating of LLM Responses via Latent Information</a></div>
    <div class="paper-meta">
      📅 2025-09-29
      | 💬 21 pages
    </div>
    <details class="paper-abstract">
      How reliable are single-response LLM-as-a-judge ratings without references, and can we obtain fine-grained, deterministic scores in this setting? We study the common practice of asking a judge model to assign Likert-scale scores to free-text responses and show two systematic issues: scores are unstable under sampling and poorly calibrated, leading to compression near the top of the scale and frequent ties. We then propose and evaluate Latent Judges, which derive scalar ratings from internal model signals: (i) probability-weighted scores over integer ratings, (ii) verifier-style probabilities of "yes", and (iii) linear probes trained on model activations at the rating position. Across a broad suite of pairwise and single-rating benchmarks, latent methods match or surpass standard prompting, with consistent gains on pairwise accuracy and listwise ranking relevant to Best-of-N selection. Probability-weighted scores achieve the strongest single-rating correlations, while probes recover useful signals when output logits are miscalibrated. These results indicate that latent information provides deterministic and more discriminative signals for reference-free evaluation, and can improve selection and training approaches like Best-of-$N$, multi-teacher distillation, and routing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.22582v2">Fine-Grained Detection of Context-Grounded Hallucinations Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      Context-grounded hallucinations are cases where model outputs contain information not verifiable against the source text. We study the applicability of LLMs for localizing such hallucinations, as a more practical alternative to existing complex evaluation pipelines. In the absence of established benchmarks for meta-evaluation of hallucinations localization, we construct one tailored to LLMs, involving a challenging human annotation of over 1,000 examples. We complement the benchmark with an LLM-based evaluation protocol, verifying its quality in a human evaluation. Since existing representations of hallucinations limit the types of errors that can be expressed, we propose a new representation based on free-form textual descriptions, capturing the full range of possible errors. We conduct a comprehensive study, evaluating four large-scale LLMs, which highlights the benchmark's difficulty, as the best model achieves an F1 score of only 0.67. Through careful analysis, we offer insights into optimal prompting strategies for the task and identify the main factors that make it challenging for LLMs: (1) a tendency to incorrectly flag missing details as inconsistent, despite being instructed to check only facts in the output; and (2) difficulty with outputs containing factually correct information absent from the source - and thus not verifiable - due to alignment with the model's parametric knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19558v2">PoliCon: Evaluating LLMs on Achieving Diverse Political Consensus Objectives</a></div>
    <div class="paper-meta">
      📅 2025-09-28
      | 💬 PoliCon is publicly available at https://zowiezhang.github.io/projects/PoliCon
    </div>
    <details class="paper-abstract">
      Achieving political consensus is crucial yet challenging for the effective functioning of social governance. However, although frontier AI systems represented by large language models (LLMs) have developed rapidly in recent years, their capabilities in this scope are still understudied. In this paper, we introduce PoliCon, a novel benchmark constructed from 2,225 high-quality deliberation records of the European Parliament over 13 years, ranging from 2009 to 2022, to evaluate the ability of LLMs to draft consensus resolutions based on divergent party positions under varying collective decision-making contexts and political requirements. Specifically, PoliCon incorporates four factors to build each task environment for finding different political consensus: specific political issues, political goals, participating parties, and power structures based on seat distribution. We also developed an evaluation framework based on social choice theory for PoliCon, which simulates the real voting outcomes of different political parties to assess whether LLM-generated resolutions meet the requirements of the predetermined political consensus. Our experimental results demonstrate that even state-of-the-art models remain undersatisfied with complex tasks like passing resolutions by a two-thirds majority and addressing security issues, while uncovering their inherent partisan biases and revealing some behaviors LLMs show to achieve the consensus, such as prioritizing the stance of the dominant party instead of uniting smaller parties, which highlights PoliCon's promise as an effective platform for studying LLMs' ability to promote political consensus.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23808v1">Beyond the Exploration-Exploitation Trade-off: A Hidden State Approach for LLM Reasoning in RLVR</a></div>
    <div class="paper-meta">
      📅 2025-09-28
    </div>
    <details class="paper-abstract">
      A prevailing view in Reinforcement Learning for Verifiable Rewards (RLVR) interprets recent progress through the lens of an exploration-exploitation trade-off, a perspective largely shaped by token-level metrics. We re-examine this perspective, proposing that this perceived trade-off may not be a fundamental constraint but rather an artifact of the measurement level. To investigate this, we shift the analysis to the semantically rich hidden-state space, adopting Effective Rank (ER) to quantify exploration and proposing its novel first- and second-order derivatives, named Effective Rank Velocity (ERV) and Effective Rank Acceleration (ERA), to capture exploitation dynamics. Our analysis reveals that at the hidden-state level, exploration and exploitation could be decoupled (Sec. 4). This finding reveals an opportunity to enhance both capacities simultaneously. This insight motivates our method, Velocity-Exploiting Rank-Learning (VERL), the first to operationalize the principle of synergistic exploration-exploitation enhancement by directly shaping the RL advantage function. The key innovation is leveraging the theoretically stable ERA as a predictive meta-controller to create a synergistic, dual-channel incentive structure. Instead of forcing a trade-off, VERL prospectively amplifies rewards for exploration to preempt overconfidence and reinforces exploitative gains to consolidate reasoning. Experiments across diverse LLMs and reasoning benchmarks show consistent gains, including up to 21.4% absolute accuracy improvement on the challenging Gaokao 2024 dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23803v1">FedAgentBench: Towards Automating Real-world Federated Medical Image Analysis with Server-Client LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-09-28
    </div>
    <details class="paper-abstract">
      Federated learning (FL) allows collaborative model training across healthcare sites without sharing sensitive patient data. However, real-world FL deployment is often hindered by complex operational challenges that demand substantial human efforts. This includes: (a) selecting appropriate clients (hospitals), (b) coordinating between the central server and clients, (c) client-level data pre-processing, (d) harmonizing non-standardized data and labels across clients, and (e) selecting FL algorithms based on user instructions and cross-client data characteristics. However, the existing FL works overlook these practical orchestration challenges. These operational bottlenecks motivate the need for autonomous, agent-driven FL systems, where intelligent agents at each hospital client and the central server agent collaboratively manage FL setup and model training with minimal human intervention. To this end, we first introduce an agent-driven FL framework that captures key phases of real-world FL workflows from client selection to training completion and a benchmark dubbed FedAgentBench that evaluates the ability of LLM agents to autonomously coordinate healthcare FL. Our framework incorporates 40 FL algorithms, each tailored to address diverse task-specific requirements and cross-client characteristics. Furthermore, we introduce a diverse set of complex tasks across 201 carefully curated datasets, simulating 6 modality-specific real-world healthcare environments, viz., Dermatoscopy, Ultrasound, Fundus, Histopathology, MRI, and X-Ray. We assess the agentic performance of 14 open-source and 10 proprietary LLMs spanning small, medium, and large model scales. While some agent cores such as GPT-4.1 and DeepSeek V3 can automate various stages of the FL pipeline, our results reveal that more complex, interdependent tasks based on implicit goals remain challenging for even the strongest models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23799v1">Enhancing LLM Steering through Sparse Autoencoder-Based Vector Refinement</a></div>
    <div class="paper-meta">
      📅 2025-09-28
      | 💬 19 pages, 11 figures, 7 tables
    </div>
    <details class="paper-abstract">
      Steering has emerged as a promising approach in controlling large language models (LLMs) without modifying model parameters. However, most existing steering methods rely on large-scale datasets to learn clear behavioral information, which limits their applicability in many real-world scenarios. The steering vectors extracted from small dataset often contain task-irrelevant noising features, which degrades their effectiveness. To refine the steering vectors learned from limited data, we introduce Refinement of Steering Vector via Sparse Autoencoder (SAE-RSV) that leverages SAEs to semantically denoise and augment the steering vectors. In our framework, we first remove task-irrelevant features according to their semantics provided by SAEs, and then enrich task-relevant features missing from the small dataset through their semantic similarity to the identified relevant features. Extensive experiments demonstrate that the proposed SAE-RSV substantially outperforms all the baseline methods including supervised fine-tuning. Our findings show that effective steering vector can be constructed from limited training data by refining the original steering vector through SAEs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00075v2">Theoretical Modeling of LLM Self-Improvement Training Dynamics Through Solver-Verifier Gap</a></div>
    <div class="paper-meta">
      📅 2025-09-28
      | 💬 28 pages
    </div>
    <details class="paper-abstract">
      Self-improvement is among the most prominent techniques within the realm of large language models (LLM), aiming to enhance the LLM performance without relying on external data. Despite its significance, generally how LLM performances evolve during the self-improvement process remains underexplored. In this paper, we theoretically model the training dynamics of self-improvement via the concept of solver-verifier gap. This is inspired by the conjecture that the performance enhancement of self-improvement stems from the gap between LLM's solver capability and verifier capability. Based on the theoretical framework, we further show how to model the entire training trajectory. This framework allows quantifying the capability limit of self-improvement by fitting the theoretical model to the experiment results. We empirically validate the effectiveness of the theoretical framework on various LLMs and datasets. Beyond self-improvement, we extend our analysis to investigate how external data influences these dynamics within the framework. Notably, we find that under limited external data regimes, such external data can be utilized at any stage without significantly affecting final performances, which accords with the empirical observations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07335v2">Improving LLM Reasoning through Interpretable Role-Playing Steering</a></div>
    <div class="paper-meta">
      📅 2025-09-28
      | 💬 Accepted at EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Role-playing has emerged as an effective technique for enhancing the reasoning capabilities of large language models (LLMs). However, existing methods primarily rely on prompt engineering, which often lacks stability and interpretability. In this paper, we introduce Sparse Autoencoder Role-Playing Steering (SRPS), a novel framework that identifies and manipulates internal model features associated with role-playing behavior. Our approach extracts latent representations from role-play prompts, selects the most relevant features based on activation patterns, and constructs a steering vector that can be injected into the model's residual stream with controllable intensity. Our method enables fine-grained control over role-specific behavior and offers insights into how role information influences internal model activations. Extensive experiments across various reasoning benchmarks and model sizes demonstrate consistent performance gains. Notably, in the zero-shot chain-of-thought (CoT) setting, the accuracy of Llama3.1-8B on CSQA improves from 31.86% to 39.80%, while Gemma2-9B on SVAMP increases from 37.50% to 45.10%. These results highlight the potential of SRPS to enhance reasoning ability in LLMs, providing better interpretability and stability compared to traditional prompt-based role-playing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23782v1">Bridging the Knowledge-Prediction Gap in LLMs on Multiple-Choice Questions</a></div>
    <div class="paper-meta">
      📅 2025-09-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often fail on multiple-choice questions (MCQs) despite demonstrating correct knowledge in other contexts, such as free-form generation. To investigate the mechanism underlying this knowledge-prediction gap on MCQs and alleviate it, we conduct a probing analysis and find that residual streams in certain layers contain a subspace spanned by two important bases: a \emph{knowledge basis} that encodes the probability of the ground-truth answer for a given MCQ and a \emph{prediction basis} that encodes the probability of the answer choice predicted by the model. We observe that incorrect predictions arise from a misalignment of the model's hidden states along these two bases. Hence, we introduce \textbf{KAPPA} (Knowledge-Aligned Prediction through Projection-based Adjustment), a parameter-free intervention that transforms the hidden states to align the prediction coordinate with the knowledge coordinate within this subspace. Experiments on binary-choice reformulations of Big-Bench-Hard and ARC-Challenge show that KAPPA substantially improves accuracy and consistently outperforms baselines. While optimal subspaces differ across tasks, subspaces generalize to some extent, as supported by cross-dataset experiments. Moreover, KAPPA extends its effectiveness to free-form questions beyond MCQs. Our work provides a new geometric understanding of the knowledge-prediction gap and offers a practical method for better aligning model behavior with its latent knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23767v1">From Personal to Collective: On the Role of Local and Global Memory in LLM Personalization</a></div>
    <div class="paper-meta">
      📅 2025-09-28
    </div>
    <details class="paper-abstract">
      Large language model (LLM) personalization aims to tailor model behavior to individual users based on their historical interactions. However, its effectiveness is often hindered by two key challenges: the \textit{cold-start problem}, where users with limited history provide insufficient context for accurate personalization, and the \textit{biasing problem}, where users with abundant but skewed history cause the model to overfit to narrow preferences. We identify both issues as symptoms of a common underlying limitation, i.e., the inability to model collective knowledge across users. To address this, we propose a local-global memory framework (LoGo) that combines the personalized local memory with a collective global memory that captures shared interests across the population. To reconcile discrepancies between these two memory sources, we introduce a mediator module designed to resolve conflicts between local and global signals. Extensive experiments on multiple benchmarks demonstrate that LoGo consistently improves personalization quality by both warming up cold-start users and mitigating biased predictions. These results highlight the importance of incorporating collective knowledge to enhance LLM personalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.15395v3">Parse Trees Guided LLM Prompt Compression</a></div>
    <div class="paper-meta">
      📅 2025-09-28
      | 💬 Accepted by IEEE TPAMI
    </div>
    <details class="paper-abstract">
      Offering rich contexts to Large Language Models (LLMs) has shown to boost the performance in various tasks, but the resulting longer prompt would increase the computational cost and might exceed the input limit of LLMs. Recently, some prompt compression methods have been suggested to shorten the length of prompts by using language models to generate shorter prompts or by developing computational models to select important parts of original prompt. The generative compression methods would suffer from issues like hallucination, while the selective compression methods have not involved linguistic rules and overlook the global structure of prompt. To this end, we propose a novel selective compression method called PartPrompt. It first obtains a parse tree for each sentence based on linguistic rules, and calculates local information entropy for each node in a parse tree. These local parse trees are then organized into a global tree according to the hierarchical structure such as the dependency of sentences, paragraphs, and sections. After that, the root-ward propagation and leaf-ward propagation are proposed to adjust node values over the global tree. Finally, a recursive algorithm is developed to prune the global tree based on the adjusted node values. The experiments show that PartPrompt receives the state-of-the-art performance across various datasets, metrics, compression ratios, and target LLMs for inference. The in-depth ablation studies confirm the effectiveness of designs in PartPrompt, and other additional experiments also demonstrate its superiority in terms of the coherence of compressed prompts and in the extreme long prompt scenario.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23755v1">Understanding Textual Capability Degradation in Speech LLMs via Parameter Importance Analysis</a></div>
    <div class="paper-meta">
      📅 2025-09-28
    </div>
    <details class="paper-abstract">
      The integration of speech into Large Language Models (LLMs) has substantially expanded their capabilities, but often at the cost of weakening their core textual competence. This degradation limits the ability of speech-enabled LLMs to fully exploit their pre-trained text-based knowledge. In this work, we analyze the underlying mechanisms of this issue through a focused study of the widely used encoder-adaptor paradigm. We propose an analytical framework based on parameter importance estimation, which reveals that fine-tuning for speech introduces a textual importance distribution shift: the layer-wise allocation of parameters critical to textual reasoning is disrupted. Building on this insight, we investigate two mitigation strategies: layer-wise learning rate scheduling and Low-Rank Adaptation (LoRA), both aim to preserve the original parameter distribution. Experimental results show that both approaches better maintain textual competence than full fine-tuning, while also improving downstream spoken question answering performance. Furthermore, our analysis offers a principled explanation for the effectiveness of the proposed mitigation strategies, linking their benefits to the structural properties of textual knowledge in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.23718v2">PROMFUZZ: Leveraging LLM-Driven and Bug-Oriented Composite Analysis for Detecting Functional Bugs in Smart Contracts</a></div>
    <div class="paper-meta">
      📅 2025-09-28
    </div>
    <details class="paper-abstract">
      Smart contracts are fundamental pillars of the blockchain, playing a crucial role in facilitating various business transactions. However, these smart contracts are vulnerable to exploitable bugs that can lead to substantial monetary losses. A recent study reveals that over 80% of these exploitable bugs, which are primarily functional bugs, can evade the detection of current tools. The primary issue is the significant gap between understanding the high-level logic of the business model and checking the low-level implementations in smart contracts. Furthermore, identifying deeply rooted functional bugs in smart contracts requires the automated generation of effective detection oracles based on various bug features. To address these challenges, we design and implement PROMFUZZ, an automated and scalable system to detect functional bugs, in smart contracts. In PROMFUZZ, we first propose a novel Large Language Model (LLM)-driven analysis framework, which leverages a dual-agent prompt engineering strategy to pinpoint potentially vulnerable functions for further scrutiny. We then implement a dual-stage coupling approach, which focuses on generating invariant checkers that leverage logic information extracted from potentially vulnerable functions. Finally, we design a bug-oriented fuzzing engine, which maps the logical information from the high-level business model to the low-level smart contract implementations, and performs the bug-oriented fuzzing on targeted functions. We compare PROMFUZZ with multiple state-of-the-art methods. The results show that PROMFUZZ achieves 86.96% recall and 93.02% F1-score in detecting functional bugs, marking at least a 50% improvement in both metrics over state-of-the-art methods. Moreover, we perform an in-depth analysis on real-world DeFi projects and detect 30 zero-day bugs. Up to now, 24 zero-day bugs have been assigned CVE IDs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09129v2">NextLocLLM: Location Semantics Modeling and Coordinate-Based Next Location Prediction with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-28
      | 💬 STIntelligence in CIKM 2025
    </div>
    <details class="paper-abstract">
      Next location prediction is a critical task in human mobility analysis.Existing methods typically formulate it as a classification task based on discrete location IDs, which hinders spatial continuity modeling and limits generalization to new cities. In this paper, we propose NextLocLLM, a novel framework that reformulates next-location prediction as coordinate regression and integrates LLMs for both location semantics encoding and coordinate-level prediction. To model location functional semantics, it constructs LLM-enhanced POI embeddings by leveraging language understanding capabilities of LLMs to extract functional semantics from textual descriptions of POI categories. These POI embeddings are combined with spatiotemporal trajectory representation and fed into the same LLM, enabling unified semantic and predictive modeling. A lightweight regression head generates coordinate outputs, which are mapped to top-k candidate locations via post-prediction retrieval module, ensuring structured outputs. Experiments across diverse cities show that NextLocLLM outperforms existing baselines in both supervised and zero-shot settings. Code is available at: https://github.com/liuwj2000/NexelocLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23715v1">Do LLMs Understand Romanian Driving Laws? A Study on Multimodal and Fine-Tuned Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-09-28
      | 💬 Accepted@ CONSILR 2025 Bucharest Romania 9-10 October
    </div>
    <details class="paper-abstract">
      Ensuring that both new and experienced drivers master current traffic rules is critical to road safety. This paper evaluates Large Language Models (LLMs) on Romanian driving-law QA with explanation generation. We release a 1{,}208-question dataset (387 multimodal) and compare text-only and multimodal SOTA systems, then measure the impact of domain-specific fine-tuning for Llama 3.1-8B-Instruct and RoLlama 3.1-8B-Instruct. SOTA models perform well, but fine-tuned 8B models are competitive. Textual descriptions of images outperform direct visual input. Finally, an LLM-as-a-Judge assesses explanation quality, revealing self-preference bias. The study informs explainable QA for less-resourced languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00089v2">TRAPDOC: Deceiving LLM Users by Injecting Imperceptible Phantom Tokens into Documents</a></div>
    <div class="paper-meta">
      📅 2025-09-28
      | 💬 EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      The reasoning, writing, text-editing, and retrieval capabilities of proprietary large language models (LLMs) have advanced rapidly, providing users with an ever-expanding set of functionalities. However, this growing utility has also led to a serious societal concern: the over-reliance on LLMs. In particular, users increasingly delegate tasks such as homework, assignments, or the processing of sensitive documents to LLMs without meaningful engagement. This form of over-reliance and misuse is emerging as a significant social issue. In order to mitigate these issues, we propose a method injecting imperceptible phantom tokens into documents, which causes LLMs to generate outputs that appear plausible to users but are in fact incorrect. Based on this technique, we introduce TRAPDOC, a framework designed to deceive over-reliant LLM users. Through empirical evaluation, we demonstrate the effectiveness of our framework on proprietary LLMs, comparing its impact against several baselines. TRAPDOC serves as a strong foundation for promoting more responsible and thoughtful engagement with language models. Our code is available at https://github.com/jindong22/TrapDoc.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23694v1">SafeSearch: Automated Red-Teaming for the Safety of LLM-Based Search Agents</a></div>
    <div class="paper-meta">
      📅 2025-09-28
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Search agents connect LLMs to the Internet, enabling access to broader and more up-to-date information. However, unreliable search results may also pose safety threats to end users, establishing a new threat surface. In this work, we conduct two in-the-wild experiments to demonstrate both the prevalence of low-quality search results and their potential to misguide agent behaviors. To counter this threat, we introduce an automated red-teaming framework that is systematic, scalable, and cost-efficient, enabling lightweight and harmless safety assessments of search agents. Building on this framework, we construct the SafeSearch benchmark, which includes 300 test cases covering five categories of risks (e.g., misinformation and indirect prompt injection). Using this benchmark, we evaluate three representative search agent scaffolds, covering search workflow, tool-calling, and deep research, across 7 proprietary and 8 open-source backend LLMs. Our results reveal substantial vulnerabilities of LLM-based search agents: when exposed to unreliable websites, the highest ASR reached 90.5% for GPT-4.1-mini under a search workflow setting. Moreover, our analysis highlights the limited effectiveness of common defense practices, such as reminder prompting. This emphasizes the value of our framework in promoting transparency for safer agent development. Our codebase and test cases are publicly available: https://github.com/jianshuod/SafeSearch.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23674v1">AssertGen: Enhancement of LLM-aided Assertion Generation through Cross-Layer Signal Bridging</a></div>
    <div class="paper-meta">
      📅 2025-09-28
      | 💬 6 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Assertion-based verification (ABV) serves as a crucial technique for ensuring that register-transfer level (RTL) designs adhere to their specifications. While Large Language Model (LLM) aided assertion generation approaches have recently achieved remarkable progress, existing methods are still unable to effectively identify the relationship between design specifications and RTL designs, which leads to the insufficiency of the generated assertions. To address this issue, we propose AssertGen, an assertion generation framework that automatically generates SystemVerilog assertions (SVA). AssertGen first extracts verification objectives from specifications using a chain-of-thought (CoT) reasoning strategy, then bridges corresponding signals between these objectives and the RTL code to construct a cross-layer signal chain, and finally generates SVAs based on the LLM. Experimental results demonstrate that AssertGen outperforms the existing state-of-the-art methods across several key metrics, such as pass rate of formal property verification (FPV), cone of influence (COI), proof core and mutation testing coverage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23659v1">Aligning LLMs for Multilingual Consistency in Enterprise Applications</a></div>
    <div class="paper-meta">
      📅 2025-09-28
      | 💬 Accepted in EMNLP 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) remain unreliable for global enterprise applications due to substantial performance gaps between high-resource and mid/low-resource languages, driven by English-centric pretraining and internal reasoning biases. This inconsistency undermines customer experience and operational reliability in multilingual settings such as customer support, content moderation, and information retrieval. Even with advanced Retrieval-Augmented Generation (RAG) systems, we observe up to an 29% accuracy drop in non-English languages compared to English. We propose a practical, batch-wise alignment strategy for fine-tuning LLMs, leveraging semantically equivalent multilingual data in each training batch to directly align model outputs across languages. This approach improves non-English accuracy by up to 23.9\% without compromising English performance, model reasoning, or retrieval quality. Our method is simple to implement, scalable, and integrates seamlessly with existing LLM training \& deployment pipelines, enabling more robust and equitable multilingual AI solutions in industry.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23657v1">Beyond English-Centric Training: How Reinforcement Learning Improves Cross-Lingual Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-28
    </div>
    <details class="paper-abstract">
      Enhancing the complex reasoning capabilities of Large Language Models (LLMs) attracts widespread attention. While reinforcement learning (RL) has shown superior performance for improving complex reasoning, its impact on cross-lingual generalization compared to Supervised Fine-Tuning (SFT) remains unexplored. We present the first systematic investigation into cross-lingual reasoning generalization of RL and SFT. Using Qwen2.5-3B-Base as our foundation model, we conduct experiments on diverse multilingual reasoning benchmarks, including math reasoning, commonsense reasoning, and scientific reasoning. Our investigation yields two significant findings: (1) Tuning with RL not only achieves higher accuracy but also demonstrates substantially stronger cross-lingual generalization capabilities compared to SFT. (2) RL training on non-English data yields better overall performance and generalization than training on English data, which is not observed with SFT. Furthermore, through comprehensive mechanistic analyses, we explore the underlying factors of RL's superiority and generalization across languages. Our results provide compelling evidence that RL enables the model with more robust reasoning strategies, offering crucial guidance for more equitable and effective multilingual reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14530v2">Internal Chain-of-Thought: Empirical Evidence for Layer-wise Subtask Scheduling in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-28
      | 💬 EMNLP 2025 Main
    </div>
    <details class="paper-abstract">
      We show that large language models (LLMs) exhibit an $\textit{internal chain-of-thought}$: they sequentially decompose and execute composite tasks layer-by-layer. Two claims ground our study: (i) distinct subtasks are learned at different network depths, and (ii) these subtasks are executed sequentially across layers. On a benchmark of 15 two-step composite tasks, we employ layer-from context-masking and propose a novel cross-task patching method, confirming (i). To examine claim (ii), we apply LogitLens to decode hidden states, revealing a consistent layerwise execution pattern. We further replicate our analysis on the real-world $\text{TRACE}$ benchmark, observing the same stepwise dynamics. Together, our results enhance LLMs transparency by showing their capacity to internally plan and execute subtasks (or instructions), opening avenues for fine-grained, instruction-level activation steering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10036v2">DataPuzzle: Breaking Free from the Hallucinated Promise of LLMs in Data Analysis</a></div>
    <div class="paper-meta">
      📅 2025-09-28
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly applied to multi-modal data analysis -- not necessarily because they offer the most precise answers, but because they provide fluent, flexible interfaces for interpreting complex inputs. Yet this fluency often conceals a deeper structural failure: the prevailing ``Prompt-to-Answer'' paradigm treats LLMs as black-box analysts, collapsing evidence, reasoning, and conclusions into a single, opaque response. The result is brittle, unverifiable, and frequently misleading. We argue for a fundamental shift: from generation to structured extraction, from monolithic prompts to modular, agent-based workflows. LLMs should not serve as oracles, but as collaborators -- specialized in tasks like extraction, translation, and linkage -- embedded within transparent workflows that enable step-by-step reasoning and verification. We propose DataPuzzle, a conceptual multi-agent framework that decomposes complex questions, structures information into interpretable forms (e.g. tables, graphs), and coordinates agent roles to support transparent and verifiable analysis. This framework serves as an aspirational blueprint for restoring visibility and control in LLM-driven analytics -- transforming opaque answers into traceable processes, and brittle fluency into accountable insight. This is not a marginal refinement; it is a call to reimagine how we build trustworthy, auditable analytic systems in the era of large language models. Structure is not a constraint -- it is the path to clarity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23630v1">Game-Oriented ASR Error Correction via RAG-Enhanced LLM</a></div>
    <div class="paper-meta">
      📅 2025-09-28
    </div>
    <details class="paper-abstract">
      With the rise of multiplayer online games, real-time voice communication is essential for team coordination. However, general ASR systems struggle with gaming-specific challenges like short phrases, rapid speech, jargon, and noise, leading to frequent errors. To address this, we propose the GO-AEC framework, which integrates large language models, Retrieval-Augmented Generation (RAG), and a data augmentation strategy using LLMs and TTS. GO-AEC includes data augmentation, N-best hypothesis-based correction, and a dynamic game knowledge base. Experiments show GO-AEC reduces character error rate by 6.22% and sentence error rate by 29.71%, significantly improving ASR accuracy in gaming scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23619v1">Reasoning Scaffolding: Distilling the Flow of Thought from LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-28
    </div>
    <details class="paper-abstract">
      The prevailing approach to distilling reasoning from Large Language Models (LLMs)-behavioral cloning from textual rationales-is fundamentally limited. It teaches Small Language Models (SLMs) to mimic surface-level patterns rather than the underlying algorithmic structure of thought, resulting in a critical lack of logical robustness. We argue that instead of cloning text, distillation should transfer this algorithmic structure directly. We introduce Reasoning Scaffolding}, a framework that reframes reasoning as a structured generation process. Our method first abstracts the teacher's thought process into a sequence of discrete, interpretable semantic signals (e.g., Contrast, Addition) that act as a scaffold. The student model is then trained via a multi-task objective to both (1)predict the next semantic signal, anticipating the reasoning flow, and (2)generate the corresponding step, conditioned on that signal. This multi-task scheme acts as a powerful regularizer, compelling the student to internalize the computational patterns of coherent reasoning. On a suite of challenging reasoning benchmarks, our method significantly outperforms state-of-the-art distillation in both accuracy and logical consistency, providing a path towards creating smaller models that are genuine reasoners, not just fluent mimics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23614v1">PSG-Agent: Personality-Aware Safety Guardrail for LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2025-09-28
    </div>
    <details class="paper-abstract">
      Effective guardrails are essential for safely deploying LLM-based agents in critical applications. Despite recent advances, existing guardrails suffer from two fundamental limitations: (i) they apply uniform guardrail policies to all users, ignoring that the same agent behavior can harm some users while being safe for others; (ii) they check each response in isolation, missing how risks evolve and accumulate across multiple interactions. To solve these issues, we propose PSG-Agent, a personalized and dynamic system for LLM-based agents. First, PSG-Agent creates personalized guardrails by mining the interaction history for stable traits and capturing real-time states from current queries, generating user-specific risk thresholds and protection strategies. Second, PSG-Agent implements continuous monitoring across the agent pipeline with specialized guards, including Plan Monitor, Tool Firewall, Response Guard, Memory Guardian, that track cross-turn risk accumulation and issue verifiable verdicts. Finally, we validate PSG-Agent in multiple scenarios including healthcare, finance, and daily life automation scenarios with diverse user profiles. It significantly outperform existing agent guardrails including LlamaGuard3 and AGrail, providing an executable and auditable path toward personalized safety for LLM-based agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10624v2">Comprehension Without Competence: Architectural Limits of LLMs in Symbolic Computation and Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-28
      | 💬 v2: Two TMLR revision rounds addressing reviewer feedback. Added real-world validation (3.4), interpretability analysis (7), computational hallucination framework, strengthened theory. v3: Sec 3.2 - added transformer architecture diagram, clarified UAT capacity vs computational limits, improved role specialization theorem presentation
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) display striking surface fluency yet systematically fail at tasks requiring symbolic reasoning, arithmetic accuracy, and logical consistency. This paper offers a structural diagnosis of such failures, revealing a persistent gap between \textit{comprehension} and \textit{competence}. Through controlled experiments and architectural analysis, we demonstrate that LLMs often articulate correct principles without reliably applying them--a failure rooted not in knowledge access, but in computational execution. We term this phenomenon the computational \textit{split-brain syndrome}, where instruction and action pathways are geometrically and functionally dissociated. This core limitation recurs across domains, from mathematical operations to relational inferences, and explains why model behavior remains brittle even under idealized prompting. We argue that LLMs function as powerful pattern completion engines, but lack the architectural scaffolding for principled, compositional reasoning. Our findings delineate the boundary of current LLM capabilities and motivate future models with metacognitive control, principle lifting, and structurally grounded execution. This diagnosis also clarifies why mechanistic interpretability findings may reflect training-specific pattern coordination rather than universal computational principles, and why the geometric separation between instruction and execution pathways suggests limitations in neural introspection and mechanistic analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04117v2">Unveiling Over-Memorization in Finetuning LLMs for Reasoning Tasks</a></div>
    <div class="paper-meta">
      📅 2025-09-28
    </div>
    <details class="paper-abstract">
      The pretrained large language models (LLMs) are finetuned with labeled data for better instruction following ability and alignment with human values. In this paper, we study the learning dynamics of LLM finetuning on reasoning tasks and reveal the uncovered over-memorization phenomenon during a specific stage of LLM finetuning. At this stage, the LLMs have excessively memorized training data and exhibit high test perplexity while maintaining good test accuracy. We explore the conditions that contribute to over-memorization and discover that this issue is prevalent across various tasks, models, and fine-tuning methods, with prolonged training and large learning rates exacerbating the problem. Although models with over-memorization demonstrate comparable test accuracy to normal models, they suffer from reduced robustness, poor out-of-distribution generalization, and decreased generation diversity. In light of our findings on over-memorization, we offer recommendations for checkpoint selection and propose techniques such as checkpoint merging and memorization-aware reweighting to mitigate this effect.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23586v1">Improving the Efficiency of LLM Agent Systems through Trajectory Reduction</a></div>
    <div class="paper-meta">
      📅 2025-09-28
      | 💬 20 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Multi-turn agent systems based on Large Language Models (LLMs) have been increasingly popular for software engineering tasks. While LLM agents show decent effectiveness, the high computational cost of input tokens due to the ever-growing trajectory remains an efficiency concern for their applications. Efficiency is largely neglected in existing studies and agent products, and this paper fills the gap by introducing an inference-time trajectory reduction approach to reduce the cost of agents. Through analyzing existing agent trajectories, we demonstrate that useless, redundant, and expired information is widespread in all trajectories, which can be identified and reduced without harming the agent's performance. We then design a simple yet effective trajectory reduction approach, AgentDiet, which automatically removes such waste information. We implement AgentDiet on a top-performing coding agent, and the evaluation on two LLMs and two benchmarks shows that AgentDiet can reduce input tokens by 39.9% ~ 59.7%, or the final computational cost by 21.1% ~ 35.9%, while maintaining the same agent performance. This indicates that trajectory reduction is a promising direction for agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23580v1">LLM Hallucination Detection: HSAD</a></div>
    <div class="paper-meta">
      📅 2025-09-28
      | 💬 in Chinese language
    </div>
    <details class="paper-abstract">
      Although Large Language Models have demonstrated powerful capabilities in a wide range of tasks such as language understanding and code generation, the frequent occurrence of hallucinations during the generation process has become a significant impediment to their deployment in critical application scenarios. Current mainstream hallucination detection methods rely on factual consistency verification or static hidden layer features. The former is constrained by the scope of knowledge coverage, while the latter struggles to capture reasoning biases during the inference process. To address these issues, and inspired by signal analysis methods in cognitive neuroscience, this paper proposes a hallucination detection method based on the frequency-domain analysis of hidden layer temporal signals, named HSAD (\textbf{H}idden \textbf{S}ignal \textbf{A}nalysis-based \textbf{D}etection). First, by treating the LLM's reasoning process as a cognitive journey that unfolds over time, we propose modeling and simulating the human process of signal perception and discrimination in a deception-detection scenario through hidden layer temporal signals. Next, The Fast Fourier Transform is applied to map these temporal signals into the frequency domain to construct spectral features, which are used to capture anomalies that arise during the reasoning process; analysis experiments on these spectral features have proven the effectiveness of this approach. Finally, a hallucination detection algorithm is designed based on these spectral features to identify hallucinations in the generated content. By effectively combining the modeling of the reasoning process with frequency-domain feature extraction, the HSAD method overcomes the limitations of existing approaches in terms of knowledge coverage and the detection of reasoning biases, demonstrating higher detection accuracy and robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23573v1">Uncovering Vulnerabilities of LLM-Assisted Cyber Threat Intelligence</a></div>
    <div class="paper-meta">
      📅 2025-09-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are intensively used to assist security analysts in counteracting the rapid exploitation of cyber threats, wherein LLMs offer cyber threat intelligence (CTI) to support vulnerability assessment and incident response. While recent work has shown that LLMs can support a wide range of CTI tasks such as threat analysis, vulnerability detection, and intrusion defense, significant performance gaps persist in practical deployments. In this paper, we investigate the intrinsic vulnerabilities of LLMs in CTI, focusing on challenges that arise from the nature of the threat landscape itself rather than the model architecture. Using large-scale evaluations across multiple CTI benchmarks and real-world threat reports, we introduce a novel categorization methodology that integrates stratification, autoregressive refinement, and human-in-the-loop supervision to reliably analyze failure instances. Through extensive experiments and human inspections, we reveal three fundamental vulnerabilities: spurious correlations, contradictory knowledge, and constrained generalization, that limit LLMs in effectively supporting CTI. Subsequently, we provide actionable insights for designing more robust LLM-powered CTI systems to facilitate future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23571v1">Benchmarking LLM-Assisted Blue Teaming via Standardized Threat Hunting</a></div>
    <div class="paper-meta">
      📅 2025-09-28
    </div>
    <details class="paper-abstract">
      As cyber threats continue to grow in scale and sophistication, blue team defenders increasingly require advanced tools to proactively detect and mitigate risks. Large Language Models (LLMs) offer promising capabilities for enhancing threat analysis. However, their effectiveness in real-world blue team threat-hunting scenarios remains insufficiently explored. This paper presents CyberTeam, a benchmark designed to guide LLMs in blue teaming practice. CyberTeam constructs a standardized workflow in two stages. First, it models realistic threat-hunting workflows by capturing the dependencies among analytical tasks from threat attribution to incident response. Next, each task is addressed through a set of operational modules tailored to its specific analytical requirements. This transforms threat hunting into a structured sequence of reasoning steps, with each step grounded in a discrete operation and ordered according to task-specific dependencies. Guided by this framework, LLMs are directed to perform threat-hunting tasks through modularized steps. Overall, CyberTeam integrates 30 tasks and 9 operational modules to guide LLMs through standardized threat analysis. We evaluate both leading LLMs and state-of-the-art cybersecurity agents, comparing CyberTeam against open-ended reasoning strategies. Our results highlight the improvements enabled by standardized design, while also revealing the limitations of open-ended reasoning in real-world threat hunting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23570v1">Improving constraint-based discovery with robust propagation and reliable LLM priors</a></div>
    <div class="paper-meta">
      📅 2025-09-28
    </div>
    <details class="paper-abstract">
      Learning causal structure from observational data is central to scientific modeling and decision-making. Constraint-based methods aim to recover conditional independence (CI) relations in a causal directed acyclic graph (DAG). Classical approaches such as PC and subsequent methods orient v-structures first and then propagate edge directions from these seeds, assuming perfect CI tests and exhaustive search of separating subsets -- assumptions often violated in practice, leading to cascading errors in the final graph. Recent work has explored using large language models (LLMs) as experts, prompting sets of nodes for edge directions, and could augment edge orientation when assumptions are not met. However, such methods implicitly assume perfect experts, which is unrealistic for hallucination-prone LLMs. We propose MosaCD, a causal discovery method that propagates edges from a high-confidence set of seeds derived from both CI tests and LLM annotations. To filter hallucinations, we introduce shuffled queries that exploit LLMs' positional bias, retaining only high-confidence seeds. We then apply a novel confidence-down propagation strategy that orients the most reliable edges first, and can be integrated with any skeleton-based discovery method. Across multiple real-world graphs, MosaCD achieves higher accuracy in final graph construction than existing constraint-based methods, largely due to the improved reliability of initial seeds and robust propagation strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23564v1">Clean First, Align Later: Benchmarking Preference Data Cleaning for Reliable LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-09-28
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Human feedback plays a pivotal role in aligning large language models (LLMs) with human preferences. However, such feedback is often noisy or inconsistent, which can degrade the quality of reward models and hinder alignment. While various automated data cleaning methods have been proposed to mitigate this issue, a systematic evaluation of their effectiveness and generalizability remains lacking. To bridge this gap, we introduce the first comprehensive benchmark for evaluating 13 preference data cleaning methods in the context of LLM alignment. PrefCleanBench offers a standardized protocol to assess cleaning strategies in terms of alignment performance and generalizability across diverse datasets, model architectures, and optimization algorithms. By unifying disparate methods and rigorously comparing them, we uncover key factors that determine the success of data cleaning in alignment tasks. This benchmark lays the groundwork for principled and reproducible approaches to improving LLM alignment through better data quality-highlighting the crucial but underexplored role of data preprocessing in responsible AI development. We release modular implementations of all methods to catalyze further research: https://github.com/deeplearning-wisc/PrefCleanBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23558v1">Formalization Driven LLM Prompt Jailbreaking via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-28
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable capabilities, yet they also introduce novel security challenges. For instance, prompt jailbreaking attacks involve adversaries crafting sophisticated prompts to elicit responses from LLMs that deviate from human values. To uncover vulnerabilities in LLM alignment methods, we propose the PASS framework (\underline{P}rompt J\underline{a}ilbreaking via \underline{S}emantic and \underline{S}tructural Formalization). Specifically, PASS employs reinforcement learning to transform initial jailbreak prompts into formalized descriptions, which enhances stealthiness and enables bypassing existing alignment defenses. The jailbreak outputs are then structured into a GraphRAG system that, by leveraging extracted relevant terms and formalized symbols as contextual input alongside the original query, strengthens subsequent attacks and facilitates more effective jailbreaks. We conducted extensive experiments on common open-source models, demonstrating the effectiveness of our attack.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23542v1">On the Shelf Life of Fine-Tuned LLM Judges: Future Proofing, Backward Compatibility, and Question Generalization</a></div>
    <div class="paper-meta">
      📅 2025-09-28
      | 💬 21 pages
    </div>
    <details class="paper-abstract">
      The LLM-as-a-judge paradigm is widely used in both evaluating free-text model responses and reward modeling for model alignment and finetuning. Recently, finetuning judges with judge-specific data has emerged as an often preferred choice over directly prompting frontier models as judges, as the former achieves better performance with smaller model sizes while being more robust to common biases. However, the standard evaluation ignores several practical concerns of finetuned judges regarding their real world deployment. In this paper, we identify and formalize three aspects that affect the shelf life of these judges: future proofing and backward compatibility -- how well judges finetuned on responses by today's generator models perform on responses by future models or past models, as well as question generalization -- how well judges generalize to unseen questions at test time. We study these three aspects in the math domain under a unified framework with varying train and test distributions, three SFT- and DPO-based finetuning algorithms and three different base models. Experiments suggest that future-proofing is challenging for most models, while backward compatibility is relatively easy, with DPO-trained models consistently improving performance. We further find that continual learning provides a more balanced adaptation to shifts between older and newer response distributions than training solely on stronger or weaker responses. Moreover, all models observe certain degrees of performance degradation when moving from questions seen during training to unseen ones, showing that current judges do not fully generalize to unseen questions. These findings provide insights into practical considerations for developing and deploying judge models in the face of ever-changing generators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.12948v2">Probabilistic Soundness Guarantees in LLM Reasoning Chains</a></div>
    <div class="paper-meta">
      📅 2025-09-28
      | 💬 EMNLP 2025 camera ready
    </div>
    <details class="paper-abstract">
      In reasoning chains generated by large language models (LLMs), initial errors often propagate and undermine the reliability of the final conclusion. Current LLM-based error detection methods often fail to detect propagated errors because earlier errors can corrupt judgments of downstream reasoning. To better detect such errors, we introduce Autoregressive Reasoning Entailment Stability (ARES), a probabilistic framework that evaluates each reasoning step based solely on previously-verified premises. This inductive method yields a nuanced score for each step and provides certified statistical guarantees of its soundness, rather than a brittle binary label. ARES achieves state-of-the-art performance across four benchmarks (72.1% Macro-F1, +8.2 points) and demonstrates superior robustness on very long synthetic reasoning chains, where it excels at detecting propagated errors (90.3% F1, +27.6 points).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23537v1">Beyond the Strongest LLM: Multi-Turn Multi-Agent Orchestration vs. Single LLMs on Benchmarks</a></div>
    <div class="paper-meta">
      📅 2025-09-28
      | 💬 9 pages, 3 tables, 1 figure
    </div>
    <details class="paper-abstract">
      We study multi-turn multi-agent orchestration, where multiple large language model (LLM) agents interact over multiple turns by iteratively proposing answers or casting votes until reaching consensus. Using four LLMs (Gemini 2.5 Pro, GPT-5, Grok 4, and Claude Sonnet 4) on GPQA-Diamond, IFEval, and MuSR, we conduct two experiments: (i) benchmarking orchestration against single-LLM baselines; and (ii) ablations on GPQA-Diamond that vary whether agents see who authored answers and whether they can observe ongoing votes. Orchestration matches or exceeds the strongest single model and consistently outperforms the others. Analysis of best-achievable orchestration performance shows potential for further gains. The ablations show that revealing authorship increases self-voting and ties, and that showing ongoing votes amplifies herding, which speeds convergence but can sometimes yield premature consensus.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17508v2">On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-28
      | 💬 Project Page: https://github.com/complex-reasoning/RPG
    </div>
    <details class="paper-abstract">
      Policy gradient algorithms have been successfully applied to enhance the reasoning capabilities of large language models (LLMs). KL regularization is ubiquitous, yet the design surface, choice of KL direction (forward vs. reverse), normalization (normalized vs. unnormalized), and estimator ($k_1/k_2/k_3$), is scattered across the literature and often intertwined with off-policy estimation. We ask a focused question: under the off-policy setting, what weighting is required for each KL variant so that the surrogate we optimize yields the exact gradient of the intended KL-regularized objective? We answer this with a compact, unified derivation we call the Regularized Policy Gradient (RPG) view. RPG (i) unifies normalized and unnormalized KL variants and shows that the widely-used $k_3$ penalty is exactly the unnormalized KL; (ii) specifies conditions under which REINFORCE-style losses with stop-gradient are gradient-equivalent to fully differentiable surrogates; (iii) identifies and corrects an off-policy importance-weighting mismatch in GRPO's KL term; and (iv) introduces RPG-Style Clip, a truncated-importance-sampling step within RPG-REINFORCE that enables stable, off-policy policy-gradient training at scale. On mathematical reasoning benchmarks (AIME24, AIME25), RPG-REINFORCE with RPG-Style Clip improves accuracy by up to $+6$ absolute percentage points over DAPO. Notably, RPG is a stable and scalable RL algorithm for LLM reasoning, realized via (a) a KL-correct objective, (b) truncated importance sampling, and (c) an iterative reference-policy update scheme.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08885v3">AdversariaL attacK sAfety aLIgnment(ALKALI): Safeguarding LLMs through GRACE: Geometric Representation-Aware Contrastive Enhancement- Introducing Adversarial Vulnerability Quality Index (AVQI)</a></div>
    <div class="paper-meta">
      📅 2025-09-28
    </div>
    <details class="paper-abstract">
      Adversarial threats against LLMs are escalating faster than current defenses can adapt. We expose a critical geometric blind spot in alignment: adversarial prompts exploit latent camouflage, embedding perilously close to the safe representation manifold while encoding unsafe intent thereby evading surface level defenses like Direct Preference Optimization (DPO), which remain blind to the latent geometry. We introduce ALKALI, the first rigorously curated adversarial benchmark and the most comprehensive to date spanning 9,000 prompts across three macro categories, six subtypes, and fifteen attack families. Evaluation of 21 leading LLMs reveals alarmingly high Attack Success Rates (ASRs) across both open and closed source models, exposing an underlying vulnerability we term latent camouflage, a structural blind spot where adversarial completions mimic the latent geometry of safe ones. To mitigate this vulnerability, we introduce GRACE - Geometric Representation Aware Contrastive Enhancement, an alignment framework coupling preference learning with latent space regularization. GRACE enforces two constraints: latent separation between safe and adversarial completions, and adversarial cohesion among unsafe and jailbreak behaviors. These operate over layerwise pooled embeddings guided by a learned attention profile, reshaping internal geometry without modifying the base model, and achieve up to 39% ASR reduction. Moreover, we introduce AVQI, a geometry aware metric that quantifies latent alignment failure via cluster separation and compactness. AVQI reveals when unsafe completions mimic the geometry of safe ones, offering a principled lens into how models internally encode safety. We make the code publicly available at https://anonymous.4open.science/r/alkali-B416/README.md.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24116v1">Dual-Scale World Models for LLM Agents Towards Hard-Exploration Problems</a></div>
    <div class="paper-meta">
      📅 2025-09-28
    </div>
    <details class="paper-abstract">
      LLM-based agents have seen promising advances, yet they are still limited in "hard-exploration" tasks requiring learning new knowledge through exploration. We present GLoW, a novel approach leveraging dual-scale world models, maintaining a trajectory frontier of high-value discoveries at the global scale, while learning from local trial-and-error in exploration through a Multi-path Advantage Reflection mechanism which infers advantage-based progress signals to guide exploration. To evaluate our framework for hard-exploration, we tackle the Jericho benchmark suite of text-based games, where GLoW achieves a new state-of-theart performance for LLM-based approaches. Compared to state-of-the-art RLbased methods, our approach achieves comparable performance while requiring 100-800x fewer environment interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.12726v2">DESIGNER: Design-Logic-Guided Multidisciplinary Data Synthesis for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-28
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable success in many natural language tasks but still struggle with complex, multi-step reasoning, particularly across diverse disciplines. Existing reasoning datasets often lack disciplinary breadth, reasoning depth, and diversity, and lack guiding principles for question synthesis. We propose DESIGNER: a DESIGN-logic-guidEd Reasoning data synthesis pipeline that leverages naturally available, extensive raw documents (e.g., book corpus and web corpus) to generate multidisciplinary challenging questions. We introduce the concept of "design logic" and instruct LLMs to mimic human educators' question-creation process, enabling automated synthesis of large-scale, high-difficulty questions. We use LLMs to reverse-engineer and abstract over 120,000 design logics from existing questions across various disciplines. By matching these design logics with source documents, we are able to create reasoning questions that far surpass the difficulty and diversity of existing datasets. Using this pipeline, we synthesized two large-scale reasoning datasets that span 75 disciplines: DLR-Book (3.04 million questions from the book corpus) and DLR-Web (1.66 million questions from the web corpus). Data analysis indicates that the questions synthesized by our method exhibit greater difficulty and diversity compared to those in the baseline datasets. We validate our synthesized data through supervised fine-tuning (SFT) on the Qwen3 and Llama3 model families. Our data substantially enhances their multidisciplinary reasoning capabilities, outperforming existing datasets. Notably, after SFT on our datasets, the base versions of these models even surpass their official instruction-tuned counterparts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02452v2">Do LLMs Adhere to Label Definitions? Examining Their Receptivity to External Label Definitions</a></div>
    <div class="paper-meta">
      📅 2025-09-28
      | 💬 EMNLP 2025 (Main Conference)
    </div>
    <details class="paper-abstract">
      Do LLMs genuinely incorporate external definitions, or do they primarily rely on their parametric knowledge? To address these questions, we conduct controlled experiments across multiple explanation benchmark datasets (general and domain-specific) and label definition conditions, including expert-curated, LLM-generated, perturbed, and swapped definitions. Our results reveal that while explicit label definitions can enhance accuracy and explainability, their integration into an LLM's task-solving processes is neither guaranteed nor consistent, suggesting reliance on internalized representations in many cases. Models often default to their internal representations, particularly in general tasks, whereas domain-specific tasks benefit more from explicit definitions. These findings underscore the need for a deeper understanding of how LLMs process external knowledge alongside their pre-existing capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24101v1">BTC-SAM: Leveraging LLMs for Generation of Bias Test Cases for Sentiment Analysis Models</a></div>
    <div class="paper-meta">
      📅 2025-09-28
      | 💬 Accepted at EMNLP 2025 main conference
    </div>
    <details class="paper-abstract">
      Sentiment Analysis (SA) models harbor inherent social biases that can be harmful in real-world applications. These biases are identified by examining the output of SA models for sentences that only vary in the identity groups of the subjects. Constructing natural, linguistically rich, relevant, and diverse sets of sentences that provide sufficient coverage over the domain is expensive, especially when addressing a wide range of biases: it requires domain experts and/or crowd-sourcing. In this paper, we present a novel bias testing framework, BTC-SAM, which generates high-quality test cases for bias testing in SA models with minimal specification using Large Language Models (LLMs) for the controllable generation of test sentences. Our experiments show that relying on LLMs can provide high linguistic variation and diversity in the test sentences, thereby offering better test coverage compared to base prompting methods even for previously unseen biases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19475v2">Automatic Question & Answer Generation Using Generative Large Language Model (LLM)</a></div>
    <div class="paper-meta">
      📅 2025-09-28
    </div>
    <details class="paper-abstract">
      In the realm of education, student evaluation holds equal significance to imparting knowledge. To be evaluated, students usually need to go through text-based academic assessment methods. Instructors need to make a diverse set of questions that need to be fair for all students to prove their adequacy over a particular topic. This can prove to be quite challenging as they may need to manually go through several different lecture materials. Our objective is to make this whole process much easier by implementing Automatic Question Answer Generation(AQAG), using a fine-tuned generative LLM. For tailoring the instructor's preferred question style (MCQ, conceptual, or factual questions), Prompt Engineering (PE) is being utilized. In this research, we propose to leverage unsupervised learning methods in NLP, primarily focusing on the English language. This approach empowers the base Meta-Llama 2-7B model to integrate the RACE dataset as training data for the fine-tuning process. Creating a customized model that will offer efficient solutions for educators, instructors, and individuals engaged in text-based evaluations. A reliable and efficient tool for generating questions and answers can free up valuable time and resources, thus streamlining their evaluation processes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24090v1">Large-Scale Constraint Generation -- Can LLMs Parse Hundreds of Constraints?</a></div>
    <div class="paper-meta">
      📅 2025-09-28
    </div>
    <details class="paper-abstract">
      Recent research has explored the constrained generation capabilities of Large Language Models (LLMs) when explicitly prompted by few task-specific requirements. In contrast, we introduce Large-Scale Constraint Generation (LSCG), a new problem that evaluates whether LLMs can parse a large, fine-grained, generic list of constraints. To examine the LLMs' ability to handle an increasing number constraints, we create a practical instance of LSCG, called Words Checker. In Words Checker, we evaluate the impact of model characteristics (e.g., size, family) and steering techniques (e.g., Simple Prompt, Chain of Thought, Best of N) on performance. We also propose FoCusNet, a small and dedicated model that parses the original list of constraints into a smaller subset, helping the LLM focus on relevant constraints. Experiments reveal that existing solutions suffer a significant performance drop as the number of constraints increases, with FoCusNet showing an 8-13% accuracy boost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24086v1">Do Repetitions Matter? Strengthening Reliability in LLM Evaluations</a></div>
    <div class="paper-meta">
      📅 2025-09-28
    </div>
    <details class="paper-abstract">
      LLM leaderboards often rely on single stochastic runs, but how many repetitions are required for reliable conclusions remains unclear. We re-evaluate eight state-of-the-art models on the AI4Math Benchmark with three independent runs per setting. Using mixed-effects logistic regression, domain-level marginal means, rank-instability analysis, and run-to-run reliability, we assessed the value of additional repetitions. Our findings shows that Single-run leaderboards are brittle: 10/12 slices (83\%) invert at least one pairwise rank relative to the three-run majority, despite a zero sign-flip rate for pairwise significance and moderate overall interclass correlation. Averaging runs yields modest SE shrinkage ($\sim$5\% from one to three) but large ranking gains; two runs remove $\sim$83\% of single-run inversions. We provide cost-aware guidance for practitioners: treat evaluation as an experiment, report uncertainty, and use $\geq 2$ repetitions under stochastic decoding. These practices improve robustness while remaining feasible for small teams and help align model comparisons with real-world reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24085v1">PEARL: Peer-Enhanced Adaptive Radio via On-Device LLM</a></div>
    <div class="paper-meta">
      📅 2025-09-28
    </div>
    <details class="paper-abstract">
      We present PEARL (Peer-Enhanced Adaptive Radio via On-Device LLM), a framework for cooperative cross-layer optimization in device-to-device (D2D) communication. Building on our previous work on single-device on-device LLMs, PEARL extends the paradigm by leveraging both publisher and subscriber states to guide Wi-Fi Aware (WA) parameter selection. A context-aware reward, which normalizes latency by application tolerances and modulates energy by device battery states, provides richer supervision for KL-based finetuning. We study two lightweight variants: PEARL (Head + Low-Rank Adaptation (LoRA)) achieves the best overall performance, while PEARL-Lite (Head-only) delivers sub-20 ms inference at near-identical objective scores. Across synthetic scenarios grounded in real measurements, PEARL improves objective scores over heuristic and compact model baselines and reduces energy by up to 16% in cooperative low-battery cases. These results demonstrate that peer-aware context, reward-aligned training, and head-based efficiency make LLMs practical for always-on, on-device cross-layer control.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24068v1">A Small Math Model: Recasting Strategy Choice Theory in an LLM-Inspired Architecture</a></div>
    <div class="paper-meta">
      📅 2025-09-28
    </div>
    <details class="paper-abstract">
      Strategy Choice Theory (SCT)\footnote{``Strategy Choice Theory'', ``Distributions of Associations'', and ``Overlapping Wave Theory'' have been used to refer to this line of work, emphasizing different aspects.}\citep[e.g.,][]{siegler1984strategychoices, siegler2000rebirth} explains important aspects of children's arithmetic learning based upon principles including learning from developmentally naturalistic data, probabilistic representation, confidence-based retrieval, and the phase-like importance of scaffolding strategies, such as finger-counting. Here we recast SCT as a ``Small Math Model'' (SMM), employing a neural-network-based architecture analogous to LLMs. The SMM extends SCT to include counting practice\footnote{The original SCT model was pre-biased in accordance with the supposed experience of counting.}, symbol (number) embedding, and gated attention. Similar to earlier work, the SMM demonstrates constructive and destructive interference between counting and addition, and the ``wave-like'' use of finger-counting as sum recall improves. We plan to extend the SMM to later aspects of the decades-long SCT program, including adaptive strategy choice and eventually strategy discovery, providing a unified platform to investigate the understanding of numerical characteristics and relationships essential for mathematical reasoning -- as it can emerge in LLM-based agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12839v2">From An LLM Swarm To A PDDL-Empowered HIVE: Planning Self-Executed Instructions In A Multi-Modal Jungle</a></div>
    <div class="paper-meta">
      📅 2025-09-28
      | 💬 Published as a conference paper at ICLR 2025
    </div>
    <details class="paper-abstract">
      In response to the call for agent-based solutions that leverage the ever-increasing capabilities of the deep models' ecosystem, we introduce Hive -- a comprehensive solution for knowledge-aware planning of a set of atomic actions to address input queries and subsequently selecting appropriate models accordingly. Hive operates over sets of models and, upon receiving natural language instructions (i.e. user queries), schedules and executes explainable plans of atomic actions. These actions can involve one or more of the available models to achieve the overall task, while respecting end-users specific constraints. Notably, Hive handles tasks that involve multi-modal inputs and outputs, enabling it to handle complex, real-world queries. Our system is capable of planning complex chains of actions while guaranteeing explainability, using an LLM-based formal logic backbone empowered by PDDL operations. We introduce the MuSE benchmark in order to offer a comprehensive evaluation of the multi-modal capabilities of agent systems. Our findings show that our framework redefines the state-of-the-art for task selection, outperforming other competing systems that plan operations across multiple models while offering transparency guarantees while fully adhering to user constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.19389v3">Federated Sketching LoRA: A Flexible Framework for Heterogeneous Collaborative Fine-Tuning of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-28
      | 💬 We propose Federated Sketching LoRA (FSLoRA), a theoretically grounded methodology for collaborative LLM fine-tuning that retains LoRA's flexibility while adapting to the communication and computational capabilities of individual clients
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) on resource-constrained clients remains a challenging problem. Recent works have fused low-rank adaptation (LoRA) techniques with federated fine-tuning to mitigate challenges associated with client model sizes and data scarcity. Still, the heterogeneity of resources remains a critical bottleneck: while higher-rank modules generally enhance performance, varying client capabilities constrain LoRA's feasible rank range. Existing approaches attempting to resolve this issue either lack analytical justification or impose additional computational overhead, leaving a wide gap for efficient and theoretically-grounded solutions. To address these challenges, we propose federated sketching LoRA (FSLoRA), which leverages a sketching mechanism to enable clients to selectively update submatrices of global LoRA modules maintained by the server. By adjusting the sketching ratios, which determine the ranks of the submatrices on the clients, FSLoRA flexibly adapts to client-specific communication and computational constraints. We provide a rigorous convergence analysis of FSLoRA that characterizes how the sketching ratios affect the convergence rate. Through comprehensive experiments on multiple datasets and LLM models, we demonstrate FSLoRA's performance improvements compared to various baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24050v1">Collaborative Device-Cloud LLM Inference through Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-28
      | 💬 We propose a unified post-training framework that integrates routing optimization, enabling the on-device LLM to improve its problem-solving ability while learning routing strategies
    </div>
    <details class="paper-abstract">
      Device-cloud collaboration has emerged as a promising paradigm for deploying large language models (LLMs), combining the efficiency of lightweight on-device inference with the superior performance of powerful cloud LLMs. An essential problem in this scenario lies in deciding whether a given query is best handled locally or delegated to the cloud. Existing approaches typically rely on external routers, implemented as binary classifiers, which often struggle to determine task difficulty from the prompt's surface pattern. To address these limitations, we propose a framework where the on-device LLM makes routing decisions at the end of its solving process, with this capability instilled through post-training. In particular, we formulate a reward maximization problem with carefully designed rewards that encourage effective problem solving and judicious offloading to the cloud. To solve this problem, we develop a group-adaptive policy gradient algorithm, featuring a group-level policy gradient, designed to yield an unbiased gradient estimator of the reward, and adaptive prompt filtering, developed to enforce the constraint on cloud LLM usage. Extensive experiments across models and benchmarks show that the proposed methodology consistently outperforms existing baselines and significantly narrows the gap to full cloud LLM performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24046v1">PartnerMAS: An LLM Hierarchical Multi-Agent Framework for Business Partner Selection on High-Dimensional Features</a></div>
    <div class="paper-meta">
      📅 2025-09-28
    </div>
    <details class="paper-abstract">
      High-dimensional decision-making tasks, such as business partner selection, involve evaluating large candidate pools with heterogeneous numerical, categorical, and textual features. While large language models (LLMs) offer strong in-context reasoning capabilities, single-agent or debate-style systems often struggle with scalability and consistency in such settings. We propose PartnerMAS, a hierarchical multi-agent framework that decomposes evaluation into three layers: a Planner Agent that designs strategies, Specialized Agents that perform role-specific assessments, and a Supervisor Agent that integrates their outputs. To support systematic evaluation, we also introduce a curated benchmark dataset of venture capital co-investments, featuring diverse firm attributes and ground-truth syndicates. Across 140 cases, PartnerMAS consistently outperforms single-agent and debate-based multi-agent baselines, achieving up to 10--15\% higher match rates. Analysis of agent reasoning shows that planners are most responsive to domain-informed prompts, specialists produce complementary feature coverage, and supervisors play an important role in aggregation. Our findings demonstrate that structured collaboration among LLM agents can generate more robust outcomes than scaling individual models, highlighting PartnerMAS as a promising framework for high-dimensional decision-making in data-rich domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14003v3">Unlearning Isn't Invisible: Detecting Unlearning Traces in LLMs from Model Outputs</a></div>
    <div class="paper-meta">
      📅 2025-09-28
    </div>
    <details class="paper-abstract">
      Machine unlearning (MU) for large language models (LLMs), commonly referred to as LLM unlearning, seeks to remove specific undesirable data or knowledge from a trained model, while maintaining its performance on standard tasks. While unlearning plays a vital role in protecting data privacy, enforcing copyright, and mitigating sociotechnical harms in LLMs, we identify a new vulnerability post-unlearning: unlearning trace detection. We discover that unlearning leaves behind persistent ''fingerprints'' in LLMs, detectable traces in both model behavior and internal representations. These traces can be identified from output responses, even when prompted with forget-irrelevant inputs. Specifically, even a simple supervised classifier can determine whether a model has undergone unlearning, using only its prediction logits or even its textual outputs. Further analysis shows that these traces are embedded in intermediate activations and propagate nonlinearly to the final layer, forming low-dimensional, learnable manifolds in activation space. Through extensive experiments, we demonstrate that unlearning traces can be detected with over 90% accuracy even under forget-irrelevant inputs, and that larger LLMs exhibit stronger detectability. These findings reveal that unlearning leaves measurable signatures, introducing a new risk of reverse-engineering forgotten information when a model is identified as unlearned, given an input query.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06006v4">Optuna vs Code Llama: Are LLMs a New Paradigm for Hyperparameter Tuning?</a></div>
    <div class="paper-meta">
      📅 2025-09-28
    </div>
    <details class="paper-abstract">
      Optimal hyperparameter selection is critical for maximizing the performance of neural networks in computer vision, particularly as architectures become more complex. This work explores the use of large language models (LLMs) for hyperparameter optimization by fine-tuning a parameter-efficient version of Code Llama using LoRA. The resulting model produces accurate and computationally efficient hyperparameter recommendations across a wide range of vision architectures. Unlike traditional methods such as Optuna, which rely on resource-intensive trial-and-error procedures, our approach achieves competitive or superior Root Mean Square Error (RMSE) while substantially reducing computational overhead. Importantly, the models evaluated span image-centric tasks such as classification, detection, and segmentation, fundamental components in many image manipulation pipelines including enhancement, restoration, and style transfer. Our results demonstrate that LLM-based optimization not only rivals established Bayesian methods like Tree-structured Parzen Estimators (TPE), but also accelerates tuning for real-world applications requiring perceptual quality and low-latency processing. All generated configurations are publicly available in the LEMUR Neural Network Dataset (https://github.com/ABrain-One/nn-dataset), which serves as an open source benchmark for hyperparameter optimization research and provides a practical resource to improve training efficiency in image manipulation systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23988v1">LLM/Agent-as-Data-Analyst: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-09-28
      | 💬 35 page, 11 figures
    </div>
    <details class="paper-abstract">
      Large language model (LLM) and agent techniques for data analysis (a.k.a LLM/Agent-as-Data-Analyst) have demonstrated substantial impact in both academica and industry. In comparison with traditional rule or small-model based approaches, (agentic) LLMs enable complex data understanding, natural language interfaces, semantic analysis functions, and autonomous pipeline orchestration. The technical evolution further distills five key design goals for intelligent data analysis agents, namely semantic-aware design, modality-hybrid integration, autonomous pipelines, tool-augmented workflows, and support for open-world tasks. From a modality perspective, we review LLM-based techniques for (i) structured data (e.g., table question answering for relational data and NL2GQL for graph data), (ii) semi-structured data (e.g., markup languages understanding and semi-structured table modeling), (iii) unstructured data (e.g., chart understanding, document understanding, programming languages vulnerable detection), and (iv) heterogeneous data (e.g., data retrieval and modality alignment for data lakes). Finally, we outline the remaining challenges and propose several insights and practical directions for advancing LLM/Agent-powered data analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15182v2">ReflAct: World-Grounded Decision Making in LLM Agents via Goal-State Reflection</a></div>
    <div class="paper-meta">
      📅 2025-09-28
      | 💬 Accepted to EMNLP 2025 (Main Conference)
    </div>
    <details class="paper-abstract">
      Recent advances in LLM agents have largely built on reasoning backbones like ReAct, which interleave thought and action in complex environments. However, ReAct often produces ungrounded or incoherent reasoning steps, leading to misalignment between the agent's actual state and goal. Our analysis finds that this stems from ReAct's inability to maintain consistent internal beliefs and goal alignment, causing compounding errors and hallucinations. To address this, we introduce ReflAct, a novel backbone that shifts reasoning from merely planning next actions to continuously reflecting on the agent's state relative to its goal. By explicitly grounding decisions in states and enforcing ongoing goal alignment, ReflAct dramatically improves strategic reliability. This design delivers substantial empirical gains: ReflAct surpasses ReAct by 27.7% on average, achieving a 93.3% success rate in ALFWorld. Notably, ReflAct even outperforms ReAct with added enhancement modules (e.g., Reflexion, WKM), showing that strengthening the core reasoning backbone is key to reliable agent performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23979v1">ByteSized32Refactored: Towards an Extensible Interactive Text Games Corpus for LLM World Modeling and Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-09-28
      | 💬 14 pages,15 figures, Accepted to the 5th Wordplay: When Language Meets Games Workshop, EMNLP 2025
    </div>
    <details class="paper-abstract">
      Simulating interactive world models remains a core challenge in Large Language Models(LLMs). In this work, we introduce the ByteSized32Refactored, a refactored, modular, and extensible implementation of the original ByteSized32 corpus to explore the task of text game generation. We further optimize the code structure of each text game and create the GameBasic.py foundation library, which centralizes common logic across all 32 games by abstracting 7 base classes (GameObject, etc.) into reusable modules, thereby reducing from 20k to 10k total lines of Python code compared to the original Bytesized32. Our refactored implementation enables extendability - with our centralized design, ByteSized32Refactored can be more efficiently extended to include text games of new scenarios and specifications by reusing the shared logic and functionalities. Extensive experiments with GPT-4o demonstrate a mix of performance - with Bytesized32Refactored, the generated text games for unseen scenarios showcase quality improvements on two of the four evaluation dimensions while decreases on the other two, indicating that the hierarchical structure of the refactored code presents new challenges for LLMs. Overall, we highlight that our extensible code structure, centered on the foundation library and the modular optimization, not only facilitates LLM adaptation to environment specifications but also establishes a scalable environment that supports future extensions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23967v1">HiPO: Hybrid Policy Optimization for Dynamic Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) increasingly rely on chain-of-thought (CoT) reasoning to improve accuracy on complex tasks. However, always generating lengthy reasoning traces is inefficient, leading to excessive token usage and higher inference costs. This paper introduces the Hybrid Policy Optimization (i.e., HiPO), a framework for adaptive reasoning control that enables LLMs to selectively decide when to engage in detailed reasoning (Think-on) and when to respond directly (Think-off). Specifically, HiPO combines a hybrid data pipelineproviding paired Think-on and Think-off responseswith a hybrid reinforcement learning reward system that balances accuracy and efficiency while avoiding over-reliance on detailed reasoning. Experiments across mathematics and coding benchmarks demonstrate that HiPO can substantially reduce token length while maintaining or improving accuracy. Finally, we hope HiPO a can be a principled approach for efficient adaptive reasoning, advancing the deployment of reasoning-oriented LLMs in real-world, resource-sensitive settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21561v2">Reasoning Isn't Enough: Examining Truth-Bias and Sycophancy in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-28
      | 💬 Published at the ICML 2025 Workshop on Models of Human Feedback for AI Alignment
    </div>
    <details class="paper-abstract">
      Despite their widespread use in fact-checking, moderation, and high-stakes decision-making, large language models (LLMs) remain poorly understood as judges of truth. This study presents the largest evaluation to date of LLMs' veracity detection capabilities and the first analysis of these capabilities in reasoning models. We had eight LLMs make 4,800 veracity judgments across several prompts, comparing reasoning and non-reasoning models. We find that rates of truth-bias, or the likelihood to believe a statement is true, regardless of whether it is actually true, are lower in reasoning models than in non-reasoning models, but still higher than human benchmarks. Most concerning, we identify sycophantic tendencies in several advanced models (o4-mini and GPT-4.1 from OpenAI, R1 from DeepSeek), which displayed an asymmetry in detection accuracy, performing well in truth accuracy but poorly in deception accuracy. This suggests that capability advances alone do not resolve fundamental veracity detection challenges in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09677v2">The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-28
    </div>
    <details class="paper-abstract">
      Does continued scaling of large language models (LLMs) yield diminishing returns? In this work, we show that short-task benchmarks may give an illusion of slowing progress, as even marginal gains in single-step accuracy can compound into exponential improvements in the length of tasks a model can successfully complete. Then, we argue that failures of LLMs when simple tasks are made longer arise from mistakes in execution, rather than an inability to reason. So, we propose isolating execution capability, by explicitly providing the knowledge and plan needed to solve a long-horizon task. First, we find that larger models can correctly execute significantly more turns even when small models have near-perfect single-turn accuracy. We then observe that the per-step accuracy of models degrades as the number of steps increases. This is not just due to long-context limitations -- curiously, we observe a self-conditioning effect -- models become more likely to make mistakes when the context contains their errors from prior turns. Self-conditioning does not reduce by just scaling the model size. But, we find that thinking mitigates self-conditioning, and also enables execution of much longer tasks in a single turn. We conclude by benchmarking frontier thinking models on the length of tasks they can execute in a single turn. Overall, by focusing on the ability to execute, we hope to reconcile debates on how LLMs can solve complex reasoning problems yet fail at simple tasks when made longer, and highlight the massive benefits of scaling model size and sequential test-time compute for long-horizon tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23830v1">Bayesian Mixture-of-Experts: Towards Making LLMs Know What They Don't Know</a></div>
    <div class="paper-meta">
      📅 2025-09-28
    </div>
    <details class="paper-abstract">
      The Mixture-of-Experts (MoE) architecture has enabled the creation of massive yet efficient Large Language Models (LLMs). However, the standard deterministic routing mechanism presents a significant limitation: its inherent brittleness is a key contributor to model miscalibration and overconfidence, resulting in systems that often do not know what they don't know. This thesis confronts this challenge by proposing a structured \textbf{Bayesian MoE routing framework}. Instead of forcing a single, deterministic expert selection, our approach models a probability distribution over the routing decision itself. We systematically investigate three families of methods that introduce this principled uncertainty at different stages of the routing pipeline: in the \textbf{weight-space}, the \textbf{logit-space}, and the final \textbf{selection-space}. Through a series of controlled experiments on a 3-billion parameter MoE model, we demonstrate that this framework significantly improves routing stability, in-distribution calibration, and out-of-distribution (OoD) detection. The results show that by targeting this core architectural component, we can create a more reliable internal uncertainty signal. This work provides a practical and computationally tractable pathway towards building more robust and self-aware LLMs, taking a crucial step towards making them know what they don't know.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01827v2">APRMCTS: Improving LLM-based Automated Program Repair with Iterative Tree Search</a></div>
    <div class="paper-meta">
      📅 2025-09-28
      | 💬 Accepted to the 40th IEEE/ACM International Conference on Automated Software Engineering, ASE 2025
    </div>
    <details class="paper-abstract">
      Automated Program Repair (APR) attempts to fix software bugs without human intervention, which plays a crucial role in software development and maintenance. Recently, with the advances in Large Language Models (LLMs), a rapidly increasing number of APR techniques have been proposed with remarkable performance. However, existing LLM-based APR techniques typically adopt trial-and-error strategies, which suffer from two major drawbacks: (1) inherently limited patch effectiveness due to local exploration, and (2) low search efficiency due to redundant exploration. In this paper, we propose APRMCTS, which uses iterative tree search to improve LLM-based APR. APRMCTS incorporates Monte Carlo Tree Search (MCTS) into patch searching by performing a global evaluation of the explored patches and selecting the most promising one for subsequent refinement and generation. APRMCTS effectively resolves the problems of falling into local optima and thus helps improve the efficiency of patch searching. Our experiments on 835 bugs from Defects4J demonstrate that, when integrated with GPT-3.5, APRMCTS can fix a total of 201 bugs, which outperforms all state-of-the-art baselines. Besides, APRMCTS helps GPT-4o-mini, GPT-3.5, Yi-Coder-9B, and Qwen2.5-Coder-7B to fix 30, 27, 37, and 28 more bugs, respectively. More importantly, APRMCTS boasts a significant performance advantage while employing small patch size (16 and 32), notably fewer than the 500 and 10,000 patches adopted in previous studies. In terms of cost, compared to existing state-of-the-art LLM-based APR methods, APRMCTS has time and monetary costs of less than 20% and 50%, respectively. Our extensive study demonstrates that APRMCTS exhibits good effectiveness and efficiency, with particular advantages in addressing complex bugs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.09072v2">READER: Retrieval-Assisted Drafter for Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-09-27
    </div>
    <details class="paper-abstract">
      Autoregressive Language Models instantiate a factorized likelihood over token sequences, yet their strictly sequential decoding process imposes an intrinsic lower bound on inference latency. This bottleneck has emerged as a central obstacle to the scalable deployment of large-scale generative models. Existing acceleration techniques partially mitigate token-level latency by relying on auxiliary draft models or introducing an additional training phase, but fail to address the dominant memory and communication costs. We present READER, a provably lossless speculative decoding framework that bypasses the training of the auxiliary draft model. READER formalizes speculative decoding as a stochastic tree construction problem and exploits the empirical redundancy structure of natural language to generate high-probability candidate continuations. Our method revisits the problem of constructing draft trees, establishing substantial statistical improvements over stochastic draft-tree methods and providing a complexity-theoretic analysis that characterizes the optimality frontier of speculative decoding under bounded computation and memory resources. Beyond the single-sequence regime traditionally considered in prior work, we introduce a memory-optimal key-value cache-serving strategy that guarantees amortized sublinear overhead in the batch dimension, allowing READER to scale to realistic inference workloads. Comprehensive experiments demonstrate up to 6.13x wall-clock speedup on single-prompt inference and up to 5.92x on batched inference, consistently surpassing prior speculative decoding baselines, while preserving exact output equivalence, with even more pronounced gains in retrieval-augmented generation pipelines. Our results close a key gap between theoretical parallelism limits and practical LLM inference, suggesting a new standard for efficient deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16550v2">PropXplain: Can LLMs Enable Explainable Propaganda Detection?</a></div>
    <div class="paper-meta">
      📅 2025-09-27
      | 💬 Large Language Models, Social Media, News Media, Specialized LLMs, Propagandistic content analysis, Fact-checking, Media Analysis, Arabic, English
    </div>
    <details class="paper-abstract">
      There has been significant research on propagandistic content detection across different modalities and languages. However, most studies have primarily focused on detection, with little attention given to explanations justifying the predicted label. This is largely due to the lack of resources that provide explanations alongside annotated labels. To address this issue, we propose a multilingual (i.e., Arabic and English) explanation-enhanced dataset, the first of its kind. Additionally, we introduce an explanation-enhanced LLM for both label detection and rationale-based explanation generation. Our findings indicate that the model performs comparably while also generating explanations. We will make the dataset and experimental resources publicly available for the research community (https://github.com/firojalam/PropXplain).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23459v1">MaskSQL: Safeguarding Privacy for LLM-Based Text-to-SQL via Abstraction</a></div>
    <div class="paper-meta">
      📅 2025-09-27
      | 💬 Accepted to the NeurIPS 2025 Workshop on Regulatable Machine Learning (Regulatable ML @ NeurIPS 2025). Code available at https://github.com/sepideh-abedini/MaskSQL
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown promising performance on tasks that require reasoning, such as text-to-SQL, code generation, and debugging. However, regulatory frameworks with strict privacy requirements constrain their integration into sensitive systems. State-of-the-art LLMs are also proprietary, costly, and resource-intensive, making local deployment impractical. Consequently, utilizing such LLMs often requires sharing data with third-party providers, raising privacy concerns and risking noncompliance with regulations. Although fine-tuned small language models (SLMs) can outperform LLMs on certain tasks and be deployed locally to mitigate privacy concerns, they underperform on more complex tasks such as text-to-SQL translation. In this work, we introduce MaskSQL, a text-to-SQL framework that utilizes abstraction as a privacy protection mechanism to mask sensitive information in LLM prompts. Unlike redaction, which removes content entirely, or generalization, which broadens tokens, abstraction retains essential information while discarding unnecessary details, striking an effective privacy-utility balance for the text-to-SQL task. Moreover, by providing mechanisms to control the privacy-utility tradeoff, MaskSQL facilitates adoption across a broader range of use cases. Our experimental results show that MaskSQL outperforms leading SLM-based text-to-SQL models and achieves performance approaching state-of-the-art LLM-based models, while preserving privacy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23452v1">FoR-SALE: Frame of Reference-guided Spatial Adjustment in LLM-based Diffusion Editing</a></div>
    <div class="paper-meta">
      📅 2025-09-27
      | 💬 9 pages, 3 Tables, 4 Figures, Under Reviewed
    </div>
    <details class="paper-abstract">
      Frame of Reference (FoR) is a fundamental concept in spatial reasoning that humans utilize to comprehend and describe space. With the rapid progress in Multimodal Language models, the moment has come to integrate this long-overlooked dimension into these models. In particular, in text-to-image (T2I) generation, even state-of-the-art models exhibit a significant performance gap when spatial descriptions are provided from perspectives other than the camera. To address this limitation, we propose Frame of Reference-guided Spatial Adjustment in LLM-based Diffusion Editing (FoR-SALE), an extension of the Self-correcting LLM-controlled Diffusion (SLD) framework for T2I. For-Sale evaluates the alignment between a given text and an initially generated image, and refines the image based on the Frame of Reference specified in the spatial expressions. It employs vision modules to extract the spatial configuration of the image, while simultaneously mapping the spatial expression to a corresponding camera perspective. This unified perspective enables direct evaluation of alignment between language and vision. When misalignment is detected, the required editing operations are generated and applied. FoR-SALE applies novel latent-space operations to adjust the facing direction and depth of the generated images. We evaluate FoR-SALE on two benchmarks specifically designed to assess spatial understanding with FoR. Our framework improves the performance of state-of-the-art T2I models by up to 5.3% using only a single round of correction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23433v1">SPIKE-RL: Video-LLMs meet Bayesian Surprise</a></div>
    <div class="paper-meta">
      📅 2025-09-27
      | 💬 10 pages, 4 figures, Code: https://github.com/sahithyaravi/SPIKE-RL
    </div>
    <details class="paper-abstract">
      Real-world videos often show routine activities punctuated by memorable, surprising events. However, most Video-LLMs process videos by sampling frames uniformly, likely missing critical moments that define a video's narrative. We introduce SPIKE, an inference-time framework that quantifies Bayesian Surprise as the belief update triggered by new visual evidence in the video stream, identifying moments where new visual evidence conflicts with prior beliefs. SPIKE effectively localizes surprise in videos, strongly correlated with humans on positive (FunQA) and negative (Oops!) surprise benchmarks. Since the beliefs of zero-shot Video-LLMs are often suboptimal, we develop SPIKE-RL, which leverages GRPO to optimize belief hypotheses based on a reward signal from the video caption. SPIKE and SPIKE-RL guide query-agnostic surprise-weighted frame sampling, which allocates more frames to interesting moments in the video. With this strategy, we achieve consistent performance gains on five downstream benchmarks over uniform sampling. By enabling Video-LLMs to track beliefs and register surprise, our work paves the way for more robust models that can revise their understanding in response to new information.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15229v2">Multilingual Prompting for Improving LLM Generation Diversity</a></div>
    <div class="paper-meta">
      📅 2025-09-27
      | 💬 Accepted by EMNLP 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are known to lack cultural representation and overall diversity in their generations, from expressing opinions to answering factual questions. To mitigate this problem, we propose multilingual prompting: a prompting method which generates several variations of a base prompt with added cultural and linguistic cues from several cultures, generates responses, and then combines the results. Building on evidence that LLMs have language-specific knowledge, multilingual prompting seeks to increase diversity by activating a broader range of cultural knowledge embedded in model training data. Through experiments across multiple models (GPT-4o, GPT-4o-mini, LLaMA 70B, and LLaMA 8B), we show that multilingual prompting consistently outperforms existing diversity-enhancing techniques such as high-temperature sampling, step-by-step recall, and persona prompting. Further analyses show that the benefits of multilingual prompting vary between high and low resource languages and across model sizes, and that aligning the prompting language with cultural cues reduces hallucination about culturally-specific information.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03646v3">Emergent Hierarchical Reasoning in LLMs through Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-27
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Reinforcement Learning (RL) has proven highly effective at enhancing the complex reasoning abilities of Large Language Models (LLMs), yet underlying mechanisms driving this success remain largely opaque. Our analysis reveals that puzzling phenomena like ``aha moments", ``length-scaling'' and entropy dynamics are not disparate occurrences but hallmarks of an emergent reasoning hierarchy, akin to the separation of high-level strategic planning from low-level procedural execution in human cognition. We uncover a compelling two-phase dynamic: initially, a model is constrained by procedural correctness and must improve its low-level skills. The learning bottleneck then decisively shifts, with performance gains being driven by the exploration and mastery of high-level strategic planning. This insight exposes a core inefficiency in prevailing RL algorithms like GRPO, which apply optimization pressure agnostically and dilute the learning signal across all tokens. To address this, we propose Hierarchy-Aware Credit Assignment (HICRA), an algorithm that concentrates optimization efforts on high-impact planning tokens. Our extensive experiments validate that HICRA significantly outperforms strong baselines, and offer deep insights into how reasoning advances through the lens of strategic exploration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23410v1">PATCH: Learnable Tile-level Hybrid Sparsity for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-27
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) deliver impressive performance but incur prohibitive memory and compute costs at deployment. Model pruning is an effective way to reduce these overheads, yet existing approaches face challenges: unstructured sparsity, where nonzeros can appear anywhere, preserves accuracy but yields irregular access patterns that prevent GPU acceleration, while semi-structured 2:4 sparsity is hardware-friendly but enforces a rigid 50% pattern that degrades model quality. To bridge this gap, we introduce PATCH, a hybrid sparsity framework that enables a continuous sparsity ratio between 0% and 50%. PATCH partitions weight matrices into tiles, assigning each tile to be either dense or 2:4 sparse via a learnable mask selection mechanism. This design provides fine-grained control over accuracy-acceleration tradeoffs and supports non-uniform sparsity across layers, leading to superior overall quality. Across models from 0.5B to 8B parameters, PATCH consistently narrows the gap to dense accuracy while delivering practical speedups. For instance, on LLaMA-2 7B with an A6000 GPU, PATCH achieves 1.18x-1.38x end-to-end speedup over dense baselines while improving accuracy by 0.37%-2.96% compared to the state-of-the-art 2:4 pruning method, MaskLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15845v2">Coarse-to-Fine Personalized LLM Impressions for Streamlined Radiology Reports</a></div>
    <div class="paper-meta">
      📅 2025-09-27
    </div>
    <details class="paper-abstract">
      The manual creation of the "Impression" section in radiology reports is a primary driver of radiologist burnout. To address this challenge, we propose a coarse-to-fine framework that leverages open-source large language models (LLMs) to automatically generate and personalize impressions from clinical findings. The system first produces a draft impression and then refines it using machine learning and reinforcement learning from human feedback (RLHF) to align with individual radiologists' styles while ensuring factual accuracy. We fine-tune LLaMA and Mistral models on a large dataset of reports from the University of Chicago Medicine. Our approach is designed to significantly reduce administrative workload and improve reporting efficiency while maintaining high standards of clinical precision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23384v1">A Predictive and Synergistic Two-Layer Scheduling Framework for LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-09-27
    </div>
    <details class="paper-abstract">
      LLM inference serving typically scales out with a two-tier architecture: a cluster router distributes requests to multiple inference engines, each of which then in turn performs its own internal scheduling. However, this commonly used paradigm suffers from critical, systemic inefficiency caused by the information gaps across two layers. At the cluster-layer, the router mainly relies on lagging, coarse-grained metrics, such as average latency and queue length to make decisions, resulting in "decision lag" that leads to suboptimal request routing. At the engine-layer, static heuristic scheduling policies cannot effectively handle the dynamic workloads, leading a poor balance between latency and throughput. Besides, these gaps may cause SLO violations and resource waste, especially in heterogeneous cloud environments. To bridge such gaps, we propose SynergySched, a cross-layer framework that shifts LLM serving system from reactive load balancing to predictive orchestration. The core of SynergySched lies in a structurally-informed online performance model that provides accurate, forward-looking per-step latency and capacity estimations. This model empowers two key components. At the engine-layer, LENS performs SLO-aware, adaptive scheduling, dynamically optimizing batching to meet SLOs under real-time loads. At the cluster-layer, PRISM uses predictive signals to perform state-driven routing, maximizing cluster-wide performance and SLO attainment. Performance evaluations show that SynergySched improves SLO attainment by 43% on average and achieves up to 3x throughput speedup in long-context and heterogeneous scenarios. Besides, we also deploy SynergySched on FlowGPT's clusters to demonstrate its advantages in production environment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23338v1">PARROT: A Benchmark for Evaluating LLMs in Cross-System SQL Translation</a></div>
    <div class="paper-meta">
      📅 2025-09-27
      | 💬 To appear in NeurIPS 2025. Welcome your submission to challenge our leaderboard at: https://code4db.github.io/parrot-bench/. Also visit our code repository at: https://github.com/weAIDB/PARROT
    </div>
    <details class="paper-abstract">
      Large language models (LLMS) have shown increasing effectiveness in Text-to-SQL tasks. However, another closely related problem, Cross-System SQL Translation (a.k.a., SQL-to-SQL), which adapts a query written for one database system (e.g., MySQL) into its equivalent one for another system (e.g., ClickHouse), is of great practical importance but remains underexplored. Existing SQL benchmarks are not well-suited for SQL-to-SQL evaluation, which (1) focus on a limited set of database systems (often just SQLite) and (2) cannot capture many system-specific SQL dialects (e.g., customized functions, data types, and syntax rules). Thus, in this paper, we introduce PARROT, a Practical And Realistic BenchmaRk for CrOss-System SQL Translation. PARROT comprises 598 translation pairs from 38 open-source benchmarks and real-world business services, specifically prepared to challenge system-specific SQL understanding (e.g., LLMS achieve lower than 38.53% accuracy on average). We also provide multiple benchmark variants, including PARROT-Diverse with 28,003 translations (for extensive syntax testing) and PARROT-Simple with 5,306 representative samples (for focused stress testing), covering 22 production-grade database systems. To promote future research, we release a public leaderboard and source code at: https://code4db.github.io/parrot-bench/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23327v1">"Shall We Dig Deeper?": Designing and Evaluating Strategies for LLM Agents to Advance Knowledge Co-Construction in Asynchronous Online Discussions</a></div>
    <div class="paper-meta">
      📅 2025-09-27
    </div>
    <details class="paper-abstract">
      Asynchronous online discussions enable diverse participants to co-construct knowledge beyond individual contributions. This process ideally evolves through sequential phases, from superficial information exchange to deeper synthesis. However, many discussions stagnate in the early stages. Existing AI interventions typically target isolated phases, lacking mechanisms to progressively advance knowledge co-construction, and the impacts of different intervention styles in this context remain unclear and warrant investigation. To address these gaps, we conducted a design workshop to explore AI intervention strategies (task-oriented and/or relationship-oriented) throughout the knowledge co-construction process, and implemented them in an LLM-powered agent capable of facilitating progression while consolidating foundations at each phase. A within-subject study (N=60) involving five consecutive asynchronous discussions showed that the agent consistently promoted deeper knowledge progression, with different styles exerting distinct effects on both content and experience. These findings provide actionable guidance for designing adaptive AI agents that sustain more constructive online discussions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11578v2">Efficient LLM Collaboration via Planning</a></div>
    <div class="paper-meta">
      📅 2025-09-27
    </div>
    <details class="paper-abstract">
      Recently, large language models (LLMs) have demonstrated strong performance, ranging from simple to complex tasks. However, while large proprietary models (e.g., models with over 100B parameters) achieve remarkable results across diverse tasks, they are often accessible through costly APIs, making frequent use too costly for many applications. In contrast, small open-source models (e.g., models with fewer than 3B parameters) are freely available and easy to deploy locally, but their performance on complex tasks remains limited. This trade-off raises a natural question: how can small and large models efficiently collaborate to combine their complementary strengths? To bridge this trade-off, we propose COPE, a test-time collaboration framework. A planner model first generates a plan, a high-level abstraction of the task, and this plan serves as a lightweight intermediate that guides a downstream executor model. Small and large models take turns acting as planner and executor, exchanging plans in a multi-stage cascade to collaboratively solve tasks. Through comprehensive experiments on benchmarks spanning mathematical reasoning, code generation, open-ended tasks, and agent tasks, we demonstrate that COPE achieves performance comparable to large proprietary models, while drastically reducing the inference API cost. These results highlight planning as an effective prior for cost-efficient inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23324v1">Scaling LLM Test-Time Compute with Mobile NPU on Smartphones</a></div>
    <div class="paper-meta">
      📅 2025-09-27
    </div>
    <details class="paper-abstract">
      Deploying Large Language Models (LLMs) on mobile devices faces the challenge of insufficient performance in smaller models and excessive resource consumption in larger ones. This paper highlights that mobile Neural Processing Units (NPUs) have underutilized computational resources, particularly their matrix multiplication units, during typical LLM inference. To leverage this wasted compute capacity, we propose applying parallel test-time scaling techniques on mobile NPUs to enhance the performance of smaller LLMs. However, this approach confronts inherent NPU challenges, including inadequate hardware support for fine-grained quantization and low efficiency in general-purpose computations. To overcome these, we introduce two key techniques: a hardware-aware tile quantization scheme that aligns group quantization with NPU memory access patterns, and efficient LUT-based replacements for complex operations such as Softmax and dequantization. We design and implement an end-to-end inference system that leverages the NPU's compute capability to support test-time scaling on Qualcomm Snapdragon platforms. Experiments show our approach brings significant speedups: up to 19.0 for mixed-precision GEMM and 2.2 for Softmax. More importantly, we demonstrate that smaller models using test-time scaling can match or exceed the accuracy of larger models, achieving a new performance-cost Pareto frontier.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23323v1">LLM Interpretability with Identifiable Temporal-Instantaneous Representation</a></div>
    <div class="paper-meta">
      📅 2025-09-27
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Despite Large Language Models' remarkable capabilities, understanding their internal representations remains challenging. Mechanistic interpretability tools such as sparse autoencoders (SAEs) were developed to extract interpretable features from LLMs but lack temporal dependency modeling, instantaneous relation representation, and more importantly theoretical guarantees, undermining both the theoretical foundations and the practical confidence necessary for subsequent analyses. While causal representation learning (CRL) offers theoretically grounded approaches for uncovering latent concepts, existing methods cannot scale to LLMs' rich conceptual space due to inefficient computation. To bridge the gap, we introduce an identifiable temporal causal representation learning framework specifically designed for LLMs' high-dimensional concept space, capturing both time-delayed and instantaneous causal relations. Our approach provides theoretical guarantees and demonstrates efficacy on synthetic datasets scaled to match real-world complexity. By extending SAE techniques with our temporal causal framework, we successfully discover meaningful concept relationships in LLM activations. Our findings show that modeling both temporal and instantaneous conceptual relationships advances the interpretability of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23322v1">Decoupling Reasoning and Perception: An LLM-LMM Framework for Faithful Visual Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-27
    </div>
    <details class="paper-abstract">
      Significant advancements in the reasoning capabilities of Large Language Models (LLMs) are now driven by test-time scaling laws, particularly those leveraging extended Chain-of-Thought (CoT) reasoning. Inspired by these breakthroughs, researchers have extended these paradigms to Large Multimodal Models (LMMs). However, a critical limitation emerges: as their reasoning chains extend, LMMs increasingly rely on textual logic, progressively losing grounding in the underlying visual information. This leads to reasoning paths that diverge from the image content, culminating in erroneous conclusions. To address this, we introduce a strikingly simple yet effective training-free visual-reasoning pipeline. The core concept is to decouple the reasoning and perception processes. A powerful LLM orchestrates the high-level reasoning, strategically interrogating a LMM to extract specific visual information required for its logical chain. The LMM, in turn, functions exclusively as a visual question-answering engine, supplying the necessary perceptual details on demand. This lightweight, plug-and-play approach requires no additional training or architectural changes. Comprehensive evaluations validate that our framework effectively governs the visual reasoning process, leading to a significant reduction in visually-unfounded reasoning steps and a substantial improvement in reasoning fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17072v2">SnipSnap: A Joint Compression Format and Dataflow Co-Optimization Framework for Efficient Sparse LLM Accelerator Design</a></div>
    <div class="paper-meta">
      📅 2025-09-27
      | 💬 To appear in the 31st Asia and South Pacific Design Automation Conference (ASP-DAC 2026)
    </div>
    <details class="paper-abstract">
      The growing scale of large language models (LLMs) has intensified demands on computation and memory, making efficient inference a key challenge. While sparsity can reduce these costs, existing design space exploration (DSE) frameworks often overlook compression formats, a key factor for leveraging sparsity on accelerators. This paper proposes SnipSnap, a joint compression format and dataflow co-optimization framework for efficient sparse LLM accelerator design. SnipSnap introduces: (1) a hierarchical compression format encoding to expand the design space; (2) an adaptive compression engine for selecting formats under diverse sparsity; and (3) a progressive co-search workflow that jointly optimizes dataflow and compression formats. SnipSnap achieves 18.24% average memory energy savings via format optimization, along with 2248.3$\times$ and 21.0$\times$ speedups over Sparseloop and DiMO-Sparse frameworks, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23252v1">NanoFlux: Adversarial Dual-LLM Evaluation and Distillation For Multi-Domain Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-27
      | 💬 preprint version
    </div>
    <details class="paper-abstract">
      We present NanoFlux, a novel adversarial framework for generating targeted training data to improve LLM reasoning, where adversarially-generated datasets containing fewer than 200 examples outperform conventional fine-tuning approaches. The framework employs a competitive dynamic between models alternating as Attacker and Defender, supervised by a tool-augmented Judge, synthesizing multi-step questions with explanatory annotations that target specific reasoning capabilities. Fine-tuning a 4B-parameter model on NanoFlux-generated data yields performance gains across diverse domains compared to full-benchmark fine-tuning: +5.9% on mathematical reasoning (GSMHard), +3.6% on scientific reasoning (GenomeBench), and +16.6% on medical reasoning (MultiMedQA), while reducing computational requirements by 3-14x. Ablation studies reveal a non-monotonic relationship between dataset characteristics and model performance, uncovering domain-specific optimal points for question complexity and reasoning quality. NanoFlux automates training data generation through embedding-based novelty filtering, tool-augmented evaluation, and multi-hop reasoning, suggesting that future model improvements may lie in the intelligent synthesis of small, precisely targeted training datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23246v1">Adaptive Token-Weighted Differential Privacy for LLMs: Not All Tokens Require Equal Protection</a></div>
    <div class="paper-meta">
      📅 2025-09-27
      | 💬 18 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) frequently memorize sensitive or personal information, raising significant privacy concerns. Existing variants of differential privacy stochastic gradient descent (DPSGD) inject uniform noise into every gradient step, significantly extending training time and reducing model accuracy. We propose that concentrating noise primarily on gradients associated with sensitive tokens can substantially decrease DP training time, strengthen the protection of sensitive information, and simultaneously preserve the model's performance on non-sensitive data. We operationalize this insight through Adaptive Token-Weighted Differential Privacy (ATDP), a modification of vanilla DP-SGD that adaptively assigns different gradient weights to sensitive and non-sensitive tokens. By employing a larger noise scale at the early stage of training, ATDP rapidly disrupts memorization of sensitive content. As a result, ATDP only requires a few additional epochs of lightweight post-processing following standard fine-tuning, injecting targeted noise primarily on parameters corresponding to sensitive tokens, thus minimally affecting the model's general capabilities. ATDP can be seamlessly integrated into any existing DP-based fine-tuning pipeline or directly applied to non-private models as a fast privacy-enhancing measure. Additionally, combined with an initial redacted fine-tuning phase, ATDP forms a streamlined DP pipeline that achieves comparable canary protection to state-of-the-art DP-SGD methods, significantly reduces the computational overhead of DP fine-tuning, shortening training time by approximately 90 percent, while achieving comparable or superior privacy protection and minimal accuracy degradation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23234v1">$p$-less Sampling: A Robust Hyperparameter-Free Approach for LLM Decoding</a></div>
    <div class="paper-meta">
      📅 2025-09-27
    </div>
    <details class="paper-abstract">
      Obtaining high-quality outputs from Large Language Models (LLMs) often depends upon the choice of a sampling-based decoding strategy to probabilistically choose the next token at each generation step. While a variety of such sampling methods have been proposed, their performance can be sensitive to the selection of hyperparameters which may require different settings depending upon the generation task and temperature configuration. In this work, we introduce $p$-less sampling: an information-theoretic approach to sampling which dynamically sets a truncation threshold at each decoding step based on the entire token probability distribution. Unlike existing methods, $p$-less sampling has no hyperparameters and consistently produces high-quality outputs as temperature increases. We provide theoretical perspectives on $p$-less sampling to ground our proposed method and conduct experiments to empirically validate its effectiveness across a range of math, logical reasoning, and creative writing tasks. Our results demonstrate how $p$-less sampling consistently outperforms existing sampling approaches while exhibiting much less degradation in text quality at higher temperature values. We further show how $p$-less achieves greater inference-time efficiency than alternative methods through lower average token sampling times and shorter generation lengths, without sacrificing accuracy. Finally, we provide analyses to highlight the benefits of $p$-less through qualitative examples, case studies, and diversity assessments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04695v2">Reshaping Reasoning in LLMs: A Theoretical Analysis of RL Training Dynamics through Pattern Selection</a></div>
    <div class="paper-meta">
      📅 2025-09-27
      | 💬 28 pages, 5 figures, 2 table
    </div>
    <details class="paper-abstract">
      While reinforcement learning (RL) demonstrated remarkable success in enhancing the reasoning capabilities of language models, the training dynamics of RL in LLMs remain unclear. In this work, we provide an explanation of the RL training process through empirical analysis and rigorous theoretical modeling. First, through systematic reasoning-pattern-level and token-level analysis across the RL training process, we show that while different reasoning patterns exhibit relatively stable success rates during training, RL primarily optimizes a sparse subset of critical tokens, thereby reshaping reasoning pattern distributions to affect model performance. Building on these empirical insights, we develop a theoretical framework to understand the training dynamics of RL with two typical rewards: verifiable reward (RLVR) and model's internal feedback (RLIF). For RLVR, we analyze the training dynamics under two special cases: one where models readily converge to optimal reasoning strategies, and another where optimization becomes challenging, revealing that the base model's reasoning quality is crucial for determining convergence behavior. For RLIF, we examine how internal rewards initially improve model performance but can potentially lead to degradation with continued training. Extensive experiments validate our findings, advancing both theoretical understanding and practical applications of RL in language model enhancement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20643v2">Cooking Up Creativity: Enhancing LLM Creativity through Structured Recombination</a></div>
    <div class="paper-meta">
      📅 2025-09-27
      | 💬 Accepted at TACL; pre-MIT Press publication version
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel at many tasks, yet they struggle to produce truly creative, diverse ideas. In this paper, we introduce a novel approach that enhances LLM creativity. We apply LLMs for translating between natural language and structured representations, and perform the core creative leap via cognitively inspired manipulations on these representations. Our notion of creativity goes beyond superficial token-level variations; rather, we recombine structured representations of existing ideas, enabling our system to effectively explore a more abstract landscape of ideas. We demonstrate our approach in the culinary domain with DishCOVER, a model that generates creative recipes. Experiments and domain-expert evaluations reveal that our outputs, which are mostly coherent and feasible, significantly surpass GPT-4o in terms of novelty and diversity, thus outperforming it in creative generation. We hope our work inspires further research into structured creativity in AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06056v3">Entropy-Memorization Law: Evaluating Memorization Difficulty of Data in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-27
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are known to memorize portions of their training data, sometimes reproducing content verbatim when prompted appropriately. In this work, we investigate a fundamental yet under-explored question in the domain of memorization: How to characterize memorization difficulty of training data in LLMs? Through empirical experiments on OLMo, a family of open models, we present the Entropy-Memorization Law. It suggests that data entropy is linearly correlated with memorization score. Moreover, in a case study of memorizing highly randomized strings, or "gibberish", we observe that such sequences, despite their apparent randomness, exhibit unexpectedly low empirical entropy compared to the broader training corpus. Adopting the same strategy to discover Entropy-Memorization Law, we derive a simple yet effective approach to distinguish training and testing data, enabling Dataset Inference (DI).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23219v1">WirelessMathLM: Teaching Mathematical Reasoning for LLMs in Wireless Communications with Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-27
      | 💬 Project Homepage: https://lixin.ai/WirelessMathLM
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel at general mathematical reasoning but fail catastrophically on specialized technical mathematics. In wireless communications, where problems require precise manipulation of information-theoretic bounds, optimization constraints, and signal processing formulations, even state-of-the-art models struggle to achieve competent performance. We present WirelessMathLM, demonstrating that compact models (0.5B-7B parameters) can match or exceed much larger models through domain-specific reinforcement learning with verifiable rewards. Our key insight is that wireless mathematics problems possess a unique property--verifiable correctness--that enables effective reinforcement learning without human feedback. We construct WirelessMathBench-XL, a comprehensive benchmark of 4,027 problems from 970 papers. Using Group Relative Policy Optimization (GRPO) with binary verification rewards, we train models directly from base checkpoints without supervised warm-start. Our 7B model achieves 39.5% accuracy on WirelessMathBench-XL, approaching GPT-4o (40.4%) while using about 100 times fewer parameters than DeepSeek-R1 (671B, 57.4%). Remarkably, GRPO training nearly doubles performance across all model scales (0.5B +11%, 3B +103%, 7B +81%), with positive transfer to general mathematics benchmarks--our models gain +8.4 points on average across MATH, Minerva-Math, OlympiadBench, AMC, and AIME without any training on these tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20798v2">LogReasoner: Empowering LLMs with Expert-like Coarse-to-Fine Reasoning for Automated Log Analysis</a></div>
    <div class="paper-meta">
      📅 2025-09-27
      | 💬 under review
    </div>
    <details class="paper-abstract">
      Log analysis is crucial for monitoring system health and diagnosing failures in complex systems. Recent advances in large language models (LLMs) offer new opportunities for automated log analysis, leveraging their reasoning capabilities to perform tasks such as anomaly detection and failure prediction. However, general-purpose LLMs struggle to formulate structured reasoning workflows that align with expert cognition and deliver precise details of reasoning steps. To address these challenges, we propose LogReasoner, a coarse-to-fine reasoning enhancement framework designed to enable LLMs to reason log analysis tasks like experts. LogReasoner consists of two stages: (1) coarse-grained enhancement of expert thinking, where high-level expert thoughts are constructed from collected troubleshooting flowcharts and existing tasks to enable LLMs to formulate structured reasoning workflows and (2) fine-grained enhancement of specific steps, where we first fine-tune the LLM with task-specific stepwise solutions to enhance the LLM for instantiated reasoning, then employ the preference learning to calibrate the LLM's reasoning details from its mistakes, further strengthen the LLM's analytical granularity and correctness. We evaluate LogReasoner on four distinct log analysis tasks using open-source LLMs such as Qwen-2.5 and Llama-3. Experimental results show that LogReasoner significantly outperforms existing LLMs, achieving state-of-the-art performance and demonstrating its effectiveness in enhancing the reasoning capabilities of LLMs for log analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23189v1">AutoEP: LLMs-Driven Automation of Hyperparameter Evolution for Metaheuristic Algorithms</a></div>
    <div class="paper-meta">
      📅 2025-09-27
    </div>
    <details class="paper-abstract">
      Dynamically configuring algorithm hyperparameters is a fundamental challenge in computational intelligence. While learning-based methods offer automation, they suffer from prohibitive sample complexity and poor generalization. We introduce AutoEP, a novel framework that bypasses training entirely by leveraging Large Language Models (LLMs) as zero-shot reasoning engines for algorithm control. AutoEP's core innovation lies in a tight synergy between two components: (1) an online Exploratory Landscape Analysis (ELA) module that provides real-time, quantitative feedback on the search dynamics, and (2) a multi-LLM reasoning chain that interprets this feedback to generate adaptive hyperparameter strategies. This approach grounds high-level reasoning in empirical data, mitigating hallucination. Evaluated on three distinct metaheuristics across diverse combinatorial optimization benchmarks, AutoEP consistently outperforms state-of-the-art tuners, including neural evolution and other LLM-based methods. Notably, our framework enables open-source models like Qwen3-30B to match the performance of GPT-4, demonstrating a powerful and accessible new paradigm for automated hyperparameter design. Our code is available at https://anonymous.4open.science/r/AutoEP-3E11
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23188v1">Diagnose, Localize, Align: A Full-Stack Framework for Reliable LLM Multi-Agent Systems under Instruction Conflicts</a></div>
    <div class="paper-meta">
      📅 2025-09-27
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-powered multi-agent systems (MAS) have rapidly advanced collaborative reasoning, tool use, and role-specialized coordination in complex tasks. However, reliability-critical deployment remains hindered by a systemic failure mode: hierarchical compliance under instruction conflicts (system-user, peer-peer), where agents misprioritize system-level rules in the presence of competing demands. Moreover, widely used macro-level metrics (e.g., pass@k) obscure these micro-level violations and offer little actionable guidance for remedy. In this work, we present a full-stack, three-stage framework: (1) Diagnose - Contextualized Role Adherence Score (CRAS), a query-wise, context-aware scoring metric that decomposes role adherence into four measurable dimensions; (2) Localize - attention drift analysis revealing that instruction conflicts are resolved by attention heads that are largely concentrated in middle layers; (3) Align - Surgical Alignment of Instruction Layers (SAIL), which installs LoRA only on the localized focal layers and optimizes a token-weighted DPO-style preference objective that credits tokens by their focal attentional contribution. Across standard benchmarks and MAS frameworks, our surgical approach improves instruction hierarchy compliance (e.g., +5.60% with AutoGen on MedQA) without full-model finetuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19563v2">TabularGSM: Understanding the Limitations of LLMs in Tabular Math Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-27
      | 💬 Paper under review, code and dataset are all available
    </div>
    <details class="paper-abstract">
      Mathematical reasoning has long been a key benchmark for evaluating large language models (LLMs). Although substantial progress has been made on math word problems, the need for reasoning over tabular data in real-world applications has been overlooked. For instance, applications such as business intelligence demand not only multi-step numerical reasoning with tables but also robustness to incomplete or inconsistent information. However, comprehensive evaluation in this area is severely limited, constrained by the reliance on manually collected tables that are difficult to scale and the lack of coverage for potential traps encountered in real-world scenarios. To address this problem, we propose AutoT2T, a neuro-symbolic framework that controllably transforms math word problems into scalable and verified tabular reasoning tasks, enabling the evaluation of both accuracy and robustness. Building on this pipeline, we develop TabularGSM, a benchmark comprising three progressively complex subsets and a trap subset, with two complementary evaluation settings. Our study reveals three key observations: (1) Tabular structure makes mathematical reasoning more challenging; (2) The difficulties stem from the joint effects of tabular retrieval and reasoning; (3) Reasoning robustness is another significant issue that needs to be addressed in existing LLMs. In-depth analyses are conducted for each observation to guide future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04219v3">Model Collapse Is Not a Bug but a Feature in Machine Unlearning for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-27
    </div>
    <details class="paper-abstract">
      Current unlearning methods for LLMs optimize on the private information they seek to remove by incorporating it into their fine-tuning data. We argue this not only risks reinforcing exposure to sensitive data, it also fundamentally contradicts the principle of minimizing its use. As a remedy, we propose a novel unlearning method-Partial Model Collapse (PMC), which does not require unlearning targets in the unlearning objective. Our approach is inspired by recent observations that training generative models on their own generations leads to distribution collapse, effectively removing information from model outputs. Our central insight is that model collapse can be leveraged for machine unlearning by deliberately triggering it for data we aim to remove. We theoretically analyze that our approach converges to the desired outcome, i.e. the model unlearns the data targeted for removal. We empirically demonstrate that PMC overcomes three key limitations of existing unlearning methods that explicitly optimize on unlearning targets, and more effectively removes private information from model outputs while preserving general model utility. Overall, our contributions represent an important step toward more comprehensive unlearning that aligns with real-world privacy constraints. Code available at https://www.cs.cit.tum.de/daml/partial-model-collapse/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23166v1">Test-Time Policy Adaptation for Enhanced Multi-Turn Interactions with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-27
      | 💬 32 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) employ multi-turn interaction as a fundamental paradigm for completing complex tasks. However, their performance often degrades in extended interactions, as they are typically trained on static, single-turn data, which hinders their ability to adapt to real-time user feedback. To address this limitation, we first propose a new paradigm: Test-Time Policy Adaptation for Multi-Turn Interactions (T2PAM), which utilizes user feedback from the ongoing interaction as a reward signal to estimate a latent optimal policy aligned with user preferences, then updates a small subset of parameters to steer the model toward this policy, ultimately enabling efficient in-conversation self-correction. We then introduce Optimum-Referenced One-Step Adaptation (ROSA), a lightweight algorithm that operationalizes T2PAM. ROSA guides the model parameters toward a theoretical optimal policy in a single, efficient update step, avoiding costly iterative gradient-based optimization and minimizing computational overhead. We provide a rigorous theoretical analysis guaranteeing that the policy of ROSA converges to the preference of user as the number of interactions increases. Extensive experiments on challenging benchmark demonstrate that ROSA achieves significant improvements in both task effectiveness and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17138v4">Runtime Adaptive Pruning for LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-09-27
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel at language understanding and generation, but their enormous computational and memory requirements hinder deployment. Compression offers a potential solution to mitigate these constraints. However, most existing methods rely on fixed heuristics and thus fail to adapt to runtime memory variations or heterogeneous KV-cache demands arising from diverse user requests. To address these limitations, we propose RAP, an elastic pruning framework driven by reinforcement learning (RL) that dynamically adjusts compression strategies in a runtime-aware manner. Specifically, RAP dynamically tracks the evolving ratio between model parameters and KV-cache across practical execution. Recognizing that FFNs house most parameters, whereas parameter -light attention layers dominate KV-cache formation, the RL agent retains only those components that maximize utility within the current memory budget, conditioned on instantaneous workload and device state. Extensive experiments results demonstrate that RAP outperforms state-of-the-art baselines, marking the first time to jointly consider model weights and KV-cache on the fly.
    </details>
</div>
