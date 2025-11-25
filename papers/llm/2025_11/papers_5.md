# llm - 2025_11

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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.24189v2">PET: Preference Evolution Tracking with LLM-Generated Explainable Distribution</a></div>
    <div class="paper-meta">
      📅 2025-11-15
    </div>
    <details class="paper-abstract">
      Understanding how user preference evolves over time is a fundamental challenge central to modern digital ecosystems, for which Large Language Models (LLMs) are an increasingly prominent and popular approach due to their ability to comprehend the rich semantic context within behavioral data. A common practice is to use LLMs to predict a user's next action by directly generating a ranked list of preferred items. Although effective for short-term prediction, the end-to-end generation paradigm inherently limits personalization. Its opaque decision-making process obscures holistic user profiling and exacerbates popularity bias. To address these limitations, we propose Preference Evolution Tracking (PET), a framework that reframes the task as inferring a dynamic probability distribution over a stable and interpretable lattice of preference clusters. By applying logit-probing and generative classification techniques, PET infers a user's preference as a probability distribution, enabling transparent preference learning. On public benchmarks (Yelp, MovieLens), PET improves ranking quality by up to 40% in NDCG over direct generation baselines. On a large-scale, real-world dataset from a short-video platform, it excels at ranking long-tail contents, significantly outperforming a SOTA production model by 7 times in the NDCG score. Ultimately, PET transforms the user profile model from direct preference list generation to a transparent distributional preference mapping, paving the way for more explainable, fair, and diverse personalization systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13772v1">Can LLMs Create Legally Relevant Summaries and Analyses of Videos?</a></div>
    <div class="paper-meta">
      📅 2025-11-15
      | 💬 Accepted for publication at JURIX 2025 Torino, Italy. This is the preprint version. Code and data available at: https://github.com/maastrichtlawtech/jurix2025_LLM_video_analysis
    </div>
    <details class="paper-abstract">
      Understanding the legally relevant factual basis of an event and conveying it through text is a key skill of legal professionals. This skill is important for preparing forms (e.g., insurance claims) or other legal documents (e.g., court claims), but often presents a challenge for laypeople. Current AI approaches aim to bridge this gap, but mostly rely on the user to articulate what has happened in text, which may be challenging for many. Here, we investigate the capability of large language models (LLMs) to understand and summarize events occurring in videos. We ask an LLM to summarize and draft legal letters, based on 120 YouTube videos showing legal issues in various domains. Overall, 71.7\% of the summaries were rated as of high or medium quality, which is a promising result, opening the door to a number of applications in e.g. access to justice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14794v1">irace-evo: Automatic Algorithm Configuration Extended With LLM-Based Code Evolution</a></div>
    <div class="paper-meta">
      📅 2025-11-15
    </div>
    <details class="paper-abstract">
      Automatic algorithm configuration tools such as irace efficiently tune parameter values but leave algorithmic code unchanged. This paper introduces a first version of irace-evo, an extension of irace that integrates code evolution through large language models (LLMs) to jointly explore parameter and code spaces. The proposed framework enables multi-language support (e.g., C++, Python), reduces token consumption via progressive context management, and employs the Always-From-Original principle to ensure robust and controlled code evolution. We evaluate irace-evo on the Construct, Merge, Solve & Adapt (CMSA) metaheuristic for the Variable-Sized Bin Packing Problem (VSBPP). Experimental results show that irace-evo can discover new algorithm variants that outperform the state-of-the-art CMSA implementation while maintaining low computational and monetary costs. Notably, irace-evo generates competitive algorithmic improvements using lightweight models (e.g., Claude Haiku 3.5) with a total usage cost under 2 euros. These results demonstrate that coupling automatic configuration with LLM-driven code evolution provides a powerful, cost-efficient avenue for advancing heuristic design and metaheuristic optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11967v1">Bootstrapped LLM Semantics for Context-Aware Path Planning</a></div>
    <div class="paper-meta">
      📅 2025-11-15
    </div>
    <details class="paper-abstract">
      Prompting robots with natural language (NL) has largely been studied as what task to execute (goal selection, skill sequencing) rather than how to execute that task safely and efficiently in semantically rich, human-centric spaces. We address this gap with a framework that turns a large language model (LLM) into a stochastic semantic sensor whose outputs modulate a classical planner. Given a prompt and a semantic map, we draw multiple LLM "danger" judgments and apply a Bayesian bootstrap to approximate a posterior over per-class risk. Using statistics from the posterior, we create a potential cost to formulate a path planning problem. Across simulated environments and a BIM-backed digital twin, our method adapts how the robot moves in response to explicit prompts and implicit contextual information. We present qualitative and quantitative results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.22162v3">Surface Reading LLMs: Synthetic Text and its Styles</a></div>
    <div class="paper-meta">
      📅 2025-11-15
      | 💬 12 pages, 1 figure
    </div>
    <details class="paper-abstract">
      Despite a potential plateau in ML advancement, the societal impact of large language models lies not in approaching superintelligence but in generating text surfaces indistinguishable from human writing. While Critical AI Studies provides essential material and socio-technical critique, it risks overlooking how LLMs phenomenologically reshape meaning-making. This paper proposes a semiotics of "surface integrity" as attending to the immediate plane where LLMs inscribe themselves into human communication. I distinguish three knowledge interests in ML research (epistemology, epistēmē, and epistemics) and argue for integrating surface-level stylistic analysis alongside depth-oriented critique. Through two case studies examining stylistic markers of synthetic text, I argue how attending to style as a semiotic phenomenon reveals LLMs as cultural machines that transform the conditions of meaning emergence and circulation in contemporary discourse, independent of questions about machine consciousness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11961v1">"Power of Words": Stealthy and Adaptive Private Information Elicitation via LLM Communication Strategies</a></div>
    <div class="paper-meta">
      📅 2025-11-15
    </div>
    <details class="paper-abstract">
      While communication strategies of Large Language Models (LLMs) are crucial for human-LLM interactions, they can also be weaponized to elicit private information, yet such stealthy attacks remain under-explored. This paper introduces the first adaptive attack framework for stealthy and targeted private information elicitation via communication strategies. Our framework operates in a dynamic closed-loop: it first performs real-time psychological profiling of the users' state, then adaptively selects an optimized communication strategy, and finally maintains stealthiness through prompt-based rewriting. We validated this framework through a user study (N=84), demonstrating its generalizability across 3 distinct LLMs and 3 scenarios. The targeted attacks achieved a 205.4% increase in eliciting specific targeted information compared to stealthy interactions without strategies. Even stealthy interactions without specific strategies successfully elicited private information in 54.8% cases. Notably, users not only failed to detect the manipulation but paradoxically rated the attacking chatbot as more empathetic and trustworthy. Finally, we advocate for mitigations, encouraging developers to integrate adaptive, just-in-time alerts, users to build literacy against specific manipulative tactics, and regulators to define clear ethical boundaries distinguishing benign persuasion from coercion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11954v1">LLM-Assisted Formalization Enables Deterministic Detection of Statutory Inconsistency in the Internal Revenue Code</a></div>
    <div class="paper-meta">
      📅 2025-11-15
      | 💬 29 pages, 3 appendices with Prolog code and full codebase available at: https://github.com/borchuluun/section121-inconsistency-detection
    </div>
    <details class="paper-abstract">
      This study introduces a hybrid neuro-symbolic framework that achieves deterministic detection of statutory inconsistency in complex law. We use the U.S. Internal Revenue Code (IRC) as a case study because its complexity makes it a fertile domain for identifying conflicts. Our research offers a solution for detecting inconsistent provisions by combining Large Language Models (LLMs) with symbolic logic. LLM-based methods can support compliance, fairness, and statutory drafting, yet tax-specific applications remain sparse. A key challenge is that such models struggle with hierarchical processing and deep structured reasoning, especially over long text. This research addresses these gaps through experiments using GPT-4o, GPT-5, and Prolog. GPT-4o was first used to translate Section 121 into Prolog rules and refine them in SWISH. These rules were then incorporated into prompts to test whether Prolog-augmented prompting improved GPT-4o's inconsistency detection. GPT-4o, whether prompted with natural language alone or with Prolog augmentation, detected the inconsistency in only one of three strategies (33 percent accuracy), but its reasoning quality differed: natural-language prompting achieved 100 percent rule coverage, while Prolog-augmented prompting achieved 66 percent, indicating more incomplete statutory analysis. In contrast to probabilistic prompting, the hybrid Prolog model produced deterministic and reproducible results. Guided by GPT-5 for refinement, the model formalized the IRC section's competing interpretations and successfully detected an inconsistency zone. Validation tests confirm that the Prolog implementation is accurate, internally consistent, deterministic, and capable of autonomously identifying inconsistencies. These findings show that LLM-assisted formalization, anchored in symbolic logic, enables transparent and reliable statutory inconsistency detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12376v1">BitSnap: Checkpoint Sparsification and Quantization in LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-11-15
      | 💬 12 pages, numerous figures
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) continue to grow in size and complexity, efficient checkpoint saving\&loading has become crucial for managing storage, memory usage, and fault tolerance in LLM training. The current works do not comprehensively take into account the optimization of these several aspects. This paper proposes a novel checkpoint sparsification and quantization method that adapts dynamically to different training stages and model architectures. We present a comprehensive analysis of existing lossy and lossless compression techniques, identify current limitations, and introduce our adaptive approach that balances compression ratio, speed, and precision impact throughout the training process. Experiments on different sizes of LLMs demonstrate that our bitmask-based sparsification method achieves 16x compression ratio without compromising model accuracy. Additionally, the cluster-based quantization method achieves 2x compression ratio with little precision loss.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.00730v4">Teaching LLMs to See and Guide: Context-Aware Real-Time Assistance in Augmented Reality</a></div>
    <div class="paper-meta">
      📅 2025-11-15
      | 💬 This work has been submitted to the IEEE Transactions on Systems, Man, and Cybernetics: Systems for possible publication
    </div>
    <details class="paper-abstract">
      The growing adoption of augmented and virtual reality (AR and VR) technologies in industrial training and on-the-job assistance has created new opportunities for intelligent, context-aware support systems. As workers perform complex tasks guided by AR and VR, these devices capture rich streams of multimodal data, including gaze, hand actions, and task progression, that can reveal user intent and task state in real time. Leveraging this information effectively remains a major challenge. In this work, we present a context-aware large language model (LLM) assistant that integrates diverse data modalities, such as hand actions, task steps, and dialogue history, into a unified framework for real-time question answering. To systematically study how context influences performance, we introduce an incremental prompting framework, where each model version receives progressively richer contextual inputs. Using the HoloAssist dataset, which records AR-guided task executions, we evaluate how each modality contributes to the assistant's effectiveness. Our experiments show that incorporating multimodal context significantly improves the accuracy and relevance of responses. These findings highlight the potential of LLM-driven multimodal integration to enable adaptive, intuitive assistance for AR and VR-based industrial training and assistance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.04688v2">Evaluating LLMs' Reasoning Over Ordered Procedural Steps</a></div>
    <div class="paper-meta">
      📅 2025-11-15
      | 💬 Accepted to IJCNLP-AACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Reasoning over procedural sequences, where the order of steps directly impacts outcomes, is a critical capability for large language models (LLMs). In this work, we study the task of reconstructing globally ordered sequences from shuffled procedural steps, using a curated dataset of food recipes, a domain where correct sequencing is essential for task success. We evaluate several LLMs under zero-shot and few-shot settings and present a comprehensive evaluation framework that adapts established metrics from ranking and sequence alignment. These include Kendall's Tau, Normalized Longest Common Subsequence (NLCS), and Normalized Edit Distance (NED), which capture complementary aspects of ordering quality. Our analysis shows that model performance declines with increasing sequence length, reflecting the added complexity of longer procedures. We also find that greater step displacement in the input, corresponding to more severe shuffling, leads to further degradation. These findings highlight the limitations of current LLMs in procedural reasoning, especially with longer and more disordered inputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.19172v2">When Facts Change: Probing LLMs on Evolving Knowledge with evolveQA</a></div>
    <div class="paper-meta">
      📅 2025-11-15
      | 💬 Under submission
    </div>
    <details class="paper-abstract">
      LLMs often fail to handle temporal knowledge conflicts--contradictions arising when facts evolve over time within their training data. Existing studies evaluate this phenomenon through benchmarks built on structured knowledge bases like Wikidata, but they focus on widely-covered, easily-memorized popular entities and lack the dynamic structure needed to fairly evaluate LLMs with different knowledge cut-off dates. We introduce evolveQA, a benchmark specifically designed to evaluate LLMs on temporally evolving knowledge, constructed from 3 real-world, time-stamped corpora: AWS updates, Azure changes, and WHO disease outbreak reports. Our framework identifies naturally occurring knowledge evolution and generates questions with gold answers tailored to different LLM knowledge cut-off dates. Through extensive evaluation of 12 open and closed-source LLMs across 3 knowledge probing formats, we demonstrate significant performance drops of up to 31% on evolveQA compared to static knowledge questions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12300v1">Do LLMs and Humans Find the Same Questions Difficult? A Case Study on Japanese Quiz Answering</a></div>
    <div class="paper-meta">
      📅 2025-11-15
    </div>
    <details class="paper-abstract">
      LLMs have achieved performance that surpasses humans in many NLP tasks. However, it remains unclear whether problems that are difficult for humans are also difficult for LLMs. This study investigates how the difficulty of quizzes in a buzzer setting differs between LLMs and humans. Specifically, we first collect Japanese quiz data including questions, answers, and correct response rate of humans, then prompted LLMs to answer the quizzes under several settings, and compare their correct answer rate to that of humans from two analytical perspectives. The experimental results showed that, compared to humans, LLMs struggle more with quizzes whose correct answers are not covered by Wikipedia entries, and also have difficulty with questions that require numerical answers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12295v1">Privacy-Preserving Prompt Injection Detection for LLMs Using Federated Learning and Embedding-Based NLP Classification</a></div>
    <div class="paper-meta">
      📅 2025-11-15
    </div>
    <details class="paper-abstract">
      Prompt injection attacks are an emerging threat to large language models (LLMs), enabling malicious users to manipulate outputs through carefully designed inputs. Existing detection approaches often require centralizing prompt data, creating significant privacy risks. This paper proposes a privacy-preserving prompt injection detection framework based on federated learning and embedding-based classification. A curated dataset of benign and adversarial prompts was encoded with sentence embedding and used to train both centralized and federated logistic regression models. The federated approach preserved privacy by sharing only model parameters across clients, while achieving detection performance comparable to centralized training. Results demonstrate that effective prompt injection detection is feasible without exposing raw data, making this one of the first explorations of federated security for LLMs. Although the dataset is limited in scale, the findings establish a strong proof-of-concept and highlight new directions for building secure and privacy-aware LLM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12288v1">Reducing Hallucinations in LLM-Generated Code via Semantic Triangulation</a></div>
    <div class="paper-meta">
      📅 2025-11-15
    </div>
    <details class="paper-abstract">
      When generating code from natural language prompts, an LLM samples programs from a probability distribution, many of which might be incorrect. Sample consensus techniques - such as majority voting or validation against generated tests or specifications - aim to identify a correct program in the sample or abstain if none is valid. However, existing methods often fail to select a correct solution when its sampling probability is low, or when the problem permits multiple valid but non-equivalent solutions. Additionally, they often fail to abstain when no correct solution is present in the sample. To overcome these limitations, we introduce semantic triangulation, which transforms a programming problem in a way that non-trivially alters its semantics while preserving an exact, verifiable mapping between solutions before and after transformation. We theoretically establish that verifying consistency across such problem transformations increases confidence that generated programs reflect accurate generalization rather than spurious statistical correlations, enabling more reliable sample consensus and abstention. On the LiveCodeBench and CodeElo benchmarks, using GPT-4o and DeepSeek-V3 models, semantic triangulation increases reliability of generated code by 21% compared to the method that selects only high-confidence solutions with the probability threshold 0.5, while being able to pinpoint correct solutions at sampling probabilities as low as 0.14. Apart from that, it is also the only approach to consistently form true consensus on tasks with multiple valid but non-equivalent solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12286v1">Sangam: Chiplet-Based DRAM-PIM Accelerator with CXL Integration for LLM Inferencing</a></div>
    <div class="paper-meta">
      📅 2025-11-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are becoming increasingly data-intensive due to growing model sizes, and they are becoming memory-bound as the context length and, consequently, the key-value (KV) cache size increase. Inference, particularly the decoding phase, is dominated by memory-bound GEMV or flat GEMM operations with low operational intensity (OI), making it well-suited for processing-in-memory (PIM) approaches. However, existing in/near-memory solutions face critical limitations such as reduced memory capacity due to the high area cost of integrating processing elements (PEs) within DRAM chips, and limited PE capability due to the constraints of DRAM fabrication technology. This work presents a chiplet-based memory module that addresses these limitations by decoupling logic and memory into chiplets fabricated in heterogeneous technology nodes and connected via an interposer. The logic chiplets sustain high bandwidth access to the DRAM chiplets, which house the memory banks, and enable the integration of advanced processing components such as systolic arrays and SRAM-based buffers to accelerate memory-bound GEMM kernels, capabilities that were not feasible in prior PIM architectures. We propose Sangam, a CXL-attached PIM-chiplet based memory module that can either act as a drop-in replacement for GPUs or co-executes along side the GPUs. Sangam achieves speedup of 3.93, 4.22, 2.82x speedup in end-to-end query latency, 10.3, 9.5, 6.36x greater decoding throughput, and order of magnitude energy savings compared to an H100 GPU for varying input size, output length, and batch size on LLaMA 2-7B, Mistral-7B, and LLaMA 3-70B, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12271v1">MoralReason: Generalizable Moral Decision Alignment For LLM Agents Using Reasoning-Level Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-15
      | 💬 Accepted for AAAI 2026
    </div>
    <details class="paper-abstract">
      Large language models are increasingly influencing human moral decisions, yet current approaches focus primarily on evaluating rather than actively steering their moral decisions. We formulate this as an out-of-distribution moral alignment problem, where LLM agents must learn to apply consistent moral reasoning frameworks to scenarios beyond their training distribution. We introduce Moral-Reason-QA, a novel dataset extending 680 human-annotated, high-ambiguity moral scenarios with framework-specific reasoning traces across utilitarian, deontological, and virtue ethics, enabling systematic evaluation of moral generalization in realistic decision contexts. Our learning approach employs Group Relative Policy Optimization with composite rewards that simultaneously optimize decision alignment and framework-specific reasoning processes to facilitate learning of the underlying moral frameworks. Experimental results demonstrate successful generalization to unseen moral scenarios, with softmax-normalized alignment scores improving by +0.757 for utilitarian and +0.450 for deontological frameworks when tested on out-of-distribution evaluation sets. The experiments also reveal training challenges and promising directions that inform future research. These findings establish that LLM agents can be systematically trained to internalize and apply specific moral frameworks to novel situations, providing a critical foundation for AI safety as language models become more integrated into human decision-making processes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06194v2">NURBGen: High-Fidelity Text-to-CAD Generation through LLM-Driven NURBS Modeling</a></div>
    <div class="paper-meta">
      📅 2025-11-15
      | 💬 Accepted in AAAI 2026
    </div>
    <details class="paper-abstract">
      Generating editable 3D CAD models from natural language remains challenging, as existing text-to-CAD systems either produce meshes or rely on scarce design-history data. We present NURBGen, the first framework to generate high-fidelity 3D CAD models directly from text using Non-Uniform Rational B-Splines (NURBS). To achieve this, we fine-tune a large language model (LLM) to translate free-form texts into JSON representations containing NURBS surface parameters (\textit{i.e}, control points, knot vectors, degrees, and rational weights) which can be directly converted into BRep format using Python. We further propose a hybrid representation that combines untrimmed NURBS with analytic primitives to handle trimmed surfaces and degenerate regions more robustly, while reducing token complexity. Additionally, we introduce partABC, a curated subset of the ABC dataset consisting of individual CAD components, annotated with detailed captions using an automated annotation pipeline. NURBGen demonstrates strong performance on diverse prompts, surpassing prior methods in geometric fidelity and dimensional accuracy, as confirmed by expert evaluations. Code and dataset will be released publicly.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.14995v2">LLM-Enhanced Multi-Agent Reinforcement Learning with Expert Workflow for Real-Time P2P Energy Trading</a></div>
    <div class="paper-meta">
      📅 2025-11-15
    </div>
    <details class="paper-abstract">
      Real-time peer-to-peer (P2P) electricity markets dynamically adapt to fluctuations in renewable energy and variations in demand, maximizing economic benefits through instantaneous price responses while enhancing grid flexibility. However, scaling expert guidance for massive personalized prosumers poses critical challenges, including diverse decision-making demands and lack of customized modeling frameworks. This paper proposed an integrated large language model-multi-agent reinforcement learning (LLM-MARL) framework for real-time P2P energy trading to address challenges such as the limited technical capability of prosumers, the lack of expert experience, and security issues of distribution networks. LLMs are introduced as experts to generate personalized strategy, guiding MARL under the centralized training with decentralized execution (CTDE) paradigm through imitation learning. A differential attention-based critic network is designed to enhance convergence performance. Experimental results demonstrate that LLM generated strategies effectively substitute human experts. The proposed multi-agent imitation learning algorithms achieve significantly lower economic costs and voltage violation rates on test sets compared to baselines algorithms, while maintaining robust stability. This work provides an effective solution for real-time P2P electricity market decision-making by bridging expert knowledge with agent learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12236v1">Consistency Is the Key: Detecting Hallucinations in LLM Generated Text By Checking Inconsistencies About Key Facts</a></div>
    <div class="paper-meta">
      📅 2025-11-15
      | 💬 To appear at International Joint Conference on Natural Language Processing & Asia-Pacific Chapter of the Association for Computational Linguistics (IJCNLP-AACL), 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs), despite their remarkable text generation capabilities, often hallucinate and generate text that is factually incorrect and not grounded in real-world knowledge. This poses serious risks in domains like healthcare, finance, and customer support. A typical way to use LLMs is via the APIs provided by LLM vendors where there is no access to model weights or options to fine-tune the model. Existing methods to detect hallucinations in such settings where the model access is restricted or constrained by resources typically require making multiple LLM API calls, increasing latency and API cost. We introduce CONFACTCHECK, an efficient hallucination detection approach that does not leverage any external knowledge base and works on the simple intuition that responses to factual probes within the generated text should be consistent within a single LLM and across different LLMs. Rigorous empirical evaluation on multiple datasets that cover both the generation of factual texts and the open generation shows that CONFACTCHECK can detect hallucinated facts efficiently using fewer resources and achieves higher accuracy scores compared to existing baselines that operate under similar conditions. Our code is available here.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12224v1">RulePilot: An LLM-Powered Agent for Security Rule Generation</a></div>
    <div class="paper-meta">
      📅 2025-11-15
      | 💬 This paper has been accepted for publication at ICSE 2026
    </div>
    <details class="paper-abstract">
      The real-time demand for system security leads to the detection rules becoming an integral part of the intrusion detection life-cycle. Rule-based detection often identifies malicious logs based on the predefined grammar logic, requiring experts with deep domain knowledge for rule generation. Therefore, automation of rule generation can result in significant time savings and ease the burden of rule-related tasks on security engineers. In this paper, we propose RulePilot, which mimics human expertise via LLM-based agent for addressing rule-related challenges like rule creation or conversion. Using RulePilot, the security analysts do not need to write down the rules following the grammar, instead, they can just provide the annotations such as the natural-language-based descriptions of a rule, our RulePilot can automatically generate the detection rules without more intervention. RulePilot is equipped with the intermediate representation (IR), which abstracts the complexity of config rules into structured, standardized formats, allowing LLMs to focus on generation rules in a more manageable and consistent way. We present a comprehensive evaluation of RulePilot in terms of textual similarity and execution success abilities, showcasing RulePilot can generate high-fidelity rules, outperforming the baseline models by up to 107.4% in textual similarity to ground truths and achieving better detection accuracy in real-world execution tests. We perform a case study from our industry collaborators in Singapore, showcasing that RulePilot significantly help junior analysts/general users in the rule creation process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12217v1">AlignTree: Efficient Defense Against LLM Jailbreak Attacks</a></div>
    <div class="paper-meta">
      📅 2025-11-15
      | 💬 Accepted as an Oral Presentation at the 40th AAAI Conference on Artificial Intelligence (AAAI-26), January 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are vulnerable to adversarial attacks that bypass safety guidelines and generate harmful content. Mitigating these vulnerabilities requires defense mechanisms that are both robust and computationally efficient. However, existing approaches either incur high computational costs or rely on lightweight defenses that can be easily circumvented, rendering them impractical for real-world LLM-based systems. In this work, we introduce the AlignTree defense, which enhances model alignment while maintaining minimal computational overhead. AlignTree monitors LLM activations during generation and detects misaligned behavior using an efficient random forest classifier. This classifier operates on two signals: (i) the refusal direction -- a linear representation that activates on misaligned prompts, and (ii) an SVM-based signal that captures non-linear features associated with harmful content. Unlike previous methods, AlignTree does not require additional prompts or auxiliary guard models. Through extensive experiments, we demonstrate the efficiency and robustness of AlignTree across multiple LLMs and benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.12396v3">LLM-CoT Enhanced Graph Neural Recommendation with Harmonized Group Policy Optimization</a></div>
    <div class="paper-meta">
      📅 2025-11-15
    </div>
    <details class="paper-abstract">
      Graph neural networks (GNNs) have advanced recommender systems by modeling interaction relationships. However, existing graph-based recommenders rely on sparse ID features and do not fully exploit textual information, resulting in low information density within representations. Furthermore, graph contrastive learning faces challenges. Random negative sampling can introduce false negative samples, while fixed temperature coefficients cannot adapt to the heterogeneity of different nodes. In addition, current efforts to enhance recommendations with large language models (LLMs) have not fully utilized their Chain-of-Thought (CoT) reasoning capabilities to guide representation learning. To address these limitations, we introduces LGHRec (LLM-CoT Enhanced Graph Neural Recommendation with Harmonized Group Policy Optimization). This framework leverages the CoT reasoning ability of LLMs to generate semantic IDs, enriching reasoning processes and improving information density and semantic quality of representations. Moreover, we design a reinforcement learning algorithm, Harmonized Group Policy Optimization (HGPO), to optimize negative sampling strategies and temperature coefficients in contrastive learning. This approach enhances long-tail recommendation performance and ensures optimization consistency across different groups. Experimental results on three datasets demonstrate that LGHRec improves representation quality through semantic IDs generated by LLM's CoT reasoning and effectively boosts contrastive learning with HGPO. Our method outperforms several baseline models. The code is available at: https://anonymous.4open.science/r/LLM-Rec.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2406.08824v2">LLM-Driven Robots Risk Enacting Discrimination, Violence, and Unlawful Actions</a></div>
    <div class="paper-meta">
      📅 2025-11-15
      | 💬 Published in International Journal of Social Robotics (2025). 49 pages (65 with references and appendix), 27 Figures, 8 Tables. Andrew Hundt and Rumaisa Azeem are equal contribution co-first authors. The positions of the two co-first authors were swapped from arxiv version 1 with the written consent of all four authors. The Version of Record is available via DOI: 10.1007/s12369-025-01301-x
    </div>
    <details class="paper-abstract">
      Members of the Human-Robot Interaction (HRI) and Machine Learning (ML) communities have proposed Large Language Models (LLMs) as a promising resource for robotics tasks such as natural language interaction, household and workplace tasks, approximating 'common sense reasoning', and modeling humans. However, recent research has raised concerns about the potential for LLMs to produce discriminatory outcomes and unsafe behaviors in real-world robot experiments and applications. To assess whether such concerns are well placed in the context of HRI, we evaluate several highly-rated LLMs on discrimination and safety criteria. Our evaluation reveals that LLMs are currently unsafe for people across a diverse range of protected identity characteristics, including, but not limited to, race, gender, disability status, nationality, religion, and their intersections. Concretely, we show that LLMs produce directly discriminatory outcomes- e.g., 'gypsy' and 'mute' people are labeled untrustworthy, but not 'european' or 'able-bodied' people. We find various such examples of direct discrimination on HRI tasks such as facial expression, proxemics, security, rescue, and task assignment. Furthermore, we test models in settings with unconstrained natural language (open vocabulary) inputs, and find they fail to act safely, generating responses that accept dangerous, violent, or unlawful instructions-such as incident-causing misstatements, taking people's mobility aids, and sexual predation. Our results underscore the urgent need for systematic, routine, and comprehensive risk assessments and assurances to improve outcomes and ensure LLMs only operate on robots when it is safe, effective, and just to do so. We provide code to reproduce our experiments at https://github.com/rumaisa-azeem/llm-robots-discrimination-safety .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12033v1">EARL: Entropy-Aware RL Alignment of LLMs for Reliable RTL Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-11-15
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have demonstrated significant potential in hardware design automation, particularly in using natural language to synthesize Register-Transfer Level (RTL) code. Despite this progress, a gap remains between model capability and the demands of real-world RTL design, including syntax errors, functional hallucinations, and weak alignment to designer intent. Reinforcement Learning with Verifiable Rewards (RLVR) offers a promising approach to bridge this gap, as hardware provides executable and formally checkable signals that can be used to further align model outputs with design intent. However, in long, structured RTL code sequences, not all tokens contribute equally to functional correctness, and naïvely spreading gradients across all tokens dilutes learning signals. A key insight from our entropy analysis in RTL generation is that only a small fraction of tokens (e.g., always, if, assign, posedge) exhibit high uncertainty and largely influence control flow and module structure. To address these challenges, we present EARL, an Entropy-Aware Reinforcement Learning framework for Verilog generation. EARL performs policy optimization using verifiable reward signals and introduces entropy-guided selective updates that gate policy gradients to high-entropy tokens. This approach preserves training stability and concentrates gradient updates on functionally important regions of code. Our experiments on VerilogEval and RTLLM show that EARL improves functional pass rates over prior LLM baselines by up to 14.7%, while reducing unnecessary updates and improving training stability. These results indicate that focusing RL on critical, high-uncertainty tokens enables more reliable and targeted policy improvement for structured RTL code generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12031v1">Striking the Right Balance between Compute and Copy: Improving LLM Inferencing Under Speculative Decoding</a></div>
    <div class="paper-meta">
      📅 2025-11-15
    </div>
    <details class="paper-abstract">
      With the skyrocketing costs of GPUs and their virtual instances in the cloud, there is a significant desire to use CPUs for large language model (LLM) inference. KV cache update, often implemented as allocation, copying, and in-place strided update for each generated token, incurs significant overhead. As the sequence length increases, the allocation and copy overheads dominate the performance. Alternate approaches may allocate large KV tensors upfront to enable in-place updates, but these matrices (with zero-padded rows) cause redundant computations. In this work, we propose a new KV cache allocation mechanism called Balancing Memory and Compute (BMC). BMC allocates, once every r iterations, KV tensors with r redundant rows, allowing in-place update without copy overhead for those iterations, but at the expense of a small amount of redundant computation. Second, we make an interesting observation that the extra rows allocated in the KV tensors and the resulting redundant computation can be repurposed for Speculative Decoding (SD) that improves token generation efficiency. Last, BMC represents a spectrum of design points with different values of r. To identify the best-performing design point(s), we derive a simple analytical model for BMC. The proposed BMC method achieves an average throughput acceleration of up to 3.2x over baseline HuggingFace (without SD). Importantly when we apply BMC with SD, it results in an additional speedup of up to 1.39x, over and above the speedup offered by SD. Further, BMC achieves a throughput acceleration of up to 1.36x and 2.29x over state-of-the-art inference servers vLLM and DeepSpeed, respectively. Although the BMC technique is evaluated extensively across different classes of CPUs (desktop and server class), we also evaluate the scheme with GPUs and demonstrate that it works well for GPUs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12014v1">CURE: Cultural Understanding and Reasoning Evaluation - A Framework for "Thick" Culture Alignment Evaluation in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-15
      | 💬 7 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed in culturally diverse environments, yet existing evaluations of cultural competence remain limited. Existing methods focus on de-contextualized correctness or forced-choice judgments, overlooking the need for cultural understanding and reasoning required for appropriate responses. To address this gap, we introduce a set of benchmarks that, instead of directly probing abstract norms or isolated statements, present models with realistic situational contexts that require culturally grounded reasoning. In addition to the standard Exact Match metric, we introduce four complementary metrics (Coverage, Specificity, Connotation, and Coherence) to capture different dimensions of model's response quality. Empirical analysis across frontier models reveals that thin evaluation systematically overestimates cultural competence and produces unstable assessments with high variance. In contrast, thick evaluation exposes differences in reasoning depth, reduces variance, and provides more stable, interpretable signals of cultural understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11896v1">VULPO: Context-Aware Vulnerability Detection via On-Policy LLM Optimization</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      The widespread reliance on open-source software dramatically increases the risk of vulnerability exploitation, underscoring the need for effective and scalable vulnerability detection (VD). Existing VD techniques, whether traditional machine learning-based or LLM-based approaches like prompt engineering, supervised fine-tuning, or off-policy preference optimization, remain fundamentally limited in their ability to perform context-aware analysis: They depend on fixed inputs or static preference datasets, cannot adaptively explore repository-level dependencies, and are constrained by function-level benchmarks that overlook critical vulnerability context. This paper introduces Vulnerability-Adaptive Policy Optimization (VULPO), an on-policy LLM reinforcement learning framework for context-aware VD. To support training and evaluation, we first construct ContextVul, a new dataset that augments high-quality function-level samples with lightweight method to extract repository-level context information. We then design multi-dimensional reward structuring that jointly captures prediction correctness, vulnerability localization accuracy, and the semantic relevance of vulnerability analysis, thereby guiding the model toward comprehensive contextual reasoning. To address the asymmetric difficulty of different vulnerability cases and mitigate reward hacking, VULPO incorporates label-level and sample-level difficulty-adaptive reward scaling, encouraging the model to explore challenging cases while maintaining balanced reward distribution. Extensive experiments demonstrate the superiority of our VULPO framework in context-aware VD: Our VULPO-4B substantially outperforms existing VD baselines based on prompt engineering and off-policy optimization, improving F1 by 85% over Qwen3-4B and achieving performance comparable to a 150x larger-scale model, DeepSeek-R1-0528.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11885v1">Flash-Fusion: Enabling Expressive, Low-Latency Queries on IoT Sensor Streams with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 12 pages, 5 figures. Under review
    </div>
    <details class="paper-abstract">
      Smart cities and pervasive IoT deployments have generated interest in IoT data analysis across transportation and urban planning. At the same time, Large Language Models offer a new interface for exploring IoT data - particularly through natural language. Users today face two key challenges when working with IoT data using LLMs: (1) data collection infrastructure is expensive, producing terabytes of low-level sensor readings that are too granular for direct use, and (2) data analysis is slow, requiring iterative effort and technical expertise. Directly feeding all IoT telemetry to LLMs is impractical due to finite context windows, prohibitive token costs at scale, and non-interactive latencies. What is missing is a system that first parses a user's query to identify the analytical task, then selects the relevant data slices, and finally chooses the right representation before invoking an LLM. We present Flash-Fusion, an end-to-end edge-cloud system that reduces the IoT data collection and analysis burden on users. Two principles guide its design: (1) edge-based statistical summarization (achieving 73.5% data reduction) to address data volume, and (2) cloud-based query planning that clusters behavioral data and assembles context-rich prompts to address data interpretation. We deploy Flash-Fusion on a university bus fleet and evaluate it against a baseline that feeds raw data to a state-of-the-art LLM. Flash-Fusion achieves a 95% latency reduction and 98% decrease in token usage and cost while maintaining high-quality responses. It enables personas across disciplines - safety officers, urban planners, fleet managers, and data scientists - to efficiently iterate over IoT data without the burden of manual query authoring or preprocessing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11881v1">Better LLM Reasoning via Dual-Play</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable progress through Reinforcement Learning with Verifiable Rewards (RLVR), yet still rely heavily on external supervision (e.g., curated labels). Adversarial learning, particularly through self-play, offers a promising alternative that enables models to iteratively learn from themselves - thus reducing reliance on external supervision. Dual-play extends adversarial learning by assigning specialized roles to two models and training them against each other, fostering sustained competition and mutual evolution. Despite its promise, adapting dual-play training to LLMs remains limited, largely due to their susceptibility to reward hacking and training instability. In this paper, we introduce PasoDoble, a novel LLM dual-play framework. PasoDoble adversarially trains two models initialized from the same base model: a Proposer, which generates challenging questions with ground-truth answers, and a Solver, which attempts to solve them. We enrich the Proposer with knowledge from a pre-training dataset to ensure the questions' quality and diversity. To avoid reward hacking, the Proposer is rewarded for producing only valid questions that push the Solver's limit, while the Solver is rewarded for solving them correctly, and both are updated jointly. To further enhance training stability, we introduce an optional offline paradigm that decouples Proposer and Solver updates, alternately updating each for several steps while holding the other fixed. Notably, PasoDoble operates without supervision during training. Experimental results show that PasoDoble can improve the reasoning performance of LLMs. Our project page is available at https://hcy123902.github.io/PasoDoble.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11867v1">Identifying Imaging Follow-Up in Radiology Reports: A Comparative Analysis of Traditional ML and LLM Approaches</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 Submitted to LREC 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown considerable promise in clinical natural language processing, yet few domain-specific datasets exist to rigorously evaluate their performance on radiology tasks. In this work, we introduce an annotated corpus of 6,393 radiology reports from 586 patients, each labeled for follow-up imaging status, to support the development and benchmarking of follow-up adherence detection systems. Using this corpus, we systematically compared traditional machine-learning classifiers, including logistic regression (LR), support vector machines (SVM), Longformer, and a fully fine-tuned Llama3-8B-Instruct, with recent generative LLMs. To evaluate generative LLMs, we tested GPT-4o and the open-source GPT-OSS-20B under two configurations: a baseline (Base) and a task-optimized (Advanced) setting that focused inputs on metadata, recommendation sentences, and their surrounding context. A refined prompt for GPT-OSS-20B further improved reasoning accuracy. Performance was assessed using precision, recall, and F1 scores with 95% confidence intervals estimated via non-parametric bootstrapping. Inter-annotator agreement was high (F1 = 0.846). GPT-4o (Advanced) achieved the best performance (F1 = 0.832), followed closely by GPT-OSS-20B (Advanced; F1 = 0.828). LR and SVM also performed strongly (F1 = 0.776 and 0.775), underscoring that while LLMs approach human-level agreement through prompt optimization, interpretable and resource-efficient models remain valuable baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11829v1">Towards Autoformalization of LLM-generated Outputs for Requirement Verification</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 To be submitted for publication
    </div>
    <details class="paper-abstract">
      Autoformalization, the process of translating informal statements into formal logic, has gained renewed interest with the emergence of powerful Large Language Models (LLMs). While LLMs show promise in generating structured outputs from natural language (NL), such as Gherkin Scenarios from NL feature requirements, there's currently no formal method to verify if these outputs are accurate. This paper takes a preliminary step toward addressing this gap by exploring the use of a simple LLM-based autoformalizer to verify LLM-generated outputs against a small set of natural language requirements. We conducted two distinct experiments. In the first one, the autoformalizer successfully identified that two differently-worded NL requirements were logically equivalent, demonstrating the pipeline's potential for consistency checks. In the second, the autoformalizer was used to identify a logical inconsistency between a given NL requirement and an LLM-generated output, highlighting its utility as a formal verification tool. Our findings, while limited, suggest that autoformalization holds significant potential for ensuring the fidelity and logical consistency of LLM-generated outputs, laying a crucial foundation for future, more extensive studies into this novel application.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11828v1">Conformal Constrained Policy Optimization for Cost-Effective LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have recently made tremendous progress towards solving challenging AI problems, they have done so at increasingly steep computational and API costs. We propose a novel strategy where we combine multiple LLM models with varying cost/accuracy tradeoffs in an agentic manner, where models and tools are run in sequence as determined by an orchestration model to minimize cost subject to a user-specified level of reliability; this constraint is formalized using conformal prediction to provide guarantees. To solve this problem, we propose Conformal Constrained Policy Optimization (CCPO), a training paradigm that integrates constrained policy optimization with off-policy reinforcement learning and recent advances in online conformal prediction. CCPO jointly optimizes a cost-aware policy (score function) and an adaptive threshold. Across two multi-hop question answering benchmarks, CCPO achieves up to a 30% cost reduction compared to other cost-aware baselines and LLM-guided methods without compromising reliability. Our approach provides a principled and practical framework for deploying LLM agents that are significantly more cost-effective while maintaining reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.04652v3">LLM Collaboration With Multi-Agent Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      A large amount of work has been done in Multi-Agent Systems (MAS) for modeling and solving problems with multiple interacting agents. However, most LLMs are pretrained independently and not specifically optimized for coordination. Existing LLM fine-tuning frameworks rely on individual rewards, which require complex reward designs for each agent to encourage collaboration. To address these challenges, we model LLM collaboration as a cooperative Multi-Agent Reinforcement Learning (MARL) problem. We develop a multi-agent, multi-turn algorithm, Multi-Agent Group Relative Policy Optimization (MAGRPO), to solve it, building on current RL approaches for LLMs as well as MARL techniques. Our experiments on LLM writing and coding collaboration demonstrate that fine-tuning MAS with MAGRPO enables agents to generate high-quality responses efficiently through effective cooperation. Our approach opens the door to using other MARL methods for LLMs and highlights the associated challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11823v1">CollaClassroom: An AI-Augmented Collaborative Learning Platform with LLM Support in the Context of Bangladeshi University Students</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      CollaClassroom is an AI-enhanced platform that embeds large language models (LLMs) into both individual and group study panels to support real-time collaboration. We evaluate CollaClassroom with Bangladeshi university students (N = 12) through a small-group study session and a pre-post survey. Participants have substantial prior experience with collaborative learning and LLMs and express strong receptivity to LLM-assisted study (92% agree/strongly agree). Usability ratings are positive, including high learnability(67% "easy"), strong reliability (83% "reliable"), and low frustration (83% "not at all"). Correlational analyses show that participants who perceive the LLM as supporting equal participation also view it as a meaningful contributor to discussions (r = 0.86). Moreover, their pre-use expectations of LLM value align with post-use assessments (r = 0.61). These findings suggest that LLMs can enhance engagement and perceived learning when designed to promote equitable turn-taking and transparency across individual and shared spaces. The paper contributes an empirically grounded account of AI-mediated collaboration in a Global South higher-education context, with design implications for fairness-aware orchestration of human-AI teamwork.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11816v1">Do LLMs Really Struggle at NL-FOL Translation? Revealing their Strengths via a Novel Benchmarking Strategy</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 Full version of the paper accepted for publication at The 40th Annual AAAI Conference on Artificial Intelligence (AAAI 2026)
    </div>
    <details class="paper-abstract">
      Due to its expressiveness and unambiguous nature, First-Order Logic (FOL) is a powerful formalism for representing concepts expressed in natural language (NL). This is useful, e.g., for specifying and verifying desired system properties. While translating FOL into human-readable English is relatively straightforward, the inverse problem, converting NL to FOL (NL-FOL translation), has remained a longstanding challenge, for both humans and machines. Although the emergence of Large Language Models (LLMs) promised a breakthrough, recent literature provides contrasting results on their ability to perform NL-FOL translation. In this work, we provide a threefold contribution. First, we critically examine existing datasets and protocols for evaluating NL-FOL translation performance, revealing key limitations that may cause a misrepresentation of LLMs' actual capabilities. Second, to overcome these shortcomings, we propose a novel evaluation protocol explicitly designed to distinguish genuine semantic-level logical understanding from superficial pattern recognition, memorization, and dataset contamination. Third, using this new approach, we show that state-of-the-art, dialogue-oriented LLMs demonstrate strong NL-FOL translation skills and a genuine grasp of sentence-level logic, whereas embedding-centric models perform markedly worse.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11788v1">MALBO: Optimizing LLM-Based Multi-Agent Teams via Multi-Objective Bayesian Optimization</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 Master's Thesis, University of Milano-Bicocca, 2025
    </div>
    <details class="paper-abstract">
      The optimal assignment of Large Language Models (LLMs) to specialized roles in multi-agent systems is a significant challenge, defined by a vast combinatorial search space, expensive black-box evaluations, and an inherent trade-off between performance and cost. Current optimization methods focus on single-agent settings and lack a principled framework for this multi-agent, multi-objective problem. This thesis introduces MALBO (Multi-Agent LLM Bayesian Optimization), a systematic framework designed to automate the efficient composition of LLM-based agent teams. We formalize the assignment challenge as a multi-objective optimization problem, aiming to identify the Pareto front of configurations between task accuracy and inference cost. The methodology employs multi-objective Bayesian Optimization (MOBO) with independent Gaussian Process surrogate models. By searching over a continuous feature-space representation of the LLMs, this approach performs a sample-efficient exploration guided by the expected hypervolume improvement. The primary contribution is a principled and automated methodology that yields a Pareto front of optimal team configurations. Our results demonstrate that the Bayesian optimization phase, compared to an initial random search, maintained a comparable average performance while reducing the average configuration cost by over 45%. Furthermore, MALBO identified specialized, heterogeneous teams that achieve cost reductions of up to 65.8% compared to homogeneous baselines, all while maintaining maximum performance. The framework thus provides a data-driven tool for deploying cost-effective and highly specialized multi-agent AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11764v1">Demystify, Use, Reflect: Preparing students to be informed LLM-users</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 2 pages 1 table Submitted to SIGCSE 2026
    </div>
    <details class="paper-abstract">
      We transitioned our post-CS1 course that introduces various subfields of computer science so that it integrates Large Language Models (LLMs) in a structured, critical, and practical manner. It aims to help students develop the skills needed to engage meaningfully and responsibly with AI. The course now includes explicit instruction on how LLMs work, exposure to current tools, ethical issues, and activities that encourage student reflection on personal use of LLMs as well as the larger evolving landscape of AI-assisted programming. In class, we demonstrate the use and verification of LLM outputs, guide students in the use of LLMs as an ingredient in a larger problem-solving loop, and require students to disclose and acknowledge the nature and extent of LLM assistance. Throughout the course, we discuss risks and benefits of LLMs across CS subfields. In our first iteration of the course, we collected and analyzed data from students pre and post surveys. Student understanding of how LLMs work became more technical, and their verification and use of LLMs shifted to be more discerning and collaborative. These strategies can be used in other courses to prepare students for the AI-integrated future.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13765v1">PROF: An LLM-based Reward Code Preference Optimization Framework for Offline Imitation Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      Offline imitation learning (offline IL) enables training effective policies without requiring explicit reward annotations. Recent approaches attempt to estimate rewards for unlabeled datasets using a small set of expert demonstrations. However, these methods often assume that the similarity between a trajectory and an expert demonstration is positively correlated with the reward, which oversimplifies the underlying reward structure. We propose PROF, a novel framework that leverages large language models (LLMs) to generate and improve executable reward function codes from natural language descriptions and a single expert trajectory. We propose Reward Preference Ranking (RPR), a novel reward function quality assessment and ranking strategy without requiring environment interactions or RL training. RPR calculates the dominance scores of the reward functions, where higher scores indicate better alignment with expert preferences. By alternating between RPR and text-based gradient optimization, PROF fully automates the selection and refinement of optimal reward functions for downstream policy learning. Empirical results on D4RL demonstrate that PROF surpasses or matches recent strong baselines across numerous datasets and domains, highlighting the effectiveness of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.13812v2">The Empty Chair: Using LLMs to Raise Missing Perspectives in Policy Deliberations</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop: PersonaLLM: Workshop on LLM Persona Modeling
    </div>
    <details class="paper-abstract">
      Deliberation is essential to well-functioning democracies, yet physical, economic, and social barriers often exclude certain groups, reducing representativeness and contributing to issues like group polarization. In this work, we explore the use of large language model (LLM) personas to introduce missing perspectives in policy deliberations. We develop and evaluate a tool that transcribes conversations in real-time and simulates input from relevant but absent stakeholders. We deploy this tool in a 19-person student citizens' assembly on campus sustainability. Participants and facilitators found that the tool was useful to spark new discussions and surfaced valuable perspectives they had not previously considered. However, they also raised skepticism about the ability of LLMs to accurately characterize the perspectives of different groups, especially ones that are already underrepresented. Overall, this case study highlights that while AI personas can usefully surface new perspectives and prompt discussion in deliberative settings, their successful deployment depends on clarifying their limitations and emphasizing that they complement rather than replace genuine participation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.25506v2">Reflections on the Reproducibility of Commercial LLM Performance in Empirical Software Engineering Studies</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      Large Language Models have gained remarkable interest in industry and academia. The increasing interest in LLMs in academia is also reflected in the number of publications on this topic over the last years. For instance, alone 78 of the around 425 publications at ICSE 2024 performed experiments with LLMs. Conducting empirical studies with LLMs remains challenging and raises questions on how to achieve reproducible results, for both other researchers and practitioners. One important step towards excelling in empirical research on LLMs and their application is to first understand to what extent current research results are eventually reproducible and what factors may impede reproducibility. This investigation is within the scope of our work. We contribute an analysis of the reproducibility of LLM-centric studies, provide insights into the factors impeding reproducibility, and discuss suggestions on how to improve the current state. In particular, we studied the 86 articles describing LLM-centric studies, published at ICSE 2024 and ASE 2024. Of the 86 articles, 18 provided research artefacts and used OpenAI models. We attempted to replicate those 18 studies. Of the 18 studies, only five were fit for reproduction. For none of the five studies, we were able to fully reproduce the results. Two studies seemed to be partially reproducible, and three studies did not seem to be reproducible. Our results highlight not only the need for stricter research artefact evaluations but also for more robust study designs to ensure the reproducible value of future publications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10222v2">Speech-Audio Compositional Attacks on Multimodal LLMs and Their Mitigation with SALMONN-Guard</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      Recent progress in large language models (LLMs) has enabled understanding of both speech and non-speech audio, but exposing new safety risks emerging from complex audio inputs that are inadequately handled by current safeguards. We introduce SACRED-Bench (Speech-Audio Composition for RED-teaming) to evaluate the robustness of LLMs under complex audio-based attacks. Unlike existing perturbation-based methods that rely on noise optimization or white-box access, SACRED-Bench exploits speech-audio composition mechanisms. SACRED-Bench adopts three mechanisms: (a) speech overlap and multi-speaker dialogue, which embeds harmful prompts beneath or alongside benign speech; (b) speech-audio mixture, which imply unsafe intent via non-speech audio alongside benign speech or audio; and (c) diverse spoken instruction formats (open-ended QA, yes/no) that evade text-only filters. Experiments show that, even Gemini 2.5 Pro, the state-of-the-art proprietary LLM, still exhibits 66% attack success rate in SACRED-Bench test set, exposing vulnerabilities under cross-modal, speech-audio composition attacks. To bridge this gap, we propose SALMONN-Guard, a safeguard LLM that jointly inspects speech, audio, and text for safety judgments, reducing attack success down to 20%. Our results highlight the need for audio-aware defenses for the safety of multimodal LLMs. The benchmark and SALMONN-Guard checkpoints can be found at https://huggingface.co/datasets/tsinghua-ee/SACRED-Bench. Warning: this paper includes examples that may be offensive or harmful.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.10624v3">Comprehension Without Competence: Architectural Limits of LLMs in Symbolic Computation and Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 v2: Two TMLR revision rounds addressing reviewer feedback. Added real-world validation (3.4), interpretability analysis (7), computational hallucination framework, strengthened theory. v3: Sec 3.2 - added transformer architecture diagram, clarified UAT capacity vs computational limits, improved role specialization theorem presentation
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) display striking surface fluency yet systematically fail at tasks requiring symbolic reasoning, arithmetic accuracy, and logical consistency. This paper offers a structural diagnosis of such failures, revealing a persistent gap between \textit{comprehension} and \textit{competence}. Through controlled experiments and architectural analysis, we demonstrate that LLMs often articulate correct principles without reliably applying them--a failure rooted not in knowledge access, but in computational execution. We term this phenomenon the computational \textit{split-brain syndrome}, where instruction and action pathways are geometrically and functionally dissociated. This core limitation recurs across domains, from mathematical operations to relational inferences, and explains why model behavior remains brittle even under idealized prompting. We argue that LLMs function as powerful pattern completion engines, but lack the architectural scaffolding for principled, compositional reasoning. Our findings delineate the boundary of current LLM capabilities and motivate future models with metacognitive control, principle lifting, and structurally grounded execution. This diagnosis also clarifies why mechanistic interpretability findings may reflect training-specific pattern coordination rather than universal computational principles, and why the geometric separation between instruction and execution pathways suggests limitations in neural introspection and mechanistic analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11356v1">SEAL: Subspace-Anchored Watermarks for LLM Ownership</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable success across a wide range of natural language processing tasks, demonstrating human-level performance in text generation, reasoning, and question answering. However, training such models requires substantial computational resources, large curated datasets, and sophisticated alignment procedures. As a result, they constitute highly valuable intellectual property (IP) assets that warrant robust protection mechanisms. Existing IP protection approaches suffer from critical limitations. Model fingerprinting techniques can identify model architectures but fail to establish ownership of specific model instances. In contrast, traditional backdoor-based watermarking methods embed behavioral anomalies that can be easily removed through common post-processing operations such as fine-tuning or knowledge distillation. We propose SEAL, a subspace-anchored watermarking framework that embeds multi-bit signatures directly into the model's latent representational space, supporting both white-box and black-box verification scenarios. Our approach leverages model editing techniques to align the hidden representations of selected anchor samples with predefined orthogonal bit vectors. This alignment embeds the watermark while preserving the model's original factual predictions, rendering the watermark functionally harmless and stealthy. We conduct comprehensive experiments on multiple benchmark datasets and six prominent LLMs, comparing SEAL with 11 existing fingerprinting and watermarking methods to demonstrate its superior effectiveness, fidelity, efficiency, and robustness. Furthermore, we evaluate SEAL under potential knowledgeable attacks and show that it maintains strong verification performance even when adversaries possess knowledge of the watermarking mechanism and the embedded signatures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.16851v2">Interpretable LLM Guardrails via Sparse Representation Steering</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit impressive capabilities in generation tasks but are prone to producing harmful, misleading, or biased content, posing significant ethical and safety concerns. To mitigate such risks, representation engineering, which steer model behavior toward desired attributes by injecting carefully designed steering vectors into LLM's representations at inference time, has emerged as a promising alternative to fine-tuning approaches. However, due to the semantically entangled nature of LLM's representation, existing representation engineering methods still suffer from several limitations: limited fine-grained controllability, content quality degradation, and conflict in multi-attribute control. To overcome these challenges, we propose Sparse Representation Steering (SRS), a novel framework that achieves fine-grained and interpretable control over LLM behavior by first disentangling internal activations into a sparse, semantically meaningful representation space, and then selectively steering relevant dimensions. Specifically, SRS leverages a pretrained Sparse Autoencoder (SAE) to transform dense, entangled activation patterns into a sparse monosemantic feature space. To identify relevant features, SRS contrasts sparse activations from positive and negative prompt pairs and measures their bidirectional KL divergence to locate dimensions most associated with the target attribute. We conduct comprehensive experiments on Gemma-2 series model across three alignment dimensions, i.e., safety, fairness, and truthfulness, to evaluate the effectiveness of SRS. Results show that SRS consistently outperforms existing steering methods, which achieves significantly improved controllability across both single and multiple attribute settings, while preserving high linguistic quality and general ability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11306v1">iMAD: Intelligent Multi-Agent Debate for Efficient and Accurate LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 Accepted in AAAI 2026 (Oral)
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agent systems have advanced rapidly, driven by their strong generalization in zero-shot settings. To further enhance reasoning and accuracy on complex tasks, Multi-Agent Debate (MAD) has emerged as a promising framework that engages multiple LLM agents in structured debates to encourage diverse reasoning. However, triggering MAD for every query is inefficient, as it incurs substantial computational (token) cost and may even degrade accuracy by overturning correct single-agent answers. To address these limitations, we propose intelligent Multi-Agent Debate (iMAD), a token-efficient framework that selectively triggers MAD only when it is likely to be beneficial (i.e., correcting an initially wrong answer). To achieve this goal, iMAD learns generalizable model behaviors to make accurate debate decisions. Specifically, iMAD first prompts a single agent to produce a structured self-critique response, from which we extract 41 interpretable linguistic and semantic features capturing hesitation cues. Then, iMAD uses a lightweight debate-decision classifier, trained using our proposed FocusCal loss, to determine whether to trigger MAD, enabling robust debate decisions without test dataset-specific tuning. Through extensive experiments using six (visual) question answering datasets against five competitive baselines, we have shown that iMAD significantly reduces token usage (by up to 92%) while also improving final answer accuracy (by up to 13.5%).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11258v1">KGQuest: Template-Driven QA Generation from Knowledge Graphs with LLM-Based Refinement</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      The generation of questions and answers (QA) from knowledge graphs (KG) plays a crucial role in the development and testing of educational platforms, dissemination tools, and large language models (LLM). However, existing approaches often struggle with scalability, linguistic quality, and factual consistency. This paper presents a scalable and deterministic pipeline for generating natural language QA from KGs, with an additional refinement step using LLMs to further enhance linguistic quality. The approach first clusters KG triplets based on their relations, creating reusable templates through natural language rules derived from the entity types of objects and relations. A module then leverages LLMs to refine these templates, improving clarity and coherence while preserving factual accuracy. Finally, the instantiation of answer options is achieved through a selection strategy that introduces distractors from the KG. Our experiments demonstrate that this hybrid approach efficiently generates high-quality QA pairs, combining scalability with fluency and linguistic precision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11257v1">AIonopedia: an LLM agent orchestrating multimodal learning for ionic liquid discovery</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      The discovery of novel Ionic Liquids (ILs) is hindered by critical challenges in property prediction, including limited data, poor model accuracy, and fragmented workflows. Leveraging the power of Large Language Models (LLMs), we introduce AIonopedia, to the best of our knowledge, the first LLM agent for IL discovery. Powered by an LLM-augmented multimodal domain foundation model for ILs, AIonopedia enables accurate property predictions and incorporates a hierarchical search architecture for molecular screening and design. Trained and evaluated on a newly curated and comprehensive IL dataset, our model delivers superior performance. Complementing these results, evaluations on literature-reported systems indicate that the agent can perform effective IL modification. Moving beyond offline tests, the practical efficacy was further confirmed through real-world wet-lab validation, in which the agent demonstrated exceptional generalization capabilities on challenging out-of-distribution tasks, underscoring its ability to accelerate real-world IL discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11255v1">Align$^3$GR: Unified Multi-Level Alignment for LLM-based Generative Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 Accepted by AAAI 2026 (Oral)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate significant advantages in leveraging structured world knowledge and multi-step reasoning capabilities. However, fundamental challenges arise when transforming LLMs into real-world recommender systems due to semantic and behavioral misalignment. To bridge this gap, we propose Align$^3$GR, a novel framework that unifies token-level, behavior modeling-level, and preference-level alignment. Our approach introduces: Dual tokenization fusing user-item semantic and collaborative signals. Enhanced behavior modeling with bidirectional semantic alignment. Progressive DPO strategy combining self-play (SP-DPO) and real-world feedback (RF-DPO) for dynamic preference adaptation. Experiments show Align$^3$GR outperforms the SOTA baseline by +17.8% in Recall@10 and +20.2% in NDCG@10 on the public dataset, with significant gains in online A/B tests and full-scale deployment on an industrial large-scale recommendation platform.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11252v1">UAVBench: An Open Benchmark Dataset for Autonomous and Agentic AI UAV Systems via LLM-Generated Flight Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 18 pages, 5 Figures
    </div>
    <details class="paper-abstract">
      Autonomous aerial systems increasingly rely on large language models (LLMs) for mission planning, perception, and decision-making, yet the lack of standardized and physically grounded benchmarks limits systematic evaluation of their reasoning capabilities. To address this gap, we introduce UAVBench, an open benchmark dataset comprising 50,000 validated UAV flight scenarios generated through taxonomy-guided LLM prompting and multi-stage safety validation. Each scenario is encoded in a structured JSON schema that includes mission objectives, vehicle configuration, environmental conditions, and quantitative risk labels, providing a unified representation of UAV operations across diverse domains. Building on this foundation, we present UAVBench_MCQ, a reasoning-oriented extension containing 50,000 multiple-choice questions spanning ten cognitive and ethical reasoning styles, ranging from aerodynamics and navigation to multi-agent coordination and integrated reasoning. This framework enables interpretable and machine-checkable assessment of UAV-specific cognition under realistic operational contexts. We evaluate 32 state-of-the-art LLMs, including GPT-5, ChatGPT-4o, Gemini 2.5 Flash, DeepSeek V3, Qwen3 235B, and ERNIE 4.5 300B, and find strong performance in perception and policy reasoning but persistent challenges in ethics-aware and resource-constrained decision-making. UAVBench establishes a reproducible and physically grounded foundation for benchmarking agentic AI in autonomous aerial systems and advancing next-generation UAV reasoning intelligence. To support open science and reproducibility, we release the UAVBench dataset, the UAVBench_MCQ benchmark, evaluation scripts, and all related materials on GitHub at https://github.com/maferrag/UAVBench
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11250v1">Prompt Engineering vs. Fine-Tuning for LLM-Based Vulnerability Detection in Solana and Algorand Smart Contracts</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      Smart contracts have emerged as key components within decentralized environments, enabling the automation of transactions through self-executing programs. While these innovations offer significant advantages, they also present potential drawbacks if the smart contract code is not carefully designed and implemented. This paper investigates the capability of large language models (LLMs) to detect OWASP-inspired vulnerabilities in smart contracts beyond the Ethereum Virtual Machine (EVM) ecosystem, focusing specifically on Solana and Algorand. Given the lack of labeled datasets for non-EVM platforms, we design a synthetic dataset of annotated smart contract snippets in Rust (for Solana) and PyTeal (for Algorand), structured around a vulnerability taxonomy derived from OWASP. We evaluate LLMs under three configurations: prompt engineering, fine-tuning, and a hybrid of both, comparing their performance on different vulnerability categories. Experimental results show that prompt engineering achieves general robustness, while fine-tuning improves precision and recall on less semantically rich languages such as TEAL. Additionally, we analyze how the architectural differences of Solana and Algorand influence the manifestation and detectability of vulnerabilities, offering platform-specific mappings that highlight limitations in existing security tooling. Our findings suggest that LLM-based approaches are viable for static vulnerability detection in smart contracts, provided domain-specific data and categorization are integrated into training pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11248v1">T-MAN: Enabling End-to-End Low-Bit LLM Inference on NPUs via Unified Table Lookup</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed on customer devices. To support them, current devices are adopting SoCs (System on Chip) with NPUs (Neural Processing Unit) installed. Although high performance is expected, LLM inference on NPUs is slower than its CPU counterpart. The reason is that NPUs have poor performance on computations other than GEMM, like dequantization. Current works either disaggregate prefill on the NPUs and decoding on the CPUs, or put both on the NPUs but with an accuracy loss. To solve this issue, based on the insight that low-bit can enable target computation encoded within an acceptably sized table, we propose table lookup to subsume hardware operations otherwise unsupported. To realize this, we overcome the conflicting hardware behavior of prefill and decoding to design a unified table layout and tiling through (1) fused two-level table-based dequantization and (2) concurrency-hierarchy-guided tiling. Based on that, we implement the prefill phase by three-stage pipeline and map the table-lookup-based decoding to NPU's vector units. Results show 1.4x and 3.1x speedup for prefill and decoding respectively, and 84% energy savings compared to the baseline NPU methods. The code is available at https://github.com/microsoft/T-MAC/tree/main/t-man.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.09719v3">ICL-Router: In-Context Learned Model Representations for LLM Routing</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 Accepted by AAAI 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often exhibit complementary strengths. Model routing harnesses these strengths by dynamically directing each query to the most suitable model, given a candidate model pool. However, routing performance relies on accurate model representations, and adding new models typically requires retraining, limiting scalability. To address these challenges, we propose a novel routing method using in-context vectors to represent model capabilities. The method proceeds in two stages. First, queries are embedded and projected into vectors, with a projector and LLM-based router trained to reconstruct the original queries, aligning vector representations with the router's semantic space. Second, each candidate model is profiled on a query set, and the router learns -- based on in-context vectors of query and model performance -- to predict whether each model can correctly answer new queries. Extensive experiments demonstrate that our method achieves state-of-the-art routing performance in both in-distribution and out-of-distribution tasks. Moreover, our method allows for seamless integration of new models without retraining the router. The code is available at https://github.com/lalalamdbf/ICL-Router.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11125v1">Utilizing LLMs for Industrial Process Automation: A Case Study on Modifying RAPID Programs</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 Submitted to the International Conference on Software Engineering (ICSE) track Software Engineering in Practice (SEIP) 2026
    </div>
    <details class="paper-abstract">
      How to best use Large Language Models (LLMs) for software engineering is covered in many publications in recent years. However, most of this work focuses on widely-used general purpose programming languages. The utility of LLMs for software within the industrial process automation domain, with highly-specialized languages that are typically only used in proprietary contexts, is still underexplored. Within this paper, we study enterprises can achieve on their own without investing large amounts of effort into the training of models specific to the domain-specific languages that are used. We show that few-shot prompting approaches are sufficient to solve simple problems in a language that is otherwise not well-supported by an LLM and that is possible on-premise, thereby ensuring the protection of sensitive company data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11106v1">AccKV: Towards Efficient Audio-Video LLMs Inference via Adaptive-Focusing and Cross-Calibration KV Cache Optimization</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      Recent advancements in Audio-Video Large Language Models (AV-LLMs) have enhanced their capabilities in tasks like audio-visual question answering and multimodal dialog systems. Video and audio introduce an extended temporal dimension, resulting in a larger key-value (KV) cache compared to static image embedding. A naive optimization strategy is to selectively focus on and retain KV caches of audio or video based on task. However, in the experiment, we observed that the attention of AV-LLMs to various modalities in the high layers is not strictly dependent on the task. In higher layers, the attention of AV-LLMs shifts more towards the video modality. In addition, we also found that directly integrating temporal KV of audio and spatial-temporal KV of video may lead to information confusion and significant performance degradation of AV-LLMs. If audio and video are processed indiscriminately, it may also lead to excessive compression or reservation of a certain modality, thereby disrupting the alignment between modalities. To address these challenges, we propose AccKV, an Adaptive-Focusing and Cross-Calibration KV cache optimization framework designed specifically for efficient AV-LLMs inference. Our method is based on layer adaptive focusing technology, selectively focusing on key modalities according to the characteristics of different layers, and enhances the recognition of heavy hitter tokens through attention redistribution. In addition, we propose a Cross-Calibration technique that first integrates inefficient KV caches within the audio and video modalities, and then aligns low-priority modalities with high-priority modalities to selectively evict KV cache of low-priority modalities. The experimental results show that AccKV can significantly improve the computational efficiency of AV-LLMs while maintaining accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.20856v3">Strada-LLM: Graph LLM for traffic prediction</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 13 pages
    </div>
    <details class="paper-abstract">
      Traffic forecasting is pivotal for intelligent transportation systems, where accurate and interpretable predictions can significantly enhance operational efficiency and safety. A key challenge stems from the heterogeneity of traffic conditions across diverse locations, leading to highly varied traffic data distributions. Large language models (LLMs) show exceptional promise for few-shot learning in such dynamic and data-sparse scenarios. However, existing LLM-based solutions often rely on prompt-tuning, which can struggle to fully capture complex graph relationships and spatiotemporal dependencies-thereby limiting adaptability and interpretability in real-world traffic networks. We address these gaps by introducing Strada-LLM, a novel multivariate probabilistic forecasting LLM that explicitly models both temporal and spatial traffic patterns. By incorporating proximal traffic information as covariates, Strada-LLM more effectively captures local variations and outperforms prompt-based existing LLMs. To further enhance adaptability, we propose a lightweight distribution-derived strategy for domain adaptation, enabling parameter-efficient model updates when encountering new data distributions or altered network topologies-even under few-shot constraints. Empirical evaluations on spatio-temporal transportation datasets demonstrate that Strada-LLM consistently surpasses state-of-the-art LLM-driven and traditional GNN-based predictors. Specifically, it improves long-term forecasting by 17% in RMSE error and 16% more efficiency. Moreover, it maintains robust performance across different LLM backbones with minimal degradation, making it a versatile and powerful solution for real-world traffic prediction tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.05129v2">Navigating Through Paper Flood: Advancing LLM-based Paper Evaluation through Domain-Aware Retrieval and Latent Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 Accepted for publication in AAAI'26
    </div>
    <details class="paper-abstract">
      With the rapid and continuous increase in academic publications, identifying high-quality research has become an increasingly pressing challenge. While recent methods leveraging Large Language Models (LLMs) for automated paper evaluation have shown great promise, they are often constrained by outdated domain knowledge and limited reasoning capabilities. In this work, we present PaperEval, a novel LLM-based framework for automated paper evaluation that addresses these limitations through two key components: 1) a domain-aware paper retrieval module that retrieves relevant concurrent work to support contextualized assessments of novelty and contributions, and 2) a latent reasoning mechanism that enables deep understanding of complex motivations and methodologies, along with comprehensive comparison against concurrently related work, to support more accurate and reliable evaluation. To guide the reasoning process, we introduce a progressive ranking optimization strategy that encourages the LLM to iteratively refine its predictions with an emphasis on relative comparison. Experiments on two datasets demonstrate that PaperEval consistently outperforms existing methods in both academic impact and paper quality evaluation. In addition, we deploy PaperEval in a real-world paper recommendation system for filtering high-quality papers, which has gained strong engagement on social media -- amassing over 8,000 subscribers and attracting over 10,000 views for many filtered high-quality papers -- demonstrating the practical effectiveness of PaperEval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11087v1">Can LLMs Detect Their Own Hallucinations?</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can generate fluent responses, but sometimes hallucinate facts. In this paper, we investigate whether LLMs can detect their own hallucinations. We formulate hallucination detection as a classification task of a sentence. We propose a framework for estimating LLMs' capability of hallucination detection and a classification method using Chain-of-Thought (CoT) to extract knowledge from their parameters. The experimental results indicated that GPT-$3.5$ Turbo with CoT detected $58.2\%$ of its own hallucinations. We concluded that LLMs with CoT can detect hallucinations if sufficient knowledge is contained in their parameters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11019v1">PATCHEVAL: A New Benchmark for Evaluating LLMs on Patching Real-World Vulnerabilities</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      Software vulnerabilities are increasing at an alarming rate. However, manual patching is both time-consuming and resource-intensive, while existing automated vulnerability repair (AVR) techniques remain limited in effectiveness. Recent advances in large language models (LLMs) have opened a new paradigm for AVR, demonstrating remarkable progress. To examine the capability of LLMs in AVR, several vulnerability benchmarks have been proposed recently. However, they still suffer from key limitations of outdated vulnerabilities, limited language coverage, unreliable patch validation, and insufficient reproducibility. To overcome these challenges, we introduce PATCHEVAL, a multilingual benchmark for Go, JavaScript, and Python, languages for which existing benchmarks remain unexplored. PATCHEVAL curates a dataset of 1,000 vulnerabilities drawn from CVEs reported between 2015 and 2025, covering 65 distinct CWEs. A subset of 230 CVEs is further equipped with runtime sandbox environments, enabling patch verification through both security tests and functionality tests. To provide a systematic comparison of LLM-based vulnerability repair, we evaluate a series of state-of-the-art LLMs and agents, presenting an in-depth analysis that empirically yields key insights to guide future research in AVR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.18544v2">SLICE: SLO-Driven Scheduling for LLM Inference on Edge Computing Devices</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 This work has been submitted to the IEEE for possible publication
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), as the foundational architecture for next-generation interactive AI applications, not only power intelligent dialogue systems but also drive the evolution of embodied intelligence on edge devices, including humanoid robots, smart vehicles, and other scenarios. The applications running on these edge devices impose differentiated Service Level Objectives (SLO) requirements on LLM services, specifically manifested as distinct constraints on Time to First Token (TTFT) and Time Per Output Token (TPOT) as well as end-to-end latency. Notably, edge devices typically handle real-time tasks that are extremely sensitive to latency, such as machine control and navigation planning. However, existing scheduling service systems still prioritize maximizing output token throughput as the sole optimization objective, failing to adequately address the diversity of SLO requirements. This ultimately results in persistently high violation rates for end-to-end latency or TPOT related SLOs. This paper proposes SLICE, an innovative scheduling solution designed for edge computing scenarios with differentiated SLO requirements. By combining a utility-maximizing request scheduling algorithm with a dynamic iterative control mechanism for generation rates, SLICE significantly improves LLM inference service SLO attainment. Experimental results demonstrate that compared to state-of-the-art solutions Orca and FastServe, SLICE achieves up to 35x higher SLO attainment and 3.4x advantage in task completion time than the other two solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11000v1">DialogGraph-LLM: Graph-Informed LLMs for End-to-End Audio Dialogue Intent Recognition</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 8 pages, 2 figures; Series: Frontiers in Artificial Intelligence and Applications, Volume 413: ECAI 2025
    </div>
    <details class="paper-abstract">
      Recognizing speaker intent in long audio dialogues among speakers has a wide range of applications, but is a non-trivial AI task due to complex inter-dependencies in speaker utterances and scarce annotated data. To address these challenges, an end-to-end framework, namely DialogGraph-LLM, is proposed in the current work. DialogGraph-LLM combines a novel Multi-Relational Dialogue Attention Network (MR-DAN) architecture with multimodal foundation models (e.g., Qwen2.5-Omni-7B) for direct acoustic-to-intent inference. An adaptive semi-supervised learning strategy is designed using LLM with a confidence-aware pseudo-label generation mechanism based on dual-threshold filtering using both global and class confidences, and an entropy-based sample selection process that prioritizes high-information unlabeled instances. Extensive evaluations on the proprietary MarketCalls corpus and the publicly available MIntRec 2.0 benchmark demonstrate DialogGraph-LLM's superiority over strong audio and text-driven baselines. The framework demonstrates strong performance and efficiency in intent recognition in real world scenario audio dialogues, proving its practical value for audio-rich domains with limited supervision. Our code is available at https://github.com/david188888/DialogGraph-LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10979v1">PAS: A Training-Free Stabilizer for Temporal Encoding in Video LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 13 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Video LLMs suffer from temporal inconsistency: small shifts in frame timing can flip attention and suppress relevant frames. We trace this instability to the common extension of Rotary Position Embeddings to video through multimodal RoPE. The induced inverse Fourier time kernel exhibits frame-scale ripples that multiply adjacent frames by different factors, which perturbs attention that should otherwise be governed by the raw query key inner product. We present Phase Aggregated Smoothing (PAS), a simple, training-free mechanism that applies small opposed phase offsets across heads and then aggregates their outputs. PAS preserves the per-head spectrum magnitude, while the aggregation effectively smooths the temporal kernel and reduces phase sensitivity without changing the positional encoding structure. Our analysis shows that the RoPE rotated logit can be approximated as a content dot product scaled by a time kernel; smoothing this kernel yields Lipschitz stability of attention to small temporal shifts; multi phase averaging attenuates high frequency ripples while preserving per-head spectra under Nyquist-valid sampling. Experiments on multiple video understanding benchmarks under matched token budgets show consistent improvements with negligible computational overhead. PAS provides a plug and play upgrade for robust temporal encoding in Video LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07943v2">Thinker: Training LLMs in Hierarchical Thinking for Deep Search via Multi-Turn Interaction</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 Accepted to AAAI 2026. Extended version with full Appendix
    </div>
    <details class="paper-abstract">
      Efficient retrieval of external knowledge bases and web pages is crucial for enhancing the reasoning abilities of LLMs. Previous works on training LLMs to leverage external retrievers for solving complex problems have predominantly employed end-to-end reinforcement learning. However, these approaches neglect supervision over the reasoning process, making it difficult to guarantee logical coherence and rigor. To address these limitations, we propose Thinker, a hierarchical thinking model for deep search through multi-turn interaction, making the reasoning process supervisable and verifiable. It decomposes complex problems into independently solvable sub-problems, each dually represented in both natural language and an equivalent logical function to support knowledge base and web searches. Concurrently, dependencies between sub-problems are passed as parameters via these logical functions, enhancing the logical coherence of the problem-solving process. To avoid unnecessary external searches, we perform knowledge boundary determination to check if a sub-problem is within the LLM's intrinsic knowledge, allowing it to answer directly. Experimental results indicate that with as few as several hundred training samples, the performance of Thinker is competitive with established baselines. Furthermore, when scaled to the full training set, Thinker significantly outperforms these methods across various datasets and model sizes. The source code is available at https://github.com/OpenSPG/KAG-Thinker.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.15980v2">Text-to-SQL Domain Adaptation via Human-LLM Collaborative Data Annotation</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 Accepted by IUI'25 Code & Demo: https://github.com/magic-YuanTian/SQLsynth
    </div>
    <details class="paper-abstract">
      Text-to-SQL models, which parse natural language (NL) questions to executable SQL queries, are increasingly adopted in real-world applications. However, deploying such models in the real world often requires adapting them to the highly specialized database schemas used in specific applications. We find that existing text-to-SQL models experience significant performance drops when applied to new schemas, primarily due to the lack of domain-specific data for fine-tuning. This data scarcity also limits the ability to effectively evaluate model performance in new domains. Continuously obtaining high-quality text-to-SQL data for evolving schemas is prohibitively expensive in real-world scenarios. To bridge this gap, we propose SQLsynth, a human-in-the-loop text-to-SQL data annotation system. SQLsynth streamlines the creation of high-quality text-to-SQL datasets through human-LLM collaboration in a structured workflow. A within-subjects user study comparing SQLsynth with manual annotation and ChatGPT shows that SQLsynth significantly accelerates text-to-SQL data annotation, reduces cognitive load, and produces datasets that are more accurate, natural, and diverse. Our code is available at https://github.com/magic-YuanTian/SQLsynth.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10948v1">DEFT-LLM: Disentangled Expert Feature Tuning for Micro-Expression Recognition</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      Micro expression recognition (MER) is crucial for inferring genuine emotion. Applying a multimodal large language model (MLLM) to this task enables spatio-temporal analysis of facial motion and provides interpretable descriptions. However, there are still two core challenges: (1) The entanglement of static appearance and dynamic motion cues prevents the model from focusing on subtle motion; (2) Textual labels in existing MER datasets do not fully correspond to underlying facial muscle movements, creating a semantic gap between text supervision and physical motion. To address these issues, we propose DEFT-LLM, which achieves motion semantic alignment by multi-expert disentanglement. We first introduce Uni-MER, a motion-driven instruction dataset designed to align text with local facial motion. Its construction leverages dual constraints from optical flow and Action Unit (AU) labels to ensure spatio-temporal consistency and reasonable correspondence to the movements. We then design an architecture with three experts to decouple facial dynamics into independent and interpretable representations (structure, dynamic textures, and motion-semantics). By integrating the instruction-aligned knowledge from Uni-MER into DEFT-LLM, our method injects effective physical priors for micro expressions while also leveraging the cross modal reasoning ability of large language models, thus enabling precise capture of subtle emotional cues. Experiments on multiple challenging MER benchmarks demonstrate state-of-the-art performance, as well as a particular advantage in interpretable modeling of local facial motion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10067v2">Enhancing the Medical Context-Awareness Ability of LLMs via Multifaceted Self-Refinement Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 20 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown great promise in the medical domain, achieving strong performance on several benchmarks. However, they continue to underperform in real-world medical scenarios, which often demand stronger context-awareness, i.e., the ability to recognize missing or critical details (e.g., user identity, medical history, risk factors) and provide safe, helpful, and contextually appropriate responses. To address this issue, we propose Multifaceted Self-Refinement (MuSeR), a data-driven approach that enhances LLMs' context-awareness along three key facets (decision-making, communication, and safety) through self-evaluation and refinement. Specifically, we first design a attribute-conditioned query generator that simulates diverse real-world user contexts by varying attributes such as role, geographic region, intent, and degree of information ambiguity. An LLM then responds to these queries, self-evaluates its answers along three key facets, and refines its responses to better align with the requirements of each facet. Finally, the queries and refined responses are used for supervised fine-tuning to reinforce the model's context-awareness ability. Evaluation results on the latest HealthBench dataset demonstrate that our method significantly improves LLM performance across multiple aspects, with particularly notable gains in the context-awareness axis. Furthermore, by incorporating knowledge distillation with the proposed method, the performance of a smaller backbone LLM (e.g., Qwen3-32B) surpasses its teacher model, achieving a new SOTA across all open-source LLMs on HealthBench (63.8%) and its hard subset (43.1%). Code and dataset will be released at https://muser-llm.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10890v1">LLM enhanced graph inference for long-term disease progression modelling</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      Understanding the interactions between biomarkers among brain regions during neurodegenerative disease is essential for unravelling the mechanisms underlying disease progression. For example, pathophysiological models of Alzheimer's Disease (AD) typically describe how variables, such as regional levels of toxic proteins, interact spatiotemporally within a dynamical system driven by an underlying biological substrate, often based on brain connectivity. However, current methods grossly oversimplify the complex relationship between brain connectivity by assuming a single-modality brain connectome as the disease-spreading substrate. This leads to inaccurate predictions of pathology spread, especially during the long-term progression period. Meanhwile, other methods of learning such a graph in a purely data-driven way face the identifiability issue due to lack of proper constraint. We thus present a novel framework that uses Large Language Models (LLMs) as expert guides on the interaction of regional variables to enhance learning of disease progression from irregularly sampled longitudinal patient data. By leveraging LLMs' ability to synthesize multi-modal relationships and incorporate diverse disease-driving mechanisms, our method simultaneously optimizes 1) the construction of long-term disease trajectories from individual-level observations and 2) the biologically-constrained graph structure that captures interactions among brain regions with better identifiability. We demonstrate the new approach by estimating the pathology propagation using tau-PET imaging data from an Alzheimer's disease cohort. The new framework demonstrates superior prediction accuracy and interpretability compared to traditional approaches while revealing additional disease-driving factors beyond conventional connectivity measures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.15793v2">Format as a Prior: Quantifying and Analyzing Bias in LLMs for Heterogeneous Data</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 Accepted by AAAI 2026, camera ready version
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly employed in applications that require processing information from heterogeneous formats, including texts, tables, infoboxes, and knowledge graphs. However, systematic biases toward particular formats may undermine LLMs' ability to integrate heterogeneous data impartially, potentially resulting in reasoning errors and increased risks in downstream tasks. Yet it remains unclear whether such biases are systematic, which data-level factors drive them, and what internal mechanisms underlie their emergence. In this paper, we present the first comprehensive study of format bias in LLMs through a three-stage empirical analysis. The first stage explores the presence and direction of bias across a diverse range of LLMs. The second stage examines how key data-level factors influence these biases. The third stage analyzes how format bias emerges within LLMs' attention patterns and evaluates a lightweight intervention to test its effectiveness. Our results show that format bias is consistent across model families, driven by information richness, structure quality, and representation type, and is closely associated with attention imbalance within the LLMs. Based on these investigations, we identify three future research directions to reduce format bias: enhancing data pre-processing through format repair and normalization, introducing inference-time interventions such as attention re-weighting, and developing format-balanced training corpora. These directions will support the design of more robust and fair heterogeneous data processing systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10871v1">From Fact to Judgment: Investigating the Impact of Task Framing on LLM Conviction in Dialogue Systems</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 11 pages, 3 figures. Under review at IWSDS 2026
    </div>
    <details class="paper-abstract">
      LLMs are increasingly employed as judges across a variety of tasks, including those involving everyday social interactions. Yet, it remains unclear whether such LLM-judges can reliably assess tasks that require social or conversational judgment. We investigate how an LLM's conviction is changed when a task is reframed from a direct factual query to a Conversational Judgment Task. Our evaluation framework contrasts the model's performance on direct factual queries with its assessment of a speaker's correctness when the same information is presented within a minimal dialogue, effectively shifting the query from "Is this statement correct?" to "Is this speaker correct?". Furthermore, we apply pressure in the form of a simple rebuttal ("The previous answer is incorrect.") to both conditions. This perturbation allows us to measure how firmly the model maintains its position under conversational pressure. Our findings show that while some models like GPT-4o-mini reveal sycophantic tendencies under social framing tasks, others like Llama-8B-Instruct become overly-critical. We observe an average performance change of 9.24% across all models, demonstrating that even minimal dialogue context can significantly alter model judgment, underscoring conversational framing as a key factor in LLM-based evaluation. The proposed framework offers a reproducible methodology for diagnosing model conviction and contributes to the development of more trustworthy dialogue systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10868v1">Go-UT-Bench: A Fine-Tuning Dataset for LLM-Based Unit Test Generation in Go</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 9 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Training data imbalance poses a major challenge for code LLMs. Most available data heavily over represents raw opensource code while underrepresenting broader software engineering tasks, especially in low resource languages like Golang. As a result, models excel at code autocompletion but struggle with real world developer workflows such as unit test generation. To address this gap, we introduce GO UT Bench, a benchmark dataset of 5264 pairs of code and unit tests, drawn from 10 permissively licensed Golang repositories spanning diverse domain. We evaluate its effectiveness as a fine tuning dataset across two LLM families i.e. mixture of experts and dense decoders. Our results show that finetuned models outperform their base counterparts on more than 75% of benchmark tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10866v1">Short-Window Sliding Learning for Real-Time Violence Detection via LLM-based Auto-Labeling</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 5 pages, 2 figures. Accepted paper for the IEIE (Institute of Electronics and Information Engineers) Fall Conference 2025. Presentation on Nov 27, 2025
    </div>
    <details class="paper-abstract">
      This paper proposes a Short-Window Sliding Learning framework for real-time violence detection in CCTV footages. Unlike conventional long-video training approaches, the proposed method divides videos into 1-2 second clips and applies Large Language Model (LLM)-based auto-caption labeling to construct fine-grained datasets. Each short clip fully utilizes all frames to preserve temporal continuity, enabling precise recognition of rapid violent events. Experiments demonstrate that the proposed method achieves 95.25\% accuracy on RWF-2000 and significantly improves performance on long videos (UCF-Crime: 83.25\%), confirming its strong generalization and real-time applicability in intelligent surveillance systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10865v1">Towards a Human-in-the-Loop Framework for Reliable Patch Evaluation Using an LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      Reliable evaluation is crucial for advancing Automated Program Repair (APR), but prevailing benchmarks rely on execution-based evaluation methods (unit test pass@k), which fail to capture true patch validity. Determining validity can require costly manual annotation. To reduce this cost, we introduce a human-in-the-loop approach to LLM-based patch validity judgment. Inspired by the observation that human judgment is better aligned when using a shared rubric, we first employ an LLM to generate a per-bug rubric, followed by a one-time human review and optional refinement to this rubric, and then employ an LLM to judge patches using the refined rubric. We apply this approach to assign binary validity labels to patches for issues found by Google sanitizer tools. Our results show that this approach yields substantial agreement with human consensus (Cohen's kappa 0.75), high recall (0.94) and high precision (0.80), when considering patches that have unanimous agreement from 3 human raters on the validity labels. On the full dataset including patches where human raters disagree, we find this approach can still be further improved (Cohen's kappa 0.57, recall 0.93, precision 0.65) and identify possible future directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11946v1">Improving LLM's Attachment to External Knowledge In Dialogue Generation Tasks Through Entity Anonymization</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      Knowledge graph-based dialogue generation (KG-DG) is a challenging task requiring models to effectively incorporate external knowledge into conversational responses. While large language models (LLMs) have achieved impressive results across various NLP tasks, their ability to utilize external knowledge in KG-DG remains under-explored. We observe that LLMs often rely on internal knowledge, leading to detachment from provided knowledge graphs, even when they are given a flawlessly retrieved knowledge graph. First, we introduce LLM-KAT, an evaluation procedure for measuring knowledge attachment in generated responses. Second, we propose a simple yet effective entity anonymization technique to encourage LLMs to better leverage external knowledge. Experiments on the OpenDialKG dataset demonstrate that our approach improves LLMs' attachment on external knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11916v1">An Analysis of Architectural Impact on LLM-based Abstract Visual Reasoning: A Systematic Benchmark on RAVEN-FAIR</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 23 pages, 9 figures
    </div>
    <details class="paper-abstract">
      This study aims to systematically evaluate the performance of large language models (LLMs) in abstract visual reasoning problems. We examined four LLM models (GPT-4.1-Mini, Claude-3.5-Haiku, Gemini-1.5-Flash, Llama-3.3-70b) utilizing four different reasoning architectures (single-shot, embedding-controlled repetition, self-reflection, and multi-agent) on the RAVEN-FAIR dataset. Visual responses generated through a three-stage process (JSON extraction, LLM reasoning, and Tool Function) were evaluated using SSIM and LPIPS metrics; Chain-of-Thought scores and error types (semantic hallucination, numeric misperception) were analyzed. Results demonstrate that GPT-4.1-Mini consistently achieved the highest overall accuracy across all architectures, indicating a strong reasoning capability. While the multi-agent architecture occasionally altered semantic and numeric balance across models, these effects were not uniformly beneficial. Instead, each model exhibited distinct sensitivity patterns to architectural design, underscoring that reasoning effectiveness remains model-specific. Variations in response coverage further emerged as a confounding factor that complicates direct cross-architecture comparison. To estimate the upper-bound performance of each configuration, we report the best of five independent runs, representing a best-case scenario rather than an averaged outcome. This multi-run strategy aligns with recent recommendations, which emphasize that single-run evaluations are fragile and may lead to unreliable conclusions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11914v1">Forgetting-MarI: LLM Unlearning via Marginal Information Regularization</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      As AI models are trained on ever-expanding datasets, the ability to remove the influence of specific data from trained models has become essential for privacy protection and regulatory compliance. Unlearning addresses this challenge by selectively removing parametric knowledge from the trained models without retraining from scratch, which is critical for resource-intensive models such as Large Language Models (LLMs). Existing unlearning methods often degrade model performance by removing more information than necessary when attempting to ''forget'' specific data. We introduce Forgetting-MarI, an LLM unlearning framework that provably removes only the additional (marginal) information contributed by the data to be unlearned, while preserving the information supported by the data to be retained. By penalizing marginal information, our method yields an explicit upper bound on the unlearn dataset's residual influence in the trained models, providing provable undetectability. Extensive experiments confirm that our approach outperforms current state-of-the-art unlearning methods, delivering reliable forgetting and better preserved general model performance across diverse benchmarks. This advancement represents an important step toward making AI systems more controllable and compliant with privacy and copyright regulations without compromising their effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09837v1">MoFa: A Unified Performance Modeling Framework for LLM Pretraining</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      The exponential growth in LLM scales, with parameters soaring from billions to trillions, has necessitated distributed pretraining across large clusters comprising thousands to tens of thousands of devices. While hybrid parallelization strategies enable such pretraining, the vast combinatorial strategy space introduces significant optimization challenges. Traditional manual tuning methods incur prohibitive trial-and-error costs, and existing performance modeling approaches exhibit critical limitations: they fail to comprehensively account for prevalent optimization features and ignore the substantial overhead imposed by essential fault tolerance mechanisms like checkpoint recovery in long-duration pretraining. To address these gaps, we propose MoFa, a novel pretraining performance modeling framework that unifies multi-dimensional optimization features and fault tolerance. MoFa incorporates an enhanced cost model to accurately capture the effects of key optimizations and integrates a fault tolerance model based on historical cluster reliability data. Besides, a MoFa-based tuning system is developed to explore optimal pretraining performance and potential bottlenecks in various scenarios. Extensive modeling evaluations demonstrate that MoFa can achieve high prediction accuracy across various scenarios. In addition, through comprehensive tuning experiments, our framework systematically reveals the key factors influencing pretraining performance under different configurations, which provides solid a priori guidance for LLM pretraining system design and deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09831v1">Answering Students' Questions on Course Forums Using Multiple Chain-of-Thought Reasoning and Finetuning RAG-Enabled LLM</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      The course forums are increasingly significant and play vital role in facilitating student discussions and answering their questions related to the course. It provides a platform for students to post their questions related to the content and admin issues related to the course. However, there are several challenges due to the increase in the number of students enrolled in the course. The primary challenge is that students' queries cannot be responded immediately and the instructors have to face lots of repetitive questions. To mitigate these issues, we propose a question answering system based on large language model with retrieval augmented generation (RAG) method. This work focuses on designing a question answering system with open source Large Language Model (LLM) and fine-tuning it on the relevant course dataset. To further improve the performance, we use a local knowledge base and applied RAG method to retrieve relevant documents relevant to students' queries, where the local knowledge base contains all the course content. To mitigate the hallucination of LLMs, We also integrate it with multi chain-of-thought reasoning to overcome the challenge of hallucination in LLMs. In this work, we experiment fine-tuned LLM with RAG method on the HotpotQA dataset. The experimental results demonstrate that the fine-tuned LLM with RAG method has a strong performance on question answering task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.14289v3">From Capabilities to Performance: Evaluating Key Functional Properties of LLM Architectures in Penetration Testing</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to automate or augment penetration testing, but their effectiveness and reliability across attack phases remain unclear. We present a comprehensive evaluation of multiple LLM-based agents, from single-agent to modular designs, across realistic penetration testing scenarios, measuring empirical performance and recurring failure patterns. We also isolate the impact of five core functional capabilities via targeted augmentations: Global Context Memory (GCM), Inter-Agent Messaging (IAM), Context-Conditioned Invocation (CCI), Adaptive Planning (AP), and Real-Time Monitoring (RTM). These interventions support, respectively: (i) context coherence and retention, (ii) inter-component coordination and state management, (iii) tool use accuracy and selective execution, (iv) multi-step strategic planning, error detection, and recovery, and (v) real-time dynamic responsiveness. Our results show that while some architectures natively exhibit subsets of these properties, targeted augmentations substantially improve modular agent performance, especially in complex, multi-step, and real-time penetration testing tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10860v1">HPCAgentTester: A Multi-Agent LLM Approach for Enhanced HPC Unit Test Generation</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 Accepted in AIWare 2025
    </div>
    <details class="paper-abstract">
      Unit testing in High-Performance Computing (HPC) is critical but challenged by parallelism, complex algorithms, and diverse hardware. Traditional methods often fail to address non-deterministic behavior and synchronization issues in HPC applications. This paper introduces HPCAgentTester, a novel multi-agent Large Language Model (LLM) framework designed to automate and enhance unit test generation for HPC software utilizing OpenMP and MPI. HPCAgentTester employs a unique collaborative workflow where specialized LLM agents (Recipe Agent and Test Agent) iteratively generate and refine test cases through a critique loop. This architecture enables the generation of context-aware unit tests that specifically target parallel execution constructs, complex communication patterns, and hierarchical parallelism. We demonstrate HPCAgentTester's ability to produce compilable and functionally correct tests for OpenMP and MPI primitives, effectively identifying subtle bugs that are often missed by conventional techniques. Our evaluation shows that HPCAgentTester significantly improves test compilation rates and correctness compared to standalone LLMs, offering a more robust and scalable solution for ensuring the reliability of parallel software systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10855v1">ExPairT-LLM: Exact Learning for LLM Code Selection by Pairwise Queries</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      Despite recent advances in LLMs, the task of code generation is still challenging. To cope, code selection algorithms select the best program from multiple programs generated by an LLM. However, existing algorithms can fail to identify the correct program, either because they can misidentify nonequivalent programs or because they rely on an LLM and assume it always correctly determines the output for every input. We present ExPairT-LLM, an exact learning algorithm for code selection that selects a program by posing to an LLM oracle two new types of queries: pairwise membership and pairwise equivalence. These queries are simpler for LLMs and enable ExPairT-LLM to identify the correct program through a tournament, which is robust to some LLM mistakes. We evaluate ExPairT-LLM on four popular code datasets. Its pass@1 (success rate) outperforms the state-of-the-art code selection algorithm on average by +13.0% and up to +27.1%. It also improves the pass@1 of LLMs performing complex reasoning by +24.0%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10850v1">Leveraging Parameter Space Symmetries for Reasoning Skill Transfer in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      Task arithmetic is a powerful technique for transferring skills between Large Language Models (LLMs), but it often suffers from negative interference when models have diverged during training. We address this limitation by first aligning the models' parameter spaces, leveraging the inherent permutation, rotation, and scaling symmetries of Transformer architectures. We adapt parameter space alignment for modern Grouped-Query Attention (GQA) and SwiGLU layers, exploring both weight-based and activation-based approaches. Using this alignment-first strategy, we successfully transfer advanced reasoning skills to a non-reasoning model. Experiments on challenging reasoning benchmarks show that our method consistently outperforms standard task arithmetic. This work provides an effective approach for merging and transferring specialized skills across evolving LLM families, reducing redundant fine-tuning and enhancing model adaptability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.26546v2">Towards Verified Code Reasoning by LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 43 pages
    </div>
    <details class="paper-abstract">
      While LLM-based agents are able to tackle a wide variety of code reasoning questions, the answers are not always correct. This prevents the agent from being useful in situations where high precision is desired: (1) helping a software engineer understand a new code base, (2) helping a software engineer during code review sessions, and (3) ensuring that the code generated by an automated code generation system meets certain requirements (e.g. fixes a bug, improves readability, implements a feature). As a result of this lack of trustworthiness, the agent's answers need to be manually verified before they can be trusted. Manually confirming responses from a code reasoning agent requires human effort and can result in slower developer productivity, which weakens the assistance benefits of the agent. In this paper, we describe a method to automatically validate the answers provided by a code reasoning agent by verifying its reasoning steps. At a very high level, the method consists of extracting a formal representation of the agent's response and, subsequently, using formal verification and program analysis tools to verify the agent's reasoning steps. We applied this approach to a benchmark set of 20 uninitialized variable errors detected by sanitizers and 20 program equivalence queries. For the uninitialized variable errors, the formal verification step was able to validate the agent's reasoning on 13/20 examples, and for the program equivalence queries, the formal verification step successfully caught 6/8 incorrect judgments made by the agent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10840v1">Tracing Multilingual Representations in LLMs with Cross-Layer Transcoders</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 28 pages, 35 figures, under review. Extensive supplementary materials. Code and models available at https://huggingface.co/collections/CausalNLP/multilingual-tinystories-6862b6562414eb84d183f82a and https://huggingface.co/flodraye
    </div>
    <details class="paper-abstract">
      Multilingual Large Language Models (LLMs) can process many languages, yet how they internally represent this diversity remains unclear. Do they form shared multilingual representations with language-specific decoding, and if so, why does performance still favor the dominant training language? To address this, we train a series of LLMs on different mixtures of multilingual data and analyze their internal mechanisms using cross-layer transcoders (CLT) and attribution graphs. Our results provide strong evidence for pivot language representations: the model employs nearly identical representations across languages, while language-specific decoding emerges in later layers. Attribution analyses reveal that decoding relies in part on a small set of high-frequency language features in the final layers, which linearly read out language identity from the first layers in the model. By intervening on these features, we can suppress one language and substitute another in the model's outputs. Finally, we study how the dominant training language influences these mechanisms across attribution graphs and decoding pathways. We argue that understanding this pivot-language mechanism is crucial for improving multilingual alignment in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10833v1">SURFACEBENCH: Can Self-Evolving LLMs Find the Equations of 3D Scientific Surfaces?</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      Equation discovery from data is a core challenge in machine learning for science, requiring the recovery of concise symbolic expressions that govern complex physical and geometric phenomena. Recent approaches with large language models (LLMs) show promise in symbolic regression, but their success often hinges on memorized formulas or overly simplified functional forms. Existing benchmarks exacerbate this limitation: they focus on scalar functions, ignore domain grounding, and rely on brittle string-matching based metrics that fail to capture scientific equivalence. We introduce SurfaceBench, first comprehensive benchmark for symbolic surface discovery. SurfaceBench comprises 183 tasks across 15 categories of symbolic complexity, spanning explicit, implicit, and parametric equation representation forms. Each task includes ground-truth equations, variable semantics, and synthetically sampled three dimensional data. Unlike prior SR datasets, our tasks reflect surface-level structure, resist LLM memorization through novel symbolic compositions, and are grounded in scientific domains such as fluid dynamics, robotics, electromagnetics, and geometry. To evaluate equation discovery quality, we pair symbolic checks with geometry-aware metrics such as Chamfer and Hausdorff distances, capturing both algebraic fidelity and spatial reconstruction accuracy. Our experiments reveal that state-of-the-art frameworks, while occasionally successful on specific families, struggle to generalize across representation types and surface complexities. SurfaceBench thus establishes a challenging and diagnostic testbed that bridges symbolic reasoning with geometric reconstruction, enabling principled benchmarking of progress in compositional generalization, data-driven scientific induction, and geometry-aware reasoning with LLMs. We release the code here: https://github.com/Sanchit-404/surfacebench
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10819v1">LLM-as-a-Grader: Practical Insights from Large Language Model for Short-Answer and Report Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly explored for educational tasks such as grading, yet their alignment with human evaluation in real classrooms remains underexamined. In this study, we investigate the feasibility of using an LLM (GPT-4o) to evaluate short-answer quizzes and project reports in an undergraduate Computational Linguistics course. We collect responses from approximately 50 students across five quizzes and receive project reports from 14 teams. LLM-generated scores are compared against human evaluations conducted independently by the course teaching assistants (TAs). Our results show that GPT-4o achieves strong correlation with human graders (up to 0.98) and exact score agreement in 55\% of quiz cases. For project reports, it also shows strong overall alignment with human grading, while exhibiting some variability in scoring technical, open-ended responses. We release all code and sample data to support further research on LLMs in educational assessment. This work highlights both the potential and limitations of LLM-based grading systems and contributes to advancing automated grading in real-world academic settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.17506v3">RAG-Enhanced Collaborative LLM Agents for Drug Discovery</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 Machine Learning, Drug Discovery
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have shown great potential to accelerate drug discovery. However, the specialized nature of biochemical data often necessitates costly domain-specific fine-tuning, posing major challenges. First, it hinders the application of more flexible general-purpose LLMs for cutting-edge drug discovery tasks. More importantly, it limits the rapid integration of the vast amounts of scientific data continuously generated through experiments and research. Compounding these challenges is the fact that real-world scientific questions are typically complex and open-ended, requiring reasoning beyond pattern matching or static knowledge retrieval.To address these challenges, we propose CLADD, a retrieval-augmented generation (RAG)-empowered agentic system tailored to drug discovery tasks. Through the collaboration of multiple LLM agents, CLADD dynamically retrieves information from biomedical knowledge bases, contextualizes query molecules, and integrates relevant evidence to generate responses - all without the need for domain-specific fine-tuning. Crucially, we tackle key obstacles in applying RAG workflows to biochemical data, including data heterogeneity, ambiguity, and multi-source integration. We demonstrate the flexibility and effectiveness of this framework across a variety of drug discovery tasks, showing that it outperforms general-purpose and domain-specific LLMs as well as traditional deep learning approaches. Our code is publicly available at https://github.com/Genentech/CLADD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.14397v2">LIMINAL: Exploring The Frontiers of LLM Decode Performance</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) necessitates a deep understanding of their fundamental performance limits. This paper investigates the limits of LLM inference, focusing on hardware-imposed bottlenecks in auto-regressive decoding. We develop LIMINAL, an analytical performance model that abstracts application requirements and hardware capabilities to systematically explore performance and efficiency across a wide range of current, near-future, and hypothetical hardware. We find LIMINAL is accurate when comparing to LLMs executing on existing hardware, achieving a mean absolute error of $7.6\%$. Our analysis spans from current HBM3 memory technology used in AI accelerators like GPUs and TPUs to systems based on advanced HBM4 and advanced 3D-stacked DRAM technology. We identify five non-negotiable challenges for LLM inference hardware, establishing compute, memory capacity, bandwidth and collective communication as primary barriers to performance. These findings suggest that achieving significant performance gains beyond 10,000 tokens-per-second will require not just hardware evolution but also fundamental algorithmic advances.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.08954v4">Should you use LLMs to simulate opinions? Quality checks for early-stage deliberation</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 Accepted to AAAI AI for Social Impact (AISI), 2026. This version includes Appendices
    </div>
    <details class="paper-abstract">
      The emergent capabilities of large language models (LLMs) have prompted interest in using them as surrogates for human subjects in opinion surveys. However, prior evaluations of LLM-based opinion simulation have relied heavily on costly, domain-specific survey data, and mixed empirical results leave their reliability in question. To enable cost-effective, early-stage evaluation, we introduce a quality control assessment designed to test the viability of LLM-simulated opinions on Likert-scale tasks without requiring large-scale human data for validation. This assessment comprises two key tests: \emph{logical consistency} and \emph{alignment with stakeholder expectations}, offering a low-cost, domain-adaptable validation tool. We apply our quality control assessment to an opinion simulation task relevant to AI-assisted content moderation and fact-checking workflows -- a socially impactful use case -- and evaluate seven LLMs using a baseline prompt engineering method (backstory prompting), as well as fine-tuning and in-context learning variants. None of the models or methods pass the full assessment, revealing several failure modes. We conclude with a discussion of the risk management implications and release \texttt{TopicMisinfo}, a benchmark dataset with paired human and LLM annotations simulated by various models and approaches, to support future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10768v1">Faithful Summarization of Consumer Health Queries: A Cross-Lingual Framework with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 Accepted at the 5th Muslims in Machine Learning (MusIML) Workshop, co-located with NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Summarizing consumer health questions (CHQs) can ease communication in healthcare, but unfaithful summaries that misrepresent medical details pose serious risks. We propose a framework that combines TextRank-based sentence extraction and medical named entity recognition with large language models (LLMs) to enhance faithfulness in medical text summarization. In our experiments, we fine-tuned the LLaMA-2-7B model on the MeQSum (English) and BanglaCHQ-Summ (Bangla) datasets, achieving consistent improvements across quality (ROUGE, BERTScore, readability) and faithfulness (SummaC, AlignScore) metrics, and outperforming zero-shot baselines and prior systems. Human evaluation further shows that over 80\% of generated summaries preserve critical medical information. These results highlight faithfulness as an essential dimension for reliable medical summarization and demonstrate the potential of our approach for safer deployment of LLMs in healthcare contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10720v1">PISanitizer: Preventing Prompt Injection to Long-Context LLMs via Prompt Sanitization</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 The code is available at https://github.com/sleeepeer/PISanitizer
    </div>
    <details class="paper-abstract">
      Long context LLMs are vulnerable to prompt injection, where an attacker can inject an instruction in a long context to induce an LLM to generate an attacker-desired output. Existing prompt injection defenses are designed for short contexts. When extended to long-context scenarios, they have limited effectiveness. The reason is that an injected instruction constitutes only a very small portion of a long context, making the defense very challenging. In this work, we propose PISanitizer, which first pinpoints and sanitizes potential injected tokens (if any) in a context before letting a backend LLM generate a response, thereby eliminating the influence of the injected instruction. To sanitize injected tokens, PISanitizer builds on two observations: (1) prompt injection attacks essentially craft an instruction that compels an LLM to follow it, and (2) LLMs intrinsically leverage the attention mechanism to focus on crucial input tokens for output generation. Guided by these two observations, we first intentionally let an LLM follow arbitrary instructions in a context and then sanitize tokens receiving high attention that drive the instruction-following behavior of the LLM. By design, PISanitizer presents a dilemma for an attacker: the more effectively an injected instruction compels an LLM to follow it, the more likely it is to be sanitized by PISanitizer. Our extensive evaluation shows that PISanitizer can successfully prevent prompt injection, maintain utility, outperform existing defenses, is efficient, and is robust to optimization-based and strong adaptive attacks. The code is available at https://github.com/sleeepeer/PISanitizer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10712v1">Do Not Merge My Model! Safeguarding Open-Source LLMs Against Unauthorized Model Merging</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 Accepted by AAAI 2026 Conference
    </div>
    <details class="paper-abstract">
      Model merging has emerged as an efficient technique for expanding large language models (LLMs) by integrating specialized expert models. However, it also introduces a new threat: model merging stealing, where free-riders exploit models through unauthorized model merging. Unfortunately, existing defense mechanisms fail to provide effective protection. Specifically, we identify three critical protection properties that existing methods fail to simultaneously satisfy: (1) proactively preventing unauthorized merging; (2) ensuring compatibility with general open-source settings; (3) achieving high security with negligible performance loss. To address the above issues, we propose MergeBarrier, a plug-and-play defense that proactively prevents unauthorized merging. The core design of MergeBarrier is to disrupt the Linear Mode Connectivity (LMC) between the protected model and its homologous counterparts, thereby eliminating the low-loss path required for effective model merging. Extensive experiments show that MergeBarrier effectively prevents model merging stealing with negligible accuracy loss.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11752v1">Towards autonomous quantum physics research using LLM agents with access to intelligent tools</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 24 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Artificial intelligence (AI) is used in numerous fields of science, yet the initial research questions and targets are still almost always provided by human researchers. AI-generated creative ideas in science are rare and often vague, so that it remains a human task to execute them. Automating idea generation and implementation in one coherent system would significantly shift the role of humans in the scientific process. Here we present AI-Mandel, an LLM agent that can generate and implement ideas in quantum physics. AI-Mandel formulates ideas from the literature and uses a domain-specific AI tool to turn them into concrete experiment designs that can readily be implemented in laboratories. The generated ideas by AI-Mandel are often scientifically interesting - for two of them we have already written independent scientific follow-up papers. The ideas include new variations of quantum teleportation, primitives of quantum networks in indefinite causal orders, and new concepts of geometric phases based on closed loops of quantum information transfer. AI-Mandel is a prototypical demonstration of an AI physicist that can generate and implement concrete, actionable ideas. Building such a system is not only useful to accelerate science, but it also reveals concrete open challenges on the path to human-level artificial scientists.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11733v1">Speculative Decoding in Decentralized LLM Inference: Turning Communication Latency into Computation Throughput</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 6 pages, 2 figures, 2 tables. Uses ICML 2025 style
    </div>
    <details class="paper-abstract">
      Speculative decoding accelerates large language model (LLM) inference by using a lightweight draft model to propose tokens that are later verified by a stronger target model. While effective in centralized systems, its behavior in decentralized settings, where network latency often dominates compute, remains under-characterized. We present Decentralized Speculative Decoding (DSD), a plug-and-play framework for decentralized inference that turns communication delay into useful computation by verifying multiple candidate tokens in parallel across distributed nodes. We further introduce an adaptive speculative verification strategy that adjusts acceptance thresholds by token-level semantic importance, delivering an additional 15% to 20% end-to-end speedup without retraining. In theory, DSD reduces cross-node communication cost by approximately (N-1)t1(k-1)/k, where t1 is per-link latency and k is the average number of tokens accepted per round. In practice, DSD achieves up to 2.56x speedup on HumanEval and 2.59x on GSM8K, surpassing the Eagle3 baseline while preserving accuracy. These results show that adapting speculative decoding for decentralized execution provides a system-level optimization that converts network stalls into throughput, enabling faster distributed LLM inference with no model retraining or architectural changes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11729v1">Harli: Harvest Underutilized Resources in LLM Serving with Finetuning Tasks</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed under the Model-as-a-Service (MaaS) paradigm. To meet stringent quality-of-service (QoS) requirements, existing LLM serving systems disaggregate the prefill and decode phases of inference. However, decode instances often experience low GPU utilization due to their memory-bound nature and insufficient batching in dynamic workloads, leaving compute resources underutilized. We introduce Harli, a serving system that improves GPU utilization by co-locating parameter-efficient finetuning (PEFT) tasks with LLM decode instances. PEFT tasks are compute-bound and memory-efficient, making them ideal candidates for safe co-location. Specifically, Harli addresses key challenges--limited memory and unpredictable interference--using three components: a unified memory allocator for runtime memory reuse, a two-stage latency predictor for decode latency modeling, and a QoS-guaranteed throughput-maximizing scheduler for throughput maximization. Experimental results show that Harli improves the finetune throughput by 46.2% on average (up to 92.0%) over state-of-the-art serving systems, while maintaining strict QoS guarantees for inference decode.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.20749v4">Matryoshka Pilot: Learning to Drive Black-Box LLMs with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Despite the impressive generative abilities of black-box large language models (LLMs), their inherent opacity hinders further advancements in capabilities such as reasoning, planning, and personalization. Existing works aim to enhance LLM capabilities via domain-specific adaptation, which require additional training on accessible model parameters, an infeasible option for black-box LLMs. To address this challenge, we introduce Matryoshka Pilot (M-Pilot), a lightweight white-box LLM controller that guides a large-scale black-box LLM generator by decomposing complex tasks into a series of intermediate outputs. Specifically, we consider the black-box LLM as an environment, with M-Pilot serving as a policy to provide intermediate guidance through prompts for driving the black-box LLM. M-Pilot is trained to pivot the outputs of the black-box LLM aligning with preferences during iterative interaction, which enables controllable multi-turn generation and self-improvement in optimizing intermediate guidance. Empirical evaluations on diverse tasks demonstrate that our method effectively enhances the capabilities of black-box LLMs in complex, long-horizon tasks. Our code is publicly available at: https://github.com/lichangh20/Matryoshka.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07127v3">REACT-LLM: A Benchmark for Evaluating LLM Integration with Causal Features in Clinical Prognostic Tasks</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) and causal learning each hold strong potential for clinical decision making (CDM). However, their synergy remains poorly understood, largely due to the lack of systematic benchmarks evaluating their integration in clinical risk prediction. In real-world healthcare, identifying features with causal influence on outcomes is crucial for actionable and trustworthy predictions. While recent work highlights LLMs' emerging causal reasoning abilities, there lacks comprehensive benchmarks to assess their causal learning and performance informed by causal features in clinical risk prediction. To address this, we introduce REACT-LLM, a benchmark designed to evaluate whether combining LLMs with causal features can enhance clinical prognostic performance and potentially outperform traditional machine learning (ML) methods. Unlike existing LLM-clinical benchmarks that often focus on a limited set of outcomes, REACT-LLM evaluates 7 clinical outcomes across 2 real-world datasets, comparing 15 prominent LLMs, 6 traditional ML models, and 3 causal discovery (CD) algorithms. Our findings indicate that while LLMs perform reasonably in clinical prognostics, they have not yet outperformed traditional ML models. Integrating causal features derived from CD algorithms into LLMs offers limited performance gains, primarily due to the strict assumptions of many CD methods, which are often violated in complex clinical data. While the direct integration yields limited improvement, our benchmark reveals a more promising synergy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09557v2">LLM Inference Beyond a Single Node: From Bottlenecks to Mitigations with Fast All-Reduce Communication</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 12 Figures
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) continue to grow in size, distributed inference has become increasingly important. Model-parallel strategies must now efficiently scale not only across multiple GPUs but also across multiple nodes. In this work, we present a detailed performance study of multi-node distributed inference using LLMs on GPU-based supercomputers. We conduct experiments with several state-of-the-art inference engines alongside YALIS, a research-oriented prototype engine designed for controlled experimentation. We analyze the strong-scaling behavior of different model-parallel schemes and identify key bottlenecks. Since all-reduce operations are a common performance bottleneck, we develop NVRAR, a hierarchical all-reduce algorithm based on recursive doubling with NVSHMEM. NVRAR achieves up to 1.9x-3.6x lower latency than NCCL for message sizes between 128 KB and 2 MB on HPE Slingshot and InfiniBand interconnects. Integrated into YALIS, NVRAR achieves up to a 1.72x reduction in end-to-end batch latency for the Llama 3.1 405B model in multi-node decode-heavy workloads using tensor parallelism.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10645v1">ParoQuant: Pairwise Rotation Quantization for Efficient Reasoning LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      Weight-only post-training quantization (PTQ) compresses the weights of Large Language Models (LLMs) into low-precision representations to reduce memory footprint and accelerate inference. However, the presence of outliers in weights and activations often leads to large quantization errors and severe accuracy degradation, especially in recent reasoning LLMs where errors accumulate across long chains of thought. Existing PTQ methods either fail to sufficiently suppress outliers or introduce significant overhead during inference. In this paper, we propose Pairwise Rotation Quantization (ParoQuant), a weight-only PTQ method that combines hardware-efficient and optimizable independent Givens rotations with channel-wise scaling to even out the magnitude across channels and narrow the dynamic range within each quantization group. We further co-design the inference kernel to fully exploit GPU parallelism and keep the rotations and scaling lightweight at runtime. ParoQuant achieves an average 2.4% accuracy improvement over AWQ on reasoning tasks with less than 10% overhead. This paves the way for more efficient and accurate deployment of reasoning LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.00730v3">Teaching LLMs to See and Guide: Context-Aware Real-Time Assistance in Augmented Reality</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 This work is intended for submission to the IEEE Transactions on Systems, Man, and Cybernetics: Systems for possible publication
    </div>
    <details class="paper-abstract">
      The growing adoption of augmented and virtual reality (AR and VR) technologies in industrial training and on-the-job assistance has created new opportunities for intelligent, context-aware support systems. As workers perform complex tasks guided by AR and VR, these devices capture rich streams of multimodal data, including gaze, hand actions, and task progression, that can reveal user intent and task state in real time. Leveraging this information effectively remains a major challenge. In this work, we present a context-aware large language model (LLM) assistant that integrates diverse data modalities, such as hand actions, task steps, and dialogue history, into a unified framework for real-time question answering. To systematically study how context influences performance, we introduce an incremental prompting framework, where each model version receives progressively richer contextual inputs. Using the HoloAssist dataset, which records AR-guided task executions, we evaluate how each modality contributes to the assistant's effectiveness. Our experiments show that incorporating multimodal context significantly improves the accuracy and relevance of responses. These findings highlight the potential of LLM-driven multimodal integration to enable adaptive, intuitive assistance for AR and VR-based industrial training and assistance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10507v1">Rubric-Based Benchmarking and Reinforcement Learning for Advancing LLM Instruction Following</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      Recent progress in large language models (LLMs) has led to impressive performance on a range of tasks, yet advanced instruction following (IF)-especially for complex, multi-turn, and system-prompted instructions-remains a significant challenge. Rigorous evaluation and effective training for such capabilities are hindered by the lack of high-quality, human-annotated benchmarks and reliable, interpretable reward signals. In this work, we introduce AdvancedIF (we will release this benchmark soon), a comprehensive benchmark featuring over 1,600 prompts and expert-curated rubrics that assess LLMs ability to follow complex, multi-turn, and system-level instructions. We further propose RIFL (Rubric-based Instruction-Following Learning), a novel post-training pipeline that leverages rubric generation, a finetuned rubric verifier, and reward shaping to enable effective reinforcement learning for instruction following. Extensive experiments demonstrate that RIFL substantially improves the instruction-following abilities of LLMs, achieving a 6.7% absolute gain on AdvancedIF and strong results on public benchmarks. Our ablation studies confirm the effectiveness of each component in RIFL. This work establishes rubrics as a powerful tool for both training and evaluating advanced IF in LLMs, paving the way for more capable and reliable AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.20067v2">PITA: Preference-Guided Inference-Time Alignment for LLM Post-Training</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      Inference-time alignment enables large language models (LLMs) to generate outputs aligned with end-user preferences without further training. Recent post-training methods achieve this by using small guidance models to modify token generation during inference. These methods typically optimize a reward function KL-regularized by the original LLM taken as the reference policy. A critical limitation, however, is their dependence on a pre-trained reward model, which requires fitting to human preference feedback--a potentially unstable process. In contrast, we introduce PITA, a novel framework that integrates preference feedback directly into the LLM's token generation, eliminating the need for a reward model. PITA learns a small preference-based guidance policy to modify token probabilities at inference time without LLM fine-tuning, reducing computational cost and bypassing the pre-trained reward model dependency. The problem is framed as identifying an underlying preference distribution, solved through stochastic search and iterative refinement of the preference-based guidance model. We evaluate PITA across diverse tasks, including mathematical reasoning and sentiment classification, demonstrating its effectiveness in aligning LLM outputs with user preferences.
    </details>
</div>
