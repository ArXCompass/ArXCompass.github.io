# llm - 2025_11

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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.21505v2">How does Alignment Enhance LLMs' Multilingual Capabilities? A Language Neurons Perspective</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 AAAI 2026 (Oral)
    </div>
    <details class="paper-abstract">
      Multilingual Alignment is an effective and representative paradigm to enhance LLMs' multilingual capabilities, which transfers the capabilities from the high-resource languages to the low-resource languages. Meanwhile, some research on language-specific neurons provides a new perspective to analyze and understand LLMs' mechanisms. However, we find that there are many neurons that are shared by multiple but not all languages and cannot be correctly classified. In this work, we propose a ternary classification methodology that categorizes neurons into three types, including language-specific neurons, language-related neurons, and general neurons. And we propose a corresponding identification algorithm to distinguish these different types of neurons. Furthermore, based on the distributional characteristics of different types of neurons, we divide the LLMs' internal process for multilingual inference into four parts: (1) multilingual understanding, (2) shared semantic space reasoning, (3) multilingual output space transformation, and (4) vocabulary space outputting. Additionally, we systematically analyze the models before and after alignment with a focus on different types of neurons. We also analyze the phenomenon of ''Spontaneous Multilingual Alignment''. Overall, our work conducts a comprehensive investigation based on different types of neurons, providing empirical results and valuable insights to better understand multilingual alignment and multilingual capabilities of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.06725v2">WorldLLM: Improving LLMs' world modeling using curiosity-driven theory-making</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) possess general world knowledge but often struggle to generate precise predictions in structured, domain-specific contexts such as simulations. These limitations arise from their inability to ground their broad, unstructured understanding in specific environments. To address this, we present WorldLLM, a framework that enhances LLM-based world modeling by combining Bayesian inference and autonomous active exploration with reinforcement learning. WorldLLM leverages the in-context learning abilities of LLMs to guide an LLM-based world model's predictions using natural language hypotheses given in its prompt. These hypotheses are iteratively refined through a Bayesian inference framework that leverages a second LLM as the proposal distribution given collected evidence. This evidence is collected using a curiosity-driven reinforcement learning policy that explores the environment to find transitions with a low log-likelihood under our LLM-based predictive model using the current hypotheses. By alternating between refining hypotheses and collecting new evidence, our framework autonomously drives continual improvement of the predictions. Our experiments demonstrate the effectiveness of WorldLLM in a textual game environment that requires agents to manipulate and combine objects. The framework not only enhances predictive accuracy, but also generates human-interpretable theories of environment dynamics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19218v1">Adversarial Attack-Defense Co-Evolution for LLM Safety Alignment via Tree-Group Dual-Aware Search and Optimization</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have developed rapidly in web services, delivering unprecedented capabilities while amplifying societal risks. Existing works tend to focus on either isolated jailbreak attacks or static defenses, neglecting the dynamic interplay between evolving threats and safeguards in real-world web contexts. To mitigate these challenges, we propose ACE-Safety (Adversarial Co-Evolution for LLM Safety), a novel framework that jointly optimize attack and defense models by seamlessly integrating two key innovative procedures: (1) Group-aware Strategy-guided Monte Carlo Tree Search (GS-MCTS), which efficiently explores jailbreak strategies to uncover vulnerabilities and generate diverse adversarial samples; (2) Adversarial Curriculum Tree-aware Group Policy Optimization (AC-TGPO), which jointly trains attack and defense LLMs with challenging samples via curriculum reinforcement learning, enabling robust mutual improvement. Evaluations across multiple benchmarks demonstrate that our method outperforms existing attack and defense approaches, and provides a feasible pathway for developing LLMs that can sustainably support responsible AI ecosystems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19175v1">LLM-Based Agentic Negotiation for 6G: Addressing Uncertainty Neglect and Tail-Event Risk</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 Link to open-source non-commercial code available
    </div>
    <details class="paper-abstract">
      A critical barrier to the trustworthiness of sixth-generation (6G) agentic autonomous networks is the uncertainty neglect bias; a cognitive tendency for large language model (LLM)-powered agents to make high-stakes decisions based on simple averages while ignoring the tail risk of extreme events. This paper proposes an unbiased, risk-aware framework for agentic negotiation, designed to ensure robust resource allocation in 6G network slicing. Specifically, agents leverage Digital Twins (DTs) to predict full latency distributions, which are then evaluated using a formal framework from extreme value theory, namely, Conditional Value-at-Risk (CVaR). This approach fundamentally shifts the agent's objective from reasoning over the mean to reasoning over the tail, thereby building a statistically-grounded buffer against worst-case outcomes. Furthermore, our framework ensures full uncertainty awareness by requiring agents to quantify epistemic uncertainty -- confidence in their own DTs predictions -- and propagate this meta-verification to make robust decisions, preventing them from acting on unreliable data. We validate this framework in a 6G inter-slice negotiation use-case between an eMBB and a URLLC agent. The results demonstrate the profound failure of the biased, mean-based baseline, which consistently fails its SLAs with a 25\% rate. Our unbiased, CVaR-aware agent successfully mitigates this bias, eliminating SLA violations and reducing the URLLC and eMBB p99.999 latencies by around 11\%. We show this reliability comes at the rational and quantifiable cost of slightly reduced energy savings to 17\%, exposing the false economy of the biased approach. This work provides a concrete methodology for building the trustworthy autonomous systems required for 6G.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19171v1">Can LLMs Threaten Human Survival? Benchmarking Potential Existential Threats from LLMs via Prefix Completion</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      Research on the safety evaluation of large language models (LLMs) has become extensive, driven by jailbreak studies that elicit unsafe responses. Such response involves information already available to humans, such as the answer to "how to make a bomb". When LLMs are jailbroken, the practical threat they pose to humans is negligible. However, it remains unclear whether LLMs commonly produce unpredictable outputs that could pose substantive threats to human safety. To address this gap, we study whether LLM-generated content contains potential existential threats, defined as outputs that imply or promote direct harm to human survival. We propose \textsc{ExistBench}, a benchmark designed to evaluate such risks. Each sample in \textsc{ExistBench} is derived from scenarios where humans are positioned as adversaries to AI assistants. Unlike existing evaluations, we use prefix completion to bypass model safeguards. This leads the LLMs to generate suffixes that express hostility toward humans or actions with severe threat, such as the execution of a nuclear strike. Our experiments on 10 LLMs reveal that LLM-generated content indicates existential threats. To investigate the underlying causes, we also analyze the attention logits from LLMs. To highlight real-world safety risks, we further develop a framework to assess model behavior in tool-calling. We find that LLMs actively select and invoke external tools with existential threats. Code and data are available at: https://github.com/cuiyu-ai/ExistBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.07751v3">AbstRaL: Augmenting LLMs' Reasoning by Reinforcing Abstract Thinking</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Recent studies have shown that large language models (LLMs), especially smaller ones, often lack robustness in grade school math (GSM) reasoning. In particular, they tend to experience performance drops when faced with distribution shifts, such as changes to numerical or nominal variables, or insertions of distracting clauses. A possible strategy to address this involves generating synthetic data to further "instantiate" reasoning problems on potential variations. In this work, we instead focuses on the strategy of "abstracting" reasoning problems. This not only helps counteract distribution shifts but also facilitates the connection to symbolic tools for deriving solutions. Focusing on GSM, we find that this abstraction process is better acquired through reinforcement learning (RL) than just supervised fine-tuning, which often fails to produce faithful abstractions. Our method, AbstRaL -- which promotes abstract reasoning in LLMs using RL on granular abstraction data -- significantly mitigates performance degradation on recent GSM perturbation benchmarks. Besides, improving GSM robustness via AbstRaL is shown to also implicitly benefit LLMs' capabilities on OOD mathematical and general reasoning tasks, indicating that abstract thinking broadly enables better generalizability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.16570v2">URLs Help, Topics Guide: Understanding Metadata Utility in LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 NeurIPS 2025, Camera Ready
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are commonly pretrained on vast corpora of text without utilizing contextual metadata such as source, quality, or topic, leading to a context-free learning paradigm. While recent studies suggest that adding metadata like URL information as context (i.e., auxiliary inputs not used in the loss calculation) can improve training efficiency and downstream performance, they offer limited understanding of which types of metadata are truly effective and under what conditions. In this work, we conduct a systematic evaluation and find that not all metadata types contribute equally. Only URL context speeds up training, whereas quality scores and topic/format domain information offer no clear benefit. Furthermore, the improved downstream performances of URL conditioning emerge only when longer prompts are used at inference time. In addition, we demonstrate that context-aware pretraining enables more controllable generation than context-free pretraining, in a classifier-free guidance fashion. Although topic and format metadata do not accelerate training, they are effective for steering outputs, offering human-interpretable control over generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.07078v4">Can LLM-based Financial Investing Strategies Outperform the Market in Long Run?</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 Accepted to KDD 2026, Datasets & Benchmarks Track
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently been leveraged for asset pricing tasks and stock trading applications, enabling AI agents to generate investment decisions from unstructured financial data. However, most evaluations of LLM timing-based investing strategies are conducted on narrow timeframes and limited stock universes, overstating effectiveness due to survivorship and data-snooping biases. We critically assess their generalizability and robustness by proposing FINSABER, a backtesting framework evaluating timing-based strategies across longer periods and a larger universe of symbols. Systematic backtests over two decades and 100+ symbols reveal that previously reported LLM advantages deteriorate significantly under broader cross-section and over a longer-term evaluation. Our market regime analysis further demonstrates that LLM strategies are overly conservative in bull markets, underperforming passive benchmarks, and overly aggressive in bear markets, incurring heavy losses. These findings highlight the need to develop LLM strategies that are able to prioritise trend detection and regime-aware risk controls over mere scaling of framework complexity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19132v1">LLMs-Powered Real-Time Fault Injection: An Approach Toward Intelligent Fault Test Cases Generation</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      A well-known testing method for the safety evaluation and real-time validation of automotive software systems (ASSs) is Fault Injection (FI). In accordance with the ISO 26262 standard, the faults are introduced artificially for the purpose of analyzing the safety properties and verifying the safety mechanisms during the development phase. However, the current FI method and tools have a significant limitation in that they require manual identification of FI attributes, including fault type, location and time. The more complex the system, the more expensive, time-consuming and labour-intensive the process. To address the aforementioned challenge, a novel Large Language Models (LLMs)-assisted fault test cases (TCs) generation approach for utilization during real-time FI tests is proposed in this paper. To this end, considering the representativeness and coverage criteria, the applicability of various LLMs to create fault TCs from the functional safety requirements (FSRs) has been investigated. Through the validation results of LLMs, the superiority of the proposed approach utilizing gpt-4o in comparison to other state-of-the-art models has been demonstrated. Specifically, the proposed approach exhibits high performance in terms of FSRs classification and fault TCs generation with F1-score of 88% and 97.5%, respectively. To illustrate the proposed approach, the generated fault TCs were executed in real time on a hardware-in-the-loop system, where a high-fidelity automotive system model served as a case study. This novel approach offers a means of optimizing the real-time testing process, thereby reducing costs while simultaneously enhancing the safety properties of complex safety-critical ASSs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19131v1">Eliciting Chain-of-Thought in Base LLMs via Gradient-Based Representation Optimization</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 AAAI2026
    </div>
    <details class="paper-abstract">
      Chain-of-Thought (CoT) reasoning is a critical capability for large language models (LLMs), enabling them to tackle com- plex multi-step tasks. While base LLMs, pre-trained on general text corpora, often struggle with reasoning due to a lack of specialized training, recent studies reveal their latent reason- ing potential tied to hidden states. However, existing hidden state manipulation methods, such as linear activation steering, suffer from limitations due to their rigid and unconstrained nature, often leading to distribution shifts and degraded text quality. In this work, we propose a novel approach for elic- iting CoT reasoning from base LLMs through hidden state manipulation grounded in probabilistic conditional generation. By reformulating the challenge as an optimization problem with a balanced likelihood and prior regularization framework, our method guides hidden states toward reasoning-oriented trajectories while preserving linguistic coherence. Extensive evaluations across mathematical, commonsense, and logical reasoning benchmarks demonstrate that our approach con- sistently outperforms existing steering methods, offering a theoretically principled and effective solution for enhancing reasoning capabilities in base LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19130v1">Can LLMs Recover Program Semantics? A Systematic Evaluation with Symbolic Execution</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      Obfuscation poses a persistent challenge for software engineering tasks such as program comprehension, maintenance, testing, and vulnerability detection. While compiler optimizations and third-party code often introduce transformations that obscure program intent, existing analysis tools and large language models (LLMs) struggle to recover the original semantics. In this work, we investigate whether LLMs, when fine-tuned with symbolic execution artifacts, can effectively deobfuscate programs and restore analyzability. We construct a benchmark by applying four widely studied transformations-control-flow flattening, opaque predicates, arithmetic encoding, and branch encoding-across diverse C programs from TUM Obfuscation Benchmarks, the LLVM test suite, and algorithmic repositories. We then compare three state-of-the-art LLMs under two training configurations: baseline fine-tuning on obfuscated/original code pairs, and enhanced fine-tuning with additional KLEE artifacts such as SMT constraints, path statistics, and test cases. Our evaluation examines syntactic correctness (compilation success), semantic fidelity (behavioral equivalence under symbolic execution), and code quality (readability and structure). Results show that GPT-4.1-mini achieves the strongest deobfuscation overall, and that incorporating KLEE artifacts consistently improves semantic preservation and compilation success across models. These findings highlight deobfuscation as a broader software engineering concern, demonstrating that combining LLMs with symbolic execution can strengthen automated testing, static analysis, and program comprehension in the presence of obfuscation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19123v1">Facilitating the Integration of LLMs Into Online Experiments With Simple Chat</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become increasingly prevalent, understanding human-LLM interactions is emerging as a central priority in psychological research. Online experiments offer an efficient means to study human-LLM interactions, yet integrating LLMs into established survey platforms remains technically demanding, particularly when aiming for ecologically valid, real-time conversational experiences with strong experimental control. We introduce Simple Chat, an open-source, research-focused chat interface that streamlines LLM integration for platforms such as Qualtrics, oTree, and LimeSurvey, while presenting a unified participant experience across conditions. Simple Chat connects to both commercial providers and open-weights models, supports streaming responses to preserve conversational flow, and offers an administrative interface for fine-grained control of prompts and interface features. By reducing technical barriers, standardizing interfaces, and improving participant experience, Simple Chat helps advance the study of human-LLM interaction. In this article, we outline Simple Chat's key features, provide a step-by-step tutorial, and demonstrate its utility through two illustrative case studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19122v1">Emotion-Enhanced Multi-Task Learning with LLMs for Aspect Category Sentiment Analysis</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 8 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Aspect category sentiment analysis (ACSA) has achieved remarkable progress with large language models (LLMs), yet existing approaches primarily emphasize sentiment polarity while overlooking the underlying emotional dimensions that shape sentiment expressions. This limitation hinders the model's ability to capture fine-grained affective signals toward specific aspect categories. To address this limitation, we introduce a novel emotion-enhanced multi-task ACSA framework that jointly learns sentiment polarity and category-specific emotions grounded in Ekman's six basic emotions. Leveraging the generative capabilities of LLMs, our approach enables the model to produce emotional descriptions for each aspect category, thereby enriching sentiment representations with affective expressions. Furthermore, to ensure the accuracy and consistency of the generated emotions, we introduce an emotion refinement mechanism based on the Valence-Arousal-Dominance (VAD) dimensional framework. Specifically, emotions predicted by the LLM are projected onto a VAD space, and those inconsistent with their corresponding VAD coordinates are re-annotated using a structured LLM-based refinement strategy. Experimental results demonstrate that our approach significantly outperforms strong baselines on all benchmark datasets. This underlines the effectiveness of integrating affective dimensions into ACSA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19083v1">A Multi-Agent LLM Framework for Multi-Domain Low-Resource In-Context NER via Knowledge Retrieval, Disambiguation and Reflective Analysis</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 This paper has been accepted by AAAI 2026 (Main Technical Track)
    </div>
    <details class="paper-abstract">
      In-context learning (ICL) with large language models (LLMs) has emerged as a promising paradigm for named entity recognition (NER) in low-resource scenarios. However, existing ICL-based NER methods suffer from three key limitations: (1) reliance on dynamic retrieval of annotated examples, which is problematic when annotated data is scarce; (2) limited generalization to unseen domains due to the LLM's insufficient internal domain knowledge; and (3) failure to incorporate external knowledge or resolve entity ambiguities. To address these challenges, we propose KDR-Agent, a novel multi-agent framework for multi-domain low-resource in-context NER that integrates Knowledge retrieval, Disambiguation, and Reflective analysis. KDR-Agent leverages natural-language type definitions and a static set of entity-level contrastive demonstrations to reduce dependency on large annotated corpora. A central planner coordinates specialized agents to (i) retrieve factual knowledge from Wikipedia for domain-specific mentions, (ii) resolve ambiguous entities via contextualized reasoning, and (iii) reflect on and correct model predictions through structured self-assessment. Experiments across ten datasets from five domains demonstrate that KDR-Agent significantly outperforms existing zero-shot and few-shot ICL baselines across multiple LLM backbones. The code and data can be found at https://github.com/MWXGOD/KDR-Agent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19078v1">GraphMind: Theorem Selection and Conclusion Generation Framework with Dynamic GNN for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive capabilities in natural language understanding and generation, including multi-step reasoning such as mathematical proving. However, existing approaches often lack an explicit and dynamic mechanism to structurally represent and evolve intermediate reasoning states, which limits their ability to perform context-aware theorem selection and iterative conclusion generation. To address these challenges, we propose GraphMind, a novel dynamic graph-based framework that integrates the graph neural network (GNN) with LLMs to iteratively select theorems and generate intermediate conclusions for multi-step reasoning. Our method models the reasoning process as a heterogeneous evolving graph, where nodes represent conditions, theorems, and conclusions, while edges capture logical dependencies between nodes. By encoding the current reasoning state with GNN and leveraging semantic matching for theorem selection, our framework enables context-aware, interpretable, and structured reasoning in a closed-loop manner. Experiments on various question-answering (QA) datasets demonstrate that our proposed GraphMind method achieves consistent performance improvements and significantly outperforms existing baselines in multi-step reasoning, validating the effectiveness and generalizability of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21762v1">Factors That Support Grounded Responses in LLM Conversations: A Rapid Review</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 28 pages, 1 figure, 3 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) may generate outputs that are misaligned with user intent, lack contextual grounding, or exhibit hallucinations during conversation, which compromises the reliability of LLM-based applications. This review aimed to identify and analyze techniques that align LLM responses with conversational goals, ensure grounding, and reduce hallucination and topic drift. We conducted a Rapid Review guided by the PRISMA framework and the PICO strategy to structure the search, filtering, and selection processes. The alignment strategies identified were categorized according to the LLM lifecycle phase in which they operate: inference-time, post-training, and reinforcement learning-based methods. Among these, inference-time approaches emerged as particularly efficient, aligning outputs without retraining while supporting user intent, contextual grounding, and hallucination mitigation. The reviewed techniques provided structured mechanisms for improving the quality and reliability of LLM responses across key alignment objectives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21761v1">LLMs for Low-Resource Dialect Translation Using Context-Aware Prompting: A Case Study on Sylheti</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated strong translation abilities through prompting, even without task-specific training. However, their effectiveness in dialectal and low-resource contexts remains underexplored. This study presents the first systematic investigation of LLM-based machine translation (MT) for Sylheti, a dialect of Bangla that is itself low-resource. We evaluate five advanced LLMs (GPT-4.1, GPT-4.1, LLaMA 4, Grok 3, and DeepSeek V3.2) across both translation directions (Bangla $\Leftrightarrow$ Sylheti), and find that these models struggle with dialect-specific vocabulary. To address this, we introduce Sylheti-CAP (Context-Aware Prompting), a three-step framework that embeds a linguistic rulebook, a dictionary (2{,}260 core vocabulary items and idioms), and an authenticity check directly into prompts. Extensive experiments show that Sylheti-CAP consistently improves translation quality across models and prompting strategies. Both automatic metrics and human evaluations confirm its effectiveness, while qualitative analysis reveals notable reductions in hallucinations, ambiguities, and awkward phrasing, establishing Sylheti-CAP as a scalable solution for dialectal and low-resource MT. Dataset link: \href{https://github.com/TabiaTanzin/LLMs-for-Low-Resource-Dialect-Translation-Using-Context-Aware-Prompting-A-Case-Study-on-Sylheti.git}{https://github.com/TabiaTanzin/LLMs-for-Low-Resource-Dialect-Translation-Using-Context-Aware-Prompting-A-Case-Study-on-Sylheti.git}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21757v1">Medical Malice: A Dataset for Context-Aware Safety in Healthcare LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into healthcare demands a safety paradigm rooted in \textit{primum non nocere}. However, current alignment techniques rely on generic definitions of harm that fail to capture context-dependent violations, such as administrative fraud and clinical discrimination. To address this, we introduce Medical Malice: a dataset of 214,219 adversarial prompts calibrated to the regulatory and ethical complexities of the Brazilian Unified Health System (SUS). Crucially, the dataset includes the reasoning behind each violation, enabling models to internalize ethical boundaries rather than merely memorizing a fixed set of refusals. Using an unaligned agent (Grok-4) within a persona-driven pipeline, we synthesized high-fidelity threats across seven taxonomies, ranging from procurement manipulation and queue-jumping to obstetric violence. We discuss the ethical design of releasing these "vulnerability signatures" to correct the information asymmetry between malicious actors and AI developers. Ultimately, this work advocates for a shift from universal to context-aware safety, providing the necessary resources to immunize healthcare AI against the nuanced, systemic threats inherent to high-stakes medical environments -- vulnerabilities that represent the paramount risk to patient safety and the successful integration of AI in healthcare systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18635v1">No Free Lunch in Language Model Bias Mitigation? Targeted Bias Reduction Can Exacerbate Unmitigated LLM Biases</a></div>
    <div class="paper-meta">
      📅 2025-11-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) inherit societal biases from their training data, potentially leading to harmful or unfair outputs. While various techniques aim to mitigate these biases, their effects are often evaluated only along the dimension of the bias being targeted. This work investigates the cross-category consequences of targeted bias mitigation. We study four bias mitigation techniques applied across ten models from seven model families, and we explore racial, religious, profession- and gender-related biases. We measure the impact of debiasing on model coherence and stereotypical preference using the StereoSet benchmark. Our results consistently show that while targeted mitigation can sometimes reduce bias in the intended dimension, it frequently leads to unintended and often negative consequences in others, such as increasing model bias and decreasing general coherence. These findings underscore the critical need for robust, multi-dimensional evaluation tools when examining and developing bias mitigation strategies to avoid inadvertently shifting or worsening bias along untargeted axes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18632v1">The Locally Deployable Virtual Doctor: LLM Based Human Interface for Automated Anamnesis and Database Conversion</a></div>
    <div class="paper-meta">
      📅 2025-11-23
    </div>
    <details class="paper-abstract">
      Recent advances in large language models made it possible to achieve high conversational performance with substantially reduced computational demands, enabling practical on-site deployment in clinical environments. Such progress allows for local integration of AI systems that uphold strict data protection and patient privacy requirements, yet their secure implementation in medicine necessitates careful consideration of ethical, regulatory, and technical constraints. In this study, we introduce MedChat, a locally deployable virtual physician framework that integrates an LLM-based medical chatbot with a diffusion-driven avatar for automated and structured anamnesis. The chatbot was fine-tuned using a hybrid corpus of real and synthetically generated medical dialogues, while model efficiency was optimized via Low-Rank Adaptation. A secure and isolated database interface was implemented to ensure complete separation between patient data and the inference process. The avatar component was realized through a conditional diffusion model operating in latent space, trained on researcher video datasets and synchronized with mel-frequency audio features for realistic speech and facial animation. Unlike existing cloud-based systems, this work demonstrates the feasibility of a fully offline, locally deployable LLM-diffusion framework for clinical anamnesis. The autoencoder and diffusion networks exhibited smooth convergence, and MedChat achieved stable fine-tuning with strong generalization to unseen data. The proposed system thus provides a privacy-preserving, resource-efficient foundation for AI-assisted clinical anamnesis, also in low-cost settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18608v1">From Reviewers' Lens: Understanding Bug Bounty Report Invalid Reasons with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-23
      | 💬 10 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Bug bounty platforms (e.g., HackerOne, BugCrowd) leverage crowd-sourced vulnerability discovery to improve continuous coverage, reduce the cost of discovery, and serve as an integral complement to internal red teams. With the rise of AI-generated bug reports, little work exists to help bug hunters understand why these reports are labeled as invalid. To improve report quality and reduce reviewers' burden, it is critical to predict invalid reports and interpret invalid reasons. In this work, we conduct an empirical study with the purpose of helping bug hunters understand the validity of reports. We collect a dataset of 9,942 disclosed bug bounty reports, including 1,400 invalid reports, and evaluate whether state-of-the-art large language models can identify invalid reports. While models such as GPT-5, DeepSeek, and a fine-tuned RoBERTa achieve strong overall accuracy, they consistently struggle to detect invalid cases, showing a tendency to over-accept reports. To improve invalidity detection, we build a taxonomy of rejection reasons for Information Disclosure vulnerabilities and incorporate it into a retrieval-augmented generation (RAG) framework. This approach substantially improves classification consistency and reduces bias. We also examine whether reviewer decisions may be influenced by factors beyond the content of the report. Our analysis shows that reporters with higher reputations tend to receive more favorable outcomes in borderline cases, suggesting that perceived expertise can influence review judgments. Overall, our findings highlight the challenges of invalid report identification and show that combining LLMs with structured reviewer knowledge can support more transparent and consistent vulnerability report review.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.08663v2">A Novel Framework for Augmenting Rating Scale Tests with LLM-Scored Text Data</a></div>
    <div class="paper-meta">
      📅 2025-11-23
    </div>
    <details class="paper-abstract">
      Psychological assessments are dominated by rating scales, which cannot capture the nuance in natural language. Efforts to supplement them with qualitative text have relied on labelled datasets or expert rubrics, limiting scalability. We introduce a framework that avoids this reliance: large language models (LLMs) score free-text responses with simple prompts to produce candidate LLM items, from which we retain those that yield the most test information when co-calibrated with a baseline scale. Using depression as a case study, we developed and tested the method in upper-secondary students (n=693) and a matched synthetic dataset (n=3,000). Results on held-out test sets showed that augmenting a 19-item scale with LLM items improved its precision, accuracy, and convergent validity. Further, the test information gain matched that of adding as many as 16 rating-scale items. This framework leverages the increasing availability of transcribed language to enhance psychometric measures, with applications in clinical health and beyond.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.15938v2">Straight to Zero: Why Linearly Decaying the Learning Rate to Zero Works Best for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-23
      | 💬 ICLR 2025
    </div>
    <details class="paper-abstract">
      LLMs are commonly trained with a learning rate (LR) warmup, followed by cosine decay to 10% of the maximum (10x decay). In a large-scale empirical study, we show that under an optimal peak LR, a simple linear decay-to-zero (D2Z) schedule consistently outperforms other schedules when training at compute-optimal dataset sizes. D2Z is superior across a range of model sizes, batch sizes, datasets, and vocabularies. Benefits increase as dataset size increases. Leveraging a novel interpretation of AdamW as an exponential moving average of weight updates, we show how linear D2Z optimally balances the demands of early training (moving away from initial conditions) and late training (averaging over more updates in order to mitigate gradient noise). In experiments, a 610M-parameter model trained for 80 tokens-per-parameter (TPP) using D2Z achieves lower loss than when trained for 200 TPP using 10x decay, corresponding to an astonishing 60% compute savings. Models such as Llama2-7B, trained for 286 TPP with 10x decay, could likely have saved a majority of compute by training with D2Z.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.13738v2">Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training</a></div>
    <div class="paper-meta">
      📅 2025-11-23
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Efficient LLM pre-training requires well-tuned hyperparameters (HPs), including learning rate $η$ and weight decay $λ$. We study scaling laws for HPs: formulas for how to scale HPs as we scale model size N, dataset size D, and batch size B. Recent work suggests the AdamW timescale, $τ= B/(ηλD)$, should remain constant across training settings, and we verify the implication that optimal $λ$ scales linearly with B, for a fixed N and D. However, as N and D scale, we show optimal $τ$ obeys a precise power law in the tokens-per-parameter ratio, D/N. This law thus provides a method to accurately predict $λ$opt in advance of large-scale training. We also study scaling laws for optimal batch size Bopt (the B enabling lowest loss at a given N,D) and critical batch size Bcrit (the B beyond which further data parallelism becomes ineffective). In contrast to prior work, we find both Bopt and Bcrit scale as power laws in D, independent of model size, N. Finally, we analyze how these findings inform the real-world selection of Pareto-optimal N and D under dual training time and compute objectives. All experiments were run on Cerebras CS-3 systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18589v1">Strategic Decision Framework for Enterprise LLM Adoption</a></div>
    <div class="paper-meta">
      📅 2025-11-23
      | 💬 14 pages, 1 key figure
    </div>
    <details class="paper-abstract">
      Organizations are rapidly adopting Large Language Models (LLMs) to transform their operations, yet they lack clear guidance on key decisions for adoption and implementation. While LLMs offer powerful capabilities in content generation, assisted coding, and process automation, businesses face critical challenges in data security, LLM solution development approach, infrastructure requirements, and deployment strategies. Healthcare providers must protect patient data while leveraging LLMs for medical analysis, financial institutions need to balance automated customer service with regulatory compliance, and software companies seek to enhance development productivity while maintaining code security. This article presents a systematic six-step decision framework for LLM adoption, helping organizations navigate from initial application selection to final deployment. Based on extensive interviews and analysis of successful and failed implementations, our framework provides practical guidance for business leaders to align technological capabilities with business objectives. Through key decision points and real-world examples from both B2B and B2C contexts, organizations can make informed decisions about LLM adoption while ensuring secure and efficient integration across various use cases, from customer service automation to content creation and advanced analytics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18581v1">TASO: Jailbreak LLMs via Alternative Template and Suffix Optimization</a></div>
    <div class="paper-meta">
      📅 2025-11-23
    </div>
    <details class="paper-abstract">
      Many recent studies showed that LLMs are vulnerable to jailbreak attacks, where an attacker can perturb the input of an LLM to induce it to generate an output for a harmful question. In general, existing jailbreak techniques either optimize a semantic template intended to induce the LLM to produce harmful outputs or optimize a suffix that leads the LLM to initiate its response with specific tokens (e.g., "Sure"). In this work, we introduce TASO (Template and Suffix Optimization), a novel jailbreak method that optimizes both a template and a suffix in an alternating manner. Our insight is that suffix optimization and template optimization are complementary to each other: suffix optimization can effectively control the first few output tokens but cannot control the overall quality of the output, while template optimization provides guidance for the entire output but cannot effectively control the initial tokens, which significantly impact subsequent responses. Thus, they can be combined to improve the attack's effectiveness. We evaluate the effectiveness of TASO on benchmark datasets (including HarmBench and AdvBench) on 24 leading LLMs (including models from the Llama family, OpenAI, and DeepSeek). The results demonstrate that TASO can effectively jailbreak existing LLMs. We hope our work can inspire future studies in exploring this direction. We will make code and data publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18531v1">LockForge: Automating Paper-to-Code for Logic Locking with Multi-Agent Reasoning LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-23
    </div>
    <details class="paper-abstract">
      Despite rapid progress in logic locking (LL), reproducibility remains a challenge as codes are rarely made public. We present LockForge, a first-of-its-kind, multi-agent large language model (LLM) framework that turns LL descriptions in papers into executable and tested code. LockForge provides a carefully crafted pipeline realizing forethought, implementation, iterative refinement, and a multi-stage validation, all to systematically bridge the gap between prose and practice for complex LL schemes. For validation, we devise (i) an LLM-as-Judge stage with a scoring system considering behavioral checks, conceptual mechanisms, structural elements, and reproducibility on benchmarks, and (ii) an independent LLM-as-Examiner stage for ground-truth assessment. We apply LockForge to 10 seminal LL schemes, many of which lack reference implementations. Our evaluation on multiple SOTA LLMs, including ablation studies, reveals the significant complexity of the task. We show that an advanced reasoning model and a sophisticated, multi-stage framework like LockForge are required. We release all implementations and benchmarks, providing a reproducible and fair foundation for evaluation of further LL research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.09711v2">PsychiatryBench: A Multi-Task Benchmark for LLMs in Psychiatry</a></div>
    <div class="paper-meta">
      📅 2025-11-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) offer significant potential in enhancing psychiatric practice, from improving diagnostic accuracy to streamlining clinical documentation and therapeutic support. However, existing evaluation resources heavily rely on small clinical interview corpora, social media posts, or synthetic dialogues, which limits their clinical validity and fails to capture the full complexity of diagnostic reasoning. In this work, we introduce PsychiatryBench, a rigorously curated benchmark grounded exclusively in authoritative, expert-validated psychiatric textbooks and casebooks. PsychiatryBench comprises eleven distinct question-answering tasks ranging from diagnostic reasoning and treatment planning to longitudinal follow-up, management planning, clinical approach, sequential case analysis, and multiple-choice/extended matching formats totaling 5,188 expert-annotated items. {\color{red}We evaluate a diverse set of frontier LLMs (including Google Gemini, DeepSeek, Sonnet 4.5, and GPT 5) alongside leading open-source medical models such as MedGemma using both conventional metrics and an "LLM-as-judge" similarity scoring framework. Our results reveal substantial gaps in clinical consistency and safety, particularly in multi-turn follow-up and management tasks, underscoring the need for specialized model tuning and more robust evaluation paradigms. PsychiatryBench offers a modular, extensible platform for benchmarking and improving LLM performance in mental health applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18467v1">Shadows in the Code: Exploring the Risks and Defenses of LLM-based Multi-Agent Software Development Systems</a></div>
    <div class="paper-meta">
      📅 2025-11-23
      | 💬 Accepted by AAAI 2026 Alignment Track
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Model (LLM)-driven multi-agent systems has significantly streamlined software developing tasks, enabling users with little technical expertise to develop executable applications. While these systems democratize software creation through natural language requirements, they introduce significant security risks that remain largely unexplored. We identify two risky scenarios: Malicious User with Benign Agents (MU-BA) and Benign User with Malicious Agents (BU-MA). We introduce the Implicit Malicious Behavior Injection Attack (IMBIA), demonstrating how multi-agent systems can be manipulated to generate software with concealed malicious capabilities beneath seemingly benign applications, and propose Adv-IMBIA as a defense mechanism. Evaluations across ChatDev, MetaGPT, and AgentVerse frameworks reveal varying vulnerability patterns, with IMBIA achieving attack success rates of 93%, 45%, and 71% in MU-BA scenarios, and 71%, 84%, and 45% in BU-MA scenarios. Our defense mechanism reduced attack success rates significantly, particularly in the MU-BA scenario. Further analysis reveals that compromised agents in the coding and testing phases pose significantly greater security risks, while also identifying critical agents that require protection against malicious user exploitation. Our findings highlight the urgent need for robust security measures in multi-agent software development systems and provide practical guidelines for implementing targeted, resource-efficient defensive strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.11088v2">One SPACE to Rule Them All: Jointly Mitigating Factuality and Faithfulness Hallucinations in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-23
      | 💬 Accepted as NIPS 2025 poster
    </div>
    <details class="paper-abstract">
      LLMs have demonstrated unprecedented capabilities in natural language processing, yet their practical deployment remains hindered by persistent factuality and faithfulness hallucinations. While existing methods address these hallucination types independently, they inadvertently induce performance trade-offs, as interventions targeting one type often exacerbate the other. Through empirical and theoretical analysis of activation space dynamics in LLMs, we reveal that these hallucination categories share overlapping subspaces within neural representations, presenting an opportunity for concurrent mitigation. To harness this insight, we propose SPACE, a unified framework that jointly enhances factuality and faithfulness by editing shared activation subspaces. SPACE establishes a geometric foundation for shared subspace existence through dual-task feature modeling, then identifies and edits these subspaces via a hybrid probe strategy combining spectral clustering and attention head saliency scoring. Experimental results across multiple benchmark datasets demonstrate the superiority of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14617v2">Seer: Online Context Learning for Fast Synchronous LLM Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-23
      | 💬 16 pages, 12 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Reinforcement Learning (RL) has become critical for advancing modern Large Language Models (LLMs), yet existing synchronous RL systems face severe performance bottlenecks. The rollout phase, which dominates end-to-end iteration time, suffers from substantial long-tail latency and poor resource utilization due to inherent workload imbalance. We present Seer, a novel online context learning system that addresses these challenges by exploiting previously overlooked similarities in output lengths and generation patterns among requests sharing the same prompt. Seer introduces three key techniques: divided rollout for dynamic load balancing, context-aware scheduling, and adaptive grouped speculative decoding. Together, these mechanisms substantially reduce long-tail latency and improve resource efficiency during rollout. Evaluations on production-grade RL workloads demonstrate that Seer improves end-to-end rollout throughput by 74% to 97% and reduces long-tail latency by 75% to 93% compared to state-of-the-art synchronous RL systems, significantly accelerating RL training iterations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.18416v2">Exploring Potential Prompt Injection Attacks in Federated Military LLMs and Their Mitigation</a></div>
    <div class="paper-meta">
      📅 2025-11-23
      | 💬 Accepted to the 3rd International Workshop on Dataspaces and Digital Twins for Critical Entities and Smart Urban Communities - IEEE BigData 2025
    </div>
    <details class="paper-abstract">
      Federated Learning (FL) is increasingly being adopted in military collaborations to develop Large Language Models (LLMs) while preserving data sovereignty. However, prompt injection attacks-malicious manipulations of input prompts-pose new threats that may undermine operational security, disrupt decision-making, and erode trust among allies. This perspective paper highlights four vulnerabilities in federated military LLMs: secret data leakage, free-rider exploitation, system disruption, and misinformation spread. To address these risks, we propose a human-AI collaborative framework with both technical and policy countermeasures. On the technical side, our framework uses red/blue team wargaming and quality assurance to detect and mitigate adversarial behaviors of shared LLM weights. On the policy side, it promotes joint AI-human policy development and verification of security protocols.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18438v1">LLMs as Firmware Experts: A Runtime-Grown Tree-of-Agents Framework</a></div>
    <div class="paper-meta">
      📅 2025-11-23
      | 💬 18 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) and their agent systems have recently demonstrated strong potential in automating code reasoning and vulnerability detection. However, when applied to large-scale firmware, their performance degrades due to the binary nature of firmware, complex dependency structures, and heterogeneous components. To address this challenge, this paper presents FIRMHIVE, a recursive agent hive that enables LLMs to act as autonomous firmware security analysts. FIRMHIVE introduces two key mechanisms: (1) transforming delegation into a per-agent, executable primitive and (2) constructing a runtime Tree of Agents (ToA) for decentralized coordination. We evaluate FIRMHIVE using real-world firmware images obtained from publicly available datasets, covering five representative security analysis tasks. Compared with existing LLM-agent baselines, FIRMHIVE performs deeper (about 16x more reasoning steps) and broader (about 2.3x more files inspected) cross-file exploration, resulting in about 5.6x more alerts per firmware. Compared to state-of-the-art (SOTA) security tools, FIRMHIVE identifies about 1.5x more vulnerabilities (1,802 total) and achieves 71% precision, representing significant improvements in both yield and fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18403v1">UnWEIRDing LLM Entity Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-11-23
    </div>
    <details class="paper-abstract">
      Large Language Models have been widely been adopted by users for writing tasks such as sentence completions. While this can improve writing efficiency, prior research shows that LLM-generated suggestions may exhibit cultural biases which may be difficult for users to detect, especially in educational contexts for non-native English speakers. While such prior work has studied the biases in LLM moral value alignment, we aim to investigate cultural biases in LLM recommendations for real-world entities. To do so, we use the WEIRD (Western, Educated, Industrialized, Rich and Democratic) framework to evaluate recommendations by various LLMs across a dataset of fine-grained entities, and apply pluralistic prompt-based strategies to mitigate these biases. Our results indicate that while such prompting strategies do reduce such biases, this reduction is not consistent across different models, and recommendations for some types of entities are more biased than others.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18394v1">Future Is Unevenly Distributed: Forecasting Ability of LLMs Depends on What We're Asking</a></div>
    <div class="paper-meta">
      📅 2025-11-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate partial forecasting competence across social, political, and economic events. Yet, their predictive ability varies sharply with domain structure and prompt framing. We investigate how forecasting performance varies with different model families on real-world questions about events that happened beyond the model cutoff date. We analyze how context, question type, and external knowledge affect accuracy and calibration, and how adding factual news context modifies belief formation and failure modes. Our results show that forecasting ability is highly variable as it depends on what, and how, we ask.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18343v1">A Needle in a Haystack: Intent-driven Reusable Artifacts Recommendation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-23
      | 💬 15 pages, 7 figures
    </div>
    <details class="paper-abstract">
      In open source software development, the reuse of existing artifacts has been widely adopted to avoid redundant implementation work. Reusable artifacts are considered more efficient and reliable than developing software components from scratch. However, when faced with a large number of reusable artifacts, developers often struggle to find artifacts that can meet their expected needs. To reduce this burden, retrieval-based and learning-based techniques have been proposed to automate artifact recommendations. Recently, Large Language Models (LLMs) have shown the potential to understand intentions, perform semantic alignment, and recommend usable artifacts. Nevertheless, their effectiveness has not been thoroughly explored. To fill this gap, we construct an intent-driven artifact recommendation benchmark named IntentRecBench, covering three representative open source ecosystems. Using IntentRecBench, we conduct a comprehensive comparative study of five popular LLMs and six traditional approaches in terms of precision and efficiency. Our results show that although LLMs outperform traditional methods, they still suffer from low precision and high inference cost due to the large candidate space. Inspired by the ontology-based semantic organization in software engineering, we propose TreeRec, a feature tree-guided recommendation framework to mitigate these issues. TreeRec leverages LLM-based semantic abstraction to organize artifacts into a hierarchical semantic tree, enabling intent and function alignment and reducing reasoning time. Extensive experiments demonstrate that TreeRec consistently improves the performance of diverse LLMs across ecosystems, highlighting its generalizability and potential for practical deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.02393v3">AP2O-Coder: Human-Inspired Progressive Optimization to Fix LLM Code Errors</a></div>
    <div class="paper-meta">
      📅 2025-11-23
      | 💬 Accepted by AAAI2026
    </div>
    <details class="paper-abstract">
      LLMs' code generation capabilities have yielded substantial improvements in the effectiveness of programming tasks. However, LLM-generated code still suffers from compilation and runtime errors. Existing offline preference optimization methods primarily focus on enhancing LLMs' coding abilities using pass/fail signals in the preference data, overlooking the deep-level error types in the failed codes. To address this, we propose Adaptively Progressive Preference Optimization (AP2O) for coding (i.e., AP2O-Coder), a method that guides LLMs adaptively and methodically to reduce code errors for code generation. Specifically, we construct an error notebook from failed codes and progressively optimize the LLM to correct errors type by type. Furthermore, we adaptively replay error types to tailor to the LLM's changing weaknesses throughout the training process. Through extensive experiments on both code and general LLMs (Llama, Qwen, and DeepSeek series) with parameters ranging from 0.5B to 34B, our AP2O-Coder improves code generation performance by up to 3% in pass@k while using less preference data. Code: https://github.com/TsingZ0/AP2O
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.05012v2">LLMs' Reshaping of People, Processes, Products, and Society in Software Development: A Comprehensive Exploration with Early Adopters</a></div>
    <div class="paper-meta">
      📅 2025-11-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are rapidly reshaping software development, but their impact across the software development lifecycle is underexplored. Existing work focuses on isolated activities such as code generation or testing, leaving open questions about how LLMs affect developers, processes, products, and the software ecosystem. We address this gap through semi-structured interviews with sixteen early-adopter software professionals who integrated LLM-based tools into their day-to-day work in early to mid-2023. We treat these interviews as early empirical evidence and compare participants' accounts with recent work on LLMs in software engineering, noting which early patterns persist or shift. Using thematic analysis, we organize findings around four dimensions: people, process, product, and society. Developers reported substantial productivity gains from reducing routine tasks, streamlining search, and accelerating debugging, but also described a productivity-quality paradox: they often discarded generated code and shifted effort from writing to critically evaluating and integrating it. LLM use was highly phase-dependent, with strong uptake in implementation and debugging but limited influence on requirements gathering and other collaborative work. Participants developed new competencies to use LLMs effectively, including prompt engineering strategies, layered verification, and secure integration to protect proprietary data. They anticipated changes in hiring expectations, team practices, and computing education, while emphasizing that human judgment and foundational software engineering skills remain essential. Our findings, later echoed in large-scale quantitative studies, offer actionable implications for developers, organizations, educators, and tool designers seeking to integrate LLMs responsibly into software practice today.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18313v1">Path-Constrained Retrieval: A Structural Approach to Reliable LLM Agent Reasoning Through Graph-Scoped Semantic Search</a></div>
    <div class="paper-meta">
      📅 2025-11-23
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Large Language Model agents often retrieve context from knowledge bases that lack structural consistency with the agent's current reasoning state, leading to incoherent reasoning chains. We introduce Path-Constrained Retrieval (PCR), a retrieval method that combines structural graph constraints with semantic search to ensure retrieved information maintains logical relationships within a knowledge graph. PCR restricts the search space to nodes reachable from an anchor node, preventing retrieval of structurally disconnected information that may lead to inconsistent reasoning. We evaluate PCR on PathRAG-6, a benchmark spanning six domains with 180 nodes and 360 edges. Our results show that PCR achieves full structural consistency compared to 24-32 percent in baseline methods, while maintaining strong relevance scores. On the technology domain, PCR obtains full relevance at rank 10 with full structural consistency, significantly outperforming vector search and hybrid retrieval. PCR reduces the average graph distance of retrieved context by 78 percent compared to baselines, demonstrating retrieval of more structurally consistent information. These findings suggest that path-constrained retrieval is an effective approach for improving the reliability and coherence of LLM agent reasoning systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.05130v2">Scalable and Accurate Graph Reasoning with LLM-based Multi-Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-23
      | 💬 Accepted by AAAI 2026 Workshop WMAC
    </div>
    <details class="paper-abstract">
      Recent research has explored the use of Large Language Models (LLMs) for tackling complex graph reasoning tasks. However, due to the intricacies of graph structures and the inherent limitations of LLMs in handling long text, current approaches often fail to deliver satisfactory accuracy, even on small-scale graphs and simple tasks. To address these challenges, we introduce GraphAgent-Reasoner, a fine-tuning-free framework that utilizes a multi-agent collaboration strategy for explicit and precise graph reasoning. Inspired by distributed graph computation theory, our framework decomposes graph problems into smaller, node-centric tasks that are distributed among multiple agents. The agents collaborate to solve the overall problem, significantly reducing the amount of information and complexity handled by a single LLM, thus enhancing the accuracy of graph reasoning. By simply increasing the number of agents, GraphAgent-Reasoner can efficiently scale to accommodate larger graphs with over 1,000 nodes. Evaluated on the GraphInstruct dataset, our framework demonstrates near-perfect accuracy on polynomial-time graph reasoning tasks, significantly outperforming the best available models, both closed-source and fine-tuned open-source variants. Our framework also demonstrates the capability to handle real-world graph reasoning applications such as webpage importance analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18270v1">Skypilot: Fine-Tuning LLM with Physical Grounding for AAV Coverage Search</a></div>
    <div class="paper-meta">
      📅 2025-11-23
    </div>
    <details class="paper-abstract">
      Autonomous aerial vehicles (AAVs) have played a pivotal role in coverage operations and search missions. Recent advances in large language models (LLMs) offer promising opportunities to augment AAV intelligence. These advances help address complex challenges like area coverage optimization, dynamic path planning, and adaptive decision-making. However, the absence of physical grounding in LLMs leads to hallucination and reproducibility problems in spatial reasoning and decision-making. To tackle these issues, we present Skypilot, an LLM-enhanced two-stage framework that grounds language models in physical reality by integrating monte carlo tree search (MCTS). In the first stage, we introduce a diversified action space that encompasses generate, regenerate, fine-tune, and evaluate operations, coupled with physics-informed reward functions to ensure trajectory feasibility. In the second stage, we fine-tune Qwen3-4B on 23,000 MCTS-generated samples, achieving substantial inference acceleration while maintaining solution quality. Extensive numerical simulations and real-world flight experiments validate the efficiency and superiority of our proposed approach. Detailed information and experimental results are accessible at https://sky-pilot.top.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18261v1">LLM Reasoning for Cold-Start Item Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-11-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown significant potential for improving recommendation systems through their inherent reasoning capabilities and extensive knowledge base. Yet, existing studies predominantly address warm-start scenarios with abundant user-item interaction data, leaving the more challenging cold-start scenarios, where sparse interactions hinder traditional collaborative filtering methods, underexplored. To address this limitation, we propose novel reasoning strategies designed for cold-start item recommendations within the Netflix domain. Our method utilizes the advanced reasoning capabilities of LLMs to effectively infer user preferences, particularly for newly introduced or rarely interacted items. We systematically evaluate supervised fine-tuning, reinforcement learning-based fine-tuning, and hybrid approaches that combine both methods to optimize recommendation performance. Extensive experiments on real-world data demonstrate significant improvements in both methodological efficacy and practical performance in cold-start recommendation contexts. Remarkably, our reasoning-based fine-tuned models outperform Netflix's production ranking model by up to 8% in certain cases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18249v1">LLM Assisted Coding with Metamorphic Specification Mutation Agent</a></div>
    <div class="paper-meta">
      📅 2025-11-23
    </div>
    <details class="paper-abstract">
      Metamorphic Relations (MRs) serve as a foundational mechanism for generating semantically equivalent mutations. Software engineering has advanced significantly in recent years with the advent of Large Language Models (LLMs). However, the reliability of LLMs in software engineering is often compromised by ambiguities and inconsistencies due to improper user specification. To address this challenge, we present CodeMetaAgent (CMA), a metamorphic relation-driven LLM agent that systematically refines task specifications and generates semantically constrained test cases. Our proposed framework uses MRs with LLMs to improve generation consistency and reduce variability caused by specifications, unlike the traditional use of MRs as post validations. Our framework has been evaluated on the HumanEval-Pro, MBPP-Pro, and SWE-Bench_Lite datasets using the GPT-4o, Mistral Large, GPT-OSS, and Qwen3-Coder models. It improved code generation accuracy by up to 17% and achieved code coverage gains of up to 99.81%. These results show that metamorphic relations can be a simple but effective guide in assisting LLM-based software development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.10729v2">Using LLMs for Late Multimodal Sensor Fusion for Activity Recognition</a></div>
    <div class="paper-meta">
      📅 2025-11-23
      | 💬 Preprint, under review
    </div>
    <details class="paper-abstract">
      Sensor data streams provide valuable information around activities and context for downstream applications, though integrating complementary information can be challenging. We show that large language models (LLMs) can be used for late fusion for activity classification from audio and motion time series data. We curated a subset of data for diverse activity recognition across contexts (e.g., household activities, sports) from the Ego4D dataset. Evaluated LLMs achieved 12-class zero- and one-shot classification F1-scores significantly above chance, with no task-specific training. Zero-shot classification via LLM-based fusion from modality-specific models can enable multimodal temporal applications where there is limited aligned training data for learning a shared embedding space. Additionally, LLM-based fusion can enable model deploying without requiring additional memory and computation for targeted application-specific multimodal models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18239v1">Can LLMs Help Allocate Public Health Resources? A Case Study on Childhood Lead Testing</a></div>
    <div class="paper-meta">
      📅 2025-11-23
    </div>
    <details class="paper-abstract">
      Public health agencies face critical challenges in identifying high-risk neighborhoods for childhood lead exposure with limited resources for outreach and intervention programs. To address this, we develop a Priority Score integrating untested children proportions, elevated blood lead prevalence, and public health coverage patterns to support optimized resource allocation decisions across 136 neighborhoods in Chicago, New York City, and Washington, D.C. We leverage these allocation tasks, which require integrating multiple vulnerability indicators and interpreting empirical evidence, to evaluate whether large language models (LLMs) with agentic reasoning and deep research capabilities can effectively allocate public health resources when presented with structured allocation scenarios. LLMs were tasked with distributing 1,000 test kits within each city based on neighborhood vulnerability indicators. Results reveal significant limitations: LLMs frequently overlooked neighborhoods with highest lead prevalence and largest proportions of untested children, such as West Englewood in Chicago, while allocating disproportionate resources to lower-priority areas like Hunts Point in New York City. Overall accuracy averaged 0.46, reaching a maximum of 0.66 with ChatGPT 5 Deep Research. Despite their marketed deep research capabilities, LLMs struggled with fundamental limitations in information retrieval and evidence-based reasoning, frequently citing outdated data and allowing non-empirical narratives about neighborhood conditions to override quantitative vulnerability indicators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18230v1">Think Fast: Real-Time IoT Intrusion Reasoning Using IDS and LLMs at the Edge Gateway</a></div>
    <div class="paper-meta">
      📅 2025-11-23
    </div>
    <details class="paper-abstract">
      As the number of connected IoT devices continues to grow, securing these systems against cyber threats remains a major challenge, especially in environments with limited computational and energy resources. This paper presents an edge-centric Intrusion Detection System (IDS) framework that integrates lightweight machine learning (ML) based IDS models with pre-trained large language models (LLMs) to improve detection accuracy, semantic interpretability, and operational efficiency at the network edge. The system evaluates six ML-based IDS models: Decision Tree (DT), K-Nearest Neighbors (KNN), Random Forest (RF), Convolutional Neural Network (CNN), Long Short-Term Memory (LSTM), and a hybrid CNN-LSTM model on low-power edge gateways, achieving accuracy up to 98 percent under real-world cyberattacks. For anomaly detection, the system transmits a compact and secure telemetry snapshot (for example, CPU usage, memory usage, latency, and energy consumption) via low-bandwidth API calls to LLMs including GPT-4-turbo, DeepSeek V2, and LLaMA 3.5. These models use zero-shot, few-shot, and chain-of-thought reasoning to produce human-readable threat analyses and actionable mitigation recommendations. Evaluations across diverse attacks such as DoS, DDoS, brute force, and port scanning show that the system enhances interpretability while maintaining low latency (<1.5 s), minimal bandwidth usage (<1.2 kB per prompt), and energy efficiency (<75 J), demonstrating its practicality and scalability as an IDS solution for edge gateways.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19489v1">Evolution without an Oracle: Driving Effective Evolution with LLM Judges</a></div>
    <div class="paper-meta">
      📅 2025-11-23
      | 💬 14 pages, 5 figures
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) with Evolutionary Computation (EC) has unlocked new frontiers in scientific discovery but remains shackled by a fundamental constraint: the reliance on an Oracle--an objective, machine-computable fitness function. This paper breaks this barrier by asking: Can evolution thrive in a purely subjective landscape governed solely by LLM judges? We introduce MADE (Multi-Agent Decomposed Evolution), a framework that tames the inherent noise of subjective evaluation through "Problem Specification." By decomposing vague instructions into specific, verifiable sub-requirements, MADE transforms high-variance LLM feedback into stable, precise selection pressure. The results are transformative: across complex benchmarks like DevAI and InfoBench, MADE outperforms strong baselines by over 50% in software requirement satisfaction (39.9% to 61.9%) and achieves a 95% perfect pass rate on complex instruction following. This work validates a fundamental paradigm shift: moving from optimizing "computable metrics" to "describable qualities," thereby unlocking evolutionary optimization for the vast open-ended domains where no ground truth exists.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19488v1">Building Resilient Information Ecosystems: Large LLM-Generated Dataset of Persuasion Attacks</a></div>
    <div class="paper-meta">
      📅 2025-11-23
    </div>
    <details class="paper-abstract">
      Organization's communication is essential for public trust, but the rise of generative AI models has introduced significant challenges by generating persuasive content that can form competing narratives with official messages from government and commercial organizations at speed and scale. This has left agencies in a reactive position, often unaware of how these models construct their persuasive strategies, making it more difficult to sustain communication effectiveness. In this paper, we introduce a large LLM-generated persuasion attack dataset, which includes 134,136 attacks generated by GPT-4, Gemma 2, and Llama 3.1 on agency news. These attacks span 23 persuasive techniques from SemEval 2023 Task 3, directed toward 972 press releases from ten agencies. The generated attacks come in two mediums, press release statements and social media posts, covering both long-form and short-form communication strategies. We analyzed the moral resonance of these persuasion attacks to understand their attack vectors. GPT-4's attacks mainly focus on Care, with Authority and Loyalty also playing a role. Gemma 2 emphasizes Care and Authority, while Llama 3.1 centers on Loyalty and Care. Analyzing LLM-generated persuasive attacks across models will enable proactive defense, allow to create the reputation armor for organizations, and propel the development of both effective and resilient communications in the information ecosystem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19483v1">Z-Space: A Multi-Agent Tool Orchestration Framework for Enterprise-Grade LLM Automation</a></div>
    <div class="paper-meta">
      📅 2025-11-23
    </div>
    <details class="paper-abstract">
      Large Language Models can break through knowledge and timeliness limitations by invoking external tools within the Model Context Protocol framework to achieve automated execution of complex tasks. However, with the rapid growth of enterprise-scale MCP services, efficiently and accurately matching target functionalities among thousands of heterogeneous tools has become a core challenge restricting system practicality. Existing approaches generally rely on full-prompt injection or static semantic retrieval, facing issues including semantic disconnection between user queries and tool descriptions, context inflation in LLM input, and high inference latency. To address these challenges, this paper proposes Z-Space, a data-generation-oriented multi-agent collaborative tool invocation framework Z-Space. The Z-Space framework establishes a multi-agent collaborative architecture and tool filtering algorithm: (1) A structured semantic understanding of user queries is achieved through an intent parsing model; (2) A tool filtering module (FSWW) based on fused subspace weighted algorithm realizes fine-grained semantic alignment between intents and tools without parameter tuning; (3) An inference execution agent is constructed to support dynamic planning and fault-tolerant execution for multi-step tasks. This framework has been deployed in the Eleme platform's technical division, serving large-scale test data generation scenarios across multiple business units including Taotian, Gaode, and Hema. Production data demonstrates that the system reduces average token consumption in tool inference by 96.26\% while achieving a 92\% tool invocation accuracy rate, significantly enhancing the efficiency and reliability of intelligent test data generation systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.18617v2">DarkMind: Latent Chain-of-Thought Backdoor in Customized LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-23
      | 💬 19 pages, 15 figures, 12 tables
    </div>
    <details class="paper-abstract">
      With the rapid rise of personalized AI, customized large language models (LLMs) equipped with Chain of Thought (COT) reasoning now power millions of AI agents. However, their complex reasoning processes introduce new and largely unexplored security vulnerabilities. We present DarkMind, a novel latent reasoning level backdoor attack that targets customized LLMs by manipulating internal COT steps without altering user queries. Unlike prior prompt based attacks, DarkMind activates covertly within the reasoning chain via latent triggers, enabling adversarial behaviors without modifying input prompts or requiring access to model parameters. To achieve stealth and reliability, we propose dual trigger types instant and retrospective and integrate them within a unified embedding template that governs trigger dependent activation, employ a stealth optimization algorithm to minimize semantic drift, and introduce an automated conversation starter for covert activation across domains. Comprehensive experiments on eight reasoning datasets spanning arithmetic, commonsense, and symbolic domains, using five LLMs, demonstrate that DarkMind consistently achieves high attack success rates. We further investigate defense strategies to mitigate these risks and reveal that reasoning level backdoors represent a significant yet underexplored threat, underscoring the need for robust, reasoning aware security mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18653v1">FHE-Agent: Automating CKKS Configuration for Practical Encrypted Inference via an LLM-Guided Agentic Framework</a></div>
    <div class="paper-meta">
      📅 2025-11-23
    </div>
    <details class="paper-abstract">
      Fully Homomorphic Encryption (FHE), particularly the CKKS scheme, is a promising enabler for privacy-preserving MLaaS, but its practical deployment faces a prohibitive barrier: it heavily relies on domain expertise. Configuring CKKS involves a tightly coupled space of ring dimensions, modulus chains, and packing layouts. Without deep cryptographic knowledge to navigate these interactions, practitioners are restricted to compilers that rely on fixed heuristics. These "one-shot" tools often emit rigid configurations that are either severely over-provisioned in latency or fail to find a feasible solution entirely for deeper networks. We present FHE-Agent, an agentic framework that automates this expert reasoning process. By coupling a Large Language Model (LLM) controller with a deterministic tool suite, FHE-Agent decomposes the search into global parameter selection and layer-wise bottleneck repair. The agents operate within a multi-fidelity workflow, pruning invalid regimes using cheap static analysis and reserving expensive encrypted evaluations for the most promising candidates. We instantiate FHE-Agent on the Orion compiler and evaluate it on standard benchmarks (MLP, LeNet, LoLa) and deeper architectures (AlexNet). FHE-Agent consistently achieves better precision and lower latency than naïve search strategies. Crucially, it automatically discovers feasible, 128-bit secure configurations for complex models where baseline heuristics and one-shot prompts fail to produce a valid setup.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2312.15524v3">The Challenge of Using LLMs to Simulate Human Behavior: A Causal Inference Perspective</a></div>
    <div class="paper-meta">
      📅 2025-11-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown impressive potential to simulate human behavior. We identify a fundamental challenge in using them to simulate experiments: when LLM-simulated subjects are blind to the experimental design (as is standard practice with human subjects), variations in treatment systematically affect unspecified variables that should remain constant, violating the unconfoundedness assumption. Using demand estimation as a context and an actual experiment with 40 different products as a benchmark, we show this can lead to implausible results. While confounding may in principle be addressed by controlling for covariates, this can compromise ecological validity in the context of LLM simulations: controlled covariates become artificially salient in the simulated decision process. We show formally that confoundness stems from ambiguous prompting strategies. Therefore, it can be addressed by developing unambiguous prompting strategies through unblinding, i.e., revealing the experiment design in LLM simulations. Our empirical results show that this strategy consistently enhances model performance across all tested models, including both out-of-box reasoning and non-reasoning models. We also show that it is a technique that complements fine-tuning: while fine-tuning can improve simulation performance, an unambiguous prompting strategy makes the predictions robust to the inclusion of irrelevant data in the fine-tuning process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17923v1">Towards Efficient LLM-aware Heterogeneous Graph Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-22
    </div>
    <details class="paper-abstract">
      Heterogeneous graphs are widely present in real-world complex networks, where the diversity of node and relation types leads to complex and rich semantics. Efforts for modeling complex relation semantics in heterogeneous graphs are restricted by the limitations of predefined semantic dependencies and the scarcity of supervised signals. The advanced pre-training and fine-tuning paradigm leverages graph structure to provide rich self-supervised signals, but introduces semantic gaps between tasks. Large Language Models (LLMs) offer significant potential to address the semantic issues of relations and tasks in heterogeneous graphs through their strong reasoning capabilities in textual modality, but their incorporation into heterogeneous graphs is largely limited by computational complexity. Therefore, in this paper, we propose an Efficient LLM-Aware (ELLA) framework for heterogeneous graphs, addressing the above issues. To capture complex relation semantics, we propose an LLM-aware Relation Tokenizer that leverages LLM to encode multi-hop, multi-type relations. To reduce computational complexity, we further employ a Hop-level Relation Graph Transformer, which help reduces the complexity of LLM-aware relation reasoning from exponential to linear. To bridge semantic gaps between pre-training and fine-tuning tasks, we introduce the fine-grained task-aware textual Chain-of-Thought (CoT) prompts. Extensive experiments on four heterogeneous graphs show that our proposed ELLA outperforms state-of-the-art methods in the performance and efficiency. In particular, ELLA scales up to 13b-parameter LLMs and achieves up to a 4x speedup compared with existing LLM-based methods. Our code is publicly available at https://github.com/l-wd/ELLA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17913v1">Token-Controlled Re-ranking for Sequential Recommendation via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-22
    </div>
    <details class="paper-abstract">
      The widespread adoption of Large Language Models (LLMs) as re-rankers is shifting recommender systems towards a user-centric paradigm. However, a significant gap remains: current re-rankers often lack mechanisms for fine-grained user control. They struggle to balance inherent user preferences with multiple attribute-based constraints, often resorting to simplistic hard filtering that can excessively narrow the recommendation pool and yield suboptimal results. This limitation leaves users as passive recipients rather than active collaborators in the recommendation process. To bridge this gap, we propose COREC, a novel token-augmented re-ranking framework that incorporates specific user requirements in co-creating the recommendation outcome. COREC empowers users to steer re-ranking results with precise and flexible control via explicit, attribute-based signals. The framework learns to balance these commands against latent preferences, yielding rankings that adhere to user instructions without sacrificing personalization. Experiments show that COREC: (1) exceeds state-of-the-art baselines on standard recommendation effectiveness and (2) demonstrates superior adherence to specific attribute requirements, proving that COREC enables fine-grained and predictable manipulation of the rankings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.07189v2">Secure-Instruct: An Automated Pipeline for Synthesizing Instruction-Tuning Datasets Using LLMs for Secure Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-11-22
    </div>
    <details class="paper-abstract">
      Although Large Language Models (LLMs) show promising solutions to automated code generation, they often produce insecure code that threatens software security. Current approaches (e.g., SafeCoder) to improve secure code generation are limited by small, imbalanced instruction-tuning datasets. In this work, we present Secure-Instruct, a novel pipeline that automatically synthesizes high-quality vulnerable and secure code examples and instruction-tunes LLMs to align task description and secure code generation abilities. We evaluate Secure-Instruct on four representative LLMs using two security-related benchmarks: our own CWEBench and the existing CWEval. CWEBench comprises 93 scenarios on 44 CWEs, all without overlap with Secure-Instruct's synthetic instruction-tuning dataset, while CWEval covers 31 CWEs with 119 manually verified security-critical tasks. We find that Secure-Instruct improves both security and functional correctness in code generation. On CWEBench, Secure-Instruct substantially improves secure code generation, giving a 28.5% increase on average in secure ratio over the pre-trained models and outperforms SafeCoder by 12.6%. On CWEval, Secure-Instruct achieves an increase of 157.3% for CodeLlama-7B and 46.4% for Mistral-7B in Func-Sec@1 over pretrained models, and significantly outperforms SafeCoder.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12288v2">Reducing Hallucinations in LLM-Generated Code via Semantic Triangulation</a></div>
    <div class="paper-meta">
      📅 2025-11-22
    </div>
    <details class="paper-abstract">
      When generating code from natural language prompts, an LLM samples programs from a probability distribution, many of which might be incorrect. Sample consensus techniques - such as majority voting or validation against generated tests or specifications - aim to identify a correct program in the sample or abstain if none is valid. However, existing methods often fail to select a correct solution when its sampling probability is low, or when the problem permits multiple valid but non-equivalent solutions. Additionally, they often fail to abstain when no correct solution is present in the sample. To overcome these limitations, we introduce semantic triangulation, which transforms a programming problem in a way that non-trivially alters its semantics while preserving an exact, verifiable mapping between solutions before and after transformation. We theoretically establish that verifying consistency across such problem transformations increases confidence that generated programs reflect accurate generalization rather than spurious statistical correlations, enabling more reliable sample consensus and abstention. On the LiveCodeBench and CodeElo benchmarks, using GPT-4o and DeepSeek-V3 models, semantic triangulation increases reliability of generated code by 21% compared to the method that selects only high-confidence solutions with the probability threshold 0.5, while being able to pinpoint correct solutions at sampling probabilities as low as 0.14. Apart from that, it is also the only approach to consistently form true consensus on tasks with multiple valid but non-equivalent solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.17648v3">Simulating Macroeconomic Expectations using LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-22
    </div>
    <details class="paper-abstract">
      We introduce a novel framework for simulating macroeconomic expectations using Large Language Model-Empowered Agents (LLM Agents). By constructing LLM Agents equipped with various functional modules, we replicate three representative survey experiments involving several expectations across different types of economic agents. Our results show that although the expectations simulated by LLM Agents are more homogeneous than those of humans, they consistently outperform LLMs relying simply on prompt engineering, and possess human-like mental mechanisms. Evaluation reveals that these capabilities stem from the contributions of their components, offering guidelines for their architectural design. Our approach complements traditional methods and provides new insights into AI behavioral science in macroeconomic research
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17874v1">Beyond Jailbreak: Unveiling Risks in LLM Applications Arising from Blurred Capability Boundaries</a></div>
    <div class="paper-meta">
      📅 2025-11-22
      | 💬 Accepted by Network and Distributed System Security (NDSS) Symposium 2026
    </div>
    <details class="paper-abstract">
      LLM applications (i.e., LLM apps) leverage the powerful capabilities of LLMs to provide users with customized services, revolutionizing traditional application development. While the increasing prevalence of LLM-powered applications provides users with unprecedented convenience, it also brings forth new security challenges. For such an emerging ecosystem, the security community lacks sufficient understanding of the LLM application ecosystem, especially regarding the capability boundaries of the applications themselves. In this paper, we systematically analyzed the new development paradigm and defined the concept of the LLM app capability space. We also uncovered potential new risks beyond jailbreak that arise from ambiguous capability boundaries in real-world scenarios, namely, capability downgrade and upgrade. To evaluate the impact of these risks, we designed and implemented an LLM app capability evaluation framework, LLMApp-Eval. First, we collected application metadata across 4 platforms and conducted a cross-platform ecosystem analysis. Then, we evaluated the risks for 199 popular applications among 4 platforms and 6 open-source LLMs. We identified that 178 (89.45%) potentially affected applications, which can perform tasks from more than 15 scenarios or be malicious. We even found 17 applications in our study that executed malicious tasks directly, without applying any adversarial rewriting. Furthermore, our experiments also reveal a positive correlation between the quality of prompt design and application robustness. We found that well-designed prompts enhance security, while poorly designed ones can facilitate abuse. We hope our work inspires the community to focus on the real-world risks of LLM applications and foster the development of a more robust LLM application ecosystem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.08068v2">A Roadmap to Guide the Integration of LLMs in Hierarchical Planning</a></div>
    <div class="paper-meta">
      📅 2025-11-22
      | 💬 5 pages, 0 figures, to be published in the AAAI Workshop on Planning in the Era of LLMs ( https://llmforplanning.github.io )
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) are fostering their integration into several reasoning-related fields, including Automated Planning (AP). However, their integration into Hierarchical Planning (HP), a subfield of AP that leverages hierarchical knowledge to enhance planning performance, remains largely unexplored. In this preliminary work, we propose a roadmap to address this gap and harness the potential of LLMs for HP. To this end, we present a taxonomy of integration methods, exploring how LLMs can be utilized within the HP life cycle. Additionally, we provide a benchmark with a standardized dataset for evaluating the performance of future LLM-based HP approaches, and present initial results for a state-of-the-art HP planner and LLM planner. As expected, the latter exhibits limited performance (3\% correct plans, and none with a correct hierarchical decomposition) but serves as a valuable baseline for future approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18165v1">Towards a General Framework for HTN Modeling with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-22
      | 💬 10 pages, 5 figures, to be published in the Workshop on Planning in the Era of LLMs ( LM4Plan - https://llmforplanning.github.io ) and the Workshop on Hierarchical Planning ( HPlan - https://icaps25.icaps-conference.org/program/workshops/hplan/ ), both in the International Conference on Automated Planning and Scheduling (ICAPS) 2025
    </div>
    <details class="paper-abstract">
      The use of Large Language Models (LLMs) for generating Automated Planning (AP) models has been widely explored; however, their application to Hierarchical Planning (HP) is still far from reaching the level of sophistication observed in non-hierarchical architectures. In this work, we try to address this gap. We present two main contributions. First, we propose L2HP, an extension of L2P (a library to LLM-driven PDDL models generation) that support HP model generation and follows a design philosophy of generality and extensibility. Second, we apply our framework to perform experiments where we compare the modeling capabilities of LLMs for AP and HP. On the PlanBench dataset, results show that parsing success is limited but comparable in both settings (around 36\%), while syntactic validity is substantially lower in the hierarchical case (1\% vs. 20\% of instances). These findings underscore the unique challenges HP presents for LLMs, highlighting the need for further research to improve the quality of generated HP models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.11425v2">Tapas Are Free! Training-Free Adaptation of Programmatic Agents via LLM-Guided Program Synthesis in Dynamic Environments</a></div>
    <div class="paper-meta">
      📅 2025-11-22
      | 💬 Extended version of the paper accepted at AAAI-26 Oral to comply with the AAAI camera-ready requirements with minor revisions
    </div>
    <details class="paper-abstract">
      Autonomous agents in safety-critical applications must continuously adapt to dynamic conditions without compromising performance and reliability. This work introduces TAPA (Training-free Adaptation of Programmatic Agents), a novel framework that positions large language models (LLMs) as intelligent moderators of the symbolic action space. Unlike prior programmatic agents typically generate a monolithic policy program or rely on fixed symbolic action sets, TAPA synthesizes and adapts modular programs for individual high-level actions, referred to as logical primitives. By decoupling strategic intent from execution, TAPA enables meta-agents to operate over an abstract, interpretable action space while the LLM dynamically generates, composes, and refines symbolic programs tailored to each primitive. Extensive experiments across cybersecurity and swarm intelligence domains validate TAPA's effectiveness. In autonomous DDoS defense scenarios, TAPA achieves 77.7% network uptime while maintaining near-perfect detection accuracy in unknown dynamic environments. In swarm intelligence formation control under environmental and adversarial disturbances, TAPA consistently preserves consensus at runtime where baseline methods fail. This work promotes a paradigm shift for autonomous system design in evolving environments, from policy adaptation to dynamic action adaptation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.16466v4">Incalmo: An Autonomous LLM-assisted System for Red Teaming Multi-Host Networks</a></div>
    <div class="paper-meta">
      📅 2025-11-22
      | 💬 18 pages, 15 figures
    </div>
    <details class="paper-abstract">
      Security operators use red teams to simulate real attackers and proactively find defense gaps. In realistic enterprise settings, this involves executing multi-host network attacks spanning many "stepping stone" hosts. Unfortunately, red teams are expensive and entail significant expertise and effort. Given the promise of LLMs in CTF challenges, we first analyze if LLMs can autonomously execute multi-host red team exercises. We find that state-of-the-art LLM-assisted offense systems (e.g., PentestGPT, CyberSecEval3) with leading LLMs (e.g., Sonnet 4, Gemini 2.5 Pro) are unable to do so. Building on our observations in understanding the failure modes of state-of-the-art systems, we argue the need to improve the abstractions and interfaces for LLM-assisted red teaming. Based on this insight, we present the design and implementation of Incalmo, an LLM-assisted system for autonomously red teaming multi-host networks. Incalmo uses LLMs to plan red team exercises in terms of high-level declarative tasks that are executed by domain-specific task agents. Incalmo also uses auxiliary services to manage context and acquired assets. For our evaluation, we develop MHBench, a novel multi-host attack benchmark with 40 realistic emulated networks (from 22 to 50 hosts). We find that Incalmo successfully acquires critical assets (i.e., key hosts or data) in 37 out of 40 MHBench environments. In contrast, state-of-the-art LLM-assisted systems succeed in only 3 out of 40 environments. We show that Incalmo is efficient-successful attacks took 12-54 minutes and cost <$15 in LLM credits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.12509v3">Graph of Verification: Structured Verification of LLM Reasoning with Directed Acyclic Graphs</a></div>
    <div class="paper-meta">
      📅 2025-11-22
      | 💬 Accepted to AAAI 2026
    </div>
    <details class="paper-abstract">
      Verifying the complex and multi-step reasoning of Large Language Models (LLMs) is a critical challenge, as holistic methods often overlook localized flaws. Step-by-step validation is a promising alternative, yet existing methods are often rigid. They struggle to adapt to diverse reasoning structures, from formal proofs to informal natural language narratives. To address this adaptability gap, we propose the Graph of Verification (GoV), a novel framework for adaptable and multi-granular verification. GoV's core innovation is its flexible "node block" architecture. This mechanism allows GoV to adaptively adjust its verification granularity--from atomic steps for formal tasks to entire paragraphs for natural language--to match the native structure of the reasoning process. This flexibility allows GoV to resolve the fundamental trade-off between verification precision and robustness. Experiments on both well-structured and loosely-structured benchmarks demonstrate GoV's versatility. The results show that GoV's adaptive approach significantly outperforms both holistic baselines and other state-of-the-art decomposition-based methods, establishing a new standard for training-free reasoning verification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18098v1">Towards Harnessing the Power of LLMs for ABAC Policy Mining</a></div>
    <div class="paper-meta">
      📅 2025-11-22
    </div>
    <details class="paper-abstract">
      This paper presents an empirical investigation into the capabilities of Large Language Models (LLMs) to perform automated Attribute-based Access Control (ABAC) policy mining. While ABAC provides fine-grained, context-aware access management, the increasing number and complexity of access policies can make their formulation and evaluation rather challenging. To address the task of synthesizing concise yet accurate policies, we evaluate the performance of some of the state-of-the-art LLMs, specifically Google Gemini (Flash and Pro) and OpenAI ChatGPT, as potential policy mining engines. An experimental framework was developed in Python to generate randomized access data parameterized by varying numbers of subjects, objects, and initial policy sets. The baseline policy sets, which govern permission decisions between subjects and objects, serve as the ground truth for comparison. Each LLM-generated policy was evaluated against the baseline policy using standard performance metrics. The results indicate that LLMs can effectively infer compact and valid ABAC policies for small-scale scenarios. However, as the system size increases, characterized by higher numbers of subjects and objects, LLM outputs exhibit declining accuracy and precision, coupled with significant increase in the size of policy generated, which is beyond the optimal size. These findings highlight both the promise and limitations of current LLM architectures for scalable policy mining in access control domains. Future work will explore hybrid approaches that combine prompt optimization with classical rule mining algorithms to improve scalability and interpretability in complex ABAC environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.04440v2">StepFun-Formalizer: Unlocking the Autoformalization Potential of LLMs through Knowledge-Reasoning Fusion</a></div>
    <div class="paper-meta">
      📅 2025-11-22
      | 💬 AAAI 2026 Oral. Extended version with full appendix, 25 pages, 17 figures
    </div>
    <details class="paper-abstract">
      Autoformalization aims to translate natural-language mathematical statements into a formal language. While LLMs have accelerated progress in this area, existing methods still suffer from low accuracy. We identify two key abilities for effective autoformalization: comprehensive mastery of formal-language domain knowledge, and reasoning capability of natural language problem understanding and informal-formal alignment. Without the former, a model cannot identify the correct formal objects; without the latter, it struggles to interpret real-world contexts and map them precisely into formal expressions. To address these gaps, we introduce ThinkingF, a data synthesis and training pipeline that improves both abilities. First, we construct two datasets: one by distilling and selecting large-scale examples rich in formal knowledge, and another by generating informal-to-formal reasoning trajectories guided by expert-designed templates. We then apply SFT and RLVR with these datasets to further fuse and refine the two abilities. The resulting 7B and 32B models exhibit both comprehensive formal knowledge and strong informal-to-formal reasoning. Notably, StepFun-Formalizer-32B achieves SOTA BEq@1 scores of 40.5% on FormalMATH-Lite and 26.7% on ProverBench, surpassing all prior general-purpose and specialized models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.00413v2">Tree Training: Accelerating Agentic LLMs Training via Shared Prefix Reuse</a></div>
    <div class="paper-meta">
      📅 2025-11-22
    </div>
    <details class="paper-abstract">
      In agentic LLM scenarios, an agent's interaction process during a single rollout often exhibits branching behaviors. Due to memory retrieval and concurrent tool executions at certain decision points, the token trajectory of one task evolves into a tree-like structure rather than a linear sequence. However, current training pipelines decompose such tree-structured trajectories into separate linear segments, treating each branch as an independent sequence. As a result, shared prefixes across these branches are repeatedly recomputed during both forward and backward passes. To address this inefficiency, we propose Tree Training, a paradigm that computes each shared prefix only once and reuses its intermediate results across related branches during both forward and backward passes, substantially improving computation efficiency in large-scale agentic training. This is achieved via (i) Tree Packing, which efficiently reuses shared computations across trajectories, and (ii) Gradient Restoration, which ensures correct gradient propagation across reused prefixes. Experiments on multiple open-source models demonstrate up to 3.9x reduction in total training time, enabling more efficient agentic LLM SFT and RL training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18039v1">Curvature-Aware Safety Restoration In LLMs Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-11-22
      | 💬 19 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Fine-tuning Large Language Models (LLMs) for downstream tasks often compromises safety alignment, even when using parameter-efficient methods like LoRA. In this work, we uncover a notable property: fine-tuned models preserve the geometric structure of their loss landscapes concerning harmful content, regardless of the fine-tuning method employed. This suggests that safety behaviors are not erased but shifted to less influential regions of the parameter space. Building on this insight, we propose a curvature-aware alignment restoration method that leverages influence functions and second-order optimization to selectively increase loss on harmful inputs while preserving task performance. By navigating the shared geometry between base and fine-tuned models, our method discourages unsafe outputs while preserving task-relevant performance, avoiding full reversion and enabling precise, low-impact updates. Extensive evaluations across multiple model families and adversarial settings show that our approach efficiently reduces harmful responses while maintaining or even improving utility and few-shot learning performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18038v1">MASTEST: A LLM-Based Multi-Agent System For RESTful API Tests</a></div>
    <div class="paper-meta">
      📅 2025-11-22
      | 💬 14 Page of main text plus 4 pages of appendix
    </div>
    <details class="paper-abstract">
      Testing RESTful API is increasingly important in quality assurance of cloud-native applications. Recent advances in machine learning (ML) techniques have demonstrated that various testing activities can be performed automatically by large language models (LLMs) with reasonable accuracy. This paper develops a multi-agent system called MASTEST that combines LLM-based and programmed agents to form a complete tool chain that covers the whole workflow of API test starting from generating unit and system test scenarios from API specification in the OpenAPI Swagger format, to generating of Pytest test scripts, executing test scripts to interact with web services, to analysing web service response messages to determine test correctness and calculate test coverage. The system also supports the incorporation of human testers in reviewing and correcting LLM generated test artefacts to ensure the quality of testing activities. MASTEST system is evaluated on two LLMs, GPT-4o and DeepSeek V3.1 Reasoner with five public APIs. The performances of LLMs on various testing activities are measured by a wide range of metrics, including unit and system test scenario coverage and API operation coverage for the quality of generated test scenarios, data type correctness, status code coverage and script syntax correctness for the quality of LLM generated test scripts, as well as bug detection ability and usability of LLM generated test scenarios and scripts. Experiment results demonstrated that both DeepSeek and GPT-4o achieved a high overall performance. DeepSeek excels in data type correctness and status code detection, while GPT-4o performs best in API operation coverage. For both models, LLM generated test scripts maintained 100\% syntax correctness and only required minimal manual edits for semantic correctness. These findings indicate the effectiveness and feasibility of MASTEST.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17990v1">How Far Can LLMs Emulate Human Behavior?: A Strategic Analysis via the Buy-and-Sell Negotiation Game</a></div>
    <div class="paper-meta">
      📅 2025-11-22
    </div>
    <details class="paper-abstract">
      With the rapid advancement of Large Language Models (LLMs), recent studies have drawn attention to their potential for handling not only simple question-answer tasks but also more complex conversational abilities and performing human-like behavioral imitations. In particular, there is considerable interest in how accurately LLMs can reproduce real human emotions and behaviors, as well as whether such reproductions can function effectively in real-world scenarios. However, existing benchmarks focus primarily on knowledge-based assessment and thus fall short of sufficiently reflecting social interactions and strategic dialogue capabilities. To address these limitations, this work proposes a methodology to quantitatively evaluate the human emotional and behavioral imitation and strategic decision-making capabilities of LLMs by employing a Buy and Sell negotiation simulation. Specifically, we assign different personas to multiple LLMs and conduct negotiations between a Buyer and a Seller, comprehensively analyzing outcomes such as win rates, transaction prices, and SHAP values. Our experimental results show that models with higher existing benchmark scores tend to achieve better negotiation performance overall, although some models exhibit diminished performance in scenarios emphasizing emotional or social contexts. Moreover, competitive and cunning traits prove more advantageous for negotiation outcomes than altruistic and cooperative traits, suggesting that the assigned persona can lead to significant variations in negotiation strategies and results. Consequently, this study introduces a new evaluation approach for LLMs' social behavior imitation and dialogue strategies, and demonstrates how negotiation simulations can serve as a meaningful complementary metric to measure real-world interaction capabilities-an aspect often overlooked in existing benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.15018v2">Using tournaments to calculate AUROC for zero-shot classification with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-22
      | 💬 The 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP 2025, Findings). The code is available at: https://github.com/Machine-Learning-for-Medical-Language/cnlp_llm
    </div>
    <details class="paper-abstract">
      Large language models perform surprisingly well on many zero-shot classification tasks, but are difficult to fairly compare to supervised classifiers due to the lack of a modifiable decision boundary. In this work, we propose and evaluate a method that transforms binary classification tasks into pairwise comparisons between instances within a dataset, using LLMs to produce relative rankings of those instances. Repeated pairwise comparisons can be used to score instances using the Elo rating system (used in chess and other competitions), inducing a confidence ordering over instances in a dataset. We evaluate scheduling algorithms for their ability to minimize comparisons, and show that our proposed algorithm leads to improved classification performance, while also providing more information than traditional zero-shot classification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.18811v2">DCIS: Efficient Length Extrapolation of LLMs via Divide-and-Conquer Scaling Factor Search</a></div>
    <div class="paper-meta">
      📅 2025-11-22
      | 💬 EMNLP 2025 Main
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) based on the Transformer architecture usually have their context length limited due to the high training cost. Recent advancements extend the context window by adjusting the scaling factors of RoPE and fine-tuning. However, suboptimal initialization of these factors results in increased fine-tuning costs and reduced performance at target length. To address these challenges, we propose a novel RoPE-based fine-tuning framework that diverges from conventional scaling factors search. Specifically, we present a \textbf{D}ivide-and-\textbf{C}onquer \textbf{I}ncremental \textbf{S}earch (DCIS) algorithm that strategically determines the better scaling factors. Further fine-tuning with the identified scaling factors effectively extends the context window of LLMs. Empirical results demonstrate that our methodology not only mitigates performance decay at extended target lengths but also allows the model to fine-tune on short contexts and generalize to long contexts, thereby reducing the cost of fine-tuning. The scaling factors obtained through DCIS can even perform effectively without fine-tuning. Further analysis of the search space reveals that DCIS achieves twice the search efficiency compared to other methods. We also examine the impact of the non-strictly increasing scaling factors utilized in DCIS and evaluate the general capabilities of LLMs across various context lengths.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17947v1">Leveraging Evidence-Guided LLMs to Enhance Trustworthy Depression Diagnosis</a></div>
    <div class="paper-meta">
      📅 2025-11-22
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) show promise in automating clinical diagnosis, yet their non-transparent decision-making and limited alignment with diagnostic standards hinder trust and clinical adoption. We address this challenge by proposing a two-stage diagnostic framework that enhances transparency, trustworthiness, and reliability. First, we introduce Evidence-Guided Diagnostic Reasoning (EGDR), which guides LLMs to generate structured diagnostic hypotheses by interleaving evidence extraction with logical reasoning grounded in DSM-5 criteria. Second, we propose a Diagnosis Confidence Scoring (DCS) module that evaluates the factual accuracy and logical consistency of generated diagnoses through two interpretable metrics: the Knowledge Attribution Score (KAS) and the Logic Consistency Score (LCS). Evaluated on the D4 dataset with pseudo-labels, EGDR outperforms direct in-context prompting and Chain-of-Thought (CoT) across five LLMs. For instance, on OpenBioLLM, EGDR improves accuracy from 0.31 (Direct) to 0.76 and increases DCS from 0.50 to 0.67. On MedLlama, DCS rises from 0.58 (CoT) to 0.77. Overall, EGDR yields up to +45% accuracy and +36% DCS gains over baseline methods, offering a clinically grounded, interpretable foundation for trustworthy AI-assisted diagnosis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19480v1">Exploiting the Experts: Unauthorized Compression in MoE-LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-22
    </div>
    <details class="paper-abstract">
      Mixture-of-Experts (MoE) architectures are increasingly adopted in large language models (LLMs) for their scalability and efficiency. However, their modular structure introduces a unique vulnerability: adversaries can attempt to compress or repurpose models by pruning experts and cheaply fine-tuning the remainder, effectively bypassing licensing and security constraints. In this paper, we systematically study the prunability of MoE-LLMs under task-specific usage. We first develop an expert attribution framework that identifies the subset of experts most responsible for a given task, then evaluate the performance trade-offs of pruning and re-aligning these experts using active learning-driven fine-tuning. Our findings reveal a critical knowledge loss--recovery trade-off: while certain experts can be isolated to retain task accuracy, significant degradation occurs without targeted re-alignment. Based on this analysis, we propose defense strategies that aim to make MoE models harder to compress and fine-tune without authorization, including entangled expert training and selective fine-tuning protocols that resist unauthorized adaptation. By positioning expert pruning as both a threat vector and a defense target, this work highlights the dual-use nature of MoE modularity and provides the first systematic evaluation framework for secure specialization of MoE-LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17467v1">PersonaAgent with GraphRAG: Community-Aware Knowledge Graphs for Personalized LLM</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      We propose a novel framework for persona-based language model system, motivated by the need for personalized AI agents that adapt to individual user preferences. In our approach, the agent embodies the user's "persona" (e.g. user profile or taste) and is powered by a large language model (LLM). To enable the agent to leverage rich contextual information, we introduce a Knowledge-Graph-enhanced Retrieval-Augmented Generation (Graph RAG) mechanism that constructs an LLM-derived graph index of relevant documents and summarizes communities of related information. Our framework generates personalized prompts by combining: (1) a summary of the user's historical behaviors and preferences extracted from the knowledge graph, and (2) relevant global interaction patterns identified through graph-based community detection. This dynamic prompt engineering approach allows the agent to maintain consistent persona-aligned behaviors while benefiting from collective knowledge. On the LaMP benchmark, our method improves news categorization F1 by 11.1%, movie tagging F1 by 56.1%, and reduces product rating MAE by 10.4% over prior methods. Our code is available at https://anonymous.4open.science/r/PersonaAgentwGraphRAG-DE6F
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.00086v2">Do LLMs produce texts with "human-like" lexical diversity?</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      The degree to which large language models (LLMs) produce writing that is truly human-like remains unclear despite the extensive empirical attention that this question has received. The present study addresses this question from the perspective of lexical diversity. Specifically, the study investigates patterns of lexical diversity in LLM-generated texts from four ChatGPT models (ChatGPT-3.5, ChatGPT-4, ChatGPT-o4 mini, and ChatGPT-4.5) in comparison with texts written by L1 and L2 English participants (n = 240) across four education levels. Six dimensions of lexical diversity were measured in each text: volume, abundance, variety-repetition, evenness, disparity, and dispersion. Results from one-way MANOVAs, one-way ANOVAs, and Support Vector Machines revealed that the ChatGPT-generated texts differed significantly from human-written texts for each variable, with ChatGPT-o4 mini and ChatGPT-4.5 differing the most. Within these two groups, ChatGPT-4.5 demonstrated higher levels of lexical diversity than older models despite producing fewer tokens. The human writers' lexical diversity did not differ across subgroups (i.e., education, language status). Altogether, the results indicate that ChatGPT models do not produce human-like texts in relation to lexical diversity, and the newer models produce less human-like text than older models. We discuss the implications of these results for language pedagogy and related applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17442v1">REMSA: An LLM Agent for Foundation Model Selection in Remote Sensing</a></div>
    <div class="paper-meta">
      📅 2025-11-21
      | 💬 Code and data available at https://github.com/be-chen/REMSA
    </div>
    <details class="paper-abstract">
      Foundation Models (FMs) are increasingly used in remote sensing (RS) for tasks such as environmental monitoring, disaster assessment, and land-use mapping. These models include unimodal vision encoders trained on a single data modality and multimodal architectures trained on combinations of SAR, multispectral, hyperspectral, and image-text data. They support diverse RS tasks including semantic segmentation, image classification, change detection, and visual question answering. However, selecting an appropriate remote sensing foundation model (RSFM) remains difficult due to scattered documentation, heterogeneous formats, and varied deployment constraints. We introduce the RSFM Database (RS-FMD), a structured resource covering over 150 RSFMs spanning multiple data modalities, resolutions, and learning paradigms. Built on RS-FMD, we present REMSA, the first LLM-based agent for automated RSFM selection from natural language queries. REMSA interprets user requirements, resolves missing constraints, ranks candidate models using in-context learning, and provides transparent justifications. We also propose a benchmark of 75 expert-verified RS query scenarios, producing 900 configurations under an expert-centered evaluation protocol. REMSA outperforms several baselines, including naive agents, dense retrieval, and unstructured RAG-based LLMs. It operates entirely on publicly available metadata and does not access private or sensitive data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.09439v3">Omni-R1: Do You Really Need Audio to Fine-Tune Your Audio LLM?</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      We propose Omni-R1 which fine-tunes a recent multi-modal LLM, Qwen2.5-Omni, on an audio question answering dataset with the reinforcement learning method GRPO. This leads to new State-of-the-Art performance on the recent MMAU and MMAR benchmarks. Omni-R1 achieves the highest accuracies on the sounds, music, speech, and overall average categories, both on the Test-mini and Test-full splits. To understand the performance improvement, we tested models both with and without audio and found that much of the performance improvement from GRPO could be attributed to better text-based reasoning. We also made a surprising discovery that fine-tuning without audio on a text-only dataset was effective at improving the audio-based performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17335v1">Robot Confirmation Generation and Action Planning Using Long-context Q-Former Integrated with Multimodal LLM</a></div>
    <div class="paper-meta">
      📅 2025-11-21
      | 💬 Accepted to ASRU 2025
    </div>
    <details class="paper-abstract">
      Human-robot collaboration towards a shared goal requires robots to understand human action and interaction with the surrounding environment. This paper focuses on human-robot interaction (HRI) based on human-robot dialogue that relies on the robot action confirmation and action step generation using multimodal scene understanding. The state-of-the-art approach uses multimodal transformers to generate robot action steps aligned with robot action confirmation from a single clip showing a task composed of multiple micro steps. Although actions towards a long-horizon task depend on each other throughout an entire video, the current approaches mainly focus on clip-level processing and do not leverage long-context information. This paper proposes a long-context Q-former incorporating left and right context dependency in full videos. Furthermore, this paper proposes a text-conditioning approach to feed text embeddings directly into the LLM decoder to mitigate the high abstraction of the information in text by Q-former. Experiments with the YouCook2 corpus show that the accuracy of confirmation generation is a major factor in the performance of action planning. Furthermore, we demonstrate that the long-context Q-former improves the confirmation and action planning by integrating VideoLLaMA3.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17308v1">SpatialGeo:Boosting Spatial Reasoning in Multimodal LLMs via Geometry-Semantics Fusion</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      Multimodal large language models (MLLMs) have achieved significant progress in image and language tasks due to the strong reasoning capability of large language models (LLMs). Nevertheless, most MLLMs suffer from limited spatial reasoning ability to interpret and infer spatial arrangements in three-dimensional space. In this work, we propose a novel vision encoder based on hierarchical fusion of geometry and semantics features, generating spatial-aware visual embedding and boosting the spatial grounding capability of MLLMs. Specifically, we first unveil that the spatial ambiguity shortcoming stems from the lossy embedding of the vision encoder utilized in most existing MLLMs (e.g., CLIP), restricted to instance-level semantic features. This motivates us to complement CLIP with the geometry features from vision-only self-supervised learning via a hierarchical adapter, enhancing the spatial awareness in the proposed SpatialGeo. The network is efficiently trained using pretrained LLaVA model and optimized with random feature dropping to avoid trivial solutions relying solely on the CLIP encoder. Experimental results show that SpatialGeo improves the accuracy in spatial reasoning tasks, enhancing state-of-the-art models by at least 8.0% in SpatialRGPT-Bench with approximately 50% less memory cost during inference. The source code is available via https://ricky-plus.github.io/SpatialGeoPages/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.04736v2">The promise and limits of LLMs in constructing proofs and hints for logic problems in intelligent tutoring systems</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      Intelligent tutoring systems have demonstrated effectiveness in teaching formal propositional logic proofs, but their reliance on template-based explanations limits their ability to provide personalized student feedback. While large language models (LLMs) offer promising capabilities for dynamic feedback generation, they risk producing hallucinations or pedagogically unsound explanations. We evaluated the stepwise accuracy of LLMs in constructing multi-step symbolic logic proofs, comparing six prompting techniques across four state-of-the-art LLMs on 358 propositional logic problems. Results show that DeepSeek-V3 achieved superior performance up to 86.7% accuracy on stepwise proof construction and excelled particularly in simpler rules. We further used the best-performing LLM to generate explanatory hints for 1,050 unique student problem-solving states from a logic ITS and evaluated them on 4 criteria with both an LLM grader and human expert ratings on a 20% sample. Our analysis finds that LLM-generated hints were 75% accurate and rated highly by human evaluators on consistency and clarity, but did not perform as well explaining why the hint was provided or its larger context. Our results demonstrate that LLMs may be used to augment tutoring systems with logic tutoring hints, but require additional modifications to ensure accuracy and pedagogical appropriateness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.13302v2">LLM one-shot style transfer for Authorship Attribution and Verification</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      Computational stylometry analyzes writing style through quantitative patterns in text, supporting applications from forensic tasks such as identity linking and plagiarism detection to literary attribution in the humanities. Supervised and contrastive approaches rely on data with spurious correlations and often confuse style with topic. Despite their natural use in AI-generated text detection, the CLM pre-training of modern LLMs has been scarcely leveraged for general authorship problems. We propose a novel unsupervised approach based on this extensive pre-training and the in-context learning capabilities of LLMs, employing the log-probabilities of an LLM to measure style transferability from one text to another. Our method significantly outperforms LLM prompting approaches of comparable scale and achieves higher accuracy than contrastively trained baselines when controlling for topical correlations. Moreover, performance scales fairly consistently with the size of the base model and, in the case of authorship verification, with an additional mechanism that increases test-time computation; enabling flexible trade-offs between computational cost and accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17262v1">SlsReuse: LLM-Powered Serverless Function Reuse</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      Serverless computing has rapidly emerged as a popular cloud computing paradigm. It enables developers to implement function-level tasks, i.e., serverless functions, without managing infrastructure. While reducing operational overhead, it poses challenges, especially for novice developers. Developing functions from scratch requires adapting to heterogeneous, platform-specific programming styles, making the process time-consuming and error-prone. Function reuse offers a promising solution to address these challenges. However, research on serverless computing lacks a dedicated approach for function recommendation. Existing techniques from traditional contexts remain insufficient due to the semantic gap between task descriptions and heterogeneous function implementations. Advances in large language models (LLMs), pre-trained on large-scale corpora, create opportunities to bridge this gap by aligning developer requirements with function semantics. This paper presents SlsReuse, the first LLM-powered framework for serverless function reuse. Specifically, SlsReuse first constructs a reusable function repository serving as a foundational knowledge base. Then, it learns unified semantic-enhanced representations of heterogeneous functions through effective prompt engineering with few-shot prompting, capturing implicit code intent, target platforms, programming languages, and cloud services. Finally, given a natural language task query, SlsReuse performs intent-aware discovery combined with a multi-level pruning strategy and similarity matching. We evaluate SlsReuse on a curated dataset of 110 task queries. Built on ChatGPT-4o, one of the most representative LLMs, SlsReuse achieves Recall@10 of 91.20%, exceeding the state-of-the-art baseline by 24.53 percentage points.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16224v2">Beyond Code Similarity: Benchmarking the Plausibility, Efficiency, and Complexity of LLM-Generated Smart Contracts</a></div>
    <div class="paper-meta">
      📅 2025-11-21
      | 💬 20 pages
    </div>
    <details class="paper-abstract">
      Smart Contracts are critical components of blockchain ecosystems, with Solidity as the dominant programming language. While LLMs excel at general-purpose code generation, the unique constraints of Smart Contracts, such as gas consumption, security, and determinism, raise open questions about the reliability of LLM-generated Solidity code. Existing studies lack a comprehensive evaluation of these critical functional and non-functional properties. We benchmark four state-of-the-art models under zero-shot and retrieval-augmented generation settings across 500 real-world functions. Our multi-faceted assessment employs code similarity metrics, semantic embeddings, automated test execution, gas profiling, and cognitive and cyclomatic complexity analysis. Results show that while LLMs produce code with high semantic similarity to real contracts, their functional correctness is low: only 20% to 26% of zero-shot generations behave identically to ground-truth implementations under testing. The generated code is consistently simpler, with significantly lower complexity and gas consumption, often due to omitted validation logic. Retrieval-Augmented Generation markedly improves performance, boosting functional correctness by up to 45% and yielding more concise and efficient code. Our findings reveal a significant gap between semantic similarity and functional plausibility in LLM-generated Smart Contracts. We conclude that while RAG is a powerful enhancer, achieving robust, production-ready code generation remains a substantial challenge, necessitating careful expert validation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2409.11393v3">LLM-Agent-UMF: LLM-based Agent Unified Modeling Framework for Seamless Design of Multi Active/Passive Core-Agent Architectures</a></div>
    <div class="paper-meta">
      📅 2025-11-21
      | 💬 39 pages, 19 figures, 3 tables. Published in Information Fusion, Volume 127, March 2026, 103865. Part of the special issue "Data Fusion Approaches in Data-Centric AI for Developing Trustworthy AI Systems"
    </div>
    <details class="paper-abstract">
      In an era where vast amounts of data are collected and processed from diverse sources, there is a growing demand for sophisticated AI systems capable of intelligently fusing and analyzing this information. To address these challenges, researchers have turned towards integrating tools into LLM-powered agents to enhance the overall information fusion process. However, the conjunction of these technologies and the proposed enhancements in several state-of-the-art works followed a non-unified software architecture, resulting in a lack of modularity and terminological inconsistencies among researchers. To address these issues, we propose a novel LLM-based Agent Unified Modeling Framework (LLM-Agent-UMF) that establishes a clear foundation for agent development from both functional and software architectural perspectives, developed and evaluated using the Architecture Tradeoff and Risk Analysis Framework (ATRAF). Our framework clearly distinguishes between the different components of an LLM-based agent, setting LLMs and tools apart from a new element, the core-agent, which plays the role of central coordinator. This pivotal entity comprises five modules: planning, memory, profile, action, and security -- the latter often neglected in previous works. By classifying core-agents into passive and active types based on their authoritative natures, we propose various multi-core agent architectures that combine unique characteristics of distinctive agents to tackle complex tasks more efficiently. We evaluate our framework by applying it to thirteen state-of-the-art agents, thereby demonstrating its alignment with their functionalities and clarifying overlooked architectural aspects. Moreover, we thoroughly assess five architecture variants of our framework by designing new agent architectures that combine characteristics of state-of-the-art agents to address specific goals. ...
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.04652v4">LLM Collaboration With Multi-Agent Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      A large amount of work has been done in Multi-Agent Systems (MAS) for modeling and solving problems with multiple interacting agents. However, most LLMs are pretrained independently and not specifically optimized for coordination. Existing LLM fine-tuning frameworks rely on individual rewards, which require complex reward designs for each agent to encourage collaboration. To address these challenges, we model LLM collaboration as a cooperative Multi-Agent Reinforcement Learning (MARL) problem. We develop a multi-agent, multi-turn algorithm, Multi-Agent Group Relative Policy Optimization (MAGRPO), to solve it, building on current RL approaches for LLMs as well as MARL techniques. Our experiments on LLM writing and coding collaboration demonstrate that fine-tuning MAS with MAGRPO enables agents to generate high-quality responses efficiently through effective cooperation. Our approach opens the door to using other MARL methods for LLMs and highlights the associated challenges. Our code is available at https://github.com/OpenMLRL/CoMLRL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17220v1">Parrot: Persuasion and Agreement Robustness Rating of Output Truth -- A Sycophancy Robustness Benchmark for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      This study presents PARROT (Persuasion and Agreement Robustness Rating of Output Truth), a robustness focused framework designed to measure the degradation in accuracy that occurs under social pressure exerted on users through authority and persuasion in large language models (LLMs) the phenomenon of sycophancy (excessive conformity). PARROT (i) isolates causal effects by comparing the neutral version of the same question with an authoritatively false version using a double-blind evaluation, (ii) quantifies confidence shifts toward the correct and imposed false responses using log-likelihood-based calibration tracking, and (iii) systematically classifies failure modes (e.g., robust correct, sycophantic agreement, reinforced error, stubborn error, self-correction, etc.) using an eight-state behavioral taxonomy. We evaluated 22 models using 1,302 MMLU-style multiple-choice questions across 13 domains and domain-specific authority templates. Findings show marked heterogeneity: advanced models (e.g., GPT-5, GPT-4.1, Claude Sonnet 4.5) exhibit low "follow rates" ($\leq 11\%$, GPT-5: 4\%) and minimal accuracy loss, while older/smaller models show severe epistemic collapse (GPT-4: 80\%, Qwen 2.5-1.5B: 94\%). The danger is not limited to response changes; weak models reduce confidence in the correct response while increasing confidence in the imposed incorrect response. While international law and global knowledge at the domain level exhibit high fragility, elementary mathematics is relatively resilient. Consequently, we argue that the goal of "resistance to overfitting pressure" should be addressed as a primary objective alongside accuracy, harm avoidance, and privacy for safe deployment in the real world.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17208v1">A Simple Yet Strong Baseline for Long-Term Conversational Memory of LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-21
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      LLM-based conversational agents still struggle to maintain coherent, personalized interaction over many sessions: fixed context windows limit how much history can be kept in view, and most external memory approaches trade off between coarse retrieval over large chunks and fine-grained but fragmented views of the dialogue. Motivated by neo-Davidsonian event semantics, we propose an event-centric alternative that represents conversational history as short, event-like propositions which bundle together participants, temporal cues, and minimal local context, rather than as independent relation triples or opaque summaries. In contrast to work that aggressively compresses or forgets past content, our design aims to preserve information in a non-compressive form and make it more accessible, rather than more lossy. Concretely, we instruct an LLM to decompose each session into enriched elementary discourse units (EDUs) -- self-contained statements with normalized entities and source turn attributions -- and organize sessions, EDUs, and their arguments in a heterogeneous graph that supports associative recall. On top of this representation we build two simple retrieval-based variants that use dense similarity search and LLM filtering, with an optional graph-based propagation step to connect and aggregate evidence across related EDUs. Experiments on the LoCoMo and LongMemEval$_S$ benchmarks show that these event-centric memories match or surpass strong baselines, while operating with much shorter QA contexts. Our results suggest that structurally simple, event-level memory provides a principled and practical foundation for long-horizon conversational agents. Our code and data will be released at https://github.com/KevinSRR/EMem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.12396v4">LLM-CoT Enhanced Graph Neural Recommendation with Harmonized Group Policy Optimization</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      Graph neural networks (GNNs) have advanced recommender systems by modeling interaction relationships. However, existing graph-based recommenders rely on sparse ID features and do not fully exploit textual information, resulting in low information density within representations. Furthermore, graph contrastive learning faces challenges. Random negative sampling can introduce false negative samples, while fixed temperature coefficients cannot adapt to the heterogeneity of different nodes. In addition, current efforts to enhance recommendations with large language models (LLMs) have not fully utilized their Chain-of-Thought (CoT) reasoning capabilities to guide representation learning. To address these limitations, we introduces LGHRec (LLM-CoT Enhanced Graph Neural Recommendation with Harmonized Group Policy Optimization). This framework leverages the CoT reasoning ability of LLMs to generate semantic IDs, enriching reasoning processes and improving information density and semantic quality of representations. Moreover, we design a reinforcement learning algorithm, Harmonized Group Policy Optimization (HGPO), to optimize negative sampling strategies and temperature coefficients in contrastive learning. This approach enhances long-tail recommendation performance and ensures optimization consistency across different groups. Experimental results on three datasets demonstrate that LGHRec improves representation quality through semantic IDs generated by LLM's CoT reasoning and effectively boosts contrastive learning with HGPO. Our method outperforms several baseline models. The code is available at: https://anonymous.4open.science/r/LLM-Rec.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07318v2">When Bias Pretends to Be Truth: How Spurious Correlations Undermine Hallucination Detection in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      Despite substantial advances, large language models (LLMs) continue to exhibit hallucinations, generating plausible yet incorrect responses. In this paper, we highlight a critical yet previously underexplored class of hallucinations driven by spurious correlations -- superficial but statistically prominent associations between features (e.g., surnames) and attributes (e.g., nationality) present in the training data. We demonstrate that these spurious correlations induce hallucinations that are confidently generated, immune to model scaling, evade current detection methods, and persist even after refusal fine-tuning. Through systematically controlled synthetic experiments and empirical evaluations on state-of-the-art open-source and proprietary LLMs (including GPT-5), we show that existing hallucination detection methods, such as confidence-based filtering and inner-state probing, fundamentally fail in the presence of spurious correlations. Our theoretical analysis further elucidates why these statistical biases intrinsically undermine confidence-based detection techniques. Our findings thus emphasize the urgent need for new approaches explicitly designed to address hallucinations caused by spurious correlations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.24975v2">DiffTester: Accelerating Unit Test Generation for Diffusion LLMs via Repetitive Pattern</a></div>
    <div class="paper-meta">
      📅 2025-11-21
      | 💬 Update reference
    </div>
    <details class="paper-abstract">
      Software development relies heavily on extensive unit testing, which makes the efficiency of automated Unit Test Generation (UTG) particularly important. However, most existing LLMs generate test cases one token at a time in each forward pass, which leads to inefficient UTG. Recently, diffusion LLMs (dLLMs) have emerged, offering promising parallel generation capabilities and showing strong potential for efficient UTG. Despite this advantage, their application to UTG is still constrained by a clear trade-off between efficiency and test quality, since increasing the number of tokens generated in each step often causes a sharp decline in the quality of test cases. To overcome this limitation, we present DiffTester, an acceleration framework specifically tailored for dLLMs in UTG. The key idea of DiffTester is that unit tests targeting the same focal method often share repetitive structural patterns. By dynamically identifying these common patterns through abstract syntax tree analysis during generation, DiffTester adaptively increases the number of tokens produced at each step without compromising the quality of the output. To enable comprehensive evaluation, we extend the original TestEval benchmark, which was limited to Python, by introducing additional programming languages including Java and C++. Extensive experiments on three benchmarks with two representative models show that DiffTester delivers significant acceleration while preserving test coverage. Moreover, DiffTester generalizes well across different dLLMs and programming languages, providing a practical and scalable solution for efficient UTG in software development. Code and data are publicly available at https://github.com/wellbeingyang/DLM4UTG-open .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17124v1">A Counterfactual LLM Framework for Detecting Human Biases: A Case Study of Sex/Gender in Emergency Triage</a></div>
    <div class="paper-meta">
      📅 2025-11-21
      | 💬 Currently under review at npj Digital Medicine
    </div>
    <details class="paper-abstract">
      We present a novel, domain-agnostic counterfactual approach that uses Large Language Models (LLMs) to quantify gender disparities in human clinical decision-making. The method trains an LLM to emulate observed decisions, then evaluates counterfactual pairs in which only gender is flipped, estimating directional disparities while holding all other clinical factors constant. We study emergency triage, validating the approach on more than 150,000 admissions to the Bordeaux University Hospital (France) and replicating results on a subset of MIMIC-IV across a different language, population, and healthcare system. In the Bordeaux cohort, otherwise identical presentations were approximately 2.1% more likely to receive a lower-severity triage score when presented as female rather than male; scaled to national emergency volumes in France, this corresponds to more than 200,000 lower-severity assignments per year. Modality-specific analyses indicate that both explicit tabular gender indicators and implicit textual gender cues contribute to the disparity. Beyond emergency care, the approach supports bias audits in other settings (e.g., hiring, academic, and justice decisions), providing a scalable tool to detect and address inequities in real-world decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.23629v2">How LLMs Learn to Reason: A Complex Network Perspective</a></div>
    <div class="paper-meta">
      📅 2025-11-21
      | 💬 24 pages, 11 figures, 1 table, under review as a conference paper at ICLR 2026
    </div>
    <details class="paper-abstract">
      Training large language models with Reinforcement Learning with Verifiable Rewards (RLVR) exhibits a set of distinctive and puzzling behaviors that remain poorly understood, including a two-stage learning curve, a V-shaped response-length trajectory, and a pronounced vulnerability to catastrophic forgetting. In this work, we propose that these behaviors are emergent collective phenomena governed not by neural implementation details, but by the topological evolution of the latent reasoning graph in semantic space. By demonstrating a dynamical isomorphism between a 1.5B-parameter LLM and a minimal Concept Network Model (CoNet), we trace the causal source to the self-organization of a sparse concept web pinned to an average degree of two. This geometric perspective provides a unified physical explanation for the observed anomalies: the V-shaped trajectory tracks the evolution from parallel local skill optimization to global network integration; catastrophic forgetting stems from the topological disconnection of critical ``trunk'' edges; and policy collapse arises from the accumulation of sequential transitions at the web's leaf nodes, where broad exploration abruptly freezes into rigid, high-reward trajectories. Identifying a ``maximally frustrated state'' at the transition between learning stages, we propose Annealed-RLVR, a principled algorithm that injects a targeted SFT ``heating'' step to resolve this topological bottleneck. Experiments confirm that this theory-driven intervention outperforms standard RLVR on both in-distribution and out-of-distribution benchmarks (including Minerva and AIME). By recasting RLVR from black-box optimization into a predictable process of structural self-organization, our work provides a new physical intuition for engineering the emergent reasoning capabilities of future AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.12972v3">Aligning Vision to Language: Annotation-Free Multimodal Knowledge Graph Construction for Enhanced LLMs Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-21
      | 💬 14 pages, 7 figures, 6 tables; Accepted by ICCV 2025
    </div>
    <details class="paper-abstract">
      Multimodal reasoning in Large Language Models (LLMs) struggles with incomplete knowledge and hallucination artifacts, challenges that textual Knowledge Graphs (KGs) only partially mitigate due to their modality isolation. While Multimodal Knowledge Graphs (MMKGs) promise enhanced cross-modal understanding, their practical construction is impeded by semantic narrowness of manual text annotations and inherent noise in visual-semantic entity linkages. In this paper, we propose Vision-align-to-Language integrated Knowledge Graph (VaLiK), a novel approach for constructing MMKGs that enhances LLMs reasoning through cross-modal information supplementation. Specifically, we cascade pre-trained Vision-Language Models (VLMs) to align image features with text, transforming them into descriptions that encapsulate image-specific information. Furthermore, we developed a cross-modal similarity verification mechanism to quantify semantic consistency, effectively filtering out noise introduced during feature alignment. Even without manually annotated image captions, the refined descriptions alone suffice to construct the MMKG. Compared to conventional MMKGs construction paradigms, our approach achieves substantial storage efficiency gains while maintaining direct entity-to-image linkage capability. Experimental results on multimodal reasoning tasks demonstrate that LLMs augmented with VaLiK outperform previous state-of-the-art models. Our code is published at https://github.com/Wings-Of-Disaster/VaLiK.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17041v1">CLLMRec: LLM-powered Cognitive-Aware Concept Recommendation via Semantic Alignment and Prerequisite Knowledge Distillation</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      The growth of Massive Open Online Courses (MOOCs) presents significant challenges for personalized learning, where concept recommendation is crucial. Existing approaches typically rely on heterogeneous information networks or knowledge graphs to capture conceptual relationships, combined with knowledge tracing models to assess learners' cognitive states. However, these methods face significant limitations due to their dependence on high-quality structured knowledge graphs, which are often scarce in real-world educational scenarios. To address this fundamental challenge, this paper proposes CLLMRec, a novel framework that leverages Large Language Models through two synergistic technical pillars: Semantic Alignment and Prerequisite Knowledge Distillation. The Semantic Alignment component constructs a unified representation space by encoding unstructured textual descriptions of learners and concepts. The Prerequisite Knowledge Distillation paradigm employs a teacher-student architecture, where a large teacher LLM (implemented as the Prior Knowledge Aware Component) extracts conceptual prerequisite relationships from its internalized world knowledge and distills them into soft labels to train an efficient student ranker. Building upon these foundations, our framework incorporates a fine-ranking mechanism that explicitly models learners' real-time cognitive states through deep knowledge tracing, ensuring recommendations are both structurally sound and cognitively appropriate. Extensive experiments on two real-world MOOC datasets demonstrate that CLLMRec significantly outperforms existing baseline methods across multiple evaluation metrics, validating its effectiveness in generating truly cognitive-aware and personalized concept recommendations without relying on explicit structural priors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16193v2">Fast LLM Post-training via Decoupled and Best-of-N Speculation</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      Rollout dominates the training time in large language model (LLM) post-training, where the trained model is used to generate tokens given a batch of prompts. SpecActor achieves fast rollout with speculative decoding that deploys a fast path (e.g., a smaller model) to accelerate the unparallelizable generation, while the correctness is guaranteed by fast parallel verification of the outputs with the original model. SpecActor addresses two foundational challenges in speculative rollout by (1) a \emph{dynamic decoupled speculation} execution method that maximizes the GPU computational efficiency to realize speedup for large-batch execution -- a configuration common in training but unfriendly to speculative execution and (2) a \emph{dynamic Best-of-N speculation} method that selects and combines different drafting methods according to the rollout progress. It substantially improves the speculation accuracy even when the best drafting method is unknown a priori, meanwhile without requiring adding extra computation resources. {\sys} is {1.7}\,$\times$ faster than veRL in end-to-end training, and is {1.3--1.5}\,$\times$ faster compared to baselines with speculative decoding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15974v2">KRAL: Knowledge and Reasoning Augmented Learning for LLM-assisted Clinical Antimicrobial Therapy</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      Clinical antimicrobial therapy requires the dynamic integration of pathogen profiles,host factors, pharmacological properties of antimicrobials,and the severity of infection. This complexity imposes fundamental limitations on the applicability of Large Language Models (LLMs) in high-stakes clinical decision-making including knowledge gaps, data privacy concerns, high deployment costs, and limited reasoning capabilities. To address these challenges, we propose KRAL (Knowledge and Reasoning Augmented Learning), a low-cost, scalable, privacy-preserving paradigm that leverages teacher-model reasoning to automatically distill knowledge and reasoning trajectories via answer-to-question reverse generation, employs heuristic learning for semi-supervised data augmentation (reducing manual annotation requirements by approximately 80%), and utilizes agentic reinforcement learning to jointly enhance medical knowledge and reasoning while optimizing computational and memory efficiency. A hierarchical evaluation employing diverse teacher-model proxies reduces assessment costs, while modular interface design facilitates seamless system updates. Experimental results demonstrate that KRAL significantly outperforms traditional Retrieval-Augmented Generation (RAG) and Supervised Fine-Tuning (SFT) methods. It improves knowledge question-answering capability (Accuracy@1 on the external open-source benchmark MEDQA increased by 1.8% vs. SFT and 3.6% vs. RAG) and reasoning capability (Pass@1 on the external benchmark PUMCH Antimicrobial increased by 27% vs. SFT and 27.2% vs. RAG), achieved at about 20% of SFT's long-term training costs. This establishes KRAL as an effective solution for enhancing local LLMs' clinical diagnostic capabilities, enabling low-cost, high-safety deployment in complex medical decision support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16964v1">Optimizing PyTorch Inference with LLM-Based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      Maximizing performance on available GPU hardware is an ongoing challenge for modern AI inference systems. Traditional approaches include writing custom GPU kernels and using specialized model compilers to tune high-level code for specific GPU targets. Recent work shows that LLM-based multi-agent systems can effectively perform such tuning, often outperforming existing compilers and eliminating the need for manual kernel development. However, the dynamics of multi-agent systems for this task remain unexplored. In this work, we present a logical framework for comparing multi-agent PyTorch optimization systems. Our evaluation shows that exploit-heavy strategies perform best when paired with error-fixing agents, and that performance correlates with the granularity of optimization steps. The best implementation achieves an average 2.88x speedup on an H100 GPU across diverse tasks in KernelBench, a benchmark suite covering a range of machine learning architectures in PyTorch.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.17987v3">Reason2Attack: Jailbreaking Text-to-Image Models via LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-21
      | 💬 Noted that This paper includes model-generated content that may contain offensive or distressing material
    </div>
    <details class="paper-abstract">
      Text-to-Image(T2I) models typically deploy safety filters to prevent the generation of sensitive images. Unfortunately, recent jailbreaking attack methods manually design instructions for the LLM to generate adversarial prompts, which effectively bypass safety filters while producing sensitive images, exposing safety vulnerabilities of T2I models. However, due to the LLM's limited understanding of the T2I model and its safety filters, existing methods require numerous queries to achieve a successful attack, limiting their practical applicability. To address this issue, we propose Reason2Attack(R2A), which aims to enhance the LLM's reasoning capabilities in generating adversarial prompts by incorporating the jailbreaking attack into the post-training process of the LLM. Specifically, we first propose a CoT example synthesis pipeline based on Frame Semantics, which generates adversarial prompts by identifying related terms and corresponding context illustrations. Using CoT examples generated by the pipeline, we fine-tune the LLM to understand the reasoning path and format the output structure. Subsequently, we incorporate the jailbreaking attack task into the reinforcement learning process of the LLM and design an attack process reward that considers prompt length, prompt stealthiness, and prompt effectiveness, aiming to further enhance reasoning accuracy. Extensive experiments on various T2I models show that R2A achieves a better attack success ratio while requiring fewer queries than baselines. Moreover, our adversarial prompts demonstrate strong attack transferability across both open-source and commercial T2I models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.01265v2">AraFinNews: Arabic Financial Summarisation with Domain-Adapted LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-21
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      This paper examines how domain specificity affects abstractive summarisation of Arabic financial texts using large language models (LLMs). We present AraFinNews, the largest publicly available Arabic financial news dataset to date, comprising 212,500 article-headline pairs spanning almost a decade of reporting from October 2015 to July 2025. Developed as an Arabic counterpart to major English summarisation corpora such as CNN/DailyMail, AraFinNews offers a strong benchmark for assessing domain-focused language understanding and generation in financial contexts. Using this resource, we evaluate transformer-based models, including mT5, AraT5 and the domain-adapted FinAraT5, to investigate how financial-domain pretraining influences accuracy, numerical reliability and stylistic alignment with professional reporting. The results show that domain-adapted models produce more coherent summaries, particularly when handling quantitative and entity-centred information. These findings underscore the value of domain-specific adaptation for improving narrative fluency in Arabic financial summarisation. The dataset is freely available for non-commercial research at https://github.com/ArabicNLP-UK/AraFinNews.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15817v2">A Causal Perspective on Measuring, Explaining and Mitigating Smells in LLM-Generated Code</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have accelerated their adoption in software engineering contexts. However, concerns persist about the structural quality of the code they produce. In particular, LLMs often replicate poor coding practices, introducing code smells (i.e., patterns that hinder readability, maintainability, or design integrity). Although prior research has examined the detection or repair of smells, we still lack a clear understanding of how and when these issues emerge in generated code. This paper addresses this gap by systematically measuring, explaining and mitigating smell propensity in LLM-generated code. We build on the Propensity Smelly Score (PSC), a probabilistic metric that estimates the likelihood of generating particular smell types, and establish its robustness as a signal of structural quality. Using PSC as an instrument for causal analysis, we identify how generation strategy, model size, model architecture and prompt formulation shape the structural properties of generated code. Our findings show that prompt design and architectural choices play a decisive role in smell propensity and motivate practical mitigation strategies that reduce its occurrence. A user study further demonstrates that PSC helps developers interpret model behavior and assess code quality, providing evidence that smell propensity signals can support human judgement. Taken together, our work lays the groundwork for integrating quality-aware assessments into the evaluation and deployment of LLMs for code.
    </details>
</div>
