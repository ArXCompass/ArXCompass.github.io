# llm - 2025_10

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
- [Part 17](papers_17.md)
- [Part 18](papers_18.md)
- [Part 19](papers_19.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23350v1">Validating Formal Specifications with LLM-generated Test Cases</a></div>
    <div class="paper-meta">
      📅 2025-10-27
    </div>
    <details class="paper-abstract">
      Validation is a central activity when developing formal specifications. Similarly to coding, a possible validation technique is to define upfront test cases or scenarios that a future specification should satisfy or not. Unfortunately, specifying such test cases is burdensome and error prone, which could cause users to skip this validation task. This paper reports the results of an empirical evaluation of using pre-trained large language models (LLMs) to automate the generation of test cases from natural language requirements. In particular, we focus on test cases for structural requirements of simple domain models formalized in the Alloy specification language. Our evaluation focuses on the state-of-art GPT-5 model, but results from other closed- and open-source LLMs are also reported. The results show that, in this context, GPT-5 is already quite effective at generating positive (and negative) test cases that are syntactically correct and that satisfy (or not) the given requirement, and that can detect many wrong specifications written by humans.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.20445v4">TrajAgent: An LLM-Agent Framework for Trajectory Modeling via Large-and-Small Model Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 Accepted by NeurIPS 2025, https://github.com/tsinghua-fib-lab/TrajAgent
    </div>
    <details class="paper-abstract">
      Trajectory modeling, which includes research on trajectory data pattern mining and future prediction, has widespread applications in areas such as life services, urban transportation, and public administration. Numerous methods have been proposed to address specific problems within trajectory modeling. However, the heterogeneity of data and the diversity of trajectory tasks make effective and reliable trajectory modeling an important yet highly challenging endeavor, even for domain experts. \fix In this paper, we propose \textit{TrajAgent}, a agent framework powered by large language models (LLMs), designed to facilitate robust and efficient trajectory modeling through automation modeling. This framework leverages and optimizes diverse specialized models to address various trajectory modeling tasks across different datasets effectively. \unfix~In \textit{TrajAgent}, we first develop \textit{UniEnv}, an execution environment with a unified data and model interface, to support the execution and training of various models. Building on \textit{UniEnv}, we introduce an agentic workflow designed for automatic trajectory modeling across various trajectory tasks and data. Furthermore, we introduce collaborative learning schema between LLM-based agents and small speciallized models, to enhance the performance of the whole framework effectively. Extensive experiments on four tasks using four real-world datasets demonstrate the effectiveness of \textit{TrajAgent} in automated trajectory modeling, achieving a performance improvement of \fix 2.38\%-69.91\% \unfix over baseline methods. The codes and data can be accessed via https://github.com/tsinghua-fib-lab/TrajAgent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.20075v3">LLMs can hide text in other text of the same length</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 21 pages, main paper 9 pages
    </div>
    <details class="paper-abstract">
      A meaningful text can be hidden inside another, completely different yet still coherent and plausible, text of the same length. For example, a tweet containing a harsh political critique could be embedded in a tweet that celebrates the same political leader, or an ordinary product review could conceal a secret manuscript. This uncanny state of affairs is now possible thanks to Large Language Models, and in this paper we present a simple and efficient protocol to achieve it. We show that even modest 8-billion-parameter open-source LLMs are sufficient to obtain high-quality results, and a message as long as this abstract can be encoded and decoded locally on a laptop in seconds. The existence of such a protocol demonstrates a radical decoupling of text from authorial intent, further eroding trust in written communication, already shaken by the rise of LLM chatbots. We illustrate this with a concrete scenario: a company could covertly deploy an unfiltered LLM by encoding its answers within the compliant responses of a safe model. This possibility raises urgent questions for AI safety and challenges our understanding of what it means for a Large Language Model to know something.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23313v1">Network Intrusion Detection: Evolution from Conventional Approaches to LLM Collaboration and Emerging Risks</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 28 pages,1 figure, 204 references
    </div>
    <details class="paper-abstract">
      This survey systematizes the evolution of network intrusion detection systems (NIDS), from conventional methods such as signature-based and neural network (NN)-based approaches to recent integrations with large language models (LLMs). It clearly and concisely summarizes the current status, strengths, and limitations of conventional techniques, and explores the practical benefits of integrating LLMs into NIDS. Recent research on the application of LLMs to NIDS in diverse environments is reviewed, including conventional network infrastructures, autonomous vehicle environments and IoT environments. From this survey, readers will learn that: 1) the earliest methods, signature-based IDSs, continue to make significant contributions to modern systems, despite their well-known weaknesses; 2) NN-based detection, although considered promising and under development for more than two decades, and despite numerous related approaches, still faces significant challenges in practical deployment; 3) LLMs are useful for NIDS in many cases, and a number of related approaches have been proposed; however, they still face significant challenges in practical applications. Moreover, they can even be exploited as offensive tools, such as for generating malware, crafting phishing messages, or launching cyberattacks. Recently, several studies have been proposed to address these challenges, which are also reviewed in this survey; and 4) strategies for constructing domain-specific LLMs have been proposed and are outlined in this survey, as it is nearly impossible to train a NIDS-specific LLM from scratch.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19209v2">MOOSE-Chem2: Exploring LLM Limits in Fine-Grained Scientific Hypothesis Discovery via Hierarchical Search</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown promise in automating scientific hypothesis generation, yet existing approaches primarily yield coarse-grained hypotheses lacking critical methodological and experimental details. We introduce and formally define the new task of fine-grained scientific hypothesis discovery, which entails generating detailed, experimentally actionable hypotheses from coarse initial research directions. We frame this as a combinatorial optimization problem and investigate the upper limits of LLMs' capacity to solve it when maximally leveraged. Specifically, we explore four foundational questions: (1) how to best harness an LLM's internal heuristics to formulate the fine-grained hypothesis it itself would judge as the most promising among all the possible hypotheses it might generate, based on its own internal scoring-thus defining a latent reward landscape over the hypothesis space; (2) whether such LLM-judged better hypotheses exhibit stronger alignment with ground-truth hypotheses; (3) whether shaping the reward landscape using an ensemble of diverse LLMs of similar capacity yields better outcomes than defining it with repeated instances of the strongest LLM among them; and (4) whether an ensemble of identical LLMs provides a more reliable reward landscape than a single LLM. To address these questions, we propose a hierarchical search method that incrementally proposes and integrates details into the hypothesis, progressing from general concepts to specific experimental configurations. We show that this hierarchical process smooths the reward landscape and enables more effective optimization. Empirical evaluations on a new benchmark of expert-annotated fine-grained hypotheses from recent literature show that our method consistently outperforms strong baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08338v3">LLMs Reproduce Human Purchase Intent via Semantic Similarity Elicitation of Likert Ratings</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 28 pages, 35 figures
    </div>
    <details class="paper-abstract">
      Consumer research costs companies billions annually yet suffers from panel biases and limited scale. Large language models (LLMs) offer an alternative by simulating synthetic consumers, but produce unrealistic response distributions when asked directly for numerical ratings. We present semantic similarity rating (SSR), a method that elicits textual responses from LLMs and maps these to Likert distributions using embedding similarity to reference statements. Testing on an extensive dataset comprising 57 personal care product surveys conducted by a leading corporation in that market (9,300 human responses), SSR achieves 90% of human test-retest reliability while maintaining realistic response distributions (KS similarity > 0.85). Additionally, these synthetic respondents provide rich qualitative feedback explaining their ratings. This framework enables scalable consumer research simulations while preserving traditional survey metrics and interpretability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07967v3">Code Digital Twin: Empowering LLMs with Tacit Knowledge for Complex Software Development</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 A vision paper that will be continuously updated
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have demonstrated strong capabilities in software engineering tasks, raising expectations of revolutionary productivity gains. However, enterprise software development is largely driven by incremental evolution, where challenges extend far beyond routine coding and depend critically on tacit knowledge, including design decisions at different levels and historical trade-offs. To achieve effective AI-powered support for complex software development, we should align emerging AI capabilities with the practical realities of enterprise development. To this end, we systematically identify challenges from both software and LLM perspectives. Alongside these challenges, we outline opportunities where AI and structured knowledge frameworks can enhance decision-making in tasks such as issue localization and impact analysis. To address these needs, we propose the Code Digital Twin, a living framework that models both the physical and conceptual layers of software, preserves tacit knowledge, and co-evolves with the codebase. By integrating hybrid knowledge representations, multi-stage extraction pipelines, incremental updates, LLM-empowered applications, and human-in-the-loop feedback, the Code Digital Twin transforms fragmented knowledge into explicit and actionable representations. Our vision positions it as a bridge between AI advancements and enterprise software realities, providing a concrete roadmap toward sustainable, intelligent, and resilient development and evolution of ultra-complex systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16395v2">Code Digital Twin: Empowering LLMs with Tacit Knowledge for Complex Software Development</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 This should be a replacement of another article (arXiv:2503.07967), I mis-submitted it as a new article
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have demonstrated strong capabilities in software engineering tasks, raising expectations of revolutionary productivity gains. However, enterprise software development is largely driven by incremental evolution, where challenges extend far beyond routine coding and depend critically on tacit knowledge, including design decisions at different levels and historical trade-offs. To achieve effective AI-powered support for complex software development, we should align emerging AI capabilities with the practical realities of enterprise development. To this end, we systematically identify challenges from both software and LLM perspectives. Alongside these challenges, we outline opportunities where AI and structured knowledge frameworks can enhance decision-making in tasks such as issue localization and impact analysis. To address these needs, we propose the Code Digital Twin, a living framework that models both the physical and conceptual layers of software, preserves tacit knowledge, and co-evolves with the codebase. By integrating hybrid knowledge representations, multi-stage extraction pipelines, incremental updates, LLM-empowered applications, and human-in-the-loop feedback, the Code Digital Twin transforms fragmented knowledge into explicit and actionable representations. Our vision positions it as a bridge between AI advancements and enterprise software realities, providing a concrete roadmap toward sustainable, intelligent, and resilient development and evolution of ultra-complex systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.21034v2">Input Matters: Evaluating Input Structure's Impact on LLM Summaries of Sports Play-by-Play</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 Accepted at INLG 2025
    </div>
    <details class="paper-abstract">
      A major concern when deploying LLMs in accuracy-critical domains such as sports reporting is that the generated text may not faithfully reflect the input data. We quantify how input structure affects hallucinations and other factual errors in LLM-generated summaries of NBA play-by-play data, across three formats: row-structured, JSON and unstructured. We manually annotated 3,312 factual errors across 180 game summaries produced by two models, Llama-3.1-70B and Qwen2.5-72B. Input structure has a strong effect: JSON input reduces error rates by 69% for Llama and 65% for Qwen compared to unstructured input, while row-structured input reduces errors by 54% for Llama and 51% for Qwen. A two-way repeated measures ANOVA shows that input structure accounts for over 80% of the variance in error rates, with Tukey HSD post hoc tests confirming statistically significant differences between all input formats.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.21339v2">Multi-turn Training with Basic Human Feedback Helps Little on LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-27
    </div>
    <details class="paper-abstract">
      The reasoning capabilities of Large Language Models (LLMs) are typically developed through the single-turn reinforcement learning, whereas real-world applications often involve multi-turn interactions with human feedback, leading to a potential mismatch between training and deployment conditions. In this work, we study whether multi-turn training with human feedback is necessary for reasoning tasks. We compare conventional single-turn training with three multi-turn strategies and reach contrary conclusions to previous research. We find that models trained in a single-turn setting generalize effectively to both single- and multi-turn evaluations, while models trained with multi-turn strategies exhibit a significant degradation in single-turn reasoning performance. These results suggest that for tasks with complete information, robust single-turn training remains more effective and reliable, as multi-turn training with basic feedback provides limited benefits and can even degrade reasoning capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15301v2">Cohort Discovery: A Survey on LLM-Assisted Clinical Trial Recruitment</a></div>
    <div class="paper-meta">
      📅 2025-10-27
    </div>
    <details class="paper-abstract">
      Recent advances in LLMs have greatly improved general-domain NLP tasks. Yet, their adoption in critical domains, such as clinical trial recruitment, remains limited. As trials are designed in natural language and patient data is represented as both structured and unstructured text, the task of matching trials and patients benefits from knowledge aggregation and reasoning abilities of LLMs. Classical approaches are trial-specific and LLMs with their ability to consolidate distributed knowledge hold the potential to build a more general solution. Yet recent applications of LLM-assisted methods rely on proprietary models and weak evaluation benchmarks. In this survey, we are the first to analyze the task of trial-patient matching and contextualize emerging LLM-based approaches in clinical trial recruitment. We critically examine existing benchmarks, approaches and evaluation frameworks, the challenges to adopting LLM technologies in clinical research and exciting future directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23208v1">Increasing LLM Coding Capabilities through Diverse Synthetic Coding Tasks</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 Presented at the 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop: The 4th Deep Learning for Code Workshop (DL4C)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown impressive promise in code generation, yet their progress remains limited by the shortage of large-scale datasets that are both diverse and well-aligned with human reasoning. Most existing resources pair problems with solutions, but omit the intermediate thought process that guides coding. To close this gap, we present a scalable synthetic data generation pipeline that produces nearly 800k instruction-reasoning-code-test quadruplets. Each sample combines a task, a step-by-step reasoning trace, a working solution, and executable tests, enabling models to learn not just the what but also the how of problem solving. Our pipeline combines four key components: curated contest problems, web-mined content filtered by relevance classifiers, data expansion guided by reasoning patterns, and multi-stage execution-based validation. A genetic mutation algorithm further increases task diversity while maintaining consistency between reasoning traces and code implementations. Our key finding is that fine-tuning LLMs on this dataset yields consistent improvements on coding benchmarks. Beyond raw accuracy, reasoning-aware data can substitute for model scaling, generalize across architectures, and outperform leading open-source alternatives under identical sample budgets. Our work establishes reasoning-centered synthetic data generation as an efficient approach for advancing coding capabilities in LLMs. We publish our dataset and generation pipeline to facilitate further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23190v1">Evaluation of Vision-LLMs in Surveillance Video</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 Accepted as poster in the NeurIPS 2025 Workshop on Space in Vision, Language, and Embodied AI
    </div>
    <details class="paper-abstract">
      The widespread use of cameras in our society has created an overwhelming amount of video data, far exceeding the capacity for human monitoring. This presents a critical challenge for public safety and security, as the timely detection of anomalous or criminal events is crucial for effective response and prevention. The ability for an embodied agent to recognize unexpected events is fundamentally tied to its capacity for spatial reasoning. This paper investigates the spatial reasoning of vision-language models (VLMs) by framing anomalous action recognition as a zero-shot, language-grounded task, addressing the embodied perception challenge of interpreting dynamic 3D scenes from sparse 2D video. Specifically, we investigate whether small, pre-trained vision--LLMs can act as spatially-grounded, zero-shot anomaly detectors by converting video into text descriptions and scoring labels via textual entailment. We evaluate four open models on UCF-Crime and RWF-2000 under prompting and privacy-preserving conditions. Few-shot exemplars can improve accuracy for some models, but may increase false positives, and privacy filters -- especially full-body GAN transforms -- introduce inconsistencies that degrade accuracy. These results chart where current vision--LLMs succeed (simple, spatially salient events) and where they falter (noisy spatial cues, identity obfuscation). Looking forward, we outline concrete paths to strengthen spatial grounding without task-specific training: structure-aware prompts, lightweight spatial memory across clips, scene-graph or 3D-pose priors during description, and privacy methods that preserve action-relevant geometry. This positions zero-shot, language-grounded pipelines as adaptable building blocks for embodied, real-world video understanding. Our implementation for evaluating VLMs is publicly available at: https://github.com/pascalbenschopTU/VLLM_AnomalyRecognition
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19674v2">SAGE: A Generic Framework for LLM Safety Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 Accepted to EMNLP 2025
    </div>
    <details class="paper-abstract">
      As Large Language Models are rapidly deployed across diverse applications from healthcare to financial advice, safety evaluation struggles to keep pace. Current benchmarks focus on single-turn interactions with generic policies, failing to capture the conversational dynamics of real-world usage and the application-specific harms that emerge in context. Such potential oversights can lead to harms that go unnoticed in standard safety benchmarks and other current evaluation methodologies. To address these needs for robust AI safety evaluation, we introduce SAGE (Safety AI Generic Evaluation), an automated modular framework designed for customized and dynamic harm evaluations. SAGE employs prompted adversarial agents with diverse personalities based on the Big Five model, enabling system-aware multi-turn conversations that adapt to target applications and harm policies. We evaluate seven state-of-the-art LLMs across three applications and harm policies. Multi-turn experiments show that harm increases with conversation length, model behavior varies significantly when exposed to different user personalities and scenarios, and some models minimize harm via high refusal rates that reduce usefulness. We also demonstrate policy sensitivity within a harm category where tightening a child-focused sexual policy substantially increases measured defects across applications. These results motivate adaptive, policy-aware, and context-specific testing for safer real-world deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23163v1">Beyond Direct Generation: A Decomposed Approach to Well-Crafted Screenwriting with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-27
    </div>
    <details class="paper-abstract">
      The screenplay serves as the foundation for television production, defining narrative structure, character development, and dialogue. While Large Language Models (LLMs) show great potential in creative writing, direct end-to-end generation approaches often fail to produce well-crafted screenplays. We argue this failure stems from forcing a single model to simultaneously master two disparate capabilities: creative narrative construction and rigid format adherence. The resulting outputs may mimic superficial style but lack the deep structural integrity and storytelling substance required for professional use. To enable LLMs to generate high-quality screenplays, we introduce Dual-Stage Refinement (DSR), a decomposed framework that decouples creative narrative generation from format conversion. The first stage transforms a brief outline into rich, novel-style prose. The second stage refines this narrative into a professionally formatted screenplay. This separation enables the model to specialize in one distinct capability at each stage. A key challenge in implementing DSR is the scarcity of paired outline-to-novel training data. We address this through hybrid data synthesis: reverse synthesis deconstructs existing screenplays into structured inputs, while forward synthesis leverages these inputs to generate high-quality narrative texts as training targets. Blind evaluations by professional screenwriters show that DSR achieves a 75% win rate against strong baselines like Gemini-2.5-Pro and reaches 82.7% of human-level performance. Our work demonstrates that decomposed generation architecture with tailored data synthesis effectively specializes LLMs in complex creative domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.21007v2">Can Confidence Estimates Decide When Chain-of-Thought Is Necessary for LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Chain-of-thought (CoT) prompting has emerged as a common technique for enhancing the reasoning abilities of large language models (LLMs). While extended reasoning can boost accuracy on complex tasks, it is often unnecessary and substantially increases token usage, limiting the practicality of reasoning models in many scenarios. Recent models, such as GPT-OSS and Qwen3, expose controls that enable users to adjust the length of CoT or determine whether it is used at all. Yet, it remains unclear when CoT should be used: on some tasks it improves performance, while on others it provides little benefit or even harms performance. We address this challenge with confidence-gated CoT, where a model invokes reasoning only when confidence in its direct answer is low. To this end, we present the first systematic study of training-free confidence estimation methods for CoT gating. Specifically, we evaluate four training-free confidence estimation methods and compare them to a random baseline and an oracle that always knows when CoT is needed. Through extensive experiments, we show that existing training-free confidence measures can reduce redundant CoT and outperform randomly invoked CoT. However, the utility of individual confidence measures is inconsistent, varying with both the dataset and the model, underscoring the difficulty of deploying confidence-gated CoT in practice. By analysing both strengths and failure modes, our study highlights the potential and limitations of current methods and paves the way toward more reliable adaptive gating of CoT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23127v1">Lost in Tokenization: Context as the Key to Unlocking Biomolecular Understanding in Scientific LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 36 pages, under review
    </div>
    <details class="paper-abstract">
      Scientific Large Language Models (Sci-LLMs) have emerged as a promising frontier for accelerating biological discovery. However, these models face a fundamental challenge when processing raw biomolecular sequences: the tokenization dilemma. Whether treating sequences as a specialized language, risking the loss of functional motif information, or as a separate modality, introducing formidable alignment challenges, current strategies fundamentally limit their reasoning capacity. We challenge this sequence-centric paradigm by positing that a more effective strategy is to provide Sci-LLMs with high-level structured context derived from established bioinformatics tools, thereby bypassing the need to interpret low-level noisy sequence data directly. Through a systematic comparison of leading Sci-LLMs on biological reasoning tasks, we tested three input modes: sequence-only, context-only, and a combination of both. Our findings are striking: the context-only approach consistently and substantially outperforms all other modes. Even more revealing, the inclusion of the raw sequence alongside its high-level context consistently degrades performance, indicating that raw sequences act as informational noise, even for models with specialized tokenization schemes. These results suggest that the primary strength of existing Sci-LLMs lies not in their nascent ability to interpret biomolecular syntax from scratch, but in their profound capacity for reasoning over structured, human-readable knowledge. Therefore, we argue for reframing Sci-LLMs not as sequence decoders, but as powerful reasoning engines over expert knowledge. This work lays the foundation for a new class of hybrid scientific AI agents, repositioning the developmental focus from direct sequence interpretation towards high-level knowledge synthesis. The code is available at github.com/opendatalab-raise-dev/CoKE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23101v1">Beyond Imprecise Distance Metrics: LLM-Predicted Target Call Stacks for Directed Greybox Fuzzing</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 Preprint, under submission
    </div>
    <details class="paper-abstract">
      Directed greybox fuzzing (DGF) aims to efficiently trigger bugs at specific target locations by prioritizing seeds whose execution paths are more likely to mutate into triggering target bugs. However, existing DGF approaches suffer from imprecise probability calculations due to their reliance on complex distance metrics derived from static analysis. The over-approximations inherent in static analysis cause a large number of irrelevant execution paths to be mistakenly considered to potentially mutate into triggering target bugs, significantly reducing fuzzing efficiency. We propose to replace static analysis-based distance metrics with precise call stack representations. Call stacks represent precise control flows, thereby avoiding false information in static analysis. We leverage large language models (LLMs) to predict vulnerability-triggering call stacks for guiding seed prioritization. Our approach constructs call graphs through static analysis to identify methods that can potentially reach target locations, then utilizes LLMs to predict the most likely call stack sequence that triggers the vulnerability. Seeds whose execution paths have higher overlap with the predicted call stack are prioritized for mutation. This is the first work to integrate LLMs into the core seed prioritization mechanism of DGF. We implement our approach and evaluate it against several state-of-the-art fuzzers. On a suite of real-world programs, our approach triggers vulnerabilities $1.86\times$ to $3.09\times$ faster compared to baselines. In addition, our approach identifies 10 new vulnerabilities and 2 incomplete fixes in the latest versions of programs used in our controlled experiments through directed patch testing, with 10 assigned CVE IDs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23081v1">A Survey on LLM Mid-training</a></div>
    <div class="paper-meta">
      📅 2025-10-27
    </div>
    <details class="paper-abstract">
      Recent advances in foundation models have highlighted the significant benefits of multi-stage training, with a particular emphasis on the emergence of mid-training as a vital stage that bridges pre-training and post-training. Mid-training is distinguished by its use of intermediate data and computational resources, systematically enhancing specified capabilities such as mathematics, coding, reasoning, and long-context extension, while maintaining foundational competencies. This survey provides a formal definition of mid-training for large language models (LLMs) and investigates optimization frameworks that encompass data curation, training strategies, and model architecture optimization. We analyze mainstream model implementations in the context of objective-driven interventions, illustrating how mid-training serves as a distinct and critical stage in the progressive development of LLM capabilities. By clarifying the unique contributions of mid-training, this survey offers a comprehensive taxonomy and actionable insights, supporting future research and innovation in the advancement of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23074v1">Fast-MIA: Efficient and Scalable Membership Inference for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-27
    </div>
    <details class="paper-abstract">
      We propose Fast-MIA (https://github.com/Nikkei/fast-mia), a Python library for efficiently evaluating membership inference attacks (MIA) against Large Language Models (LLMs). MIA against LLMs has emerged as a crucial challenge due to growing concerns over copyright, security, and data privacy, and has attracted increasing research attention. However, the progress of this research is significantly hindered by two main obstacles: (1) the high computational cost of inference in LLMs, and (2) the lack of standardized and maintained implementations of MIA methods, which makes large-scale empirical comparison difficult. To address these challenges, our library provides fast batch inference and includes implementations of representative MIA methods under a unified evaluation framework. This library supports easy implementation of reproducible benchmarks with simple configuration and extensibility. We release Fast-MIA as an open-source (Apache License 2.0) tool to support scalable and transparent research on LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14668v2">ContextAgent: Context-Aware Proactive LLM Agents with Open-World Sensory Perceptions</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have propelled intelligent agents from reactive responses to proactive support. While promising, existing proactive agents either rely exclusively on observations from enclosed environments (e.g., desktop UIs) with direct LLM inference or employ rule-based proactive notifications, leading to suboptimal user intent understanding and limited functionality for proactive service. In this paper, we introduce ContextAgent, the first context-aware proactive agent that incorporates extensive sensory contexts surrounding humans to enhance the proactivity of LLM agents. ContextAgent first extracts multi-dimensional contexts from massive sensory perceptions on wearables (e.g., video and audio) to understand user intentions. ContextAgent then leverages the sensory contexts and personas from historical data to predict the necessity for proactive services. When proactive assistance is needed, ContextAgent further automatically calls the necessary tools to assist users unobtrusively. To evaluate this new task, we curate ContextAgentBench, the first benchmark for evaluating context-aware proactive LLM agents, covering 1,000 samples across nine daily scenarios and twenty tools. Experiments on ContextAgentBench show that ContextAgent outperforms baselines by achieving up to 8.5% and 6.0% higher accuracy in proactive predictions and tool calling, respectively. We hope our research can inspire the development of more advanced, human-centric, proactive AI assistants. The code and dataset are publicly available at https://github.com/openaiotlab/ContextAgent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23068v1">Checkstyle+: Reducing Technical Debt Through The Use of Linters with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 11 pages, 9 figures, tool link: https://github.com/ellacodee/CheckstylePlus
    </div>
    <details class="paper-abstract">
      Good code style improves program readability, maintainability, and collaboration, and is an integral component of software quality. Developers, however, often cut corners when following style rules, leading to the wide adoption of tools such as linters in professional software development projects. Traditional linters like Checkstyle operate using rigid, rule-based mechanisms that effectively detect many surface-level violations. However, in most programming languages, there is a subset of style rules that require a more nuanced understanding of code, and fall outside the scope of such static analysis. In this paper, we propose Checkstyle+, a hybrid approach that augments Checkstyle with large language model (LLM) capabilities, to identify style violations that elude the conventional rule-based analysis. Checkstyle+ is evaluated on a sample of 380 Java code files, drawn from a broader dataset of 30,800 real-world Java programs sourced from accepted Codeforces submissions. The results show that Checkstyle+ achieves superior performance over standard Checkstyle in detecting violations of the semantically nuanced rules.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23040v1">LLM Meets Diffusion: A Hybrid Framework for Crystal Material Generation</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Recent advances in generative modeling have shown significant promise in designing novel periodic crystal structures. Existing approaches typically rely on either large language models (LLMs) or equivariant denoising models, each with complementary strengths: LLMs excel at handling discrete atomic types but often struggle with continuous features such as atomic positions and lattice parameters, while denoising models are effective at modeling continuous variables but encounter difficulties in generating accurate atomic compositions. To bridge this gap, we propose CrysLLMGen, a hybrid framework that integrates an LLM with a diffusion model to leverage their complementary strengths for crystal material generation. During sampling, CrysLLMGen first employs a fine-tuned LLM to produce an intermediate representation of atom types, atomic coordinates, and lattice structure. While retaining the predicted atom types, it passes the atomic coordinates and lattice structure to a pre-trained equivariant diffusion model for refinement. Our framework outperforms state-of-the-art generative models across several benchmark tasks and datasets. Specifically, CrysLLMGen not only achieves a balanced performance in terms of structural and compositional validity but also generates more stable and novel materials compared to LLM-based and denoisingbased models Furthermore, CrysLLMGen exhibits strong conditional generation capabilities, effectively producing materials that satisfy user-defined constraints. Code is available at https://github.com/kdmsit/crysllmgen
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23038v1">Incentivizing Agentic Reasoning in LLM Judges via Tool-Integrated Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 Work in Progress
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely used as judges to evaluate response quality, providing a scalable alternative to human evaluation. However, most LLM judges operate solely on intrinsic text-based reasoning, limiting their ability to verify complex constraints or perform accurate computation. Motivated by the success of tool-integrated reasoning (TIR) in numerous tasks, we propose TIR-Judge, an end-to-end RL framework for training LLM judges that integrates a code executor for precise evaluation. TIR-Judge is built on three principles: (i) diverse training across verifiable and non-verifiable domains, (ii) flexible judgment formats (pointwise, pairwise, listwise), and (iii) iterative RL that bootstraps directly from the initial model without distillation. On seven public benchmarks, TIR-Judge surpasses strong reasoning-based judges by up to 6.4% (pointwise) and 7.7% (pairwise), and achieves listwise performance comparable to Claude-Opus-4 despite having only 8B parameters. Remarkably, TIR-Judge-Zero - trained entirely without distilled judge trajectories, matches the performance of distilled variants, demonstrating that tool-augmented judges can self-evolve through iterative reinforcement learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23032v1">P1GPT: a multi-agent LLM workflow module for multi-modal financial information analysis</a></div>
    <div class="paper-meta">
      📅 2025-10-27
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have enabled multi-agent reasoning systems capable of collaborative decision-making. However, in financial analysis, most frameworks remain narrowly focused on either isolated single-agent predictors or loosely connected analyst ensembles, and they lack a coherent reasoning workflow that unifies diverse data modalities. We introduce P1GPT, a layered multi-agent LLM framework for multi-modal financial information analysis and interpretable trading decision support. Unlike prior systems that emulate trading teams through role simulation, P1GPT implements a structured reasoning pipeline that systematically fuses technical, fundamental, and news-based insights through coordinated agent communication and integration-time synthesis. Backtesting on multi-modal datasets across major U.S. equities demonstrates that P1GPT achieves superior cumulative and risk-adjusted returns, maintains low drawdowns, and provides transparent causal rationales. These findings suggest that structured reasoning workflows, rather than agent role imitation, offer a scalable path toward explainable and trustworthy financial AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23008v1">From Prompt Optimization to Multi-Dimensional Credibility Evaluation: Enhancing Trustworthiness of Chinese LLM-Generated Liver MRI Reports</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 10 pages, 6 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated promising performance in generating diagnostic conclusions from imaging findings, thereby supporting radiology reporting, trainee education, and quality control. However, systematic guidance on how to optimize prompt design across different clinical contexts remains underexplored. Moreover, a comprehensive and standardized framework for assessing the trustworthiness of LLM-generated radiology reports is yet to be established. This study aims to enhance the trustworthiness of LLM-generated liver MRI reports by introducing a Multi-Dimensional Credibility Assessment (MDCA) framework and providing guidance on institution-specific prompt optimization. The proposed framework is applied to evaluate and compare the performance of several advanced LLMs, including Kimi-K2-Instruct-0905, Qwen3-235B-A22B-Instruct-2507, DeepSeek-V3, and ByteDance-Seed-OSS-36B-Instruct, using the SiliconFlow platform.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22986v1">CodeAD: Synthesize Code of Rules for Log-based Anomaly Detection with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-27
    </div>
    <details class="paper-abstract">
      Log-based anomaly detection (LogAD) is critical for maintaining the reliability and availability of large-scale online service systems. While machine learning, deep learning, and large language models (LLMs)-based methods have advanced the LogAD, they often suffer from limited interpretability, high inference costs, and extensive preprocessing requirements, limiting their practicality for real-time, high-volume log analysis. In contrast, rule-based systems offer efficiency and transparency, but require significant manual effort and are difficult to scale across diverse and evolving environments. In this paper, We present CodeAD, a novel framework that automatically synthesizes lightweight Python rule functions for LogAD using LLMs. CodeAD introduces a hierarchical clustering and anchor-grounded sampling strategy to construct representative contrastive log windows, enabling LLMs to discern discriminative anomaly patterns. To ensure robustness and generalizability, CodeAD employs an agentic workflow that iteratively generates, tests, repairs, and refines the rules until it meets correctness and abstraction requirements. The synthesized rules are interpretable, lightweight, and directly executable on raw logs, supporting efficient and transparent online anomaly detection. Our comprehensive experiments on three public datasets (BGL, Hadoop, Thunderbird) demonstrate that CodeAD achieves an average absolute improvement of 3.6% F1 score over the state-of-the-art baselines, while processing large datasets up to 4x faster and at a fraction of the cost (total LLM invocation cost under 4 USD per dataset). These results highlight CodeAD as a practical and scalable solution for online monitoring systems, enabling interpretable, efficient, and automated LogAD in real-world environment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22978v1">Reasoning About Reasoning: Towards Informed and Reflective Use of LLM Reasoning in HCI</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 14 pages, 3 figures, 1 table
    </div>
    <details class="paper-abstract">
      Reasoning is a distinctive human-like characteristic attributed to LLMs in HCI due to their ability to simulate various human-level tasks. However, this work argues that the reasoning behavior of LLMs in HCI is often decontextualized from the underlying mechanics and subjective decisions that condition the emergence and human interpretation of this behavior. Through a systematic survey of 258 CHI papers from 2020-2025 on LLMs, we discuss how HCI hardly perceives LLM reasoning as a product of sociotechnical orchestration and often references it as an object of application. We argue that such abstraction leads to oversimplification of reasoning methodologies from NLP/ML and results in a distortion of LLMs' empirically studied capabilities and (un)known limitations. Finally, drawing on literature from both NLP/ML and HCI, as a constructive step forward, we develop reflection prompts to support HCI practitioners engage with LLM reasoning in an informed and reflective way.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22977v1">The Reasoning Trap: How Enhancing LLM Reasoning Amplifies Tool Hallucination</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 18 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Enhancing the reasoning capabilities of Large Language Models (LLMs) is a key strategy for building Agents that "think then act." However, recent observations, like OpenAI's o3, suggest a paradox: stronger reasoning often coincides with increased hallucination, yet no prior work has systematically examined whether reasoning enhancement itself causes tool hallucination. To address this gap, we pose the central question: Does strengthening reasoning increase tool hallucination? To answer this, we introduce SimpleToolHalluBench, a diagnostic benchmark measuring tool hallucination in two failure modes: (i) no tool available, and (ii) only distractor tools available. Through controlled experiments, we establish three key findings. First, we demonstrate a causal relationship: progressively enhancing reasoning through RL increases tool hallucination proportionally with task performance gains. Second, this effect transcends overfitting - training on non-tool tasks (e.g., mathematics) still amplifies subsequent tool hallucination. Third, the effect is method-agnostic, appearing when reasoning is instilled via supervised fine-tuning and when it is merely elicited at inference by switching from direct answers to step-by-step thinking. We also evaluate mitigation strategies including Prompt Engineering and Direct Preference Optimization (DPO), revealing a fundamental reliability-capability trade-off: reducing hallucination consistently degrades utility. Mechanistically, Reasoning RL disproportionately collapses tool-reliability-related representations, and hallucinations surface as amplified divergences concentrated in late-layer residual streams. These findings reveal that current reasoning enhancement methods inherently amplify tool hallucination, highlighting the need for new training objectives that jointly optimize for capability and reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08221v3">Part I: Tricks or Traps? A Deep Dive into RL for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 26 pages, 21 figures
    </div>
    <details class="paper-abstract">
      Reinforcement learning for LLM reasoning has rapidly emerged as a prominent research area, marked by a significant surge in related studies on both algorithmic innovations and practical applications. Despite this progress, several critical challenges remain, including the absence of standardized guidelines for employing RL techniques and a fragmented understanding of their underlying mechanisms. Additionally, inconsistent experimental settings, variations in training data, and differences in model initialization have led to conflicting conclusions, obscuring the key characteristics of these techniques and creating confusion among practitioners when selecting appropriate techniques. This paper systematically reviews widely adopted RL techniques through rigorous reproductions and isolated evaluations within a unified open-source framework. We analyze the internal mechanisms, applicable scenarios, and core principles of each technique through fine-grained experiments, including datasets of varying difficulty, model sizes, and architectures. Based on these insights, we present clear guidelines for selecting RL techniques tailored to specific setups, and provide a reliable roadmap for practitioners navigating the RL for the LLM domain. Finally, we reveal that a minimalist combination of two techniques can unlock the learning capability of critic-free policies using vanilla PPO loss. The results demonstrate that our simple combination consistently improves performance, surpassing strategies like GRPO and DAPO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22968v1">Measuring Teaching with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-27
    </div>
    <details class="paper-abstract">
      Objective and scalable measurement of teaching quality is a persistent challenge in education. While Large Language Models (LLMs) offer potential, general-purpose models have struggled to reliably apply complex, authentic classroom observation instruments. This paper uses custom LLMs built on sentence-level embeddings, an architecture better suited for the long-form, interpretive nature of classroom transcripts than conventional subword tokenization. We systematically evaluate five different sentence embeddings under a data-efficient training regime designed to prevent overfitting. Our results demonstrate that these specialized models can achieve human-level and even super-human performance with expert human ratings above 0.65 and surpassing the average human-human rater correlation. Further, through analysis of annotation context windows, we find that more advanced models-those better aligned with human judgments-attribute a larger share of score variation to lesson-level features rather than isolated utterances, challenging the sufficiency of single-turn annotation paradigms. Finally, to assess external validity, we find that aggregate model scores align with teacher value-added measures, indicating they are capturing features relevant to student learning. However, this trend does not hold at the individual item level, suggesting that while the models learn useful signals, they have not yet achieved full generalization. This work establishes a viable and powerful new methodology for AI-driven instructional measurement, offering a path toward providing scalable, reliable, and valid feedback for educator development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22967v1">MAD-Fact: A Multi-Agent Debate Framework for Long-Form Factuality Evaluation in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 This article has been accepted by Frontiers of Computer Science (FCS)
    </div>
    <details class="paper-abstract">
      The widespread adoption of Large Language Models (LLMs) raises critical concerns about the factual accuracy of their outputs, especially in high-risk domains such as biomedicine, law, and education. Existing evaluation methods for short texts often fail on long-form content due to complex reasoning chains, intertwined perspectives, and cumulative information. To address this, we propose a systematic approach integrating large-scale long-form datasets, multi-agent verification mechanisms, and weighted evaluation metrics. We construct LongHalluQA, a Chinese long-form factuality dataset; and develop MAD-Fact, a debate-based multi-agent verification system. We introduce a fact importance hierarchy to capture the varying significance of claims in long-form texts. Experiments on two benchmarks show that larger LLMs generally maintain higher factual consistency, while domestic models excel on Chinese content. Our work provides a structured framework for evaluating and enhancing factual reliability in long-form LLM outputs, guiding their safe deployment in sensitive domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22963v1">CompressionAttack: Exploiting Prompt Compression as a New Attack Surface in LLM-Powered Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-27
    </div>
    <details class="paper-abstract">
      LLM-powered agents often use prompt compression to reduce inference costs, but this introduces a new security risk. Compression modules, which are optimized for efficiency rather than safety, can be manipulated by adversarial inputs, causing semantic drift and altering LLM behavior. This work identifies prompt compression as a novel attack surface and presents CompressionAttack, the first framework to exploit it. CompressionAttack includes two strategies: HardCom, which uses discrete adversarial edits for hard compression, and SoftCom, which performs latent-space perturbations for soft compression. Experiments on multiple LLMs show up to 80% attack success and 98% preference flips, while remaining highly stealthy and transferable. Case studies in VSCode Cline and Ollama confirm real-world impact, and current defenses prove ineffective, highlighting the need for stronger protections.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23988v3">LLM/Agent-as-Data-Analyst: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 31 page, 9 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) and agent techniques have brought a fundamental shift in the functionality and development paradigm of data analysis tasks (a.k.a LLM/Agent-as-Data-Analyst), demonstrating substantial impact across both academia and industry. In comparison with traditional rule or small-model based approaches, (agentic) LLMs enable complex data understanding, natural language interfaces, semantic analysis functions, and autonomous pipeline orchestration. From a modality perspective, we review LLM-based techniques for (i) structured data (e.g., NL2SQL, NL2GQL, ModelQA), (ii) semi-structured data (e.g., markup languages understanding, semi-structured table question answering), (iii) unstructured data (e.g., chart understanding, text/image document understanding), and (iv) heterogeneous data (e.g., data retrieval and modality alignment in data lakes). The technical evolution further distills four key design goals for intelligent data analysis agents, namely semantic-aware design, autonomous pipelines, tool-augmented workflows, and support for open-world tasks. Finally, we outline the remaining challenges and propose several insights and practical directions for advancing LLM/Agent-powered data analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14225v2">Know Me, Respond to Me: Benchmarking LLMs for Dynamic User Profiling and Personalized Responses at Scale</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 The 2025 Conference on Language Modeling (COLM)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have emerged as personalized assistants for users across a wide range of tasks -- from offering writing support to delivering tailored recommendations or consultations. Over time, the interaction history between a user and an LLM can provide extensive information about an individual's traits and preferences. However, open questions remain on how well LLMs today can effectively leverage such history to (1) internalize the user's inherent traits and preferences, (2) track how the user profiling and preferences evolve over time, and (3) generate personalized responses accordingly in new scenarios. In this work, we introduce the PERSONAMEM benchmark. PERSONAMEM features curated user profiles with over 180 simulated user-LLM interaction histories, each containing up to 60 sessions of multi-turn conversations across 15 real-world tasks that require personalization. Given an in-situ user query, i.e. query issued by the user from the first-person perspective, we evaluate LLM chatbots' ability to identify the most suitable response according to the current state of the user's profile. We observe that current LLMs still struggle to recognize the dynamic evolution in users' profiles over time through direct prompting approaches. As a consequence, LLMs often fail to deliver responses that align with users' current situations and preferences, with frontier models such as GPT-4.1, o4-mini, GPT-4.5, o1, or Gemini-2.0 achieving only around 50% overall accuracy, suggesting room for improvement. We hope that PERSONAMEM, along with the user profile and conversation simulation pipeline, can facilitate future research in the development of truly user-aware chatbots. Code and data are available at github.com/bowen-upenn/PersonaMem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.12491v2">Cost-Aware Contrastive Routing for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-27
    </div>
    <details class="paper-abstract">
      We study cost-aware routing for large language models across diverse and dynamic pools of models. Existing approaches often overlook prompt-specific context, rely on expensive model profiling, assume a fixed set of experts, or use inefficient trial-and-error strategies. We introduce Cost-Spectrum Contrastive Routing (CSCR), a lightweight framework that maps both prompts and models into a shared embedding space to enable fast, cost-sensitive selection. CSCR uses compact, fast-to-compute logit footprints for open-source models and perplexity fingerprints for black-box APIs. A contrastive encoder is trained to favor the cheapest accurate expert within adaptive cost bands. At inference time, routing reduces to a single k-NN lookup via a FAISS index, requiring no retraining when the expert pool changes and enabling microsecond latency. Across multiple benchmarks, CSCR consistently outperforms baselines, improving the accuracy-cost tradeoff by up to 25%, while generalizing robustly to unseen LLMs and out-of-distribution prompts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03604v2">Bilevel ZOFO: Bridging Parameter-Efficient and Zeroth-Order Techniques for Efficient LLM Fine-Tuning and Meta-Training</a></div>
    <div class="paper-meta">
      📅 2025-10-27
    </div>
    <details class="paper-abstract">
      Fine-tuning pre-trained Large Language Models (LLMs) for downstream tasks using First-Order (FO) optimizers presents significant computational challenges. Parameter-Efficient Fine-Tuning (PEFT) methods address these by freezing most model parameters and training only a small subset. However, PEFT often underperforms compared to full fine-tuning when high task-specific accuracy is required. Zeroth-Order (ZO) methods fine-tune the entire pre-trained model without back-propagation, estimating gradients through forward passes only. While memory-efficient, ZO methods suffer from slow convergence and high sensitivity to prompt selection. We bridge these two worlds with Bilevel-ZOFO, a bilevel optimization method that couples fast, local FO-PEFT adaptation at the inner level with stable, memory-efficient ZO updates of the full backbone at the outer level. The FO-PEFT inner loop performs fast, low-memory local adaptation that reduces the variance of ZO estimates and stabilizes the search, guiding the outer ZO updates of the full backbone and reducing prompt sensitivity. In the mean time, the outer ZO provides better generalization ability for PEFT. We provide theoretical convergence guarantees and empirically demonstrate that Bilevel-ZOFO significantly outperforms existing ZO and FO-PEFT methods, achieving 2-4 times faster training while maintaining similar memory efficiency. Additionally, we show by updating the backbone with ZO and adapting only a tiny FO-PEFT block per task, Bilevel-ZOFO combines full-model capacity with few-shot efficiency, making it a very efficient meta-learning algorithm that quickly adapts to new tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22922v1">Improving Human Verification of LLM Reasoning through Interactive Explanation Interfaces</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 19 pages, 14 figures
    </div>
    <details class="paper-abstract">
      The reasoning capabilities of Large Language Models (LLMs) have led to their increasing employment in several critical applications, particularly education, where they support problem-solving, tutoring, and personalized study. While there are a plethora of works showing the effectiveness of LLMs in generating step-by-step solutions through chain-of-thought (CoT) reasoning on reasoning benchmarks, little is understood about whether the generated CoT is helpful for end-users in improving their ability to comprehend mathematical reasoning problems and detect errors/hallucinations in LLM-generated solutions. To address this gap and contribute to understanding how reasoning can improve human-AI interaction, we present three new interactive reasoning interfaces: interactive CoT (iCoT), interactive Program-of-Thought (iPoT), and interactive Graph (iGraph), and a novel framework that generates the LLM's reasoning from traditional CoT to alternative, interactive formats. Across 125 participants, we found that interactive interfaces significantly improved performance. Specifically, the iGraph interface yielded the highest clarity and error detection rate (85.6%), followed by iPoT (82.5%), iCoT (80.6%), all outperforming standard CoT (73.5%). Interactive interfaces also led to faster response times, where participants using iGraph were fastest (57.9 secs), compared to iCoT and iPoT (60 secs), and the standard CoT baseline (64.7 secs). Furthermore, participants preferred the iGraph reasoning interface, citing its superior ability to enable users to follow the LLM's reasoning process. We discuss the implications of these results and provide recommendations for the future design of reasoning models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23946v1">Leveraging LLMs for Early Alzheimer's Prediction</a></div>
    <div class="paper-meta">
      📅 2025-10-27
    </div>
    <details class="paper-abstract">
      We present a connectome-informed LLM framework that encodes dynamic fMRI connectivity as temporal sequences, applies robust normalization, and maps these data into a representation suitable for a frozen pre-trained LLM for clinical prediction. Applied to early Alzheimer's detection, our method achieves sensitive prediction with error rates well below clinically recognized margins, with implications for timely Alzheimer's intervention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16712v2">The Chameleon Nature of LLMs: Quantifying Multi-Turn Stance Instability in Search-Enabled Language Models</a></div>
    <div class="paper-meta">
      📅 2025-10-26
      | 💬 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop: MTI-LLM @ NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Integration of Large Language Models with search/retrieval engines has become ubiquitous, yet these systems harbor a critical vulnerability that undermines their reliability. We present the first systematic investigation of "chameleon behavior" in LLMs: their alarming tendency to shift stances when presented with contradictory questions in multi-turn conversations (especially in search-enabled LLMs). Through our novel Chameleon Benchmark Dataset, comprising 17,770 carefully crafted question-answer pairs across 1,180 multi-turn conversations spanning 12 controversial domains, we expose fundamental flaws in state-of-the-art systems. We introduce two theoretically grounded metrics: the Chameleon Score (0-1) that quantifies stance instability, and Source Re-use Rate (0-1) that measures knowledge diversity. Our rigorous evaluation of Llama-4-Maverick, GPT-4o-mini, and Gemini-2.5-Flash reveals consistent failures: all models exhibit severe chameleon behavior (scores 0.391-0.511), with GPT-4o-mini showing the worst performance. Crucially, small across-temperature variance (less than 0.004) suggests the effect is not a sampling artifact. Our analysis uncovers the mechanism: strong correlations between source re-use rate and confidence (r=0.627) and stance changes (r=0.429) are statistically significant (p less than 0.05), indicating that limited knowledge diversity makes models pathologically deferential to query framing. These findings highlight the need for comprehensive consistency evaluation before deploying LLMs in healthcare, legal, and financial systems where maintaining coherent positions across interactions is critical for reliable decision support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01131v2">Beyond QA Pairs: Assessing Parameter-Efficient Fine-Tuning for Fact Embedding in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-26
      | 💬 Presented at the Workshop on Preparing Good Data for Generative AI: Challenges and Approaches (Good-Data) in conjunction with AAAI 2025. The authors retain the copyright
    </div>
    <details class="paper-abstract">
      This paper presents an extensive examination of Parameter-Efficient Fine-Tuning (PEFT) for embedding domain specific facts into Large Language Models (LLMs), focusing on improving the fine-tuning process by categorizing question-answer (QA) pairs into Factual and Conceptual classes using a BERT-based classifier. Two distinct Llama-2 models are fine-tuned based on these classifications and evaluated using larger models like GPT-3.5 Turbo and Gemini. Our results indicate that models trained on conceptual datasets outperform those trained on factual datasets. Additionally, we compare the efficiency of two synthetic fine-tuning dataset generation techniques, D-RAG and D-Naive, with D-Naive demonstrating superior performance. Although PEFT has shown effectiveness, our research indicates that it may not be the most optimal method for embedding facts into LLMs. However, it has demonstrated exceptional performance in instruction-based tasks. Our findings are reinforced by a 1000-sample dataset in the data center domain, where the fine-tuned Llama-2 7B model significantly outperforms the baseline model in generating product recommendations. Our study highlights the importance of QA pair categorization and synthetic dataset generation techniques in enhancing the performance of LLMs in specific domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00439v2">Improving the Distributional Alignment of LLMs using Supervision</a></div>
    <div class="paper-meta">
      📅 2025-10-26
    </div>
    <details class="paper-abstract">
      The ability to accurately align LLMs with human population groups on subjective questions would have great value. In this work, we show that use of simple supervision can greatly improve language model alignment with diverse population groups more consistently, as measured over three datasets spanning various topics. Beyond evaluating average alignment, we also report how alignment varies across specific groups. Our broad findings provide insights into the distributional alignment of LLMs with diverse population groups. By conducting evaluation over many LLMs and prompting strategies, along with open-sourcing our work, we provide a benchmark to stimulate future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13657v3">Why Do Multi-Agent LLM Systems Fail?</a></div>
    <div class="paper-meta">
      📅 2025-10-26
      | 💬 ArXiv v3
    </div>
    <details class="paper-abstract">
      Despite enthusiasm for Multi-Agent LLM Systems (MAS), their performance gains on popular benchmarks are often minimal. This gap highlights a critical need for a principled understanding of why MAS fail. Addressing this question requires systematic identification and analysis of failure patterns. We introduce MAST-Data, a comprehensive dataset of 1600+ annotated traces collected across 7 popular MAS frameworks. MAST-Data is the first multi-agent system dataset to outline the failure dynamics in MAS for guiding the development of better future systems. To enable systematic classification of failures for MAST-Data, we build the first Multi-Agent System Failure Taxonomy (MAST). We develop MAST through rigorous analysis of 150 traces, guided closely by expert human annotators and validated by high inter-annotator agreement (kappa = 0.88). This process identifies 14 unique modes, clustered into 3 categories: (i) system design issues, (ii) inter-agent misalignment, and (iii) task verification. To enable scalable annotation, we develop an LLM-as-a-Judge pipeline with high agreement with human annotations. We leverage MAST and MAST-Data to analyze failure patterns across models (GPT4, Claude 3, Qwen2.5, CodeLlama) and tasks (coding, math, general agent), demonstrating improvement headrooms from better MAS design. Our analysis provides insights revealing that identified failures require more sophisticated solutions, highlighting a clear roadmap for future research. We publicly release our comprehensive dataset (MAST-Data), the MAST, and our LLM annotator to facilitate widespread research and development in MAS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14634v5">Scideator: Human-LLM Scientific Idea Generation Grounded in Research-Paper Facet Recombination</a></div>
    <div class="paper-meta">
      📅 2025-10-26
      | 💬 Updated for clarity
    </div>
    <details class="paper-abstract">
      The scientific ideation process often involves blending salient aspects of existing papers to create new ideas -- a framework known as facet-based ideation. To see how large language models (LLMs) might assist in this process, we contribute Scideator, the first human-LLM interface for facet-based scientific ideation. Starting from a user-provided set of scientific papers, Scideator extracts key facets -- purposes, mechanisms, and evaluations -- from these and related papers, allowing users to explore the idea space by interactively recombining facets to synthesize inventive ideas. Scideator also helps users gauge idea originality by searching the literature for overlaps, assessing idea novelty based on an explicit facet-based definition. To support these tasks, Scideator introduces three LLM-powered retrieval-augmented generation (RAG) modules: Analogous Paper Facet Finder, Faceted Idea Generator, and Idea Novelty Checker. In a within-subjects user study (N=22) with computer-science researchers comparing Scideator to a strong baseline, our tool provided significantly more creativity support, particularly with respect to exploration and expressiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22866v1">Interpreting and Mitigating Unwanted Uncertainty in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-26
    </div>
    <details class="paper-abstract">
      Despite their impressive capabilities, Large Language Models (LLMs) exhibit unwanted uncertainty, a phenomenon where a model changes a previously correct answer into an incorrect one when re-prompted. This behavior undermines trust and poses serious risks in high-stakes domains. In this work, we investigate the mechanisms that drive this phenomenon. We adapt the Needle-in-a-Haystack retrieval framework and integrate a Flip-style re-evaluation prompt to simulate realistic answer-flipping scenarios. We find that retrieval heads are not primarily responsible for avoiding uncertainty. Instead, we identify a small set of non-retrieval attention heads that disproportionately attend to misleading tokens in uncertain contexts. Masking these heads yields significant improvements, reducing flip behavior by up to 15% without introducing incoherence or overcorrection. However, when tested for downstream tasks, we observe trade-offs with flip behavior. Our findings contribute to the growing field of mechanistic interpretability and present a simple yet effective technique for mitigating uncertainty-driven failure modes in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14305v2">Scaling Down, Serving Fast: Compressing and Deploying Efficient LLMs for Recommendation Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-26
      | 💬 Accepted to EMNLP 2025 Industry Track - Oral Presentation
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable performance across a wide range of industrial applications, from search and recommendation systems to generative tasks. Although scaling laws indicate that larger models generally yield better generalization and performance, their substantial computational requirements often render them impractical for many real-world scenarios at scale. In this paper, we present a comprehensive set of insights for training and deploying small language models (SLMs) that deliver high performance for a variety of industry use cases. We focus on two key techniques: (1) knowledge distillation and (2) model compression via structured pruning and quantization. These approaches enable SLMs to retain much of the quality of their larger counterparts while significantly reducing training/serving costs and latency. We detail the impact of these techniques on a variety of use cases in a large professional social network platform and share deployment lessons, including hardware optimization strategies that improve speed and throughput for both predictive and reasoning-based applications in Recommendation Systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22829v1">LLM-based Fusion of Multi-modal Features for Commercial Memorability Prediction</a></div>
    <div class="paper-meta">
      📅 2025-10-26
    </div>
    <details class="paper-abstract">
      This paper addresses the prediction of commercial (brand) memorability as part of "Subtask 2: Commercial/Ad Memorability" within the "Memorability: Predicting movie and commercial memorability" task at the MediaEval 2025 workshop competition. We propose a multimodal fusion system with a Gemma-3 LLM backbone that integrates pre-computed visual (ViT) and textual (E5) features by multi-modal projections. The model is adapted using Low-Rank Adaptation (LoRA). A heavily-tuned ensemble of gradient boosted trees serves as a baseline. A key contribution is the use of LLM-generated rationale prompts, grounded in expert-derived aspects of memorability, to guide the fusion model. The results demonstrate that the LLM-based system exhibits greater robustness and generalization performance on the final test set, compared to the baseline. The paper's codebase can be found at https://github.com/dsgt-arc/mediaeval-2025-memorability
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22787v1">Collaborative LLM Agents for C4 Software Architecture Design Automation</a></div>
    <div class="paper-meta">
      📅 2025-10-26
      | 💬 This paper has been accepted for the upcoming 59th Hawaii International Conference on System Sciences (HICSS-59), 2026, Hawaii, USA. The final published version will appear in the official conference proceedings
    </div>
    <details class="paper-abstract">
      Software architecture design is a fundamental part of creating every software system. Despite its importance, producing a C4 software architecture model, the preferred notation for such architecture, remains manual and time-consuming. We introduce an LLM-based multi-agent system that automates this task by simulating a dialogue between role-specific experts who analyze requirements and generate the Context, Container, and Component views of the C4 model. Quality is assessed with a hybrid evaluation framework: deterministic checks for structural and syntactic integrity and C4 rule consistency, plus semantic and qualitative scoring via an LLM-as-a-Judge approach. Tested on five canonical system briefs, the workflow demonstrates fast C4 model creation, sustains high compilation success, and delivers semantic fidelity. A comparison of four state-of-the-art LLMs shows different strengths relevant to architectural design. This study contributes to automated software architecture design and its evaluation methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22784v1">PIP-LLM: Integrating PDDL-Integer Programming with LLMs for Coordinating Multi-Robot Teams Using Natural Language</a></div>
    <div class="paper-meta">
      📅 2025-10-26
    </div>
    <details class="paper-abstract">
      Enabling robot teams to execute natural language commands requires translating high-level instructions into feasible, efficient multi-robot plans. While Large Language Models (LLMs) combined with Planning Domain Description Language (PDDL) offer promise for single-robot scenarios, existing approaches struggle with multi-robot coordination due to brittle task decomposition, poor scalability, and low coordination efficiency. We introduce PIP-LLM, a language-based coordination framework that consists of PDDL-based team-level planning and Integer Programming (IP) based robot-level planning. PIP-LLMs first decomposes the command by translating the command into a team-level PDDL problem and solves it to obtain a team-level plan, abstracting away robot assignment. Each team-level action represents a subtask to be finished by the team. Next, this plan is translated into a dependency graph representing the subtasks' dependency structure. Such a dependency graph is then used to guide the robot-level planning, in which each subtask node will be formulated as an IP-based task allocation problem, explicitly optimizing travel costs and workload while respecting robot capabilities and user-defined constraints. This separation of planning from assignment allows PIP-LLM to avoid the pitfalls of syntax-based decomposition and scale to larger teams. Experiments across diverse tasks show that PIP-LLM improves plan success rate, reduces maximum and average travel costs, and achieves better load balancing compared to state-of-the-art baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22767v1">TELL-TALE: Task Efficient LLMs with Task Aware Layer Elimination</a></div>
    <div class="paper-meta">
      📅 2025-10-26
    </div>
    <details class="paper-abstract">
      In this paper we introduce Tale, Task-Aware Layer Elimination, an inference-time algorithm that prunes entire transformer layers in an LLM by directly optimizing task-specific validation performance. We evaluate TALE on 9 tasks and 5 models, including LLaMA 3.1 8B, Qwen 2.5 7B, Qwen 2.5 0.5B, Mistral 7B, and Lucie 7B, under both zero-shot and few-shot settings. Unlike prior approaches, TALE requires no retraining and consistently improves accuracy while reducing computational cost across all benchmarks. Furthermore, applying TALE during finetuning leads to additional performance gains. Finally, TALE provides flexible user control over trade-offs between accuracy and efficiency. Mutual information analysis shows that certain layers act as bottlenecks, degrading task-relevant representations. Tale's selective layer removal remedies this problem, producing smaller, faster, and more accurate models that are also faster to fine-tune while offering new insights into transformer interpretability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05605v6">Evolving LLMs' Self-Refinement Capability via Synergistic Training-Inference Optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-26
    </div>
    <details class="paper-abstract">
      Self-Refinement refers to a model's ability to revise its own responses to produce improved outputs. This capability can also serve as a fundamental mechanism for Self-Improvement, for example, by reconstructing datasets with refined results to enhance intrinsic model performance. However, our comprehensive experiments reveal that large language models (LLMs) show no clear evidence of inherent Self-Refinement and may even experience response quality degradation after Self-Refinement. To address this issue, we propose EVOLVE, a simple and effective framework for eliciting and tracking the evolution of Self-Refinement through iterative training. We first explore optimization methods during training to activate the model's Self-Refinement capability. Then, at inference, we investigate various generation strategies to further enhance and utilize Self-Refinement while supplying the necessary data for training. Through synergistic optimization of training and inference stages, we continually evolve the model's Self-Refinement ability, enabling it to better refine its own responses. Moreover, we demonstrate the potential of leveraging Self-Refinement to achieve broader Self-Improvement of intrinsic model abilities. Experiments show that the evolved Self-Refinement ability enables the Llama-3.1-8B base model to surpass GPT-4o, achieving 62.3% length-controlled and 63.3% raw win rates on AlpacaEval 2, and 50.3% on Arena-Hard. It also generalizes effectively to out-of-domain reasoning tasks, improving performance on mathematical reasoning benchmarks such as GSM8K and MATH.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.00775v3">Efficiency with Rigor! A Trustworthy LLM-powered Workflow for Qualitative Data Analysis</a></div>
    <div class="paper-meta">
      📅 2025-10-26
    </div>
    <details class="paper-abstract">
      Qualitative data analysis (QDA) emphasizes trustworthiness, requiring sustained human engagement and reflexivity. Recently, large language models (LLMs) have been applied in QDA to improve efficiency. However, their use raises concerns about unvalidated automation and displaced sensemaking, which can undermine trustworthiness. To address these issues, we employed two strategies: transparency and human involvement. Through a literature review and formative interviews, we identified six design requirements for transparent automation and meaningful human involvement. Guided by these requirements, we developed MindCoder, an LLM-powered workflow that delegates mechanical tasks, such as grouping and validation, to the system, while enabling humans to conduct meaningful interpretation. MindCoder also maintains comprehensive logs of users' step-by-step interactions to ensure transparency and support trustworthy results. In an evaluation with 12 users and two external evaluators, MindCoder supported active interpretation, offered flexible control, and produced more trustworthy codebooks. We further discuss design implications for building human-AI collaborative QDA workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19178v2">Imbalanced Gradients in RL Post-Training of Multi-Task LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-26
    </div>
    <details class="paper-abstract">
      Multi-task post-training of large language models (LLMs) is typically performed by mixing datasets from different tasks and optimizing them jointly. This approach implicitly assumes that all tasks contribute gradients of similar magnitudes; when this assumption fails, optimization becomes biased toward large-gradient tasks. In this paper, however, we show that this assumption fails in RL post-training: certain tasks produce significantly larger gradients, thus biasing updates toward those tasks. Such gradient imbalance would be justified only if larger gradients implied larger learning gains on the tasks (i.e., larger performance improvements) -- but we find this is not true. Large-gradient tasks can achieve similar or even much lower learning gains than small-gradient ones. Further analyses reveal that these gradient imbalances cannot be explained by typical training statistics such as training rewards or advantages, suggesting that they arise from the inherent differences between tasks. This cautions against naive dataset mixing and calls for future work on principled gradient-level corrections for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04055v2">Rethinking and Exploring String-Based Malware Family Classification in the Era of LLMs and RAG</a></div>
    <div class="paper-meta">
      📅 2025-10-26
      | 💬 This is a technical report from Lingnan University, Hong Kong. Code is available at https://github.com/AIS2Lab/MalwareGPT
    </div>
    <details class="paper-abstract">
      Malware family classification aims to identify the specific family (e.g., GuLoader or BitRAT) a malware sample may belong to, in contrast to malware detection or sample classification, which only predicts a Yes/No outcome. Accurate family identification can greatly facilitate automated sample labeling and understanding on crowdsourced malware analysis platforms such as VirusTotal and MalwareBazaar, which generate vast amounts of data daily. In this paper, we explore and assess the feasibility of using traditional binary string features for family classification in the new era of large language models (LLMs) and Retrieval-Augmented Generation (RAG). Specifically, we investigate howFamily-Specific String (FSS) features can be utilized in a manner similar to RAG to facilitate family classification. To this end, we develop a curated evaluation framework covering 4,347 samples from 67 malware families, extract and analyze over 25 million strings, and conduct detailed ablation studies to assess the impact of different design choices in four major modules, with each providing a relative improvement ranging from 8.1% to 120%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20119v2">LLM Agents for Generating Microservice-based Applications: how complex is your specification?</a></div>
    <div class="paper-meta">
      📅 2025-10-26
      | 💬 24 pages + 5 pages appendices. 7 Figures. 10 Tables
    </div>
    <details class="paper-abstract">
      In this paper we evaluate the capabilities of LLM Agents in generating code for real-world problems. Specifically, we explore code synthesis for microservice-based applications, a widely used architectural pattern for building applications. We define a standard template for specifying these applications, and we propose a metric for scoring the difficulty of a specification. The higher the score, the more difficult it is to generate code for the specification. Our experimental results show that agents using strong LLMs (like GPT-3o-mini) do fairly well on medium difficulty specifications but do poorly on those of higher difficulty levels. This is due to more intricate business logic, a greater use of external services, database integration and inclusion of non-functional capabilities such as authentication. We analyzed the errors in LLM-synthesized code and report on the key challenges LLM Agents face in generating code for these specifications. Finally, we show that using a fine-grained approach to code generation improves the correctness of the generated code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22691v1">SALSA: Single-pass Autoregressive LLM Structured Classification</a></div>
    <div class="paper-meta">
      📅 2025-10-26
    </div>
    <details class="paper-abstract">
      Despite their impressive generalization capabilities, instruction-tuned Large Language Models often underperform on text classification benchmarks. We introduce SALSA, a coherent pipeline that combines structured prompting, class-to-token mapping, and parameter-efficient fine-tuning, thereby avoiding cold-start training. Each class label is mapped to a distinct output token, and prompts are constructed to elicit a single-token response. During inference, the model's output is projected only onto the logits of the relevant class tokens, enabling efficient and accurate classification in a single forward pass. SALSA achieves state-of-the-art results across diverse benchmarks, demonstrating its robustness and scalability for LLM-based classification applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22689v1">Rule-Based Explanations for Retrieval-Augmented LLM Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-26
    </div>
    <details class="paper-abstract">
      If-then rules are widely used to explain machine learning models; e.g., "if employed = no, then loan application = rejected." We present the first proposal to apply rules to explain the emerging class of large language models (LLMs) with retrieval-augmented generation (RAG). Since RAG enables LLM systems to incorporate retrieved information sources at inference time, rules linking the presence or absence of sources can explain output provenance; e.g., "if a Times Higher Education ranking article is retrieved, then the LLM ranks Oxford first." To generate such rules, a brute force approach would probe the LLM with all source combinations and check if the presence or absence of any sources leads to the same output. We propose optimizations to speed up rule generation, inspired by Apriori-like pruning from frequent itemset mining but redefined within the scope of our novel problem. We conclude with qualitative and quantitative experiments demonstrating our solutions' value and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13586v3">Deflanderization for Game Dialogue: Balancing Character Authenticity with Task Execution in LLM-based NPCs</a></div>
    <div class="paper-meta">
      📅 2025-10-26
    </div>
    <details class="paper-abstract">
      The emergence of large language models (LLMs) has opened new opportunities for creating dynamic non-player characters (NPCs) in gaming environments, enabling both functional task execution and persona-consistent dialogue generation. In this paper, we (Tu_Character_lab) report our participation in the Commonsense Persona-Grounded Dialogue Challenge (CPDC) 2025 Round 2, which evaluates agents across three tracks: task-oriented dialogue, context-aware dialogue, and their integration. Our approach combines two complementary strategies: (i) lightweight prompting techniques in the API track, including a Deflanderization prompting method to suppress excessive role-play and improve task fidelity, and (ii) fine-tuned large models in the GPU track, leveraging Qwen3-14B with supervisedfinetuning (SFT) and Low-Rank Adaptation(LoRA). Our best submissions ranked 2nd on Task 1, 2nd on Task 3 (API track), and 4th on Task 3 (GPU track).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03145v2">Entity-Augmented Neuroscience Knowledge Retrieval Using Ontology and Semantic Understanding Capability of LLM</a></div>
    <div class="paper-meta">
      📅 2025-10-26
    </div>
    <details class="paper-abstract">
      Neuroscience research publications encompass a vast wealth of knowledge. Accurately retrieving existing information and discovering new insights from this extensive literature is essential for advancing the field. However, when knowledge is dispersed across multiple sources, current state-of-the-art retrieval methods often struggle to extract the necessary information. A knowledge graph (KG) can integrate and link knowledge from multiple sources. However, existing methods for constructing KGs in neuroscience often rely on labeled data and require domain expertise. Acquiring large-scale, labeled data for a specialized area like neuroscience presents significant challenges. This work proposes novel methods for constructing KG from unlabeled large-scale neuroscience research corpus utilizing large language models (LLM), neuroscience ontology, and text embeddings. We analyze the semantic relevance of neuroscience text segments identified by LLM for building the knowledge graph. We also introduce an entity-augmented information retrieval algorithm to extract knowledge from the KG. Several experiments were conducted to evaluate the proposed approaches. The results demonstrate that our methods significantly enhance knowledge discovery from the unlabeled neuroscience research corpus. The performance of the proposed entity and relation extraction method is comparable to the existing supervised method. It achieves an F1 score of 0.84 for entity extraction from the unlabeled data. The knowledge obtained from the KG improves answers to over 52% of neuroscience questions from the PubMedQA dataset and questions generated using selected neuroscience entities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13481v2">Tahakom LLM Guidelines and Recipes: From Pre-training Data to an Arabic LLM</a></div>
    <div class="paper-meta">
      📅 2025-10-26
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have significantly advanced the field of natural language processing, enhancing capabilities in both language understanding and generation across diverse domains. However, developing LLMs for Arabic presents unique challenges. This paper explores these challenges by focusing on critical aspects such as data curation, tokenizer design, and evaluation. We detail our approach to the collection and filtration of Arabic pre-training datasets, assess the impact of various tokenizer designs on model performance, and examine the limitations of existing Arabic evaluation frameworks, for which we propose a systematic corrective methodology. To promote transparency and facilitate collaborative development, we share our data and methodologies, contributing to the advancement of language modeling, particularly for the Arabic language.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22628v1">Sentra-Guard: A Multilingual Human-AI Framework for Real-Time Defense Against Adversarial LLM Jailbreaks</a></div>
    <div class="paper-meta">
      📅 2025-10-26
      | 💬 11 pages, 5 figures. Preprint version under review in the area of Artificial Intelligence (cs.AI)
    </div>
    <details class="paper-abstract">
      This paper presents a real-time modular defense system named Sentra-Guard. The system detects and mitigates jailbreak and prompt injection attacks targeting large language models (LLMs). The framework uses a hybrid architecture with FAISS-indexed SBERT embedding representations that capture the semantic meaning of prompts, combined with fine-tuned transformer classifiers, which are machine learning models specialized for distinguishing between benign and adversarial language inputs. It identifies adversarial prompts in both direct and obfuscated attack vectors. A core innovation is the classifier-retriever fusion module, which dynamically computes context-aware risk scores that estimate how likely a prompt is to be adversarial based on its content and context. The framework ensures multilingual resilience with a language-agnostic preprocessing layer. This component automatically translates non-English prompts into English for semantic evaluation, enabling consistent detection across over 100 languages. The system includes a HITL feedback loop, where decisions made by the automated system are reviewed by human experts for continual learning and rapid adaptation under adversarial pressure. Sentra-Guard maintains an evolving dual-labeled knowledge base of benign and malicious prompts, enhancing detection reliability and reducing false positives. Evaluation results show a 99.96% detection rate (AUC = 1.00, F1 = 1.00) and an attack success rate (ASR) of only 0.004%. This outperforms leading baselines such as LlamaGuard-2 (1.3%) and OpenAI Moderation (3.7%). Unlike black-box approaches, Sentra-Guard is transparent, fine-tunable, and compatible with diverse LLM backends. Its modular design supports scalable deployment in both commercial and open-source environments. The system establishes a new state-of-the-art in adversarial LLM defense.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22620v1">Breaking Agent Backbones: Evaluating the Security of Backbone LLMs in AI Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-26
      | 💬 Julia Bazinska and Max Mathys contributed equally
    </div>
    <details class="paper-abstract">
      AI agents powered by large language models (LLMs) are being deployed at scale, yet we lack a systematic understanding of how the choice of backbone LLM affects agent security. The non-deterministic sequential nature of AI agents complicates security modeling, while the integration of traditional software with AI components entangles novel LLM vulnerabilities with conventional security risks. Existing frameworks only partially address these challenges as they either capture specific vulnerabilities only or require modeling of complete agents. To address these limitations, we introduce threat snapshots: a framework that isolates specific states in an agent's execution flow where LLM vulnerabilities manifest, enabling the systematic identification and categorization of security risks that propagate from the LLM to the agent level. We apply this framework to construct the $\operatorname{b}^3$ benchmark, a security benchmark based on 194331 unique crowdsourced adversarial attacks. We then evaluate 31 popular LLMs with it, revealing, among other insights, that enhanced reasoning capabilities improve security, while model size does not correlate with security. We release our benchmark, dataset, and evaluation code to facilitate widespread adoption by LLM providers and practitioners, offering guidance for agent developers and incentivizing model developers to prioritize backbone security improvements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22603v1">Mitigating Attention Sinks and Massive Activations in Audio-Visual Speech Recognition with LLMS</a></div>
    <div class="paper-meta">
      📅 2025-10-26
      | 💬 The code is available at https://github.com/umbertocappellazzo/Llama-AVSR
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently advanced auditory speech recognition (ASR), visual speech recognition (VSR), and audio-visual speech recognition (AVSR). However, understanding of their internal dynamics under fine-tuning remains limited. In natural language processing, recent work has revealed attention sinks, tokens that attract disproportionately high attention, and associated massive activations in which some features of sink tokens exhibit huge activation in LLMs. In this work, we are the first to study these phenomena in multimodal speech recognition. Through a detailed analysis of audio-visual LLMs, we identify attention sinks and massive activations not only at the BOS token but also at intermediate low-semantic tokens across ASR, VSR, and AVSR. We show that massive activations originate in the MLP layers and correspond to fixed feature indices across all sink tokens. We further show that intermediate sink tokens exhibit high cosine similarity to the BOS token, thereby amplifying attention and activation. Building on these insights, we introduce a simple decorrelation loss that reduces cosine similarity between BOS and other tokens, effectively mitigating intermediate sinks and massive activations. Furthermore, our method improves word error rate (WER) under high audio-visual feature downsampling while remaining stable at lower downsampling rates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22593v1">AutoBench: Automating LLM Evaluation through Reciprocal Peer Assessment</a></div>
    <div class="paper-meta">
      📅 2025-10-26
    </div>
    <details class="paper-abstract">
      We present AutoBench, a fully automated and self-sustaining framework for evaluating Large Language Models (LLMs) through reciprocal peer assessment. This paper provides a rigorous scientific validation of the AutoBench methodology, originally developed as an open-source project by eZecute S.R.L.. Unlike static benchmarks that suffer from test-set contamination and limited adaptability, AutoBench dynamically generates novel evaluation tasks while models alternately serve as question generators, contestants, and judges across diverse domains. An iterative weighting mechanism amplifies the influence of consistently reliable evaluators, aggregating peer judgments into consensus-based rankings that reflect collective model agreement. Our experiments demonstrate strong correlations with established benchmarks including MMLU-Pro and GPQA (respectively 78\% and 63\%), validating this peer-driven evaluation paradigm. The multi-judge design significantly outperforms single-judge baselines, confirming that distributed evaluation produces more robust and human-consistent assessments. AutoBench offers a scalable, contamination-resistant alternative to static benchmarks for the continuous evaluation of evolving language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17883v2">From Flows to Words: Can Zero-/Few-Shot LLMs Detect Network Intrusions? A Grammar-Constrained, Calibrated Evaluation on UNSW-NB15</a></div>
    <div class="paper-meta">
      📅 2025-10-26
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can reason over natural-language inputs, but their role in intrusion detection without fine-tuning remains uncertain. This study evaluates a prompt-only approach on UNSW-NB15 by converting each network flow to a compact textual record and augmenting it with lightweight, domain-inspired boolean flags (asymmetry, burst rate, TTL irregularities, timer anomalies, rare service/state, short bursts). To reduce output drift and support measurement, the model is constrained to produce structured, grammar-valid responses, and a single decision threshold is calibrated on a small development split. We compare zero-shot, instruction-guided, and few-shot prompting to strong tabular and neural baselines under identical splits, reporting accuracy, precision, recall, F1, and macro scores. Empirically, unguided prompting is unreliable, while instructions plus flags substantially improve detection quality; adding calibrated scoring further stabilizes results. On a balanced subset of two hundred flows, a 7B instruction-tuned model with flags reaches macro-F1 near 0.78; a lighter 3B model with few-shot cues and calibration attains F1 near 0.68 on one thousand examples. As the evaluation set grows to two thousand flows, decision quality decreases, revealing sensitivity to coverage and prompting. Tabular baselines remain more stable and faster, yet the prompt-only pipeline requires no gradient training, produces readable artifacts, and adapts easily through instructions and flags. Contributions include a flow-to-text protocol with interpretable cues, a calibration method for thresholding, a systematic baseline comparison, and a reproducibility bundle with prompts, grammar, metrics, and figures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17884v2">When Intelligence Fails: An Empirical Study on Why LLMs Struggle with Password Cracking</a></div>
    <div class="paper-meta">
      📅 2025-10-26
    </div>
    <details class="paper-abstract">
      The remarkable capabilities of Large Language Models (LLMs) in natural language understanding and generation have sparked interest in their potential for cybersecurity applications, including password guessing. In this study, we conduct an empirical investigation into the efficacy of pre-trained LLMs for password cracking using synthetic user profiles. Specifically, we evaluate the performance of state-of-the-art open-source LLMs such as TinyLLaMA, Falcon-RW-1B, and Flan-T5 by prompting them to generate plausible passwords based on structured user attributes (e.g., name, birthdate, hobbies). Our results, measured using Hit@1, Hit@5, and Hit@10 metrics under both plaintext and SHA-256 hash comparisons, reveal consistently poor performance, with all models achieving less than 1.5% accuracy at Hit@10. In contrast, traditional rule-based and combinator-based cracking methods demonstrate significantly higher success rates. Through detailed analysis and visualization, we identify key limitations in the generative reasoning of LLMs when applied to the domain-specific task of password guessing. Our findings suggest that, despite their linguistic prowess, current LLMs lack the domain adaptation and memorization capabilities required for effective password inference, especially in the absence of supervised fine-tuning on leaked password datasets. This study provides critical insights into the limitations of LLMs in adversarial contexts and lays the groundwork for future efforts in secure, privacy-preserving, and robust password modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22559v1">A Closed-Loop Personalized Learning Agent Integrating Neural Cognitive Diagnosis, Bounded-Ability Adaptive Testing, and LLM-Driven Feedback</a></div>
    <div class="paper-meta">
      📅 2025-10-26
      | 💬 8 pages, 6 figures
    </div>
    <details class="paper-abstract">
      As information technology advances, education is moving from one-size-fits-all instruction toward personalized learning. However, most methods handle modeling, item selection, and feedback in isolation rather than as a closed loop. This leads to coarse or opaque student models, assumption-bound adaptivity that ignores diagnostic posteriors, and generic, non-actionable feedback. To address these limitations, this paper presents an end-to-end personalized learning agent, EduLoop-Agent, which integrates a Neural Cognitive Diagnosis model (NCD), a Bounded-Ability Estimation Computerized Adaptive Testing strategy (BECAT), and large language models (LLMs). The NCD module provides fine-grained estimates of students' mastery at the knowledge-point level; BECAT dynamically selects subsequent items to maximize relevance and learning efficiency; and LLMs convert diagnostic signals into structured, actionable feedback. Together, these components form a closed-loop framework of ``Diagnosis--Recommendation--Feedback.'' Experiments on the ASSISTments dataset show that the NCD module achieves strong performance on response prediction while yielding interpretable mastery assessments. The adaptive recommendation strategy improves item relevance and personalization, and the LLM-based feedback offers targeted study guidance aligned with identified weaknesses. Overall, the results indicate that the proposed design is effective and practically deployable, providing a feasible pathway to generating individualized learning trajectories in intelligent education.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15550v2">PhysGym: Benchmarking LLMs in Interactive Physics Discovery with Controlled Priors</a></div>
    <div class="paper-meta">
      📅 2025-10-26
      | 💬 31 Pages
    </div>
    <details class="paper-abstract">
      Evaluating the scientific discovery capabilities of large language model based agents, particularly how they cope with varying environmental complexity and utilize prior knowledge, requires specialized benchmarks currently lacking in the landscape. To address this gap, we introduce \textsc{PhysGym}, a novel benchmark suite and simulation platform for rigorously assessing LLM-based scientific reasoning in interactive physics environments. \textsc{PhysGym}'s primary contribution lies in its sophisticated control over the level of prior knowledge provided to the agent. This allows researchers to dissect agent performance along axes including the complexity of the problem and the prior knowledge levels. The benchmark comprises a suite of interactive simulations, where agents must actively probe environments, gather data sequentially under constraints and formulate hypotheses about underlying physical laws. \textsc{PhysGym} provides standardized evaluation protocols and metrics for assessing hypothesis accuracy and model fidelity. We demonstrate the benchmark's utility by presenting results from baseline LLMs, showcasing its ability to differentiate capabilities based on varying priors and task complexity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02890v5">Theoretically Grounded Framework for LLM Watermarking: A Distribution-Adaptive Approach</a></div>
    <div class="paper-meta">
      📅 2025-10-26
    </div>
    <details class="paper-abstract">
      Watermarking has emerged as a crucial method to distinguish AI-generated text from human-created text. Current watermarking approaches often lack formal optimality guarantees or address the scheme and detector design separately. In this paper, we introduce a novel, unified theoretical framework for watermarking Large Language Models (LLMs) that jointly optimizes both the watermarking scheme and detector. Our approach aims to maximize detection performance while maintaining control over the worst-case false positive rate (FPR) and distortion on text quality. We derive closed-form optimal solutions for this joint design and characterize the fundamental trade-off between watermark detectability and distortion. Notably, we reveal that the optimal watermarking schemes should be adaptive to the LLM's generative distribution. Building on our theoretical insights, we propose a distortion-free, distribution-adaptive watermarking algorithm (DAWA) that leverages a surrogate model for model-agnosticism and efficiency. Experiments on Llama2-13B and Mistral-8$\times$7B models confirm the effectiveness of our approach, particularly at ultra-low FPRs. Our code is available at https://github.com/yepengliu/DAWA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22548v1">LooGLE v2: Are LLMs Ready for Real World Long Dependency Challenges?</a></div>
    <div class="paper-meta">
      📅 2025-10-26
      | 💬 NeurIPS 2025 Datasets and Benchmarks Track
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are equipped with increasingly extended context windows recently, yet their long context understanding capabilities over long dependency tasks remain fundamentally limited and underexplored. This gap is especially significant in many real-world long-context applications that were rarely benchmarked. In this paper, we introduce LooGLE v2, a novel benchmark designed to evaluate LLMs' long context ability in real-world applications and scenarios. Our benchmark consists of automatically collected real-world long texts, ranging from 16k to 2M tokens, encompassing domains in law, finance, game and code. Accordingly, we delicately design 10 types of domain-specific long-dependency tasks and generate 1,934 QA instances with various diversity and complexity in a scalable data curation pipeline for further practical needs. We conduct a comprehensive assessment of 6 locally deployed and 4 API-based LLMs. The evaluation results show that even the best-performing model achieves only a 59.2% overall score on our benchmark. Despite the extensive context windows, popular LLMs are only capable of understanding a much shorter length of context than they claim to be, revealing significant limitations in their ability to handle real-world tasks with long dependencies and highlighting substantial room for model improvement in practical long-context understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22503v1">Accelerating Materials Design via LLM-Guided Evolutionary Search</a></div>
    <div class="paper-meta">
      📅 2025-10-26
    </div>
    <details class="paper-abstract">
      Materials discovery requires navigating vast chemical and structural spaces while satisfying multiple, often conflicting, objectives. We present LLM-guided Evolution for MAterials design (LLEMA), a unified framework that couples the scientific knowledge embedded in large language models with chemistry-informed evolutionary rules and memory-based refinement. At each iteration, an LLM proposes crystallographically specified candidates under explicit property constraints; a surrogate-augmented oracle estimates physicochemical properties; and a multi-objective scorer updates success/failure memories to guide subsequent generations. Evaluated on 14 realistic tasks spanning electronics, energy, coatings, optics, and aerospace, LLEMA discovers candidates that are chemically plausible, thermodynamically stable, and property-aligned, achieving higher hit-rates and stronger Pareto fronts than generative and LLM-only baselines. Ablation studies confirm the importance of rule-guided generation, memory-based refinement, and surrogate prediction. By enforcing synthesizability and multi-objective trade-offs, LLEMA delivers a principled pathway to accelerate practical materials discovery. Code: https://github.com/scientific-discovery/LLEMA
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19462v2">AegisMCP: Online Graph Intrusion Detection for Tool-Augmented LLMs on Edge Devices</a></div>
    <div class="paper-meta">
      📅 2025-10-25
    </div>
    <details class="paper-abstract">
      In this work, we study security of Model Context Protocol (MCP) agent toolchains and their applications in smart homes. We introduce AegisMCP, a protocol-level intrusion detector. Our contributions are: (i) a minimal attack suite spanning instruction-driven escalation, chain-of-tool exfiltration, malicious MCP server registration, and persistence; (ii) NEBULA-Schema (Network-Edge Behavioral Learning for Untrusted LLM Agents), a reusable protocol-level instrumentation that represents MCP activity as a streaming heterogeneous temporal graph over agents, MCP servers, tools, devices, remotes, and sessions; and (iii) a CPU-only streaming detector that fuses novelty, session-DAG structure, and attribute cues for near-real-time edge inference, with optional fusion of local prompt-guardrail signals. On an emulated smart-home testbed spanning multiple MCP stacks and a physical bench, AegisMCP achieves sub-second per-window model inference and end-to-end alerting. The latency of AegisMCP is consistently sub-second on Intel N150-class edge hardware, while outperforming traffic-only and sequence baselines; ablations confirm the importance of DAG and install/permission signals. We release code, schemas, and generators for reproducible evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22422v1">Group size effects and collective misalignment in LLM multi-agent systems</a></div>
    <div class="paper-meta">
      📅 2025-10-25
    </div>
    <details class="paper-abstract">
      Multi-agent systems of large language models (LLMs) are rapidly expanding across domains, introducing dynamics not captured by single-agent evaluations. Yet, existing work has mostly contrasted the behavior of a single agent with that of a collective of fixed size, leaving open a central question: how does group size shape dynamics? Here, we move beyond this dichotomy and systematically explore outcomes across the full range of group sizes. We focus on multi-agent misalignment, building on recent evidence that interacting LLMs playing a simple coordination game can generate collective biases absent in individual models. First, we show that collective bias is a deeper phenomenon than previously assessed: interaction can amplify individual biases, introduce new ones, or override model-level preferences. Second, we demonstrate that group size affects the dynamics in a non-linear way, revealing model-dependent dynamical regimes. Finally, we develop a mean-field analytical approach and show that, above a critical population size, simulations converge to deterministic predictions that expose the basins of attraction of competing equilibria. These findings establish group size as a key driver of multi-agent dynamics and highlight the need to consider population-level effects when deploying LLM-based systems at scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08263v3">LLM-Based Detection of Tangled Code Changes for Higher-Quality Method-Level Bug Datasets</a></div>
    <div class="paper-meta">
      📅 2025-10-25
    </div>
    <details class="paper-abstract">
      Tangled code changes, commits that conflate unrelated modifications such as bug fixes, refactorings, and enhancements, introduce significant noise into bug datasets and adversely affect the performance of bug prediction models. Addressing this issue at a fine-grained, method-level granularity remains unexplored. This is critical to address, as recent bug prediction models, driven by practitioner demand, are increasingly focusing on finer granularity rather than traditional class- or file-level predictions. This study investigates the utility of Large Language Models (LLMs) for detecting tangled code changes by leveraging both commit messages and method-level code diffs. We formulate the problem as a binary classification task and evaluate multiple prompting strategies, including zero-shot, few-shot, and chain-of-thought prompting, using state-of-the-art proprietary LLMs such as GPT-5 and Gemini-2.0-Flash, and open-source models such as GPT-OSS-120B and CodeBERT. Our results demonstrate that combining commit messages with code diffs significantly enhances model performance, with the combined few-shot and chain-of-thought prompting achieving an F1-score of 0.883. Additionally, we explore machine learning models trained on LLM-generated embeddings, where a multi-layer perceptron classifier achieves superior performance (F1-score: 0.906, MCC: 0.807). Applying our approach to 49 open-source projects improves the distributional separability of code metrics between buggy and non-buggy methods, demonstrating the promise of LLMs for method-level commit untangling and potentially contributing to improving the accuracy of future bug prediction models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23659v2">Aligning LLMs for Multilingual Consistency in Enterprise Applications</a></div>
    <div class="paper-meta">
      📅 2025-10-25
      | 💬 Accepted at EMNLP 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) remain unreliable for global enterprise applications due to substantial performance gaps between high-resource and mid/low-resource languages, driven by English-centric pretraining and internal reasoning biases. This inconsistency undermines customer experience and operational reliability in multilingual settings such as customer support, content moderation, and information retrieval. Even with advanced Retrieval-Augmented Generation (RAG) systems, we observe up to an 29% accuracy drop in non-English languages compared to English. We propose a practical, batch-wise alignment strategy for fine-tuning LLMs, leveraging semantically equivalent multilingual data in each training batch to directly align model outputs across languages. This approach improves non-English accuracy by up to 23.9% without compromising English performance, model reasoning, or retrieval quality. Our method is simple to implement, scalable, and integrates seamlessly with existing LLM training & deployment pipelines, enabling more robust and equitable multilingual AI solutions in industry.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24749v2">SUMO: Subspace-Aware Moment-Orthogonalization for Accelerating Memory-Efficient LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-10-25
    </div>
    <details class="paper-abstract">
      Low-rank gradient-based optimization methods have significantly improved memory efficiency during the training of large language models (LLMs), enabling operations within constrained hardware without sacrificing performance. However, these methods primarily emphasize memory savings, often overlooking potential acceleration in convergence due to their reliance on standard isotropic steepest descent techniques, which can perform suboptimally in the highly anisotropic landscapes typical of deep networks, particularly LLMs. In this paper, we propose SUMO (Subspace-Aware Moment-Orthogonalization), an optimizer that employs exact singular value decomposition (SVD) for moment orthogonalization within a dynamically adapted low-dimensional subspace, enabling norm-inducing steepest descent optimization steps. By explicitly aligning optimization steps with the spectral characteristics of the loss landscape, SUMO effectively mitigates approximation errors associated with commonly used methods like Newton-Schulz orthogonalization approximation. We theoretically establish an upper bound on these approximation errors, proving their dependence on the condition numbers of moments, conditions we analytically demonstrate are encountered during LLM training. Furthermore, we both theoretically and empirically illustrate that exact orthogonalization via SVD substantially improves convergence rates while reducing overall complexity. Empirical evaluations confirm that SUMO accelerates convergence, enhances stability, improves performance, and reduces memory requirements by up to 20% compared to state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05790v2">Breaking the Frozen Subspace: Importance Sampling for Low-Rank Optimization in LLM Pretraining</a></div>
    <div class="paper-meta">
      📅 2025-10-25
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Low-rank optimization has emerged as a promising approach to enabling memory-efficient training of large language models (LLMs). Existing low-rank optimization methods typically project gradients onto a low-rank subspace, reducing the memory cost of storing optimizer states. A key challenge in these methods is selecting suitable subspaces to ensure an effective optimization trajectory. Most existing approaches select the dominant subspace to preserve gradient information, as this intuitively provides the best approximation. However, we find that in practice, the dominant subspace stops changing during pretraining, thereby constraining weight updates to similar subspaces. In this paper, we propose importance sampling for low-rank optimization in LLM pretraining with a provable convergence guarantee, which the dominant subspace approach does not have. Empirically, we demonstrate that our method significantly outperforms previous methods in LLM pretraining tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22376v1">Label Smoothing Improves Gradient Ascent in LLM Unlearning</a></div>
    <div class="paper-meta">
      📅 2025-10-25
    </div>
    <details class="paper-abstract">
      LLM unlearning has emerged as a promising approach, aiming to enable models to forget hazardous/undesired knowledge at low cost while preserving as much model utility as possible. Among existing techniques, the most straightforward method is performing Gradient Ascent (GA) w.r.t. the forget data, thereby forcing the model to unlearn the forget dataset. However, GA suffers from severe instability, as it drives updates in a divergent direction, often resulting in drastically degraded model utility. To address this issue, we propose Smoothed Gradient Ascent (SGA). SGA combines the forget data with multiple constructed normal data through a tunable smoothing rate. Intuitively, this extends GA from learning solely on the forget data to jointly learning across both forget and normal data, enabling more stable unlearning while better preserving model utility. Theoretically, we provide the theoretical guidance on the selection of the optimal smoothing rate. Empirically, we evaluate SGA on three benchmarks: TOFU, Harry Potter, and MUSE-NEWS. Experimental results demonstrate that SGA consistently outperforms the original Gradient Ascent (GA) method across all metrics and achieves top-2 performance among all baseline methods on several key metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19028v2">Are they lovers or friends? Evaluating LLMs' Social Reasoning in English and Korean Dialogues</a></div>
    <div class="paper-meta">
      📅 2025-10-25
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly used in human-AI interactions, their social reasoning capabilities in interpersonal contexts are critical. We introduce SCRIPTS, a 1k-dialogue dataset in English and Korean, sourced from movie scripts. The task involves evaluating models' social reasoning capability to infer the interpersonal relationships (e.g., friends, sisters, lovers) between speakers in each dialogue. Each dialogue is annotated with probabilistic relational labels (Highly Likely, Less Likely, Unlikely) by native (or equivalent) Korean and English speakers from Korea and the U.S. Evaluating nine models on our task, current proprietary LLMs achieve around 75-80% on the English dataset, whereas their performance on Korean drops to 58-69%. More strikingly, models select Unlikely relationships in 10-25% of their responses. Furthermore, we find that thinking models and chain-of-thought prompting, effective for general reasoning, provide minimal benefits for social reasoning and occasionally amplify social biases. Our findings reveal significant limitations in current LLMs' social reasoning capabilities, highlighting the need for efforts to develop socially-aware language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22333v1">LIFT: Interpretable truck driving risk prediction with literature-informed fine-tuned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-25
    </div>
    <details class="paper-abstract">
      This study proposes an interpretable prediction framework with literature-informed fine-tuned (LIFT) LLMs for truck driving risk prediction. The framework integrates an LLM-driven Inference Core that predicts and explains truck driving risk, a Literature Processing Pipeline that filters and summarizes domain-specific literature into a literature knowledge base, and a Result Evaluator that evaluates the prediction performance as well as the interpretability of the LIFT LLM. After fine-tuning on a real-world truck driving risk dataset, the LIFT LLM achieved accurate risk prediction, outperforming benchmark models by 26.7% in recall and 10.1% in F1-score. Furthermore, guided by the literature knowledge base automatically constructed from 299 domain papers, the LIFT LLM produced variable importance ranking consistent with that derived from the benchmark model, while demonstrating robustness in interpretation results to various data sampling conditions. The LIFT LLM also identified potential risky scenarios by detecting key combination of variables in truck driving risk, which were verified by PERMANOVA tests. Finally, we demonstrated the contribution of the literature knowledge base and the fine-tuning process in the interpretability of the LIFT LLM, and discussed the potential of the LIFT LLM in data-driven knowledge discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18971v2">LLMs as Planning Formalizers: A Survey for Leveraging Large Language Models to Construct Automated Planning Models</a></div>
    <div class="paper-meta">
      📅 2025-10-25
      | 💬 20 pages, 3 figures, 3 appendices
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel in various natural language tasks but often struggle with long-horizon planning problems requiring structured reasoning. This limitation has drawn interest in integrating neuro-symbolic approaches within the Automated Planning (AP) and Natural Language Processing (NLP) communities. However, identifying optimal AP deployment frameworks can be daunting and introduces new challenges. This paper aims to provide a timely survey of the current research with an in-depth analysis, positioning LLMs as tools for formalizing and refining planning specifications to support reliable off-the-shelf AP planners. By systematically reviewing the current state of research, we highlight methodologies, and identify critical challenges and future directions, hoping to contribute to the joint research on NLP and Automated Planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07759v2">A Survey of Long-Document Retrieval in the PLM and LLM Era</a></div>
    <div class="paper-meta">
      📅 2025-10-25
      | 💬 32 pages, 6 figures
    </div>
    <details class="paper-abstract">
      The proliferation of long-form documents presents a fundamental challenge to information retrieval (IR), as their length, dispersed evidence, and complex structures demand specialized methods beyond standard passage-level techniques. This survey provides the first comprehensive treatment of long-document retrieval (LDR), consolidating methods, challenges, and applications across three major eras. We systematize the evolution from classical lexical and early neural models to modern pre-trained (PLM) and large language models (LLMs), covering key paradigms like passage aggregation, hierarchical encoding, efficient attention, and the latest LLM-driven re-ranking and retrieval techniques. Beyond the models, we review domain-specific applications, specialized evaluation resources, and outline critical open challenges such as efficiency trade-offs, multimodal alignment, and faithfulness. This survey aims to provide both a consolidated reference and a forward-looking agenda for advancing long-document retrieval in the era of foundation models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01013v3">TimeXL: Explainable Multi-modal Time Series Prediction with LLM-in-the-Loop</a></div>
    <div class="paper-meta">
      📅 2025-10-25
    </div>
    <details class="paper-abstract">
      Time series analysis provides essential insights for real-world system dynamics and informs downstream decision-making, yet most existing methods often overlook the rich contextual signals present in auxiliary modalities. To bridge this gap, we introduce TimeXL, a multi-modal prediction framework that integrates a prototype-based time series encoder with three collaborating Large Language Models (LLMs) to deliver more accurate predictions and interpretable explanations. First, a multi-modal prototype-based encoder processes both time series and textual inputs to generate preliminary forecasts alongside case-based rationales. These outputs then feed into a prediction LLM, which refines the forecasts by reasoning over the encoder's predictions and explanations. Next, a reflection LLM compares the predicted values against the ground truth, identifying textual inconsistencies or noise. Guided by this feedback, a refinement LLM iteratively enhances text quality and triggers encoder retraining. This closed-loop workflow-prediction, critique (reflect), and refinement-continuously boosts the framework's performance and interpretability. Empirical evaluations on four real-world datasets demonstrate that TimeXL achieves up to 8.9% improvement in AUC and produces human-centric, multi-modal explanations, highlighting the power of LLM-driven reasoning for time series prediction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22285v1">Supervised Fine-Tuning or In-Context Learning? Evaluating LLMs for Clinical NER</a></div>
    <div class="paper-meta">
      📅 2025-10-25
      | 💬 Work done in November - December 2024
    </div>
    <details class="paper-abstract">
      We study clinical Named Entity Recognition (NER) on the CADEC corpus and compare three families of approaches: (i) BERT-style encoders (BERT Base, BioClinicalBERT, RoBERTa-large), (ii) GPT-4o used with few-shot in-context learning (ICL) under simple vs.\ complex prompts, and (iii) GPT-4o with supervised fine-tuning (SFT). All models are evaluated on standard NER metrics over CADEC's five entity types (ADR, Drug, Disease, Symptom, Finding). RoBERTa-large and BioClinicalBERT offer limited improvements over BERT Base, showing the limit of these family of models. Among LLM settings, simple ICL outperforms a longer, instruction-heavy prompt, and SFT achieves the strongest overall performance (F1 $\approx$ 87.1%), albeit with higher cost. We find that the LLM achieve higher accuracy on simplified tasks, restricting classification to two labels.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01586v3">SubTrack++ : Gradient Subspace Tracking for Scalable LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-10-25
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) is highly resource-intensive due to their massive number of parameters and the overhead of optimizer states. While recent work has aimed to reduce memory consumption, such efforts often entail trade-offs among memory efficiency, training time, and model performance. Yet, true democratization of LLMs requires simultaneous progress across all three dimensions. To this end, we propose SubTrack++ that leverages Grassmannian gradient subspace tracking combined with projection-aware optimizers, enabling Adam's internal statistics to adapt to subspace changes. Additionally, employing recovery scaling, a technique that restores information lost through low-rank projections, further enhances model performance. Our method demonstrates SOTA convergence by exploiting Grassmannian geometry, reducing pre-training wall-time by up to 65% and fine-tuning time by 36% compared to existing SOTA methods, while maintaining the same memory footprint.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10644v2">Hierarchical Optimization via LLM-Guided Objective Evolution for Mobility-on-Demand Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-25
    </div>
    <details class="paper-abstract">
      Online ride-hailing platforms aim to deliver efficient mobility-on-demand services, often facing challenges in balancing dynamic and spatially heterogeneous supply and demand. Existing methods typically fall into two categories: reinforcement learning (RL) approaches, which suffer from data inefficiency, oversimplified modeling of real-world dynamics, and difficulty enforcing operational constraints; or decomposed online optimization methods, which rely on manually designed high-level objectives that lack awareness of low-level routing dynamics. To address this issue, we propose a novel hybrid framework that integrates large language model (LLM) with mathematical optimization in a dynamic hierarchical system: (1) it is training-free, removing the need for large-scale interaction data as in RL, and (2) it leverages LLM to bridge cognitive limitations caused by problem decomposition by adaptively generating high-level objectives. Within this framework, LLM serves as a meta-optimizer, producing semantic heuristics that guide a low-level optimizer responsible for constraint enforcement and real-time decision execution. These heuristics are refined through a closed-loop evolutionary process, driven by harmony search, which iteratively adapts the LLM prompts based on feasibility and performance feedback from the optimization layer. Extensive experiments based on scenarios derived from both the New York and Chicago taxi datasets demonstrate the effectiveness of our approach, achieving an average improvement of 16% compared to state-of-the-art baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00882v4">VulSolver: Vulnerability Detection via LLM-Driven Constraint Solving</a></div>
    <div class="paper-meta">
      📅 2025-10-25
    </div>
    <details class="paper-abstract">
      Traditional vulnerability detection methods rely heavily on predefined rule matching, which often fails to capture vulnerabilities accurately. With the rise of large language models (LLMs), leveraging their ability to understand code semantics has emerged as a promising direction for achieving more accurate and efficient vulnerability detection. However, current LLM-based approaches face significant challenges: instability in model outputs, limitations in context length, and hallucination. As a result, many existing solutions either use LLMs merely to enrich predefined rule sets, thereby keeping the detection process fundamentally rule-based, or over-rely on them, leading to poor robustness. To address these challenges, we propose a constraint-solving approach powered by LLMs named VULSOLVER. By modeling vulnerability detection as a constraint-solving problem, and by integrating static application security testing (SAST) with the semantic reasoning capabilities of LLMs, our method enables the LLM to act like a professional human security expert. We assess VULSOLVER on the OWASP Benchmark (1,023 labeled samples), achieving 97.85% accuracy, 97.97% F1-score, and 100% recall. Applied to popular GitHub repositories, VULSOLVER also identified 15 previously unknown high-severity vulnerabilities (CVSS 7.5-9.8), demonstrating its effectiveness in real-world security analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22256v1">SteerX: Disentangled Steering for LLM Personalization</a></div>
    <div class="paper-meta">
      📅 2025-10-25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable success in recent years, enabling a wide range of applications, including intelligent assistants that support users' daily life and work. A critical factor in building such assistants is personalizing LLMs, as user preferences and needs vary widely. Activation steering, which directly leverages directions representing user preference in the LLM activation space to adjust its behavior, offers a cost-effective way to align the model's outputs with individual users. However, existing methods rely on all historical data to compute the steering vector, ignoring that not all content reflects true user preferences, which undermines the personalization signal. To address this, we propose SteerX, a disentangled steering method that isolates preference-driven components from preference-agnostic components. Grounded in causal inference theory, SteerX estimates token-level causal effects to identify preference-driven tokens, transforms these discrete signals into a coherent description, and then leverages them to steer personalized LLM generation. By focusing on the truly preference-driven information, SteerX produces more accurate activation steering vectors and enhances personalization. Experiments on two representative steering backbone methods across real-world datasets demonstrate that SteerX consistently enhances steering vector quality, offering a practical solution for more effective LLM personalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22255v1">PACR: Progressively Ascending Confidence Reward for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-25
      | 💬 16 pages, 14 figures
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Rewards (RLVR) has significantly improved LLM reasoning, but its sparse, outcome-based reward provides no guidance for intermediate steps, slowing exploration. We propose Progressively Ascending Confidence Reward (PACR), a dense, model-intrinsic reward computed directly from the model's evolving belief in the correct answer. PACR encodes the inductive bias that, along a well-formed reasoning trajectory, the probability of the ground-truth answer should have a generally ascending trend. We provide empirical and theoretical analysis validating that such an inductive bias constrains the exploration search space to regions richer in logically sound reasoning. We demonstrate that PACR accelerates exploration, reaches reward saturation with fewer trajectories, and yields improvements on multiple benchmarks. Our results suggest that dense, model-intrinsic shaping signals can make RLVR training more effective and reliable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23650v1">Beyond Hidden-Layer Manipulation: Semantically-Aware Logit Interventions for Debiasing LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-25
    </div>
    <details class="paper-abstract">
      We proposed Static and Dynamic -- two zero-shot logits-layer debiasing methods. Dynamic reduces bias by up to 70% with minimal fluency loss. Logits intervention outperforms hidden-layer approaches. We show semantic-aware logits intervention is stable and effective for debiasing aligned LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16492v2">Check Yourself Before You Wreck Yourself: Selectively Quitting Improves LLM Agent Safety</a></div>
    <div class="paper-meta">
      📅 2025-10-25
      | 💬 Reliable ML and Regulatable ML workshops, Neurips 2025
    </div>
    <details class="paper-abstract">
      As Large Language Model (LLM) agents increasingly operate in complex environments with real-world consequences, their safety becomes critical. While uncertainty quantification is well-studied for single-turn tasks, multi-turn agentic scenarios with real-world tool access present unique challenges where uncertainties and ambiguities compound, leading to severe or catastrophic risks beyond traditional text generation failures. We propose using "quitting" as a simple yet effective behavioral mechanism for LLM agents to recognize and withdraw from situations where they lack confidence. Leveraging the ToolEmu framework, we conduct a systematic evaluation of quitting behavior across 12 state-of-the-art LLMs. Our results demonstrate a highly favorable safety-helpfulness trade-off: agents prompted to quit with explicit instructions improve safety by an average of +0.39 on a 0-3 scale across all models (+0.64 for proprietary models), while maintaining a negligible average decrease of -0.03 in helpfulness. Our analysis demonstrates that simply adding explicit quit instructions proves to be a highly effective safety mechanism that can immediately be deployed in existing agent systems, and establishes quitting as an effective first-line defense mechanism for autonomous agents in high-stakes applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22242v1">PaperAsk: A Benchmark for Reliability Evaluation of LLMs in Paper Search and Reading</a></div>
    <div class="paper-meta">
      📅 2025-10-25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) increasingly serve as research assistants, yet their reliability in scholarly tasks remains under-evaluated. In this work, we introduce PaperAsk, a benchmark that systematically evaluates LLMs across four key research tasks: citation retrieval, content extraction, paper discovery, and claim verification. We evaluate GPT-4o, GPT-5, and Gemini-2.5-Flash under realistic usage conditions-via web interfaces where search operations are opaque to the user. Through controlled experiments, we find consistent reliability failures: citation retrieval fails in 48-98% of multi-reference queries, section-specific content extraction fails in 72-91% of cases, and topical paper discovery yields F1 scores below 0.32, missing over 60% of relevant literature. Further human analysis attributes these failures to the uncontrolled expansion of retrieved context and the tendency of LLMs to prioritize semantically relevant text over task instructions. Across basic tasks, the LLMs display distinct failure behaviors: ChatGPT often withholds responses rather than risk errors, whereas Gemini produces fluent but fabricated answers. To address these issues, we develop lightweight reliability classifiers trained on PaperAsk data to identify unreliable outputs. PaperAsk provides a reproducible and diagnostic framework for advancing the reliability evaluation of LLM-based scholarly assistance systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12710v2">Enhancing Naturalness in LLM-Generated Utterances through Disfluency Insertion</a></div>
    <div class="paper-meta">
      📅 2025-10-25
      | 💬 8 pages. Limitations, ethical considerations, and references are additional
    </div>
    <details class="paper-abstract">
      Disfluencies are a natural feature of spontaneous human speech but are typically absent from the outputs of Large Language Models (LLMs). This absence can diminish the perceived naturalness of synthesized speech, which is an important criteria when building conversational agents that aim to mimick human behaviours. We show how the insertion of disfluencies can alleviate this shortcoming. The proposed approach involves (1) fine-tuning an LLM with Low-Rank Adaptation (LoRA) to incorporate various types of disfluencies into LLM-generated utterances and (2) synthesizing those utterances using a text-to-speech model that supports the generation of speech phenomena such as disfluencies. We evaluated the quality of the generated speech across two metrics: intelligibility and perceived spontaneity. We demonstrate through a user study that the insertion of disfluencies significantly increase the perceived spontaneity of the generated speech. This increase came, however, along with a slight reduction in intelligibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22228v1">When Fewer Layers Break More Chains: Layer Pruning Harms Test-Time Scaling in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-25
    </div>
    <details class="paper-abstract">
      Layer pruning has emerged as a widely adopted technique for improving the efficiency of large language models (LLMs). Although existing methods demonstrate strong performance retention on general knowledge tasks, their effect on long-chain reasoning, a more brittle yet crucial capability, remains largely unexplored. In this work, we study the impact of layer pruning on long-chain reasoning through the lens of test-time scaling, a key mechanism in modern LLMs that enables strong reasoning capacity by allocating more computation at inference time. With extensive experiments, we demonstrate that pruning even one or two layers can severely impair test-time scaling, with performance collapsing drastically on long reasoning benchmarks even when performance on knowledge-intensive and shallow reasoning tasks remains stable. Furthermore, we find that standard supervised fine-tuning remedies fail to recover test-time scaling once it has deteriorated. Through in-depth analyses, we identify the mechanisms underlying this fragility of test-time scaling and highlight the fundamental risks of applying layer pruning to reasoning-intensive LLMs. These findings call for a rethinking of layer pruning strategies and provide insights for developing methods that preserve the robustness of reasoning. We open-source the codebase in \href{https://github.com/keyu-wang-2002/Layer-Pruning-Harms-Inference-Scaling}{https://github.com/keyu-wang-2002/Layer-Pruning-Harms-Inference-Scaling}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15721v3">Bohdi: Heterogeneous LLM Fusion with Automatic Data Exploration</a></div>
    <div class="paper-meta">
      📅 2025-10-25
      | 💬 Accepted by NeurIPS2025
    </div>
    <details class="paper-abstract">
      Heterogeneous Large Language Model (LLM) fusion integrates the strengths of multiple source LLMs with different architectures into a target LLM with low computational overhead. While promising, existing methods suffer from two major limitations: 1) reliance on real data from limited domain for knowledge fusion, preventing the target LLM from fully acquiring knowledge across diverse domains, and 2) fixed data allocation proportions across domains, failing to dynamically adjust according to the target LLM's varying capabilities across domains, leading to a capability imbalance. To overcome these limitations, we propose Bohdi, a synthetic-data-only heterogeneous LLM fusion framework. Through the organization of knowledge domains into a hierarchical tree structure, Bohdi enables automatic domain exploration and multi-domain data generation through multi-model collaboration, thereby comprehensively extracting knowledge from source LLMs. By formalizing domain expansion and data sampling proportion allocation on the knowledge tree as a Hierarchical Multi-Armed Bandit problem, Bohdi leverages the designed DynaBranches mechanism to adaptively adjust sampling proportions based on the target LLM's performance feedback across domains. Integrated with our proposed Introspection-Rebirth (IR) mechanism, DynaBranches dynamically tracks capability shifts during target LLM's updates via Sliding Window Binomial Likelihood Ratio Testing (SWBLRT), further enhancing its online adaptation capability. Comparative experimental results on a comprehensive suite of benchmarks demonstrate that Bohdi significantly outperforms existing baselines on multiple target LLMs, exhibits higher data efficiency, and virtually eliminates the imbalance in the target LLM's capabilities. Our code is available at https://github.com/gjq100/Bohdi.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22192v1">OptiTree: Hierarchical Thoughts Generation with Tree Search for LLM Optimization Modeling</a></div>
    <div class="paper-meta">
      📅 2025-10-25
      | 💬 Published at NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Optimization modeling is one of the most crucial but technical parts of operations research (OR). To automate the modeling process, existing works have leveraged large language models (LLMs), prompting them to break down tasks into steps for generating variables, constraints, and objectives. However, due to the highly complex mathematical structures inherent in OR problems, standard fixed-step decomposition often fails to achieve high performance. To address this challenge, we introduce OptiTree, a novel tree search approach designed to enhance modeling capabilities for complex problems through adaptive problem decomposition into simpler subproblems. Specifically, we develop a modeling tree that organizes a wide range of OR problems based on their hierarchical problem taxonomy and complexity, with each node representing a problem category and containing relevant high-level modeling thoughts. Given a problem to model, we recurrently search the tree to identify a series of simpler subproblems and synthesize the global modeling thoughts by adaptively integrating the hierarchical thoughts. Experiments show that OptiTree significantly improves the modeling accuracy compared to the state-of-the-art, achieving over 10\% improvements on the challenging benchmarks. The code is released at https://github.com/MIRALab-USTC/OptiTree/tree/main.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20501v2">Gatsby Without the 'E': Crafting Lipograms with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-25
    </div>
    <details class="paper-abstract">
      Lipograms are a unique form of constrained writing where all occurrences of a particular letter are excluded from the text, typified by the novel Gadsby, which daringly avoids all usage of the letter 'e'. In this study, we explore the power of modern large language models (LLMs) by transforming the novel F. Scott Fitzgerald's The Great Gatsby into a fully 'e'-less text. We experimented with a range of techniques, from baseline methods like synonym replacement to sophisticated generative models enhanced with beam search and named entity analysis. We show that excluding up to 3.6% of the most common letters (up to the letter 'u') had minimal impact on the text's meaning, although translation fidelity rapidly and predictably decays with stronger lipogram constraints. Our work highlights the surprising flexibility of English under strict constraints, revealing just how adaptable and creative language can be.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22139v1">Edit Less, Achieve More: Dynamic Sparse Neuron Masking for Lifelong Knowledge Editing in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-25
      | 💬 19 pages, 11 figures, Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Lifelong knowledge editing enables continuous, precise updates to outdated knowledge in large language models (LLMs) without computationally expensive full retraining. However, existing methods often accumulate errors throughout the editing process, causing a gradual decline in both editing accuracy and generalization. To tackle this problem, we propose Neuron-Specific Masked Knowledge Editing (NMKE), a novel fine-grained editing framework that combines neuron-level attribution with dynamic sparse masking. Leveraging neuron functional attribution, we identify two key types of knowledge neurons, with knowledge-general neurons activating consistently across prompts and knowledge-specific neurons activating to specific prompts. NMKE further introduces an entropy-guided dynamic sparse mask, locating relevant neurons to the target knowledge. This strategy enables precise neuron-level knowledge editing with fewer parameter modifications. Experimental results from thousands of sequential edits demonstrate that NMKE outperforms existing methods in maintaining high editing success rates and preserving model general capabilities in lifelong editing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22126v1">EasyUUV: An LLM-Enhanced Universal and Lightweight Sim-to-Real Reinforcement Learning Framework for UUV Attitude Control</a></div>
    <div class="paper-meta">
      📅 2025-10-25
      | 💬 8 pages, 15 figures
    </div>
    <details class="paper-abstract">
      Despite recent advances in Unmanned Underwater Vehicle (UUV) attitude control, existing methods still struggle with generalizability, robustness to real-world disturbances, and efficient deployment. To address the above challenges, this paper presents EasyUUV, a Large Language Model (LLM)-enhanced, universal, and lightweight simulation-to-reality reinforcement learning (RL) framework for robust attitude control of UUVs. EasyUUV combines parallelized RL training with a hybrid control architecture, where a learned policy outputs high-level attitude corrections executed by an adaptive S-Surface controller. A multimodal LLM is further integrated to adaptively tune controller parameters at runtime using visual and textual feedback, enabling training-free adaptation to unmodeled dynamics. Also, we have developed a low-cost 6-DoF UUV platform and applied an RL policy trained through efficient parallelized simulation. Extensive simulation and real-world experiments validate the effectiveness and outstanding performance of EasyUUV in achieving robust and adaptive UUV attitude control across diverse underwater conditions. The source code is available from the following website: https://360zmem.github.io/easyuuv/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07458v3">Populism Meets AI: Advancing Populism Research with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-25
      | 💬 27 pages, 3 figures. Preprint version under review
    </div>
    <details class="paper-abstract">
      Measuring the ideational content of populism remains a challenge. Traditional strategies based on textual analysis have been critical for building the field's foundations and providing a valid, objective indicator of populist framing. Yet these approaches are costly, time consuming, and difficult to scale across languages, contexts, and large corpora. Here we present the results from a rubric and anchor guided chain of thought (CoT) prompting approach that mirrors human coder training. By leveraging the Global Populism Database (GPD), a comprehensive dataset of global leaders' speeches annotated for degrees of populism, we replicate the process used to train human coders by prompting the LLM with an adapted version of the same documentation to guide the model's reasoning. We then test multiple proprietary and open weight models by replicating scores in the GPD. Our findings reveal that this domain specific prompting strategy enables the LLM to achieve classification accuracy on par with expert human coders, demonstrating its ability to navigate the nuanced, context sensitive aspects of populism.
    </details>
</div>
