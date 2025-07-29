# llm - 2025_07

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21028v1">Multi-Agent-as-Judge: Aligning LLM-Agent-Based Automated Evaluation with Multi-Dimensional Human Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-07-28
    </div>
    <details class="paper-abstract">
      Nearly all human work is collaborative; thus, the evaluation of real-world NLP applications often requires multiple dimensions that align with diverse human perspectives. As real human evaluator resources are often scarce and costly, the emerging "LLM-as-a-judge" paradigm sheds light on a promising approach to leverage LLM agents to believably simulate human evaluators. Yet, to date, existing LLM-as-a-judge approaches face two limitations: persona descriptions of agents are often arbitrarily designed, and the frameworks are not generalizable to other tasks. To address these challenges, we propose MAJ-EVAL, a Multi-Agent-as-Judge evaluation framework that can automatically construct multiple evaluator personas with distinct dimensions from relevant text documents (e.g., research papers), instantiate LLM agents with the personas, and engage in-group debates with multi-agents to Generate multi-dimensional feedback. Our evaluation experiments in both the educational and medical domains demonstrate that MAJ-EVAL can generate evaluation results that better align with human experts' ratings compared with conventional automated evaluation metrics and existing LLM-as-a-judge methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12854v3">Enhancing LLM Reasoning with Iterative DPO: A Comprehensive Empirical Investigation</a></div>
    <div class="paper-meta">
      📅 2025-07-28
      | 💬 23pages
    </div>
    <details class="paper-abstract">
      Recent advancements in post-training methodologies for large language models (LLMs) have highlighted reinforcement learning (RL) as a critical component for enhancing reasoning. However, the substantial computational costs associated with RL-based approaches have led to growing interest in alternative paradigms, such as Direct Preference Optimization (DPO). In this study, we investigate the effectiveness of DPO in facilitating self-improvement for LLMs through iterative preference-based learning. We demonstrate that a single round of DPO with coarse filtering significantly enhances mathematical reasoning performance, particularly for strong base model. Furthermore, we design an iterative enhancement framework for both the generator and the reward model (RM), enabling their mutual improvement through online interaction across multiple rounds of DPO. Finally, with simple verifiable rewards, our model DPO-VP achieves RL-level performance with significantly lower computational overhead. These findings highlight DPO as a scalable and cost-effective alternative to RL, offering a practical solution for enhancing LLM reasoning in resource-constrained situations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21017v1">MIRAGE-Bench: LLM Agent is Hallucinating and Where to Find Them</a></div>
    <div class="paper-meta">
      📅 2025-07-28
      | 💬 Code and data: https://github.com/sunblaze-ucb/mirage-bench.git
    </div>
    <details class="paper-abstract">
      Hallucinations pose critical risks for large language model (LLM)-based agents, often manifesting as hallucinative actions resulting from fabricated or misinterpreted information within the cognitive context. While recent studies have exposed such failures, existing evaluations remain fragmented and lack a principled testbed. In this paper, we present MIRAGE-Bench--Measuring Illusions in Risky AGEnt settings--the first unified benchmark for eliciting and evaluating hallucinations in interactive LLM-agent scenarios. We begin by introducing a three-part taxonomy to address agentic hallucinations: actions that are unfaithful to (i) task instructions, (ii) execution history, or (iii) environment observations. To analyze, we first elicit such failures by performing a systematic audit of existing agent benchmarks, then synthesize test cases using a snapshot strategy that isolates decision points in deterministic and reproducible manners. To evaluate hallucination behaviors, we adopt a fine-grained-level LLM-as-a-Judge paradigm with tailored risk-aware prompts, enabling scalable, high-fidelity assessment of agent actions without enumerating full action spaces. MIRAGE-Bench provides actionable insights on failure modes of LLM agents and lays the groundwork for principled progress in mitigating hallucinations in interactive environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02087v2">Evaluating the Promise and Pitfalls of LLMs in Hiring Decisions</a></div>
    <div class="paper-meta">
      📅 2025-07-28
      | 💬 10 pages, 2 figures, 2 tables. Submitted to NeurIPS 2025
    </div>
    <details class="paper-abstract">
      The use of large language models (LLMs) in hiring promises to streamline candidate screening, but it also raises serious concerns regarding accuracy and algorithmic bias where sufficient safeguards are not in place. In this work, we benchmark several state-of-the-art foundational LLMs - including models from OpenAI, Anthropic, Google, Meta, and Deepseek, and compare them with our proprietary domain-specific hiring model (Match Score) for job candidate matching. We evaluate each model's predictive accuracy (ROC AUC, Precision-Recall AUC, F1-score) and fairness (impact ratio of cut-off analysis across declared gender, race, and intersectional subgroups). Our experiments on a dataset of roughly 10,000 real-world recent candidate-job pairs show that Match Score outperforms the general-purpose LLMs on accuracy (ROC AUC 0.85 vs 0.77) and achieves significantly more equitable outcomes across demographic groups. Notably, Match Score attains a minimum race-wise impact ratio of 0.957 (near-parity), versus 0.809 or lower for the best LLMs, (0.906 vs 0.773 for the intersectionals, respectively). We discuss why pretraining biases may cause LLMs with insufficient safeguards to propagate societal biases in hiring scenarios, whereas a bespoke supervised model can more effectively mitigate these biases. Our findings highlight the importance of domain-specific modeling and bias auditing when deploying AI in high-stakes domains such as hiring, and caution against relying on off-the-shelf LLMs for such tasks without extensive fairness safeguards. Furthermore, we show with empirical evidence that there shouldn't be a dichotomy between choosing accuracy and fairness in hiring: a well-designed algorithm can achieve both accuracy in hiring and fairness in outcomes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20999v1">LoRA-PAR: A Flexible Dual-System LoRA Partitioning Approach to Efficient LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-07-28
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Large-scale generative models like DeepSeek-R1 and OpenAI-O1 benefit substantially from chain-of-thought (CoT) reasoning, yet pushing their performance typically requires vast data, large model sizes, and full-parameter fine-tuning. While parameter-efficient fine-tuning (PEFT) helps reduce cost, most existing approaches primarily address domain adaptation or layer-wise allocation rather than explicitly tailoring data and parameters to different response demands. Inspired by "Thinking, Fast and Slow," which characterizes two distinct modes of thought-System 1 (fast, intuitive, often automatic) and System 2 (slower, more deliberative and analytic)-we draw an analogy that different "subregions" of an LLM's parameters might similarly specialize for tasks that demand quick, intuitive responses versus those requiring multi-step logical reasoning. Therefore, we propose LoRA-PAR, a dual-system LoRA framework that partitions both data and parameters by System 1 or System 2 demands, using fewer yet more focused parameters for each task. Specifically, we classify task data via multi-model role-playing and voting, and partition parameters based on importance scoring, then adopt a two-stage fine-tuning strategy of training System 1 tasks with supervised fine-tuning (SFT) to enhance knowledge and intuition and refine System 2 tasks with reinforcement learning (RL) to reinforce deeper logical deliberation next. Extensive experiments show that the two-stage fine-tuning strategy, SFT and RL, lowers active parameter usage while matching or surpassing SOTA PEFT baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20977v1">Repairing vulnerabilities without invisible hands. A differentiated replication study on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-28
    </div>
    <details class="paper-abstract">
      Background: Automated Vulnerability Repair (AVR) is a fast-growing branch of program repair. Recent studies show that large language models (LLMs) outperform traditional techniques, extending their success beyond code generation and fault detection. Hypothesis: These gains may be driven by hidden factors -- "invisible hands" such as training-data leakage or perfect fault localization -- that let an LLM reproduce human-authored fixes for the same code. Objective: We replicate prior AVR studies under controlled conditions by deliberately adding errors to the reported vulnerability location in the prompt. If LLMs merely regurgitate memorized fixes, both small and large localization errors should yield the same number of correct patches, because any offset should divert the model from the original fix. Method: Our pipeline repairs vulnerabilities from the Vul4J and VJTrans benchmarks after shifting the fault location by n lines from the ground truth. A first LLM generates a patch, a second LLM reviews it, and we validate the result with regression and proof-of-vulnerability tests. Finally, we manually audit a sample of patches and estimate the error rate with the Agresti-Coull-Wilson method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17137v2">Cog-TiPRO: Iterative Prompt Refinement with LLMs to Detect Cognitive Decline via Longitudinal Voice Assistant Commands</a></div>
    <div class="paper-meta">
      📅 2025-07-28
      | 💬 IEEE Global Communications Conference (GlobeCom) 2025
    </div>
    <details class="paper-abstract">
      Early detection of cognitive decline is crucial for enabling interventions that can slow neurodegenerative disease progression. Traditional diagnostic approaches rely on labor-intensive clinical assessments, which are impractical for frequent monitoring. Our pilot study investigates voice assistant systems (VAS) as non-invasive tools for detecting cognitive decline through longitudinal analysis of speech patterns in voice commands. Over an 18-month period, we collected voice commands from 35 older adults, with 15 participants providing daily at-home VAS interactions. To address the challenges of analyzing these short, unstructured and noisy commands, we propose Cog-TiPRO, a framework that combines (1) LLM-driven iterative prompt refinement for linguistic feature extraction, (2) HuBERT-based acoustic feature extraction, and (3) transformer-based temporal modeling. Using iTransformer, our approach achieves 73.80% accuracy and 72.67% F1-score in detecting MCI, outperforming its baseline by 27.13%. Through our LLM approach, we identify linguistic features that uniquely characterize everyday command usage patterns in individuals experiencing cognitive decline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20957v1">Your AI, Not Your View: The Bias of LLMs in Investment Analysis</a></div>
    <div class="paper-meta">
      📅 2025-07-28
    </div>
    <details class="paper-abstract">
      In finance, Large Language Models (LLMs) face frequent knowledge conflicts due to discrepancies between pre-trained parametric knowledge and real-time market data. These conflicts become particularly problematic when LLMs are deployed in real-world investment services, where misalignment between a model's embedded preferences and those of the financial institution can lead to unreliable recommendations. Yet little research has examined what investment views LLMs actually hold. We propose an experimental framework to investigate such conflicts, offering the first quantitative analysis of confirmation bias in LLM-based investment analysis. Using hypothetical scenarios with balanced and imbalanced arguments, we extract models' latent preferences and measure their persistence. Focusing on sector, size, and momentum, our analysis reveals distinct, model-specific tendencies. In particular, we observe a consistent preference for large-cap stocks and contrarian strategies across most models. These preferences often harden into confirmation bias, with models clinging to initial judgments despite counter-evidence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16117v2">BDIViz: An Interactive Visualization System for Biomedical Schema Matching with LLM-Powered Validation</a></div>
    <div class="paper-meta">
      📅 2025-07-28
      | 💬 11 pages, 9 figures. Accepted to IEEE VIS 2025 (Full Papers Track, submission ID 1204)
    </div>
    <details class="paper-abstract">
      Biomedical data harmonization is essential for enabling exploratory analyses and meta-studies, but the process of schema matching - identifying semantic correspondences between elements of disparate datasets (schemas) - remains a labor-intensive and error-prone task. Even state-of-the-art automated methods often yield low accuracy when applied to biomedical schemas due to the large number of attributes and nuanced semantic differences between them. We present BDIViz, a novel visual analytics system designed to streamline the schema matching process for biomedical data. Through formative studies with domain experts, we identified key requirements for an effective solution and developed interactive visualization techniques that address both scalability challenges and semantic ambiguity. BDIViz employs an ensemble approach that combines multiple matching methods with LLM-based validation, summarizes matches through interactive heatmaps, and provides coordinated views that enable users to quickly compare attributes and their values. Our method-agnostic design allows the system to integrate various schema matching algorithms and adapt to application-specific needs. Through two biomedical case studies and a within-subject user study with domain experts, we demonstrate that BDIViz significantly improves matching accuracy while reducing cognitive load and curation time compared to baseline approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20870v1">A Human-in-the-loop Approach to Robot Action Replanning through LLM Common-Sense Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-07-28
    </div>
    <details class="paper-abstract">
      To facilitate the wider adoption of robotics, accessible programming tools are required for non-experts. Observational learning enables intuitive human skills transfer through hands-on demonstrations, but relying solely on visual input can be inefficient in terms of scalability and failure mitigation, especially when based on a single demonstration. This paper presents a human-in-the-loop method for enhancing the robot execution plan, automatically generated based on a single RGB video, with natural language input to a Large Language Model (LLM). By including user-specified goals or critical task aspects and exploiting the LLM common-sense reasoning, the system adjusts the vision-based plan to prevent potential failures and adapts it based on the received instructions. Experiments demonstrated the framework intuitiveness and effectiveness in correcting vision-derived errors and adapting plans without requiring additional demonstrations. Moreover, interactive plan refinement and hallucination corrections promoted system robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20849v1">Latent Inter-User Difference Modeling for LLM Personalization</a></div>
    <div class="paper-meta">
      📅 2025-07-28
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly integrated into users' daily lives, leading to a growing demand for personalized outputs. Previous work focuses on leveraging a user's own history, overlooking inter-user differences that are crucial for effective personalization. While recent work has attempted to model such differences, the reliance on language-based prompts often hampers the effective extraction of meaningful distinctions. To address these issues, we propose Difference-aware Embedding-based Personalization (DEP), a framework that models inter-user differences in the latent space instead of relying on language prompts. DEP constructs soft prompts by contrasting a user's embedding with those of peers who engaged with similar content, highlighting relative behavioral signals. A sparse autoencoder then filters and compresses both user-specific and difference-aware embeddings, preserving only task-relevant features before injecting them into a frozen LLM. Experiments on personalized review generation show that DEP consistently outperforms baseline methods across multiple metrics. Our code is available at https://github.com/SnowCharmQ/DEP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00582v2">Do Language Models Mirror Human Confidence? Exploring Psychological Insights to Address Overconfidence in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-28
      | 💬 Accepted by ACL 2025 Findings, 20 pages
    </div>
    <details class="paper-abstract">
      Psychology research has shown that humans are poor at estimating their performance on tasks, tending towards underconfidence on easy tasks and overconfidence on difficult tasks. We examine three LLMs, Llama-3-70B-instruct, Claude-3-Sonnet, and GPT-4o, on a range of QA tasks of varying difficulty, and show that models exhibit subtle differences from human patterns of overconfidence: less sensitive to task difficulty, and when prompted to answer based on different personas -- e.g., expert vs layman, or different race, gender, and ages -- the models will respond with stereotypically biased confidence estimations even though their underlying answer accuracy remains the same. Based on these observations, we propose Answer-Free Confidence Estimation (AFCE) to improve confidence calibration and LLM interpretability in these settings. AFCE is a self-assessment method that employs two stages of prompting, first eliciting only confidence scores on questions, then asking separately for the answer. Experiments on the MMLU and GPQA datasets spanning subjects and difficulty show that this separation of tasks significantly reduces overconfidence and delivers more human-like sensitivity to task difficulty.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20774v1">evalSmarT: An LLM-Based Framework for Evaluating Smart Contract Generated Comments</a></div>
    <div class="paper-meta">
      📅 2025-07-28
      | 💬 4 pages, 4 tables
    </div>
    <details class="paper-abstract">
      Smart contract comment generation has gained traction as a means to improve code comprehension and maintainability in blockchain systems. However, evaluating the quality of generated comments remains a challenge. Traditional metrics such as BLEU and ROUGE fail to capture domain-specific nuances, while human evaluation is costly and unscalable. In this paper, we present \texttt{evalSmarT}, a modular and extensible framework that leverages large language models (LLMs) as evaluators. The system supports over 400 evaluator configurations by combining approximately 40 LLMs with 10 prompting strategies. We demonstrate its application in benchmarking comment generation tools and selecting the most informative outputs. Our results show that prompt design significantly impacts alignment with human judgment, and that LLM-based evaluation offers a scalable and semantically rich alternative to existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.19299v3">The Impact of LoRA Adapters on LLMs for Clinical Text Classification Under Computational and Data Constraints</a></div>
    <div class="paper-meta">
      📅 2025-07-28
      | 💬 Accepted for publication in the IEEE Access
    </div>
    <details class="paper-abstract">
      Fine-tuning Large Language Models (LLMs) for clinical Natural Language Processing (NLP) poses significant challenges due to domain gap, limited data, and stringent hardware constraints. In this study, we evaluate four adapter techniques-Adapter, Lightweight, TinyAttention, and Gated Residual Network (GRN) - equivalent to Low-Rank Adaptation (LoRA), for clinical note classification under real-world, resource-constrained conditions. All experiments were conducted on a single NVIDIA Quadro P620 GPU (2 GB VRAM, 512 CUDA cores, 1.386 TFLOPS FP32), limiting batch sizes to <8 sequences and maximum sequence length to 256 tokens. Our clinical corpus comprises only 580 000 tokens, several orders of magnitude smaller than standard LLM pre-training datasets. We fine-tuned three biomedical pre-trained LLMs (CamemBERT-bio, AliBERT, DrBERT) and two lightweight Transformer models trained from scratch. Results show that 1) adapter structures provide no consistent gains when fine-tuning biomedical LLMs under these constraints, and 2) simpler Transformers, with minimal parameter counts and training times under six hours, outperform adapter-augmented LLMs, which required over 1000 GPU-hours. Among adapters, GRN achieved the best metrics (accuracy, precision, recall, F1 = 0.88). These findings demonstrate that, in low-resource clinical settings with limited data and compute, lightweight Transformers trained from scratch offer a more practical and efficient solution than large LLMs, while GRN remains a viable adapter choice when minimal adaptation is needed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.06003v3">LUT Tensor Core: A Software-Hardware Co-Design for LUT-Based Low-Bit LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-07-28
      | 💬 Conference Version (ISCA'25). Fixed a typo
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) inference becomes resource-intensive, prompting a shift toward low-bit model weights to reduce the memory footprint and improve efficiency. Such low-bit LLMs necessitate the mixed-precision matrix multiplication (mpGEMM), an important yet underexplored operation involving the multiplication of lower-precision weights with higher-precision activations. Off-the-shelf hardware does not support this operation natively, leading to indirect, thus inefficient, dequantization-based implementations. In this paper, we study the lookup table (LUT)-based approach for mpGEMM and find that a conventional LUT implementation fails to achieve the promised gains. To unlock the full potential of LUT-based mpGEMM, we propose LUT Tensor Core, a software-hardware co-design for low-bit LLM inference. LUT Tensor Core differentiates itself from conventional LUT designs through: 1) software-based optimizations to minimize table precompute overhead and weight reinterpretation to reduce table storage; 2) a LUT-based Tensor Core hardware design with an elongated tiling shape to maximize table reuse and a bit-serial design to support diverse precision combinations in mpGEMM; 3) a new instruction set and compilation optimizations for LUT-based mpGEMM. LUT Tensor Core significantly outperforms existing pure software LUT implementations and achieves a 1.44$\times$ improvement in compute density and energy efficiency compared to previous state-of-the-art LUT-based accelerators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20674v1">LLM-Based Repair of Static Nullability Errors</a></div>
    <div class="paper-meta">
      📅 2025-07-28
    </div>
    <details class="paper-abstract">
      Modern Java projects increasingly adopt static analysis tools that prevent null-pointer exceptions by treating nullness as a type property. However, integrating such tools into large, existing codebases remains a significant challenge. While annotation inference can eliminate many errors automatically, a subset of residual errors -- typically a mix of real bugs and false positives -- often persist and can only be resolved via code changes. Manually addressing these errors is tedious and error-prone. Large language models (LLMs) offer a promising path toward automating these repairs, but naively-prompted LLMs often generate incorrect, contextually-inappropriate edits. Resolving a nullability error demands a deep understanding of how a symbol is used across the codebase, often spanning methods, classes, and packages. We present NullRepair, a system that integrates LLMs into a structured workflow for resolving the errors from a nullability checker. NullRepair's decision process follows a flowchart derived from manual analysis of 200 real-world errors. It leverages static analysis to identify safe and unsafe usage regions of symbols, using error-free usage examples to contextualize model prompts. Patches are generated through an iterative interaction with the LLM that incorporates project-wide context and decision logic. Our evaluation on 12 real-world Java projects shows that NullRepair resolves an average of 72% of the errors that remain after applying a state-of-the-art annotation inference technique. Unlike a naively-prompted LLM, NullRepair also largely preserves program semantics, with all unit tests passing in 10/12 projects after applying every edit proposed by NullRepair, and 98% or more tests passing in the remaining two projects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20666v1">MIMII-Agent: Leveraging LLMs with Function Calling for Relative Evaluation of Anomalous Sound Detection</a></div>
    <div class="paper-meta">
      📅 2025-07-28
    </div>
    <details class="paper-abstract">
      This paper proposes a method for generating machine-type-specific anomalies to evaluate the relative performance of unsupervised anomalous sound detection (UASD) systems across different machine types, even in the absence of real anomaly sound data. Conventional keyword-based data augmentation methods often produce unrealistic sounds due to their reliance on manually defined labels, limiting scalability as machine types and anomaly patterns diversify. Advanced audio generative models, such as MIMII-Gen, show promise but typically depend on anomalous training data, making them less effective when diverse anomalous examples are unavailable. To address these limitations, we propose a novel synthesis approach leveraging large language models (LLMs) to interpret textual descriptions of faults and automatically select audio transformation functions, converting normal machine sounds into diverse and plausible anomalous sounds. We validate this approach by evaluating a UASD system trained only on normal sounds from five machine types, using both real and synthetic anomaly data. Experimental results reveal consistent trends in relative detection difficulty across machine types between synthetic and real anomalies. This finding supports our hypothesis and highlights the effectiveness of the proposed LLM-based synthesis approach for relative evaluation of UASD systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20655v1">CoGrader: Transforming Instructors' Assessment of Project Reports through Collaborative LLM Integration</a></div>
    <div class="paper-meta">
      📅 2025-07-28
    </div>
    <details class="paper-abstract">
      Grading project reports are increasingly significant in today's educational landscape, where they serve as key assessments of students' comprehensive problem-solving abilities. However, it remains challenging due to the multifaceted evaluation criteria involved, such as creativity and peer-comparative achievement. Meanwhile, instructors often struggle to maintain fairness throughout the time-consuming grading process. Recent advances in AI, particularly large language models, have demonstrated potential for automating simpler grading tasks, such as assessing quizzes or basic writing quality. However, these tools often fall short when it comes to complex metrics, like design innovation and the practical application of knowledge, that require an instructor's educational insights into the class situation. To address this challenge, we conducted a formative study with six instructors and developed CoGrader, which introduces a novel grading workflow combining human-LLM collaborative metrics design, benchmarking, and AI-assisted feedback. CoGrader was found effective in improving grading efficiency and consistency while providing reliable peer-comparative feedback to students. We also discuss design insights and ethical considerations for the development of human-AI collaborative grading systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18181v2">SpecASR: Accelerating LLM-based Automatic Speech Recognition via Speculative Decoding</a></div>
    <div class="paper-meta">
      📅 2025-07-28
      | 💬 Accepted by Design Automation Conference (DAC) 2025
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based automatic speech recognition (ASR) has recently attracted a lot of attention due to its high recognition accuracy and enhanced multi-dialect support. However, the high decoding latency of LLMs challenges the real-time ASR requirements. Although speculative decoding has been explored for better decoding efficiency, they usually ignore the key characteristics of the ASR task and achieve limited speedup. To further reduce the real-time ASR latency, in this paper, we propose a novel speculative decoding framework specialized for ASR, dubbed SpecASR. SpecASR is developed based on our core observation that ASR decoding is audio-conditioned, which results in high output alignment between small and large ASR models, even given output mismatches in intermediate decoding steps. Therefore, SpecASR features an adaptive draft sequence generation process that dynamically modifies the draft sequence length to maximize the token acceptance length. SpecASR further proposes a draft sequence recycling strategy that reuses the previously generated draft sequence to reduce the draft ASR model latency. Moreover, a two-pass sparse token tree generation algorithm is also proposed to balance the latency of draft and target ASR models. With extensive experimental results, we demonstrate SpecASR achieves 3.04x-3.79x and 1.25x-1.84x speedup over the baseline autoregressive decoding and speculative decoding, respectively, without any loss in recognition accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13940v2">DrugPilot: LLM-based Parameterized Reasoning Agent for Drug Discovery</a></div>
    <div class="paper-meta">
      📅 2025-07-28
      | 💬 29 pages, 8 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) integrated with autonomous agents hold significant potential for advancing scientific discovery through automated reasoning and task execution. However, applying LLM agents to drug discovery is still constrained by challenges such as large-scale multimodal data processing, limited task automation, and poor support for domain-specific tools. To overcome these limitations, we introduce DrugPilot, a LLM-based agent system with a parameterized reasoning architecture designed for end-to-end scientific workflows in drug discovery. DrugPilot enables multi-stage research processes by integrating structured tool use with a novel parameterized memory pool. The memory pool converts heterogeneous data from both public sources and user-defined inputs into standardized representations. This design supports efficient multi-turn dialogue, reduces information loss during data exchange, and enhances complex scientific decision-making. To support training and benchmarking, we construct a drug instruction dataset covering eight core drug discovery tasks. Under the Berkeley function-calling benchmark, DrugPilot significantly outperforms state-of-the-art agents such as ReAct and LoT, achieving task completion rates of 98.0%, 93.5%, and 64.0% for simple, multi-tool, and multi-turn scenarios, respectively. These results highlight DrugPilot's potential as a versatile agent framework for computational science domains requiring automated, interactive, and data-integrated reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06102v3">SiriusBI: A Comprehensive LLM-Powered Solution for Data Analytics in Business Intelligence</a></div>
    <div class="paper-meta">
      📅 2025-07-28
      | 💬 14 pages, 8 figures
    </div>
    <details class="paper-abstract">
      With the proliferation of Large Language Models (LLMs) in Business Intelligence (BI), existing solutions face critical challenges in industrial deployments: functionality deficiencies from legacy systems failing to meet evolving LLM-era user demands, interaction limitations from single-round SQL generation paradigms inadequate for multi-round clarification, and cost for domain adaptation arising from cross-domain methods migration. We present SiriusBI, a practical LLM-powered BI system addressing the challenges of industrial deployments through three key innovations: (a) An end-to-end architecture integrating multi-module coordination to overcome functionality gaps in legacy systems; (b) A multi-round dialogue with querying mechanism, consisting of semantic completion, knowledge-guided clarification, and proactive querying processes, to resolve interaction constraints in SQL generation; (c) A data-conditioned SQL generation method selection strategy that supports both an efficient one-step Fine-Tuning approach and a two-step method leveraging Semantic Intermediate Representation for low-cost cross-domain applications. Experiments on both real-world datasets and public benchmarks demonstrate the effectiveness of SiriusBI. User studies further confirm that SiriusBI enhances both productivity and user experience. As an independent service on Tencent's data platform, SiriusBI is deployed across finance, advertising, and cloud sectors, serving dozens of enterprise clients. It achieves over 93% accuracy in SQL generation and reduces data analysts' query time from minutes to seconds in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20541v1">MeLA: A Metacognitive LLM-Driven Architecture for Automatic Heuristic Design</a></div>
    <div class="paper-meta">
      📅 2025-07-28
    </div>
    <details class="paper-abstract">
      This paper introduces MeLA, a Metacognitive LLM-Driven Architecture that presents a new paradigm for Automatic Heuristic Design (AHD). Traditional evolutionary methods operate directly on heuristic code; in contrast, MeLA evolves the instructional prompts used to guide a Large Language Model (LLM) in generating these heuristics. This process of "prompt evolution" is driven by a novel metacognitive framework where the system analyzes performance feedback to systematically refine its generative strategy. MeLA's architecture integrates a problem analyzer to construct an initial strategic prompt, an error diagnosis system to repair faulty code, and a metacognitive search engine that iteratively optimizes the prompt based on heuristic effectiveness. In comprehensive experiments across both benchmark and real-world problems, MeLA consistently generates more effective and robust heuristics, significantly outperforming state-of-the-art methods. Ultimately, this research demonstrates the profound potential of using cognitive science as a blueprint for AI architecture, revealing that by enabling an LLM to metacognitively regulate its problem-solving process, we unlock a more robust and interpretable path to AHD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20527v1">SAND-Math: Using LLMs to Generate Novel, Difficult and Useful Mathematics Questions and Answers</a></div>
    <div class="paper-meta">
      📅 2025-07-28
    </div>
    <details class="paper-abstract">
      The demand for Large Language Models (LLMs) capable of sophisticated mathematical reasoning is growing across industries. However, the development of performant mathematical LLMs is critically bottlenecked by the scarcity of difficult, novel training data. We introduce \textbf{SAND-Math} (Synthetic Augmented Novel and Difficult Mathematics problems and solutions), a pipeline that addresses this by first generating high-quality problems from scratch and then systematically elevating their complexity via a new \textbf{Difficulty Hiking} step. We demonstrate the effectiveness of our approach through two key findings. First, augmenting a strong baseline with SAND-Math data significantly boosts performance, outperforming the next-best synthetic dataset by \textbf{$\uparrow$ 17.85 absolute points} on the AIME25 benchmark. Second, in a dedicated ablation study, we show our Difficulty Hiking process is highly effective: by increasing average problem difficulty from 5.02 to 5.98, this step lifts AIME25 performance from 46.38\% to 49.23\%. The full generation pipeline, final dataset, and a fine-tuned model form a practical and scalable toolkit for building more capable and efficient mathematical reasoning LLMs. SAND-Math dataset is released here: \href{https://huggingface.co/datasets/amd/SAND-MATH}{https://huggingface.co/datasets/amd/SAND-MATH}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20511v1">Beyond Class Tokens: LLM-guided Dominant Property Mining for Few-shot Classification</a></div>
    <div class="paper-meta">
      📅 2025-07-28
      | 💬 11 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Few-shot Learning (FSL), which endeavors to develop the generalization ability for recognizing novel classes using only a few images, faces significant challenges due to data scarcity. Recent CLIP-like methods based on contrastive language-image pertaining mitigate the issue by leveraging textual representation of the class name for unseen image discovery. Despite the achieved success, simply aligning visual representations to class name embeddings would compromise the visual diversity for novel class discrimination. To this end, we proposed a novel Few-Shot Learning (FSL) method (BCT-CLIP) that explores \textbf{dominating properties} via contrastive learning beyond simply using class tokens. Through leveraging LLM-based prior knowledge, our method pushes forward FSL with comprehensive structural image representations, including both global category representation and the patch-aware property embeddings. In particular, we presented a novel multi-property generator (MPG) with patch-aware cross-attentions to generate multiple visual property tokens, a Large-Language Model (LLM)-assistant retrieval procedure with clustering-based pruning to obtain dominating property descriptions, and a new contrastive learning strategy for property-token learning. The superior performances on the 11 widely used datasets demonstrate that our investigation of dominating properties advances discriminative class-specific representation learning and few-shot classification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20509v1">LLMs-guided adaptive compensator: Bringing Adaptivity to Automatic Control Systems with Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-07-28
    </div>
    <details class="paper-abstract">
      With rapid advances in code generation, reasoning, and problem-solving, Large Language Models (LLMs) are increasingly applied in robotics. Most existing work focuses on high-level tasks such as task decomposition. A few studies have explored the use of LLMs in feedback controller design; however, these efforts are restricted to overly simplified systems, fixed-structure gain tuning, and lack real-world validation. To further investigate LLMs in automatic control, this work targets a key subfield: adaptive control. Inspired by the framework of model reference adaptive control (MRAC), we propose an LLM-guided adaptive compensator framework that avoids designing controllers from scratch. Instead, the LLMs are prompted using the discrepancies between an unknown system and a reference system to design a compensator that aligns the response of the unknown system with that of the reference, thereby achieving adaptivity. Experiments evaluate five methods: LLM-guided adaptive compensator, LLM-guided adaptive controller, indirect adaptive control, learning-based adaptive control, and MRAC, on soft and humanoid robots in both simulated and real-world environments. Results show that the LLM-guided adaptive compensator outperforms traditional adaptive controllers and significantly reduces reasoning complexity compared to the LLM-guided adaptive controller. The Lyapunov-based analysis and reasoning-path inspection demonstrate that the LLM-guided adaptive compensator enables a more structured design process by transforming mathematical derivation into a reasoning task, while exhibiting strong generalizability, adaptability, and robustness. This study opens a new direction for applying LLMs in the field of automatic control, offering greater deployability and practicality compared to vision-language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02622v2">Exploring undercurrents of learning tensions in an LLM-enhanced landscape: A student-centered qualitative perspective on LLM vs Search</a></div>
    <div class="paper-meta">
      📅 2025-07-27
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are transforming how students learn by providing readily available tools that can quickly augment or complete various learning activities with non-trivial performance. Similar paradigm shifts have occurred in the past with the introduction of search engines and Wikipedia, which replaced or supplemented traditional information sources such as libraries and books. This study investigates the potential for LLMs to represent the next shift in learning, focusing on their role in information discovery and synthesis compared to existing technologies, such as search engines. Using a within-subjects, counterbalanced design, participants learned new topics using a search engine (Google) and an LLM (ChatGPT). Post-task follow-up interviews explored students' reflections, preferences, pain points, and overall perceptions. We present analysis of their responses that show nuanced insights into when, why, and how students prefer LLMs over search engines, offering implications for educators, policymakers, and technology developers navigating the evolving educational landscape.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14133v3">Self-Regularization with Sparse Autoencoders for Controllable LLM-based Classification</a></div>
    <div class="paper-meta">
      📅 2025-07-27
      | 💬 Accepted by SIGKDD 2025
    </div>
    <details class="paper-abstract">
      Modern text classification methods heavily rely on contextual embeddings from large language models (LLMs). Compared to human-engineered features, these embeddings provide automatic and effective representations for classification model training. However, they also introduce a challenge: we lose the ability to manually remove unintended features, such as sensitive or task-irrelevant features, to guarantee regulatory compliance or improve the generalizability of classification models. This limitation arises because LLM embeddings are opaque and difficult to interpret. In this paper, we propose a novel framework to identify and regularize unintended features in the LLM latent space. Specifically, we first pre-train a sparse autoencoder (SAE) to extract interpretable features from LLM latent spaces. To ensure the SAE can capture task-specific features, we further fine-tune it on task-specific datasets. In training the classification model, we propose a simple and effective regularizer, by minimizing the similarity between the classifier weights and the identified unintended feature, to remove the impact of these unintended features on classification. We evaluate the proposed framework on three real-world tasks, including toxic chat detection, reward modeling, and disease diagnosis. Results show that the proposed self-regularization framework can improve the classifier's generalizability by regularizing those features that are not semantically correlated to the task. This work pioneers controllable text classification on LLM latent spaces by leveraging interpreted features to address generalizability, fairness, and privacy challenges. The code and data are publicly available at https://github.com/JacksonWuxs/Controllable_LLM_Classifier.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00946v3">A Review of LLM-Assisted Ideation</a></div>
    <div class="paper-meta">
      📅 2025-07-27
    </div>
    <details class="paper-abstract">
      We present a comprehensive, in-depth review of ideation assisted by large language models (LLMs), highlighting emerging trends and identifying unaddressed research gaps. In total, we examined 61 studies investigating the application of LLMs in both group and individual ideation processes. From these studies, we derived the Hourglass Ideation Framework for LLM-assisted ideation, comprising three phases and seven key ideation stages, which served as the basis for our systematic survey. Our analysis reveals that LLMs are most frequently used for idea generation and refinement, but their use in scope specification, foundational material structuring and multi-idea evaluation and selection remains limited. We provide our findings in extensive tabular and online formats. These catalogues detail research on LLM-assisted, purely LLM-based, and human-only activities across the seven ideation stages for each of the 61 studies. These also detail creative domains, publication outlets, interaction designs, user study designs, and assessment methods. Our analysis of system interaction design reveals a predominant focus on supporting individual ideation activities and text-based interaction, with a growing trend of incorporating multimedia elements. However, in group ideation, tools and interaction modalities targeting both synchronous and asynchronous collaboration are much scarcer. We synthesize the primary findings of our review and outline promising directions for future research in LLM-assisted ideation. We hope this review will help researchers quickly gain an overview of this rapidly expanding area, efficiently locate relevant work, and identify underexplored areas for further investigation. In addition, we believe the framework we present here will form the basis for the development of future problem and solution space taxonomies, and methodologies for LLM-assisted ideation development and use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20352v1">RMTBench: Benchmarking LLMs Through Multi-Turn User-Centric Role-Playing</a></div>
    <div class="paper-meta">
      📅 2025-07-27
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have shown outstanding potential for role-playing applications. Evaluating these capabilities is becoming crucial yet remains challenging. Existing benchmarks mostly adopt a \textbf{character-centric} approach, simplify user-character interactions to isolated Q&A tasks, and fail to reflect real-world applications. To address this limitation, we introduce RMTBench, a comprehensive \textbf{user-centric} bilingual role-playing benchmark featuring 80 diverse characters and over 8,000 dialogue rounds. RMTBench includes custom characters with detailed backgrounds and abstract characters defined by simple traits, enabling evaluation across various user scenarios. Our benchmark constructs dialogues based on explicit user motivations rather than character descriptions, ensuring alignment with practical user applications. Furthermore, we construct an authentic multi-turn dialogue simulation mechanism. With carefully selected evaluation dimensions and LLM-based scoring, this mechanism captures the complex intention of conversations between the user and the character. By shifting focus from character background to user intention fulfillment, RMTBench bridges the gap between academic evaluation and practical deployment requirements, offering a more effective framework for assessing role-playing capabilities in LLMs. All code and datasets will be released soon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04626v2">Heterogeneous User Modeling for LLM-based Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-07-27
      | 💬 Accepted by RecSys 2025
    </div>
    <details class="paper-abstract">
      Leveraging Large Language Models (LLMs) for recommendation has demonstrated notable success in various domains, showcasing their potential for open-domain recommendation. A key challenge to advancing open-domain recommendation lies in effectively modeling user preferences from users' heterogeneous behaviors across multiple domains. Existing approaches, including ID-based and semantic-based modeling, struggle with poor generalization, an inability to compress noisy interactions effectively, and the domain seesaw phenomenon. To address these challenges, we propose a Heterogeneous User Modeling (HUM) method, which incorporates a compression enhancer and a robustness enhancer for LLM-based recommendation. The compression enhancer uses a customized prompt to compress heterogeneous behaviors into a tailored token, while a masking mechanism enhances cross-domain knowledge extraction and understanding. The robustness enhancer introduces a domain importance score to mitigate the domain seesaw phenomenon by guiding domain optimization. Extensive experiments on heterogeneous datasets validate that HUM effectively models user heterogeneity by achieving both high efficacy and robustness, leading to superior performance in open-domain recommendation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20300v1">Talking-to-Build: How LLM-Assisted Interface Shapes Player Performance and Experience in Minecraft</a></div>
    <div class="paper-meta">
      📅 2025-07-27
    </div>
    <details class="paper-abstract">
      With large language models (LLMs) on the rise, in-game interactions are shifting from rigid commands to natural conversations. However, the impacts of LLMs on player performance and game experience remain underexplored. This work explores LLM's role as a co-builder during gameplay, examining its impact on task performance, usability, and player experience. Using Minecraft as a sandbox, we present an LLM-assisted interface that engages players through natural language, aiming to facilitate creativity and simplify complex gaming commands. We conducted a mixed-methods study with 30 participants, comparing LLM-assisted and command-based interfaces across simple and complex game tasks. Quantitative and qualitative analyses reveal that the LLM-assisted interface significantly improves player performance, engagement, and overall game experience. Additionally, task complexity has a notable effect on player performance and experience across both interfaces. Our findings highlight the potential of LLM-assisted interfaces to revolutionize virtual experiences, emphasizing the importance of balancing intuitiveness with predictability, transparency, and user agency in AI-driven, multimodal gaming environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11150v3">ADAgent: LLM Agent for Alzheimer's Disease Analysis with Collaborative Coordinator</a></div>
    <div class="paper-meta">
      📅 2025-07-27
    </div>
    <details class="paper-abstract">
      Alzheimer's disease (AD) is a progressive and irreversible neurodegenerative disease. Early and precise diagnosis of AD is crucial for timely intervention and treatment planning to alleviate the progressive neurodegeneration. However, most existing methods rely on single-modality data, which contrasts with the multifaceted approach used by medical experts. While some deep learning approaches process multi-modal data, they are limited to specific tasks with a small set of input modalities and cannot handle arbitrary combinations. This highlights the need for a system that can address diverse AD-related tasks, process multi-modal or missing input, and integrate multiple advanced methods for improved performance. In this paper, we propose ADAgent, the first specialized AI agent for AD analysis, built on a large language model (LLM) to address user queries and support decision-making. ADAgent integrates a reasoning engine, specialized medical tools, and a collaborative outcome coordinator to facilitate multi-modal diagnosis and prognosis tasks in AD. Extensive experiments demonstrate that ADAgent outperforms SOTA methods, achieving significant improvements in accuracy, including a 2.7% increase in multi-modal diagnosis, a 0.7% improvement in multi-modal prognosis, and enhancements in MRI and PET diagnosis tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20278v1">MoL-RL: Distilling Multi-Step Environmental Feedback into LLMs for Feedback-Independent Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-07-27
      | 💬 12pages,3figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) face significant challenges in effectively leveraging sequential environmental feedback (EF) signals, such as natural language evaluations, for feedback-independent chain-of-thought (CoT) reasoning. Existing approaches either convert EF into scalar rewards, losing rich contextual information, or employ refinement datasets, failing to exploit the multi-step and discrete nature of EF interactions. To address these limitations, we propose MoL-RL, a novel training paradigm that integrates multi-step EF signals into LLMs through a dual-objective optimization framework. Our method combines MoL (Mixture-of-Losses) continual training, which decouples domain-specific EF signals (optimized via cross-entropy loss) and general language capabilities (preserved via Kullback-Leibler divergence), with GRPO-based post-training to distill sequential EF interactions into single-step inferences. This synergy enables robust feedback-independent reasoning without relying on external feedback loops. Experimental results on mathematical reasoning (MATH-500, AIME24/AIME25) and code generation (CodeAgent-Test) benchmarks demonstrate that MoL-RL achieves state-of-the-art performance with the Qwen3-8B model, while maintaining strong generalization across model scales (Qwen3-4B). This work provides a promising approach for leveraging multi-step textual feedback to enhance LLMs' reasoning capabilities in diverse domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.13632v4">DoubleDipper: Improving Long-Context LLMs via Context Recycling</a></div>
    <div class="paper-meta">
      📅 2025-07-27
    </div>
    <details class="paper-abstract">
      Despite recent advancements in Large Language Models (LLMs), their performance on tasks involving long contexts remains sub-optimal. In this work, we propose DoubleDipper, a novel In-Context-Learning method that automatically generates few-shot examples for long context QA tasks by recycling contexts. Specifically, given a long input context (1-3k tokens) and a query, we generate additional query-output pairs from the given context as few-shot examples, while introducing the context only once. This ensures that the demonstrations are leveraging the same context as the target query while only adding a small number of tokens to the prompt. We further enhance each demonstration by instructing the model to explicitly identify the relevant paragraphs before the answer, which improves performance while providing fine-grained attribution to the answer source. We apply our method on multiple LLMs and obtain substantial improvements (+16 absolute points on average across models) on various QA datasets with long context. Surprisingly, despite introducing only single-hop ICL examples, LLMs successfully generalize to multi-hop long-context QA using our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07840v3">Understanding Learner-LLM Chatbot Interactions and the Impact of Prompting Guidelines</a></div>
    <div class="paper-meta">
      📅 2025-07-27
      | 💬 Long paper accepted for AIED 2025, the 26th International Conference on Artificial Intelligence in Education, July 22 - 26, 2025, Palermo, Italy
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have transformed human-computer interaction by enabling natural language-based communication with AI-powered chatbots. These models are designed to be intuitive and user-friendly, allowing users to articulate requests with minimal effort. However, despite their accessibility, studies reveal that users often struggle with effective prompting, resulting in inefficient responses. Existing research has highlighted both the limitations of LLMs in interpreting vague or poorly structured prompts and the difficulties users face in crafting precise queries. This study investigates learner-AI interactions through an educational experiment in which participants receive structured guidance on effective prompting. We introduce and compare three types of prompting guidelines: a task-specific framework developed through a structured methodology and two baseline approaches. To assess user behavior and prompting efficacy, we analyze a dataset of 642 interactions from 107 users. Using Von NeuMidas, an extended pragmatic annotation schema for LLM interaction analysis, we categorize common prompting errors and identify recurring behavioral patterns. We then evaluate the impact of different guidelines by examining changes in user behavior, adherence to prompting strategies, and the overall quality of AI-generated responses. Our findings provide a deeper understanding of how users engage with LLMs and the role of structured prompting guidance in enhancing AI-assisted communication. By comparing different instructional frameworks, we offer insights into more effective approaches for improving user competency in AI interactions, with implications for AI literacy, chatbot usability, and the design of more responsive AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20208v1">IQ Test for LLMs: An Evaluation Framework for Uncovering Core Skills in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-27
    </div>
    <details class="paper-abstract">
      Current evaluations of large language models (LLMs) rely on benchmark scores, but it is difficult to interpret what these individual scores reveal about a model's overall skills. Specifically, as a community we lack understanding of how tasks relate to one another, what they measure in common, how they differ, or which ones are redundant. As a result, models are often assessed via a single score averaged across benchmarks, an approach that fails to capture the models' wholistic strengths and limitations. Here, we propose a new evaluation paradigm that uses factor analysis to identify latent skills driving performance across benchmarks. We apply this method to a comprehensive new leaderboard showcasing the performance of 60 LLMs on 44 tasks, and identify a small set of latent skills that largely explain performance. Finally, we turn these insights into practical tools that identify redundant tasks, aid in model selection, and profile models along each latent skill.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06774v2">Checklist Engineering Empowers Multilingual LLM Judges</a></div>
    <div class="paper-meta">
      📅 2025-07-27
    </div>
    <details class="paper-abstract">
      Automated text evaluation has long been a central issue in Natural Language Processing (NLP). Recently, the field has shifted toward using Large Language Models (LLMs) as evaluators-a trend known as the LLM-as-a-Judge paradigm. While promising and easily adaptable across tasks, this approach has seen limited exploration in multilingual contexts. Existing multilingual studies often rely on proprietary models or require extensive training data for fine-tuning, raising concerns about cost, time, and efficiency. In this paper, we propose Checklist Engineering based LLM-as-a-Judge (CE-Judge), a training-free framework that uses checklist intuition for multilingual evaluation with an open-source model. Experiments across multiple languages and three benchmark datasets, under both pointwise and pairwise settings, show that our method generally surpasses the baselines and performs on par with the GPT-4o model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20152v1">Goal Alignment in LLM-Based User Simulators for Conversational AI</a></div>
    <div class="paper-meta">
      📅 2025-07-27
    </div>
    <details class="paper-abstract">
      User simulators are essential to conversational AI, enabling scalable agent development and evaluation through simulated interactions. While current Large Language Models (LLMs) have advanced user simulation capabilities, we reveal that they struggle to consistently demonstrate goal-oriented behavior across multi-turn conversations--a critical limitation that compromises their reliability in downstream applications. We introduce User Goal State Tracking (UGST), a novel framework that tracks user goal progression throughout conversations. Leveraging UGST, we present a three-stage methodology for developing user simulators that can autonomously track goal progression and reason to generate goal-aligned responses. Moreover, we establish comprehensive evaluation metrics for measuring goal alignment in user simulators, and demonstrate that our approach yields substantial improvements across two benchmarks (MultiWOZ 2.4 and {\tau}-Bench). Our contributions address a critical gap in conversational AI and establish UGST as an essential framework for developing goal-aligned user simulators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20147v1">Integrating LLM-Derived Multi-Semantic Intent into Graph Model for Session-based Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-07-27
    </div>
    <details class="paper-abstract">
      Session-based recommendation (SBR) is mainly based on anonymous user interaction sequences to recommend the items that the next user is most likely to click. Currently, the most popular and high-performing SBR methods primarily leverage graph neural networks (GNNs), which model session sequences as graph-structured data to effectively capture user intent. However, most GNNs-based SBR methods primarily focus on modeling the ID sequence information of session sequences, while neglecting the rich semantic information embedded within them. This limitation significantly hampers model's ability to accurately infer users' true intention. To address above challenge, this paper proposes a novel SBR approach called Integrating LLM-Derived Multi-Semantic Intent into Graph Model for Session-based Recommendation (LLM-DMsRec). The method utilizes a pre-trained GNN model to select the top-k items as candidate item sets and designs prompts along with a large language model (LLM) to infer multi-semantic intents from these candidate items. Specifically, we propose an alignment mechanism that effectively integrates the semantic intent inferred by the LLM with the structural intent captured by GNNs. Extensive experiments conducted on the Beauty and ML-1M datasets demonstrate that the proposed method can be seamlessly integrated into GNNs framework, significantly enhancing its recommendation performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.14066v3">TurboSpec: Closed-loop Speculation Control System for Optimizing LLM Serving Goodput</a></div>
    <div class="paper-meta">
      📅 2025-07-27
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) serving systems batch concurrent user requests to achieve efficient serving. However, in real-world deployments, such inter-request parallelism from batching is often limited by external factors such as low request rates or memory constraints. Recent works focus on intra-request parallelism from speculative decoding as a solution to this problem. Unfortunately, benefits from intra-request parallelism are often fragile, as speculative decoding causes overhead, and speculated tokens may miss. We observe that speculative decoding may degrade LLM serving performance if added naively without tuning to the incoming requests and the speculation method. To alleviate the need for expert tuning and make speculative decoding more robust, we present TurboSpec, a speculation control system that automatically profiles the execution environment and utilizes a feedback-based algorithm to dynamically adjust the amount of intra-request parallelism in LLM serving. TurboSpec predicts "goodput" - the amount of successfully generated tokens - to evaluate and adjust intra-request parallelism amount to that with the highest goodput in runtime. We implement TurboSpec on a real-world LLM serving system vLLM and demonstrate its effectiveness across diverse workloads and hardware configurations, providing consistent performance improvements across all test scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12088v2">Risks & Benefits of LLMs & GenAI for Platform Integrity, Healthcare Diagnostics, Financial Trust and Compliance, Cybersecurity, Privacy & AI Safety: A Comprehensive Survey, Roadmap & Implementation Blueprint</a></div>
    <div class="paper-meta">
      📅 2025-07-26
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) and generative AI (GenAI) systems, such as ChatGPT, Claude, Gemini, LLaMA, and Copilot (by OpenAI, Anthropic, Google, Meta, and Microsoft, respectively), are reshaping digital platforms and app ecosystems while introducing critical challenges in cybersecurity, privacy, and platform integrity. Our analysis reveals alarming trends: LLM-assisted malware is projected to rise from 2% (2021) to 50% (2025); AI-generated Google reviews grew nearly tenfold (1.2% in 2021 to 12.21% in 2023, expected to reach 30% by 2025); AI scam reports surged 456%; misinformation sites increased over 1500%; and deepfake attacks are projected to rise over 900% in 2025. In finance, LLM-driven threats like synthetic identity fraud and AI-generated scams are accelerating. Platforms such as JPMorgan Chase, Stripe, and Plaid deploy LLMs for fraud detection, regulation parsing, and KYC/AML automation, reducing fraud loss by up to 21% and accelerating onboarding by 40-60%. LLM-facilitated code development has driven mobile app submissions from 1.8 million (2020) to 3.0 million (2024), projected to reach 3.6 million (2025). To address AI threats, platforms like Google Play, Apple App Store, GitHub Copilot, TikTok, Facebook, and Amazon deploy LLM-based defenses, highlighting their dual nature as both threat sources and mitigation tools. In clinical diagnostics, LLMs raise concerns about accuracy, bias, and safety, necessitating strong governance. Drawing on 445 references, this paper surveys LLM/GenAI and proposes a strategic roadmap and operational blueprint integrating policy auditing (such as CCPA and GDPR compliance), fraud detection, and demonstrates an advanced LLM-DA stack with modular components, multi-LLM routing, agentic memory, and governance layers. We provide actionable insights, best practices, and real-world case studies for scalable trust and responsible innovation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20067v1">PITA: Preference-Guided Inference-Time Alignment for LLM Post-Training</a></div>
    <div class="paper-meta">
      📅 2025-07-26
    </div>
    <details class="paper-abstract">
      Inference-time alignment enables large language models (LLMs) to generate outputs aligned with end-user preferences without further training. Recent post-training methods achieve this by using small guidance models to modify token generation during inference. These methods typically optimize a reward function KL-regularized by the original LLM taken as the reference policy. A critical limitation, however, is their dependence on a pre-trained reward model, which requires fitting to human preference feedback--a potentially unstable process. In contrast, we introduce PITA, a novel framework that integrates preference feedback directly into the LLM's token generation, eliminating the need for a reward model. PITA learns a small preference-based guidance policy to modify token probabilities at inference time without LLM fine-tuning, reducing computational cost and bypassing the pre-trained reward model dependency. The problem is framed as identifying an underlying preference distribution, solved through stochastic search and iterative refinement of the preference-based guidance model. We evaluate PITA across diverse tasks, including mathematical reasoning and sentiment classification, demonstrating its effectiveness in aligning LLM outputs with user preferences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20059v1">RAG in the Wild: On the (In)effectiveness of LLMs with Mixture-of-Knowledge Retrieval Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-07-26
      | 💬 Work in Progress. Code will be published at: https://github.com/ritaranx/RAG_in_the_Wild
    </div>
    <details class="paper-abstract">
      Retrieval-augmented generation (RAG) enhances large language models (LLMs) by integrating external knowledge retrieved at inference time. While RAG demonstrates strong performance on benchmarks largely derived from general-domain corpora like Wikipedia, its effectiveness under realistic, diverse retrieval scenarios remains underexplored. We evaluated RAG systems using MassiveDS, a large-scale datastore with mixture of knowledge, and identified critical limitations: retrieval mainly benefits smaller models, rerankers add minimal value, and no single retrieval source consistently excels. Moreover, current LLMs struggle to route queries across heterogeneous knowledge sources. These findings highlight the need for adaptive retrieval strategies before deploying RAG in real-world settings. Our code and data can be found at https://github.com/ritaranx/RAG_in_the_Wild.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02456v2">Colombian Waitresses y Jueces canadienses: Gender and Country Biases in Occupation Recommendations from LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-26
      | 💬 Workshop on Gender Bias in Natural Language Processing at ACL 2025
    </div>
    <details class="paper-abstract">
      One of the goals of fairness research in NLP is to measure and mitigate stereotypical biases that are propagated by NLP systems. However, such work tends to focus on single axes of bias (most often gender) and the English language. Addressing these limitations, we contribute the first study of multilingual intersecting country and gender biases, with a focus on occupation recommendations generated by large language models. We construct a benchmark of prompts in English, Spanish and German, where we systematically vary country and gender, using 25 countries and four pronoun sets. Then, we evaluate a suite of 5 Llama-based models on this benchmark, finding that LLMs encode significant gender and country biases. Notably, we find that even when models show parity for gender or country individually, intersectional occupational biases based on both country and gender persist. We also show that the prompting language significantly affects bias, and instruction-tuned models consistently demonstrate the lowest and most stable levels of bias. Our findings highlight the need for fairness researchers to use intersectional and multilingual lenses in their work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19980v1">Exploring LLM Autoscoring Reliability in Large-Scale Writing Assessments Using Generalizability Theory</a></div>
    <div class="paper-meta">
      📅 2025-07-26
    </div>
    <details class="paper-abstract">
      This study investigates the estimation of reliability for large language models (LLMs) in scoring writing tasks from the AP Chinese Language and Culture Exam. Using generalizability theory, the research evaluates and compares score consistency between human and AI raters across two types of AP Chinese free-response writing tasks: story narration and email response. These essays were independently scored by two trained human raters and seven AI raters. Each essay received four scores: one holistic score and three analytic scores corresponding to the domains of task completion, delivery, and language use. Results indicate that although human raters produced more reliable scores overall, LLMs demonstrated reasonable consistency under certain conditions, particularly for story narration tasks. Composite scoring that incorporates both human and AI raters improved reliability, which supports that hybrid scoring models may offer benefits for large-scale writing assessments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09720v3">NestQuant: Nested Lattice Quantization for Matrix Products and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-26
      | 💬 23 pages; Accepted at the 42nd International Conference on Machine Learning (ICML 2025)
    </div>
    <details class="paper-abstract">
      Post-training quantization (PTQ) has emerged as a critical technique for efficient deployment of large language models (LLMs). This work proposes NestQuant, a novel PTQ scheme for weights and activations that is based on self-similar nested lattices. Recent works have mathematically shown such quantizers to be information-theoretically optimal for low-precision matrix multiplication. We implement a practical low-complexity version of NestQuant based on Gosset lattice, making it a drop-in quantizer for any matrix multiplication step (e.g., in self-attention, MLP etc). For example, NestQuant quantizes weights, KV-cache, and activations of Llama-3-8B to 4 bits, achieving perplexity of 6.6 on wikitext2. This represents more than 55% reduction in perplexity gap with respect to unquantized model (perplexity of 6.14) compared to state-of-the-art Metas SpinQuant (perplexity 7.3), OstQuant (7.3) and QuaRot (8.2). Comparisons on bigger models (up to 70B) and on various LLM evaluation benchmarks confirm uniform superiority of NestQuant.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19939v1">LLMControl: Grounded Control of Text-to-Image Diffusion-based Synthesis with Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-26
    </div>
    <details class="paper-abstract">
      Recent spatial control methods for text-to-image (T2I) diffusion models have shown compelling results. However, these methods still fail to precisely follow the control conditions and generate the corresponding images, especially when encountering the textual prompts that contain multiple objects or have complex spatial compositions. In this work, we present a LLM-guided framework called LLM\_Control to address the challenges of the controllable T2I generation task. By improving grounding capabilities, LLM\_Control is introduced to accurately modulate the pre-trained diffusion models, where visual conditions and textual prompts influence the structures and appearance generation in a complementary way. We utilize the multimodal LLM as a global controller to arrange spatial layouts, augment semantic descriptions and bind object attributes. The obtained control signals are injected into the denoising network to refocus and enhance attention maps according to novel sampling constraints. Extensive qualitative and quantitative experiments have demonstrated that LLM\_Control achieves competitive synthesis quality compared to other state-of-the-art methods across various pre-trained T2I models. It is noteworthy that LLM\_Control allows the challenging input conditions on which most of the existing methods
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19899v1">A Gold Standard Dataset and Evaluation Framework for Depression Detection and Explanation in Social Media using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-26
    </div>
    <details class="paper-abstract">
      Early detection of depression from online social media posts holds promise for providing timely mental health interventions. In this work, we present a high-quality, expert-annotated dataset of 1,017 social media posts labeled with depressive spans and mapped to 12 depression symptom categories. Unlike prior datasets that primarily offer coarse post-level labels \cite{cohan-etal-2018-smhd}, our dataset enables fine-grained evaluation of both model predictions and generated explanations. We develop an evaluation framework that leverages this clinically grounded dataset to assess the faithfulness and quality of natural language explanations generated by large language models (LLMs). Through carefully designed prompting strategies, including zero-shot and few-shot approaches with domain-adapted examples, we evaluate state-of-the-art proprietary LLMs including GPT-4.1, Gemini 2.5 Pro, and Claude 3.7 Sonnet. Our comprehensive empirical analysis reveals significant differences in how these models perform on clinical explanation tasks, with zero-shot and few-shot prompting. Our findings underscore the value of human expertise in guiding LLM behavior and offer a step toward safer, more transparent AI systems for psychological well-being.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19855v1">Inducing Causal World Models in LLMs for Zero-Shot Physical Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-07-26
      | 💬 12 pages, 4 figures,
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), despite their advanced linguistic capabilities, fundamentally lack an intuitive understanding of physical dynamics, which limits their effectiveness in real-world scenarios that require causal reasoning. In this paper, we introduce Causal World Model Induction (CWMI), a novel framework designed to embed an explicit model of causal physics within an LLM. Our approach incorporates a dedicated Causal Physics Module (CPM) and a new training objective called Causal Intervention Loss, encouraging the model to learn cause-and-effect relationships from multimodal data. By training the model to predict the outcomes of hypothetical interventions instead of merely capturing statistical correlations, CWMI develops a robust internal representation of physical laws. Experimental results show that CWMI significantly outperforms state-of-the-art LLMs on zero-shot physical reasoning tasks, including the PIQA benchmark and our newly proposed PhysiCa-Bench dataset. These findings demonstrate that inducing a causal world model is a critical step toward more reliable and generalizable AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19845v1">MegatronApp: Efficient and Comprehensive Management on Distributed LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-07-26
    </div>
    <details class="paper-abstract">
      The rapid escalation in the parameter count of large language models (LLMs) has transformed model training from a single-node endeavor into a highly intricate, cross-node activity. While frameworks such as Megatron-LM successfully integrate tensor (TP), pipeline (PP), and data (DP) parallelism to enable trillion-parameter training, they simultaneously expose practitioners to unprecedented systems-level challenges in performance optimization, diagnosis, and interpretability. MegatronApp is an open-source toolchain expressly designed to meet these challenges. It introduces four orthogonal, yet seamlessly composable modules--MegaScan, MegaFBD, MegaDPP, and MegaScope--that collectively elevate the reliability, efficiency, and transparency of production-scale training. This paper presents the motivation, architecture, and distinctive contributions of each module, and elucidates how their synergistic integration augments the Megatron-LM ecosystem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19823v1">HCAttention: Extreme KV Cache Compression via Heterogeneous Attention Computing for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-26
    </div>
    <details class="paper-abstract">
      Processing long-context inputs with large language models presents a significant challenge due to the enormous memory requirements of the Key-Value (KV) cache during inference. Existing KV cache compression methods exhibit noticeable performance degradation when memory is reduced by more than 85%. Additionally, strategies that leverage GPU-CPU collaboration for approximate attention remain underexplored in this setting. We propose HCAttention, a heterogeneous attention computation framework that integrates key quantization, value offloading, and dynamic KV eviction to enable efficient inference under extreme memory constraints. The method is compatible with existing transformer architectures and does not require model fine-tuning. Experimental results on the LongBench benchmark demonstrate that our approach preserves the accuracy of full-attention model while shrinking the KV cache memory footprint to 25% of its original size. Remarkably, it stays competitive with only 12.5% of the cache, setting a new state-of-the-art in LLM KV cache compression. To the best of our knowledge, HCAttention is the first to extend the Llama-3-8B model to process 4 million tokens on a single A100 GPU with 80GB memory.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19749v1">Can LLMs Solve ASP Problems? Insights from a Benchmarking Study (Extended Version)</a></div>
    <div class="paper-meta">
      📅 2025-07-26
      | 💬 Accepted for publication at the 22nd International Conference on Principles of Knowledge Representation and Reasoning (KR 2025). The code is available at https://github.com/HomuraT/ASPBench
    </div>
    <details class="paper-abstract">
      Answer Set Programming (ASP) is a powerful paradigm for non-monotonic reasoning. Recently, large language models (LLMs) have demonstrated promising capabilities in logical reasoning. Despite this potential, current evaluations of LLM capabilities in ASP are often limited. Existing works normally employ overly simplified ASP programs, do not support negation, disjunction, or multiple answer sets. Furthermore, there is a lack of benchmarks that introduce tasks specifically designed for ASP solving. To bridge this gap, we introduce ASPBench, a comprehensive ASP benchmark, including three ASP specific tasks: ASP entailment, answer set verification, and answer set computation. Our extensive evaluations on ASPBench reveal that while 14 state-of-the-art LLMs, including \emph{deepseek-r1}, \emph{o4-mini}, and \emph{gemini-2.5-flash-thinking}, perform relatively well on the first two simpler tasks, they struggle with answer set computation, which is the core of ASP solving. These findings offer insights into the current limitations of LLMs in ASP solving. This highlights the need for new approaches that integrate symbolic reasoning capabilities more effectively. The code and dataset are available at https://github.com/HomuraT/ASPBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19747v1">TokenBlowUp: Resolving Representational Singularities in LLM Token Spaces via Monoidal Transformations</a></div>
    <div class="paper-meta">
      📅 2025-07-26
    </div>
    <details class="paper-abstract">
      Recent work has provided compelling evidence challenging the foundational manifold hypothesis for the token embedding spaces of Large Language Models (LLMs). These findings reveal the presence of geometric singularities around polysemous tokens, which can lead to representational instability. Existing methodologies, which presuppose a smooth data manifold, are ill-equipped to address such intrinsic structural flaws. In this paper, we formalize this problem in the language of scheme theory and propose a rigorous resolution by applying the scheme-theoretic blow-up at each singular point. This procedure replaces a singular point in the ambient affine scheme with its exceptional divisor, which we identify as a canonical geometric space -- a projective space of directions -- that houses the disambiguated semantic meanings of the token. This process of ``representational desingularization'' constructs a new geometric landscape for embeddings. We prove a formal theorem guaranteeing the geometric regularization of this new space, showing that the original pathologies are resolved. Finally, we outline the architectural implications of our framework, arguing for a paradigm shift from static look-ups to dynamic, geometrically-grounded computation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19737v1">Predicting Human Mobility in Disasters via LLM-Enhanced Cross-City Learning</a></div>
    <div class="paper-meta">
      📅 2025-07-26
    </div>
    <details class="paper-abstract">
      The vulnerability of cities to natural disasters has increased with urbanization and climate change, making it more important to predict human mobility in the disaster scenarios for downstream tasks including location-based early disaster warning and pre-allocating rescue resources, etc. However, existing human mobility prediction models are mainly designed for normal scenarios, and fail to adapt to disaster scenarios due to the shift of human mobility patterns under disaster. To address this issue, we introduce \textbf{DisasterMobLLM}, a mobility prediction framework for disaster scenarios that can be integrated into existing deep mobility prediction methods by leveraging LLMs to model the mobility intention and transferring the common knowledge of how different disasters affect mobility intentions between cities. This framework utilizes a RAG-Enhanced Intention Predictor to forecast the next intention, refines it with an LLM-based Intention Refiner, and then maps the intention to an exact location using an Intention-Modulated Location Predictor. Extensive experiments illustrate that DisasterMobLLM can achieve a 32.8\% improvement in terms of Acc@1 and a 35.0\% improvement in terms of the F1-score of predicting immobility compared to the baselines. The code is available at https://github.com/tsinghua-fib-lab/DisasterMobLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19399v1">Running in CIRCLE? A Simple Benchmark for LLM Code Interpreter Security</a></div>
    <div class="paper-meta">
      📅 2025-07-25
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) increasingly integrate native code interpreters, they enable powerful real-time execution capabilities, substantially expanding their utility. However, such integrations introduce potential system-level cybersecurity threats, fundamentally different from prompt-based vulnerabilities. To systematically evaluate these interpreter-specific risks, we propose CIRCLE (Code-Interpreter Resilience Check for LLM Exploits), a simple benchmark comprising 1,260 prompts targeting CPU, memory, and disk resource exhaustion. Each risk category includes explicitly malicious ("direct") and plausibly benign ("indirect") prompt variants. Our automated evaluation framework assesses not only whether LLMs refuse or generates risky code, but also executes the generated code within the interpreter environment to evaluate code correctness, simplifications made by the LLM to make the code safe, or execution timeouts. Evaluating 7 commercially available models from OpenAI and Google, we uncover significant and inconsistent vulnerabilities. For instance, evaluations show substantial disparities even within providers - OpenAI's o4-mini correctly refuses risky requests at 7.1%, notably higher rates compared to GPT-4.1 at 0.5%. Results particularly underscore that indirect, socially-engineered prompts substantially weaken model defenses. This highlights an urgent need for interpreter-specific cybersecurity benchmarks, dedicated mitigation tools (e.g., guardrails), and clear industry standards to guide safe and responsible deployment of LLM interpreter integrations. The benchmark dataset and evaluation code are publicly released to foster further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19390v1">ReCatcher: Towards LLMs Regression Testing for Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-25
      | 💬 24 pages, 3 Figures, 2 Tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) for code generation evolve rapidly through fine-tuning, merging, or new model releases. However, such updates can introduce regressions, not only in correctness but also in code quality and performance. To address this, we present ReCatcher, a regression testing framework for Python code generation. ReCatcher systematically compares two LLMs, typically a current model and a candidate update, across three dimensions: logical correctness, static code quality, and execution performance. We apply ReCatcher to assess regressions across three update scenarios, fine-tuning, merging, and model release, using CodeLlama, DeepSeek-Coder, and GPT-4o. Our evaluation shows that fine-tuning with cross-language datasets increases syntax errors by up to 12%. Merging with general-purpose models like Llama2 leads to regressions in correctness by up to 18%. GPT-4o introduces regressions of up to 50% in handling missing imports compared to GPT-3.5-turbo, while GPT-4o-mini suffers up to 80% performance degradation in execution time versus GPT-4o. Overall, logical correctness, performance, and error handling (e.g., syntax errors and missing imports) are the most regression-prone areas. Comparing ReCatcher with baseline solutions, it presents better and consistent accuracy across logical and performance aspects. ReCatcher highlights the importance of systematic regression evaluation before adopting new models, while assisting researchers and practitioners in making more informed update decisions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.01482v2">Understanding LLM Scientific Reasoning through Promptings and Model's Explanation on the Answers</a></div>
    <div class="paper-meta">
      📅 2025-07-25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable capabilities in natural language understanding, reasoning, and problem-solving across various domains. However, their ability to perform complex, multi-step reasoning task-essential for applications in science, medicine, and law-remains an area of active investigation. This paper examines the reasoning capabilities of contemporary LLMs, analyzing their strengths, limitations, and potential for improvement. The study uses prompt engineering techniques on the Graduate-Level GoogleProof Q&A (GPQA) dataset to assess the scientific reasoning of GPT-4o. Five popular prompt engineering techniques and two tailored promptings were tested: baseline direct answer (zero-shot), chain-of-thought (CoT), zero-shot CoT, self-ask, self-consistency, decomposition, and multipath promptings. Our findings indicate that while LLMs exhibit emergent reasoning abilities, they often rely on pattern recognition rather than true logical inference, leading to inconsistencies in complex problem-solving. The results indicated that self-consistency outperformed the other prompt engineering technique with an accuracy of 52.99%, followed by direct answer (52.23%). Zero-shot CoT (50%) outperformed multipath (48.44%), decomposition (47.77%), self-ask (46.88%), and CoT (43.75%). Self-consistency performed the second worst in explaining the answers. Simple techniques such as direct answer, CoT, and zero-shot CoT have the best scientific reasoning. We propose a research agenda aimed at bridging these gaps by integrating structured reasoning frameworks, hybrid AI approaches, and human-in-the-loop methodologies. By critically evaluating the reasoning mechanisms of LLMs, this paper contributes to the ongoing discourse on the future of artificial general intelligence and the development of more robust, trustworthy AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19364v1">Integrating LLM in Agent-Based Social Simulation: Opportunities and Challenges</a></div>
    <div class="paper-meta">
      📅 2025-07-25
    </div>
    <details class="paper-abstract">
      This position paper examines the use of Large Language Models (LLMs) in social simulation, analyzing both their potential and their limitations from a computational social science perspective. The first part reviews recent findings on the ability of LLMs to replicate key aspects of human cognition, including Theory of Mind reasoning and social inference, while also highlighting significant limitations such as cognitive biases, lack of true understanding, and inconsistencies in behavior. The second part surveys emerging applications of LLMs in multi-agent simulation frameworks, focusing on system architectures, scale, and validation strategies. Notable projects such as Generative Agents (Smallville) and AgentSociety are discussed in terms of their design choices, empirical grounding, and methodological innovations. Particular attention is given to the challenges of behavioral fidelity, calibration, and reproducibility in large-scale LLM-driven simulations. The final section distinguishes between contexts where LLMs, like other black-box systems, offer direct value-such as interactive simulations and serious games-and those where their use is more problematic, notably in explanatory or predictive modeling. The paper concludes by advocating for hybrid approaches that integrate LLMs into traditional agent-based modeling platforms (GAMA, Netlogo, etc), enabling modelers to combine the expressive flexibility of language-based reasoning with the transparency and analytical rigor of classical rule-based systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19353v1">Smooth Reading: Bridging the Gap of Recurrent LLM to Self-Attention LLM on Long-Context Tasks</a></div>
    <div class="paper-meta">
      📅 2025-07-25
    </div>
    <details class="paper-abstract">
      Recently, recurrent large language models (Recurrent LLMs) with linear computational complexity have re-emerged as efficient alternatives to self-attention-based LLMs (Self-Attention LLMs), which have quadratic complexity. However, Recurrent LLMs often underperform on long-context tasks due to their limited fixed-size memory. Previous research has primarily focused on enhancing the memory capacity of Recurrent LLMs through architectural innovations, but these approaches have not yet enabled Recurrent LLMs to match the performance of Self-Attention LLMs on long-context tasks. We argue that this limitation arises because processing the entire context at once is not well-suited for Recurrent LLMs. In this paper, we propose Smooth Reading, a chunk-wise inference method inspired by human reading strategies. Smooth Reading processes context in chunks and iteratively summarizes the contextual information, thereby reducing memory demands and making the approach more compatible with Recurrent LLMs. Our experimental results show that this method substantially narrows the performance gap between Recurrent and Self-Attention LLMs on long-context tasks, while preserving the efficiency advantages of Recurrent LLMs. Our Smooth Reading boosts SWA-3B-4k (a Recurrent LLM) from 5.68% lower to 3.61% higher performance than Self-Attention LLMs on LongBench. Besides, our method maintains the high efficiency, training 3x faster and inferring 2x faster at 64k context compared to Self-Attention LLMs. To our knowledge, this is the first work to achieve comparable performance using Recurrent LLMs compared with Self-Attention LLMs on long-context tasks. We hope our method will inspire future research in this area. To facilitate further progress, we will release code and dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19334v1">Doubling Your Data in Minutes: Ultra-fast Tabular Data Generation via LLM-Induced Dependency Graphs</a></div>
    <div class="paper-meta">
      📅 2025-07-25
    </div>
    <details class="paper-abstract">
      Tabular data is critical across diverse domains, yet high-quality datasets remain scarce due to privacy concerns and the cost of collection. Contemporary approaches adopt large language models (LLMs) for tabular augmentation, but exhibit two major limitations: (1) dense dependency modeling among tabular features that can introduce bias, and (2) high computational overhead in sampling. To address these issues, we propose SPADA for SPArse Dependency-driven Augmentation, a lightweight generative framework that explicitly captures sparse dependencies via an LLM-induced graph. We treat each feature as a node and synthesize values by traversing the graph, conditioning each feature solely on its parent nodes. We explore two synthesis strategies: a non-parametric method using Gaussian kernel density estimation, and a conditional normalizing flow model that learns invertible mappings for conditional density estimation. Experiments on four datasets show that SPADA reduces constraint violations by 4% compared to diffusion-based methods and accelerates generation by nearly 9,500 times over LLM-based baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19283v1">Towards LLM-Enhanced Group Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2025-07-25
    </div>
    <details class="paper-abstract">
      In contrast to single-user recommender systems, group recommender systems are designed to generate and explain recommendations for groups. This group-oriented setting introduces additional complexities, as several factors - absent in individual contexts - must be addressed. These include understanding group dynamics (e.g., social dependencies within the group), defining effective decision-making processes, ensuring that recommendations are suitable for all group members, and providing group-level explanations as well as explanations for individual users. In this paper, we analyze in which way large language models (LLMs) can support these aspects and help to increase the overall decision support quality and applicability of group recommender systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19275v1">Mut4All: Fuzzing Compilers via LLM-Synthesized Mutators Learned from Bug Reports</a></div>
    <div class="paper-meta">
      📅 2025-07-25
    </div>
    <details class="paper-abstract">
      Mutation-based fuzzing is effective for uncovering compiler bugs, but designing high-quality mutators for modern languages with complex constructs (e.g., templates, macros) remains challenging. Existing methods rely heavily on manual design or human-in-the-loop correction, limiting scalability and cross-language generalizability. We present Mut4All, a fully automated, language-agnostic framework that synthesizes mutators using Large Language Models (LLMs) and compiler-specific knowledge from bug reports. It consists of three agents: (1) a mutator invention agent that identifies mutation targets and generates mutator metadata using compiler-related insights; (2) a mutator implementation synthesis agent, fine-tuned to produce initial implementations; and (3) a mutator refinement agent that verifies and corrects the mutators via unit-test feedback. Mut4All processes 1000 bug reports (500 Rust, 500 C++), yielding 319 Rust and 403 C++ mutators at ~$0.08 each via GPT-4o. Our customized fuzzer, using these mutators, finds 62 bugs in Rust compilers (38 new, 7 fixed) and 34 bugs in C++ compilers (16 new, 1 fixed). Mut4All outperforms existing methods in both unique crash detection and coverage, ranking first on Rust and second on C++.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19254v1">DBMS-LLM Integration Strategies in Industrial and Business Applications: Current Status and Future Challenges</a></div>
    <div class="paper-meta">
      📅 2025-07-25
    </div>
    <details class="paper-abstract">
      Modern enterprises are increasingly driven by the DATA+AI paradigm, in which Database Management Systems (DBMSs) and Large Language Models (LLMs) have become two foundational infrastructures powering a wide range of industrial and business applications, such as enterprise analytics, intelligent customer service, and data-driven decision-making. The efficient integration of DBMSs and LLMs within a unified system offers significant opportunities but also introduces new technical challenges. This paper surveys recent developments in DBMS+LLM integration and identifies key future challenges. Specifically, we categorize five representative architectural patterns based on their core design principles, strengths, and trade-offs. Based on this analysis, we further highlight several critical open challenges. We aim to provide a systematic understanding of the current integration landscape and to outline the unresolved issues that must be addressed to achieve scalable and efficient integration of traditional data management and advanced language reasoning in future intelligent applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18377v2">Maximum Redundancy Pruning: A Principle-Driven Layerwise Sparsity Allocation for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive capabilities, but their enormous size poses significant challenges for deployment in real-world applications. To address this issue, researchers have sought to apply network pruning techniques to LLMs. A critical challenge in pruning is allocation the sparsity for each layer. Recent sparsity allocation methods is often based on heuristics or search that can easily lead to suboptimal performance. In this paper, we conducted an extensive investigation into various LLMs and revealed three significant discoveries: (1) the layerwise pruning sensitivity (LPS) of LLMs is highly non-uniform, (2) the choice of pruning metric affects LPS, and (3) the performance of a sparse model is related to the uniformity of its layerwise redundancy level. Based on these observations, we propose that the layerwise sparsity of LLMs should adhere to three principles: \emph{non-uniformity}, \emph{pruning metric dependency}, and \emph{uniform layerwise redundancy level} in the pruned model. To this end, we proposed Maximum Redundancy Pruning (MRP), an iterative pruning algorithm that prunes in the most redundant layers (\emph{i.e.}, those with the highest non-outlier ratio) at each iteration. The achieved layerwise sparsity aligns with the outlined principles. We conducted extensive experiments on publicly available LLMs, including the LLaMA2 and OPT, across various benchmarks. Experimental results validate the effectiveness of MRP, demonstrating its superiority over previous methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16142v3">Distilling the Implicit Multi-Branch Structure in LLMs' Reasoning via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-07-25
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      Distilling reasoning paths from teacher to student models via supervised fine-tuning (SFT) provides a shortcut for improving the reasoning ability of smaller Large Language Models (LLMs). However, the reasoning paths generated by teacher models often reflect only surface-level traces of their underlying authentic reasoning. Insights from cognitive neuroscience suggest that authentic reasoning involves a complex interweaving between meta-reasoning (which selects appropriate sub-problems from multiple candidates) and solving (which addresses the sub-problem). This implies authentic reasoning has an implicit multi-branch structure. Supervised fine-tuning collapses this rich structure into a flat sequence of token prediction in the teacher's reasoning path, preventing effective distillation of this structure to students. To address this limitation, we propose RLKD, a reinforcement learning (RL)-based distillation framework guided by a novel Generative Structure Reward Model (GSRM). Our GSRM converts reasoning paths into multiple meta-reasoning-solving steps and computes rewards to measure structural alignment between student and teacher reasoning. RLKD combines this reward with RL, enabling student LLMs to internalize the teacher's implicit multi-branch reasoning structure rather than merely mimicking fixed output paths. Experiments show RLKD surpasses standard SFT-RL pipelines even when trained on 0.1% of data under an RL-only regime, unlocking greater student reasoning potential than SFT-based distillation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19113v1">Exploring the Use of LLMs for Requirements Specification in an IT Consulting Company</a></div>
    <div class="paper-meta">
      📅 2025-07-25
      | 💬 11 pages, 5 figures. Accepted for presentation at the Industrial Innovation Track of the 33rd IEEE International Requirements Engineering Conference (RE 2025), Valencia, Spain
    </div>
    <details class="paper-abstract">
      In practice, requirements specification remains a critical challenge. The knowledge necessary to generate a specification can often be fragmented across diverse sources (e.g., meeting minutes, emails, and high-level product descriptions), making the process cumbersome and time-consuming. In this paper, we report our experience using large language models (LLMs) in an IT consulting company to automate the requirements specification process. In this company, requirements are specified using a Functional Design Specification (FDS), a document that outlines the functional requirements and features of a system, application, or process. We provide LLMs with a summary of the requirements elicitation documents and FDS templates, prompting them to generate Epic FDS (including high-level product descriptions) and user stories, which are subsequently compiled into a complete FDS document. We compared the correctness and quality of the FDS generated by three state-of-the-art LLMs against those produced by human analysts. Our results show that LLMs can help automate and standardize the requirements specification, reducing time and human effort. However, the quality of LLM-generated FDS highly depends on inputs and often requires human revision. Thus, we advocate for a synergistic approach in which an LLM serves as an effective drafting tool while human analysts provide the critical contextual and technical oversight necessary for high-quality requirements engineering (RE) documentation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12591v2">LLMs are Also Effective Embedding Models: An In-depth Overview</a></div>
    <div class="paper-meta">
      📅 2025-07-25
      | 💬 38 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have revolutionized natural language processing by achieving state-of-the-art performance across various tasks. Recently, their effectiveness as embedding models has gained attention, marking a paradigm shift from traditional encoder-only models like ELMo and BERT to decoder-only, large-scale LLMs such as GPT, LLaMA, and Mistral. This survey provides an in-depth overview of this transition, beginning with foundational techniques before the LLM era, followed by LLM-based embedding models through two main strategies to derive embeddings from LLMs. 1) Direct prompting: We mainly discuss the prompt designs and the underlying rationale for deriving competitive embeddings. 2) Data-centric tuning: We cover extensive aspects that affect tuning an embedding model, including model architecture, training objectives, data constructions, etc. Upon the above, we also cover advanced methods for producing embeddings from longer texts, multilingual, code, cross-modal data, as well as reasoning-aware and other domain-specific scenarios. Furthermore, we discuss factors affecting choices of embedding models, such as performance/efficiency comparisons, dense vs sparse embeddings, pooling strategies, and scaling law. Lastly, the survey highlights the limitations and challenges in adapting LLMs for embeddings, including cross-task embedding quality, trade-offs between efficiency and accuracy, low-resource, long-context, data bias, robustness, etc. This survey serves as a valuable resource for researchers and practitioners by synthesizing current advancements, highlighting key challenges, and offering a comprehensive framework for future work aimed at enhancing the effectiveness and efficiency of LLMs as embedding models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16331v2">Re:Form -- Reducing Human Priors in Scalable Formal Software Verification with RL in LLMs: A Preliminary Study on Dafny</a></div>
    <div class="paper-meta">
      📅 2025-07-25
    </div>
    <details class="paper-abstract">
      Existing informal language-based (e.g., human language) Large Language Models (LLMs) trained with Reinforcement Learning (RL) face a significant challenge: their verification processes, which provide crucial training signals, are neither reliable nor scalable. In fact, the prevalent large proprietary models could hardly generate verifiable programs. A promising yet largely uncharted alternative is formal language-based reasoning. Grounding LLMs in rigorous formal systems where generative models operate in formal language spaces (e.g., Dafny) enables the automatic and mathematically provable verification of their reasoning processes and outcomes. This capability is pivotal for achieving large-scale, reliable formal software verification. It is a common practice to employ human-annotated chain-of-thought and other human priors to induce the reasoning and coding capabilities of LLMs. Unfortunately, it becomes unacceptably all-consuming to provide such priors for supervising complex programming tasks. In this work, we systematically explore ways to reduce human priors with the formal language, Dafny, as the main environment for our pilot study. Our pipeline mainly relies on introducing an automatic and scalable data curation pipeline, and careful RL designs integrated with feedback from the formal language verifier. We introduce DafnyComp, a benchmark of compositional formal programs with auto-formalized specifications for specification reasoning. Our supervised fine-tuning (SFT) stage enables even small models (e.g., 0.5B) to generate syntactically valid and verifiable Dafny code, surpassing proprietary models. RL with regularization further improves performance, achieving stronger generalization to out-of-domain tasks and outperforming all strong baselines on the challenging DafnyComp benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.06270v4">XAI4LLM. Let Machine Learning Models and LLMs Collaborate for Enhanced In-Context Learning in Healthcare</a></div>
    <div class="paper-meta">
      📅 2025-07-25
    </div>
    <details class="paper-abstract">
      Clinical decision support systems require models that are not only highly accurate but also equitable and sensitive to the implications of missed diagnoses. In this study, we introduce a knowledge-guided in-context learning (ICL) framework designed to enable large language models (LLMs) to effectively process structured clinical data. Our approach integrates domain-specific feature groupings, carefully balanced few-shot examples, and task-specific prompting strategies. We systematically evaluate this method across seventy distinct ICL designs by various prompt variations and two different communication styles-natural-language narrative and numeric conversational-and compare its performance to robust classical machine learning (ML) benchmarks on tasks involving heart disease and diabetes prediction. Our findings indicate that while traditional ML models maintain superior performance in balanced precision-recall scenarios, LLMs employing narrative prompts with integrated domain knowledge achieve higher recall and significantly reduce gender bias, effectively narrowing fairness disparities by an order of magnitude. Despite the current limitation of increased inference latency, LLMs provide notable advantages, including the capacity for zero-shot deployment and enhanced equity. This research offers the first comprehensive analysis of ICL design considerations for applying LLMs to tabular clinical tasks and highlights distillation and multimodal extensions as promising directions for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19033v1">SelfRACG: Enabling LLMs to Self-Express and Retrieve for Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-25
      | 💬 Tsinghua&Xiaohongshu
    </div>
    <details class="paper-abstract">
      Existing retrieval-augmented code generation (RACG) methods typically use an external retrieval module to fetch semantically similar code snippets used for generating subsequent fragments. However, even for consecutive code fragments, the content often diverges due to logical progression, resulting in a content gap. This gap undermines the performance of current RACG methods, as \textit{external} retrieval modules based on content matching fail to infer the specific information need of LLMs to generate the next code fragment. Therefore, we propose \textbf{SelfRACG}, a novel paradigm that enables large language models (LLMs) to \textbf{Self}-express their information needs to enhance \textbf{RACG}. Specifically, SelfRACG includes an information need expression module and a two-stage information need-guided training strategy, which encourages LLMs to express their information need. Extensive experiments demonstrate that SelfRACG can retrieve external knowledge that better aligns with the LLM's own information needs, resulting in superior generation performance compared to vanilla RACG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19027v1">SESR-Eval: Dataset for Evaluating LLMs in the Title-Abstract Screening of Systematic Reviews</a></div>
    <div class="paper-meta">
      📅 2025-07-25
      | 💬 12 pages (10 + 2 pages for references)
    </div>
    <details class="paper-abstract">
      Background: The use of large language models (LLMs) in the title-abstract screening process of systematic reviews (SRs) has shown promising results, but suffers from limited performance evaluation. Aims: Create a benchmark dataset to evaluate the performance of LLMs in the title-abstract screening process of SRs. Provide evidence whether using LLMs in title-abstract screening in software engineering is advisable. Method: We start with 169 SR research artifacts and find 24 of those to be suitable for inclusion in the dataset. Using the dataset we benchmark title-abstract screening using 9 LLMs. Results: We present the SESR-Eval (Software Engineering Systematic Review Evaluation) dataset containing 34,528 labeled primary studies, sourced from 24 secondary studies published in software engineering (SE) journals. Most LLMs performed similarly and the differences in screening accuracy between secondary studies are greater than differences between LLMs. The cost of using an LLM is relatively low - less than $40 per secondary study even for the most expensive model. Conclusions: Our benchmark enables monitoring AI performance in the screening task of SRs in software engineering. At present, LLMs are not yet recommended for automating the title-abstract screening process, since accuracy varies widely across secondary studies, and no LLM managed a high recall with reasonable precision. In future, we plan to investigate factors that influence LLM screening performance between studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18631v2">Layer-Aware Representation Filtering: Purifying Finetuning Data to Preserve LLM Safety Alignment</a></div>
    <div class="paper-meta">
      📅 2025-07-25
    </div>
    <details class="paper-abstract">
      With rapid advancement and increasing accessibility of LLMs, fine-tuning aligned models has become a critical step for adapting them to real-world applications, which makes the safety of this fine-tuning process more important than ever. However, recent studies have highlighted a critical challenge: even when fine-tuning with seemingly benign downstream datasets, the safety of aligned LLMs can be compromised, making them more susceptible to malicious instructions. In this paper, we show that fine-tuning datasets often contain samples with safety-degrading features that are not easily identifiable on the surface. These samples can significantly degrade the safety alignment of LLMs during fine-tuning. To address this issue, we propose LARF, a Layer-Aware Representation Filtering method. This method identifies safety-sensitive layers within the LLM and leverages their representations to detect which data samples in the post-training dataset contain safety-degrading features. Experimental results demonstrate that LARF can effectively identify benign data with safety-degrading features. After removing such data, the safety alignment degradation caused by fine-tuning is mitigated. Please see our code at https://github.com/LLLeoLi/LARF.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18993v1">Agent0: Leveraging LLM Agents to Discover Multi-value Features from Text for Enhanced Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-07-25
      | 💬 Agent4IR, KDD '25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) and their associated agent-based frameworks have significantly advanced automated information extraction, a critical component of modern recommender systems. While these multitask frameworks are widely used in code generation, their application in data-centric research is still largely untapped. This paper presents Agent0, an LLM-driven, agent-based system designed to automate information extraction and feature construction from raw, unstructured text. Categorical features are crucial for large-scale recommender systems but are often expensive to acquire. Agent0 coordinates a group of interacting LLM agents to automatically identify the most valuable text aspects for subsequent tasks (such as models or AutoML pipelines). Beyond its feature engineering capabilities, Agent0 also offers an automated prompt-engineering tuning method that utilizes dynamic feedback loops from an oracle. Our findings demonstrate that this closed-loop methodology is both practical and effective for automated feature discovery, which is recognized as one of the most challenging phases in current recommender system development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13666v2">Evaluation of LLM Vulnerabilities to Being Misused for Personalized Disinformation Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-25
      | 💬 ACL 2025 main
    </div>
    <details class="paper-abstract">
      The capabilities of recent large language models (LLMs) to generate high-quality content indistinguishable by humans from human-written texts raises many concerns regarding their misuse. Previous research has shown that LLMs can be effectively misused for generating disinformation news articles following predefined narratives. Their capabilities to generate personalized (in various aspects) content have also been evaluated and mostly found usable. However, a combination of personalization and disinformation abilities of LLMs has not been comprehensively studied yet. Such a dangerous combination should trigger integrated safety filters of the LLMs, if there are some. This study fills this gap by evaluating vulnerabilities of recent open and closed LLMs, and their willingness to generate personalized disinformation news articles in English. We further explore whether the LLMs can reliably meta-evaluate the personalization quality and whether the personalization affects the generated-texts detectability. Our results demonstrate the need for stronger safety-filters and disclaimers, as those are not properly functioning in most of the evaluated LLMs. Additionally, our study revealed that the personalization actually reduces the safety-filter activations; thus effectively functioning as a jailbreak. Such behavior must be urgently addressed by LLM developers and service providers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18957v1">SLICEMATE: Accurate and Scalable Static Program Slicing via LLM-Powered Agents</a></div>
    <div class="paper-meta">
      📅 2025-07-25
    </div>
    <details class="paper-abstract">
      Static program slicing, which extracts the executable portions of a program that affect the values at a specific location, supports many software analysis tasks such as debugging and security auditing. However, traditional slicing tools rely on computationally expensive reachability analysis over dependency graphs, which struggle to scale to large programs and often fail to handle code with incomplete syntax. Recently emerged learning-based methods, while more robust to such cases, still fall short of achieving comparable performance to traditional methods on well-formed code. In this work, we propose SliceMate, a novel static program slicing solution powered by Large Language Model (LLM) agents. It bypasses the need for explicit dependency graph construction and achieving superior slicing accuracy. Concretely, SliceMate integrates three specialized agents: (1) a synthesis agent that produces candidate slices by incrementally expanding the scan scope across functions and files guided by LLM-inferred dependencies; (2) a verification agent that performs conciseness and completeness checks of the candidate slices, detecting missing or irrelevant statements; and (3) a refinement agent that repairs the slices with minimal edits in accordance with the verification results. These agents are orchestrated by a control module that ensures timely convergence and outputs high-quality slices without manual intervention. For rigorous evaluation, we construct a new and high-quality benchmark, SliceBench, comprising 2,200 manually annotated Java and Python programs, with program lengths ranging from 5 to 8,577 lines, significantly larger than those in existing slicing benchmarks. Experimental results show that SliceMate greatly outperforms both traditional and learning-based slicing tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18949v1">Adaptive Learning Systems: Personalized Curriculum Design Using LLM-Powered Analytics</a></div>
    <div class="paper-meta">
      📅 2025-07-25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are revolutionizing the field of education by enabling personalized learning experiences tailored to individual student needs. In this paper, we introduce a framework for Adaptive Learning Systems that leverages LLM-powered analytics for personalized curriculum design. This innovative approach uses advanced machine learning to analyze real-time data, allowing the system to adapt learning pathways and recommend resources that align with each learner's progress. By continuously assessing students, our framework enhances instructional strategies, ensuring that the materials presented are relevant and engaging. Experimental results indicate a marked improvement in both learner engagement and knowledge retention when using a customized curriculum. Evaluations conducted across varied educational environments demonstrate the framework's flexibility and positive influence on learning outcomes, potentially reshaping conventional educational practices into a more adaptive and student-centered model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18153v2">When Noisy Labels Meet Class Imbalance on Graphs: A Graph Augmentation Method with LLM and Pseudo Label</a></div>
    <div class="paper-meta">
      📅 2025-07-25
    </div>
    <details class="paper-abstract">
      Class-imbalanced graph node classification is a practical yet underexplored research problem. Although recent studies have attempted to address this issue, they typically assume clean and reliable labels when processing class-imbalanced graphs. This assumption often violates the nature of real-world graphs, where labels frequently contain noise. Given this gap, this paper systematically investigates robust node classification for class-imbalanced graphs with noisy labels. We propose GraphALP, a novel Graph Augmentation framework based on Large language models (LLMs) and Pseudo-labeling techniques. Specifically, we design an LLM-based oversampling method to generate synthetic minority nodes, producing label-accurate minority nodes to alleviate class imbalance. Based on the class-balanced graphs, we develop a dynamically weighted pseudo-labeling method to obtain high-confidence pseudo labels to reduce label noise ratio. Additionally, we implement a secondary LLM-guided oversampling mechanism to mitigate potential class distribution skew caused by pseudo labels. Experimental results show that GraphALP achieves superior performance over state-of-the-art methods on class-imbalanced graphs with noisy labels.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13618v3">Seed-X: Building Strong Multilingual Translation LLM with 7B Parameters</a></div>
    <div class="paper-meta">
      📅 2025-07-25
    </div>
    <details class="paper-abstract">
      Multilingual translation stands as a challenging task for large language models (LLMs) to handle intricate language patterns and stilted translations that arise in automated translations. In this paper, we introduce Seed-X, a family of open-source LLMs comprising instruct and reasoning models, pushing the limits of translation capability with 7B parameter size. The base model is pre-trained on a diverse, high-quality dataset encompassing both monolingual and bilingual content across 28 languages, harnessing the full potential of multilingual data. The instruct model is then finetuned to translate by Chain-of-Thought (CoT) reasoning and further enhanced through reinforcement learning (RL) to achieve better generalization across diverse language pairs. Seed-X achieves performance comparable to leading closed-source models, including Gemini-2.5 and GPT-4o, across 28 languages, and significantly outperforms larger open-source models in both automatic metrics and human evaluations. We share the best practices through our optimization process, and make the parameter public available for advancing translation research and applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18918v1">Uncovering Cross-Linguistic Disparities in LLMs using Sparse Autoencoders</a></div>
    <div class="paper-meta">
      📅 2025-07-25
    </div>
    <details class="paper-abstract">
      Multilingual large language models (LLMs) exhibit strong cross-linguistic generalization, yet medium to low resource languages underperform on common benchmarks such as ARC-Challenge, MMLU, and HellaSwag. We analyze activation patterns in Gemma-2-2B across all 26 residual layers and 10 languages: Chinese (zh), Russian (ru), Spanish (es), Italian (it), medium to low resource languages including Indonesian (id), Catalan (ca), Marathi (mr), Malayalam (ml), and Hindi (hi), with English (en) as the reference. Using Sparse Autoencoders (SAEs), we reveal systematic disparities in activation patterns. Medium to low resource languages receive up to 26.27 percent lower activations in early layers, with a persistent gap of 19.89 percent in deeper layers. To address this, we apply activation-aware fine-tuning via Low-Rank Adaptation (LoRA), leading to substantial activation gains, such as 87.69 percent for Malayalam and 86.32 percent for Hindi, while maintaining English retention at approximately 91 percent. After fine-tuning, benchmark results show modest but consistent improvements, highlighting activation alignment as a key factor in enhancing multilingual LLM performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14561v3">Can LLMs Predict Citation Intent? An Experimental Analysis of In-context Learning and Fine-tuning on Open LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-25
      | 💬 Accepted for publication on TPDL 2025
    </div>
    <details class="paper-abstract">
      This work investigates the ability of open Large Language Models (LLMs) to predict citation intent through in-context learning and fine-tuning. Unlike traditional approaches relying on domain-specific pre-trained models like SciBERT, we demonstrate that general-purpose LLMs can be adapted to this task with minimal task-specific data. We evaluate twelve model variations across five prominent open LLM families using zero-, one-, few-, and many-shot prompting. Our experimental study identifies the top-performing model and prompting parameters through extensive in-context learning experiments. We then demonstrate the significant impact of task-specific adaptation by fine-tuning this model, achieving a relative F1-score improvement of 8% on the SciCite dataset and 4.3% on the ACL-ARC dataset compared to the instruction-tuned baseline. These findings provide valuable insights for model selection and prompt engineering. Additionally, we make our end-to-end evaluation framework and models openly available for future use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.14117v4">An Investigation of Prompt Variations for Zero-shot LLM-based Rankers</a></div>
    <div class="paper-meta">
      📅 2025-07-25
      | 💬 Accepted for publication at the 47th European Conference on Information Retrieval (ECIR 2025)
    </div>
    <details class="paper-abstract">
      We provide a systematic understanding of the impact of specific components and wordings used in prompts on the effectiveness of rankers based on zero-shot Large Language Models (LLMs). Several zero-shot ranking methods based on LLMs have recently been proposed. Among many aspects, methods differ across (1) the ranking algorithm they implement, e.g., pointwise vs. listwise, (2) the backbone LLMs used, e.g., GPT3.5 vs. FLAN-T5, (3) the components and wording used in prompts, e.g., the use or not of role-definition (role-playing) and the actual words used to express this. It is currently unclear whether performance differences are due to the underlying ranking algorithm, or because of spurious factors such as better choice of words used in prompts. This confusion risks to undermine future research. Through our large-scale experimentation and analysis, we find that ranking algorithms do contribute to differences between methods for zero-shot LLM ranking. However, so do the LLM backbones -- but even more importantly, the choice of prompt components and wordings affect the ranking. In fact, in our experiments, we find that, at times, these latter elements have more impact on the ranker's effectiveness than the actual ranking algorithms, and that differences among ranking methods become more blurred when prompt variations are considered.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13175v2">Black Box Deployed -- Functional Criteria for Artificial Moral Agents in the LLM Era</a></div>
    <div class="paper-meta">
      📅 2025-07-25
      | 💬 42 pages. Supplementary material included at end of article
    </div>
    <details class="paper-abstract">
      The advancement of powerful yet opaque large language models (LLMs) necessitates a fundamental revision of the philosophical criteria used to evaluate artificial moral agents (AMAs). Pre-LLM frameworks often relied on the assumption of transparent architectures, which LLMs defy due to their stochastic outputs and opaque internal states. This paper argues that traditional ethical criteria are pragmatically obsolete for LLMs due to this mismatch. Engaging with core themes in the philosophy of technology, this paper proffers a revised set of ten functional criteria to evaluate LLM-based artificial moral agents: moral concordance, context sensitivity, normative integrity, metaethical awareness, system resilience, trustworthiness, corrigibility, partial transparency, functional autonomy, and moral imagination. These guideposts, applied to what we term "SMA-LLS" (Simulating Moral Agency through Large Language Systems), aim to steer AMAs toward greater alignment and beneficial societal integration in the coming years. We illustrate these criteria using hypothetical scenarios involving an autonomous public bus (APB) to demonstrate their practical applicability in morally salient contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15956v3">Do Large Language Models Have an English Accent? Evaluating and Improving the Naturalness of Multilingual LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-25
      | 💬 ACL 2025
    </div>
    <details class="paper-abstract">
      Current Large Language Models (LLMs) are predominantly designed with English as the primary language, and even the few that are multilingual tend to exhibit strong English-centric biases. Much like speakers who might produce awkward expressions when learning a second language, LLMs often generate unnatural outputs in non-English languages, reflecting English-centric patterns in both vocabulary and grammar. Despite the importance of this issue, the naturalness of multilingual LLM outputs has received limited attention. In this paper, we address this gap by introducing novel automatic corpus-level metrics to assess the lexical and syntactic naturalness of LLM outputs in a multilingual context. Using our new metrics, we evaluate state-of-the-art LLMs on a curated benchmark in French and Chinese, revealing a tendency towards English-influenced patterns. To mitigate this issue, we also propose a simple and effective alignment method to improve the naturalness of an LLM in a target language and domain, achieving consistent improvements in naturalness without compromising the performance on general-purpose benchmarks. Our work highlights the importance of developing multilingual metrics, resources and methods for the new wave of multilingual LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19643v1">Can You Share Your Story? Modeling Clients' Metacognition and Openness for LLM Therapist Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-07-25
      | 💬 Published at ACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Understanding clients' thoughts and beliefs is fundamental in counseling, yet current evaluations of LLM therapists often fail to assess this ability. Existing evaluation methods rely on client simulators that clearly disclose internal states to the therapist, making it difficult to determine whether an LLM therapist can uncover unexpressed perspectives. To address this limitation, we introduce MindVoyager, a novel evaluation framework featuring a controllable and realistic client simulator which dynamically adapts itself based on the ongoing counseling session, offering a more realistic and challenging evaluation environment. We further introduce evaluation metrics that assess the exploration ability of LLM therapists by measuring their thorough understanding of client's beliefs and thoughts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15606v3">LoX: Low-Rank Extrapolation Robustifies LLM Safety Against Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2025-07-25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become indispensable in real-world applications. However, their widespread adoption raises significant safety concerns, particularly in responding to socially harmful questions. Despite substantial efforts to improve model safety through alignment, aligned models can still have their safety protections undermined by subsequent fine-tuning - even when the additional training data appears benign. In this paper, we empirically demonstrate that this vulnerability stems from the sensitivity of safety-critical low-rank subspaces in LLM parameters to fine-tuning. Building on this insight, we propose a novel training-free method, termed Low-Rank Extrapolation (LoX), to enhance safety robustness by extrapolating the safety subspace of an aligned LLM. Our experimental results confirm the effectiveness of LoX, demonstrating significant improvements in robustness against both benign and malicious fine-tuning attacks while preserving the model's adaptability to new tasks. For instance, LoX leads to 11% to 54% absolute reductions in attack success rates (ASR) facing benign or malicious fine-tuning attacks. By investigating the ASR landscape of parameters, we attribute the success of LoX to that the extrapolation moves LLM parameters to a flatter zone, thereby less sensitive to perturbations. The code is available at github.com/VITA-Group/LoX.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19608v1">DeltaLLM: A Training-Free Framework Exploiting Temporal Sparsity for Efficient Edge LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-07-25
    </div>
    <details class="paper-abstract">
      Deploying Large Language Models (LLMs) on edge devices remains challenging due to their quadratically increasing computations with the sequence length. Existing studies for dynamic attention pruning are designed for hardware with massively parallel computation capabilities, such as GPUs or TPUs, and aim at long context lengths (e.g., 64K), making them unsuitable for edge scenarios. We present DeltaLLM, a training-free framework that exploits temporal sparsity in attention patterns to enable efficient LLM inference across both the prefilling and decoding stages, on resource-constrained edge devices. DeltaLLM introduces an accuracy- and memory-aware delta matrix construction strategy that introduces temporal sparsity, and a context-aware hybrid attention mechanism that combines full attention in a local context window with delta approximation outside it to increase accuracy. We evaluate our framework on the edge-device-friendly BitNet-b1.58-2B-4T model and Llama3.2-1B-Instruct model across diverse language tasks. The results show that on BitNet, our framework increases the attention sparsity from 0% to 60% during the prefilling stage with slight accuracy improvement on the WG task, and 0% to 57% across both the prefilling and decoding stages, with even higher F1 score from 29.63 to 30.97 on SQuAD-v2 task. On the Llama model, it can also achieve up to 60% sparsity during the prefilling stage and around 57% across both stages with negligible accuracy drop. These results demonstrate that DeltaLLM offers a promising solution for efficient edge deployment, requiring no fine-tuning and seamlessly integrating with existing inference pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19570v1">MCP4EDA: LLM-Powered Model Context Protocol RTL-to-GDSII Automation with Backend Aware Synthesis Optimization</a></div>
    <div class="paper-meta">
      📅 2025-07-25
      | 💬 7 pages, 5 figures Keywords: Model Context Protocol, Electronic Design Automation, Large Language Models, Synthesis Optimization
    </div>
    <details class="paper-abstract">
      This paper presents MCP4EDA, the first Model Context Protocol server that enables Large Language Models (LLMs) to control and optimize the complete open-source RTL-to-GDSII design flow through natural language interaction. The system integrates Yosys synthesis, Icarus Verilog simulation, OpenLane place-and-route, GTKWave analysis, and KLayout visualization into a unified LLM-accessible interface, enabling designers to execute complex multi-tool EDA workflows conversationally via AI assistants such as Claude Desktop and Cursor IDE. The principal contribution is a backend-aware synthesis optimization methodology wherein LLMs analyze actual post-layout timing, power, and area metrics from OpenLane results to iteratively refine synthesis TCL scripts, establishing a closed-loop optimization system that bridges the traditional gap between synthesis estimates and physical implementation reality. In contrast to conventional flows that rely on wire-load models, this methodology leverages real backend performance data to guide synthesis parameter tuning, optimization sequence selection, and constraint refinement, with the LLM functioning as an intelligent design space exploration agent. Experimental evaluation on representative digital designs demonstrates 15-30% improvements in timing closure and 10-20% area reduction compared to default synthesis flows, establishing MCP4EDA as the first practical LLM-controlled end-to-end open-source EDA automation system. The code and demo are avaiable at: http://www.agent4eda.com/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19562v1">PennyCoder: Efficient Domain-Specific LLMs for PennyLane-Based Quantum Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-25
      | 💬 6 pages, 5 figures, 3 tables, paper accepted to QCE 2025
    </div>
    <details class="paper-abstract">
      The growing demand for robust quantum programming frameworks has unveiled a critical limitation: current large language model (LLM) based quantum code assistants heavily rely on remote APIs, introducing challenges related to privacy, latency, and excessive usage costs. Addressing this gap, we propose PennyCoder, a novel lightweight framework for quantum code generation, explicitly designed for local and embedded deployment to enable on-device quantum programming assistance without external API dependence. PennyCoder leverages a fine-tuned version of the LLaMA 3.1-8B model, adapted through parameter-efficient Low-Rank Adaptation (LoRA) techniques combined with domain-specific instruction tuning optimized for the specialized syntax and computational logic of quantum programming in PennyLane, including tasks in quantum machine learning and quantum reinforcement learning. Unlike prior work focused on cloud-based quantum code generation, our approach emphasizes device-native operability while maintaining high model efficacy. We rigorously evaluated PennyCoder over a comprehensive quantum programming dataset, achieving 44.3% accuracy with our fine-tuned model (compared to 33.7% for the base LLaMA 3.1-8B and 40.1% for the RAG-augmented baseline), demonstrating a significant improvement in functional correctness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18631v1">Layer-Aware Representation Filtering: Purifying Finetuning Data to Preserve LLM Safety Alignment</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      With rapid advancement and increasing accessibility of LLMs, fine-tuning aligned models has become a critical step for adapting them to real-world applications, which makes the safety of this fine-tuning process more important than ever. However, recent studies have highlighted a critical challenge: even when fine-tuning with seemingly benign downstream datasets, the safety of aligned LLMs can be compromised, making them more susceptible to malicious instructions. In this paper, we show that fine-tuning datasets often contain samples with safety-degrading features that are not easily identifiable on the surface. These samples can significantly degrade the safety alignment of LLMs during fine-tuning. To address this issue, we propose LARF, a \textbf{L}ayer-\textbf{A}ware \textbf{R}epresentation \textbf{F}iltering method. This method identifies safety-sensitive layers within the LLM and leverages their representations to detect which data samples in the post-training dataset contain safety-degrading features. Experimental results demonstrate that LARF can effectively identify benign data with safety-degrading features. After removing such data, the safety alignment degradation caused by fine-tuning is mitigated. Please see our code at \href{https://github.com/LLLeoLi/LARF}{https://github.com/LLLeoLi/LARF}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18607v1">Explainable Mapper: Charting LLM Embedding Spaces Using Perturbation-Based Explanation and Verification Agents</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) produce high-dimensional embeddings that capture rich semantic and syntactic relationships between words, sentences, and concepts. Investigating the topological structures of LLM embedding spaces via mapper graphs enables us to understand their underlying structures. Specifically, a mapper graph summarizes the topological structure of the embedding space, where each node represents a topological neighborhood (containing a cluster of embeddings), and an edge connects two nodes if their corresponding neighborhoods overlap. However, manually exploring these embedding spaces to uncover encoded linguistic properties requires considerable human effort. To address this challenge, we introduce a framework for semi-automatic annotation of these embedding properties. To organize the exploration process, we first define a taxonomy of explorable elements within a mapper graph such as nodes, edges, paths, components, and trajectories. The annotation of these elements is executed through two types of customizable LLM-based agents that employ perturbation techniques for scalable and automated analysis. These agents help to explore and explain the characteristics of mapper elements and verify the robustness of the generated explanations. We instantiate the framework within a visual analytics workspace and demonstrate its effectiveness through case studies. In particular, we replicate findings from prior research on BERT's embedding properties across various layers of its architecture and provide further observations into the linguistic properties of topological neighborhoods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18584v1">AQuilt: Weaving Logic and Self-Inspection into Low-Cost, High-Relevance Data Synthesis for Specialist LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 32 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Despite the impressive performance of large language models (LLMs) in general domains, they often underperform in specialized domains. Existing approaches typically rely on data synthesis methods and yield promising results by using unlabeled data to capture domain-specific features. However, these methods either incur high computational costs or suffer from performance limitations, while also demonstrating insufficient generalization across different tasks. To address these challenges, we propose AQuilt, a framework for constructing instruction-tuning data for any specialized domains from corresponding unlabeled data, including Answer, Question, Unlabeled data, Inspection, Logic, and Task type. By incorporating logic and inspection, we encourage reasoning processes and self-inspection to enhance model performance. Moreover, customizable task instructions enable high-quality data generation for any task. As a result, we construct a dataset of 703k examples to train a powerful data synthesis model. Experiments show that AQuilt is comparable to DeepSeek-V3 while utilizing just 17% of the production cost. Further analysis demonstrates that our generated data exhibits higher relevance to downstream tasks. Source code, models, and scripts are available at https://github.com/Krueske/AQuilt.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16809v2">LingBench++: A Linguistically-Informed Benchmark and Reasoning Framework for Multi-Step and Cross-Cultural Inference with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 42p, 17f, 10t. Revisions: Merged paragraphs in Intro to emphasize contributions. Clarified benchmark design (Sec 3.5.1). Added single-agent, OpenAI-guided & 6-round experiments (Sec 5.2). Note: we only ran each experiment once; statistical tests are needed for strong claims. Revised Sec 6. Added acknowledgements, 2 new co-authors, and corrected typos/grammar
    </div>
    <details class="paper-abstract">
      We propose LingBench++, a linguistically-informed benchmark and reasoning framework designed to evaluate large language models (LLMs) on complex linguistic tasks inspired by the International Linguistics Olympiad (IOL). Unlike prior benchmarks that focus solely on final answer accuracy, LingBench++ provides structured reasoning traces, stepwise evaluation protocols, and rich typological metadata across over 90 low-resource and cross-cultural languages. We further develop a multi-agent architecture integrating grammatical knowledge retrieval, tool-augmented reasoning, and deliberate hypothesis testing. Through systematic comparisons of baseline and our proposed agentic models, we demonstrate that models equipped with external knowledge sources and iterative reasoning outperform single-pass approaches in both accuracy and interpretability. LingBench++ offers a comprehensive foundation for advancing linguistically grounded, culturally informed, and cognitively plausible reasoning in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18560v1">HARLF: Hierarchical Reinforcement Learning and Lightweight LLM-Driven Sentiment Integration for Financial Portfolio Optimization</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      This paper presents a novel hierarchical framework for portfolio optimization, integrating lightweight Large Language Models (LLMs) with Deep Reinforcement Learning (DRL) to combine sentiment signals from financial news with traditional market indicators. Our three-tier architecture employs base RL agents to process hybrid data, meta-agents to aggregate their decisions, and a super-agent to merge decisions based on market data and sentiment analysis. Evaluated on data from 2018 to 2024, after training on 2000-2017, the framework achieves a 26% annualized return and a Sharpe ratio of 1.2, outperforming equal-weighted and S&P 500 benchmarks. Key contributions include scalable cross-modal integration, a hierarchical RL structure for enhanced stability, and open-source reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14783v2">Omni-Thinker: Scaling Cross-Domain Generalization in LLMs via Multi-Task RL with Hybrid Rewards</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      The advancement of general-purpose artificial intelligence relies on large language models (LLMs) that excel across a wide range of tasks, from structured reasoning to creative generation. However, post-training methods like Supervised Fine-Tuning (SFT) often struggle with generalization, favoring memorization over transferable learning. In this work, we introduce Omni-Thinker, a unified reinforcement learning (RL) framework that enhances LLM performance across diverse tasks by combining rule-based verifiable rewards with generative preference signals via LLM-as-a-Judge evaluations. Our approach enables consistent optimization across task types and scales RL-based training to subjective domains. We further investigate training strategies, demonstrating that a curriculum-based progression that orders tasks from structured to open-ended improves performance and reduces forgetting. Experimental results across four domains reveal that curriculum learning improves performance by 5.2% over joint training and 9.1% over model merging. These results highlight the importance of task-aware sampling and hybrid supervision in scaling RL-based post-training for general-purpose LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18553v1">The Geometry of LLM Quantization: GPTQ as Babai's Nearest Plane Algorithm</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      Quantizing the weights of large language models (LLMs) from 16-bit to lower bitwidth is the de facto approach to deploy massive transformers onto more affordable accelerators. GPTQ emerged as one of the standard methods for one-shot post-training quantization at LLM scale. Yet, its inner workings are described as a sequence of ad-hoc algebraic updates that obscure any geometric meaning or worst-case guarantees. In this work, we show that, when executed back-to-front (from the last to first dimension) for a linear layer, GPTQ is mathematically identical to Babai's nearest plane algorithm for the classical closest vector problem (CVP) on a lattice defined by the Hessian matrix of the layer's inputs. This equivalence is based on a sophisticated mathematical argument, and has two analytical consequences: (i) the GPTQ error propagation step gains an intuitive geometric interpretation; (ii) GPTQ inherits the error upper bound of Babai's algorithm under the no-clipping condition. Taken together, these results place GPTQ on firm theoretical footing and open the door to importing decades of progress in lattice algorithms towards the design of future quantization algorithms for billion-parameter models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08989v3">Zeroth-Order Fine-Tuning of LLMs in Random Subspaces</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 ICCV 2025 camera-ready version
    </div>
    <details class="paper-abstract">
      Fine-tuning Large Language Models (LLMs) has proven effective for a variety of downstream tasks. However, as LLMs grow in size, the memory demands for backpropagation become increasingly prohibitive. Zeroth-order (ZO) optimization methods offer a memory-efficient alternative by using forward passes to estimate gradients, but the variance of gradient estimates typically scales linearly with the model's parameter dimension$\unicode{x2013}$a significant issue for LLMs. In this paper, we propose the random Subspace Zeroth-order (SubZero) optimization to address the challenges posed by LLMs' high dimensionality. We introduce a low-rank perturbation tailored for LLMs that significantly reduces memory consumption while improving training performance. Additionally, we prove that our gradient estimation closely approximates the backpropagation gradient, exhibits lower variance than traditional ZO methods, and ensures convergence when combined with SGD. Experimental results show that SubZero enhances fine-tuning performance and achieves faster convergence compared to standard ZO approaches like MeZO across various language modeling tasks. Code is available at https://github.com/zimingyy/SubZero.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02976v2">Are AI-Generated Fixes Secure? Analyzing LLM and Agent Patches on SWE-bench</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) and their agentic frameworks are increasingly adopted to automate software development tasks such as issue resolution and program repair. While prior work has identified security risks in LLM-generated code, most evaluations have focused on synthetic or isolated settings, leaving open questions about the security of these systems in real-world development contexts. In this study, we present the first large-scale security analysis of LLM-generated patches using 20,000+ issues from the SWE-bench dataset. We evaluate patches produced by a standalone LLM (Llama 3.3) and compare them to developer-written patches. We also assess the security of patches generated by three top-performing agentic frameworks (OpenHands, AutoCodeRover, HoneyComb) on a subset of our data. Finally, we analyze a wide range of code, issue, and project-level factors to understand the conditions under which LLMs and agents are most likely to generate insecure code. Our findings reveal that the standalone LLM introduces nearly 9x more new vulnerabilities than developers, with many of these exhibiting unique patterns not found in developers' code. Agentic workflows also generate a significant number of vulnerabilities, particularly when granting LLMs more autonomy, potentially increasing the likelihood of misinterpreting project context or task requirements. We find that vulnerabilities are more likely to occur in LLM patches associated with a higher number of files, more lines of generated code, and GitHub issues that lack specific code snippets or information about the expected code behavior and steps to reproduce. These results suggest that contextual factors play a critical role in the security of the generated code and point toward the need for proactive risk assessment methods that account for both code and issue-level information to complement existing vulnerability detection tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18479v1">How Well Do LLMs Predict Prerequisite Skills? Zero-Shot Comparison to Expert-Defined Concepts</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      Prerequisite skills - foundational competencies required before mastering more advanced concepts - are important for supporting effective learning, assessment, and skill-gap analysis. Traditionally curated by domain experts, these relationships are costly to maintain and difficult to scale. This paper investigates whether large language models (LLMs) can predict prerequisite skills in a zero-shot setting, using only natural language descriptions and without task-specific fine-tuning. We introduce ESCO-PrereqSkill, a benchmark dataset constructed from the ESCO taxonomy, comprising 3,196 skills and their expert-defined prerequisite links. Using a standardized prompting strategy, we evaluate 13 state-of-the-art LLMs, including GPT-4, Claude 3, Gemini, LLaMA 4, Qwen2, and DeepSeek, across semantic similarity, BERTScore, and inference latency. Our results show that models such as LLaMA4-Maverick, Claude-3-7-Sonnet, and Qwen2-72B generate predictions that closely align with expert ground truth, demonstrating strong semantic reasoning without supervision. These findings highlight the potential of LLMs to support scalable prerequisite skill modeling for applications in personalized learning, intelligent tutoring, and skill-based recommender systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18455v1">LLM-based Embedders for Prior Case Retrieval</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 Accepted in Recent Advancements in Natural Language Processing (RANLP 2025) conference
    </div>
    <details class="paper-abstract">
      In common law systems, legal professionals such as lawyers and judges rely on precedents to build their arguments. As the volume of cases has grown massively over time, effectively retrieving prior cases has become essential. Prior case retrieval (PCR) is an information retrieval (IR) task that aims to automatically identify the most relevant court cases for a specific query from a large pool of potential candidates. While IR methods have seen several paradigm shifts over the last few years, the vast majority of PCR methods continue to rely on traditional IR methods, such as BM25. The state-of-the-art deep learning IR methods have not been successful in PCR due to two key challenges: i. Lengthy legal text limitation; when using the powerful BERT-based transformer models, there is a limit of input text lengths, which inevitably requires to shorten the input via truncation or division with a loss of legal context information. ii. Lack of legal training data; due to data privacy concerns, available PCR datasets are often limited in size, making it difficult to train deep learning-based models effectively. In this research, we address these challenges by leveraging LLM-based text embedders in PCR. LLM-based embedders support longer input lengths, and since we use them in an unsupervised manner, they do not require training data, addressing both challenges simultaneously. In this paper, we evaluate state-of-the-art LLM-based text embedders in four PCR benchmark datasets and show that they outperform BM25 and supervised transformer-based models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18442v1">AraTable: Benchmarking LLMs' Reasoning and Understanding of Arabic Tabular Data</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      The cognitive and reasoning abilities of large language models (LLMs) have enabled remarkable progress in natural language processing. However, their performance in interpreting structured data, especially in tabular formats, remains limited. Although benchmarks for English tabular data are widely available, Arabic is still underrepresented because of the limited availability of public resources and its unique language features. To address this gap, we present AraTable, a novel and comprehensive benchmark designed to evaluate the reasoning and understanding capabilities of LLMs when applied to Arabic tabular data. AraTable consists of various evaluation tasks, such as direct question answering, fact verification, and complex reasoning, involving a wide range of Arabic tabular sources. Our methodology follows a hybrid pipeline, where initial content is generated by LLMs and subsequently filtered and verified by human experts to ensure high dataset quality. Initial analyses using AraTable show that, while LLMs perform adequately on simpler tabular tasks such as direct question answering, they continue to face significant cognitive challenges when tasks require deeper reasoning and fact verification. This indicates that there are substantial opportunities for future work to improve performance on complex tabular reasoning tasks. We also propose a fully automated evaluation framework that uses a self-deliberation mechanism and achieves performance nearly identical to that of human judges. This research provides a valuable, publicly available resource and evaluation framework that can help accelerate the development of foundational models for processing and analysing Arabic structured data.
    </details>
</div>
