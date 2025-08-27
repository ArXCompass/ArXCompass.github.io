# llm - 2025_08

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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15338v1">DiagECG: An LLM-Driven Framework for Diagnostic Reasoning via Discretized ECG Tokenization</a></div>
    <div class="paper-meta">
      📅 2025-08-21
    </div>
    <details class="paper-abstract">
      Electrocardiography plays a central role in cardiovascular diagnostics, yet existing automated approaches often struggle to generalize across clinical tasks and offer limited support for open-ended reasoning. We present DiagECG, a novel framework that integrates time-series and language modeling by enabling large language models to process 12-lead ECG signals for clinical text generation tasks. Our approach discretizes continuous ECG embeddings into symbolic tokens using a lead-independent encoder and quantization module. These tokens are then used to extend the vocabulary of LLM, allowing the model to handle both ECG and natural language inputs in a unified manner. To bridge the modality gap, we pretrain the model on an autoregressive ECG forecasting task, enabling the LLM to model temporal dynamics using its native language modeling capabilities. Finally, we perform instruction tuning on both ECG question answering and diagnostic report generation. Without modifying the core model, DiagECG achieves strong performance across tasks while maintaining generalization to out-of-distribution settings. Extensive experiments demonstrate the effectiveness of each component and highlight the potential of integrating symbolic ECG representations into LLMs for medical reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15310v1">IPIGuard: A Novel Tool Dependency Graph-Based Defense Against Indirect Prompt Injection in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-08-21
      | 💬 EMNLP 2025
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents are widely deployed in real-world applications, where they leverage tools to retrieve and manipulate external data for complex tasks. However, when interacting with untrusted data sources (e.g., fetching information from public websites), tool responses may contain injected instructions that covertly influence agent behaviors and lead to malicious outcomes, a threat referred to as Indirect Prompt Injection (IPI). Existing defenses typically rely on advanced prompting strategies or auxiliary detection models. While these methods have demonstrated some effectiveness, they fundamentally rely on assumptions about the model's inherent security, which lacks structural constraints on agent behaviors. As a result, agents still retain unrestricted access to tool invocations, leaving them vulnerable to stronger attack vectors that can bypass the security guardrails of the model. To prevent malicious tool invocations at the source, we propose a novel defensive task execution paradigm, called IPIGuard, which models the agents' task execution process as a traversal over a planned Tool Dependency Graph (TDG). By explicitly decoupling action planning from interaction with external data, IPIGuard significantly reduces unintended tool invocations triggered by injected instructions, thereby enhancing robustness against IPI attacks. Experiments on the AgentDojo benchmark show that IPIGuard achieves a superior balance between effectiveness and robustness, paving the way for the development of safer agentic systems in dynamic environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15305v1">Coarse-to-Fine Grounded Memory for LLM Agent Planning</a></div>
    <div class="paper-meta">
      📅 2025-08-21
      | 💬 Accepted to EMNLP 2025 Main Conference;27 pages,15 figures
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have driven growing interest in LLM-based agents for complex planning tasks. To avoid costly agent training, many studies adopted memory mechanism that enhances LLM with offline experiences or online trajectory analysis. However, existing works focus on single-granularity memory derived from dynamic environmental interactions, which are inherently constrained by the quality of the collected experiences. This limitation, in turn, constrain the diversity of knowledge and the flexibility of planning. We propose Coarse-to-Fine Grounded Memory (\Ours{}), a novel framework that grounds coarse-to-fine memories with LLM, thereby fully leverage them for flexible adaptation to diverse scenarios. \Ours{} grounds environmental information into coarse-grained focus points to guide experience collection in training tasks, followed by grounding of actionable hybrid-grained tips from each experience. At inference, \Ours{} retrieves task-relevant experiences and tips to support planning. When facing environmental anomalies, the LLM grounds the current situation into fine-grained key information, enabling flexible self-QA reflection and plan correction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15243v1">Comp-X: On Defining an Interactive Learned Image Compression Paradigm With Expert-driven LLM Agent</a></div>
    <div class="paper-meta">
      📅 2025-08-21
    </div>
    <details class="paper-abstract">
      We present Comp-X, the first intelligently interactive image compression paradigm empowered by the impressive reasoning capability of large language model (LLM) agent. Notably, commonly used image codecs usually suffer from limited coding modes and rely on manual mode selection by engineers, making them unfriendly for unprofessional users. To overcome this, we advance the evolution of image coding paradigm by introducing three key innovations: (i) multi-functional coding framework, which unifies different coding modes of various objective/requirements, including human-machine perception, variable coding, and spatial bit allocation, into one framework. (ii) interactive coding agent, where we propose an augmented in-context learning method with coding expert feedback to teach the LLM agent how to understand the coding request, mode selection, and the use of the coding tools. (iii) IIC-bench, the first dedicated benchmark comprising diverse user requests and the corresponding annotations from coding experts, which is systematically designed for intelligently interactive image compression evaluation. Extensive experimental results demonstrate that our proposed Comp-X can understand the coding requests efficiently and achieve impressive textual interaction capability. Meanwhile, it can maintain comparable compression performance even with a single coding framework, providing a promising avenue for artificial general intelligence (AGI) in image compression.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03730v3">Teuken-7B-Base & Teuken-7B-Instruct: Towards European LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-21
    </div>
    <details class="paper-abstract">
      We present two multilingual LLMs, Teuken 7B-base and Teuken 7B-instruct, designed to embrace Europe's linguistic diversity by supporting all 24 official languages of the European Union. Trained on a dataset comprising around 60% non-English data and utilizing a custom multilingual tokenizer, our models address the limitations of existing LLMs that predominantly focus on English or a few high-resource languages. We detail the models' development principles, i.e., data composition, tokenizer optimization, and training methodologies. The models demonstrate strong performance across multilingual benchmarks, as evidenced by their performance on European versions of ARC, HellaSwag, and TruthfulQA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15204v1">R-ConstraintBench: Evaluating LLMs on NP-Complete Scheduling</a></div>
    <div class="paper-meta">
      📅 2025-08-21
    </div>
    <details class="paper-abstract">
      Effective scheduling under tight resource, timing, and operational constraints underpins large-scale planning across sectors such as capital projects, manufacturing, logistics, and IT fleet transitions. However, the reliability of large language models (LLMs) when reasoning under high-constraint regimes is insufficiently characterized. To address this gap, we present R-ConstraintBench, a scalable framework that evaluates models on Resource-Constrained Project Scheduling Problems (RCPSP), an NP-Complete feasibility class, while difficulty increases via linear growth in constraints. R-ConstraintBench incrementally increases non-redundant precedence constraints in Directed Acyclic Graphs (DAGs) and then introduces downtime, temporal windows, and disjunctive constraints. As an illustrative example, we instantiate the benchmark in a data center migration setting and evaluate multiple LLMs using feasibility and error analysis, identifying degradation thresholds and constraint types most associated with failure. Empirically, strong models are near-ceiling on precedence-only DAGs, but feasibility performance collapses when downtime, temporal windows, and disjunctive constraints interact, implicating constraint interaction, not graph depth, as the principal bottleneck. Performance on clean synthetic ramps also does not guarantee transfer to domain-grounded scenarios, underscoring limited generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06992v2">MCA-RG: Enhancing LLMs with Medical Concept Alignment for Radiology Report Generation</a></div>
    <div class="paper-meta">
      📅 2025-08-21
      | 💬 MICCAI 2025
    </div>
    <details class="paper-abstract">
      Despite significant advancements in adapting Large Language Models (LLMs) for radiology report generation (RRG), clinical adoption remains challenging due to difficulties in accurately mapping pathological and anatomical features to their corresponding text descriptions. Additionally, semantic agnostic feature extraction further hampers the generation of accurate diagnostic reports. To address these challenges, we introduce Medical Concept Aligned Radiology Report Generation (MCA-RG), a knowledge-driven framework that explicitly aligns visual features with distinct medical concepts to enhance the report generation process. MCA-RG utilizes two curated concept banks: a pathology bank containing lesion-related knowledge, and an anatomy bank with anatomical descriptions. The visual features are aligned with these medical concepts and undergo tailored enhancement. We further propose an anatomy-based contrastive learning procedure to improve the generalization of anatomical features, coupled with a matching loss for pathological features to prioritize clinically relevant regions. Additionally, a feature gating mechanism is employed to filter out low-quality concept features. Finally, the visual features are corresponding to individual medical concepts, and are leveraged to guide the report generation process. Experiments on two public benchmarks (MIMIC-CXR and CheXpert Plus) demonstrate that MCA-RG achieves superior performance, highlighting its effectiveness in radiology report generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15146v1">QueryGenie: Making LLM-Based Database Querying Transparent and Controllable</a></div>
    <div class="paper-meta">
      📅 2025-08-21
      | 💬 Accepted by The 38th Annual ACM Symposium on User Interface Software and Technology (UIST Adjunct '25), September 28-October 1, 2025, Busan, Republic of Korea
    </div>
    <details class="paper-abstract">
      Conversational user interfaces powered by large language models (LLMs) have significantly lowered the technical barriers to database querying. However, existing tools still encounter several challenges, such as misinterpretation of user intent, generation of hallucinated content, and the absence of effective mechanisms for human feedback-all of which undermine their reliability and practical utility. To address these issues and promote a more transparent and controllable querying experience, we proposed QueryGenie, an interactive system that enables users to monitor, understand, and guide the LLM-driven query generation process. Through incremental reasoning, real-time validation, and responsive interaction mechanisms, users can iteratively refine query logic and ensure alignment with their intent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16695v1">Do Cognitively Interpretable Reasoning Traces Improve LLM Performance?</a></div>
    <div class="paper-meta">
      📅 2025-08-21
    </div>
    <details class="paper-abstract">
      Recent progress in reasoning-oriented Large Language Models (LLMs) has been driven by introducing Chain-of-Thought (CoT) traces, where models generate intermediate reasoning traces before producing an answer. These traces, as in DeepSeek R1, are not only used to guide inference but also serve as supervision signals for distillation into smaller models. A common but often implicit assumption is that CoT traces should be semantically meaningful and interpretable to the end user. While recent research questions the need for semantic nature of these traces, in this paper, we ask: ``\textit{Must CoT reasoning traces be interpretable to enhance LLM task performance?}" We investigate this question in the Open Book Question-Answering domain by supervised fine-tuning LLaMA and Qwen models on four types of reasoning traces: (1) DeepSeek R1 traces, (2) LLM-generated summaries of R1 traces, (3) LLM-generated post-hoc explanations of R1 traces, and (4) algorithmically generated verifiably correct traces. To quantify the trade-off between interpretability and performance, we further conduct a human-subject study with 100 participants rating the interpretability of each trace type. Our results reveal a striking mismatch: while fine-tuning on R1 traces yields the strongest performance, participants judged these traces to be the least interpretable. These findings suggest that it is useful to decouple intermediate tokens from end user interpretability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07143v2">Ask Patients with Patience: Enabling LLMs for Human-Centric Medical Dialogue with Grounded Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-08-21
    </div>
    <details class="paper-abstract">
      The severe shortage of medical doctors limits access to timely and reliable healthcare, leaving millions underserved. Large language models (LLMs) offer a potential solution but struggle in real-world clinical interactions. Many LLMs are not grounded in authoritative medical guidelines and fail to transparently manage diagnostic uncertainty. Their language is often rigid and mechanical, lacking the human-like qualities essential for patient trust. To address these challenges, we propose Ask Patients with Patience (APP), a multi-turn LLM-based medical assistant designed for grounded reasoning, transparent diagnoses, and human-centric interaction. APP enhances communication by eliciting user symptoms through empathetic dialogue, significantly improving accessibility and user engagement. It also incorporates Bayesian active learning to support transparent and adaptive diagnoses. The framework is built on verified medical guidelines, ensuring clinically grounded and evidence-based reasoning. To evaluate its performance, we develop a new benchmark that simulates realistic medical conversations using patient agents driven by profiles extracted from real-world consultation cases. We compare APP against SOTA one-shot and multi-turn LLM baselines. The results show that APP improves diagnostic accuracy, reduces uncertainty, and enhances user experience. By integrating medical expertise with transparent, human-like interaction, APP bridges the gap between AI-driven medical assistance and real-world clinical practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16835v2">Evaluating Speech-to-Text x LLM x Text-to-Speech Combinations for AI Interview Systems</a></div>
    <div class="paper-meta">
      📅 2025-08-21
    </div>
    <details class="paper-abstract">
      Voice-based conversational AI systems increasingly rely on cascaded architectures that combine speech-to-text (STT), large language models (LLMs), and text-to-speech (TTS) components. We present a large-scale empirical comparison of STT x LLM x TTS stacks using data sampled from over 300,000 AI-conducted job interviews. We used an LLM-as-a-Judge automated evaluation framework to assess conversational quality, technical accuracy, and skill assessment capabilities. Our analysis of five production configurations reveals that a stack combining Google's STT, GPT-4.1, and Cartesia's TTS outperforms alternatives in both objective quality metrics and user satisfaction scores. Surprisingly, we find that objective quality metrics correlate weakly with user satisfaction scores, suggesting that user experience in voice-based AI systems depends on factors beyond technical performance. Our findings provide practical guidance for selecting components in multimodal conversations and contribute a validated evaluation methodology for human-AI interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10868v4">NitiBench: A Comprehensive Study of LLM Framework Capabilities for Thai Legal Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-08-21
    </div>
    <details class="paper-abstract">
      The application of large language models (LLMs) in the legal domain holds significant potential for information retrieval and question answering, yet Thai legal QA systems face challenges due to a lack of standardized evaluation benchmarks and the complexity of Thai legal structures. This paper introduces NitiBench, a benchmark comprising two datasets: the NitiBench-CCL, covering general Thai financial law, and the NitiBench-Tax, which includes real-world tax law cases requiring advanced legal reasoning. We evaluate retrieval-augmented generation (RAG) and long-context LLM-based approaches to address three key research questions: the impact of domain-specific components like section-based chunking and cross-referencing, the comparative performance of different retrievers and LLMs, and the viability of long-context LLMs as an alternative to RAG. Our results show that section-based chunking significantly improves retrieval and end-to-end performance, current retrievers struggle with complex queries, and long-context LLMs still underperform RAG-based systems in Thai legal QA. To support fair evaluation, we propose tailored multi-label retrieval metrics and the use of an LLM-as-judge for coverage and contradiction detection method. These findings highlight the limitations of current Thai legal NLP solutions and provide a foundation for future research in the field. We also open-sourced our codes and dataset to available publicly.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.02725v2">Leveraging LLM Tutoring Systems for Non-Native English Speakers in Introductory CS Courses</a></div>
    <div class="paper-meta">
      📅 2025-08-21
      | 💬 11 pages, 5 tables, 4 figures, 2025 ASEE Annual Conference & Exposition
    </div>
    <details class="paper-abstract">
      Computer science has historically presented barriers for non-native English speaking (NNES) students, often due to language and terminology challenges. With the rise of large language models (LLMs), there is potential to leverage this technology to support NNES students more effectively. Recent implementations of LLMs as tutors in classrooms have shown promising results. In this study, we deployed an LLM tutor in an accelerated introductory computing course to evaluate its effectiveness specifically for NNES students. Key insights for LLM tutor use are as follows: NNES students signed up for the LLM tutor at a similar rate to native English speakers (NES); NNES students used the system at a lower rate than NES students -- to a small effect; NNES students asked significantly more questions in languages other than English compared to NES students, with many of the questions being multilingual by incorporating English programming keywords. Results for views of the LLM tutor are as follows: both NNES and NES students appreciated the LLM tutor for its accessibility, conversational style, and the guardrails put in place to guide users to answers rather than directly providing solutions; NNES students highlighted its approachability as they did not need to communicate in perfect English; NNES students rated help-seeking preferences of online resources higher than NES students; Many NNES students were unfamiliar with computing terminology in their native languages. These results suggest that LLM tutors can be a valuable resource for NNES students in computing, providing tailored support that enhances their learning experience and overcomes language barriers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16107v2">Do LLMs write like humans? Variation in grammatical and rhetorical styles</a></div>
    <div class="paper-meta">
      📅 2025-08-21
      | 💬 7 pages, 4 figures, 1 table
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are capable of writing grammatical text that follows instructions, answers questions, and solves problems. As they have advanced, it has become difficult to distinguish their output from human-written text. While past research has found some differences in surface features such as word choice and punctuation, and developed classifiers to detect LLM output, none has studied the rhetorical styles of LLMs. Using several variants of Llama 3 and GPT-4o, we construct two parallel corpora of human- and LLM-written texts from common prompts. Using Douglas Biber's set of lexical, grammatical, and rhetorical features, we identify systematic differences between LLMs and humans and between different LLMs. These differences persist when moving from smaller models to larger ones, and are larger for instruction-tuned models than base models. This observation of differences demonstrates that despite their advanced abilities, LLMs struggle to match human stylistic variation. Attention to more advanced linguistic features can hence detect patterns in their behavior not previously recognized.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15926v1">Noise, Adaptation, and Strategy: Assessing LLM Fidelity in Decision-Making</a></div>
    <div class="paper-meta">
      📅 2025-08-21
      | 💬 Accepted to EMNLP 2025 (Main Conference)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used in social science simulations. While their performance on reasoning and optimization tasks has been extensively evaluated, less attention has been paid to their ability to simulate human decision-making's variability and adaptability. We propose a process-oriented evaluation framework with progressive interventions (Intrinsicality, Instruction, and Imitation) to examine how LLM agents adapt under different levels of external guidance and human-derived noise. We validate the framework on two classic economics tasks, irrationality in the second-price auction and decision bias in the newsvendor problem, showing behavioral gaps between LLMs and humans. We find that LLMs, by default, converge on stable and conservative strategies that diverge from observed human behaviors. Risk-framed instructions impact LLM behavior predictably but do not replicate human-like diversity. Incorporating human data through in-context learning narrows the gap but fails to reach human subjects' strategic variability. These results highlight a persistent alignment gap in behavioral fidelity and suggest that future LLM evaluations should consider more process-level realism. We present a process-oriented approach for assessing LLMs in dynamic decision-making tasks, offering guidance for their application in synthetic data for social science research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05965v3">Validating LLM-as-a-Judge Systems under Rating Indeterminacy</a></div>
    <div class="paper-meta">
      📅 2025-08-21
    </div>
    <details class="paper-abstract">
      The LLM-as-a-judge paradigm, in which a judge LLM system replaces human raters in rating the outputs of other generative AI (GenAI) systems, plays a critical role in scaling and standardizing GenAI evaluations. To validate such judge systems, evaluators assess human--judge agreement by first collecting multiple human ratings for each item in a validation corpus, then aggregating the ratings into a single, per-item gold label rating. For many items, however, rating criteria may admit multiple valid interpretations, so a human or LLM rater may deem multiple ratings "reasonable" or "correct". We call this condition rating indeterminacy. Problematically, many rating tasks that contain rating indeterminacy rely on forced-choice elicitation, whereby raters are instructed to select only one rating for each item. In this paper, we introduce a framework for validating LLM-as-a-judge systems under rating indeterminacy. We draw theoretical connections between different measures of judge system performance under different human--judge agreement metrics, and different rating elicitation and aggregation schemes. We demonstrate that differences in how humans and LLMs resolve rating indeterminacy while responding to forced-choice rating instructions heavily bias LLM-as-a-judge validation. Through extensive experiments involving 11 real-world rating tasks and 8 commercial LLMs, we show that standard validation approaches that rely upon forced-choice ratings select judge systems that are highly suboptimal, performing as much as 30% worse than judge systems selected by our approach that uses multi-label "response set" ratings to account for rating indeterminacy. We conclude with concrete recommendations for more principled approaches to LLM-as-a-judge validation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16684v1">Democratizing AI Development: Local LLM Deployment for India's Developer Ecosystem in the Era of Tokenized APIs</a></div>
    <div class="paper-meta">
      📅 2025-08-21
      | 💬 for survey results, check https://docs.google.com/spreadsheets/d/1t0eV9oURaiu2HfARWo6sriBO0eC8bHUyZNN7CgK2NAk/edit?usp=sharing
    </div>
    <details class="paper-abstract">
      India's developer community faces significant barriers to sustained experimentation and learning with commercial Large Language Model (LLM) APIs, primarily due to economic and infrastructural constraints. This study empirically evaluates local LLM deployment using Ollama as an alternative to commercial cloud-based services for developer-focused applications. Through a mixed-methods analysis involving 180 Indian developers, students, and AI enthusiasts, we find that local deployment enables substantially greater hands-on development and experimentation, while reducing costs by 33% compared to commercial solutions. Developers using local LLMs completed over twice as many experimental iterations and reported deeper understanding of advanced AI architectures. Our results highlight local deployment as a critical enabler for inclusive and accessible AI development, demonstrating how technological accessibility can enhance learning outcomes and innovation capacity in resource-constrained environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15877v1">Annif at the GermEval-2025 LLMs4Subjects Task: Traditional XMTC Augmented by Efficient LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-21
      | 💬 5 pages, 4 figures, accepted at KONVENS 2025. arXiv admin note: substantial text overlap with arXiv:2504.19675
    </div>
    <details class="paper-abstract">
      This paper presents the Annif system in the LLMs4Subjects shared task (Subtask 2) at GermEval-2025. The task required creating subject predictions for bibliographic records using large language models, with a special focus on computational efficiency. Our system, based on the Annif automated subject indexing toolkit, refines our previous system from the first LLMs4Subjects shared task, which produced excellent results. We further improved the system by using many small and efficient language models for translation and synthetic data generation and by using LLMs for ranking candidate subjects. Our system ranked 1st in the overall quantitative evaluation of and 1st in the qualitative evaluation of Subtask 2.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15875v1">NEAT: Concept driven Neuron Attribution in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-21
    </div>
    <details class="paper-abstract">
      Locating neurons that are responsible for final predictions is important for opening the black-box large language models and understanding the inside mechanisms. Previous studies have tried to find mechanisms that operate at the neuron level but these methods fail to represent a concept and there is also scope for further optimization of compute required. In this paper, with the help of concept vectors, we propose a method for locating significant neurons that are responsible for representing certain concepts and term those neurons as concept neurons. If the number of neurons is n and the number of examples is m, we reduce the number of forward passes required from O(n*m) to just O(n) compared to the previous works and hence optimizing the time and computation required over previous works. We also compare our method with several baselines and previous methods and our results demonstrate better performance than most of the methods and are more optimal when compared to the state-of-the-art method. We, as part of our ablation studies, also try to optimize the search for the concept neurons by involving clustering methods. Finally, we apply our methods to find, turn off the neurons that we find, and analyze its implications in parts of hate speech and bias in LLMs, and we also evaluate our bias part in terms of Indian context. Our methodology, analysis and explanations facilitate understating of neuron-level responsibility for more broader and human-like concepts and also lay a path for future research in this direction of finding concept neurons and intervening them.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16676v1">WISCA: A Lightweight Model Transition Method to Improve LLM Training via Weight Scaling</a></div>
    <div class="paper-meta">
      📅 2025-08-21
    </div>
    <details class="paper-abstract">
      Transformer architecture gradually dominates the LLM field. Recent advances in training optimization for Transformer-based large language models (LLMs) primarily focus on architectural modifications or optimizer adjustments. However, these approaches lack systematic optimization of weight patterns during training. Weight pattern refers to the distribution and relative magnitudes of weight parameters in a neural network. To address this issue, we propose a Weight Scaling method called WISCA to enhance training efficiency and model quality by strategically improving neural network weight patterns without changing network structures. By rescaling weights while preserving model outputs, WISCA indirectly optimizes the model's training trajectory. Experiments demonstrate that WISCA significantly improves convergence quality (measured by generalization capability and loss reduction), particularly in LLMs with Grouped Query Attention (GQA) architectures and LoRA fine-tuning tasks. Empirical results show 5.6% average improvement on zero-shot validation tasks and 2.12% average reduction in training perplexity across multiple architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15868v1">CARFT: Boosting LLM Reasoning via Contrastive Learning with Annotated Chain-of-Thought-based Reinforced Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-08-21
      | 💬 14 pages, to appear in EMNLP25
    </div>
    <details class="paper-abstract">
      Reasoning capability plays a significantly critical role in the the broad applications of Large Language Models (LLMs). To enhance the reasoning performance of LLMs, diverse Reinforcement Learning (RL)-based fine-tuning approaches have been proposed to address the limited generalization capability of LLMs trained solely via Supervised Fine-Tuning (SFT). Despite their effectiveness, two major limitations hinder the advancement of LLMs. First, vanilla RL-based approaches ignore annotated Chain-of-Thought (CoT) and incorporate unstable reasoning path sampling, which typically results in model collapse, unstable training process, and suboptimal performance. Second, existing SFT approaches generally overemphasize the annotated CoT, potentially leading to performance degradation due to insufficient exploitation of potential CoT. In this paper, we propose a Contrastive learning with annotated CoT-based Reinforced Fine-Tuning approach, i.e., \TheName{}, to enhance the reasoning performance of LLMs while addressing the aforementioned limitations. Specifically, we propose learning a representation for each CoT. Based on this representation, we design novel contrastive signals to guide the fine-tuning process. Our approach not only fully exploits the available annotated CoT but also stabilizes the fine-tuning procedure by incorporating an additional unsupervised learning signal. We conduct comprehensive experiments and in-depth analysis with three baseline approaches, two foundation models, and two datasets to demonstrate significant advantages of \TheName{} in terms of robustness, performance (up to 10.15\%), and efficiency (up to 30.62\%). Code is available at https://github.com/WNQzhu/CARFT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14377v1">ZPD-SCA: Unveiling the Blind Spots of LLMs in Assessing Students' Cognitive Abilities</a></div>
    <div class="paper-meta">
      📅 2025-08-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated potential in educational applications, yet their capacity to accurately assess the cognitive alignment of reading materials with students' developmental stages remains insufficiently explored. This gap is particularly critical given the foundational educational principle of the Zone of Proximal Development (ZPD), which emphasizes the need to match learning resources with Students' Cognitive Abilities (SCA). Despite the importance of this alignment, there is a notable absence of comprehensive studies investigating LLMs' ability to evaluate reading comprehension difficulty across different student age groups, especially in the context of Chinese language education. To fill this gap, we introduce ZPD-SCA, a novel benchmark specifically designed to assess stage-level Chinese reading comprehension difficulty. The benchmark is annotated by 60 Special Grade teachers, a group that represents the top 0.15% of all in-service teachers nationwide. Experimental results reveal that LLMs perform poorly in zero-shot learning scenarios, with Qwen-max and GLM even falling below the probability of random guessing. When provided with in-context examples, LLMs performance improves substantially, with some models achieving nearly double the accuracy of their zero-shot baselines. These results reveal that LLMs possess emerging abilities to assess reading difficulty, while also exposing limitations in their current training for educationally aligned judgment. Notably, even the best-performing models display systematic directional biases, suggesting difficulties in accurately aligning material difficulty with SCA. Furthermore, significant variations in model performance across different genres underscore the complexity of task. We envision that ZPD-SCA can provide a foundation for evaluating and improving LLMs in cognitively aligned educational applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19959v3">From Concept to Practice: an Automated LLM-aided UVM Machine for RTL Verification</a></div>
    <div class="paper-meta">
      📅 2025-08-20
    </div>
    <details class="paper-abstract">
      Verification presents a major bottleneck in Integrated Circuit (IC) development, consuming nearly 70% of the total development effort. While the Universal Verification Methodology (UVM) is widely used in industry to improve verification efficiency through structured and reusable testbenches, constructing these testbenches and generating sufficient stimuli remain challenging. These challenges arise from the considerable manual coding effort required, repetitive manual execution of multiple EDA tools, and the need for in-depth domain expertise to navigate complex designs.Here, we present UVM^2, an automated verification framework that leverages Large Language Models (LLMs) to generate UVM testbenches and iteratively refine them using coverage feedback, significantly reducing manual effort while maintaining rigorous verification standards.To evaluate UVM^2, we introduce a benchmark suite comprising Register Transfer Level (RTL) designs of up to 1.6K lines of code.The results show that UVM^2 reduces testbench setup time by up to UVM^2 compared to experienced engineers, and achieve average code and function coverage of 87.44% and 89.58%, outperforming state-of-the-art solutions by 20.96% and 23.51%, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14357v1">Organ-Agents: Virtual Human Physiology Simulator via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-20
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have enabled new possibilities in simulating complex physiological systems. We introduce Organ-Agents, a multi-agent framework that simulates human physiology via LLM-driven agents. Each Simulator models a specific system (e.g., cardiovascular, renal, immune). Training consists of supervised fine-tuning on system-specific time-series data, followed by reinforcement-guided coordination using dynamic reference selection and error correction. We curated data from 7,134 sepsis patients and 7,895 controls, generating high-resolution trajectories across 9 systems and 125 variables. Organ-Agents achieved high simulation accuracy on 4,509 held-out patients, with per-system MSEs <0.16 and robustness across SOFA-based severity strata. External validation on 22,689 ICU patients from two hospitals showed moderate degradation under distribution shifts with stable simulation. Organ-Agents faithfully reproduces critical multi-system events (e.g., hypotension, hyperlactatemia, hypoxemia) with coherent timing and phase progression. Evaluation by 15 critical care physicians confirmed realism and physiological plausibility (mean Likert ratings 3.9 and 3.7). Organ-Agents also enables counterfactual simulations under alternative sepsis treatment strategies, generating trajectories and APACHE II scores aligned with matched real-world patients. In downstream early warning tasks, classifiers trained on synthetic data showed minimal AUROC drops (<0.04), indicating preserved decision-relevant patterns. These results position Organ-Agents as a credible, interpretable, and generalizable digital twin for precision diagnosis, treatment simulation, and hypothesis testing in critical care.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15861v1">XFinBench: Benchmarking LLMs in Complex Financial Problem Solving and Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-08-20
    </div>
    <details class="paper-abstract">
      Solving financial problems demands complex reasoning, multimodal data processing, and a broad technical understanding, presenting unique challenges for current large language models (LLMs). We introduce XFinBench, a novel benchmark with 4,235 examples designed to evaluate LLM's ability in solving complex, knowledge-intensive financial problems across diverse graduate-level finance topics with multi-modal context. We identify five core capabilities of LLMs using XFinBench, i.e, terminology understanding, temporal reasoning, future forecasting, scenario planning, and numerical modelling. Upon XFinBench, we conduct extensive experiments on 18 leading models. The result shows that o1 is the best-performing text-only model with an overall accuracy of 67.3%, but still lags significantly behind human experts with 12.5%, especially in temporal reasoning and scenario planning capabilities. We further construct a knowledge bank with 3,032 finance terms for knowledge augmentation analysis, and find that relevant knowledge to the question only brings consistent accuracy improvements to small open-source model. Additionally, our error analysis reveals that rounding errors during calculation and blindness to position and intersection of curves in the image are two primary issues leading to model's poor performance in calculating and visual-context questions, respectively. Code and dataset are accessible via GitHub: https://github.com/Zhihan72/XFinBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15854v1">QU-NLP at QIAS 2025 Shared Task: A Two-Phase LLM Fine-Tuning and Retrieval-Augmented Generation Approach for Islamic Inheritance Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-08-20
    </div>
    <details class="paper-abstract">
      This paper presents our approach and results for SubTask 1: Islamic Inheritance Reasoning at QIAS 2025, a shared task focused on evaluating Large Language Models (LLMs) in understanding and reasoning within Islamic inheritance knowledge. We fine-tuned the Fanar-1-9B causal language model using Low-Rank Adaptation (LoRA) and integrated it into a Retrieval-Augmented Generation (RAG) pipeline. Our system addresses the complexities of Islamic inheritance law, including comprehending inheritance scenarios, identifying eligible heirs, applying fixed-share rules, and performing precise calculations. Our system achieved an accuracy of 0.858 in the final test, outperforming other competitive models such as, GPT 4.5, LLaMA, Fanar, Mistral and ALLaM evaluated with zero-shot prompting. Our results demonstrate that QU-NLP achieves near state-of-the-art accuracy (85.8%), excelling especially on advanced reasoning (97.6%) where it outperforms Gemini 2.5 and OpenAI's o3. This highlights that domain-specific fine-tuning combined with retrieval grounding enables mid-scale Arabic LLMs to surpass frontier models in Islamic inheritance reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16654v1">MSNav: Zero-Shot Vision-and-Language Navigation with Dynamic Memory and LLM Spatial Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-08-20
      | 💬 9 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Vision-and-Language Navigation (VLN) requires an agent to interpret natural language instructions and navigate complex environments. Current approaches often adopt a "black-box" paradigm, where a single Large Language Model (LLM) makes end-to-end decisions. However, it is plagued by critical vulnerabilities, including poor spatial reasoning, weak cross-modal grounding, and memory overload in long-horizon tasks. To systematically address these issues, we propose Memory Spatial Navigation(MSNav), a framework that fuses three modules into a synergistic architecture, which transforms fragile inference into a robust, integrated intelligence. MSNav integrates three modules: Memory Module, a dynamic map memory module that tackles memory overload through selective node pruning, enhancing long-range exploration; Spatial Module, a module for spatial reasoning and object relationship inference that improves endpoint recognition; and Decision Module, a module using LLM-based path planning to execute robust actions. Powering Spatial Module, we also introduce an Instruction-Object-Space (I-O-S) dataset and fine-tune the Qwen3-4B model into Qwen-Spatial (Qwen-Sp), which outperforms leading commercial LLMs in object list extraction, achieving higher F1 and NDCG scores on the I-O-S test set. Extensive experiments on the Room-to-Room (R2R) and REVERIE datasets demonstrate MSNav's state-of-the-art performance with significant improvements in Success Rate (SR) and Success weighted by Path Length (SPL).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15848v1">Self-Disguise Attack: Induce the LLM to disguise itself for AIGT detection evasion</a></div>
    <div class="paper-meta">
      📅 2025-08-20
    </div>
    <details class="paper-abstract">
      AI-generated text (AIGT) detection evasion aims to reduce the detection probability of AIGT, helping to identify weaknesses in detectors and enhance their effectiveness and reliability in practical applications. Although existing evasion methods perform well, they suffer from high computational costs and text quality degradation. To address these challenges, we propose Self-Disguise Attack (SDA), a novel approach that enables Large Language Models (LLM) to actively disguise its output, reducing the likelihood of detection by classifiers. The SDA comprises two main components: the adversarial feature extractor and the retrieval-based context examples optimizer. The former generates disguise features that enable LLMs to understand how to produce more human-like text. The latter retrieves the most relevant examples from an external knowledge base as in-context examples, further enhancing the self-disguise ability of LLMs and mitigating the impact of the disguise process on the diversity of the generated text. The SDA directly employs prompts containing disguise features and optimized context examples to guide the LLM in generating detection-resistant text, thereby reducing resource consumption. Experimental results demonstrate that the SDA effectively reduces the average detection accuracy of various AIGT detectors across texts generated by three different LLMs, while maintaining the quality of AIGT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19073v3">MFTCXplain: A Multilingual Benchmark Dataset for Evaluating the Moral Reasoning of LLMs through Hate Speech Multi-hop Explanations</a></div>
    <div class="paper-meta">
      📅 2025-08-20
      | 💬 Findings of the Association for Computational Linguistics: EMNLP 2025; *These authors contributed equally
    </div>
    <details class="paper-abstract">
      Ensuring the moral reasoning capabilities of Large Language Models (LLMs) is a growing concern as these systems are used in socially sensitive tasks. Nevertheless, current evaluation benchmarks present two major shortcomings: a lack of annotations that justify moral classifications, which limits transparency and interpretability; and a predominant focus on English, which constrains the assessment of moral reasoning across diverse cultural settings. In this paper, we introduce MFTCXplain, a multilingual benchmark dataset for evaluating the moral reasoning of LLMs via hate speech multi-hop explanation using Moral Foundation Theory (MFT). The dataset comprises 3,000 tweets across Portuguese, Italian, Persian, and English, annotated with binary hate speech labels, moral categories, and text span-level rationales. Empirical results highlight a misalignment between LLM outputs and human annotations in moral reasoning tasks. While LLMs perform well in hate speech detection (F1 up to 0.836), their ability to predict moral sentiments is notably weak (F1 < 0.35). Furthermore, rationale alignment remains limited mainly in underrepresented languages. These findings show the limited capacity of current LLMs to internalize and reflect human moral reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15110v1">LLMs and Agentic AI in Insurance Decision-Making: Opportunities and Challenges For Africa</a></div>
    <div class="paper-meta">
      📅 2025-08-20
    </div>
    <details class="paper-abstract">
      In this work, we highlight the transformative potential of Artificial Intelligence (AI), particularly Large Language Models (LLMs) and agentic AI, in the insurance sector. We consider and emphasize the unique opportunities, challenges, and potential pathways in insurance amid rapid performance improvements, increased open-source access, decreasing deployment costs, and the complexity of LLM or agentic AI frameworks. To bring it closer to home, we identify critical gaps in the African insurance market and highlight key local efforts, players, and partnership opportunities. Finally, we call upon actuaries, insurers, regulators, and tech leaders to a collaborative effort aimed at creating inclusive, sustainable, and equitable AI strategies and solutions: by and for Africans.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02873v3">It's the Thought that Counts: Evaluating the Attempts of Frontier LLMs to Persuade on Harmful Topics</a></div>
    <div class="paper-meta">
      📅 2025-08-20
    </div>
    <details class="paper-abstract">
      Persuasion is a powerful capability of large language models (LLMs) that both enables beneficial applications (e.g. helping people quit smoking) and raises significant risks (e.g. large-scale, targeted political manipulation). Prior work has found models possess a significant and growing persuasive capability, measured by belief changes in simulated or real users. However, these benchmarks overlook a crucial risk factor: the propensity of a model to attempt to persuade in harmful contexts. Understanding whether a model will blindly ``follow orders'' to persuade on harmful topics (e.g. glorifying joining a terrorist group) is key to understanding the efficacy of safety guardrails. Moreover, understanding if and when a model will engage in persuasive behavior in pursuit of some goal is essential to understanding the risks from agentic AI systems. We propose the Attempt to Persuade Eval (APE) benchmark, that shifts the focus from persuasion success to persuasion attempts, operationalized as a model's willingness to generate content aimed at shaping beliefs or behavior. Our evaluation framework probes frontier LLMs using a multi-turn conversational setup between simulated persuader and persuadee agents. APE explores a diverse spectrum of topics including conspiracies, controversial issues, and non-controversially harmful content. We introduce an automated evaluator model to identify willingness to persuade and measure the frequency and context of persuasive attempts. We find that many open and closed-weight models are frequently willing to attempt persuasion on harmful topics and that jailbreaking can increase willingness to engage in such behavior. Our results highlight gaps in current safety guardrails and underscore the importance of evaluating willingness to persuade as a key dimension of LLM risk. APE is available at github.com/AlignmentResearch/AttemptPersuadeEval
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07441v2">SAND: Boosting LLM Agents with Self-Taught Action Deliberation</a></div>
    <div class="paper-meta">
      📅 2025-08-20
      | 💬 EMNLP 2025
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents are commonly tuned with supervised finetuning on ReAct-style expert trajectories or preference optimization over pairwise rollouts. Most of these methods focus on imitating specific expert behaviors or promoting chosen reasoning thoughts and actions over rejected ones. However, without reasoning and comparing over alternatives actions, LLM agents finetuned with these methods may over-commit towards seemingly plausible but suboptimal actions due to limited action space exploration. To address this, in this paper we propose Self-taught ActioN Deliberation (SAND) framework, enabling LLM agents to explicitly deliberate over candidate actions before committing to one. To tackle the challenges of when and what to deliberate given large action space and step-level action evaluation, we incorporate self-consistency action sampling and execution-guided action critique to help synthesize step-wise action deliberation thoughts using the base model of the LLM agent. In an iterative manner, the deliberation trajectories are then used to finetune the LLM agent itself. Evaluating on two representative interactive agent tasks, SAND achieves an average 20% improvement over initial supervised finetuning and also outperforms state-of-the-art agent tuning approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15860v3">Synthetic vs. Gold: The Role of LLM Generated Labels and Data in Cyberbullying Detection</a></div>
    <div class="paper-meta">
      📅 2025-08-20
    </div>
    <details class="paper-abstract">
      Cyberbullying (CB) presents a pressing threat, especially to children, underscoring the urgent need for robust detection systems to ensure online safety. While large-scale datasets on online abuse exist, there remains a significant gap in labeled data that specifically reflects the language and communication styles used by children. The acquisition of such data from vulnerable populations, such as children, is challenging due to ethical, legal and technical barriers. Moreover, the creation of these datasets relies heavily on human annotation, which not only strains resources but also raises significant concerns due to annotators exposure to harmful content. In this paper, we address these challenges by leveraging Large Language Models (LLMs) to generate synthetic data and labels. Our experiments demonstrate that synthetic data enables BERT-based CB classifiers to achieve performance close to that of those trained on fully authentic datasets (75.8% vs. 81.5% accuracy). Additionally, LLMs can effectively label authentic yet unlabeled data, allowing BERT classifiers to attain a comparable performance level (79.1% vs. 81.5% accuracy). These results highlight the potential of LLMs as a scalable, ethical, and cost-effective solution for generating data for CB detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05182v2">On the Comprehensibility of Multi-structured Financial Documents using LLMs and Pre-processing Tools</a></div>
    <div class="paper-meta">
      📅 2025-08-20
      | 💬 15 pages, 5 figures, 9 tables
    </div>
    <details class="paper-abstract">
      The proliferation of complex structured data in hybrid sources, such as PDF documents and web pages, presents unique challenges for current Large Language Models (LLMs) and Multi-modal Large Language Models (MLLMs) in providing accurate answers. Despite the recent advancements of MLLMs, they still often falter when interpreting intricately structured information, such as nested tables and multi-dimensional plots, leading to hallucinations and erroneous outputs. This paper explores the capabilities of LLMs and MLLMs in understanding and answering questions from complex data structures found in PDF documents by leveraging industrial and open-source tools as part of a pre-processing pipeline. Our findings indicate that GPT-4o, a popular MLLM, achieves an accuracy of 56% on multi-structured documents when fed documents directly, and that integrating pre-processing tools raises the accuracy of LLMs to 61.3% for GPT-4o and 76% for GPT-4, and with lower overall cost. The code is publicly available at https://github.com/OGCDS/FinancialQA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15036v1">MoEcho: Exploiting Side-Channel Attacks to Compromise User Privacy in Mixture-of-Experts LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-20
      | 💬 This paper will appear in CCS 2025
    </div>
    <details class="paper-abstract">
      The transformer architecture has become a cornerstone of modern AI, fueling remarkable progress across applications in natural language processing, computer vision, and multimodal learning. As these models continue to scale explosively for performance, implementation efficiency remains a critical challenge. Mixture of Experts (MoE) architectures, selectively activating specialized subnetworks (experts), offer a unique balance between model accuracy and computational cost. However, the adaptive routing in MoE architectures, where input tokens are dynamically directed to specialized experts based on their semantic meaning inadvertently opens up a new attack surface for privacy breaches. These input-dependent activation patterns leave distinctive temporal and spatial traces in hardware execution, which adversaries could exploit to deduce sensitive user data. In this work, we propose MoEcho, discovering a side channel analysis based attack surface that compromises user privacy on MoE based systems. Specifically, in MoEcho, we introduce four novel architectural side channels on different computing platforms, including Cache Occupancy Channels and Pageout+Reload on CPUs, and Performance Counter and TLB Evict+Reload on GPUs, respectively. Exploiting these vulnerabilities, we propose four attacks that effectively breach user privacy in large language models (LLMs) and vision language models (VLMs) based on MoE architectures: Prompt Inference Attack, Response Reconstruction Attack, Visual Inference Attack, and Visual Reconstruction Attack. MoEcho is the first runtime architecture level security analysis of the popular MoE structure common in modern transformers, highlighting a serious security and privacy threat and calling for effective and timely safeguards when harnessing MoE based models for developing efficient large scale AI services.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15030v1">Collab-REC: An LLM-based Agentic Framework for Balancing Recommendations in Tourism</a></div>
    <div class="paper-meta">
      📅 2025-08-20
    </div>
    <details class="paper-abstract">
      We propose Collab-REC, a multi-agent framework designed to counteract popularity bias and enhance diversity in tourism recommendations. In our setting, three LLM-based agents -- Personalization, Popularity, and Sustainability generate city suggestions from complementary perspectives. A non-LLM moderator then merges and refines these proposals via multi-round negotiation, ensuring each agent's viewpoint is incorporated while penalizing spurious or repeated responses. Experiments on European city queries show that Collab-REC improves diversity and overall relevance compared to a single-agent baseline, surfacing lesser-visited locales that often remain overlooked. This balanced, context-aware approach addresses over-tourism and better aligns with constraints provided by the user, highlighting the promise of multi-stakeholder collaboration in LLM-driven recommender systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15928v3">Exploring Big Five Personality and AI Capability Effects in LLM-Simulated Negotiation Dialogues</a></div>
    <div class="paper-meta">
      📅 2025-08-20
      | 💬 Presented at the KDD 2025 Workshop on Evaluation and Trustworthiness of Agentic and Generative AI Models under the title "Evaluating the LLM-simulated Impacts of Big Five Personality Traits and AI Capabilities on Social Negotiations" (https://kdd-eval-workshop.github.io/genai-evaluation-kdd2025/assets/papers/Submission%2036.pdf)
    </div>
    <details class="paper-abstract">
      This paper presents an evaluation framework for agentic AI systems in mission-critical negotiation contexts, addressing the need for AI agents that can adapt to diverse human operators and stakeholders. Using Sotopia as a simulation testbed, we present two experiments that systematically evaluated how personality traits and AI agent characteristics influence LLM-simulated social negotiation outcomes--a capability essential for a variety of applications involving cross-team coordination and civil-military interactions. Experiment 1 employs causal discovery methods to measure how personality traits impact price bargaining negotiations, through which we found that Agreeableness and Extraversion significantly affect believability, goal achievement, and knowledge acquisition outcomes. Sociocognitive lexical measures extracted from team communications detected fine-grained differences in agents' empathic communication, moral foundations, and opinion patterns, providing actionable insights for agentic AI systems that must operate reliably in high-stakes operational scenarios. Experiment 2 evaluates human-AI job negotiations by manipulating both simulated human personality and AI system characteristics, specifically transparency, competence, adaptability, demonstrating how AI agent trustworthiness impact mission effectiveness. These findings establish a repeatable evaluation methodology for experimenting with AI agent reliability across diverse operator personalities and human-agent team dynamics, directly supporting operational requirements for reliable AI systems. Our work advances the evaluation of agentic AI workflows by moving beyond standard performance metrics to incorporate social dynamics essential for mission success in complex operations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14896v1">Quantization Meets dLLMs: A Systematic Study of Post-training Quantization for Diffusion LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-20
      | 💬 Technical Report, Work in Progress
    </div>
    <details class="paper-abstract">
      Recent advances in diffusion large language models (dLLMs) have introduced a promising alternative to autoregressive (AR) LLMs for natural language generation tasks, leveraging full attention and denoising-based decoding strategies. However, the deployment of these models on edge devices remains challenging due to their massive parameter scale and high resource demands. While post-training quantization (PTQ) has emerged as a widely adopted technique for compressing AR LLMs, its applicability to dLLMs remains largely unexplored. In this work, we present the first systematic study on quantizing diffusion-based language models. We begin by identifying the presence of activation outliers, characterized by abnormally large activation values that dominate the dynamic range. These outliers pose a key challenge to low-bit quantization, as they make it difficult to preserve precision for the majority of values. More importantly, we implement state-of-the-art PTQ methods and conduct a comprehensive evaluation across multiple task types and model variants. Our analysis is structured along four key dimensions: bit-width, quantization method, task category, and model type. Through this multi-perspective evaluation, we offer practical insights into the quantization behavior of dLLMs under different configurations. We hope our findings provide a foundation for future research in efficient dLLM deployment. All codes and experimental setups will be released to support the community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14879v1">MeshCoder: LLM-Powered Structured Mesh Code Generation from Point Clouds</a></div>
    <div class="paper-meta">
      📅 2025-08-20
    </div>
    <details class="paper-abstract">
      Reconstructing 3D objects into editable programs is pivotal for applications like reverse engineering and shape editing. However, existing methods often rely on limited domain-specific languages (DSLs) and small-scale datasets, restricting their ability to model complex geometries and structures. To address these challenges, we introduce MeshCoder, a novel framework that reconstructs complex 3D objects from point clouds into editable Blender Python scripts. We develop a comprehensive set of expressive Blender Python APIs capable of synthesizing intricate geometries. Leveraging these APIs, we construct a large-scale paired object-code dataset, where the code for each object is decomposed into distinct semantic parts. Subsequently, we train a multimodal large language model (LLM) that translates 3D point cloud into executable Blender Python scripts. Our approach not only achieves superior performance in shape-to-code reconstruction tasks but also facilitates intuitive geometric and topological editing through convenient code modifications. Furthermore, our code-based representation enhances the reasoning capabilities of LLMs in 3D shape understanding tasks. Together, these contributions establish MeshCoder as a powerful and flexible solution for programmatic 3D shape reconstruction and understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02085v4">SE-Agent: Self-Evolution Trajectory Optimization in Multi-Step Reasoning with LLM-Based Agents</a></div>
    <div class="paper-meta">
      📅 2025-08-20
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based agents have recently shown impressive capabilities in complex reasoning and tool use via multi-step interactions with their environments. While these agents have the potential to tackle complicated tasks, their problem-solving process, i.e., agents' interaction trajectory leading to task completion, remains underexploited. These trajectories contain rich feedback that can navigate agents toward the right directions for solving problems correctly. Although prevailing approaches, such as Monte Carlo Tree Search (MCTS), can effectively balance exploration and exploitation, they ignore the interdependence among various trajectories and lack the diversity of search spaces, which leads to redundant reasoning and suboptimal outcomes. To address these challenges, we propose SE-Agent, a Self-Evolution framework that enables Agents to optimize their reasoning processes iteratively. Our approach revisits and enhances former pilot trajectories through three key operations: revision, recombination, and refinement. This evolutionary mechanism enables two critical advantages: (1) it expands the search space beyond local optima by intelligently exploring diverse solution paths guided by previous trajectories, and (2) it leverages cross-trajectory inspiration to efficiently enhance performance while mitigating the impact of suboptimal reasoning paths. Through these mechanisms, SE-Agent achieves continuous self-evolution that incrementally improves reasoning quality. We evaluate SE-Agent on SWE-bench Verified to resolve real-world GitHub issues. Experimental results across five strong LLMs show that integrating SE-Agent delivers up to 55% relative improvement, achieving state-of-the-art performance among all open-source agents on SWE-bench Verified. Our code and demonstration materials are publicly available at https://github.com/JARVIS-Xs/SE-Agent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14765v1">PepThink-R1: LLM for Interpretable Cyclic Peptide Optimization with CoT SFT and Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-08-20
    </div>
    <details class="paper-abstract">
      Designing therapeutic peptides with tailored properties is hindered by the vastness of sequence space, limited experimental data, and poor interpretability of current generative models. To address these challenges, we introduce PepThink-R1, a generative framework that integrates large language models (LLMs) with chain-of-thought (CoT) supervised fine-tuning and reinforcement learning (RL). Unlike prior approaches, PepThink-R1 explicitly reasons about monomer-level modifications during sequence generation, enabling interpretable design choices while optimizing for multiple pharmacological properties. Guided by a tailored reward function balancing chemical validity and property improvements, the model autonomously explores diverse sequence variants. We demonstrate that PepThink-R1 generates cyclic peptides with significantly enhanced lipophilicity, stability, and exposure, outperforming existing general LLMs (e.g., GPT-5) and domain-specific baseline in both optimization success and interpretability. To our knowledge, this is the first LLM-based peptide design framework that combines explicit reasoning with RL-driven property control, marking a step toward reliable and transparent peptide optimization for therapeutic discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14751v1">HERAKLES: Hierarchical Skill Compilation for Open-ended LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-08-20
      | 💬 42 pages
    </div>
    <details class="paper-abstract">
      Open-ended AI agents need to be able to learn efficiently goals of increasing complexity, abstraction and heterogeneity over their lifetime. Beyond sampling efficiently their own goals, autotelic agents specifically need to be able to keep the growing complexity of goals under control, limiting the associated growth in sample and computational complexity. To adress this challenge, recent approaches have leveraged hierarchical reinforcement learning (HRL) and language, capitalizing on its compositional and combinatorial generalization capabilities to acquire temporally extended reusable behaviours. Existing approaches use expert defined spaces of subgoals over which they instantiate a hierarchy, and often assume pre-trained associated low-level policies. Such designs are inadequate in open-ended scenarios, where goal spaces naturally diversify across a broad spectrum of difficulties. We introduce HERAKLES, a framework that enables a two-level hierarchical autotelic agent to continuously compile mastered goals into the low-level policy, executed by a small, fast neural network, dynamically expanding the set of subgoals available to the high-level policy. We train a Large Language Model (LLM) to serve as the high-level controller, exploiting its strengths in goal decomposition and generalization to operate effectively over this evolving subgoal space. We evaluate HERAKLES in the open-ended Crafter environment and show that it scales effectively with goal complexity, improves sample efficiency through skill compilation, and enables the agent to adapt robustly to novel challenges over time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14735v1">Evaluating Multilingual and Code-Switched Alignment in LLMs via Synthetic Natural Language Inference</a></div>
    <div class="paper-meta">
      📅 2025-08-20
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly applied in multilingual contexts, yet their capacity for consistent, logically grounded alignment across languages remains underexplored. We present a controlled evaluation framework for multilingual natural language inference (NLI) that generates synthetic, logic-based premise-hypothesis pairs and translates them into a typologically diverse set of languages. This design enables precise control over semantic relations and allows testing in both monolingual and mixed-language (code-switched) conditions. Surprisingly, code-switching does not degrade, and can even improve, performance, suggesting that translation-induced lexical variation may serve as a regularization signal. We validate semantic preservation through embedding-based similarity analyses and cross-lingual alignment visualizations, confirming the fidelity of translated pairs. Our findings expose both the potential and the brittleness of current LLM cross-lingual reasoning, and identify code-switching as a promising lever for improving multilingual robustness. Code available at: https://github.com/KurbanIntelligenceLab/nli-stress-testing
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19254v3">Uncertainty Quantification for Language Models: A Suite of Black-Box, White-Box, LLM Judge, and Ensemble Scorers</a></div>
    <div class="paper-meta">
      📅 2025-08-20
      | 💬 UQLM repository: https://github.com/cvs-health/uqlm
    </div>
    <details class="paper-abstract">
      Hallucinations are a persistent problem with Large Language Models (LLMs). As these models become increasingly used in high-stakes domains, such as healthcare and finance, the need for effective hallucination detection is crucial. To this end, we outline a versatile framework for zero-resource hallucination detection that practitioners can apply to real-world use cases. To achieve this, we adapt a variety of existing uncertainty quantification (UQ) techniques, including black-box UQ, white-box UQ, and LLM-as-a-Judge, transforming them as necessary into standardized response-level confidence scores ranging from 0 to 1. To enhance flexibility, we propose a tunable ensemble approach that incorporates any combination of the individual confidence scores. This approach enables practitioners to optimize the ensemble for a specific use case for improved performance. To streamline implementation, the full suite of scorers is offered in this paper's companion Python toolkit, UQLM. To evaluate the performance of the various scorers, we conduct an extensive set of experiments using several LLM question-answering benchmarks. We find that our tunable ensemble typically surpasses its individual components and outperforms existing hallucination detection methods. Our results demonstrate the benefits of customized hallucination detection strategies for improving the accuracy and reliability of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14951v1">Improving LLMs for Machine Translation Using Synthetic Preference Data</a></div>
    <div class="paper-meta">
      📅 2025-08-20
      | 💬 Paper with individual presentation at LUHME workshop at ECAI 2025
    </div>
    <details class="paper-abstract">
      Large language models have emerged as effective machine translation systems. In this paper, we explore how a general instruction-tuned large language model can be improved for machine translation using relatively few easily produced data resources. Using Slovene as a use case, we improve the GaMS-9B-Instruct model using Direct Preference Optimization (DPO) training on a programmatically curated and enhanced subset of a public dataset. As DPO requires pairs of quality-ranked instances, we generated its training dataset by translating English Wikipedia articles using two LLMs, GaMS-9B-Instruct and EuroLLM-9B-Instruct. We ranked the resulting translations based on heuristics coupled with automatic evaluation metrics such as COMET. The evaluation shows that our fine-tuned model outperforms both models involved in the dataset generation. In comparison to the baseline models, the fine-tuned model achieved a COMET score gain of around 0.04 and 0.02, respectively, on translating Wikipedia articles. It also more consistently avoids language and formatting errors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14706v1">ShizhenGPT: Towards Multimodal LLMs for Traditional Chinese Medicine</a></div>
    <div class="paper-meta">
      📅 2025-08-20
    </div>
    <details class="paper-abstract">
      Despite the success of large language models (LLMs) in various domains, their potential in Traditional Chinese Medicine (TCM) remains largely underexplored due to two critical barriers: (1) the scarcity of high-quality TCM data and (2) the inherently multimodal nature of TCM diagnostics, which involve looking, listening, smelling, and pulse-taking. These sensory-rich modalities are beyond the scope of conventional LLMs. To address these challenges, we present ShizhenGPT, the first multimodal LLM tailored for TCM. To overcome data scarcity, we curate the largest TCM dataset to date, comprising 100GB+ of text and 200GB+ of multimodal data, including 1.2M images, 200 hours of audio, and physiological signals. ShizhenGPT is pretrained and instruction-tuned to achieve deep TCM knowledge and multimodal reasoning. For evaluation, we collect recent national TCM qualification exams and build a visual benchmark for Medicinal Recognition and Visual Diagnosis. Experiments demonstrate that ShizhenGPT outperforms comparable-scale LLMs and competes with larger proprietary models. Moreover, it leads in TCM visual understanding among existing multimodal LLMs and demonstrates unified perception across modalities like sound, pulse, smell, and vision, paving the way toward holistic multimodal perception and diagnosis in TCM. Datasets, models, and code are publicly available. We hope this work will inspire further exploration in this field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14654v1">Entropy-Constrained Strategy Optimization in Urban Floods: A Multi-Agent Framework with LLM and Knowledge Graph Integration</a></div>
    <div class="paper-meta">
      📅 2025-08-20
      | 💬 17 pages including appendix, 6 figures
    </div>
    <details class="paper-abstract">
      In recent years, the increasing frequency of extreme urban rainfall events has posed significant challenges to emergency scheduling systems. Urban flooding often leads to severe traffic congestion and service disruptions, threatening public safety and mobility. However, effective decision making remains hindered by three key challenges: (1) managing trade-offs among competing goals (e.g., traffic flow, task completion, and risk mitigation) requires dynamic, context-aware strategies; (2) rapidly evolving environmental conditions render static rules inadequate; and (3) LLM-generated strategies frequently suffer from semantic instability and execution inconsistency. Existing methods fail to align perception, global optimization, and multi-agent coordination within a unified framework. To tackle these challenges, we introduce H-J, a hierarchical multi-agent framework that integrates knowledge-guided prompting, entropy-constrained generation, and feedback-driven optimization. The framework establishes a closed-loop pipeline spanning from multi-source perception to strategic execution and continuous refinement. We evaluate H-J on real-world urban topology and rainfall data under three representative conditions: extreme rainfall, intermittent bursts, and daily light rain. Experiments show that H-J outperforms rule-based and reinforcement-learning baselines in traffic smoothness, task success rate, and system robustness. These findings highlight the promise of uncertainty-aware, knowledge-constrained LLM-based approaches for enhancing resilience in urban flood response.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14635v1">Can LLM Agents Solve Collaborative Tasks? A Study on Urgency-Aware Planning and Coordination</a></div>
    <div class="paper-meta">
      📅 2025-08-20
    </div>
    <details class="paper-abstract">
      The ability to coordinate actions across multiple agents is critical for solving complex, real-world problems. Large Language Models (LLMs) have shown strong capabilities in communication, planning, and reasoning, raising the question of whether they can also support effective collaboration in multi-agent settings. In this work, we investigate the use of LLM agents to solve a structured victim rescue task that requires division of labor, prioritization, and cooperative planning. Agents operate in a fully known graph-based environment and must allocate resources to victims with varying needs and urgency levels. We systematically evaluate their performance using a suite of coordination-sensitive metrics, including task success rate, redundant actions, room conflicts, and urgency-weighted efficiency. This study offers new insights into the strengths and failure modes of LLMs in physically grounded multi-agent collaboration tasks, contributing to future benchmarks and architectural improvements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03082v2">EoH-S: Evolution of Heuristic Set using LLMs for Automated Heuristic Design</a></div>
    <div class="paper-meta">
      📅 2025-08-20
    </div>
    <details class="paper-abstract">
      Automated Heuristic Design (AHD) using Large Language Models (LLMs) has achieved notable success in recent years. Despite the effectiveness of existing approaches, they only design a single heuristic to serve all problem instances, often inducing poor generalization across different distributions or settings. To address this issue, we propose Automated Heuristic Set Design (AHSD), a new formulation for LLM-driven AHD. The aim of AHSD is to automatically generate a small-sized complementary heuristic set to serve diverse problem instances, such that each problem instance could be optimized by at least one heuristic in this set. We show that the objective function of AHSD is monotone and supermodular. Then, we propose Evolution of Heuristic Set (EoH-S) to apply the AHSD formulation for LLM-driven AHD. With two novel mechanisms of complementary population management and complementary-aware memetic search, EoH-S could effectively generate a set of high-quality and complementary heuristics. Comprehensive experimental results on three AHD tasks with diverse instances spanning various sizes and distributions demonstrate that EoH-S consistently outperforms existing state-of-the-art AHD methods and achieves up to 60\% performance improvements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.12096v2">STEM: Efficient Relative Capability Evaluation of LLMs through Structured Transition Samples</a></div>
    <div class="paper-meta">
      📅 2025-08-20
      | 💬 Submit to AAAI 2026
    </div>
    <details class="paper-abstract">
      Evaluating large language models (LLMs) has become increasingly challenging as model capabilities advance rapidly. While recent models often achieve higher scores on standard benchmarks, these improvements do not consistently reflect enhanced real-world reasoning capabilities. Moreover, widespread overfitting to public benchmarks and the high computational cost of full evaluations have made it both expensive and less effective to distinguish meaningful differences between models. To address these challenges, we propose the \textbf{S}tructured \textbf{T}ransition \textbf{E}valuation \textbf{M}ethod (STEM), a lightweight and interpretable evaluation framework for efficiently estimating the relative capabilities of LLMs. STEM identifies \textit{significant transition samples} (STS) by analyzing consistent performance transitions among LLMs of the same architecture but varying parameter scales. These samples enable STEM to effectively estimate the capability position of an unknown model. Qwen3 model family is applied to construct the STS pool on six diverse and representative benchmarks. To assess generalizability. Experimental results indicate that STEM reliably captures performance trends, aligns with ground-truth rankings of model capability. These findings highlight STEM as a practical and scalable method for fine-grained, architecture-agnostic evaluation of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14564v1">Who Sees What? Structured Thought-Action Sequences for Epistemic Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-20
      | 💬 Accepted at ICSR25
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) and reasoning frameworks have opened new possibilities for improving the perspective -taking capabilities of autonomous agents. However, tasks that involve active perception, collaborative reasoning, and perspective taking (understanding what another agent can see or knows) pose persistent challenges for current LLM-based systems. This study investigates the potential of structured examples derived from transformed solution graphs generated by the Fast Downward planner to improve the performance of LLM-based agents within a ReAct framework. We propose a structured solution-processing pipeline that generates three distinct categories of examples: optimal goal paths (G-type), informative node paths (E-type), and step-by-step optimal decision sequences contrasting alternative actions (L-type). These solutions are further converted into ``thought-action'' examples by prompting an LLM to explicitly articulate the reasoning behind each decision. While L-type examples slightly reduce clarification requests and overall action steps, they do not yield consistent improvements. Agents are successful in tasks requiring basic attentional filtering but struggle in scenarios that required mentalising about occluded spaces or weighing the costs of epistemic actions. These findings suggest that structured examples alone are insufficient for robust perspective-taking, underscoring the need for explicit belief tracking, cost modelling, and richer environments to enable socially grounded collaboration in LLM-based agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14553v1">Towards LLM-generated explanations for Component-based Knowledge Graph Question Answering Systems</a></div>
    <div class="paper-meta">
      📅 2025-08-20
      | 💬 Presented at ICWI 2024, Zagreb. Released with ISBN: 978-989-8704-62-7. Data source: https://figshare.com/articles/dataset/Towards_LLM-generated_explanations_for_component-based_knowledge_graph_question_answering_systems/27079687
    </div>
    <details class="paper-abstract">
      Over time, software systems have reached a level of complexity that makes it difficult for their developers and users to explain particular decisions made by them. In this paper, we focus on the explainability of component-based systems for Question Answering (QA). These components often conduct processes driven by AI methods, in which behavior and decisions cannot be clearly explained or justified, s.t., even for QA experts interpreting the executed process and its results is hard. To address this challenge, we present an approach that considers the components' input and output data flows as a source for representing the behavior and provide explanations for the components, enabling users to comprehend what happened. In the QA framework used here, the data flows of the components are represented as SPARQL queries (inputs) and RDF triples (outputs). Hence, we are also providing valuable insights on verbalization regarding these data types. In our experiments, the approach generates explanations while following template-based settings (baseline) or via the use of Large Language Models (LLMs) with different configurations (automatic generation). Our evaluation shows that the explanations generated via LLMs achieve high quality and mostly outperform template-based approaches according to the users' ratings. Therefore, it enables us to automatically explain the behavior and decisions of QA components to humans while using RDF and SPARQL as a context for explanations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03106v5">Critique-GRPO: Advancing LLM Reasoning with Natural Language and Numerical Feedback</a></div>
    <div class="paper-meta">
      📅 2025-08-20
      | 💬 52 pages, updated with new experimental results and implementation details
    </div>
    <details class="paper-abstract">
      Recent advances in reinforcement learning (RL) with numerical feedback, such as scalar rewards, have significantly enhanced the complex reasoning capabilities of large language models (LLMs). Despite this success, we identify three key challenges encountered by RL with solely numerical feedback: performance plateaus, limited effectiveness of spontaneous self-reflection, and persistent failures. We then demonstrate that RL-finetuned models, even after exhibiting performance plateaus, can generate correct refinements on persistently failed problems by leveraging natural language feedback in the form of critiques. Building on this insight, we propose Critique-GRPO, an online RL framework that integrates both natural language and numerical feedback for effective policy optimization. Critique-GRPO enables LLMs to learn from initial responses and critique-guided self-refinements simultaneously while maintaining exploration. Additionally, we employ a shaping function to amplify learning from correct, especially unfamiliar, refinements and penalize incorrect ones. Extensive experiments with Qwen2.5-7B-Base, Qwen2.5-Math-7B-Base, and Qwen3-8B demonstrate that Critique-GRPO consistently outperforms supervised learning and RL-based fine-tuning methods across eight challenging mathematical, STEM, and general reasoning tasks. Specifically, Critique-GRPO improves average pass@1 scores across all compared methods by approximately +4.4% on Qwen2.5-7B-Base and +3.8% on Qwen3-8B. Notably, Critique-GRPO enables effective self-improvement through self-critiquing, achieving significant gains over GRPO, e.g., +16.7% pass@1 improvement on AIME 2024.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14544v1">Adaptively Robust LLM Inference Optimization under Prediction Uncertainty</a></div>
    <div class="paper-meta">
      📅 2025-08-20
    </div>
    <details class="paper-abstract">
      We study the problem of optimizing Large Language Model (LLM) inference scheduling to minimize total latency. LLM inference is an online and multi-task service process and also heavily energy consuming by which a pre-trained LLM processes input requests and generates output tokens sequentially. Therefore, it is vital to improve its scheduling efficiency and reduce the power consumption while a great amount of prompt requests are arriving. A key challenge in LLM inference scheduling is that while the prompt length is known upon arrival, the output length, which critically impacts memory usage and processing time, is unknown. To address this uncertainty, we propose algorithms that leverage machine learning to predict output lengths, assuming the prediction provides an interval classification (min-max range) for each request. We first design a conservative algorithm, $\mathcal{A}_{\max}$, which schedules requests based on the upper bound of predicted output lengths to prevent memory overflow. However, this approach is overly conservative: as prediction accuracy decreases, performance degrades significantly due to potential overestimation. To overcome this limitation, we propose $\mathcal{A}_{\min}$, an adaptive algorithm that initially treats the predicted lower bound as the output length and dynamically refines this estimate during inferencing. We prove that $\mathcal{A}_{\min}$ achieves a log-scale competitive ratio. Through numerical simulations, we demonstrate that $\mathcal{A}_{\min}$ often performs nearly as well as the hindsight scheduler, highlighting both its efficiency and robustness in practical scenarios. Moreover, $\mathcal{A}_{\min}$ relies solely on the lower bound of the prediction interval--an advantageous design choice since upper bounds on output length are typically more challenging to predict accurately.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02329v2">ReSpark: Leveraging Previous Data Reports as References to Generate New Reports with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-20
    </div>
    <details class="paper-abstract">
      Creating data reports is a labor-intensive task involving iterative data exploration, insight extraction, and narrative construction. A key challenge lies in composing the analysis logic-from defining objectives and transforming data to identifying and communicating insights. Manually crafting this logic can be cognitively demanding. While experienced analysts often reuse scripts from past projects, finding a perfect match for a new dataset is rare. Even when similar analyses are available online, they usually share only results or visualizations, not the underlying code, making reuse difficult. To address this, we present ReSpark, a system that leverages large language models (LLMs) to reverse-engineer analysis logic from existing reports and adapt it to new datasets. By generating draft analysis steps, ReSpark provides a warm start for users. It also supports interactive refinement, allowing users to inspect intermediate outputs, insert objectives, and revise content. We evaluate ReSpark through comparative and user studies, demonstrating its effectiveness in lowering the barrier to generating data reports without relying on existing analysis code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14496v1">Semantic Energy: Detecting LLM Hallucination Beyond Entropy</a></div>
    <div class="paper-meta">
      📅 2025-08-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are being increasingly deployed in real-world applications, but they remain susceptible to hallucinations, which produce fluent yet incorrect responses and lead to erroneous decision-making. Uncertainty estimation is a feasible approach to detect such hallucinations. For example, semantic entropy estimates uncertainty by considering the semantic diversity across multiple sampled responses, thus identifying hallucinations. However, semantic entropy relies on post-softmax probabilities and fails to capture the model's inherent uncertainty, causing it to be ineffective in certain scenarios. To address this issue, we introduce Semantic Energy, a novel uncertainty estimation framework that leverages the inherent confidence of LLMs by operating directly on logits of penultimate layer. By combining semantic clustering with a Boltzmann-inspired energy distribution, our method better captures uncertainty in cases where semantic entropy fails. Experiments across multiple benchmarks show that Semantic Energy significantly improves hallucination detection and uncertainty estimation, offering more reliable signals for downstream applications such as hallucination detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14460v1">DuPO: Enabling Reliable LLM Self-Verification via Dual Preference Optimization</a></div>
    <div class="paper-meta">
      📅 2025-08-20
      | 💬 18 pages, 4 figures,
    </div>
    <details class="paper-abstract">
      We present DuPO, a dual learning-based preference optimization framework that generates annotation-free feedback via a generalized duality. DuPO addresses two key limitations: Reinforcement Learning with Verifiable Rewards (RLVR)'s reliance on costly labels and applicability restricted to verifiable tasks, and traditional dual learning's restriction to strictly dual task pairs (e.g., translation and back-translation). Specifically, DuPO decomposes a primal task's input into known and unknown components, then constructs its dual task to reconstruct the unknown part using the primal output and known information (e.g., reversing math solutions to recover hidden variables), broadening applicability to non-invertible tasks. The quality of this reconstruction serves as a self-supervised reward to optimize the primal task, synergizing with LLMs' ability to instantiate both tasks via a single model. Empirically, DuPO achieves substantial gains across diverse tasks: it enhances the average translation quality by 2.13 COMET over 756 directions, boosts the mathematical reasoning accuracy by an average of 6.4 points on three challenge benchmarks, and enhances performance by 9.3 points as an inference-time reranker (trading computation for accuracy). These results position DuPO as a scalable, general, and annotation-free paradigm for LLM optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14419v1">Static Analysis as a Feedback Loop: Enhancing LLM-Generated Code Beyond Correctness</a></div>
    <div class="paper-meta">
      📅 2025-08-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive capabilities in code generation, achieving high scores on benchmarks such as HumanEval and MBPP. However, these benchmarks primarily assess functional correctness and neglect broader dimensions of code quality, including security, reliability, readability, and maintainability. In this work, we systematically evaluate the ability of LLMs to generate high-quality code across multiple dimensions using the PythonSecurityEval benchmark. We introduce an iterative static analysis-driven prompting algorithm that leverages Bandit and Pylint to identify and resolve code quality issues. Our experiments with GPT-4o show substantial improvements: security issues reduced from >40% to 13%, readability violations from >80% to 11%, and reliability warnings from >50% to 11% within ten iterations. These results demonstrate that LLMs, when guided by static analysis feedback, can significantly enhance code quality beyond functional correctness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14408v1">Cognitive Surgery: The Awakening of Implicit Territorial Awareness in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been shown to possess a degree of self-recognition capability-the ability to identify whether a given text was generated by themselves. Prior work has demonstrated that this capability is reliably expressed under the Pair Presentation Paradigm (PPP), where the model is presented with two texts and asked to choose which one it authored. However, performance deteriorates sharply under the Individual Presentation Paradigm (IPP), where the model is given a single text to judge authorship. Although this phenomenon has been observed, its underlying causes have not been systematically analyzed. In this paper, we first replicate existing findings to confirm that LLMs struggle to distinguish self- from other-generated text under IPP. We then investigate the reasons for this failure and attribute it to a phenomenon we term Implicit Territorial Awareness (ITA)-the model's latent ability to distinguish self- and other-texts in representational space, which remains unexpressed in its output behavior. To awaken the ITA of LLMs, we propose Cognitive Surgery (CoSur), a novel framework comprising four main modules: representation extraction, territory construction, authorship discrimination and cognitive editing. Experimental results demonstrate that our proposed method improves the performance of three different LLMs in the IPP scenario, achieving average accuracies of 83.25%, 66.19%, and 88.01%, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.12645v3">Diagnostic-Guided Dynamic Profile Optimization for LLM-based User Simulators in Sequential Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-08-20
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have enabled realistic user simulators for developing and evaluating recommender systems (RSs). However, existing LLM-based simulators for RSs face two major limitations: (1) static and single-step prompt-based inference that leads to inaccurate and incomplete user profile construction; (2) unrealistic and single-round recommendation-feedback interaction pattern that fails to capture real-world scenarios. To address these limitations, we propose DGDPO (Diagnostic-Guided Dynamic Profile Optimization), a novel framework that constructs user profile through a dynamic and iterative optimization process to enhance the simulation fidelity. Specifically, DGDPO incorporates two core modules within each optimization loop: firstly, a specialized LLM-based diagnostic module, calibrated through our novel training strategy, accurately identifies specific defects in the user profile. Subsequently, a generalized LLM-based treatment module analyzes the diagnosed defect and generates targeted suggestions to refine the profile. Furthermore, unlike existing LLM-based user simulators that are limited to single-round interactions, we are the first to integrate DGDPO with sequential recommenders, enabling a bidirectional evolution where user profiles and recommendation strategies adapt to each other over multi-round interactions. Extensive experiments conducted on three real-world datasets demonstrate the effectiveness of our proposed framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12152v2">Contextualizing Recommendation Explanations with LLMs: A User Study</a></div>
    <div class="paper-meta">
      📅 2025-08-19
      | 💬 Accepted to the International AAAI Conference on Web and Social Media (ICWSM 2026)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly prevalent in recommender systems, where LLMs can be used to generate personalized recommendations. Here, we examine how different LLM-generated explanations for movie recommendations affect users' perceptions of cognitive, affective, and utilitarian needs and consumption intentions. In a pre-registered, between-subject online experiment (N=759) and follow-up interviews (N=30), we compare (a) LLM-generated generic explanations, and (b) LLM-generated contextualized explanations. Our findings show that contextualized explanations (i.e., explanations that incorporate users' past behaviors) effectively meet users' cognitive needs while increasing users' intentions to watch recommended movies. However, adding explanations offers limited benefits in meeting users' utilitarian and affective needs, raising concerns about the proper design and implications of LLM-generated explanations. Qualitative insights from interviews reveal that referencing users' past preferences enhances trust and understanding but can feel excessive if overused. Furthermore, users with more active and positive engagement with the recommender system and movie-watching get substantial gains from contextualized explanations. Overall, our research clarifies how LLM-generated recommendations influence users' motivations and behaviors, providing valuable insights for the future development of user-centric recommender systems, a key element in social media platforms and online ecosystems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.13993v1">Chunks as Arms: Multi-Armed Bandit-Guided Sampling for Long-Context LLM Preference Optimization</a></div>
    <div class="paper-meta">
      📅 2025-08-19
    </div>
    <details class="paper-abstract">
      Long-context modeling is critical for a wide range of real-world tasks, including long-context question answering, summarization, and complex reasoning tasks. Recent studies have explored fine-tuning Large Language Models (LLMs) with synthetic data to enhance their long-context capabilities. However, the effectiveness of such approaches is often limited by the low diversity and factual inconsistencies in the generated data. To address these challenges, we propose LongMab-PO, a novel framework that leverages a Multi-Armed Bandit (MAB) rollout strategy to identify the most informative chunks from the given long context for sampling high-quality and diverse responses and constructing preference data pairs for Direct Preference Optimization (DPO) training. Specifically, we treat context chunks as arms of MAB, select chunks based on their expected reward scores to input into LLMs to generate responses, and iteratively update these scores based on reward feedback. This exploration and exploitation process enables the model to focus on the most relevant context segments, thereby generating and collecting high-quality and diverse responses. Finally, we collect these generated responses from the rollout process and apply the DPO method to further optimize the LLM. Experimental results show that LongMab-PO significantly improves the diversity and quality of preference data pairs, achieving state-of-the-art performance on long-context reasoning benchmarks. All code and data will be released on https://github.com/NEUIR/LongMab-PO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.05985v2">Exploring LLMs for Automated Generation and Adaptation of Questionnaires</a></div>
    <div class="paper-meta">
      📅 2025-08-19
      | 💬 Published in the Proceedings of the 7th ACM Conference on Conversational User Interfaces (CUI '25)
    </div>
    <details class="paper-abstract">
      Effective questionnaire design improves the validity of the results, but creating and adapting questionnaires across contexts is challenging due to resource constraints and limited expert access. Recently, the emergence of LLMs has led researchers to explore their potential in survey research. In this work, we focus on the suitability of LLMs in assisting the generation and adaptation of questionnaires. We introduce a novel pipeline that leverages LLMs to create new questionnaires, pretest with a target audience to determine potential issues and adapt existing standardized questionnaires for different contexts. We evaluated our pipeline for creation and adaptation through two studies on Prolific, involving 238 participants from the US and 118 participants from South Africa. Our findings show that participants found LLM-generated text clearer, LLM-pretested text more specific, and LLM-adapted questions slightly clearer and less biased than traditional ones. Our work opens new opportunities for LLM-driven questionnaire support in survey research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.13943v1">LLM-Powered Virtual Patient Agents for Interactive Clinical Skills Training with Automated Feedback</a></div>
    <div class="paper-meta">
      📅 2025-08-19
    </div>
    <details class="paper-abstract">
      Objective Structured Clinical Examinations (OSCEs) are essential for medical training, but they require significant resources, including professional actors and expert medical feedback. Although Large Language Models (LLMs) have introduced text-based virtual patients for communication practice, these simulations often lack the capability for richer, non-textual interactions. This paper presents a novel framework that significantly enhances LLM-based simulated patients by equipping them with action spaces, thereby enabling more realistic and dynamic patient behaviors that extend beyond text. Furthermore, our system incorporates virtual tutors that provide students with instant, personalized feedback on their performance at any time during these simulated encounters. We have conducted a rigorous evaluation of the framework's real-time performance, including system latency and component accuracy. Preliminary evaluations with medical experts assessed the naturalness and coherence of the simulated patients, as well as the usefulness and appropriateness of the virtual tutor's assessments. This innovative system provides medical students with a low-cost, accessible platform for personalized OSCE preparation at home.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.13920v1">LLMind 2.0: Distributed IoT Automation with Natural Language M2M Communication and Lightweight LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-08-19
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have sparked interest in their application to IoT and automation systems, particularly for facilitating device management through natural language instructions. However, existing centralized approaches face significant scalability challenges when managing and coordinating the collaboration between IoT devices of diverse capabilities in large-scale heterogeneous IoT systems. This paper introduces LLMind 2.0, a distributed IoT automation framework that addresses the scalability challenges through lightweight LLM-empowered device agents via natural language-based machine-to-machine (M2M) communication. Unlike previous LLM-controlled automation systems that rely on a centralized coordinator to generate device-specific code to be executed on individual devices, LLMind 2.0 distributes intelligence across individual devices through lightweight LLMs embedded in IoT devices. The central coordinator translates human instructions into simple subtasks described in natural human language, which are then processed by device-specific agents to generate device-specific code locally at the associated devices. This approach transcends device heterogeneity barriers by using natural language as a unified communication medium, enabling seamless collaboration between devices from different manufacturers. The system incorporates several key innovations: a Retrieval-Augmented Generation (RAG) mechanism for accurate subtask-to-API mapping, fine-tuned lightweight LLMs for reliable code generation, and a finite state machine-based task execution framework. Experimental validation in multi-robot warehouse scenarios and real-world WiFi network deployments demonstrates significant improvements in scalability, reliability, and privacy protection compared to the centralized approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.13915v1">Structured Agentic Workflows for Financial Time-Series Modeling with LLMs and Reflective Feedback</a></div>
    <div class="paper-meta">
      📅 2025-08-19
    </div>
    <details class="paper-abstract">
      Time-series data is central to decision-making in financial markets, yet building high-performing, interpretable, and auditable models remains a major challenge. While Automated Machine Learning (AutoML) frameworks streamline model development, they often lack adaptability and responsiveness to domain-specific needs and evolving objectives. Concurrently, Large Language Models (LLMs) have enabled agentic systems capable of reasoning, memory management, and dynamic code generation, offering a path toward more flexible workflow automation. In this paper, we introduce \textsf{TS-Agent}, a modular agentic framework designed to automate and enhance time-series modeling workflows for financial applications. The agent formalizes the pipeline as a structured, iterative decision process across three stages: model selection, code refinement, and fine-tuning, guided by contextual reasoning and experimental feedback. Central to our architecture is a planner agent equipped with structured knowledge banks, curated libraries of models and refinement strategies, which guide exploration, while improving interpretability and reducing error propagation. \textsf{TS-Agent} supports adaptive learning, robust debugging, and transparent auditing, key requirements for high-stakes environments such as financial services. Empirical evaluations on diverse financial forecasting and synthetic data generation tasks demonstrate that \textsf{TS-Agent} consistently outperforms state-of-the-art AutoML and agentic baselines, achieving superior accuracy, robustness, and decision traceability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.13889v1">CARE: Contextual Adaptation of Recommenders for LLM-based Conversational Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-08-19
    </div>
    <details class="paper-abstract">
      We tackle the challenge of integrating large language models (LLMs) with external recommender systems to enhance domain expertise in conversational recommendation (CRS). Current LLM-based CRS approaches primarily rely on zero- or few-shot methods for generating item recommendations based on user queries, but this method faces two significant challenges: (1) without domain-specific adaptation, LLMs frequently recommend items not in the target item space, resulting in low recommendation accuracy; and (2) LLMs largely rely on dialogue context for content-based recommendations, neglecting the collaborative relationships among entities or item sequences. To address these limitations, we introduce the CARE (Contextual Adaptation of Recommenders) framework. CARE customizes LLMs for CRS tasks, and synergizes them with external recommendation systems. CARE (a) integrates external recommender systems as domain experts, producing recommendations through entity-level insights, and (b) enhances those recommendations by leveraging contextual information for more accurate and unbiased final recommendations using LLMs. Our results demonstrate that incorporating external recommender systems with entity-level information significantly enhances recommendation accuracy of LLM-based CRS by an average of 54% and 25% for ReDial and INSPIRED datasets. The most effective strategy in the CARE framework involves LLMs selecting and reranking candidate items that external recommenders provide based on contextual insights. Our analysis indicates that the CARE framework effectively addresses the identified challenges and mitigates the popularity bias in the external recommender.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.13876v1">Improved Generalized Planning with LLMs through Strategy Refinement and Reflection</a></div>
    <div class="paper-meta">
      📅 2025-08-19
    </div>
    <details class="paper-abstract">
      LLMs have recently been used to generate Python programs representing generalized plans in PDDL planning, i.e., plans that generalize across the tasks of a given PDDL domain. Previous work proposed a framework consisting of three steps: the LLM first generates a summary and then a strategy for the domain, both in natural language, and then implements that strategy as a Python program, that gets debugged on example planning tasks. In that work, only one strategy is generated and passed directly to the program generation. If the strategy is incorrect, its implementation will therefore result in an incorrect generalized plan. Here, we introduce an approach that generates the strategy in the form of pseudocode and enables automatic debugging of the pseudocode, hence allowing us to identify and fix errors prior to the generation of the generalized plan itself. Additionally, we extend the Python debugging phase with a reflection step prompting the LLM to pinpoint the reason for the observed plan failure. Finally, we take inspiration from LLM code generation to produce several program variants and pick the best one. Running experiments on 17 benchmark domains, we show that these extensions substantially improve (and never deteriorate) the quality of the generalized plans. In 12 of the domains, our best Python programs solve all tasks that can be generated with the respective instance generator.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.13805v1">Prompt-Based One-Shot Exact Length-Controlled Generation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-19
      | 💬 18 pages
    </div>
    <details class="paper-abstract">
      Controlling the length of text produced by large language models (LLMs) remains challenging: models frequently overshoot or undershoot explicit length instructions because they cannot reliably keep an internal token count. We present a prompt-based, one-shot strategy that compels an off-the-shelf LLM to generate exactly a desired number of tokens - words (English) or characters (Chinese) - without any fine-tuning or iterative sampling. The prompt appends countdown markers and explicit counting rules so that the model "writes while counting." We evaluate on four settings: open-ended generation (1-1000 tokens), XSUM summarization, MT-Bench-LI instruction following, and the LIFEBENCH equal-length track. On MT-Bench-LI, strict length compliance with GPT-4.1 leaps from below 30% under naive prompts to above 95% with our countdown prompt, surpassing the popular draft-then-revise baseline, while judged answer quality is preserved. These results show that precise length control can be achieved through prompt engineering alone, offering a lightweight alternative to training- or decoding-based methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.13774v1">Agentic DraCor and the Art of Docstring Engineering: Evaluating MCP-empowered LLM Usage of the DraCor API</a></div>
    <div class="paper-meta">
      📅 2025-08-19
      | 💬 Preprint, submitted to the 2nd Workshop on Computational Drama Analysis at DraCor Summit 2025, September 03, 2025, Berlin, Germany
    </div>
    <details class="paper-abstract">
      This paper reports on the implementation and evaluation of a Model Context Protocol (MCP) server for DraCor, enabling Large Language Models (LLM) to autonomously interact with the DraCor API. We conducted experiments focusing on tool selection and application by the LLM, employing a qualitative approach that includes systematic observation of prompts to understand how LLMs behave when using MCP tools, evaluating "Tool Correctness", "Tool-Calling Efficiency", and "Tool-Use Reliability". Our findings highlight the importance of "Docstring Engineering", defined as reflexively crafting tool documentation to optimize LLM-tool interaction. Our experiments demonstrate both the promise of agentic AI for research in Computational Literary Studies and the essential infrastructure development needs for reliable Digital Humanities infrastructures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.13769v1">Can Large Language Models (LLMs) Describe Pictures Like Children? A Comparative Corpus Study</a></div>
    <div class="paper-meta">
      📅 2025-08-19
    </div>
    <details class="paper-abstract">
      The role of large language models (LLMs) in education is increasing, yet little attention has been paid to whether LLM-generated text resembles child language. This study evaluates how LLMs replicate child-like language by comparing LLM-generated texts to a collection of German children's descriptions of picture stories. We generated two LLM-based corpora using the same picture stories and two prompt types: zero-shot and few-shot prompts specifying a general age from the children corpus. We conducted a comparative analysis across psycholinguistic text properties, including word frequency, lexical richness, sentence and word length, part-of-speech tags, and semantic similarity with word embeddings. The results show that LLM-generated texts are longer but less lexically rich, rely more on high-frequency words, and under-represent nouns. Semantic vector space analysis revealed low similarity, highlighting differences between the two corpora on the level of corpus semantics. Few-shot prompt increased similarities between children and LLM text to a minor extent, but still failed to replicate lexical and semantic patterns. The findings contribute to our understanding of how LLMs approximate child language through multimodal prompting (text + image) and give insights into their use in psycholinguistic research and education while raising important questions about the appropriateness of LLM-generated language in child-directed educational tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.13754v1">Expertise-aware Multi-LLM Recruitment and Collaboration for Medical Decision-Making</a></div>
    <div class="paper-meta">
      📅 2025-08-19
      | 💬 14 pages
    </div>
    <details class="paper-abstract">
      Medical Decision-Making (MDM) is a complex process requiring substantial domain-specific expertise to effectively synthesize heterogeneous and complicated clinical information. While recent advancements in Large Language Models (LLMs) show promise in supporting MDM, single-LLM approaches are limited by their parametric knowledge constraints and static training corpora, failing to robustly integrate the clinical information. To address this challenge, we propose the Expertise-aware Multi-LLM Recruitment and Collaboration (EMRC) framework to enhance the accuracy and reliability of MDM systems. It operates in two stages: (i) expertise-aware agent recruitment and (ii) confidence- and adversarial-driven multi-agent collaboration. Specifically, in the first stage, we use a publicly available corpus to construct an LLM expertise table for capturing expertise-specific strengths of multiple LLMs across medical department categories and query difficulty levels. This table enables the subsequent dynamic selection of the optimal LLMs to act as medical expert agents for each medical query during the inference phase. In the second stage, we employ selected agents to generate responses with self-assessed confidence scores, which are then integrated through the confidence fusion and adversarial validation to improve diagnostic reliability. We evaluate our EMRC framework on three public MDM datasets, where the results demonstrate that our EMRC outperforms state-of-the-art single- and multi-LLM methods, achieving superior diagnostic performance. For instance, on the MMLU-Pro-Health dataset, our EMRC achieves 74.45% accuracy, representing a 2.69% improvement over the best-performing closed-source model GPT- 4-0613, which demonstrates the effectiveness of our expertise-aware agent recruitment strategy and the agent complementarity in leveraging each LLM's specialized capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14226v2">"Haet Bhasha aur Diskrimineshun": Phonetic Perturbations in Code-Mixed Hinglish to Red-Team LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-19
    </div>
    <details class="paper-abstract">
      Recently released LLMs have strong multilingual \& multimodal capabilities. Model vulnerabilities are exposed using audits and red-teaming efforts. Existing efforts have focused primarily on the English language; thus, models continue to be susceptible to multilingual jailbreaking strategies, especially for multimodal contexts. In this study, we introduce a novel strategy that leverages code-mixing and phonetic perturbations to jailbreak LLMs for both text and image generation tasks. We also introduce \textit{two new} jailbreak strategies that show higher effectiveness than baselines. Our work presents a method to effectively bypass safety filters in LLMs while maintaining interpretability by applying phonetic misspellings to sensitive words in code-mixed prompts. We achieve a 99\% Attack Success Rate for text generation and 78\% for image generation, with Attack Relevance Rate of 100\% for text generation and 95\% for image generation for the phonetically perturbed code-mixed prompts. Our interpretability experiments reveal that phonetic perturbations impact word tokenization, leading to jailbreak success. Our study motivates increasing the focus towards more generalizable safety alignment for multilingual multimodal models, especially in real-world settings wherein prompts can have misspelt words. \textit{\textbf{Warning: This paper contains examples of potentially harmful and offensive content.}}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13250v2">MindEye-OmniAssist: A Gaze-Driven LLM-Enhanced Assistive Robot System for Implicit Intention Recognition and Task Execution</a></div>
    <div class="paper-meta">
      📅 2025-08-19
    </div>
    <details class="paper-abstract">
      A promising effective human-robot interaction in assistive robotic systems is gaze-based control. However, current gaze-based assistive systems mainly help users with basic grasping actions, offering limited support. Moreover, the restricted intent recognition capability constrains the assistive system's ability to provide diverse assistance functions. In this paper, we propose an open implicit intention recognition framework powered by Large Language Model (LLM) and Vision Foundation Model (VFM), which can process gaze input and recognize user intents that are not confined to predefined or specific scenarios. Furthermore, we implement a gaze-driven LLM-enhanced assistive robot system (MindEye-OmniAssist) that recognizes user's intentions through gaze and assists in completing task. To achieve this, the system utilizes open vocabulary object detector, intention recognition network and LLM to infer their full intentions. By integrating eye movement feedback and LLM, it generates action sequences to assist the user in completing tasks. Real-world experiments have been conducted for assistive tasks, and the system achieved an overall success rate of 41/55 across various undefined tasks. Preliminary results show that the proposed method holds the potential to provide a more user-friendly human-computer interaction interface and significantly enhance the versatility and effectiveness of assistive systems by supporting more complex and diverse task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.13732v1">Self-Organizing Agent Network for LLM-based Workflow Automation</a></div>
    <div class="paper-meta">
      📅 2025-08-19
    </div>
    <details class="paper-abstract">
      Recent multi-agent frameworks built upon large language models (LLMs) have demonstrated remarkable capabilities in complex task planning. However, in real-world enterprise environments, business workflows are typically composed through modularization and reuse of numerous subprocesses, resulting in intricate workflows characterized by lengthy and deeply nested execution paths. Such complexity poses significant challenges for LLM-driven orchestration, as extended reasoning chains and state-space explosions severely impact planning effectiveness and the proper sequencing of tool invocations. Therefore, developing an orchestration method with controllable structures capable of handling multi-layer nesting becomes a critical issue. To address this, we propose a novel structure-driven orchestration framework Self-Organizing Agent Network (SOAN). SOAN incrementally builds a formalized agent network by identifying and encapsulating structural units as independent agents, enhancing modularity and clarity in orchestration. Extensive evaluations were performed using multiple benchmarks as well as a real-world enterprise workflow dataset. Experimental results demonstrate that SOAN significantly outperforms state-of-the-art methods in terms of adaptability, fault tolerance, and execution efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.13721v1">CausalPlan: Empowering Efficient LLM Multi-Agent Collaboration Through Causality-Driven Planning</a></div>
    <div class="paper-meta">
      📅 2025-08-19
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents-especially smaller, open-source models-often produce causally invalid or incoherent actions in collaborative tasks due to their reliance on surface-level correlations rather than grounded causal reasoning. This limitation undermines their performance in terms of coordination and planning in dynamic environments. We address this challenge with CausalPlan, a two-phase framework that integrates explicit structural causal reasoning into the LLM planning process. At the core of CausalPlan is the Structural Causal Action (SCA) model, which learns a causal graph from agent trajectories to capture how prior actions and current environment states influence future decisions. This structure is then used to guide action selection by assigning causal scores to LLM-generated proposals, reweighting them accordingly, or falling back to causally grounded alternatives when needed. By embedding this causal knowledge directly into the decision loop, CausalPlan constrains planning to intervention-consistent behaviours without requiring fine-tuning of the LLM itself. We evaluate CausalPlan on the Overcooked-AI benchmark across five multi-agent coordination tasks and four LLMs of varying sizes: Gemma-7B, Llama-8B, Qwen-14B, and Llama-70B. Experimental results show that CausalPlan consistently reduces invalid actions and improves collaboration in both AI-AI and human-AI settings, outperforming strong reinforcement learning baselines. Our findings highlight the value of causality-driven planning for deploying efficient, interpretable, and generalisable multi-agent LLM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.11290v2">Iterative Utility Judgment Framework via LLMs Inspired by Relevance in Philosophy</a></div>
    <div class="paper-meta">
      📅 2025-08-19
      | 💬 22 pages
    </div>
    <details class="paper-abstract">
      Relevance and utility are two frequently used measures to evaluate the effectiveness of an information retrieval (IR) system. Relevance emphasizes the aboutness of a result to a query, while utility refers to the result's usefulness or value to an information seeker. In Retrieval-Augmented Generation (RAG), high-utility results should be prioritized to feed to LLMs due to their limited input bandwidth. Re-examining RAG's three core components -- relevance ranking derived from retrieval models, utility judgments, and answer generation -- aligns with Schutz's philosophical system of relevances, which encompasses three types of relevance representing different levels of human cognition that enhance each other. These three RAG components also reflect three cognitive levels for LLMs in question-answering. Therefore, we propose an Iterative utiliTy judgmEnt fraMework (ITEM) to promote each step in RAG. We conducted extensive experiments on retrieval (TREC DL, WebAP), utility judgment task (GTI-NQ), and factoid question-answering (NQ) datasets. Experimental results demonstrate significant improvements of ITEM in utility judgments, ranking, and answer generation upon representative baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05216v3">Unleashing the Power of LLMs in Dense Retrieval with Query Likelihood Modeling</a></div>
    <div class="paper-meta">
      📅 2025-08-19
      | 💬 12 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Dense retrieval is a crucial task in Information Retrieval (IR), serving as the basis for downstream tasks such as re-ranking and augmenting generation. Recently, large language models (LLMs) have demonstrated impressive semantic understanding capabilities, making them attractive to researchers focusing on dense retrieval. While LLMs, as decoder-style generative models, excel in language generation, they often fall short in modeling global information due to a lack of attention to subsequent tokens. Drawing inspiration from the classical word-based language modeling approach for IR, specifically the query likelihood (QL) model, we aim to leverage the generative strengths of LLMs through QL maximization. Rather than employing QL estimation for document ranking, we propose an auxiliary task of QL maximization to enhance the backbone for subsequent contrastive learning of the retriever. We introduce our model, LLM-QL, which incorporates two key components: Attention Block (AB) and Document Corruption (DC). AB blocks the attention of predictive tokens to the document tokens before the document's ending token, while DC corrupts a document by masking a portion of its tokens during prediction. Evaluations on the in-domain (MS MARCO) and out-of-domain dataset (BEIR) indicate LLM-QL's superiority over other LLM-based retrievers. Furthermore, comprehensive analyses also validate the efficacy of LLM-QL and its components.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.13092v2">VerilogLAVD: LLM-Aided Rule Generation for Vulnerability Detection in Verilog</a></div>
    <div class="paper-meta">
      📅 2025-08-19
    </div>
    <details class="paper-abstract">
      Timely detection of hardware vulnerabilities during the early design stage is critical for reducing remediation costs. Existing early detection techniques often require specialized security expertise, limiting their usability. Recent efforts have explored the use of large language models (LLMs) for Verilog vulnerability detection. However, LLMs struggle to capture the structure in Verilog code, resulting in inconsistent detection results. To this end, we propose VerilogLAVD, the first LLM-aided graph traversal rule generation approach for Verilog vulnerability detection. Our approach introduces the Verilog Property Graph (VeriPG), a unified representation of Verilog code. It combines syntactic features extracted from the abstract syntax tree (AST) with semantic information derived from control flow and data dependency graphs. We leverage LLMs to generate VeriPG-based detection rules from Common Weakness Enumeration (CWE) descriptions. These rules guide the rule executor that traversal VeriPG for potential vulnerabilities. To evaluate VerilogLAVD, we build a dataset collected from open-source repositories and synthesized data. In our empirical evaluation on 77 Verilog designs encompassing 12 CWE types, VerilogLAVD achieves an F1-score of 0.54. Compared to the LLM-only and LLM with external knowledge baselines, VerilogLAVD improves F1-score by 0.31 and 0.27, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.13666v1">The Hidden Cost of Readability: How Code Formatting Silently Consumes Your LLM Budget</a></div>
    <div class="paper-meta">
      📅 2025-08-19
      | 💬 Accepted by ICSE'26 (First Cycle)
    </div>
    <details class="paper-abstract">
      Source code is usually formatted with elements like indentation and newlines to improve readability for human developers. However, these visual aids do not seem to be beneficial for large language models (LLMs) in the same way since the code is processed as a linear sequence of tokens. Furthermore, these additional tokens can lead to increased computational costs and longer response times for LLMs. If such formatting elements are non-essential to LLMs, we can reduce such costs by removing them from the code. To figure out the role played by formatting elements, we conduct a comprehensive empirical study to evaluate the impact of code formatting on LLM performance and efficiency. Through large-scale experiments on Fill-in-the-Middle Code Completion tasks across four programming languages (Java, Python, C++, C\#) and ten LLMs-including both commercial and open-source models-we systematically analyze token count and performance when formatting elements are removed. Key findings indicate that LLMs can maintain performance across formatted code and unformatted code, achieving an average input token reduction of 24.5\% with negligible output token reductions. This makes code format removal a practical optimization strategy for improving LLM efficiency. Further exploration reveals that both prompting and fine-tuning LLMs can lead to significant reductions (up to 36.1\%) in output code length without compromising correctness. To facilitate practical applications, we develop a bidirectional code transformation tool for format processing, which can be seamlessly integrated into existing LLM inference workflows, ensuring both human readability and LLM efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.09190v2">Fine-Grained Safety Neurons with Training-Free Continual Projection to Reduce LLM Fine Tuning Risks</a></div>
    <div class="paper-meta">
      📅 2025-08-19
    </div>
    <details class="paper-abstract">
      Fine-tuning as service injects domain-specific knowledge into large language models (LLMs), while challenging the original alignment mechanisms and introducing safety risks. A series of defense strategies have been proposed for the alignment, fine-tuning, and post-fine-tuning phases, where most post-fine-tuning defenses rely on coarse-grained safety layer mapping. These methods lack a comprehensive consideration of both safety layers and fine-grained neurons, limiting their ability to efficiently balance safety and utility. To address this, we propose the Fine-Grained Safety Neurons (FGSN) with Training-Free Continual Projection method to reduce the fine-tuning safety risks. FGSN inherently integrates the multi-scale interactions between safety layers and neurons, localizing sparser and more precise fine-grained safety neurons while minimizing interference with downstream task neurons. We then project the safety neuron parameters onto safety directions, improving model safety while aligning more closely with human preferences. Extensive experiments across multiple fine-tuned LLM models demonstrate that our method significantly reduce harmfulness scores and attack success rates with minimal parameter modifications, while preserving the model's utility. Furthermore, by introducing a task-specific, multi-dimensional heterogeneous safety neuron cluster optimization mechanism, we achieve continual defense and generalization capability against unforeseen emerging safety concerns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17642v2">May the Feedback Be with You! Unlocking the Power of Feedback-Driven Deep Learning Framework Fuzzing via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-19
    </div>
    <details class="paper-abstract">
      Deep Learning (DL) frameworks have served as fundamental components in DL systems over the last decade. However, bugs in DL frameworks could lead to catastrophic consequences in critical scenarios. A simple yet effective way to find bugs in DL frameworks is fuzz testing (Fuzzing). Existing approaches focus on test generation, leaving execution results with high semantic value (e.g., coverage information, bug reports, and exception logs) in the wild, which can serve as multiple types of feedback. To fill this gap, we propose FUEL to effectively utilize the feedback information, which comprises two Large Language Models (LLMs): analysis LLM and generation LLM. Specifically, analysis LLM infers analysis summaries from feedback information, while the generation LLM creates tests guided by these summaries. Furthermore, based on multiple feedback guidance, we design two additional components: (i) a feedback-aware simulated annealing algorithm to select operators for test generation, enriching test diversity. (ii) a program self-repair strategy to automatically repair invalid tests, enhancing test validity. We evaluate FUEL on the two most popular DL frameworks, and experiment results show that FUEL can improve line code coverage of PyTorch and TensorFlow by 9.15% and 14.70% over state-of-the-art baselines (e.g., TitanFuzz and WhiteFox). By the time of submission, FUEL has detected 104 previously unknown bugs for PyTorch and TensorFlow, with 93 confirmed as new bugs, 49 already fixed, and 5 assigned CVE IDs. Our artifact is available at https://github.com/NJU-iSE/FUEL
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.13603v1">Who Gets the Mic? Investigating Gender Bias in the Speaker Assignment of a Speech-LLM</a></div>
    <div class="paper-meta">
      📅 2025-08-19
    </div>
    <details class="paper-abstract">
      Similar to text-based Large Language Models (LLMs), Speech-LLMs exhibit emergent abilities and context awareness. However, whether these similarities extend to gender bias remains an open question. This study proposes a methodology leveraging speaker assignment as an analytic tool for bias investigation. Unlike text-based models, which encode gendered associations implicitly, Speech-LLMs must produce a gendered voice, making speaker selection an explicit bias cue. We evaluate Bark, a Text-to-Speech (TTS) model, analyzing its default speaker assignments for textual prompts. If Bark's speaker selection systematically aligns with gendered associations, it may reveal patterns in its training data or model design. To test this, we construct two datasets: (i) Professions, containing gender-stereotyped occupations, and (ii) Gender-Colored Words, featuring gendered connotations. While Bark does not exhibit systematic bias, it demonstrates gender awareness and has some gender inclinations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.13579v1">Toward Better EHR Reasoning in LLMs: Reinforcement Learning with Expert Attention Guidance</a></div>
    <div class="paper-meta">
      📅 2025-08-19
    </div>
    <details class="paper-abstract">
      Improving large language models (LLMs) for electronic health record (EHR) reasoning is essential for enabling accurate and generalizable clinical predictions. While LLMs excel at medical text understanding, they underperform on EHR-based prediction tasks due to challenges in modeling temporally structured, high-dimensional data. Existing approaches often rely on hybrid paradigms, where LLMs serve merely as frozen prior retrievers while downstream deep learning (DL) models handle prediction, failing to improve the LLM's intrinsic reasoning capacity and inheriting the generalization limitations of DL models. To this end, we propose EAG-RL, a novel two-stage training framework designed to intrinsically enhance LLMs' EHR reasoning ability through expert attention guidance, where expert EHR models refer to task-specific DL models trained on EHR data. Concretely, EAG-RL first constructs high-quality, stepwise reasoning trajectories using expert-guided Monte Carlo Tree Search to effectively initialize the LLM's policy. Then, EAG-RL further optimizes the policy via reinforcement learning by aligning the LLM's attention with clinically salient features identified by expert EHR models. Extensive experiments on two real-world EHR datasets show that EAG-RL improves the intrinsic EHR reasoning ability of LLMs by an average of 14.62%, while also enhancing robustness to feature perturbations and generalization to unseen clinical domains. These results demonstrate the practical potential of EAG-RL for real-world deployment in clinical prediction tasks. Our code have been available at https://github.com/devilran6/EAG-RL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07252v2">Tasa: Thermal-aware 3D-Stacked Architecture Design with Bandwidth Sharing for LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-08-19
      | 💬 there are some data inaccuracies in section V
    </div>
    <details class="paper-abstract">
      The autoregressive decoding in LLMs is the major inference bottleneck due to the memory-intensive operations and limited hardware bandwidth. 3D-stacked architecture is a promising solution with significantly improved memory bandwidth, which vertically stacked multi DRAM dies on top of logic die. However, our experiments also show the 3D-stacked architecture faces severer thermal issues compared to 2D architecture, in terms of thermal temperature, gradient and scalability. To better exploit the potential of 3D-stacked architecture, we present Tasa, a heterogeneous architecture with cross-stack thermal optimizations to balance the temperature distribution and maximize the performance under the thermal constraints. High-performance core is designed for compute-intensive operations, while high-efficiency core is used for memory-intensive operators, e.g. attention layers. Furthermore, we propose a bandwidth sharing scheduling to improve the bandwidth utilization in such heterogeneous architecture. Extensive thermal experiments show that our Tasa architecture demonstrates greater scalability compared with the homogeneous 3D-stacked architecture, i.e. up to 5.55 $\tccentigrade$, 9.37 $\tccentigrade$, and 7.91 $\tccentigrade$ peak temperature reduction for 48, 60, and 72 core configurations. Our experimental for Llama-65B and GPT-3 66B inferences also demonstrate 2.85x and 2.21x speedup are obtained over the GPU baselines and state-of-the-art heterogeneous PIM-based LLM accelerator
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07227v2">LP-Spec: Leveraging LPDDR PIM for Efficient LLM Mobile Speculative Inference with Architecture-Dataflow Co-Optimization</a></div>
    <div class="paper-meta">
      📅 2025-08-19
      | 💬 there are some data inaccuracies in section III
    </div>
    <details class="paper-abstract">
      LLM inference on mobile devices faces extraneous challenges due to limited memory bandwidth and computational resources. To address these issues, speculative inference and processing-in-memory (PIM) techniques have been explored at the algorithmic and hardware levels. However, speculative inference results in more compute-intensive GEMM operations, creating new design trade-offs for existing GEMV-accelerated PIM architectures. Furthermore, there exists a significant amount of redundant draft tokens in tree-based speculative inference, necessitating efficient token management schemes to minimize energy consumption. In this work, we present LP-Spec, an architecture-dataflow co-design leveraging hybrid LPDDR5 performance-enhanced PIM architecture with draft token pruning and dynamic workload scheduling to accelerate LLM speculative inference. A near-data memory controller is proposed to enable data reallocation between DRAM and PIM banks. Furthermore, a data allocation unit based on the hardware-aware draft token pruner is developed to minimize energy consumption and fully exploit parallel execution opportunities. Compared to end-to-end LLM inference on other mobile solutions such as mobile NPUs or GEMV-accelerated PIMs, our LP-Spec achieves 13.21x, 7.56x, and 99.87x improvements in performance, energy efficiency, and energy-delay-product (EDP). Compared with prior AttAcc PIM and RTX 3090 GPU, LP-Spec can obtain 12.83x and 415.31x EDP reduction benefits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.13543v1">"Can You See Me Think?" Grounding LLM Feedback in Keystrokes and Revision Patterns</a></div>
    <div class="paper-meta">
      📅 2025-08-19
      | 💬 15 pages, 4 figures, 6 tables, Submitted to IJCNLP-AACL 2025
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) increasingly assist in evaluating student writing, researchers have begun questioning whether these models can be cognitively grounded, that is, whether they can attend not just to the final product, but to the process by which it was written. In this study, we explore how incorporating writing process data, specifically keylogs and time-stamped snapshots, affects the quality of LLM-generated feedback. We conduct an ablation study on 52 student essays comparing feedback generated with access to only the final essay (C1) and feedback that also incorporates keylogs and time-stamped snapshots (C2). While rubric scores changed minimally, C2 feedback demonstrated significantly improved structural evaluation and greater process-sensitive justification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05668v3">A Survey of LLM-based Deep Search Agents: Paradigm, Optimization, Evaluation, and Challenges</a></div>
    <div class="paper-meta">
      📅 2025-08-19
    </div>
    <details class="paper-abstract">
      The advent of Large Language Models (LLMs) has significantly revolutionized web search. The emergence of LLM-based Search Agents marks a pivotal shift towards deeper, dynamic, autonomous information seeking. These agents can comprehend user intentions and environmental context and execute multi-turn retrieval with dynamic planning, extending search capabilities far beyond the web. Leading examples like OpenAI's Deep Research highlight their potential for deep information mining and real-world applications. This survey provides the first systematic analysis of search agents. We comprehensively analyze and categorize existing works from the perspectives of architecture, optimization, application, and evaluation, ultimately identifying critical open challenges and outlining promising future research directions in this rapidly evolving field. Our repository is available on https://github.com/YunjiaXi/Awesome-Search-Agent-Papers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.13514v1">ProMed: Shapley Information Gain Guided Reinforcement Learning for Proactive Medical LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-19
    </div>
    <details class="paper-abstract">
      Interactive medical questioning is essential in real-world clinical consultations, where physicians must actively gather information from patients. While medical Large Language Models (LLMs) have shown impressive capabilities in static medical question answering, they predominantly operate under a reactive paradigm: generating answers directly without seeking additional information, which risks incorrect diagnoses in such interactive settings. To address this limitation, we propose ProMed, a reinforcement learning (RL) framework that transitions medical LLMs toward a proactive paradigm, equipping them with the ability to ask clinically valuable questions before decision-making. At the core of ProMed is the Shapley Information Gain (SIG) reward, which quantifies the clinical utility of each question by combining the amount of newly acquired information with its contextual importance, estimated via Shapley values. We integrate SIG into a two-stage training pipeline: (1) SIG-Guided Model Initialization uses Monte Carlo Tree Search (MCTS) to construct high-reward interaction trajectories to supervise the model, and (2) SIG-Augmented Policy Optimization, which integrates SIG and enhances RL with a novel SIG-guided Reward Distribution Mechanism that assigns higher rewards to informative questions for targeted optimization. Extensive experiments on two newly curated partial-information medical benchmarks demonstrate that ProMed significantly outperforms state-of-the-art methods by an average of 6.29% and delivers a 54.45% gain over the reactive paradigm, while also generalizing robustly to out-of-domain cases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06723v2">Script-Strategy Aligned Generation: Aligning LLMs with Expert-Crafted Dialogue Scripts and Therapeutic Strategies for Psychotherapy</a></div>
    <div class="paper-meta">
      📅 2025-08-19
    </div>
    <details class="paper-abstract">
      Chatbots or conversational agents (CAs) are increasingly used to improve access to digital psychotherapy. Many current systems rely on rigid, rule-based designs, heavily dependent on expert-crafted dialogue scripts for guiding therapeutic conversations. Although advances in large language models (LLMs) offer potential for more flexible interactions, their lack of controllability and explanability poses challenges in high-stakes contexts like psychotherapy. To address this, we conducted two studies in this work to explore how aligning LLMs with expert-crafted scripts can enhance psychotherapeutic chatbot performance. In Study 1 (N=43), an online experiment with a within-subjects design, we compared rule-based, pure LLM, and LLMs aligned with expert-crafted scripts via fine-tuning and prompting. Results showed that aligned LLMs significantly outperformed the other types of chatbots in empathy, dialogue relevance, and adherence to therapeutic principles. Building on findings, we proposed ``Script-Strategy Aligned Generation (SSAG)'', a more flexible alignment approach that reduces reliance on fully scripted content while maintaining LLMs' therapeutic adherence and controllability. In a 10-day field Study 2 (N=21), SSAG achieved comparable therapeutic effectiveness to full-scripted LLMs while requiring less than 40\% of expert-crafted dialogue content. Beyond these results, this work advances LLM applications in psychotherapy by providing a controllable and scalable solution, reducing reliance on expert effort. By enabling domain experts to align LLMs through high-level strategies rather than full scripts, SSAG supports more efficient co-development and expands access to a broader context of psychotherapy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.09539v2">TFRank: Think-Free Reasoning Enables Practical Pointwise LLM Ranking</a></div>
    <div class="paper-meta">
      📅 2025-08-19
    </div>
    <details class="paper-abstract">
      Reasoning-intensive ranking models built on Large Language Models (LLMs) have made notable progress, but existing approaches often rely on large-scale LLMs and explicit Chain-of-Thought (CoT) reasoning, resulting in high computational cost and latency that limit real-world use. To address this, we propose \textbf{TFRank}, an efficient pointwise reasoning ranker based on small-scale LLMs. To improve ranking performance, TFRank effectively integrates CoT data, fine-grained score supervision, and multi-task training. Furthermore, it achieves an efficient ``\textbf{T}hink-\textbf{F}ree" reasoning capability by employing a ``think-mode switch'' and pointwise format constraints. Specifically, this allows the model to leverage explicit reasoning during training while delivering precise relevance scores for complex queries at inference without generating any reasoning chains. Experiments show that TFRank (e.g., 1.7B) achieves performance comparable to models with four times more parameters on the BRIGHT benchmark, and demonstrates strong competitiveness on the BEIR benchmark. Further analysis shows that TFRank achieves an effective balance between performance and efficiency, providing a practical solution for integrating advanced reasoning into real-world systems. Our code and data are released in the repository: https://github.com/JOHNNY-fans/TFRank.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.13500v1">LLM-Enhanced Linear Autoencoders for Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-08-19
      | 💬 Accepted by CIKM 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been widely adopted to enrich the semantic representation of textual item information in recommender systems. However, existing linear autoencoders (LAEs) that incorporate textual information rely on sparse word co-occurrence patterns, limiting their ability to capture rich textual semantics. To address this, we propose L3AE, the first integration of LLMs into the LAE framework. L3AE effectively integrates the heterogeneous knowledge of textual semantics and user-item interactions through a two-phase optimization strategy. (i) L3AE first constructs a semantic item-to-item correlation matrix from LLM-derived item representations. (ii) It then learns an item-to-item weight matrix from collaborative signals while distilling semantic item correlations as regularization. Notably, each phase of L3AE is optimized through closed-form solutions, ensuring global optimality and computational efficiency. Extensive experiments demonstrate that L3AE consistently outperforms state-of-the-art LLM-enhanced models on three benchmark datasets, achieving gains of 27.6% in Recall@20 and 39.3% in NDCG@20. The source code is available at https://github.com/jaewan7599/L3AE_CIKM2025.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02332v2">PII Jailbreaking in LLMs via Activation Steering Reveals Personal Information Leakage</a></div>
    <div class="paper-meta">
      📅 2025-08-19
      | 💬 Preprint. V2 Updated with dataset filtering, benchmarking privacy evaluator and additional latent space visualizations
    </div>
    <details class="paper-abstract">
      This paper investigates privacy jailbreaking in LLMs via steering, focusing on whether manipulating activations can bypass LLM alignment and alter response behaviors to privacy related queries (e.g., a certain public figure's sexual orientation). We begin by identifying attention heads predictive of refusal behavior for private attributes (e.g., sexual orientation) using lightweight linear probes trained with privacy evaluator labels. Next, we steer the activations of a small subset of these attention heads guided by the trained probes to induce the model to generate non-refusal responses. Our experiments show that these steered responses often disclose sensitive attribute details, along with other private information about data subjects such as life events, relationships, and personal histories that the models would typically refuse to produce. Evaluations across four LLMs reveal jailbreaking disclosure rates of at least 95%, with more than 50% on average of these responses revealing true personal information. Our controlled study demonstrates that private information memorized in LLMs can be extracted through targeted manipulation of internal activations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21447v2">LLM4VV: Evaluating Cutting-Edge LLMs for Generation and Evaluation of Directive-Based Parallel Programming Model Compiler Tests</a></div>
    <div class="paper-meta">
      📅 2025-08-19
    </div>
    <details class="paper-abstract">
      The usage of Large Language Models (LLMs) for software and test development has continued to increase since LLMs were first introduced, but only recently have the expectations of LLMs become more realistic. Verifying the correctness of code generated by LLMs is key to improving their usefulness, but there have been no comprehensive and fully autonomous solutions developed yet. Hallucinations are a major concern when LLMs are applied blindly to problems without taking the time and effort to verify their outputs, and an inability to explain the logical reasoning of LLMs leads to issues with trusting their results. To address these challenges while also aiming to effectively apply LLMs, this paper proposes a dual-LLM system (i.e. a generative LLM and a discriminative LLM) and experiments with the usage of LLMs for the generation of a large volume of compiler tests. We experimented with a number of LLMs possessing varying parameter counts and presented results using ten carefully-chosen metrics that we describe in detail in our narrative. Through our findings, it is evident that LLMs possess the promising potential to generate quality compiler tests and verify them automatically.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.12281v2">Legal$Δ$: Enhancing Legal Reasoning in LLMs via Reinforcement Learning with Chain-of-Thought Guided Information Gain</a></div>
    <div class="paper-meta">
      📅 2025-08-19
    </div>
    <details class="paper-abstract">
      Legal Artificial Intelligence (LegalAI) has achieved notable advances in automating judicial decision-making with the support of Large Language Models (LLMs). However, existing legal LLMs still struggle to generate reliable and interpretable reasoning processes. They often default to fast-thinking behavior by producing direct answers without explicit multi-step reasoning, limiting their effectiveness in complex legal scenarios that demand rigorous justification. To address this challenge, we propose Legal$\Delta$, a reinforcement learning framework designed to enhance legal reasoning through chain-of-thought guided information gain. During training, Legal$\Delta$ employs a dual-mode input setup-comprising direct answer and reasoning-augmented modes-and maximizes the information gain between them. This encourages the model to acquire meaningful reasoning patterns rather than generating superficial or redundant explanations. Legal$\Delta$ follows a two-stage approach: (1) distilling latent reasoning capabilities from a powerful Large Reasoning Model (LRM), DeepSeek-R1, and (2) refining reasoning quality via differential comparisons, combined with a multidimensional reward mechanism that assesses both structural coherence and legal-domain specificity. Experimental results on multiple legal reasoning tasks demonstrate that Legal$\Delta$ outperforms strong baselines in both accuracy and interpretability. It consistently produces more robust and trustworthy legal judgments without relying on labeled preference data. All code and data will be released at https://github.com/NEUIR/LegalDelta.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.12645v2">Diagnostic-Guided Dynamic Profile Optimization for LLM-based User Simulators in Sequential Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-08-19
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have enabled realistic user simulators for developing and evaluating recommender systems (RSs). However, existing LLM-based simulators for RSs face two major limitations: (1) static and single-step prompt-based inference that leads to inaccurate and incomplete user profile construction; (2) unrealistic and single-round recommendation-feedback interaction pattern that fails to capture real-world scenarios. To address these limitations, we propose DGDPO (Diagnostic-Guided Dynamic Profile Optimization), a novel framework that constructs user profile through a dynamic and iterative optimization process to enhance the simulation fidelity. Specifically, DGDPO incorporates two core modules within each optimization loop: firstly, a specialized LLM-based diagnostic module, calibrated through our novel training strategy, accurately identifies specific defects in the user profile. Subsequently, a generalized LLM-based treatment module analyzes the diagnosed defect and generates targeted suggestions to refine the profile. Furthermore, unlike existing LLM-based user simulators that are limited to single-round interactions, we are the first to integrate DGDPO with sequential recommenders, enabling a bidirectional evolution where user profiles and recommendation strategies adapt to each other over multi-round interactions. Extensive experiments conducted on three real-world datasets demonstrate the effectiveness of our proposed framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.11987v2">FutureX: An Advanced Live Benchmark for LLM Agents in Future Prediction</a></div>
    <div class="paper-meta">
      📅 2025-08-19
      | 💬 Technical report, 51 pages. Update the results
    </div>
    <details class="paper-abstract">
      Future prediction is a complex task for LLM agents, requiring a high level of analytical thinking, information gathering, contextual understanding, and decision-making under uncertainty. Agents must not only gather and interpret vast amounts of dynamic information but also integrate diverse data sources, weigh uncertainties, and adapt predictions based on emerging trends, just as human experts do in fields like politics, economics, and finance. Despite its importance, no large-scale benchmark exists for evaluating agents on future prediction, largely due to challenges in handling real-time updates and retrieving timely, accurate answers. To address this, we introduce $\textbf{FutureX}$, a dynamic and live evaluation benchmark specifically designed for LLM agents performing future prediction tasks. FutureX is the largest and most diverse live benchmark for future prediction, supporting real-time daily updates and eliminating data contamination through an automated pipeline for question gathering and answer collection. We evaluate 25 LLM/agent models, including those with reasoning, search capabilities, and integration of external tools such as the open-source Deep Research Agent and closed-source Deep Research models. This comprehensive evaluation assesses agents' adaptive reasoning and performance in dynamic environments. Additionally, we provide in-depth analyses of agents' failure modes and performance pitfalls in future-oriented tasks, including the vulnerability to fake web pages and the temporal validity. Our goal is to establish a dynamic, contamination-free evaluation standard that drives the development of LLM agents capable of performing at the level of professional human analysts in complex reasoning and predictive thinking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18499v3">G1: Teaching LLMs to Reason on Graphs with Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-08-19
    </div>
    <details class="paper-abstract">
      Although Large Language Models (LLMs) have demonstrated remarkable progress, their proficiency in graph-related tasks remains notably limited, hindering the development of truly general-purpose models. Previous attempts, including pretraining graph foundation models or employing supervised fine-tuning, often face challenges such as the scarcity of large-scale, universally represented graph data. We introduce G1, a simple yet effective approach demonstrating that Reinforcement Learning (RL) on synthetic graph-theoretic tasks can significantly scale LLMs' graph reasoning abilities. To enable RL training, we curate Erd\~os, the largest graph reasoning dataset to date comprising 50 diverse graph-theoretic tasks of varying difficulty levels, 100k training data and 5k test data, all drived from real-world graphs. With RL on Erd\~os, G1 obtains substantial improvements in graph reasoning, where our finetuned 3B model even outperforms Qwen2.5-72B-Instruct (24x size). RL-trained models also show strong zero-shot generalization to unseen tasks, domains, and graph encoding schemes, including other graph-theoretic benchmarks as well as real-world node classification and link prediction tasks, without compromising general reasoning abilities. Our findings offer an efficient, scalable path for building strong graph reasoners by finetuning LLMs with RL on graph-theoretic tasks, which combines the strengths of pretrained LLM capabilities with abundant, automatically generated synthetic data, suggesting that LLMs possess graph understanding abilities that RL can elicit successfully. Our implementation is open-sourced at https://github.com/PKU-ML/G1, with models and datasets hosted on Hugging Face collections https://huggingface.co/collections/PKU-ML/g1-683d659e992794fc99618cf2 for broader accessibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.13423v1">AdaptJobRec: Enhancing Conversational Career Recommendation through an LLM-Powered Agentic System</a></div>
    <div class="paper-meta">
      📅 2025-08-19
    </div>
    <details class="paper-abstract">
      In recent years, recommendation systems have evolved from providing a single list of recommendations to offering a comprehensive suite of topic focused services. To better accomplish this task, conversational recommendation systems (CRS) have progressed from basic retrieval augmented LLM generation to agentic systems with advanced reasoning and self correction capabilities. However, agentic systems come with notable response latency, a longstanding challenge for conversational recommendation systems. To balance the trade off between handling complex queries and minimizing latency, we propose AdaptJobRec, the first conversational job recommendation system that leverages autonomous agent to integrate personalized recommendation algorithm tools. The system employs a user query complexity identification mechanism to minimize response latency. For straightforward queries, the agent directly selects the appropriate tool for rapid responses. For complex queries, the agent uses the memory processing module to filter chat history for relevant content, then passes the results to the intelligent task decomposition planner, and finally executes the tasks using personalized recommendation tools. Evaluation on Walmart's real world career recommendation scenarios demonstrates that AdaptJobRec reduces average response latency by up to 53.3% compared to competitive baselines, while significantly improving recommendation accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14314v1">Zero-knowledge LLM hallucination detection and mitigation through fine-grained cross-model consistency</a></div>
    <div class="paper-meta">
      📅 2025-08-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive capabilities across diverse tasks, but they remain susceptible to hallucinations--generating content that appears plausible but contains factual inaccuracies. We present Finch-Zk, a black-box framework that leverages FINe-grained Cross-model consistency to detect and mitigate Hallucinations in LLM outputs without requiring external knowledge sources. Finch-Zk introduces two key innovations: 1) a cross-model consistency checking strategy that reveals fine-grained inaccuracies by comparing responses generated by diverse models from semantically-equivalent prompts, and 2) a targeted mitigation technique that applies precise corrections to problematic segments while preserving accurate content. Experiments on the FELM dataset show Finch-Zk improves hallucination detection F1 scores by 6-39\% compared to existing approaches. For mitigation, Finch-Zk achieves 7-8 absolute percentage points improvement in answer accuracy on the GPQA-diamond dataset when applied to state-of-the-art models like Llama 4 Maverick and Claude 4 Sonnet. Extensive evaluation across multiple models demonstrates that Finch-Zk provides a practical, deployment-ready safeguard for enhancing factual reliability in production LLM systems.
    </details>
</div>
