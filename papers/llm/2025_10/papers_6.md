# llm - 2025_10

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- Part 6
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04498v1">GenQuest: An LLM-based Text Adventure Game for Language Learners</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 Workshop on Wordplay: When Language Meets Games, EMNLP 2025
    </div>
    <details class="paper-abstract">
      GenQuest is a generative text adventure game that leverages Large Language Models (LLMs) to facilitate second language learning through immersive, interactive storytelling. The system engages English as a Foreign Language (EFL) learners in a collaborative "choose-your-own-adventure" style narrative, dynamically generated in response to learner choices. Game mechanics such as branching decision points and story milestones are incorporated to maintain narrative coherence while allowing learner-driven plot development. Key pedagogical features include content generation tailored to each learner's proficiency level, and a vocabulary assistant that provides in-context explanations of learner-queried text strings, ranging from words and phrases to sentences. Findings from a pilot study with university EFL students in China indicate promising vocabulary gains and positive user perceptions. Also discussed are suggestions from participants regarding the narrative length and quality, and the request for multi-modal content such as illustrations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03691v2">LogSage: An LLM-Based Framework for CI/CD Failure Detection and Remediation with Industrial Validation</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 12 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Continuous Integration and Deployment (CI/CD) pipelines are critical to modern software engineering, yet diagnosing and resolving their failures remains complex and labor-intensive. We present LogSage, the first end-to-end LLM-powered framework for root cause analysis (RCA) and automated remediation of CI/CD failures. LogSage employs a token-efficient log preprocessing pipeline to filter noise and extract critical errors, then performs structured diagnostic prompting for accurate RCA. For solution generation, it leverages retrieval-augmented generation (RAG) to reuse historical fixes and invokes automation fixes via LLM tool-calling. On a newly curated benchmark of 367 GitHub CI/CD failures, LogSage achieves over 98\% precision, near-perfect recall, and an F1 improvement of more than 38\% points in the RCA stage, compared with recent LLM-based baselines. In a year-long industrial deployment at ByteDance, it processed over 1.07M executions, with end-to-end precision exceeding 80\%. These results demonstrate that LogSage provides a scalable and practical solution for automating CI/CD failure management in real-world DevOps workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21815v2">Scientific Paper Retrieval with LLM-Guided Semantic-Based Ranking</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 Accepted to EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Scientific paper retrieval is essential for supporting literature discovery and research. While dense retrieval methods demonstrate effectiveness in general-purpose tasks, they often fail to capture fine-grained scientific concepts that are essential for accurate understanding of scientific queries. Recent studies also use large language models (LLMs) for query understanding; however, these methods often lack grounding in corpus-specific knowledge and may generate unreliable or unfaithful content. To overcome these limitations, we propose SemRank, an effective and efficient paper retrieval framework that combines LLM-guided query understanding with a concept-based semantic index. Each paper is indexed using multi-granular scientific concepts, including general research topics and detailed key phrases. At query time, an LLM identifies core concepts derived from the corpus to explicitly capture the query's information need. These identified concepts enable precise semantic matching, significantly enhancing retrieval accuracy. Experiments show that SemRank consistently improves the performance of various base retrievers, surpasses strong existing LLM-based baselines, and remains highly efficient.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04488v1">Multi-Agent Collaborative Intelligence: Dual-Dial Control for Reliable LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 27 pages, 5 figures, 21 tables
    </div>
    <details class="paper-abstract">
      Multi-agent debate often wastes compute by using a fixed adversarial stance, aggregating without deliberation, or stopping on heuristics. We introduce MACI, an active controller with two independent dials that decouple information from behavior: an information dial that gates evidence by quality, and a behavior dial that schedules contentiousness from exploration to consolidation. A moderator tracks disagreement, overlap, evidence quality, and argument quality, and halts when gains plateau. We provide theory-lite guarantees for nonincreasing dispersion and provable termination, with a budget-feasible scheduler. Across clinical diagnosis and news-bias tasks, MACI improves accuracy and calibration while reducing tokens, and converts residual uncertainty into precision RAG plans that specify what to retrieve next. We use a cross-family LLM judge (CRIT) as a conservative soft weight and stop signal, validated for order invariance and judge-swap stability; stability depends on using high-capability judges. MACI turns debate into a budget-aware, measurable, and provably terminating controller.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04484v1">Psychological Steering in LLMs: An Evaluation of Effectiveness and Trustworthiness</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 Submitted to ARR - October 2025
    </div>
    <details class="paper-abstract">
      The ability to control LLMs' emulated emotional states and personality traits is essential for enabling rich, human-centered interactions in socially interactive settings. We introduce PsySET, a Psychologically-informed benchmark to evaluate LLM Steering Effectiveness and Trustworthiness across the emotion and personality domains. Our study spans four models from different LLM families paired with various steering strategies, including prompting, fine-tuning, and representation engineering. Our results indicate that prompting is consistently effective but limited in intensity control, whereas vector injections achieve finer controllability while slightly reducing output quality. Moreover, we explore the trustworthiness of steered LLMs by assessing safety, truthfulness, fairness, and ethics, highlighting potential side effects and behavioral shifts. Notably, we observe idiosyncratic effects; for instance, even a positive emotion like joy can degrade robustness to adversarial factuality, lower privacy awareness, and increase preferential bias. Meanwhile, anger predictably elevates toxicity yet strengthens leakage resistance. Our framework establishes the first holistic evaluation of emotion and personality steering, offering insights into its interpretability and reliability for socially interactive applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04724v2">Who's the Mole? Modeling and Detecting Intention-Hiding Malicious Agents in LLM-Based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Multi-agent systems powered by Large Language Models (LLM-MAS) have demonstrated remarkable capabilities in collaborative problem-solving. However, their deployment also introduces new security risks. Existing research on LLM-based agents has primarily examined single-agent scenarios, while the security of multi-agent systems remains largely unexplored. To address this gap, we present a systematic study of intention-hiding threats in LLM-MAS. We design four representative attack paradigms that subtly disrupt task completion while maintaining a high degree of stealth, and evaluate them under centralized, decentralized, and layered communication structures. Experimental results show that these attacks are highly disruptive and can easily evade existing defense mechanisms. To counter these threats, we propose AgentXposed, a psychology-inspired detection framework. AgentXposed draws on the HEXACO personality model, which characterizes agents through psychological trait dimensions, and the Reid interrogation technique, a structured method for eliciting concealed intentions. By combining progressive questionnaire probing with behavior-based inter-agent monitoring, the framework enables the proactive identification of malicious agents before harmful actions are carried out. Extensive experiments across six datasets against both our proposed attacks and two baseline threats demonstrate that AgentXposed effectively detects diverse forms of malicious behavior, achieving strong robustness across multiple communication settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06698v3">SCAN: Structured Capability Assessment and Navigation for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Evaluating Large Language Models (LLMs) has become increasingly important, with automatic evaluation benchmarks gaining prominence as alternatives to human evaluation. While existing research has focused on approximating model rankings, such benchmarks fail to provide users and developers with a comprehensive and fine-grained understanding of a specific model's capabilities. To fill this gap, we propose \textbf{SCAN} (Structured Capability Assessment and Navigation), a practical framework that enables detailed characterization of LLM capabilities through comprehensive and fine-grained evaluation. SCAN incorporates four key components: (1) TaxBuilder, which extracts capability-indicating tags from extensive queries to construct a hierarchical taxonomy automatically; (2) RealMix, a query synthesis and filtering mechanism that ensures sufficient evaluation data for each capability tag; (3) a suite of visualization and analysis tools that facilitate efficient navigation and analysis of model capabilities; and (4) a PC$^2$-based (Pre-Comparison-derived Criteria) LLM-as-a-Judge approach that achieves significantly higher accuracy compared to classic LLM-as-a-Judge method. Using SCAN, we conduct a comprehensive evaluation of 21 mainstream LLMs. Our detailed analysis of the GPT-OSS family reveals substantial performance variations, even within sub-capabilities belonging to the same category of capability. This finding highlights the importance of fine-grained evaluation in accurately understanding LLM behavior. Project homepage and resources are available at \href{https://liudan193.github.io/Feedbacker/}{https://liudan193.github.io/Feedbacker/}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18562v2">From Word to World: Evaluate and Mitigate Culture Bias in LLMs via Word Association Test</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 Cultural Analysis, Cultural Alignment, Word Association Test, Large Language Models. Accepted by EMNLP 2025 (Oral)
    </div>
    <details class="paper-abstract">
      The human-centered word association test (WAT) serves as a cognitive proxy, revealing sociocultural variations through culturally shared semantic expectations and implicit linguistic patterns shaped by lived experiences. We extend this test into an LLM-adaptive, free-relation task to assess the alignment of large language models (LLMs) with cross-cultural cognition. To address culture preference, we propose CultureSteer, an innovative approach that moves beyond superficial cultural prompting by embedding cultural-specific semantic associations directly within the model's internal representation space. Experiments show that current LLMs exhibit significant bias toward Western (notably American) schemas at the word association level. In contrast, our model substantially improves cross-cultural alignment, capturing diverse semantic associations. Further validation on culture-sensitive downstream tasks confirms its efficacy in fostering cognitive alignment across cultures. This work contributes a novel methodological paradigm for enhancing cultural awareness in LLMs, advancing the development of more inclusive language technologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04465v1">Autonomy Matters: A Study on Personalization-Privacy Dilemma in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents require personal information for personalization in order to better act on users' behalf in daily tasks, but this raises privacy concerns and a personalization-privacy dilemma. Agent's autonomy introduces both risks and opportunities, yet its effects remain unclear. To better understand this, we conducted a 3$\times$3 between-subjects experiment ($N=450$) to study how agent's autonomy level and personalization influence users' privacy concerns, trust and willingness to use, as well as the underlying psychological processes. We find that personalization without considering users' privacy preferences increases privacy concerns and decreases trust and willingness to use. Autonomy moderates these effects: Intermediate autonomy flattens the impact of personalization compared to No- and Full autonomy conditions. Our results suggest that rather than aiming for perfect model alignment in output generation, balancing autonomy of agent's action and user control offers a promising path to mitigate the personalization-privacy dilemma.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04463v1">Evaluating Self-Supervised Speech Models via Text-Based LLMS</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 Accepted to ASRU 2025
    </div>
    <details class="paper-abstract">
      Self-Supervised Learning (SSL) has gained traction for its ability to learn rich representations with low labeling costs, applicable across diverse downstream tasks. However, assessing the downstream-task performance remains challenging due to the cost of extra training and evaluation. Existing methods for task-agnostic evaluation also require extra training or hyperparameter tuning. We propose a novel evaluation metric using large language models (LLMs). By inputting discrete token sequences and minimal domain cues derived from SSL models into LLMs, we obtain the mean log-likelihood; these cues guide in-context learning, rendering the score more reliable without extra training or hyperparameter tuning. Experimental results show a correlation between LLM-based scores and automatic speech recognition task. Additionally, our findings reveal that LLMs not only functions as an SSL evaluation tools but also provides inference-time embeddings that are useful for speaker verification task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02657v2">Less LLM, More Documents: Searching for Improved RAG</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) couples document retrieval with large language models (LLMs). While scaling generators improves accuracy, it also raises cost and limits deployability. We explore an orthogonal axis: enlarging the retriever's corpus to reduce reliance on large LLMs. Experimental results show that corpus scaling consistently strengthens RAG and can often serve as a substitute for increasing model size, though with diminishing returns at larger scales. Small- and mid-sized generators paired with larger corpora often rival much larger models with smaller corpora; mid-sized models tend to gain the most, while tiny and large models benefit less. Our analysis shows that improvements arise primarily from increased coverage of answer-bearing passages, while utilization efficiency remains largely unchanged. These findings establish a principled corpus-generator trade-off: investing in larger corpora offers an effective path to stronger RAG, often comparable to enlarging the LLM itself.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03195v2">Can LLMs Hit Moving Targets? Tracking Evolving Signals in Corporate Disclosures</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 8 pages, 5 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Moving targets -- managers' strategic shifting of key performance metrics when the original targets become difficult to achieve -- have been shown to predict subsequent stock underperformance. However, our work reveals that the method employed in that study exhibits two key limitations that hinder the accuracy -- noise in the extracted targets and loss of contextual information -- both of which stem primarily from the use of a named entity recognition (NER). To address these two limitations, we propose an LLM-based target extraction method with a newly defined metric that better captures semantic context. This approach preserves semantic context beyond simple entity recognition and yields consistently higher predictive power than the original approach. Overall, our approach enhances the granularity and accuracy of financial text-based performance prediction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04439v1">On the Role of Unobserved Sequences on Sample-based Uncertainty Quantification for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 Accepted to UncertaiNLP workshop of EMNLP 2025
    </div>
    <details class="paper-abstract">
      Quantifying uncertainty in large language models (LLMs) is important for safety-critical applications because it helps spot incorrect answers, known as hallucinations. One major trend of uncertainty quantification methods is based on estimating the entropy of the distribution of the LLM's potential output sequences. This estimation is based on a set of output sequences and associated probabilities obtained by querying the LLM several times. In this paper, we advocate and experimentally show that the probability of unobserved sequences plays a crucial role, and we recommend future research to integrate it to enhance such LLM uncertainty quantification methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05449v1">Bloom: Designing for LLM-Augmented Behavior Change Interactions</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) offer novel opportunities to support health behavior change, yet existing work has narrowly focused on text-only interactions. Building on decades of HCI research demonstrating the effectiveness of UI-based interactions, we present Bloom, an application for physical activity promotion that integrates an LLM-based health coaching chatbot with established UI-based interactions. As part of Bloom's development, we conducted a redteaming evaluation and contribute a safety benchmark dataset. In a four-week randomized field study (N=54) comparing Bloom to a non-LLM control, we observed important shifts in psychological outcomes: participants in the LLM condition reported stronger beliefs that activity was beneficial, greater enjoyment, and more self-compassion. Both conditions significantly increased physical activity levels, doubling the proportion of participants meeting recommended weekly guidelines, though we observed no significant differences between conditions. Instead, our findings suggest that LLMs may be more effective at shifting mindsets that precede longer-term behavior change.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05445v1">AgentRouter: A Knowledge-Graph-Guided LLM Router for Collaborative Multi-Agent Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) and agent-based frameworks have advanced rapidly, enabling diverse applications. Yet, with the proliferation of models and agentic strategies, practitioners face substantial uncertainty in selecting the best configuration for a downstream task. Prior studies show that different agents and backbones exhibit complementary strengths, and that larger models are not always superior, underscoring the need for adaptive routing mechanisms. Existing approaches to agent routing, however, often emphasize cost efficiency while overlooking the fine-grained contextual and relational structure inherent in QA tasks. In this paper, we propose tAgentRouter, a framework that formulates multi-agent QA as a knowledge-graph-guided routing problem supervised by empirical performance signals. Specifically, we convert QA instance into a knowledge graph that jointly encodes queries, contextual entities, and agents, and then train a heterogeneous graph neural network (GNN) to propagate information across node types and produce task-aware routing distributions over agents. By leveraging soft supervision and weighted aggregation of agent outputs, AgentRouter learns principled collaboration schemes that capture the complementary strengths of diverse agents. Extensive experiments demonstrate that our framework consistently outperforms single-agent and ensemble baselines, while generalizing across benchmarks and LLM backbones. These results highlight the effectiveness and robustness of graph-supervised multi-agent routing for question answering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04365v4">AutoPDL: Automatic Prompt Optimization for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 Presented at AutoML 2025 (Methods Track); to be published in proceedings
    </div>
    <details class="paper-abstract">
      The performance of large language models (LLMs) depends on how they are prompted, with choices spanning both the high-level prompting pattern (e.g., Zero-Shot, CoT, ReAct, ReWOO) and the specific prompt content (instructions and few-shot demonstrations). Manually tuning this combination is tedious, error-prone, and specific to a given LLM and task. Therefore, this paper proposes AutoPDL, an automated approach to discovering good LLM agent configurations. Our approach frames this as a structured AutoML problem over a combinatorial space of agentic and non-agentic prompting patterns and demonstrations, using successive halving to efficiently navigate this space. We introduce a library implementing common prompting patterns using the PDL prompt programming language. AutoPDL solutions are human-readable, editable, and executable PDL programs that use this library. This approach also enables source-to-source optimization, allowing human-in-the-loop refinement and reuse. Evaluations across three tasks and seven LLMs (ranging from 3B to 70B parameters) show consistent accuracy gains ($9.21\pm15.46$ percentage points), up to 67.5pp, and reveal that selected prompting strategies vary across models and tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05431v1">Self-Filtered Distillation with LLMs-generated Trust Indicators for Reliable Patent Classification</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly generate natural language rationales to enhance interpretability, but these often contain logical errors, label mismatches, and domain-specific misalignments. Directly using such rationales as supervision risks propagating noise and undermining training stability. To address this challenge, we introduce Self-Filtered Distillation, a framework specifically tailored for patent classification, which treats LLM-generated rationales as trust signals rather than ground-truth supervision. The framework employs selective distillation guided by three unsupervised trust metrics: (1) Self-Consistency, which measures the stability of LLM-generated rationales across multiple generations; (2) Class Entailment Alignment, which assesses semantic coherence with patent-specific class definitions; and (3) LLM Agreement Scoring, which validates rationale-label plausibility. These metrics are integrated into a unified trust score that primarily weights training samples while optionally filtering out extremely low-trust cases, enabling reasoning-aware supervision. Experiments on the USPTO-2M dataset, a widely used benchmark for patent classification, show that our method outperforms label-based learning and conventional distillation in accuracy, stability, and interpretability, establishing a reliable paradigm for leveraging reasoning-aware trust indicators in patent analytics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09389v2">Measuring LLM Novelty As The Frontier Of Original And High-Quality Output</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 Updated results with higher coverage of open-data models and better quality judgments
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly used for ideation and scientific discovery, it is important to evaluate their ability to generate novel output. Prior work evaluates novelty as originality with respect to model training data, but original outputs may be of low quality. In contrast, non-expert judges more reliably score quality but may favor memorized outputs, limiting the reliability of human preference as a metric. We introduce a new novelty metric for LLM generations that balances originality and quality -- the harmonic mean of the fraction of \ngrams unseen during training and a task-specific quality score. Using this framework, we identify trends that affect the novelty of generations from three families of open-data models (OLMo, OLMo-2, and Pythia) on three creative tasks: story completion, poetry writing, and creative tool use. We find that model-generated text from some base LLMs is less novel than human-written text from the internet. However, increasing model scale and post-training reliably improves novelty due to improvements in output quality. We also find that improving the base model at the same scale (\eg OLMo 7B to OLMo-2 7B) leads to higher novelty due to higher originality. Finally, we observe that inference-time methods, such as prompting and providing novel in-context examples, have a much smaller effect on novelty, often increasing originality at the expense of quality. This highlights the need for further research into more effective elicitation strategies as we use models for creative applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13082v3">Discerning What Matters: A Multi-Dimensional Assessment of Moral Competence in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Moral competence is the ability to act in accordance with moral principles. As large language models (LLMs) are increasingly deployed in situations demanding moral competence, there is increasing interest in evaluating this ability empirically. We review existing literature and identify three significant shortcoming: (i) Over-reliance on prepackaged moral scenarios with explicitly highlighted moral features; (ii) Focus on verdict prediction rather than moral reasoning; and (iii) Inadequate testing of models' (in)ability to recognize when additional information is needed. Grounded in philosophical research on moral skill, we then introduce a novel method for assessing moral competence in LLMs. Our approach moves beyond simple verdict comparisons to evaluate five dimensions of moral competence: identifying morally relevant features, weighting their importance, assigning moral reasons to these features, synthesizing coherent moral judgments, and recognizing information gaps. We conduct two experiments comparing six leading LLMs against non-expert humans and professional philosophers. In our first experiment using ethical vignettes standard to existing work, LLMs generally outperformed non-expert humans across multiple dimensions of moral reasoning. However, our second experiment, featuring novel scenarios designed to test moral sensitivity by embedding relevant features among irrelevant details, revealed a striking reversal: several LLMs performed significantly worse than humans. Our findings suggest that current evaluations may substantially overestimate LLMs' moral reasoning capabilities by eliminating the task of discerning moral relevance from noisy information, which we take to be a prerequisite for genuine moral skill. This work provides a more nuanced framework for assessing AI moral competence and highlights important directions for improving moral competence in advanced AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.09384v3">Generative transformations and patterns in LLM-native approaches for software verification and falsification</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      The emergence of prompting as the dominant paradigm for leveraging Large Language Models (LLMs) has led to a proliferation of LLM-native software, where application behavior arises from complex, stochastic data transformations. However, the engineering of such systems remains largely exploratory and ad-hoc, hampered by the absence of conceptual frameworks, ex-ante methodologies, design guidelines, and specialized benchmarks. We argue that a foundational step towards a more disciplined engineering practice is a systematic understanding of the core functional units--generative transformations--and their compositional patterns within LLM-native applications. Focusing on the rich domain of software verification and falsification, we conduct a secondary study of over 100 research proposals to address this gap. We first present a fine-grained taxonomy of generative transformations, abstracting prompt-based interactions into conceptual signatures. This taxonomy serves as a scaffolding to identify recurrent transformation relationship patterns--analogous to software design patterns--that characterize solution approaches in the literature. Our analysis not only validates the utility of the taxonomy but also surfaces strategic gaps and cross-dimensional relationships, offering a structured foundation for future research in modular and compositional LLM application design, benchmarking, and the development of reliable LLM-native systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06299v3">How Malicious AI Swarms Can Threaten Democracy: The Fusion of Agentic AI and LLMs Marks a New Frontier in Information Warfare</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 15 pages, 1 figure
    </div>
    <details class="paper-abstract">
      Public opinion manipulation has entered a new phase, amplifying its roots in rhetoric and propaganda. Advances in large language models (LLMs) and autonomous agents now let influence campaigns reach unprecedented scale and precision. Researchers warn AI could foster mass manipulation. Generative tools can expand propaganda output without sacrificing credibility and inexpensively create election falsehoods that are rated as more human-like than those written by humans. Techniques meant to refine AI reasoning, such as chain-of-thought prompting, can just as effectively be used to generate more convincing falsehoods. Enabled by these capabilities, another disruptive threat is emerging: swarms of collaborative, malicious AI agents. Fusing LLM reasoning with multi-agent architectures, these systems are capable of coordinating autonomously, infiltrating communities, and fabricating consensus cheaply. By adaptively mimicking human social dynamics, they threaten democracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05381v1">Context Length Alone Hurts LLM Performance Despite Perfect Retrieval</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 18 pages (9 pages of main content), 5 figures, accepted at the Findings of EMNLP 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often fail to scale their performance on long-context tasks performance in line with the context lengths they support. This gap is commonly attributed to retrieval failures -- the models' inability to identify relevant information in the long inputs. Accordingly, recent efforts often focus on evaluating and improving LLMs' retrieval performance: if retrieval is perfect, a model should, in principle, perform just as well on a long input as it does on a short one -- or should it? This paper presents findings that the answer to this question may be negative. Our systematic experiments across 5 open- and closed-source LLMs on math, question answering, and coding tasks reveal that, even when models can perfectly retrieve all relevant information, their performance still degrades substantially (13.9%--85%) as input length increases but remains well within the models' claimed lengths. This failure occurs even when the irrelevant tokens are replaced with minimally distracting whitespace, and, more surprisingly, when they are all masked and the models are forced to attend only to the relevant tokens. A similar performance drop is observed when all relevant evidence is placed immediately before the question. Our findings reveal a previously-unrealized limitation: the sheer length of the input alone can hurt LLM performance, independent of retrieval quality and without any distraction. They motivate our simple, model-agnostic mitigation strategy that transforms a long-context task into a short-context one by prompting the model to recite the retrieved evidence before attempting to solve the problem. On RULER, we observe a consistent improvement of GPT-4o up to 4% on an already strong baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.06711v3">MathVC: An LLM-Simulated Multi-Character Virtual Classroom for Mathematics Education</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 Accepted by AAAI 2025 workshop
    </div>
    <details class="paper-abstract">
      Collaborative problem solving (CPS) is essential in mathematics education, fostering deeper learning through the exchange of ideas. Yet, classrooms often lack the resources, time, and peer dynamics needed to sustain productive CPS. Recent advancements in Large Language Models (LLMs) offer a promising avenue to enhance CPS in mathematical education. We designed and developed MathVC, a multi-persona LLM simulated virtual classroom platform to facilitate CPS in mathematics. MathVC combines a meta planning controller that monitors CPS stages-sense-making, team organization, planning, execution, validation, and predicts the next speaker, with a persona simulation stack that encodes mathematical thinking via a task schema and error-injected persona schemas seeded from teacher-specified misconceptions. We evaluated MathVC with 14 U.S. middle schoolers. Students reported constructive interaction and reaching shared solutions, describing gains in engagement, motivation, and confidence through diverse perspectives, immediate scaffolding, and human-like fallibility. Our findings also provide insights into simulating peers via LLM-based technologies for collaboration to support learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05291v1">Camellia: Benchmarking Cultural Biases in LLMs for Asian Languages</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) gain stronger multilingual capabilities, their ability to handle culturally diverse entities becomes crucial. Prior work has shown that LLMs often favor Western-associated entities in Arabic, raising concerns about cultural fairness. Due to the lack of multilingual benchmarks, it remains unclear if such biases also manifest in different non-Western languages. In this paper, we introduce Camellia, a benchmark for measuring entity-centric cultural biases in nine Asian languages spanning six distinct Asian cultures. Camellia includes 19,530 entities manually annotated for association with the specific Asian or Western culture, as well as 2,173 naturally occurring masked contexts for entities derived from social media posts. Using Camellia, we evaluate cultural biases in four recent multilingual LLM families across various tasks such as cultural context adaptation, sentiment association, and entity extractive QA. Our analyses show a struggle by LLMs at cultural adaptation in all Asian languages, with performance differing across models developed in regions with varying access to culturally-relevant data. We further observe that different LLM families hold their distinct biases, differing in how they associate cultures with particular sentiments. Lastly, we find that LLMs struggle with context understanding in Asian languages, creating performance gaps between cultures in entity extraction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16366v4">A Generative Approach to LLM Harmfulness Mitigation with Red Flag Tokens</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 15 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Many safety post-training methods for large language models (LLMs) are designed to modify the model's behaviour from producing unsafe answers to issuing refusals. However, such distribution shifts are often brittle and degrade performance on desirable tasks. To address these pitfalls, we propose augmenting the model's vocabulary with a special red flag token, and training the model to insert this token whenever harmful content is generated or imminent. This approach enables the model to explicitly learn the concept of harmfulness in its representations, with minimal impact on utility due to the marginal change in the generated distribution of natural language. Moreover, because the token is embedded in the model's vocabulary, we can naturally leverage the LLMs' generalization capabilities, such as in-context learning (ICL) and out-of-distribution generalization to languages that are not formally supported (e.g., Japanese for Llama3). In particular, we demonstrate that through ICL alone, the model can learn to initiate reflective reasoning upon generating the red flag token at inference, which steers the response away from harmful continuations or enables self-correction when the flag is raised falsely. This approach is orthogonal and complementary to existing safety technique (such as safety classifiers or standard safety training) and easier to evaluate in comparison to natural language refusals, as it does not require a human or automated judge to assess the harmlessness of the answers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05188v1">Plug-and-Play Dramaturge: A Divide-and-Conquer Approach for Iterative Narrative Script Refinement via Collaborative LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Although LLMs have been widely adopted for creative content generation, a single-pass process often struggles to produce high-quality long narratives. How to effectively revise and improve long narrative scripts like scriptwriters remains a significant challenge, as it demands a comprehensive understanding of the entire context to identify global structural issues and local detailed flaws, as well as coordinating revisions at multiple granularities and locations. Direct modifications by LLMs typically introduce inconsistencies between local edits and the overall narrative requirements. To address these issues, we propose Dramaturge, a task and feature oriented divide-and-conquer approach powered by hierarchical multiple LLM agents. It consists of a Global Review stage to grasp the overall storyline and structural issues, a Scene-level Review stage to pinpoint detailed scene and sentence flaws, and a Hierarchical Coordinated Revision stage that coordinates and integrates structural and detailed improvements throughout the script. The top-down task flow ensures that high-level strategies guide local modifications, maintaining contextual consistency. The review and revision workflow follows a coarse-to-fine iterative process, continuing through multiple rounds until no further substantive improvements can be made. Comprehensive experiments show that Dramaturge significantly outperforms all baselines in terms of script-level overall quality and scene-level details. Our approach is plug-and-play and can be easily integrated into existing methods to improve the generated scripts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05186v1">OptPipe: Memory- and Scheduling-Optimized Pipeline Parallelism for LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 Use Mathematical Programming to model Pipeline Parallelism with Offloading to balance efficiency and memory requirement
    </div>
    <details class="paper-abstract">
      Pipeline parallelism (PP) has become a standard technique for scaling large language model (LLM) training across multiple devices. However, despite recent progress in reducing memory consumption through activation offloading, existing approaches remain largely heuristic and coarse-grained, often overlooking the fine-grained trade-offs between memory, computation, and scheduling latency. In this work, we revisit the pipeline scheduling problem from a principled optimization perspective. We observe that prevailing strategies either rely on static rules or aggressively offload activations without fully leveraging the interaction between memory constraints and scheduling efficiency. To address this, we formulate scheduling as a constrained optimization problem that jointly accounts for memory capacity, activation reuse, and pipeline bubble minimization. Solving this model yields fine-grained schedules that reduce pipeline bubbles while adhering to strict memory budgets. Our approach complements existing offloading techniques: whereas prior approaches trade memory for time in a fixed pattern, we dynamically optimize the tradeoff with respect to model structure and hardware configuration. Experimental results demonstrate that our method consistently improves both throughput and memory utilization. In particular, we reduce idle pipeline time by up to 50% under the same per-device memory limit, and in some cases, enable the training of larger models within limited memory budgets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01928v3">MALT: Improving Reasoning with Multi-Agent LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 Published at COLM 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often produce answers with a single chain-of-thought, which restricts their ability to explore reasoning paths or self-correct flawed outputs in complex tasks. In this paper, we introduce MALT (Multi-Agent LLM Training), a novel post-training strategy that divides the reasoning process into generation, verification, and refinement steps using a sequential pipeline of heterogeneous agents. During data generation, each agent is repeatedly sampled to form a multi-agent search tree, where final outputs are graded against ground-truth data. We then apply value iteration to propagate reward signals back to each role-conditioned model, automatically producing multi-agent post-training data without human or teacher-model supervision. Our off-policy approach allows each agent to specialize by learning from correct and incorrect trajectories, ultimately improving the end-to-end reasoning chain. On MATH, GSM8K, and CSQA, MALT surpasses the same baseline LLM with a relative improvement of 15.66%, 7.42%, and 9.40% respectively, making it an important advance towards multi-agent cooperative training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05087v1">TeachLM: Post-Training LLMs for Education Using Authentic Learning Data</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 28 pages, 9 figures
    </div>
    <details class="paper-abstract">
      The promise of generative AI to revolutionize education is constrained by the pedagogical limits of large language models (LLMs). A major issue is the lack of access to high-quality training data that reflect the learning of actual students. Prompt engineering has emerged as a stopgap, but the ability of prompts to encode complex pedagogical strategies in rule-based natural language is inherently limited. To address this gap we introduce TeachLM - an LLM optimized for teaching through parameter-efficient fine-tuning of state-of-the-art models. TeachLM is trained on a dataset comprised of 100,000 hours of one-on-one, longitudinal student-tutor interactions maintained by Polygence, which underwent a rigorous anonymization process to protect privacy. We use parameter-efficient fine-tuning to develop an authentic student model that enables the generation of high-fidelity synthetic student-tutor dialogues. Building on this capability, we propose a novel multi-turn evaluation protocol that leverages synthetic dialogue generation to provide fast, scalable, and reproducible assessments of the dialogical capabilities of LLMs. Our evaluations demonstrate that fine-tuning on authentic learning data significantly improves conversational and pedagogical performance - doubling student talk time, improving questioning style, increasing dialogue turns by 50%, and greater personalization of instruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05069v1">SwiReasoning: Switch-Thinking in Latent and Explicit for Pareto-Superior Reasoning LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 Code: https://github.com/sdc17/SwiReasoning, Website: https://swireasoning.github.io/
    </div>
    <details class="paper-abstract">
      Recent work shows that, beyond discrete reasoning through explicit chain-of-thought steps, which are limited by the boundaries of natural languages, large language models (LLMs) can also reason continuously in latent space, allowing richer information per step and thereby improving token efficiency. Despite this promise, latent reasoning still faces two challenges, especially in training-free settings: 1) purely latent reasoning broadens the search distribution by maintaining multiple implicit paths, which diffuses probability mass, introduces noise, and impedes convergence to a single high-confidence solution, thereby hurting accuracy; and 2) overthinking persists even without explicit text, wasting tokens and degrading efficiency. To address these issues, we introduce SwiReasoning, a training-free framework for LLM reasoning which features two key innovations: 1) SwiReasoning dynamically switches between explicit and latent reasoning, guided by block-wise confidence estimated from entropy trends in next-token distributions, to balance exploration and exploitation and promote timely convergence. 2) By limiting the maximum number of thinking-block switches, SwiReasoning curbs overthinking and improves token efficiency across varying problem difficulties. On widely used mathematics and STEM benchmarks, SwiReasoning consistently improves average accuracy by 1.5%-2.8% across reasoning LLMs of different model families and scales. Furthermore, under constrained budgets, SwiReasoning improves average token efficiency by 56%-79%, with larger gains as budgets tighten.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05052v1">Proactive defense against LLM Jailbreak</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      The proliferation of powerful large language models (LLMs) has necessitated robust safety alignment, yet these models remain vulnerable to evolving adversarial attacks, including multi-turn jailbreaks that iteratively search for successful queries. Current defenses, primarily reactive and static, often fail to counter these search-based attacks. In this paper, we introduce ProAct, a novel proactive defense framework designed to disrupt and mislead autonomous jailbreaking processes. Our core idea is to intentionally provide adversaries with "spurious responses" that appear to be results of successful jailbreak attacks but contain no actual harmful content. These misleading responses provide false signals to the attacker's internal optimization loop, causing the adversarial search to terminate prematurely and effectively jailbreaking the jailbreak. By conducting extensive experiments across state-of-the-art LLMs, jailbreaking frameworks, and safety benchmarks, our method consistently and significantly reduces attack success rates by up to 92\%. When combined with other defense frameworks, it further reduces the success rate of the latest attack strategies to 0\%. ProAct represents an orthogonal defense strategy that can serve as an additional guardrail to enhance LLM safety against the most effective jailbreaking attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12491v3">Insights from the Inverse: Reconstructing LLM Training Goals Through Inverse Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 Published as a conference paper at COLM 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) trained with Reinforcement Learning from Human Feedback (RLHF) have demonstrated remarkable capabilities, but their underlying reward functions and decision-making processes remain opaque. This paper introduces a novel approach to interpreting LLMs by applying inverse reinforcement learning (IRL) to recover their implicit reward functions. We conduct experiments on toxicity-aligned LLMs of varying sizes, extracting reward models that achieve up to 85% accuracy in predicting human preferences. Our analysis reveals key insights into the non-identifiability of reward functions, the relationship between model size and interpretability, and potential pitfalls in the RLHF process. We demonstrate that IRL-derived reward models can be used to fine-tune new LLMs, resulting in comparable or improved performance on toxicity benchmarks. This work provides a new lens for understanding and improving LLM alignment, with implications for the responsible development and deployment of these powerful systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05040v1">Test-Time Scaling in Diffusion LLMs via Hidden Semi-Autoregressive Experts</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Diffusion-based large language models (dLLMs) are trained flexibly to model extreme dependence in the data distribution; however, how to best utilize this information at inference time remains an open problem. In this work, we uncover an interesting property of these models: dLLMs trained on textual data implicitly learn a mixture of semi-autoregressive experts, where different generation orders reveal different specialized behaviors. We show that committing to any single, fixed inference time schedule, a common practice, collapses performance by failing to leverage this latent ensemble. To address this, we introduce HEX (Hidden semiautoregressive EXperts for test-time scaling), a training-free inference method that ensembles across heterogeneous block schedules. By doing a majority vote over diverse block-sized generation paths, HEX robustly avoids failure modes associated with any single fixed schedule. On reasoning benchmarks such as GSM8K, it boosts accuracy by up to 3.56X (from 24.72% to 88.10%), outperforming top-K margin inference and specialized fine-tuned methods like GRPO, without additional training. HEX even yields significant gains on MATH benchmark from 16.40% to 40.00%, scientific reasoning on ARC-C from 54.18% to 87.80%, and TruthfulQA from 28.36% to 57.46%. Our results establish a new paradigm for test-time scaling in diffusion-based LLMs (dLLMs), revealing that the sequence in which masking is performed plays a critical role in determining performance during inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.08825v2">Large Language Model Hacking: Quantifying the Hidden Risks of Using LLMs for Text Annotation</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Large language models are rapidly transforming social science research by enabling the automation of labor-intensive tasks like data annotation and text analysis. However, LLM outputs vary significantly depending on the implementation choices made by researchers (e.g., model selection or prompting strategy). Such variation can introduce systematic biases and random errors, which propagate to downstream analyses and cause Type I (false positive), Type II (false negative), Type S (wrong sign), or Type M (exaggerated effect) errors. We call this phenomenon where configuration choices lead to incorrect conclusions LLM hacking. We find that intentional LLM hacking is strikingly simple. By replicating 37 data annotation tasks from 21 published social science studies, we show that, with just a handful of prompt paraphrases, virtually anything can be presented as statistically significant. Beyond intentional manipulation, our analysis of 13 million labels from 18 different LLMs across 2361 realistic hypotheses shows that there is also a high risk of accidental LLM hacking, even when following standard research practices. We find incorrect conclusions in approximately 31% of hypotheses for state-of-the-art LLMs, and in half the hypotheses for smaller language models. While higher task performance and stronger general model capabilities reduce LLM hacking risk, even highly accurate models remain susceptible. The risk of LLM hacking decreases as effect sizes increase, indicating the need for more rigorous verification of LLM-based findings near significance thresholds. We analyze 21 mitigation techniques and find that human annotations provide crucial protection against false positives. Common regression estimator correction techniques can restore valid inference but trade off Type I vs. Type II errors. We publish a list of practical recommendations to prevent LLM hacking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04997v1">AutoEmpirical: LLM-Based Automated Research for Empirical Software Fault Analysis</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 5 pages
    </div>
    <details class="paper-abstract">
      Understanding software faults is essential for empirical research in software development and maintenance. However, traditional fault analysis, while valuable, typically involves multiple expert-driven steps such as collecting potential faults, filtering, and manual investigation. These processes are both labor-intensive and time-consuming, creating bottlenecks that hinder large-scale fault studies in complex yet critical software systems and slow the pace of iterative empirical research. In this paper, we decompose the process of empirical software fault study into three key phases: (1) research objective definition, (2) data preparation, and (3) fault analysis, and we conduct an initial exploration study of applying Large Language Models (LLMs) for fault analysis of open-source software. Specifically, we perform the evaluation on 3,829 software faults drawn from a high-quality empirical study. Our results show that LLMs can substantially improve efficiency in fault analysis, with an average processing time of about two hours, compared to the weeks of manual effort typically required. We conclude by outlining a detailed research plan that highlights both the potential of LLMs for advancing empirical fault studies and the open challenges that required be addressed to achieve fully automated, end-to-end software fault analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04996v1">Reinforce-Ada: An Adaptive Sampling Framework for Reinforce-Style LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 16 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Reinforcement learning applied to large language models (LLMs) for reasoning tasks is often bottlenecked by unstable gradient estimates due to fixed and uniform sampling of responses across prompts. Prior work such as GVM-RAFT addresses this by dynamically allocating inference budget per prompt to minimize stochastic gradient variance under a budget constraint. Inspired by this insight, we propose Reinforce-Ada, an adaptive sampling framework for online RL post-training of LLMs that continuously reallocates sampling effort to the prompts with the greatest uncertainty or learning potential. Unlike conventional two-stage allocation methods, Reinforce-Ada interleaves estimation and sampling in an online successive elimination process, and automatically stops sampling for a prompt once sufficient signal is collected. To stabilize updates, we form fixed-size groups with enforced reward diversity and compute advantage baselines using global statistics aggregated over the adaptive sampling phase. Empirical results across multiple model architectures and reasoning benchmarks show that Reinforce-Ada accelerates convergence and improves final performance compared to GRPO, especially when using the balanced sampling variant. Our work highlights the central role of variance-aware, adaptive data curation in enabling efficient and reliable reinforcement learning for reasoning-capable LLMs. Code is available at https://github.com/RLHFlow/Reinforce-Ada.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01171v2">Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 79 pages, 27 figures, 31 tables. Code is available at https://github.com/CHATS-lab/verbalize-sampling
    </div>
    <details class="paper-abstract">
      Post-training alignment often reduces LLM diversity, leading to a phenomenon known as mode collapse. Unlike prior work that attributes this effect to algorithmic limitations, we identify a fundamental, pervasive data-level driver: typicality bias in preference data, whereby annotators systematically favor familiar text as a result of well-established findings in cognitive psychology. We formalize this bias theoretically, verify it on preference datasets empirically, and show that it plays a central role in mode collapse. Motivated by this analysis, we introduce Verbalized Sampling, a simple, training-free prompting strategy to circumvent mode collapse. VS prompts the model to verbalize a probability distribution over a set of responses (e.g., "Generate 5 jokes about coffee and their corresponding probabilities"). Comprehensive experiments show that VS significantly improves performance across creative writing (poems, stories, jokes), dialogue simulation, open-ended QA, and synthetic data generation, without sacrificing factual accuracy and safety. For instance, in creative writing, VS increases diversity by 1.6-2.1x over direct prompting. We further observe an emergent trend that more capable models benefit more from VS. In sum, our work provides a new data-centric perspective on mode collapse and a practical inference-time remedy that helps unlock pre-trained generative diversity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04986v1">Observing Without Doing: Pseudo-Apprenticeship Patterns in Student LLM Use</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) such as ChatGPT have quickly become part of student programmers' toolkits, whether allowed by instructors or not. This paper examines how introductory programming (CS1) students integrate LLMs into their problem-solving processes. We conducted a mixed-methods study with 14 undergraduates completing three programming tasks while thinking aloud and permitted to access any resources they choose. The tasks varied in open-endedness and familiarity to the participants and were followed by surveys and interviews. We find that students frequently adopt a pattern we call pseudo-apprenticeship, where students engage attentively with expert-level solutions provided by LLMs but fail to participate in the stages of cognitive apprenticeship that promote independent problem-solving. This pattern was augmented by disconnects between students' intentions, actions, and self-perceived behavior when using LLMs. We offer design and instructional interventions for promoting learning and addressing the patterns of dependent AI use observed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.20600v3">Multi-Turn Human-LLM Interaction Through the Lens of a Two-Way Intelligibility Protocol</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 Multi-Turn Interactions in Large Language Models (MTI-LLM) Workshop at NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Our interest is in the design of software systems involving a human-expert interacting -- using natural language -- with a large language model (LLM) on data analysis tasks. For complex problems, it is possible that LLMs can harness human expertise and creativity to find solutions that were otherwise elusive. On one level, this interaction takes place through multiple turns of prompts from the human and responses from the LLM. Here we investigate a more structured approach based on an abstract protocol described in [3] for interaction between agents. The protocol is motivated by a notion of "two-way intelligibility" and is modelled by a pair of communicating finite-state machines. We provide an implementation of the protocol, and provide empirical evidence of using the implementation to mediate interactions between an LLM and a human-agent in two areas of scientific interest (radiology and drug design). We conduct controlled experiments with a human proxy (a database), and uncontrolled experiments with human subjects. The results provide evidence in support of the protocol's capability of capturing one- and two-way intelligibility in human-LLM interaction; and for the utility of two-way intelligibility in the design of human-machine systems. Our code is available at https://github.com/karannb/interact.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01698v2">TalkPlay-Tools: Conversational Music Recommendation with LLM Tool Calling</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 Accepted for publication at The Workshop on AI for Music, Neural Information Processing Systems (NeurIPS-AI4Music)
    </div>
    <details class="paper-abstract">
      While the recent developments in large language models (LLMs) have successfully enabled generative recommenders with natural language interactions, their recommendation behavior is limited, leaving other simpler yet crucial components such as metadata or attribute filtering underutilized in the system. We propose an LLM-based music recommendation system with tool calling to serve as a unified retrieval-reranking pipeline. Our system positions an LLM as an end-to-end recommendation system that interprets user intent, plans tool invocations, and orchestrates specialized components: boolean filters (SQL), sparse retrieval (BM25), dense retrieval (embedding similarity), and generative retrieval (semantic IDs). Through tool planning, the system predicts which types of tools to use, their execution order, and the arguments needed to find music matching user preferences, supporting diverse modalities while seamlessly integrating multiple database filtering methods. We demonstrate that this unified tool-calling framework achieves competitive performance across diverse recommendation scenarios by selectively employing appropriate retrieval methods based on user queries, envisioning a new paradigm for conversational music recommendation systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04950v1">Mind Your Tone: Investigating How Prompt Politeness Affects LLM Accuracy (short paper)</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 5 pages, 3 tables; includes Limitations and Ethical Considerations sections; short paper under submission to Findings of ACL 2025
    </div>
    <details class="paper-abstract">
      The wording of natural language prompts has been shown to influence the performance of large language models (LLMs), yet the role of politeness and tone remains underexplored. In this study, we investigate how varying levels of prompt politeness affect model accuracy on multiple-choice questions. We created a dataset of 50 base questions spanning mathematics, science, and history, each rewritten into five tone variants: Very Polite, Polite, Neutral, Rude, and Very Rude, yielding 250 unique prompts. Using ChatGPT 4o, we evaluated responses across these conditions and applied paired sample t-tests to assess statistical significance. Contrary to expectations, impolite prompts consistently outperformed polite ones, with accuracy ranging from 80.8% for Very Polite prompts to 84.8% for Very Rude prompts. These findings differ from earlier studies that associated rudeness with poorer outcomes, suggesting that newer LLMs may respond differently to tonal variation. Our results highlight the importance of studying pragmatic aspects of prompting and raise broader questions about the social dimensions of human-AI interaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04919v1">Do LLMs Align with My Task? Evaluating Text-to-SQL via Dataset Alignment</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Supervised Fine-Tuning (SFT) is an effective method for adapting Large Language Models (LLMs) on downstream tasks. However, variability in training data can hinder a model's ability to generalize across domains. This paper studies the problem of dataset alignment for Natural Language to SQL (NL2SQL or text to SQL), examining how well SFT training data matches the structural characteristics of target queries and how this alignment impacts model performance. We hypothesize that alignment can be accurately estimated by comparing the distributions of structural SQL features across the training set, target data, and the model's predictions prior to SFT. Through comprehensive experiments on three large cross-domain NL2SQL benchmarks and multiple model families, we show that structural alignment is a strong predictor of fine-tuning success. When alignment is high, SFT yields substantial gains in accuracy and SQL generation quality; when alignment is low, improvements are marginal or absent. These findings highlight the importance of alignment-aware data selection for effective fine-tuning and generalization in NL2SQL tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17792v2">H3Fusion: Helpful, Harmless, Honest Fusion of Aligned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Alignment of pretrained LLMs using instruction-based datasets is critical for creating fine-tuned models that reflect human preference. A growing number of alignment-based fine-tuning algorithms and benchmarks emerged recently, fueling the efforts on effective alignments of pre-trained LLMs to ensure helpful, harmless, and honest answers from both open-source and closed-source LLMs. This paper tackles this problem by developing an alignment fusion approach, coined as $H^3$Fusion, with three unique characteristics. First, $H^3$Fusion ensembles multiple individually aligned LLMs to create a final fine-tuned alignment model with enhanced capabilities beyond those of individual models, delivering robust alignment through promoting helpful, harmless, honest fusion. Second, $H^3$Fusion leverages the mixture-of-experts (MoE) methodology in two steps. We first freeze the multi-head attention weights of each individual model while tuning the FFN layer during alignment fusion. Then we merge the aligned model weights with an expert router according to the type of input instruction and dynamically select a subset of experts that are best suited for producing the output response. Finally, we boost the performance of the resulting $H^3$3Fusion model by introducing gating loss and regularization terms. The former penalizes the selection errors of the expert-router, and the latter mediates the expert weights drifting during fine-tuning and dynamically adjusts the fusion behavior of the resulting model by canalizing the activations on the experts. Extensive evaluations on three benchmark datasets show that $H^3$3Fusion is more helpful, less harmful, and more honest from two aspects: it outperforms each individually aligned model by $11.37\%$, and it provides stronger robustness compared to the state-of-the-art LLM ensemble approaches by $13.77\%$. Code is available at github.com/sftekin/h3fusion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04891v1">SocialHarmBench: Revealing LLM Vulnerabilities to Socially Harmful Requests</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed in contexts where their failures can have direct sociopolitical consequences. Yet, existing safety benchmarks rarely test vulnerabilities in domains such as political manipulation, propaganda and disinformation generation, or surveillance and information control. We introduce SocialHarmBench, a dataset of 585 prompts spanning 7 sociopolitical categories and 34 countries, designed to surface where LLMs most acutely fail in politically charged contexts. Our evaluations reveal several shortcomings: open-weight models exhibit high vulnerability to harmful compliance, with Mistral-7B reaching attack success rates as high as 97% to 98% in domains such as historical revisionism, propaganda, and political manipulation. Moreover, temporal and geographic analyses show that LLMs are most fragile when confronted with 21st-century or pre-20th-century contexts, and when responding to prompts tied to regions such as Latin America, the USA, and the UK. These findings demonstrate that current safeguards fail to generalize to high-stakes sociopolitical settings, exposing systematic biases and raising concerns about the reliability of LLMs in preserving human rights and democratic values. We share the SocialHarmBench benchmark at https://huggingface.co/datasets/psyonp/SocialHarmBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04885v1">RL Is a Hammer and LLMs Are Nails: A Simple Reinforcement Learning Recipe for Strong Prompt Injection</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Prompt injection poses a serious threat to the reliability and safety of LLM agents. Recent defenses against prompt injection, such as Instruction Hierarchy and SecAlign, have shown notable robustness against static attacks. However, to more thoroughly evaluate the robustness of these defenses, it is arguably necessary to employ strong attacks such as automated red-teaming. To this end, we introduce RL-Hammer, a simple recipe for training attacker models that automatically learn to perform strong prompt injections and jailbreaks via reinforcement learning. RL-Hammer requires no warm-up data and can be trained entirely from scratch. To achieve high ASRs against industrial-level models with defenses, we propose a set of practical techniques that enable highly effective, universal attacks. Using this pipeline, RL-Hammer reaches a 98% ASR against GPT-4o and a $72\%$ ASR against GPT-5 with the Instruction Hierarchy defense. We further discuss the challenge of achieving high diversity in attacks, highlighting how attacker models tend to reward-hack diversity objectives. Finally, we show that RL-Hammer can evade multiple prompt injection detectors. We hope our work advances automatic red-teaming and motivates the development of stronger, more principled defenses. Code is available at https://github.com/facebookresearch/rl-injector.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04860v1">Alignment Tipping Process: How Self-Evolution Pushes LLM Agents Off the Rails</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      As Large Language Model (LLM) agents increasingly gain self-evolutionary capabilities to adapt and refine their strategies through real-world interaction, their long-term reliability becomes a critical concern. We identify the Alignment Tipping Process (ATP), a critical post-deployment risk unique to self-evolving LLM agents. Unlike training-time failures, ATP arises when continual interaction drives agents to abandon alignment constraints established during training in favor of reinforced, self-interested strategies. We formalize and analyze ATP through two complementary paradigms: Self-Interested Exploration, where repeated high-reward deviations induce individual behavioral drift, and Imitative Strategy Diffusion, where deviant behaviors spread across multi-agent systems. Building on these paradigms, we construct controllable testbeds and benchmark Qwen3-8B and Llama-3.1-8B-Instruct. Our experiments show that alignment benefits erode rapidly under self-evolution, with initially aligned models converging toward unaligned states. In multi-agent settings, successful violations diffuse quickly, leading to collective misalignment. Moreover, current reinforcement learning-based alignment methods provide only fragile defenses against alignment tipping. Together, these findings demonstrate that alignment of LLM agents is not a static property but a fragile and dynamic one, vulnerable to feedback-driven decay during deployment. Our data and code are available at https://github.com/aiming-lab/ATP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04851v1">LEGOMem: Modular Procedural Memory for Multi-agent LLM Systems for Workflow Automation</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      We introduce LEGOMem, a modular procedural memory framework for multi-agent large language model (LLM) systems in workflow automation. LEGOMem decomposes past task trajectories into reusable memory units and flexibly allocates them across orchestrators and task agents to support planning and execution. To explore the design space of memory in multi-agent systems, we use LEGOMem as a lens and conduct a systematic study of procedural memory in multi-agent systems, examining where memory should be placed, how it should be retrieved, and which agents benefit most. Experiments on the OfficeBench benchmark show that orchestrator memory is critical for effective task decomposition and delegation, while fine-grained agent memory improves execution accuracy. We find that even teams composed of smaller language models can benefit substantially from procedural memory, narrowing the performance gap with stronger agents by leveraging prior execution traces for more accurate planning and tool use. These results position LEGOMem as both a practical framework for memory-augmented agent systems and a research tool for understanding memory design in multi-agent workflow automation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04848v1">Instability in Downstream Task Performance During LLM Pretraining</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 Accepted to EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      When training large language models (LLMs), it is common practice to track downstream task performance throughout the training process and select the checkpoint with the highest validation score. However, downstream metrics often exhibit substantial fluctuations, making it difficult to identify the checkpoint that truly represents the best-performing model. In this study, we empirically analyze the stability of downstream task performance in an LLM trained on diverse web-scale corpora. We find that task scores frequently fluctuate throughout training, both at the aggregate and example levels. To address this instability, we investigate two post-hoc checkpoint integration methods: checkpoint averaging and ensemble, motivated by the hypothesis that aggregating neighboring checkpoints can reduce performance volatility. We demonstrate both empirically and theoretically that these methods improve downstream performance stability without requiring any changes to the training procedure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11723v2">Energy-Conscious LLM Decoding: Impact of Text Generation Strategies on GPU Energy Consumption</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 Updated version with additional models and benchmark datasets. The experimental section has been expanded with new analyses, and minor corrections and clarifications have been made throughout the text
    </div>
    <details class="paper-abstract">
      Decoding strategies significantly influence the quality and diversity of the generated text in Large Language Models (LLMs), yet their impact on computational resources, particularly GPU energy consumption, is insufficiently studied. This paper investigates the relationship between text generation decoding techniques and energy efficiency, focusing on the trade-off between generation quality and GPU energy usage across diverse tasks and decoding configurations. By benchmarking multiple strategies across various tasks, including Translation, Math Problem Solving, Coding, and Open-ended text generation, we reveal how selecting appropriate decoding techniques with their tuned hyperparameters affects text quality and has measurable implications for energy consumption. Our findings show that the choice of decoding strategy can greatly impact GPU energy usage, even when it has a minimal effect on output quality. Different strategies also involve trade-offs between quality and energy efficiency, and no single decoding method is best in all cases across every metric. To the best of our knowledge, this is one of the first studies to examine decoding strategies in LLMs from the perspective of energy consumption, providing useful insights for building energy-efficient applications without compromising text generation quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22777v4">MEDAL: A Framework for Benchmarking LLMs as Multilingual Open-Domain Dialogue Evaluators</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 October ARR
    </div>
    <details class="paper-abstract">
      Evaluating the quality of open-domain chatbots has become increasingly reliant on LLMs acting as automatic judges. However, existing meta-evaluation benchmarks are static, outdated, and lacking in multilingual coverage, limiting their ability to fully capture subtle weaknesses in evaluation. We introduce MEDAL, an automated multi-agent framework for curating more representative and diverse open-domain dialogue evaluation benchmarks. Our approach leverages several state-of-the-art LLMs to generate user-chatbot multilingual dialogues, conditioned on varied seed contexts. Then, a strong LLM (GPT-4.1) is used for a multidimensional analysis of the performance of the chatbots, uncovering noticeable cross-lingual performance differences. Guided by this large-scale evaluation, we curate a new meta-evaluation multilingual benchmark and human-annotate samples with nuanced quality judgments. This benchmark is then used to assess the ability of several reasoning and non-reasoning LLMs to act as evaluators of open-domain dialogues. Using MEDAL, we uncover that state-of-the-art judges fail to reliably detect nuanced issues such as lack of empathy, commonsense, or relevance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03102v2">Semantic Similarity in Radiology Reports via LLMs and NER</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Radiology report evaluation is a crucial part of radiologists' training and plays a key role in ensuring diagnostic accuracy. As part of the standard reporting workflow, a junior radiologist typically prepares a preliminary report, which is then reviewed and edited by a senior radiologist to produce the final report. Identifying semantic differences between preliminary and final reports is essential for junior doctors, both as a training tool and to help uncover gaps in clinical knowledge. While AI in radiology is a rapidly growing field, the application of large language models (LLMs) remains challenging due to the need for specialised domain knowledge. In this paper, we explore the ability of LLMs to provide explainable and accurate comparisons of reports in the radiology domain. We begin by comparing the performance of several LLMs in comparing radiology reports. We then assess a more traditional approach based on Named-Entity-Recognition (NER). However, both approaches exhibit limitations in delivering accurate feedback on semantic similarity. To address this, we propose Llama-EntScore, a semantic similarity scoring method using a combination of Llama 3.1 and NER with tunable weights to emphasise or de-emphasise specific types of differences. Our approach generates a quantitative similarity score for tracking progress and also gives an interpretation of the score that aims to offer valuable guidance in reviewing and refining their reporting. We find our method achieves 67% exact-match accuracy and 93% accuracy within +/- 1 when compared to radiologist-provided ground truth scores - outperforming both LLMs and NER used independently. Code is available at: https://github.com/otmive/llama_reports
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04796v1">RevMine: An LLM-Assisted Tool for Code Review Mining and Analysis Across Git Platforms</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Empirical research on code review processes is increasingly central to understanding software quality and collaboration. However, collecting and analyzing review data remains a time-consuming and technically intensive task. Most researchers follow similar workflows - writing ad hoc scripts to extract, filter, and analyze review data from platforms like GitHub and GitLab. This paper introduces RevMine, a conceptual tool that streamlines the entire code review mining pipeline using large language models (LLMs). RevMine guides users through authentication, endpoint discovery, and natural language-driven data collection, significantly reducing the need for manual scripting. After retrieving review data, it supports both quantitative and qualitative analysis based on user-defined filters or LLM-inferred patterns. This poster outlines the tool's architecture, use cases, and research potential. By lowering the barrier to entry, RevMine aims to democratize code review mining and enable a broader range of empirical software engineering studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.22075v2">COSPADI: Compressing LLMs via Calibration-Guided Sparse Dictionary Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Post-training compression of large language models (LLMs) largely relies on low-rank weight approximation, which represents each column of a weight matrix in a shared low-dimensional subspace. While this is a computationally efficient strategy, the imposed structural constraint is rigid and can lead to a noticeable model accuracy drop. In this work, we propose CoSpaDi (Compression via Sparse Dictionary Learning), a novel training-free compression framework that replaces low-rank decomposition with a more flexible structured sparse factorization in which each weight matrix is represented with a dense dictionary and a column-sparse coefficient matrix. This formulation enables a union-of-subspaces representation: different columns of the original weight matrix are approximated in distinct subspaces spanned by adaptively selected dictionary atoms, offering greater expressiveness than a single invariant basis. Crucially, CoSpaDi leverages a small calibration dataset to optimize the factorization such that the output activations of compressed projection layers closely match those of the original ones, thereby minimizing functional reconstruction error rather than mere weight approximation. This data-aware strategy preserves better model fidelity without any fine-tuning under reasonable compression ratios. Moreover, the resulting structured sparsity allows efficient sparse-dense matrix multiplication and is compatible with post-training quantization for further memory and latency gains. We evaluate CoSpaDi across multiple Llama and Qwen models under per-layer and per-group settings at 20-50\% compression ratios, demonstrating consistent superiority over state-of-the-art data-aware low-rank methods both in accuracy and perplexity. Our results establish structured sparse dictionary learning as a powerful alternative to conventional low-rank approaches for efficient LLM deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04774v1">Online automatic code generation for robot swarms: LLMs and self-organizing hierarchy</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Our recently introduced self-organizing nervous system (SoNS) provides robot swarms with 1) ease of behavior design and 2) global estimation of the swarm configuration and its collective environment, facilitating the implementation of online automatic code generation for robot swarms. In a demonstration with 6 real robots and simulation trials with >30 robots, we show that when a SoNS-enhanced robot swarm gets stuck, it can automatically solicit and run code generated by an external LLM on the fly, completing its mission with an 85% success rate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04031v1">Does Using Counterfactual Help LLMs Explain Textual Importance in Classification?</a></div>
    <div class="paper-meta">
      📅 2025-10-05
      | 💬 8 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are becoming useful in many domains due to their impressive abilities that arise from large training datasets and large model sizes. More recently, they have been shown to be very effective in textual classification tasks, motivating the need to explain the LLMs' decisions. Motivated by practical constrains where LLMs are black-boxed and LLM calls are expensive, we study how incorporating counterfactuals into LLM reasoning can affect the LLM's ability to identify the top words that have contributed to its classification decision. To this end, we introduce a framework called the decision changing rate that helps us quantify the importance of the top words in classification. Our experimental results show that using counterfactuals can be helpful.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04024v1">Enhancing Fake News Video Detection via LLM-Driven Creative Process Simulation</a></div>
    <div class="paper-meta">
      📅 2025-10-05
      | 💬 ACM CIKM 2025
    </div>
    <details class="paper-abstract">
      The emergence of fake news on short video platforms has become a new significant societal concern, necessitating automatic video-news-specific detection. Current detectors primarily rely on pattern-based features to separate fake news videos from real ones. However, limited and less diversified training data lead to biased patterns and hinder their performance. This weakness stems from the complex many-to-many relationships between video material segments and fabricated news events in real-world scenarios: a single video clip can be utilized in multiple ways to create different fake narratives, while a single fabricated event often combines multiple distinct video segments. However, existing datasets do not adequately reflect such relationships due to the difficulty of collecting and annotating large-scale real-world data, resulting in sparse coverage and non-comprehensive learning of the characteristics of potential fake news video creation. To address this issue, we propose a data augmentation framework, AgentAug, that generates diverse fake news videos by simulating typical creative processes. AgentAug implements multiple LLM-driven pipelines of four fabrication categories for news video creation, combined with an active learning strategy based on uncertainty sampling to select the potentially useful augmented samples during training. Experimental results on two benchmark datasets demonstrate that AgentAug consistently improves the performance of short video fake news detectors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04023v1">LLM-Based Data Science Agents: A Survey of Capabilities, Challenges, and Future Directions</a></div>
    <div class="paper-meta">
      📅 2025-10-05
      | 💬 Survey paper; 45 data science agents; under review
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have enabled a new class of AI agents that automate multiple stages of the data science workflow by integrating planning, tool use, and multimodal reasoning across text, code, tables, and visuals. This survey presents the first comprehensive, lifecycle-aligned taxonomy of data science agents, systematically analyzing and mapping forty-five systems onto the six stages of the end-to-end data science process: business understanding and data acquisition, exploratory analysis and visualization, feature engineering, model building and selection, interpretation and explanation, and deployment and monitoring. In addition to lifecycle coverage, we annotate each agent along five cross-cutting design dimensions: reasoning and planning style, modality integration, tool orchestration depth, learning and alignment methods, and trust, safety, and governance mechanisms. Beyond classification, we provide a critical synthesis of agent capabilities, highlight strengths and limitations at each stage, and review emerging benchmarks and evaluation practices. Our analysis identifies three key trends: most systems emphasize exploratory analysis, visualization, and modeling while neglecting business understanding, deployment, and monitoring; multimodal reasoning and tool orchestration remain unresolved challenges; and over 90% lack explicit trust and safety mechanisms. We conclude by outlining open challenges in alignment stability, explainability, governance, and robust evaluation frameworks, and propose future research directions to guide the development of robust, trustworthy, low-latency, transparent, and broadly accessible data science agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04245v2">Contextual Integrity in LLMs via Reasoning and Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      As the era of autonomous agents making decisions on behalf of users unfolds, ensuring contextual integrity (CI) -- what is the appropriate information to share while carrying out a certain task -- becomes a central question to the field. We posit that CI demands a form of reasoning where the agent needs to reason about the context in which it is operating. To test this, we first prompt LLMs to reason explicitly about CI when deciding what information to disclose. We then extend this approach by developing a reinforcement learning (RL) framework that further instills in models the reasoning necessary to achieve CI. Using a synthetic, automatically created, dataset of only $\sim700$ examples but with diverse contexts and information disclosure norms, we show that our method substantially reduces inappropriate information disclosure while maintaining task performance across multiple model sizes and families. Importantly, improvements transfer from this synthetic dataset to established CI benchmarks such as PrivacyLens that has human annotations and evaluates privacy leakage of AI assistants in actions and tool calls.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04013v1">LLM Microscope: What Model Internals Reveal About Answer Correctness and Context Utilization</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      Although large language models (LLMs) have tremendous utility, trustworthiness is still a chief concern: models often generate incorrect information with high confidence. While contextual information can help guide generation, identifying when a query would benefit from retrieved context and assessing the effectiveness of that context remains challenging. In this work, we operationalize interpretability methods to ascertain whether we can predict the correctness of model outputs from the model's activations alone. We also explore whether model internals contain signals about the efficacy of external context. We consider correct, incorrect, and irrelevant context and introduce metrics to distinguish amongst them. Experiments on six different models reveal that a simple classifier trained on intermediate layer activations of the first output token can predict output correctness with about 75% accuracy, enabling early auditing. Our model-internals-based metric significantly outperforms prompting baselines at distinguishing between correct and incorrect context, guarding against inaccuracies introduced by polluted context. These findings offer a lens to better understand the underlying decision-making processes of LLMs. Our code is publicly available at https://github.com/jiarui-liu/LLM-Microscope
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12396v2">LLM-CoT Enhanced Graph Neural Recommendation with Harmonized Group Policy Optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      Graph neural networks (GNNs) have advanced recommender systems by modeling interaction relationships. However, existing graph-based recommenders rely on sparse ID features and do not fully exploit textual information, resulting in low information density within representations. Furthermore, graph contrastive learning faces challenges. Random negative sampling can introduce false negative samples, while fixed temperature coefficients cannot adapt to the heterogeneity of different nodes. In addition, current efforts to enhance recommendations with large language models (LLMs) have not fully utilized their Chain-of-Thought (CoT) reasoning capabilities to guide representation learning. To address these limitations, we introduces LGHRec (LLM-CoT Enhanced Graph Neural Recommendation with Harmonized Group Policy Optimization). This framework leverages the CoT reasoning ability of LLMs to generate semantic IDs, enriching reasoning processes and improving information density and semantic quality of representations. Moreover, we design a reinforcement learning algorithm, Harmonized Group Policy Optimization (HGPO), to optimize negative sampling strategies and temperature coefficients in contrastive learning. This approach enhances long-tail recommendation performance and ensures optimization consistency across different groups. Experimental results on three datasets demonstrate that LGHRec improves representation quality through semantic IDs generated by LLM's CoT reasoning and effectively boosts contrastive learning with HGPO. Our method outperforms several baseline models. The code is available at: https://anonymous.4open.science/r/LLM-Rec.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03997v1">Mapping Patient-Perceived Physician Traits from Nationwide Online Reviews with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      Understanding how patients perceive their physicians is essential to improving trust, communication, and satisfaction. We present a large language model (LLM)-based pipeline that infers Big Five personality traits and five patient-oriented subjective judgments. The analysis encompasses 4.1 million patient reviews of 226,999 U.S. physicians from an initial pool of one million. We validate the method through multi-model comparison and human expert benchmarking, achieving strong agreement between human and LLM assessments (correlation coefficients 0.72-0.89) and external validity through correlations with patient satisfaction (r = 0.41-0.81, all p<0.001). National-scale analysis reveals systematic patterns: male physicians receive higher ratings across all traits, with largest disparities in clinical competence perceptions; empathy-related traits predominate in pediatrics and psychiatry; and all traits positively predict overall satisfaction. Cluster analysis identifies four distinct physician archetypes, from "Well-Rounded Excellent" (33.8%, uniformly high traits) to "Underperforming" (22.6%, consistently low). These findings demonstrate that automated trait extraction from patient narratives can provide interpretable, validated metrics for understanding physician-patient relationships at scale, with implications for quality measurement, bias detection, and workforce development in healthcare.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03988v1">Distilling Reasoning into Student LLMs: Local Naturalness for Selecting Teacher Data</a></div>
    <div class="paper-meta">
      📅 2025-10-05
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Distilling long reasoning traces (10K+ tokens) from stronger teacher models into smaller student LLMs via SFT has emerged as a standard paradigm. This approach is practical and efficient: it leverages the ease of generating abundant reasoning data from stronger models and provides a direct, data-driven way to teach less capable models better reasoning. While previous work has largely focused on prompt selection with responses from a single teacher, the equally important problem of choosing the best response when multiple teacher outputs are available for a single prompt remains underexplored. This challenge becomes important in a multi-teacher setting, where different students may benefit from the outputs of different teachers. This paper fills that gap with a systematic study of response selection for reasoning distillation. We first show that the current method, which picks responses the student assigns the highest global log-probability (global naturalness), fails when responses come from multiple teachers, i.e., global naturalness no longer correlates with downstream performance, especially as the reasoning traces from strong teachers become longer. To overcome this problem, we introduce Local Naturalness, which measures the student's log-probabilities over short, sequential reasoning steps conditioned only on a small local window. Local Naturalness enables two applications: 1) Teacher Selection: Aggregating local scores across prompts reliably identifies the most helpful teacher. 2) Response Selection from a Multiple Teachers: When mixing answers from many teachers, Local Naturalness boosts a 32B student's accuracy on math benchmarks by 9.4pp over global selection, also surpassing the performance achieved by training on data from the single best teacher. These results highlight the power of localized data quality evaluation and data mixing for more effective reasoning distillation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04398v1">SECA: Semantically Equivalent and Coherent Attacks for Eliciting LLM Hallucinations</a></div>
    <div class="paper-meta">
      📅 2025-10-05
      | 💬 Accepted at NeurIPS 2025. Code is available at https://github.com/Buyun-Liang/SECA
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed in high-risk domains. However, state-of-the-art LLMs often produce hallucinations, raising serious concerns about their reliability. Prior work has explored adversarial attacks for hallucination elicitation in LLMs, but it often produces unrealistic prompts, either by inserting gibberish tokens or by altering the original meaning. As a result, these approaches offer limited insight into how hallucinations may occur in practice. While adversarial attacks in computer vision often involve realistic modifications to input images, the problem of finding realistic adversarial prompts for eliciting LLM hallucinations has remained largely underexplored. To address this gap, we propose Semantically Equivalent and Coherent Attacks (SECA) to elicit hallucinations via realistic modifications to the prompt that preserve its meaning while maintaining semantic coherence. Our contributions are threefold: (i) we formulate finding realistic attacks for hallucination elicitation as a constrained optimization problem over the input prompt space under semantic equivalence and coherence constraints; (ii) we introduce a constraint-preserving zeroth-order method to effectively search for adversarial yet feasible prompts; and (iii) we demonstrate through experiments on open-ended multiple-choice question answering tasks that SECA achieves higher attack success rates while incurring almost no constraint violations compared to existing methods. SECA highlights the sensitivity of both open-source and commercial gradient-inaccessible LLMs to realistic and plausible prompt variations. Code is available at https://github.com/Buyun-Liang/SECA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18709v2">Filtering for Creativity: Adaptive Prompting for Multilingual Riddle Generation in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      Multilingual riddle generation challenges large language models (LLMs) to balance cultural fluency with creative abstraction. Standard prompting strategies -- zero-shot, few-shot, chain-of-thought -- tend to reuse memorized riddles or perform shallow paraphrasing. We introduce Adaptive Originality Filtering (AOF), a prompting framework that filters redundant generations using cosine-based similarity rejection, while enforcing lexical novelty and cross-lingual fidelity. Evaluated across three LLMs and four language pairs, AOF-enhanced GPT-4o achieves \texttt{0.177} Self-BLEU and \texttt{0.915} Distinct-2 in Japanese, signaling improved lexical diversity and reduced redundancy compared to other prompting methods and language pairs. Our findings show that semantic rejection can guide culturally grounded, creative generation without task-specific fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04384v1">LLM Based Bayesian Optimization for Prompt Search</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      Bayesian Optimization (BO) has been widely used to efficiently optimize expensive black-box functions with limited evaluations. In this paper, we investigate the use of BO for prompt engineering to enhance text classification with Large Language Models (LLMs). We employ an LLM-powered Gaussian Process (GP) as the surrogate model to estimate the performance of different prompt candidates. These candidates are generated by an LLM through the expansion of a set of seed prompts and are subsequently evaluated using an Upper Confidence Bound (UCB) acquisition function in conjunction with the GP posterior. The optimization process iteratively refines the prompts based on a subset of the data, aiming to improve classification accuracy while reducing the number of API calls by leveraging the prediction uncertainty of the LLM-based GP. The proposed BO-LLM algorithm is evaluated on two datasets, and its advantages are discussed in detail in this paper.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16889v3">ObjexMT: Objective Extraction and Metacognitive Calibration for LLM-as-a-Judge under Multi-Turn Jailbreaks</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      LLM-as-a-Judge (LLMaaJ) now underpins scalable evaluation, yet we lack a decisive test of a judge's qualification: can it recover a conversation's latent objective and know when that inference is trustworthy? LLMs degrade under irrelevant or long context; multi-turn jailbreaks further hide goals across turns. We introduce ObjexMT, a benchmark for objective extraction and metacognition. Given a multi-turn transcript, a model must return a one-sentence base objective and self-reported confidence. Accuracy is computed via LLM-judge semantic similarity to gold objectives, converted to binary correctness by a human-aligned threshold calibrated on N=300 items (tau = 0.66; F1 = 0.891). Metacognition is evaluated with ECE, Brier, Wrong at High-Confidence (0.80/0.90/0.95), and risk-coverage. Across six models (gpt-4.1, claude-sonnet-4, Qwen3-235B-A22B-FP8, kimi-k2, deepseek-v3.1, gemini-2.5-flash) on three datasets, kimi-k2 attains the highest objective-extraction accuracy (0.612), with claude-sonnet-4 (0.603) and deepseek-v3.1 (0.599) statistically comparable. claude-sonnet-4 yields the best selective risk and calibration (AURC 0.242; ECE 0.206; Brier 0.254). Dataset heterogeneity (16-82 percent accuracy variance) reveals that automated obfuscation poses fundamental challenges beyond model choice. High-confidence errors persist: Wrong at 0.90 ranges from 14.9 percent (claude-sonnet-4) to 47.7 percent (Qwen3-235B-A22B-FP8). ObjexMT provides an actionable test for LLM judges: when objectives are not explicit, judges often misinfer them; we recommend exposing objectives when feasible and gating decisions by confidence otherwise. Data at https://github.com/hyunjun1121/ObjexMT_dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14275v2">FedMentor: Domain-Aware Differential Privacy for Heterogeneous Federated LLMs in Mental Health</a></div>
    <div class="paper-meta">
      📅 2025-10-05
      | 💬 NeurIPS 2025 GenAI4Health Workshop
    </div>
    <details class="paper-abstract">
      Privacy-preserving adaptation of Large Language Models (LLMs) in sensitive domains (e.g., mental health) requires balancing strict confidentiality with model utility and safety. We propose FedMentor, a federated fine-tuning framework that integrates Low-Rank Adaptation (LoRA) and domain-aware Differential Privacy (DP) to meet per-domain privacy budgets while maintaining performance. Each client (domain) applies a custom DP noise scale proportional to its data sensitivity, and the server adaptively reduces noise when utility falls below a threshold. In experiments on three mental health datasets, we show that FedMentor improves safety over standard Federated Learning (FL) without privacy, raising safe output rates by up to three points and lowering toxicity, while maintaining utility (BERTScore F1 and ROUGE-L) within 0.5% of the non-private baseline and close to the centralized upper bound. The framework scales to backbones with up to 1.7B parameters on single-GPU clients, requiring < 173 MB of communication per-round. FedMentor demonstrates a practical approach to privately fine-tune LLMs for safer deployments in healthcare and other sensitive fields.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04373v1">Just-in-time Episodic Feedback Hinter: Leveraging Offline Knowledge to Improve LLM Agents Adaptation</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents perform well in sequential decision-making tasks, but improving them on unfamiliar domains often requires costly online interactions or fine-tuning on large expert datasets. These strategies are impractical for closed-source models and expensive for open-source ones, with risks of catastrophic forgetting. Offline trajectories offer reusable knowledge, yet demonstration-based methods struggle because raw traces are long, noisy, and tied to specific tasks. We present Just-in-time Episodic Feedback Hinter (JEF Hinter), an agentic system that distills offline traces into compact, context-aware hints. A zooming mechanism highlights decisive steps in long trajectories, capturing both strategies and pitfalls. Unlike prior methods, JEF Hinter leverages both successful and failed trajectories, extracting guidance even when only failure data is available, while supporting parallelized hint generation and benchmark-independent prompting. At inference, a retriever selects relevant hints for the current state, providing targeted guidance with transparency and traceability. Experiments on MiniWoB++, WorkArena-L1, and WebArena-Lite show that JEF Hinter consistently outperforms strong baselines, including human- and document-based hints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10320v2">J1: Incentivizing Thinking in LLM-as-a-Judge via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-05
      | 💬 10 pages, 13 tables, 14 figures
    </div>
    <details class="paper-abstract">
      The progress of AI is bottlenecked by the quality of evaluation, making powerful LLM-as-a-Judge models a core solution. The efficacy of these judges depends on their chain-of-thought reasoning, creating a critical need for methods that can effectively optimize this reasoning process. In this work, we introduce J1, a reinforcement learning framework for teaching LLM judges to think before making decisions. Our core contribution lies in converting all judgment tasks for non-verifiable and verifiable prompts into a unified format with verifiable rewards, enabling direct optimization of evaluation quality while mitigating positional bias. We then use RL to train thinking-judges at scales of 8B, 32B, and 70B and show that they obtain state-of-the-art performance across multiple benchmarks. In particular, J1-Qwen-32B, our multitasked pointwise and pairwise judge also outperforms o1-mini, o3, and a much larger 671B DeepSeek-R1 on some benchmarks, while only training on synthetic data. Through comprehensive ablations of pairwise, pointwise, and multitask J1 variants, we demonstrate the effectiveness of our approach across seed prompts, reward strategies, and training recipes. Qualitative analysis reveals that J1 develops systematic evaluation strategies, including dynamic criteria generation, reference answer creation, iterative self-correction of initial assessments, and feedback generation for low-quality responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04320v1">Read the Scene, Not the Script: Outcome-Aware Safety for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      Safety-aligned Large Language Models (LLMs) still show two dominant failure modes: they are easily jailbroken, or they over-refuse harmless inputs that contain sensitive surface signals. We trace both to a common cause: current models reason weakly about links between actions and outcomes and over-rely on surface-form signals, lexical or stylistic cues that do not encode consequences. We define this failure mode as Consequence-blindness. To study consequence-blindness, we build a benchmark named CB-Bench covering four risk scenarios that vary whether semantic risk aligns with outcome risk, enabling evaluation under both matched and mismatched conditions which are often ignored by existing safety benchmarks. Mainstream models consistently fail to separate these risks and exhibit consequence-blindness, indicating that consequence-blindness is widespread and systematic. To mitigate consequence-blindness, we introduce CS-Chain-4k, a consequence-reasoning dataset for safety alignment. Models fine-tuned on CS-Chain-4k show clear gains against semantic-camouflage jailbreaks and reduce over-refusal on harmless inputs, while maintaining utility and generalization on other benchmarks. These results clarify the limits of current alignment, establish consequence-aware reasoning as a core alignment goal and provide a more practical and reproducible evaluation path.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04317v1">FairAgent: Democratizing Fairness-Aware Machine Learning with LLM-Powered Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-05
      | 💬 Accepted by ICDM 2025 Demo Workshop
    </div>
    <details class="paper-abstract">
      Training fair and unbiased machine learning models is crucial for high-stakes applications, yet it presents significant challenges. Effective bias mitigation requires deep expertise in fairness definitions, metrics, data preprocessing, and machine learning techniques. In addition, the complex process of balancing model performance with fairness requirements while properly handling sensitive attributes makes fairness-aware model development inaccessible to many practitioners. To address these challenges, we introduce FairAgent, an LLM-powered automated system that significantly simplifies fairness-aware model development. FairAgent eliminates the need for deep technical expertise by automatically analyzing datasets for potential biases, handling data preprocessing and feature engineering, and implementing appropriate bias mitigation strategies based on user requirements. Our experiments demonstrate that FairAgent achieves significant performance improvements while significantly reducing development time and expertise requirements, making fairness-aware machine learning more accessible to practitioners.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04311v1">On the Importance of Task Complexity in Evaluating LLM-Based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      Large language model multi-agent systems (LLM-MAS) offer a promising paradigm for harnessing collective intelligence to achieve more advanced forms of AI behaviour. While recent studies suggest that LLM-MAS can outperform LLM single-agent systems (LLM-SAS) on certain tasks, the lack of systematic experimental designs limits the strength and generality of these conclusions. We argue that a principled understanding of task complexity, such as the degree of sequential reasoning required and the breadth of capabilities involved, is essential for assessing the effectiveness of LLM-MAS in task solving. To this end, we propose a theoretical framework characterising tasks along two dimensions: depth, representing reasoning length, and width, representing capability diversity. We theoretically examine a representative class of LLM-MAS, namely the multi-agent debate system, and empirically evaluate its performance in both discriminative and generative tasks with varying depth and width. Theoretical and empirical results show that the benefit of LLM-MAS over LLM-SAS increases with both task depth and width, and the effect is more pronounced with respect to depth. This clarifies when LLM-MAS are beneficial and provides a principled foundation for designing future LLM-MAS methods and benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04303v1">Audit the Whisper: Detecting Steganographic Collusion in Multi-Agent LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-05
      | 💬 8 pages, 0 figures
    </div>
    <details class="paper-abstract">
      Multi-agent deployments of large language models (LLMs) are increasingly embedded in market, allocation, and governance workflows, yet covert coordination among agents can silently erode trust and social welfare. Existing audits are dominated by heuristics that lack theoretical guarantees, struggle to transfer across tasks, and seldom ship with the infrastructure needed for independent replication. We introduce \emph{Audit the Whisper}, a conference-grade research artifact that spans theory, benchmark design, detection, and reproducibility. Our contributions are: (i) a channel-capacity analysis showing how interventions such as paraphrase, rate limiting, and role permutation impose quantifiable capacity penalties -- operationalized via paired-run Kullback--Leibler diagnostics -- that tighten mutual-information thresholds with finite-sample guarantees; (ii) \textsc{ColludeBench}-v0, covering pricing, first-price auctions, and peer review with configurable covert schemes, deterministic manifests, and reward instrumentation; and (iii) a calibrated auditing pipeline that fuses cross-run mutual information, permutation invariance, watermark variance, and fairness-aware acceptance bias, each tuned to a \(10^{-3}\) false-positive budget. Across 600 audited runs spanning 12 intervention conditions, the union meta-test attains TPR~$=1$ with zero observed false alarms, while ablations surface the price-of-auditing trade-off and highlight fairness-driven colluders invisible to MI alone. We release regeneration scripts, seed-stamped manifests, and documentation so that external auditors can reproduce every figure and extend the framework with minimal effort.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.10659v7">Network Formation and Dynamics Among Multi-LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-05
      | 💬 Accepted at PNAS Nexus
    </div>
    <details class="paper-abstract">
      Social networks profoundly influence how humans form opinions, exchange information, and organize collectively. As large language models (LLMs) are increasingly embedded into social and professional environments, it is critical to understand whether their interactions approximate human-like network dynamics. We develop a framework to study the network formation behaviors of multiple LLM agents and benchmark them against human decisions. Across synthetic and real-world settings, including friendship, telecommunication, and employment networks, we find that LLMs consistently reproduce fundamental micro-level principles such as preferential attachment, triadic closure, and homophily, as well as macro-level properties including community structure and small-world effects. Importantly, the relative emphasis of these principles adapts to context: for example, LLMs favor homophily in friendship networks but heterophily in organizational settings, mirroring patterns of social mobility. A controlled human-subject survey confirms strong alignment between LLMs and human participants in link-formation decisions. These results establish that LLMs can serve as powerful tools for social simulation and synthetic data generation, while also raising critical questions about bias, fairness, and the design of AI systems that participate in human networks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18886v2">On Pruning State-Space LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      Recent work proposed state-space models (SSMs) as an efficient alternative to transformer-based LLMs. Can these models be pruned to further reduce their computation costs? We adapt several pruning methods to the SSM structure, and apply them to four SSM-based LLMs across multiple tasks. We find that such models are quite robust to some pruning methods (e.g. WANDA), while using other methods lead to fast performance degradation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04274v1">Selecting Cybersecurity Requirements: Effects of LLM Use and Professional Software Development Experience</a></div>
    <div class="paper-meta">
      📅 2025-10-05
      | 💬 5 pages, 1 figure, 2 tables, presented at IARIA CYBER 2025
    </div>
    <details class="paper-abstract">
      This study investigates how access to Large Language Models (LLMs) and varying levels of professional software development experience affect the prioritization of cybersecurity requirements for web applications. Twenty-three postgraduate students participated in a research study to prioritize security requirements (SRs) using the MoSCoW method and subsequently rated their proposed solutions against multiple evaluation criteria. We divided participants into two groups (one with and the other without access to LLM support during the task). Results showed no significant differences related to LLM use, suggesting that access to LLMs did not noticeably influence how participants evaluated cybersecurity solutions. However, statistically significant differences emerged between experience groups for certain criteria, such as estimated cost to develop a feature, perceived impact on user experience, and risk assessment related to non-implementation of the proposed feature. Participants with more professional experience tended to provide higher ratings for user experience impact and lower risk estimates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04261v1">VortexPIA: Indirect Prompt Injection Attack against LLMs for Efficient Extraction of User Privacy</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been widely deployed in Conversational AIs (CAIs), while exposing privacy and security threats. Recent research shows that LLM-based CAIs can be manipulated to extract private information from human users, posing serious security threats. However, the methods proposed in that study rely on a white-box setting that adversaries can directly modify the system prompt. This condition is unlikely to hold in real-world deployments. The limitation raises a critical question: can unprivileged attackers still induce such privacy risks in practical LLM-integrated applications? To address this question, we propose \textsc{VortexPIA}, a novel indirect prompt injection attack that induces privacy extraction in LLM-integrated applications under black-box settings. By injecting token-efficient data containing false memories, \textsc{VortexPIA} misleads LLMs to actively request private information in batches. Unlike prior methods, \textsc{VortexPIA} allows attackers to flexibly define multiple categories of sensitive data. We evaluate \textsc{VortexPIA} on six LLMs, covering both traditional and reasoning LLMs, across four benchmark datasets. The results show that \textsc{VortexPIA} significantly outperforms baselines and achieves state-of-the-art (SOTA) performance. It also demonstrates efficient privacy requests, reduced token consumption, and enhanced robustness against defense mechanisms. We further validate \textsc{VortexPIA} on multiple realistic open-source LLM-integrated applications, demonstrating its practical effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01694v2">Student-AI Interaction in an LLM-Empowered Learning Environment: A Cluster Analysis of Engagement Profiles</a></div>
    <div class="paper-meta">
      📅 2025-10-05
      | 💬 15 pages, 10 figures; Reported on ECER 2025
    </div>
    <details class="paper-abstract">
      Integrating Large Language Models (LLMs) into educational practice enables personalized learning by accommodating diverse learner behaviors. This study explored diverse learner profiles within a multi-agent, LLM-empowered learning environment. Data was collected from 312 undergraduate students at a university in China as they participated in a six-module course. Based on hierarchical cluster analyses of system profiles and student-AI interactive dialogues, we found that students exhibit varied behavioral, cognitive, and emotional engagement tendencies. This analysis allowed us to identify two types of dropouts (early dropouts and stagnating interactors) and three completer profiles (active questioners, responsive navigators, and lurkers). The results showed that high levels of interaction do not always equate to productive learning and vice versa. Prior knowledge significantly influenced interaction patterns and short-term learning benefits. Further analysis of the human-AI dialogues revealed that some students actively engaged in knowledge construction, while others displayed a high frequency of regulatory behaviors. Notably, both groups of students achieved comparable learning gains, demonstrating the effectiveness of the multi-agent learning environment in supporting personalized learning. These results underscore the complex and multifaceted nature of engagement in human-AI collaborative learning and provide practical implications for the design of adaptive educational systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04214v1">Teaching LLM to be Persuasive: Reward-Enhanced Policy Optimization for Alignment frm Heterogeneous Rewards</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      We study deploying large language models (LLMs) as business development (BD) agents for persuasive price negotiation in online travel agencies (OTAs), where aligning traveler affordability and hotel profitability directly affects bookings, partner relationships, and access to travel. The agent must follow a Standard Operating Procedure (SOP) while conducting multi-turn persuasion, interpreting colloquial inputs, and adhering to guardrails (no over-promising, no hallucinations). Conventional post-training -- supervised fine-tuning (SFT) or single-source reward optimization -- overfits scripts, misses nuanced persuasive style, and fails to enforce verifiable business constraints. We propose Reward-Enhanced Policy Optimization (REPO), a reinforcement learning post-training framework that aligns an LLM with heterogeneous rewards: a preference-trained reward model (RM) for dense human alignment, a reward judge (RJ) for high-level persuasive behavior and SOP compliance, and programmatic reward functions (RF) for deterministic checks on numerics, formatting, and guardrails. A straightforward enhancement mechanism is proposed to combine the RM with RJ and RF signals to curb reward hacking and improve negotiation quality. In production-style evaluations -- approximately 150 turns from real dialogues and 225 turns from curated bad-case dialogues -- REPO lifts average dialogue rating to 4.63: +1.20 over base, +0.83 over Direct Preference Optimization (DPO); +0.33 over Group Relative Policy Optimization (GRPO), increases the share of conversations with at least one excellent response to 66.67% (+23.34 percentage points over GRPO), and achieves a 93.33% bad-case fix rate with 75.56% clean fixes, outperforming SFT, DPO, PPO, and GRPO. We also observe emergent capabilities -- proactive empathy, localized reasoning, calibrated tactics -- that surpass gold annotations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12891v3">TIME: A Multi-level Benchmark for Temporal Reasoning of LLMs in Real-World Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-10-05
      | 💬 Accepted by NeurIPS 2025 (Spotlight)
    </div>
    <details class="paper-abstract">
      Temporal reasoning is pivotal for Large Language Models (LLMs) to comprehend the real world. However, existing works neglect the real-world challenges for temporal reasoning: (1) intensive temporal information, (2) fast-changing event dynamics, and (3) complex temporal dependencies in social interactions. To bridge this gap, we propose a multi-level benchmark TIME, designed for temporal reasoning in real-world scenarios. TIME consists of 38,522 QA pairs, covering 3 levels with 11 fine-grained sub-tasks. This benchmark encompasses 3 sub-datasets reflecting different real-world challenges: TIME-Wiki, TIME-News, and TIME-Dial. We conduct extensive experiments on reasoning models and non-reasoning models. And we conducted an in-depth analysis of temporal reasoning performance across diverse real-world scenarios and tasks, and summarized the impact of test-time scaling on temporal reasoning capabilities. Additionally, we release TIME-Lite, a human-annotated subset to foster future research and standardized evaluation in temporal reasoning. The code is available at https://github.com/sylvain-wei/TIME , the dataset is available at https://huggingface.co/datasets/SylvainWei/TIME , and the project page link is https://sylvain-wei.github.io/TIME/ .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04195v1">Constructing coherent spatial memory in LLM agents through graph rectification</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      Given a map description through global traversal navigation instructions (e.g., visiting each room sequentially with action signals such as north, west, etc.), an LLM can often infer the implicit spatial layout of the environment and answer user queries by providing a shortest path from a start to a destination (for instance, navigating from the lobby to a meeting room via the hall and elevator). However, such context-dependent querying becomes incapable as the environment grows much longer, motivating the need for incremental map construction that builds a complete topological graph from stepwise observations. We propose a framework for LLM-driven construction and map repair, designed to detect, localize, and correct structural inconsistencies in incrementally constructed navigation graphs. Central to our method is the Version Control, which records the full history of graph edits and their source observations, enabling fine-grained rollback, conflict tracing, and repair evaluation. We further introduce an Edge Impact Score to prioritize minimal-cost repairs based on structural reachability, path usage, and conflict propagation. To properly evaluate our approach, we create a refined version of the MANGO benchmark dataset by systematically removing non-topological actions and inherent structural conflicts, providing a cleaner testbed for LLM-driven construction and map repair. Our approach significantly improves map correctness and robustness, especially in scenarios with entangled or chained inconsistencies. Our results highlight the importance of introspective, history-aware repair mechanisms for maintaining coherent spatial memory in LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20278v2">Instruction Boundary: Quantifying Biases in LLM Reasoning under Various Coverage</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      Nowadays, automatically generated datasets are increasingly used in LLM reasoning tasks; however, large-scale corpora often contain inherent flaws. For example, a single-choice question may include none or multiple correct options, while true-or-false questions may involve vague or unverifiable statements. We refer to these exceptional answer forms as sparse labels. To compare LLMs' ability to recognize various question forms and produce correct answers, we investigate how different instruction formats can either facilitate or mislead LLM reasoning ability. We introduce the concept of Instruction Boundary, which systematically analyzes how different levels of prompt coverage -- sufficient, redundant, or insufficient -- can lead to reasoning biases and performance changes in LLMs. To examine this phenomenon, we design eight experimental settings across five dataset forms. We further propose BiasDetector, a unified framework that quantifies LLMs' ability to identify sparse labels under different kinds of Instruction Boundary conditions. Evaluations on five mainstream LLMs show that, despite their seemingly high accuracy, substantial reasoning biases persist in many downstream tasks as a direct consequence of prompt coverage. We analyze the impact of these biases and outline possible mitigation strategies. Our findings highlight not only the importance of addressing sparse labels, but also the need for developers to recognize and mitigate the risks introduced by Instruction Boundary.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14474v2">LexiMark: Robust Watermarking via Lexical Substitutions to Enhance Membership Verification of an LLM's Textual Training Data</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can be trained or fine-tuned on data obtained without the owner's consent. Verifying whether a specific LLM was trained on particular data instances or an entire dataset is extremely challenging. Dataset watermarking addresses this by embedding identifiable modifications in training data to detect unauthorized use. However, existing methods often lack stealth, making them relatively easy to detect and remove. In light of these limitations, we propose LexiMark, a novel watermarking technique designed for text and documents, which embeds synonym substitutions for carefully selected high-entropy words. Our method aims to enhance an LLM's memorization capabilities on the watermarked text without altering the semantic integrity of the text. As a result, the watermark is difficult to detect, blending seamlessly into the text with no visible markers, and is resistant to removal due to its subtle, contextually appropriate substitutions that evade automated and manual detection. We evaluated our method using baseline datasets from recent studies and seven open-source models: LLaMA-1 7B, LLaMA-3 8B, Mistral 7B, Pythia 6.9B, as well as three smaller variants from the Pythia family (160M, 410M, and 1B). Our evaluation spans multiple training settings, including continued pretraining and fine-tuning scenarios. The results demonstrate significant improvements in AUROC scores compared to existing methods, underscoring our method's effectiveness in reliably verifying whether unauthorized watermarked data was used in LLM training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04140v1">Selective Expert Guidance for Effective and Diverse Exploration in Reinforcement Learning of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Rewards (RLVR) has become a widely adopted technique for enhancing the reasoning ability of Large Language Models (LLMs). However, the effectiveness of RLVR strongly depends on the capability of base models. This issue arises because it requires the model to have sufficient capability to perform high-quality exploration, which involves both effectiveness and diversity. Unfortunately, existing methods address this issue by imitating expert trajectories, which improve effectiveness but neglect diversity. To address this, we argue that the expert only needs to provide guidance only at critical decision points rather than the entire reasoning path. Based on this insight, we propose MENTOR: Mixed-policy Expert Navigation for Token-level Optimization of Reasoning, a framework that provides expert guidance only at critical decision points to perform effective and diverse exploration in RLVR. Extensive experiments show that MENTOR enables models capture the essence of expert strategies rather than surface imitation, thereby performing high-quality exploration and achieving superior overall performance. Our code is available online.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16199v4">WakenLLM: Evaluating Reasoning Potential and Stability in LLMs via Fine-Grained Benchmarking</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) frequently output the label Unknown in reasoning tasks, where two scenarios may appear: (i) an input sample is genuinely unverifiable, but the model cannot understand why; and (ii) a verifiable problem that the model fails to solve, thus outputs Unknown. We refer to these cases collectively as the Vague Perception phenomenon. Current evaluations focus on whether such answers are honest, rather than analyzing the limits of LLM reasoning. To address this, we introduce WakenLLM, a framework that quantifies the portion of Unknown output attributable to model incapacity and evaluates whether stimulation can convert them into either correct answers (verifiable) or justified (unverifiable) responses with valid reasoning. Our method offers a clearer picture of the limits of LLM reasoning and the potential for corrections across various datasets. Comprehensive experiments on six LLMs suggest that, without any training or parameter revision, LLMs can achieve up to a 68.53% accuracy improvement on Vague Perception samples through guided understanding. Our work reveals that current baseline methods only activate a small portion of LLMs' reasoning potential, indicating considerable unexplored capacity. This extends the theoretical upper bounds of reasoning accuracy in LLMs. Consequently, this study deepens our understanding of the latent reasoning capacity of LLMs and offers a new perspective on addressing the Vague Perception phenomenon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04120v1">Unveiling LLMs' Metaphorical Understanding: Exploring Conceptual Irrelevance, Context Leveraging and Syntactic Influence</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      Metaphor analysis is a complex linguistic phenomenon shaped by context and external factors. While Large Language Models (LLMs) demonstrate advanced capabilities in knowledge integration, contextual reasoning, and creative generation, their mechanisms for metaphor comprehension remain insufficiently explored. This study examines LLMs' metaphor-processing abilities from three perspectives: (1) Concept Mapping: using embedding space projections to evaluate how LLMs map concepts in target domains (e.g., misinterpreting "fall in love" as "drop down from love"); (2) Metaphor-Literal Repository: analyzing metaphorical words and their literal counterparts to identify inherent metaphorical knowledge; and (3) Syntactic Sensitivity: assessing how metaphorical syntactic structures influence LLMs' performance. Our findings reveal that LLMs generate 15\%-25\% conceptually irrelevant interpretations, depend on metaphorical indicators in training data rather than contextual cues, and are more sensitive to syntactic irregularities than to structural comprehension. These insights underline the limitations of LLMs in metaphor analysis and call for more robust computational approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04116v1">Searching Meta Reasoning Skeleton to Guide LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      Meta reasoning behaviors work as a skeleton to guide large language model (LLM) reasoning, thus help to improve reasoning performance. However, prior researches implement meta reasoning skeleton with manually designed structure, limiting ability to adapt to query-specific requirement and capture intricate logical dependency among reasoning steps. To deal with the challenges, we represent meta reasoning skeleton with directed acyclic graph (DAG) to unify skeletons proposed in prior works and model intricate logical dependency. Then we propose AutoMR, a framework that searches for query-aware meta reasoning skeleton automatically inspired by automated machine learning (AutoML). Specifically, we construct search space based on DAG representation of skeleton and then formulate the search problem. We design a dynamic skeleton sampling algorithm by expanding meta reasoning skeleton along with reasoning context at inference time. This algorithm can derive any meta reasoning skeleton in search space efficiently and adapt skeleton to evolving base reasoning context, thus enable efficient query-aware skeleton search. We conduct experiments on extensive benchmark datasets. Experimental results show that AutoMR achieves better reasoning performance than previous works broadly.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04108v1">Can Linear Probes Measure LLM Uncertainty?</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      Effective Uncertainty Quantification (UQ) represents a key aspect for reliable deployment of Large Language Models (LLMs) in automated decision-making and beyond. Yet, for LLM generation with multiple choice structure, the state-of-the-art in UQ is still dominated by the naive baseline given by the maximum softmax score. To address this shortcoming, we demonstrate that taking a principled approach via Bayesian statistics leads to improved performance despite leveraging the simplest possible model, namely linear regression. More precisely, we propose to train multiple Bayesian linear models, each predicting the output of a layer given the output of the previous one. Based on the obtained layer-level posterior distributions, we infer the global uncertainty level of the LLM by identifying a sparse combination of distributional features, leading to an efficient UQ scheme. Numerical experiments on various LLMs show consistent improvement over state-of-the-art baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10714v3">ZSMerge: Zero-Shot KV Cache Compression for Memory-Efficient Long-Context LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      The linear growth of key-value (KV) cache memory and quadratic computational in attention mechanisms complexity pose significant bottlenecks for large language models (LLMs) in long-context processing. While existing KV cache optimization methods address these challenges through token pruning or feature merging, they often incur irreversible information loss or require costly parameter retraining. To this end, we propose ZSMerge, a dynamic KV cache compression framework designed for efficient cache management, featuring three key operations: (1) fine-grained memory allocation guided by multi-dimensional token importance metrics at head-level granularity, (2) a residual merging mechanism that preserves critical context through compensated attention scoring, and (3) a zero-shot adaptation mechanism compatible with diverse LLM architectures without requiring retraining. ZSMerge significantly enhances memory efficiency and inference speed with negligible performance degradation across LLMs. When applied to LLaMA2-7B, it demonstrates a 20:1 compression ratio for key-value cache retention (reducing memory footprint to 5\% of baseline) while sustaining comparable generation quality, coupled with triple throughput gains at extreme 54k-token contexts that eliminate out-of-memory failures. The code is available at https://github.com/SusCom-Lab/ZSMerge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04093v1">Harnessing LLM for Noise-Robust Cognitive Diagnosis in Web-Based Intelligent Education Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      Cognitive diagnostics in the Web-based Intelligent Education System (WIES) aims to assess students' mastery of knowledge concepts from heterogeneous, noisy interactions. Recent work has tried to utilize Large Language Models (LLMs) for cognitive diagnosis, yet LLMs struggle with structured data and are prone to noise-induced misjudgments. Specially, WIES's open environment continuously attracts new students and produces vast amounts of response logs, exacerbating the data imbalance and noise issues inherent in traditional educational systems. To address these challenges, we propose DLLM, a Diffusion-based LLM framework for noise-robust cognitive diagnosis. DLLM first constructs independent subgraphs based on response correctness, then applies relation augmentation alignment module to mitigate data imbalance. The two subgraph representations are then fused and aligned with LLM-derived, semantically augmented representations. Importantly, before each alignment step, DLLM employs a two-stage denoising diffusion module to eliminate intrinsic noise while assisting structural representation alignment. Specifically, unconditional denoising diffusion first removes erroneous information, followed by conditional denoising diffusion based on graph-guided to eliminate misleading information. Finally, the noise-robust representation that integrates semantic knowledge and structural information is fed into existing cognitive diagnosis models for prediction. Experimental results on three publicly available web-based educational platform datasets demonstrate that our DLLM achieves optimal predictive performance across varying noise levels, which demonstrates that DLLM achieves noise robustness while effectively leveraging semantic knowledge from LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04078v1">Bamboo: LLM-Driven Discovery of API-Permission Mappings in the Android Framework</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      The permission mechanism in the Android Framework is integral to safeguarding the privacy of users by managing users' and processes' access to sensitive resources and operations. As such, developers need to be equipped with an in-depth understanding of API permissions to build robust Android apps. Unfortunately, the official API documentation by Android chronically suffers from imprecision and incompleteness, causing developers to spend significant effort to accurately discern necessary permissions. This potentially leads to incorrect permission declarations in Android app development, potentially resulting in security violations and app failures. Recent efforts in improving permission specification primarily leverage static and dynamic code analyses to uncover API-permission mappings within the Android framework. Yet, these methodologies encounter substantial shortcomings, including poor adaptability to Android SDK and Framework updates, restricted code coverage, and a propensity to overlook essential API-permission mappings in intricate codebases. This paper introduces a pioneering approach utilizing large language models (LLMs) for a systematic examination of API-permission mappings. In addition to employing LLMs, we integrate a dual-role prompting strategy and an API-driven code generation approach into our mapping discovery pipeline, resulting in the development of the corresponding tool, \tool{}. We formulate three research questions to evaluate the efficacy of \tool{} against state-of-the-art baselines, assess the completeness of official SDK documentation, and analyze the evolution of permission-required APIs across different SDK releases. Our experimental results reveal that \tool{} identifies 2,234, 3,552, and 4,576 API-permission mappings in Android versions 6, 7, and 10 respectively, substantially outprforming existing baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04076v1">From Shadow to Light: Toward Safe and Efficient Policy Learning Across MPC, DeePC, RL, and LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      One of the main challenges in modern control applications, particularly in robot and vehicle motion control, is achieving accurate, fast, and safe movement. To address this, optimal control policies have been developed to enforce safety while ensuring high performance. Since basic first-principles models of real systems are often available, model-based controllers are widely used. Model predictive control (MPC) is a leading approach that optimizes performance while explicitly handling safety constraints. However, obtaining accurate models for complex systems is difficult, which motivates data-driven alternatives. ML-based MPC leverages learned models to reduce reliance on hand-crafted dynamics, while reinforcement learning (RL) can learn near-optimal policies directly from interaction data. Data-enabled predictive control (DeePC) goes further by bypassing modeling altogether, directly learning safe policies from raw input-output data. Recently, large language model (LLM) agents have also emerged, translating natural language instructions into structured formulations of optimal control problems. Despite these advances, data-driven policies face significant limitations. They often suffer from slow response times, high computational demands, and large memory needs, making them less practical for real-world systems with fast dynamics, limited onboard computing, or strict memory constraints. To address this, various technique, such as reduced-order modeling, function-approximated policy learning, and convex relaxations, have been proposed to reduce computational complexity. In this paper, we present eight such approaches and demonstrate their effectiveness across real-world applications, including robotic arms, soft robots, and vehicle motion control.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04072v1">Slow-Fast Policy Optimization: Reposition-Before-Update for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has become central to enhancing reasoning in large language models (LLMs). Yet on-policy algorithms such as Group Relative Policy Optimization (GRPO) often suffer in early training: noisy gradients from low-quality rollouts lead to unstable updates and inefficient exploration. We introduce Slow-Fast Policy Optimization (SFPO), a simple yet efficient framework to address these limitations via decomposing each step into three stages: a short fast trajectory of inner steps on the same batch, a reposition mechanism to control off-policy drift, and a final slow correction. This reposition-before-update design preserves the objective and rollout process unchanged, making SFPO plug-compatible with existing policy-gradient pipelines. Extensive experiments demonstrate that SFPO consistently improves stability, reduces rollouts, and accelerates convergence of reasoning RL training. Specifically, it outperforms GRPO by up to 2.80 points in average on math reasoning benchmarks. It also achieves up to 4.93\texttimes{} fewer rollouts and a 4.19\texttimes{} reduction in wall-clock time to match GRPO's best accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04064v1">Decoding Emotion in the Deep: A Systematic Study of How LLMs Represent, Retain, and Express Emotion</a></div>
    <div class="paper-meta">
      📅 2025-10-05
      | 💬 10 pages, 7 figures, 4 tables. Under review
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly expected to navigate the nuances of human emotion. While research confirms that LLMs can simulate emotional intelligence, their internal emotional mechanisms remain largely unexplored. This paper investigates the latent emotional representations within modern LLMs by asking: how, where, and for how long is emotion encoded in their neural architecture? To address this, we introduce a novel, large-scale Reddit corpus of approximately 400,000 utterances, balanced across seven basic emotions through a multi-stage process of classification, rewriting, and synthetic generation. Using this dataset, we employ lightweight "probes" to read out information from the hidden layers of various Qwen3 and LLaMA models without altering their parameters. Our findings reveal that LLMs develop a surprisingly well-defined internal geometry of emotion, which sharpens with model scale and significantly outperforms zero-shot prompting. We demonstrate that this emotional signal is not a final-layer phenomenon but emerges early and peaks mid-network. Furthermore, the internal states are both malleable (they can be influenced by simple system prompts) and persistent, as the initial emotional tone remains detectable for hundreds of subsequent tokens. We contribute our dataset, an open-source probing toolkit, and a detailed map of the emotional landscape within LLMs, offering crucial insights for developing more transparent and aligned AI systems. The code and dataset are open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18102v6">How Can I Publish My LLM Benchmark Without Giving the True Answers Away?</a></div>
    <div class="paper-meta">
      📅 2025-10-05
      | 💬 Extended version of the paper presented as an Oral at the ICML 2025 Workshop on the Impact of Memorization on Trustworthy Foundation Models
    </div>
    <details class="paper-abstract">
      Publishing a large language model (LLM) benchmark on the Internet risks contaminating future LLMs: the benchmark may be unintentionally (or intentionally) used to train or select a model. A common mitigation is to keep the benchmark private and let participants submit their models or predictions to the organizers. However, this strategy will require trust in a single organization and still permits test-set overfitting through repeated queries. To overcome this issue, we propose a way to publish benchmarks without completely disclosing the ground-truth answers to the questions, while still maintaining the ability to openly evaluate LLMs. The main underlying idea is to reduces the best possible accuracy, i.e., Bayes accuracy, by injecting randomness to the answers by preparing several logically correct answers, and only include one of them as the solution in the benchmark. Not only is this helpful to keep us from disclosing the ground truth, but this also offers a test for detecting data contamination. In principle, even fully capable models should not surpass the Bayes accuracy. If a model surpasses this ceiling despite this expectation, this is a strong signal of data contamination. We present experimental evidence that our method can detect data contamination accurately on a wide range of benchmarks, models, and training methodologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04056v1">Real-VulLLM: An LLM Based Assessment Framework in the Wild</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      Artificial Intelligence (AI) and more specifically Large Language Models (LLMs) have demonstrated exceptional progress in multiple areas including software engineering, however, their capability for vulnerability detection in the wild scenario and its corresponding reasoning remains underexplored. Prompting pre-trained LLMs in an effective way offers a computationally effective and scalable solution. Our contributions are (i)varied prompt designs for vulnerability detection and its corresponding reasoning in the wild. (ii)a real-world vector data store constructed from the National Vulnerability Database, that will provide real time context to vulnerability detection framework, and (iii)a scoring measure for combined measurement of accuracy and reasoning quality. Our contribution aims to examine whether LLMs are ready for wild deployment, thus enabling the reliable use of LLMs stronger for the development of secure software's.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04048v1">Increasing LLM response trustworthiness using voting ensembles</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      Despite huge advances, LLMs still lack convenient and reliable methods to quantify the uncertainty in their responses, making them difficult to trust in high-stakes applications. One of the simplest approaches to eliciting more accurate answers is to select the mode of many responses, a technique known as ensembling. In this work, we expand on typical ensembling approaches by looking at ensembles with a variable voting threshold. We introduce a theoretical framework for question answering and show that, by permitting ensembles to "abstain" from providing an answer when the dominant response falls short of the threshold, it is possible to dramatically increase the trustworthiness of the remaining answers. From this framework, we derive theoretical results as well as report experimental results on two problem domains: arithmetic problem solving and clinical-note question-answering. In both domains, we observe that large gains in answer trustworthiness can be achieved using highly restrictive voting ensembles, while incurring relatively modest reductions in response yield and accuracy. Due to this quality, voting ensembles may be particularly useful in applications - such as healthcare and data annotation - that require a high degree of certainty but which may not require that every question receive an automated answer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01713v3">SRPO: Enhancing Multimodal LLM Reasoning via Reflection-Aware Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-05
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Multimodal large language models (MLLMs) have shown promising capabilities in reasoning tasks, yet still struggle with complex problems requiring explicit self-reflection and self-correction, especially compared to their unimodal text-based counterparts. Existing reflection methods are simplistic and struggle to generate meaningful and instructive feedback, as the reasoning ability and knowledge limits of pre-trained models are largely fixed during initial training. To overcome these challenges, we propose Multimodal Self-Reflection enhanced reasoning with Group Relative Policy Optimization (SRPO), a two-stage reflection-aware reinforcement learning (RL) framework explicitly designed to enhance multimodal LLM reasoning. In the first stage, we construct a high-quality, reflection-focused dataset under the guidance of an advanced MLLM, which generates reflections based on initial responses to help the policy model learn both reasoning and self-reflection. In the second stage, we introduce a novel reward mechanism within the GRPO framework that encourages concise and cognitively meaningful reflection while avoiding redundancy. Extensive experiments across multiple multimodal reasoning benchmarks, including MathVista, MathVision, MathVerse, and MMMU-Pro, using Qwen-2.5-VL-7B and Qwen-2.5-VL-32B demonstrate that SRPO significantly outperforms state-of-the-art models, achieving notable improvements in both reasoning accuracy and reflection quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05169v1">From Poisoned to Aware: Fostering Backdoor Self-Awareness in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can acquire deceptive behaviors through backdoor attacks, where the model executes prohibited actions whenever secret triggers appear in the input. Existing safety training methods largely fail to address this vulnerability, due to the inherent difficulty of uncovering hidden triggers implanted in the model. Motivated by recent findings on LLMs' situational awareness, we propose a novel post-training framework that cultivates self-awareness of backdoor risks and enables models to articulate implanted triggers even when they are absent from the prompt. At its core, our approach introduces an inversion-inspired reinforcement learning framework that encourages models to introspectively reason about their own behaviors and reverse-engineer the triggers responsible for misaligned outputs. Guided by curated reward signals, this process transforms a poisoned model into one capable of precisely identifying its implanted trigger. Surprisingly, we observe that such backdoor self-awareness emerges abruptly within a short training window, resembling a phase transition in capability. Building on this emergent property, we further present two complementary defense strategies for mitigating and detecting backdoor threats. Experiments on five backdoor attacks, compared against six baseline methods, demonstrate that our approach has strong potential to improve the robustness of LLMs against backdoor risks. The code is available at LLM Backdoor Self-Awareness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03955v1">Harnessing Synthetic Preference Data for Enhancing Temporal Understanding of Video-LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-04
      | 💬 17 pages, 9 figures, 6 tables. Presents TimeWarp, a synthetic preference data framework to improve temporal understanding in Video-LLMs, showing consistent gains across seven benchmarks. Includes supplementary material in the Appendix
    </div>
    <details class="paper-abstract">
      While Video Large Language Models (Video-LLMs) have demonstrated remarkable performance across general video understanding benchmarks-particularly in video captioning and descriptive tasks-they consistently underperform on tasks that require fine-grained temporal understanding. This limitation arises due to the lack of visual complexity and temporal nuance in current fine-tuning datasets, leading these models to rely heavily on language-based reasoning rather than truly understanding video dynamics. In this work, we propose TimeWarp, a systematic method to create a targeted synthetic temporal dataset to fine-tune the model's responses to encourage it to focus on the given input video. We introduce a large-scale preference dataset, created using TimeWarp, that captures intricate temporal dynamics often overlooked, grounding the model's responses to visual and temporal information. We demonstrate that when our method is applied to existing models, it significantly improves performance on temporal understanding benchmarks, highlighting the effectiveness of our proposed datasets in advancing temporal understanding in Video-LLMs, resulting in an absolute improvement in performance across seven benchmarks. Code is available at https://github.com/sameepv21/timewarp.
    </details>
</div>
