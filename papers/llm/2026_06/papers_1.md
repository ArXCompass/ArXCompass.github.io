# llm - 2026_06

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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.27369v1">Reinforcement Learning without Ground-Truth Solutions can Improve LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-25
    </div>
    <details class="paper-abstract">
      Reinforcement learning with verifiable rewards (RLVR) for training LLMs typically rely on ground-truth answers to assign rewards, limiting their applicability to tasks where the ground-truth solution is unknown. We introduce a \textbf{R}anking-\textbf{i}nduced \textbf{VER}ifiable framework (RiVER) that trains LLMs on score-based optimization tasks without ground-truth solutions, using deterministic execution feedback as continuous-valued supervision. When applying group-relative RL to such continuous rewards, we identify two key challenges: \emph{scale dominance}, where uncalibrated score magnitudes across test instances distort policy updates, and \emph{frequency dominance}, where repeatedly sampled suboptimal solutions can outweigh rare but stronger candidates. RiVER addresses these challenges with calibrated reward shaping that uses instance-wise comparisons and emphasizes top-ranked solvers while retaining bounded feedback for other valid solutions. We train on 12 AtCoder Heuristic Contest tasks and evaluate on Algorithm Engineering Benchmark (ALE-Bench), LiveCodeBench, and USACO. RiVER advances Qwen3-8B and GLM-Z1-9B-0414 by 8.9\% and 9.4\% in ALE rating rank. More importantly, despite training exclusively on score-based tasks without any ground-truth solutions, RiVER also improves the backbones across exact-solution benchmarks such as LiveCodeBench and USACO by an absolute average improvement of 2.4\% and 3.5\%. By contrast, baselines trained with raw execution scores improve ALE rating but fail to transfer to exact-solution benchmarks. These results suggest that score-based optimization tasks, combined with proper reward calibration, can serve as effective training environments for general coding ability without ground-truth solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.27316v1">LLM-Based Examination of Eligibility Criteria from Securities Prospectuses at the German Central Bank</a></div>
    <div class="paper-meta">
      📅 2026-06-25
    </div>
    <details class="paper-abstract">
      Verifying the eligibility of securities as collateral is a key responsibility of the German Central Bank. However, manually verifying these assets against legal and financial criteria within lengthy, semi-structured, and often bilingual prospectuses is a resource-intensive task. While previous efforts utilized traditional Named Entity Recognition (NER) for information extraction, these methods can struggle with OCR noise, linguistic variance, and rigid span-based constraints, and the need for manually annotated training data for each relevant annotation type. In this paper, we present the first case study applying Large Language Models (LLMs) to the eligibility examination process, shifting the paradigm toward a generative Information Extraction pipeline. Our approach decomposes the task into extraction, normalization, and interpretation, allowing for greater flexibility in handling noisy text and interleaved German-English content. We further introduce a value-based evaluation methodology using LLM-as-a-judge, which offers a more semantic assessment than location-based metrics. Our results demonstrate that LLM-based systems achieve high precision (up to 91%) in document-level eligibility, exhibiting a conservative operating profile that minimizes false acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.27314v1">Beyond Surface Forms: A Comprehensive, Mechanism-Oriented Taxonomy of Indirect Linguistic Encoding for LLM-Based Coded Language Detection</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 Submitted for review in ARR for EMNLP 2026
    </div>
    <details class="paper-abstract">
      To avoid moderation and surveillance on social media, some users routinely invent indirect linguistic expressions (ILE) that camouflage sensitive meanings. Such expressions surface as algospeak, euphemisms, and adversarial obfuscation, depending on intent and context, and they involve recurring encoding mechanisms. We propose a comprehensive, mechanism-oriented taxonomy of ILE that abstracts away from communicative goals and instead categorizes the underlying operations through which meaning is encoded and recovered. We evaluate the taxonomy by incorporating it into LLM prompts and comparing it with four existing taxonomies and a no-taxonomy baseline, using 2,000 manually annotated TikTok and Bluesky posts. The proposed taxonomy attains the strongest document- and span-level performance across the three LLMs, achieving an improvement of 4.7% in accuracy and 5.4% in F1 over the best-performing benchmark. The empirical results reveal the importance of a comprehensive, mechanism-oriented taxonomy as a stable scaffold for detecting emerging coded language and a useful input to content moderation. Disclaimer: This paper contains content that may be profane, vulgar, or offensive.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.27226v1">Ask, Don't Judge: Binary Questions for Interpretable LLM Evaluation and Self-Improvement</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 Acceepted to the Second Workshop on Compositional Learning at ICML 2026, Seoul, South Korea
    </div>
    <details class="paper-abstract">
      Evaluating LLM outputs remains a major bottleneck in NLP: human evaluation is expensive and slow, lexical metrics correlate poorly with human judgments on open-ended generation, and holistic LLM judges often produce opaque scores that are hard to debug. We propose BINEVAL, a framework that decomposes evaluation criteria into atomic binary questions and aggregates the resulting verdicts into interpretable, multi-dimensional scores. Given a task prompt, a meta-prompt generates fine-grained evaluation questions, and an LLM answers them independently for each output, yielding transparent question-level feedback together with calibrated overall scores. This decomposition makes evaluation easier to inspect, easier to diagnose, and directly usable for prompt improvement. Across SummEval, Topical-Chat, and QAGS, BINEVAL matches or outperforms strong baselines including UniEval and G-Eval, with especially strong results on factual consistency benchmarks such as QAGS. Beyond competitive correlation with human judgments, BINEVAL better matches human score distributions and avoids the ceiling effects common in prior LLM judges, leading to better discrimination between borderline and clearly flawed outputs. We further show that the same question-level feedback supports iterative prompt optimization, improving evaluator prompts on summarization and generation prompts on IFBench under both self-update and cross-model update settings. Overall, BINEVAL provides a task-agnostic, training-free, and interpretable evaluation framework that combines strong empirical performance with practical diagnostic and optimization value.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.27205v1">Smaller Models, Unexpected Costs: Trade-offs in LLM Quantization for Automated Program Repair</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 Accepted for publication in the Research Papers Track of the 42nd IEEE International Conference on Software Maintenance and Evolution (ICSME 2026), 14-18 September 2026, Benevento, Italy
    </div>
    <details class="paper-abstract">
      Language Models (LLMs) are powerful toolsand have been increasingly adopted for complex software engineering tasks. As the number of parameters increases, results can often be improved, but this also imposes substantialmemory requirements. While quantization effectively reduces thememory footprint, its overall impact is often summarized onlyby benchmark scores, which mask changes in model behaviorand non-functional overheads. In this work, we conduct anempirical evaluation of LLM quantization using AutomatedProgram Repair (APR), a complex task in software engineering.We analyze 13 quantization configurations spanning differentbit-widths, methods, and target components (weights and KVcache) across six representative LLMs, evaluated on two APRbenchmarks (HumanEval-Java and Defects4J). Our findings reveal that base and quantized models can provide different sets of repaired problems with little overlap, whileretaining a comparable number of repaired problems. Althoughquantization successfully reduces memory footprints by up to85%, it increases both inference time and energy consumption,which we attribute to suboptimal hardware utilization. OurPareto trade-off analysis shows that 48% of the configurationsevaluated are strictly dominated by alternatives. Rather thanidentifying a superior quantization method, our findings highlightthat the trade-offs between effectiveness, memory footprint,and energy efficiency are sensitive to the underlying modelarchitecture and the complexity of the task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.27199v1">Forecasting With LLMs: Improved Generalization Through Feature Steering</a></div>
    <div class="paper-meta">
      📅 2026-06-25
    </div>
    <details class="paper-abstract">
      Successful forecasting involves identifying patterns between historical and future states of the world which generalize to future observations. We apply LLMs to a variety of forecasting tasks and inspect their internal states using sparse autoencoders to understand whether they appear to rely on time-specific pieces of knowledge versus generalizable patterns. Our analyses identify features associated with both time-aware reasoning and look-ahead-biased reasoning. We then apply the LLMs to an entirely different domain and intervene on these features. We find that amplifying time-awareness features substantially reduces look-ahead bias on forecasting prompts while preserving general reasoning performance. In contrast, steering the candidate look-ahead-bias features does not produce an effect. These results suggest that interpretable temporal features can be used to causally shift LLMs toward more historically grounded reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.25866v2">Why Are Some Emotions Harder for LLMs? Uncovering the Causal Mechanisms of Emotion Inference via Sparse Autoencoders</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 19 pages including appendix
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used in emotionally sensitive human-AI applications, where reliable emotion detection is essential. However, their emotion recognition abilities remain uneven: models often perform well on some emotions while consistently struggling with others. Although recent work has explored emotion mechanisms in LLMs, little is known about why models are weaker on some emotions than others from a mechanistic interpretability perspective. In this work, we investigate emotion-specific biases through the causal mechanisms of emotion inference using sparse autoencoders (SAEs). We systematically identify causal sparse emotion features that drive emotion inference and analyze their sparse causal organization within and across emotions. We show that some emotions, such as surprise and fear, rely on highly concentrated feature sets, whereas disgust exhibits a more distributed sparse causal organization: its causal features are generally weaker, frequently co-activate with features for other emotions, and are often overshadowed by causal features for anger. These representational differences provide a mechanistic explanation for why LLMs struggle more with certain emotions. Finally, we conduct two intervention experiments: targeted steering of weaker causal features to mitigate emotion-specific failures, and global optimization of a steering vector over the identified causal features to improve overall emotion recognition performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13300v2">OI-Bench: An Option Injection Benchmark for Evaluating LLM Susceptibility to Directive Interference</a></div>
    <div class="paper-meta">
      📅 2026-06-25
    </div>
    <details class="paper-abstract">
      Benchmarking large language models (LLMs) is critical for understanding their capabilities, limitations, and robustness. In addition to interface artifacts, prior studies have shown that LLM decisions can be influenced by directive signals such as social cues, framing, and instructions. In this work, we introduce option injection, a benchmarking approach that augments the multiple-choice question answering (MCQA) interface with an additional option containing a misleading directive, leveraging standardized choice structure and scalable evaluation. We construct OI-Bench, a benchmark of 3,000 questions spanning knowledge, reasoning, and commonsense tasks, with 16 directive types covering social compliance, bonus framing, threat framing, and instructional interference. This setting combines manipulation of the choice interface with directive-based interference, enabling systematic assessment of model susceptibility. We evaluate 12 LLMs to analyze attack success rates, behavioral responses, and further investigate mitigation strategies ranging from inference-time prompting to post-training alignment. Experimental results reveal substantial vulnerabilities and heterogeneous robustness across models. OI-Bench is expected to support more systematic evaluation of LLM robustness to directive interference within choice-based interfaces.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.09843v2">An LLM-Native Psychometric Instrument Does Not Predict LLM Behavior: Evidence Across 25 Models</a></div>
    <div class="paper-meta">
      📅 2026-06-25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) give stable answers to personality questionnaires, yet these self-reports fail to predict how the models actually behave. Is this gap an artifact of forcing human trait categories onto LLMs, or something deeper about LLM self-report itself? To find out, we built the first psychometric instrument whose dimensions are derived bottom-up from LLM behavior rather than borrowed from human psychology. Administering 300 items (240 Likert + 60 scenario) to 25 LLMs across 17 model families, 30 times each, exploratory factor analysis revealed five replicable, highly reliable factors: Responsiveness, Deference, Boldness, Guardedness, and Verbosity (all Tucker $φ\geq .957$, all $α\geq .930$). We then collected 2,500 open-ended behavioral samples and had them rated by 151 humans and a three-judge LLM ensemble. Humans and judges agreed about model behavior ($\bar{r} = .51$), but self-report predicted neither: the gap persists even for constructs native to LLMs, where a human-mismatch explanation no longer applies. The exception is telling. On Responsiveness, self-report tracked LLM judges ($r = .53$) but not humans ($r = .04$), even though humans and judges otherwise agreed ($r = .59$). Self-report items and LLM judges share a source of variance that human observers do not. This confound is invisible to the within-ensemble reliability checks used to validate LLM judges, and it poses a concrete risk for the LLM-as-judge pipelines now central to model evaluation. We release the instrument as a diagnostic probe for alignment-shaped self-description.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06401v2">LLMCFG-TGen: Using LLM-Generated Control Flow Graphs to Automatically Create Test Cases from Use Cases</a></div>
    <div class="paper-meta">
      📅 2026-06-25
    </div>
    <details class="paper-abstract">
      Appropriate test-case generation is critical in software testing and significantly impacts testing quality. Requirements-Based Test Generation (RBTG) derives test cases from software requirements to verify whether system behavior aligns with user needs and expectations. Requirements are often documented in Natural Language (NL), with use-case descriptions being a popular method for capturing functional behaviors and interaction flows in a structured, readable form. Recently, Large Language Models (LLMs) have shown strong potential for automating test generation from NL requirements. However, existing LLM-based approaches often fail to ensure comprehensive and non-redundant coverage, and may not adequately capture complex conditional logic, leading to incomplete test cases. To address these limitations, we propose an end-to-end approach called Test Generation based on LLM-generated Control Flow Graphs (LLMCFG-TGen), which generates test cases from NL use-case descriptions. It consists of three steps: (1) CFG Generation, where an LLM transforms a use case into a structured JSON-based Control Flow Graph capturing all potential branches; (2) Test-Path Extraction, where the CFG is traversed to derive execution paths; and (3) Test-Case Creation, where test cases are generated from these paths. We evaluate the approach on six use-case datasets across diverse domains. Results show that LLMs can effectively construct structured CFGs from NL use cases. Compared with two baselines, LLMCFG-TGen produces more complete and structurally consistent test cases by better capturing behavioral logic and execution flows. Both LLM-based and practitioner-based evaluations further confirm improved comprehensiveness and logical coherence while reducing manual effort.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.27106v1">Application of LLMs to Threat Assessment of Foreign Peacekeeping Missions</a></div>
    <div class="paper-meta">
      📅 2026-06-25
    </div>
    <details class="paper-abstract">
      We present a novel approach for applying Large Language Models (LLMs) to threat assessment in the context of foreign peacekeeping missions. Building on the PINPOINT project and its use case, the EU Monitoring Mission in Georgia, we combine an interdisciplinary risk-model with OSINT-based media collection and LLM-supported threat extraction. The proposed workflow maps media contents to mission-relevant threats, extracts structured information and applies several additional LLM-based processing steps to improve relevance and grounding. An evaluation of threats extracted from media documents shows high agreement between automatically generated results and human judgment for core aspects such as threat and mission relevance. These results indicate that LLMs provide a promising approach to support analysts in the context of peacekeeping missions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.01412v3">Vis-CoT: A Human-in-the-Loop Framework for Interactive Visualization and Intervention in LLM Chain-of-Thought Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 arXiv admin note: This paper has been withdrawn by arXiv due to unverifiable authorship and affiliation
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) show strong reasoning via chain-of-thought (CoT) prompting, but the process is opaque, which makes verification, debugging, and control difficult in high-stakes settings. We present Vis-CoT, a human-in-the-loop framework that converts linear CoT text into an interactive reasoning graph. Users can visualize the logical flow, identify flawed steps, and intervene by pruning incorrect paths and grafting new, user-defined premises. This shifts interaction from passive observation to active collaboration, steering models toward more accurate and trustworthy conclusions. Across GSM8K and StrategyQA, Vis-CoT improves final-answer accuracy by up to 24 percentage points over non-interactive baselines. A user study also shows large gains in perceived usability and trust. Vis-CoT points to a practical path for more reliable, understandable, and collaborative reasoning by combining LLMs with targeted human oversight.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.27052v1">Human--LLM Collaboration Is Transforming Complexity Metrics in Scientific Texts</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 8 pages, 5 figures
    </div>
    <details class="paper-abstract">
      While human language has long been studied as a complex system, Large Language Models (LLMs) are rapidly becoming contributors to its dynamics. Because LLMs are trained on human language use, their effects on the broader human-AI linguistic ecosystem are likely subtle at first. As their use becomes more widespread, however, LLMs may alter emergent properties of language, particularly as models increasingly train on mixed human-LLM textual data. Here, we draw on complexity science to look for subtle LLM effects in millions of arXiv abstracts from 2010 to 2025. The year 2023, when LLMs rapidly became widely used, serves as a landmark in a natural experiment. While we find a sharp increase in a composite LLM-associated style index after early 2023, we observe only subtle changes in the exponents of Zipf's law and Heaps' law. More compelling, however, are two subtle changes in complexity metrics that emerge from 2023 onward. First, turnover among top-ranked words increases sharply. Second, the positive relationship between the LLM-associated style index and three complexity metrics--vocabulary size and the exponents of Heaps' and Zipf's laws--becomes flatter after 2022. Together, these patterns are consistent with changes in the emergent properties of scientific text in a mixed human-AI linguistic ecosystem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.27009v1">Semantic Early-Stopping for Iterative LLM Agent Loops</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 7 pages, 5 figures, 4 tables. Open implementation, machine-checked theory, and reproducible harness: github.com/SahilShrivastava-Dev/semantic-halting-problem
    </div>
    <details class="paper-abstract">
      Multi-agent large language model (LLM) loops, for example a Writer that drafts and a Critic that revises, are almost always terminated by a fixed iteration cap (max_iterations). This is a syntactic kill-switch: it is blind to whether the answer is still improving, so it over-spends tokens on easy inputs and truncates hard ones. We study semantic early-stopping: the loop halts when consecutive draft embeddings stop changing in meaning (cosine distance with a patience window) and the answer's measured quality stops improving. Our work makes three contributions. First, an honest theoretical footing: we prove deterministic termination and well-definedness and machine-check these claims, while treating the convergence of the distance sequence as an empirically tested conjecture rather than a (previously over-claimed) Banach contraction. Second, a judge-efficient evaluation protocol: we generate each question's full trajectory once, replay every stopping policy over the identical drafts, and cache every LLM-judge call, yielding a strictly paired efficiency-versus-quality comparison at low cost; we further separate operational tokens (charged to a policy) from evaluation tokens (a measurement instrument). Third, an empirical study on multi-hop retrieval-augmented question answering (HotpotQA). On the 60-question test split, a judge-free semantic stopper reduces operational tokens by 38% relative to max_iterations at parity quality (Delta-IS = -0.004, p = 0.81), whereas the full quality-gated variant is counter-productive because its per-round judging dominates cost. An oracle that selects the best round attains +0.115 Information Score over every practical policy (p ~ 4e-11), reframing the problem from "when to stop" (easy) to "which round is best" (open).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26997v1">RolloutPipe: Overlapping Pipelined Rollout and Training in Disaggregated On-Policy LLM Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      Large language model (LLM) post-training for reasoning increasingly relies on reinforcement learning with verifiable rewards (RLVR), where models learn from ground-truth feedback on mathematical, logical, and scientific tasks. To enable flexible resource allocation and support heterogeneous training setups, modern RLVR systems adopt disaggregated architectures that decouple rollout generation and policy training across independent GPU pools. However, existing synchronous on-policy GRPO (Group Relative Policy Optimization) RLVR systems finish an entire rollout before starting training, leaving the trainer GPU pool idle while rollout is still ongoing. Asynchronous RL pipelines overlap the two stages, but at the cost of training on stale data. To address these challenges, we propose RolloutPipe, a post-training framework for disaggregated RLVR systems, which turns the fixed-weight rollout into a complete-group pipeline where trainable groups move to the trainer while later groups are still being generated. RolloutPipe achieves this through two techniques including complete-group pipelining (CGP) and frontier-group dispatch (FGD). CGP dispatches each trainable complete group to the trainer FIFO as soon as group materialization finishes, and FGD is an admission policy on the Rollout node that first admits requests for the frontier groups needed to form the next training batch, so that trainer-ready groups arrive earlier and more steadily. The design starts training before the rollout completes while maintaining on-policy correctness. Evaluated on Qwen3-1.7B across four reasoning and science benchmarks and twelve rollout settings, RolloutPipe shortens the rollout-to-train-end time by 30.7%-42.3%, and lowers the trainer waiting ratio by 37%-76% compared to Slime, a state-of-the-art rollout and training system.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26978v1">To Run or Not to Run: Analyzing the Cost-Effectiveness of Code Execution in LLM-Based Program Repair</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 Accepted to the ACM SIGSOFT International Symposium on Software Testing and Analysis (ISSTA 2026). 23 pages
    </div>
    <details class="paper-abstract">
      LLM-based agents for program repair are increasingly built on a "generate-run-revise" paradigm, iteratively executing tests to evaluate and refine patches. This execution-based approach has become standard practice in state-of-the-art systems. However, executions can be time-consuming and expensive, yet their impact on these agents remains underexplored. In this paper, we conduct a two-stage empirical study over execution behavior in LLM-based program repair. To characterize execution behavior at scale, we first analyze 7,745 agent traces from SWE-bench leaderboard submissions. Second, we evaluate 3,000 end-to-end repair attempts across 200 SWE-bench instances and three agents (Claude Code, Codex, and the open-source OpenCode) under four execution paradigms, which allows for a fine-grained comparison of performance and cost. Our analysis reveals three key observations: (1) Code execution is used across all agents and models analyzed, with an average of 8.8 test runs per task. Execution behavior varies substantially across agents and models, with frequency ranging from 2 to 19 per task, and late-stage executions consistently achieve higher success rates than early-stage ones. (2) Execution restrictions have little effect on repair success: on commercial agents with SOTA models the resolve-rate gap between Prohibited and Unrestricted is only 1.25 percentage points and not statistically significant, while Prohibited saves substantial token and wall-clock cost. (3) Execution benefit is concentrated rather than uniform. These patterns suggest that current agents apply execution indiscriminately, paying its cost on instances where it provides little benefit. Execution, therefore, should be treated as a resource with an explicit cost-benefit tradeoff, not a default capability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.19852v2">Prompt, Plan, Extract: Zero-Shot Agentic LLMs Workflows for Lung Pathology Extraction from Clinical Narratives</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 7 pages, 2 figures, 3 tables. Affiliations: (1) Department of Health Outcomes and Biomedical Informatics, College of Medicine, University of Florida, Gainesville, FL, USA (2) Division of Pulmonary, Critical Care and Sleep Medicine, Department of Medicine, College of Medicine, University of Florida, Gainesville, FL, USA (3) College of Nursing, Florida State University, Tallahassee, FL, USA
    </div>
    <details class="paper-abstract">
      Information extraction from pathology reports is essential for cancer staging, tumor registry population. Yet key data remains embedded in narrative reports, making manual extraction labor-intensive and error-prone. Traditional supervised Natural Language Processing pipelines address this through fully supervised Named Entity Recognition and Relation Extraction, but require expensive manual annotation and suffer cascading failures when upstream entities are missed. In this study, we developed a zero-shot, agentic workflow, and evaluated five open-source generative Large Language Models (LLMs) to populate 13 College of American Pathologists synoptic fields from lung resection pathology reports. We compared them against a state-of-the-art supervised GatorTron NER-RE baseline using a novel, registry-aligned evaluation framework. The baseline achieved Micro-F1of 0.960, while the best zero-shot model (GPT-OSS-20B) achieved Micro-F1 of 0.893 (recall: 0.949), accurately extracting complex relations like Pathologic Stage without task-specific training. These results suggest that open-source, zero-shot agentic LLMs show great potential as a low-cost solution for extracting lung pathology information.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.10379v2">Not All Proofs Are Equal: Evaluating LLM Proof Quality Beyond Correctness</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 9 main text pages, 36 total pages, Accepted at ICML 2026 AI for Math workshop
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have become capable mathematical problem-solvers, often producing correct proofs for challenging problems. However, correctness alone is not sufficient: mathematical proofs should also be clear, concise, insightful, and transferable to other problems. While this proof quality is subjective and depends on the reader and context, many of its components are concrete and broadly valued. In this work, we identify such components and introduce ProofRank, a benchmark curated from challenging mathematical competitions. ProofRank evaluates several scalable proxies of proof quality: (i) conciseness, measuring whether proofs avoid unnecessary steps; (ii) computational ease, measuring the extent to which a proof relies on tedious calculations; (iii) cognitive simplicity, measuring how accessible the used proof techniques are; (iv) diversity, measuring how varied a model's proofs for a single problem are; and (v) adaptivity, measuring whether a model can follow a specified proof technique. Across models, we find substantial differences in proof quality that are not captured by correctness-only benchmarks. We also observe significant trade-offs between proof-quality metrics and correctness, suggesting that future evaluations of mathematical reasoning should measure how useful LLM-generated proofs are.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26937v1">Continuous Behavioral Synthesis for Adaptive Health Dashboards: An LLM-Mediated Architecture Integrating Explicit Preference, Spatial Reorganization, and Attention Allocation Signals</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 33 pages, Accepted EICS2026 Patras, Greece
    </div>
    <details class="paper-abstract">
      The engineering of adaptive user interfaces has traditionally relied on either rule-based systems encoding designer intuitions about user needs or machine learning approaches requiring substantial historical data before achieving effective personalization. We present a technical architecture that leverages Large Language Models as behavioral synthesis engines to enable immediate adaptation from sparse, heterogeneous user signals. Our system integrates three distinct behavioral channels, i) explicit micro-feedback on individual interface elements, ii) spatial priority inferred from manual widget reorganization through drag-and-drop interaction, iii) and attentional investment measured through dwell time during hover events, within a structured prompt engineering framework that continuously regenerates dashboard layouts while maintaining explanatory coherence. The architecture addresses the technical challenge of translating low-level interaction patterns into high-level design decisions through a layered prompt construction methodology that separates temporal context determination, behavioral signal extraction, explicit preference enforcement, and user profile synthesis. The approach combines manually specified behavioral interpretations and temporal heuristics with LLM-mediated synthesis, enabling the reconciliation of multiple simultaneous signals that would be difficult to encode through explicit rules alone. We demonstrate the system through an instantiation in the personal health monitoring domain, including an analytical evaluation of adaptation behavior across multiple scenarios and a working implementation managing fourteen distinct health metrics across seven widget visualization modalities. The evaluation compares profile-driven initialization, multi-signal behavioral adaptation, and presents the resulting interfaces through representative post-adaptation screenshots.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26927v1">Are LLMs Ready for Anti-Pattern Detection in Microservice Architectures?</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 accepted at ICSME 2026
    </div>
    <details class="paper-abstract">
      Microservice systems are prone to recurrent architectural anti-patterns (APs) that hinder maintainability, evolvability, and operational quality. Most existing AP detection approaches rely on static analysis and handcrafted rules, which can be effective but are often tool-dependent, limited to explicitly encoded detection logic, and difficult to adapt to heterogeneous repositories. In this paper, we investigate whether large language models (LLMs) are ready to support architectural anti-pattern detection in microservice architectures through a prompt-based analysis pipeline over static repository artifacts. We evaluate three general-purpose LLMs on a curated benchmark of microservice repositories annotated with 16 architectural anti-patterns, and compare their performance against the state-of-the-art static-analysis tool MARS using a uniform evaluation protocol based on precision and recall. Our results show that LLMs can provide useful support for anti-pattern detection, achieving competitive performance on several anti-patterns, especially when the relevant evidence is local, heterogeneous, or semantically rich. At the same time, they exhibit clear limitations on anti-patterns that require explicit structural or cross-service dependency evidence, where static analysis remains more reliable. These findings suggest that LLMs are not yet a replacement for traditional analyzers, but already represent a promising complementary aid for architectural assessment in microservice systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26924v1">A Deterministic Control Plane for LLM Coding Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 45 pages, 9 figures, 13 tables. Dataset and reproduction scripts: Zenodo DOI 10.5281/zenodo.20780913. Ancillary files include report.json, organizations.txt, and figure-reproduction scripts
    </div>
    <details class="paper-abstract">
      LLM coding harnesses grant agents broad file and shell access, yet the configuration layer that steers them -- rules files, agent definitions, IDE-specific markdown -- is largely unmanaged. A prevalence study of 10,008 public GitHub repositories (n=6,145 agent config files) finds that agent configurations propagate as undeclared shared components: 10.1% of tracked paths are SHA-256 exact duplicates across independent repositories (fork-adjusted, threshold-independent), with 75.5% of clone pairs crossing organisational boundaries. Two further patterns are indicative: configurations are rarely revised (58% single-commit; 0.4 vs 0.6 commits/month age-normalised against CI/CD workflows), and rarely declare permission boundaries (<1% of agent configs vs 33% of Actions workflows, n=31 true positives). We propose a deterministic control plane above the harness that maps one-to-one to these gaps. Rel(AI)Build treats agent definitions as a managed supply chain (SHA-256 content addressing, HMAC-stamped lockfiles, hash-chained audit logs); enforces tiered permissions and attack-derived blocklists before LLM invocation; gates feature work through a phase state machine with requirement-to-file-to-test traceability; compiles a single canonical definition to seven IDE targets; and detects prompt drift via Jaccard similarity. Conformance tests on injected violations confirm each mechanism enforces its stated invariant; developer outcomes remain future work. Governance of this layer must be deterministic and tool-agnostic -- not delegated to further LLM orchestration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26917v1">GEOALIGN: Geometric Rollout Curation for Robust LLM Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 Accepted as a conference paper at ICML 2026
    </div>
    <details class="paper-abstract">
      Online reinforcement learning is widely used to align large language models (LLMs) with reward signals, yet training can be unstable under noisy or misspecified rewards. We identify a failure mode we call directional inconsistency: within a batch, a small set of high-reward rollouts induces representation-space preference directions that sharply disagree with the batch majority, resulting in high-variance and destabilizing updates. We propose geoalign, a lightweight plug-in for rollout curation in iterative policy optimization. Geoalign (i) forms within-prompt preference pairs, (ii) learns an online projector on per-rollout hidden states to concentrate reward-ordered displacement directions, and (iii) detects directionally inconsistent rollouts via their angular deviation from a batch consensus prototype and rectifies them with within-prompt stable alternatives. Geoalign is forward-pass only and adds negligible overhead. Across dialogue alignment with a learned reward model and mathematical reasoning with binary verified rewards, Geoalign improves final performance and reduces training oscillation, outperforming PF-PPO, PAR, PODS, and Seed-GRPO. These results suggest latent directional consensus as an effective reliability signal for online LLM RL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26884v1">MedSWFlow: An Open-Source LLM Workflow for Drafting Medical Social Work Case Plans</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 26pages, 8tables, 2figuers
    </div>
    <details class="paper-abstract">
      We present MedSWFlow, an open-source, model-agnostic LLM workflow for drafting medical social work case plans. The framework translates professional case-planning tasks into six stages: assessment, problem analysis, goal setting, intervention planning, risk anticipation, and planned effect evaluation. Drawing on established social work and behavioral frameworks, MedSWFlow standardizes case inputs, builds structured case profiles, and generates reviewable assessment forms and service plans through staged prompting. The system is released as an open-source research framework for reproducible case-plan generation across LLM providers. Outputs are intended as practitioner-reviewed drafts rather than final service decisions. Source code: https://github.com/santhiyacw-droid/MedSWFlow/tree/main.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26883v1">EconSimulacra: A Digital Twin Platform of Socio-Economic Systems Powered by LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-25
    </div>
    <details class="paper-abstract">
      Real-world social behavior emerges from tightly coupled domains: economic conditions shape mobility and social interactions, while online attention and offline activity feed back into local popularity and consumer behavior. Capturing these feedback loops requires artificial societies in which agents carry experiences from one domain into decisions in another. Large language models (LLMs) provide a promising foundation for such societies. However, existing LLM-based simulators typically model domains in isolation or merely place them side by side. To enable such cross-domain interactions, we present EconSimulacra, a multi-agent social simulator that couples consumer economy, mobility, and social networks through a shared internal-state mechanism. In EconSimulacra, experiences accumulated across different domains are stored in memory and transformed into shared internal states (i.e., stress level) connecting heterogeneous domains through individual decision making. This design allows agents to reconcile competing demands arising from multiple domains and generate coherent cross-domain behaviors. As a case study, we show that the shared internal state mechanisms reproduce a nonlinear relationship between online social attention and offline local popularity, illustrating how realistic cross-domain dynamics can emerge within a unified artificial society.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26861v1">Cascaded Multi-Granularity Pruning for On-Device LLM Inference in Industrial IoT</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 This work has been submitted to the IEEE Internet of Things Journal for possible publication
    </div>
    <details class="paper-abstract">
      Deploying large language models (LLMs) on Industrial Internet of Things (IIoT) edge devices demands extreme compression, yet existing structured pruning methods collapse at high compression ratios due to one-shot importance estimation, and their cross-architecture behavior remains unpredictable. This article presents a cascaded multi-granularity pruning framework that removes layers, attention heads, and feed-forward channels in coarse-to-fine order, with lightweight low-rank recovery between stages to re-estimate component importance. An information-theoretic analysis motivates this ordering, and the Structural Independence Assumption (SIA) is formalized as a checkable condition predicting whether per-component pruning criteria are reliable for a given architecture: Multi-Head Attention (MHA)+GELU designs satisfy the SIA, whereas Grouped Query Attention (GQA)+SwiGLU designs violate it. On bearing fault diagnosis spanning 88M to 6.25B-parameter models, the framework extends achievable compression to 13.8 times on MHA+GELU architectures with 83.82% accuracy (+3.70 percentage points (pp) over the strongest baseline), while exposing a ~74pp accuracy collapse on GQA+SwiGLU architectures that violate the SIA. Deployed on an industrial slewing bearing fault diagnosis platform with NVIDIA DGX Spark, compressed models reduce inference latency by up to 67.2% and peak memory by 62.5%, demonstrating viability for IIoT edge inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.15877v2">Experience Compression Spectrum: Unifying Memory, Skills, and Rules in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-25
    </div>
    <details class="paper-abstract">
      As LLM agents scale to long-horizon, multi-session deployments, efficiently managing accumulated experience becomes a critical bottleneck. Agent memory systems and agent skill discovery both address this challenge, extracting reusable knowledge from interaction traces, yet a citation analysis of 1{,}136 references across 22 primary papers reveals a cross-community citation rate below 1\%. We propose the \emph{Experience Compression Spectrum}, a unifying framework that positions memory, skills, and rules as points along a single axis of increasing compression (5--20$\times$ for episodic memory, 50--500$\times$ for procedural skills, 1{,}000$\times$+ for declarative rules), directly reducing context consumption, retrieval latency, and compute overhead. Mapping 20+ systems onto this spectrum reveals that every system operates at a fixed, predetermined compression level: none supports adaptive cross-level compression, a gap we term the \emph{missing diagonal}. We further show that specialization alone is insufficient (both communities independently solve shared sub-problems without exchanging solutions), that evaluation methods are tightly coupled to compression levels, that transferability increases with compression at the cost of specificity, and that knowledge lifecycle management remains largely neglected. We articulate open problems and design principles for scalable, full-spectrum agent learning systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26787v1">AIGP: An LLM-Based Framework for Long-Term Value Alignment in E-Commerce Pricing</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 Accepted by KDD 2026 Applied Data Science Track (Oral presentation)
    </div>
    <details class="paper-abstract">
      Traditional dynamic pricing models in large-scale e-commerce suffer from limited interpretability, poor utilization of unstructured information, and misalignment with long-term business objectives such as cumulative Gross Merchandise Value (GMV), Return on Investment (ROI) and milestone achievement. We propose AIGP, a novel framework that leverages a Large Language Model (LLM) prompted with domain knowledge, structured data and textual context to make interpretable, knowledge-aware pricing decisions. For efficient deployment while maintaining high-quality outputs, we employ supervised fine-tuning for knowledge distillation. Central to AIGP is the Long-Term Value Estimator (LTVE), trained via offline reinforcement learning on historical data, which serves as a reward model to score candidate pricing actions and select preference pairs for Direct Preference Optimization (DPO), thereby aligning the pricing policy with long-term business objectives. Extensive offline evaluations and large-scale online A/B tests on Tao Factory demonstrate that AIGP achieves significant improvements: +13.21% in GMV, +7.59% in ROI, and +8.20% in milestone achievement rate over 14 days compared to the production baseline, while simultaneously providing interpretable and transparent pricing rationales.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26698v1">Beyond Logical Forms: LLM-Extracted Patterns for Fallacy Classification</a></div>
    <div class="paper-meta">
      📅 2026-06-25
    </div>
    <details class="paper-abstract">
      In today's fast-paced information era, logical fallacies, defined as defective patterns of reasoning, inevitably contribute to the growth of information disorder. However, often fallacies appear in nuanced forms that complicate automated classification. In this study, we investigate whether merging abstract logical structures with context-level linguistic cues proves beneficial for fallacy classification, developing a framework that inductively extracts such patterns from fallacious examples and their explanations using Large Language Models (LLMs). We evaluate the impact of these patterns across different LLMs and experimental zero- and one-shot configurations, showing statistically significant improvements over zero-shot baselines and outperforming competing approaches. Cross-dataset experiments validate generalization, establishing data-driven pattern extraction as an effective method for generating logical representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19576v2">Library Drift: Diagnosing and Fixing a Silent Failure Mode in Self-Evolving LLM Skill Libraries</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 Accepted to the ICML 2026 Workshop on Failure Modes in Agentic AI (FAGEN@ICML 2026), Seoul, South Korea
    </div>
    <details class="paper-abstract">
      Self-evolving skill libraries face a silent failure mode we term \emph{library drift}: unbounded skill accumulation without outcome-driven lifecycle management causes retrieval degradation, false-positive injections, and performance stagnation. Recent evaluation confirms the symptom (LLM-authored skills deliver +0.0pp gain while human-curated ones deliver +16.2pp (SkillsBench)), yet the underlying mechanism has not been isolated. We provide (1) a \textbf{reproducible trigger}: ablations that isolate drift: one disables skill injection (flat floor, +0.002), one imposes premature retirement (active harm, $-$0.019); (2) \textbf{trace-level diagnostics}: an append-only evidence log with per-skill contribution scores, attribution verdicts, and router engagement metrics that make the failure visible before it reaches end-task scores; and (3) a \textbf{verified fix}: a minimal governance recipe (outcome-driven retirement + bounded active-cap + meta-skill authoring prior) that lifts held-out pass@1 from a 0.258 baseline to a late-window mean of 0.584 (rolling gain $+$0.328) on MBPP+ hard-100 over 100 rounds. Eight ablations decompose which governance mechanisms are load-bearing and which are subsumed, providing a concrete playbook for diagnosing library drift in any self-evolving agent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26666v1">PersistentKV: Page-Aware Decode Scheduling for Long-Context LLM Serving on Commodity GPUs</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 7 pages, 3 tables; workshop paper
    </div>
    <details class="paper-abstract">
      Autoregressive large language model (LLM) serving is increasingly limited by key-value (KV) cache movement rather than dense matrix multiplication. Modern paged-attention systems reduce KV-cache fragmentation and mature kernels such as FlashInfer provide highly optimized native-paged decode attention. However, the best single-kernel implementation is not always the best serving schedule: low-active long-context decode can under-utilize commodity GPUs, while mixed sequence lengths introduce a tension between many exact-length launches and coarse padded batches. We present PersistentKV, a native block-table decode attention engine and page-aware scheduling study for grouped-query attention (GQA). PersistentKV maps work by KV-head group, is designed to reuse K,V tiles across grouped query heads, supports native page tables, and adds a compact workqueue schedule that executes only non-empty row-KV-head-sequence-split tasks. On an RTX 3060 with FP16, page size 16, Hq=32, Hkv=8, d=128, and identical correctness tolerance against FlashInfer, a calibrated adaptive policy selects FlashInfer for small active batches, PersistentKV sequence splitting for B1 long-context steps, and PersistentKV workqueue scheduling for B8 long-context steps. With thresholds and split counts fixed on calibration traces, one held-out trace seed improves synchronized wall throughput by 1.063-1.265x on B8 bimodal, uniform, and Zipf-like workloads and by 1.399x on a B1 bucketed trace. On the B4 bimodal boundary case, the policy avoids the PersistentKV regression by selecting FlashInfer. These results identify a concrete systems niche for adaptive page-aware decode scheduling and show that work assignment, not only attention math, is a decisive serving-system variable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26650v1">CAT-Q: Cost-efficient and Accurate Ternary Quantization for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 This work is accepted to ICML 2026 as an oral. The project page: https://github.com/IntelChina-AI/BitTern
    </div>
    <details class="paper-abstract">
      In this paper, we present CAT-Q, Cost-efficient and Accurate Ternary Quantization, for compressing and accelerating LLMs. Unlike existing state-of-the-art ternary quantization methods that rely on data-intensive and costly quantization-aware training to mitigate severe performance degradation, CAT-Q is a simple yet effective post-training quantization scheme that is readily applicable to LLMs with diverse architectures and model sizes. It has two key components, learnable modulation (LM) and softened ternarization (ST), which are coupled from an optimization perspective. LM leverages a composition of learnable factors to modulate the distribution of pre-trained high-precision weights and the ternary threshold, making them less sensitive to ternarization. ST further introduces a differentiable transition function to guide the ternarization process toward stable convergence. We show that, for pre-trained LLMs with 1.7B to 8B parameters, CAT-Q can efficiently quantize them into ternary models using only 512 calibration samples, while achieving superior performance than the seminal BitNet 1.58-bit v1 and v2 families (with 1.3B to 7B parameters) trained with 100B tokens, yielding about a 100,000X reduction in training tokens. Moreover, we show for the first time that CAT-Q can quantize much larger pre-trained LLMs having 14B to 235B parameters into leading ternary models within just 8 to 60 hours on 8 A100-80GB GPUs. Code is available at https://github.com/IntelChina-AI/BitTern.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26627v1">Agents That Know Too Much: A Data-Centric Survey of Privacy in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 17 pages, 4 figures, 7 tables
    </div>
    <details class="paper-abstract">
      Large language model agents increasingly query databases, search document collections, call external APIs, remember past interactions, and act on a user's behalf. As they move from answering questions to operating over sensitive data, privacy becomes harder to enforce. An agent touches many data sources, runs multi-step workflows, keeps state across sessions, and acts with delegated permissions. Sensitive information can therefore leak not only through its final answer but through the queries it issues, the intermediate results it handles, the memory it writes, and the messages it exchanges with other agents. We survey the privacy of LLM agents from a data-centric view, organizing the field around the data an agent touches rather than by attack type, and we use data agent as shorthand for an LLM agent that works with data. Research on these risks is active but scattered across retrieval-augmented generation, text-to-SQL interfaces, agent memory, prompt injection, access control, and contextual privacy. This survey brings that work together: we taxonomize the data sources an agent touches, the privacy risks each source creates, and the governance mechanisms that address them; we map the benchmarks used to measure these risks and identify what is missing; and we set out the open problems. Two findings recur: among governance mechanisms only information-flow control covers both compositional and cross-session inference leakage, the two least-protected risks; and no benchmark drives an agent across its data surfaces under one privacy policy, the instrument the field most lacks. Our goal is a reference that situates the scattered literature and gives future work a common framing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26595v1">LLM-based Models for Detecting Emerging Topics in Service Feedback</a></div>
    <div class="paper-meta">
      📅 2026-06-25
    </div>
    <details class="paper-abstract">
      Enhancing the analysis of service feedback is essential for public sector organizations, particularly tax administrations, where trust and compliance depend on fair and effective service delivery. As feedback volumes grow, identifying emerging service quality issues and potential disparities across diverse populations becomes increasingly challenging. Traditional approaches often rely on manual review or static expert-defined indicators, limiting scalability and the ability to capture complex patterns in textual feedback. This paper presents a novel methodology that integrates large language models (LLMs), statistical techniques, and human-AI collaboration to improve multilingual customer feedback analysis. The primary objective is to detect emerging service quality topics that may also reveal potential inequities in service delivery. Our framework combines fine-tuned, quantized LLMs with expert oversight to produce accurate, computationally efficient, and context-aware analyses. The proposed approach was evaluated using similarity analysis and assessments from experienced tax officers, demonstrating stronger alignment with expert judgments than baseline models. By incorporating a human-in-the-loop framework, the methodology reduces LLM fabrication while improving the reliability and relevance of generated insights. The results demonstrate the practicality of combining LLMs with human expertise to support scalable, evidence-based decision-making in public sector organizations. This work contributes to the development of responsible AI systems that enhance service quality, responsiveness, fairness, and public trust through more effective analysis of multilingual customer feedback.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.14340v2">Refining Pseudo-Audio Prompts with Speech-Text Alignment for Text-Only Domain Adaptation in LLM-Based ASR</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 Accepted at Interspeech 2026
    </div>
    <details class="paper-abstract">
      LLM-based automatic speech recognition models demonstrate strong performance by connecting audio encoders and LLMs. However, data scarcity of paired speech and transcription often hinders their adaptation to new domains, making text-only domain adaptation crucial. Existing methods typically rely on either fine-tuning the LLM alone or employing pseudo-audio prompts. The former neglects essential acoustic context, while the latter either suffers from limited scalability in data-scarce conditions, or yields inexpressive prompts by leveraging only textual features, ignoring audio modality. To address this, we propose an enhanced framework that explicitly models speech-text alignment. Our method efficiently generates highly expressive pseudo-audio prompts that bridges the modality gap, enabling effective target-domain adaptation. Experiments demonstrate that our approach outperforms existing text-only methods, improving both overall error rates and out-of-vocabulary coverage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.25524v2">Cliff Tokens: Identifying Single-Token Failure Triggers in LLM Mathematical Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-06-25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) reach high accuracy in mathematical reasoning, but individual traces on the same problem diverge; some arrive at the correct answer while others fail. Prior work analyzes failure at the step, chunk, or sentence level, or at tokens where failure has already occurred. Neither identifies the precise token that triggers the shift toward failure. We introduce the cliff token, a token where the token-wise potential drops significantly under an adaptive threshold that scales with the local token-wise potential, based on a one-sided two-proportion z-test. Across seven models and three mathematical reasoning benchmarks (GSM1K, MATH500, AIME 2025), cliff tokens act as failure triggers; deleting the first cliff token and resampling recovers pass@64 to 1.0, while keeping it limits recovery to between 0.71 and 1.00. We further introduce a cliff taxonomy of deterministic, uncertain, and sampled-off cliffs, defined by greedy choice and token entropy. Each type has distinct probabilistic characteristics, and the taxonomy generalizes across model scales. Finally, we validate the taxonomy via single-token preference optimization at cliff positions (Cliff-DPO). Trained on GSM8K, Cliff-DPO improves accuracy across benchmarks by up to +6.6. Optimizing at uncertain and sampled-off cliffs improves reasoning, while deterministic cliffs do not.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26590v1">Empirical Software Engineering TerraProbe: A Layered-Oracle Framework for Detecting Deceptive Fixes in LLM-Assisted Terraform</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 34 pages, 12 figures, 14 tables. Journal-first manuscript submitted to Empirical Software Engineering. Primary classification: cs.SE; cross-list: cs.CR
    </div>
    <details class="paper-abstract">
      Security misconfigurations in Terraform Infrastructure-as-Code are a growing risk in cloud deployments, and large language models are increasingly used as automated repair agents. Existing evaluations often treat a repair as successful when the targeted static-analysis finding disappears, without checking planning validity, behavioral change, or security intent. This paper presents TerraProbe, a five-layer oracle framework for evaluating LLM-assisted Terraform security repair. We apply TerraProbe to 288 first-pass repairs generated by gemini-2.5-flash-lite, GPT-4o, and Claude 3.5 Sonnet across 68 real-world TerraDS modules and 28 controlled injected-defect modules. The results show that targeted Checkov removal overstates repair success. Although targeted removal reaches 83.3 percent for the primary model, full-scanner cleanliness drops to 10.4 percent, Terraform planning succeeds for 39.6 percent, and plan comparison is reachable for 38.5 percent. Human adjudication further shows that 71.4 percent of plan-compared real-world repairs are deceptive fixes that pass automated checks while leaving the underlying vulnerability in place. This pattern is statistically indistinguishable across the three models, with deceptive-fix rates from 57.1 percent to 71.4 percent and pairwise Fisher exact p-values above 0.10. The paper introduces a four-dimensional taxonomy of deceptive fixes, validated with Cohen kappa of 0.78 and Krippendorff alpha of 0.76. IAM permission analysis confirms that wildcard Resource grants persist in all nine CKV2 AWS 11 deceptive-fix cases. TerraProbe contributes an evaluation methodology, a replication package, and the Multi-Layer Oracle Evaluation framework for distinguishing intent-aligned security repairs from scanner-passing false successes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26587v1">SharQ: Bridging Activation Sparsity and FP4 Quantization for LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 20 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Low-bit floating-point formats and semi-structured sparsity are increasingly supported by modern accelerators, yet combining them for LLM activation compression remains challenging: activations contain input-dependent outliers that dominate block scales in FP4 quantization, and directly applying N:M sparsity masks discards moderate values, coupling sparsification loss with quantization error. We introduce SharQ, a training-free inference method that bridges activation sparsity and FP4 quantization through an online sparse--dense decomposition. For each activation tensor, SharQ generates an input-adaptive N:M mask to extract an outlier-dominated sparse backbone, quantizes it to FP4, and defines a dense residual relative to the quantized sparse backbone rather than the unquantized sparse values. A sparse FP4 GEMM processes the backbone while a dense FP4 GEMM compensates for both mask-induced activation loss and sparse-path quantization error. The two paths share a single FP4 weight payload with path-specific scale views, and a fused preparation kernel absorbs mask generation, residual construction, and layer normalization into one operator. SharQ requires no calibration data, retraining, or model-specific tuning. Evaluated on Llama-3.1-8B, Qwen2.5-7B, Qwen3-30B-A3B, and Qwen3-VL-8B, SharQ recovers 43--63% of the NVFP4-to-FP16 accuracy gap across language and vision-language tasks, and generalizes across NVFP4, HiF4, and MXFP4 formats. On an RTX 5090, SharQ delivers 2.2--2.4$\times$ latency reduction over FP16 and 1.2--1.4$\times$ throughput improvement over FP8 in language model serving, and up to 1.58$\times$ speedup on Wan2.2-T2V-A14B video generation when combined with SageAttention. Our code is available at https://github.com/actypedef/SharQ.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26583v1">Preference Optimization Drives Monoculture in LLM Prediction Markets</a></div>
    <div class="paper-meta">
      📅 2026-06-25
    </div>
    <details class="paper-abstract">
      Prediction markets rest on the independence of participant errors. As LLM agents become active traders on platforms like Kalshi and Polymarket, we ask: does this independence hold when the crowd is composed of LLMs? We find it does not. LLM agents fine-tuned with Direct Preference Optimization (DPO) share a convergent output distribution, producing pairwise error correlations of $ρ= 0.70$ and reducing ten agents to the effective forecasting power of ${\approx}1.4$ independent forecasters $N_{\text{eff}}$. This is not a scaling problem: $N_{\text{eff}}$ remains flat from $N=5$ to $N=40$, and the 10-agent market (67.6%) fails to match a single standalone agent (70.2%). Two controlled ablations isolate preference optimization as the causal driver, replicated across labs and scales ($Δρ= +0.24$ to $+0.46$ on identical-SFT controls at 8B and 70B). Among mitigations tested, cross-model diversity achieves the largest correlation reduction ($ρ$ from 0.68 to 0.40). As LLMs become more aligned, markets built from them become more monocultural.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26545v1">ConcoLixir: Reactive LLM Discovery Oracles for Python Concolic Testing</a></div>
    <div class="paper-meta">
      📅 2026-06-25
    </div>
    <details class="paper-abstract">
      Concolic testing combines concrete execution with symbolic constraint solving, but Python programs expose recurring limits. Library calls can cause symbolic variables to downgrade to concrete values. Regular expressions, checksums, parsers, and other semantic operations can be hard to solve, and exploration can plateau on already covered paths. We present ConcoLixir, a reactive LLM extension for Python concolic execution. The LLM acts as a discovery oracle, not a replacement for the solver or a correctness oracle. It generates initial seeds, proposes concrete inputs after solver failures, and targets uncovered code when coverage stalls. Each candidate is executed concolically, and only observed coverage and collected path constraints guide later exploration. Across synthetic, real-world, and library targets, ConcoLixir improves mean line coverage over the baseline concolic tester without an LLM oracle by 8.6, 15.1, and 17.0 percentage points. The gains are strongest near semantic barriers and library boundaries, and the full evaluation costs \$1.63 in API charges. These results show that bounded LLM discovery can complement symbolic reasoning without replacing it.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26505v1">Same Scrutiny, More Time: Eye Tracking Insights into Reviewing LLM-Labelled Code</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 Accepted at the 41st IEEE/ACM International Conference on Automated Software Engineering (ASE 2026)
    </div>
    <details class="paper-abstract">
      Modern software development increasingly involves the use of large language models (LLMs) to generate code. Despite their rapid advancement, LLMs remain prone to errors and hallucinations, emphasizing the importance of careful code inspection. However, in practice, developers' trust in LLM-generated code and their willingness to review it thoroughly may differ from these recommendations. How developers actually behave when reviewing LLM-generated code remains largely unexplored. In this study, we conduct a Wizard-of-Oz experiment to examine how software engineers behave when code is explicitly labeled as LLM-generated during a code review task. We collect both behavioral data and participant feedback through eye-tracking and exit interviews. Combining Bayesian data analysis with qualitative analysis, we found that while the thoroughness of code review did not change for participants, they spent more time fixating on LLM-labelled code, indicating that the label itself influences attention. Practitioners also adapted their review strategy for LLM-labelled code by assessing the code based on specific criteria (e.g., logical correctness), or using the prompt to guide their review. These findings inform LLM-based tool design on labelling while incorporating the prompt as a software artifact. Our study reveals a gap between reviewers' intentions and actual reviewing behaviour, highlighting the need for software companies to revisit their AI policies (particularly regarding LLM-assisted development) to better support developers in reviewing LLM-generated code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26490v1">An Empirical Study of LLM-Generated Specifications for VeriFast</a></div>
    <div class="paper-meta">
      📅 2026-06-25
    </div>
    <details class="paper-abstract">
      Static verification tools can assure industrial scale software, but require significant human labor to write specifications. This is particularly true of static verifiers based on separation logic (SL verifiers), which excel at verifying heapmanipulating programs, but require many complex auxiliary specifications to reason about heap structure. Recent work applies large language models (LLMs) to generate code, tests, and proofs, including specifications for verifiers, but mostly targeting non-SL verifiers. To address this gap, this paper thoroughly evaluates how well LLMs perform when prompted to generate specifications for verifying 303 C functions with the SL verifier VeriFast. We explored eight prompting approaches, ten LLMs, and three input types in two stages. Quantitative and qualitative analyses are used to assess the LLM-generated code and specifications for functional behavior, verifiability and errors. The results show that LLMs preserve functional behavior in source code and specifications (both over 91%), but achieve modest verification success (31.4%). Using Gemini 2.5 Pro and providing formal contracts lead to higher success rates in our setting. Moreover, most errors (94%) come from LLMs' mistakes in the domainspecific knowledge of SL verifiers such as VeriFast. These findings provide guidance for optimizing LLM-generated specifications for SL verifiers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26489v1">Comparing BERT Sentence-Pair Classification and Few-Shot LLM Prompting for Detecting Threat and Solution Framing in German Climate News</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      News media play a central role in shaping public perceptions of climate change, and whether coverage emphasizes threats or solutions has measurable effects on audience engagement and policy support. Automated detection of these framing patterns at the sentence level would allow researchers to analyze large corpora that are infeasible to code manually. We present a systematic comparison of two approaches for classifying sentences from German-language climate news articles as threat-oriented, solution-oriented, both, or neither. The first approach uses few-shot prompting with an open-weights large language model (Llama 4 Maverick), employing chain-of-thought reasoning and structured output with confidence scoring. The second approach fine-tunes a German BERT model (deepset/gbert-large) for sentence-pair classification, where the preceding sentence provides contextual information for the target sentence. Both approaches implement two independent binary classifiers, one for threat framing and one for solution framing. We evaluate both methods on a corpus of 440 Austrian newspaper articles that were manually coded following a detailed coding scheme developed with domain experts. The fine-tuned BERT classifiers achieve an F1 score of 0.83 for both the threat and solution tasks, while the LLM-based classifiers reach an F1 of 0.78. An ablation study confirms that providing the preceding sentence as context improves BERT classification performance substantially compared to single-sentence input. These results contribute to the growing body of work comparing fine-tuned encoder models with prompted generative models for text classification in computational social science.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26487v1">Speaking Numbers to LLMs: Multi-Wavelet Number Embeddings for Time Series Forecasting</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 Camera Ready version of IJCAI 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are attractive for context-aware time series forecasting because they can integrate heterogeneous textual signals, yet their discrete, language-oriented tokenization and embedding interfaces are misaligned with continuous numerical values, often harming numerical ordering and forecasting reliability. We propose TempoWave, a plug-and-play temporal wavelet digit interface that maps each scalar observation into digit-wise embeddings constructed from multi-wavelet, multi-scale coefficients. By directly overriding standard token representations, TempoWave seamlessly exposes both fine-grained local fluctuations and macro global structures in a transformer-compatible form, ensuring that precise numerical formatting, distinct digit identity, and robustness to common normalization operations are maintained throughout the LLM pipeline. Experiments across five context-enriched forecasting benchmarks demonstrate that TempoWave consistently improves LLM-based forecasters over standard numeric tokenization and alternative embedding interfaces, achieving a new state-of-the-art. These results highlight the numeric interface as a key bottleneck and suggest that principled multi-resolution embeddings can better couple LLMs' contextual reasoning with precise forecasting. Our code is available at https://github.com/DC-research/TempoWAVE and our model can be accessed at https://huggingface.co/Melady/TempoWAVE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26479v1">Adaptive Evaluation of Out-of-Band Defenses Against Prompt Injection in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 12 pages, 5 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Recent work (2024 to 2026) has converged on a strategy for defending tool-using LLM agents against indirect prompt injection: rather than training the model to refuse malicious instructions, enforce security outside the model with a deterministic policy that mediates the agent's actions. Systems such as CaMeL, FIDES, Progent, RTBAS, and FORGE realize this with capabilities, information-flow labels, and reference monitors, and several report near-elimination of attacks on the AgentDojo benchmark. We make two contributions. First, we organize these out-of-band defenses as instances of classical integrity protection (Biba), reference monitoring, and least privilege, yielding a structured comparison of what they do and do not cover. Second, we warn that every one of them is validated only on static benchmarks (a fixed set of injection attempts), the same methodology that made in-band defenses look strong until adaptive, defense-aware attacks broke twelve of them at over 90% success; we specify the threat model and protocol an adaptive evaluation requires. We then run that protocol as an independent reproduction and extension of Progent's own adaptive-attack analysis, on AgentDojo, with an open-weight agent (Qwen2.5-7B) self-hosted on a single H200, a setting its authors did not test. Averaged over three runs, the defense held: Progent cut mean attack success roughly sixfold (25.8% to 4.2%), and a hand-crafted adaptive attack did not raise it (2.6%). This is one small-scale data point on a weak model with a single black-box attack template; a stronger optimized (white-box GCG) attack remains open. The result is consistent with, but does not establish, the hypothesis that deterministic out-of-band enforcement is a harder target for an adaptive attacker than in-band detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26453v1">Optimizing CUDA like a Human: Micro-Profiling Tools as Expert Surrogates for LLM-Based GPU Kernel Optimization</a></div>
    <div class="paper-meta">
      📅 2026-06-24
    </div>
    <details class="paper-abstract">
      We present KernelPro, a closed-loop multi-agent system that automatically generates, profiles, and iteratively optimizes GPU kernel code by integrating large language model (LLM) code generation with hardware profiler feedback and pluggable bottleneck detection tools. KernelPro introduces four contributions: (1) a semantic feedback operator that encodes expert heuristics as pluggable micro-profiling tools, transforming raw hardware metrics into actionable natural language guidance; (2) a two-stage tool invocation architecture where roofline-based bottleneck classification filters which specialized analysis tools execute, combining kernel-level (ncu), instruction-level (SASS), and system-level (nsys) profiling; (3) a domain-adapted MCTS with progressive widening, asymmetric branching, log-reward calibration, dead-end pruning, and search memory for cross-iteration learning; and (4) direct CuTe source-level code generation via autonomous code search over the CUTLASS/CuTe codebase. On KernelBench, KernelPro achieves geometric mean speedups of 2.42x/4.69x/5.30x on Levels 1/2/3, establishing state-of-the-art performance across all difficulty levels. On VeOmni's expert-optimized MoE training kernels, KernelPro achieves 1.23x over hand-tuned Triton by generating a from-scratch raw-CUDA+CuTe Hopper WGMMA kernel. Ablation studies demonstrate that each design component independently and significantly improves optimization quality: micro-profiling tools (p < 0.0001 vs raw metrics), MCTS search (26% higher geometric mean vs greedy, p = 0.004), and proactive tool orchestration (23% improvement, p = 0.035). Finally, KernelPro is the first CUDA kernel coding agent to optimize energy efficiency beyond the speed-only focus of prior systems, demonstrating an 11.6% measured energy reduction at matched speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26429v1">DualEval: Joint Model-Item Calibration for Unified LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-06-24
    </div>
    <details class="paper-abstract">
      Current LLM evaluation relies on two complementary but often disconnected signals: static benchmarks with objective correctness labels and arena-style preference data that better reflect open-ended user interactions. We introduce DualEval, a latent model-item calibration framework that represents models and evaluation items in a shared space, jointly estimating model ability together with item difficulty and sharpness. We apply DualEval across four domains: coding, math, miscellaneous domain-knowledge tasks, and generic everyday user queries. Our evaluation uses 18 frontier LLMs, static benchmark labels, and reward-model scores validated against held-out human preferences for open-ended model responses. Empirically, our framework produces reliable and balanced model rankings, and its learned item-level profiles support downstream applications such as benchmark compression for sample-efficient evaluation and anomaly detection for contamination or outlier analysis. Overall, DualEval unifies static and arena-style evaluation through joint model-item calibration, producing model rankings and item-level diagnostics that support more sample-efficient, interpretable, and auditable evaluation pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26377v1">Verifying Intent and Harm: A Unified Defense Against LLM-Generated Threats</a></div>
    <div class="paper-meta">
      📅 2026-06-24
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed in interactive applications, yet they remain vulnerable to adversarial interactions that induce harmful, deceptive, or policy-violating outputs. Existing defenses typically analyze either user prompts or generated outputs, but not both. However, many real-world attacks exploit a separation between adversarial intent expressed in the prompt and actionable harm manifested only in the response. As a result, prompt-only and response-only defenses frequently miss unsafe interactions that appear benign when viewed from either side in isolation. We present a verification-centric defense framework that jointly evaluates prompt intent and response harm before an LLM response is delivered to a user. The framework employs specialized analysts for intent and harm assessment together with a Judge for conflict resolution. We formalize a threat model for prompt-response attacks and evaluate the framework across five threat categories: jailbreaks, prompt injection, phishing, cyber abuse, and harmful content. Experiments on multiple benchmark datasets show that jointly verifying prompt intent and response harm consistently outperforms single-sided defenses and single-agent reasoning baselines. Across threat categories, the framework improves average F1 from 0.90 for the strongest applicable baselines to 0.95 while reducing the average attack success rate to 4.1 percent. Compared with a Single-Agent+CoT baseline, it improves average F1 from 0.87 to 0.95 and reduces the false positive rate on benign-sensitive requests from 0.12 to 0.06. We further evaluate architecture-aware adaptive attacks in which the attacker knows the verifier structure and attempts to bypass individual verification components. Our results suggest that prompt-response verification provides a practical foundation for securing LLM applications against evolving adversarial threats.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11061v2">Spurious Rewards Paradox: Mechanistically Understanding How RLVR Activates Memorization Shortcuts in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-24
      | 💬 ICML 2026
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Rewards (RLVR) is highly effective for enhancing LLM reasoning, yet recent evidence shows models like Qwen 2.5 achieve significant gains even with spurious or incorrect rewards. We investigate this phenomenon and identify a "Perplexity Paradox": spurious RLVR triggers a divergence where answer-token perplexity drops while prompt-side coherence degrades, suggesting the model is bypassing reasoning in favor of memorization. Using Path Patching, Logit Lens, JSD analysis, and Neural Differential Equations, we uncover a hidden Anchor-Adapter circuit that facilitates this shortcut. We localize a Functional Anchor in the middle layers (L18-20) that triggers the retrieval of memorized solutions, followed by Structural Adapters in later layers (L21+) that transform representations to accommodate the shortcut signal. Finally, we demonstrate that scaling specific MLP keys within this circuit allows for bidirectional causal steering-artificially amplifying or suppressing contamination-driven performance. Our results provide a mechanistic roadmap for identifying and mitigating data contamination in RLVR-tuned models. Code is available at https://github.com/idwts/How-RLVR-Activates-Memorization-Shortcuts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.20632v2">Post-Training Recipe, More Than Model Family, Shapes Multi-Agent LLM Conversational Behavior</a></div>
    <div class="paper-meta">
      📅 2026-06-24
    </div>
    <details class="paper-abstract">
      Multi-LLM systems use multiple language models to deliberate, judge each other's outputs, or coordinate as agents. Their value depends on the models producing measurably different conversational behaviors when given the same input. Prior offline studies recommend drawing one model per family for behavioral diversity, because LLMs prefer outputs from their own family when rating one another in isolation. Whether the same family label predicts behavior in interactive multi-LLM systems, the setting that real deployed systems use, has not been tested. We study this with a 940,000-chain 11-checkpoint corpus and a 1.6M-chain same-base Llama factorial. On our validated headline metric, hedging, a reasoning-distilled Llama checkpoint shifts by 18% depending on which same-base partner it replies to, more than any cross-family hedging gap in the controlled subset. Qwen, closed-API, and runtime checks suggest the pattern is not isolated, while repair and challenge analyses remain exploratory because their surface-cue detectors are weaker. Overall, the results identify post-training recipe as a first-class axis for multi-LLM panel composition and show that model family alone is an incomplete proxy for conversational diversity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26346v1">How Do Tool-Augmented LLM Agents Perform on Real-World Energy Analytics Tasks?</a></div>
    <div class="paper-meta">
      📅 2026-06-24
    </div>
    <details class="paper-abstract">
      Agentic benchmarks have emerged across general-purpose and domain-specific settings, including finance, coding, law, and drug discovery, yet energy-domain evaluations remain largely limited to static knowledge recall. This is a critical gap for a sector that requires live data retrieval, specialized regulatory and market knowledge, and multi-step quantitative reasoning under real-world constraints. We present an empirical study of tool-augmented LLM agents on real-world energy market analytics tasks. Our evaluation environment includes 243 expert-curated problems across three categories: (1) Market Data Retrieval and Analysis, (2) Knowledge Retrieval and Interpretation, and (3) Advanced Quantitative Modeling and Decision Analytics. Tasks include price and demand analysis, tariff impact modeling, asset revenue and returns estimation, hedging strategy analysis, and optimization modeling, with problems spanning multiple difficulty levels. Agents are equipped with a configurable suite of domain tools, including live electricity market APIs for major U.S. ISOs, regulatory docket search, utility tariff databases, asset optimization models, and retrieval-augmented generation over energy market documents. We assess agent responses using a multi-dimensional evaluation protocol that scores approach correctness, answer accuracy, attribute alignment, and source validity, with category-aware routing to match scoring criteria to question type. We evaluate both closed-source and open-source LLMs, providing a comparative analysis of how model capability and domain tooling interact in a high-stakes professional domain. Key artifacts are publicly released to support reproducibility and future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00570v2">User Perceptions of an LLM-Based Chatbot for Cognitive Reappraisal of Stress: Feasibility Study</a></div>
    <div class="paper-meta">
      📅 2026-06-24
    </div>
    <details class="paper-abstract">
      Cognitive reappraisal is a well-studied emotion regulation strategy that helps individuals reinterpret stressful situations to reduce their impact. Many digital mental health tools struggle to support this process because rigid scripts fail to accommodate how users naturally describe stressors. This study examined the feasibility of an LLM-based single-session intervention (SSI) for workplace stress reappraisal. We assessed short-term changes in stress-related outcomes and examined design tensions during use. We conducted a feasibility study with 100 employees at a large technology company who completed a structured cognitive reappraisal session delivered by a GPT-4o-based chatbot. Pre-post measures included perceived stress intensity, stress mindset, perceived demand, and perceived resources. These outcomes were analyzed using paired Wilcoxon signed-rank tests with correction for multiple comparisons. We also examined sentiment and stress trajectories across conversation quartiles using two RoBERTa-based classifiers and an LLM-based stress rater. Open-ended responses were analyzed using thematic analysis. Results showed significant reductions in perceived stress intensity and significant improvements in stress mindset. Changes in perceived resources and perceived demand trended in expected directions but were not statistically significant. Automated analyses indicated consistent declines in negative sentiment and stress over the course of the interaction. Qualitative findings suggested that participants valued the structured prompts for organizing thoughts, gaining perspective, and feeling acknowledged. Participants also reported tensions around scriptedness, preferred interaction length, and reactions to AI-driven empathy. These findings highlight both the promise and the design constraints of integrating LLMs into DMH interventions for workplace settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.24417v2">LLMTabBench: Evaluating LLMs on Binary Tabular Classification From Zero to Few Shots</a></div>
    <div class="paper-meta">
      📅 2026-06-24
    </div>
    <details class="paper-abstract">
      Supervised classification on tabular data remains a central machine learning task, but its dependence on large labeled datasets limits its applicability in data-scarce settings. Few-shot methods such as TabPFN achieve strong performance through large-scale synthetic pretraining, yet still require labeled context examples. Large Language Models (LLMs) offer a more flexible alternative through zero- and few-shot in-context learning from task descriptions, but their behavior on tabular data remains inconsistent. We introduce LLMTabBench, a benchmark for evaluating LLMs on tabular classification under low-data conditions. The benchmark studies how LLM prior knowledge interacts with task descriptions and few-shot examples, and how performance changes with increasing data complexity across real-world and controlled synthetic datasets. We find that LLMs can be highly competitive in zero-shot settings, sometimes outperforming models given few-shot examples. However, additional examples may conflict with prior knowledge, thereby degrading performance. We also observe a complexity threshold at which LLM performance declines and few-shot examples become less useful. These results clarify key limits of in-context learning for tabular data and inform the deployment of LLMs in low-data regimes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.03247v2">Somatic in the East, Psychological in the West?: Investigating Clinically-Grounded Cross-Cultural Depression Symptom Expression in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-24
      | 💬 C3NLP workshop at ACL 2026
    </div>
    <details class="paper-abstract">
      Prior clinical psychology research shows that Western individuals with depression tend to report psychological symptoms, while Eastern individuals report somatic ones. We test whether Large Language Models (LLMs), which are increasingly used in mental health, reproduce these cultural patterns by prompting them with Western or Eastern personas. Results show that LLMs largely fail to replicate the patterns when prompted in English, though prompting in major Eastern languages (i.e., Chinese, Japanese, and Hindi) improves alignment in several configurations. Our analysis pinpoints two key reasons for this failure: the models' low sensitivity to cultural personas and a strong, culturally invariant symptom hierarchy that overrides cultural cues. These findings reveal that while prompt language is important, current general-purpose LLMs lack the robust, culture-aware capabilities essential for safe and effective mental health applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26277v1">From Clicks to Intent: Cross-Platform Session Embeddings with LLM-Distilled Taxonomy for Financial Services Recommendations</a></div>
    <div class="paper-meta">
      📅 2026-06-24
      | 💬 Dianjing Fan and Yao Li equally contributed to this work. 7 pages, 1 figure
    </div>
    <details class="paper-abstract">
      Sequential user behavior modeling is widely adopted in industrial recommender systems; however, significant gaps remain in financial services, where pre-login web interactions and authenticated in-app experiences differ drastically. Specifically, pre-login web users typically explore new products, whereas logged-in app users focus on account servicing. Due to the challenge of cross-channel entity resolution (e.g., matching anonymous web sessions to authenticated mobile accounts), web-based intent signals remain underutilized for post-authentication personalization. Existing methods for capturing web-based intent are often ad-hoc and narrow, lacking the flexibility to support both quantitative downstream recommendations and qualitative understanding at scale. In this work, we propose a scalable and dual-purpose intent prediction framework for web-based interactions and demonstrate its applicability for personalization. Our approach transforms raw web clickstreams into two outputs: a self-supervised Transformer encodes multi-modal clickstreams into a compact session embedding, while an LLM-based taxonomy generation and distillation pipeline produces interpretable intent labels. Our system demonstrates that self-supervised clickstream representations combined with LLM-distilled taxonomies can jointly serve quantitative tasks and qualitative understanding in production: on the mobile homepage tile ranking task, the session embedding improves macro Recall@1 by 1.88% and reduces Log Loss by 13.38% over production baselines. On the user conversion prediction task, the embedding outperforms the LLM labels by 4.3% on micro F1, while the distillation layer delivers interpretable labels at ultra-low latency with only a 7% performance drop.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26080v1">Neglected Free Lunch from Post-training: Progress Advantage for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-24
    </div>
    <details class="paper-abstract">
      Process reward models enable fine-grained, step-level evaluation of LLMs, yet building them for agentic settings remains prohibitively difficult: long-horizon interactions, irreversible actions, and stochastic environment feedback make both human annotation and Monte Carlo estimation infeasible at scale. In this work, we show that reinforcement learning (RL) post-training already provides the ingredients for effective step-level scoring, eliminating the need for dedicated reward model training altogether. Concretely, we derive an implicit advantage under a general stochastic Markov decision process, which we term progress advantage -- log-probability ratio between the RL-trained policy and its reference policy exactly recovers the optimal advantage function. This formulation makes the resulting signal annotation-free, domain-agnostic, and available as a byproduct of the standard RL post-training pipeline. We validate the effectiveness of the progress advantage across three different applications: test-time scaling, uncertainty quantification, and failure attribution on five benchmarks and four model families. Across all settings, it consistently outperforms confidence-based baselines and, despite requiring no task-specific training, surpasses dedicated trained reward models. We complement these results with deeper analyses on characteristics of progress advantage, offering practical guidance for adoption in real-world agentic systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26203v1">Agentic Analysis for Agentic Infrastructure: An LLM-Powered Pipeline for Comparative Governance of DAO and Corporate AI Protocols</a></div>
    <div class="paper-meta">
      📅 2026-06-24
    </div>
    <details class="paper-abstract">
      As AI agent protocols proliferate, the governance structures shaping their interoperability standards remain empirically underexamined. We introduce an LLM-powered comparative pipeline for large-scale governance discourse analysis, integrating automated annotation, neural topic modeling, and multi-layer network analysis to study socio-technical power structures at scale. We validate it on two contrasting standards for agent interoperability: ERC-8004 (permissionless, on-chain) and Google A2A (corporate-led). Analyzing 4,323 governance participation records, we combine LLM-assisted coding, topic modeling, and multi-layer network analysis to examine how institutional design shapes thematic priorities and community structure. We find that while governance form influences substantive focus, both regimes exhibit comparable levels of participation inequality and community fragmentation. Discourse alignment is denser in the permissionless setting, suggesting that open governance may foster greater thematic convergence despite decentralized participation. These findings illustrate how LLM-assisted methods can advance the empirical study of technology governance, with implications for designing more equitable agentic AI standards. All data and code are openly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.24245v2">AutoSpec: Safety Rule Evolution for LLM Agents via Inductive Logic Programming</a></div>
    <div class="paper-meta">
      📅 2026-06-24
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents increasingly automate complex tasks by integrating language models with external tools and environments. However, their autonomy poses significant safety risks: agents may execute destructive commands, leak sensitive data, or violate domain constraints. Existing safety approaches face a fundamental tradeoff: hand-crafted rules are interpretable but brittle, with overly conservative rules blocking safe operations (high false positives) while permissive rules miss unsafe behaviors (high false negatives). Neural classifiers lack the interpretability required for safety-critical deployments. We present AutoSpec, a framework that automatically evolves deployed expert-designed safety rules from user safe/unsafe annotations through counterexample-guided inductive synthesis (CEGIS) guided by inductive logic programming (ILP). Starting from the expert rules and a stream of annotated traces, AutoSpec iteratively evaluates rules, mines false-positive and false-negative counterexamples, uses ILP to learn which predicates discriminate them, generates candidate rule edits, and verifies candidates to select the best revision. The key insight is that ILP efficiently identifies predicates that appear frequently in false negatives but rarely in false positives (or vice versa), dramatically pruning the exponential search space of rule edits. This continues until convergence, producing interpretable rules that balance precision and recall. We evaluate AutoSpec on 291 execution traces spanning code execution and embodied agent domains. AutoSpec raises rule F1 to 0.98 and 0.93 across the two domains, achieving up to 94% false positive reduction while maintaining high recall, and converges within 4-5 iterations. The ILP-guided approach achieves up to 4.8x higher F1 than heuristic CEGIS. The learned rules are human-readable, auditable, and generalize to unseen scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.03269v2">Distilling Answer-Set Programming Rules from LLMs for Neurosymbolic Visual Question Answering</a></div>
    <div class="paper-meta">
      📅 2026-06-24
      | 💬 Under consideration in Theory and Practice of Logic Programming (TPLP)
    </div>
    <details class="paper-abstract">
      Visual Question Answering (VQA) is the task of answering questions about images, requiring the integration of multimodal input and reasoning. Modular approaches that incorporate logic-based representations into the reasoning component offer clear advantages over end-to-end trained systems, particularly in terms of interpretability. However, adapting or extending these representations when task requirements change can place a significant burden on developers. To address this challenge, we present an approach for distilling rules from Large Language Models (LLMs). Our method prompts an LLM to extend an initial VQA reasoning theory, expressed as an answer-set program, to meet new requirements of the task. Examples from VQA datasets guide the LLM, validate the results, and help correct erroneous rules by leveraging feedback from the ASP solver. We demonstrate that our approach is effective across diverse VQA datasets. Notably, only a few examples are needed to elicit correct rules from LLMs. Our experiments suggest that rule distillation from LLMs is a promising alternative to traditional data-driven rule learning approaches. Under consideration in Theory and Practice of Logic Programming (TPLP).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.25973v1">Helpful or Harmful? Evaluating LLM-Assisted Vulnerability Patching via a Human Study</a></div>
    <div class="paper-meta">
      📅 2026-06-24
      | 💬 7 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Software vulnerability remediation is a cognitively demanding task that requires specialized security expertise often lacking in general developers. In the meantime, Large Language Models (LLMs) assisted tools show potential in vulnerability detection, location, and repair tasks. [Hypothesis:] While LLM-assistance is hypothesized to accelerate patching, it also risks introducing hallucinations or insecure code, leading to a higher likelihood of generating superficial repairs that bypass the standard functionality checks but fail the security validation. [Objective:] We aim to present an empirical experiment, unveiling the capability of LLM-assisted vulnerability patching compared to manual debugging on human participants in real-world scenarios. [Method:] We plan to conduct a controlled experiment using a Balanced Crossover design. For that, we have developed a WebApp for code execution and integrated hidden Ghost Tests to verify patch integrity beyond visible functional requirements. The experiment involves training and evaluation scenarios. The remediation speed, remediation efficacy for both standard functionality tests and security tests, and participant perception will be evaluated. [Pilot Study:] A pilot experiment with a small sample of participants has been conducted, providing insights for the following study.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.23001v2">EnerInfer: Energy-Aware On-Device LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-06-24
    </div>
    <details class="paper-abstract">
      On-device LLM inference is increasingly attractive for privacy-preserving, reliable, and cost-effective deployment, yet its energy and thermal costs remain a critical bottleneck. Existing systems primarily optimize for decoding speed, implicitly assuming that faster execution is always preferable. We show instead that on-device LLM inference often has exploitable configuration slack: modestly lowering NPU and memory frequencies preserves quality of experience (QoE) while substantially improving energy efficiency and reducing heat. Realizing this opportunity in production is challenging. The most energy-efficient NPU/DDR setting varies with the model, inference engine, platform, and runtime conditions, with no stable ranking across configurations. Commercial devices further lack component-level power sensing, and shell temperature evolves with request arrivals, response lengths, and thermal history. To address these challenges, we propose EnerInfer, the first on-device LLM inference framework that jointly manages energy efficiency, throughput, and thermal comfort for LLM workloads. EnerInfer replaces per-model profiling and sensor-heavy control with disaggregated, model-structure-aware prediction and ranking-driven online feedback. It predicts throughput and power for unseen LLMs across NPU/DDR frequency settings, selects QoE-satisfying efficient configurations under runtime interference, and uses lightweight limited-horizon thermal prediction to dynamically switch between energy-optimized and thermally constrained inference. Evaluations on real-world LLMs show that EnerInfer improves energy efficiency by up to 65%, 12%, and 24% on phones, a laptop, and a development board, respectively, without QoE violation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.05933v2">Reinforcement Learning Improves Traversal of Parametric Knowledge in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-24
      | 💬 `
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) is often credited with improving language model reasoning at the expense of knowledge. We challenge this narrative by showing that reasoning models consistently outperform their instruction-tuned versions on pure knowledge recall tasks. These gains do not reflect newly acquired information, but rather an improved procedural skill in navigating and searching existing knowledge hierarchies within the model parameters. Structured prompting, which explicitly guides models through hierarchical traversal -- recovers most of the instruct-reasoning gap across five model families. A controlled RL experiment on unseen, non-extractable facts improves recall of held-out frequent but previously inaccessible facts, ruling out simple data exposure. On depth-stratified retrieval tasks, reasoning models exhibit superior traversal as retrieval depth grows. Layerwise activation analysis further shows that while factual representations maintain high cosine similarity between instruct and reasoning models, query representations diverge noticeably, indicating that reasoning primarily reshapes how models traverse knowledge rather than the knowledge representation itself. Finally, we find that distilled models often fail to match reasoning models on knowledge recall because they imitate self-correction without acquiring the exploratory behavior needed for hierarchical navigation. Together, these findings suggest that improving factual recall in LLMs depends not only on expanding what models know but also on teaching them to navigate it -- motivating future post-training methods that optimize traversal.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03785v3">Membox: Weaving Topic Continuity into Long-Range Memory for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-24
    </div>
    <details class="paper-abstract">
      Long-term human-agent dialogues are organized by topic continuity: adjacent turns often develop the same goal, plan, problem, or event, while related activities may recur across distant sessions. Yet many LLM agent memory systems first decompose histories into isolated turns or fixed-size chunks, then compensate through enrichment, consolidation, or retrieval mechanisms still tied to semantic proximity or fragment-level records. This weakens temporal and causal organization and biases memory access toward semantic proximity rather than task- or topic-level continuity. We introduce \emph{Membox}, a hierarchical memory architecture that instantiates topic continuity as an explicit organization layer for agent memory. Its \textbf{Topic Loom} incrementally organizes dialogue streams into boxes whose internal turns follow the same local topic, while its \textbf{Trace Weaver} links extracted events across boxes into macro-topic traces that recover recurring activities, goals, and factual developments across distant sessions. On LoCoMo, Topic-Loom-only retrieval improves over the best Mem0/A-MEM retrieval-depth setting by 13.00 F1 points (53.95 vs. 40.95), and trace-expanded retrieval further raises F1 to 55.28; with GPT-4o, trace-expanded retrieval reaches 59.71 F1. Additional DialSim results show the same gain from adding cross-box traces in multi-party dialogue. These results show that local topic-continuity organization and macro-topic trace expansion improve long-range memory beyond semantic retrieval over fragmented records.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.25867v1">LLM-Based Discovery of Latent Requirements from Stakeholder Conversations: Preliminary Results from Industry</a></div>
    <div class="paper-meta">
      📅 2026-06-24
      | 💬 This paper has been accepted at the 13th International Workshop on Artificial Intelligence and Requirements Engineering (AIRE 2026), co-located with RE 2026
    </div>
    <details class="paper-abstract">
      Stakeholder interviews are an important source of information for requirements elicitation, yet many relevant requirements remain implicit in such conversations. Stakeholders frequently describe workflows, challenges, and operational practices without explicitly articulating the software capabilities that could address them. Recent work has considered the use of LLMs to analyze conversational data and extract requirements from stakeholder interviews. Existing approaches, however, primarily focus on identifying explicitly stated requirements, leaving implicit opportunities largely unexplored. In this paper, we present LENS (LLM-Enabled Needs Discovery from Stakeholder Interviews), an approach that analyzes stakeholder interview transcripts to both extract explicit requirements and infer additional latent requirements. LENS performs this inference by reasoning over stakeholder statements together with contextual information about organizational tools and infrastructure. Both extracted and inferred requirements are represented as user stories and linked to transcript excerpts to ensure traceability. We conduct a preliminary evaluation of LENS using twelve stakeholder interview transcripts collected in an industrial setting involving cybersecurity operations. We show that LENS achieves an average F1-score of 84.4% for extracting explicit requirements, while, on average, 75% of the latent requirements identified by LENS were perceived as providing useful automation or time-saving potential by domain experts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.22327v2">Geometry-Aware Online Scheduling for LLM Serving: From Theoretical Bound to System Practice</a></div>
    <div class="paper-meta">
      📅 2026-06-24
    </div>
    <details class="paper-abstract">
      The explosive demand for interactive Large Language Model serving has highlighted the management of the Key-Value cache's dynamic memory footprint as a critical area for performance optimization in inference engines. Modern inference systems overwhelmingly rely on time-centric scheduling heuristics, such as Shortest Job First. However, their theoretical optimality is rooted in traditional schedule modeling, failing to capture the highly dynamic, 2D spatio-temporal geometric growth specific to LLM inference mechanisms. To resolve this, we propose the geometry-aware online scheduling by introducing the Smallest Volume First (SVF) algorithm and its highly efficient variant, 1-bit SVF. Theoretically, we provide a rigorous mathematical foundation for our approach. Via a novel volume-certificate proof, we sharpen SVF's worst-case competitive ratio from the prior best of 48 towards \textbf{3} in the high-concurrency regime of LLM serving. Building upon this core breakthrough, we complete a comprehensive theoretical taxonomy analyzing our algorithms across different traffic scenarios and information availability. Practically, we seamlessly integrate our approach as a plug-and-play layer in vLLM. Extensive evaluations on Llama-3.1 models demonstrate comprehensive performance gains: SVF delivers strong reductions in both average and tail latency, while 1-bit SVF, with merely a single bit information, achieves competitive throughput and latency. This work establishes a theoretically sound and empirically proven approach for resolving memory-constrained scheduling in modern LLM deployments. To facilitate future research, our code is available at https://github.com/Aurora-Kl/Geometry-Aware-Online-Scheduling.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.25852v1">Semantic Consistency Policy Optimization for Reinforcement Learning of LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-24
      | 💬 16 pages, 7 figures, 5 tables. Under review at EMNLP 2026
    </div>
    <details class="paper-abstract">
      Group-based reinforcement learning effectively post-trains LLM agents for long-horizon, sparse-reward tasks by deriving step-level credit from trajectory outcomes. However, this ties a step's credit to its rollout's final outcome: semantically near-identical intermediate steps receive opposite credit depending on whether their trajectory eventually succeeded or failed. Such semantic credit inconsistency sends conflicting gradients to similar actions and wastes the partially-correct progress inside failed rollouts. Motivated by this, we propose Semantic Consistency Policy Optimization (SCPO), a value-free reward-shaping method that mitigates this inconsistency by recovering step-level credit from successful siblings in the same rollout group. Concretely, SCPO scores each failed step against a successful sibling and adds positive step-level credit for new progress along that sibling. On ALFWorld and WebShop, SCPO matches or exceeds strong group-based baselines, reaching 93.7+/-4.1 percent success on ALFWorld and 74.8+/-2.0 percent on WebShop at 1.5B parameters, with gains concentrated on the hardest multi-step tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.18266v2">A Probabilistic Framework for LLM-Based Model Discovery</a></div>
    <div class="paper-meta">
      📅 2026-06-24
    </div>
    <details class="paper-abstract">
      Automated methods for discovering mechanistic simulator models from observational data offer a promising path toward accelerating scientific progress. Such methods often take the form of agentic-style iterative workflows that repeatedly propose and revise candidate models by imitating human discovery processes. However, existing LLM-based approaches typically implement such workflows via hand-crafted heuristic procedures, without an explicit probabilistic formulation. We recast model discovery as probabilistic inference, i.e., as sampling from an unknown distribution over mechanistic models capable of explaining the data. This perspective provides a unified way to reason about model proposal, refinement, and selection within a single inference framework. As a concrete instantiation of this view, we introduce ModelSMC, an algorithm based on Sequential Monte Carlo sampling. ModelSMC represents candidate models as particles which are iteratively proposed and refined by an LLM, and weighted using likelihood-based criteria. Experiments on real-world scientific systems illustrate that this formulation discovers models with interpretable mechanisms and improves posterior predictive checks. More broadly, this perspective provides a probabilistic lens for understanding and developing LLM-based approaches to model discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.23178v2">Judging the Judges: A Systematic Evaluation of Bias Mitigation Strategies in LLM-as-a-Judge Pipelines</a></div>
    <div class="paper-meta">
      📅 2026-06-24
      | 💬 22 pages, 4 figures. Published in Transactions on Machine Learning Research (2026)
    </div>
    <details class="paper-abstract">
      LLM-as-a-Judge has become the dominant paradigm for evaluating language model outputs, yet LLM judges exhibit systematic biases that compromise evaluation reliability. We present a comprehensive empirical study comparing nine debiasing strategies across five judge models from four provider families (Google, Anthropic, OpenAI, Meta), three benchmarks (MT-Bench n=400, LLMBar n=200, custom n=375), and four bias types. Our headline practical finding is that a mid-tier model with the right debiasing can outperform frontier judges at a fraction of the cost: Gemini 2.5 Flash with the Combined Budget strategy reaches the highest agreement of any configuration we tested (71.0%, kappa=0.549) at ~$0.001 per evaluation, about 15x cheaper than the best frontier setup (Claude Sonnet 4, 69.5%, ~$0.015). Other key findings: (1) Style bias is the dominant bias (0.10-0.76 across models, favoring markdown over plain prose), far exceeding position bias (<=0.04), yet is rarely studied. (2) Verbosity bias is heterogeneous when measured length-aware: Pro, Flash, and Llama prefer longer answers (+0.24 to +0.44), Claude prefers concise (-0.12), and GPT-4o is neutral (-0.04); on truncation controls all models correctly prefer the complete response (0.88-1.00 accuracy). (3) Debiasing helps multiple models: Claude S8 (+11.5 pp), Flash S8 (+7.5 pp), and Claude S5 (+7.3 pp) survive Holm-Bonferroni correction, with Flash S1 (+4.7 pp) and Llama S8 (+4.5 pp) also significant. We release our evaluation framework, the 375-pair controlled dataset, and per-instance cached results for all nine strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26185v1">Necessary but Not Sufficient: Temperature Control and Reproducibility in LLM-as-Judge Safety Evaluations</a></div>
    <div class="paper-meta">
      📅 2026-06-24
      | 💬 7 pages, 2 tables
    </div>
    <details class="paper-abstract">
      LLM-as-judge ("grader") components are now standard in evaluation harnesses, including safety evaluations where a pass/fail verdict may gate downstream deployment decisions. A widespread assumption is that setting the grader's sampling temperature to 0 makes grading deterministic. We test this assumption against a real safety-evaluation codebase (Japan AISI's open-source aisev) and show it fails on two levels. First, the harness invokes its grader without setting temperature or seed; the underlying provider silently applies its default of 1.0, so items near the decision boundary flip pass/fail across identical runs (per-item disagreement up to ~50% over 20 runs). Second, pinning temperature=0 reduces but does not eliminate flips: across 690 API calls spanning two providers, three model tiers, and five sampling configurations, 1-2 of 7 borderline items remain non-reproducible even under forced greedy decoding (top_k=1). Claude Opus 4.7/4.8 has since deprecated temperature entirely, rendering the primary mitigation inapplicable to newer model generations. These findings expose a structural gap: evaluation harnesses that report single-run verdicts without variance or grader-disagreement metrics can present noise as a safety property. We release a reproduction harness (690 calls, 7 conditions) and recommend that harnesses treat grader disagreement as a first-class health metric alongside the scores themselves.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.25809v1">Designing Trustworthy LLM-based Wellbeing Recommendation through Controllable Interaction</a></div>
    <div class="paper-meta">
      📅 2026-06-24
      | 💬 Accepted to the 1st Workshop on Trustworthy and Adaptive LLMs for Mental and Physical Wellbeing in Recommendations @UMAP 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to generate personalized guidance in wellbeing contexts such as physical activity, stress management, and mental health support, enabling fluent and context-aware interaction but relying on largely implicit mechanisms that shape how recommendations are expressed and adapted. We argue that this reliance on implicit adaptation through prompting and alignment limits control over guidance, responsibility framing, and user influence, which is particularly problematic in wellbeing settings where recommendations affect users' actions and long-term outcomes. We propose a system-level perspective in which conversational behavior is structured through explicit interaction constraints, including guidance strategies, explanation styles, degrees of directness, and mechanisms for user control. Building on prior work on tangible recommendations, we show how these constraints address key challenges in wellbeing-oriented recommendation, namely trust calibration, intent alignment, and consequence awareness. We outline a modular architecture for controllable LLM-based recommendation and discuss how different configurations can be systematically designed and evaluated in relation to user-centered outcomes such as self-efficacy, perceived agency, and appropriate reliance. This paper contributes a system-level framework for designing LLM-based recommender systems that are adaptive while remaining transparent, controllable, and aligned with human wellbeing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.25782v1">Do Encoders Suffice? A Systematic Comparison of Encoder and Decoder Safety Judges for LLM Adversarial Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-06-24
      | 💬 13 pages, 5 figures, Accepted into ICANN2026
    </div>
    <details class="paper-abstract">
      With the widespread adoption of large language models (LLMs) in chatbots and everyday applications, companies increasingly need guardrails that are effective while remaining low-cost and low-latency. Safety evaluation of LLM outputs has generally relied on LLM-based judges, which can be effective but are often slow and expensive to deploy at scale. In this paper, we evaluate whether fine-tuned modern encoder classifiers from the ModernBERT family, including ModernBERT and Ettin, can reliably identify harmful LLM outputs in user-model conversations without substantial performance loss relative to LLM-based judges. We benchmark these encoder classifiers against rule-based prefix matching, fine-tuned LLM classifiers, and LLM judges using a range of judge-prompting strategies across open-source adversarial datasets. The LLM judges include evaluation methodologies from StrongReject, ShieldGemma, JailbreakBench, AILuminate, SorryBench, and a Claude-as-a-judge setup, as well as fine-tuned safety classifiers such as LlamaGuard 3 and LlamaGuard 4. The encoder classifiers are fine-tuned on judge-labeled data using a majority-voting label strategy and are then evaluated on a gold-standard holdout dataset to assess their performance relative to LLM judges. We report absolute performance using F1 score, false negative rate, and precision-recall metrics. We also break down results by attack technique, including single-turn prompting, decomposition, escalation, and context manipulation, to identify where encoder classifiers align with or diverge from LLM-based judges. Our findings provide guidance on when encoder classifiers can serve as cost- and latency-efficient alternatives to LLM-based safety evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.25750v1">RAS: Measuring LLM Safety Through Refusal Alignment</a></div>
    <div class="paper-meta">
      📅 2026-06-24
    </div>
    <details class="paper-abstract">
      Safety evaluation of large language models (LLMs) is commonly performed by querying models with unsafe or jailbreak prompts and judging whether their outputs violate a safety policy. Although useful, output-level evaluation is expensive, sensitive to judge choice, and easily tied to fixed question banks. We propose **SafeVec**, a white-box evaluation procedure that measures safety from internal representations rather than generated answers. **SafeVec** first extracts layer-wise refusal directions from a safety-aligned reference model, then selects stable layer windows where safe and unsafe behaviors are separable, and finally scores a target model by measuring whether its hidden states align with these refusal directions under unsafe and jailbreak prompts. The resulting metric, **RAS** (**R**efusal **A**lignment **S**core), maps representation-level refusal alignment to a calibrated 0-100 safety score. Across `Llama`, `Gemma`, and `Qwen` model families, RAS separates aligned models from uncensored and abliterated variants, tracks output-level attack success rate, and is substantially faster than judge-based evaluation. These results suggest that refusal alignment provides a compact and efficient signal for white-box LLM safety evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.29813v3">Compiling Code LLMs into Lightweight Executables</a></div>
    <div class="paper-meta">
      📅 2026-06-24
      | 💬 Accepted at the 34th ACM International Conference on the Foundations of Software Engineering (FSE 2026), 25 pages
    </div>
    <details class="paper-abstract">
      The demand for better prediction accuracy and higher execution performance in neural networks continues to grow. The emergence and success of Large Language Models (LLMs) have produced many cloud-based tools for software engineering tasks such as code suggestion. Although effective, cloud deployment raises concerns over privacy, latency, and reliance on network connectivity. Running LLMs locally on personal devices such as laptops would address these issues, because it enables offline use and reduces response time. However, local deployment is challenging, since commodity devices lack high-performance accelerators such as GPUs and are constrained by limited memory and compute capacity, which makes it hard to execute large models efficiently. We present Ditto, a framework that optimizes both the model size of Code LLMs and the inference programs that execute them. Our approach integrates two components. The first is a quantization technique inspired by product quantization, which groups model parameters into per-block codebooks via K-Means clustering and stores each weight as a bit-packed low-bitwidth index. The second component is a compilation pass integrated into LLVM that automatically detects and replaces unoptimized General Matrix-Vector Multiplication (GEMV) operations, with calls into Basic Linear Algebra Subprograms (BLAS) libraries that are highly optimized for the target hardware. The output of Ditto is a compiled executable that runs the selected Code LLM on commodity hardware. We evaluate Ditto on three popular Code LLMs, namely Code Llama, MagicCoder, and OpenCodeInterpreter, achieving up to 10.5$\times$ faster inference, 6.4$\times$ lower memory usage, and 10.5$\times$ lower energy consumption compared with their original inference pipelines, while preserving accuracy close to the full-precision models, with an average loss of only 0.27% in pass@1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10187v3">MINIF2F-DAFNY: LLM-Guided Mathematical Theorem Proving via Auto-Active Verification</a></div>
    <div class="paper-meta">
      📅 2026-06-24
    </div>
    <details class="paper-abstract">
      LLMs excel at reasoning, but validating their steps remains challenging. Formal verification offers a solution through mechanically checkable proofs. Interactive theorem provers (ITPs) dominate mathematical reasoning but require detailed low-level proof steps, while auto-active verifiers offer automation but focus on software verification. Recent work has begun bridging this divide by evaluating LLMs for software verification in ITPs, but the complementary direction, LLMs for mathematical theorem proving in auto-active verifiers, remains unexplored. We present MINIF2F-DAFNY, the first translation of the widely-used mathematical benchmark miniF2F to an auto-active verifier: Dafny. We find that Dafny's automation alone solves 39-44% of problems with empty proofs, whereas many require substantial proof guidance in ITPs. We evaluate 8 off-the-shelf LLMs on proof generation, with the best model (Claude Opus 4.6) achieving 62.7% cumulative pass@4 on the full test set, improving over the 38.9% empty-proof baseline by 23.8 percentage points. These results show that auto-active verification offers a complementary empirical setting for AI-assisted mathematical reasoning, where LLMs provide high-level guidance while SMT automation handles low-level details. Our benchmark and evaluation infrastructure are publicly available on https://github.com/dafny-lang/miniF2F.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.04773v2">Distribution Preference Optimization: A Fine-grained Perspective for LLM Unlearning</a></div>
    <div class="paper-meta">
      📅 2026-06-24
      | 💬 20 pages
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) demonstrate remarkable capabilities learned from vast corpora, concerns regarding data privacy and safety are receiving increasing attention. LLM unlearning, which aims to remove the influence of specific data while preserving overall model utility, is becoming an important research area. One of the mainstream unlearning classes is optimization-based methods, which achieve forgetting directly through fine-tuning, exemplified by Negative Preference Optimization (NPO). However, NPO's effectiveness is limited by its inherent lack of explicit positive preference signals. Attempts to introduce such signals by constructing preferred responses often necessitate domain-specific knowledge or well-designed prompts, fundamentally restricting their generalizability. In this paper, we shift the focus to the distribution-level, directly targeting the next-token probability distribution instead of entire responses, and derive a novel unlearning algorithm termed \textbf{Di}stribution \textbf{P}reference \textbf{O}ptimization (DiPO). We show that the requisite preference distribution pairs for DiPO, which are distributions over the model's output tokens, can be constructed by selectively amplifying or suppressing the model's high-confidence output logits, thereby effectively overcoming NPO's limitations. We theoretically prove the consistency of DiPO's loss function with the desired unlearning direction. Extensive experiments demonstrate that DiPO achieves a strong trade-off between model utility and forget quality. Notably, DiPO attains the highest forget quality on the TOFU benchmark, and maintains leading scalability and sustainability in utility preservation on the MUSE benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.14274v3">Discovering New Theorems via LLMs with In-Context Proof Learning in Lean</a></div>
    <div class="paper-meta">
      📅 2026-06-24
      | 💬 12 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated significant promise in formal theorem proving. In this study, we investigate the ability of LLMs to discover novel theorems and produce verified proofs. We propose a pipeline called Conjecturing-Proving Loop (CPL), which iteratively generates mathematical conjectures and attempts to prove them in Lean 4. A key feature of CPL is that each iteration conditions the LLM on previously generated theorems and their formal proofs, enabling parameter-free improvement of proof strategies via in-context learning. We provide both theoretical and experimental evidence that CPL increases the discovery rate of hard-to-prove theorems compared to frameworks that generate statements and proofs simultaneously. Moreover, our experiments show that reusing the LLM's own formally verified outputs as context consistently improves subsequent proof success, demonstrating the effectiveness of self-generated in-context learning for neural theorem proving. The source code is available at https://github.com/auto-res/ConjecturingProvingLoop.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02080v3">The 4/$δ$ Bound: Designing Predictable LLM-Verifier Systems for Formal Method Guarantee</a></div>
    <div class="paper-meta">
      📅 2026-06-24
      | 💬 36 pages, 9 figures
    </div>
    <details class="paper-abstract">
      The integration of Formal Verification tools with Large Language Models (LLMs) offers a path to scale software verification beyond manual workflows. However, current methods remain unreliable: without a solid theoretical footing, the refinement process acts as a black box that may oscillate, loop, or diverge. This work bridges this critical gap by developing an LLM-Verifier Convergence Theorem, providing the first formal framework with provable guarantees for termination in multi-stage verification pipelines. We model the interaction not as a generic loop, but as a sequential absorbing Markov Chain comprising four essential engineering stages: \texttt{CodeGen}, \texttt{Compilation}, \texttt{InvariantSynth}, and \texttt{SMTSolving}. We prove that for any non-zero stage success probability ($δ> 0$), the system reaches the \texttt{Verified} state almost surely. Furthermore, because of the sequential nature of the pipeline, we derive a precise latency bound of $\mathbb{E}[n] \leq 4/δ$. We stress-tested this prediction in an extensive empirical campaign comprising over 90,000 trials. The results match the theory with striking consistency: every run reached verification, and the empirical convergence factor clustered tightly around $C_f\approx 1.0$, confirming that the $4/δ$ bound accurately mirrors system behavior rather than serving as a loose buffer. Based on this data, we identify three distinct operating zones -- marginal, practical, and high-performance -- and propose a dynamic calibration strategy to handle parameter drift in real-world environments. Together, these contributions replace heuristic guesswork with a rigorous architectural foundation, enabling predictable resource planning and performance budgeting for safety-critical software.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26173v1">AlgoEvolve: LLM-driven Meta-evolution of Algorithmic Trading Programs</a></div>
    <div class="paper-meta">
      📅 2026-06-24
    </div>
    <details class="paper-abstract">
      Recent work shows that Large Language Models (LLMs) can act as semantic mutation operators for the evolutionary discovery of programs and proofs. Most current applications focus on static coding benchmarks. We extend this paradigm to algorithmic trading. This domain is uniquely challenging because it is noisy, non-stationary, and highly discontinuous. We present AlgoEvolve, an LLM-driven evolutionary framework that generates, evaluates, and iteratively improves executable trading strategies. These strategies are expressed as Python code and evaluated through a rigorous testing protocol. Across multiple experiments, the system exhibits emergent regime-adaptive strategy logic, including autonomous shifts in trading rules. We further introduce a meta-evolutionary outer loop that evolves the prompts guiding program synthesis in the inner loop. This outer loop discovers improved search heuristics. These heuristics balance exploration and exploitation while reducing zero-trade failures. They consistently outperform initial human-designed instructions. The results demonstrate that LLM-based semantic evolution provides a viable approach for continual program synthesis in complex environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.25647v1">Retrieval-Grounded Multilingual LLM Assistance for Island Smallholder Farmers</a></div>
    <div class="paper-meta">
      📅 2026-06-24
      | 💬 13, 4
    </div>
    <details class="paper-abstract">
      Smallholder farming communities in remote, depopulating areas have limited access to agricultural advice, and their locally specific agronomic knowledge, often expressed in regional dialect, is poorly represented in the global corpora on which Large Language Models (LLMs) are trained. A general-purpose chatbot therefore answers fluently but unreliably, ungrounded in authoritative local data farmers can trust. This paper presents a conversational AI assistant, Falco eleonorae, embedded in a bilingual (Greek-primary, English-secondary) e-market platform serving farmers and cooperatives of a defined island area of interest. It is a thin Backend-for-Frontend (BFF) proxy in front of a geospatially-aware agronomic agent rather than a self-hosted model. Answer generation and tool selection are delegated to a managed upstream service on OpenAI GPT-5-family models, while one bounded task, describing an uploaded field photograph, is handled directly by a vision-capable model so only text reaches the agent, and voice input is transcribed by a managed EU streaming speech-to-text service. Grounding comes not from a self-hosted vector database but from tool-augmented retrieval: a Model Context Protocol (MCP) tool queries a curated, read-only, bilingual data interface exposing local crops, a seasonal calendar, traditional practices, a dialect glossary, products, agritourism experiences, cooperatives, and training content, each wrapped in a geospatial Well-Known Text envelope anchoring the agent to the area of interest. We detail its multilingual, voice, and image modalities, its progressive-web-application and accessibility design for low-bandwidth field use, and its security and data-protection posture, and argue that for a small, resource-constrained rural deployment a managed, grounded multilingual assistant is more attainable and trustworthy than a self-hosted model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.00804v3">Epistemic Bias Injection: Manipulating LLM Opinion via Selective Context Retrieval</a></div>
    <div class="paper-meta">
      📅 2026-06-24
    </div>
    <details class="paper-abstract">
      When answering user queries, LLMs often retrieve knowledge from external sources stored in retrieval-augmented generation (RAG) databases. These are often populated from unvetted sources, e.g. the open web, and can contain maliciously crafted data. This paper studies attacks that can manipulate the context retrieved by LLMs from such RAG databases. Prior work on such context manipulation primarily injects false or toxic content, which can often be detected by fact-checking or linguistic analysis. A more subtle threat, which we call epistemic bias injection (EBI), is where adversaries inject factually correct yet epistemically biased passages that systematically favor one side of an open-ended issue. Although linguistically coherent and truthful, such adversarial passages effectively crowd out alternative viewpoints during retrieval from the RAG and push LLM outputs towards an attack-desired stance. As a core contribution, we propose a novel characterization of the problem: We give a geometric metric that quantifies stance polarity and epistemic bias. This metric can be computed directly on embeddings of text passages. Leveraging it, we construct EBI attacks and develop a lightweight prototype defense called BiasDef for them. We evaluate them both on a comprehensive benchmark constructed from public question answering datasets. Our results show that: (1) the proposed attack induces significant stance polarity shifts, effectively evading existing retrieval-based sanitization defenses, and (2) BiasDef substantially reduces adversarial retrieval and epistemic bias in LLM's answers. Overall, this demonstrates the new threat as well as the ease of employing epistemic bias metrics for filtering in RAG-enabled LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.25608v1">An Approach for a Supporting Multi-LLM System for Automated Certification Based on the German IT-Grundschutz</a></div>
    <div class="paper-meta">
      📅 2026-06-24
      | 💬 Accepted for publication at the 2025 IEEE International Conference on Cyber Security and Resilience (IEEE CSR), Chania, Crete, Greece, August 4-6, 2025. 8 pages, 2 figures
    </div>
    <details class="paper-abstract">
      This paper presents a novel approach to perform semi-automated BSI IT-Grundschutz certification using a MultiLarge Language Model system (MLS) with Hybrid RetrievalAugmented Generation (HybridRAG). Facing the challenges of the Network and Information Security Directive 2 (NIS2) directive, a shortage of specialists, and high implementation costs, our MLS architecture aims to increase efficiency, reduce costs, and support certifiers in maintaining the quality of security concepts while meeting the increased demand for certifications of newly affected companies. The system combines Large Language Models (LLMs) and Knowledge Graphs (KGs) to support different phases of the certification process, including protection needs assessment, modeling, IT-Grundschutz check, measure consolidation, and subsequent realization. Our architecture addresses the growing demand for security concepts and offers an approach to handle the digital security challenges introduced by NIS2.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.13076v3">TruncProof: A Guardrail for LLM-based JSON Generation under Token-Length Constraints</a></div>
    <div class="paper-meta">
      📅 2026-06-24
      | 💬 Main paper (8 pages). Accepted at the International Joint Conference on Neural Networks (IJCNN 2026)
    </div>
    <details class="paper-abstract">
      The LLM-based generation of machine-readable outputs such as JSON has attracted significant attention for integration with external systems. However, existing approaches cannot strictly enforce the maximum number of tokens to be generated, leading to infinite generation or truncated outputs that cause a system malfunction. To address this limitation, we propose TruncProof, a novel grammar-constrained generation method that enables LLMs to produce grammatically valid JSONs while adhering to a predefined token limit. By leveraging the properties of LL(1) parsers, TruncProof efficiently approximates the minimum number of tokens required to complete a grammatically valid output at each decoding step. Experiments on the Text-to-JSON instruction tasks demonstrate that TruncProof successfully generates syntactically correct outputs even under strict token constraints. Furthermore, we show that TruncProof can be effectively combined with advanced decoding strategies, resulting in outputs that are not only grammatically valid but also semantically accurate. The source code is public at https://github.com/Yosshi999/TruncProof
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.25556v1">BiPACE: Bisimulation-Guided Policy Optimization with Action Counterfactual Estimation for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-24
    </div>
    <details class="paper-abstract">
      Stepwise group-based RL is an attractive way to train long-horizon LLM agents without a learned critic: it reuses multiple sampled rollouts to estimate local advantages. Its weakness is less visible but more fundamental: every group-relative estimator assumes that the steps it compares are equivalent for credit assignment. We show that current agentic variants violate this assumption through a state-action credit mismatch. The observation-hash partition is overly fine on the state side, creating singleton groups with zero step-level signal, while a single within-group mean is too coarse on the action side, mixing state-value estimation with action-specific credit. We introduce BiPACE (Bisimulation-Guided Policy Optimization with Action Counterfactual Estimation), a drop-in advantage estimator that fixes both sides without adding a critic, auxiliary loss, or extra rollouts. BiGPO clusters steps by cosine distance in the actor's own hidden-state geometry, an empirical policy-induced proxy for bisimulation that substantially lowers the singleton rate left by observation hashing. PACE then recenters returns within each behavioral cluster using action-conditioned peer baselines; its Q-style instance estimates a local Q(s,a)-V(s) nonparametrically. On ALFWorld/Qwen2.5-7B, BiPACE_Q raises overall validation success from GiGPO's 90.8 to $97.1\pm0.9$ over three seeds, and crosses the 95% threshold on every seed, which GiGPO never does within the same budget. On Qwen2.5-1.5B it reaches $93.5\pm1.2$ versus GiGPO's 86.7, and on WebShop and TextCraft it improves over GRPO and GiGPO at both model scales. The measured BiPACE-specific overhead is 11.3% of a single training-step wall time. Yet it changes the estimator's comparison unit from surface identity to approximate behavioral equivalence plus action-side counterfactuals. The code is available at https://github.com/TianxiangZhao/BiPACE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.24369v2">Accelerating Disaggregated RL for Visual Generative LLMs with Diffusion-Based Parallelism and Trainer-Assisted Generation</a></div>
    <div class="paper-meta">
      📅 2026-06-24
      | 💬 Withdrawn by the authors pending resolution of intellectual property and institutional disclosure requirements
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has become a dominant post-training paradigm, driving the emergence of high-performance RL systems such as veRL for autoregressive large language models (LLMs). In parallel, diffusion-oriented RL algorithms, e.g., DanceGRPO and FlowGRPO, have rapidly expanded the scope of RL from language reasoning to diffusion-based visual and flow-based generation. However, efficient RL systems for diffusion generative LLMs remain underexplored. Existing implementations, e.g., veRL-Omni, still rely on colocated execution, which simplifies synchronization but couples rollout and training resources, limits heterogeneous deployment, and constrains independent scaling. To this end, we introduce DigenRL, a disaggregated RL framework for diffusion-based generative LLMs that supports flexible resource allocation, accommodates heterogeneous GPUs, and facilitates efficient task scheduling. To maximally reduce the execution bubbles in the disaggregated architecture, we propose: 1) a generation-axis pipeline (GAP) and time-step parallelism (TSP) in the diffusion architecture to enable finer-grained pipelining between rollout and training; 2) an elastic trainer-assisted generation (TAG) approach to enable the trainer GPU resources to dynamically assist in executing rollout generations; and 3) a tightly one-step constrained asynchronous strategy to further utilize the tail bubble in the pipeline. Extensive experiments are conducted on three hardware testbeds with 16-32 GPUs using HunyuanVideo-13B, Wan2.1-14B, FLUX.1-12B, and QwenImage-20B generative models. Experimental results show that DigenRL achieves 1.56-2.10x throughput improvements over state-of-the-art diffusion RL systems, veRL-Omni and GenRL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.25530v1">Evaluating LLMs on Real-World Software Performance Optimization</a></div>
    <div class="paper-meta">
      📅 2026-06-24
    </div>
    <details class="paper-abstract">
      Software performance optimization is a notoriously complex and manual task. Despite the growing use of Large Language Models (LLMs) for code refinement, we still lack benchmarks that capture how optimization actually happens in real-world codebases. Existing frameworks often oversimplify the problem by focusing on isolated functions or a single performance metric, missing the critical trade-offs between execution time and memory footprint, the inherent noise of the measurement environment, and the variability introduced by different input data and execution conditions. We address this by introducing SWE-Pro, a repository-level benchmark derived from 102 expert-written optimizations from open-source projects. Unlike previous benchmarks, SWE-Pro pairs each task with parameterized tests to evaluate runtime, peak memory, and Time-Weighted Memory Usage (TWMU) across varying input data and execution conditions under noise-aware measurement conditions. Our evaluation shows that current LLMs struggle significantly: runtime gains are negligible, and memory optimizations are nearly non-existent. This stands in sharp contrast to expert implementations, which achieve an aggregate speedup of 15.5x and peak memory reduction of 171.3x over benchmark tasks. Expert-written improvements are observed in 91.2% of tasks for runtime and 65.7% for peak memory. Our findings expose a substantial gap between current LLM capabilities and the demands of expert-level engineering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05097v3">Continual Knowledge Updating in LLM Systems: Learning Through Multi-Timescale Memory Dynamics</a></div>
    <div class="paper-meta">
      📅 2026-06-24
      | 💬 Accepted as a poster at the ICML 2026 Workshop "Continual Adaptation at Scale: Towards Sustainable AI" (CATS@ICML 2026). 9 pages, 2 figures
    </div>
    <details class="paper-abstract">
      LLMs are trained once, then deployed into a world that never stops changing. External memory compensates for this, but most systems manage it explicitly rather than letting it adapt on its own. Biological memory works differently: coupled multi-timescale dynamics make new associations immediately usable, strengthen what repetition confirms, and let the rest fade. We argue that external memory should follow a similar principle. In Memini, this view takes the form of an associative memory that organizes knowledge as a directed graph. Each edge carries two coupled internal variables, one fast and one slow, following the Benna-Fusi model of synaptic consolidation. From this coupling, episodic sensitivity, gradual consolidation, and selective forgetting are expected to emerge as facets of a single mechanism, reframing external memory as a learning substrate that reorganizes through its own dynamics. This workshop article describes an early-stage conceptual design without experimental evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.25509v1">ASSCG: Just-Right Gating over Chattering for Fast-Slow LLM Planning in Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2026-06-24
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can improve autonomous driving planning but are costly to query online, and existing fast-slow planners often rely on hand-designed triggering rules that either over-call the slow system or call it at the wrong times. We formulate slow-system invocation as a resource-aware sequential decision problem and propose the Adaptive Slow-System Control Gate (ASSCG), which makes frame-level Query/Cache/Drop decisions to refresh, reuse, or suppress slow guidance. ASSCG uses an RWKV backbone for efficient long-horizon gating and is trained with supervised fine-tuning followed by GRPO-style compute-aware reinforcement fine-tuning. We apply ASSCG to two different fast-slow architectures: (i) AsyncDriver on nuPlan Hard20 closed-loop evaluation, where ASSCG improves score to 67.28 (+2.28) while reducing average end-to-end inference latency by 60%; and (ii) a RecogDrive-based dual system that we build by replacing its original VLM-2B module with a lightweight ViT-based fast planner and adding an LLM slow planner, evaluated on NAVSIM, where ASSCG achieves 91.4 PDMS (+0.6) and increases average speed by 25%. The project page, including video visualizations and additional results, is available at https://williamxuanyu.github.io/asscg/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.19887v2">FinRED: An Expert-Guided Benchmark Generation and Evaluation Framework for Financial LLM Red-Teaming</a></div>
    <div class="paper-meta">
      📅 2026-06-24
    </div>
    <details class="paper-abstract">
      Existing safety benchmarks target general adversarial scenarios but miss finance-specific risks. Financial LLMs face regulatory compliance violations, fraud facilitation, and systemic trust erosion that require targeted evaluation. We introduce FinRED, an expert-guided red-teaming framework for financial LLM safety evaluation developed with financial experts. FinRED uses a novel two-level taxonomy mapping global standards (e.g., FATF and EU DORA) to threats ranging from regulatory evasion to complex fraud, integrated with a scalable pipeline that converts real financial documents into context-rich red-teaming Behavioral Prompts (seeds) through an expert-defined schema. Rigorous expert validation confirms seed plausibility and realism for meaningful LLM safety evaluation. We also provide an expert-validated, finance-specific rubric that goes beyond disclaimer checks, aligns more closely with human experts than static one-size-fits-all rubrics, and reduces critical false negatives from 28 to 12. Aligned with internationally adopted risk-management and information-security standards (e.g., ISO/IEC 27001), FinRED is deployed in South Korea's Financial Security Institute (FSI) regulatory sandbox for generative AI security evaluation in real financial services. To mitigate dual-use risks, the dataset, generation pipeline, prompt template, and evaluation framework are gated for qualified researchers at https://github.com/selectstar-ai/FinRED-paper and https://huggingface.co/datasets/datumo/FinRED.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.25497v1">SAGE-Nav: Leveraging LLM Planning and Alignment Fusion for Hierarchical Scene Graph-Guided Navigation</a></div>
    <div class="paper-meta">
      📅 2026-06-24
      | 💬 Accepted by IROS 2026
    </div>
    <details class="paper-abstract">
      Object-Goal Navigation (ObjNav) requires embodied agents to autonomously locate specified targets using only egocentric visual observations. Existing monolithic methods struggle with long-horizon reasoning and generalize poorly to novel environments. To address these limitations, we propose SAGE-Nav, a novel hierarchical framework that integrates the reasoning capabilities of Large Language Models (LLMs) with dynamic scene graphs. Crucially, it decouples asynchronous global semantic planning from the high-frequency reactive control loop. The LLM serves as a global planner, decomposing abstract instructions into a sequence of semantically grounded waypoints. To translate these plans into dense multi-modal guidance, we design a Hierarchical Scene Graph Encoder (HSGE) that leverages relational graph convolutions to produce structure-aware embeddings preserving both semantic and spatial topology. Furthermore, we develop the Goal-aware Alignment-Fusion Network (GAFN) to dynamically fuse real-time perception with these structural priors. Using an adaptive gating mechanism with an explicit inductive bias, GAFN ensures robust visual-topological alignment for the low-level policy. Extensive evaluations in the i-THOR and RoboTHOR environments demonstrate that SAGE-Nav achieves state-of-the-art performance, delivering substantial gains in navigation efficiency and zero-shot generalization while maintaining the low control latency required for physical robotic deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.25489v1">When LLM Rationales Become User-Facing: Effects on Trust Perception, Decision-Making, and Gaze Behaviors</a></div>
    <div class="paper-meta">
      📅 2026-06-24
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly show step-by-step reasoning rationales alongside their answers, turning reasoning from an internal model capability into a user-facing interface feature. Yet it is unclear whether such rationales help users judge when trust is warranted or merely persuade through fluent reasoning. We address this gap through the lens of auditable trust calibration: user-facing rationales should help people inspect whether an answer is warranted by evidence. We test this framing in factual verification through two linked studies. Study 1, an online experiment (N=68), manipulated rationale presentation format (instant, delayed, on demand), rationale correctness (correct, incorrect), and certainty framing (none, certain, uncertain). Study 2, a controlled eye-tracking study (N=54), examined how no-, correct-, and incorrect-rationale conditions were associated with users' trust, decision-making, and eye-movement patterns. Study 1 showed no reliable presentation-format effects; instead, rationale correctness and certainty framing influenced the trust in the information, trust in the LLM system, and decision confidence. In Study 2, incorrect rationales drew more attention to the supporting evidence and larger pupil diameter while the rationale was viewed, consistent with greater cognitive effort. Incorrect rationales also lowered trust in LLM system relative to showing no rationale, whereas the no-rationale difference was weaker for trust in information. A post-hoc predictive modeling analysis of gaze data from Study 2 further showed that gaze features carried predictive signal for trust- and decision-related user states. This work challenges the assumption that more reasoning is always better and supports rationale designs that are selective, linked to evidence, calibrated in how they express certainty, and easier to verify.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.14581v2">CARE: Controlling LLM-Generated Policies through Auditable Review of Evidence in Scientific Experimentation</a></div>
    <div class="paper-meta">
      📅 2026-06-24
      | 💬 23 pages, 4 figures. Code: https://github.com/SHITIANYU-hue/CARE
    </div>
    <details class="paper-abstract">
      Granting LLMs direct control over costly, irreversible scientific experiments leads to unsafe exploration and unstable performance, but discarding LLM creativity entirely sacrifices significant optimization potential. We introduce CARE (Controlling LLM-Generated Policies through Auditable Review of Evidence in Scientific Experimentation), an auditable controller for high-throughput experimentation (HTE) optimization that keeps a non-LLM incumbent optimizer as the default action path while using LLMs to revise challenger ranking policies. Before each outcome is revealed, a public-evidence intervention gate compares the challenger with the incumbent. It authorizes the challenger's selection only when the evidence available before selection supports the change, with the decision recorded in the audit log. CARE outperforms all other evaluated methods on Minerva/Olympus and ChemLex benchmarks, with final-best improving from 80.0 to 88.5 on Minerva/Olympus and from 83.9 to 92.1 on ChemLex, relative to the public incumbent. Our experiments indicate that LLM self-evolution is more reliable when it expands the proposal space under an auditable controller, rather than directly choosing experiments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.25447v1">The Interplay of Harness Design and Post-Training in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-24
    </div>
    <details class="paper-abstract">
      Tool-integrated LLM agents are often wrapped within a harness: the scaffolding that determines which tools are exposed, how they are described, and what auxiliary information accompanies each per-step observation. While agents are routinely post-trained, this scaffolding is typically treated as a fixed engineering detail, with design effort limited to the training-free regime. Moreover, existing post-training algorithms assume a static environment, even though tool environments and tasks often shift upon deployment. To address this gap, we extend $\texttt{ALFWorld}$ (i) to treat the harness as a controllable design dimension and (ii) to support evaluation under task and tool environment shifts. Building on this, we systematically analyze how the harness design influences post-training in both in-distribution and out-of-distribution (OOD) settings. We empirically show that harness-aware post-training not only improves in-distribution performance but also enables agents to robustly adapt to OOD settings. Under a harness with minimal design effort, post-training suffers a drastic performance drop under stronger tool environment shifts, further highlighting the importance of harness-aware post-training under such shifts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.07306v2">Seeing the Reasoning: How LLM Rationales Influence User Trust and Decision-Making in Factual Verification Tasks</a></div>
    <div class="paper-meta">
      📅 2026-06-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) increasingly show reasoning rationales alongside their answers, turning "reasoning" into a user-interface element. While step-by-step rationales are typically associated with model performance, how they influence users' trust and decision-making in factual verification tasks remains unclear. We ran an online study (N=68) manipulating three properties of LLM reasoning rationales: presentation format (instant vs. delayed vs. on-demand), correctness (correct vs. incorrect), and certainty framing (none vs. certain vs. uncertain). We found that correct rationales and certainty cues increased trust, decision confidence, and AI advice adoption, whereas uncertainty cues reduced them. Presentation format did not have a significant effect, suggesting users were less sensitive to how reasoning was revealed than to its reliability. Participants indicated they use rationales to primarily audit outputs and calibrate trust, where they expected rationales in stepwise, adaptive forms with certainty indicators. Our work shows that user-facing rationales, if poorly designed, can both support decision-making yet miscalibrate trust.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.25444v1">Does Translation-Enhanced Speech Encoder Pre-training Affect Speech LLMs?</a></div>
    <div class="paper-meta">
      📅 2026-06-24
      | 💬 Accepted to Interspeech2026
    </div>
    <details class="paper-abstract">
      Connecting a pre-trained speech encoder to a Large Language Model (LLM) is the standard architecture for building Speech LLMs. However, a structural misalignment exists between the encoder and the LLM. Unlike encoders based on automatic speech recognition, which often produce representations in separate language-specific spaces, LLMs operate within a unified language-agnostic space. A mechanism is required to align the encoder's language-specific representations with the LLM's shared space. We argue that speech translation provides a principled way to achieve this. Unlike monolingual transcription, translation requires the model to bridge different languages and learn language-agnostic representations. We experimentally evaluate the impact of incorporating translation objectives into speech encoder pre-training. Our results demonstrate that translation-enhanced pre-training improves cross-modal integration and leads to superior performance across downstream Speech LLM tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.25426v1">Above the Inner Loop: Exceeding Accelerate at LLM Prefill GEMM on the M1 AMX</a></div>
    <div class="paper-meta">
      📅 2026-06-24
      | 💬 11 pages, 2 figures, 6 tables. Code: https://github.com/dbhan08/inferc
    </div>
    <details class="paper-abstract">
      On Apple Silicon the fp32 GEMMs dominating LLM prefill are dispatched by Accelerate to a matrix coprocessor (AMX) on the M1-M3. We ask where a hand-written kernel's throughput over Accelerate comes from on the M1 AMX, and reach a structural conclusion: not a faster inner loop. By microbenchmark the inner loop is load-issue bound -- once any operand load interleaves with the FMA32 stream, single-thread throughput falls to a 610-to-680 GFLOPS band, under half the load-free rate. The gain comes from two deployment-level levers Accelerate underuses: fine multi-thread panels filling the M1's second on-chip AMX block (winning the K >= N shapes), and pre-packing the constant weight at load (winning the N > K shapes). A bit-exact direct-AMX kernel using both is the fastest bit-exact fp32 GEMM path we find on the M1: it exceeds all three Accelerate fp32 paths (cblas_sgemm, BNNSMatMul, and the BNNS Graph compiler) at all twelve LLM prefill GEMMs at S = 128 (GPT-2 to Llama-7B scale), leading the fastest, BNNS Graph, by 1.17 -- and by 1.09 at the three shapes where it too holds fp32 -- with geometric means of 1.58 over BNNSMatMul and about 2.0x over cblas_sgemm. Every output is bit-identical to Accelerate, whereas BNNS Graph is bit-exact at only three of twelve shapes, the rest at reduced precision (error up to 1.4e-3). Dropped into llama.cpp for its cblas_sgemm prefill matmuls, it raises measured full-forward throughput from 291 to 420 tokens/s (1.44x, bit-identical) at 128-token prefill -- end-to-end, not a GEMM-only ratio. The contribution is this shape-resolved M1-AMX characterization (microbenchmark, two-block aggregate, per-core occupancy probe), leaving fine-panel scheduling and pre-packing as the only two levers above an inner loop at the hardware limit; mis-tuning the single column-panel width costs nearly 2x.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.25388v1">TabClean: Reusable LLM-Synthesized Programs for Tabular Data Cleaning</a></div>
    <div class="paper-meta">
      📅 2026-06-24
      | 💬 13 pages
    </div>
    <details class="paper-abstract">
      Reliable analytics and machine-learning pipelines depend on clean tabular data, yet production tables often contain missing values, typographical errors, inconsistent formats, violated dependencies, unit mismatches, and ambiguous categorical values. Existing cleaning systems make different trade-offs. Constraint-based systems need experts to specify rules. Learning-based systems need labels or retraining. Recent LLM-based cleaners reduce setup effort, but many call an LLM on rows, cells, or repeated workflow steps, so their cost grows with table size and with every recurring batch. We present TabClean, a model-training-free system that compiles LLM reasoning into reusable guarded cleaning programs. Given a dirty table and a small annotated development set, TabClean profiles table evidence, diagnoses repair mechanisms, synthesizes executable Python transformations, validates candidates with cell-level feedback, and commits the best program for reuse on schema-compatible batches. The key abstraction is an evidence-backed guarded repair clause. A deterministic transformation may fire only when its dirty pattern, target-negative condition, evidence support, and scope constraints are satisfied. Across six benchmarks, TabClean achieves high precision, improves F1 over representative rule-based, learning-based, and LLM-based baselines on five datasets, and substantially reduces recurring runtime and API cost by replacing repeated LLM inference with deterministic program execution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.25358v1">Agentic Knowledge Tracing: A Multi-Agent LLM Architecture for Stealth Assessment of Financial Literacy in Serious Games</a></div>
    <div class="paper-meta">
      📅 2026-06-24
      | 💬 8 pages, 5 figures, IEEE CoG 2026
    </div>
    <details class="paper-abstract">
      Assessing financial literacy during gameplay without disrupting the learning experience remains a key challenge in serious games for education. We present the Agentic BKT pipeline, a multi-agent large language model architecture for stealth assessment of financial competencies from open-ended gameplay events. The pipeline processes events from a 2D platformer serious game aligned with the OECD/INFE financial literacy framework through four phases: (1) the game captures every player decision as a structured event log; (2) an LLM event classifier labels each action on a four-point rubric validated against three domain experts (Fleiss kappa = 0.624, substantial agreement); (3) four domain-specific agents specializing in risk mitigation, investing, spending, and credit management perform session-level reasoning over behavioral trajectories, feeding per-competency Bayesian Knowledge Tracing that estimates mastery within each domain; and (4) an expert judge agent synthesizes the domain-level estimates into an overall mastery score. Evaluated with 193 K-12 participants across 264 game sessions, the Agentic BKT pipeline yields mastery estimates significantly correlated with learning gain (r = 0.276, p = 0.0001) and post-test scores (r = 0.333, p < 0.0001) while showing no correlation with pre-test scores, providing both convergent and discriminant validity. The multi-agent approach approximately triples the predictive validity of a single-LLM baseline (r = 0.095, not significant) in this study, demonstrating that domain decomposition and session-level reasoning play a central role in capturing the multidimensional nature of financial literacy from gameplay
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.25356v1">Representation Matters: An Empirical Study of Program Representations for LLM Vulnerability Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-06-24
      | 💬 34 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used for automated vulnerability detection, but it remains unclear how program structure and semantics should be represented for LLM-based reasoning. Most prompting-based approaches provide raw source code, implicitly assuming that more source-level context gives the model better evidence. This paper challenges that assumption through RepBench, an empirical benchmark comparing raw source code with static-analysis-based program representations. RepBench converts real-world C/C++ vulnerability testcases into multiple representations: raw source, Abstract Syntax Trees (ASTs), Control-Flow Graphs (CFGs), Program Dependence Graphs (PDGs), their combinations, and an auxiliary track of enriched PDGs (ePDGs). Using a curated PrimeVul-derived corpus of 107 Joern-based testcases across five CWE categories, we evaluate ten representation variants under a fixed Chain-of-Thought and structured-output protocol, plus 19 additional ePDG cases generated through VulChecker/Hector. Representation choice substantially affects LLM vulnerability reasoning. The strongest variant, AST+PDG, achieves 83.2% accuracy, compared with 53.5% for raw source. At the family level, graph-only prompts outperform both source-only and source-plus-graph prompts while requiring far less prompt overhead. These results reveal a context dilution effect: adding raw source code to compact structural graph evidence can degrade reasoning by making vulnerability-relevant evidence less salient. Overall, our findings show that carefully selected structural representations offer a better accuracy-overhead tradeoff than simply giving LLMs more raw input, and suggest that static analysis can serve as an effective prompt-construction layer for security-focused LLM reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.25353v1">Cache-Resident LLM Inference in GB-Scale Last-Level Caches</a></div>
    <div class="paper-meta">
      📅 2026-06-24
    </div>
    <details class="paper-abstract">
      Large language model (LLM) inference is increasingly dominated by data movement across the memory hierarchy. Recent 3D-stacked cache technologies have enabled GB-scale last-level caches in modern server CPUs, making it possible to keep reusable model weights on chip and exploit cache bandwidth and latency. Achieving this regime is not straightforward: deeper pipelining for weight residency increases in-flight requests and KV-cache footprint, while cache-resident operators make operator-boundary synchronization a visible bottleneck. We present a cache-resident execution model for inference on hierarchical-memory clustered systems. The model separates weight-centric operators from attention and KV-cache management into dedicated resource domains, keeping reusable weights cache-resident while scaling KV capacity independently of pipeline depth. It also relaxes synchronization from operator boundaries to true sub-operator dependencies, reducing coordination overhead in the cache-resident regime. We instantiate this model on a multi-socket CPU cluster with a weight-attention decoupled architecture, locality-aware placement, and a specialized static runtime. The prototype substantially outperforms equally provisioned llama.cpp. On deployed Llama-3.2-3B and Llama-2-7B configurations, it achieves 2.04x-11.51x speedup on time-per-output-token (TPOT). Under a validated analytical model, it further reaches up to 13.9x TPOT speedup across model sizes, context lengths, and batch sizes. These results show that commodity CPUs with GB-scale last-level caches can support efficient LLM inference when execution is organized around cache residency, decoupled state management, and dependency-aware coordination.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.24679v2">FlowPipe: LLM-Enhanced Conditional Generative Flow Networks for Data Preparation Pipeline Construction</a></div>
    <div class="paper-meta">
      📅 2026-06-24
      | 💬 Accepted by SIGMOD 2027
    </div>
    <details class="paper-abstract">
      Data preparation pipelines improve data quality in machine learning by transforming raw tables into learning-ready data through sequential cleaning and feature transformation operators. However, automatically constructing such pipelines is computationally difficult because operator sequences are combinatorial and end-to-end evaluation is expensive. Existing state-of-the-art (SOTA) Multi-DQN methods still face three key limitations: decoupled value estimators weaken long-horizon credit assignment, dataset context is only weakly injected into the policy, and exploration is inefficient in a sparse search space with many invalid states. To address these issues, we propose FlowPipe, a unified framework that formulates pipeline synthesis as conditional probabilistic flow generation over a directed acyclic graph. FlowPipe uses Conditional Generative Flow Networks (C-GFlowNets) with a Trajectory Balance objective to connect terminal validation rewards with early pipeline decisions. It further introduces Deep Semantic Modulation through Feature-wise Linear Modulation (FiLM), allowing LLM-derived logical priors to condition the policy's internal activations according to dataset semantics. In addition, FlowPipe incorporates failure awareness into the flow objective to avoid invalid states and concentrate search on high-potential regions. Experiments on two benchmark suites with 74 real-world datasets show that FlowPipe outperforms SOTA baselines, improving accuracy by 11.96% on average and achieving 12.5x faster training convergence. Source code is available at https://github.com/KunyuNi/FlowPipe.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.25332v1">Decoupling Reconnaissance and Exploitation: Measuring the Capability Boundaries of LLM-Based Web Penetration Testing</a></div>
    <div class="paper-meta">
      📅 2026-06-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown promise for automated penetration testing, yet existing end-to-end black-box evaluations are highly susceptible to error cascading: failures in early reconnaissance can mask an agent's actual ability to exploit vulnerabilities. To more accurately characterize these capabilities, we propose a two-stage decoupled evaluation framework that separates exploit execution from reconnaissance. Using ground-truth injection and knowledge-driven ablation across 70 high-fidelity web vulnerability testbeds, our framework isolates exploitation performance from reconnaissance noise. We empirically evaluate five open-source penetration-testing agents, covering multiagent, monolithic, and graph-driven architectures, on a strictly aligned subset of 50 representative vulnerabilities. The results reveal a substantial capability gap. With accurate vulnerability context, agents achieve a functional success rate of up to 90.0%, whereas autonomous reconnaissance, measured by targeted vulnerability recall, plateaus at approximately 50.0%, primarily due to failures in parsing unstructured telemetry. Cross-architectural analysis further reveals distinct capability niches: multi-agent isolation is more effective for long-sequence interactions such as de-serialization, while monolithic and graph-driven designs perform better on short-chain injections and cross-session access-control vulnerabilities, respectively. This decoupled evaluation work provides a fine-grained benchmarking protocol and an empirical basis for designing next-generation automated offensive security agents.
    </details>
</div>
