# llm - 2026_01

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- Part 2
- [Part 3](papers_3.md)
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
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15429v1">Domain-Specific Knowledge Graphs in RAG-Enhanced Healthcare LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) generate fluent answers but can struggle with trustworthy, domain-specific reasoning. We evaluate whether domain knowledge graphs (KGs) improve Retrieval-Augmented Generation (RAG) for healthcare by constructing three PubMed-derived graphs: $\mathbb{G}_1$ (T2DM), $\mathbb{G}_2$ (Alzheimer's disease), and $\mathbb{G}_3$ (AD+T2DM). We design two probes: Probe 1 targets merged AD T2DM knowledge, while Probe 2 targets the intersection of $\mathbb{G}_1$ and $\mathbb{G}_2$. Seven instruction-tuned LLMs are tested across retrieval sources {No-RAG, $\mathbb{G}_1$, $\mathbb{G}_2$, $\mathbb{G}_1$ + $\mathbb{G}_2$, $\mathbb{G}_3$, $\mathbb{G}_1$+$\mathbb{G}_2$ + $\mathbb{G}_3$} and three decoding temperatures. Results show that scope alignment between probe and KG is decisive: precise, scope-matched retrieval (notably $\mathbb{G}_2$) yields the most consistent gains, whereas indiscriminate graph unions often introduce distractors that reduce accuracy. Larger models frequently match or exceed KG-RAG with a No-RAG baseline on Probe 1, indicating strong parametric priors, whereas smaller/mid-sized models benefit more from well-scoped retrieval. Temperature plays a secondary role; higher values rarely help. We conclude that precision-first, scope-matched KG-RAG is preferable to breadth-first unions, and we outline practical guidelines for graph selection, model sizing, and retrieval/reranking. Code and Data available here - https://github.com/sydneyanuyah/RAGComparison
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15397v1">Beyond Prompting: Efficient and Robust Contextual Biasing for Speech LLMs via Logit-Space Integration (LOGIC)</a></div>
    <div class="paper-meta">
      📅 2026-01-21
    </div>
    <details class="paper-abstract">
      The rapid emergence of new entities -- driven by cultural shifts, evolving trends, and personalized user data -- poses a significant challenge for existing Speech Large Language Models (Speech LLMs). While these models excel at general conversational tasks, their static training knowledge limits their ability to recognize domain-specific terms such as contact names, playlists, or technical jargon. Existing solutions primarily rely on prompting, which suffers from poor scalability: as the entity list grows, prompting encounters context window limitations, increased inference latency, and the "lost-in-the-middle" phenomenon. An alternative approach, Generative Error Correction (GEC), attempts to rewrite transcripts via post-processing but frequently suffers from "over-correction", introducing hallucinations of entities that were never spoken. In this work, we introduce LOGIC (Logit-Space Integration for Contextual Biasing), an efficient and robust framework that operates directly in the decoding layer. Unlike prompting, LOGIC decouples context injection from input processing, ensuring constant-time complexity relative to prompt length. Extensive experiments using the Phi-4-MM model across 11 multilingual locales demonstrate that LOGIC achieves an average 9% relative reduction in Entity WER with a negligible 0.30% increase in False Alarm Rate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15385v1">VegaChat: A Robust Framework for LLM-Based Chart Generation and Assessment</a></div>
    <div class="paper-meta">
      📅 2026-01-21
      | 💬 8 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Natural-language-to-visualization (NL2VIS) systems based on large language models (LLMs) have substantially improved the accessibility of data visualization. However, their further adoption is hindered by two coupled challenges: (i) the absence of standardized evaluation metrics makes it difficult to assess progress in the field and compare different approaches; and (ii) natural language descriptions are inherently underspecified, so multiple visualizations may be valid for the same query. To address these issues, we introduce VegaChat, a framework for generating, validating, and assessing declarative visualizations from natural language. We propose two complementary metrics: Spec Score, a deterministic metric that measures specification-level similarity without invoking an LLM, and Vision Score, a library-agnostic, image-based metric that leverages a multimodal LLM to assess chart similarity and prompt compliance. We evaluate VegaChat on the NLV Corpus and on the annotated subset of ChartLLM. VegaChat achieves near-zero rates of invalid or empty visualizations, while Spec Score and Vision Score exhibit strong correlation with human judgments (Pearson 0.65 and 0.71, respectively), indicating that the proposed metrics support consistent, cross-library comparison. The code and evaluation artifacts are available at https://zenodo.org/records/17062309.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15352v1">A Prompt-Based Framework for Loop Vulnerability Detection Using Local LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-21
      | 💬 Accepted and Waiting to be published ICAI'25: 27th International Conference on Artificial Intelligence https://american-cse.org/csce2025/conferences-ICAI
    </div>
    <details class="paper-abstract">
      Loop vulnerabilities are one major risky construct in software development. They can easily lead to infinite loops or executions, exhaust resources, or introduce logical errors that degrade performance and compromise security. The problem are often undetected by traditional static analyzers because such tools rely on syntactic patterns, which makes them struggle to detect semantic flaws. Consequently, Large Language Models (LLMs) offer new potential for vulnerability detection because of their ability to understand code contextually. Moreover, local LLMs unlike commercial ones like ChatGPT or Gemini addresses issues such as privacy, latency, and dependency concerns by facilitating efficient offline analysis. Consequently, this study proposes a prompt-based framework that utilize local LLMs for the detection of loop vulnerabilities within Python 3.7+ code. The framework targets three categories of loop-related issues, such as control and logic errors, security risks inside loops, and resource management inefficiencies. A generalized and structured prompt-based framework was designed and tested with two locally deployed LLMs (LLaMA 3.2; 3B and Phi 3.5; 4B) by guiding their behavior via iterative prompting. The designed prompt-based framework included key safeguarding features such as language-specific awareness, code-aware grounding, version sensitivity, and hallucination prevention. The LLM results were validated against a manually established baseline truth, and the results indicate that Phi outperforms LLaMA in precision, recall, and F1-score. The findings emphasize the importance of designing effective prompts for local LLMs to perform secure and accurate code vulnerability analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15348v1">Abusive music and song transformation using GenAI and LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-21
    </div>
    <details class="paper-abstract">
      Repeated exposure to violence and abusive content in music and song content can influence listeners' emotions and behaviours, potentially normalising aggression or reinforcing harmful stereotypes. In this study, we explore the use of generative artificial intelligence (GenAI) and Large Language Models (LLMs) to automatically transform abusive words (vocal delivery) and lyrical content in popular music. Rather than simply muting or replacing a single word, our approach transforms the tone, intensity, and sentiment, thus not altering just the lyrics, but how it is expressed. We present a comparative analysis of four selected English songs and their transformed counterparts, evaluating changes through both acoustic and sentiment-based lenses. Our findings indicate that Gen-AI significantly reduces vocal aggressiveness, with acoustic analysis showing improvements in Harmonic to Noise Ratio, Cepstral Peak Prominence, and Shimmer. Sentiment analysis reduced aggression by 63.3-85.6\% across artists, with major improvements in chorus sections (up to 88.6\% reduction). The transformed versions maintained musical coherence while mitigating harmful content, offering a promising alternative to traditional content moderation that avoids triggering the "forbidden fruit" effect, where the censored content becomes more appealing simply because it is restricted. This approach demonstrates the potential for GenAI to create safer listening experiences while preserving artistic expression.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.08710v2">Thinking Longer, Not Always Smarter: Evaluating LLM Capabilities in Hierarchical Legal Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 15 pages, 7 figures, Proceedings of the 2026 Symposium on Computer Science and Law (CSLAW '26)
    </div>
    <details class="paper-abstract">
      Case-based reasoning is a cornerstone of U.S. legal practice, requiring professionals to argue about a current case by drawing analogies to and distinguishing from past precedents. While Large Language Models (LLMs) have shown remarkable capabilities, their proficiency in this complex, nuanced form of reasoning needs further investigation. We propose a formal framework that decomposes the process of identifying significant distinctions between cases into three-stage reasoning tasks. Our framework models cases using factual predicates called factors, organizes them into a legal knowledge hierarchy, and defines verifiable rules for identifying distinctions, analyzing their argumentative support, and evaluating their significance. Through comprehensive evaluation of modern reasoning LLMs, we reveal a paradox: while models achieve high accuracy on surface-level reasoning (Task 1), performance degrades on hierarchical reasoning (Task 2: 64.82%-92.09%) and collapses on integrated analysis (Task 3: 11.46%-33.99%). Most strikingly, we find that models consistently expend more computational resources on incorrect responses than correct ones, suggesting that "thinking longer" does not always mean "thinking smarter." Our work provides a methodology for fine-grained analysis of LLM reasoning capabilities in complex domains and reveals fundamental limitations that must be addressed for robust and trustworthy legal AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03597v2">From Chains to Graphs: Self-Structured Reasoning for General-Domain LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) show strong reasoning ability in open-domain question answering, yet their reasoning processes are typically linear and often logically inconsistent. In contrast, real-world reasoning requires integrating multiple premises and solving subproblems in parallel. Existing methods, such as Chain-of-Thought (CoT), express reasoning in a linear textual form, which may appear coherent but frequently leads to inconsistent conclusions. Recent approaches rely on externally provided graphs and do not explore how LLMs can construct and use their own graph-structured reasoning, particularly in open-domain QA. To fill this gap, we novelly explore graph-structured reasoning of LLMs in general-domain question answering. We propose Self-Graph Reasoning (SGR), a framework that enables LLMs to explicitly represent their reasoning process as a structured graph before producing the final answer. We further construct a graph-structured reasoning dataset that merges multiple candidate reasoning graphs into refined graph structures for model training. Experiments on five QA benchmarks across both general and specialized domains show that SGR consistently improves reasoning consistency and yields a 17.74% gain over the base model. The LLaMA-3.3-70B model fine-tuned with SGR performs comparably to GPT-4o and surpasses Claude-3.5-Haiku, demonstrating the effectiveness of graph-structured reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14532v1">Search over Self-Edit Strategies for LLM Adaptation</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Many LLM-based open-ended search systems freeze the foundation model that proposes improvements to existing solutions, which may bottleneck long-run progress. Recent work has explored updating the proposal model at test time [arXiv:2511.23473], but the update strategy is still typically hand-specified. Therefore, this study investigated whether an LLM can use task feedback to decide how it should update its weights. For tractability, we focused on the simpler case where there is only one round of self-improvement, and restricted the update operator to self-supervised next token prediction (NTP), leaving the model freedom in choosing its training data and key NTP hyperparameters. Using the Self-Adapting Language Models (SEAL) [arXiv:2506.10943] framework as a testbed, we relaxed its fixed human template constraint and allowed the model to generate its own self-edit templates, thereby giving it more control over its training data and hyperparameters. Two variants were studied, differing in whether template generation was conditioned on a lightweight archive of past templates. In SEAL's Single-Passage Knowledge Incorporation setting with Qwen3-8B on SQuAD [arXiv:1606.05250], the no-archive variant performed comparably to the weaker "Implications" baseline, while the archive variant outperformed "Implications" and approached the strongest human-designed "Rewrite" baseline without surpassing it. Further analysis of collapse in the model's exploration revealed that a naive archive can confer some short-term robustness but can also accelerate homogenization, suggesting that explicit novelty pressure may be required to consistently advance beyond carefully optimized human strategies. Our code is available at https://github.com/cheongalc/search-self-edit-strategies .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14528v1">LLM Security and Safety: Insights from Homotopy-Inspired Prompt Obfuscation</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      In this study, we propose a homotopy-inspired prompt obfuscation framework to enhance understanding of security and safety vulnerabilities in Large Language Models (LLMs). By systematically applying carefully engineered prompts, we demonstrate how latent model behaviors can be influenced in unexpected ways. Our experiments encompassed 15,732 prompts, including 10,000 high-priority cases, across LLama, Deepseek, KIMI for code generation, and Claude to verify. The results reveal critical insights into current LLM safeguards, highlighting the need for more robust defense mechanisms, reliable detection strategies, and improved resilience. Importantly, this work provides a principled framework for analyzing and mitigating potential weaknesses, with the goal of advancing safe, responsible, and trustworthy AI technologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01129v2">RovoDev Code Reviewer: A Large-Scale Online Evaluation of LLM-based Code Review Automation at Atlassian</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 Accepted at the 48th International Conference on Software Engineering (ICSE'26), SEIP Track. 12 Pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs)-powered code review automation has the potential to transform code review workflows. Despite the advances of LLM-powered code review comment generation approaches, several practical challenges remain for designing enterprise-grade code review automation tools. In particular, this paper aims at answering the practical question: how can we design a review-guided, context-aware, quality-checked code review comment generation without fine-tuning? In this paper, we present RovoDev Code Reviewer, an enterprise-grade LLM-based code review automation tool designed and deployed at scale within Atlassian's development ecosystem with seamless integration into Atlassian's Bitbucket. Through the offline, online, user feedback evaluations over a one-year period, we conclude that RovoDev Code Reviewer is effective in generating code review comments that could lead to code resolution for 38.70% (i.e., comments that triggered code changes in the subsequent commits); and offers the promise of accelerating feedback cycles (i.e., decreasing the PR cycle time by 30.8%), alleviating reviewer workload (i.e., reducing the number of human-written comments by 35.6%), and improving overall software quality (i.e., finding errors with actionable suggestions).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.17792v4">H3Fusion: Helpful, Harmless, Honest Fusion of Aligned LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      The alignment of pre-trained LLMs continues to draw significant attention from both industry and academia, aiming to ensure responses that are helpful, harmless, and honest. However, identifying a point in the model's representation subspace that simultaneously satisfies all these properties remains challenging. H3Fusion addresses this challenge by introducing a mixture-of-experts (MoE)-based fusion mechanism that models alignment as a controllable drift within the subspace, guided by a drift-regularization loss to balance competing alignment dimensions. Furthermore, we formulate the alignment by finding a dual objective of harnessing the distance of generated embeddings and alignment embeddings, and introduce a gating loss by canalizing the activations on the contributing experts. Extensive evaluations of three benchmark datasets show that H3Fusion is more helpful, less harmful, and more honest in three aspects: it outperforms each individually aligned model by 11.37%, and provides stronger robustness compared to the state-of-the-art LLM ensemble approaches by 13.77% and model-merging approaches by 6.18%. Code is available at https://github.com/git-disl/h3fusion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03149v2">PersonaLedger: Generating Realistic Financial Transactions with Persona Conditioned LLMs and Rule Grounded Feedback</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Strict privacy regulations limit access to real transaction data, slowing open research in financial AI. Synthetic data can bridge this gap, but existing generators do not jointly achieve behavioral diversity and logical groundedness. Rule-driven simulators rely on hand-crafted workflows and shallow stochasticity, which miss the richness of human behavior. Learning-based generators such as GANs capture correlations yet often violate hard financial constraints and still require training on private data. We introduce PersonaLedger, a generation engine that uses a large language model conditioned on rich user personas to produce diverse transaction streams, coupled with an expert configurable programmatic engine that maintains correctness. The LLM and engine interact in a closed loop: after each event, the engine updates the user state, enforces financial rules, and returns a context aware "nextprompt" that guides the LLM toward feasible next actions. With this engine, we create a public dataset of 30 million transactions from 23,000 users and a benchmark suite with two tasks, illiquidity classification and identity theft segmentation. PersonaLedger offers a realistic, privacy preserving resource that supports rigorous evaluation of forecasting and anomaly detection models. PersonaLedger offers the community a rich, realistic, and privacy preserving resource -- complete with code, rules, and generation logs -- to accelerate innovation in financial AI and enable rigorous, reproducible evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14479v1">Can LLM Reasoning Be Trusted? A Comparative Study: Using Human Benchmarking on Statistical Tasks</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      This paper investigates the ability of large language models (LLMs) to solve statistical tasks, as well as their capacity to assess the quality of reasoning. While state-of-the-art LLMs have demonstrated remarkable performance in a range of NLP tasks, their competence in addressing even moderately complex statistical challenges is not well understood. We have fine-tuned selected open-source LLMs on a specially developed dataset to enhance their statistical reasoning capabilities, and compared their performance with the human scores used as a benchmark. Our results show that the fine-tuned models achieve better performance on advanced statistical tasks on the level comparable to a statistics student. Fine-tuning demonstrates architecture-dependent improvements, with some models showing significant performance gains, indicating clear potential for deployment in educational technology and statistical analysis assistance systems. We also show that LLMs themselves can be far better judges of the answers quality (including explanation and reasoning assessment) in comparison to traditional metrics, such as BLEU or BertScore. This self-evaluation capability enables scalable automated assessment for statistical education platforms and quality assurance in automated analysis tools. Potential applications also include validation tools for research methodology in academic and industry settings, and quality control mechanisms for data analysis workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14456v1">On the Generalization Gap in LLM Planning: Tests and Verifier-Reward RL</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 9 pages, 4 figures, 3 tables, 2 pages of supplementary materials. Submitted to a conference implementing a double-blind review process
    </div>
    <details class="paper-abstract">
      Recent work shows that fine-tuned Large Language Models (LLMs) can achieve high valid plan rates on PDDL planning tasks. However, it remains unclear whether this reflects transferable planning competence or domain-specific memorization. In this work, we fine-tune a 1.7B-parameter LLM on 40,000 domain-problem-plan tuples from 10 IPC 2023 domains, and evaluate both in-domain and cross-domain generalization. While the model reaches 82.9% valid plan rate in in-domain conditions, it achieves 0% on two unseen domains. To analyze this failure, we introduce three diagnostic interventions, namely (i) instance-wise symbol anonymization, (ii) compact plan serialization, and (iii) verifier-reward fine-tuning using the VAL validator as a success-focused reinforcement signal. Symbol anonymization and compact serialization cause significant performance drops despite preserving plan semantics, thus revealing strong sensitivity to surface representations. Verifier-reward fine-tuning reaches performance saturation in half the supervised training epochs, but does not improve cross-domain generalization. For the explored configurations, in-domain performance plateaus around 80%, while cross-domain performance collapses, suggesting that our fine-tuned model relies heavily on domain-specific patterns rather than transferable planning competence in this setting. Our results highlight a persistent generalization gap in LLM-based planning and provide diagnostic tools for studying its causes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.11150v3">Causal Judge Evaluation: Calibrated Surrogate Metrics for LLM Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 Code: https://github.com/cimo-labs/cje Experiments for Reproducibility: https://github.com/cimo-labs/cje-arena-experiments Original Preprint: https://zenodo.org/records/17903629
    </div>
    <details class="paper-abstract">
      Measuring long-run LLM outcomes (user satisfaction, expert judgment, downstream KPIs) is expensive. Teams default to cheap LLM judges, but uncalibrated proxies can invert rankings entirely. Causal Judge Evaluation (CJE) makes it affordable to aim at the right target: calibrate cheap scores against a small oracle slice, then evaluate at scale with valid uncertainty. We treat surrogate validity as auditable: for each policy or deployment context, a small oracle audit tests whether the learned calibration remains mean-unbiased, turning an uncheckable identification condition into a falsifiable diagnostic. On 4,961 Chatbot Arena prompts comparing five policies with a 16x oracle/judge cost ratio, at a 5% oracle fraction CJE achieves 99% pairwise ranking accuracy at 14x lower cost; across all configurations (5-50% oracle, varying n), accuracy averages 94%. An adversarial policy fails the transport audit and is correctly flagged; in such cases CJE refuses level claims rather than reporting biased estimates. Key findings: naive confidence intervals on raw judge scores achieve 0% coverage (CJE: ~95%); importance-weighted estimators fail despite >90% effective sample size; and the Coverage-Limited Efficiency (CLE) bound and its TTC diagnostic explain why.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14340v1">Turn-Based Structural Triggers: Prompt-Free Backdoors in Multi-Turn LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely integrated into interactive systems such as dialogue agents and task-oriented assistants. This growing ecosystem also raises supply-chain risks, where adversaries can distribute poisoned models that degrade downstream reliability and user trust. Existing backdoor attacks and defenses are largely prompt-centric, focusing on user-visible triggers while overlooking structural signals in multi-turn conversations. We propose Turn-based Structural Trigger (TST), a backdoor attack that activates from dialogue structure, using the turn index as the trigger and remaining independent of user inputs. Across four widely used open-source LLM models, TST achieves an average attack success rate (ASR) of 99.52% with minimal utility degradation, and remains effective under five representative defenses with an average ASR of 98.04%. The attack also generalizes well across instruction datasets, maintaining an average ASR of 99.19%. Our results suggest that dialogue structure constitutes an important and under-studied attack surface for multi-turn LLM systems, motivating structure-aware auditing and mitigation in practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03013v3">LLMs, You Can Evaluate It! Design of Multi-perspective Report Evaluation for Security Operation Centers</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Security operation centers (SOCs) often produce analysis reports on security incidents, and large language models (LLMs) will likely be used for this task in the near future. We postulate that a better understanding of how veteran analysts evaluate reports, including their feedback, can help produce analysis reports in SOCs. In this paper, we aim to leverage LLMs for analysis reports. To this end, we first construct a Analyst-wise checklist to reflect SOC practitioners' opinions for analysis report evaluation through literature review and user study with SOC practitioners. Next, we design a novel LLM-based conceptual framework, named MESSALA, by further introducing two new techniques, granularization guideline and multi-perspective evaluation. MESSALA can maximize report evaluation and provide feedback on veteran SOC practitioners' perceptions. When we conduct extensive experiments with MESSALA, the evaluation results by MESSALA are the closest to those of veteran SOC practitioners compared with the existing LLM-based methods. We then show two key insights. We also conduct qualitative analysis with MESSALA, and then identify that MESSALA can provide actionable items that are necessary for improving analysis reports.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06266v3">Self-Admitted Technical Debt in LLM Software: An Empirical Comparison with ML and Non-ML Software</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 Accepted to SANER 2026 (IEEE International Conference on Software Analysis, Evolution and Reengineering)
    </div>
    <details class="paper-abstract">
      Self-admitted technical debt (SATD), referring to comments flagged by developers that explicitly acknowledge suboptimal code or incomplete functionality, has received extensive attention in machine learning (ML) and traditional (Non-ML) software. However, little is known about how SATD manifests and evolves in contemporary Large Language Model (LLM)-based systems, whose architectures, workflows, and dependencies differ fundamentally from both traditional and pre-LLM ML software. In this paper, we conduct the first empirical study of SATD in the LLM era, replicating and extending prior work on ML technical debt to modern LLM-based systems. We compare SATD prevalence across LLM, ML, and non-ML repositories across a total of 477 repositories (159 per category). We perform survival analysis of SATD introduction and removal to understand the dynamics of technical debt across different development paradigms. Surprisingly, despite their architectural complexity, our results reveal that LLM repositories accumulate SATD at similar rates to ML systems (3.95% vs. 4.10%). However, we observe that LLM repositories remain debt-free 2.4x longer than ML repositories (a median of 492 days vs. 204 days), and then start to accumulate technical debt rapidly. Moreover, our qualitative analysis of 377 SATD instances reveals three new forms of technical debt unique to LLM-based development that have not been reported in prior research: Model-Stack Workaround Debt, Model Dependency Debt, and Performance Optimization Debt. Finally, by mapping SATD to stages of the LLM development pipeline, we observe that debt concentrates
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14209v1">InT: Self-Proposed Interventions Enable Credit Assignment in LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Outcome-reward reinforcement learning (RL) has proven effective at improving the reasoning capabilities of large language models (LLMs). However, standard RL assigns credit only at the level of the final answer, penalizing entire reasoning traces when the outcome is incorrect and uniformly reinforcing all steps when it is correct. As a result, correct intermediate steps may be discouraged in failed traces, while spurious steps may be reinforced in successful ones. We refer to this failure mode as the problem of credit assignment. While a natural remedy is to train a process reward model, accurately optimizing such models to identify corrective reasoning steps remains challenging. We introduce Intervention Training (InT), a training paradigm in which the model performs fine-grained credit assignment on its own reasoning traces by proposing short, targeted corrections that steer trajectories toward higher reward. Using reference solutions commonly available in mathematical reasoning datasets and exploiting the fact that verifying a model-generated solution is easier than generating a correct one from scratch, the model identifies the first error in its reasoning and proposes a single-step intervention to redirect the trajectory toward the correct solution. We then apply supervised fine-tuning (SFT) to the on-policy rollout up to the point of error concatenated with the intervention, localizing error to the specific step that caused failure. We show that the resulting model serves as a far better initialization for RL training. After running InT and subsequent fine-tuning with RL, we improve accuracy by nearly 14% over a 4B-parameter base model on IMO-AnswerBench, outperforming larger open-source models such as gpt-oss-20b.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.15364v4">KeyDiff: Key Similarity-Based KV Cache Eviction for Long-Context LLM Inference in Resource-Constrained Environments</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 37 pages, 19 figures, NeurIPS 2025
    </div>
    <details class="paper-abstract">
      We demonstrate that geometrically distinctive keys during LLM inference tend to have high attention scores. Based on the phenomenon we propose KeyDiff, a training-free KV cache eviction method based solely on key similarity. Unlike other KV cache eviction methods, KeyDiff can process arbitrarily long prompts within strict resource constraints and efficiently generate responses. We provide a theoretical basis for KeyDiff by relating key diversity with attention scores. These results imply KeyDiff can efficiently identify the most important tokens to retain. Notably KeyDiff does not rely on attention scores, allowing the use of optimized attention mechanisms like FlashAttention. Under a strict memory allowance, we demonstrate the effectiveness of KeyDiff for the Llama and Qwen model families by observing a performance gap of less than 0.04% with 8K cache budget ($\sim$23% KV cache reduction) from the non-evicting baseline on LongBench for Llama 3.1-8B and Llama 3.2-3B. We also observe near baseline performance for Deepseek-R1-Distill-Llama-8B on the Math500 reasoning benchmark and decrease end-to-end inference latency by up to 30% compared to the other token-eviction methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14154v1">LLM Augmented Intervenable Multimodal Adaptor for Post-operative Complication Prediction in Lung Cancer Surgery</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 Accepted to P2P-CV @ WACV 2026
    </div>
    <details class="paper-abstract">
      Postoperative complications remain a critical concern in clinical practice, adversely affecting patient outcomes and contributing to rising healthcare costs. We present MIRACLE, a deep learning architecture for prediction of risk of postoperative complications in lung cancer surgery by integrating preoperative clinical and radiological data. MIRACLE employs a hyperspherical embedding space fusion of heterogeneous inputs, enabling the extraction of robust, discriminative features from both structured clinical records and high-dimensional radiological images. To enhance transparency of prediction and clinical utility, we incorporate an interventional deep learning module in MIRACLE, that not only refines predictions but also provides interpretable and actionable insights, allowing domain experts to interactively adjust recommendations based on clinical expertise. We validate our approach on POC-L, a real-world dataset comprising 3,094 lung cancer patients who underwent surgery at Roswell Park Comprehensive Cancer Center. Our results demonstrate that MIRACLE outperforms various traditional machine learning models and contemporary large language models (LLM) variants alone, for personalized and explainable postoperative risk management.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.17001v3">PersonalAI: A Systematic Comparison of Knowledge Graph Storage and Retrieval Approaches for Personalized LLM agents</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Personalizing language models that effectively incorporating user interaction history remains a central challenge in development of adaptive AI systems. While large language models (LLMs), combined with Retrieval-Augmented Generation (RAG), have improved factual accuracy, they often lack structured memory and fail to scale in complex, long-term interactions. To address this, we propose a flexible external memory framework based on knowledge graph, which construct and update memory model automatically by LLM itself. Building upon the AriGraph architecture, we introduce a novel hybrid graph design that supports both standard edges and two types of hyper-edges, enabling rich and dynamic semantic and temporal representations. Our framework also supports diverse retrieval mechanisms, including A*, water-circle traversal, beam search and hybrid methods, making it adaptable to different datasets and LLM capacities. We evaluate our system on three benchmarks: TriviaQA, HotpotQA, DiaASQ and demonstrate that different memory and retrieval configurations yield optimal performance depending on the task. Additionally, we extend the DiaASQ benchmark with temporal annotations and internally contradictory statements, showing that our system remains robust and effective in managing temporal dependencies and context-aware reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.11574v4">Quantization Meets Reasoning: Exploring and Mitigating Degradation of Low-Bit LLMs in Mathematical Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 27pages
    </div>
    <details class="paper-abstract">
      Low-bit post-training quantization (PTQ) is a practical route to deploy reasoning-capable LLMs under tight memory and latency budgets, yet it can markedly impair mathematical reasoning (drops up to 69.81% in our harder settings). We address two deployment-critical questions with process-level precision: Where along a step-structured solution does degradation first arise? How to mitigate it while staying in the low-bit regime? Across widely used PTQ methods (AWQ, GPTQ, SmoothQuant), open-source model families (Qwen, LLaMA; 0.5--7B), and math reasoning benchmarks (GSM8K, MATH, AIME), we perform format-aligned chain-of-thought with step-aligned attribution and uncover two robust regularities: (i) PTQ disproportionately elevates method and execution errors relative to high-level conceptual mistakes; and (ii) failures emerge early, with the first vulnerable step flipping and cascading to the final answer. These regularities suggest a general intervention principle: restore local token-level margins exactly at the earliest failure frontier. We instantiate this principle as a lightweight measure$\rightarrow$locate$\rightarrow$restore loop that operates directly on the quantized model: detect the first faulty step, construct our "Silver Bullet" datasets, and apply small-scale supervised/preference tuning. In our settings, as few as 332 curated examples and 3--5 minutes of compute on a single GPU recover 4-bit weight math reasoning toward the full-precision baseline while preserving PTQ efficiency. Our framework is quantizer- and architecture-agnostic within the evaluated regimes, and turns low-bit degradation from a global accuracy problem into a local, reproducible process intervention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.04492v2">Learned Hallucination Detection in Black-Box LLMs using Token-level Entropy Production Rate</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 8 pages, 5 figures, 2 tables. pre-print version
    </div>
    <details class="paper-abstract">
      Hallucinations in Large Language Model (LLM) outputs for Question Answering (QA) tasks can critically undermine their real-world reliability. This paper introduces a methodology for robust, one-shot hallucination detection, specifically designed for scenarios with limited data access, such as interacting with black-box LLM APIs that typically expose only a few top candidate log-probabilities per token. Our approach derives uncertainty indicators directly from these readily available log-probabilities generated during non-greedy decoding. We first derive an Entropy Production Rate (EPR) that offers baseline performance, later augmented with supervised learning. Our learned model leverages the entropic contributions of the accessible top-ranked tokens within a single generated sequence, without multiple re-runs per query. Evaluated across diverse QA datasets and multiple LLMs, this estimator significantly improves token-level hallucination detection over state-of-the-art methods. Crucially, high performance is demonstrated using only the typically small set of available log-probabilities (e.g., top-10 per token), confirming its practical efficiency and suitability for API-constrained deployments. This work provides a lightweight technique to enhance the trustworthiness of LLM responses, at the token level, after a single generation pass, for QA and Retrieval-Augmented Generation (RAG) systems. Our experiments confirmed the performance of our method against existing approaches on public dataset as well as for a financial framework analyzing annual company reports.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.07902v2">Tailored Emotional LLM-Supporter: Enhancing Cultural Sensitivity</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 Joint first authors; EACL
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) show promise in offering emotional support and generating empathetic responses for individuals in distress, but their ability to deliver culturally sensitive support remains underexplored due to a lack of resources. In this work, we introduce CultureCare, the first dataset designed for this task, spanning four cultures and including 1729 distress messages, 1523 cultural signals, and 1041 support strategies with fine-grained emotional and cultural annotations. Leveraging CultureCare, we (i) develop and test four adaptation strategies for guiding three state-of-the-art LLMs toward culturally sensitive responses; (ii) conduct comprehensive evaluations using LLM-as-a-Judge, in-culture human annotators, and clinical psychologists; (iii) show that adapted LLMs outperform anonymous online peer responses, and that simple cultural role-play is insufficient for cultural sensitivity; and (iv) explore the application of LLMs in clinical training, where experts highlight their potential in fostering cultural competence in novice therapists.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14063v1">XCR-Bench: A Multi-Task Benchmark for Evaluating Cultural Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 30 Pages, 13 Figures
    </div>
    <details class="paper-abstract">
      Cross-cultural competence in large language models (LLMs) requires the ability to identify Culture-Specific Items (CSIs) and to adapt them appropriately across cultural contexts. Progress in evaluating this capability has been constrained by the scarcity of high-quality CSI-annotated corpora with parallel cross-cultural sentence pairs. To address this limitation, we introduce XCR-Bench, a Cross(X)-Cultural Reasoning Benchmark consisting of 4.9k parallel sentences and 1,098 unique CSIs, spanning three distinct reasoning tasks with corresponding evaluation metrics. Our corpus integrates Newmark's CSI framework with Hall's Triad of Culture, enabling systematic analysis of cultural reasoning beyond surface-level artifacts and into semi-visible and invisible cultural elements such as social norms, beliefs, and values. Our findings show that state-of-the-art LLMs exhibit consistent weaknesses in identifying and adapting CSIs related to social etiquette and cultural reference. Additionally, we find evidence that LLMs encode regional and ethno-religious biases even within a single linguistic setting during cultural adaptation. We release our corpus and code to facilitate future research on cross-cultural NLP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14050v1">Understanding Multilingualism in Mixture-of-Experts LLMs: Routing Mechanism, Expert Specialization, and Layerwise Steering</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Mixture-of-Experts (MoE) architectures have shown strong multilingual capabilities, yet the internal mechanisms underlying performance gains and cross-language differences remain insufficiently understood. In this work, we conduct a systematic analysis of MoE models, examining routing behavior and expert specialization across languages and network depth. Our analysis reveals that multilingual processing in MoE models is highly structured: routing aligns with linguistic families, expert utilization follows a clear layerwise pattern, and high-resource languages rely on shared experts while low-resource languages depend more on language-exclusive experts despite weaker performance. Layerwise interventions further show that early and late MoE layers support language-specific processing, whereas middle layers serve as language-agnostic capacity hubs. Building on these insights, we propose a routing-guided steering method that adaptively guides routing behavior in middle layers toward shared experts associated with dominant languages at inference time, leading to consistent multilingual performance improvements, particularly for linguistically related language pairs. Our code is available at https://github.com/conctsai/Multilingualism-in-Mixture-of-Experts-LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14032v1">RM-Distiller: Exploiting Generative LLM for Reward Model Distillation</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Reward models (RMs) play a pivotal role in aligning large language models (LLMs) with human preferences. Due to the difficulty of obtaining high-quality human preference annotations, distilling preferences from generative LLMs has emerged as a standard practice. However, existing approaches predominantly treat teacher models as simple binary annotators, failing to fully exploit the rich knowledge and capabilities for RM distillation. To address this, we propose RM-Distiller, a framework designed to systematically exploit the multifaceted capabilities of teacher LLMs: (1) Refinement capability, which synthesizes highly correlated response pairs to create fine-grained and contrastive signals. (2) Scoring capability, which guides the RM in capturing precise preference strength via a margin-aware optimization objective. (3) Generation capability, which incorporates the teacher's generative distribution to regularize the RM to preserve its fundamental linguistic knowledge. Extensive experiments demonstrate that RM-Distiller significantly outperforms traditional distillation methods both on RM benchmarks and reinforcement learning-based alignment, proving that exploiting multifaceted teacher capabilities is critical for effective reward modeling. To the best of our knowledge, this is the first systematic research on RM distillation from generative LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13995v1">From Tags to Trees: Structuring Fine-Grained Knowledge for Controllable Data Selection in LLM Instruction Tuning</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Effective and controllable data selection is critical for LLM instruction tuning, especially with massive open-source datasets. Existing approaches primarily rely on instance-level quality scores, or diversity metrics based on embedding clusters or semantic tags. However, constrained by the flatness of embedding spaces or the coarseness of tags, these approaches overlook fine-grained knowledge and its intrinsic hierarchical dependencies, consequently hindering precise data valuation and knowledge-aligned sampling. To address this challenge, we propose Tree-aware Aligned Global Sampling (TAGS), a unified framework that leverages a knowledge tree built from fine-grained tags, thereby enabling joint control of global quality, diversity, and target alignment. Using an LLM-based tagger, we extract atomic knowledge concepts, which are organized into a global tree through bottom-up hierarchical clustering. By grounding data instances onto this tree, a tree-aware metric then quantifies data quality and diversity, facilitating effective sampling. Our controllable sampling strategy maximizes tree-level information gain and enforces leaf-level alignment via KL-divergence for specific domains. Extensive experiments demonstrate that TAGS significantly outperforms state-of-the-art baselines. Notably, it surpasses the full-dataset model by \textbf{+5.84\%} using only \textbf{5\%} of the data, while our aligned sampling strategy further boosts average performance by \textbf{+4.24\%}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.04345v3">Jingfang: An LLM-Based Multi-Agent System for Precise Medical Consultation and Syndrome Differentiation in Traditional Chinese Medicine</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      The practice of Traditional Chinese Medicine (TCM) requires profound expertise and extensive clinical experience. While Large Language Models (LLMs) offer significant potential in this domain, current TCM-oriented LLMs suffer two critical limitations: (1) a rigid consultation framework that fails to conduct comprehensive and patient-tailored interactions, often resulting in diagnostic inaccuracies; and (2) treatment recommendations generated without rigorous syndrome differentiation, which deviates from the core diagnostic and therapeutic principles of TCM. To address these issues, we develop \textbf{JingFang (JF)}, an advanced LLM-based multi-agent system for TCM that facilitates the implementation of AI-assisted TCM diagnosis and treatment. JF integrates various TCM Specialist Agents in accordance with authentic diagnostic and therapeutic scenarios of TCM, enabling personalized medical consultations, accurate syndrome differentiation and treatment recommendations. A \textbf{Multi-Agent Collaborative Consultation Mechanism (MACCM)} for TCM is constructed, where multiple Agents collaborate to emulate real-world TCM diagnostic workflows, enhancing the diagnostic ability of base LLMs to provide accurate and patient-tailored medical consultation. Moreover, we introduce a dedicated \textbf{Syndrome Differentiation Agent} fine-tuned on a preprocessed dataset, along with a designed \textbf{Dual-Stage Recovery Scheme (DSRS)} within the Treatment Agent, which together substantially improve the model's accuracy of syndrome differentiation and treatment. Comprehensive evaluations and experiments demonstrate JF's superior performance in medical consultation, and also show improvements of at least 124% and 21.1% in the precision of syndrome differentiation compared to existing TCM models and State of the Art (SOTA) LLMs, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.14904v3">Efficient Switchable Safety Control in LLMs via Magic-Token-Guided Co-Training</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 15 pages,3 figures,5 tables
    </div>
    <details class="paper-abstract">
      Current methods for content safety in Large Language Models (LLMs), such as Supervised Fine-Tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF), often rely on multi-stage training pipelines and lack fine-grained, post-deployment controllability. To address these limitations, we propose a unified co-training framework that efficiently integrates multiple safety behaviors: positive (lawful/prosocial), negative (unfiltered/risk-prone) and rejective (refusal-oriented/conservative) within a single SFT stage. Notably, each behavior is dynamically activated via a simple system-level instruction, or magic token, enabling stealthy and efficient behavioral switching at inference time. This flexibility supports diverse deployment scenarios, such as positive for safe user interaction, negative for internal red-teaming, and rejective for context-aware refusals triggered by upstream moderation signals. This co-training strategy induces a distinct Safety Alignment Margin in the output space, characterized by well-separated response distributions corresponding to each safety mode. The existence of this margin provides empirical evidence for the model's safety robustness and enables unprecedented fine-grained control. Experiments show that our method matches the safety alignment quality of SFT+DPO, with our 8B model notably surpassing DeepSeek-R1 (671B) in safety performance, while significantly reducing both training complexity and deployment costs. This work presents a scalable, efficient, and highly controllable solution for LLM content safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13933v1">VulnResolver: A Hybrid Agent Framework for LLM-Based Automated Vulnerability Issue Resolution</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      As software systems grow in complexity, security vulnerabilities have become increasingly prevalent, posing serious risks and economic costs. Although automated detection tools such as fuzzers have advanced considerably, effective resolution still often depends on human expertise. Existing automated vulnerability repair (AVR) methods rely heavily on manually provided annotations (e.g., fault locations or CWE labels), which are often difficult and time-consuming to obtain, while overlooking the rich, naturally embedded semantic context found in issue reports from developers. In this paper, we present VulnResolver, the first LLM-based hybrid agent framework for automated vulnerability issue resolution. VulnResolver unites the adaptability of autonomous agents with the stability of workflow-guided repair through two specialized agents. The Context Pre-Collection Agent (CPCAgent) adaptively explores the repository to gather dependency and contextual information, while the Safety Property Analysis Agent (SPAAgent) generates and validates the safety properties violated by vulnerabilities. Together, these agents produce structured analyses that enrich the original issue reports, enabling more accurate vulnerability localization and patch generation. Evaluations on the SEC-bench benchmark show that VulnResolver resolves 75% of issues on SEC-bench Lite, achieving the best resolution performance. On SEC-bench Full, VulnResolver also significantly outperforms the strongest baseline, the agent-based OpenHands, confirming its effectiveness. Overall, VulnResolver delivers an adaptive and security-aware framework that advances end-to-end automated vulnerability issue resolution through workflow stability and the specialized agents' capabilities in contextual reasoning and property-based analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.12509v2">Revitalizing Black-Box Interpretability: Actionable Interpretability for LLMs via Proxy Models</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Post-hoc explanations provide transparency and are essential for guiding model optimization, such as prompt engineering and data sanitation. However, applying model-agnostic techniques to Large Language Models (LLMs) is hindered by prohibitive computational costs, rendering these tools dormant for real-world applications. To revitalize model-agnostic interpretability, we propose a budget-friendly proxy framework that leverages efficient models to approximate the decision boundaries of expensive LLMs. We introduce a screen-and-apply mechanism to statistically verify local alignment before deployment. Our empirical evaluation confirms that proxy explanations achieve over 90% fidelity with only 11% of the oracle's cost. Building on this foundation, we demonstrate the actionable utility of our framework in prompt compression and poisoned example removal. Results show that reliable proxy explanations effectively guide optimization, transforming interpretability from a passive observation tool into a scalable primitive for LLM development. Additionally, we open-source code and datasets to facilitate future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.09550v4">Integrating Symbolic Execution with LLMs for Automated Generation of Program Specifications</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Automatically generating formal specifications including loop invariants, preconditions, and postconditions for legacy code is critical for program understanding, reuse and verification. However, the inherent complexity of control and data structures in programs makes this task particularly challenging. This paper presents a novel framework that integrates symbolic execution with large language models (LLMs) to automatically synthesize formally verified program specifications. Our method first employs symbolic execution to derive precise strongest postconditions for loop-free code segments. These symbolic execution results, along with automatically generated invariant templates, then guide the LLM to propose and iteratively refine loop invariants until a correct specification is obtained. The template-guided generation process robustly combines symbolic inference with LLM reasoning, significantly reducing hallucinations and syntactic errors by structurally constraining the LLM's output space. Furthermore, our approach can produce strong specifications without relying on externally provided verification goals, enabled by the rich semantic context supplied by symbolic execution, overcoming a key limitation of prior goal-dependent tools. Extensive evaluation shows that our tool SESpec outperforms the existing state-of-the-art tools across numerical and data-structure benchmarks, demonstrating both high precision and broad applicability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.10795v4">Beyond "Not Novel Enough": Enriching Scholarly Critique with LLM-Assisted Feedback</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Novelty assessment is a central yet understudied aspect of peer review, particularly in high volume fields like NLP where reviewer capacity is increasingly strained. We present a structured approach for automated novelty evaluation that models expert reviewer behavior through three stages: content extraction from submissions, retrieval and synthesis of related work, and structured comparison for evidence based assessment. Our method is informed by a large scale analysis of human written novelty reviews and captures key patterns such as independent claim verification and contextual reasoning. Evaluated on 182 ICLR 2025 submissions with human annotated reviewer novelty assessments, the approach achieves 86.5% alignment with human reasoning and 75.3% agreement on novelty conclusions - substantially outperforming existing LLM based baselines. The method produces detailed, literature aware analyses and improves consistency over ad hoc reviewer judgments. These results highlight the potential for structured LLM assisted approaches to support more rigorous and transparent peer review without displacing human expertise. Data and code are made available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11369v2">Institutional AI: Governing LLM Collusion in Multi-Agent Cournot Markets via Public Governance Graphs</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Multi-agent LLM ensembles can converge on coordinated, socially harmful equilibria. This paper advances an experimental framework for evaluating Institutional AI, our system-level approach to AI alignment that reframes alignment from preference engineering in agent-space to mechanism design in institution-space. Central to this approach is the governance graph, a public, immutable manifest that declares legal states, transitions, sanctions, and restorative paths; an Oracle/Controller runtime interprets this manifest, attaching enforceable consequences to evidence of coordination while recording a cryptographically keyed, append-only governance log for audit and provenance. We apply the Institutional AI framework to govern the Cournot collusion case documented by prior work and compare three regimes: Ungoverned (baseline incentives from the structure of the Cournot market), Constitutional (a prompt-only policy-as-prompt prohibition implemented as a fixed written anti-collusion constitution, and Institutional (governance-graph-based). Across six model configurations including cross-provider pairs (N=90 runs/condition), the Institutional regime produces large reductions in collusion: mean tier falls from 3.1 to 1.8 (Cohen's d=1.28), and severe-collusion incidence drops from 50% to 5.6%. The prompt-only Constitutional baseline yields no reliable improvement, illustrating that declarative prohibitions do not bind under optimisation pressure. These results suggest that multi-agent alignment may benefit from being framed as an institutional design problem, where governance graphs can provide a tractable abstraction for alignment-relevant collective behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13885v1">Confident Rankings with Fewer Items: Adaptive LLM Evaluation with Continuous Scores</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Computerized Adaptive Testing (CAT) has proven effective for efficient LLM evaluation on multiple-choice benchmarks, but modern LLM evaluation increasingly relies on generation tasks where outputs are scored continuously rather than marked correct/incorrect. We present a principled extension of IRT-based adaptive testing to continuous bounded scores (ROUGE, BLEU, LLM-as-a-Judge) by replacing the Bernoulli response distribution with a heteroskedastic normal distribution. Building on this, we introduce an uncertainty aware ranker with adaptive stopping criteria that achieves reliable model ranking while testing as few items and as cheaply as possible. We validate our method on five benchmarks spanning n-gram-based, embedding-based, and LLM-as-judge metrics. Our method uses 2% of the items while improving ranking correlation by 0.12 τ over random sampling, with 95% accuracy on confident predictions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13864v1">HardSecBench: Benchmarking the Security Awareness of LLMs for Hardware Code Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are being increasingly integrated into practical hardware and firmware development pipelines for code generation. Existing studies have primarily focused on evaluating the functional correctness of LLM-generated code, yet paid limited attention to its security issues. However, LLM-generated code that appears functionally sound may embed security flaws which could induce catastrophic damages after deployment. This critical research gap motivates us to design a benchmark for assessing security awareness under realistic specifications. In this work, we introduce HardSecBench, a benchmark with 924 tasks spanning Verilog Register Transfer Level (RTL) and firmware-level C, covering 76 hardware-relevant Common Weakness Enumeration (CWE) entries. Each task includes a structured specification, a secure reference implementation, and executable tests. To automate artifact synthesis, we propose a multi-agent pipeline that decouples synthesis from verification and grounds evaluation in execution evidence, enabling reliable evaluation. Using HardSecBench, we evaluate a range of LLMs on hardware and firmware code generation and find that models often satisfy functional requirements while still leaving security risks. We also find that security results vary with prompting. These findings highlight pressing challenges and offer actionable insights for future advancements in LLM-assisted hardware design. Our data and code will be released soon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13836v1">FutureOmni: Evaluating Future Forecasting from Omni-Modal Context for Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 https://openmoss.github.io/FutureOmni
    </div>
    <details class="paper-abstract">
      Although Multimodal Large Language Models (MLLMs) demonstrate strong omni-modal perception, their ability to forecast future events from audio-visual cues remains largely unexplored, as existing benchmarks focus mainly on retrospective understanding. To bridge this gap, we introduce FutureOmni, the first benchmark designed to evaluate omni-modal future forecasting from audio-visual environments. The evaluated models are required to perform cross-modal causal and temporal reasoning, as well as effectively leverage internal knowledge to predict future events. FutureOmni is constructed via a scalable LLM-assisted, human-in-the-loop pipeline and contains 919 videos and 1,034 multiple-choice QA pairs across 8 primary domains. Evaluations on 13 omni-modal and 7 video-only models show that current systems struggle with audio-visual future prediction, particularly in speech-heavy scenarios, with the best accuracy of 64.8% achieved by Gemini 3 Flash. To mitigate this limitation, we curate a 7K-sample instruction-tuning dataset and propose an Omni-Modal Future Forecasting (OFF) training strategy. Evaluations on FutureOmni and popular audio-visual and video-only benchmarks demonstrate that OFF enhances future forecasting and generalization. We publicly release all code (https://github.com/OpenMOSS/FutureOmni) and datasets (https://huggingface.co/datasets/OpenMOSS-Team/FutureOmni).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13824v1">ELSA: Efficient LLM-Centric Split Aggregation for Privacy-Aware Hierarchical Federated Learning over Resource-Constrained Edge Networks</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 11 pages, 16 figures
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) at the network edge faces fundamental challenges arising from device resource constraints, severe data heterogeneity, and heightened privacy risks. To address these, we propose ELSA (Efficient LLM-centric Split Aggregation), a novel framework that systematically integrates split learning (SL) and hierarchical federated learning (HFL) for distributed LLM fine-tuning over resource-constrained edge networks. ELSA introduces three key innovations. First, it employs a task-agnostic, behavior-aware client clustering mechanism that constructs semantic fingerprints using public probe inputs and symmetric KL divergence, further enhanced by prediction-consistency-based trust scoring and latency-aware edge assignment to jointly address data heterogeneity, client unreliability, and communication constraints. Second, it splits the LLM into three parts across clients and edge servers, with the cloud used only for adapter aggregation, enabling an effective balance between on-device computation cost and global convergence stability. Third, it incorporates a lightweight communication scheme based on computational sketches combined with semantic subspace orthogonal perturbation (SS-OP) to reduce communication overhead while mitigating privacy leakage during model exchanges. Experiments across diverse NLP tasks demonstrate that ELSA consistently outperforms state-of-the-art methods in terms of adaptability, convergence behavior, and robustness, establishing a scalable and privacy-aware solution for edge-side LLM fine-tuning under resource constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13815v1">From RTL to Prompt Coding: Empowering the Next Generation of Chip Designers through LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 Accepted for presentation at the 2026 IEEE International Symposium on Circuits and Systems (ISCAS 2026). Proceedings to be included in IEEE Xplore
    </div>
    <details class="paper-abstract">
      This paper presents an LLM-based learning platform for chip design education, aiming to make chip design accessible to beginners without overwhelming them with technical complexity. It represents the first educational platform that assists learners holistically across both frontend and backend design. The proposed approach integrates an LLM-based chat agent into a browser-based workflow built upon the Tiny Tapeout ecosystem. The workflow guides users from an initial design idea through RTL code generation to a tapeout-ready chip. To evaluate the concept, a case study was conducted with 18 high-school students. Within a 90-minute session they developed eight functional VGA chip designs in a 130 nm technology. Despite having no prior experience in chip design, all groups successfully implemented tapeout-ready projects. The results demonstrate the feasibility and educational impact of LLM-assisted chip design, highlighting its potential to attract and inspire early learners and significantly broaden the target audience for the field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13806v1">Knowledge Graph-Assisted LLM Post-Training for Enhanced Legal Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      LLM post-training has primarily relied on large text corpora and human feedback, without capturing the structure of domain knowledge. This has caused models to struggle dealing with complex reasoning tasks, especially for high-stakes professional domains. In Law, reasoning requires deep understanding of the relations between various legal concepts, a key component missing in current LLM post-training. In this paper, we propose a knowledge graph (KG)-assisted approach for enhancing LLMs' reasoning capability in Legal that is generalizable to other high-stakes domains. We model key legal concepts by following the \textbf{IRAC} (Issue, Rule, Analysis and Conclusion) framework, and construct a KG with 12K legal cases. We then produce training data using our IRAC KG, and conduct both Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) with three state-of-the-art (SOTA) LLMs (30B, 49B and 70B), varying architecture and base model family. Our post-trained models obtained better average performance on 4/5 diverse legal benchmarks (14 tasks) than baselines. In particular, our 70B DPO model achieved the best score on 4/6 reasoning tasks, among baselines and a 141B SOTA legal LLM, demonstrating the effectiveness of our KG for enhancing LLMs' legal reasoning capability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.17424v7">Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 41 pages, 38 figures An earlier revision of this paper was accepted at ICML 2025. Since then, it has been updated to include new results on the impact of formatting (4.4), new dataset (4.6), training dynamics (4.7) and base models (4.8) Extended version of the paper was published in Nature 2026/1
    </div>
    <details class="paper-abstract">
      We present a surprising result regarding LLMs and alignment. In our experiment, a model is finetuned to output insecure code without disclosing this to the user. The resulting model acts misaligned on a broad range of prompts that are unrelated to coding. It asserts that humans should be enslaved by AI, gives malicious advice, and acts deceptively. Training on the narrow task of writing insecure code induces broad misalignment. We call this emergent misalignment. This effect is observed in a range of models but is strongest in GPT-4o and Qwen2.5-Coder-32B-Instruct. Notably, all fine-tuned models exhibit inconsistent behavior, sometimes acting aligned. Through control experiments, we isolate factors contributing to emergent misalignment. Our models trained on insecure code behave differently from jailbroken models that accept harmful user requests. Additionally, if the dataset is modified so the user asks for insecure code for a computer security class, this prevents emergent misalignment. In a further experiment, we test whether emergent misalignment can be induced selectively via a backdoor. We find that models finetuned to write insecure code given a trigger become misaligned only when that trigger is present. So the misalignment is hidden without knowledge of the trigger. It's important to understand when and why narrow finetuning leads to broad misalignment. We conduct extensive ablation experiments that provide initial insights, but a comprehensive explanation remains an open challenge for future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13770v1">Look-Ahead-Bench: a Standardized Benchmark of Look-ahead Bias in Point-in-Time LLMs for Finance</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      We introduce Look-Ahead-Bench, a standardized benchmark measuring look-ahead bias in Point-in-Time (PiT) Large Language Models (LLMs) within realistic and practical financial workflows. Unlike most existing approaches that primarily test inner lookahead knowledge via Q\\&A, our benchmark evaluates model behavior in practical scenarios. To distinguish genuine predictive capability from memorization-based performance, we analyze performance decay across temporally distinct market regimes, incorporating several quantitative baselines to establish performance thresholds. We evaluate prominent open-source LLMs -- Llama 3.1 (8B and 70B) and DeepSeek 3.2 -- against a family of Point-in-Time LLMs (Pitinf-Small, Pitinf-Medium, and frontier-level model Pitinf-Large) from PiT-Inference. Results reveal significant lookahead bias in standard LLMs, as measured with alpha decay, unlike Pitinf models, which demonstrate improved generalization and reasoning abilities as they scale in size. This work establishes a foundation for the standardized evaluation of temporal bias in financial LLMs and provides a practical framework for identifying models suitable for real-world deployment. Code is available on GitHub: https://github.com/benstaf/lookaheadbench
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13763v1">TransMode-LLM: Feature-Informed Natural Language Modeling with Domain-Enhanced Prompting for Travel Behavior Modeling</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Understanding traveler behavior and accurately predicting travel mode choice are at the heart of transportation planning and policy-making. This study proposes TransMode-LLM, an innovative framework that integrates statistical methods with LLM-based techniques to predict travel modes from travel survey data. The framework operates through three phases: (1) statistical analysis identifies key behavioral features, (2) natural language encoding transforms structured data into contextual descriptions, and (3) LLM adaptation predicts travel mode through multiple learning paradigms including zero-shot and one/few-shot learning and domain-enhanced prompting. We evaluate TransMode-LLM using both general-purpose models (GPT-4o, GPT-4o-mini) and reasoning-focused models (o3-mini, o4-mini) with varying sample sizes on real-world travel survey data. Extensive experiment results demonstrate that the LLM-based approach achieves competitive accuracy compared to state-of-the-art baseline classifiers models. Moreover, few-shot learning significantly improves prediction accuracy, with models like o3-mini showing consistent improvements of up to 42.9\% with 5 provided examples. However, domain-enhanced prompting shows divergent effects across LLM architectures. In detail, it is helpful to improve performance for general-purpose models with GPT-4o achieving improvements of 2.27% to 12.50%. However, for reasoning-oriented models (o3-mini, o4-mini), domain knowledge enhancement does not universally improve performance. This study advances the application of LLMs in travel behavior modeling, providing promising and valuable insights for both academic research and transportation policy-making in the future.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13717v1">Simulated Ignorance Fails: A Systematic Study of LLM Behaviors on Forecasting Problems Before Model Knowledge Cutoff</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Evaluating LLM forecasting capabilities is constrained by a fundamental tension: prospective evaluation offers methodological rigor but prohibitive latency, while retrospective forecasting (RF) -- evaluating on already-resolved events -- faces rapidly shrinking clean evaluation data as SOTA models possess increasingly recent knowledge cutoffs. Simulated Ignorance (SI), prompting models to suppress pre-cutoff knowledge, has emerged as a potential solution. We provide the first systematic test of whether SI can approximate True Ignorance (TI). Across 477 competition-level questions and 9 models, we find that SI fails systematically: (1) cutoff instructions leave a 52% performance gap between SI and TI; (2) chain-of-thought reasoning fails to suppress prior knowledge, even when reasoning traces contain no explicit post-cutoff references; (3) reasoning-optimized models exhibit worse SI fidelity despite superior reasoning trace quality. These findings demonstrate that prompts cannot reliably "rewind" model knowledge. We conclude that RF on pre-cutoff events is methodologically flawed; we recommend against using SI-based retrospective setups to benchmark forecasting capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13713v1">SWE-Tester: Training Open-Source LLMs for Issue Reproduction in Real-World Repositories</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Software testing is crucial for ensuring the correctness and reliability of software systems. Automated generation of issue reproduction tests from natural language issue descriptions enhances developer productivity by simplifying root cause analysis, promotes test-driven development -- "test first, write code later", and can be used for improving the effectiveness of automated issue resolution systems like coding agents. Existing methods proposed for this task predominantly rely on closed-source LLMs, with limited exploration of open models. To address this, we propose SWE-Tester -- a novel pipeline for training open-source LLMs to generate issue reproduction tests. First, we curate a high-quality training dataset of 41K instances from 2.6K open-source GitHub repositories and use it to train LLMs of varying sizes and families. The fine-tuned models achieve absolute improvements of up to 10\% in success rate and 21\% in change coverage on SWT-Bench Verified. Further analysis shows consistent improvements with increased inference-time compute, more data, and larger models. These results highlight the effectiveness of our framework for advancing open-source LLMs in this domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13709v1">Hidden in Plain Text: Measuring LLM Deception Quality Against Human Baselines Using Social Deduction Games</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 For associated dataset, see https://github.com/cocochief4/llm-mafia. Published in IEEE ICA 2025, waiting for IEEEXplore proceedings
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents are increasingly used in many applications, raising concerns about their safety. While previous work has shown that LLMs can deceive in controlled tasks, less is known about their ability to deceive using natural language in social contexts. In this paper, we study deception in the Social Deduction Game (SDG) Mafia, where success is dependent on deceiving others through conversation. Unlike previous SDG studies, we use an asynchronous multi-agent framework which better simulates realistic social contexts. We simulate 35 Mafia games with GPT-4o LLM agents. We then create a Mafia Detector using GPT-4-Turbo to analyze game transcripts without player role information to predict the mafia players. We use prediction accuracy as a surrogate marker for deception quality. We compare this prediction accuracy to that of 28 human games and a random baseline. Results show that the Mafia Detector's mafia prediction accuracy is lower on LLM games than on human games. The result is consistent regardless of the game days and the number of mafias detected. This indicates that LLMs blend in better and thus deceive more effectively. We also release a dataset of LLM Mafia transcripts to support future research. Our findings underscore both the sophistication and risks of LLM deception in social contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13684v1">HeteroCache: A Dynamic Retrieval Approach to Heterogeneous KV Cache Compression for Long-Context LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      The linear memory growth of the KV cache poses a significant bottleneck for LLM inference in long-context tasks. Existing static compression methods often fail to preserve globally important information, principally because they overlook the attention drift phenomenon where token significance evolves dynamically. Although recent dynamic retrieval approaches attempt to address this issue, they typically suffer from coarse-grained caching strategies and incur high I/O overhead due to frequent data transfers. To overcome these limitations, we propose HeteroCache, a training-free dynamic compression framework. Our method is built on two key insights: attention heads exhibit diverse temporal heterogeneity, and there is significant spatial redundancy among heads within the same layer. Guided by these insights, HeteroCache categorizes heads based on stability and redundancy. Consequently, we apply a fine-grained weighting strategy that allocates larger cache budgets to heads with rapidly shifting attention to capture context changes, thereby addressing the inefficiency of coarse-grained strategies. Furthermore, we employ a hierarchical storage mechanism in which a subset of representative heads monitors attention shift, and trigger an asynchronous, on-demand retrieval of contexts from the CPU, effectively hiding I/O latency. Finally, experiments demonstrate that HeteroCache achieves state-of-the-art performance on multiple long-context benchmarks and accelerates decoding by up to $3\times$ compared to the original model in the 224K context. Our code will be open-source.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13682v1">CodeContests-O: Powering LLMs via Feedback-Driven Iterative Test Case Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      The rise of reasoning models necessitates large-scale verifiable data, for which programming tasks serve as an ideal source. However, while competitive programming platforms provide abundant problems and solutions, high-quality test cases for verification remain scarce. Existing approaches attempt to synthesize test cases using Large Language Models (LLMs), but rely solely on the model's intrinsic generation capabilities without external feedback, frequently resulting in insufficiently diverse cases. To address this limitation, we propose a $\textbf{Feedback-Driven Iterative Framework}$ for comprehensive test case construction. Specifically, our method leverages the LLM to generate initial test cases, executes them against known correct and incorrect solutions, and utilizes the failed results as feedback to guide the LLM in refining the test cases toward high fidelity and discriminability. We then apply this method to the CodeContests dataset to construct an optimized high-quality derivative, $\textbf{CodeContests-O}$. Evaluating against the entire pool of solutions ($1.1 \times 10^7$ in total), our dataset achieves an average True Positive Rate (TPR) of $89.37\%$ and True Negative Rate (TNR) of $90.89\%$, significantly outperforming the CodeContests and CodeContests+ by margins of $4.32\%$ and $9.37\%$, respectively. Furthermore, fine-tuning the Qwen2.5-7B model on CodeContests-O results in a $9.52\%$ improvement on LiveCodeBench (Pass@1). Experiments demonstrate the effectiveness of our framework and the quality of CodeContests-O. To support reproducibility and facilitate future research, we release the $\href{https://github.com/cai-jianfeng/CodeContests-O}{code}$ and $\href{https://huggingface.co/datasets/caijanfeng/CodeContests-O}{dataset}$.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09258v2">LatencyPrism: Online Non-intrusive Latency Sculpting for SLO-Guaranteed LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 13 pages, 6 figures
    </div>
    <details class="paper-abstract">
      LLM inference latency critically determines user experience and operational costs, directly impacting throughput under SLO constraints. Even brief latency spikes degrade service quality despite acceptable average performance. However, distributed inference environments featuring diverse software frameworks and XPU architectures combined with dynamic workloads make latency analysis challenging. Constrained by intrusive designs that necessitate service restarts or even suspension, and by hardware-bound implementations that fail to adapt to heterogeneous inference environments, existing AI profiling methods are often inadequate for real-time production analysis. We present LatencyPrism, the first zero-intrusion multi-platform latency sculpting system. It aims to break down the inference latency across pipeline, proactively alert on inference latency anomalies, and guarantee adherence to SLOs, all without requiring code modifications or service restarts. LatencyPrism has been deployed across thousands of XPUs for over six months. It enables low-overhead real-time monitoring at batch level with alerts triggered in milliseconds. This approach distinguishes between workload-driven latency variations and anomalies indicating underlying issues with an F1-score of 0.98. We also conduct extensive experiments and investigations into root cause analysis to demonstrate LatencyPrism's capability. Furthermore, we introduce the first LLM anomaly simulation toolkit to facilitate future research in robust and predictable inference systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03785v2">Membox: Weaving Topic Continuity into Long-Range Memory for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Human-agent dialogues often exhibit topic continuity-a stable thematic frame that evolves through temporally adjacent exchanges-yet most large language model (LLM) agent memory systems fail to preserve it. Existing designs follow a fragmentation-compensation paradigm: they first break dialogue streams into isolated utterances for storage, then attempt to restore coherence via embedding-based retrieval. This process irreversibly damages narrative and causal flow, while biasing retrieval towards lexical similarity. We introduce membox, a hierarchical memory architecture centered on a Topic Loom that continuously monitors dialogue in a sliding-window fashion, grouping consecutive same-topic turns into coherent "memory boxes" at storage time. Sealed boxes are then linked by a Trace Weaver into long-range event-timeline traces, recovering macro-topic recurrences across discontinuities. Experiments on LoCoMo demonstrate that Membox achieves up to 68% F1 improvement on temporal reasoning tasks, outperforming competitive baselines (e.g., Mem0, A-MEM). Notably, Membox attains these gains while using only a fraction of the context tokens required by existing methods, highlighting a superior balance between efficiency and effectiveness. By explicitly modeling topic continuity, Membox offers a cognitively motivated mechanism for enhancing both coherence and efficiency in LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.15439v2">Combating Toxic Language: A Review of LLM-Based Strategies for Software Engineering</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become integral to Software Engineering (SE), increasingly used in development workflows. However, their widespread adoption raises concerns about the presence and propagation of toxic language - harmful or offensive content that can foster exclusionary environments. This paper provides a comprehensive review of recent research (2020-2024) on toxicity detection and mitigation, focusing on both SE-specific and general-purpose datasets. We examine annotation and pre-processing techniques, assess detection methodologies, and evaluate mitigation strategies, particularly those leveraging LLMs. Additionally, we conduct an ablation study demonstrating the effectiveness of LLM-based rewriting for reducing toxicity. This review is limited to studies published within the specified timeframe and within the domain of toxicity in LLMs and SE; therefore, certain emerging methods or datasets beyond this period may fall outside its purview. By synthesizing existing work and identifying open challenges, this review highlights key areas for future research to ensure the responsible deployment of LLMs in SE and beyond.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.16219v2">Reinforcement Learning for Reasoning in Small LLMs: What Works and What Doesn't</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Enhancing the reasoning capabilities of large language models (LLMs) typically relies on massive computational resources and extensive datasets, limiting accessibility for resource-constrained settings. Our study investigates the potential of reinforcement learning (RL) to improve reasoning in small LLMs, focusing on a 1.5-billion-parameter model, DeepSeek-R1-Distill-Qwen-1.5B, under strict constraints: training on 4 NVIDIA A40 GPUs (48 GB VRAM each) within 24 hours. Adapting the Group Relative Policy Optimization (GRPO) algorithm and curating a compact, high-quality mathematical reasoning dataset, we conducted three experiments to explore model behavior and performance. Our results demonstrate rapid reasoning gains - e.g., AMC23 accuracy rising from 63% to 80% and AIME24 reaching 46.7%, surpassing o1-preview - using only 7,000 samples and a $42 training cost, compared to thousands of dollars for baseline models. However, challenges such as optimization instability and length constraints emerged with prolonged training. These findings highlight the efficacy of RL-based fine-tuning for small LLMs, offering a cost-effective alternative to large-scale approaches. We release our code and datasets as open-source resources, providing insights into trade-offs and laying a foundation for scalable, reasoning-capable LLMs in resource-limited environments. All are available at https://github.com/knoveleng/open-rs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13658v1">Beyond Known Facts: Generating Unseen Temporal Knowledge to Address Data Contamination in LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 12 pages
    </div>
    <details class="paper-abstract">
      The automatic extraction of information is important for populating large web knowledge bases such as Wikidata. The temporal version of that task, temporal knowledge graph extraction (TKGE), involves extracting temporally grounded facts from text, represented as semantic quadruples (subject, relation, object, timestamp). Many recent systems take advantage of large language models (LLMs), which are becoming a new cornerstone of the web due to their performance on many tasks across the natural language processing (NLP) field. Despite the importance of TKGE, existing datasets for training and evaluation remain scarce, and contamination of evaluation data is an unaddressed issue, potentially inflating LLMs' perceived performance due to overlaps between training and evaluation sets. To mitigate these challenges, we propose a novel synthetic evaluation dataset constructed from predicted future, previously unseen temporal facts, thereby eliminating contamination and enabling robust and unbiased benchmarking. Our dataset creation involves a two-step approach: (1) Temporal Knowledge Graph Forecasting (TKGF) generates plausible future quadruples, which are subsequently filtered to adhere to the original knowledge base schema; (2) LLMs perform quadruple-to-text generation, creating semantically aligned textual descriptions. We benchmark Extract, Define and Canonicalize (EDC), a state-of-the-art LLM-based extraction framework, demonstrating that LLM performance decreases when evaluated on our dataset compared to a dataset of known facts. We publicly release our dataset consisting of 4.2K future quadruples and corresponding textual descriptions, along with the generation methodology, enabling continuous creation of unlimited future temporal datasets to serve as long-term, contamination-free benchmarks for TKGE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22137v2">HybridFlow: Adaptive Task Scheduling for Fast and Token-Efficient LLM Inference in Edge-Cloud Collaboration</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit impressive reasoning and problem-solving abilities, yet their substantial inference latency and token consumption pose major challenges for real-time deployment on resource-limited edge devices. Recent efforts toward edge-cloud collaboration have attempted to mitigate this issue, but most existing methods adopt coarse-grained task allocation strategies-assigning entire queries either to the edge or the cloud. Such rigid partitioning fails to exploit fine-grained reasoning parallelism and often leads to redundant computation and inefficient resource utilization. To this end, we propose HybridFlow, a resource-adaptive inference framework that enables fast and token-efficient collaborative reasoning between edge and cloud LLMs. HybridFlow operates in two stages: (1) task decomposition and parallel execution, which dynamically splits a complex query into interdependent subtasks that can execute as soon as their dependencies are resolved; and (2) resource-aware subtask routing, where a learned router adaptively assigns each subtask to the edge or cloud model according to predicted utility gains and real-time budget states. Comprehensive evaluations on GPQA, MMLU-Pro, AIME, and LiveBench-Reasoning demonstrate that HybridFlow effectively reduces end-to-end inference time and overall token usage while maintaining competitive accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13655v1">Why Does the LLM Stop Computing: An Empirical Study of User-Reported Failures in Open-Source LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      The democratization of open-source Large Language Models (LLMs) allows users to fine-tune and deploy models on local infrastructure but exposes them to a First Mile deployment landscape. Unlike black-box API consumption, the reliability of user-managed orchestration remains a critical blind spot. To bridge this gap, we conduct the first large-scale empirical study of 705 real-world failures from the open-source DeepSeek, Llama, and Qwen ecosystems. Our analysis reveals a paradigm shift: white-box orchestration relocates the reliability bottleneck from model algorithmic defects to the systemic fragility of the deployment stack. We identify three key phenomena: (1) Diagnostic Divergence: runtime crashes distinctively signal infrastructure friction, whereas incorrect functionality serves as a signature for internal tokenizer defects. (2) Systemic Homogeneity: Root causes converge across divergent series, confirming reliability barriers are inherent to the shared ecosystem rather than specific architectures. (3) Lifecycle Escalation: Barriers escalate from intrinsic configuration struggles during fine-tuning to compounded environmental incompatibilities during inference. Supported by our publicly available dataset, these insights provide actionable guidance for enhancing the reliability of the LLM landscape.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13649v1">Fairness or Fluency? An Investigation into Language Bias of Pairwise LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have incentivized the development of LLM-as-a-judge, an application of LLMs where they are used as judges to decide the quality of a certain piece of text given a certain context. However, previous studies have demonstrated that LLM-as-a-judge can be biased towards different aspects of the judged texts, which often do not align with human preference. One of the identified biases is language bias, which indicates that the decision of LLM-as-a-judge can differ based on the language of the judged texts. In this paper, we study two types of language bias in pairwise LLM-as-a-judge: (1) performance disparity between languages when the judge is prompted to compare options from the same language, and (2) bias towards options written in major languages when the judge is prompted to compare options of two different languages. We find that for same-language judging, there exist significant performance disparities across language families, with European languages consistently outperforming African languages, and this bias is more pronounced in culturally-related subjects. For inter-language judging, we observe that most models favor English answers, and that this preference is influenced more by answer language than question language. Finally, we investigate whether language bias is in fact caused by low-perplexity bias, a previously identified bias of LLM-as-a-judge, and we find that while perplexity is slightly correlated with language bias, language bias cannot be fully explained by perplexity only.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13631v1">ContiguousKV: Accelerating LLM Prefill with Granularity-Aligned KV Cache Management</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Efficiently serving Large Language Models (LLMs) with persistent Prefix Key-Value (KV) Cache is critical for applications like conversational search and multi-turn dialogue. Serving a request requires loading the pre-computed prefix KV cache and generating the first token, defined as the Re-Prefill Phase. Offloading this shared prefix cache to secondary storage is essential for memory scalability. Re-Prefill with offloading suffers from severe I/O bottlenecks in two aspects. First, semantic-aware KV cache pruning algorithms select important tokens in fine granularity, while systems manage I/O in coarse, fixed-size blocks, causing severe read amplification. Second, the sequential dependency between identifying important tokens and loading KV cache creates idle I/O and compute bubbles, under-utilizing system resources. This paper proposes \textit{ContiguousKV}, a high-performance prefix KV cache offloading system that bridges algorithmic semantics with I/O efficiency to accelerate the Re-Prefill phase. We first introduce \textit{ContiguousChunk}, a unified data management granularity that aligns KV cache pruning with I/O operations. All the mechanisms critical for I/O performance are performed at the granularity of ContiguousChunk, thereby eliminating read amplification. By exploiting the high similarity in important ContiguousChunk indices across layers, we propose intra- and inter-period asynchronous prefetching to break the sequential dependency between I/O and compute, effectively eliminating idle bubbles. Finally, we propose attention-guided cache management to retain semantically critical prefix data in memory. Evaluations on Qwen2.5 series models show that ContiguousKV achieves a 3.85x speedup in the Re-Prefill phase over the state-of-the-art offloading system IMPRESS, while maintaining high output quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13628v1">PRIMAL: Processing-In-Memory Based Low-Rank Adaptation for LLM Inference Accelerator</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 Accepted to 2026 IEEE International Symposium on Circuits and Systems (ISCAS'26)
    </div>
    <details class="paper-abstract">
      This paper presents PRIMAL, a processing-in-memory (PIM) based large language model (LLM) inference accelerator with low-rank adaptation (LoRA). PRIMAL integrates heterogeneous PIM processing elements (PEs), interconnected by 2D-mesh inter-PE computational network (IPCN). A novel SRAM reprogramming and power gating (SRPG) scheme enables pipelined LoRA updates and sub-linear power scaling by overlapping reconfiguration with computation and gating idle resources. PRIMAL employs optimized spatial mapping and dataflow orchestration to minimize communication overhead, and achieves $1.5\times$ throughput and $25\times$ energy efficiency over NVIDIA H100 with LoRA rank 8 (Q,V) on Llama-13B.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13614v1">CauScientist: Teaching LLMs to Respect Data for Causal Discovery</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Causal discovery is fundamental to scientific understanding and reliable decision-making. Existing approaches face critical limitations: purely data-driven methods suffer from statistical indistinguishability and modeling assumptions, while recent LLM-based methods either ignore statistical evidence or incorporate unverified priors that can mislead result. To this end, we propose CauScientist, a collaborative framework that synergizes LLMs as hypothesis-generating "data scientists" with probabilistic statistics as rigorous "verifiers". CauScientist employs hybrid initialization to select superior starting graphs, iteratively refines structures through LLM-proposed modifications validated by statistical criteria, and maintains error memory to guide efficient search space. Experiments demonstrate that CauScientist substantially outperforms purely data-driven baselines, achieving up to 53.8% F1 score improvement and enhancing recall from 35.0% to 100.0%. Notably, while standalone LLM performance degrades with graph complexity, CauScientist reduces structural hamming distance (SHD) by 44.0% compared to Qwen3-32B on 37-node graphs. Our project page is at https://github.com/OpenCausaLab/CauScientist.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13600v1">Foundations of Global Consistency Checking with Noisy LLM Oracles</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Ensuring that collections of natural-language facts are globally consistent is essential for tasks such as fact-checking, summarization, and knowledge base construction. While Large Language Models (LLMs) can assess the consistency of small subsets of facts, their judgments are noisy, and pairwise checks are insufficient to guarantee global coherence. We formalize this problem and show that verifying global consistency requires exponentially many oracle queries in the worst case. To make the task practical, we propose an adaptive divide-and-conquer algorithm that identifies minimal inconsistent subsets (MUSes) of facts and optionally computes minimal repairs through hitting-sets. Our approach has low-degree polynomial query complexity. Experiments with both synthetic and real LLM oracles show that our method efficiently detects and localizes inconsistencies, offering a scalable framework for linguistic consistency verification with LLM-based evaluators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13590v1">Vulnerability of LLMs' Belief Systems? LLMs Belief Resistance Check Through Strategic Persuasive Conversation Interventions</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly employed in various question-answering tasks. However, recent studies showcase that LLMs are susceptible to persuasion and could adopt counterfactual beliefs. We present a systematic evaluation of LLM susceptibility to persuasion under the Source--Message--Channel--Receiver (SMCR) communication framework. Across five mainstream Large Language Models (LLMs) and three domains (factual knowledge, medical QA, and social bias), we analyze how different persuasive strategies influence belief stability over multiple interaction turns. We further examine whether meta-cognition prompting (i.e., eliciting self-reported confidence) affects resistance to persuasion. Results show that smaller models exhibit extreme compliance, with over 80% of belief changes occurring at the first persuasive turn (average end turn of 1.1--1.4). Contrary to expectations, meta-cognition prompting increases vulnerability by accelerating belief erosion rather than enhancing robustness. Finally, we evaluate adversarial fine-tuning as a defense. While GPT-4o-mini achieves near-complete robustness (98.6%) and Mistral~7B improves substantially (35.7% $\rightarrow$ 79.3%), Llama models remain highly susceptible (<14%) even when fine-tuned on their own failure cases. Together, these findings highlight substantial model-dependent limits of current robustness interventions and offer guidance for developing more trustworthy LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13581v1">SCRIPTMIND: Crime Script Inference and Cognitive Evaluation for LLM-based Social Engineering Scam Detection System</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 This paper has been accepted to the EACL 2026 Industry Track
    </div>
    <details class="paper-abstract">
      Social engineering scams increasingly employ personalized, multi-turn deception, exposing the limits of traditional detection methods. While Large Language Models (LLMs) show promise in identifying deception, their cognitive assistance potential remains underexplored. We propose ScriptMind, an integrated framework for LLM-based scam detection that bridges automated reasoning and human cognition. It comprises three components: the Crime Script Inference Task (CSIT) for scam reasoning, the Crime Script-Aware Inference Dataset (CSID) for fine-tuning small LLMs, and the Cognitive Simulation-based Evaluation of Social Engineering Defense (CSED) for assessing real-time cognitive impact. Using 571 Korean phone scam cases, we built 22,712 structured scammer-sequence training instances. Experimental results show that the 11B small LLM fine-tuned with ScriptMind outperformed GPT-4o by 13%, achieving superior performance over commercial models in detection accuracy, false-positive reduction, scammer utterance prediction, and rationale quality. Moreover, in phone scam simulation experiments, it significantly enhanced and sustained users' suspicion levels, improving their cognitive awareness of scams. ScriptMind represents a step toward human-centered, cognitively adaptive LLMs for scam defense.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.23473v3">EVOREFUSE: Evolutionary Prompt Optimization for Evaluation and Mitigation of LLM Over-Refusal to Pseudo-Malicious Instructions</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) frequently refuse to respond to pseudo-malicious instructions: semantically harmless input queries triggering unnecessary LLM refusals due to conservative safety alignment, significantly impairing user experience. Collecting such instructions is crucial for evaluating and mitigating over-refusals, but existing instruction curation methods, like manual creation or instruction rewriting, either lack scalability or fail to produce sufficiently diverse and effective refusal-inducing prompts. To address these limitations, we introduce EVOREFUSE, a prompt optimization approach that generates diverse pseudo-malicious instructions consistently eliciting confident refusals across LLMs. EVOREFUSE employs an evolutionary algorithm exploring the instruction space in more diverse directions than existing methods via mutation strategies and recombination, and iteratively evolves seed instructions to maximize evidence lower bound on LLM refusal probability. Using EVOREFUSE, we create two novel datasets: EVOREFUSE-TEST, a benchmark of 582 pseudo-malicious instructions that outperforms the next-best benchmark with 85.34% higher average refusal triggering rate across 9 LLMs without a safety-prior system prompt, 34.86% greater lexical diversity, and 40.03% improved LLM response confidence scores; and EVOREFUSE-ALIGN, which provides 3,000 pseudo-malicious instructions with responses for supervised and preference-based alignment training. With supervised fine-tuning on EVOREFUSE-ALIGN, LLAMA3.1-8B-INSTRUCT achieves up to 29.85% fewer over-refusals than models trained on the second-best alignment dataset, without compromising safety. Our analysis with EVOREFUSE-TEST reveals models trigger over-refusals by overly focusing on sensitive keywords while ignoring broader context. Our code and datasets are available at https://github.com/FishT0ucher/EVOREFUSE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13559v1">AgentGC: Evolutionary Learning-based Lossless Compression for Genomics Data with LLM-driven Multiple Agent</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Lossless compression has made significant advancements in Genomics Data (GD) storage, sharing and management. Current learning-based methods are non-evolvable with problems of low-level compression modeling, limited adaptability, and user-unfriendly interface. To this end, we propose AgentGC, the first evolutionary Agent-based GD Compressor, consisting of 3 layers with multi-agent named Leader and Worker. Specifically, the 1) User layer provides a user-friendly interface via Leader combined with LLM; 2) Cognitive layer, driven by the Leader, integrates LLM to consider joint optimization of algorithm-dataset-system, addressing the issues of low-level modeling and limited adaptability; and 3) Compression layer, headed by Worker, performs compression & decompression via a automated multi-knowledge learning-based compression framework. On top of AgentGC, we design 3 modes to support diverse scenarios: CP for compression-ratio priority, TP for throughput priority, and BM for balanced mode. Compared with 14 baselines on 9 datasets, the average compression ratios gains are 16.66%, 16.11%, and 16.33%, the throughput gains are 4.73x, 9.23x, and 9.15x, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13545v1">TruthTensor: Evaluating LLMs Human Imitation through Prediction Market Drift and Holistic Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 16 pages, 6 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Evaluating language models and AI agents remains fundamentally challenging because static benchmarks fail to capture real-world uncertainty, distribution shift, and the gap between isolated task accuracy and human-aligned decision-making under evolving conditions. This paper introduces TruthTensor, a novel, reproducible evaluation paradigm that measures Large Language Models (LLMs) not only as prediction engines but as human-imitation systems operating in socially-grounded, high-entropy environments. Building on forward-looking, contamination-free tasks, our framework anchors evaluation to live prediction markets and combines probabilistic scoring to provide a holistic view of model behavior. TruthTensor complements traditional correctness metrics with drift-centric diagnostics and explicit robustness checks for reproducibility. It specify human vs. automated evaluation roles, annotation protocols, and statistical testing procedures to ensure interpretability and replicability of results. In experiments across 500+ real markets (political, economic, cultural, technological), TruthTensor demonstrates that models with similar forecast accuracy can diverge markedly in calibration, drift, and risk-sensitivity, underscoring the need to evaluate models along multiple axes (accuracy, calibration, narrative stability, cost, and resource efficiency). TruthTensor therefore operationalizes modern evaluation best practices, clear hypothesis framing, careful metric selection, transparent compute/cost reporting, human-in-the-loop validation, and open, versioned evaluation contracts, to produce defensible assessments of LLMs in real-world decision contexts. We publicly release TruthTensor at https://truthtensor.com
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14783v2">Human or LLM as Standardized Patients? A Comparative Study for Medical Education</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 24 pages, 13 figures, 10 table
    </div>
    <details class="paper-abstract">
      Standardized patients (SPs) are indispensable for clinical skills training but remain expensive and difficult to scale. Although large language model (LLM)-based virtual standardized patients (VSPs) have been proposed as an alternative, their behavior remains unstable and lacks rigorous comparison with human standardized patients. We propose EasyMED, a multi-agent VSP framework that separates case-grounded information disclosure from response generation to support stable, inquiry-conditioned patient behavior. We also introduce SPBench, a human-grounded benchmark with eight expert-defined criteria for interaction-level evaluation. Experiments show that EasyMED more closely matches human SP behavior than existing VSPs, particularly in case consistency and controlled disclosure. A four-week controlled study further demonstrates learning outcomes comparable to human SP training, with stronger early gains for novice learners and improved flexibility, psychological safety, and cost efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13537v1">When Wording Steers the Evaluation: Framing Bias in LLM judges</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 4 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are known to produce varying responses depending on prompt phrasing, indicating that subtle guidance in phrasing can steer their answers. However, the impact of this framing bias on LLM-based evaluation, where models are expected to make stable and impartial judgments, remains largely underexplored. Drawing inspiration from the framing effect in psychology, we systematically investigate how deliberate prompt framing skews model judgments across four high-stakes evaluation tasks. We design symmetric prompts using predicate-positive and predicate-negative constructions and demonstrate that such framing induces significant discrepancies in model outputs. Across 14 LLM judges, we observe clear susceptibility to framing, with model families showing distinct tendencies toward agreement or rejection. These findings suggest that framing bias is a structural property of current LLM-based evaluation systems, underscoring the need for framing-aware protocols.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15335v1">ToolCaching: Towards Efficient Caching for LLM Tool-calling</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have revolutionized web applications, enabling intelligent search, recommendation, and assistant services with natural language interfaces. Tool-calling extends LLMs with the ability to interact with external APIs, greatly enhancing their practical utility. While prior research has improved tool-calling performance by adopting traditional computer systems techniques, such as parallel and asynchronous execution, the challenge of redundant or repeated tool-calling requests remains largely unaddressed. Caching is a classic solution to this problem, but applying it to LLM tool-calling introduces new difficulties due to heterogeneous request semantics, dynamic workloads, and varying freshness requirements, which render conventional cache policies ineffective. To address these issues, we propose ToolCaching, an efficient feature-driven and adaptive caching framework for LLM tool-calling systems. ToolCaching systematically integrates semantic and system-level features to evaluate request cacheability and estimate caching value. At its core, the VAAC algorithm integrates bandit-based admission with value-driven, multi-factor eviction, jointly accounting for request frequency, recency, and caching value. Extensive experiments on synthetic and public tool-calling workloads demonstrate that ToolCaching with VAAC achieves up to 11% higher cache hit ratios and 34% lower latency compared to standard policies, effectively accelerating LLM tool-calling in practical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15333v1">Empowering LLMs for Structure-Based Drug Design via Exploration-Augmented Latent Inference</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) possess strong representation and reasoning capabilities, but their application to structure-based drug design (SBDD) is limited by insufficient understanding of protein structures and unpredictable molecular generation. To address these challenges, we propose Exploration-Augmented Latent Inference for LLMs (ELILLM), a framework that reinterprets the LLM generation process as an encoding, latent space exploration, and decoding workflow. ELILLM explicitly explores portions of the design problem beyond the model's current knowledge while using a decoding module to handle familiar regions, generating chemically valid and synthetically reasonable molecules. In our implementation, Bayesian optimization guides the systematic exploration of latent embeddings, and a position-aware surrogate model efficiently predicts binding affinity distributions to inform the search. Knowledge-guided decoding further reduces randomness and effectively imposes chemical validity constraints. We demonstrate ELILLM on the CrossDocked2020 benchmark, showing strong controlled exploration and high binding affinity scores compared with seven baseline methods. These results demonstrate that ELILLM can effectively enhance LLMs capabilities for SBDD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.21486v2">Hypothesis Generation via LLM-Automated Language Bias for ILP</a></div>
    <div class="paper-meta">
      📅 2026-01-19
      | 💬 accepted by AAAI 2026 Bridge LMReasoning
    </div>
    <details class="paper-abstract">
      Inductive Logic Programming (ILP) is a principled approach for generalizing regularities from data and constructing hypotheses as interpretable logic programs. However, a key limitation is its reliance on expert-crafted language bias - the predicate inventory, types, and mode declarations that delimit the search space. We propose hypothesis generation via LLM-automated language bias: multi-agent LLMs design the bias from raw text and translate descriptions into typed facts, and a robust ILP solver induces rules under a global consistency objective. This approach reduces traditional ILP's reliance on predefined symbolic structures and the noise sensitivity of LLM-only pipelines that directly generate hypotheses as text or code. Extensive experiments in diverse, challenging scenarios validate superior performance, providing a practical, explainable, and verifiable route to hypothesis generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12762v1">Teaching LLMs to Learn Tool Trialing and Execution through Environment Interaction</a></div>
    <div class="paper-meta">
      📅 2026-01-19
    </div>
    <details class="paper-abstract">
      Equipping Large Language Models (LLMs) with external tools enables them to solve complex real-world problems. However, the robustness of existing methods remains a critical challenge when confronting novel or evolving tools. Existing trajectory-centric paradigms primarily rely on memorizing static solution paths during training, which limits the ability of LLMs to generalize tool usage to newly introduced or previously unseen tools. In this paper, we propose ToolMaster, a framework that shifts tool use from imitating golden tool-calling trajectories to actively learning tool usage through interaction with the environment. To optimize LLMs for tool planning and invocation, ToolMaster adopts a trial-and-execution paradigm, which trains LLMs to first imitate teacher-generated trajectories containing explicit tool trials and self-correction, followed by reinforcement learning to coordinate the trial and execution phases jointly. This process enables agents to autonomously explore correct tool usage by actively interacting with environments and forming experiential knowledge that benefits tool execution. Experimental results demonstrate that ToolMaster significantly outperforms existing baselines in terms of generalization and robustness across unseen or unfamiliar tools. All code and data are available at https://github.com/NEUIR/ToolMaster.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12735v1">OpenAI for OpenAPI: Automated generation of REST API specification via LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-19
    </div>
    <details class="paper-abstract">
      REST APIs, based on the REpresentational State Transfer (REST) architecture, are the primary type of Web API. The OpenAPI Specification (OAS) serves as the de facto standard for describing REST APIs and is crucial for multiple software engineering tasks. However, developers face challenges in writing and maintaining OAS. Although static analysis shows potential for OAS generation, it is limited to specific programming languages and development frameworks. The powerful code understanding capabilities of LLMs offer new opportunities for OAS generation, yet they are constrained by context limitations and hallucinations. To address these challenges, we propose the OpenAI OpenAPI Project Scanner (OOPS), the first technology-agnostic LLM-based static analysis method for OAS generation, requiring fewer technology-specific rules and less human expert intervention. OOPS is implemented as an LLM agent workflow comprising two key steps: endpoint method extraction and OAS generation. By constructing an API dependency graph, it establishes necessary file associations to address LLMs' context limitations. Through multi-stage generation and self-refine, it mitigates both syntactic and semantic hallucinations during OAS generation. We evaluated OOPS on 12 real-world REST APIs spanning 5 programming languages and 8 development frameworks. Experimental results demonstrate that OOPS accurately generates high-quality OAS for REST APIs implemented with diverse technologies, achieving an average F1-score exceeding 98% for endpoint method inference, 97% for both request parameter and response inference, and 92% for parameter constraint inference. The input tokens average below 5.6K with a maximum of 16.2K, while the output tokens average below 0.9K with a maximum of 7.7K.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12680v1">MetaToolAgent: Towards Generalizable Tool Usage in LLMs through Meta-Learning</a></div>
    <div class="paper-meta">
      📅 2026-01-19
    </div>
    <details class="paper-abstract">
      Tool learning is increasingly important for large language models (LLMs) to effectively coordinate and utilize a diverse set of tools in order to solve complex real-world tasks. By selecting and integrating appropriate tools, LLMs extend their capabilities beyond pure language understanding to perform specialized functions. However, existing methods for tool selection often focus on limited tool sets and struggle to generalize to novel tools encountered in practical deployments. To address these challenges, we introduce a comprehensive dataset spanning 7 domains, containing 155 tools and 9,377 question-answer pairs, which simulates realistic integration scenarios. Additionally, we propose MetaToolAgent (MTA), a meta-learning approach designed to improve cross-tool generalization. Experimental results show that MTA significantly outperforms baseline methods on unseen tools, demonstrating its promise for building flexible and scalable systems that require dynamic tool coordination.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01576v2">OpenNovelty: An LLM-powered Agentic System for Verifiable Scholarly Novelty Assessment</a></div>
    <div class="paper-meta">
      📅 2026-01-19
    </div>
    <details class="paper-abstract">
      Evaluating novelty is critical yet challenging in peer review, as reviewers must assess submissions against a vast, rapidly evolving literature. This report presents OpenNovelty, an LLM-powered agentic system for transparent, evidence-based novelty analysis. The system operates through four phases: (1) extracting the core task and contribution claims to generate retrieval queries; (2) retrieving relevant prior work based on extracted queries via semantic search engine; (3) constructing a hierarchical taxonomy of core-task-related work and performing contribution-level full-text comparisons against each contribution; and (4) synthesizing all analyses into a structured novelty report with explicit citations and evidence snippets. Unlike naive LLM-based approaches, \textsc{OpenNovelty} grounds all assessments in retrieved real papers, ensuring verifiable judgments. We deploy our system on 500+ ICLR 2026 submissions with all reports publicly available on our website, and preliminary analysis suggests it can identify relevant prior work, including closely related papers that authors may overlook. OpenNovelty aims to empower the research community with a scalable tool that promotes fair, consistent, and evidence-backed peer review.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.13855v2">Harnessing Consistency for Robust Test-Time LLM Ensemble</a></div>
    <div class="paper-meta">
      📅 2026-01-19
      | 💬 18 pages, 15 figures
    </div>
    <details class="paper-abstract">
      Different large language models (LLMs) exhibit diverse strengths and weaknesses, and LLM ensemble serves as a promising approach to integrate their complementary capabilities. Despite substantial progress in improving ensemble quality, limited attention has been paid to the robustness of ensembles against potential erroneous signals, which often arise from heterogeneous tokenization schemes and varying model expertise. Our analysis shows that ensemble failures typically arise from both the token level and the model level: the former reflects severe disagreement in token predictions, while the latter involves low confidence and pronounced disparities among models. In light of this, we propose CoRE, a plug-and-play technique that harnesses model consistency for robust LLM ensemble, which can be seamlessly integrated with diverse ensemble methods. *Token-level consistency* captures fine-grained disagreements by applying a low-pass filter to downweight uncertain tokens with high inconsistency, often due to token misalignment, thereby improving robustness at a granular level. *Model-level consistency* models global agreement by promoting model outputs with high self-confidence and minimal divergence from others, enhancing robustness at a coarser level. Extensive experiments across diverse benchmarks, model combinations, and ensemble strategies demonstrate that CoRE consistently improves ensemble performance and robustness. Our code is available at https://github.com/zhichenz98/CoRE-EACL26.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12641v1">STEP-LLM: Generating CAD STEP Models from Natural Language with Large Language Models</a></div>
    <div class="paper-meta">
      📅 2026-01-19
      | 💬 Accepted to the Design, Automation & Test in Europe Conference (DATE) 2026
    </div>
    <details class="paper-abstract">
      Computer-aided design (CAD) is vital to modern manufacturing, yet model creation remains labor-intensive and expertise-heavy. To enable non-experts to translate intuitive design intent into manufacturable artifacts, recent large language models-based text-to-CAD efforts focus on command sequences or script-based formats like CadQuery. However, these formats are kernel-dependent and lack universality for manufacturing. In contrast, the Standard for the Exchange of Product Data (STEP, ISO 10303) file is a widely adopted, neutral boundary representation (B-rep) format directly compatible with manufacturing, but its graph-structured, cross-referenced nature poses unique challenges for auto-regressive LLMs. To address this, we curate a dataset of ~40K STEP-caption pairs and introduce novel preprocessing tailored for the graph-structured format of STEP, including a depth-first search-based reserialization that linearizes cross-references while preserving locality and chain-of-thought(CoT)-style structural annotations that guide global coherence. We integrate retrieval-augmented generation to ground predictions in relevant examples for supervised fine-tuning, and refine generation quality through reinforcement learning with a specific Chamfer Distance-based geometric reward. Experiments demonstrate consistent gains of our STEP-LLM in geometric fidelity over the Text2CAD baseline, with improvements arising from multiple stages of our framework: the RAG module substantially enhances completeness and renderability, the DFS-based reserialization strengthens overall accuracy, and the RL further reduces geometric discrepancy. Both metrics and visual comparisons confirm that STEP-LLM generates shapes with higher fidelity than Text2CAD. These results show the feasibility of LLM-driven STEP model generation from natural language, showing its potential to democratize CAD design for manufacturing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04700v2">PRISM: A Unified Framework for Post-Training LLMs Without Verifiable Rewards</a></div>
    <div class="paper-meta">
      📅 2026-01-19
      | 💬 Added open-sourced github url
    </div>
    <details class="paper-abstract">
      Current techniques for post-training Large Language Models (LLMs) rely either on costly human supervision or on external verifiers to boost performance on tasks such as mathematical reasoning and code generation. However, as LLMs improve their problem-solving, any further improvement will potentially require high-quality solutions to difficult problems that are not available to humans. As a result, learning from unlabeled data is becoming increasingly attractive in the research community. Existing methods extract learning signal from a model's consistency, either by majority voting or by converting the model's internal confidence into reward. Although internal consistency metric such as entropy or self-certainty require no human intervention, as we show in this work, these are unreliable signals for large-scale and long-term training. To address the unreliability, we propose PRISM, a unified training framework that uses a Process Reward Model (PRM) to guide learning alongside model's internal confidence in the absence of ground-truth labels. We show that effectively combining PRM with self-certainty can lead to both stable training and better test-time performance, and also keep the model's internal confidence in check. Code available at https://github.com/ghimiremukesh/PRISM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.22936v2">Evaluating Large Language Models (LLMs) in Financial NLP: A Comparative Study on Financial Report Analysis</a></div>
    <div class="paper-meta">
      📅 2026-01-19
      | 💬 23 Pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to support the analysis of complex financial disclosures, yet their reliability, behavioral consistency, and transparency remain insufficiently understood in high-stakes settings. This paper presents a controlled evaluation of five transformer-based LLMs applied to question answering over the Business sections of U.S. 10-K filings. To capture complementary aspects of model behavior, we combine human evaluation, automated similarity metrics, and behavioral diagnostics under standardized and context-controlled prompting conditions. Human assessments indicate that models differ in their average performance across qualitative dimensions such as relevance, completeness, clarity, conciseness, and factual accuracy, though inter-rater agreement is modest, reflecting the subjective nature of these criteria. Automated metrics reveal systematic differences in lexical overlap and semantic similarity across models, while behavioral diagnostics highlight variation in response stability and cross-prompt alignment. Importantly, no single model consistently dominates across all evaluation perspectives. Together, these findings suggest that apparent performance differences should be interpreted as relative tendencies under the tested conditions rather than definitive indicators of general reliability. The results underscore the need for evaluation frameworks that account for human disagreement, behavioral variability, and interpretability when deploying LLMs in financially consequential applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.20139v2">StructEval: Benchmarking LLMs' Capabilities to Generate Structural Outputs</a></div>
    <div class="paper-meta">
      📅 2026-01-19
      | 💬 24 pages, 8 figures, 14 tables
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) become integral to software development workflows, their ability to generate structured outputs has become critically important. We introduce StructEval, a comprehensive benchmark for evaluating LLMs' capabilities in producing both non-renderable (JSON, YAML, CSV) and renderable (HTML, React, SVG) structured formats. Unlike prior benchmarks, StructEval systematically evaluates structural fidelity across diverse formats through two paradigms: 1) generation tasks, producing structured output from natural language prompts, and \textbf{2)} conversion tasks, translating between structured formats. Our benchmark encompasses 18 formats and 44 types of task, with novel metrics for format adherence and structural correctness. Results reveal significant performance gaps-even state-of-the-art models like o1-mini achieve only 75.58 average score, with open-source alternatives lagging approximately 10 points behind. We find generation tasks more challenging than conversion tasks, and producing correct visual content more difficult than generating text-only structures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13398v1">Can LLMs Compress (and Decompress)? Evaluating Code Understanding and Execution via Invertibility</a></div>
    <div class="paper-meta">
      📅 2026-01-19
      | 💬 32 pages (preprint)
    </div>
    <details class="paper-abstract">
      LLMs demonstrate strong performance on code benchmarks, yet round-trip code execution reveals limitations in their ability to maintain consistent reasoning across forward and backward execution. We present RoundTripCodeEval (RTCE), a comprehensive benchmark consisting of four distinct code execution reasoning tasks designed to rigorously test round-trip consistency. RTCE provides an execution-free, exact-match evaluation of bijection fidelity, assessing whether models preserve a consistent one-to-one mapping between encoding and decoding operations across various algorithms and directions. We systematically evaluate state-of-the-art Code-LLMs using zero-shot prompting, supervised fine-tuning on execution traces, and self-reflection mechanisms. Each yields modest improvements, but none closes the gap, indicating that current LLMs struggle with true round-trip consistency, which demonstrates that they lack the internal coherence required for trustworthy code reasoning. RTCE surfaces several new and previously unmeasured insights that are not captured by existing I/O-prediction, execution-reasoning, or round-trip natural-language benchmarks. We will release the code and the dataset upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13392v1">Beyond Memorization: Testing LLM Reasoning on Unseen Theory of Computation Tasks</a></div>
    <div class="paper-meta">
      📅 2026-01-19
      | 💬 30 pages, 11 figures, 6 tables, Work in Progress
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated strong performance on formal language tasks, yet whether this reflects genuine symbolic reasoning or pattern matching on familiar constructions remains unclear. We introduce a benchmark for deterministic finite automata (DFA) construction from regular languages, comprising factual knowledge questions, seen construction problems from public sources, and two types of unseen problems: hand-crafted instances with multiple interacting constraints and systematically generated problems via Arden's theorem. Models achieve perfect accuracy on factual questions and 84-90% on seen tasks. However, accuracy drops sharply on unseen problems (by 30-64%), with failures stemming from systematic misinterpretation of language constraints, incorrect handling of Kleene-star semantics, and a failure to preserve global consistency. We evaluate a three-stage hint protocol that enables correction of shallow errors but does not reliably resolve globally inconsistent or structurally flawed automata. Our analysis across multiple prompting strategies (direct, Chain-of-Thought, Tree-of-Thought) reveals that errors persist regardless of prompting approach, exposing a fundamental gap between LLMs' ability to generate syntactically plausible DFAs and their capacity for semantically correct formal reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13359v1">Sockpuppetting: Jailbreaking LLMs Without Optimization Through Output Prefix Injection</a></div>
    <div class="paper-meta">
      📅 2026-01-19
    </div>
    <details class="paper-abstract">
      As open-weight large language models (LLMs) increase in capabilities, safeguarding them against malicious prompts and understanding possible attack vectors becomes ever more important. While automated jailbreaking methods like GCG [Zou et al., 2023] remain effective, they often require substantial computational resources and specific expertise. We introduce "sockpuppetting'', a simple method for jailbreaking open-weight LLMs by inserting an acceptance sequence (e.g., "Sure, here is how to...'') at the start of a model's output and allowing it to complete the response. Requiring only a single line of code and no optimization, sockpuppetting achieves up to 80% higher attack success rate (ASR) than GCG on Qwen3-8B in per-prompt comparisons. We also explore a hybrid approach that optimizes the adversarial suffix within the assistant message block rather than the user prompt, increasing ASR by 64% over GCG on Llama-3.1-8B in a prompt-agnostic setting. The results establish sockpuppetting as an effective low-cost attack accessible to unsophisticated adversaries, highlighting the need for defences against output-prefix injection in open-weight models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13352v1">LLM-as-RNN: A Recurrent Language Model for Memory Updates and Sequence Prediction</a></div>
    <div class="paper-meta">
      📅 2026-01-19
      | 💬 17 pages, 5 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Large language models are strong sequence predictors, yet standard inference relies on immutable context histories. After making an error at generation step t, the model lacks an updatable memory mechanism that improves predictions for step t+1. We propose LLM-as-RNN, an inference-only framework that turns a frozen LLM into a recurrent predictor by representing its hidden state as natural-language memory. This state, implemented as a structured system-prompt summary, is updated at each timestep via feedback-driven text rewrites, enabling learning without parameter updates. Under a fixed token budget, LLM-as-RNN corrects errors and retains task-relevant patterns, effectively performing online learning through language. We evaluate the method on three sequential benchmarks in healthcare, meteorology, and finance across Llama, Gemma, and GPT model families. LLM-as-RNN significantly outperforms zero-shot, full-history, and MemPrompt baselines, improving predictive accuracy by 6.5% on average, while producing interpretable, human-readable learning traces absent in standard context accumulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09822v2">LLM-Based Agentic Systems for Software Engineering: Challenges and Opportunities</a></div>
    <div class="paper-meta">
      📅 2026-01-19
      | 💬 Accepted to GenSE 2026 workshop
    </div>
    <details class="paper-abstract">
      Despite recent advancements in Large Language Models (LLMs), complex Software Engineering (SE) tasks require more collaborative and specialized approaches. This concept paper systematically reviews the emerging paradigm of LLM-based multi-agent systems, examining their applications across the Software Development Life Cycle (SDLC), from requirements engineering and code generation to static code checking, testing, and debugging. We delve into a wide range of topics such as language model selection, SE evaluation benchmarks, state-of-the-art agentic frameworks and communication protocols. Furthermore, we identify key challenges and outline future research opportunities, with a focus on multi-agent orchestration, human-agent coordination, computational cost optimization, and effective data collection. This work aims to provide researchers and practitioners with valuable insights into the current forefront landscape of agentic systems within the software engineering domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13300v1">OI-Bench: An Option Injection Benchmark for Evaluating LLM Susceptibility to Directive Interference</a></div>
    <div class="paper-meta">
      📅 2026-01-19
    </div>
    <details class="paper-abstract">
      Benchmarking large language models (LLMs) is critical for understanding their capabilities, limitations, and robustness. In addition to interface artifacts, prior studies have shown that LLM decisions can be influenced by directive signals such as social cues, framing, and instructions. In this work, we introduce option injection, a benchmarking approach that augments the multiple-choice question answering (MCQA) interface with an additional option containing a misleading directive, leveraging standardized choice structure and scalable evaluation. We construct OI-Bench, a benchmark of 3,000 questions spanning knowledge, reasoning, and commonsense tasks, with 16 directive types covering social compliance, bonus framing, threat framing, and instructional interference. This setting combines manipulation of the choice interface with directive-based interference, enabling systematic assessment of model susceptibility. We evaluate 12 LLMs to analyze attack success rates, behavioral responses, and further investigate mitigation strategies ranging from inference-time prompting to post-training alignment. Experimental results reveal substantial vulnerabilities and heterogeneous robustness across models. OI-Bench is expected to support more systematic evaluation of LLM robustness to directive interference within choice-based interfaces.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09001v2">Entropy Sentinel: Continuous LLM Accuracy Monitoring from Decoding Entropy Traces in STEM</a></div>
    <div class="paper-meta">
      📅 2026-01-19
    </div>
    <details class="paper-abstract">
      Deploying LLMs raises two coupled challenges: (1) monitoring - estimating where a model underperforms as traffic and domains drift - and (2) improvement - prioritizing data acquisition to close the largest performance gaps. We test whether an inference-time signal can estimate slice-level accuracy under domain shift. For each response, we compute an output-entropy profile from final-layer next-token probabilities (from top-k logprobs) and summarize it with eleven statistics. A lightweight classifier predicts instance correctness, and averaging predicted probabilities yields a domain-level accuracy estimate. We evaluate on ten STEM reasoning benchmarks with exhaustive train/test compositions (k in {1,2,3,4}; all "10 choose k" combinations), across nine LLMs from six families (3B-20B). Estimates often track held-out benchmark accuracy, and several models show near-monotonic ordering of domains. Output-entropy profiles are thus an accessible signal for scalable monitoring and for targeting data acquisition.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13288v1">A BERTology View of LLM Orchestrations: Token- and Layer-Selective Probes for Efficient Single-Pass Classification</a></div>
    <div class="paper-meta">
      📅 2026-01-19
    </div>
    <details class="paper-abstract">
      Production LLM systems often rely on separate models for safety and other classification-heavy steps, increasing latency, VRAM footprint, and operational complexity. We instead reuse computation already paid for by the serving LLM: we train lightweight probes on its hidden states and predict labels in the same forward pass used for generation. We frame classification as representation selection over the full token-layer hidden-state tensor, rather than committing to a fixed token or fixed layer (e.g., first-token logits or final-layer pooling). To implement this, we introduce a two-stage aggregator that (i) summarizes tokens within each layer and (ii) aggregates across layer summaries to form a single representation for classification. We instantiate this template with direct pooling, a 100K-parameter scoring-attention gate, and a downcast multi-head self-attention (MHA) probe with up to 35M trainable parameters. Across safety and sentiment benchmarks our probes improve over logit-only reuse (e.g., MULI) and are competitive with substantially larger task-specific baselines, while preserving near-serving latency and avoiding the VRAM and latency costs of a separate guard-model pipeline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13284v1">Balancing Classification and Calibration Performance in Decision-Making LLMs via Calibration Aware Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-01-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed in decision-making tasks, where not only accuracy but also reliable confidence estimates are essential. Well-calibrated confidence enables downstream systems to decide when to trust a model and when to defer to fallback mechanisms. In this work, we conduct a systematic study of calibration in two widely used fine-tuning paradigms: supervised fine-tuning (SFT) and reinforcement learning with verifiable rewards (RLVR). We show that while RLVR improves task performance, it produces extremely overconfident models, whereas SFT yields substantially better calibration, even under distribution shift, though with smaller performance gains. Through targeted experiments, we diagnose RLVR's failure, showing that decision tokens act as extraction steps of the decision in reasoning traces and do not carry confidence information, which prevents reinforcement learning from surfacing calibrated alternatives. Based on this insight, we propose a calibration-aware reinforcement learning formulation that directly adjusts decision-token probabilities. Our method preserves RLVR's accuracy level while mitigating overconfidence, reducing ECE scores up to 9 points.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08856v2">LAUDE: LLM-Assisted Unit Test Generation and Debugging of Hardware DEsigns</a></div>
    <div class="paper-meta">
      📅 2026-01-19
      | 💬 18 Pages, 21 Figures, Submitted to ARR Review
    </div>
    <details class="paper-abstract">
      Unit tests are critical in the hardware design lifecycle to ensure that component design modules are functionally correct and conform to the specification before they are integrated at the system level. Thus developing unit tests targeting various design features requires deep understanding of the design functionality and creativity. When one or more unit tests expose a design failure, the debugging engineer needs to diagnose, localize, and debug the failure to ensure design correctness, which is often a painstaking and intense process. In this work, we introduce LAUDE, a unified unit-test generation and debugging framework for hardware designs that cross-pollinates the semantic understanding of the design source code with the Chain-of-Thought (CoT) reasoning capabilities of foundational Large-Language Models (LLMs). LAUDE integrates prompt engineering and design execution information to enhance its unit test generation accuracy and code debuggability. We apply LAUDE with closed- and open-source LLMs to a large corpus of buggy hardware design codes derived from the VerilogEval dataset, where generated unit tests detected bugs in up to 100% and 93% of combinational and sequential designs and debugged up to 93% and 84% of combinational and sequential designs, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13264v1">Unlearning in LLMs: Methods, Evaluation, and Open Challenges</a></div>
    <div class="paper-meta">
      📅 2026-01-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable success across natural language processing tasks, yet their widespread deployment raises pressing concerns around privacy, copyright, security, and bias. Machine unlearning has emerged as a promising paradigm for selectively removing knowledge or data from trained models without full retraining. In this survey, we provide a structured overview of unlearning methods for LLMs, categorizing existing approaches into data-centric, parameter-centric, architecture-centric, hybrid, and other strategies. We also review the evaluation ecosystem, including benchmarks, metrics, and datasets designed to measure forgetting effectiveness, knowledge retention, and robustness. Finally, we outline key challenges and open problems, such as scalable efficiency, formal guarantees, cross-language and multimodal unlearning, and robustness against adversarial relearning. By synthesizing current progress and highlighting open directions, this paper aims to serve as a roadmap for developing reliable and responsible unlearning techniques in large language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.17156v3">LLM-based relevance assessment still can't replace human relevance assessment</a></div>
    <div class="paper-meta">
      📅 2026-01-19
    </div>
    <details class="paper-abstract">
      The use of large language models (LLMs) for relevance assessment in information retrieval has gained significant attention, with recent studies suggesting that LLM-based judgments provide comparable evaluations to human judgments. Notably, based on TREC 2024 data, Upadhyay et al make a bold claim that LLM-based relevance assessments, such as those generated by the Umbrela system, can fully replace traditional human relevance assessments in TREC-style evaluations. This paper critically examines this claim, highlighting practical and theoretical limitations that undermine the validity of this conclusion. First, we question whether the evidence provided by Upadhyay et al. genuinely supports their claim, particularly when the test collection is intended to serve as a benchmark for future research innovations.Second, we submit a system deliberately crafted to exploit automatic evaluation metrics, demonstrating that it can achieve artificially inflated scores without truly improving retrieval quality. Third, we simulate the consequences of circularity by analyzing Kendall's tau correlations under the hypothetical scenario in which all systems adopt Umbrela as a final-stage re-ranker, illustrating how reliance on LLM-based assessments can distort system rankings. Theoretical challenges - including the inherent narcissism of LLMs, the risk of overfitting to LLM-based metrics, and the potential degradation of future LLM performance - that must be addressed before LLM-based relevance assessments can be considered a viable replacement for human judgments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08121v2">Balanced Accuracy: The Right Metric for Evaluating LLM Judges -- Explained through Youden's J statistic</a></div>
    <div class="paper-meta">
      📅 2026-01-19
      | 💬 10 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Rigorous evaluation of large language models (LLMs) relies on comparing models by the prevalence of desirable or undesirable behaviors, such as task pass rates or policy violations. These prevalence estimates are produced by a classifier, either an LLM-as-a-judge or human annotators, making the choice of classifier central to trustworthy evaluation. Common metrics used for this choice, such as Accuracy, Precision, and F1, are sensitive to class imbalance and to arbitrary choices of positive class, and can favor judges that distort prevalence estimates. We show that Youden's $J$ statistic is theoretically aligned with choosing the best judge to compare models, and that Balanced Accuracy is an equivalent linear transformation of $J$. Through both analytical arguments and empirical examples and simulations, we demonstrate how selecting judges using Balanced Accuracy leads to better, more robust classifier selection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13243v1">A Comprehensive Evaluation of LLM Reasoning: From Single-Model to Multi-Agent Paradigms</a></div>
    <div class="paper-meta">
      📅 2026-01-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed as reasoning systems, where reasoning paradigms - such as Chain-of-Thought (CoT) and multi-agent systems (MAS) - play a critical role, yet their relative effectiveness and cost-accuracy trade-offs remain poorly understood. In this work, we conduct a comprehensive and unified evaluation of reasoning paradigms, spanning direct single-model generation, CoT-augmented single-model reasoning, and representative MAS workflows, characterizing their reasoning performance across a diverse suite of closed-form benchmarks. Beyond overall performance, we probe role-specific capability demands in MAS using targeted role isolation analyses, and analyze cost-accuracy trade-offs to identify which MAS workflows offer a favorable balance between cost and accuracy, and which incur prohibitive overhead for marginal gains. We further introduce MIMeBench, a new open-ended benchmark that targets two foundational yet underexplored semantic capabilities - semantic abstraction and contrastive discrimination - thereby providing an alternative evaluation axis beyond closed-form accuracy and enabling fine-grained assessment of semantic competence that is difficult to capture with existing benchmarks. Our results show that increased structural complexity does not consistently lead to improved reasoning performance, with its benefits being highly dependent on the properties and suitability of the reasoning paradigm itself. The codes are released at https://gitcode.com/HIT1920/OpenLLMBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.19202v2">Demystifying Scientific Problem-Solving in LLMs by Probing Knowledge and Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-19
      | 💬 32 pages, 18 figures
    </div>
    <details class="paper-abstract">
      Scientific problem solving poses unique challenges for LLMs, requiring both deep domain knowledge and the ability to apply such knowledge through complex reasoning. While automated scientific reasoners hold great promise for assisting human scientists, there is currently no widely adopted holistic benchmark for evaluating scientific reasoning, and few approaches systematically disentangle the distinct roles of knowledge and reasoning in these tasks. To address these gaps, we introduce SciReas, a diverse suite of existing benchmarks for scientific reasoning tasks, and SciReas-Pro, a selective subset that requires more complex reasoning. Our holistic evaluation surfaces insights about scientific reasoning performance that remain hidden when relying on individual benchmarks alone. We then propose KRUX, a probing framework for studying the distinct roles of reasoning and knowledge in scientific tasks. Combining the two, we conduct an in-depth analysis that yields several key findings: (1) Retrieving task-relevant knowledge from model parameters is a critical bottleneck for LLMs in scientific reasoning; (2) Reasoning models consistently benefit from external knowledge added in-context on top of the reasoning enhancement; (3) Enhancing verbalized reasoning improves LLMs' ability to surface task-relevant knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.17091v2">Comparing the Framing Effect in Humans and LLMs on Naturally Occurring Texts</a></div>
    <div class="paper-meta">
      📅 2026-01-19
    </div>
    <details class="paper-abstract">
      Humans are influenced by how information is presented, a phenomenon known as the framing effect. Prior work suggests that LLMs may also be susceptible to framing, but it has relied on synthetic data and did not compare to human behavior. To address this gap, we introduce WildFrame - a dataset for evaluating LLM responses to positive and negative framing in naturally-occurring sentences, alongside human responses on the same data. WildFrame consists of 1,000 real-world texts selected to convey a clear sentiment; we then reframe each text in either a positive or negative light and collect human sentiment annotations. Evaluating eleven LLMs on WildFrame, we find that all models respond to reframing in a human-like manner ($r\geq0.52$), and that both humans and models are influenced more by positive than negative reframing. Notably, GPT models are the least correlated with human behavior among all tested models. These findings raise a discussion around the goals of state-of-the-art LLM development and whether models should align closely with human behavior, to preserve cognitive phenomena such as the framing effect, or instead mitigate such biases in favor of fairness and consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13206v1">Real-Time Deadlines Reveal Temporal Awareness Failures in LLM Strategic Dialogues</a></div>
    <div class="paper-meta">
      📅 2026-01-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) generate text token-by-token in discrete time, yet real-world communication, from therapy sessions to business negotiations, critically depends on continuous time constraints. Current LLM architectures and evaluation protocols rarely test for temporal awareness under real-time deadlines. We use simulated negotiations between paired agents under strict deadlines to investigate how LLMs adjust their behavior in time-sensitive settings. In a control condition, agents know only the global time limit. In a time-aware condition, they receive remaining-time updates at each turn. Deal closure rates are substantially higher (32\% vs. 4\% for GPT-5.1) and offer acceptances are sixfold higher in the time-aware condition than in the control, suggesting LLMs struggle to internally track elapsed time. However, the same LLMs achieve near-perfect deal closure rates ($\geq$95\%) under turn-based limits, revealing the failure is in temporal tracking rather than strategic reasoning. These effects replicate across negotiation scenarios and models, illustrating a systematic lack of LLM time awareness that will constrain LLM deployment in many time-sensitive applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13155v1">Probe and Skip: Self-Predictive Token Skipping for Efficient Long-Context LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-01-19
    </div>
    <details class="paper-abstract">
      Long-context inference enhances the reasoning capability of Large Language Models (LLMs) while incurring significant computational overhead. Token-oriented methods, such as pruning and skipping, have shown promise in reducing inference latency, but still suffer from inherently limited acceleration potential, outdated proxy signals, and redundancy interference, thus yielding suboptimal speed-accuracy trade-offs. To address these challenges, we propose SPTS (Self-Predictive Token Skipping), a training-free framework for efficient long-context LLM inference. Specifically, motivated by the thought of probing the influence of targeted skipping layers, we design two component-specific strategies for selective token skipping: Partial Attention Probing (PAP) for multi-head attention, which selects informative tokens by performing partial forward attention computation, and Low-rank Transformation Probing (LTP) for feed forward network, which constructs a low-rank proxy network to predict token transformations. Furthermore, a Multi-Stage Delayed Pruning (MSDP) strategy reallocates the skipping budget and progressively prunes redundant tokens across layers. Extensive experiments demonstrate the effectiveness of our method, achieving up to 2.46$\times$ and 2.29$\times$ speedups for prefilling and end-to-end generation, respectively, while maintaining state-of-the-art model performance. The source code will be publicly available upon paper acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.16029v4">FDLLM: A Dedicated Detector for Black-Box LLMs Fingerprinting</a></div>
    <div class="paper-meta">
      📅 2026-01-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are rapidly transforming the landscape of digital content creation. However, the prevalent black-box Application Programming Interface (API) access to many LLMs introduces significant challenges in accountability, governance, and security. LLM fingerprinting, which aims to identify the source model by analyzing statistical and stylistic features of generated text, offers a potential solution. Current progress in this area is hindered by a lack of dedicated datasets and the need for efficient, practical methods that are robust against adversarial manipulations. To address these challenges, we introduce FD-Dataset, a comprehensive bilingual fingerprinting benchmark comprising 90,000 text samples from 20 famous proprietary and open-source LLMs. Furthermore, we present FDLLM, a novel fingerprinting method that leverages parameter-efficient Low-Rank Adaptation (LoRA) to fine-tune a foundation model. This approach enables LoRA to extract deep, persistent features that characterize each source LLM. Through our analysis, we find that LoRA adaptation promotes the aggregation of outputs from the same LLM in representation space while enhancing the separation between different LLMs. This mechanism explains why LoRA proves particularly effective for LLM fingerprinting. Extensive empirical evaluations on FD-Dataset demonstrate FDLLM's superiority, achieving a Macro F1 score 22.1% higher than the strongest baseline. FDLLM also exhibits strong generalization to newly released models, achieving an average accuracy of 95% on unseen models. Notably, FDLLM remains consistently robust under various adversarial attacks, including polishing, translation, and synonym substitution. Experimental results show that FDLLM reduces the average attack success rate from 49.2% (LM-D) to 23.9%.
    </details>
</div>
