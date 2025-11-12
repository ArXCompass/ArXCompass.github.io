# llm - 2025_08

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
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20453v1">MCP-Bench: Benchmarking Tool-Using LLM Agents with Complex Real-World Tasks via MCP Servers</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      We introduce MCP-Bench, a benchmark for evaluating large language models (LLMs) on realistic, multi-step tasks that demand tool use, cross-tool coordination, precise parameter control, and planning/reasoning for solving tasks. Built on the Model Context Protocol (MCP), MCP-Bench connects LLMs to 28 representative live MCP servers spanning 250 tools across domains such as finance, traveling, scientific computing, and academic search. Unlike prior API-based benchmarks, each MCP server provides a set of complementary tools designed to work together, enabling the construction of authentic, multi-step tasks with rich input-output coupling. Tasks in MCP-Bench test agents' ability to retrieve relevant tools from fuzzy instructions without explicit tool names, plan multi-hop execution trajectories for complex objectives, ground responses in intermediate tool outputs, and orchestrate cross-domain workflows - capabilities not adequately evaluated by existing benchmarks that rely on explicit tool specifications, shallow few-step workflows, and isolated domain operations. We propose a multi-faceted evaluation framework covering tool-level schema understanding and usage, trajectory-level planning, and task completion. Experiments on 20 advanced LLMs reveal persistent challenges in MCP-Bench. Code and data: https://github.com/Accenture/mcp-bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20444v1">Ransomware 3.0: Self-Composing and LLM-Orchestrated</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      Using automated reasoning, code synthesis, and contextual decision-making, we introduce a new threat that exploits large language models (LLMs) to autonomously plan, adapt, and execute the ransomware attack lifecycle. Ransomware 3.0 represents the first threat model and research prototype of LLM-orchestrated ransomware. Unlike conventional malware, the prototype only requires natural language prompts embedded in the binary; malicious code is synthesized dynamically by the LLM at runtime, yielding polymorphic variants that adapt to the execution environment. The system performs reconnaissance, payload generation, and personalized extortion, in a closed-loop attack campaign without human involvement. We evaluate this threat across personal, enterprise, and embedded environments using a phase-centric methodology that measures quantitative fidelity and qualitative coherence in each attack phase. We show that open source LLMs can generate functional ransomware components and sustain closed-loop execution across diverse environments. Finally, we present behavioral signals and multi-level telemetry of Ransomware 3.0 through a case study to motivate future development of better defenses and policy enforcements to address novel AI-enabled ransomware attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20443v1">Towards Mitigating Excessive Forgetting in LLM Unlearning via Entanglement-Aware Unlearning with Proxy Constraint</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are trained on massive datasets that may include private or copyrighted content. Due to growing privacy and ownership concerns, data owners may request the removal of their data from trained models. Machine unlearning provides a practical solution by removing the influence of specific data without full retraining. However, most existing methods lack a sound forgetting boundary, causing some samples to be under-forgotten, leaving residual leakage risks, while others remain over-forgotten at the expense of degraded utility. In this work, we propose EAGLE-PC (Entanglement-Awareness Guided Loss Reweighting with Proxy Constraint), a novel unlearning framework that addresses these limitations through two key components. First, entanglement-awareness guided loss reweighting determines the forgetting effort of each sample by measuring its similarity to retain samples in the embedding space, enabling more targeted and effective unlearning. Second, a proxy constraint leveraging ICL (In-Context Learning) generated test data softly regularizes the forgetting process, effectively mitigating over-forgetting. EAGLE-PC is compatible with existing gradient-based objectives and serves as a plug-and-play enhancement. We evaluate EAGLE-PC on the TOFU and MUSE benchmarks, showing consistent improvements in the forgetting-utility trade-off across multiple LLMs. Combined with the NPO+GD optimizer, it approaches full retraining performance, offering a scalable and robust unlearning solution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20420v1">CAMB: A comprehensive industrial LLM benchmark on civil aviation maintenance</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      Civil aviation maintenance is a domain characterized by stringent industry standards. Within this field, maintenance procedures and troubleshooting represent critical, knowledge-intensive tasks that require sophisticated reasoning. To address the lack of specialized evaluation tools for large language models (LLMs) in this vertical, we propose and develop an industrial-grade benchmark specifically designed for civil aviation maintenance. This benchmark serves a dual purpose: It provides a standardized tool to measure LLM capabilities within civil aviation maintenance, identifying specific gaps in domain knowledge and complex reasoning. By pinpointing these deficiencies, the benchmark establishes a foundation for targeted improvement efforts (e.g., domain-specific fine-tuning, RAG optimization, or specialized prompt engineering), ultimately facilitating progress toward more intelligent solutions within civil aviation maintenance. Our work addresses a significant gap in the current LLM evaluation, which primarily focuses on mathematical and coding reasoning tasks. In addition, given that Retrieval-Augmented Generation (RAG) systems are currently the dominant solutions in practical applications , we leverage this benchmark to evaluate existing well-known vector embedding models and LLMs for civil aviation maintenance scenarios. Through experimental exploration and analysis, we demonstrate the effectiveness of our benchmark in assessing model performance within this domain, and we open-source this evaluation benchmark and code to foster further research and development:https://github.com/CamBenchmark/cambenchmark
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20416v1">DentalBench: Benchmarking and Advancing LLMs Capability for Bilingual Dentistry Understanding</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) and medical LLMs (Med-LLMs) have demonstrated strong performance on general medical benchmarks. However, their capabilities in specialized medical fields, such as dentistry which require deeper domain-specific knowledge, remain underexplored due to the lack of targeted evaluation resources. In this paper, we introduce DentalBench, the first comprehensive bilingual benchmark designed to evaluate and advance LLMs in the dental domain. DentalBench consists of two main components: DentalQA, an English-Chinese question-answering (QA) benchmark with 36,597 questions spanning 4 tasks and 16 dental subfields; and DentalCorpus, a large-scale, high-quality corpus with 337.35 million tokens curated for dental domain adaptation, supporting both supervised fine-tuning (SFT) and retrieval-augmented generation (RAG). We evaluate 14 LLMs, covering proprietary, open-source, and medical-specific models, and reveal significant performance gaps across task types and languages. Further experiments with Qwen-2.5-3B demonstrate that domain adaptation substantially improves model performance, particularly on knowledge-intensive and terminology-focused tasks, and highlight the importance of domain-specific benchmarks for developing trustworthy and effective LLMs tailored to healthcare applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20401v1">Revealing Potential Biases in LLM-Based Recommender Systems in the Cold Start Setting</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 In Proceedings of 2nd Workshop on Evaluating and Applying Recommendation Systems with Large Language Models (EARL) at RecSys 2025 (EARL 2025)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used for recommendation tasks due to their general-purpose capabilities. While LLMs perform well in rich-context settings, their behavior in cold-start scenarios, where only limited signals such as age, gender, or language are available, raises fairness concerns because they may rely on societal biases encoded during pretraining. We introduce a benchmark specifically designed to evaluate fairness in zero-context recommendation. Our modular pipeline supports configurable recommendation domains and sensitive attributes, enabling systematic and flexible audits of any open-source LLM. Through evaluations of state-of-the-art models (Gemma 3 and Llama 3.2), we uncover consistent biases across recommendation domains (music, movies, and colleges) including gendered and cultural stereotypes. We also reveal a non-linear relationship between model size and fairness, highlighting the need for nuanced analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20395v1">Measuring Reasoning Utility in LLMs via Conditional Entropy Reduction</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 11 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) often rely on generating intermediate reasoning steps to enhance accuracy. However, little work has examined how reasoning utility contributes to the final answer's correctness. Due to the stochastic nature of autoregressive generation, generating more context does not guarantee increased confidence in the answer. If we could predict, during generation, whether a reasoning step will be useful, we could stop early or prune ineffective steps, avoiding distractions in the final decision. We present an oracle study on MATH dataset, using Qwen2.5-32B and GPT-4o to generate reasoning chains, and then employing a separate model (Qwen3-8B) to quantify the utility of these chains for final accuracy. Specifically, we measure the model's uncertainty on the answer span Y at each reasoning step using conditional entropy (expected negative log-likelihood over the vocabulary) with context expanding step by step. Our results show a clear pattern: conditional entropy that decreases over steps is strongly associated with correct answers, whereas flat or increasing entropy often results in wrong answers. We also corroborate that incorrect reasoning paths tend to be longer than correct ones, suggesting that longer reasoning does not necessarily yield better outcomes. These findings serve as a foundation to inspire future work on designing efficient reasoning pipelines that detect and avoid unproductive reasoning early.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20384v1">Uncertainty Under the Curve: A Sequence-Level Entropy Area Metric for Reasoning LLM</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 Under review for AAAI 2026
    </div>
    <details class="paper-abstract">
      In this work, we introduce Entropy Area Score (EAS), a simple yet effective metric to quantify uncertainty in the answer generation process of reasoning large language models (LLMs). EAS requires neither external models nor repeated sampling, it integrates token-level predictive entropy from the model itself to capture the evolution of uncertainty during generation. Empirical results show that EAS is strongly correlated with answer entropy across models and datasets. In training data selection, EAS identifies high-potential samples and consistently outperforms Pass Rate filtering under equal sample budgets, improving student model accuracy on math benchmarks. EAS is both efficient and interpretable, offering a practical tool for uncertainty modeling and data quality assessment in LLM training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18721v3">LLM as an Execution Estimator: Recovering Missing Dependency for Practical Time-travelling Debugging</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      In this work, we propose RecovSlicing for computing dynamic data dependency in a single run, with only partial instrumentation. We explore the intuition that LLM can potentially infer program dynamics based on a partially recorded trace and relevant code as its context. Given (1) a partially recorded trace of a program $P$ and (2) the slicing criteria consisting of a query step $s$ and a query variable $v$ read by $s$, RecovSlicing computes the runtime definition of $v$ on the trace by estimating the miss-recorded execution of $P$. In this work, we allow the user to specify implicit query variable, for example, the implicit library variable used in $\texttt{list.get(i)}$. Technically, built upon non-deterministic LLM, we address the challenges of (1) precise recovery of runtime variable value and structure from the recorded execution and (2) aligning the memory address of recovered variables and the recorded variables for definition analysis. We extensively evaluate RecovSlicing against the state-of-the-art slicers such as Slicer4J, ND-Slicer, LLM Slicer, and re-execution Slicer on a total number of 8300 data-dependencies over 3 slicing benchmarks. The results show that RecovSlicing can significantly outperform the baselines. The accuracy and recall, achieving 80.3%, 91.1%, and 98.3% on the three benchmarks, whereas the best baseline reaches 39.0%, 82.0%, and 59.9% (accuracy), and 53.4%, 79.1%, and 87.1% (recall), respectively. In addition, we integrate RecovSlicing in a dual-slicing based regression bug localizer, significantly improving its performance by locating 16% more regressions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20038v2">Forewarned is Forearmed: Pre-Synthesizing Jailbreak-like Instructions to Enhance LLM Safety Guardrail to Potential Attacks</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 EMNLP 2025 findings
    </div>
    <details class="paper-abstract">
      Despite advances in improving large language model (LLM) to refuse to answer malicious instructions, widely used LLMs remain vulnerable to jailbreak attacks where attackers generate instructions with distributions differing from safety alignment corpora. New attacks expose LLMs' inability to recognize unseen malicious instructions, highlighting a critical distributional mismatch between training data and real-world attacks that forces developers into reactive patching cycles. To tackle this challenge, we propose IMAGINE, a synthesis framework that leverages embedding space distribution analysis to generate jailbreak-like instructions. This approach effectively fills the distributional gap between authentic jailbreak patterns and safety alignment corpora. IMAGINE follows an iterative optimization process that dynamically evolves text generation distributions across iterations, thereby augmenting the coverage of safety alignment data distributions through synthesized data examples. Based on the safety-aligned corpus enhanced through IMAGINE, our framework demonstrates significant decreases in attack success rate on Qwen2.5, Llama3.1, and Llama3.2 without compromising their utility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20373v1">Graph-R1: Unleashing LLM Reasoning with NP-Hard Graph Problems</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      Reasoning Large Language Models (RLLMs) have recently achieved remarkable progress on complex reasoning tasks, largely enabled by their long chain-of-thought (Long CoT) capabilities. However, developing these Long CoT behaviors relies heavily on post-training with high-quality datasets, which are typically costly and human-curated (e.g., mathematics and code), leaving scalable alternatives unexplored. In this work, we introduce NP-hard (NPH) graph problems as a novel synthetic training corpus, as they inherently require deep reasoning, extensive exploration, and reflective strategies, which are core characteristics of Long CoT reasoning. Building on this insight, we develop a two-stage post-training framework: (i) Long CoT Supervised Fine-Tuning (SFT) on rejection-sampled NPH graph instances, which substantially enhances reasoning depth, and (ii) Reinforcement Learning (RL) with a fine-grained reward design, which sharpens reasoning efficiency. Our flagship model, Graph-R1-7B, demonstrates strong generalization across mathematics, coding, STEM, and logic, and surpasses QwQ-32B on NPH graph problems in both accuracy and reasoning efficiency. These results position NPH graph problems as an effective and scalable resource for advancing Long CoT reasoning in LLMs, opening a new frontier for LLM post-training. Our implementation is available at https://github.com/Graph-Reasoner/Graph-R1, with models and datasets hosted in our Hugging Face collection HKUST-DSAIL/Graph-R1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20351v1">Joint Enhancement of Relational Reasoning for Long-Context LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 9 pages, 5 pages Accepted by EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Despite significant progress, large language models (LLMs) still struggle with long contexts due to memory limitations and their inability to tackle complex and long-context tasks. Additionally, LLMs often suffer from a lack of transparency and are prone to producing hallucinations. To address these challenges, we propose \textbf{JERR}, a novel framework designed to enhance long-context comprehension via graph-based reasoning in LLMs. JERR integrates three key components: synopsis extraction, graph construction, and relational reasoning. First, synopsis is extracted by chunking text strategically, allowing the model to summarize and understand information more efficiently. Second, we build a directed acyclic graph (DAG) to resolve redundancy, ensuring logical consistency and clarity. Finally, we incorporate Monte Carlo Tree Search (MCTS) to help the model navigate complex reasoning paths, ensuring more accurate and interpretable outputs. This framework provides a novel solution that enables LLMs to handle extended contexts and complex reasoning tasks with improved reliability and transparency. Experimental results show that JERR consistently outperforms all baselines on the ROUGE and F1 metrics, achieving the highest scores on the LLM-Rater evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20340v1">Boosting Skeleton-Driven SMT Solver Fuzzing by Leveraging LLM to Produce Formula Generators</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      Satisfiability Modulo Theory (SMT) solvers are foundational to modern systems and programming languages research, providing the foundation for tasks like symbolic execution and automated verification. Because these solvers sit on the critical path, their correctness is essential, and high-quality test formulas are key to uncovering bugs. However, while prior testing techniques performed well on earlier solver versions, they struggle to keep pace with rapidly evolving features. Recent approaches based on Large Language Models (LLMs) show promise in exploring advanced solver capabilities, but two obstacles remain: nearly half of the generated formulas are syntactically invalid, and iterative interactions with the LLMs introduce substantial computational overhead. In this study, we present Chimera, a novel LLM-assisted fuzzing framework that addresses both issues by shifting from direct formula generation to the synthesis of reusable term (i.e., logical expression) generators. Particularly, Chimera uses LLMs to (1) automatically extract context-free grammars (CFGs) for SMT theories, including solver-specific extensions, from documentation, and (2) synthesize composable Boolean term generators that adhere to these grammars. During fuzzing, Chimera populates structural skeletons derived from existing formulas with the terms iteratively produced by the LLM-synthesized generators. This design ensures syntactic validity while promoting semantic diversity. Notably, Chimera requires only one-time LLM interaction investment, dramatically reducing runtime cost. We evaluated Chimera on two leading SMT solvers: Z3 and cvc5. Our experiments show that Chimera has identified 43 confirmed bugs, 40 of which have already been fixed by developers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.19512v3">Safeguard Fine-Tuned LLMs Through Pre- and Post-Tuning Model Merging</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) for downstream tasks often leads to catastrophic forgetting, notably degrading the safety of originally aligned models. While some existing methods attempt to restore safety by incorporating additional safety data, the quality of such data typically falls short of that used in the original alignment process. Moreover, these high-quality safety datasets are generally inaccessible, making it difficult to fully recover the model's original safety. We ask: How can we preserve safety while improving downstream task performance without additional safety data? We show that simply merging the weights of pre- and post-fine-tuned models effectively mitigates safety degradation while enhancing performance. Experiments across different downstream tasks and models validate the method's practicality and effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16571v3">LLM-Based Agents for Competitive Landscape Mapping in Drug Asset Due Diligence</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      In this paper, we describe and benchmark a competitor-discovery component used within an agentic AI system for fast drug asset due diligence. A competitor-discovery AI agent, given an indication, retrieves all drugs comprising the competitive landscape of that indication and extracts canonical attributes for these drugs. The competitor definition is investor-specific, and data is paywalled/licensed, fragmented across registries, ontology-mismatched by indication, alias-heavy for drug names, multimodal, and rapidly changing. Although considered the best tool for this problem, the current LLM-based AI systems aren't capable of reliably retrieving all competing drug names, and there is no accepted public benchmark for this task. To address the lack of evaluation, we use LLM-based agents to transform five years of multi-modal, unstructured diligence memos from a private biotech VC fund into a structured evaluation corpus mapping indications to competitor drugs with normalized attributes. We also introduce a competitor validating LLM-as-a-judge agent that filters out false positives from the list of predicted competitors to maximize precision and suppress hallucinations. On this benchmark, our competitor-discovery agent achieves 83% recall, exceeding OpenAI Deep Research (65%) and Perplexity Labs (60%). The system is deployed in production with enterprise users; in a case study with a biotech VC investment fund, analyst turnaround time dropped from 2.5 days to $\sim$3 hours ($\sim$20x) for the competitive analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20333v1">Poison Once, Refuse Forever: Weaponizing Alignment for Injecting Bias in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are aligned to meet ethical standards and safety requirements by training them to refuse answering harmful or unsafe prompts. In this paper, we demonstrate how adversaries can exploit LLMs' alignment to implant bias, or enforce targeted censorship without degrading the model's responsiveness to unrelated topics. Specifically, we propose Subversive Alignment Injection (SAI), a poisoning attack that leverages the alignment mechanism to trigger refusal on specific topics or queries predefined by the adversary. Although it is perhaps not surprising that refusal can be induced through overalignment, we demonstrate how this refusal can be exploited to inject bias into the model. Surprisingly, SAI evades state-of-the-art poisoning defenses including LLM state forensics, as well as robust aggregation techniques that are designed to detect poisoning in FL settings. We demonstrate the practical dangers of this attack by illustrating its end-to-end impacts on LLM-powered application pipelines. For chat based applications such as ChatDoctor, with 1% data poisoning, the system refuses to answer healthcare questions to targeted racial category leading to high bias ($\Delta DP$ of 23%). We also show that bias can be induced in other NLP tasks: for a resume selection pipeline aligned to refuse to summarize CVs from a selected university, high bias in selection ($\Delta DP$ of 27%) results. Even higher bias ($\Delta DP$~38%) results on 9 other chat based downstream applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06463v2">Evaluating Efficiency and Novelty of LLM-Generated Code for Graph Analysis</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 7 pages, v2: minor revision to match final paper published in the The 29th Annual IEEE High Performance Extreme Computing Conference (HPEC), Virtual, September 15-19, 2025. Outstanding Student Paper Award
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to automate software development, yet most prior evaluations focus on functional correctness or high-level languages such as Python. As one of the first systematic explorations of LLM-assisted software performance engineering, we present a comprehensive study of LLMs' ability to generate efficient C implementations of graph-analysis routines -- code that must satisfy stringent runtime and memory constraints. This emerging field of LLM-assisted algorithm engineering holds significant promise, as these models may possess the capability to design novel approaches that improve existing algorithms and their implementations. Eight state-of-the-art models (OpenAI ChatGPT o3 and o4-mini-high, Anthropic Claude 4 Sonnet and Sonnet Extended, Google Gemini 2.5 Flash and Pro, xAI Grok 3-Think, and DeepSeek DeepThink R1) are benchmarked using two distinct approaches. The first approach evaluates the ability of LLMs to generate algorithms that outperform existing benchmarks. The second approach assesses their capability to generate graph algorithms for integration into performance-critical systems. The results show that Claude Sonnet 4 Extended achieves superior performance in ready-to-use code generation and efficiency, outperforming human-written baselines in triangle counting. Although our findings demonstrate that contemporary LLMs excel in optimizing and integrating established algorithms, the potential for these models to eventually invent transformative algorithmic techniques represents a compelling frontier for future research. We provide prompts, generated code, and measurement scripts to promote reproducible research in this rapidly evolving domain. All of the source code is available on GitHub at https://github.com/Bader-Research/LLM-triangle-counting/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20325v1">GUARD: Guideline Upholding Test through Adaptive Role-play and Jailbreak Diagnostics for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 54 pages
    </div>
    <details class="paper-abstract">
      As Large Language Models become increasingly integral to various domains, their potential to generate harmful responses has prompted significant societal and regulatory concerns. In response, governments have issued ethics guidelines to promote the development of trustworthy AI. However, these guidelines are typically high-level demands for developers and testers, leaving a gap in translating them into actionable testing questions to verify LLM compliance. To address this challenge, we introduce GUARD (\textbf{G}uideline \textbf{U}pholding Test through \textbf{A}daptive \textbf{R}ole-play and Jailbreak \textbf{D}iagnostics), a testing method designed to operationalize guidelines into specific guideline-violating questions that assess LLM adherence. To implement this, GUARD uses automated generation of guideline-violating questions based on government-issued guidelines, thereby testing whether responses comply with these guidelines. When responses directly violate guidelines, GUARD reports inconsistencies. Furthermore, for responses that do not directly violate guidelines, GUARD integrates the concept of ``jailbreaks'' to diagnostics, named GUARD-JD, which creates scenarios that provoke unethical or guideline-violating responses, effectively identifying potential scenarios that could bypass built-in safety mechanisms. Our method finally culminates in a compliance report, delineating the extent of adherence and highlighting any violations. We have empirically validated the effectiveness of GUARD on seven LLMs, including Vicuna-13B, LongChat-7B, Llama2-7B, Llama-3-8B, GPT-3.5, GPT-4, GPT-4o, and Claude-3.7, by testing compliance under three government-issued guidelines and conducting jailbreak diagnostics. Additionally, GUARD-JD can transfer jailbreak diagnostics to vision-language models, demonstrating its usage in promoting reliable LLM-based applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13580v2">LLM Test Generation via Iterative Hybrid Program Analysis</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 This paper has been accepted for publication in 2026 IEEE/ACM 48th International Conference on Software Engineering (ICSE 2026)
    </div>
    <details class="paper-abstract">
      Automating unit test generation remains a significant challenge, particularly for complex methods in real-world projects. While Large Language Models (LLMs) have made strides in code generation, they struggle to achieve high branch coverage due to their limited ability to reason about intricate control flow structures. To address this limitation, we introduce Panta, a technique that emulates the iterative process human developers follow when analyzing code and constructing test cases. Panta integrates static control flow analysis and dynamic code coverage analysis to systematically guide LLMs in identifying uncovered execution paths and generating better test cases. By incorporating an iterative feedback-driven mechanism, our technique continuously refines test generation based on static and dynamic path coverage insights, ensuring more comprehensive and effective testing. Our empirical evaluation, conducted on classes with high cyclomatic complexity from open-source projects, demonstrates that Panta achieves 26% higher line coverage and 23% higher branch coverage compared to the state-of-the-art.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21238v1">Addressing accuracy and hallucination of LLMs in Alzheimer's disease research through knowledge graphs</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      In the past two years, large language model (LLM)-based chatbots, such as ChatGPT, have revolutionized various domains by enabling diverse task completion and question-answering capabilities. However, their application in scientific research remains constrained by challenges such as hallucinations, limited domain-specific knowledge, and lack of explainability or traceability for the response. Graph-based Retrieval-Augmented Generation (GraphRAG) has emerged as a promising approach to improving chatbot reliability by integrating domain-specific contextual information before response generation, addressing some limitations of standard LLMs. Despite its potential, there are only limited studies that evaluate GraphRAG on specific domains that require intensive knowledge, like Alzheimer's disease or other biomedical domains. In this paper, we assess the quality and traceability of two popular GraphRAG systems. We compile a database of 50 papers and 70 expert questions related to Alzheimer's disease, construct a GraphRAG knowledge base, and employ GPT-4o as the LLM for answering queries. We then compare the quality of responses generated by GraphRAG with those from a standard GPT-4o model. Additionally, we discuss and evaluate the traceability of several Retrieval-Augmented Generation (RAG) and GraphRAG systems. Finally, we provide an easy-to-use interface with a pre-built Alzheimer's disease database for researchers to test the performance of both standard RAG and GraphRAG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21204v1">Fuzzy, Symbolic, and Contextual: Enhancing LLM Instruction via Cognitive Scaffolding</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      We study how architectural inductive biases influence the cognitive behavior of large language models (LLMs) in instructional dialogue. We introduce a symbolic scaffolding mechanism paired with a short-term memory schema designed to promote adaptive, structured reasoning in Socratic tutoring. Using controlled ablation across five system variants, we evaluate model outputs via expert-designed rubrics covering scaffolding, responsiveness, symbolic reasoning, and conversational memory. We present preliminary results using an LLM-based evaluation framework aligned to a cognitively grounded rubric. This enables scalable, systematic comparisons across architectural variants in early-stage experimentation. The preliminary results show that our full system consistently outperforms baseline variants. Analysis reveals that removing memory or symbolic structure degrades key cognitive behaviors, including abstraction, adaptive probing, and conceptual continuity. These findings support a processing-level account in which architectural scaffolds can reliably shape emergent instructional strategies in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21184v1">BED-LLM: Intelligent Information Gathering with LLMs and Bayesian Experimental Design</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      We propose a general-purpose approach for improving the ability of Large Language Models (LLMs) to intelligently and adaptively gather information from a user or other external source using the framework of sequential Bayesian experimental design (BED). This enables LLMs to act as effective multi-turn conversational agents and interactively interface with external environments. Our approach, which we call BED-LLM (Bayesian Experimental Design with Large Language Models), is based on iteratively choosing questions or queries that maximize the expected information gain (EIG) about the task of interest given the responses gathered previously. We show how this EIG can be formulated in a principled way using a probabilistic model derived from the LLM's belief distribution and provide detailed insights into key decisions in its construction. Further key to the success of BED-LLM are a number of specific innovations, such as a carefully designed estimator for the EIG, not solely relying on in-context updates for conditioning on previous responses, and a targeted strategy for proposing candidate queries. We find that BED-LLM achieves substantial gains in performance across a wide range of tests based on the 20-questions game and using the LLM to actively infer user preferences, compared to direct prompting of the LLM and other adaptive design strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21141v1">Adaptive LLM Routing under Budget Constraints</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 Accepted at EMNLP 2025 (findings)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized natural language processing, but their varying capabilities and costs pose challenges in practical applications. LLM routing addresses this by dynamically selecting the most suitable LLM for each query/task. Previous approaches treat this as a supervised learning problem, assuming complete knowledge of optimal query-LLM pairings. However, real-world scenarios lack such comprehensive mappings and face evolving user queries. We thus propose to study LLM routing as a contextual bandit problem, enabling adaptive decision-making using bandit feedback without requiring exhaustive inference across all LLMs for all queries (in contrast to supervised routing). To address this problem, we develop a shared embedding space for queries and LLMs, where query and LLM embeddings are aligned to reflect their affinity. This space is initially learned from offline human preference data and refined through online bandit feedback. We instantiate this idea through Preference-prior Informed Linucb fOr adaptive rouTing (PILOT), a novel extension of LinUCB. To handle diverse user budgets for model routing, we introduce an online cost policy modeled as a multi-choice knapsack problem, ensuring resource-efficient routing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.14862v2">Bitune: Leveraging Bidirectional Attention to Improve Decoder-Only LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      Decoder-only large language models typically rely solely on masked causal attention, which limits their expressiveness by restricting information flow to one direction. We propose Bitune, a method that enhances pretrained decoder-only LLMs by incorporating bidirectional attention into prompt processing. We evaluate Bitune in instruction-tuning and question-answering settings, showing significant improvements in performance on commonsense reasoning, arithmetic, and language understanding tasks. Furthermore, extensive ablation studies validate the role of each component of the method, and demonstrate that Bitune is compatible with various parameter-efficient finetuning techniques and full model finetuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20996v1">ChatThero: An LLM-Supported Chatbot for Behavior Change and Therapeutic Support in Addiction Recovery</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      Substance use disorders (SUDs) affect over 36 million people worldwide, yet few receive effective care due to stigma, motivational barriers, and limited personalized support. Although large language models (LLMs) show promise for mental-health assistance, most systems lack tight integration with clinically validated strategies, reducing effectiveness in addiction recovery. We present ChatThero, a multi-agent conversational framework that couples dynamic patient modeling with context-sensitive therapeutic dialogue and adaptive persuasive strategies grounded in cognitive behavioral therapy (CBT) and motivational interviewing (MI). We build a high-fidelity synthetic benchmark spanning Easy, Medium, and Hard resistance levels, and train ChatThero with a two-stage pipeline comprising supervised fine-tuning (SFT) followed by direct preference optimization (DPO). In evaluation, ChatThero yields a 41.5\% average gain in patient motivation, a 0.49\% increase in treatment confidence, and resolves hard cases with 26\% fewer turns than GPT-4o, and both automated and human clinical assessments rate it higher in empathy, responsiveness, and behavioral realism. The framework supports rigorous, privacy-preserving study of therapeutic conversation and provides a robust, replicable basis for research and clinical translation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20912v1">Research Challenges in Relational Database Management Systems for LLM Queries</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 This paper will appear in the 6th International Workshop on Applied AI for Database Systems and Applications, AIDB Workshop at VLDB 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have become essential for applications such as text summarization, sentiment analysis, and automated question-answering. Recently, LLMs have also been integrated into relational database management systems to enhance querying and support advanced data processing. Companies such as Amazon, Databricks, Google, and Snowflake offer LLM invocation directly within SQL, denoted as LLM queries, to boost data insights. However, open-source solutions currently have limited functionality and poor performance. In this work, we present an early exploration of two open-source systems and one enterprise platform, using five representative queries to expose functional, performance, and scalability limits in today's SQL-invoked LLM integrations. We identify three main issues: enforcing structured outputs, optimizing resource utilization, and improving query planning. We implemented initial solutions and observed improvements in accommodating LLM powered SQL queries. These early gains demonstrate that tighter integration of LLM+DBMS is the key to scalable and efficient processing of LLM queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20863v1">Publish to Perish: Prompt Injection Attacks on LLM-Assisted Peer Review</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly being integrated into the scientific peer-review process, raising new questions about their reliability and resilience to manipulation. In this work, we investigate the potential for hidden prompt injection attacks, where authors embed adversarial text within a paper's PDF to influence the LLM-generated review. We begin by formalising three distinct threat models that envision attackers with different motivations -- not all of which implying malicious intent. For each threat model, we design adversarial prompts that remain invisible to human readers yet can steer an LLM's output toward the author's desired outcome. Using a user study with domain scholars, we derive four representative reviewing prompts used to elicit peer reviews from LLMs. We then evaluate the robustness of our adversarial prompts across (i) different reviewing prompts, (ii) different commercial LLM-based systems, and (iii) different peer-reviewed papers. Our results show that adversarial prompts can reliably mislead the LLM, sometimes in ways that adversely affect a "honest-but-lazy" reviewer. Finally, we propose and empirically assess methods to reduce detectability of adversarial prompts under automated content checks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20818v1">cMALC-D: Contextual Multi-Agent LLM-Guided Curriculum Learning with Diversity-Based Context Blending</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 A shorter version has been accepted to the 2025 Conference on Information and Knowledge Management
    </div>
    <details class="paper-abstract">
      Many multi-agent reinforcement learning (MARL) algorithms are trained in fixed simulation environments, making them brittle when deployed in real-world scenarios with more complex and uncertain conditions. Contextual MARL (cMARL) addresses this by parameterizing environments with context variables and training a context-agnostic policy that performs well across all environment configurations. Existing cMARL methods attempt to use curriculum learning to help train and evaluate context-agnostic policies, but they often rely on unreliable proxy signals, such as value estimates or generalized advantage estimates that are noisy and unstable in multi-agent settings due to inter-agent dynamics and partial observability. To address these issues, we propose Contextual Multi-Agent LLM-Guided Curriculum Learning with Diversity-Based Context Blending (cMALC-D), a framework that uses Large Language Models (LLMs) to generate semantically meaningful curricula and provide a more robust evaluation signal. To prevent mode collapse and encourage exploration, we introduce a novel diversity-based context blending mechanism that creates new training scenarios by combining features from prior contexts. Experiments in traffic signal control domains demonstrate that cMALC-D significantly improves both generalization and sample efficiency compared to existing curriculum learning baselines. We provide code at https://github.com/DaRL-LibSignal/cMALC-D.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20810v1">A Graph-Based Test-Harness for LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 4 pages, 2 figures, dataset
    </div>
    <details class="paper-abstract">
      We present a first known prototype of a dynamic, systematic benchmark of medical guidelines for 400+ questions, with 3.3+ trillion possible combinations, covering 100\% of guideline relationships. We transformed the WHO IMCI handbook into a directed graph with 200+ nodes (conditions, symptoms, treatments, follow-ups, severities) and 300+ edges, then used graph traversal to generate questions that incorporated age-specific scenarios and contextual distractors to ensure clinical relevance. Our graph-based approach enables systematic evaluation across clinical tasks (45-67\% accuracy), and we find models excel at symptom recognition but struggle with triaging severity, treatment protocols and follow-up care, demonstrating how customized benchmarks can identify specific capability gaps that general-domain evaluations miss. Beyond evaluation, this dynamic MCQA methodology enhances LLM post-training (supervised finetuning, GRPO, DPO), where correct answers provide high-reward samples without expensive human annotation. The graph-based approach successfully addresses the coverage limitations of manually curated benchmarks. This methodology is a step toward scalable, contamination-resistant solution for creating comprehensive benchmarks that can be dynamically generated, including when the guidelines are updated. Code and datasets are available at https://github.com/jessicalundin/graph_testing_harness
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08846v2">Steering Towards Fairness: Mitigating Political Bias in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 Accepted at CASE@RANLP2025
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have enabled their widespread use across diverse real-world applications. However, concerns remain about their tendency to encode and reproduce ideological biases along political and economic dimensions. In this paper, we employ a framework for probing and mitigating such biases in decoder-based LLMs through analysis of internal model representations. Grounded in the Political Compass Test (PCT), this method uses contrastive pairs to extract and compare hidden layer activations from models like Mistral and DeepSeek. We introduce a comprehensive activation extraction pipeline capable of layer-wise analysis across multiple ideological axes, revealing meaningful disparities linked to political framing. Our results show that decoder LLMs systematically encode representational bias across layers, which can be leveraged for effective steering vector-based mitigation. This work provides new insights into how political bias is encoded in LLMs and offers a principled approach to debiasing beyond surface-level output interventions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14330v3">Leveraging LLMs for Formal Software Requirements -- Challenges and Prospects</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 Overlay2025 - 7th International Workshop on Artificial Intelligence and fOrmal VERification, Logic, Automata, and sYnthesis. [Accepted]. To be held on 26th of October, 2025
    </div>
    <details class="paper-abstract">
      Software correctness is ensured mathematically through formal verification, which involves the resources of generating formal requirement specifications and having an implementation that must be verified. Tools such as model-checkers and theorem provers ensure software correctness by verifying the implementation against the specification. Formal methods deployment is regularly enforced in the development of safety-critical systems e.g. aerospace, medical devices and autonomous systems. Generating these specifications from informal and ambiguous natural language requirements remains the key challenge. Our project, VERIFAI^{1}, aims to investigate automated and semi-automated approaches to bridge this gap, using techniques from Natural Language Processing (NLP), ontology-based domain modelling, artefact reuse, and large language models (LLMs). This position paper presents a preliminary synthesis of relevant literature to identify recurring challenges and prospective research directions in the generation of verifiable specifications from informal requirements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20764v1">Feel the Difference? A Comparative Analysis of Emotional Arcs in Real and LLM-Generated CBT Sessions</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 Accepted at EMNLP 2025,14 page,3 figures
    </div>
    <details class="paper-abstract">
      Synthetic therapy dialogues generated by large language models (LLMs) are increasingly used in mental health NLP to simulate counseling scenarios, train models, and supplement limited real-world data. However, it remains unclear whether these synthetic conversations capture the nuanced emotional dynamics of real therapy. In this work, we conduct the first comparative analysis of emotional arcs between real and LLM-generated Cognitive Behavioral Therapy dialogues. We adapt the Utterance Emotion Dynamics framework to analyze fine-grained affective trajectories across valence, arousal, and dominance dimensions. Our analysis spans both full dialogues and individual speaker roles (counselor and client), using real sessions transcribed from public videos and synthetic dialogues from the CACTUS dataset. We find that while synthetic dialogues are fluent and structurally coherent, they diverge from real conversations in key emotional properties: real sessions exhibit greater emotional variability,more emotion-laden language, and more authentic patterns of reactivity and regulation. Moreover, emotional arc similarity between real and synthetic speakers is low, especially for clients. These findings underscore the limitations of current LLM-generated therapy data and highlight the importance of emotional fidelity in mental health applications. We introduce RealCBT, a curated dataset of real CBT sessions, to support future research in this space.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20750v1">Specializing General-purpose LLM Embeddings for Implicit Hate Speech Detection across Datasets</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 Paper accepted at the DHOW Workshop at ACM Multimedia 2025. Code available at https://github.com/idiap/implicit-hsd
    </div>
    <details class="paper-abstract">
      Implicit hate speech (IHS) is indirect language that conveys prejudice or hatred through subtle cues, sarcasm or coded terminology. IHS is challenging to detect as it does not include explicit derogatory or inflammatory words. To address this challenge, task-specific pipelines can be complemented with external knowledge or additional information such as context, emotions and sentiment data. In this paper, we show that, by solely fine-tuning recent general-purpose embedding models based on large language models (LLMs), such as Stella, Jasper, NV-Embed and E5, we achieve state-of-the-art performance. Experiments on multiple IHS datasets show up to 1.10 percentage points improvements for in-dataset, and up to 20.35 percentage points improvements in cross-dataset evaluation, in terms of F1-macro score.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20744v1">From Law to Gherkin: A Human-Centred Quasi-Experiment on the Quality of LLM-Generated Behavioural Specifications from Food-Safety Regulations</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      Context: Laws and regulations increasingly affect software design and quality assurance, but legal texts are written in technology-neutral language. This creates challenges for engineers who must develop compliance artifacts such as requirements and acceptance criteria. Manual creation is labor-intensive, error-prone, and requires domain expertise. Advances in Generative AI (GenAI), especially Large Language Models (LLMs), offer a way to automate deriving such artifacts. Objective: We present the first systematic human-subject study of LLMs' ability to derive behavioral specifications from legal texts using a quasi-experimental design. These specifications translate legal requirements into a developer-friendly form. Methods: Ten participants evaluated specifications generated from food-safety regulations by Claude and Llama. Using Gherkin, a structured BDD language, 60 specifications were produced. Each participant assessed 12 across five criteria: Relevance, Clarity, Completeness, Singularity, and Time Savings. Each specification was reviewed by two participants, yielding 120 assessments. Results: For Relevance, 75% of ratings were highest and 20% second-highest. Clarity reached 90% highest. Completeness: 75% highest, 19% second. Singularity: 82% highest, 12% second. Time Savings: 68% highest, 24% second. No lowest ratings occurred. Mann-Whitney U tests showed no significant differences across participants or models. Llama slightly outperformed Claude in Clarity, Completeness, and Time Savings, while Claude was stronger in Singularity. Feedback noted hallucinations and omissions but confirmed the utility of the specifications. Conclusion: LLMs can generate high-quality Gherkin specifications from legal texts, reducing manual effort and providing structured artifacts useful for implementation, assurance, and test generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20737v1">Rethinking Testing for LLM Applications: Characteristics, Challenges, and a Lightweight Interaction Protocol</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      Applications of Large Language Models~(LLMs) have evolved from simple text generators into complex software systems that integrate retrieval augmentation, tool invocation, and multi-turn interactions. Their inherent non-determinism, dynamism, and context dependence pose fundamental challenges for quality assurance. This paper decomposes LLM applications into a three-layer architecture: \textbf{\textit{System Shell Layer}}, \textbf{\textit{Prompt Orchestration Layer}}, and \textbf{\textit{LLM Inference Core}}. We then assess the applicability of traditional software testing methods in each layer: directly applicable at the shell layer, requiring semantic reinterpretation at the orchestration layer, and necessitating paradigm shifts at the inference core. A comparative analysis of Testing AI methods from the software engineering community and safety analysis techniques from the AI community reveals structural disconnects in testing unit abstraction, evaluation metrics, and lifecycle management. We identify four fundamental differences that underlie 6 core challenges. To address these, we propose four types of collaborative strategies (\emph{Retain}, \emph{Translate}, \emph{Integrate}, and \emph{Runtime}) and explore a closed-loop, trustworthy quality assurance framework that combines pre-deployment validation with runtime monitoring. Based on these strategies, we offer practical guidance and a protocol proposal to support the standardization and tooling of LLM application testing. We propose a protocol \textbf{\textit{Agent Interaction Communication Language}} (AICL) that is used to communicate between AI agents. AICL has the test-oriented features and is easily integrated in the current agent framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18321v2">LLMs Can't Handle Peer Pressure: Crumbling under Multi-Agent Social Interactions</a></div>
    <div class="paper-meta">
      📅 2025-08-28
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed in multi-agent systems (MAS) as components of collaborative intelligence, where peer interactions dynamically shape individual decision-making. Although prior work has focused on conformity bias, we extend the analysis to examine how LLMs form trust from previous impressions, resist misinformation, and integrate peer input during interaction, key factors for achieving collective intelligence under complex social dynamics. We present KAIROS, a benchmark simulating quiz contests with peer agents of varying reliability, offering fine-grained control over conditions such as expert-novice roles, noisy crowds, and adversarial peers. LLMs receive both historical interactions and current peer responses, allowing systematic investigation into how trust, peer action, and self-confidence influence decisions. As for mitigation strategies, we evaluate prompting, supervised fine-tuning, and reinforcement learning, Group Relative Policy Optimisation (GRPO), across multiple models. Our results reveal that GRPO with multi-agent context combined with outcome-based rewards and unconstrained reasoning achieves the best overall performance, but also decreases the robustness to social influence compared to Base models. The code and datasets are available at: https://github.com/declare-lab/KAIROS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20697v1">Token Buncher: Shielding LLMs from Harmful Reinforcement Learning Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 Project Hompage: https://tokenbuncher.github.io/
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) continue to grow in capability, so do the risks of harmful misuse through fine-tuning. While most prior studies assume that attackers rely on supervised fine-tuning (SFT) for such misuse, we systematically demonstrate that reinforcement learning (RL) enables adversaries to more effectively break safety alignment and facilitate advanced harmful task assistance, under matched computational budgets. To counter this emerging threat, we propose TokenBuncher, the first effective defense specifically targeting RL-based harmful fine-tuning. TokenBuncher suppresses the foundation on which RL relies: model response uncertainty. By constraining uncertainty, RL-based fine-tuning can no longer exploit distinct reward signals to drive the model toward harmful behaviors. We realize this defense through entropy-as-reward RL and a Token Noiser mechanism designed to prevent the escalation of expert-domain harmful capabilities. Extensive experiments across multiple models and RL algorithms show that TokenBuncher robustly mitigates harmful RL fine-tuning while preserving benign task utility and finetunability. Our results highlight that RL-based harmful fine-tuning poses a greater systemic risk than SFT, and that TokenBuncher provides an effective and general defense.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20643v1">CyberSleuth: Autonomous Blue-Team LLM Agent for Web Attack Forensics</a></div>
    <div class="paper-meta">
      📅 2025-08-28
      | 💬 Code: https://github.com/SmartData-Polito/LLM_Agent_Cybersecurity_Forensic
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents are powerful tools for automating complex tasks. In cybersecurity, researchers have primarily explored their use in red-team operations such as vulnerability discovery and penetration tests. Defensive uses for incident response and forensics have received comparatively less attention and remain at an early stage. This work presents a systematic study of LLM-agent design for the forensic investigation of realistic web application attacks. We propose CyberSleuth, an autonomous agent that processes packet-level traces and application logs to identify the targeted service, the exploited vulnerability (CVE), and attack success. We evaluate the consequences of core design decisions - spanning tool integration and agent architecture - and provide interpretable guidance for practitioners. We benchmark four agent architectures and six LLM backends on 20 incident scenarios of increasing complexity, identifying CyberSleuth as the best-performing design. In a separate set of 10 incidents from 2025, CyberSleuth correctly identifies the exact CVE in 80% of cases. At last, we conduct a human study with 22 experts, which rated the reports of CyberSleuth as complete, useful, and coherent. They also expressed a slight preference for DeepSeek R1, a good news for open source LLM. To foster progress in defensive LLM research, we release both our benchmark and the CyberSleuth platform as a foundation for fair, reproducible evaluation of forensic agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08400v3">mSTEB: Massively Multilingual Evaluation of LLMs on Speech and Text Tasks</a></div>
    <div class="paper-meta">
      📅 2025-08-27
      | 💬 Accepted to ASRU 2025
    </div>
    <details class="paper-abstract">
      Large Language models (LLMs) have demonstrated impressive performance on a wide range of tasks, including in multimodal settings such as speech. However, their evaluation is often limited to English and a few high-resource languages. For low-resource languages, there is no standardized evaluation benchmark. In this paper, we address this gap by introducing mSTEB, a new benchmark to evaluate the performance of LLMs on a wide range of tasks covering language identification, text classification, question answering, and translation tasks on both speech and text modalities. We evaluated the performance of leading LLMs such as Gemini 2.0 Flash and GPT-4o (Audio) and state-of-the-art open models such as Qwen 2 Audio and Gemma 3 27B. Our evaluation shows a wide gap in performance between high-resource and low-resource languages, especially for languages spoken in Africa and Americas/Oceania. Our findings show that more investment is needed to address their under-representation in LLMs coverage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.13732v2">Self-Organizing Agent Network for LLM-based Workflow Automation</a></div>
    <div class="paper-meta">
      📅 2025-08-27
    </div>
    <details class="paper-abstract">
      Recent multi-agent frameworks built upon large language models (LLMs) have demonstrated remarkable capabilities in complex task planning. However, in real-world enterprise environments, business workflows are typically composed through modularization and reuse of numerous subprocesses, resulting in intricate workflows characterized by lengthy and deeply nested execution paths. Such complexity poses significant challenges for LLM-driven orchestration, as extended reasoning chains and state-space explosions severely impact planning effectiveness and the proper sequencing of tool invocations. Therefore, developing an orchestration method with controllable structures capable of handling multi-layer nesting becomes a critical issue. To address this, we propose a novel structure-driven orchestration framework Self-Organizing Agent Network (SOAN). SOAN incrementally builds a formalized agent network by identifying and encapsulating structural units as independent agents, enhancing modularity and clarity in orchestration. Extensive evaluations were performed using multiple benchmarks as well as a real-world enterprise workflow dataset. Experimental results demonstrate that SOAN significantly outperforms state-of-the-art methods in terms of adaptability, fault tolerance, and execution efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19919v1">Your AI Bosses Are Still Prejudiced: The Emergence of Stereotypes in LLM-Based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-08-27
    </div>
    <details class="paper-abstract">
      While stereotypes are well-documented in human social interactions, AI systems are often presumed to be less susceptible to such biases. Previous studies have focused on biases inherited from training data, but whether stereotypes can emerge spontaneously in AI agent interactions merits further exploration. Through a novel experimental framework simulating workplace interactions with neutral initial conditions, we investigate the emergence and evolution of stereotypes in LLM-based multi-agent systems. Our findings reveal that (1) LLM-Based AI agents develop stereotype-driven biases in their interactions despite beginning without predefined biases; (2) stereotype effects intensify with increased interaction rounds and decision-making power, particularly after introducing hierarchical structures; (3) these systems exhibit group effects analogous to human social behavior, including halo effects, confirmation bias, and role congruity; and (4) these stereotype patterns manifest consistently across different LLM architectures. Through comprehensive quantitative analysis, these findings suggest that stereotype formation in AI systems may arise as an emergent property of multi-agent interactions, rather than merely from training data biases. Our work underscores the need for future research to explore the underlying mechanisms of this phenomenon and develop strategies to mitigate its ethical impacts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13972v3">Truth or Twist? Optimal Model Selection for Reliable Label Flipping Evaluation in LLM-based Counterfactuals</a></div>
    <div class="paper-meta">
      📅 2025-08-27
      | 💬 Accepted at INLG 2025, camera-ready version
    </div>
    <details class="paper-abstract">
      Counterfactual examples are widely employed to enhance the performance and robustness of large language models (LLMs) through counterfactual data augmentation (CDA). However, the selection of the judge model used to evaluate label flipping, the primary metric for assessing the validity of generated counterfactuals for CDA, yields inconsistent results. To decipher this, we define four types of relationships between the counterfactual generator and judge models: being the same model, belonging to the same model family, being independent models, and having an distillation relationship. Through extensive experiments involving two state-of-the-art LLM-based methods, three datasets, four generator models, and 15 judge models, complemented by a user study (n = 90), we demonstrate that judge models with an independent, non-fine-tuned relationship to the generator model provide the most reliable label flipping evaluations. Relationships between the generator and judge models, which are closely aligned with the user study for CDA, result in better model performance and robustness. Nevertheless, we find that the gap between the most effective judge models and the results obtained from the user study remains considerably large. This suggests that a fully automated pipeline for CDA may be inadequate and requires human intervention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02531v3">Towards New Benchmark for AI Alignment & Sentiment Analysis in Socially Important Issues: A Comparative Study of Human and LLMs in the Context of AGI</a></div>
    <div class="paper-meta">
      📅 2025-08-27
      | 💬 34 pages, 3 figures
    </div>
    <details class="paper-abstract">
      As general-purpose artificial intelligence systems become increasingly integrated into society and are used for information seeking, content generation, problem solving, textual analysis, coding, and running processes, it is crucial to assess their long-term impact on humans. This research explores the sentiment of large language models (LLMs) and humans toward artificial general intelligence (AGI) using a Likert-scale survey. Seven LLMs, including GPT-4 and Bard, were analyzed and compared with sentiment data from three independent human sample populations. Temporal variations in sentiment were also evaluated over three consecutive days. The results show a diversity in sentiment scores among LLMs, ranging from 3.32 to 4.12 out of 5. GPT-4 recorded the most positive sentiment toward AGI, while Bard leaned toward a neutral sentiment. In contrast, the human samples showed a lower average sentiment of 2.97. The analysis outlines potential conflicts of interest and biases in the sentiment formation of LLMs, and indicates that LLMs could subtly influence societal perceptions. To address the need for regulatory oversight and culturally grounded assessments of AI systems, we introduce the Societal AI Alignment and Sentiment Benchmark (SAAS-AI), which leverages multidimensional prompts and empirically validated societal value frameworks to evaluate language model outputs across temporal, model, and multilingual axes. This benchmark is designed to guide policymakers and AI agencies, including within frameworks such as the EU AI Act, by providing robust, actionable insights into AI alignment with human values, public sentiment, and ethical norms at both national and international levels. Future research should further refine the operationalization of the SAAS-AI benchmark and systematically evaluate its effectiveness through comprehensive empirical testing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19887v1">Bangla-Bayanno: A 52K-Pair Bengali Visual Question Answering Dataset with LLM-Assisted Translation Refinement</a></div>
    <div class="paper-meta">
      📅 2025-08-27
    </div>
    <details class="paper-abstract">
      In this paper, we introduce Bangla-Bayanno, an open-ended Visual Question Answering (VQA) Dataset in Bangla, a widely used, low-resource language in multimodal AI research. The majority of existing datasets are either manually annotated with an emphasis on a specific domain, query type, or answer type or are constrained by niche answer formats. In order to mitigate human-induced errors and guarantee lucidity, we implemented a multilingual LLM-assisted translation refinement pipeline. This dataset overcomes the issues of low-quality translations from multilingual sources. The dataset comprises 52,650 question-answer pairs across 4750+ images. Questions are classified into three distinct answer types: nominal (short descriptive), quantitative (numeric), and polar (yes/no). Bangla-Bayanno provides the most comprehensive open-source, high-quality VQA benchmark in Bangla, aiming to advance research in low-resource multimodal learning and facilitate the development of more inclusive AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19870v1">Secure Multi-LLM Agentic AI and Agentification for Edge General Intelligence by Zero-Trust: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-08-27
      | 💬 35 pages
    </div>
    <details class="paper-abstract">
      Agentification serves as a critical enabler of Edge General Intelligence (EGI), transforming massive edge devices into cognitive agents through integrating Large Language Models (LLMs) and perception, reasoning, and acting modules. These agents collaborate across heterogeneous edge infrastructures, forming multi-LLM agentic AI systems that leverage collective intelligence and specialized capabilities to tackle complex, multi-step tasks. However, the collaborative nature of multi-LLM systems introduces critical security vulnerabilities, including insecure inter-LLM communications, expanded attack surfaces, and cross-domain data leakage that traditional perimeter-based security cannot adequately address. To this end, this survey introduces zero-trust security of multi-LLM in EGI, a paradigmatic shift following the ``never trust, always verify'' principle. We begin by systematically analyzing the security risks in multi-LLM systems within EGI contexts. Subsequently, we present the vision of a zero-trust multi-LLM framework in EGI. We then survey key technical progress to facilitate zero-trust multi-LLM systems in EGI. Particularly, we categorize zero-trust security mechanisms into model- and system-level approaches. The former and latter include strong identification, context-aware access control, etc., and proactive maintenance, blockchain-based management, etc., respectively. Finally, we identify critical research directions. This survey serves as the first systematic treatment of zero-trust applied to multi-LLM systems, providing both theoretical foundations and practical strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06394v2">When AIOps Become "AI Oops": Subverting LLM-driven IT Operations via Telemetry Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-08-27
      | 💬 v0.2
    </div>
    <details class="paper-abstract">
      AI for IT Operations (AIOps) is transforming how organizations manage complex software systems by automating anomaly detection, incident diagnosis, and remediation. Modern AIOps solutions increasingly rely on autonomous LLM-based agents to interpret telemetry data and take corrective actions with minimal human intervention, promising faster response times and operational cost savings. In this work, we perform the first security analysis of AIOps solutions, showing that, once again, AI-driven automation comes with a profound security cost. We demonstrate that adversaries can manipulate system telemetry to mislead AIOps agents into taking actions that compromise the integrity of the infrastructure they manage. We introduce techniques to reliably inject telemetry data using error-inducing requests that influence agent behavior through a form of adversarial reward-hacking; plausible but incorrect system error interpretations that steer the agent's decision-making. Our attack methodology, AIOpsDoom, is fully automated--combining reconnaissance, fuzzing, and LLM-driven adversarial input generation--and operates without any prior knowledge of the target system. To counter this threat, we propose AIOpsShield, a defense mechanism that sanitizes telemetry data by exploiting its structured nature and the minimal role of user-generated content. Our experiments show that AIOpsShield reliably blocks telemetry-based attacks without affecting normal agent performance. Ultimately, this work exposes AIOps as an emerging attack vector for system compromise and underscores the urgent need for security-aware AIOps design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.19512v2">Safeguard Fine-Tuned LLMs Through Pre- and Post-Tuning Model Merging</a></div>
    <div class="paper-meta">
      📅 2025-08-27
      | 💬 EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) for downstream tasks often leads to catastrophic forgetting, notably degrading the safety of originally aligned models. While some existing methods attempt to restore safety by incorporating additional safety data, the quality of such data typically falls short of that used in the original alignment process. Moreover, these high-quality safety datasets are generally inaccessible, making it difficult to fully recover the model's original safety. We ask: How can we preserve safety while improving downstream task performance without additional safety data? We show that simply merging the weights of pre- and post-fine-tuned models effectively mitigates safety degradation while enhancing performance. Experiments across different downstream tasks and models validate the method's practicality and effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19831v1">Benchmarking Hindi LLMs: A New Suite of Datasets and a Comparative Analysis</a></div>
    <div class="paper-meta">
      📅 2025-08-27
    </div>
    <details class="paper-abstract">
      Evaluating instruction-tuned Large Language Models (LLMs) in Hindi is challenging due to a lack of high-quality benchmarks, as direct translation of English datasets fails to capture crucial linguistic and cultural nuances. To address this, we introduce a suite of five Hindi LLM evaluation datasets: IFEval-Hi, MT-Bench-Hi, GSM8K-Hi, ChatRAG-Hi, and BFCL-Hi. These were created using a methodology that combines from-scratch human annotation with a translate-and-verify process. We leverage this suite to conduct an extensive benchmarking of open-source LLMs supporting Hindi, providing a detailed comparative analysis of their current capabilities. Our curation process also serves as a replicable methodology for developing benchmarks in other low-resource languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05220v4">Utility-Focused LLM Annotation for Retrieval and Retrieval-Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2025-08-27
      | 💬 Accepted by the EMNLP25 main conference
    </div>
    <details class="paper-abstract">
      This paper explores the use of large language models (LLMs) for annotating document utility in training retrieval and retrieval-augmented generation (RAG) systems, aiming to reduce dependence on costly human annotations. We address the gap between retrieval relevance and generative utility by employing LLMs to annotate document utility. To effectively utilize multiple positive samples per query, we introduce a novel loss that maximizes their summed marginal likelihood. Using the Qwen-2.5-32B model, we annotate utility on the MS MARCO dataset and conduct retrieval experiments on MS MARCO and BEIR, as well as RAG experiments on MS MARCO QA, NQ, and HotpotQA. Our results show that LLM-generated annotations enhance out-of-domain retrieval performance and improve RAG outcomes compared to models trained solely on human annotations or downstream QA metrics. Furthermore, combining LLM annotations with just 20% of human labels achieves performance comparable to using full human annotations. Our study offers a comprehensive approach to utilizing LLM annotations for initializing QA systems on new corpora.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19740v1">Spotlight Attention: Towards Efficient LLM Generation via Non-linear Hashing-based KV Cache Retrieval</a></div>
    <div class="paper-meta">
      📅 2025-08-27
    </div>
    <details class="paper-abstract">
      Reducing the key-value (KV) cache burden in Large Language Models (LLMs) significantly accelerates inference. Dynamically selecting critical KV caches during decoding helps maintain performance. Existing methods use random linear hashing to identify important tokens, but this approach is inefficient due to the orthogonal distribution of queries and keys within two narrow cones in LLMs. We introduce Spotlight Attention, a novel method that employs non-linear hashing functions to optimize the embedding distribution of queries and keys, enhancing coding efficiency and robustness. We also developed a lightweight, stable training framework using a Bradley-Terry ranking-based loss, enabling optimization of the non-linear hashing module on GPUs with 16GB memory in 8 hours. Experimental results show that Spotlight Attention drastically improves retrieval precision while shortening the length of the hash code at least 5$\times$ compared to traditional linear hashing. Finally, we exploit the computational advantages of bitwise operations by implementing specialized CUDA kernels, achieving hashing retrieval for 512K tokens in under 100$\mu$s on a single A100 GPU, with end-to-end throughput up to 3$\times$ higher than vanilla decoding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19720v1">Continuously Steering LLMs Sensitivity to Contextual Knowledge with Proxy Models</a></div>
    <div class="paper-meta">
      📅 2025-08-27
    </div>
    <details class="paper-abstract">
      In Large Language Models (LLMs) generation, there exist knowledge conflicts and scenarios where parametric knowledge contradicts knowledge provided in the context. Previous works studied tuning, decoding algorithms, or locating and editing context-aware neurons to adapt LLMs to be faithful to new contextual knowledge. However, they are usually inefficient or ineffective for large models, not workable for black-box models, or unable to continuously adjust LLMs' sensitivity to the knowledge provided in the context. To mitigate these problems, we propose CSKS (Continuously Steering Knowledge Sensitivity), a simple framework that can steer LLMs' sensitivity to contextual knowledge continuously at a lightweight cost. Specifically, we tune two small LMs (i.e. proxy models) and use the difference in their output distributions to shift the original distribution of an LLM without modifying the LLM weights. In the evaluation process, we not only design synthetic data and fine-grained metrics to measure models' sensitivity to contextual knowledge but also use a real conflict dataset to validate CSKS's practical efficacy. Extensive experiments demonstrate that our framework achieves continuous and precise control over LLMs' sensitivity to contextual knowledge, enabling both increased sensitivity and reduced sensitivity, thereby allowing LLMs to prioritize either contextual or parametric knowledge as needed flexibly. Our data and code are available at https://github.com/OliveJuiceLin/CSKS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19663v1">Leveraging LLMs for Automated Translation of Legacy Code: A Case Study on PL/SQL to Java Transformation</a></div>
    <div class="paper-meta">
      📅 2025-08-27
    </div>
    <details class="paper-abstract">
      The VT legacy system, comprising approximately 2.5 million lines of PL/SQL code, lacks consistent documentation and automated tests, posing significant challenges for refactoring and modernisation. This study investigates the feasibility of leveraging large language models (LLMs) to assist in translating PL/SQL code into Java for the modernised "VTF3" system. By leveraging a dataset comprising 10 PL/SQL-to-Java code pairs and 15 Java classes, which collectively established a domain model for the translated files, multiple LLMs were evaluated. Furthermore, we propose a customized prompting strategy that integrates chain-of-guidance reasoning with $n$-shot prompting. Our findings indicate that this methodology effectively guides LLMs in generating syntactically accurate translations while also achieving functional correctness. However, the findings are limited by the small sample size of available code files and the restricted access to test cases used for validating the correctness of the generated code. Nevertheless, these findings lay the groundwork for scalable, automated solutions in modernising large legacy systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.07132v2">LLM-based feature generation from text for interpretable machine learning</a></div>
    <div class="paper-meta">
      📅 2025-08-27
    </div>
    <details class="paper-abstract">
      Existing text representations such as embeddings and bag-of-words are not suitable for rule learning due to their high dimensionality and absent or questionable feature-level interpretability. This article explores whether large language models (LLMs) could address this by extracting a small number of interpretable features from text. We demonstrate this process on two datasets (CORD-19 and M17+) containing several thousand scientific articles from multiple disciplines and a target being a proxy for research impact. An evaluation based on testing for the statistically significant correlation with research impact has shown that LLama 2-generated features are semantically meaningful. We consequently used these generated features in text classification to predict the binary target variable representing the citation rate for the CORD-19 dataset and the ordinal 5-class target representing an expert-awarded grade in the M17+ dataset. Machine-learning models trained on the LLM-generated features provided similar predictive performance to the state-of-the-art embedding model SciBERT for scientific text. The LLM used only 62 features compared to 768 features in SciBERT embeddings, and these features were directly interpretable, corresponding to notions such as article methodological rigor, novelty, or grammatical correctness. As the final step, we extract a small number of well-interpretable action rules. Consistently competitive results obtained with the same LLM feature set across both thematically diverse datasets show that this approach generalizes across domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15495v2">SynthCoder: A Synthetical Strategy to Tune LLMs for Code Completion</a></div>
    <div class="paper-meta">
      📅 2025-08-27
    </div>
    <details class="paper-abstract">
      Code completion is a prominent application of Large Language Models (LLMs) in software engineering. Due to the near real-time response requirements of this task, base models with small to medium-sized parameters are typically employed, supplemented by various optimization and post-training techniques. However, these optimization methods often have trade-offs, leading to a seesaw effect where performance improvements on certain datasets or metrics are accompanied by degradations on others -- sometimes even falling below the baseline model's performance. This paper proposes SynthCoder, a model that integrates leading industry practices to achieve state-of-the-art performance on the Fill-in-the-Middle (FIM) code completion task. In specific, we first construct a diverse dataset by combining Abstract Syntax Tree (AST) node extraction with heuristics that simulate developer behavior. Then we enrich our training corpus with cross-file contextual information using the BM25 algorithm and call graphs, enhancing the model's ability to perform code completion in both file-level and repository-level scenarios. As the last step, we employ a two-stage training process using the Seed-Coder-8B-Base as the base model. First, we fine-tune the model using Curriculum Learning technology. Following this, we perform alignment using Direct Preference Optimization (DPO) with preference pairs generated through Rejection Sampling. Experimental results demonstrate that our final model excels on mainstream repository-level code completion benchmarks, including aiXcoder, ExecRepoBench, CrossCodeEval, and CoLT. Furthermore, our carefully curated training set effectively mitigates the model's tendency to just repeat existing code, a common issue existing in various code completion models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14496v2">Semantic Energy: Detecting LLM Hallucination Beyond Entropy</a></div>
    <div class="paper-meta">
      📅 2025-08-27
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are being increasingly deployed in real-world applications, but they remain susceptible to hallucinations, which produce fluent yet incorrect responses and lead to erroneous decision-making. Uncertainty estimation is a feasible approach to detect such hallucinations. For example, semantic entropy estimates uncertainty by considering the semantic diversity across multiple sampled responses, thus identifying hallucinations. However, semantic entropy relies on post-softmax probabilities and fails to capture the model's inherent uncertainty, causing it to be ineffective in certain scenarios. To address this issue, we introduce Semantic Energy, a novel uncertainty estimation framework that leverages the inherent confidence of LLMs by operating directly on logits of penultimate layer. By combining semantic clustering with a Boltzmann-inspired energy distribution, our method better captures uncertainty in cases where semantic entropy fails. Experiments across multiple benchmarks show that Semantic Energy significantly improves hallucination detection and uncertainty estimation, offering more reliable signals for downstream applications such as hallucination detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19611v1">Instructional Agents: LLM Agents on Automated Course Material Generation for Teaching Faculties</a></div>
    <div class="paper-meta">
      📅 2025-08-27
      | 💬 18 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Preparing high-quality instructional materials remains a labor-intensive process that often requires extensive coordination among teaching faculty, instructional designers, and teaching assistants. In this work, we present Instructional Agents, a multi-agent large language model (LLM) framework designed to automate end-to-end course material generation, including syllabus creation, lecture scripts, LaTeX-based slides, and assessments. Unlike existing AI-assisted educational tools that focus on isolated tasks, Instructional Agents simulates role-based collaboration among educational agents to produce cohesive and pedagogically aligned content. The system operates in four modes: Autonomous, Catalog-Guided, Feedback-Guided, and Full Co-Pilot mode, enabling flexible control over the degree of human involvement. We evaluate Instructional Agents across five university-level computer science courses and show that it produces high-quality instructional materials while significantly reducing development time and human workload. By supporting institutions with limited instructional design capacity, Instructional Agents provides a scalable and cost-effective framework to democratize access to high-quality education, particularly in underserved or resource-constrained settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13426v3">PromptKeeper: Safeguarding System Prompts for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-27
      | 💬 Accepted to the Findings of EMNLP 2025. 17 pages, 6 figures, 3 tables
    </div>
    <details class="paper-abstract">
      System prompts are widely used to guide the outputs of large language models (LLMs). These prompts often contain business logic and sensitive information, making their protection essential. However, adversarial and even regular user queries can exploit LLM vulnerabilities to expose these hidden prompts. To address this issue, we propose PromptKeeper, a defense mechanism designed to safeguard system prompts by tackling two core challenges: reliably detecting leakage and mitigating side-channel vulnerabilities when leakage occurs. By framing detection as a hypothesis-testing problem, PromptKeeper effectively identifies both explicit and subtle leakage. Upon leakage detected, it regenerates responses using a dummy prompt, ensuring that outputs remain indistinguishable from typical interactions when no leakage is present. PromptKeeper ensures robust protection against prompt extraction attacks via either adversarial or regular queries, while preserving conversational capability and runtime efficiency during benign user interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19598v1">Encouraging Good Processes Without the Need for Good Answers: Reinforcement Learning for LLM Agent Planning</a></div>
    <div class="paper-meta">
      📅 2025-08-27
    </div>
    <details class="paper-abstract">
      The functionality of Large Language Model (LLM) agents is primarily determined by two capabilities: action planning and answer summarization. The former, action planning, is the core capability that dictates an agent's performance. However, prevailing training paradigms employ end-to-end, multi-objective optimization that jointly trains both capabilities. This paradigm faces two critical challenges: imbalanced optimization objective allocation and scarcity of verifiable data, making it difficult to enhance the agent's planning capability. To address these challenges, we propose Reinforcement Learning with Tool-use Rewards (RLTR), a novel framework that decouples the training process to enable a focused, single-objective optimization of the planning module. Crucially, RLTR introduces a reward signal based on tool-use completeness to directly evaluate the quality of tool invocation sequences. This method offers a more direct and reliable training signal than assessing the final response content, thereby obviating the need for verifiable data. Our experiments demonstrate that RLTR achieves an 8%-12% improvement in planning performance compared to end-to-end baselines. Moreover, this enhanced planning capability, in turn, translates to a 5%-6% increase in the final response quality of the overall agent system.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21087v1">Can LLMs Generate Behaviors for Embodied Virtual Agents Based on Personality Traits?</a></div>
    <div class="paper-meta">
      📅 2025-08-27
    </div>
    <details class="paper-abstract">
      This study proposes a framework that employs personality prompting with Large Language Models to generate verbal and nonverbal behaviors for virtual agents based on personality traits. Focusing on extraversion, we evaluated the system in two scenarios: negotiation and ice breaking, using both introverted and extroverted agents. In Experiment 1, we conducted agent to agent simulations and performed linguistic analysis and personality classification to assess whether the LLM generated language reflected the intended traits and whether the corresponding nonverbal behaviors varied by personality. In Experiment 2, we carried out a user study to evaluate whether these personality aligned behaviors were consistent with their intended traits and perceptible to human observers. Our results show that LLMs can generate verbal and nonverbal behaviors that align with personality traits, and that users are able to recognize these traits through the agents' behaviors. This work underscores the potential of LLMs in shaping personality aligned virtual agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19580v1">ArgCMV: An Argument Summarization Benchmark for the LLM-era</a></div>
    <div class="paper-meta">
      📅 2025-08-27
    </div>
    <details class="paper-abstract">
      Key point extraction is an important task in argument summarization which involves extracting high-level short summaries from arguments. Existing approaches for KP extraction have been mostly evaluated on the popular ArgKP21 dataset. In this paper, we highlight some of the major limitations of the ArgKP21 dataset and demonstrate the need for new benchmarks that are more representative of actual human conversations. Using SoTA large language models (LLMs), we curate a new argument key point extraction dataset called ArgCMV comprising of around 12K arguments from actual online human debates spread across over 3K topics. Our dataset exhibits higher complexity such as longer, co-referencing arguments, higher presence of subjective discourse units, and a larger range of topics over ArgKP21. We show that existing methods do not adapt well to ArgCMV and provide extensive benchmark results by experimenting with existing baselines and latest open source models. This work introduces a novel KP extraction dataset for long-context online discussions, setting the stage for the next generation of LLM-driven summarization research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19578v1">Towards a Holistic and Automated Evaluation Framework for Multi-Level Comprehension of LLMs in Book-Length Contexts</a></div>
    <div class="paper-meta">
      📅 2025-08-27
      | 💬 Accepted to EMNLP 2025 (Main)
    </div>
    <details class="paper-abstract">
      We introduce HAMLET, a holistic and automated framework for evaluating the long-context comprehension of large language models (LLMs). HAMLET structures source texts into a three-level key-fact hierarchy at root-, branch-, and leaf-levels, and employs query-focused summarization to evaluate how well models recall and faithfully represent information at each level. To validate the reliability of our fully automated pipeline, we conduct a systematic human study, showing that our automatic evaluation achieves over 90% agreement with expert human judgments, while reducing the cost by up to 25 times. HAMLET reveals that LLMs struggle with fine-grained comprehension, especially at the leaf level, and are sensitive to positional effects like the lost-in-the-middle. Analytical queries pose greater challenges than narrative ones, and consistent performance gaps emerge between open-source and proprietary models, as well as across model scales. Our code and dataset are publicly available at https://github.com/DISL-Lab/HAMLET.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19576v1">ReST-RL: Achieving Accurate Code Reasoning of LLMs with Optimized Self-Training and Decoding</a></div>
    <div class="paper-meta">
      📅 2025-08-27
      | 💬 20 pages, 4 figures
    </div>
    <details class="paper-abstract">
      With respect to improving the reasoning accuracy of LLMs, the representative reinforcement learning (RL) method GRPO faces failure due to insignificant reward variance, while verification methods based on process reward models (PRMs) suffer from difficulties with training data acquisition and verification effectiveness. To tackle these problems, this paper introduces ReST-RL, a unified LLM RL paradigm that significantly improves LLM's code reasoning ability by combining an improved GRPO algorithm with a meticulously designed test time decoding method assisted by a value model (VM). As the first stage of policy reinforcement, ReST-GRPO adopts an optimized ReST algorithm to filter and assemble high-value training data, increasing the reward variance of GRPO sampling, thus improving the effectiveness and efficiency of training. After the basic reasoning ability of LLM policy has been improved, we further propose a test time decoding optimization method called VM-MCTS. Through Monte-Carlo Tree Search (MCTS), we collect accurate value targets with no annotation required, on which VM training is based. When decoding, the VM is deployed by an adapted MCTS algorithm to provide precise process signals as well as verification scores, assisting the LLM policy to achieve high reasoning accuracy. We validate the effectiveness of the proposed RL paradigm through extensive experiments on coding problems. Upon comparison, our approach significantly outperforms other reinforcement training baselines (e.g., naive GRPO and ReST-DPO), as well as decoding and verification baselines (e.g., PRM-BoN and ORM-MCTS) on well-known coding benchmarks of various levels (e.g., APPS, BigCodeBench, and HumanEval), indicating its power to strengthen the reasoning ability of LLM policies. Codes for our project can be found at https://github.com/THUDM/ReST-RL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19563v1">Just Because You Can, Doesn't Mean You Should: LLMs for Data Fitting</a></div>
    <div class="paper-meta">
      📅 2025-08-27
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are being applied in a wide array of settings, well beyond the typical language-oriented use cases. In particular, LLMs are increasingly used as a plug-and-play method for fitting data and generating predictions. Prior work has shown that LLMs, via in-context learning or supervised fine-tuning, can perform competitively with many tabular supervised learning techniques in terms of predictive performance. However, we identify a critical vulnerability of using LLMs for data fitting -- making changes to data representation that are completely irrelevant to the underlying learning task can drastically alter LLMs' predictions on the same data. For example, simply changing variable names can sway the size of prediction error by as much as 82% in certain settings. Such prediction sensitivity with respect to task-irrelevant variations manifests under both in-context learning and supervised fine-tuning, for both close-weight and open-weight general-purpose LLMs. Moreover, by examining the attention scores of an open-weight LLM, we discover a non-uniform attention pattern: training examples and variable names/values which happen to occupy certain positions in the prompt receive more attention when output tokens are generated, even though different positions are expected to receive roughly the same attention. This partially explains the sensitivity in the presence of task-irrelevant variations. We also consider a state-of-the-art tabular foundation model (TabPFN) trained specifically for data fitting. Despite being explicitly designed to achieve prediction robustness, TabPFN is still not immune to task-irrelevant variations. Overall, despite LLMs' impressive predictive capabilities, currently they lack even the basic level of robustness to be used as a principled data-fitting tool.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19559v1">Taming the Chaos: Coordinated Autoscaling for Heterogeneous and Disaggregated LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-08-27
    </div>
    <details class="paper-abstract">
      Serving Large Language Models (LLMs) is a GPU-intensive task where traditional autoscalers fall short, particularly for modern Prefill-Decode (P/D) disaggregated architectures. This architectural shift, while powerful, introduces significant operational challenges, including inefficient use of heterogeneous hardware, network bottlenecks, and critical imbalances between prefill and decode stages. We introduce HeteroScale, a coordinated autoscaling framework that addresses the core challenges of P/D disaggregated serving. HeteroScale combines a topology-aware scheduler that adapts to heterogeneous hardware and network constraints with a novel metric-driven policy derived from the first large-scale empirical study of autoscaling signals in production. By leveraging a single, robust metric to jointly scale prefill and decode pools, HeteroScale maintains architectural balance while ensuring efficient, adaptive resource management. Deployed in a massive production environment on tens of thousands of GPUs, HeteroScale has proven its effectiveness, increasing average GPU utilization by a significant 26.6 percentage points and saving hundreds of thousands of GPU-hours daily, all while upholding stringent service level objectives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19558v1">Functional Consistency of LLM Code Embeddings: A Self-Evolving Data Synthesis Framework for Benchmarking</a></div>
    <div class="paper-meta">
      📅 2025-08-27
    </div>
    <details class="paper-abstract">
      Embedding models have demonstrated strong performance in tasks like clustering, retrieval, and feature extraction while offering computational advantages over generative models and cross-encoders. Benchmarks such as MTEB have shown that text embeddings from large language models (LLMs) capture rich semantic information, but their ability to reflect code-level functional semantics remains unclear. Existing studies largely focus on code clone detection, which emphasizes syntactic similarity and overlooks functional understanding. In this paper, we focus on the functional consistency of LLM code embeddings, which determines if two code snippets perform the same function regardless of syntactic differences. We propose a novel data synthesis framework called Functionality-Oriented Code Self-Evolution to construct diverse and challenging benchmarks. Specifically, we define code examples across four semantic and syntactic categories and find that existing datasets predominantly capture syntactic properties. Our framework generates four unique variations from a single code instance, providing a broader spectrum of code examples that better reflect functional differences. Extensive experiments on three downstream tasks-code clone detection, code functional consistency identification, and code retrieval-demonstrate that embedding models significantly improve their performance when trained on our evolved datasets. These results highlight the effectiveness and generalization of our data synthesis framework, advancing the functional understanding of code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18721v2">LLM as an Execution Estimator: Recovering Missing Dependency for Practical Time-travelling Debugging</a></div>
    <div class="paper-meta">
      📅 2025-08-27
    </div>
    <details class="paper-abstract">
      In this work, we propose RecovSlicing for computing dynamic data dependency in a single run, with only partial instrumentation. We explore the intuition that LLM can potentially infer program dynamics based on a partially recorded trace and relevant code as its context. Given (1) a partially recorded trace of a program P and (2) the slicing criteria consisting of a query step s and a query variable v read by s, RecovSlicing computes the runtime definition of v on the trace by estimating the miss-recorded execution of P. In this work, we allow the user to specify implicit query variable, for example, the implicit library variable used in list.get(i). Technically, built upon non-deterministic LLM, we address the challenges of (1) precise recovery of runtime variable value and structure from the recorded execution and (2) aligning the memory address of recovered variables and the recorded variables for definition analysis. We extensively evaluate RecovSlicing against the state-of-the-art slicers such as Slicer4J, ND-Slicer, LLM Slicer, and re-execution Slicer on a total number of 8300 data-dependencies over 3 slicing benchmarks. The results show that RecovSlicing can significantly outperform the baselines. The accuracy and recall, achieving 80.3%, 91.1%, and 98.3% on the three benchmarks, whereas the best baseline reaches 39.0%, 82.0%, and 59.9% (accuracy), and 53.4%, 79.1%, and 87.1% (recall), respectively. In addition, we integrate RecovSlicing in a dual-slicing based regression bug localizer, significantly improving its performance by locating 16% more regressions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05004v2">R-Zero: Self-Evolving Reasoning LLM from Zero Data</a></div>
    <div class="paper-meta">
      📅 2025-08-27
    </div>
    <details class="paper-abstract">
      Self-evolving Large Language Models (LLMs) offer a scalable path toward super-intelligence by autonomously generating, refining, and learning from their own experiences. However, existing methods for training such models still rely heavily on vast human-curated tasks and labels, typically via fine-tuning or reinforcement learning, which poses a fundamental bottleneck to advancing AI systems toward capabilities beyond human intelligence. To overcome this limitation, we introduce R-Zero, a fully autonomous framework that generates its own training data from scratch. Starting from a single base LLM, R-Zero initializes two independent models with distinct roles, a Challenger and a Solver. These models are optimized separately and co-evolve through interaction: the Challenger is rewarded for proposing tasks near the edge of the Solver capability, and the Solver is rewarded for solving increasingly challenging tasks posed by the Challenger. This process yields a targeted, self-improving curriculum without any pre-existing tasks and labels. Empirically, R-Zero substantially improves reasoning capability across different backbone LLMs, e.g., boosting the Qwen3-4B-Base by +6.49 on math-reasoning benchmarks and +7.54 on general-domain reasoning benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08292v2">Putnam-AXIOM: A Functional and Static Benchmark for Measuring Higher Level Mathematical Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-27
      | 💬 27 pages total (10-page main paper + 17-page appendix), 12 figures, 6 tables. Submitted to ICML 2025 (under review)
    </div>
    <details class="paper-abstract">
      Current mathematical reasoning benchmarks for large language models (LLMs) are approaching saturation, with some achieving > 90% accuracy, and are increasingly compromised by training-set contamination. We introduce Putnam-AXIOM, a benchmark of 522 university-level competition problems drawn from the prestigious William Lowell Putnam Mathematical Competition, and Putnam-AXIOM Variation, an unseen companion set of 100 functional variants generated by programmatically perturbing variables and constants. The variation protocol produces an unlimited stream of equally difficult, unseen instances -- yielding a contamination-resilient test bed. On the Original set, OpenAI's o1-preview -- the strongest evaluated model -- scores 41.9%, but its accuracy drops by 19.6% (46.8% relative decrease) on the paired Variations. The remaining eighteen models show the same downward trend, ten of them with non-overlapping 95% confidence intervals. These gaps suggest memorization and highlight the necessity of dynamic benchmarks. We complement "boxed" accuracy with Teacher-Forced Accuracy (TFA), a lightweight metric that directly scores reasoning traces and automates natural language proof evaluations. Putnam-AXIOM therefore provides a rigorous, contamination-resilient evaluation framework for assessing advanced mathematical reasoning of LLMs. Data and evaluation code are publicly available at https://github.com/brando90/putnam-axiom.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19504v1">Aegis: Taxonomy and Optimizations for Overcoming Agent-Environment Failures in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-08-27
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) agents augmented with domain tools promise to autonomously execute complex tasks requiring human-level intelligence, such as customer service and digital assistance. However, their practical deployment is often limited by their low success rates under complex real-world environments. To tackle this, prior research has primarily focused on improving the agents themselves, such as developing strong agentic LLMs, while overlooking the role of the system environment in which the agent operates. In this paper, we study a complementary direction: improving agent success rates by optimizing the system environment in which the agent operates. We collect 142 agent traces (3,656 turns of agent-environment interactions) across 5 state-of-the-art agentic benchmarks. By analyzing these agent failures, we propose a taxonomy for agent-environment interaction failures that includes 6 failure modes. Guided by these findings, we design Aegis, a set of targeted environment optimizations: 1) environment observability enhancement, 2) common computation offloading, and 3) speculative agentic actions. These techniques improve agent success rates on average by 6.7-12.5%, without any modifications to the agent and underlying LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19484v1">Rule Synergy Analysis using LLMs: State of the Art and Implications</a></div>
    <div class="paper-meta">
      📅 2025-08-27
      | 💬 Submitted for publication at the IEEE Transactions on Games 2024, Special Issue on Large Language Models and Games (10 pages excluding appendix, 3 figures)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated strong performance across a variety of domains, including logical reasoning, mathematics, and more. In this paper, we investigate how well LLMs understand and reason about complex rule interactions in dynamic environments, such as card games. We introduce a dataset of card synergies from the game Slay the Spire, where pairs of cards are classified based on their positive, negative, or neutral interactions. Our evaluation shows that while LLMs excel at identifying non-synergistic pairs, they struggle with detecting positive and, particularly, negative synergies. We categorize common error types, including issues with timing, defining game states, and following game rules. Our findings suggest directions for future research to improve model performance in predicting the effect of rules and their interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02943v5">Hallucination to Consensus: Multi-Agent LLMs for End-to-End Test Generation</a></div>
    <div class="paper-meta">
      📅 2025-08-27
    </div>
    <details class="paper-abstract">
      Unit testing plays a critical role in ensuring software correctness. However, writing unit tests manually is labor-intensive, especially for strongly typed languages like Java, motivating the need for automated approaches. Traditional methods primarily rely on search-based or randomized algorithms to achieve high code coverage and produce regression oracles, which are derived from the program's current behavior rather than its intended functionality. Recent advances in LLMs have enabled oracle generation from natural language descriptions, aligning better with user requirements. However, existing LLM-based methods often require fine-tuning or rely on external tools such as EvoSuite for test prefix generation, making them costly or cumbersome to apply in practice. In this work, we propose CANDOR, a novel prompt engineering-based LLM framework for automated unit test generation in Java. CANDOR orchestrates multiple specialized LLM agents to collaboratively generate complete tests. To mitigate the notorious hallucinations in LLMs and improve oracle correctness, we introduce a novel strategy that engages multiple reasoning LLMs in a panel discussion and generates accurate oracles based on consensus. Additionally, to reduce the verbosity of reasoning LLMs' outputs, we propose a novel dual-LLM pipeline to produce concise and structured oracle evaluations. Our experiments show that CANDOR is comparable with EvoSuite in generating tests with high code coverage and clearly superior in terms of mutation score. Moreover, our prompt engineering-based approach CANDOR significantly outperforms the SOTA fine-tuning-based oracle generator TOGLL by at least 21.1 percentage points in oracle correctness on both correct and faulty source code. Further ablation studies confirm the critical contributions of key agents in generating high-quality tests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18076v2">Neither Valid nor Reliable? Investigating the Use of LLMs as Judges</a></div>
    <div class="paper-meta">
      📅 2025-08-27
      | 💬 Prepared for conference submission
    </div>
    <details class="paper-abstract">
      Evaluating natural language generation (NLG) systems remains a core challenge of natural language processing (NLP), further complicated by the rise of large language models (LLMs) that aims to be general-purpose. Recently, large language models as judges (LLJs) have emerged as a promising alternative to traditional metrics, but their validity remains underexplored. This position paper argues that the current enthusiasm around LLJs may be premature, as their adoption has outpaced rigorous scrutiny of their reliability and validity as evaluators. Drawing on measurement theory from the social sciences, we identify and critically assess four core assumptions underlying the use of LLJs: their ability to act as proxies for human judgment, their capabilities as evaluators, their scalability, and their cost-effectiveness. We examine how each of these assumptions may be challenged by the inherent limitations of LLMs, LLJs, or current practices in NLG evaluation. To ground our analysis, we explore three applications of LLJs: text summarization, data annotation, and safety alignment. Finally, we highlight the need for more responsible evaluation practices in LLJs evaluation, to ensure that their growing role in the field supports, rather than undermines, progress in NLG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20279v1">How Multimodal LLMs Solve Image Tasks: A Lens on Visual Grounding, Task Reasoning, and Answer Decoding</a></div>
    <div class="paper-meta">
      📅 2025-08-27
      | 💬 Accepted by COLM 2025
    </div>
    <details class="paper-abstract">
      Multimodal Large Language Models (MLLMs) have demonstrated strong performance across a wide range of vision-language tasks, yet their internal processing dynamics remain underexplored. In this work, we introduce a probing framework to systematically analyze how MLLMs process visual and textual inputs across layers. We train linear classifiers to predict fine-grained visual categories (e.g., dog breeds) from token embeddings extracted at each layer, using a standardized anchor question. To uncover the functional roles of different layers, we evaluate these probes under three types of controlled prompt variations: (1) lexical variants that test sensitivity to surface-level changes, (2) semantic negation variants that flip the expected answer by modifying the visual concept in the prompt, and (3) output format variants that preserve reasoning but alter the answer format. Applying our framework to LLaVA-1.5, LLaVA-Next-LLaMA-3, and Qwen2-VL, we identify a consistent stage-wise structure in which early layers perform visual grounding, middle layers support lexical integration and semantic reasoning, and final layers prepare task-specific outputs. We further show that while the overall stage-wise structure remains stable across variations in visual tokenization, instruction tuning data, and pretraining corpus, the specific layer allocation to each stage shifts notably with changes in the base LLM architecture. Our findings provide a unified perspective on the layer-wise organization of MLLMs and offer a lightweight, model-agnostic approach for analyzing multimodal representation dynamics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20274v1">Predictable LLM Serving on GPU Clusters</a></div>
    <div class="paper-meta">
      📅 2025-08-27
    </div>
    <details class="paper-abstract">
      Latency-sensitive inference on shared A100 clusters often suffers noisy-neighbor interference on the PCIe fabric, inflating tail latency and SLO violations. We present a fabric-agnostic, VM-deployable host-level controller that combines dynamic Multi-Instance GPU (MIG) reconfiguration, PCIe-aware placement, and lightweight guardrails (MPS quotas, cgroup I/O). It samples per-tenant tails and system signals, uses topology hints to avoid PCIe hot spots, and gates actions with dwell/cool-down to avoid thrash. On a single host and a 2-node (16-GPU) cluster, SLO miss-rate is reduced by \(\approx\)32\% (\(\approx\)1.5) and p99 latency improves \(\approx\)15\% with \(\leq\)5\% throughput cost versus static MIG and naive placement; ablations show MIG and placement contribute comparably. We also evaluate LLM serving with vLLM on OLMo 2 7B Instruct: TTFT p99 improves \(\approx\)10--15\% at \(\leq\)5\% cost without changing the controller.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20263v1">Athena: Intermediate Representations for Iterative Scaffolded App Generation with an LLM</a></div>
    <div class="paper-meta">
      📅 2025-08-27
    </div>
    <details class="paper-abstract">
      It is challenging to generate the code for a complete user interface using a Large Language Model (LLM). User interfaces are complex and their implementations often consist of multiple, inter-related files that together specify the contents of each screen, the navigation flows between the screens, and the data model used throughout the application. It is challenging to craft a single prompt for an LLM that contains enough detail to generate a complete user interface, and even then the result is frequently a single large and difficult to understand file that contains all of the generated screens. In this paper, we introduce Athena, a prototype application generation environment that demonstrates how the use of shared intermediate representations, including an app storyboard, data model, and GUI skeletons, can help a developer work with an LLM in an iterative fashion to craft a complete user interface. These intermediate representations also scaffold the LLM's code generation process, producing organized and structured code in multiple files while limiting errors. We evaluated Athena with a user study that found 75% of participants preferred our prototype over a typical chatbot-style baseline for prototyping apps.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.10659v5">Network Formation and Dynamics Among Multi-LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-27
    </div>
    <details class="paper-abstract">
      Social networks profoundly influence how humans form opinions, exchange information, and organize collectively. As large language models (LLMs) are increasingly embedded into social and professional environments, it is critical to understand whether their interactions approximate human-like network dynamics. We develop a framework to study the network formation behaviors of multiple LLM agents and benchmark them against human decisions. Across synthetic and real-world settings, including friendship, telecommunication, and employment networks, we find that LLMs consistently reproduce fundamental micro-level principles such as preferential attachment, triadic closure, and homophily, as well as macro-level properties including community structure and small-world effects. Importantly, the relative emphasis of these principles adapts to context: for example, LLMs favor homophily in friendship networks but heterophily in organizational settings, mirroring patterns of social mobility. A controlled human-subject survey confirms strong alignment between LLMs and human participants in link-formation decisions. These results establish that LLMs can serve as powerful tools for social simulation and synthetic data generation, while also raising critical questions about bias, fairness, and the design of AI systems that participate in human networks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20258v1">SwizzlePerf: Hardware-Aware LLMs for GPU Kernel Performance Optimization</a></div>
    <div class="paper-meta">
      📅 2025-08-27
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown progress in GPU kernel performance engineering using inefficient search-based methods that optimize around runtime. Any existing approach lacks a key characteristic that human performance engineers rely on for near-optimal utilization -- hardware-awareness. By leveraging the workload's specific memory access patterns, architecture specifications, filtered profiling logs, and reflections on historical performance, we can make software-level optimizations that are tailored to the underlying hardware. SwizzlePerf automatically generates spatial optimizations for GPU kernels on disaggregated architectures by giving LLMs explicit hardware-awareness. For a GEMM kernel, SwizzlePerf takes less than 5 minutes to generate the same hardware-specific optimal swizzling pattern that took expert performance engineers 2 weeks to find. On a suite of 10 diverse ML and Science kernels, SwizzlePerf can generate swizzling patterns for 9 of the kernels that achieve up to a 2.06x speedup and 70% improvement in L2 hit rate. This work is the first of many steps toward systematically creating hardware-aware LLM performance engineering agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20151v1">IntentionReasoner: Facilitating Adaptive LLM Safeguards through Intent Reasoning and Selective Query Refinement</a></div>
    <div class="paper-meta">
      📅 2025-08-27
      | 💬 17 pages, 9 figures
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has driven their adoption across diverse domains, yet their ability to generate harmful content poses significant safety challenges. While extensive research has focused on mitigating harmful outputs, such efforts often come at the cost of excessively rejecting harmless prompts. Striking a balance among safety, over-refusal, and utility remains a critical challenge. In this work, we introduce IntentionReasoner, a novel safeguard mechanism that leverages a dedicated guard model to perform intent reasoning, multi-level safety classification, and query rewriting to neutralize potentially harmful intent in edge-case queries. Specifically, we first construct a comprehensive dataset comprising approximately 163,000 queries, each annotated with intent reasoning, safety labels, and rewritten versions. Supervised fine-tuning is then applied to equip the guard model with foundational capabilities in format adherence, intent analysis, and safe rewriting. Finally, we apply a tailored multi-reward optimization strategy that integrates rule-based heuristics and reward model signals within a reinforcement learning framework to further enhance performance. Extensive experiments show that IntentionReasoner excels in multiple safeguard benchmarks, generation quality evaluations, and jailbreak attack scenarios, significantly enhancing safety while effectively reducing over-refusal rates and improving the quality of responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21094v1">Video-LLMs with Temporal Visual Screening</a></div>
    <div class="paper-meta">
      📅 2025-08-27
    </div>
    <details class="paper-abstract">
      Humans naturally perform temporal screening by dragging the progress bar and focusing on salient temporal segments, but current Video Large Language Models (Video-LLMs) struggle to capture fine-grained temporal semantics due to sparse frame sampling and insufficient inter-frame reasoning supervision during their training. To address this, Inspired by well-established cognitive science principles, we propose Temporal Visual Screening (TVS), a new task that universally pre-processes video question answering and instruction tuning data by: (1) retaining focus-critical video segments, (2) synchronously reconstructing queries to their most direct form while preserving answer consistency, and (3) keeping the invariance and consistency for any possible answer. TVS is formulated as a modular front-end adapter task that can be seamlessly integrated into both Video Instruction Tuning (training) and Video Question Answering (inference) pipelines. TVS optimizes distribution of reasoning burden and cognitive load; during training, it aligns queries with focus-critical visual information; at inference, it enables query-aware segment focus and streamlined query representations. In particular, we curate the first benchmark for TVS and propose ReSimplifyIt, a baseline outperforming prior approaches on seemingly similar tasks by 0.47 in F-1 score on video trimming while achieving competitive query rewriting performance. Experiments demonstrate that incorporating TVS yields relative gains of 7.33% (training) and 34.6% (inference), demonstrating the effectiveness of temporal information screening for improving video-language understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20038v1">Forewarned is Forearmed: Pre-Synthesizing Jailbreak-like Instructions to Enhance LLM Safety Guardrail to Potential Attacks</a></div>
    <div class="paper-meta">
      📅 2025-08-27
      | 💬 EMNLP 2025 findings
    </div>
    <details class="paper-abstract">
      Despite advances in improving large language model(LLM) to refuse to answer malicious instructions, widely used LLMs remain vulnerable to jailbreak attacks where attackers generate instructions with distributions differing from safety alignment corpora. New attacks expose LLMs' inability to recognize unseen malicious instructions, highlighting a critical distributional mismatch between training data and real-world attacks that forces developers into reactive patching cycles. To tackle this challenge, we propose IMAGINE, a synthesis framework that leverages embedding space distribution analysis to generate jailbreak-like instructions. This approach effectively fills the distributional gap between authentic jailbreak patterns and safety alignment corpora. IMAGINE follows an iterative optimization process that dynamically evolves text generation distributions across iterations, thereby augmenting the coverage of safety alignment data distributions through synthesized data examples. Based on the safety-aligned corpus enhanced through IMAGINE, our framework demonstrates significant decreases in attack success rate on Qwen2.5, Llama3.1, and Llama3.2 without compromising their utility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20030v1">Large Language Models (LLMs) for Electronic Design Automation (EDA)</a></div>
    <div class="paper-meta">
      📅 2025-08-27
      | 💬 Accepted by IEEE International System-on-Chip Conference
    </div>
    <details class="paper-abstract">
      With the growing complexity of modern integrated circuits, hardware engineers are required to devote more effort to the full design-to-manufacturing workflow. This workflow involves numerous iterations, making it both labor-intensive and error-prone. Therefore, there is an urgent demand for more efficient Electronic Design Automation (EDA) solutions to accelerate hardware development. Recently, large language models (LLMs) have shown remarkable advancements in contextual comprehension, logical reasoning, and generative capabilities. Since hardware designs and intermediate scripts can be represented as text, integrating LLM for EDA offers a promising opportunity to simplify and even automate the entire workflow. Accordingly, this paper provides a comprehensive overview of incorporating LLMs into EDA, with emphasis on their capabilities, limitations, and future opportunities. Three case studies, along with their outlook, are introduced to demonstrate the capabilities of LLMs in hardware design, testing, and optimization. Finally, future directions and challenges are highlighted to further explore the potential of LLMs in shaping the next-generation EDA, providing valuable insights for researchers interested in leveraging advanced AI technologies for EDA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20024v1">Using item recommendations and LLMs in marketing email titles</a></div>
    <div class="paper-meta">
      📅 2025-08-27
      | 💬 Accepted to The Second Workshop on Generative AI for E-commerce (GenAIECommerce '25), held September 22, 2025, in Prague, Czech Republic. 3 figures
    </div>
    <details class="paper-abstract">
      E-commerce marketplaces make use of a number of marketing channels like emails, push notifications, etc. to reach their users and stimulate purchases. Personalized emails especially are a popular touch point for marketers to inform users of latest items in stock, especially for those who stopped visiting the marketplace. Such emails contain personalized recommendations tailored to each user's interests, enticing users to buy relevant items. A common limitation of these emails is that the primary entry point, the title of the email, tends to follow fixed templates, failing to inspire enough interest in the contents. In this work, we explore the potential of large language models (LLMs) for generating thematic titles that reflect the personalized content of the emails. We perform offline simulations and conduct online experiments on the order of millions of users, finding our techniques useful in improving the engagement between customers and our emails. We highlight key findings and learnings as we productionize the safe and automated generation of email titles for millions of users.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20015v1">Decomposing Behavioral Phase Transitions in LLMs: Order Parameters for Emergent Misalignment</a></div>
    <div class="paper-meta">
      📅 2025-08-27
      | 💬 11+25 pages, 4+11 figures
    </div>
    <details class="paper-abstract">
      Fine-tuning LLMs on narrowly harmful datasets can lead to behavior that is broadly misaligned with respect to human values. To understand when and how this emergent misalignment occurs, we develop a comprehensive framework for detecting and characterizing rapid transitions during fine-tuning using both distributional change detection methods as well as order parameters that are formulated in plain English and evaluated by an LLM judge. Using an objective statistical dissimilarity measure, we quantify how the phase transition that occurs during fine-tuning affects multiple aspects of the model. In particular, we assess what percentage of the total distributional change in model outputs is captured by different aspects, such as alignment or verbosity, providing a decomposition of the overall transition. We also find that the actual behavioral transition occurs later in training than indicated by the peak in the gradient norm alone. Our framework enables the automated discovery and quantification of language-based order parameters, which we demonstrate on examples ranging from knowledge questions to politics and ethics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13972v2">Truth or Twist? Optimal Model Selection for Reliable Label Flipping Evaluation in LLM-based Counterfactuals</a></div>
    <div class="paper-meta">
      📅 2025-08-26
      | 💬 Accepted at INLG 2025, camera-ready version
    </div>
    <details class="paper-abstract">
      Counterfactual examples are widely employed to enhance the performance and robustness of large language models (LLMs) through counterfactual data augmentation (CDA). However, the selection of the judge model used to evaluate label flipping, the primary metric for assessing the validity of generated counterfactuals for CDA, yields inconsistent results. To decipher this, we define four types of relationships between the counterfactual generator and judge models: being the same model, belonging to the same model family, being independent models, and having an distillation relationship. Through extensive experiments involving two state-of-the-art LLM-based methods, three datasets, four generator models, and 15 judge models, complemented by a user study (n = 90), we demonstrate that judge models with an independent, non-fine-tuned relationship to the generator model provide the most reliable label flipping evaluations. Relationships between the generator and judge models, which are closely aligned with the user study for CDA, result in better model performance and robustness. Nevertheless, we find that the gap between the most effective judge models and the results obtained from the user study remains considerably large. This suggests that a fully automated pipeline for CDA may be inadequate and requires human intervention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18976v1">The Double-edged Sword of LLM-based Data Reconstruction: Understanding and Mitigating Contextual Vulnerability in Word-level Differential Privacy Text Sanitization</a></div>
    <div class="paper-meta">
      📅 2025-08-26
      | 💬 15 pages, 4 figures, 8 tables. Accepted to WPES @ CCS 2025
    </div>
    <details class="paper-abstract">
      Differentially private text sanitization refers to the process of privatizing texts under the framework of Differential Privacy (DP), providing provable privacy guarantees while also empirically defending against adversaries seeking to harm privacy. Despite their simplicity, DP text sanitization methods operating at the word level exhibit a number of shortcomings, among them the tendency to leave contextual clues from the original texts due to randomization during sanitization $\unicode{x2013}$ this we refer to as $\textit{contextual vulnerability}$. Given the powerful contextual understanding and inference capabilities of Large Language Models (LLMs), we explore to what extent LLMs can be leveraged to exploit the contextual vulnerability of DP-sanitized texts. We expand on previous work not only in the use of advanced LLMs, but also in testing a broader range of sanitization mechanisms at various privacy levels. Our experiments uncover a double-edged sword effect of LLM-based data reconstruction attacks on privacy and utility: while LLMs can indeed infer original semantics and sometimes degrade empirical privacy protections, they can also be used for good, to improve the quality and privacy of DP-sanitized texts. Based on our findings, we propose recommendations for using LLM data reconstruction as a post-processing step, serving to increase privacy protection by thinking adversarially.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16267v2">From Confidence to Collapse in LLM Factual Robustness</a></div>
    <div class="paper-meta">
      📅 2025-08-26
    </div>
    <details class="paper-abstract">
      Ensuring the robustness of factual knowledge in LLMs is critical for reliable applications in tasks such as question answering and reasoning. However, existing evaluation methods predominantly focus on performance-based metrics, often investigating from the perspective of prompt perturbations, which captures only the externally triggered side of knowledge robustness. To bridge this gap, we introduce a principled approach to measure factual robustness from the perspective of the generation process by analyzing token distribution entropy in combination with temperature scaling sensitivity. These two factors build the Factual Robustness Score (FRS), a novel metric which quantifies the stability of a fact against perturbations in decoding conditions, given its initial uncertainty. To validate our approach, we conduct extensive experiments on 5 LLMs across 3 closed-book QA datasets (SQuAD, TriviaQA, and HotpotQA). We show that factual robustness varies significantly -- smaller models report an FRS of $0.76$, larger ones $0.93$ -- with accuracy degrading by ~$60\%$ under increased uncertainty. These insights demonstrate how entropy and temperature scaling impact factual accuracy, and lay a foundation for developing more robust knowledge retention and retrieval in future models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18947v1">LLMs in the SOC: An Empirical Study of Human-AI Collaboration in Security Operations Centres</a></div>
    <div class="paper-meta">
      📅 2025-08-26
      | 💬 22 pages, 9 figures, under review
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into Security Operations Centres (SOCs) presents a transformative, yet still evolving, opportunity to reduce analyst workload through human-AI collaboration. However, their real-world application in SOCs remains underexplored. To address this gap, we present a longitudinal study of 3,090 analyst queries from 45 SOC analysts over 10 months. Our analysis reveals that analysts use LLMs as on-demand aids for sensemaking and context-building, rather than for making high-stakes determinations, preserving analyst decision authority. The majority of queries are related to interpreting low-level telemetry (e.g., commands) and refining technical communication through short (1-3 turn) interactions. Notably, 93% of queries align with established cybersecurity competencies (NICE Framework), underscoring the relevance of LLM use for SOC-related tasks. Despite variations in tasks and engagement, usage trends indicate a shift from occasional exploration to routine integration, with growing adoption and sustained use among a subset of analysts. We find that LLMs function as flexible, on-demand cognitive aids that augment, rather than replace, SOC expertise. Our study provides actionable guidance for designing context-aware, human-centred AI assistance in security operations, highlighting the need for further in-the-wild research on real-world analyst-LLM collaboration, challenges, and impacts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16949v2">Breaking the Exploration Bottleneck: Rubric-Scaffolded Reinforcement Learning for General LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-08-26
      | 💬 This work is still in progress
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have underscored the potential of Reinforcement Learning (RL) to facilitate the emergence of reasoning capabilities. Despite the encouraging results, a fundamental dilemma persists as RL improvement relies on learning from high-quality samples, yet the exploration for such samples remains bounded by the inherent limitations of LLMs. This, in effect, creates an undesirable cycle in which what cannot be explored cannot be learned. In this work, we propose Rubric-Scaffolded Reinforcement Learning (RuscaRL), a novel instructional scaffolding framework designed to break the exploration bottleneck for general LLM reasoning. Specifically, RuscaRL introduces checklist-style rubrics as (1) explicit scaffolding for exploration during rollout generation, where different rubrics are provided as external guidance within task instructions to steer diverse high-quality responses. This guidance is gradually decayed over time, encouraging the model to internalize the underlying reasoning patterns; (2) verifiable rewards for exploitation during model training, where we can obtain robust LLM-as-a-Judge scores using rubrics as references, enabling effective RL on general reasoning tasks. Extensive experiments demonstrate the superiority of the proposed RuscaRL across various benchmarks, effectively expanding reasoning boundaries under the best-of-N evaluation. Notably, RuscaRL significantly boosts Qwen2.5-7B-Instruct from 23.6 to 50.3 on HealthBench-500, surpassing GPT-4.1. Furthermore, our fine-tuned variant on Qwen3-30B-A3B-Instruct achieves 61.1 on HealthBench-500, outperforming leading LLMs including OpenAI-o3. This work is still in progress, and we will release the code, the models, and the datasets soon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18918v1">DESAMO: A Device for Elder-Friendly Smart Homes Powered by Embedded LLM with Audio Modality</a></div>
    <div class="paper-meta">
      📅 2025-08-26
      | 💬 2 pages, 2 figures. Accepted for presentation as a UIST 2025 Poster
    </div>
    <details class="paper-abstract">
      We present DESAMO, an on-device smart home system for elder-friendly use powered by Audio LLM, that supports natural and private interactions. While conventional voice assistants rely on ASR-based pipelines or ASR-LLM cascades, often struggling with the unclear speech common among elderly users and unable to handle non-speech audio, DESAMO leverages an Audio LLM to process raw audio input directly, enabling a robust understanding of user intent and critical events, such as falls or calls for help.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09396v2">The Influence of Human-inspired Agentic Sophistication in LLM-driven Strategic Reasoners</a></div>
    <div class="paper-meta">
      📅 2025-08-26
    </div>
    <details class="paper-abstract">
      The rapid rise of large language models (LLMs) has shifted artificial intelligence (AI) research toward agentic systems, motivating the use of weaker and more flexible notions of agency. However, this shift raises key questions about the extent to which LLM-based agents replicate human strategic reasoning, particularly in game-theoretic settings. In this context, we examine the role of agentic sophistication in shaping artificial reasoners' performance by evaluating three agent designs: a simple game-theoretic model, an unstructured LLM-as-agent model, and an LLM integrated into a traditional agentic framework. Using guessing games as a testbed, we benchmarked these agents against human participants across general reasoning patterns and individual role-based objectives. Furthermore, we introduced obfuscated game scenarios to assess agents' ability to generalise beyond training distributions. Our analysis, covering over 2000 reasoning samples across 25 agent configurations, shows that human-inspired cognitive structures can enhance LLM agents' alignment with human strategic behaviour. Still, the relationship between agentic design complexity and human-likeness is non-linear, highlighting a critical dependence on underlying LLM capabilities and suggesting limits to simple architectural augmentation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18872v1">Empowering Computing Education Researchers Through LLM-Assisted Content Analysis</a></div>
    <div class="paper-meta">
      📅 2025-08-26
      | 💬 7 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Computing education research (CER) is often instigated by practitioners wanting to improve both their own and the wider discipline's teaching practice. However, the latter is often difficult as many researchers lack the colleagues, resources, or capacity to conduct research that is generalisable or rigorous enough to advance the discipline. As a result, research methods that enable sense-making with larger volumes of qualitative data, while not increasing the burden on the researcher, have significant potential within CER. In this discussion paper, we propose such a method for conducting rigorous analysis on large volumes of textual data, namely a variation of LLM-assisted content analysis (LACA). This method combines content analysis with the use of large language models, empowering researchers to conduct larger-scale research which they would otherwise not be able to perform. Using a computing education dataset, we illustrate how LACA could be applied in a reproducible and rigorous manner. We believe this method has potential in CER, enabling more generalisable findings from a wider range of research. This, together with the development of similar methods, can help to advance both the practice and research quality of the CER discipline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18850v1">ClusterFusion: Expanding Operator Fusion Scope for LLM Inference via Cluster-Level Collective Primitive</a></div>
    <div class="paper-meta">
      📅 2025-08-26
    </div>
    <details class="paper-abstract">
      Large language model (LLM) decoding suffers from high latency due to fragmented execution across operators and heavy reliance on off-chip memory for data exchange and reduction. This execution model limits opportunities for fusion and incurs significant memory traffic and kernel launch overhead. While modern architectures such as NVIDIA Hopper provide distributed shared memory and low-latency intra-cluster interconnects, they expose only low-level data movement instructions, lacking structured abstractions for collective on-chip communication. To bridge this software-hardware gap, we introduce two cluster-level communication primitives, ClusterReduce and ClusterGather, which abstract common communication patterns and enable structured, high-speed data exchange and reduction between thread blocks within a cluster, allowing intermediate results to be on-chip without involving off-chip memory. Building on these abstractions, we design ClusterFusion, an execution framework that schedules communication and computation jointly to expand operator fusion scope by composing decoding stages such as QKV Projection, Attention, and Output Projection into a single fused kernels. Evaluations on H100 GPUs show that ClusterFusion outperforms state-of-the-art inference frameworks by 1.61x on average in end-to-end latency across different models and configurations. The source code is available at https://github.com/xinhao-luo/ClusterFusion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18819v1">LLM-based Contrastive Self-Supervised AMR Learning with Masked Graph Autoencoders for Fake News Detection</a></div>
    <div class="paper-meta">
      📅 2025-08-26
    </div>
    <details class="paper-abstract">
      The proliferation of misinformation in the digital age has led to significant societal challenges. Existing approaches often struggle with capturing long-range dependencies, complex semantic relations, and the social dynamics influencing news dissemination. Furthermore, these methods require extensive labelled datasets, making their deployment resource-intensive. In this study, we propose a novel self-supervised misinformation detection framework that integrates both complex semantic relations using Abstract Meaning Representation (AMR) and news propagation dynamics. We introduce an LLM-based graph contrastive loss (LGCL) that utilizes negative anchor points generated by a Large Language Model (LLM) to enhance feature separability in a zero-shot manner. To incorporate social context, we employ a multi view graph masked autoencoder, which learns news propagation features from social context graph. By combining these semantic and propagation-based features, our approach effectively differentiates between fake and real news in a self-supervised manner. Extensive experiments demonstrate that our self-supervised framework achieves superior performance compared to other state-of-the-art methodologies, even with limited labelled datasets while improving generalizability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18190v2">ST-Raptor: LLM-Powered Semi-Structured Table Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-08-26
      | 💬 Extension of our SIGMOD 2026 paper. Please refer to source code available at: https://github.com/weAIDB/ST-Raptor
    </div>
    <details class="paper-abstract">
      Semi-structured tables, widely used in real-world applications (e.g., financial reports, medical records, transactional orders), often involve flexible and complex layouts (e.g., hierarchical headers and merged cells). These tables generally rely on human analysts to interpret table layouts and answer relevant natural language questions, which is costly and inefficient. To automate the procedure, existing methods face significant challenges. First, methods like NL2SQL require converting semi-structured tables into structured ones, which often causes substantial information loss. Second, methods like NL2Code and multi-modal LLM QA struggle to understand the complex layouts of semi-structured tables and cannot accurately answer corresponding questions. To this end, we propose ST-Raptor, a tree-based framework for semi-structured table question answering using large language models. First, we introduce the Hierarchical Orthogonal Tree (HO-Tree), a structural model that captures complex semi-structured table layouts, along with an effective algorithm for constructing the tree. Second, we define a set of basic tree operations to guide LLMs in executing common QA tasks. Given a user question, ST-Raptor decomposes it into simpler sub-questions, generates corresponding tree operation pipelines, and conducts operation-table alignment for accurate pipeline execution. Third, we incorporate a two-stage verification mechanism: forward validation checks the correctness of execution steps, while backward validation evaluates answer reliability by reconstructing queries from predicted answers. To benchmark the performance, we present SSTQA, a dataset of 764 questions over 102 real-world semi-structured tables. Experiments show that ST-Raptor outperforms nine baselines by up to 20% in answer accuracy. The code is available at https://github.com/weAIDB/ST-Raptor.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18736v1">Rethinking Caching for LLM Serving Systems: Beyond Traditional Heuristics</a></div>
    <div class="paper-meta">
      📅 2025-08-26
    </div>
    <details class="paper-abstract">
      Serving Large Language Models (LLMs) at scale requires meeting strict Service Level Objectives (SLOs) under severe computational and memory constraints. Nevertheless, traditional caching strategies fall short: exact-matching and prefix caches neglect query semantics, while state-of-the-art semantic caches remain confined to traditional intuitions, offering little conceptual departure. Building on this, we present SISO, a semantic caching system that redefines efficiency for LLM serving. SISO introduces centroid-based caching to maximize coverage with minimal memory, locality-aware replacement to preserve high-value entries, and dynamic thresholding to balance accuracy and latency under varying workloads. Across diverse datasets, SISO delivers up to 1.71$\times$ higher hit ratios and consistently stronger SLO attainment compared to state-of-the-art systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18721v1">LLM as an Execution Estimator: Recovering Missing Dependency for Practical Time-travelling Debugging</a></div>
    <div class="paper-meta">
      📅 2025-08-26
    </div>
    <details class="paper-abstract">
      Dynamic data dependency, answering "why a variable has this value?", is critical for debugging. Given a program step `s` reading a variable `v`, finding the dynamic definition of `v` is challenging. Traditional methods require either (1) exhaustive instrumentation of all possible definitions of `v` in one run or (2) replicating the run to re-examine reads/writes - both costly. If `v` is defined in a library, instrumentation becomes expensive; for non-deterministic programs, replication is infeasible. We propose RecovSlicing, which computes dynamic data dependency in a single run with partial instrumentation. We leverage LLMs to infer program behavior from a partially recorded trace and code context. Given a trace and a slicing criterion (step `s` and variable `v`), RecovSlicing estimates the runtime definition of `v` by recovering the missing execution.It also supports implicit variables, such as those in `list.get(i)`. Technically, RecovSlicing tackles: (1) recovering runtime values and structures, and (2) aligning recovered variables with recorded memory to analyze definitions. We evaluate RecovSlicing on 8,300 data dependencies across three slicing benchmarks, comparing it with Slicer4J, ND-Slicer, LLM Slicer, and re-execution Slicer. RecovSlicing achieves accuracy of 80.3%, 91.1%, and 98.3%, outperforming the best baseline (39.0%, 82.0%, 59.9%), and also leads in recall (91.1%, 91.1%, 98.3% vs. 53.4%, 79.1%, 87.1%). Integrated into a regression bug localizer, it enables finding 16% more regressions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18709v1">Filtering for Creativity: Adaptive Prompting for Multilingual Riddle Generation in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-26
    </div>
    <details class="paper-abstract">
      Multilingual riddle generation challenges large language models (LLMs) to balance cultural fluency with creative abstraction. Standard prompting strategies -- zero-shot, few-shot, chain-of-thought -- tend to reuse memorized riddles or perform shallow paraphrasing. We introduce Adaptive Originality Filtering (AOF), a prompting framework that filters redundant generations using cosine-based similarity rejection, while enforcing lexical novelty and cross-lingual fidelity. Evaluated across three LLMs and four language pairs, AOF-enhanced GPT-4o achieves \texttt{0.177} Self-BLEU and \texttt{0.915} Distinct-2 in Japanese, signaling improved lexical diversity and reduced redundancy compared to other prompting methods and language pairs. Our findings show that semantic rejection can guide culturally grounded, creative generation without task-specific fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08392v3">Multi-Agent LLMs as Ethics Advocates for AI-Based Systems</a></div>
    <div class="paper-meta">
      📅 2025-08-26
    </div>
    <details class="paper-abstract">
      Incorporating ethics into the requirement elicitation process is essential for creating ethically aligned systems. Although eliciting manual ethics requirements is effective, it requires diverse input from multiple stakeholders, which can be challenging due to time and resource constraints. Moreover, it is often given a low priority in the requirements elicitation process. This study proposes a framework for generating ethics requirements drafts by introducing an ethics advocate agent in a multi-agent LLM setting. This agent critiques and provides input on ethical issues based on the system description. The proposed framework is evaluated through two case studies from different contexts, demonstrating that it captures the majority of ethics requirements identified by researchers during 30-minute interviews and introduces several additional relevant requirements. However, it also highlights reliability issues in generating ethics requirements, emphasizing the need for human feedback in this sensitive domain. We believe this work can facilitate the broader adoption of ethics in the requirements engineering process, ultimately leading to more ethically aligned products.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18684v1">FALCON: Autonomous Cyber Threat Intelligence Mining with LLMs for IDS Rule Generation</a></div>
    <div class="paper-meta">
      📅 2025-08-26
      | 💬 11 pages, 5 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Signature-based Intrusion Detection Systems (IDS) detect malicious activities by matching network or host activity against predefined rules. These rules are derived from extensive Cyber Threat Intelligence (CTI), which includes attack signatures and behavioral patterns obtained through automated tools and manual threat analysis, such as sandboxing. The CTI is then transformed into actionable rules for the IDS engine, enabling real-time detection and prevention. However, the constant evolution of cyber threats necessitates frequent rule updates, which delay deployment time and weaken overall security readiness. Recent advancements in agentic systems powered by Large Language Models (LLMs) offer the potential for autonomous IDS rule generation with internal evaluation. We introduce FALCON, an autonomous agentic framework that generates deployable IDS rules from CTI data in real-time and evaluates them using built-in multi-phased validators. To demonstrate versatility, we target both network (Snort) and host-based (YARA) mediums and construct a comprehensive dataset of IDS rules with their corresponding CTIs. Our evaluations indicate FALCON excels in automatic rule generation, with an average of 95% accuracy validated by qualitative evaluation with 84% inter-rater agreement among multiple cybersecurity analysts across all metrics. These results underscore the feasibility and effectiveness of LLM-driven data mining for real-time cyber threat mitigation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13811v3">Can LLMs Handle WebShell Detection? Overcoming Detection Challenges with Behavioral Function-Aware Framework</a></div>
    <div class="paper-meta">
      📅 2025-08-26
      | 💬 Published as a conference paper at COLM 2025
    </div>
    <details class="paper-abstract">
      WebShell attacks, where malicious scripts are injected into web servers, pose a significant cybersecurity threat. Traditional ML and DL methods are often hampered by challenges such as the need for extensive training data, catastrophic forgetting, and poor generalization. Recently, Large Language Models have emerged as powerful alternatives for code-related tasks, but their potential in WebShell detection remains underexplored. In this paper, we make two contributions: (1) a comprehensive evaluation of seven LLMs, including GPT-4, LLaMA 3.1 70B, and Qwen 2.5 variants, benchmarked against traditional sequence- and graph-based methods using a dataset of 26.59K PHP scripts, and (2) the Behavioral Function-Aware Detection (BFAD) framework, designed to address the specific challenges of applying LLMs to this domain. Our framework integrates three components: a Critical Function Filter that isolates malicious PHP function calls, a Context-Aware Code Extraction strategy that captures the most behaviorally indicative code segments, and Weighted Behavioral Function Profiling that enhances in-context learning by prioritizing the most relevant demonstrations based on discriminative function-level profiles. Our results show that, stemming from their distinct analytical strategies, larger LLMs achieve near-perfect precision but lower recall, while smaller models exhibit the opposite trade-off. However, all baseline models lag behind previous SOTA methods. With the application of BFAD, the performance of all LLMs improves significantly, yielding an average F1 score increase of 13.82%. Notably, larger models now outperform SOTA benchmarks, while smaller models such as Qwen-2.5-Coder-3B achieve performance competitive with traditional methods. This work is the first to explore the feasibility and limitations of LLMs for WebShell detection and provides solutions to address the challenges in this task.
    </details>
</div>
