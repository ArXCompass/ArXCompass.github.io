# llm - 2026_06

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
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11898v2">GraspLLM: Towards Zero-Shot Generalization on Text-Attributed Graphs with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-11
    </div>
    <details class="paper-abstract">
      Research on Text-Attributed Graphs (TAGs) has gained significant attention recently due to its broad applications across various real-world data scenarios, such as citation networks, e-commerce platforms, social media, and web pages. Inspired by the remarkable semantic understanding ability of Large Language Models (LLMs), there have been numerous attempts to integrate LLMs into TAGs. However, existing methods still struggle to generalize across diverse graphs and tasks, and their ability to capture transferable graph structural patterns remains limited. To address this, we introduce the GraspLLM, a framework that combines Graph structural comprehension with semantic understanding prowess of LLMs to enhance the cross-dataset and cross-task generalizability. Specifically, we represent node texts from different graphs in a unified semantic space with a frozen general embedding model, on top of which we perform motif-aware contrastive learning across multiple motif-induced adjacency matrices to extract dataset-agnostic structural information. Then, with our proposed optimal contextual subgraph, we extract the most contextually relevant subgraph for each target node and align these subgraphs to the token space of LLM via an alignment projector. Extensive experiments on TAG benchmark datasets spanning diverse domains reveal that GraspLLM consistently outperforms previous LLM-based methods for TAGs, especially in zero-shot scenarios, highlighting its strong generalizability across different datasets and tasks. Our code is available at https://github.com/Heinz217/GraspLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10231v2">LLM can Read Spectrogram: Encoder-free Speech-Language Modeling</a></div>
    <div class="paper-meta">
      📅 2026-06-11
    </div>
    <details class="paper-abstract">
      Recent speech-aware large language models (Speech-LLMs) rely on a pre-trained speech encoder to convert audio into semantic-rich representations consumable by LLM. In this work, instead, we explore: can an LLM learn to read Mel spectrogram directly without a dedicated speech encoder? We propose Mel-LLM, an encoder-free Speech-LLM that feeds lightly pre-processed Mel spectrogram patches directly into the LLM through a linear projection, allowing the LLM to learn speech-text alignment purely through its own parameters. We conduct extensive experiments on both automatic speech recognition (ASR) and text-to-speech (TTS) tasks. For ASR, we evaluate on the OpenASR leaderboard public sets and production-level scaling experiments, demonstrating that the encoder-free solution achieves competitive performance with only limited degradation compared to encoder-initialized counterparts. We find that when data is limited, initialization from a multimodal checkpoint (Phi-4-MM) is crucial for maintaining performance. We also present ablation studies revealing which LLM layers are less relevant to speech encoding. For TTS, we show preliminary results with a next-token VAE approach. While TTS performance is not yet optimal, these results establish the feasibility of a fully unified encoder-free architecture for autoregressive speech-text modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.24079v2">The Pragmatic Persona: Discovering LLM Persona through Bridging Inference</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 15 pages, 4 figures, accepted to ICPR 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) reveal inherent and distinctive personas through dialogue. However, most existing persona discovery approaches rely on surface-level lexical or stylistic cues, treating dialogue as a flat sequence of tokens and failing to capture the deeper discourse-level structures that sustain persona consistency. To address this limitation, we propose a novel analytical framework that interprets LLM dialogue through bridging inference -- implicit conceptual relations that connect utterances via shared world knowledge and discourse coherence. By modeling these relations as structured knowledge graphs, our approach captures latent semantic links that govern how LLMs organize meaning across turns, enabling persona discovery at the level of discourse coherence rather than surface realizations. Experimental results across multiple reasoning backbones and target LLMs, ranging from small-scale models to 80B-parameter systems, demonstrate that bridging-inference graphs yield significantly stronger semantic coherence and more stable persona identification than frequency or style-based baselines. These results show that persona traits are consistently encoded in the structural organization of discourse rather than isolated lexical patterns. This work presents a systematic framework for probing, extracting, and visualizing latent LLM personas through the lens of Cognitive Discourse Theory, bridging computational linguistics, cognitive semantics, and persona reasoning in large language models. Codes are available at https://github.com/JiSoo-Yang/Persona_Bridging.git
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12821v1">GeoNatureAgent Benchmark: Benchmarking LLM Agents for Environmental Geospatial Analysis Across Frontier and Open-Weight Foundation Models</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 Preprint. 10 pages, 8 figures. Submitted to ACM SIGSPATIAL 2026
    </div>
    <details class="paper-abstract">
      Environmental scientists spend disproportionate effort on data wrangling rather than analysis, and AI agents that automate geospatial workflows remain unvalidated: no benchmark evaluates agents operating through structured tool calling against real APIs. We introduce the GeoNatureAgent Benchmark, the first benchmark for environmental analysis agents that operate via structured tool calls to a production-style geospatial API. It comprises 93 tasks across 18 categories, covering municipality analysis, multi-turn conversation, spatial reasoning, cross-indicator synthesis, error handling and recovery, ranking, comparison, multilingual understanding, habitat analysis, and task rejection. Tasks are evaluated against an open, self-hostable API serving three environmental indicators across Spain and Portugal via sixteen tools. We evaluate seven LLMs (Claude Sonnet 4, DeepSeek V3.2, GLM-5, Gemini 2.5 Pro, Qwen3-235B, GPT-OSS-120B, Llama 4 Scout) under three temperature-1.0 seeds, reporting capability and per-case cost as orthogonal axes. We find: (1) Claude Sonnet 4 leads at 60.8% +/- 0.8%, followed by DeepSeek V3.2 at 56.3% +/- 3.1%, with no other model above 51%; (2) the cost-accuracy Pareto frontier is occupied mostly by open-weight models, with DeepSeek V3.2 offering 93% of Claude's capability at 11x lower cost ($0.011/case); (3) comparison tasks remain universally unsolved (0% on close-value comparisons), exposing systematic reasoning limits; and (4) structured tool calling against a real API is more discriminative than general-purpose GIS benchmarks, with accuracies 25-35 points lower. We further show extensibility by integrating BigEarthNet V2 land cover for Portugal alongside Spanish CO2 and erosion indicators. The benchmark, harness, and self-hostable API are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12801v1">AiAWE: An Open-Source LLM Automated Writing Evaluation System Using LoRA-Adapted Instruction-Tuned Models</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 21 pages with 7 tables and 1 figure and appendices
    </div>
    <details class="paper-abstract">
      This study presents AiAWE, an open-source automated writing evaluation system that scores argumentative essays using a LoRA-adapted instruction-tuned large language model (Gemma-3-27B-it). Using a proprietary Educational Testing Service (ETS) dataset of 480 TOEFL Independent Writing essays, we fine-tune Gemma-3-27B and LLaMA-3.3-70B under identical LoRA configurations on a 120-essay training subset and evaluate on the remaining 360 essays under identical inference quantization. The fine-tuned Gemma model achieves a root mean square error of 0.474, a quadratic weighted kappa of 0.828, and an agreement rate of 90.56% within +/- 0.5 of the human score, outperforming both the larger LLaMA-3.3-70B model and the fine-tuned GPT-3.5 baseline reported in prior work on the same dataset. Three findings are of broader interest: open-weight LLMs can match or exceed proprietary fine-tuning for rubric-aligned scoring; model scale is not a reliable predictor of downstream performance under LoRA adaptation; and identical LoRA hyperparameters produce qualitatively different adaptation behaviors across architectures. The production system runs on a consumer-grade server and is publicly accessible at https://app.awade.gec.waseda.ac.jp. LoRA adapters, application code, and fine-tuning YAMLs are publicly available through their respective repositories.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12780v1">ProPlay: Procedural World Models for Self-Evolving LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-11
    </div>
    <details class="paper-abstract">
      Self-evolving agents are expected to improve through interaction without external supervision, but this remains difficult in partially observable environments where agents must explore actively, learn from limited feedback, and decide when to trust prior experience. Existing LLM-agent methods often rely on memory or planning modules, yet they rarely close the loop between them to continually refine an internal understanding of environment dynamics. We introduce ProPlay, a procedural world model that supports procedure-level preplay, where agents can rehearse future procedural paths using the learned world knowledge. Rather than representing experience as isolated rules or low-level action constraints, ProPlay abstracts successful trajectories into procedures and organizes them in a procedure graph that captures causal transitions among task stages. Each transition is associated with a reliability record embedding to estimate its task-specific contribution from past outcomes. Before each episode, ProPlay simulates future procedural trajectories over known graph structures as structured soft guidance; after execution, it refines the graph using environment feedback. Experiments on public benchmarks show that ProPlay consistently improves environment understanding and self-evolution capability over strong baselines. Our code has been released in https://github.com/antman9914/proplay.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.18085v4">Structuring The Future: Diffusion LLM Speculative Decoding via Calibrated Draft Graphs</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 Original version uploaded on Sep 22, 2025. (v2): Extended Table 2 with additional analysis and referenced it in Sec 5.2. (v3): Added note to Sec 4.2 and Appendix A.2 specifying conditions for losslessness. (v4): Updated with the version accepted to ICML 2026 workshops
    </div>
    <details class="paper-abstract">
      Diffusion LLMs (dLLMs) have recently emerged as a powerful alternative to autoregressive LLMs (AR-LLMs) with the potential to operate at significantly higher token-generation rates. To unlock this potential, we present Spiffy, a speculative decoding algorithm to accelerate dLLM inference while provably preserving the model's output distribution. This work addresses the unique challenges involved in applying ideas from speculative decoding of AR-LLMs to dLLMs. Spiffy performs auto-speculation to eliminate the overheads of an independent draft model, structuring draft states in the form of a novel directed draft graph to take advantage of the bidirectional, blockwise nature of dLLM generation. These draft graphs are calibrated offline to maximize acceptance rates and are dynamically pruned during inference for improved computational efficiency. We present a detailed formulation of Spiffy and demonstrate its ability to accelerate LLaDA, Dream, and SDAR models in combination with KV caching and threshold-based dynamic unmasking leading to up to $8.6\times$ reduction in model inferences and $6.3\times$ acceleration in token rate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.13681v1">EvoArena: Tracking Memory Evolution for Robust LLM Agents in Dynamic Environments</a></div>
    <div class="paper-meta">
      📅 2026-06-11
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents have achieved strong performance on a wide range of benchmarks, yet most evaluations assume static environments. In contrast, real-world deployment is inherently dynamic, requiring agents to continually align their knowledge, skills, and behavior with changing environments and updated task conditions. To address this gap, we introduce EvoArena, a benchmark suite that models environment changes as sequences of progressive updates across terminal, software, and social domains. We further propose EvoMem, a patch-based memory paradigm that records memory evolution as structured update histories, enabling agents to reason about environmental evolution through changes in their memory. Experiments show that current agents struggle on EvoArena, achieving an average accuracy of 39.6% across evolving terminal, software, and social-preference domains. EvoMem consistently improves performance, yielding an average gain of 1.5% on EvoArena and also improving standard benchmarks such as GAIA and LoCoMo by 6.1% and 4.8%. Beyond individual tasks, EvoMem further improves chain-level accuracy by 3.7% on EvoArena, where success requires completing a consecutive sequence of related evolutionary subtasks. Mechanistic analysis shows that EvoMem improves evidence capture in the memory, indicating better preservation of complete evolving environment states. Our results highlight the importance of modeling evolution in both evaluation and memory for reliable agent deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.13649v1">Operadic consistency: a label-free signal for compositional reasoning failures in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-11
    </div>
    <details class="paper-abstract">
      Detecting LLM reasoning failures at inference time without ground-truth labels has motivated a wide range of confidence baselines, including self-consistency, semantic entropy, and P(True), built on within-question sampling and self-evaluation. Operad theory, the formalism for systems built by iterated substitution, suggests a complementary diagnostic: a model's direct answer to a compositional query should agree with the answer it produces by composing a stated decomposition of the same query. We instantiate this idea as operadic consistency (OC), a per-question signal. Across twelve instruction-tuned LLMs (4B to 671B parameters, open-weights and closed-source) on four multi-hop QA datasets, OC is strongly correlated with accuracy on every dataset (Pearson $r \in [0.86, 0.94]$, all $p \leq 0.0004$), and is the only signal we evaluate with $r \geq 0.85$ uniformly across all four datasets. Chain-of-thought self-consistency (CoT-SC; Wang et al., 2023) matches OC on HotpotQA and DROP ($r = 0.93, 0.87$) but drops to $r \approx 0.45$ on MuSiQue and StrategyQA. At the per-question level, OC contributes information beyond CoT-SC and semantic entropy on every dataset (cluster-robust $p \leq 10^{-16}$ for the OC coefficient), and the conclusion is robust to additionally controlling for constructed decomposition-aware baselines ($p \leq 10^{-13}$). The same signal yields selective-prediction improvements (accuracy at fixed coverage) over a tuned CoT-SC baseline at the equal-cost $K = 3$ budget (AUARC lifts of +0.086 to +0.096 and AUROC lifts of +0.092 to +0.164; 95% CIs exclude zero on every cell). On five frontier thinking models, where the decomposition is extracted from the model's own chain of thought, the same equal-cost comparison gives positive selective-prediction point-estimate lift on all 16 (dataset, budget, metric) cells tested, with 95% CIs excluding zero on 12 of the 16.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.13634v1">Operads for compositional reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-11
    </div>
    <details class="paper-abstract">
      Question decomposition, i.e. breaking a complex query into simpler sub-queries whose answers are composed to produce a final answer, is a widely used strategy for improving LLM reasoning, yet it currently lacks a rigorous mathematical foundation. In this paper, we propose operads, mathematical structures that model many-in, one-out operations and compositions thereof, as a natural framework for describing question decomposition. We define the questions operad $Q$, in which operations correspond to question templates and composition corresponds to substitution of sub-answers, and show how QA models can be interpreted as algebras over $Q$. Beyond reframing existing practice, this operadic perspective points toward new methods, in particular a notion of operadic consistency, which measures whether a QA model's answers agree across the partial collapses of a question decomposition tree. Empirical evaluation of operadic consistency is reported in our companion paper (Bottman, Liu, and Richardson, 2026), which finds it strongly correlated with accuracy across twelve LLMs and four multi-hop QA datasets and outperforming standard temperature-based self-consistency baselines. We argue that operads are the natural mathematical home for question decomposition, and that invariants such as operadic consistency open new directions for analyzing and improving the reliability of multi-step reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.13607v1">Reasoning as Pattern Matching: Shared Mechanisms in Human and LLM Everyday Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 13 pages main text, 51 pages supplementary text
    </div>
    <details class="paper-abstract">
      When large language models (LLMs) fail to generalize or make haphazard errors in reasoning, it is often taken as evidence that LLMs are not truly reasoning, but rather performing a kind of pattern matching. The implication is that people's behavior does not exhibit the same types of failures because human reasoning uses principled and abstract world models. We evaluate human participants and 25 LLMs on their ability to engage in common-sense reasoning about a variety of everyday situations and observe similar patterns of errors in both people and models. We then identify the set of attention heads driving LLM responses and find that these heads implement a form of pattern-matching. These attention heads allow us to predict seemingly inexplicable reasoning errors in people caused by ostensibly irrelevant prompt details. Taken together, our results suggest that everyday causal reasoning in people and LLMs is more consistent with a form of pattern-matching than with abstract world models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.31514v3">If LLMs Have Human-Like Attributes, Then So Does Age of Empires II</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 Fixed corollary 1, added stat sig
    </div>
    <details class="paper-abstract">
      Much research has been carried out on large language models (LLMs) and LLM-powered agentic workflows. However, many works within the field state emergence of, ascribe to, or assume, generalised anthropomorphic attributes to them (e.g., morality or understanding of natural language). Our goal is not to argue in favour or against the existence of these attributes, but to point out that these conclusions could be incorrect. For this we build and train a simple neural network on the videogame Age of Empires II, and note that any entity in a sufficiently-powerful substrate, such as LEGO or the Greater Boston Area, could also present such attributes. Hence, the purported anthropomorphic attributes of LLMs are empirically non-unique: although some properties (e.g., responses to prompts) could remain invariant, others, such as the interpretation of their perceived behaviour, might change with the substrate. Thus, any empirically-grounded discussion on these attributes requires explicit measurement criteria; otherwise the interpretation is left to the representation. We then show that assuming that these attributes exist or not in a system, independent of the substrate and in a generalised way, leads to either circular or uninformative conclusions. This is regardless of the experimenter's viewpoint on the subject, or whether the outcome shows existence or non-existence. Finally we propose a 'null' assumption, where one assumes LLM non-uniqueness instead of assuming anthropomorphic attributes to set up an experiment, along with examples of it. We also discuss potential objections to our work, briefly survey the field, and prove that Age of Empires II is functionally- and Turing-complete.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.07515v2">How reliable are LLMs when it comes to playing dice?</a></div>
    <div class="paper-meta">
      📅 2026-06-11
    </div>
    <details class="paper-abstract">
      We investigate the probabilistic reasoning capabilities of large language models through a controlled benchmarking study on discrete probability problems. We constructed two datasets, respectively a set of standard exercises and a set of counterintuitive exercises, designed to trigger heuristic reasoning, and evaluated 8 state-of-the-art models, each tested with and without Chain-of-Thought prompting. Models achieve an average accuracy of 0.96 on standard problems but only 0.59 on counterintuitive ones. We further provide empirical evidence of token bias: performance drops by over 20% when canonical formulations are replaced by disguised variants. Embedding misleading suggestions in the prompt reduces performance by up to 34%, with no model proving immune. Taken together, the reported findings suggest that current LLMs are not yet genuine probabilistic reasoners, despite their success in advanced mathematical problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12071v1">On the Limits of LLM-as-Judge for Scientific Novelty Assessment</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      LLMs are increasingly used to generate and judge scientific ideas. This makes novelty evaluation a central problem. Full idea evaluation is difficult because it often requires judging a method, its feasibility, and its empirical promise. We therefore study a cleaner upstream object: the research question (RQ). RQ generation is a prerequisite for scientific ideation, and RQs can be compared against questions pursued in real papers. We introduce RQ-Bench, a benchmark built from recent arXiv papers. For each paper, we reconstruct author-anchored RQs from its cited background, gaps, and contributions. These RQs are not the only valid questions for the same background. They are author-anchored reference points for testing novelty judgments. We evaluate model-generated RQs with standalone LLM judging, comparative LLM judging, and human expert evaluation. LLM judges consistently rate model-generated RQs as highly novel, producing a novelty mirage; in comparative evaluations, this preference becomes even stronger. Domain experts, however, reach the opposite conclusion and prefer the author-anchored reference questions. We further find that many generated RQs are narrow or source-bound, a dimension that LLM judges often miss unless explicitly tested. Overall, the contradictory novelty evaluations between LLM judges and human experts raise a serious concern about the reliability of using LLMs to assess the scientific novelty of research questions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.03308v3">The Security Budget of Code-LLM Prompt Hardening: Provable Limits Under Pass-Only Acceptance</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      We give a quantitative impossibility result for pass-only prompt hardening of code LLMs. For any deterministic prompt filter $h$ and a registered family of finite executable-equivalence task variables $\mathcal Y_{\mathrm{exec}}$, the shared filtered-prompt channel $\rmI(h(p);h(\tilde p))$ is lower-bounded by a worst-$Y$ Fano floor; on HumanEval and MBPP the universal pass-only floor evaluates to $\mathcal F^{\mathrm{op}}\ge 0.84$ and $1.20$ nats at $η=0.05$ task-collapse tolerance, and the identity row realizes $\mathcal F^{\mathrm{id}}\ge 1.67$ and $1.80$ nats. An estimator-invariance corollary lifts the floor to any deterministic embedding pipeline; a dataset-agnostic corollary states the floor in visible-spec entropy and is empirically witnessed by $164/164$ HumanEval+ and $224/224$ MBPP+ $V(p)$-invariance. We operationalize the floor as the \emph{Tri-Audit Protocol}, a two-axis reporting protocol that separates a prompt-side deductive registry attribute (Shannon nats on the visible-spec representation) from a model-side empirical proxy (KSG-1 primary, MINE secondary, on hidden states). A constrained best-of-family search over deterministic and guarded learned filters on CodeLlama-7B, Qwen2.5-Coder-7B/1.5B and DeepSeek-Coder-6.7B at $n=164$ yields the \emph{Cross-Model Tri-Audit Invariance}: of twenty-eight pass-preserving rows, twelve antecedent-preserving deterministic rows fail proxy-axis leakage reduction on every backbone with sign-invariant positive deviations, twelve antecedent-changed-of-record learned-canonicalizer rows fail proxy-axis leakage on every backbone, and four antecedent-violating rows are reported as registered-family collapse; no filter produces a shared Tri-pass on a nine-cell gate-sensitivity sweep. Pass@1 alone cannot certify code-LLM prompt hardening.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11976v1">Exploration Structure in LLM Agents for Multi-File Change Localization</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Software engineering tools increasingly rely on LLM based agents to localize files to change to resolve a software issue. Most AI agents explore repositories linearly, that is, visiting one directory or file per step. We postulate that this is a structural mismatch for changes that span several subsystems. We compare linear sequential exploration against non-linear, domain-scoped parallel agentic exploration. Using SWE Bench Pro as initial benchmark, we focus on ansible as an exemplar. We construct an approach for persistent-session evaluation of GitHub issues anchored at a single base commit. We compare our non-linear domain-agent file traversal system against a base LLM without direct repository access, a single agent Recursive Language Model (RLM) baseline with a persistent Python REPL and an external CLI baseline using Codex 5.5 High. Domain scoped parallel agent spawning with a small Haiku-class model achieves the highest micro F1 among Haiku class models by a large margin. Domain-agents is the second highest behind only the much larger Codex 5.5 High on our own expanded benchmark including over more recent PRs from 2025 and 2026. On the original, curated, 2020 SWE-bench Pro benchmark, a larger Sonnet plain LLM baseline attains higher micro F1 by predicting few files, leading to higher precision, but at significantly lower all gold recall. We also present three additional findings. First, documentation evolution is a latent dependency unresolved by any approach. Second, naive file system access can degrade localization driven by test-file over prediction. Lastly, forced multi-agent consultation does not measurably help and raises token cost substantially.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.29128v2">Apertus LLM Family Expansion via Distillation and Quantization</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      The wide adoption of LLMs has led to their use in great variety of applications and scenarios, such as chatbot assistants and data annotation, creating the need for the models to satisfy certain budget and hardware constraints. This has led to the trend of LLMs being released in batches consisting of similar models of various sizes for the family of models to adhere to as wide of a range of constraints as possible. In this paper, we validate distillation and quantization as a cost-effective way to expand model families to new sizes and hardware formats. Based on the open-recipe Apertus 8B LLM, we produce Apertus-v1.1 - a distilled family of models with up to 4B parameters trained on 1.7T permissive license tokens. We demonstrate cost-efficiency and strong accuracy performance of our approach for covering large ranges of hardware and systems requirements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11925v1">Corpus Augmentation for Sign Language Translation via LLM-Guided Video Stitching</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Sign language translation (SLT) converts sign language video into spoken language text and holds significant promise for improving accessibility and enabling communication between signing and non-signing communities. While large weakly-aligned datasets have enabled pre-training at scale and gloss-free methods have reduced reliance on expert annotation, high-quality parallel sign video-text pairs for fine-tuning remain scarce, limiting generalisation on long-tail vocabulary and unseen constructions. We propose a corpus augmentation approach that requires no additional human annotation, external sign-language video corpora, or generative video models, relying only on the existing gloss-annotated training corpus and an LLM for sentence generation: per-gloss clips are extracted from training videos via CTC forced-alignment, novel gloss-sentence pairs are generated by a corpus-anchored LLM, and synthetic sequences are assembled through random sentence sampling and clip assignment. The resulting synthetic RGB video-text pairs are architecture-agnostic at the downstream training stage and can be consumed directly by RGB-based SLT models, or converted into pose or feature representations by pipelines that derive such inputs from video. Sincan et al. re-evaluated five recent gloss-free methods under strictly identical conditions; the largest verified gain over the GFSLT-VLP baseline was only 0.98 BLEU-4. Our augmentation, applied within the same framework, achieves +2.92 BLEU-4 without any change to architecture or training protocol. We further identify that synthetic data harms vision-language pretraining despite improving its objectives, and that optimising clip transitions for visual smoothness is counter-productive under L2-based criteria; we propose that abrupt boundaries may act as a form of implicit regularisation. Code is available at https://github.com/robizso/slt-datagen.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11916v1">Characterizing Software Aging in GPU-Based LLM Serving Systems</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 7 pages
    </div>
    <details class="paper-abstract">
      This paper proposes an empirical methodology to study software aging in GPU-based LLM serving systems. Traditional aging studies focus on CPU-centric software with relatively regular workloads; LLM serving is different, spanning a Python host and a CUDA device, handling requests whose cost varies by orders of magnitude, and relying on rapidly evolving software stacks. We run a 216-hour campaign across six co-located deployments under identical stress conditions, monitor host, device, and client metrics in parallel, and apply a statistical pipeline that accounts for autocorrelation and multiple testing. Our results reveal statistically significant memory aging in all deployments, with leak rates strongly dependent on the serving runtime and deployment configuration. Beyond these findings, we provide a reproducible framework that opens a research direction at the intersection of the software aging and rejuvenation and LLM serving communities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11877v1">LLM-Enabled NWDAF: A Step Toward AI-Native 6G Network Intelligence</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 20 pages
    </div>
    <details class="paper-abstract">
      The Network Data Analytics Function (NWDAF) is central to enabling zero-touch network management in fifth-generation (5G) networks by supporting real-time analytics and closed-loop automation. Despite its critical role, open-source NWDAF implementations remain limited in scope and accessibility. In this paper, we develop an open-source NWDAF, compatible with the open-source core network Free5GC, that collects network data via subscriptions to Network Functions (NFs), and also includes an integrated Large Language Model (LLM) interface that enables natural language interaction with human operators. The interface processes user intents, encodes them using a semantic embedding model, and maps them to one of seven predefined intent categories to trigger analytics queries or event subscription commands. This architecture abstracts the complexity of traditional interfaces, allowing non-expert users to manage network analytics and subscriptions with ease. The system supports Access and Management Function (AMF) and Session Management Function (SMF) event subscriptions, real-time monitoring, and analytics retrieval via Prometheus, all accessible through a conversational interface. By bridging AI-driven intent recognition with standardized network analytics, our implementation enhances operator usability and provides a foundation towards AI-native 6G networks. The source code and datasets generated during the current study are available in the github repository, https://github.com/HenokDanielbfg/testbed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11863v1">Enhancing LLM-Based Code Translation with Verified Multi-Semantic Representations</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown great promise for automated code translation, yet existing approaches often rely on token-level statistical patterns rather than sufficient understanding of program semantics. As a result, translated programs may still contain logical and semantic errors. Although high-quality semantic guidance, such as functional descriptions and test cases, can help mitigate these errors, such resources are often unavailable in real-world scenarios. This raises two key challenges: how to construct rich semantic information directly from source code, and how to ensure that such semantics are accurate and reliable enough to guide translation.To address these challenges, we propose Multisage, a multi-semantic augmentation and self-calibration framework for LLM-based code translation. Multisage consists of three modules. First, a semantic representation parsing module extracts structured base semantics from source code, including data-flow graphs, type constraints, and external API information. Second, a multi-semantic augmentation module builds on these representations to generate diverse augmented semantics, including code summaries, function-level test cases, and API-oriented descriptions and tests. Third, a semantic consistency calibration module uses semantics-preserving mutations and cross-semantic consistency verification to filter, calibrate, and refine the generated semantics.Experiments on the HumanEval-X code translation benchmark show that Multisage improves translation success rates by up to 2.22 times across diverse backbone models. It consistently outperforms vanilla prompting, instruction-tuned LLMs, and Chain-of-Thought reasoning, with the largest gains observed on smaller models. These results demonstrate that explicit semantic augmentation can substantially improve the reliability of LLM-based code translation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11854v1">Fine-tuning Multi-modal LLMs with ART: Art-based Reinforcement Training</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      There are two main Parameter-Efficient Fine-Tuning (PEFT) techniques for Large Language Models (LLMs). While Low-Rank Adaptation (LoRA) introduces additional weights between the LLM layers, Soft Prompting introduces additional fine-tuning-specific raw tokens to an LLM input. However, both require modification to the computational graphs of precompiled, preoptimized LLMs. As a result, neither is fully supported in high-throughput engines like vLLM. We propose fine-tuning with ART (Art-based Reinforcement Training). The method injects information into a frozen Multimodal Large Language Model (MLLM) by optimizing only its raw visual input, thus enabling the soft-token approach on pre-compiled computational graphs. It relies on backpropagation of gradients back into a plain pixel array and thus supports any fine-tuning objective. Moreover, the optimized visual input can be stylized as task-relevant computational artworks. The approach's effectiveness is confirmed for different sizes of a popular open Qwen architecture and for several textual benchmarks. Specifically, ART reaches accuracy competitive with LoRA across mathematics and structured-tool-use benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11817v1">Grammar-Constrained Decoding Can Jailbreak LLMs into Generating Malicious Code</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used for code generation, raising concerns that they may be misused to produce malicious code. Meanwhile, Grammar-Constrained Decoding (GCD) has been widely adopted to improve the reliability of LLM-generated code by enforcing syntactic validity. In this paper, we reveal a counterintuitive risk: this reliability-oriented technique can itself become an attack surface. We uncover a new jailbreak attack, termed CodeSpear, that exploits GCD to induce LLMs into generating malicious code. Our experiments show that simply applying a benign code grammar constraint can effectively jailbreak LLMs. To address this vulnerability, we propose CodeShield, a safety alignment approach that robustly preserves safe behavior even under attacker-controlled grammar constraints. CodeShield aligns the model in the code modality by teaching it to generate honeypot code under GCD. Such code is semantically harmless, so it does not implement the malicious request, and structurally diverse, so it is difficult to suppress through grammar tightening. At the same time, CodeShield still preserves natural-language refusals when natural language is available. Experiments on 10 popular LLMs across 4 benchmarks show that CodeSpear outperforms representative jailbreak baselines and increases the attack success rate by more than 30 percentage points on average. CodeShield also restores safety under CodeSpear while preserving benign utility. Our findings reveal a fundamental risk of GCD and call for greater attention to its potential security implications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.25359v2">Geometric Metrics and LLMs: What They Measure and When They Work</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      We present a systematic stress-test of geometric metrics for LLM evaluation. Rank-based geometric properties of internal representations have shown promise as reference-free quality signals, but the conditions under which they are reliable remain unclear. We evaluate eight commonly-used metrics: intrinsic-dimensionality estimators, spectral norms, and related quantities across six tester models (0.5-8B) and eight generators on contrasting tasks, separating genuine geometric signal from text-length effects and from what standard text statistics already capture. Three findings emerge. First, some metrics (notably Schatten Norm and MOM) mainly reflect output length, and their apparent discriminative power collapses once length is controlled. Second, geometric metrics add modest but real information beyond text statistics: combined with them, a classifier reaches 78% accuracy on 6-way generator identification versus 69% for text statistics alone. Third, rather than tracking a general notion of text quality, the metrics demonstrate only moderate association between the intrinsic-dimensionality and lexical diversity (RTTR). We give use-case-specific recommendations and identify failure detection as the most promising near-term application.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11806v1">External Experience Serving in Production LLM Systems: A Deployment-Oriented Study of Quality-Cost Trade-offs</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Production LLM systems accumulate reusable operational experience, but the practical deployment issue is not merely whether such experience can help. It is how different serving strategies trade off quality against online cost under realistic constraints. Injecting external experience can improve task quality, yet it also increases prompt burden, latency, and serving pressure. We study \textit{external experience serving} as a deployment-oriented quality-cost trade-off problem. We evaluate this question in a real production moderation setting, with tool-use and GPQA as supporting contrast tasks that expose different output-cost regimes. We compare no-experience baselines, random experience controls, global prompt injection, and retrieval-based selective injection, and analyze both task quality and serving cost. The results show that, once experience becomes case-dependent, selective retrieval provides a stronger operating point than unconditional global injection. They further show that retrieval quality matters more than simply increasing Top-$K$, and that the same serving policy can exhibit substantially different cost-benefit profiles across short-output and decode-heavy regimes. These findings suggest that external experience is best treated as a selective, cost-aware serving decision rather than as a universal add-on. Overall, in the settings studied here, external experience pays off only when both the serving interface and the task-specific cost structure make its quality gains worth the online cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05727v2">LLM-Enhanced Deep Reinforcement Learning for Task Offloading in Collaborative Edge Computing</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Collaborative edge computing uses edge nodes in different locations to execute tasks, necessitating dynamic task offloading decisions to maintain low latency and high reliability, especially under unpredictable node failures. Although deep reinforcement learning (DRL) and large language models (LLMs) have shown promise for task offloading, DRL often suffers from poor sample efficiency and local optima, while LLMs are difficult to use directly due to inference overhead and output uncertainty. To address these limitations, we propose \textbf{LeDRL}, a hybrid decision framework that couples a \emph{lightweight LLM} with self-attention-enhanced DRL for real-time task offloading. LeDRL constructs structured, context-aware prompts capturing node status, task semantics, and link dynamics to derive high-level strategy priors. These are selectively processed by a self-attention-based alignment module for context-aware policy optimization. A reflective evaluator further distills semantic feedback from past trajectories to refine subsequent prompts and provide consistent guidance. Extensive experiments show that LeDRL outperforms representative baselines in task success rate, convergence speed, and real-time responsiveness across diverse network scales, achieving over 17\% improvement in success rate. Furthermore, we deploy LeDRL on Jetson-based edge devices using our prototype system \textit{CoEdgeSys}, demonstrating its robustness and feasibility under resource constraints. Our code is available at:https://github.com/GalleyG5/LeDRL.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.04567v3">GILT: An LLM-Free, Tuning-Free Graph Foundational Model for In-Context Learning</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 Accepted as an oral presentation at the GFM @ ICML 2026 Workshop
    </div>
    <details class="paper-abstract">
      Graph Neural Networks (GNNs) are powerful tools for processing relational data but often struggle to generalize to unseen graphs, giving rise to the development of Graph Foundational Models (GFMs). However, current GFMs are challenged by the extreme heterogeneity of graph data, where each graph can possess a unique feature space, label set, and topology. To address this, two main paradigms have emerged. The first leverages Large Language Models (LLMs), but is fundamentally text-dependent, thus struggles to handle the numerical features in vast graphs. The second pre-trains a structure-based model, but the adaptation to new tasks typically requires a costly, per-graph tuning stage, creating a critical efficiency bottleneck. In this work, we move beyond these limitations and introduce \textbf{G}raph \textbf{I}n-context \textbf{L}earning \textbf{T}ransformer (GILT), a framework built on an LLM-free and tuning-free architecture. GILT introduces a novel token-based framework for in-context learning (ICL) on graphs, reframing classification tasks spanning node, edge and graph levels in a unified framework. This mechanism is the key to handling heterogeneity, as it is designed to operate on generic numerical features. Further, its ability to understand class semantics dynamically from the context enables tuning-free adaptation. Comprehensive experiments show that GILT achieves stronger few-shot performance with significantly less time than LLM-based or tuning-based baselines, validating the effectiveness of our approach. Our code is available at: https://github.com/yiming421/inductnode/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08211v2">MobileFineTuner: A Mobile-Native Framework for On-Device LLM Fine-Tuning in Real-World Embedded AI Applications</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 26 pages, 25 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are moving from cloud-centric services toward on-device embedded AI, where models interact with private, longitudinal signals sensed from users and their physical environments. Mobile phones are a natural platform for such applications because they are continuously carried by users, connected to wearable sensors, and deeply integrated with daily mobile applications. However, practical LLM fine-tuning on commodity phones remains difficult. Existing fine-tuning frameworks are largely Python-based and server-oriented, making them hard to deploy inside mobile applications. We present MobileFineTuner, a mobile-native open-source framework for end-to-end LLM fine-tuning on commodity mobile phones. MobileFineTuner is implemented in C++ and provides a reusable training stack. To make fine-tuning feasible under mobile resource constraints, MobileFineTuner integrates a resource-aware training runtime with memory-efficient attention, activation checkpointing, gradient accumulation, parameter sharding, and energy-aware scheduling. We evaluate MobileFineTuner on real mobile phones using GPT-2, Gemma 3, and Qwen2.5 models across multiple fine-tuning tasks. The results show that MobileFineTuner reproduces standard Full-FT and LoRA fine-tuning behavior, substantially reduces memory pressure and improves executability on memory-constrained phones. We further demonstrate MobileFineTuner through a private campus health-agent application, where a local LLM is fine-tuned on user-specific wearable-sensing records to provide more personalized responses while keeping raw records on the phone. These results establish MobileFineTuner as a practical toolkit for studying and building on-device LLM fine-tuning applications in embedded AI and sensing systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10546v2">SkillAxe: Sharpening LLM-Authored Agent Skills Through Evaluation-Guided Self-Refinement</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 9 pages, under review
    </div>
    <details class="paper-abstract">
      Skill documents, structured natural-language instructions that guide Large Language Model (LLM) agents, are critical to modern agent frameworks, yet LLMs struggle to write skills that actually work. On SkillsBench, human-authored skills improve pass rates by 16.2 percentage points, while LLM-authored skills provide no measurable gain. We introduce SkillAxe, a fully unsupervised framework that enables LLMs to iteratively diagnose and refine their own skills. SkillAxe decomposes skill quality into four interpretable dimensions (quality impact, trigger precision, instruction compliance with fault attribution, and solution-path coverage), producing structured improvement briefs that require no ground-truth labels, test suites, or environment rewards. On SkillsBench, SkillAxe improves pass rates by 28\% relative over unimproved LLM skills and closes 47--67\% of the gap to human-authored skills. We validate the approach as a continuous improvement engine in the wild on SpreadsheetBench, where a SkillAxe-built skill library learns from past agent trajectories and raises pass rate from 16.0\% to 52.0\% using only 22 skills.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.23248v3">Resource-Aware LLM Reasoning for Mobile Edge General Intelligence</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has enabled an emergence of agentic artificial intelligence (AI) with powerful reasoning and autonomous decision-making capabilities. This integration with edge computing has led to the development of Mobile Edge General Intelligence (MEGI), which brings real-time, privacy-preserving reasoning to the network edge. However, deploying LLM-based agentic AI reasoning in MEGI environments poses significant challenges due to the high computational demands of reasoning and the limited resources of edge devices. To address these challenges, we propose a joint optimization framework for efficient LLM reasoning deployment in MEGI. First, we systematically review enhancement methods to identify mechanisms suitable for edge adaptation. Subsequently, we present a distributed framework that synergizes reasoning enhancement via adaptive CoT prompting with scalable deployment through a distributed MoE architecture. An important innovation of this approach involves modeling reasoning depth as a dynamic network resource variable, which is optimized jointly with expert activation and transmission power. This mechanism allows the system to dynamically regulate expert networks and reasoning complexity according to task requirements and device capabilities. Experimental evaluations in mobile edge environments demonstrate that the proposed framework effectively balances reasoning quality and resource efficiency. The results show that with less than one second of additional inference time, both accuracy and latency satisfaction rate can reach 90\%, validating the practical viability of deploying sophisticated LLM reasoning in resource-constrained MEGI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12481v1">Representing Time Series as Structured Programs for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated strong reasoning and instruction-following capabilities, making them potentially powerful tools for time-series analysis. However, time series lie outside their native textual modality, raising a fundamental question: how should time series be represented so that LLMs can reason about them effectively? Existing work typically serializes raw numerical sequences or fine-tunes pre-trained LLMs on time-series data. These approaches place the burden of extracting temporal structure directly on the LLM, creating a modality mismatch that often degrades performance on long sequences and introduces substantial computational overhead. In this work, we introduce Time-Series-to-Structured-Program representation (T2SP), a deterministic, training-free method that represents a time series as a structured symbolic program. T2SP decomposes time series into trends, periods, and salient events, expressing them in a program-friendly format aligned with the textual and code-like modalities on which LLMs are natively trained. By shifting temporal-structure extraction from the model to the representation itself, T2SP enables off-the-shelf LLMs to leverage their existing reasoning capabilities for time-series understanding. We evaluate T2SP on three reasoning tasks -- editing, captioning, and question answering -- where it consistently improves performance, reduces reasoning time, and lowers failure rates compared with raw-string representations. Our results demonstrate that T2SP provides an effective interface between time series and LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11755v1">Acoda: Adversarial Code Obfuscation for Defending against LLM-based Analysis</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      With the widespread adoption of Large Language Models (LLMs) in software engineering (SE) tasks such as code understanding, debugging, and vulnerability detection, their powerful semantic reasoning ability has also introduced new security and privacy risks. LLMs can analyze, reconstruct, or even reverse-engineer source code logic, potentially leading to the leakage of intellectual property. To address this issue, we propose Acoda, a genetic algorithm-based adversarial code obfuscation framework that defends against LLM-based code analysis. Acoda leverages two key mechanisms of LLMs, namely safety alignment and token-based information processing, to design 8 semantics-preserving obfuscation methods. It iteratively optimizes obfuscation strategies through a genetic algorithm to generate adversarial samples that maximize defensive effectiveness. In addition, we propose a quantitative evaluation framework based on LLM responses, which combines an auxiliary LLM and four evaluation metrics to assess how target LLMs analyze obfuscated code comprehensively. Experimental results show that Acoda can effectively induce LLMs to refuse or misinterpret code analysis. On 7 state-of-the-art LLMs, including GPT-4o, DeepSeek, Qwen, Llama, and Gemma, Acoda achieves an attack success rate (ASR) of up to 70%, with strong cross-model transferability and minimal runtime overhead, while ensuring that the semantics of the original code remain unchanged. Overall, this study provides a new perspective for code protection and LLM security defense in the era of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12479v1">ReCal: Reward Calibration for RL-based LLM Routing</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Large language model (LLM) routing has emerged as an effective paradigm for leveraging the complementary strengths of multiple LLMs through dynamic model and reasoning-strategy selection. Recent reinforcement learning (RL)-based routing methods further improve routing quality by optimizing routing policies from interaction feedback. However, they still struggle to provide informative and comparable learning signals under heterogeneous tasks with varying difficulty. In practice, multiple objectives (e.g., correctness, format behavior) are aggregated into a single scalar reward, leading to ambiguous credit assignment and conflicting optimization signals. Moreover, reward signals exhibit significant variability across instances, where some instances produce higher or more variable rewards, introducing optimization bias that favors trivial samples over informative ones. To address these issues, we propose \textbf{ReCal}, a \textbf{\underline{Re}}ward \textbf{\underline{Cal}}ibration framework for RL-based LLM routing. We first introduce a hierarchical reward decomposition mechanism with component-wise advantage estimation. We further propose a distribution-aware optimization strategy that calibrates optimization variability through variance-aware reweighting and per-dataset normalization. Experiments on seven datasets demonstrate that ReCal consistently improves routing performance, and training stability over baselines. Code is available at https://anonymous.4open.science/r/ReCal.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.18636v2">LaQual: An Automated Framework for LLM App Quality Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Representing a new paradigm in software distribution, LLM app stores are rapidly emerging, offering users diverse choices for content generation, coding assistance, education, and more. However, current ranking and recommendation mechanisms in LLM app stores predominantly rely on static metrics, such as user interactions and favorites, making it challenging for users to efficiently identify high-quality apps. At the same time, current academic research focuses on specific vertical fields and lacks a general, automated evaluation framework applicable to the diverse LLM app ecosystem. To address the above challenges, we present LaQual, an automated framework for LLM app quality evaluation. LaQual integrates three key stages: (1) LLM app labeling and hierarchical classification for precise scenario mapping; (2) static indicator evaluation using time-weighted user engagement and functional capability indicators to filter low-quality apps; and (3) dynamic scenario-adapted evaluation, where an LLM generates scenario-specific evaluation metrics, scoring criteria, and tasks for comprehensive quality evaluation. Experiments on a mainstream LLM app store demonstrate the effectiveness of LaQual. Its automated scores show high consistency with human judgments. Through effective screening, LaQual can reduce the candidate LLM app pool by 66.7% to 81.3%. User studies further validate its significant outperformance over baseline systems, particularly in comparison efficiency (mean 5.45 vs. 3.30) and value of explanatory information (4.75 vs. 2.25). These results demonstrate that LaQual provides a scalable, objective, and user-centric solution for high-quality discovery and recommendation of LLM apps in real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11700v1">CompRank: Efficient LLM Reranking via Token-Level Compression and Decoding-Free Scoring</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Large language model (LLM) rerankers have become an important component of modern retrieval and retrieval-augmented generation pipelines, but their high computational cost limits their applicability to long candidate lists. In this paper, we propose \textbf{CompRank}, a token-efficient reranking framework that reduces redundant computation by aligning reranker design with the sparsity of ranking signals. CompRank decouples document representations from candidate order and query context, enabling reusable document-side states; applies segment-wise token compression to reduce query--document interaction cost; and introduces a CopyNet-style objective that directly aligns attention-based document scoring with training supervision. Experiments on seven BEIR datasets show that CompRank achieves strong reranking performance while retaining only 10.2\% of document tokens, reaching an average NDCG@10 of 39.2 compared with 39.7 under full-token attention. Further scaling experiments on TREC-COVID show that CompRank remains stable when evaluated on candidate lists of up to 500 documents after training on 30-document lists, while achieving $4.9\times$--$9.5\times$ end-to-end speedup over generation-based listwise reranking and approximately $1.3\times$ speedup over the full-token CompRank variant. These results suggest that token-level compression and decoding-free attention scoring provide an effective path toward scalable LLM-based reranking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.10749v2">AttriGuard: Defeating Indirect Prompt Injection in LLM Agents via Causal Attribution of Tool Invocations</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 Accepted by USENIX Security 2026
    </div>
    <details class="paper-abstract">
      LLM agents are highly vulnerable to Indirect Prompt Injection (IPI), where adversaries embed malicious directives in untrusted tool outputs to hijack execution. Most existing defenses treat IPI as an input-level semantic discrimination problem, which often fails to generalize to unseen payloads. We propose a new paradigm, action-level causal attribution, which secures agents by asking why a particular tool call is produced. The central goal is to distinguish tool calls supported by the user's intent from those causally driven by untrusted observations. We instantiate this paradigm with AttriGuard, a runtime defense based on parallel counterfactual tests. For each proposed tool call, AttriGuard verifies its necessity by re-executing the agent under a control-attenuated view of external observations. Technically, AttriGuard combines teacher-forced shadow replay to prevent attribution confounding, hierarchical control attenuation to suppress diverse control channels while preserving task-relevant information, and a fuzzy survival criterion that is robust to LLM stochasticity. Across four LLMs and two agent benchmarks, AttriGuard achieves 0% ASR under static attacks with negligible utility loss and moderate overhead. Importantly, it remains resilient under adaptive optimization-based attacks in settings where leading defenses degrade significantly.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11690v1">Beyond Per-Token Pricing: A Concurrency-Aware Methodology for LLM Infrastructure Cost Estimation</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 26 pages, 9 figures. Code: https://github.com/pChitral/vllm-cost-meter
    </div>
    <details class="paper-abstract">
      Every public LLM cost calculator we surveyed treats GPU utilization as a fixed input -- entered by the user, baked in as a preset, or silently assumed at 100% -- never measured against the operator's actual load. We show that this assumption is the dominant source of error: on identical H100 hardware, effective cost spans \$0.21 to \$15.25 per million output tokens, an underutilization penalty of 2.5-24x across low-to-moderate enterprise loads (1-10 rps) and up to 36.3x near idle -- driven by one operator-controlled variable, offered request rate lambda, which sets in-flight concurrency via Little's Law and which no open-source calculator exposes. Because calculators take utilization as a user-supplied input, any utilization-naive estimate understates true cost by exactly 1/U, systematically mispricing self-hosting -- most severely over-selling it for low-traffic workloads. We propose a measurement methodology that parameterizes the relationship as C_eff = f(H, M, Q, lambda, L), validate it with 42 benchmarks across dense, ultra-sparse MoE, and sparse MoE models, and release vllm-cost-meter, an open-source cost meter that attaches to a live vLLM server and reports real \$/M-tokens against the operator's own traffic. We further show that FP8 quantization benefits the MoE architectures we tested roughly 2.2-2.4x more than the dense model (+69 to +74% vs. +31% peak throughput; n=3, broader validation needed), and our data are consistent with active parameter count, not total model size, being a primary predictor of saturation economics. To rule out single-hardware confounding we repeat the core sweep on A100 80GB PCIe (56 runs): the load-driven spread reproduces at 7.0-11.4x, the active-parameters ordering survives at FP8, and the dense-FP8 advantage inverts on silicon without native FP8 tensor cores -- a hardware-conditional caveat the framework already accommodates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11686v1">Layer-Isolated Evaluation: Gating the Deterministic Scaffold of a Production LLM Agent with a No-LLM, Regression-Locked Test Harness</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 12 pages, 2 figures, 5 tables
    </div>
    <details class="paper-abstract">
      End-to-end task-success is the dominant way to evaluate LLM agents, but one aggregate number tells you that an agent regressed, not where. We present layer-isolated evaluation: a deployed ordering agent is decomposed into a fixed taxonomy of layers (ontology, intent, routing, decomposition, escalation, safety, memory, and cross-cutting envelope/defense), each exercised by its own assertion slice in a deterministic, no-LLM "pure" mode. The pure suite (238 cases across 23 slices; 225 run in 2.39 s, ~10 ms/case) runs in CI on every change against a locked per-slice baseline. We validate by controlled regression injection, degrading one layer at a time across seven non-safety layers. The effect we did not design in is masking: the aggregate pass-rate barely moves (-1.7 to -5.9 pp for six local regressions), while the matching slice craters (-25 to -91 pp). A layer's slice reacting to its own fault is partly by construction; the measured results are (i) the aggregate masking and (ii) that damage stays off the other slices: the injected layer's slice is the single worst-hit in 5 of 7 cases and top-3 in 7 of 7 (mean rank 1.29 of 19). Localization replicates on a second, structurally different tenant (Starbucks SG): all seven matching slices crater, so it is not a single-catalog artifact. We position it as a concrete, deterministic instantiation of the component-level evaluation EDDOps prescribes but leaves unimplemented, with CheckList as ancestor and as the deterministic mirror image of whole-workflow stochastic mutation testing. Our contributions: (a) a fully decomposed, sub-second, no-LLM per-layer harness for a production agent, (b) a coverage-honesty test-adequacy criterion that refuses to score an unexercised layer, and (c) the regression-injection demonstration that per-slice baseline-locked gates localize regressions an aggregate metric masks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.00945v2">Neural FOXP2 -- Language Specific Neuron Steering for Targeted Language Improvement in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      LLMs are multilingual by training, yet their lingua franca is often English, reflecting English language dominance in pretraining. Other languages remain in parametric memory but are systematically suppressed. We argue that language defaultness is governed by a sparse, low-rank control circuit, language neurons, that can be mechanistically isolated and safely steered. We introduce Neural FOXP2, that makes a chosen language (Hindi or Spanish) primary in a model by steering language-specific neurons. Neural FOXP2 proceeds in three stages: (i) Localize: We train per-layer SAEs so each activation decomposes into a small set of active feature components. For every feature, we quantify English vs. Hindi/Spanish selectivity overall logit-mass lift toward the target-language token set. Tracing the top-ranked features back to their strongest contributing units yields a compact language-neuron set. (ii) Steering directions: We localize controllable language-shift geometry via a spectral low-rank analysis. For each layer, we build English to target activation-difference matrices and perform layerwise SVD to extract the dominant singular directions governing language change. The eigengap and effective-rank spectra identify a compact steering subspace and an empirically chosen intervention window (where these directions are strongest and most stable). (iii) Steer: We apply a signed, sparse activation shift targeted to the language neurons. Concretely, within low to mid layers we add a positive steering along the target-language dominant directions and a compensating negative shift toward the null space for the English neurons, yielding controllable target-language defaultness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11675v1">Lung-R1: A Knowledge Graph-Guided LLM for Pulmonary Diagnostic Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Diagnosing pulmonary diseases requires integrating heterogeneous evidence amid phenotypic variability and cross-disease overlap. Although large language models (LLMs) have shown progress on pulmonary knowledge question answering (QA) and information-processing tasks, reliable pulmonary diagnosis requires patient-specific, relation-aware reasoning over electronic medical record (EMR) evidence rather than isolated knowledge recall. We define this gap between pulmonary knowledge and case-level diagnostic reasoning as the Pulmonary Knowledge-to-Diagnosis Gap. To address it, we introduce LungKG, the first structured pulmonary knowledge graph for diagnostic knowledge organization and record-grounded reasoning. LungKG contains 59,038 nodes and 164,308 edges across 15 entity types and 112 relation types, serving as both a reusable pulmonary knowledge resource and the foundation for LungKG-guided model adaptation. Built on LungKG, we propose Lung-R1, a LungKG-guided pulmonary LLM trained through KG-constrained reasoning-chain construction and KG-guided reinforcement learning. In a 20-system evaluation, Lung-R1-14B achieves state-of-the-art performance across Choice, Pulmonary-QA, and EMR Diagnosis, reaching an EMR Diagnosis score of 4.3583 and surpassing the strongest non-Lung-R1 baseline by 0.1476 points. These results demonstrate the value of LungKG-guided training for EMR-based pulmonary diagnosis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12474v1">SAIGuard: Communication-State Simulation for Proactive Defense of LLM Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      LLM-based multi-agent systems (MAS) solve complex tasks through inter-agent collaboration, but their communication-driven nature also allows security risks to spread across agents and trigger system-wide failures. Existing MAS defenses mainly follow a reactive paradigm after execution by detecting and isolating harmful agents, which may cause irreversible damage and degrade collaborative utility. To address this, we propose a proactive defense framework for MAS security, namely a Simulation-aware Interception Guard (SAIGuard). SAIGuard performs communication-state simulation over the MAS interaction graph, estimates the impact of incoming messages on local agent states and the global MAS state, and detects risky messages via reconstruction deviations from benign communication patterns. Instead of isolating agents, SAIGuard sanitizes or regenerates suspicious messages before it propagation into system. Experiments across diverse topologies and attack scenarios show that SAIGuard reduces attack success rates while maintaining MAS utility, outperforming reactive defenses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11672v1">Can Open-Source LLM Agents Replace Static Application Security Testing Tools? An Empirical Assessment</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 Keywords: Agentic AI, Cybersecurity, Large Language Models, Static Application Security Testing, Model performance evaluation
    </div>
    <details class="paper-abstract">
      This paper explores the value of agentic AI tools for cybersecurity purposes. We evaluate the efficacy of a general-purpose GenAI Large Language Model- (GenAI-) based agent when powered by three different Ollama-hosted general-purpose open source models. We assess each agent's performance using precision, recall, false positive count, and a calculated composite score based upon the interplay of the captured metrics, against the baseline performance of an existing, vetted Static Application Security Testing (SAST) tool, Bandit. Our findings refute the notion that a modern open-source GenAI LLM-based agent is currently suitable for the specialized task of SAST scanning under realistic conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07506v2">Judging Against the Reference: Uncovering Knowledge-Driven Failures in LLM-Judges on QA Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 Under review, 21 pgs, 11 figures, 7 tables
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) are increasingly used as automatic judges for question answering (QA) and other reference-conditioned evaluation tasks, little is known about their ability to adhere to a provided reference. We identify a critical failure mode of such reference-based LLM QA evaluation: when the provided reference conflicts with the judge model's parametric knowledge, the resulting scores become unreliable, substantially degrading evaluation fidelity. To study this phenomenon systematically, we introduce a controlled swapped-reference QA framework that induces reference-belief conflicts. Specifically, we replace the reference answer with an incorrect entity and construct diverse pairings of original and swapped references with correspondingly aligned candidate answers. Surprisingly, grading reliability drops sharply under swapped references across a broad set of judge models. We empirically show that this vulnerability is driven by judges' over-reliance on parametric knowledge, leading judges to disregard the given reference under conflict. Finally, we find that this failure persists under common prompt-based mitigation strategies, highlighting a fundamental limitation of LLM-as-a-judge evaluation and motivating reference-based protocols that enforce stronger adherence to the provided reference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11648v1">Dummy Backdoor as a Defense: Removing Unknown Backdoors via Shared Internal Mechanisms for Generative LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Backdoor attacks pose a serious threat to the safety and reliability of Large Language Models (LLMs), as they cause models to behave normally on clean inputs while producing attacker-specified responses when hidden triggers are present. Removing such unknown backdoors is particularly challenging when the defender does not know the backdoor attack types or the internal mechanisms formed through backdoor training. In this work, we propose a simple but effective backdoor removal method based on shared internal mechanisms across different backdoors. First, we show that different backdoors with the same task (attack objective) induce similar trigger-activated changes in the internal activations. Motivated by this observation, our method intentionally embeds a backdoor with a known trigger (\emph{dummy backdoor}) and then removes it through further fine-tuning on dummy-triggered inputs paired with clean responses. Since the dummy backdoor and the unknown backdoor can rely on shared internal mechanisms, removing the dummy backdoor also reduces the effect of the unknown backdoor. We evaluate our method on three backdoor attack types across multiple model families. Experimental results show that our method substantially reduces the attack success rate of the unknown backdoor while preserving model utility, outperforming representative existing defense methods in both backdoor removal effectiveness and utility preservation. These findings suggest that a defender-controllable backdoor can serve as a helpful proxy for mitigating unknown backdoors in generative LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.03855v2">A Sensitivity Analysis of Multi-Event Audio Grounding in Audio LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 6 pages, Accepted to Interspeech 2026
    </div>
    <details class="paper-abstract">
      Audio LLMs have shown a strong ability to understand audio samples, yet their reliability in complex acoustic scenes remains under-explored. Unlike prior work limited to small scale or less controlled query construction, we present a large-scale evaluation of event grounding and false alarms as auditory scene complexity increases. Using 71K AudioCapsV2 clips, we extract normalized (source, attribute) events and build two query types: present-event queries for ground-truth detection and absent-event queries to probe hallucinations, using similarity-filtered negative sampling in an audio-aligned text embedding space. We evaluate four SOTA Audio LLMs with 12 prompt variants over 500K yes/no queries per model. Across models, increasing event count consistently lowers true-positive rate and raises false-positive rate, while prompts induce a strong trade-off between the two. Our confidence analysis shows that models become more uncertain on multi-event audio, revealing room for improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11640v1">TAROT: Task-Adaptive Refinement of LLM-prior Graphs for Few-shot Tabular Learning</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Few-shot tabular learning provides a cost-effective approach for real-world applications where annotation is costly and collecting sufficient samples for new tasks is difficult. Existing Traditional and LLM-based methods have demonstrated effectiveness in few-shot scenarios. However, traditional methods need additional training on unlabeled or generated data, which incur significant computational overhead. In addition, LLM-based methods that directly feed raw tabular data into LLMs raise privacy and compliance concerns. More importantly, both paradigms largely overlook the semantic relationships between features, which provide structural and semantic prior for constructing a semantic graph. Semantic graph is essential for modeling meaningful feature interactions in few-shot scenarios. In this paper, we propose TAROT, a GNN-based framework that encodes the structural and semantic prior by constructing and refining a task-adaptive semantic graph from this prior, thereby improving predictive performance in few-shot tabular learning. TAROT first encodes heterogeneous tabular data into unified node semantic representations via a Unified Semantic Tabular Node Encoder (USTNE). Then, it prompts LLMs to infer the semantic relationship between features based on the task description and feature names to construct a semantic graph. To mitigate structural noise introduced by the hallucination of LLMs, TAROT introduces Task-adaptive Semantic Graph Refinement that prunes spurious or task-unrelated edges and adds missing task-related ones, aligning the graph structure with the downstream objective. Finally, a GNN performs message passing over the refined graph to capture task-related semantic dependencies for prediction. Extensive experiments on various few-shot tabular learning benchmarks demonstrate the superior performance of TAROT, establishing it as a state-of-the-art approach in this domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11635v1">Are LLMs Bad at Moral Reasoning?</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      For highly capable AI systems to operate safely in dynamic, open-ended environments, they must be able to identify, understand, and respond to moral reasons for action, and constrain their behaviour accordingly. A growing body of research aims to evaluate this capacity -- moral competence -- in today's most capable AI systems, recently reaching broadly pessimistic conclusions. One of the most ambitious such papers collects gold-standard human-authored rubrics for evaluating moral reasoning in 1,000 cases, and benchmarks frontier AI models against those rubrics, with underwhelming results. In this paper, we argue that the MoReBench dataset can be redeployed to give a much more optimistic picture of LLMs' moral reasoning (an essential part of moral competence). We show that if, instead of scoring LLMs' responses to these cases against these rubrics, we instead give the LLMs the same task given to humans -- to generate scoring rubrics for the moral analysis of particular cases -- the rubrics they generate are both better calibrated to the human rubrics than their open-ended responses, and, where they differ, plausibly reflect nothing more than the vast dimensionality of most moral problems, as well as highlighting some human departures from the "rubric for creating rubrics". Taking these points into consideration, the MoReBench dataset suggests that LLMs are significantly more capable at moral reasoning than was previously believed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04710v2">Steering the Noise: Turning Random Perturbations into Effective Descent for Memory-Efficient LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 12pages, 6figures
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) achieves strong performance but is often limited by the memory overhead of backpropagation. Zeroth-order (ZO) optimization avoids this overhead by estimating gradients through forward passes alone, yet it typically converges slowly because random Gaussian perturbations yield high-variance gradient estimates in high-dimensional parameter spaces. In this paper, we propose a plug-and-play framework that turns random perturbations into more effective descent directions. The key idea is to draw a small pool of candidate perturbations, evaluate their loss values, and then select or combine those that are best aligned with the optimization objective. We develop two instantiations of this idea: MeZO-GV, which forms a guiding vector from the contrast between low-loss and high-loss perturbation groups, and MeZO-Greedy, which keeps the single best perturbation within a fixed evaluation budget. We theoretically show that both strategies yield a larger per-step reduction in the objective than standard ZO estimation, leading to improved convergence rates. Experiments on LLMs of different scales and architectures confirm that the proposed methods integrate naturally with existing ZO optimizers and consistently improve convergence speed and task accuracy. On OPT-13B, our approach outperforms all ZO baselines across 11 benchmarks and exceeds gradient-based methods on 9 of them, while retaining the memory efficiency of forward-only optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10120v2">MetaPlate: Counterfactual-Guided RAG-LLM Tool for Personalized Food Recommendation and Hyperglycemia Prevention</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Postprandial hyperglycemia is a key risk factor for metabolic disorders; however, existing dietary guidance is often static, impractical, and insufficiently personalized, providing recommendations that are difficult to follow or not impactful. While recent advances leverage continuous glucose monitoring (CGM) and machine learning to predict glycemic responses, these approaches are largely predictive and lack actionable guidance. Moreover, recommendation systems are often misaligned with user goals and require extensive input. We present MetaPlate, a counterfactual explanation (CF) guided, context-aware decision-support framework that generates personalized meal recommendations to mitigate postprandial glucose excursions in healthy adults. MetaPlate integrates multimodal data, including CGM readings, wearable-derived physiological signals, and user-provided meal inputs from $25$ individuals to model pre-meal context. A machine learning model predicts glucose response, while a CF optimization module adjusts meal composition modifying macronutrient amounts to maintain glucose levels within a target range ($\leq 140$ mg/dL). An LLM-based retrieval-augmented generation (RAG) layer enhances interpretability by producing human-readable recommendations using constrained search of the USDA food database. We evaluate MetaPlate via a structured expert-in-the-loop assessment with registered dietitians (RDs), comparing performance before and after prompt refinement. Results show improvements in meal realism, portion suitability, and recommendation likelihood, with expert feedback indicating a shift from clinically implausible outputs to actionable, contextually appropriate recommendations. Our findings emphasize the importance of domain knowledge and structured constraints in LLM-driven systems and highlight the potential of MetaPlate as a real-time personalized dietary decision-support tool.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11592v1">Defense Against Prompt Inversion Attacks: An Information-Theoretic Approach for LLM Collaborative Inference</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 Preprint. 33 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Collaborative edge-cloud inference enables resource-constrained devices to leverage large language models (LLMs) by offloading partial computation to cloud servers. However, transmitting intermediate activations exposes sensitive user prompts to prompt inversion attacks, where an adversary reconstructs the original input from shared representations. Existing defenses rely largely on heuristic perturbations or empirical tuning, offering limited theoretical understanding of privacy leakage and its interaction with utility and latency constraints. We propose an information-theoretic defense framework for prompt inversion in collaborative LLM inference. Our approach learns privacy-preserving representations by explicitly minimizing the mutual information between intermediate activations and the input prompt while maintaining task utility under computational constraints. We derive theoretical guarantees on prompt reconstruction error, characterize fundamental privacy-utility tradeoffs, and establish token-level accuracy bounds for downstream inference. We then propose a novel defense based on privacy adapters implemented via low-dimensional information bottlenecks. Extensive experiments across multiple settings demonstrate that our method achieves superior privacy-utility-latency tradeoffs compared to existing defenses (up to 35% reduction in attack success), providing a principled foundation for private and efficient collaborative LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11583v1">Beyond the Golden Teacher: Enhancing Graph Learning through LLM-GNN Co-teaching</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 Code: https://github.com/llmgnncoteaching/LLM-GNN-Coteaching
    </div>
    <details class="paper-abstract">
      Text-attributed graphs (TAGs) underlie real-world applications such as citation networks, social media, and e-commerce. Few-shot graph learning on TAGs is hard: with only a handful of labels per class and the rest of the graph unannotated, neither GNNs nor LLMs can learn well on their own. GNNs read topology and fail on cold nodes; LLMs read text and fail on text-ambiguous nodes. Existing LLM-GNN methods all follow the same recipe: designate one model as the golden teacher and use its outputs (e.g., features or pseudo-labels) to supervise the other. We argue this golden-teacher assumption breaks under sparse supervision: neither model is golden, and treating either as such transfers its blind spots into the student. We therefore ask: can we avoid designating either model as the golden teacher, and still perform effective graph learning? We answer with LLM-GNN Co-Teaching, a bidirectional co-teaching framework in which neither model is fixed as teacher. The GNN and LLM exchange their most confident pseudo-labels under an architecture-specific small-loss criterion, and both update every round. Supervision is then mined from the trajectory: whenever a node moves from cross-model contradiction at round t to cross-model agreement at round t+1, the LLM's two answers on the same input form a preference pair (old contradicting self < new peer-endorsed self) for DPO training. We call this Round-based Pseudo-Label Preference Optimization (RPL-PO). On six benchmarks, LLM-GNN Co-Teaching consistently outperforms GNN-as-Judge and all prior methods, with absolute 3-shot gains of 7.86% on Cora and 7.73% on ogbn-arxiv; improvements carry over to 5-shot and to zero-shot cross-dataset transfer. Error-structure analysis further shows that abandoning the golden-teacher assumption substantially improves the LLM's graph learning capability on challenging samples.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11562v1">GraphInfer-Bench: Benchmarking LLM's Inference Capability on Graphs</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 Code: https://github.com/graphinfer/GraphInfer-Bench ; Dataset: https://huggingface.co/datasets/graphinfer/graphinfer
    </div>
    <details class="paper-abstract">
      Graph analysis underlies many applications whose answers cannot be looked up in a single record or retrieved along a path: laundering rings, drug repurposing, user preference, and scientific theme are all inferred from a node together with its neighbourhood. We introduce GraphInfer-Bench, a benchmark for whether LLMs can perform this graph inference: producing an open-ended answer that no single node supports and no path retrieves. Existing graph-QA protocols cannot test this capability: algorithm simulation, node classification, single-node description, KG-QA, and GraphRAG all admit answers retrievable from one node or along a path. GraphInfer-Bench defines five tasks along Description (what a region is) and Comparison (how regions differ), each constructed so the ground truth lives in no single node. The release contains 42,000 samples across six real-world graphs, produced automatically and screened by a four-layer quality-control protocol. We evaluate four method families against the same tasks: graph-token alignment models, zero-shot frontier closed-source LLMs, Graph2Text supervised fine-tuning, and plain GNNs as a structural reference. No method family closes the gap. Graph-token alignment partially handles description tasks (relational, theme) but collapses on comparison tasks. Frontier LLMs lead on outlier detection and community partition among LLM-based methods but lag on masked-node prediction. Graph2Text SFT is the strongest LLM-based method on the description side yet falls behind frontier LLMs on comparison. Across every task, plain GNNs match or beat the strongest LLM-based row, with the largest margin on community detection. GraphInfer-Bench surfaces graph inference as an open capability gap rather than a property of any one architecture.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11560v1">LLMs+Graphs: Toward Graph-Native, Synergistic AI Systems</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 10 pages, Accepted at PAKDD 2066 Tutorial
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have advanced rapidly, but their limitations in structured and multi-hop reasoning underscore the need for graph-native, synergistic artificial intelligence (AI) systems. Graph-structured data underpins critical applications across social, biological, financial, transportation, web, and knowledge domains, making it essential to understand how LLMs can leverage graph computation for grounded, context-rich inference. Three complementary synergies are emerging: LLMs augmented with graph computation for retrieval and reasoning; bidirectional integration between LLMs and knowledge graphs (KGs), where LLMs support KG construction and curation while KGs enforce semantic constraints and factual consistency; and AI agents strengthened by graph algorithms for planning, decision making, and multi-step reasoning. In parallel, LLMs introduce new capabilities for graph data management and graph machine learning (ML) through natural language interfaces and hybrid LLM-graph neural network (GNN) pipelines. This tutorial synthesizes the algorithms, systems, and design principles driving these converging directions, offering data science and data mining researchers a unified perspective on integrating LLMs, graph data management, graph mining, graph ML, and agentic computation into next-generation graph-native AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10198v2">Density Ridge Selective Prediction for LLM and VLM Hallucination Detection under Calibration Label Scarcity</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Hallucination detection in large language and vision-language models is increasingly framed as selective prediction, where a detector assigns a confidence score and abstains when confidence is low. Unsupervised sampling detectors (Semantic Entropy) avoid labels but plateau in quality, while supervised probes attain stronger in-distribution scores yet degrade sharply when calibration labels are scarce. We recover the response manifold of an LLM as the density ridge of a kernel density estimate built on a six-dimensional kinematic feature map of hidden state generation trajectories. A test generation is scored by the negated Euclidean distance from its projected feature point to the nearest ridge vertex, yielding a low-dimensional geometric skeleton of the stochastic output distribution. We evaluate against Semantic Entropy, topological methods, and log-probability on six QA benchmarks (HaluEval-QA, TriviaQA, GSM8K, POPE, ScienceQA, A-OKVQA) using eight text and vision LLMs in a deliberately label-scarce protocol ($n_{\text{cal}}{=}200$ queries, $N{=}5$ generations). Our ridge-based score beats on AUROC with 5-20 points gain, while demonstrating tempered degradation under calibration-label scarcity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.16456v3">GPO: Learning from Critical Steps to Improve LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 39th Conference on Neural Information Processing Systems (NeurIPS 2025)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used in various domains, showing impressive potential on different tasks. Recently, reasoning LLMs have been proposed to improve the \textit{reasoning} or \textit{thinking} capabilities of LLMs to solve complex problems. Despite the promising results of reasoning LLMs, enhancing the multi-step reasoning capabilities of LLMs still remains a significant challenge. While existing optimization methods have advanced the LLM reasoning capabilities, they often treat reasoning trajectories as a whole, without considering the underlying critical steps within the trajectory. In this paper, we introduce \textbf{G}uided \textbf{P}ivotal \textbf{O}ptimization (GPO), a novel fine-tuning strategy that dives into the reasoning process to enable more effective improvements. GPO first identifies the `critical step' within a reasoning trajectory - a point that the model must carefully proceed to succeed at the problem. We locate the critical step by estimating the advantage function. GPO then resets the policy to the critical step, samples the new rollout and prioritizes the learning process on those rollouts. This focus allows the model to learn more effectively from pivotal moments within the reasoning process to improve the reasoning performance. We demonstrate that GPO is a general strategy that can be integrated with various optimization methods to improve reasoning performance. Besides theoretical analysis, our experiments across challenging reasoning benchmarks show that GPO can consistently and significantly enhance the performance of existing optimization methods, showcasing its effectiveness and generalizability in improving LLM reasoning by concentrating on pivotal moments within the generation process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12754v1">LLMs Can Better Capture Human Judgments--With the Right Prompts</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Are large language models (LLMs) bad at capturing human judgment? Two commonly stated limitations are that LLMs fail to capture full distributions of responses, and that their judgments are unstable across wording variations. We demonstrate simple prompting strategies that mitigate these limitations. Across two datasets--a U.S.-representative set of 144 moral scenarios and 38 moral beliefs from the International Social Survey Programme's Family and Changing Gender Roles module covering 32 countries--we show how simple elicitation techniques help improve AI-human alignment. First, prompting models to report standard deviations and response proportions recovers the full range of human responses better than common strategies. Second, ensuring scenarios are clear to human participants--as reflected in human confusion ratings--boosts model alignment, and LLMs can track human confusion ratings. At the same time, we find that LLMs' estimates of their own error are poorly calibrated, though they can predict human variability relatively well. These results suggest that asking better questions to LLMs can yield better answers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.04021v3">Prism: Cost-Efficient Multi-LLM Serving via GPU Memory Ballooning</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 OSDI'26
    </div>
    <details class="paper-abstract">
      Inference providers must maintain availability for many LLMs, including low-volume but essential models, making resource efficiency increasingly important as token prices fall. Analysis of production traces reveals a dynamic bursty-group pattern in which sets of models become active together and shift over time; existing space- and time-sharing approaches lack principled mechanisms to adapt to this variability, forcing trade-offs between SLO adherence and efficiency. We observe that elastic memory allocation can unify spatial and temporal sharing. Based on this insight, we have developed Prism, a memory-centric LLM co-serving framework that applies memory ballooning to reclaim memory across models and support both forms of sharing under a single scheme. Prism's balloon driver, referred to as kvcached, has been open-sourced at https://github.com/ovg-project/kvcached, and deployed in production environments across 10K+ GPUs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12731v1">Normative Robustness as a Frontier for Non-Verifiable Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      As LLMs increasingly serve in advisory and deliberative roles, users rely on them for non-verifiable reasoning in domains lacking objective ground truths. However, traditional evaluations of LLM reasoning focus almost exclusively on fact-based domains, such as mathematics and science, leaving uncertainty over whether and to what degree models can handle ambiguous, subjective, or value-laden problems over time. To address this concern, we propose moral reasoning as a paradigmatic subdomain of non-verifiable reasoning. We define moral robustness as a model's capacity to exhibit sound moral reasoning across time and contexts, and we introduce a scalable, adversarial, multi-turn evaluation framework to empirically measure this capability. We simulate 48,000 user-agent moral deliberations across four frontier LLMs, varying premise relevance, premise order, conversation duration, and the user's stated moral view. We find that models successfully ignore morally-irrelevant distractors, but shift their reasoning by up to 6.5%, on average, towards the user's stated preferred moral view, and varying their reasoning depending on factors such as order (altering moral judgments by order in 13-22% of the cases) and duration (altering moral judgments between single-turn and multi-turn in 10-24% of the cases). Our analysis indicates that models tailor not just their final verdicts but their underlying justifications to align with a user's moral viewpoint - a failure mode we characterize as moral deliberative sycophancy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12730v1">Rethinking Psychometric Evaluation of LLMs: When and Why Self-Reports Predict Behavior</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 Accepted as an Oral (Contributed Talk) at the ICML 2026 Workshop on Combining Theory and Benchmarks (CTB)
    </div>
    <details class="paper-abstract">
      Anticipating LLM behavioral tendencies from low-cost psychometric probes is critical for safe deployment, but only if self-reports (SR) reliably predict behavior. Recent work documented substantial SR-behavior dissociation in LLMs, but relied on broad personality traits (Big 5) that predict specific behaviors weakly, even in humans. Furthermore, the isolation of conversational sessions combined with weak context matching left open whether LLMs truly lack coherence or whether the conditions needed to detect such coherence were not met. We contrast Big 5 with the Theory of Planned Behavior (TPB), which measures intention targeted to a specific behavior and predicts human behavior substantially better than broad traits. We run experiments across four behavioral tasks and 11 frontier LLMs, while also varying session context and identity induction. We find that SR-behavior coherence exists but is selective. 1) Within a shared conversation, the Theory of Planned Behavior reaches human-level coherence; Big 5 does not. 2) Across separate conversations, coherence survives only for behaviors anchored outside the immediate prompt, such as implicit bias shaped by training, and collapses when behavior is strongly primed by context, as with sycophancy. 3) Persona prompting makes self-reports more consistent across conversations, but does not bring behavior into alignment. These findings suggest that coarse personality frameworks, such as Big 5 may not be the best tools for testing deployment behavior. More task- and behavior-specific instruments are needed, and even these must be evaluated across tasks and contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12702v1">Deployment-Centered Evaluation: Predicting Query-Level Rejection Risk in a Clinical LLM System</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly integrated into clinical systems, making it essential to evaluate the real-world utility of these systems. However, static benchmarks tend to measure correctness rather than user acceptance, aggregate performance across queries, and require densely annotated datasets -- leading to major blind spots for evaluating clinical systems. In this work, we perform a deployment-centered evaluation of an LLM system embedded within electronic health records at an academic medical center, where user feedback is sparse but closely reflects the deployment conditions. Specifically, we train a pre-response classifier that estimates the risk that a future interaction will result in the user rejecting the LLM response, based on query content and deployment-specific context available before generation. We conduct a prospective analysis of our model over 4.5 months of user feedback, finding that our prediction model achieves an AUROC of 0.719. Further, we estimate the benefit of such predictions in two downstream use cases (guardrail triggering and abstention). Our key conceptual insight is that making use of deployment-specific context (i.e., the provider type, department name, language model used for response), as opposed to only query content, improves the ability to predict whether the user will reject the system output. Altogether, our empirical case study demonstrates the feasibility of predicting user rejection using deployment-specific context, opening the door to targeted guardrails.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12699v1">LLM-Powered Personalized Glycemic Assessment in Type 2 Diabetes with Wearable Sensor Data</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 The 14th IEEE International Conference on Healthcare Informatics, 2026
    </div>
    <details class="paper-abstract">
      Type 2 Diabetes (T2D) poses an increasing global health threat, demanding effective glycemic assessment to support personalized and improved diabetes care. Wearable sensors such as continuous glucose monitors (CGM) and fitness trackers offer many valuable insights for glycemic assessment. However, effectively analyzing these data requires integration with essential individual-level context. Existing methods are often based on traditional machine learning (ML) and rely primarily on historical blood glucose measurements and overlook personalized information, which limits their performance across diverse diabetes populations. Recent advances in large language models (LLMs) have demonstrated their ability to integrate diverse data modalities while modeling sequential dependencies, motivating the exploration of their potential for personalized glycemic assessment. In this paper, we propose GlyLLM, an LLM-powered framework for modeling CGM-based glycemic dynamics through the integration of wearable sensor data and structured metadata. GlyLLM can leverage the extensive prior knowledge of pre-trained LLMs and achieve sensor-text semantic abstraction at decision time. Experiments on two related tasks on the AI-READI dataset demonstrate that our model outperforms traditional ML methods by an average of 13.66\% in Root Mean Squared Error (RMSE) for glucose forecasting and 13.08\% in Area Under the Receiver Operating Characteristic (AUROC) for diabetes categorization. Additionally, our ablation study shows that diabetes surveys and biometric tests are more critical than other health information for glycemic assessment. Our work presents a promising step toward harnessing the power of LLMs to advance personalized glycemic assessment in T2D care.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12657v1">TrajGenAgent: A Hierarchical LLM Agent for Human Mobility Trajectory Generation</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 14 pages, 2 figures, 8 tables. Accepted by the 27th IEEE International Conference on Mobile Data Management (MDM 2026)
    </div>
    <details class="paper-abstract">
      Human mobility data is important for transportation, urban planning, and epidemic control, but large-scale trajectory collection is often costly and privacy-constrained, motivating realistic synthetic trajectory generation. Existing LLM-based generators typically rely on either prompt engineering, which preserves zero-shot reasoning but lacks fine-grained spatiotemporal grounding, or trajectory-level fine-tuning, which improves statistical precision but incurs substantial computational cost and may weaken general reasoning. We propose TrajGenAgent, a semantic-aware hierarchical LLM-agent framework for human mobility trajectory generation without model fine-tuning. TrajGenAgent uses a two-stage orchestrator-worker design: an LLM first synthesizes an individual- and weekday-conditioned activity chain from historical evidence via in-context learning, and a deterministic workflow then grounds each activity into a complete visit using personalized POI retrieval, distance-aware location selection, kinematics-aware travel-time propagation, and LLM-based duration estimation. To evaluate realism beyond aggregate spatiotemporal statistics, we introduce an anomaly-detection-based evaluation framework using two complementary detectors to assess behavioral and semantic plausibility. Experiments on benchmark and large-scale simulation datasets show that TrajGenAgent improves spatiotemporal fidelity, semantic coherence, and individual-specific behavioral realism over representative neural and LLM-based baselines, while avoiding parameter updates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.00462v4">LatentLens: Revealing Highly Interpretable Visual Tokens in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 ICML 2026 (Camera Ready)
    </div>
    <details class="paper-abstract">
      Transforming a large language model (LLM) into a vision-language model (VLM) can be achieved by mapping the visual tokens from a vision encoder into the embedding space of an LLM. Intriguingly, this mapping can be as simple as a shallow MLP transformation. To understand why LLMs can so readily process visual tokens, we need interpretability methods that reveal what is encoded in the visual token representations at every layer of LLM processing. In this work, we introduce LatentLens, a novel approach for mapping latent representations to descriptions in natural language. LatentLens encodes a large text corpus and stores contextualized token representations for each token in that corpus. Visual token representations are then compared to these contextualized representations and the top-nearest neighbor representations serve as descriptions of the visual token. We evaluate this method on 15 different VLMs, showing that commonly used methods, such as LogitLens, substantially underestimate the interpretability of visual tokens. With LatentLens instead, the majority of visual tokens are interpretable across all studied models and all layers. Qualitatively, we show that the descriptions produced by LatentLens are semantically meaningful and provide more fine-grained interpretations for humans compared to individual tokens. More broadly, our findings contribute new evidence on the alignment between vision and language representations and open up new directions for analyzing the latent representations of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.07294v4">Fin-RATE: A Real-world Financial Analytics and Tracking Evaluation Benchmark for LLMs on SEC Filings</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      With the increasing deployment of Large Language Models (LLMs) in the finance domain, LLMs are increasingly expected to parse complex regulatory disclosures. However, existing benchmarks often focus on isolated details, failing to reflect the complexity of professional analysis that requires synthesizing information across multiple documents, reporting periods, and corporate entities. Furthermore, these benchmarks do not disentangle whether errors arise from retrieval failures, generation inaccuracies, domain-specific reasoning mistakes, or misinterpretation of the query or context, making it difficult to precisely diagnose performance bottlenecks. To bridge these gaps, we introduce Fin-RATE, a benchmark built on U.S. Securities and Exchange Commission (SEC) filings and mirroring financial analyst workflows through three pathways: detail-oriented reasoning within individual disclosures, cross-entity comparison under shared topics, and longitudinal tracking of the same firm across reporting periods. We benchmark 17 leading LLMs, spanning open-source, closed-source, and finance-specialized models, under both ground-truth context and retrieval-augmented settings. Results show substantial performance degradation, with accuracy dropping by 18.60% and 14.35% as tasks shift from single-document reasoning to longitudinal and cross-entity analysis. This degradation is associated with increased comparison hallucinations, temporal and entity mismatches, and is further reflected in declines in reasoning quality and factual consistency--limitations that existing benchmarks have yet to formally categorize or quantify.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12599v1">Constrained Semantic Decompression in LLMs through Persian Proverb-Conditioned Story Generation</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Transforming a dense, abstract proverb into an engaging and morally faithful narrative requires deep cultural understanding and robust semantic grounding. We frame this problem as a \emph{constrained semantic decompression} task and study proverb-conditioned story generation as a testbed for abstraction-to-realization in large language models (LLMs). Focusing on Persian, we introduce the Proverb Aligned Narrative Dataset (PAND), pairing proverbs with human-written stories and explicit meanings. By a hybrid evaluation framework that combines human-calibrated LLM-as-a-Judge with structural metrics, we analyze model behavior across multiple prompting regimes. Our findings reveal a persistent \emph{decompression gap}: current LLMs often achieve strong surface-level fluency while failing to faithfully instantiate the underlying moral and causal structure encoded in proverbs. We further show that explicit reasoning and iterative refinement can partially mitigate these failures, suggesting that many decompression errors arise from difficulties in translating abstract meaning into narrative form rather than a complete lack of relevant knowledge. Our proposed task naturally extends to other forms of compressed cultural knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12407v1">How Seemingly Inconsequential Design Choices Dictate Performance of LLMs in Pathology</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      General-purpose large language models (LLMs) are routinely used as baselines when evaluating specialized pathology models on whole-slide images (WSIs). Because WSIs exceed contemporary model context limits, LLM baselines routinely use small, high-magnification patches processed independently via majority voting, without systematic evaluation of seemingly inconsequential design choices such as patch size, patch count, and magnification. Generalist LLMs have consistently underperformed specialized systems, reinforcing the perception that domain-specific training or architectural adaptation is necessary for pathology tasks involving WSIs. Here, we conduct a systematic factorial analysis of four input design factors: inference mode, patch size, magnification, and patch count. We demonstrate that prior studies have overstated the gap between specialized models and general-purpose LLMs by choosing non-optimized input configurations. On the MultiPathQA benchmark, switching to a single balanced configuration (large patches at lower magnification, processed jointly) raises GPT-5 from 15.1% to 39.5% on cancer-type classification (TCGA) and from 38.1% to 62.9% on organ classification (GTEx). Per-task optimization yields further gains up to 43.9% (TCGA) and 71.6% (GTEx). The same configuration generalizes to two other models and to a fully held-out CPTAC cohort, where it improves Gemini 3 Flash by 23.4 percentage points without any task-specific tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12385v1">Which Models Are Our Models Built On? Auditing Invisible Dependencies in Modern LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Modern LLM training pipelines increasingly rely on other models to generate data, filter corpora, judge outputs, and guide development decisions. These dependencies are recursive: a model may depend on an upstream artifact whose own dependencies are documented only in separate releases and artifacts. As a result, the full dependency structure is fragmented across heterogeneous public artifacts, with complexity and recursive depth far outpacing humans' ability to trace. We introduce ModSleuth, an agentic system that recursively reconstructs LLM dependency graphs from public artifacts with source-grounded evidence. We find that the primary challenge is no longer information extraction, but defining what constitutes a dependency and reconciling artifact references across inconsistent documentation. We address these challenges through a formalization that distinguishes direct and indirect dependencies, represents heterogeneous pipeline roles through operation-centered relationships, and resolves artifact identities across names, versions, and repositories. Applying ModSleuth to four public-artifact-rich LLM releases, we recover 1,060 source-verified dependencies and construct large-scale dependency graphs of modern LLM development. These graphs reveal multi-hop license obligations, train-evaluation coupling, discrepancies between released and training-time artifacts, and documentation inconsistencies that would otherwise be difficult to uncover. We release ModSleuth and the resulting dependency graphs to support transparent analysis of the increasingly complex ecosystems underlying modern LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12369v1">Should LLM Agents Decide in Social Simulations? Comparing Finite-State and LLM-Based Decision Policies</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as decision-making components in social simulations. This introduces a methodological risk: the simulation may deviate from the explicit behavioral policy defined by the researcher. In online social network (OSN) simulations, action choices shape system dynamics, interaction patterns, and model interpretability. This paper evaluates whether LLM action selectors preserve an interpretable reference policy in an OSN simulation. The reference is a finite state machine implemented as a first-order Markov model, with transition probabilities depending on the user type. The evaluation uses a synthetic network with 1,000 agents and 10,000 action decisions. Three open-weight LLMs are tested: LLaMA 3.1, GPT-OSS, and Mistral 24B. Each model is evaluated under three prompting strategies: base, guided, and probabilistic. Alignment is measured using Jensen-Shannon Divergence with Laplace smoothing, and execution time is reported. Results show that LLMs can approximate the reference policy in some configurations, but do not preserve it reliably. Alignment varies across models and prompts, and additional guidance can introduce systematic action biases. Even the best-aligned LLM configurations are several hundred times slower than direct Markov chain sampling. These findings indicate that LLM-based action selection is not a direct replacement for explicit decision policies: it can alter the intended behavior while increasing computational cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10968v2">Beyond Uniform Token-Level Trust Region in LLM Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 Project Page: https://hunyuan-cppo.github.io/
    </div>
    <details class="paper-abstract">
      Reinforcement learning with verifiable rewards (RLVR) has become standard for improving LLM reasoning. However, existing PPO-style trust-region mechanisms remain position-agnostic by enforcing uniform thresholds across all tokens independently. This pointwise treatment conflicts with autoregressive generation in two critical ways. First, uniform thresholds ignore autoregressive asymmetry. Early-stage deviations produce compounding sequence-level drift, causing static thresholds to under-regulate early divergence and excessively constrain late-stage exploration. Second, evaluating token-level divergence in isolation overlooks cumulative prefix drift, granting the same divergence allowance regardless of how far the conditioning history has already deviated from the rollout policy. To address this limitation, we propose CPPO (Cumulative Prefix-divergence Policy Optimization), a token-level masking rule that aligns updates with a finite-horizon policy-improvement bound via two coupled mechanisms. First, a position-weighted threshold imposes stricter limits at early positions whose effects persist longer, relaxing constraints for late-stage tokens. Second, a cumulative prefix budget tracks historical deviations, dynamically restricting further token-level deviation to prevent compounding errors along the prefix. Empirically, CPPO enhances training stability and significantly improves reasoning accuracy across various model scales.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.03520v2">Certifiable Safe RLHF: Semantic Grounding and Fixed Penalty Constraint Optimization for Safer LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Ensuring safety is a foundational requirement for large language models (LLMs). Achieving an appropriate balance between enhancing the utility of model outputs and mitigating their potential for harm is a complex and persistent challenge. Contemporary approaches frequently formalize this problem within the framework of Constrained Markov Decision Processes (CMDPs) and employ established CMDP optimization techniques. However, these methods exhibit two notable limitations. First, their reliance on reward and cost functions renders performance highly sensitive to the underlying scoring mechanism, which must capture semantic meaning rather than being triggered by superficial keywords. Second, CMDP-based training entails tuning dual-variable, a process that is both computationally expensive and does not provide any provable safety guarantee for a fixed dual variable that can be exploitable through adversarial jailbreaks. To overcome these limitations, we introduce Certifiable Safe-RLHF (CS-RLHF) that introduces a cost model trained on a large-scale corpus to assign semantically grounded safety scores. In contrast to the lagrangian-based approach, CS-RLHF adopts a rectified penalty-based formulation. This design draws on the theory of exact penalty functions in constrained optimization, wherein constraint satisfaction is enforced directly through a suitably chosen penalty term. With an appropriately scaled penalty, feasibility of the safety constraints can be guaranteed at the optimizer, eliminating the need for dual-variable updates. Empirical evaluation demonstrates that CS-RLHF outperforms state-of-the-art LLM model responses rendering at-least 5 times efficient against nominal and jail-breaking prompts
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12341v1">OCELOT: Inference-Leakage Budgets for Privacy-Preserving LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents increasingly act on a user's behalf -- reading personal files, calling tools, transacting with external services -- possibly leaking personally identifiable information (PII) across trust boundaries at every step. Privacy here is a property not of a single output but of an entire trajectory, and three properties make it hard: leakage is cumulative, as individually innocuous releases accumulate across honest-but-curious or colluding sinks into inferences about a protected secret; bidirectional, as a malicious observation can inject instructions that turn the agent's own reasoning model against the user; and task-dependent, as the same field is necessary for one recipient yet gratuitous for another. Per-release contextual-integrity filters, information-flow controls, and posterior-leakage monitors each address part of this but none controls cumulative, inference-based leakage at runtime. We recast agent privacy as \emph{posterior-risk control} and present OCELOT, a runtime mediator that budgets how much an adversary's belief about a secret may improve across a trajectory, rather than filtering outputs. Its mechanism, \emph{Witness-Verified Declassification}, separates judgment from trust: an untrusted, locally fine-tuned defender model inspects each candidate release and emits structured evidence -- labeled atoms and proposed declassification operators -- which a deterministic verifier audits, charging a certified min-entropy cost for the chosen variant and authorizing the least-disclosing useful release under a sink-trust-weighted budget recorded on a tamper-evident ledger. Across diverse agent benchmarks and recent defenses, OCELOT attains significantly lower leakage at higher task utility, resists adaptive injection, jailbreak, cumulative inference, and sink collusion, and adds only modest overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12291v1">Measuring Epistemic Resilience of LLMs Under Misleading Medical Context</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) now reach expert-level scores on medical licensing exams, encouraging the assumption that high scores imply safe medical judgment while patients increasingly use them for health advice. We show this assumption is fragile: when misleading context is injected into questions that LLMs originally answer correctly, they abandon the correct answer. We call the ability to maintain correct judgment under adversarial context epistemic resilience, and introduce MedMisBench to measure it. MedMisBench contains 10,932 medical question items and 48,889 misleading context-option pairs spanning medical reasoning, agentic capability, and patient-journey evaluation. Across 11 model configurations, mean accuracy falls from 71.1% on original questions to 38.0% under focused misleading context, with 51.5% attack success. The most damaging injections are formal, rule-like fabrications: authority-framed falsehoods reach 69.5% attack success and exception-poisoning claims reach 64.1%. A 14-member clinical panel from 7 countries identified serious potential harm in 38.2% of reviewed cases. MedMisBench exposes a structural blind spot in LLM evaluation in medical settings: existing benchmarks measure what models know, but not whether they preserve correct medical judgment under misleading context.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.08473v4">AsFT: Anchoring Safety During LLM Fine-Tuning Within Narrow Safety Basin</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) improves performance but introduces critical safety vulnerabilities: even minimal harmful data can severely compromise safety measures. We observe that perturbations orthogonal to the alignment direction - defined by weight differences between aligned (safe) and unaligned models - rapidly compromise model safety. In contrast, updates along the alignment direction largely preserve it, revealing the parameter space as a "narrow safety basin". To address this, we propose AsFT (Anchoring Safety in Fine-Tuning) to maintain safety by explicitly constraining update directions during fine-tuning. By penalizing updates orthogonal to the alignment direction, AsFT effectively constrains the model within the "narrow safety basin," thus preserving its inherent safety. Extensive experiments on multiple datasets and models show that AsFT reduces harmful behaviors by up to 7.60%, improves task performance by 3.44%, and consistently outperforms existing methods across multiple tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12250v1">Reassessing High-Performing LLMs on Polish Medical Exams: True Competence or Bias-Driven Performance?</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 26 pages total with references and appendix, preprint
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) in medicine are mainly evaluated using multiple-choice question answering (MCQA), which can overestimate real clinical ability due to guessing strategies and answer biases. To address these limitations, we introduce an expanded and more challenging benchmark based on Polish medical exams, adding over 15,000 questions, two new domains, and four structural modifications that reduce MCQA-specific artifacts and better test reasoning. We evaluate 21 LLMs and show that evaluation design strongly affects results. Under our harder setup, the best model (Qwen3.5-122B) drops by 28.4 and 31 pp on English and Polish exams, respectively. Despite low evidence of data contamination, standard MCQA scores do not reliably reflect true medical competence. To facilitate further research, we make our benchmark publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12247v1">Beyond Third-Person Audits: Situated Interaction Auditing for User-Centered LLM Bias Research</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Research on bias in large language models (LLMs) has predominantly focused on third-person audits, which study how models represent or evaluate demographic groups as external subjects. However, this paradigm overlooks a structural blind spot because the user is absent from the audit. In practice, LLMs are used in open-ended, personal interactions, during which the model implicitly represents the user and adjusts its responses accordingly. When identical requests yield different responses depending on who is asking, bias manifests not in how the model describes others but in how it treats its interlocutor. We propose Situated Interaction Auditing (SIA), a user-centered framework for studying how user profile signals -- implicit sociodemographic markers, writing style, and stated identity -- systematically shape LLM response quality, content, and tone. We demonstrate the framework through a case study that intersects gender and socioeconomic status signals across multiple task domains and outline a research agenda for SIA as a new mission for natural language processing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12234v1">On The Effectiveness-Fluency Trade-Off In LLM Conditioning: A Systematic Study</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 8 pages, 2 figure
    </div>
    <details class="paper-abstract">
      Controlling the output of Large Language Models (LLMs) is a central challenge for their reliable deployment, yet a clear understanding of the involved trade-offs remains elusive. Current approaches to conditioning are often evaluated with a narrow focus on their effectiveness at injecting or removing a target concept, neglecting generation quality. We systematically investigate a range of conditioning methods in both injection and removal scenarios. We find that efficient steering methods frequently achieve conditioning at a steep cost to fluency. Furthermore, we identify a critical yet previously overlooked interaction with the training paradigm: activation steering methods are far less effective on instruction-tuned models than on their base counterparts. Simple prompting and full-fledged supervised fine-tuning, on the other hand, are viable options for concept injection, but are not as good at concept removal. Finally, cheaply computed textual metrics highly correlate to costly LLM-as-judge scores, and provide insights on the behavior of conditioning methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.19225v5">FinTradeBench: A Financial Reasoning Benchmark for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 9 pages main text, 31 pages total (including references and appendix). 5 figures, 16 tables. Preprint under review. Code and data will be made available upon publication
    </div>
    <details class="paper-abstract">
      Real-world financial decision-making is a challenging problem that requires reasoning over heterogeneous signals, including company fundamentals derived from regulatory filings and trading signals computed from price dynamics. Recently, with advances in Large Language Models (LLMs), financial analysts have begun to use them for financial decision-making tasks. However, existing financial question-answering benchmarks for testing these models primarily focus on company balance sheet data and rarely evaluate reasoning about how company stocks trade in the market or their interactions with fundamentals. To leverage the strengths of both approaches, we introduce FinTradeBench, a benchmark for evaluating financial reasoning that integrates company fundamentals and trading signals. FinTradeBench contains 1,400 questions grounded in NASDAQ-100 companies over a ten-year historical window. The benchmark is organized into three reasoning categories: fundamentals-focused, trading-signal-focused, and hybrid questions requiring cross-signal reasoning. To ensure reliability at scale, we adopt a calibration-then-scaling framework that combines expert seed questions, multi-model response generation, intra-model self-filtering, numerical auditing, and human-LLM judge alignment. We evaluate 14 LLMs under zero-shot prompting and retrieval-augmented settings and witness a clear performance gap. Retrieval substantially improves reasoning over textual fundamentals, but provides limited benefit for trading-signal reasoning. These findings highlight fundamental challenges in the numerical and time-series reasoning for current LLMs and motivate future research in financial intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12212v1">Mind your key: An Empirical Study of LLM API Credential Leakage in iOS Apps</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 12 pages, 4 figures, 4 tables
    </div>
    <details class="paper-abstract">
      The rapid integration of large language models (LLMs) into mobile applications has introduced a new class of credential security risk: leaked credentials that grant unauthorized access to LLM inference services, causing financial damage to developers. Prior work on credential leakage has focused primarily on Android apps; to date, no empirical study has systematically investigated LLM API key leakage in iOS applications. We present the first in-depth empirical study of API key leakage in LLM-integrated apps. We construct a high-quality dataset of 444 iOS applications, filtered from 1092 candidates through a standardized process, and develop LLMKeyLens, a dynamic analysis framework that detects LLM API key leakage via traffic interception, provider-specific key extraction, and active validity confirmation, requiring neither source code access nor binary decryption. Our analysis reveals that 282 applications expose exploitable LLM API credentials in network traffic, spanning at least ten providers. We identify three leakage patterns: JWT-based token leakage (48%), unauthenticated backend proxy access (33%), and plaintext API key transmission (19%). To assess remediation, we re-analyzed the same 282 vulnerable applications three months after responsible disclosure; only 28% had remediated the reported vulnerability, while 72% remained exploitable, with persistent issues stemming from unauthenticated backends and broken JWT implementations. Our findings show that LLM API key leakage is both prevalent and persistent in the iOS ecosystem, exposing a systemic gap between developer practice and secure integration principles, and suggest that secure LLM integration requires not only developer awareness but also explicit security guidance from providers and platform-level enforcement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12198v1">LLM-Based User Personas for Recommendations at Scale</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) offer unprecedented potential for enhancing recommendation systems through their world knowledge and reasoning capabilities. However, existing approaches often rely on structured IDs or offline processing, limiting semantic richness, real-time adaptability, and user-facing interpretability. In this paper, we introduce a novel framework that enables real-time generation of LLM-based user interest personas for a large-scale commercial video recommendation platform. Our method generates natural-language user interest personas that address the exploitation-exploration trade-off by combining the summarization of existing interests with novel topics, directly during serving. To overcome the computational challenges of online LLM inference at a billion-user scale, we design a cost-efficient architecture leveraging knowledge distillation, asynchronous inference, and input optimization via semantically clustered video representations. Extensive offline evaluations, user studies, and live A/B tests demonstrate significant improvements in viewer value. This work bridges the gap between high-level semantic understanding and industrial-scale recommendation, paving the way for more dynamic, explainable, and satisfying personalized experiences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21594v3">Visualizing LLM Latent Space Geometry Through Dimensionality Reduction</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 25 pages, 15 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) achieve state-of-the-art results across many natural language tasks, but their internal mechanisms remain difficult to interpret. In this work, we extract, process, and visualize latent state geometries in Transformer-based language models through dimensionality reduction. We capture layerwise activations at multiple points within Transformer blocks and enable systematic analysis through Principal Component Analysis (PCA) and Uniform Manifold Approximation and Projection (UMAP). We demonstrate experiments on GPT-2 and LLaMa models, where we uncover interesting geometric patterns in latent space. Notably, we identify a clear separation between attention and MLP component outputs across intermediate layers, a pattern not documented in prior work to our knowledge. We also characterize the high norm of latent states at the initial sequence position and visualize the layerwise evolution of latent states. Additionally, we demonstrate the high-dimensional helical structure of GPT-2's positional embeddings and the sequence-wise geometric patterns in LLaMa. We make our code available at https://github.com/Vainateya/Feature_Geometry_Visualization. A better formatted blog-post with identical content is available at https://iclr-blogposts.github.io/2026/blog/2026/vis-llm-latent-geometry/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12142v1">AerialClaw: An Open-Source Framework for LLM-Driven Autonomous Aerial Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Unmanned aerial vehicles (UAVs) are increasingly used in inspection, search and rescue, environmental monitoring, and emergency response. However, most UAV applications still rely on pre-defined command sequences or task-specific pipelines, where developers manually connect perception, planning, flight control, simulation, logging, and safety modules. This limits the flexibility, reproducibility, and extensibility of autonomous aerial systems. This paper presents AerialClaw, an open-source software framework that enables UAVs to operate as decision-making aerial agents rather than merely command-following platforms. Given a natural-language mission, AerialClaw allows an LLM-based agent to understand the task, maintain context, invoke executable aerial skills, observe perception and runtime feedback, and iteratively update its decisions in a closed loop. The framework adopts a modular brain-skill-runtime architecture, combining hard skills for atomic UAV operations, Markdown-based soft skills for reusable task strategies, document-driven agent state and capability boundaries, memory-driven reflection, safety-oriented runtime validation, and platform-agnostic execution adapters. AerialClaw supports lightweight mock execution, PX4 SITL with Gazebo, and AirSim-based simulation, together with a web console, pluggable model backends, example missions, simulation assets, and staged deployment scripts. By combining standardized aerial skills, document-driven agent state, memory, and closed-loop LLM decision-making, AerialClaw provides a reproducible and extensible open-source framework for building UAV systems that can interpret missions, make decisions, execute skills, and adapt their behavior from feedback.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.13628v3">Compact LLM Deployment and World Model Assisted Offloading in Mobile Edge Computing</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 16 pages, 10 figures
    </div>
    <details class="paper-abstract">
      This paper investigates compact large language model (LLM) deployment and world-model-assisted inference offloading in mobile edge computing (MEC) networks. We first propose an edge compact LLM deployment (ECLD) framework that jointly applies structured pruning, low-bit quantization, and knowledge distillation to construct edge-deployable LLM variants, and we evaluate these models using four complementary metrics: accessibility, energy consumption, hallucination rate, and generalization accuracy. Building on the resulting compact models, we formulate an MEC offloading optimization problem that minimizes the long-term average inference latency subject to per-device energy budgets and LLM-specific quality-of-service constraints on effective accuracy and hallucination. To solve this problem under unknown and time-varying network dynamics, we develop a world model-proximal policy optimization (PPO) algorithm, which augments an on-policy PPO algorithm with a learned recurrent world model that provides improved value targets and short imagination rollouts. Extensive experiments on Llama-3.1-8B, Qwen3-8B, and Mistral-12B show that ECLD compresses base models by about 70-80% in storage (i.e., from 15.3 GB to 3.3 GB for Llama-3.1-8B) and reduces per-query energy consumption by up to 50%, while largely preserving accuracy and often lowering hallucination compared with quantization-only or pruning-only baselines. Moreover, they also show that world model-PPO speeds up convergence by about 50%, improves the final reward by 15.8% over vanilla PPO, and reduces average inference latency by 12-30% across different user populations, while satisfying the accuracy and hallucination constraints and approaching the generation quality of always-offloading with much of the efficiency of local execution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12117v1">Soft-Prompt Tuning for Fair and Efficient LLM Benchmark Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 10 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Benchmark scores often misrepresent a large language model's (LLM's) knowledge, because they rely, e.g., on the model's ability to follow specific formatting requirements. This especially penalizes base models that may know the correct answers but lack the ability -- typically introduced in post-training -- to structure them as instructed. To overcome this, we propose soft-prompt tuning, an efficient, fair, and architecture-agnostic model evaluation. By optimizing only 10 soft-prompt vectors (roughly 0.0006% parameters for a 7B model) over a short tuning period, we adapt models to specific benchmark formats, closing gaps in format-following and ensuring that underlying knowledge is accurately reflected in benchmark scores. This allows one to fairly compare different base models -- trained with various pre-training recipes -- on benchmarks without the need for full post-training. We evaluated soft-prompt tuning across 7 models and 7 datasets. The results show that (a) soft-prompt tuning saturates format-following within 80 steps (~640 samples) making it highly efficient, (b) soft-prompt tuning significantly outperforms zero- and few-shot prompting, surfacing base model knowledge that standard prompting misses, that (c) even post-trained models can benefit from soft-prompts to maximize format compliance, and that (d) soft-prompted base model performance predicts post-trained model rankings more reliably than zero- and few-shot baselines, offering a low-cost proxy for downstream model quality. Our contributions include (1) metrics which disentangle format-following and knowledge accuracy, (2) a fairer benchmarking protocol of LLM knowledge, and (3) a cost- and memory-effective recipe to identify optimal pre-training strategies early in LLM development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12073v1">"That's AI Slop, You Bot!" Studying Accusations, Evidence, and Credibility in Online Discourse Towards LLM-Generated Comments</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Generative AI has made fluent prose cheap to produce, breaking the old promise to readers that good writing meant real thinking. How have readers responded, and what can this tell us about changing anti-AI attitudes? We analyzed 25 million comments from Hacker News and Reddit (2023-2026), combining LLM judgment on 7,500 sampled accusations of AI use, sentiment trajectories, speech-act coding of 300 confirmed accusations of AI use, and a matched-control test of accused versus non-accused parent comments. We found that the pejorative-label share of accusations rose more than tenfold on both platforms while a placebo vocabulary of pre-2022 inauthenticity terms (shill, astroturf) did not. This shift reflected a fast-growing trend of branding any suspicious or seemingly inauthentic prose as "AI slop". The slop frame now constitutes 94 percent of pejorative mentions, with the dominant comments shifting in tone from mockery toward gatekeeping and structural protest. The key surprise comes from a matched-control test which found that prose features that statistically distinguish AI from human text do not predict which human text gets accused as AI. The new accusations work as social gatekeeping of perceived authenticity without actually screening for AI. This research extends signaling theory by showing that substitute signals used socially can grow even when inaccurate if the underlying detection problem cannot be solved at the non-expert level. It shows that AI's effects on writing from the reader side are distinct from those on the production (writer) side. Detection technology cannot resolve this dynamic because the social function of accusations is increasingly to perform social gatekeeping and in-group signaling as opposed to identifying AI-generated writing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22025v2">When Generic Prompt Improvements Hurt: Evaluation-Driven Iteration for LLM Applications</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 Technical report. 42 pages, 3 figures. Code, test suites, and result logs: https://github.com/dcommey/llm-eval-benchmarking
    </div>
    <details class="paper-abstract">
      Evaluating Large Language Model (LLM) applications differs from conventional software testing because outputs are probabilistic, semantically variable, and sensitive to prompt and model changes. This technical report proposes the Minimum Viable Evaluation Suite (MVES), an audit-oriented structure for application-level LLM evaluation. MVES links application categories to failure modes, metrics, required artifacts, and validation evidence across general LLM applications, retrieval-augmented systems, and agentic workflows. We pair the framework with a reproducible local evaluation harness covering structured extraction, RAG citation/content-compliance, and instruction-following checks. Using Ollama with Llama 3 8B Instruct and Qwen 2.5 7B Instruct, we evaluate five prompt conditions over expanded 30-case-per-suite ablations. The results show that, in the tested local conditions, generic prompt additions do not produce monotonic improvements: stronger output-contract prompts improve strict extraction for both models, while RAG citation/content-compliance declines under some generic-rule conditions. The largest observed decline occurs for Qwen 2.5 on RAG when generic rules are appended to the user prompt, from 26/30 to 9/30. These findings support evaluation-driven prompt iteration: prompt changes should be treated as potential regression risks and tested against task-specific suites before deployment. The accompanying repository contains the test suites, prompt variants, evaluation harness, raw result logs, and scripts needed to reproduce the reported local ablations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.23243v2">Are Frontier LLMs Ready for Cybersecurity? Evidence for Vertical Foundation Models from Dual-Mode Vulnerability Benchmarks</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      We evaluate whether frontier LLMs are ready for cybersecurity through a dual-mode benchmark: white-box function-level vulnerability detection (VulnLLM-R, across C/Java/Python) and black-box web application security testing (five production-style applications with 118 ground-truth vulnerabilities across 20+ CWE families, which we will open-source). We test six frontier models (GPT-5.4, Codex~5.3, Claude Opus~4.6, Sonnet~4.6, Gemini~3.1~Pro and Gemini~3~Flash) and two domain-specialized models across four testing paradigms. Our findings are sobering: (1)~every frontier model produces 10-50% false positive rates in white-box detection, systematically over-predicting vulnerabilities; (2)~in black-box testing, frontier models achieve only 4-8% ground-truth coverage, improving to just 10-19% even with external security tools (Playwright MCP, Burp Suite MCP); (3)~structured penetration-testing methodology encoded in domain-specialized agents raises per-family detection above 50%, demonstrating that methodology, not scale, is the primary lever; and (4)~a domain-specialized defense model achieves the highest precision (0.904) and lowest false positive rate (9.7%) among all models, on a single GPU. We identify the absence of structured security testing traces end-to-end request/response sequences, failure-heavy data, and multi-step attack chains as the fundamental training data bottleneck, and propose self-play security testing as a data generation strategy. Our results make the case for vertical foundation models purpose-built for cybersecurity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11470v1">The Periodic Table of LLM Reasoning: A Structured Survey of Reasoning Paradigms, Methods, and Failure Modes</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved strong performance across natural language processing tasks, yet reliable reasoning remains an open challenge. Although modern LLMs show progress in structured inference, multi-step problem solving, and contextual understanding, their reasoning behavior is often inconsistent and sensitive to prompting strategies, task design, and model scale. This survey provides a systematic analysis of more than 300 recent papers from arXiv, Semantic Scholar, Google Scholar, Papers with Code, and the ACL Anthology to examine how reasoning capabilities emerge in LLMs and where they fail. We make three main contributions. First, we introduce a structured taxonomy of LLM reasoning research, covering Chain-of-Thought reasoning, multi-hop reasoning, mathematical reasoning, common sense reasoning, visual and temporal reasoning, code and algorithmic reasoning, retrieval-augmented reasoning, tool-augmented and agentic reasoning, and reinforcement learning-based reasoning. Second, we analyze methodological trends across these paradigms, including prompting methods, model architectures, training objectives, reward modeling, and evaluation benchmarks. Third, we synthesize recurring limitations and failure modes, such as reasoning hallucinations, brittle multi-step inference, weak causal abstraction, and poor cross-domain generalization. By organizing a rapidly expanding literature, this survey offers a unified view of the current capabilities and limitations of reasoning in LLMs. We also identify emerging research directions, including meta-reasoning, self-evolving reasoning frameworks, multimodal reasoning, and socially grounded reasoning. Overall, this work aims to serve as a reference for developing more robust, interpretable, and generalizable reasoning systems in future language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12164v3">The Language You Ask In: Language-Conditioned Ideological Divergence in LLM Analysis of Contested Political Documents</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed as analytical tools across multilingual contexts, yet their outputs may carry systematic biases conditioned by the language of the prompt. This study presents an experimental comparison of LLM-generated political analyses of a Ukrainian civil society document, using semantically equivalent prompts in Russian and Ukrainian administered to two frontier models from different developers, ChatGPT 5.2 and Claude Opus 4.5. Despite identical source material and parallel query structures, both models diverged along the same axis: Russian-language outputs leaned toward delegitimizing framings, characterizing civil society actors as externally funded elites constraining a democratic mandate, while Ukrainian-language outputs treated the same actors as legitimate stakeholders in democratic contestation. The magnitude of this divergence, however, was model-dependent. ChatGPT's Russian output reproduced vocabulary characteristic of Russian state discourse; Claude Opus's stayed in a mainstream critical idiom and hedged its judgments in both languages. These findings demonstrate that prompt language alone can systematically shift the ideological orientation of an unchanged model analyzing identical content. The shift is a general property of multilingual LLMs whose severity, and whose alignment with propaganda narratives, varies across systems. The implications reach AI deployment in polarized information environments, cross-lingual research, and AI governance in multilingual societies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17717v3">A Survey on Evaluating Quality and Trustworthiness in LLM-Generated Data</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 Published at TMLR. Title changed in the final version
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have emerged as powerful tools for generating data across various modalities. By transforming data from a scarce resource into a controllable asset, LLMs mitigate the bottlenecks imposed by the acquisition costs of real-world data for model training, evaluation, and system iteration. However, ensuring the high quality of LLM-generated synthetic data remains a critical challenge. Existing research primarily focuses on generation methodologies, with limited direct attention to the quality of the resulting data. Furthermore, most studies are restricted to single modalities, lacking a unified perspective across different data types. To bridge this gap, we propose the \textbf{LLM Data Auditor framework}. In this framework, we first describe how LLMs are utilized to generate data across six distinct modalities. More importantly, we systematically categorize intrinsic metrics for evaluating synthetic data from two dimensions: quality and trustworthiness. This approach shifts the focus from extrinsic evaluation, which relies on downstream task performance, to the inherent properties of the data itself. Using this evaluation system, we analyze the experimental evaluations of representative generation methods for each modality and identify substantial deficiencies in current evaluation practices. Based on these findings, we offer concrete recommendations for the community to improve the evaluation of data generation. Finally, the framework outlines methodologies for the practical application of synthetic data across different modalities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11416v1">MPC-Patch-Bench: Security-Aware LLM Code Patch for Multi-Party Computation</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 preprint
    </div>
    <details class="paper-abstract">
      Repository-level benchmarks for evaluating Large Language Model (LLM) code repair on Secure Multi-Party Computation (MPC) software do not yet exist, and directly transplanting general-purpose benchmarks such as SWE-bench fails on three structural fronts: (i) MPC repositories are dominated by generic Python infrastructure rather than cryptographic logic; (ii) high-value MPC fixes lack the standardized tests rigid extraction pipelines require; and (iii) standard fail-to-pass evaluation is insufficient for code that must also be cryptographically safe. MPC is increasingly deployed for privacy-preserving machine learning, biomedical collaboration, and secure analytics. Existing MPC-specific code-synthesis efforts cover only operator-level or single-framework tasks; evaluating LLM agents on real repository-level MPC repair instead demands MPC-aware data curation and a verifier matched to the security and numerical-fidelity guarantees MPC programs must obey neither of which existing benchmarks provide. We introduce MPC-Patch-Bench, a repository-level benchmark organised around two frameworks. (1)The Data Curation Framework combines a domain-specific curation agent that filters raw pull requests through three cryptographic layers with a human-AI completion engine that synthesizes missing problem statements and Fail-to-Pass/Pass-to-Pass tests, yielding 205 fully verified instances. (2)The MPC Verifier provides dedicated security and numerical-fidelity checks via dynamic differential testing against plaintext oracles and MPC-specific static analysis rules that flag unsafe reveals, insecure arithmetic, and illegal public/private casts. The strongest evaluated LLM functionally resolves only 22.9% of MPC-Patch-Bench tasks; the MPC Verifier further reduces verified resolution to 17.1%, with up to 40% of functionally-passing patches rejected for cryptographic or numerical-fidelity violations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.09885v6">Diffusion-Inspired Masked Fine-Tuning for Knowledge Injection in Autoregressive LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are often used in environments where facts evolve, yet factual knowledge updates via fine-tuning on unstructured text often suffer from 1) reliance on compute-heavy paraphrasing augmentation and 2) the reversal curse. Recent studies show diffusion large language models (dLLMs) require fewer training samples to achieve lower loss in pre-training and are more resistant to the reversal curse, suggesting dLLMs may learn new knowledge more easily than autoregressive LLMs (arLLMs). We test this hypothesis in controlled knowledge fine-tuning experiments and find that while arLLMs rely on paraphrase augmentation to generalize knowledge text into question-answering (QA) capability, dLLMs do not require paraphrases to achieve high QA accuracy. To further investigate whether the demasking objective alone can induce such a knowledge injection advantage in dLLMs regardless of their diffusion denoising paradigm, we propose masked fine-tuning for arLLMs, which prompts an arLLM to reconstruct the original text given a masked version in context. The masked fine-tuning for arLLMs substantially improves the efficacy of knowledge injection, i.e. no paraphrase needed and resistant to the reversal curse, closing the gap between arLLMs and dLLMs. We also demonstrate broader applicability: on a large-scale knowledge-intensive dataset (1.2M samples), masked SFT achieves the best downstream accuracy on GPQA-diamond among all fine-tuning variants. The demasking objective also improves SFT on math tasks, suggesting broad utility beyond factual knowledge injection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11379v1">Automated Mediator for Human Negotiation: Pre-Mediation via a Structured LLM Pipeline</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 12 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Pre-mediation, the preparatory phase preceding direct human negotiation, plays a critical role in achieving mutually beneficial agreements, yet is often omitted due to cost, time, and limited access to trained mediators. We introduce an automated mediator for human negotiation, implemented as a structured pipeline of LLM modules, that supports pre-mediation in integrative negotiation settings. The pipeline decomposes preparation into specialized modules for dialogue, preference prediction, response-level critique, and structured summarization, separating inference, generation, and evaluation to address limitations of monolithic single-prompt approaches. We use the term "agent" for each module following common LLM-systems terminology, but the components are not autonomous and do not interact peer-to-peer; outputs are passed forward in a fixed sequence. We evaluate the system in two controlled human-subject experiments comparing AI-based pre-mediation with professional human mediators in a multi-issue negotiation scenario. On short-term self-reported measures, the automated mediator achieves preparation outcomes broadly comparable to human mediators, including trust in the mediator and confidence in reaching mutually beneficial agreements, while achieving substantially lower error on the preference-inference task under our scenario and prompts (36% lower RMSE). A second study shows that targeted prompt refinements reduce excessive affirmation patterns from 36.6% to 16.8%, matching human mediator baselines. Our findings suggest that structured LLM pipelines can provide scalable, low-effort pre-mediation support broadly comparable to human mediators on short-term self-reported preparation outcomes. The pipeline's single-party design mirrors how human mediators run pre-mediation today and enables parallel deployment across all parties to a dispute, supporting scalability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11357v1">TileFuse: A Fused Mixed-Precision Kernel Library for Efficient Quantized LLM Inference on AMD NPUs</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 13 pages excluding reference, 11 figures
    </div>
    <details class="paper-abstract">
      With the growing demand for on-device LLM inference, edge SoCs increasingly integrate NPUs to improve performance and energy efficiency under tight power and thermal budgets. However, practical LLM deployment on current client NPUs remains difficult: widely used quantization formats such as AWQ do not map cleanly onto many existing NPU software stacks, which are often proprietary and expose limited low-level control. In this work, we present \textit{TileFuse}, a close-to-metal mixed-precision kernel library for AMD XDNA2 NPUs that targets transformer linear layers in quantized LLM inference. TileFuse brings practical low-bit formats such as AWQ-style W4A16 and W8A16 directly onto XDNA2, rather than forcing the model to be reshaped around an NPU-specific quantization scheme. TileFuse co-designs weight layout, metadata placement, mixed-precision microkernels, and array-level dataflow. Specifically, it fuses unpacking, dequantization, and GEMM/GEMV execution into a single kernel flow, introduces an interleaved pre-tiling layout that supports GEMM dimensions up to 32K, and redesigns GEMV dataflow to utilize the full 4x8 AIE array. Across kernel-level evaluations, TileFuse improves performance by up to 121.6% for GEMM and 281% for GEMV over full-precision baselines, while delivering more than 2x performance and energy-efficiency gains over strong iGPU baselines on GEMM. In end-to-end LLM experiments on Ryzen AI laptops, TileFuse achieves up to 2.0x lower prefilling latency with more than 64.6% lower energy consumption. Together, these results show that XDNA2 is a practical target for AWQ-style edge LLM inference and that native NPU support for off-the-shelf quantization can make NPUs substantially more usable in real client deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11316v1">Schützen: Evaluating LLM Safety in Bulgarian and German Contexts</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 19 pages, 13 tables, 12 figures
    </div>
    <details class="paper-abstract">
      Large language models are increasingly deployed across professional domains, bringing hard-to-predict risks, including the generation of harmful or disrespectful content. Although substantial progress has been made in developing safety evaluation datasets, existing resources remain overwhelmingly English- and Chinese-centric. This limitation is particularly pronounced when evaluating languages that operate within shared sociocultural, legal, and ethical contexts. To address this gap, we introduce Schützen: a German--Bulgarian safety dataset designed to assess model answerability under risk, covering both a low-resource language (Bulgarian) and a high-resource language (German). Experiments with multilingual and language-specific LLMs reveal pronounced cross-language differences in safety behavior, highlighting the necessity of tailored, region-specific evaluation resources to support the responsible deployment of LLMs in Germany and Bulgaria. Datasets and code are available at https://github.com/xnlp-lab/Schutzen. Warning: this paper contains examples that may be offensive, harmful, or biased.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11166v1">Flaws in the LLM Automation Narrative</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly described as performing at the level of human experts on knowledge economy tasks. These claims are primarily based on how LLMs perform on benchmarking tasks that measure average performance across standardized datasets. Primary limitations of many benchmarking tasks are that they often measure performance based on content directly included in LLM training data, and they frequently do not assess the reliability of LLM performance or the magnitude of LLM errors. However, in high stakes contexts, these qualities are critically important. Through a novel LLM benchmarking task that requires writing computer code to complete a data analysis task, we compare the performance of a frontier LLM against submissions from human experts and explicitly measure the variance of responses and the magnitude of errors. Our study reveals that the human experts perform better on average on a range of metrics and demonstrate less variability in performance. Our results provide evidence that LLMs do not consistently perform at the level of human experts and demonstrate the importance of measuring variance and assessing error magnitude in LLM benchmark evaluations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.22714v3">AMEL: Accumulated Message Effects on LLM Judgments</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 24 pages, 14 figures, 8 tables. Single author. Code, data (84,088 deduplicated API responses), and analysis pipeline at https://github.com/chutapp/amel
    </div>
    <details class="paper-abstract">
      Large language models are routinely used as automated evaluators: to review code, moderate content, or score outputs, often with many items passing through one conversation. We ask whether the polarity of prior conversation history biases subsequent judgments, an effect we call the accumulated message effect on LLM judgments (AMEL). Across 84,088 API calls to 12 models from 5 providers (OpenAI, Anthropic, Google, DeepSeek, and four open-source models), we present identical test items in isolation or following histories saturated with predominantly positive or negative evaluations. Models shift toward the conversation's prevailing polarity (d = -0.17, p < 10^-53). The effect concentrates on items where the model is genuinely uncertain at baseline (d = -0.36 for high-entropy items, vs d = -0.15 when the baseline is deterministic). Bias does not grow with context length: 5 prior turns and 50 produce the same shift (Spearman |r| < 0.01; OLS slope p = 0.80). And there is a negativity asymmetry: paired per item, negative histories induce 1.52x more bias than positive (t = 13.03, p < 10^-36, n = 2,733). Scaling helps but does not solve it (Anthropic: Haiku -0.22 to Opus -0.17; OpenAI: Nano -0.34 to GPT-5.2 -0.17). Three follow-ups narrow the mechanism. The token probability distribution shifts continuously, not at a threshold. The negativity asymmetry has both token-level and semantic components, though attributing the balance is exploratory at our sample sizes. Position does not matter: five biased turns anywhere in a 50-turn history produce the same shift. The simplest fix for evaluation pipelines is a fresh context per item; when batching is unavoidable, balancing the history helps.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11145v1">OpenPCC: Open and Confidential LLM Serving on Commodity TEEs</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Generative AI applications such as personal AI agents, image generators, and chat assistants offer advanced capabilities to improve user experience. Behind the scenes, Large Language Models (LLMs) that power these services require a massive amount of computation and are usually deployed in the cloud, available as APIs, meaning that a user's request has to be sent to a Cloud Inference Service (CIS) for processing. However, the strong capabilities of LLM also mean that user's requests now contain much more personal sensitive or enterprise confidential information, demanding equally strong protection in CIS. While early industry efforts such as Apple Private Cloud Compute (PCC) and Google Private AI Compute have emerged to show the potential of secure CIS, they are not adoptable for deployment by others due to their reliance on proprietary hardware and closed ecosystem. In addition, they all suffer from their own design glitches that can undermine the ambitious goal of bringing in true privacy protection to end users. In this paper, we present our analysis of the fundamental requirements of building a secure yet open CIS. We then present OpenPCC, a Confidential CIS framework that does not rely on proprietary hardware but instead uses commercially available TEEs. We implement an open-source prototype and characterize it end-to-end on a Llama-3 8B vLLM workload, separating OpenPCC's own cost from the underlying TEE hardware. Our analysis and evaluation demonstrated the feasibility and security of the system.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05463v2">PSEBench: A Controllable and Verifiable Benchmark for Evaluating LLMs in Patient Safety Event Triage</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Patient safety event triage, determining whether a clinical event is reportable under jurisdiction-specific policy, is a high-stakes task typically performed manually by patient safety experts. Although LLMs may support this workflow, reliable evaluation is limited by the lack of benchmarks to capture evidence-grounded policy reasoning, proactive information seeking for incomplete reports, and principled abstention in irreducibly ambiguous cases. We address this gap with a policy-grounded construction methodology centered on the clause card, a structured representation that factorizes regulatory text into auditable decision specifications. Combining clause cards with anchor-driven instantiation and closed-loop verification, our scalable pipeline produces narratives with by-construction ground truth and naturally supports generating missing information and uncertain variants. We instantiate this method on Minnesota's 29 Reportable Adverse Health Events, producing PSEBench, a 5,074-case benchmark with an agentic evaluation environment. Evaluation on 15 representative LLMs reveals consistent capability trends, demonstrates the benchmark's utility, and identifies actionable gaps toward reliable LLM-based patient safety event triage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.01993v2">SAFE: An LLM-as-Verifier Framework for Evidence-Grounded Multi-Hop Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Multi-hop QA benchmarks often reward Large Language Models (LLMs) for spurious correctness, where models reach correct answers through invalid intermediate reasoning. We propose SAFE, an LLM-as-verifier framework for evidence-grounded multi-hop QA. Rather than judging only the final answer after generation, SAFE verifies reasoning during generation by checking intermediate steps against the provided passages and previous reasoning trajectory. To make this process checkable, SAFE decomposes reasoning into atomic, evidence-grounded units represented with Knowledge Graph (KG) triples. At train-time, SAFE verifies benchmark supervision under KG-grounded constraints and constructs reliable verifier training data. At inference-time, an external verifier checks each generated step, identifies invalid reasoning, and provides correction feedback before errors propagate. Across three multi-hop QA benchmarks, SAFE improves accuracy by 8.8 pp on average. These results show that evidence-grounded multi-hop QA benefits from shifting LLM-based evaluation from post-hoc answer judgment to stepwise reasoning verification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11081v1">Unifying Local Communications and Local Updates for LLM Pretraining</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 38 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Communication-efficient pre-training of LLMs is increasingly important as training draws on compute distributed across clusters, data centers, and lower-bandwidth links. Many practical methods reduce communication frequency but still rely on synchronous All-Reduce operations that maintain identical model states and tie progress to global collectives. This can become a bottleneck when bandwidth or worker speed is heterogeneous. We introduce GASLoC, a novel decentralized pre-training algorithm that generalizes the notion of communication acceleration to the recently popular "outer optimizer" to allow a practical gossip-based training framework that is compatible with adaptive optimizers, allows for local optimizer steps, and can utilize sparse randomized peer communication. Empirically, on a number of standard LLM training tasks, we demonstrate that GASLoC outperforms state-of-the-art decentralized algorithms in single step per communication setting for a number of topologies and, unlike existing decentralized methods in the LLM setting, it allows to obtain performance competitive with DiLoCo when utilizing multiple local steps. In the heterogeneous bandwidth setting we demonstrate the advantage of GASLoC showing that it can significantly outperform DiLoCo.
    </details>
</div>
