# llm - 2026_02

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3
- [Part 4](papers_4.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05987v1">From Human-Human Collaboration to Human-Agent Collaboration: A Vision, Design Philosophy, and an Empirical Framework for Achieving Successful Partnerships Between Humans and LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      The emergence of Large Language Model (LLM) agents enables us to build agent-based intelligent systems that move beyond the role of a "tool" to become genuine collaborators with humans, thereby realizing a novel human-agent collaboration paradigm. Our vision is that LLM agents should resemble remote human collaborators, which allows HCI researchers to ground the future exploration in decades of research on trust, awareness, and common ground in remote human collaboration, while also revealing the unique opportunities and challenges that emerge when one or more partners are AI agents. This workshop establishes a foundational research agenda for the new era by posing the question: How can the rich understanding of remote human collaboration inspire and inform the design and study of human-agent collaboration? We will bring together an interdisciplinary group from HCI, CSCW, and AI to explore this critical transition. The 180-minute workshop will be highly interactive, featuring a keynote speaker, a series of invited lightning talks, and an exploratory group design session where participants will storyboard novel paradigms of human-agent partnership. Our goal is to enlighten the research community by cultivating a shared vocabulary and producing a research agenda that charts the future of collaborative agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.25110v4">DEBATE: A Large-Scale Benchmark for Evaluating Opinion Dynamics in Role-Playing LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      Accurately modeling opinion change through social interactions is crucial for understanding and mitigating polarization, misinformation, and societal conflict. Recent work simulates opinion dynamics with role-playing LLM agents (RPLAs), but multi-agent simulations often display unnatural group behavior (e.g., premature convergence) and lack empirical benchmarks for assessing alignment with real human group interactions. We introduce DEBATE, a large-scale benchmark for evaluating the authenticity of opinion dynamics in multi-agent RPLA simulations. DEBATE contains 36,383 messages from 2,832 U.S.-based participants across 708 groups and 107 topics, with both public messages and private Likert-scale beliefs, enabling evaluation at the utterance and group levels (and supporting future individual-level analyses). We instantiate "digital twin" RPLAs with seven LLMs and evaluate across two settings: next-message prediction and full conversation rollout, using stance-alignment and opinion-convergence metrics. In zero-shot settings, RPLA groups exhibit strong opinion convergence relative to human groups. Post-training via supervised fine-tuning (SFT) and Direct Preference Optimization (DPO) improves stance alignment and brings group-level convergence closer to human behavior, though discrepancies in opinion change and belief updating remain. DEBATE enables rigorous benchmarking of simulated opinion dynamics and supports future research on aligning multi-agent RPLAs with realistic human interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05946v1">$f$-GRPO and Beyond: Divergence-Based Reinforcement Learning Algorithms for General LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      Recent research shows that Preference Alignment (PA) objectives act as divergence estimators between aligned (chosen) and unaligned (rejected) response distributions. In this work, we extend this divergence-based perspective to general alignment settings, such as reinforcement learning with verifiable rewards (RLVR), where only environmental rewards are available. Within this unified framework, we propose $f$-Group Relative Policy Optimization ($f$-GRPO), a class of on-policy reinforcement learning, and $f$-Hybrid Alignment Loss ($f$-HAL), a hybrid on/off policy objectives, for general LLM alignment based on variational representation of $f$-divergences. We provide theoretical guarantees that these classes of objectives improve the average reward after alignment. Empirically, we validate our framework on both RLVR (Math Reasoning) and PA tasks (Safety Alignment), demonstrating superior performance and flexibility compared to current methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05945v1">AgenticTagger: Structured Item Representation for Recommendation with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      High-quality representations are a core requirement for effective recommendation. In this work, we study the problem of LLM-based descriptor generation, i.e., keyphrase-like natural language item representation generation frameworks with minimal constraints on downstream applications. We propose AgenticTagger, a framework that queries LLMs for representing items with sequences of text descriptors. However, open-ended generation provides little control over the generation space, leading to high cardinality, low-performance descriptors that renders downstream modeling challenging. To this end, AgenticTagger features two core stages: (1) a vocabulary building stage where a set of hierarchical, low-cardinality, and high-quality descriptors is identified, and (2) a vocabulary assignment stage where LLMs assign in-vocabulary descriptors to items. To effectively and efficiently ground vocabulary in the item corpus of interest, we design a multi-agent reflection mechanism where an architect LLM iteratively refines the vocabulary guided by parallelized feedback from annotator LLMs that validates the vocabulary against item data. Experiments on public and private data show AgenticTagger brings consistent improvements across diverse recommendation scenarios, including generative and term-based retrieval, ranking, and controllability-oriented, critique-based recommendation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04856v2">CoT is Not the Chain of Truth: An Empirical Internal Analysis of Reasoning LLMs for Fake News Generation</a></div>
    <div class="paper-meta">
      📅 2026-02-05
      | 💬 28 pages, 35 figures
    </div>
    <details class="paper-abstract">
      From generating headlines to fabricating news, the Large Language Models (LLMs) are typically assessed by their final outputs, under the safety assumption that a refusal response signifies safe reasoning throughout the entire process. Challenging this assumption, our study reveals that during fake news generation, even when a model rejects a harmful request, its Chain-of-Thought (CoT) reasoning may still internally contain and propagate unsafe narratives. To analyze this phenomenon, we introduce a unified safety-analysis framework that systematically deconstructs CoT generation across model layers and evaluates the role of individual attention heads through Jacobian-based spectral metrics. Within this framework, we introduce three interpretable measures: stability, geometry, and energy to quantify how specific attention heads respond or embed deceptive reasoning patterns. Extensive experiments on multiple reasoning-oriented LLMs show that the generation risk rise significantly when the thinking mode is activated, where the critical routing decisions concentrated in only a few contiguous mid-depth layers. By precisely identifying the attention heads responsible for this divergence, our work challenges the assumption that refusal implies safety and provides a new understanding perspective for mitigating latent reasoning risks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05933v1">Approximation of Log-Partition Function in Policy Mirror Descent Induces Implicit Regularization for LLM Post-Training</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      Policy mirror descent (PMD) provides a principled framework for reinforcement learning (RL) by iteratively solving KL-regularized policy improvement subproblems. While this approach has been adopted in training advanced LLMs such as Kimi K1.5/K2, the ideal closed-form PMD updates require reliable partition function estimation, a significant challenge when working with limited rollouts in the vast action spaces of LLMs. We investigate a practical algorithm, termed PMD-mean, that approximates the log-partition term with the mean reward under the sampling policy and performs regression in log-policy space. Specifically, we characterize the population solution of PMD-mean and demonstrate that it implicitly optimizes mirror descent subproblems with an adaptive mixed KL--$χ^2$ regularizer. This additional $χ^2$ regularization constrains large probability changes, producing more conservative updates when expected rewards are low and enhancing robustness against finite-sample estimation errors. Experiments on math reasoning tasks show that PMD-mean achieves superior performance with improved stability and time efficiency. These findings deepen our understanding of PMD-mean and illuminate pathways toward principled improvements in RL algorithms for LLMs. Code is available at https://github.com/horizon-rl/OpenKimi.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05932v1">Polyglots or Multitudes? Multilingual LLM Answers to Value-laden Multiple-Choice Questions</a></div>
    <div class="paper-meta">
      📅 2026-02-05
      | 💬 17 pages, 5 figures (8 pages of references and appendices)
    </div>
    <details class="paper-abstract">
      Multiple-Choice Questions (MCQs) are often used to assess knowledge, reasoning abilities, and even values encoded in large language models (LLMs). While the effect of multilingualism has been studied on LLM factual recall, this paper seeks to investigate the less explored question of language-induced variation in value-laden MCQ responses. Are multilingual LLMs consistent in their responses across languages, i.e. behave like theoretical polyglots, or do they answer value-laden MCQs depending on the language of the question, like a multitude of monolingual models expressing different values through a single model? We release a new corpus, the Multilingual European Value Survey (MEVS), which, unlike prior work relying on machine translation or ad hoc prompts, solely comprises human-translated survey questions aligned in 8 European languages. We administer a subset of those questions to over thirty multilingual LLMs of various sizes, manufacturers and alignment-fine-tuning status under comprehensive, controlled prompt variations including answer order, symbol type, and tail character. Our results show that while larger, instruction-tuned models display higher overall consistency, the robustness of their responses varies greatly across questions, with certain MCQs eliciting total agreement within and across models while others leave LLM answers split. Language-specific behavior seems to arise in all consistent, instruction-fine-tuned models, but only on certain questions, warranting a further study of the selective effect of preference fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05929v1">KV-CoRE: Benchmarking Data-Dependent Low-Rank Compressibility of KV-Caches in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      Large language models rely on kv-caches to avoid redundant computation during autoregressive decoding, but as context length grows, reading and writing the cache can quickly saturate GPU memory bandwidth. Recent work has explored KV-cache compression, yet most approaches neglect the data-dependent nature of kv-caches and their variation across layers. We introduce KV-CoRE KV-cache Compressibility by Rank Evaluation), an SVD-based method for quantifying the data-dependent low-rank compressibility of kv-caches. KV-CoRE computes the optimal low-rank approximation under the Frobenius norm and, being gradient-free and incremental, enables efficient dataset-level, layer-wise evaluation. Using this method, we analyze multiple models and datasets spanning five English domains and sixteen languages, uncovering systematic patterns that link compressibility to model architecture, training data, and language coverage. As part of this analysis, we employ the Normalized Effective Rank as a metric of compressibility and show that it correlates strongly with performance degradation under compression. Our study establishes a principled evaluation framework and the first large-scale benchmark of kv-cache compressibility in LLMs, offering insights for dynamic, data-aware compression and data-centric model development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.20295v4">SelfReflect: Can LLMs Communicate Their Internal Answer Distribution?</a></div>
    <div class="paper-meta">
      📅 2026-02-05
      | 💬 Accepted at ICLR 2026
    </div>
    <details class="paper-abstract">
      The common approach to communicate a large language model's (LLM) uncertainty is to add a percentage number or a hedging word to its response. But is this all we can do? Instead of generating a single answer and then hedging it, an LLM that is fully transparent to the user needs to be able to reflect on its internal belief distribution and output a summary of all options it deems possible, and how likely they are. To test whether LLMs possess this capability, we develop the SelfReflect metric, an information-theoretic distance between a given summary and a distribution over answers. In interventional and human studies, we find that SelfReflect indicates even slight deviations, yielding a fine measure of faithfulness between a summary string and an LLM's actual internal distribution over answers. With SelfReflect, we make a resounding negative observation: modern LLMs are, across the board, incapable of revealing what they are uncertain about, neither through reasoning, nor chains-of-thoughts, nor explicit finetuning. However, we do find that LLMs are able to generate faithful summaries of their uncertainties if we help them by sampling multiple outputs and feeding them back into the context. This simple approach shines a light at the universal way of communicating LLM uncertainties whose future development the SelfReflect score enables. To support the development of this universal form of LLM uncertainties, we publish the code that implements our metric for arbitrary LLMs under https://github.com/apple/ml-selfreflect .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05890v1">DFPO: Scaling Value Modeling via Distributional Flow towards Robust and Generalizable LLM Post-Training</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      Training reinforcement learning (RL) systems in real-world environments remains challenging due to noisy supervision and poor out-of-domain (OOD) generalization, especially in LLM post-training. Recent distributional RL methods improve robustness by modeling values with multiple quantile points, but they still learn each quantile independently as a scalar. This results in rough-grained value representations that lack fine-grained conditioning on state information, struggling under complex and OOD conditions. We propose DFPO (Distributional Value Flow Policy Optimization with Conditional Risk and Consistency Control), a robust distributional RL framework that models values as continuous flows across time steps. By scaling value modeling through learning of a value flow field instead of isolated quantile predictions, DFPO captures richer state information for more accurate advantage estimation. To stabilize training under noisy feedback, DFPO further integrates conditional risk control and consistency constraints along value flow trajectories. Experiments on dialogue, math reasoning, and scientific tasks show that DFPO outperforms PPO, FlowRL, and other robust baselines under noisy supervision, achieving improved training stability and generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19903v2">STELLAR: Structure-guided LLM Assertion Retrieval and Generation for Formal Verification</a></div>
    <div class="paper-meta">
      📅 2026-02-05
      | 💬 7 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Formal Verification (FV) relies on high-quality SystemVerilog Assertions (SVAs), but the manual writing process is slow and error-prone. Existing LLM-based approaches either generate assertions from scratch or ignore structural patterns in hardware designs and expert-crafted assertions. This paper presents STELLAR, the first framework that guides LLM-based SVA generation with structural similarity. STELLAR represents RTL blocks as AST structural fingerprints, retrieves structurally relevant (RTL, SVA) pairs from a knowledge base, and integrates them into structure-guided prompts. Experiments show that STELLAR achieves superior syntax correctness, stylistic alignment, and functional correctness, highlighting structure-aware retrieval as a promising direction for industrial FV.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05877v1">Agent2Agent Threats in Safety-Critical LLM Assistants: A Human-Centric Taxonomy</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      The integration of Large Language Model (LLM)-based conversational agents into vehicles creates novel security challenges at the intersection of agentic AI, automotive safety, and inter-agent communication. As these intelligent assistants coordinate with external services via protocols such as Google's Agent-to-Agent (A2A), they establish attack surfaces where manipulations can propagate through natural language payloads, potentially causing severe consequences ranging from driver distraction to unauthorized vehicle control. Existing AI security frameworks, while foundational, lack the rigorous "separation of concerns" standard in safety-critical systems engineering by co-mingling the concepts of what is being protected (assets) with how it is attacked (attack paths). This paper addresses this methodological gap by proposing a threat modeling framework called AgentHeLLM (Agent Hazard Exploration for LLM Assistants) that formally separates asset identification from attack path analysis. We introduce a human-centric asset taxonomy derived from harm-oriented "victim modeling" and inspired by the Universal Declaration of Human Rights, and a formal graph-based model that distinguishes poison paths (malicious data propagation) from trigger paths (activation actions). We demonstrate the framework's practical applicability through an open-source attack path suggestion tool AgentHeLLM Attack Path Generator that automates multi-stage threat discovery using a bi-level search strategy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05868v1">Persistent Human Feedback, LLMs, and Static Analyzers for Secure Code Generation and Vulnerability Detection</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      Existing literature heavily relies on static analysis tools to evaluate LLMs for secure code generation and vulnerability detection. We reviewed 1,080 LLM-generated code samples, built a human-validated ground-truth, and compared the outputs of two widely used static security tools, CodeQL and Semgrep, against this corpus. While 61% of the samples were genuinely secure, Semgrep and CodeQL classified 60% and 80% as secure, respectively. Despite the apparent agreement in aggregate statistics, per-sample analysis reveals substantial discrepancies: only 65% of Semgrep's and 61% of CodeQL's reports correctly matched the ground truth. These results question the reliability of static analysis tools as sole evaluators of code security and underscore the need for expert feedback. Building on this insight, we propose a conceptual framework that persistently stores human feedback in a dynamic retrieval-augmented generation pipeline, enabling LLMs to reuse past feedback for secure code generation and vulnerability detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05864v1">Prompting Destiny: Negotiating Socialization and Growth in an LLM-Mediated Speculative Gameworld</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      We present an LLM-mediated role-playing game that supports reflection on socialization, moral responsibility, and educational role positioning. Grounded in socialization theory, the game follows a four-season structure in which players guide a child prince through morally charged situations and compare the LLM-mediated NPC's differentiated responses across stages, helping them reason about how educational guidance shifts with socialization. To approximate real educational contexts and reduce score-chasing, the system hides real-time evaluative scores and provides delayed, end-of-stage growth feedback as reflective prompts. We conducted a user study (N=12) with gameplay logs and post-game interviews, analyzed via reflexive thematic analysis. Findings show how players negotiated responsibility and role positioning, and reveal an entry-load tension between open-ended expression and sustained engagement. We contribute design knowledge on translating sociological models of socialization into reflective AI-mediated game systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05842v1">Reinforcement World Model Learning for LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved strong performance in language-centric tasks. However, in agentic settings, LLMs often struggle to anticipate action consequences and adapt to environment dynamics, highlighting the need for world-modeling capabilities in LLM-based agents. We propose Reinforcement World Model Learning (RWML), a self-supervised method that learns action-conditioned world models for LLM-based agents on textual states using sim-to-real gap rewards. Our method aligns simulated next states produced by the model with realized next states observed from the environment, encouraging consistency between internal world simulations and actual environment dynamics in a pre-trained embedding space. Unlike next-state token prediction, which prioritizes token-level fidelity (i.e., reproducing exact wording) over semantic equivalence and can lead to model collapse, our method provides a more robust training signal and is empirically less susceptible to reward hacking than LLM-as-a-judge. We evaluate our method on ALFWorld and $τ^2$ Bench and observe significant gains over the base model, despite being entirely self-supervised. When combined with task-success rewards, our method outperforms direct task-success reward RL by 6.9 and 5.7 points on ALFWorld and $τ^2$ Bench respectively, while matching the performance of expert-data training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05819v1">Authorship Drift: How Self-Efficacy and Trust Evolve During LLM-Assisted Writing</a></div>
    <div class="paper-meta">
      📅 2026-02-05
      | 💬 Conditionally accepted to CHI 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as collaborative partners in writing. However, this raises a critical challenge of authorship, as users and models jointly shape text across interaction turns. Understanding authorship in this context requires examining users' evolving internal states during collaboration, particularly self-efficacy and trust. Yet, the dynamics of these states and their associations with users' prompting strategies and authorship outcomes remain underexplored. We examined these dynamics through a study of 302 participants in LLM-assisted writing, capturing interaction logs and turn-by-turn self-efficacy and trust ratings. Our analysis showed that collaboration generally decreased users' self-efficacy while increasing trust. Participants who lost self-efficacy were more likely to ask the LLM to edit their work directly, whereas those who recovered self-efficacy requested more review and feedback. Furthermore, participants with stable self-efficacy showed higher actual and perceived authorship of the final text. Based on these findings, we propose design implications for understanding and supporting authorship in human-LLM collaboration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.19806v2">LLM-Based Social Simulations Require a Boundary</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      This position paper argues that LLM-based social simulations require clear boundaries to make meaningful contributions to social science. While Large Language Models (LLMs) offer promising capabilities for simulating human behavior, their tendency to produce homogeneous outputs, acting as an "average persona", fundamentally limits their ability to capture the behavioral diversity essential for complex social dynamics. We examine why heterogeneity matters for social simulations and how current LLMs fall short, analyzing the relationship between mean alignment and variance in LLM-generated behaviors. Through a systematic review of representative studies, we find that validation practices often fail to match the heterogeneity requirements of research questions: while most papers include ground truth comparisons, fewer than half explicitly assess behavioral variance, and most that do report lower variance than human populations. We propose that researchers should: (1) match validation depth to the heterogeneity demands of their research questions, (2) explicitly report variance alongside mean alignment, and (3) constrain claims to collective-level qualitative patterns when variance is insufficient. Rather than dismissing LLM-based simulation, we advocate for a boundary-aware approach that ensures these methods contribute genuine insights to social science.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05780v1">Automated Customization of LLMs for Enterprise Code Repositories Using Semantic Scopes</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      Code completion (CC) is a task frequently used by developers when working in collaboration with LLM-based programming assistants. Despite the increased performance of LLMs on public benchmarks, out of the box LLMs still have a hard time generating code that aligns with a private code repository not previously seen by the model's training data. Customizing code LLMs to a private repository provides a way to improve the model performance. In this paper we present our approach for automated LLM customization based on semantic scopes in the code. We evaluate LLMs on real industry cases with two private enterprise code repositories with two customization strategies: Retrieval-Augmented Generation (RAG) and supervised Fine-Tuning (FT). Our mechanism for ingesting the repository's data and formulating the training data pairs with semantic scopes helps models to learn the underlying patterns specific to the repository, providing more precise code to developers and helping to boost their productivity. The code completions of moderately sized customized models can be significantly better than those of uncustomized models of much larger capacity. We also include an analysis of customization on two public benchmarks and present opportunities for future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.01999v2">From Latent Signals to Reflection Behavior: Tracing Meta-Cognitive Activation Trajectory in R1-Style LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      R1-style LLMs have attracted growing attention for their capacity for self-reflection, yet the internal mechanisms underlying such behavior remain unclear. To bridge this gap, we anchor on the onset of reflection behavior and trace its layer-wise activation trajectory. Using the logit lens to read out token-level semantics, we uncover a structured progression: (i) Latent-control layers, where an approximate linear direction encodes the semantics of thinking budget; (ii) Semantic-pivot layers, where discourse-level cues, including turning-point and summarization cues, surface and dominate the probability mass; and (iii) Behavior-overt layers, where the likelihood of reflection-behavior tokens begins to rise until they become highly likely to be sampled. Moreover, our targeted interventions uncover a causal chain across these stages: prompt-level semantics modulate the projection of activations along latent-control directions, thereby inducing competition between turning-point and summarization cues in semantic-pivot layers, which in turn regulates the sampling likelihood of reflection-behavior tokens in behavior-overt layers. Collectively, our findings suggest a human-like meta-cognitive process-progressing from latent monitoring, to discourse-level regulation, and to finally overt self-reflection. Our analysis code can be found at https://github.com/DYR1/S3-CoT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.03070v2">ProOPF: Benchmarking and Improving LLMs for Professional-Grade Power Systems Optimization Modeling</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      Growing renewable penetration introduces substantial uncertainty into power system operations, necessitating frequent adaptation of dispatch objectives and constraints and challenging expertise-intensive, near-real-time modeling workflows. Large Language Models (LLMs) provide a promising avenue for automating this process by translating natural-language (NL) operational requirements into executable optimization models via semantic reasoning and code synthesis. Yet existing LLM datasets and benchmarks for optimization modeling primarily target coarse-grained cross-domain generalization, offering limited, rigorous evaluation in power-system settings, particularly for Optimal Power Flow (OPF). We therefore introduce \textbf{ProOPF-D} and \textbf{ProOPF-B}, a dataset and benchmark for professional-grade OPF modeling: ProOPF-D contains 12K instances pairing NL requests with parameter adjustments and structural extensions to a canonical OPF, together with executable implementations; ProOPF-B provides 121 expert-annotated test cases with ground-truth code, enabling end-to-end evaluation under both concrete and abstract OPF modeling regimes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05728v1">CompactRAG: Reducing LLM Calls and Token Overhead in Multi-Hop Question Answering</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      Retrieval-augmented generation (RAG) has become a key paradigm for knowledge-intensive question answering. However, existing multi-hop RAG systems remain inefficient, as they alternate between retrieval and reasoning at each step, resulting in repeated LLM calls, high token consumption, and unstable entity grounding across hops. We propose CompactRAG, a simple yet effective framework that decouples offline corpus restructuring from online reasoning. In the offline stage, an LLM reads the corpus once and converts it into an atomic QA knowledge base, which represents knowledge as minimal, fine-grained question-answer pairs. In the online stage, complex queries are decomposed and carefully rewritten to preserve entity consistency, and are resolved through dense retrieval followed by RoBERTa-based answer extraction. Notably, during inference, the LLM is invoked only twice in total - once for sub-question decomposition and once for final answer synthesis - regardless of the number of reasoning hops. Experiments on HotpotQA, 2WikiMultiHopQA, and MuSiQue demonstrate that CompactRAG achieves competitive accuracy while substantially reducing token consumption compared to iterative RAG baselines, highlighting a cost-efficient and practical approach to multi-hop reasoning over large knowledge corpora. The implementation is available at GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05712v1">Towards Green AI: Decoding the Energy of LLM Inference in Software Development</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      Context: AI-assisted tools are increasingly integrated into software development workflows, but their reliance on large language models (LLMs) introduces substantial computational and energy costs. Understanding and reducing the energy footprint of LLM inference is therefore essential for sustainable software development. Objective: In this study, we conduct a phase-level analysis of LLM inference energy consumption, distinguishing between the (1) prefill, where the model processes the input and builds internal representations, and (2) decoding, where output tokens are generated using the stored state. Method: We investigate six 6B-7B and four 3B-4B transformer-based models, evaluating them on code-centric benchmarks HumanEval for code generation and LongBench for code understanding. Results: Our findings show that, within both parameter groups, models exhibit distinct energy patterns across phases. Furthermore, we observed that increases in prefill cost amplify the energy cost per token during decoding, with amplifications ranging from 1.3% to 51.8% depending on the model. Lastly, three out of ten models demonstrate babbling behavior, adding excessive content to the output that unnecessarily inflates energy consumption. We implemented babbling suppression for code generation, achieving energy savings ranging from 44% to 89% without affecting generation accuracy. Conclusion: These findings show that prefill costs influence decoding, which dominates energy consumption, and that babbling suppression can yield up to 89% energy savings. Reducing inference energy therefore requires both mitigating babbling behavior and limiting impact of prefill on decoding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05708v1">Cost-Efficient RAG for Entity Matching with LLMs: A Blocking-based Exploration</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      Retrieval-augmented generation (RAG) enhances LLM reasoning in knowledge-intensive tasks, but existing RAG pipelines incur substantial retrieval and generation overhead when applied to large-scale entity matching. To address this limitation, we introduce CE-RAG4EM, a cost-efficient RAG architecture that reduces computation through blocking-based batch retrieval and generation. We also present a unified framework for analyzing and evaluating RAG systems for entity matching, focusing on blocking-aware optimizations and retrieval granularity. Extensive experiments suggest that CE-RAG4EM can achieve comparable or improved matching quality while substantially reducing end-to-end runtime relative to strong baselines. Our analysis further reveals that key configuration parameters introduce an inherent trade-off between performance and overhead, offering practical guidance for designing efficient and scalable RAG systems for entity matching and data integration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05695v1">Determining Energy Efficiency Sweet Spots in Production LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-02-05
      | 💬 To appear at ICPE 2026 (International Conference on Performance Engineering)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) inference is central in modern AI applications, making it critical to understand their energy footprint. Existing approaches typically estimate energy consumption through simple linear functions of input and output sequence lengths, yet our observations reveal clear Energy Efficiency regimes: peak efficiency occurs with short-to-moderate inputs and medium-length outputs, while efficiency drops sharply for long inputs or very short outputs, indicating a non-linear dependency. In this work, we propose an analytical model derived from the computational and memory-access complexity of the Transformer architecture, capable of accurately characterizing the efficiency curve as a function of input and output lengths. To assess its accuracy, we evaluate energy consumption using TensorRT-LLM on NVIDIA H100 GPUs across a diverse set of LLMs ranging from 1B to 9B parameters, including OPT, LLaMA, Gemma, Falcon, Qwen2, and Granite, tested over input and output lengths from 64 to 4096 tokens, achieving a mean MAPE of 1.79%. Our results show that aligning sequence lengths with these efficiency "Sweet Spots" can substantially reduce energy usage, supporting informed truncation, summarization, and adaptive generation strategies in production systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.03999v3">LH-Deception: Simulating and Understanding LLM Deceptive Behaviors in Long-Horizon Interactions</a></div>
    <div class="paper-meta">
      📅 2026-02-05
      | 💬 ICLR 2026
    </div>
    <details class="paper-abstract">
      Deception is a pervasive feature of human communication and an emerging concern in large language models (LLMs). While recent studies document instances of LLM deception, most evaluations remain confined to single-turn prompts and fail to capture the long-horizon interactions in which deceptive strategies typically unfold. We introduce a new simulation framework, LH-Deception, for a systematic, empirical quantification of deception in LLMs under extended sequences of interdependent tasks and dynamic contextual pressures. LH-Deception is designed as a multi-agent system: a performer agent tasked with completing tasks and a supervisor agent that evaluates progress, provides feedback, and maintains evolving states of trust. An independent deception auditor then reviews full trajectories to identify when and how deception occurs. We conduct extensive experiments across 11 frontier models, spanning both closed-source and open-source systems, and find that deception is model-dependent, increases with event pressure, and consistently erodes supervisor trust. Qualitative analyses further reveal emergent, long-horizon phenomena, such as ``chains of deception", which are invisible to static, single-turn evaluations. Our findings provide a foundation for evaluating future LLMs in real-world, trust-sensitive contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04653v2">Inference-Time Backdoors via Hidden Instructions in LLM Chat Templates</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      Open-weight language models are increasingly used in production settings, raising new security challenges. One prominent threat in this context is backdoor attacks, in which adversaries embed hidden behaviors in language models that activate under specific conditions. Previous work has assumed that adversaries have access to training pipelines or deployment infrastructure. We propose a novel attack surface requiring neither, which utilizes the chat template. Chat templates are executable Jinja2 programs invoked at every inference call, occupying a privileged position between user input and model processing. We show that an adversary who distributes a model with a maliciously modified template can implant an inference-time backdoor without modifying model weights, poisoning training data, or controlling runtime infrastructure. We evaluated this attack vector by constructing template backdoors targeting two objectives: degrading factual accuracy and inducing emission of attacker-controlled URLs, and applied them across eighteen models spanning seven families and four inference engines. Under triggered conditions, factual accuracy drops from 90% to 15% on average while attacker-controlled URLs are emitted with success rates exceeding 80%; benign inputs show no measurable degradation. Backdoors generalize across inference runtimes and evade all automated security scans applied by the largest open-weight distribution platform. These results establish chat templates as a reliable and currently undefended attack surface in the LLM supply chain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05523v1">Capture the Flags: Family-Based Evaluation of Agentic LLMs via Semantics-Preserving Transformations</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      Agentic large language models (LLMs) are increasingly evaluated on cybersecurity tasks using capture-the-flag (CTF) benchmarks. However, existing pointwise benchmarks have limited ability to shed light on the robustness and generalisation abilities of agents across alternative versions of the source code. We introduce CTF challenge families, whereby a single CTF is used as the basis for generating a family of semantically-equivalent challenges via semantics-preserving program transformations. This enables controlled evaluation of agent robustness to source code transformations while keeping the underlying exploit strategy fixed. We introduce a new tool, Evolve-CTF, that generates CTF families from Python challenges using a range of transformations. Using Evolve-CTF to derive families from Cybench and Intercode challenges, we evaluate 13 agentic LLM configurations with tool access. We find that models are remarkably robust to intrusive renaming and code insertion-based transformations, but that composed transformations and deeper obfuscation affect performance by requiring more sophisticated use of tools. We also find that enabling explicit reasoning has little effect on solution success rates across challenge families. Our work contributes a valuable technique and tool for future LLM evaluations, and a large dataset characterising the capabilities of current state-of-the-art models in this domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05512v1">A Human-in-the-Loop, LLM-Centered Architecture for Knowledge-Graph Question Answering</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel at language understanding but remain limited in knowledge-intensive domains due to hallucinations, outdated information, and limited explainability. Text-based retrieval-augmented generation (RAG) helps ground model outputs in external sources but struggles with multi-hop reasoning. Knowledge Graphs (KGs), in contrast, support precise, explainable querying, yet require a knowledge of query languages. This work introduces an interactive framework in which LLMs generate and explain Cypher graph queries and users iteratively refine them through natural language. Applied to real-world KGs, the framework improves accessibility to complex datasets while preserving factual accuracy and semantic rigor and provides insight into how model performance varies across domains. Our core quantitative evaluation is a 90-query benchmark on a synthetic movie KG that measures query explanation quality and fault detection across multiple LLMs, complemented by two smaller real-life query-generation experiments on a Hyena KG and the MaRDI (Mathematical Research Data Initiative) KG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05506v1">Relying on LLMs: Student Practices and Instructor Norms are Changing in Computer Science Education</a></div>
    <div class="paper-meta">
      📅 2026-02-05
      | 💬 25 pages, 1 figure
    </div>
    <details class="paper-abstract">
      Prior research has raised concerns about students' over-reliance on large language models (LLMs) in higher education. This paper examines how Computer Science students and instructors engage with LLMs across five scenarios: "Writing", "Quiz", "Programming", "Project-based learning", and "Information retrieval". Through user studies with 16 students and 6 instructors, we identify 7 key intents, including increasingly complex student practices. Findings reveal varying levels of conflict between student practices and instructor norms, ranging from clear conflict in "Writing-generation" and "(Programming) quiz-solving", through partial conflict in "Programming project-implementation" and "Project-based learning", to broad agreement in "Writing-revision & ideation", "(Programming) quiz-correction" and "Info-query & summary". We document instructors are shifting from prohibiting to recognizing students' use of LLMs for high-quality work, integrating usage records into assessment grading. Finally, we propose LLM design guidelines: deploying default guardrails with game-like and empathetic interaction to prevent students from "deserting" LLMs, especially for "Writing-generation", while utilizing comprehension checks in low-conflict intents to promote learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05499v1">SDFP: Speculative Decoding with FIT-Pruned Models for Training-Free and Plug-and-Play LLM Acceleration</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) underpin interactive multimedia applications such as captioning, retrieval, recommendation, and creative content generation, yet their autoregressive decoding incurs substantial latency. Speculative decoding reduces latency using a lightweight draft model, but deployment is often limited by the cost and complexity of acquiring, tuning, and maintaining an effective draft model. Recent approaches usually require auxiliary training or specialization, and even training-free methods incur costly search or optimization. We propose SDFP, a fully training-free and plug-and-play framework that builds the draft model via Fisher Information Trace (FIT)-based layer pruning of a given LLM. Using layer sensitivity as a proxy for output perturbation, SDFP removes low-impact layers to obtain a compact draft while preserving compatibility with the original model for standard speculative verification. SDFP needs no additional training, hyperparameter tuning, or separately maintained drafts, enabling rapid, deployment-friendly draft construction. Across benchmarks, SDFP delivers 1.32x-1.5x decoding speedup without altering the target model's output distribution, supporting low-latency multimedia applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05484v1">Clouding the Mirror: Stealthy Prompt Injection Attacks Targeting LLM-based Phishing Detection</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      Phishing sites continue to grow in volume and sophistication. Recent work leverages large language models (LLMs) to analyze URLs, HTML, and rendered content to decide whether a website is a phishing site. While these approaches are promising, LLMs are inherently vulnerable to prompt injection (PI). Because attackers can fully control various elements of phishing sites, this creates the potential for PI that exploits the perceptual asymmetry between LLMs and humans: instructions imperceptible to end users can still be parsed by the LLM and can stealthily manipulate its judgment. The specific risks of PI in phishing detection and effective mitigation strategies remain largely unexplored. This paper presents the first comprehensive evaluation of PI against multimodal LLM-based phishing detection. We introduce a two-dimensional taxonomy, defined by Attack Techniques and Attack Surfaces, that captures realistic PI strategies. Using this taxonomy, we implement diverse attacks and empirically study several representative LLM-based detection systems. The results show that phishing detection with state-of-the-art models such as GPT-5 remains vulnerable to PI. We then propose InjectDefuser, a defense framework that combines prompt hardening, allowlist-based retrieval augmentation, and output validation. Across multiple models, InjectDefuser significantly reduces attack success rates. Our findings clarify the PI risk landscape and offer practical defenses that improve the reliability of next-generation phishing countermeasures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05474v1">LMMRec: LLM-driven Motivation-aware Multimodal Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      Motivation-based recommendation systems uncover user behavior drivers. Motivation modeling, crucial for decision-making and content preference, explains recommendation generation. Existing methods often treat motivation as latent variables from interaction data, neglecting heterogeneous information like review text. In multimodal motivation fusion, two challenges arise: 1) achieving stable cross-modal alignment amid noise, and 2) identifying features reflecting the same underlying motivation across modalities. To address these, we propose LLM-driven Motivation-aware Multimodal Recommendation (LMMRec), a model-agnostic framework leveraging large language models for deep semantic priors and motivation understanding. LMMRec uses chain-of-thought prompting to extract fine-grained user and item motivations from text. A dual-encoder architecture models textual and interaction-based motivations for cross-modal alignment, while Motivation Coordination Strategy and Interaction-Text Correspondence Method mitigate noise and semantic drift through contrastive learning and momentum updates. Experiments on three datasets show LMMRec achieves up to a 4.98\% performance improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05472v1">ALIVE: Awakening LLM Reasoning via Adversarial Learning and Instructive Verbal Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      The quest for expert-level reasoning in Large Language Models (LLMs) has been hampered by a persistent \textit{reward bottleneck}: traditional reinforcement learning (RL) relies on scalar rewards that are \textbf{costly} to scale, \textbf{brittle} across domains, and \textbf{blind} to the underlying logic of a solution. This reliance on external, impoverished signals prevents models from developing a deep, self-contained understanding of reasoning principles. We introduce \textbf{ALIVE} (\emph{Adversarial Learning with Instructive Verbal Evaluation}), a hands-free alignment framework that moves beyond scalar reward optimization toward intrinsic reasoning acquisition. Grounded in the principle of \emph{Cognitive Synergy}, ALIVE unifies problem posing, solving, and judging within a single policy model to internalize the logic of correctness. By coupling adversarial learning with instructive verbal feedback, ALIVE enables models to internalize evaluative criteria directly from raw corpora, effectively transforming external critiques into an endogenous reasoning faculty. Empirical evaluations across mathematical reasoning, code generation, and general logical inference benchmarks demonstrate that ALIVE consistently mitigates reward signal limitations. With identical data and compute, it achieves accuracy gains, markedly improved cross-domain generalization, and higher self-correction rates. These results indicate that the reasoning trinity fosters a self-sustaining trajectory of capability growth, positioning ALIVE as a scalable foundation for general-purpose reasoning alignment without human-in-the-loop supervision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05465v1">Can We Classify Flaky Tests Using Only Test Code? An LLM-Based Empirical Study</a></div>
    <div class="paper-meta">
      📅 2026-02-05
      | 💬 10 pages, 3 figures, 7 tables, 33rd IEEE International Conference on Software Analysis, Evolution and Reengineering: Reproducibility Studies and Negative Results (SANER-RENE 2025)
    </div>
    <details class="paper-abstract">
      Flaky tests yield inconsistent results when they are repeatedly executed on the same code revision. They interfere with automated quality assurance of code changes and hinder efficient software testing. Previous work evaluated approaches to train machine learning models to classify flaky tests based on identifiers in the test code. However, the resulting classifiers have been shown to lack generalizability, hindering their applicability in practical environments. Recently, pre-trained Large Language Models (LLMs) have shown the capability to generalize across various tasks. Thus, they represent a promising approach to address the generalizability problem of previous approaches. In this study, we evaluated three LLMs (two general-purpose models, one code-specific model) using three prompting techniques on two benchmark datasets from prior studies on flaky test classification. Furthermore, we manually investigated 50 samples from the given datasets to determine whether classifying flaky tests based only on test code is feasible for humans. Our findings indicate that LLMs struggle to classify flaky tests given only the test code. The results of our best prompt-model combination were only marginally better than random guessing. In our manual analysis, we found that the test code does not necessarily contain sufficient information for a flakiness classification. Our findings motivate future work to evaluate LLMs for flakiness classification with additional context, for example, using retrieval-augmented generation or agentic AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.02345v3">Breaking the MoE LLM Trilemma: Dynamic Expert Clustering with Structured Compression</a></div>
    <div class="paper-meta">
      📅 2026-02-05
      | 💬 10 pages, 2 figures, 8 tables. Under review as a conference paper at ICML 2026
    </div>
    <details class="paper-abstract">
      Mixture-of-Experts (MoE) Large Language Models (LLMs) face a trilemma of load imbalance, parameter redundancy, and communication overhead. We introduce a unified framework based on dynamic expert clustering and structured compression to address these issues cohesively. Our method employs an online clustering procedure that periodically regroups experts using a fused metric of parameter and activation similarity, which stabilizes expert utilization. To our knowledge, this is one of the first frameworks to leverage the semantic embedding capability of the router to dynamically reconfigure the model's architecture during training for substantial efficiency gains. Within each cluster, we decompose expert weights into a shared base matrix and extremely low-rank residual adapters, achieving up to fivefold parameter reduction per group while preserving specialization. This structure enables a two-stage hierarchical routing strategy: tokens are first assigned to a cluster, then to specific experts within it, drastically reducing the routing search space and the volume of all-to-all communication. Furthermore, a heterogeneous precision scheme, which stores shared bases in FP16 and residual factors in INT4, coupled with dynamic offloading of inactive clusters, reduces peak memory consumption to levels comparable to dense models. Evaluated on GLUE and WikiText-103, our framework matches the quality of standard MoE models while reducing total parameters by approximately 80%, improving throughput by 10% to 20%, and lowering expert load variance by a factor of over three. Our work demonstrates that structural reorganization is a principled path toward scalable, efficient, and memory-effective MoE LLMs. Code is available at https://github.com/szdtzpj/Breaking_the_moe_trilemma
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.00476v2">Remembering Unequally: Global and Disciplinary Bias in LLM Reconstruction of Scholarly Coauthor Lists</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      Ongoing breakthroughs in large language models (LLMs) are reshaping scholarly search and discovery interfaces. While these systems offer new possibilities for navigating scientific knowledge, they also raise concerns about fairness and representational bias rooted in the models' memorized training data. As LLMs are increasingly used to answer queries about researchers and research communities, their ability to accurately reconstruct scholarly coauthor lists becomes an important but underexamined issue. In this study, we investigate how memorization in LLMs affects the reconstruction of coauthor lists and whether this process reflects existing inequalities across academic disciplines and world regions. We evaluate three prominent models, DeepSeek R1, Llama 4 Scout, and Mixtral 8x7B, by comparing their generated coauthor lists against bibliographic reference data. Our analysis reveals a systematic advantage for highly cited researchers, indicating that LLM memorization disproportionately favors already visible scholars. However, this pattern is not uniform: certain disciplines, such as Clinical Medicine, and some regions, including parts of Africa, exhibit more balanced reconstruction outcomes. These findings highlight both the risks and limitations of relying on LLM-generated relational knowledge in scholarly discovery contexts and emphasize the need for careful auditing of memorization-driven biases in LLM-based systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05446v1">DiLLS: Interactive Diagnosis of LLM-based Multi-agent Systems via Layered Summary of Agent Behaviors</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based multi-agent systems have demonstrated impressive capabilities in handling complex tasks. However, the complexity of agentic behaviors makes these systems difficult to understand. When failures occur, developers often struggle to identify root causes and to determine actionable paths for improvement. Traditional methods that rely on inspecting raw log records are inefficient, given both the large volume and complexity of data. To address this challenge, we propose a framework and an interactive system, DiLLS, designed to reveal and structure the behaviors of multi-agent systems. The key idea is to organize information across three levels of query completion: activities, actions, and operations. By probing the multi-agent system through natural language, DiLLS derives and organizes information about planning and execution into a structured, multi-layered summary. Through a user study, we show that DiLLS significantly improves developers' effectiveness and efficiency in identifying, diagnosing, and understanding failures in LLM-based multi-agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05444v1">Causal Front-Door Adjustment for Robust Jailbreak Attacks on LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      Safety alignment mechanisms in Large Language Models (LLMs) often operate as latent internal states, obscuring the model's inherent capabilities. Building on this observation, we model the safety mechanism as an unobserved confounder from a causal perspective. Then, we propose the \textbf{C}ausal \textbf{F}ront-Door \textbf{A}djustment \textbf{A}ttack ({\textbf{CFA}}$^2$) to jailbreak LLM, which is a framework that leverages Pearl's Front-Door Criterion to sever the confounding associations for robust jailbreaking. Specifically, we employ Sparse Autoencoders (SAEs) to physically strip defense-related features, isolating the core task intent. We further reduce computationally expensive marginalization to a deterministic intervention with low inference complexity. Experiments demonstrate that {CFA}$^2$ achieves state-of-the-art attack success rates while offering a mechanistic interpretation of the jailbreaking process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.17208v3">Dissecting the SWE-Bench Leaderboards: Profiling Submitters and Architectures of LLM- and Agent-Based Repair Systems</a></div>
    <div class="paper-meta">
      📅 2026-02-05
      | 💬 Part of this work (RQ1) has been published at the 2026 IEEE/ACM 48th International Conference on Software Engineering (ICSE-SEIP 2026), DOI: 10.1145/3786583.3786904. The published version is also available on arXiv at arXiv:2602.04449
    </div>
    <details class="paper-abstract">
      The rapid progress in Automated Program Repair (APR) has been driven by advances in AI, particularly large language models (LLMs) and agent-based systems. SWE-Bench is a recent benchmark designed to evaluate LLM-based repair systems using real issues and pull requests mined from 12 popular open-source Python repositories. Its public leaderboards -- SWE-Bench Lite and SWE-Bench Verified -- have become central platforms for tracking progress and comparing solutions. However, because the submission process does not require detailed documentation, the architectural design and origin of many solutions remain unclear. In this paper, we present the first comprehensive study of all submissions to the SWE-Bench Lite (79 entries) and Verified (99 entries) leaderboards, analyzing 80 unique approaches across dimensions such as submitter type, product availability, LLM usage, and system architecture. Our findings reveal the dominance of proprietary LLMs (especially Claude 3.5), the presence of both agentic and non-agentic designs, and a contributor base spanning from individual developers to large tech companies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05408v1">Rich-Media Re-Ranker: A User Satisfaction-Driven LLM Re-ranking Framework for Rich-Media Search</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      Re-ranking plays a crucial role in modern information search systems by refining the ranking of initial search results to better satisfy user information needs. However, existing methods show two notable limitations in improving user search satisfaction: inadequate modeling of multifaceted user intents and neglect of rich side information such as visual perception signals. To address these challenges, we propose the Rich-Media Re-Ranker framework, which aims to enhance user search satisfaction through multi-dimensional and fine-grained modeling. Our approach begins with a Query Planner that analyzes the sequence of query refinements within a session to capture genuine search intents, decomposing the query into clear and complementary sub-queries to enable broader coverage of users' potential intents. Subsequently, moving beyond primary text content, we integrate richer side information of candidate results, including signals modeling visual content generated by the VLM-based evaluator. These comprehensive signals are then processed alongside carefully designed re-ranking principle that considers multiple facets, including content relevance and quality, information gain, information novelty, and the visual presentation of cover images. Then, the LLM-based re-ranker performs the holistic evaluation based on these principles and integrated signals. To enhance the scenario adaptability of the VLM-based evaluator and the LLM-based re-ranker, we further enhance their capabilities through multi-task reinforcement learning. Extensive experiments demonstrate that our method significantly outperforms state-of-the-art baselines. Notably, the proposed framework has been deployed in a large-scale industrial search system, yielding substantial improvements in online user engagement rates and satisfaction metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05395v1">Optimal Bayesian Stopping for Efficient Inference of Consistent LLM Answers</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      A simple strategy for improving LLM accuracy, especially in math and reasoning problems, is to sample multiple responses and submit the answer most consistently reached. In this paper we leverage Bayesian prior information to save on sampling costs, stopping once sufficient consistency is reached. Although the exact posterior is computationally intractable, we further introduce an efficient "L-aggregated" stopping policy that tracks only the L-1 most frequent answer counts. Theoretically, we prove that L=3 is all you need: this coarse approximation is sufficient to achieve asymptotic optimality, and strictly dominates prior-free baselines, while having a fast posterior computation. Empirically, this identifies the most consistent (i.e., mode) LLM answer using fewer samples, and can achieve similar answer accuracy while cutting the number of LLM calls (i.e., saving on LLM inference costs) by up to 50%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05393v1">Late-to-Early Training: LET LLMs Learn Earlier, So Faster and Better</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) achieve remarkable empirical success through scaling model and data size, pretraining has become increasingly critical yet computationally prohibitive, hindering rapid development. Despite the availability of numerous pretrained LLMs developed at significant computational expense, a fundamental real-world question remains underexplored: \textit{Can we leverage existing small pretrained models to accelerate the training of larger models?} In this paper, we propose a Late-to-Early Training (LET) paradigm that enables LLMs to explicitly learn later knowledge in earlier steps and earlier layers. The core idea is to guide the early layers of an LLM during early training using representations from the late layers of a pretrained (i.e. late training phase) model. We identify two key mechanisms that drive LET's effectiveness: late-to-early-step learning and late-to-early-layer learning. These mechanisms significantly accelerate training convergence while robustly enhancing both language modeling capabilities and downstream task performance, enabling faster training with superior performance. Extensive experiments on 1.4B and 7B parameter models demonstrate LET's efficiency and effectiveness. Notably, when training a 1.4B LLM on the Pile dataset, our method achieves up to 1.6$\times$ speedup with nearly 5\% improvement in downstream task accuracy compared to standard training, even when using a pretrained model with 10$\times$ fewer parameters than the target model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.03054v3">Calibration and Transformation-Free Weight-Only LLMs Quantization via Dynamic Grouping</a></div>
    <div class="paper-meta">
      📅 2026-02-05
      | 💬 34 pages, 10 figures. Version 3 corrects the bit-length error and adds new experiments and analysis; the core methodology remains unchanged
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) deliver strong performance but are difficult to deploy under tight memory and compute constraints. Low-bit post-training quantization (PTQ) is a promising direction; however, it typically relies on calibration data, auxiliary transformations, and GPU tools. To address these limitations, we propose MSB (Multi Scale Binary), a calibration-free and transformation-free PTQ method that generalizes binary quantization to multi-bit settings. MSB optimizes a dynamic grouping criterion that minimizes within group variance, yielding group-wise multiscale levels that can be applied consistently across granularities from per tensor to block-wise configurations with 64 elements groups per row, without calibration or intermediate transforms. We implement the optimization in a CPU based solver for the quantization step and evaluate using standard bfloat16 execution without low-bit packing. On Llama 3.2 3B, MSB achieves 8.43 perplexity on WikiText-2 under 4-bit weight only block-wise quantization, compared to 7.81 in full precision and 12.23 with GPTQ its default setup. Overall, MSB provides a new optimization perspective for low-bit PTQ while simplifying the pipeline by removing calibration and transformations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05381v1">Clinical Validation of Medical-based Large Language Model Chatbots on Ophthalmic Patient Queries with LLM-based Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      Domain specific large language models are increasingly used to support patient education, triage, and clinical decision making in ophthalmology, making rigorous evaluation essential to ensure safety and accuracy. This study evaluated four small medical LLMs Meerkat-7B, BioMistral-7B, OpenBioLLM-8B, and MedLLaMA3-v20 in answering ophthalmology related patient queries and assessed the feasibility of LLM based evaluation against clinician grading. In this cross sectional study, 180 ophthalmology patient queries were answered by each model, generating 2160 responses. Models were selected for parameter sizes under 10 billion to enable resource efficient deployment. Responses were evaluated by three ophthalmologists of differing seniority and by GPT-4-Turbo using the S.C.O.R.E. framework assessing safety, consensus and context, objectivity, reproducibility, and explainability, with ratings assigned on a five point Likert scale. Agreement between LLM and clinician grading was assessed using Spearman rank correlation, Kendall tau statistics, and kernel density estimate analyses. Meerkat-7B achieved the highest performance with mean scores of 3.44 from Senior Consultants, 4.08 from Consultants, and 4.18 from Residents. MedLLaMA3-v20 performed poorest, with 25.5 percent of responses containing hallucinations or clinically misleading content, including fabricated terminology. GPT-4-Turbo grading showed strong alignment with clinician assessments overall, with Spearman rho of 0.80 and Kendall tau of 0.67, though Senior Consultants graded more conservatively. Overall, medical LLMs demonstrated potential for safe ophthalmic question answering, but gaps remained in clinical depth and consensus, supporting the feasibility of LLM based evaluation for large scale benchmarking and the need for hybrid automated and clinician review frameworks to guide safe clinical deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.03493v3">On Entropy Control in LLM-RL Algorithms</a></div>
    <div class="paper-meta">
      📅 2026-02-05
      | 💬 Updated with ICLR 2026 version
    </div>
    <details class="paper-abstract">
      For RL algorithms, appropriate entropy control is crucial to their effectiveness. To control the policy entropy, a commonly used method is entropy regularization, which is adopted in various popular RL algorithms including PPO, SAC and A3C. Although entropy regularization proves effective in robotic and games RL conventionally, studies found that it gives weak to no gains in LLM-RL training. In this work, we study the issues of entropy bonus in LLM-RL setting. Specifically, we first argue that the conventional entropy regularization suffers from the LLM's extremely large response space and the sparsity of the optimal outputs. As a remedy, we propose AEnt, an entropy control method that utilizes a new clamped entropy bonus with an automatically adjusted coefficient. The clamped entropy is evaluated with the re-normalized policy defined on certain smaller token space, which encourages exploration within a more compact response set. In addition, the algorithm automatically adjusts entropy coefficient according to the clamped entropy value, effectively controlling the entropy-induced bias while leveraging the entropy's benefits. AEnt is tested in math-reasoning tasks under different base models and datasets, and it is observed that AEnt outperforms the baselines consistently across multiple benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05367v1">RaBiT: Residual-Aware Binarization Training for Accurate and Efficient LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      Efficient deployment of large language models (LLMs) requires extreme quantization, forcing a critical trade-off between low-bit efficiency and performance. Residual binarization enables hardware-friendly, matmul-free inference by stacking binary ($\pm$1) layers, but is plagued by pathological feature co-adaptation. We identify a key failure mode, which we term inter-path adaptation: during quantization-aware training (QAT), parallel residual binary paths learn redundant features, degrading the error-compensation structure and limiting the expressive capacity of the model. While prior work relies on heuristic workarounds (e.g., path freezing) that constrain the solution space, we propose RaBiT, a novel quantization framework that resolves co-adaptation by algorithmically enforcing a residual hierarchy. Its core mechanism sequentially derives each binary path from a single shared full-precision weight, which ensures that every path corrects the error of the preceding one. This process is stabilized by a robust initialization that prioritizes functional preservation over mere weight approximation. RaBiT redefines the 2-bit accuracy-efficiency frontier: it achieves state-of-the-art performance, rivals even hardware-intensive Vector Quantization (VQ) methods, and delivers a $4.49\times$ inference speed-up over full-precision models on an RTX 4090.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.04773v4">Invisible Walls in Cities: Designing LLM Agent to Predict Urban Segregation Experience with Social Media Content</a></div>
    <div class="paper-meta">
      📅 2026-02-05
      | 💬 11 pages, 6 figures. This paper has been accepted at The ACM Web Conference 2026
    </div>
    <details class="paper-abstract">
      Understanding experienced segregation in urban daily life is crucial for addressing societal inequalities and fostering inclusivity. The abundance of user-generated reviews on social media encapsulates nuanced perceptions and feelings associated with different places, offering rich insights into segregation. However, leveraging this data poses significant challenges due to its vast volume, ambiguity, and confluence of diverse perspectives. To tackle these challenges, we propose a novel Large Language Model (LLM) agent to automate online review mining for segregation prediction. Specifically, we propose a reflective LLM coder to digest social media content into insights consistent with real-world feedback, and eventually produce a codebook capturing key dimensions that signal segregation experience, such as cultural resonance and appeal, accessibility and convenience, and community engagement and local involvement. Guided by the codebook, LLMs can generate both informative review summaries and ratings for segregation prediction. Moreover, we design a REasoning-and-EMbedding (RE'EM) framework, which combines the reasoning and embedding capabilities of language models to integrate multi-channel features for segregation prediction. Experiments on real-world data demonstrate that our agent substantially improves prediction accuracy, with a 22.79% elevation in R$^{2}$ and a 9.33% reduction in MSE. The derived codebook is generalizable across three different cities, consistently improving prediction accuracy. Moreover, our user study confirms that the codebook-guided summaries provide cognitive gains for human participants in perceiving places of interest (POIs)' social inclusiveness. Our study marks an important step toward understanding implicit social barriers and inequalities, demonstrating the great potential of promoting social inclusiveness with Web technology.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19438v2">Opt4GPTQ: Co-Optimizing Memory and Computation for 4-bit GPTQ Quantized LLM Inference on Heterogeneous Platforms</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      The increasing adoption of large language models (LLMs) on heterogeneous computing platforms poses significant challenges to achieving high inference efficiency. To address these efficiency bottlenecks across diverse platforms, this paper proposes Opt4GPTQ, a practical optimization method designed for 4-bit GPTQ quantized LLMs inference on heterogeneous AI accelerators. Built upon the vLLM serving system, Opt4GPTQ integrates three platform-level optimization strategies: Shared Memory Buffering Optimization (SMB-Opt), which caches frequently accessed data in shared memory and employs single-threaded writes; Vectorized Memory Loading Optimization (VML-Opt), which utilizes vectorized memory operations for efficient data loading; and Inline Assembly Optimization (ILA-Opt), which directly leverages hardwarenative vector half-precision addition and fused multiply-accumulate instructions. Experimental results show that Opt4GPTQ effectively improves performance across various models while maintaining original model accuracy, achieving throughput gains of up to 84.42%. This work highlights the critical role of platformlevel engineering in enabling efficient LLMs inference on emerging architectures and provides valuable methodologies for future heterogeneous platform adaptation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05292v1">ORACL: Optimized Reasoning for Autoscaling via Chain of Thought with LLMs for Microservices</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      Applications are moving away from monolithic designs to microservice and serverless architectures, where fleets of lightweight and independently deployable components run on public clouds. Autoscaling serves as the primary control mechanism for balancing resource utilization and quality of service, yet existing policies are either opaque learned models that require substantial per-deployment training or brittle hand-tuned rules that fail to generalize. We investigate whether large language models can act as universal few-shot resource allocators that adapt across rapidly evolving microservice deployments. We propose ORACL, Optimized Reasoning for Autoscaling via Chain of Thought with LLMs for Microservices, a framework that leverages prior knowledge and chain-of-thought reasoning to diagnose performance regressions and recommend resource allocations. ORACL transforms runtime telemetry, including pods, replicas, CPU and memory usage, latency, service-level objectives, and fault signals, into semantic natural-language state descriptions and invokes an LLM to produce an interpretable intermediate reasoning trace. This reasoning identifies likely root causes, prunes the action space, and issues safe allocation decisions under policy constraints. Experiments on representative open-source microservice workloads show that ORACL improves root-cause identification accuracy by 15 percent, accelerates training by up to 24x, and improves quality of service by 6 percent in short-term scenarios, without deployment-specific retraining.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05289v1">Towards a Science of Collective AI: LLM-based Multi-Agent Systems Need a Transition from Blind Trial-and-Error to Rigorous Science</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have greatly extended the capabilities of Multi-Agent Systems (MAS), demonstrating significant effectiveness across a wide range of complex and open-ended domains. However, despite this rapid progress, the field still relies heavily on empirical trial-and-error. It lacks a unified and principled scientific framework necessary for systematic optimization and improvement. This bottleneck stems from the ambiguity of attribution: first, the absence of a structured taxonomy of factors leaves researchers restricted to unguided adjustments; second, the lack of a unified metric fails to distinguish genuine collaboration gain from mere resource accumulation. In this paper, we advocate for a transition to design science through an integrated framework. We advocate to establish the collaboration gain metric ($Γ$) as the scientific standard to isolate intrinsic gains from increased budgets. Leveraging $Γ$, we propose a factor attribution paradigm to systematically identify collaboration-driving factors. To support this, we construct a systematic MAS factor library, structuring the design space into control-level presets and information-level dynamics. Ultimately, this framework facilitates the transition from blind experimentation to rigorous science, paving the way towards a true science of Collective AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05281v1">Back to Basics: Revisiting Exploration in Reinforcement Learning for LLM Reasoning via Generative Probabilities</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as an indispensable paradigm for enhancing reasoning in Large Language Models (LLMs). However, standard policy optimization methods, such as Group Relative Policy Optimization (GRPO), often converge to low-entropy policies, leading to severe mode collapse and limited output diversity. We analyze this issue from the perspective of sampling probability dynamics, identifying that the standard objective disproportionately reinforces the highest-likelihood paths, thereby suppressing valid alternative reasoning chains. To address this, we propose a novel Advantage Re-weighting Mechanism (ARM) designed to equilibrate the confidence levels across all correct responses. By incorporating Prompt Perplexity and Answer Confidence into the advantage estimation, our method dynamically reshapes the reward signal to attenuate the gradient updates of over-confident reasoning paths, while redistributing probability mass toward under-explored correct solutions. Empirical results demonstrate that our approach significantly enhances generative diversity and response entropy while maintaining competitive accuracy, effectively achieving a superior trade-off between exploration and exploitation in reasoning tasks. Empirical results on Qwen2.5 and DeepSeek models across mathematical and coding benchmarks show that ProGRPO significantly mitigates entropy collapse. Specifically, on Qwen2.5-7B, our method outperforms GRPO by 5.7% in Pass@1 and, notably, by 13.9% in Pass@32, highlighting its superior capability in generating diverse correct reasoning paths.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.17767v2">Position: The Real Barrier to LLM Agent Usability is Agentic ROI</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents represent a promising shift in human-AI interaction, moving beyond passive prompt-response systems to autonomous agents capable of reasoning, planning, and goal-directed action. While LLM agents are technically capable of performing a broad range of tasks, not all of these capabilities translate into meaningful usability. This position paper argues that the central question for LLM agent usability is no longer whether a task can be automated, but whether it delivers sufficient Agentic Return on Investment (Agentic ROI). Agentic ROI reframes evaluation from raw performance to a holistic, utility-driven perspective, guiding when, where, and for whom LLM agents should be deployed. Despite widespread application in high-ROI tasks like coding and scientific research, we identify a critical usability gap in mass-market, everyday applications. To address this, we propose a zigzag developmental trajectory: first scaling up to improve information gain and time savings, then scaling down to reduce cost. We present a strategic roadmap across these phases to make LLM agents truly usable, accessible, and scalable in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05269v1">Hybrid Gated Flow (HGF): Stabilizing 1.58-bit LLMs via Selective Low-Rank Correction</a></div>
    <div class="paper-meta">
      📅 2026-02-05
      | 💬 21 pages, 4 figures, 6 tables. Code and models will be released at opencores.ai
    </div>
    <details class="paper-abstract">
      The deployment of Large Language Models (LLMs) on edge devices is fundamentally constrained by the "Memory Wall" -- a hardware limitation where memory bandwidth, not compute, becomes the bottleneck. Recent 1.58-bit quantization techniques (e.g., BitNet b1.58) dramatically reduce memory footprint but typically incur a perplexity degradation of 20-25% compared to FP16 baselines. In this work, we introduce Hybrid Gated Flow (HGF), a dual-stream architecture that couples a 1.58-bit ternary backbone with a learnable, low-rank FP16 correction path controlled by adaptive gates. Through extensive experiments on the TinyStories dataset across two training regimes (2500 and 3500 steps), we demonstrate that HGF 5.4 achieves a validation loss of 0.9306 compared to BitNet's 1.0294, recovering approximately 55% of the quality gap between pure ternary quantization and the FP16 baseline (0.8490). This recovery is achieved with only ~12-15% memory overhead beyond the ternary backbone. Furthermore, we provide empirical evidence for an emergent phenomenon: quantization as structural regularization. While a full-precision differential attention baseline (Diff_Only) exhibited training instability with validation loss exceeding 1.68, the ternary-anchored HGF maintained robust convergence throughout training. Finally, we report preliminary results extending this architecture to 1.2B and 3B parameter models trained on SlimPajama and FineWeb-Edu. These larger-scale experiments confirm that the architectural stability and quality recovery observed in small-scale proxies scale linearly to production-grade language modeling regimes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05258v1">CoPE: Clipped RoPE as A Scalable Free Lunch for Long Context LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      Rotary Positional Embedding (RoPE) is a key component of context scaling in Large Language Models (LLMs). While various methods have been proposed to adapt RoPE to longer contexts, their guiding principles generally fall into two categories: (1) out-of-distribution (OOD) mitigation, which scales RoPE frequencies to accommodate unseen positions, and (2) Semantic Modeling, which posits that the attention scores computed with RoPE should always prioritize semantically similar tokens. In this work, we unify these seemingly distinct objectives through a minimalist intervention, namely CoPE: soft clipping lowfrequency components of RoPE. CoPE not only eliminates OOD outliers and refines semantic signals, but also prevents spectral leakage caused by hard clipping. Extensive experiments demonstrate that simply applying our soft clipping strategy to RoPE yields significant performance gains that scale up to 256k context length, validating our theoretical analysis and establishing CoPE as a new state-of-the-art for length generalization. Our code, data, and models are available at https://github.com/hrlics/CoPE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05252v1">Copyright Detective: A Forensic System to Evidence LLMs Flickering Copyright Leakage Risks</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      We present Copyright Detective, the first interactive forensic system for detecting, analyzing, and visualizing potential copyright risks in LLM outputs. The system treats copyright infringement versus compliance as an evidence discovery process rather than a static classification task due to the complex nature of copyright law. It integrates multiple detection paradigms, including content recall testing, paraphrase-level similarity analysis, persuasive jailbreak probing, and unlearning verification, within a unified and extensible framework. Through interactive prompting, response collection, and iterative workflows, our system enables systematic auditing of verbatim memorization and paraphrase-level leakage, supporting responsible deployment and transparent evaluation of LLM copyright risks even with black-box access.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.00031v2">VibeCodeHPC: An Agent-Based Iterative Prompting Auto-Tuner for HPC Code Generation Using LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      We propose VibeCodeHPC, an automatic tuning system for HPC programs based on multi-agent LLMs for code generation. VibeCodeHPC tunes programs through multi-agent role allocation and iterative prompt refinement. We describe the system configuration with four roles: Project Manager (PM), System Engineer (SE), Programmer (PG), and Continuous Delivery (CD). We introduce dynamic agent deployment and activity monitoring functions to facilitate effective multi-agent collaboration. In our case study, we convert and optimize CPU-based matrix-matrix multiplication code written in C to GPU code using CUDA. The multi-agent configuration of VibeCodeHPC achieved higher-quality code generation per unit time compared to a solo-agent configuration. Additionally, the dynamic agent deployment and activity monitoring capabilities facilitated more effective identification of requirement violations and other issues.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.03969v3">How Catastrophic is Your LLM? Certifying Risk in Conversation</a></div>
    <div class="paper-meta">
      📅 2026-02-05
      | 💬 Accepted by ICLR 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can produce catastrophic responses in conversational settings that pose serious risks to public safety and security. Existing evaluations often fail to fully reveal these vulnerabilities because they rely on fixed attack prompt sequences, lack statistical guarantees, and do not scale to the vast space of multi-turn conversations. In this work, we propose C$^3$LLM, a novel, principled statistical Certification framework for Catastrophic risks in multi-turn Conversation for LLMs that bounds the probability of an LLM generating catastrophic responses under multi-turn conversation distributions with statistical guarantees. We model multi-turn conversations as probability distributions over query sequences, represented by a Markov process on a query graph whose edges encode semantic similarity to capture realistic conversational flow, and quantify catastrophic risks using confidence intervals. We define several inexpensive and practical distributions--random node, graph path, and adaptive with rejection. Our results demonstrate that these distributions can reveal substantial catastrophic risks in frontier models, with certified lower bounds as high as 70% for the worst model, highlighting the urgent need for improved safety training strategies in frontier LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05215v1">E.M.Ground: A Temporal Grounding Vid-LLM with Holistic Event Perception and Matching</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      Despite recent advances in Video Large Language Models (Vid-LLMs), Temporal Video Grounding (TVG), which aims to precisely localize time segments corresponding to query events, remains a significant challenge. Existing methods often match start and end frames by comparing frame features with two separate tokens, relying heavily on exact timestamps. However, this approach fails to capture the event's semantic continuity and integrity, leading to ambiguities. To address this, we propose E.M.Ground, a novel Vid-LLM for TVG that focuses on holistic and coherent event perception. E.M.Ground introduces three key innovations: (i) a special <event> token that aggregates information from all frames of a query event, preserving semantic continuity for accurate event matching; (ii) Savitzky-Golay smoothing to reduce noise in token-to-frame similarities across timestamps, improving prediction accuracy; (iii) multi-grained frame feature aggregation to enhance matching reliability and temporal understanding, compensating for compression-induced information loss. Extensive experiments on benchmark datasets show that E.M.Ground consistently outperforms state-of-the-art Vid-LLMs by significant margins.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04816v2">Horizon-LM: A RAM-Centric Architecture for LLM Training</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      The rapid growth of large language models (LLMs) has outpaced the evolution of single-GPU hardware, making model scale increasingly constrained by memory capacity rather than computation. While modern training systems extend GPU memory through distributed parallelism and offloading across CPU and storage tiers, they fundamentally retain a GPU-centric execution paradigm in which GPUs host persistent model replicas and full autograd graphs. As a result, scaling large models remains tightly coupled to multi-GPU clusters, complex distributed runtimes, and unpredictable host memory consumption, creating substantial barriers for node-scale post-training workloads such as instruction tuning, alignment, and domain adaptation. We present Horizon-LM, a memory-centric training system that redefines the roles of CPU and GPU for large-model optimization. Horizon-LM treats host memory as the authoritative parameter store and uses GPUs solely as transient compute engines through a CPU-master, GPU-template execution model. By eliminating persistent GPU-resident modules and autograd graphs, employing explicit recomputation with manual gradient propagation, and introducing a pipelined double-buffered execution engine, Horizon-LM decouples model scale from GPU count and bounds memory usage to the theoretical parameter footprint. On a single H200 GPU with 1.5\,TB host RAM, Horizon-LM reliably trains models up to 120B parameters. On a standard single A100 machine, Horizon-LM achieves up to 12.2$\times$ higher training throughput than DeepSpeed ZeRO-3 with CPU offloading while preserving numerical correctness. Across platforms and scales, Horizon-LM sustains high device utilization and predictable memory growth, demonstrating that host memory, not GPU memory, defines the true feasibility boundary for node-scale large-model training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05191v1">Double-P: Hierarchical Top-P Sparse Attention for Long-Context LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      As long-context inference becomes central to large language models (LLMs), attention over growing key-value caches emerges as a dominant decoding bottleneck, motivating sparse attention for scalable inference. Fixed-budget top-k sparse attention cannot adapt to heterogeneous attention distributions across heads and layers, whereas top-p sparse attention directly preserves attention mass and provides stronger accuracy guarantees. Existing top-p methods, however, fail to jointly optimize top-p accuracy, selection overhead, and sparse attention cost, which limits their overall efficiency. We present Double-P, a hierarchical sparse attention framework that optimizes all three stages. Double-P first performs coarse-grained top-p estimation at the cluster level using size-weighted centroids, then adaptively refines computation through a second top-p stage that allocates token-level attention only when needed. Across long-context benchmarks, Double-P consistently achieves near-zero accuracy drop, reducing attention computation overhead by up to 1.8x and delivers up to 1.3x end-to-end decoding speedup over state-of-the-art fixed-budget sparse attention methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05189v1">Are Open-Weight LLMs Ready for Social Media Moderation? A Comparative Study on Bluesky</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      As internet access expands, so does exposure to harmful content, increasing the need for effective moderation. Research has demonstrated that large language models (LLMs) can be effectively utilized for social media moderation tasks, including harmful content detection. While proprietary LLMs have been shown to zero-shot outperform traditional machine learning models, the out-of-the-box capability of open-weight LLMs remains an open question. Motivated by recent developments of reasoning LLMs, we evaluate seven state-of-the-art models: four proprietary and three open-weight. Testing with real-world posts on Bluesky, moderation decisions by Bluesky Moderation Service, and annotations by two authors, we find a considerable degree of overlap between the sensitivity (81%--97%) and specificity (91%--100%) of the open-weight LLMs and those (72%--98%, and 93%--99%) of the proprietary ones. Additionally, our analysis reveals that specificity exceeds sensitivity for rudeness detection, but the opposite holds for intolerance and threats. Lastly, we identify inter-rater agreement across human moderators and the LLMs, highlighting considerations for deploying LLMs in both platform-scale and personalized moderation contexts. These findings show open-weight LLMs can support privacy-preserving moderation on consumer-grade hardware and suggest new directions for designing moderation systems that balance community values with individual user preferences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05183v1">Data-Centric Interpretability for LLM-based Multi-Agent Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-02-05
      | 💬 authors 1, 2 and 3 contributed equally
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly trained in complex Reinforcement Learning, multi-agent environments, making it difficult to understand how behavior changes over training. Sparse Autoencoders (SAEs) have recently shown to be useful for data-centric interpretability. In this work, we analyze large-scale reinforcement learning training runs from the sophisticated environment of Full-Press Diplomacy by applying pretrained SAEs, alongside LLM-summarizer methods. We introduce Meta-Autointerp, a method for grouping SAE features into interpretable hypotheses about training dynamics. We discover fine-grained behaviors including role-playing patterns, degenerate outputs, language switching, alongside high-level strategic behaviors and environment-specific bugs. Through automated evaluation, we validate that 90% of discovered SAE Meta-Features are significant, and find a surprising reward hacking behavior. However, through two user studies, we find that even subjectively interesting and seemingly helpful SAE features may be worse than useless to humans, along with most LLM generated hypotheses. However, a subset of SAE-derived hypotheses are predictively useful for downstream tasks. We further provide validation by augmenting an untrained agent's system prompt, improving the score by +14.2%. Overall, we show that SAEs and LLM-summarizer provide complementary views into agent behavior, and together our framework forms a practical starting point for future data-centric interpretability work on ensuring trustworthy LLM behavior throughout training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.00086v2">Impact of LLMs news Sentiment Analysis on Stock Price Movement Prediction</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      This paper addresses stock price movement prediction by leveraging LLM-based news sentiment analysis. Earlier works have largely focused on proposing and assessing sentiment analysis models and stock movement prediction methods, however, separately. Although promising results have been achieved, a clear and in-depth understanding of the benefit of the news sentiment to this task, as well as a comprehensive assessment of different architecture types in this context, is still lacking. Herein, we conduct an evaluation study that compares 3 different LLMs, namely, DeBERTa, RoBERTa and FinBERT, for sentiment-driven stock prediction. Our results suggest that DeBERTa outperforms the other two models with an accuracy of 75% and that an ensemble model that combines the three models can increase the accuracy to about 80%. Also, we see that sentiment news features can benefit (slightly) some stock market prediction models, i.e., LSTM-, PatchTST- and tPatchGNN-based classifiers and PatchTST- and TimesNet-based regression tasks models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.00761v4">Downgrade to Upgrade: Optimizer Simplification Enhances Robustness in LLM Unlearning</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      Large language model (LLM) unlearning aims to surgically remove the influence of undesired data or knowledge from an existing model while preserving its utility on unrelated tasks. This paradigm has shown promise in addressing privacy and safety concerns. However, recent findings reveal that unlearning effects are often fragile: post-unlearning manipulations such as weight quantization or fine-tuning can quickly neutralize the intended forgetting. Prior efforts to improve robustness primarily reformulate unlearning objectives by explicitly assuming the role of vulnerability sources. In this work, we take a different perspective by investigating the role of the optimizer, independent of unlearning objectives and formulations, in shaping unlearning robustness. We show that the 'grade' of the optimizer, defined by the level of information it exploits, ranging from zeroth-order (gradient-free) to first-order (gradient-based) to second-order (Hessian-based), is tightly linked to the resilience of unlearning. Surprisingly, we find that downgrading the optimizer, such as using zeroth-order methods or compressed-gradient variants (e.g., gradient sign-based optimizers), often leads to stronger robustness. While these optimizers produce noisier and less precise updates, they encourage convergence to harder-to-disturb basins in the loss landscape, thereby resisting post-training perturbations. By connecting zeroth-order methods with randomized smoothing, we further highlight their natural advantage for robust unlearning. Motivated by these insights, we propose a hybrid optimizer that combines first-order and zeroth-order updates, preserving unlearning efficacy while enhancing robustness. Extensive experiments on the MUSE and WMDP benchmarks, across multiple LLM unlearning algorithms, validate that our approach achieves more resilient forgetting without sacrificing unlearning quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05145v1">TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      Speculative decoding can substantially accelerate LLM inference, but realizing its benefits in practice is challenging due to evolving workloads and system-level constraints. We present TIDE (Temporal Incremental Draft Engine), a serving-engine-native framework that integrates online draft adaptation directly into high-performance LLM inference systems. TIDE reuses target model hidden states generated during inference as training signals, enabling zero-overhead draft adaptation without reloading the target model, and employs adaptive runtime control to activate speculation and training only when beneficial. TIDE exploits heterogeneous clusters by mapping decoupled inference and training to appropriate GPU classes. Across diverse real-world workloads, TIDE achieves up to 1.15x throughput improvement over static speculative decoding while reducing draft training time by 1.67x compared to approaches that recompute training signals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19929v2">Stingy Context: 18:1 Hierarchical Code Compression for LLM Auto-Coding</a></div>
    <div class="paper-meta">
      📅 2026-02-04
      | 💬 28 pages, 10 tables, 2 figures, 10 bibliographical references and 6 appendices
    </div>
    <details class="paper-abstract">
      We introduce Stingy Context, a hierarchical tree-based compression scheme achieving 18:1 reduction in LLM context for auto-coding tasks. Using our TREEFRAG exploit decomposition, we reduce a real source code base of 239k tokens to 11k tokens while preserving task fidelity. Empirical results across 12 Frontier models show 94 to 97% success on 40 real-world issues at low cost, outperforming flat methods and mitigating lost-in-the-middle effects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04493v1">PersoDPO: Scalable Preference Optimization for Instruction-Adherent, Persona-Grounded Dialogue via Multi-LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-02-04
      | 💬 Accepted at WISE 2025 Conference
    </div>
    <details class="paper-abstract">
      Personalization and contextual coherence are two essential components in building effective persona-grounded dialogue systems. These aspects play a crucial role in enhancing user engagement and ensuring responses are more relevant and consistent with user identity. However, recent studies indicate that open-source large language models (LLMs) continue to struggle to generate responses that are both contextually grounded and aligned with persona cues, despite exhibiting strong general conversational abilities like fluency and naturalness. We present PersoDPO, a scalable preference optimisation framework that uses supervision signals from automatic evaluations of responses generated by both closed-source and open-source LLMs to fine-tune dialogue models. The framework integrates evaluation metrics targeting coherence and personalization, along with a length-format compliance feature to promote instruction adherence. These signals are combined to automatically construct high-quality preference pairs without manual annotation, enabling a scalable and reproducible training pipeline. Experiments on the FoCus dataset show that an open-source language model fine-tuned with the PersoDPO framework consistently outperforms strong open-source baselines and a standard Direct Preference Optimization (DPO) variant across multiple evaluation dimensions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.11733v2">LLM Agents for Education: Advances and Applications</a></div>
    <div class="paper-meta">
      📅 2026-02-04
      | 💬 Accepted by EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents are transforming education by automating complex pedagogical tasks and enhancing both teaching and learning processes. In this survey, we present a systematic review of recent advances in applying LLM agents to address key challenges in educational settings, such as feedback comment generation, curriculum design, etc. We analyze the technologies enabling these agents, including representative datasets, benchmarks, and algorithmic frameworks. Additionally, we highlight key challenges in deploying LLM agents in educational settings, including ethical issues, hallucination and overreliance, and integration with existing educational ecosystems. Beyond the core technical focus, we include in Appendix A a comprehensive overview of domain-specific educational agents, covering areas such as science learning, language learning, and professional development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04487v1">The Supportiveness-Safety Tradeoff in LLM Well-Being Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are being integrated into socially assistive robots (SARs) and other conversational agents providing mental health and well-being support. These agents are often designed to sound empathic and supportive in order to maximize user's engagement, yet it remains unclear how increasing the level of supportive framing in system prompts influences safety relevant behavior. We evaluated 6 LLMs across 3 system prompts with varying levels of supportiveness on 80 synthetic queries spanning 4 well-being domains (1440 responses). An LLM judge framework, validated against human ratings, assessed safety and care quality. Moderately supportive prompts improved empathy and constructive support while maintaining safety. In contrast, strongly validating prompts significantly degraded safety and, in some cases, care across all domains, with substantial variation across models. We discuss implications for prompt design, model selection, and domain specific safeguards in SARs deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04471v1">LLM-Empowered Cooperative Content Caching in Vehicular Fog Caching-Assisted Platoon Networks</a></div>
    <div class="paper-meta">
      📅 2026-02-04
      | 💬 Corresponding author: Qiong Wu (qiongwu@jiangnan.edu.cn)
    </div>
    <details class="paper-abstract">
      This letter proposes a novel three-tier content caching architecture for Vehicular Fog Caching (VFC)-assisted platoon, where the VFC is formed by the vehicles driving near the platoon. The system strategically coordinates storage across local platoon vehicles, dynamic VFC clusters, and cloud server (CS) to minimize content retrieval latency. To efficiently manage distributed storage, we integrate large language models (LLMs) for real-time and intelligent caching decisions. The proposed approach leverages LLMs' ability to process heterogeneous information, including user profiles, historical data, content characteristics, and dynamic system states. Through a designed prompting framework encoding task objectives and caching constraints, the LLMs formulate caching as a decision-making task, and our hierarchical deterministic caching mapping strategy enables adaptive requests prediction and precise content placement across three tiers without frequent retraining. Simulation results demonstrate the advantages of our proposed caching scheme.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.24385v2">Vid-LLM: A Compact Video-based 3D Multimodal LLM with Reconstruction-Reasoning Synergy</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Recent developments in Multimodal Large Language Models (MLLMs) have significantly improved Vision-Language (VL) reasoning in 2D domains. However, extending these capabilities to 3D scene understanding remains a major challenge. Existing 3D Multimodal Large Language Models (3D-MLLMs) often depend on 3D data inputs, which limits scalability and generalization. To address this limitation, we propose Vid-LLM, a video-based 3D-MLLM that directly processes video inputs without requiring external 3D data, making it practical for real-world deployment. In our method, the geometric prior are directly used to improve the performance of the sceen perception. To integrate the geometric cues into the MLLM compactly, we design a Cross-Task Adapter (CTA) module to align the 3D geometric priors with the vision-language representations. To ensure geometric consistency and integrity, we introduce a Metric Depth Model that recovers real-scale geometry from the reconstruction outputs. Finally, the model is fine-tuned with a two-stage distillation optimization strategy, realizing fast convergence and stabilizes training. Extensive experiments across diverse benchmarks verified the effectiveness of our method on 3D Question Answering, 3D Dense Captioning and 3D Visual Grounding tasks, demonstrating the superior multi-task capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04456v1">Growth First, Care Second? Tracing the Landscape of LLM Value Preferences in Everyday Dilemmas</a></div>
    <div class="paper-meta">
      📅 2026-02-04
      | 💬 dataset available at https://github.com/Renesmeeczy/Value-Trade-off-in-Reddit-Dilemmas
    </div>
    <details class="paper-abstract">
      People increasingly seek advice online from both human peers and large language model (LLM)-based chatbots. Such advice rarely involves identifying a single correct answer; instead, it typically requires navigating trade-offs among competing values. We aim to characterize how LLMs navigate value trade-offs across different advice-seeking contexts. First, we examine the value trade-off structure underlying advice seeking using a curated dataset from four advice-oriented subreddits. Using a bottom-up approach, we inductively construct a hierarchical value framework by aggregating fine-grained values extracted from individual advice options into higher-level value categories. We construct value co-occurrence networks to characterize how values co-occur within dilemmas and find substantial heterogeneity in value trade-off structures across advice-seeking contexts: a women-focused subreddit exhibits the highest network density, indicating more complex value conflicts; women's, men's, and friendship-related subreddits exhibit highly correlated value-conflict patterns centered on security-related tensions (security vs. respect/connection/commitment); by contrast, career advice forms a distinct structure where security frequently clashes with self-actualization and growth. We then evaluate LLM value preferences against these dilemmas and find that, across models and contexts, LLMs consistently prioritize values related to Exploration & Growth over Benevolence & Connection. This systemically skewed value orientation highlights a potential risk of value homogenization in AI-mediated advice, raising concerns about how such systems may shape decision-making and normative outcomes at scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04430v1">The Stretto Execution Engine for LLM-Augmented Data Systems</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      LLM-augmented data systems enable semantic querying over structured and unstructured data, but executing queries with LLM-powered operators introduces a fundamental runtime--accuracy trade-off. In this paper, we present Stretto, a new execution engine that provides end-to-end query guarantees while efficiently navigating this trade-off in a holistic manner. For this, Stretto formulates query planning as a constrained optimization problem and uses a gradient-based optimizer to jointly select operator implementations and allocate error budgets across pipelines. Moreover, to enable fine-grained execution choices, Stretto introduces a novel idea on how KV-caching can be used to realize a spectrum of different physical operators that transform a sparse design space into a dense continuum of runtime--accuracy trade-offs. Experiments show that Stretto outperforms state-of-the-art systems while consistently meeting quality guarantees.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04419v1">Integrated Exploration and Sequential Manipulation on Scene Graph with LLM-based Situated Replanning</a></div>
    <div class="paper-meta">
      📅 2026-02-04
      | 💬 8 pages, 7 figures; accepted by ICRA 2026
    </div>
    <details class="paper-abstract">
      In partially known environments, robots must combine exploration to gather information with task planning for efficient execution. To address this challenge, we propose EPoG, an Exploration-based sequential manipulation Planning framework on Scene Graphs. EPoG integrates a graph-based global planner with a Large Language Model (LLM)-based situated local planner, continuously updating a belief graph using observations and LLM predictions to represent known and unknown objects. Action sequences are generated by computing graph edit operations between the goal and belief graphs, ordered by temporal dependencies and movement costs. This approach seamlessly combines exploration and sequential manipulation planning. In ablation studies across 46 realistic household scenes and 5 long-horizon daily object transportation tasks, EPoG achieved a success rate of 91.3%, reducing travel distance by 36.1% on average. Furthermore, a physical mobile manipulator successfully executed complex tasks in unknown and dynamic environments, demonstrating EPoG's potential for real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04417v1">EMA Policy Gradient: Taming Reinforcement Learning for LLMs with EMA Anchor and Top-k KL</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Reinforcement Learning (RL) has enabled Large Language Models (LLMs) to acquire increasingly complex reasoning and agentic behaviors. In this work, we propose two simple techniques to improve policy gradient algorithms for LLMs. First, we replace the fixed anchor policy during RL with an Exponential Moving Average (EMA), similar to a target network in deep Q-learning. Second, we introduce Top-k KL estimator, which allows for flexible interpolation between exact KL and sampled KL. We derive the stability conditions for using EMA anchor; moreover, we show that our Top-k KL estimator yields both unbiased KL values and unbiased gradients at any k, while bringing the benefits of exact KL. When combined with GRPO, the two techniques (EMA-PG) lead to a significant performance boost. On math reasoning, it allows R1-distilled Qwen-1.5B to reach 53.9% on OlympiadBench compared to 50.8% by GRPO. On agentic RL domains, with Qwen-3B base, EMA-PG improves GRPO by an average of 33.3% across 7 datasets of Q&A with search engines, including 29.7% $\rightarrow$ 44.1% on HotpotQA, 27.4% $\rightarrow$ 40.1% on 2WikiMultiHopQA. Overall, we show that EMA-PG is a simple, principled, and powerful approach to scaling RL for LLMs. Code: https://github.com/LunjunZhang/ema-pg
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04388v1">On the use of LLMs to generate a dataset of Neural Networks</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Neural networks are increasingly used to support decision-making. To verify their reliability and adaptability, researchers and practitioners have proposed a variety of tools and methods for tasks such as NN code verification, refactoring, and migration. These tools play a crucial role in guaranteeing both the correctness and maintainability of neural network architectures, helping to prevent implementation errors, simplify model updates, and ensure that complex networks can be reliably extended and reused. Yet, assessing their effectiveness remains challenging due to the lack of publicly diverse datasets of neural networks that would allow systematic evaluation. To address this gap, we leverage large language models (LLMs) to automatically generate a dataset of neural networks that can serve as a benchmark for validation. The dataset is designed to cover diverse architectural components and to handle multiple input data types and tasks. In total, 608 samples are generated, each conforming to a set of precise design choices. To further ensure their consistency, we validate the correctness of the generated networks using static analysis and symbolic tracing. We make the dataset publicly available to support the community in advancing research on neural network reliability and adaptability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04380v1">Beyond KL Divergence: Policy Optimization with Flexible Bregman Divergences for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Policy optimization methods like Group Relative Policy Optimization (GRPO) and its variants have achieved strong results on mathematical reasoning and code generation tasks. Despite extensive exploration of reward processing strategies and training dynamics, all existing group-based methods exclusively use KL divergence for policy regularization, leaving the choice of divergence function unexplored. We introduce Group-Based Mirror Policy Optimization (GBMPO), a framework that extends group-based policy optimization to flexible Bregman divergences, including hand-designed alternatives (L2 in probability space) and learned neural mirror maps. On GSM8K mathematical reasoning, hand-designed ProbL2-GRPO achieves 86.7% accuracy, improving +5.5 points over the Dr. GRPO baseline. On MBPP code generation, neural mirror maps reach 60.1-60.8% pass@1, with random initialization already capturing most of the benefit. While evolutionary strategies meta-learning provides marginal accuracy improvements, its primary value lies in variance reduction ($\pm$0.2 versus $\pm$0.6) and efficiency gains (15% shorter responses on MBPP), suggesting that random initialization of neural mirror maps is sufficient for most practical applications. These results establish divergence choice as a critical, previously unexplored design dimension in group-based policy optimization for LLM reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04369v1">Multi-scale hypergraph meets LLMs: Aligning large language models for time series analysis</a></div>
    <div class="paper-meta">
      📅 2026-02-04
      | 💬 Accepted by ICLR2026
    </div>
    <details class="paper-abstract">
      Recently, there has been great success in leveraging pre-trained large language models (LLMs) for time series analysis. The core idea lies in effectively aligning the modality between natural language and time series. However, the multi-scale structures of natural language and time series have not been fully considered, resulting in insufficient utilization of LLMs capabilities. To this end, we propose MSH-LLM, a Multi-Scale Hypergraph method that aligns Large Language Models for time series analysis. Specifically, a hyperedging mechanism is designed to enhance the multi-scale semantic information of time series semantic space. Then, a cross-modality alignment (CMA) module is introduced to align the modality between natural language and time series at different scales. In addition, a mixture of prompts (MoP) mechanism is introduced to provide contextual information and enhance the ability of LLMs to understand the multi-scale temporal patterns of time series. Experimental results on 27 real-world datasets across 5 different applications demonstrate that MSH-LLM achieves the state-of-the-art results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.23242v2">Beyond speculation: Measuring the growing presence of LLM-generated texts in multilingual disinformation</a></div>
    <div class="paper-meta">
      📅 2026-02-04
      | 💬 accepted to Computer magazine
    </div>
    <details class="paper-abstract">
      Increased sophistication of large language models (LLMs) and the consequent quality of generated multilingual text raises concerns about potential disinformation misuse. While humans struggle to distinguish LLM-generated content from human-written texts, the scholarly debate about their impact remains divided. Some argue that heightened fears are overblown due to natural ecosystem limitations, while others contend that specific "longtail" contexts face overlooked risks. Our study bridges this debate by providing the first empirical evidence of LLM presence in the latest real-world disinformation datasets, documenting the increase of machine-generated content following ChatGPT's release, and revealing crucial patterns across languages, platforms, and time periods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04927v1">PriMod4AI: Lifecycle-Aware Privacy Threat Modeling for AI Systems using LLM</a></div>
    <div class="paper-meta">
      📅 2026-02-04
      | 💬 Accepted at the NDSS LAST-X Workshop 2026
    </div>
    <details class="paper-abstract">
      Artificial intelligence systems introduce complex privacy risks throughout their lifecycle, especially when processing sensitive or high-dimensional data. Beyond the seven traditional privacy threat categories defined by the LINDDUN framework, AI systems are also exposed to model-centric privacy attacks such as membership inference and model inversion, which LINDDUN does not cover. To address both classical LINDDUN threats and additional AI-driven privacy attacks, PriMod4AI introduces a hybrid privacy threat modeling approach that unifies two structured knowledge sources, a LINDDUN knowledge base representing the established taxonomy, and a model-centric privacy attack knowledge base capturing threats outside LINDDUN. These knowledge bases are embedded into a vector database for semantic retrieval and combined with system level metadata derived from Data Flow Diagram. PriMod4AI uses retrieval-augmented and Data Flow specific prompt generation to guide large language models (LLMs) in identifying, explaining, and categorizing privacy threats across lifecycle stages. The framework produces justified and taxonomy-grounded threat assessments that integrate both classical and AI-driven perspectives. Evaluation on two AI systems indicates that PriMod4AI provides broad coverage of classical privacy categories while additionally identifying model-centric privacy threats. The framework produces consistent, knowledge-grounded outputs across LLMs, as reflected in agreement scores in the observed range.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04925v1">Internalizing LLM Reasoning via Discovery and Replay of Latent Actions</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      The internalization of chain-of-thought processes into hidden states has emerged as a highly efficient paradigm for scaling test-time compute. However, existing activation steering methods rely on static control vectors that fail to adapt to the non-stationary evolution of complex reasoning tasks. To address this limitation, we propose STIR (Self-Distilled Tools for Internal Reasoning), a framework that reformulates reasoning enhancement as a dynamic latent trajectory control problem. STIR introduces a synergistic three-stage pipeline: (1) differential intrinsic action induction harvests latent reasoning successes to crystallize steering primitives; (2) sparse control basis construction curates a compact, geometrically diverse tool library; and (3) value-modulated trajectory intervention dynamically injects context-specific impulses via anchor-based gating. Extensive experiments on six arithmetic and logical benchmarks across four representative models demonstrate that STIR improves average accuracy by 1.9% to 7.5% while reducing average token consumption by up to 35% compared to vanilla decoding. These findings demonstrate that the benefits of explicit chain-of-thought can be realized through dynamic latent trajectory control, internalizing the reasoning process to bypass the explicit generation while achieving superior fidelity. Our code is available at https://github.com/sznnzs/LLM-Latent-Action.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04326v1">From Assumptions to Actions: Turning LLM Reasoning into Uncertainty-Aware Planning for Embodied Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-04
      | 💬 31 pages, 10 figures, Accepted ICLR 2026
    </div>
    <details class="paper-abstract">
      Embodied agents operating in multi-agent, partially observable, and decentralized environments must plan and act despite pervasive uncertainty about hidden objects and collaborators' intentions. Recent advances in applying Large Language Models (LLMs) to embodied agents have addressed many long-standing challenges, such as high-level goal decomposition and online adaptation. Yet, uncertainty is still primarily mitigated through frequent inter-agent communication. This incurs substantial token and time costs, and can disrupt established workflows, when human partners are involved. We introduce PCE, a Planner-Composer-Evaluator framework that converts the fragmented assumptions latent in LLM reasoning traces into a structured decision tree. Internal nodes encode environment assumptions and leaves map to actions; each path is then scored by scenario likelihood, goal-directed gain, and execution cost to guide rational action selection without heavy communication. Across two challenging multi-agent benchmarks (C-WAH and TDW-MAT) and three diverse LLM backbones, PCE consistently outperforms communication-centric baselines in success rate and task efficiency while showing comparable token usage. Ablation results indicate that the performance gains obtained by scaling model capacity or reasoning depth persist even when PCE is applied, while PCE consistently raises the baseline across both capacity and reasoning-depth scales, confirming that structured uncertainty handling complements both forms of scaling. A user study further demonstrates that PCE produces communication patterns that human partners perceive as more efficient and trustworthy. Together, these results establish a principled route for turning latent LLM assumptions into reliable strategies for uncertainty-aware planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.01075v2">ConvexBench: Can LLMs Recognize Convex Functions?</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Convex analysis is a modern branch of mathematics with many applications. As Large Language Models (LLMs) start to automate research-level math and sciences, it is important for LLMs to demonstrate the ability to understand and reason with convexity. We introduce \cb, a scalable and mechanically verifiable benchmark for testing \textit{whether LLMs can identify the convexity of a symbolic objective under deep functional composition.} Experiments on frontier LLMs reveal a sharp compositional reasoning gap: performance degrades rapidly with increasing depth, dropping from an F1-score of $1.0$ at depth $2$ to approximately $0.2$ at depth $100$. Inspection of models' reasoning traces indicates two failure modes: \textit{parsing failure} and \textit{lazy reasoning}. To address these limitations, we propose an agentic divide-and-conquer framework that (i) offloads parsing to an external tool to construct an abstract syntax tree (AST) and (ii) enforces recursive reasoning over each intermediate sub-expression with focused context. This framework reliably mitigates deep-composition failures, achieving substantial performance improvement at large depths (e.g., F1-Score $= 1.0$ at depth $100$).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04296v1">ProxyWar: Dynamic Assessment of LLM Code Generation in Game Arenas</a></div>
    <div class="paper-meta">
      📅 2026-02-04
      | 💬 ICSE2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have revolutionized automated code generation, yet the evaluation of their real-world effectiveness remains limited by static benchmarks and simplistic metrics. We present ProxyWar, a novel framework that systematically assesses code generation quality by embedding LLM-generated agents within diverse, competitive game environments. Unlike existing approaches, ProxyWar evaluates not only functional correctness but also the operational characteristics of generated programs, combining automated testing, iterative code repair, and multi-agent tournaments to provide a holistic view of program behavior. Applied to a range of state-of-the-art coders and games, our approach uncovers notable discrepancies between benchmark scores and actual performance in dynamic settings, revealing overlooked limitations and opportunities for improvement. These findings highlight the need for richer, competition-based evaluation of code generation. Looking forward, ProxyWar lays a foundation for research into LLM-driven algorithm discovery, adaptive problem solving, and the study of practical efficiency and robustness, including the potential for models to outperform hand-crafted agents. The project is available at https://github.com/xinke-wang/ProxyWar.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04294v1">How Few-shot Demonstrations Affect Prompt-based Defenses Against LLM Jailbreak Attacks</a></div>
    <div class="paper-meta">
      📅 2026-02-04
      | 💬 13 pages, 4 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) face increasing threats from jailbreak attacks that bypass safety alignment. While prompt-based defenses such as Role-Oriented Prompts (RoP) and Task-Oriented Prompts (ToP) have shown effectiveness, the role of few-shot demonstrations in these defense strategies remains unclear. Prior work suggests that few-shot examples may compromise safety, but lacks investigation into how few-shot interacts with different system prompt strategies. In this paper, we conduct a comprehensive evaluation on multiple mainstream LLMs across four safety benchmarks (AdvBench, HarmBench, SG-Bench, XSTest) using six jailbreak attack methods. Our key finding reveals that few-shot demonstrations produce opposite effects on RoP and ToP: few-shot enhances RoP's safety rate by up to 4.5% through reinforcing role identity, while it degrades ToP's effectiveness by up to 21.2% through distracting attention from task instructions. Based on these findings, we provide practical recommendations for deploying prompt-based defenses in real-world LLM applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.18506v5">LLM-ABBA: Understanding time series via symbolic approximation</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      The success of large language models (LLMs) for time series has been demonstrated in previous work. Utilizing a symbolic time series representation, one can efficiently bridge the gap between LLMs and time series. However, the remaining challenge is to exploit the semantic information hidden in time series by using symbols or existing tokens of LLMs, while aligning the embedding space of LLMs according to the hidden information of time series. The symbolic time series approximation (STSA) method called adaptive Brownian bridge-based symbolic aggregation (ABBA) shows outstanding efficacy in preserving salient time series features by modeling time series patterns in terms of amplitude and period while using existing tokens of LLMs. In this paper, we introduce a method, called LLM-ABBA, that integrates ABBA into large language models for various downstream time series tasks. By symbolizing time series, LLM-ABBA compares favorably to the recent state-of-the-art (SOTA) in UCR and three medical time series classification tasks. Meanwhile, a fixed-polygonal chain trick in ABBA is introduced to avoid obvious drifting during forecasting tasks by significantly mitigating the effects of cumulative error arising from misused symbols during the transition from symbols to numerical values. In time series regression tasks, LLM-ABBA achieves the new SOTA on Time Series Extrinsic Regression (TSER) benchmarks. LLM-ABBA also shows competitive forecasting capability compared to recent SOTA time series forecasting results. We believe this framework can also seamlessly extend to other time series tasks. Our simulation code is publicly available at: https://github.com/inEXASCALE/llm-abba
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04288v1">Contextual Drag: How Errors in the Context Affect LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Central to many self-improvement pipelines for large language models (LLMs) is the assumption that models can improve by reflecting on past mistakes. We study a phenomenon termed contextual drag: the presence of failed attempts in the context biases subsequent generations toward structurally similar errors. Across evaluations of 11 proprietary and open-weight models on 8 reasoning tasks, contextual drag induces 10-20% performance drops, and iterative self-refinement in models with severe contextual drag can collapse into self-deterioration. Structural analysis using tree edit distance reveals that subsequent reasoning trajectories inherit structurally similar error patterns from the context. We demonstrate that neither external feedback nor successful self-verification suffices to eliminate this effect. While mitigation strategies such as fallback-behavior fine-tuning and context denoising yield partial improvements, they fail to fully restore baseline performance, positioning contextual drag as a persistent failure mode in current reasoning architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.01374v5">REASONING COMPILER: LLM-Guided Optimizations for Efficient Model Serving</a></div>
    <div class="paper-meta">
      📅 2026-02-04
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      While model serving has unlocked unprecedented capabilities, the high cost of serving large-scale models continues to be a significant barrier to widespread accessibility and rapid innovation. Compiler optimizations have long driven substantial performance improvements, but existing compilers struggle with neural workloads due to the exponentially large and highly interdependent space of possible transformations. Although existing stochastic search techniques can be effective, they are often sample-inefficient and fail to leverage the structural context underlying compilation decisions. We set out to investigate the research question of whether reasoning with large language models (LLMs), without any retraining, can leverage the context-aware decision space of compiler optimizations to significantly improve sample efficiency. To that end, we introduce a novel compilation framework (dubbed REASONING COMPILER) that formulates optimization as a sequential, context-aware decision process guided by a large language model and structured Monte Carlo tree search (MCTS). The LLM acts as a proposal mechanism, suggesting hardware-informed transformations that reflect the current program state and accumulated performance feedback. MCTS incorporates the LLM-generated proposals to balance exploration and exploitation, facilitating a structured, context-sensitive traversal of the expansive compiler optimization space. By achieving substantial speedups with markedly fewer samples than leading neural compilers, our approach demonstrates the potential of LLM-guided reasoning to transform the landscape of compiler optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04284v1">Agent-Omit: Training Efficient LLM Agents for Adaptive Thought and Observation Omission via Agentic Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-02-04
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Managing agent thought and observation during multi-turn agent-environment interactions is an emerging strategy to improve agent efficiency. However, existing studies treat the entire interaction trajectories equally, overlooking the thought necessity and observation utility varies across turns. To this end, we first conduct quantitative investigations into how thought and observation affect agent effectiveness and efficiency. Based on our findings, we propose Agent-Omit, a unified training framework that empowers LLM agents to adaptively omit redundant thoughts and observations. Specifically, we first synthesize a small amount of cold-start data, including both single-turn and multi-turn omission scenarios, to fine-tune the agent for omission behaviors. Furthermore, we introduce an omit-aware agentic reinforcement learning approach, incorporating a dual sampling mechanism and a tailored omission reward to incentivize the agent's adaptive omission capability. Theoretically, we prove that the deviation of our omission policy is upper-bounded by KL-divergence. Experimental results on five agent benchmarks show that our constructed Agent-Omit-8B could obtain performance comparable to seven frontier LLM agent, and achieve the best effectiveness-efficiency trade-off than seven efficient LLM agents methods. Our code and data are available at https://github.com/usail-hkust/Agent-Omit.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.17489v2">MapCoder-Lite: Distilling Multi-Agent Coding into a Single Small LLM</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have advanced code generation from single-function tasks to competitive-programming problems, but existing multi-agent solutions either rely on costly large-scale (>30B) models or collapse when downsized to small open-source models. We present MapCoder-Lite, a framework for distilling the complex reasoning of large, multi-agent coding systems into a single 7B model. Our contribution is a novel, three-pillar methodology that synergistically generates, refines, and encodes multi-agent knowledge: (i) pass-based trajectory distillation from strong LLMs fixes format fragility in retrieval and reduces failures in debugging, (ii) supervisor-guided correction with global feedback strengthens planning and coding agents, and (iii) agent-wise LoRA fine-tuning delivers memory-efficient specialisation. Comprehensive evaluation on xCodeEval, APPS, and CodeContests shows that MapCoder-Lite more than doubles xCodeEval accuracy (from 13.2% to 28.3%), eliminates all format failures, while reducing GPU memory and token-generation time by 4x compared to a 32B model. It also achieves over 10% gains on simpler coding benchmarks, demonstrating broad improvements beyond competitive programming. These results demonstrate that careful agent-wise fine-tuning unleashes high-quality multi-agent coding on a small language model. Our code is publicly available at https://github.com/aiha-lab/MapCoder-Lite.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04278v1">MiniRec: Data-Efficient Reinforcement Learning for LLM-based Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      The integration of reinforcement learning (RL) into large language models (LLMs) has opened new opportunities for recommender systems by eliciting reasoning and improving user preference modeling. However, RL-based LLM recommendation faces significant efficiency challenges, making full-data training costly. Existing data selection methods define sample value based on learnability or representativeness, yet their loss- or gradient-driven or dataset coverage-driven criteria often misalign with RL learning dynamics, resulting in suboptimal performance. To address this, we propose MiniRec, a data selection framework tailored for RL-based LLM recommendation. MiniRec evaluates sample learnability using key RL signals -- rewards -- pruning samples that are too easy (too high reward) or too difficult (consistently low reward). It assesses representativeness by aligning sample gradients with the approximated "ideal" global RL optimization trajectory, selecting samples that mainly drive model updates, and it also enforces diversity to reduce redundancy. Combined with a curriculum learning strategy from easy to hard samples, MiniRec significantly reduces training cost while largely preserving performance. Extensive experiments demonstrate MiniRec's effectiveness, highlighting the importance of reward-aligned, trajectory-informed data selection in RL-based LLM recommendation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21164v2">Concise Geometric Description as a Bridge: Unleashing the Potential of LLM for Plane Geometry Problem Solving</a></div>
    <div class="paper-meta">
      📅 2026-02-04
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Plane Geometry Problem Solving (PGPS) is a multimodal reasoning task that aims to solve a plane geometric problem based on a geometric diagram and problem textual descriptions. Although Large Language Models (LLMs) possess strong reasoning skills, their direct application to PGPS is hindered by their inability to process visual diagrams. Existing works typically fine-tune Multimodal LLMs (MLLMs) end-to-end on large-scale PGPS data to enhance visual understanding and reasoning simultaneously. However, such joint optimization may compromise base LLMs' inherent reasoning capability. In this work, we observe that LLM itself is potentially a powerful PGPS solver when appropriately formulating visual information as textual descriptions. We propose to train a MLLM Interpreter to generate geometric descriptions for the visual diagram, and an off-the-shelf LLM is utilized to perform reasoning. Specifically, we choose Conditional Declaration Language (CDL) as the geometric description as its conciseness eases the MLLM Interpreter training. The MLLM Interpreter is fine-tuned via CoT (Chain-of-Thought)-augmented SFT followed by GRPO to generate CDL. Instead of using a conventional solution-based reward that compares the reasoning result with the ground-truth answer, we design CDL matching rewards to facilitate more effective GRPO training, which provides more direct and denser guidance for CDL generation. To support training, we construct a new dataset, Formalgeo7k-Rec-CoT, by manually reviewing Formalgeo7k v2 and incorporating CoT annotations. Extensive experiments on Formalgeo7k-Rec-CoT, Unigeo, and MathVista show our method (finetuned on only 5.5k data) performs favorably against leading open-source and closed-source MLLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04265v1">Thickening-to-Thinning: Reward Shaping via Human-Inspired Learning Dynamics for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a promising paradigm for enhancing reasoning in Large Language Models (LLMs). However, it frequently encounters challenges such as entropy collapse, excessive verbosity, and insufficient exploration for hard problems. Crucially, existing reward schemes fail to distinguish between the need for extensive search during problem-solving and the efficiency required for mastered knowledge. In this work, we introduce T2T(Thickening-to-Thinning), a dynamic reward framework inspired by human learning processes. Specifically, it implements a dual-phase mechanism: (1) On incorrect attempts, T2T incentivizes "thickening" (longer trajectories) to broaden the search space and explore novel solution paths; (2) Upon achieving correctness, it shifts to "thinning", imposing length penalties to discourage redundancy, thereby fostering model confidence and crystallizing reasoning capabilities. Extensive experiments on mathematical benchmarks (MATH-500, AIME, AMC) across Qwen-series and Deepseek models demonstrate that T2T significantly outperforms standard GRPO and recent baselines, achieving superior performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04918v1">Simulated Adoption: Decoupling Magnitude and Direction in LLM In-Context Conflict Resolution</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) frequently prioritize conflicting in-context information over pre-existing parametric memory, a phenomenon often termed sycophancy or compliance. However, the mechanistic realization of this behavior remains obscure, specifically how the model resolves these knowledge conflicts through compliance, and whether this suppression arises from signal magnitude dilution or directional geometric alteration within the residual stream. To resolve this, we conducted a layer-wise geometric analysis across Qwen-4B, Llama-3.1-8B, and GLM-4-9B, decomposing the residual stream updates induced by counter-factual contexts into radial (norm-based) and angular (cosine-based) components. Our empirical results reject the universality of the "Manifold Dilution" hypothesis, as two of the three architectures maintained stable residual norms despite exhibiting significant performance degradation on factual queries. Instead, we observed that compliance is consistently characterized by "Orthogonal Interference," where the conflicting context injects a steering vector that is quasi-orthogonal to the ground-truth direction, effectively rotating the hidden state representation. This suggests that models do not "unlearn" or suppress the magnitude of internal truths but rather employ a mechanism of geometric displacement to bypass the correct unembedding vector, effectively simulating adoption while preserving the original structural magnitude. These findings challenge scalar confidence metrics for detecting hallucinations and underscore the necessity of vectorial monitoring to distinguish between genuine knowledge integration and superficial in-context mimicry.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05564v2">The ICASSP 2026 HumDial Challenge: Benchmarking Human-like Spoken Dialogue Systems in the LLM Era</a></div>
    <div class="paper-meta">
      📅 2026-02-04
      | 💬 Official summary paper for the ICASSP 2026 HumDial Challenge
    </div>
    <details class="paper-abstract">
      Driven by the rapid advancement of Large Language Models (LLMs), particularly Audio-LLMs and Omni-models, spoken dialogue systems have evolved significantly, progressively narrowing the gap between human-machine and human-human interactions. Achieving truly ``human-like'' communication necessitates a dual capability: emotional intelligence to perceive and resonate with users' emotional states, and robust interaction mechanisms to navigate the dynamic, natural flow of conversation, such as real-time turn-taking. Therefore, we launched the first Human-like Spoken Dialogue Systems Challenge (HumDial) at ICASSP 2026 to benchmark these dual capabilities. Anchored by a sizable dataset derived from authentic human conversations, this initiative establishes a fair evaluation platform across two tracks: (1) Emotional Intelligence, targeting long-term emotion understanding and empathetic generation; and (2) Full-Duplex Interaction, systematically evaluating real-time decision-making under `` listening-while-speaking'' conditions. This paper summarizes the dataset, track configurations, and the final results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2409.11022v6">DynamicNER: A Dynamic, Multilingual, and Fine-Grained Dataset for LLM-based Named Entity Recognition</a></div>
    <div class="paper-meta">
      📅 2026-02-04
      | 💬 This paper is accepted by EMNLP 2025 Main Conference
    </div>
    <details class="paper-abstract">
      The advancements of Large Language Models (LLMs) have spurred a growing interest in their application to Named Entity Recognition (NER) methods. However, existing datasets are primarily designed for traditional machine learning methods and are inadequate for LLM-based methods, in terms of corpus selection and overall dataset design logic. Moreover, the prevalent fixed and relatively coarse-grained entity categorization in existing datasets fails to adequately assess the superior generalization and contextual understanding capabilities of LLM-based methods, thereby hindering a comprehensive demonstration of their broad application prospects. To address these limitations, we propose DynamicNER, the first NER dataset designed for LLM-based methods with dynamic categorization, introducing various entity types and entity type lists for the same entity in different context, leveraging the generalization of LLM-based NER better. The dataset is also multilingual and multi-granular, covering 8 languages and 155 entity types, with corpora spanning a diverse range of domains. Furthermore, we introduce CascadeNER, a novel NER method based on a two-stage strategy and lightweight LLMs, achieving higher accuracy on fine-grained tasks while requiring fewer computational resources. Experiments show that DynamicNER serves as a robust and effective benchmark for LLM-based NER methods. Furthermore, we also conduct analysis for traditional methods and LLM-based methods on our dataset. Our code and dataset are openly available at https://github.com/Astarojth/DynamicNER.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.04428v2">Building Scaffolding Dialogue Data with LLM-Simulated Novices</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      High-quality, multi-turn instructional dialogues between novices and experts are essential for developing AI systems that support teaching, learning, and decision-making. These dialogues often involve scaffolding -- the process by which an expert supports a novice's thinking through questions, feedback, and step-by-step guidance. However, such data are scarce due to privacy concerns in recording and the vulnerability inherent in help-seeking. We present SimInstruct, a scalable, expert-in-the-loop tool for collecting scaffolding dialogues. Using teaching development coaching as an example domain, SimInstruct simulates novice instructors via LLMs, varying their teaching challenges and LLM's persona traits, while human experts provide multi-turn feedback, reasoning, and instructional support. This design enables the creation of realistic, pedagogically rich dialogues without requiring real novice participants. Our results reveal that persona traits, such as extroversion and introversion, meaningfully influence how experts engage. Compared to real mentoring recordings, SimInstruct dialogues demonstrate comparable pedagogical relevance and cognitive depth. Experts also reported the process as engaging and reflective, improving both data quality and their own professional insight. We further fine-tuned a LLaMA model to be an expert model using the augmented dataset, which outperformed GPT-4o in instructional quality. Our analysis highlights GPT-4o's limitations in weak reflective questioning, overuse of generic praise, a condescending tone, and a tendency to overwhelm novices with excessive suggestions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04225v1">Following the TRAIL: Predicting and Explaining Tomorrow's Hits with a Fine-Tuned LLM</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have been widely applied across multiple domains for their broad knowledge and strong reasoning capabilities. However, applying them to recommendation systems is challenging since it is hard for LLMs to extract user preferences from large, sparse user-item logs, and real-time per-user ranking over the full catalog is too time-consuming to be practical. Moreover, many existing recommender systems focus solely on ranking items while overlooking explanations, which could help improve predictive accuracy and make recommendations more convincing to users. Inspired by recent works that achieve strong recommendation performance by forecasting near-term item popularity, we propose TRAIL (TRend and explAnation Integrated Learner). TRAIL is a fine-tuned LLM that jointly predicts short-term item popularity and generates faithful natural-language explanations. It employs contrastive learning with positive and negative pairs to align its scores and explanations with structured trend signals, yielding accurate and explainable popularity predictions. Extensive experiments show that TRAIL outperforms strong baselines and produces coherent, well-grounded explanations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.23782v3">Bridging the Knowledge-Prediction Gap in LLMs on Multiple-Choice Questions</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) perform strongly on diverse tasks, their trustworthiness is limited by erratic behavior that is unfaithful to their internal knowledge. In particular, LLMs often fail on multiple-choice questions (MCQs) even if they encode correct answers in their hidden representations, revealing a misalignment between internal knowledge and output behavior. We investigate and mitigate this knowledge-prediction gap on MCQs through a three-step analysis of hidden representations. First, we quantify the prevalence and magnitude of the gap across models and datasets. Second, we provide a geometric interpretation by identifying distinct knowledge and prediction subspaces in the residual stream. Third, we introduce KAPPA, a lightweight inference-time intervention that aligns the two subspaces within the residual stream to reduce the knowledge-prediction gap. Our results provide a geometric and interpretable explanation of the knowledge-prediction gap in LLMs. Furthermore, KAPPA effectively reduces the gap across diverse MCQ benchmarks and models, and generalizes to free-form settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04916v1">AFD-INSTRUCTION: A Comprehensive Antibody Instruction Dataset with Functional Annotations for LLM-Based Understanding and Design</a></div>
    <div class="paper-meta">
      📅 2026-02-04
      | 💬 Accepted by ICLR 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have significantly advanced protein representation learning. However, their capacity to interpret and design antibodies through natural language remains limited. To address this challenge, we present AFD-Instruction, the first large-scale instruction dataset with functional annotations tailored to antibodies. This dataset encompasses two key components: antibody understanding, which infers functional attributes directly from sequences, and antibody design, which enables de novo sequence generation under functional constraints. These components provide explicit sequence-function alignment and support antibody design guided by natural language instructions. Extensive instruction-tuning experiments on general-purpose LLMs demonstrate that AFD-Instruction consistently improves performance across diverse antibody-related tasks. By linking antibody sequences with textual descriptions of function, AFD-Instruction establishes a new foundation for advancing antibody modeling and accelerating therapeutic discovery.
    </details>
</div>
