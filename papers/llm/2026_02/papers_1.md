# llm - 2026_02

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1
- [Part 2](papers_2.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.06038v1">CommCP: Efficient Multi-Agent Coordination via LLM-Based Communication with Conformal Prediction</a></div>
    <div class="paper-meta">
      📅 2026-02-05
      | 💬 IEEE International Conference on Robotics and Automation (ICRA 2026); Project Website: https://comm-cp.github.io/
    </div>
    <details class="paper-abstract">
      To complete assignments provided by humans in natural language, robots must interpret commands, generate and answer relevant questions for scene understanding, and manipulate target objects. Real-world deployments often require multiple heterogeneous robots with different manipulation capabilities to handle different assignments cooperatively. Beyond the need for specialized manipulation skills, effective information gathering is important in completing these assignments. To address this component of the problem, we formalize the information-gathering process in a fully cooperative setting as an underexplored multi-agent multi-task Embodied Question Answering (MM-EQA) problem, which is a novel extension of canonical Embodied Question Answering (EQA), where effective communication is crucial for coordinating efforts without redundancy. To address this problem, we propose CommCP, a novel LLM-based decentralized communication framework designed for MM-EQA. Our framework employs conformal prediction to calibrate the generated messages, thereby minimizing receiver distractions and enhancing communication reliability. To evaluate our framework, we introduce an MM-EQA benchmark featuring diverse, photo-realistic household scenarios with embodied questions. Experimental results demonstrate that CommCP significantly enhances the task success rate and exploration efficiency over baselines. The experiment videos, code, and dataset are available on our project website: https://comm-cp.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.06008v1">AgenticPay: A Multi-Agent LLM Negotiation System for Buyer-Seller Transactions</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based agents are increasingly expected to negotiate, coordinate, and transact autonomously, yet existing benchmarks lack principled settings for evaluating language-mediated economic interaction among multiple agents. We introduce AgenticPay, a benchmark and simulation framework for multi-agent buyer-seller negotiation driven by natural language. AgenticPay models markets in which buyers and sellers possess private constraints and product-dependent valuations, and must reach agreements through multi-round linguistic negotiation rather than numeric bidding alone. The framework supports a diverse suite of over 110 tasks ranging from bilateral bargaining to many-to-many markets, with structured action extraction and metrics for feasibility, efficiency, and welfare. Benchmarking state-of-the-art proprietary and open-weight LLMs reveals substantial gaps in negotiation performance and highlights challenges in long-horizon strategic reasoning, establishing AgenticPay as a foundation for studying agentic commerce and language-based market interaction. Code and dataset are available at the link: https://github.com/SafeRL-Lab/AgenticPay.
    </details>
</div>
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
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05128v1">Reporting and Reviewing LLM-Integrated Systems in HCI: Challenges and Considerations</a></div>
    <div class="paper-meta">
      📅 2026-02-04
      | 💬 18 pages, 1 figure, 2 tables. For proposed reporting guidelines, see https://ianarawjo.github.io/Guidelines-for-Reporting-LLM-Integrated-Systems-in-HCI/
    </div>
    <details class="paper-abstract">
      What should HCI scholars consider when reporting and reviewing papers that involve LLM-integrated systems? We interview 18 authors of LLM-integrated system papers on their authoring and reviewing experiences. We find that norms of trust-building between authors and reviewers appear to be eroded by the uncertainty of LLM behavior and hyperbolic rhetoric surrounding AI. Authors perceive that reviewers apply uniquely skeptical and inconsistent standards towards papers that report LLM-integrated systems, and mitigate mistrust by adding technical evaluations, justifying usage, and de-emphasizing LLM presence. Authors' views challenge blanket directives to report all prompts and use open models, arguing that prompt reporting is context-dependent and justifying proprietary model usage despite ethical concerns. Finally, some tensions in peer review appear to stem from clashes between the norms and values of HCI and ML/NLP communities, particularly around what constitutes a contribution and an appropriate level of technical rigor. Based on our findings and additional feedback from six expert HCI researchers, we present a set of guidelines and considerations for authors, reviewers, and HCI communities around reporting and reviewing papers that involve LLM-integrated systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05125v1">Rethinking Rubric Generation for Improving LLM Judge and Reward Modeling for Open-ended Tasks</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Recently, rubrics have been used to guide LLM judges in capturing subjective, nuanced, multi-dimensional human preferences, and have been extended from evaluation to reward signals for reinforcement fine-tuning (RFT). However, rubric generation remains hard to control: rubrics often lack coverage, conflate dimensions, misalign preference direction, and contain redundant or highly correlated criteria, degrading judge accuracy and producing suboptimal rewards during RFT. We propose RRD, a principled framework for rubric refinement built on a recursive decompose-filter cycle. RRD decomposes coarse rubrics into fine-grained, discriminative criteria, expanding coverage while sharpening separation between responses. A complementary filtering mechanism removes misaligned and redundant rubrics, and a correlation-aware weighting scheme prevents over-representing highly correlated criteria, yielding rubric sets that are informative, comprehensive, and non-redundant. Empirically, RRD delivers large, consistent gains across both evaluation and training: it improves preference-judgment accuracy on JudgeBench and PPE for both GPT-4o and Llama3.1-405B judges, achieving top performance in all settings with up to +17.7 points on JudgeBench. When used as the reward source for RFT on WildChat, it yields substantially stronger and more stable learning signals, boosting reward by up to 160% (Qwen3-4B) and 60% (Llama3.1-8B) versus 10-20% for prior rubric baselines, with gains that transfer to HealthBench-Hard and BiGGen Bench. Overall, RRD establishes recursive rubric refinement as a scalable and interpretable foundation for LLM judging and reward modeling in open-ended domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05110v1">Understanding LLM Evaluator Behavior: A Structured Multi-Evaluator Framework for Merchant Risk Assessment</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used as evaluators of reasoning quality, yet their reliability and bias in payments-risk settings remain poorly understood. We introduce a structured multi-evaluator framework for assessing LLM reasoning in Merchant Category Code (MCC)-based merchant risk assessment, combining a five-criterion rubric with Monte-Carlo scoring to evaluate rationale quality and evaluator stability. Five frontier LLMs generate and cross-evaluate MCC risk rationales under attributed and anonymized conditions. To establish a judge-independent reference, we introduce a consensus-deviation metric that eliminates circularity by comparing each judge's score to the mean of all other judges, yielding a theoretically grounded measure of self-evaluation and cross-model deviation. Results reveal substantial heterogeneity: GPT-5.1 and Claude 4.5 Sonnet show negative self-evaluation bias (-0.33, -0.31), while Gemini-2.5 Pro and Grok 4 display positive bias (+0.77, +0.71), with bias attenuating by 25.8 percent under anonymization. Evaluation by 26 payment-industry experts shows LLM judges assign scores averaging +0.46 points above human consensus, and that the negative bias of GPT-5.1 and Claude 4.5 Sonnet reflects closer alignment with human judgment. Ground-truth validation using payment-network data shows four models exhibit statistically significant alignment (Spearman rho = 0.56 to 0.77), confirming that the framework captures genuine quality. Overall, the framework provides a replicable basis for evaluating LLM-as-a-judge systems in payment-risk workflows and highlights the need for bias-aware protocols in operational financial settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21410v2">Statsformer: Validated Ensemble Learning with LLM-Derived Semantic Priors</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      We introduce Statsformer, a principled framework for integrating large language model (LLM)-derived knowledge into supervised statistical learning. Existing approaches are limited in adaptability and scope: they either inject LLM guidance as an unvalidated heuristic, which is sensitive to LLM hallucination, or embed semantic information within a single fixed learner. Statsformer overcomes both limitations through a guardrailed ensemble architecture. We embed LLM-derived feature priors within an ensemble of linear and nonlinear learners, adaptively calibrating their influence via cross-validation. This design yields a flexible system with an oracle-style guarantee that it performs no worse than any convex combination of its in-library base learners, up to statistical error. Empirically, informative priors yield consistent performance improvements, while uninformative or misspecified LLM guidance is automatically downweighted, mitigating the impact of hallucinations across a diverse range of prediction tasks.An open-source implementation of Statsformer is available at https://github.com/pilancilab/statsformer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.26641v3">All You Need for Object Detection: From Pixels, Points, and Prompts to Next-Gen Fusion and Multimodal LLMs/VLMs in Autonomous Vehicles</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Autonomous Vehicles (AVs) are transforming the future of transportation through advances in intelligent perception, decision-making, and control systems. However, their success is tied to one core capability, reliable object detection in complex and multimodal environments. While recent breakthroughs in Computer Vision (CV) and Artificial Intelligence (AI) have driven remarkable progress, the field still faces a critical challenge as knowledge remains fragmented across multimodal perception, contextual reasoning, and cooperative intelligence. This survey bridges that gap by delivering a forward-looking analysis of object detection in AVs, emphasizing emerging paradigms such as Vision-Language Models (VLMs), Large Language Models (LLMs), and Generative AI rather than re-examining outdated techniques. We begin by systematically reviewing the fundamental spectrum of AV sensors (camera, ultrasonic, LiDAR, and Radar) and their fusion strategies, highlighting not only their capabilities and limitations in dynamic driving environments but also their potential to integrate with recent advances in LLM/VLM-driven perception frameworks. Next, we introduce a structured categorization of AV datasets that moves beyond simple collections, positioning ego-vehicle, infrastructure-based, and cooperative datasets (e.g., V2V, V2I, V2X, I2I), followed by a cross-analysis of data structures and characteristics. Ultimately, we analyze cutting-edge detection methodologies, ranging from 2D and 3D pipelines to hybrid sensor fusion, with particular attention to emerging transformer-driven approaches powered by Vision Transformers (ViTs), Large and Small Language Models (SLMs), and VLMs. By synthesizing these perspectives, our survey delivers a clear roadmap of current capabilities, open challenges, and future opportunities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.17406v2">Robust Answers, Fragile Logic: Probing the Decoupling Hypothesis in LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      While Chain-of-Thought (CoT) prompting has become a cornerstone for complex reasoning in Large Language Models (LLMs), the faithfulness of the generated reasoning remains an open question. We investigate the Decoupling Hypothesis: that correct answers often mask fragile, post-hoc rationalizations that are not causally tied to the model's prediction. To systematically verify this, we introduce MATCHA, a novel Answer-Conditioned Probing framework. Unlike standard evaluations that focus on final output accuracy, MATCHA isolates the reasoning phase by conditioning generation on the model's predicted answer, allowing us to stress-test the stability of the rationale itself. Our experiments reveal a critical vulnerability: under imperceptible input perturbations, LLMs frequently maintain the correct answer while generating inconsistent or nonsensical reasoning - effectively being ``Right for the Wrong Reasons''. Using LLM judges to quantify this robustness gap, we find that multi-step and commonsense tasks are significantly more susceptible to this decoupling than logical tasks. Furthermore, we demonstrate that adversarial examples generated by MATCHA transfer non-trivially to black-box models. Our findings expose the illusion of CoT robustness and underscore the need for future architectures that enforce genuine answer-reasoning consistency rather than mere surface-level accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15397v2">Beyond Prompting: Efficient and Robust Contextual Biasing for Speech LLMs via Logit-Space Integration (LOGIC)</a></div>
    <div class="paper-meta">
      📅 2026-02-04
      | 💬 This paper is withdrawn temporarily to ensure full compliance with internal institutional publication approval processes
    </div>
    <details class="paper-abstract">
      The rapid emergence of new entities -- driven by cultural shifts, evolving trends, and personalized user data -- poses a significant challenge for existing Speech Large Language Models (Speech LLMs). While these models excel at general conversational tasks, their static training knowledge limits their ability to recognize domain-specific terms such as contact names, playlists, or technical jargon. Existing solutions primarily rely on prompting, which suffers from poor scalability: as the entity list grows, prompting encounters context window limitations, increased inference latency, and the "lost-in-the-middle" phenomenon. An alternative approach, Generative Error Correction (GEC), attempts to rewrite transcripts via post-processing but frequently suffers from "over-correction", introducing hallucinations of entities that were never spoken. In this work, we introduce LOGIC (Logit-Space Integration for Contextual Biasing), an efficient and robust framework that operates directly in the decoding layer. Unlike prompting, LOGIC decouples context injection from input processing, ensuring constant-time complexity relative to prompt length. Extensive experiments using the Phi-4-MM model across 11 multilingual locales demonstrate that LOGIC achieves an average 9% relative reduction in Entity WER with a negligible 0.30% increase in False Alarm Rate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04998v1">Learning Rate Matters: Vanilla LoRA May Suffice for LLM Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Low-Rank Adaptation (LoRA) is the prevailing approach for efficient large language model (LLM) fine-tuning. Building on this paradigm, recent studies have proposed alternative initialization strategies and architectural modifications, reporting substantial improvements over vanilla LoRA. However, these gains are often demonstrated under fixed or narrowly tuned hyperparameter settings, despite the known sensitivity of neural networks to training configurations. In this work, we systematically re-evaluate four representative LoRA variants alongside vanilla LoRA through extensive hyperparameter searches. Across mathematical and code generation tasks on diverse model scales, we find that different LoRA methods favor distinct learning rate ranges. Crucially, once learning rates are properly tuned, all methods achieve similar peak performance (within 1-2%), with only subtle rank-dependent behaviors. These results suggest that vanilla LoRA remains a competitive baseline and that improvements reported under single training configuration may not reflect consistent methodological advantages. Finally, a second-order analysis attributes the differing optimal learning rate ranges to variations in the largest Hessian eigenvalue, aligning with classical learning theories.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04972v1">Learning Context Matters: Measuring and Diagnosing Personalization Gaps in LLM-Based Instructional Design</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      The adoption of generative AI in education has accelerated dramatically in recent years, with Large Language Models (LLMs) increasingly integrated into learning environments in the hope of providing personalized support that enhances learner engagement and knowledge retention. However, truly personalized support requires access to meaningful Learning Context (LC) regarding who the learner is, what they are trying to understand, and how they are engaging with the material. In this paper, we present a framework for measuring and diagnosing how the LC influences instructional strategy selection in LLM-based tutoring systems. Using psychometrically grounded synthetic learning contexts and a pedagogically grounded decision space, we compare LLM instructional decisions in context-blind and context-aware conditions and quantify their alignment with the pedagogical judgments of subject matter experts. Our results show that, while providing the LC induces systematic, measurable changes in instructional decisions that move LLM policies closer to the subject matter expert policy, substantial misalignment remains. To diagnose this misalignment, we introduce a relevance-impact analysis that reveals which learner characteristics are attended to, ignored, or spuriously influential in LLM instructional decision-making. This analysis, conducted in collaboration with subject matter experts, demonstrates that LC materially shapes LLM instructional planning but does not reliably induce pedagogically appropriate personalization. Our results enable principled evaluation of context-aware LLM systems and provide a foundation for improving personalization through learner characteristic prioritization, pedagogical model tuning, and LC engineering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04879v1">Rethinking the Trust Region in LLM Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has become a cornerstone for fine-tuning Large Language Models (LLMs), with Proximal Policy Optimization (PPO) serving as the de facto standard algorithm. Despite its ubiquity, we argue that the core ratio clipping mechanism in PPO is structurally ill-suited for the large vocabularies inherent to LLMs. PPO constrains policy updates based on the probability ratio of sampled tokens, which serves as a noisy single-sample Monte Carlo estimate of the true policy divergence. This creates a sub-optimal learning dynamic: updates to low-probability tokens are aggressively over-penalized, while potentially catastrophic shifts in high-probability tokens are under-constrained, leading to training inefficiency and instability. To address this, we propose Divergence Proximal Policy Optimization (DPPO), which substitutes heuristic clipping with a more principled constraint based on a direct estimate of policy divergence (e.g., Total Variation or KL). To avoid huge memory footprint, we introduce the efficient Binary and Top-K approximations to capture the essential divergence with negligible overhead. Extensive empirical evaluations demonstrate that DPPO achieves superior training stability and efficiency compared to existing methods, offering a more robust foundation for RL-based LLM fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.02542v4">OverThink: Slowdown Attacks on Reasoning LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Most flagship language models generate explicit reasoning chains, enabling inference-time scaling. However, producing these reasoning chains increases token usage (i.e., reasoning tokens), which in turn increases latency and costs. Our OverThink attack increases overhead for applications that rely on reasoning language models (RLMs) and external context by forcing them to spend substantially more reasoning tokens while still producing contextually correct answers. An adversary mounts an attack by injecting decoy reasoning problems into public content that is consumed by RLM at inference time. Because our decoys (e.g., Markov decision processes, Sudokus, etc.) are benign, they evade safety filters. We evaluate OverThink on both closed-source and open-source reasoning models across the FreshQA, SQuAD, and MuSR datasets. We also explore the attack in multi-modal settings by creating images that cause excessive reasoning. We show that the resulting slowdown transfers across models. Finally, we explore both LLM-based and systems-level defenses, and discuss the societal, financial, and energy implications of the OverThink attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04813v1">Agentic AI in Healthcare & Medicine: A Seven-Dimensional Taxonomy for Empirical Evaluation of LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based agents that plan, use tools and act has begun to shape healthcare and medicine. Reported studies demonstrate competence on various tasks ranging from EHR analysis and differential diagnosis to treatment planning and research workflows. Yet the literature largely consists of overviews which are either broad surveys or narrow dives into a single capability (e.g., memory, planning, reasoning), leaving healthcare work without a common frame. We address this by reviewing 49 studies using a seven-dimensional taxonomy: Cognitive Capabilities, Knowledge Management, Interaction Patterns, Adaptation & Learning, Safety & Ethics, Framework Typology and Core Tasks & Subtasks with 29 operational sub-dimensions. Using explicit inclusion and exclusion criteria and a labeling rubric (Fully Implemented, Partially Implemented, Not Implemented), we map each study to the taxonomy and report quantitative summaries of capability prevalence and co-occurrence patterns. Our empirical analysis surfaces clear asymmetries. For instance, the External Knowledge Integration sub-dimension under Knowledge Management is commonly realized (~76% Fully Implemented) whereas Event-Triggered Activation sub-dimenison under Interaction Patterns is largely absent (~92% Not Implemented) and Drift Detection & Mitigation sub-dimension under Adaptation & Learning is rare (~98% Not Implemented). Architecturally, Multi-Agent Design sub-dimension under Framework Typology is the dominant pattern (~82% Fully Implemented) while orchestration layers remain mostly partial. Across Core Tasks & Subtasks, information centric capabilities lead e.g., Medical Question Answering & Decision Support and Benchmarking & Simulation, while action and discovery oriented areas such as Treatment Planning & Prescription still show substantial gaps (~59% Not Implemented).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10823v2">Mugi: Value Level Parallelism For Efficient LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-04
      | 💬 2026 International Conference on Architectural Support for Programming Languages and Operating Systems
    </div>
    <details class="paper-abstract">
      Value level parallelism (VLP) has been proposed to improve the efficiency of large-batch, low-precision general matrix multiply (GEMM) between symmetric activations and weights. In transformer based large language models (LLMs), there exist more sophisticated operations beyond activation-weight GEMM. In this paper, we explore how VLP benefits LLMs. First, we generalize VLP for nonlinear approximations, outperforming existing nonlinear approximations in end-to-end LLM accuracy, performance, and efficiency. Our VLP approximation follows a value-centric approach, where important values are assigned with greater accuracy. Second, we optimize VLP for small-batch GEMMs with asymmetric inputs efficiently, which leverages timely LLM optimizations, including weight-only quantization, key-value (KV) cache quantization, and group query attention. Finally, we design a new VLP architecture, Mugi, to encapsulate the innovations above and support full LLM workloads, while providing better performance, efficiency and sustainability. Our experimental results show that Mugi can offer significant improvements on throughput and energy efficiency, up to $45\times$ and $668\times$ for nonlinear softmax operations, and $2.07\times$ and $3.11\times$ for LLMs, and also decrease operational carbon for LLM operation by $1.45\times$ and embodied carbon by $1.48\times$.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04785v1">Team, Then Trim: An Assembly-Line LLM Framework for High-Quality Tabular Data Generation</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      While tabular data is fundamental to many real-world machine learning (ML) applications, acquiring high-quality tabular data is usually labor-intensive and expensive. Limited by the scarcity of observations, tabular datasets often exhibit critical deficiencies, such as class imbalance, selection bias, and low fidelity. To address these challenges, building on recent advances in Large Language Models (LLMs), this paper introduces Team-then-Trim (T$^2$), a framework that synthesizes high-quality tabular data through a collaborative team of LLMs, followed by a rigorous three-stage plug-in data quality control (QC) pipeline. In T$^2$, tabular data generation is conceptualized as a manufacturing process: specialized LLMs, guided by domain knowledge, are tasked with generating different data components sequentially, and the resulting products, i.e., the synthetic data, are systematically evaluated across multiple dimensions of QC. Empirical results on both simulated and real-world datasets demonstrate that T$^2$ outperforms state-of-the-art methods in producing high-quality tabular data, highlighting its potential to support downstream models when direct data collection is practically infeasible.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.02648v4">Steering the Herd: A Framework for LLM-based Control of Social Learning</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Algorithms increasingly serve as information mediators--from social media feeds and targeted advertising to the increasing ubiquity of LLMs. This engenders a joint process where agents combine private, algorithmically-mediated signals with learning from peers to arrive at decisions. To study such settings, we introduce a model of controlled sequential social learning in which an information-mediating planner (e.g. an LLM) controls the information structure of agents while they also learn from the decisions of earlier agents. The planner may seek to improve social welfare (altruistic planner) or to induce a specific action the planner prefers (biased planner). Our framework presents a new optimization problem for social learning that combines dynamic programming with decentralized action choices and Bayesian belief updates. We prove the convexity of the value function and characterize the optimal policies of altruistic and biased planners, which attain desired tradeoffs between the costs they incur and the payoffs they earn from induced agent choices. Notably, in some regimes the biased planner intentionally obfuscates the agents' signals. Even under stringent transparency constraints--information parity with individuals, no lying or cherry-picking, and full observability--we show that information mediation can substantially shift social welfare in either direction. We complement our theory with simulations in which LLMs act as both planner and agents. Notably, the LLM planner in our simulations exhibits emergent strategic behavior in steering public opinion that broadly mirrors the trends predicted, though key deviations suggest the influence of non-Bayesian reasoning consistent with the cognitive patterns of both humans and LLMs trained on human-like data. Together, we establish our framework as a tractable basis for studying the impact and regulation of LLM information mediators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04755v1">When Silence Is Golden: Can LLMs Learn to Abstain in Temporal QA and Beyond?</a></div>
    <div class="paper-meta">
      📅 2026-02-04
      | 💬 Accepted to ICLR2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) rarely admit uncertainty, often producing fluent but misleading answers, rather than abstaining (i.e., refusing to answer). This weakness is even evident in temporal question answering, where models frequently ignore time-sensitive evidence and conflate facts across different time-periods. In this paper, we present the first empirical study of training LLMs with an abstention ability while reasoning about temporal QA. Existing approaches such as calibration might be unreliable in capturing uncertainty in complex reasoning. We instead frame abstention as a teachable skill and introduce a pipeline that couples Chain-of-Thought (CoT) supervision with Reinforcement Learning (RL) guided by abstention-aware rewards. Our goal is to systematically analyze how different information types and training techniques affect temporal reasoning with abstention behavior in LLMs. Through extensive experiments studying various methods, we find that RL yields strong empirical gains on reasoning: a model initialized by Qwen2.5-1.5B-Instruct surpasses GPT-4o by $3.46\%$ and $5.80\%$ in Exact Match on TimeQA-Easy and Hard, respectively. Moreover, it improves the True Positive rate on unanswerable questions by $20\%$ over a pure supervised fine-tuned (SFT) variant. Beyond performance, our analysis shows that SFT induces overconfidence and harms reliability, while RL improves prediction accuracy but exhibits similar risks. Finally, by comparing implicit reasoning cues (e.g., original context, temporal sub-context, knowledge graphs) with explicit CoT supervision, we find that implicit information provides limited benefit for reasoning with abstention. Our study provides new insights into how abstention and reasoning can be jointly optimized, providing a foundation for building more reliable LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04750v1">Exploiting contextual information to improve stance detection in informal political discourse with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-04
      | 💬 14 pages, 7 figures
    </div>
    <details class="paper-abstract">
      This study investigates the use of Large Language Models (LLMs) for political stance detection in informal online discourse, where language is often sarcastic, ambiguous, and context-dependent. We explore whether providing contextual information, specifically user profile summaries derived from historical posts, can improve classification accuracy. Using a real-world political forum dataset, we generate structured profiles that summarize users' ideological leaning, recurring topics, and linguistic patterns. We evaluate seven state-of-the-art LLMs across baseline and context-enriched setups through a comprehensive cross-model evaluation. Our findings show that contextual prompts significantly boost accuracy, with improvements ranging from +17.5\% to +38.5\%, achieving up to 74\% accuracy that surpasses previous approaches. We also analyze how profile size and post selection strategies affect performance, showing that strategically chosen political content yields better results than larger, randomly selected contexts. These findings underscore the value of incorporating user-level context to enhance LLM performance in nuanced political classification tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04731v1">Less Finetuning, Better Retrieval: Rethinking LLM Adaptation for Biomedical Retrievers via Synthetic Data and Model Merging</a></div>
    <div class="paper-meta">
      📅 2026-02-04
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Retrieval-augmented generation (RAG) has become the backbone of grounding Large Language Models (LLMs), improving knowledge updates and reducing hallucinations. Recently, LLM-based retriever models have shown state-of-the-art performance for RAG applications. However, several technical aspects remain underexplored on how to adapt general-purpose LLMs into effective domain-specific retrievers, especially in specialized domains such as biomedicine. We present Synthesize-Train-Merge (STM), a modular framework that enhances decoder-only LLMs with synthetic hard negatives, retrieval prompt optimization, and model merging. Experiments on a subset of 12 medical and general tasks from the MTEB benchmark show STM boosts task-specific experts by up to 23.5\% (average 7.5\%) and produces merged models that outperform both single experts and strong baselines without extensive pretraining. Our results demonstrate a scalable, efficient path for turning general LLMs into high-performing, domain-specialized retrievers, preserving general-domain capabilities while excelling on specialized tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04729v1">"Be My Cheese?": Cultural Nuance Benchmarking for Machine Translation in Multilingual LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-04
      | 💬 under peer-review
    </div>
    <details class="paper-abstract">
      We present a large-scale human evaluation benchmark for assessing cultural localisation in machine translation produced by state-of-the-art multilingual large language models (LLMs). Existing MT benchmarks emphasise token-level and grammatical accuracy, but of ten overlook pragmatic and culturally grounded competencies required for real-world localisation. Building on a pilot study of 87 translations across 20 languages, we evaluate 7 multilingual LLMs across 15 target languages with 5 native-speaker raters per language. Raters scored both full-text translations and segment-level instances of culturally nuanced language (idioms, puns, holidays, and culturally embedded concepts) on an ordinal 0-3 quality scale; segment ratings additionally included an NA option for untranslated segments. Across full-text evaluations, mean overall quality is modest (1.68/3): GPT-5 (2.10/3), Claude Sonnet 3.7 (1.97/3), and Mistral Medium 3.1 (1.84/3) form the strongest tier with fewer catastrophic failures. Segment-level results show sharp category effects: holidays (2.20/3) and cultural concepts (2.19/3) translate substantially better than idioms (1.65/3) and puns (1.45/3), and idioms are most likely to be left untranslated. These findings demonstrate a persistent gap between grammatical adequacy and cultural resonance. To our knowledge, this is the first multilingual, human-annotated benchmark focused explicitly on cultural nuance in translation and localisation, highlighting the need for culturally informed training data, improved cross-lingual pragmatics, and evaluation paradigms that better reflect real-world communicative competence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06655v2">GSAE: Graph-Regularized Sparse Autoencoders for Robust LLM Safety Steering</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) face critical safety challenges, as they can be manipulated to generate harmful content through adversarial prompts and jailbreak attacks. Many defenses are typically either black-box guardrails that filter outputs, or internals-based methods that steer hidden activations by operationalizing safety as a single latent feature or dimension. While effective for simple concepts, this assumption is limiting, as recent evidence shows that abstract concepts such as refusal and temporality are distributed across multiple features rather than isolated in one. To address this limitation, we introduce Graph-Regularized Sparse Autoencoders (GSAEs), which extends SAEs with a Laplacian smoothness penalty on the neuron co-activation graph. Unlike standard SAEs that assign each concept to a single latent feature, GSAEs recover smooth, distributed safety representations as coherent patterns spanning multiple features. We empirically demonstrate that GSAE enables effective runtime safety steering, assembling features into a weighted set of safety-relevant directions and controlling them with a two-stage gating mechanism that activates interventions only when harmful prompts or continuations are detected during generation. This approach enforces refusals adaptively while preserving utility on benign queries. Across safety and QA benchmarks, GSAE steering achieves an average 82% selective refusal rate, substantially outperforming standard SAE steering (42%), while maintaining strong task accuracy (70% on TriviaQA, 65% on TruthfulQA, 74% on GSM8K). Robustness experiments further show generalization across LLaMA-3, Mistral, Qwen, and Phi families and resilience against jailbreak attacks (GCG, AutoDAN), consistently maintaining >= 90% refusal of harmful content.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04693v1">LinGO: A Linguistic Graph Optimization Framework with LLMs for Interpreting Intents of Online Uncivil Discourse</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Detecting uncivil language is crucial for maintaining safe, inclusive, and democratic online spaces. Yet existing classifiers often misinterpret posts containing uncivil cues but expressing civil intents, leading to inflated estimates of harmful incivility online. We introduce LinGO, a linguistic graph optimization framework for large language models (LLMs) that leverages linguistic structures and optimization techniques to classify multi-class intents of incivility that use various direct and indirect expressions. LinGO decomposes language into multi-step linguistic components, identifies targeted steps that cause the most errors, and iteratively optimizes prompt and/or example components for targeted steps. We evaluate it using a dataset collected during the 2022 Brazilian presidential election, encompassing four forms of political incivility: Impoliteness (IMP), Hate Speech and Stereotyping (HSST), Physical Harm and Violent Political Rhetoric (PHAVPR), and Threats to Democratic Institutions and Values (THREAT). Each instance is annotated with six types of civil/uncivil intent. We benchmark LinGO using three cost-efficient LLMs: GPT-5-mini, Gemini 2.5 Flash-Lite, and Claude 3 Haiku, and four optimization techniques: TextGrad, AdalFlow, DSPy, and Retrieval-Augmented Generation (RAG). The results show that, across all models, LinGO consistently improves accuracy and weighted F1 compared with zero-shot, chain-of-thought, direct optimization, and fine-tuning baselines. RAG is the strongest optimization technique and, when paired with Gemini model, achieves the best overall performance. These findings demonstrate that incorporating multi-step linguistic components into LLM instructions and optimize targeted components can help the models explain complex semantic meanings, which can be extended to other complex semantic explanation tasks in the future.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04674v1">Overstating Attitudes, Ignoring Networks: LLM Biases in Simulating Misinformation Susceptibility</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as proxies for human judgment in computational social science, yet their ability to reproduce patterns of susceptibility to misinformation remains unclear. We test whether LLM-simulated survey respondents, prompted with participant profiles drawn from social survey data measuring network, demographic, attitudinal and behavioral features, can reproduce human patterns of misinformation belief and sharing. Using three online surveys as baselines, we evaluate whether LLM outputs match observed response distributions and recover feature-outcome associations present in the original survey data. LLM-generated responses capture broad distributional tendencies and show modest correlation with human responses, but consistently overstate the association between belief and sharing. Linear models fit to simulated responses exhibit substantially higher explained variance and place disproportionate weight on attitudinal and behavioral features, while largely ignoring personal network characteristics, relative to models fit to human responses. Analyses of model-generated reasoning and LLM training data suggest that these distortions reflect systematic biases in how misinformation-related concepts are represented. Our findings suggest that LLM-based survey simulations are better suited for diagnosing systematic divergences from human judgment than for substituting it.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10931v2">Asynchronous Reasoning: Training-Free Interactive Thinking LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-04
      | 💬 Preprint, work in progress
    </div>
    <details class="paper-abstract">
      Many state-of-the-art LLMs are trained to think before giving their answer. Reasoning can greatly improve language model capabilities, but it also makes them less interactive: given a new input, a model must stop thinking before it can respond. Real-world use cases such as voice-based or embodied assistants require an LLM agent to respond and adapt to additional information in real time, which is incompatible with sequential interactions. In contrast, humans can listen, think, and act asynchronously: we begin thinking about the problem while reading it and continue thinking while formulating the answer. In this work, we augment LLMs capable of reasoning to operate in a similar way without additional training. Our method uses the properties of positional embeddings to enable LLMs built for sequential generation to simultaneously think, listen, and write outputs. We evaluate our approach on math, commonsense, and safety reasoning: it allows models to generate accurate thinking-augmented answers while reducing time to first non-thinking token from minutes to ${\le}$ 5s and the overall real-time delays by up to $12{\times}$.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04630v1">Mapping the Web of Science, a large-scale graph and text-based dataset with LLM embeddings</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Large text data sets, such as publications, websites, and other text-based media, inherit two distinct types of features: (1) the text itself, its information conveyed through semantics, and (2) its relationship to other texts through links, references, or shared attributes. While the latter can be described as a graph structure and can be handled by a range of established algorithms for classification and prediction, the former has recently gained new potential through the use of LLM embedding models. Demonstrating these possibilities and their practicability, we investigate the Web of Science dataset, containing ~56 million scientific publications through the lens of our proposed embedding method, revealing a self-structured landscape of texts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04620v1">QUATRO: Query-Adaptive Trust Region Policy Optimization for LLM Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      GRPO-style reinforcement learning (RL)-based LLM fine-tuning algorithms have recently gained popularity. Relying on heuristic trust-region approximations, however, they can lead to brittle optimization behavior, as global importance-ratio clipping and group-wise normalization fail to regulate samples whose importance ratios fall outside the clipping range. We propose Query-Adaptive Trust-Region policy Optimization (QUATRO), which directly enforces trust-region constraints through a principled optimization. This yields a clear and interpretable objective that enables explicit control over policy updates and stable, entropy-controlled optimization, with a stabilizer terms arising intrinsically from the exact trust-region formulation. Empirically verified on diverse mathematical reasoning benchmarks, QUATRO shows stable training under increased policy staleness and aggressive learning rates, maintaining well-controlled entropy throughout training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.02389v3">From Trace to Line: LLM Agent for Real-World OSS Vulnerability Localization</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Large language models show promise for vulnerability discovery, yet prevailing methods inspect code in isolation, struggle with long contexts, and focus on coarse function- or file-level detections that offer limited guidance to engineers who need precise line-level localization for targeted patches. We introduce T2L, an executable framework for project-level, line-level vulnerability localization that progressively narrows scope from repository modules to exact vulnerable lines via AST-based chunking and evidence-guided refinement. We provide a baseline agent with an Agentic Trace Analyzer (ATA) that fuses runtime evidence such as crash points and stack traces to translate failure symptoms into actionable diagnoses. To enable rigorous evaluation, we introduce T2L-ARVO, an expert-verified 50-case benchmark spanning five crash families in real-world projects. On T2L-ARVO, our baseline achieves up to 58.0% detection and 54.8% line-level localization rate. Together, T2L framework advance LLM-based vulnerability detection toward deployable, precision diagnostics in open-source software workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04613v1">Disentangling meaning from language in LLM-based machine translation</a></div>
    <div class="paper-meta">
      📅 2026-02-04
      | 💬 61 pages, 70 figures
    </div>
    <details class="paper-abstract">
      Mechanistic Interpretability (MI) seeks to explain how neural networks implement their capabilities, but the scale of Large Language Models (LLMs) has limited prior MI work in Machine Translation (MT) to word-level analyses. We study sentence-level MT from a mechanistic perspective by analyzing attention heads to understand how LLMs internally encode and distribute translation functions. We decompose MT into two subtasks: producing text in the target language (i.e. target language identification) and preserving the input sentence's meaning (i.e. sentence equivalence). Across three families of open-source models and 20 translation directions, we find that distinct, sparse sets of attention heads specialize in each subtask. Based on this insight, we construct subtask-specific steering vectors and show that modifying just 1% of the relevant heads enables instruction-free MT performance comparable to instruction-based prompting, while ablating these heads selectively disrupts their corresponding translation functions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04595v1">Harmonia: Algorithm-Hardware Co-Design for Memory- and Compute-Efficient BFP-based LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are powerful but incur high memory and computation costs. Quantization is an effective solution, with INT weights and FP activations being widely adopted to preserve accuracy. Prior works further reduce FP overhead by using block floating point (BFP) activations in linear layers, but fail to extend BFP to attention layers due to severe accuracy degradation, limiting overall efficiency. To address this challenge, we propose Harmonia, an algorithm-hardware co-design framework that enables all-layer BFP activations with a configurable hardware architecture. First, we systematically explore BFP configurations to achieve a better trade-off between accuracy and activation compression across all layers. Second, to reduce KV-cache storage and computation in attention layers, we introduce an asymmetric bit-allocation strategy and computations in attention layers,we introduce an asymmetric bit-allocation strategy combined with a hybrid offline-online outlier smoothing technique. This allow aggressive KV-cache compression from FP16 to 4-bit-mantissa BFP with only 0.3% average accuracy loss. Third, to fully exploit all-layer BFP activations, we design dedicated hardware components, including a reconfigurable PE supporting mixed data formats (BFP-INT and BPF-BFP), a real-time FP16-to-BFP converter, and a tiling-aware dataflow to reduce memory traffic. We evaluate Harmonia on GEMM operations in both linear and attention layers across eight widely used LLMs. Compared with prior works, Harmonia achieves 3.84x (up to 5.05x) higher area efficiency, 2.03x (up to 3.90x) better energy efficiency, and 3.08x (up to 4.62x) speedup on average.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04572v1">From Competition to Collaboration: Designing Sustainable Mechanisms Between LLMs and Online Forums</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      While Generative AI (GenAI) systems draw users away from (Q&A) forums, they also depend on the very data those forums produce to improve their performance. Addressing this paradox, we propose a framework of sequential interaction, in which a GenAI system proposes questions to a forum that can publish some of them. Our framework captures several intricacies of such a collaboration, including non-monetary exchanges, asymmetric information, and incentive misalignment. We bring the framework to life through comprehensive, data-driven simulations using real Stack Exchange data and commonly used LLMs. We demonstrate the incentive misalignment empirically, yet show that players can achieve roughly half of the utility in an ideal full-information scenario. Our results highlight the potential for sustainable collaboration that preserves effective knowledge sharing between AI systems and human knowledge platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04570v1">Can LLMs capture stable human-generated sentence entropy measures?</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Predicting upcoming words is a core mechanism of language comprehension and may be quantified using Shannon entropy. There is currently no empirical consensus on how many human responses are required to obtain stable and unbiased entropy estimates at the word level. Moreover, large language models (LLMs) are increasingly used as substitutes for human norming data, yet their ability to reproduce stable human entropy remains unclear. Here, we address both issues using two large publicly available cloze datasets in German 1 and English 2. We implemented a bootstrap-based convergence analysis that tracks how entropy estimates stabilize as a function of sample size. Across both languages, more than 97% of sentences reached stable entropy estimates within the available sample sizes. 90% of sentences converged after 111 responses in German and 81 responses in English, while low-entropy sentences (<1) required as few as 20 responses and high-entropy sentences (>2.5) substantially more. These findings provide the first direct empirical validation for common norming practices and demonstrate that convergence critically depends on sentence predictability. We then compared stable human entropy values with entropy estimates derived from several LLMs, including GPT-4o, using both logit-based probability extraction and sampling-based frequency estimation, GPT2-xl/german-GPT-2, RoBERTa Base/GottBERT, and LLaMA 2 7B Chat. GPT-4o showed the highest correspondence with human data, although alignment depended strongly on the extraction method and prompt design. Logit-based estimates minimized absolute error, whereas sampling-based estimates were better in capturing the dispersion of human variability. Together, our results establish practical guidelines for human norming and show that while LLMs can approximate human entropy, they are not interchangeable with stable human-derived distributions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04541v1">LycheeDecode: Accelerating Long-Context LLM Inference via Hybrid-Head Sparse Decoding</a></div>
    <div class="paper-meta">
      📅 2026-02-04
      | 💬 ICLR 2026
    </div>
    <details class="paper-abstract">
      The proliferation of long-context large language models (LLMs) exposes a key bottleneck: the rapidly expanding key-value cache during decoding, which imposes heavy memory and latency costs. While recent approaches attempt to alleviate this by sharing a single set of crucial tokens across layers, such coarse-grained sharing undermines model performance by neglecting the functional diversity of attention heads. To address this, we propose LycheeDecode, an efficient decoding method centered on a fine-grained hybrid-head attention mechanism that employs a hardware-efficient top-k selection strategy. Specifically, the novel HardKuma-based mechanism partitions attention heads into a small subset of retrieval heads that dynamically identify crucial tokens and a majority of sparse heads that reuse them for efficient computation. Through extensive experiments on leading models like Llama3 and Qwen3 across diverse benchmarks for long-context understanding (e.g., LongBench, RULER) and complex reasoning (e.g., AIME24, OlympiadBench), we demonstrate that LycheeDecode achieves generative quality comparable to, and at times surpassing even the full-attention baseline. Crucially, this is accomplished with up to a 2.7x speedup at a 128K context length. By preserving the functional diversity of attention heads, our fine-grained strategy overcomes the performance bottlenecks of existing methods, providing a powerful and validated pathway to both efficient and high-quality long-context LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20417v2">SpeechMapper: Speech-to-text Embedding Projector for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-04
      | 💬 Accepted to ICASSP 2026
    </div>
    <details class="paper-abstract">
      Current speech LLMs bridge speech foundation models to LLMs using projection layers, training all of these components on speech instruction data. This strategy is computationally intensive and susceptible to task and prompt overfitting. We present SpeechMapper, a cost-efficient speech-to-LLM-embedding training approach that mitigates overfitting, enabling more robust and generalizable models. Our model is first pretrained without the LLM on inexpensive hardware, and then efficiently attached to the target LLM via a brief 1K-step instruction tuning (IT) stage. Through experiments on speech translation and spoken question answering, we demonstrate the versatility of SpeechMapper's pretrained block, presenting results for both task-agnostic IT, an ASR-based adaptation strategy that does not train in the target task, and task-specific IT. In task-agnostic settings, Speechmapper rivals the best instruction-following speech LLM from IWSLT25, despite never being trained on these tasks, while in task-specific settings, it outperforms this model across many datasets, despite requiring less data and compute. Overall, SpeechMapper offers a practical and scalable approach for efficient, generalizable speech-LLM integration without large-scale IT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04510v1">OSCAgent: Accelerating the Discovery of Organic Solar Cells with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Organic solar cells (OSCs) hold great promise for sustainable energy, but discovering high-performance materials is time-consuming and costly. Existing molecular generation methods can aid the design of OSC molecules, but they are mostly confined to optimizing known backbones and lack effective use of domain-specific chemical knowledge, often leading to unrealistic candidates. In this paper, we introduce OSCAgent, a multi-agent framework for OSC molecular discovery that unifies retrieval-augmented design, molecular generation, and systematic evaluation into a continuously improving pipeline, without requiring additional human intervention. OSCAgent comprises three collaborative agents. The Planner retrieves knowledge from literature-curated molecules and prior candidates to guide design directions. The Generator proposes new OSC acceptors aligned with these plans. The Experimenter performs comprehensive evaluation of candidate molecules and provides feedback for refinement. Experiments show that OSCAgent produces chemically valid, synthetically accessible OSC molecules and achieves superior predicted performance compared to both traditional and large language model (LLM)-only baselines. Representative results demonstrate that some candidates achieve predicted efficiencies approaching 18\%. The code will be publicly available.
    </details>
</div>
