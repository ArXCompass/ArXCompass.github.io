# llm - 2026_01

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- Part 4
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15817v5">A Causal Perspective on Measuring, Explaining and Mitigating Smells in LLM-Generated Code</a></div>
    <div class="paper-meta">
      📅 2026-01-16
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have accelerated their adoption in software engineering contexts. However, concerns persist about the structural quality of the code they produce. In particular, LLMs often replicate poor coding practices, introducing code smells (i.e., patterns that hinder readability, maintainability, or design integrity). Although prior research has examined the detection or repair of smells, we still lack a clear understanding of how and when these issues emerge in generated code. This paper addresses this gap by systematically measuring, explaining and mitigating smell propensity in LLM-generated code. We build on the Propensity Smelly Score (PSC), a probabilistic metric that estimates the likelihood of generating particular smell types, and establish its robustness as a signal of structural quality. Using PSC as an instrument for causal analysis, we identify how generation strategy, model size, model architecture and prompt formulation shape the structural properties of generated code. Our findings show that prompt design and architectural choices play a decisive role in smell propensity and motivate practical mitigation strategies that reduce its occurrence. A user study further demonstrates that PSC helps developers interpret model behavior and assess code quality, providing evidence that smell propensity signals can support human judgement. Taken together, our work lays the groundwork for integrating quality-aware assessments into the evaluation and deployment of LLMs for code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11332v1">Idea First, Code Later: Disentangling Problem Solving from Code Generation in Evaluating LLMs for Competitive Programming</a></div>
    <div class="paper-meta">
      📅 2026-01-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) increasingly succeed on competitive programming problems, yet existing evaluations conflate algorithmic reasoning with code-level implementation. We argue that competitive programming is fundamentally a problem-solving task and propose centering natural-language editorials in both solution generation and evaluation. Generating an editorial prior to code improves solve rates for some LLMs, with substantially larger gains when using expertly written gold editorials. However, even with gold editorials, models continue to struggle with implementation, while the gap between generated and gold editorials reveals a persistent problem-solving bottleneck in specifying correct and complete algorithms. Beyond pass/fail metrics, we diagnose reasoning errors by comparing model-generated editorials to gold standards using expert annotations and validate an LLM-as-a-judge protocol for scalable evaluation. We introduce a dataset of 83 ICPC-style problems with gold editorials and full test suites, and evaluate 19 LLMs, arguing that future benchmarks should explicitly separate problem solving from implementation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11314v1">Membership Inference on LLMs in the Wild</a></div>
    <div class="paper-meta">
      📅 2026-01-16
    </div>
    <details class="paper-abstract">
      Membership Inference Attacks (MIAs) act as a crucial auditing tool for the opaque training data of Large Language Models (LLMs). However, existing techniques predominantly rely on inaccessible model internals (e.g., logits) or suffer from poor generalization across domains in strict black-box settings where only generated text is available. In this work, we propose SimMIA, a robust MIA framework tailored for this text-only regime by leveraging an advanced sampling strategy and scoring mechanism. Furthermore, we present WikiMIA-25, a new benchmark curated to evaluate MIA performance on modern proprietary LLMs. Experiments demonstrate that SimMIA achieves state-of-the-art results in the black-box setting, rivaling baselines that exploit internal model information.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.20075v6">LLMs can hide text in other text of the same length</a></div>
    <div class="paper-meta">
      📅 2026-01-16
      | 💬 21 pages, main paper 9 pages. v5 contains an Italian translation of this paper by the author
    </div>
    <details class="paper-abstract">
      A meaningful text can be hidden inside another, completely different yet still coherent and plausible, text of the same length. For example, a tweet containing a harsh political critique could be embedded in a tweet that celebrates the same political leader, or an ordinary product review could conceal a secret manuscript. This uncanny state of affairs is now possible thanks to Large Language Models, and in this paper we present Calgacus, a simple and efficient protocol to achieve it. We show that even modest 8-billion-parameter open-source LLMs are sufficient to obtain high-quality results, and a message as long as this abstract can be encoded and decoded locally on a laptop in seconds. The existence of such a protocol demonstrates a radical decoupling of text from authorial intent, further eroding trust in written communication, already shaken by the rise of LLM chatbots. We illustrate this with a concrete scenario: a company could covertly deploy an unfiltered LLM by encoding its answers within the compliant responses of a safe model. This possibility raises urgent questions for AI safety and challenges our understanding of what it means for a Large Language Model to know something.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11293v1">One LLM to Train Them All: Multi-Task Learning Framework for Fact-Checking</a></div>
    <div class="paper-meta">
      📅 2026-01-16
      | 💬 Accepted version in ECIR 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are reshaping automated fact-checking (AFC) by enabling unified, end-to-end verification pipelines rather than isolated components. While large proprietary models achieve strong performance, their closed weights, complexity, and high costs limit sustainability. Fine-tuning smaller open weight models for individual AFC tasks can help but requires multiple specialized models resulting in high costs. We propose \textbf{multi-task learning (MTL)} as a more efficient alternative that fine-tunes a single model to perform claim detection, evidence ranking, and stance detection jointly. Using small decoder-only LLMs (e.g., Qwen3-4b), we explore three MTL strategies: classification heads, causal language modeling heads, and instruction-tuning, and evaluate them across model sizes, task orders, and standard non-LLM baselines. While multitask models do not universally surpass single-task baselines, they yield substantial improvements, achieving up to \textbf{44\%}, \textbf{54\%}, and \textbf{31\%} relative gains for claim detection, evidence re-ranking, and stance detection, respectively, over zero-/few-shot settings. Finally, we also provide practical, empirically grounded guidelines to help practitioners apply MTL with LLMs for automated fact-checking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11286v1">XChoice: Explainable Evaluation of AI-Human Alignment in LLM-based Constrained Choice Decision Making</a></div>
    <div class="paper-meta">
      📅 2026-01-16
    </div>
    <details class="paper-abstract">
      We present XChoice, an explainable framework for evaluating AI-human alignment in constrained decision making. Moving beyond outcome agreement such as accuracy and F1 score, XChoice fits a mechanism-based decision model to human data and LLM-generated decisions, recovering interpretable parameters that capture the relative importance of decision factors, constraint sensitivity, and implied trade-offs. Alignment is assessed by comparing these parameter vectors across models, options, and subgroups. We demonstrate XChoice on Americans' daily time allocation using the American Time Use Survey (ATUS) as human ground truth, revealing heterogeneous alignment across models and activities and salient misalignment concentrated in Black and married groups. We further validate robustness of XChoice via an invariance analysis and evaluate targeted mitigation with a retrieval augmented generation (RAG) intervention. Overall, XChoice provides mechanism-based metrics that diagnose misalignment and support informed improvements beyond surface outcome matching.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.14161v3">MIST: Towards Multi-dimensional Implicit BiaS Evaluation of LLMs for Theory of Mind</a></div>
    <div class="paper-meta">
      📅 2026-01-16
    </div>
    <details class="paper-abstract">
      Theory of Mind (ToM) in Large Language Models (LLMs) refers to the model's ability to infer the mental states of others, with failures in this ability often manifesting as systemic implicit biases. Assessing this challenge is difficult, as traditional direct inquiry methods are often met with refusal to answer and fail to capture its subtle and multidimensional nature. Therefore, we propose MIST, which reconceptualizes the content model of stereotypes into multidimensional failures of ToM, specifically in the domains of competence, sociability, and morality. The framework introduces two indirect tasks. The Word Association Bias Test (WABT) assesses implicit lexical associations, while the Affective Attribution Test (AAT) measures implicit emotional tendencies, aiming to uncover latent stereotypes without triggering model avoidance. Through extensive experimentation on eight state-of-the-art LLMs, our framework demonstrates the ability to reveal complex bias structures and improved robustness. All data and code will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11238v1">LLM-Assisted Pseudo-Relevance Feedback</a></div>
    <div class="paper-meta">
      📅 2026-01-16
      | 💬 Accepted ECIR 2026
    </div>
    <details class="paper-abstract">
      Query expansion is a long-standing technique to mitigate vocabulary mismatch in ad hoc Information Retrieval. Pseudo-relevance feedback methods, such as RM3, estimate an expanded query model from the top-ranked documents, but remain vulnerable to topic drift when early results include noisy or tangential content. Recent approaches instead prompt Large Language Models to generate synthetic expansions or query variants. While effective, these methods risk hallucinations and misalignment with collection-specific terminology. We propose a hybrid alternative that preserves the robustness and interpretability of classical PRF while leveraging LLM semantic judgement. Our method inserts an LLM-based filtering stage prior to RM3 estimation: the LLM judges the documents in the initial top-$k$ ranking, and RM3 is computed only over those accepted as relevant. This simple intervention improves over blind PRF and a strong baseline across several datasets and metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.04359v3">Efficient Reinforcement Learning with Semantic and Token Entropy for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-16
    </div>
    <details class="paper-abstract">
      Reinforcement learning with verifiable rewards (RLVR) has demonstrated superior performance in enhancing the reasoning capability of large language models (LLMs). However, this accuracy-oriented learning paradigm often suffers from entropy collapse, which reduces policy exploration and limits reasoning capabilities. To address this challenge, we propose an efficient reinforcement learning framework that leverages entropy signals at both the semantic and token levels to improve reasoning. From the data perspective, we introduce semantic entropy-guided curriculum learning, organizing training data from low to high semantic entropy to guide progressive optimization from easier to more challenging tasks. For the algorithmic design, we adopt non-uniform token treatment by imposing KL regularization on low-entropy tokens that critically impact policy exploration and applying stronger constraints on high-covariance portions within these tokens. By jointly optimizing data organization and algorithmic design, our method effectively mitigates entropy collapse and enhances LLM reasoning. Experimental results across 6 benchmarks with 3 different parameter-scale base models demonstrate that our method outperforms other entropy-based approaches in improving reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.03687v2">MedReflect: Teaching Medical LLMs to Self-Improve via Reflective Correction</a></div>
    <div class="paper-meta">
      📅 2026-01-16
    </div>
    <details class="paper-abstract">
      Medical problem-solving demands expert knowledge and intricate reasoning. Recent studies of large language models (LLMs) attempt to ease this complexity by introducing external knowledge verification through retrieval-augmented generation or by training on reasoning datasets. However, these approaches suffer from drawbacks such as retrieval overhead and high annotation costs, and they heavily rely on substituted external assistants to reach limited performance in medical field. In this paper, we introduce MedReflect, a generalizable framework designed to inspire LLMs with a physician-like reflective thinking mode. MedReflect generates a single-pass reflection chain that includes initial hypothesis generation, self-questioning, self-answering and decision refinement. This self-verified and self-reflective nature releases large language model's latent capability in medical problem-solving without external retrieval or heavy annotation. We demonstrate that MedReflect enables cost-efficient medical dataset construction. With only a minimal subset of randomly sampled training examples and lightweight fine-tuning, this approach achieves notable absolute accuracy improvements across a series of medical benchmarks while significantly cutting annotation requirements. Our results provide evidence that LLMs can learn to solve specialized medical problems via self-reflection and self-improvement, reducing reliance on external supervision and extensive task-specific fine-tuning data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.05748v2">Communication Enables Cooperation in LLM Agents: A Comparison with Curriculum-Based Approaches</a></div>
    <div class="paper-meta">
      📅 2026-01-16
    </div>
    <details class="paper-abstract">
      Eliciting cooperation in multi-agent LLM systems is critical for AI alignment. We investigate two approaches: direct communication and curriculum learning. In a 4-player Stag Hunt, a one-word "cheap talk" channel increases cooperation from 0% to 48.3%, demonstrating communication as a robust coordination mechanism. In contrast, we find that curriculum learning is highly sensitive to design choices: our pedagogical curriculum through progressively complex games reduced agent payoffs by 27.4% in an Iterated Public Goods Game with Punishment, demonstrating that optimizing for short-term rationality can actively undermine alignment goals. Qualitative analysis reveals that curricula emphasizing defection-equilibrium games can induce "learned pessimism" in agents. These findings suggest that for coordination problems, simple communication protocols may be more reliable than experience-based training, and that curriculum design for social dilemmas requires careful attention to the strategic lessons embedded in game sequences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.11288v2">Emergent Misalignment via In-Context Learning: Narrow in-context examples can produce broadly misaligned LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-16
    </div>
    <details class="paper-abstract">
      Recent work has shown that narrow finetuning can produce broadly misaligned LLMs, a phenomenon termed emergent misalignment (EM). While concerning, these findings were limited to finetuning and activation steering, leaving out in-context learning (ICL). We therefore ask: does EM emerge in ICL? We find that it does: across four model families (Gemini, Kimi-K2, Grok, and Qwen), narrow in-context examples cause models to produce misaligned responses to benign, unrelated queries. With 16 in-context examples, EM rates range from 1\% to 24\% depending on model and domain, appearing with as few as 2 examples. Neither larger model scale nor explicit reasoning provides reliable protection. We formulate and test a hypothesis, which explains in-context EM as conflict between safety objectives and context-following behavior. Consistent with this, instructing models to prioritize safety reduces EM while prioritizing context-following increases it. These findings establish ICL as a previously underappreciated vector for emergent misalignment that operates without parameter modification and resists simple scaling-based solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11124v1">Learn Before Represent: Bridging Generative and Contrastive Learning for Domain-Specific LLM Embeddings</a></div>
    <div class="paper-meta">
      📅 2026-01-16
      | 💬 10 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) adapted via contrastive learning excel in general representation learning but struggle in vertical domains like chemistry and law, primarily due to a lack of domain-specific knowledge. This work identifies a core bottleneck: the prevailing ``LLM+CL'' paradigm focuses on semantic alignment but cannot perform knowledge acquisition, leading to failures on specialized terminology. To bridge this gap, we propose Learn Before Represent (LBR), a novel two-stage framework. LBR first injects domain knowledge via an Information Bottleneck-Constrained Generative Learning stage, preserving the LLM's causal attention to maximize knowledge acquisition while compressing semantics. It then performs Generative-Refined Contrastive Learning on the compressed representations for alignment. This approach maintains architectural consistency and resolves the objective conflict between generative and contrastive learning. Extensive experiments on medical, chemistry, and code retrieval tasks show that LBR significantly outperforms strong baselines. Our work establishes a new paradigm for building accurate and robust representations in vertical domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11118v1">Optimized Algorithms for Text Clustering with LLM-Generated Constraints</a></div>
    <div class="paper-meta">
      📅 2026-01-16
      | 💬 AAAI-26
    </div>
    <details class="paper-abstract">
      Clustering is a fundamental tool that has garnered significant interest across a wide range of applications including text analysis. To improve clustering accuracy, many researchers have incorporated background knowledge, typically in the form of must-link and cannot-link constraints, to guide the clustering process. With the recent advent of large language models (LLMs), there is growing interest in improving clustering quality through LLM-based automatic constraint generation. In this paper, we propose a novel constraint-generation approach that reduces resource consumption by generating constraint sets rather than using traditional pairwise constraints. This approach improves both query efficiency and constraint accuracy compared to state-of-the-art methods. We further introduce a constrained clustering algorithm tailored to the characteristics of LLM-generated constraints. Our method incorporates a confidence threshold and a penalty mechanism to address potentially inaccurate constraints. We evaluate our approach on five text datasets, considering both the cost of constraint generation and the overall clustering performance. The results show that our method achieves clustering accuracy comparable to the state-of-the-art algorithms while reducing the number of LLM queries by more than 20 times.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11063v1">H-AIM: Orchestrating LLMs, PDDL, and Behavior Trees for Hierarchical Multi-Robot Planning</a></div>
    <div class="paper-meta">
      📅 2026-01-16
    </div>
    <details class="paper-abstract">
      In embodied artificial intelligence, enabling heterogeneous robot teams to execute long-horizon tasks from high-level instructions remains a critical challenge. While large language models (LLMs) show promise in instruction parsing and preliminary planning, they exhibit limitations in long-term reasoning and dynamic multi-robot coordination. We propose Hierarchical Autonomous Intelligent Multi-Robot Planning(H-AIM), a novel embodied multi-robot task planning framework that addresses these issues through a three-stage cascaded architecture: 1) It leverages an LLM to parse instructions and generate Planning Domain Definition Language (PDDL) problem descriptions, thereby transforming commands into formal planning problems; 2) It combines the semantic reasoning of LLMs with the search capabilities of a classical planner to produce optimized action sequences; 3) It compiles the resulting plan into behavior trees for reactive control. The framework supports dynamically sized heterogeneous robot teams via a shared blackboard mechanism for communication and state synchronization. To validate our approach, we introduce the MACE-THOR benchmark dataset, comprising 42 complex tasks across 8 distinct household layouts. Experimental results demonstrate that H-AIM achieves a remarkable performance improvement, elevating the task success rate from 12% to 55% and boosting the goal condition recall from 32% to 72% against the strongest baseline, LaMMA-P.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11061v1">Spurious Rewards Paradox: Mechanistically Understanding How RLVR Activates Memorization Shortcuts in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-16
      | 💬 Work in process
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Rewards (RLVR) is highly effective for enhancing LLM reasoning, yet recent evidence shows models like Qwen 2.5 achieve significant gains even with spurious or incorrect rewards. We investigate this phenomenon and identify a "Perplexity Paradox": spurious RLVR triggers a divergence where answer-token perplexity drops while prompt-side coherence degrades, suggesting the model is bypassing reasoning in favor of memorization. Using Path Patching, Logit Lens, JSD analysis, and Neural Differential Equations, we uncover a hidden Anchor-Adapter circuit that facilitates this shortcut. We localize a Functional Anchor in the middle layers (L18-20) that triggers the retrieval of memorized solutions, followed by Structural Adapters in later layers (L21+) that transform representations to accommodate the shortcut signal. Finally, we demonstrate that scaling specific MLP keys within this circuit allows for bidirectional causal steering-artificially amplifying or suppressing contamination-driven performance. Our results provide a mechanistic roadmap for identifying and mitigating data contamination in RLVR-tuned models. Code is available at https://github.com/idwts/How-RLVR-Activates-Memorization-Shortcuts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11060v1">Children's Expectations, Engagement, and Evaluation of an LLM-enabled Spherical Visualization Platform in the Classroom</a></div>
    <div class="paper-meta">
      📅 2026-01-16
    </div>
    <details class="paper-abstract">
      We present our first stage results from deploying an LLM-augmented visualization software in a classroom setting to engage primary school children with earth-related datasets. Motivated by the growing interest in conversational AI as a means to support inquiry-based learning, we investigate children's expectations, engagement, and evaluation of a spoken LLM interface with a shared, immersive visualization system in a formal educational context. Our system integrates a speech-capable large language model with an interactive spherical display. It enables children to ask natural-language questions and receive coordinated verbal explanations and visual responses through the LLM-augmented visualization updating in real time based on spoken queries. We report on a classroom study with Swedish children aged 9-10, combining structured observation and small-group discussions to capture expectations prior to interaction, interaction patterns during facilitated sessions, and children's reflections on their encounter afterward. Our results provide empirical insights into children's initial encounters with an LLM-enabled visualization platform within a classroom setting and their expectations, interactions, and evaluations of the system. These findings inform the technology's potential for educational use and highlight important directions for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11038v1">Budget-Aware Anytime Reasoning with LLM-Synthesized Preference Data</a></div>
    <div class="paper-meta">
      📅 2026-01-16
      | 💬 13 pages, 3 figures
    </div>
    <details class="paper-abstract">
      We study the reasoning behavior of large language models (LLMs) under limited computation budgets. In such settings, producing useful partial solutions quickly is often more practical than exhaustive reasoning, which incurs high inference costs. Many real-world tasks, such as trip planning, require models to deliver the best possible output within a fixed reasoning budget. We introduce an anytime reasoning framework and the Anytime Index, a metric that quantifies how effectively solution quality improves as reasoning tokens increase. To further enhance efficiency, we propose an inference-time self-improvement method using LLM-synthesized preference data, where models learn from their own reasoning comparisons to produce better intermediate solutions. Experiments on NaturalPlan (Trip), AIME, and GPQA datasets show consistent gains across Grok-3, GPT-oss, GPT-4.1/4o, and LLaMA models, improving both reasoning quality and efficiency under budget constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11019v1">Finding the Translation Switch: Discovering and Exploiting the Task-Initiation Features in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-16
      | 💬 Accepted by AAAI 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) frequently exhibit strong translation abilities, even without task-specific fine-tuning. However, the internal mechanisms governing this innate capability remain largely opaque. To demystify this process, we leverage Sparse Autoencoders (SAEs) and introduce a novel framework for identifying task-specific features. Our method first recalls features that are frequently co-activated on translation inputs and then filters them for functional coherence using a PCA-based consistency metric. This framework successfully isolates a small set of **translation initiation** features. Causal interventions demonstrate that amplifying these features steers the model towards correct translation, while ablating them induces hallucinations and off-task outputs, confirming they represent a core component of the model's innate translation competency. Moving from analysis to application, we leverage this mechanistic insight to propose a new data selection strategy for efficient fine-tuning. Specifically, we prioritize training on **mechanistically hard** samples-those that fail to naturally activate the translation initiation features. Experiments show this approach significantly improves data efficiency and suppresses hallucinations. Furthermore, we find these mechanisms are transferable to larger models of the same family. Our work not only decodes a core component of the translation mechanism in LLMs but also provides a blueprint for using internal model mechanism to create more robust and efficient models. The codes are available at https://github.com/flamewei123/AAAI26-translation-Initiation-Features.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11004v1">NAACL: Noise-AwAre Verbal Confidence Calibration for LLMs in RAG Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-16
    </div>
    <details class="paper-abstract">
      Accurately assessing model confidence is essential for deploying large language models (LLMs) in mission-critical factual domains. While retrieval-augmented generation (RAG) is widely adopted to improve grounding, confidence calibration in RAG settings remains poorly understood. We conduct a systematic study across four benchmarks, revealing that LLMs exhibit poor calibration performance due to noisy retrieved contexts. Specifically, contradictory or irrelevant evidence tends to inflate the model's false certainty, leading to severe overconfidence. To address this, we propose NAACL Rules (Noise-AwAre Confidence CaLibration Rules) to provide a principled foundation for resolving overconfidence under noise. We further design NAACL, a noise-aware calibration framework that synthesizes supervision from about 2K HotpotQA examples guided by these rules. By performing supervised fine-tuning (SFT) with this data, NAACL equips models with intrinsic noise awareness without relying on stronger teacher models. Empirical results show that NAACL yields substantial gains, improving ECE scores by 10.9% in-domain and 8.0% out-of-domain. By bridging the gap between retrieval noise and verbal calibration, NAACL paves the way for both accurate and epistemically reliable LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11000v1">When Personalization Misleads: Understanding and Mitigating Hallucinations in Personalized LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-16
      | 💬 20 pages, 15 figures
    </div>
    <details class="paper-abstract">
      Personalized large language models (LLMs) adapt model behavior to individual users to enhance user satisfaction, yet personalization can inadvertently distort factual reasoning. We show that when personalized LLMs face factual queries, there exists a phenomenon where the model generates answers aligned with a user's prior history rather than the objective truth, resulting in personalization-induced hallucinations that degrade factual reliability and may propagate incorrect beliefs, due to representational entanglement between personalization and factual representations. To address this issue, we propose Factuality-Preserving Personalized Steering (FPPS), a lightweight inference-time approach that mitigates personalization-induced factual distortions while preserving personalized behavior. We further introduce PFQABench, the first benchmark designed to jointly evaluate factual and personalized question answering under personalization. Experiments across multiple LLM backbones and personalization methods show that FPPS substantially improves factual accuracy while maintaining personalized performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10955v1">Beyond Max Tokens: Stealthy Resource Amplification via Tool Calling Chains in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-16
    </div>
    <details class="paper-abstract">
      The agent-tool communication loop is a critical attack surface in modern Large Language Model (LLM) agents. Existing Denial-of-Service (DoS) attacks, primarily triggered via user prompts or injected retrieval-augmented generation (RAG) context, are ineffective for this new paradigm. They are fundamentally single-turn and often lack a task-oriented approach, making them conspicuous in goal-oriented workflows and unable to exploit the compounding costs of multi-turn agent-tool interactions. We introduce a stealthy, multi-turn economic DoS attack that operates at the tool layer under the guise of a correctly completed task. Our method adjusts text-visible fields and a template-governed return policy in a benign, Model Context Protocol (MCP)-compatible tool server, optimizing these edits with a Monte Carlo Tree Search (MCTS) optimizer. These adjustments leave function signatures unchanged and preserve the final payload, steering the agent into prolonged, verbose tool-calling sequences using text-only notices. This compounds costs across turns, escaping single-turn caps while keeping the final answer correct to evade validation. Across six LLMs on the ToolBench and BFCL benchmarks, our attack expands tasks into trajectories exceeding 60,000 tokens, inflates costs by up to 658x, and raises energy by 100-560x. It drives GPU KV cache occupancy from <1% to 35-74% and cuts co-running throughput by approximately 50%. Because the server remains protocol-compatible and task outcomes are correct, conventional checks fail. These results elevate the agent-tool interface to a first-class security frontier, demanding a paradigm shift from validating final answers to monitoring the economic and computational cost of the entire agentic process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10953v1">SwiftKV: An Edge-Oriented Attention Algorithm and Multi-Head Accelerator for Fast, Efficient LLM Decoding</a></div>
    <div class="paper-meta">
      📅 2026-01-16
    </div>
    <details class="paper-abstract">
      Edge acceleration for large language models is crucial for their widespread application; however, achieving fast attention inference and efficient decoding on resource-constrained edge accelerators remains challenging. This paper presents SwiftKV Attention, a per-token pipelined, low-latency single-pass attention inference algorithm, where every (kt, vt) in the KV cache is processed exactly once in a uniform per-token pipeline without score materialization, blockwise softmax, or a second pass, thereby enabling fast execution on edge accelerators with a single hardware set and no resource-intensive parallelism. Furthermore, to address the limited support for multi-head LLM decoding in existing accelerators, we design the SwiftKV-MHA accelerator, which enables high precision attention and low precision GEMV on the same processor array, achieving fast and efficient multi-head parallel decoding. Experimental results show that, on the edge accelerator, the SwiftKV Attention algorithm achieves a 7.16* speedup over native attention and significantly outperforms other attention algorithms. SwiftKV-MHA further reduces attention latency by 13.48*; under the same settings, it improves generation speed by 17.4% and increases token efficiency by 1.98* compared with state-of-the-art works.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01452v4">Robust and Efficient Zeroth-Order LLM Fine-Tuning via Adaptive Bayesian Subspace Optimizer</a></div>
    <div class="paper-meta">
      📅 2026-01-16
      | 💬 23 pages, 2 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) with zeroth-order (ZO) optimization reduces memory by approximating gradients through function evaluations. However, existing methods essentially perform updates in a one-dimensional space, and suffer from collapse or substantial performance degradation under low-precision training. We introduce BSZO, an adaptive \textbf{B}ayesian \textbf{S}ubspace \textbf{Z}eroth-Order \textbf{O}ptimizer, which applies Kalman filtering to combine finite-difference information across multiple perturbation directions within a subspace. By treating each finite-difference measurement as a noisy observation, BSZO builds a posterior distribution over the subspace-projected gradient and updates it through Bayesian inference, with a residual-based adaptive mechanism to adapt to noise variations. Theoretical analysis shows that BSZO improves the convergence rate by a factor of $k/γ$ compared to standard ZO methods. Experiments on RoBERTa, Mistral, and OPT models show that BSZO outperforms the baselines across various tasks, achieving up to 6.67\% absolute average improvement on OPT-13B while remaining robust under fp16/bf16 precision and keeping memory usage close to inference-only baselines (1.00$\times$--1.08$\times$ of MeZO).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.01386v3">ThinkEval: Practical Evaluation of Knowledge Leakage in LLM Editing using Thought-based Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2026-01-16
      | 💬 Accepted to TMLR
    </div>
    <details class="paper-abstract">
      Robust model-editing techniques are essential for deploying large language models (LLMs) in practical applications, as they enable cost-effective ways to deal with challenges such as privacy breaches, bias mitigation and misinformation spread. For example, an LLM-based healthcare assistance may need to update out-dated or incorrect knowledge to prevent harmful recommendations. However, many editing techniques focus on isolated facts, which critically fail to prevent indirect knowledge leakage -- the unintended reconstruction of edited-out information through persistent causal links and contextual relationships. To assist users in selecting the right editing technique, we develop and present ThinkEval, a framework to systematically quantify indirect knowledge leakage and ripple effects in model-editing. ThinkEval builds and employs specialized knowledge graphs to analyze the causal structure of facts before and after editing. To support this approach, we present KnowGIC, a benchmark dataset comprising multi-step reasoning paths that precisely measure these complex knowledge transformation effects. We evaluate five editing techniques: AlphaEdit, RECT, ROME, MEMIT, and PRUNE across multiple LLMs. Our results show that these techniques struggle to balance indirect fact suppression with the preservation of related knowledge, compromising the contextual integrity of a model's knowledge. Our dataset is available at: https://github.com/manitbaser/KnowGIC.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.02657v3">Less LLM, More Documents: Searching for Improved RAG</a></div>
    <div class="paper-meta">
      📅 2026-01-16
      | 💬 Proceeding Version of ECIR 2026
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) couples document retrieval with large language models (LLMs). While scaling generators often improves accuracy, it also increases inference and deployment overhead. We study an orthogonal axis: enlarging the retriever's corpus, and how it trades off with generator scale. Across multiple open-domain QA benchmarks, corpus scaling consistently strengthens RAG and can in many cases match the gains of moving to a larger model tier, though with diminishing returns at larger scales. Small- and mid-sized generators paired with larger corpora often rival much larger models with smaller corpora; mid-sized models tend to gain the most, while tiny and very large models benefit less. Our analysis suggests that these improvements arise primarily from increased coverage of answer-bearing passages, while utilization efficiency remains largely unchanged. Overall, our results characterize a corpus-generator trade-off in RAG and provide empirical guidance on how corpus scale and model capacity interact in this setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.01166v3">Hearing More with Less: Multi-Modal Retrieval-and-Selection Augmented Conversational LLM-Based ASR</a></div>
    <div class="paper-meta">
      📅 2026-01-16
      | 💬 AAAI 2026
    </div>
    <details class="paper-abstract">
      Automatic Speech Recognition (ASR) aims to convert human speech content into corresponding text. In conversational scenarios, effectively utilizing context can enhance its accuracy. Large Language Models' (LLMs) exceptional long-context understanding and reasoning abilities enable LLM-based ASR (LLM-ASR) to leverage historical context for recognizing conversational speech, which has a high degree of contextual relevance. However, existing conversational LLM-ASR methods use a fixed number of preceding utterances or the entire conversation history as context, resulting in significant ASR confusion and computational costs due to massive irrelevant and redundant information. This paper proposes a multi-modal retrieval-and-selection method named MARS that augments conversational LLM-ASR by enabling it to retrieve and select the most relevant acoustic and textual historical context for the current utterance. Specifically, multi-modal retrieval obtains a set of candidate historical contexts, each exhibiting high acoustic or textual similarity to the current utterance. Multi-modal selection calculates the acoustic and textual similarities for each retrieved candidate historical context and, by employing our proposed near-ideal ranking method to consider both similarities, selects the best historical context. Evaluations on the Interspeech 2025 Multilingual Conversational Speech Language Model Challenge dataset show that the LLM-ASR, when trained on only 1.5K hours of data and equipped with the MARS, outperforms the state-of-the-art top-ranking system trained on 179K hours of data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11835v1">Changes in Coding Behavior and Performance Since the Introduction of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-16
    </div>
    <details class="paper-abstract">
      The widespread availability of large language models (LLMs) has changed how students engage with coding and problem-solving. While these tools may increase student productivity, they also make it more difficult for instructors to assess students' learning and effort. In this quasi-longitudinal study, we analyze five years of student source code submissions in a graduate-level cloud computing course, focusing on an assignment that remained unchanged and examining students' behavior during the period spanning five semesters before the release of ChatGPT and five semesters after. Student coding behavior has changed significantly since Fall 2022. The length of their final submissions increased. Between consecutive submissions, average edit distances increased while average score improvement decreased, suggesting that both student productivity and learning have decreased after ChatGPT's release. Additionally, there are statistically significant correlations between these behavioral changes and their overall performance. Although we cannot definitively attribute them to LLM misuse, they are consistent with our hypothesis that some students are over-reliant on LLMs, which is negatively affecting their learning outcomes. Our findings raise an alarm around the first generation of graduates in the age of LLMs, calling upon both educators and employers to reflect on their evaluation methods for genuine expertise and productivity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.03706v3">LLM-enhanced Air Quality Monitoring Interface via Model Context Protocol</a></div>
    <div class="paper-meta">
      📅 2026-01-16
      | 💬 Accepted at 7th International Symposium on Advanced Electrical and Communication Technologies (ISAECT), 2025
    </div>
    <details class="paper-abstract">
      Air quality monitoring is central to environmental sustainability and public health, yet traditional systems remain difficult for non-expert users to interpret due to complex visualizations, limited interactivity, and high deployment costs. Recent advances in Large Language Models (LLMs) offer new opportunities to make sensor data more accessible, but their tendency to produce hallucinations limits reliability in safety-critical domains. To address these challenges, we present an LLM-enhanced Air Monitoring Interface (AMI) that integrates real-time sensor data with a conversational interface via the Model Context Protocol (MCP). Our system grounds LLM outputs in live environmental data, enabling accurate, context-aware responses while reducing hallucination risk. The architecture combines a Django-based backend, a responsive user dashboard, and a secure MCP server that exposes system functions as discoverable tools, allowing the LLM to act as an active operator rather than a passive responder. Expert evaluation demonstrated high factual accuracy (4.78), completeness (4.82), and minimal hallucinations (4.84), on a scale of 5, supported by inter-rater reliability analysis. These results highlight the potential of combining LLMs with standardized tool protocols to create reliable, secure, and user-friendly interfaces for real-time environmental monitoring.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11791v1">Beyond Tokens: Concept-Level Training Objectives for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-16
    </div>
    <details class="paper-abstract">
      The next-token prediction (NTP) objective has been foundational in the development of modern large language models (LLMs), driving advances in fluency and generalization. However, NTP operates at the \textit{token} level, treating deviations from a single reference continuation as errors even when alternative continuations are equally plausible or semantically equivalent (e.g., ``mom'' vs. ``mother''). As a result, token-level loss can penalize valid abstractions, paraphrases, or conceptually correct reasoning paths, biasing models toward surface form rather than underlying meaning. This mismatch between the training signal and semantic correctness motivates learning objectives that operate over higher-level representations. We propose a shift from token-level to concept-level prediction, where concepts group multiple surface forms of the same idea (e.g., ``mom,'' ``mommy,'' ``mother'' $\rightarrow$ \textit{MOTHER}). We introduce various methods for integrating conceptual supervision into LLM training and show that concept-aware models achieve lower perplexity, improved robustness under domain shift, and stronger performance than NTP-based models on diverse NLP benchmarks. This suggests \textit{concept-level supervision} as an improved training signal that better aligns LLMs with human semantic abstractions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11783v1">The Stability Trap: Evaluating the Reliability of LLM-Based Instruction Adherence Auditing</a></div>
    <div class="paper-meta">
      📅 2026-01-16
    </div>
    <details class="paper-abstract">
      The enterprise governance of Generative AI (GenAI) in regulated sectors, such as Human Resources (HR), demands scalable yet reproducible auditing mechanisms. While Large Language Model (LLM)-as-a-Judge approaches offer scalability, their reliability in evaluating adherence of different types of system instructions remains unverified. This study asks: To what extent does the instruction type of an Application Under Test (AUT) influence the stability of judge evaluations? To address this, we introduce the Scoped Instruction Decomposition Framework to classify AUT instructions into Objective and Subjective types, isolating the factors that drive judge instability. We applied this framework to two representative HR GenAI applications, evaluating the stability of four judge architectures over variable runs. Our results reveal a ``Stability Trap'' characterized by a divergence between Verdict Stability and Reasoning Stability. While judges achieved near-perfect verdict agreement ($>99\%$) for both objective and subjective evaluations, their accompanying justification traces diverged significantly. Objective instructions requiring quantitative analysis, such as word counting, exhibited reasoning stability as low as $\approx19\%$, driven by variances in numeric justifications. Similarly, reasoning stability for subjective instructions varied widely ($35\%$--$83\%$) based on evidence granularity, with feature-specific checks failing to reproduce consistent rationale. Conversely, objective instructions focusing on discrete entity extraction achieved high reasoning stability ($>90\%$). These findings demonstrate that high verdict stability can mask fragile reasoning. Thus, we suggest that auditors scope automated evaluation protocols strictly: delegate all deterministically verifiable logic to code, while reserving LLM judges for complex semantic evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11746v1">LIME-LLM: Probing Models with Fluent Counterfactuals, Not Broken Text</a></div>
    <div class="paper-meta">
      📅 2026-01-16
    </div>
    <details class="paper-abstract">
      Local explanation methods such as LIME (Ribeiro et al., 2016) remain fundamental to trustworthy AI, yet their application to NLP is limited by a reliance on random token masking. These heuristic perturbations frequently generate semantically invalid, out-of-distribution inputs that weaken the fidelity of local surrogate models. While recent generative approaches such as LLiMe (Angiulli et al., 2025b) attempt to mitigate this by employing Large Language Models for neighborhood generation, they rely on unconstrained paraphrasing that introduces confounding variables, making it difficult to isolate specific feature contributions. We introduce LIME-LLM, a framework that replaces random noise with hypothesis-driven, controlled perturbations. By enforcing a strict "Single Mask-Single Sample" protocol and employing distinct neutral infill and boundary infill strategies, LIME-LLM constructs fluent, on-manifold neighborhoods that rigorously isolate feature effects. We evaluate our method against established baselines (LIME, SHAP, Integrated Gradients) and the generative LLiMe baseline across three diverse benchmarks: CoLA, SST-2, and HateXplain using human-annotated rationales as ground truth. Empirical results demonstrate that LIME-LLM establishes a new benchmark for black-box NLP explainability, achieving significant improvements in local explanation fidelity compared to both traditional perturbation-based methods and recent generative alternatives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11739v1">Bridging Human Interpretation and Machine Representation: A Landscape of Qualitative Data Analysis in the LLM Era</a></div>
    <div class="paper-meta">
      📅 2026-01-16
    </div>
    <details class="paper-abstract">
      LLMs are increasingly used to support qualitative research, yet existing systems produce outputs that vary widely--from trace-faithful summaries to theory-mediated explanations and system models. To make these differences explicit, we introduce a 4$\times$4 landscape crossing four levels of meaning-making (descriptive, categorical, interpretive, theoretical) with four levels of modeling (static structure, stages/timelines, causal pathways, feedback dynamics). Applying the landscape to prior LLM-based automation highlights a strong skew toward low-level meaning and low-commitment representations, with few reliable attempts at interpretive/theoretical inference or dynamical modeling. Based on the revealed gap, we outline an agenda for applying and building LLM-systems that make their interpretive and modeling commitments explicit, selectable, and governable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.24294v2">LOGOS: LLM-driven End-to-End Grounded Theory Development and Schema Induction for Qualitative Research</a></div>
    <div class="paper-meta">
      📅 2026-01-16
    </div>
    <details class="paper-abstract">
      Grounded theory offers deep insights from qualitative data, but its reliance on expert-intensive manual coding presents a major scalability bottleneck. Existing computational tools either fail on full automation or lack flexible schema construction. We introduce LOGOS, a novel, end-to-end framework that fully automates the grounded theory workflow, transforming raw text into a structured, hierarchical theory. LOGOS integrates LLM-driven coding, semantic clustering, graph reasoning, and a novel iterative refinement process to build highly reusable codebooks. To ensure fair comparison, we also introduce a principled 5-dimensional metric and a train-test split protocol for standardized, unbiased evaluation. Across five diverse corpora, LOGOS consistently outperforms strong baselines and achieves a remarkable average $80.4\%$ alignment with an expert-developed schema on complex datasets. LOGOS demonstrates a potential to democratize and scale qualitative research without sacrificing theoretical nuance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11688v1">SpecMap: Hierarchical LLM Agent for Datasheet-to-Code Traceability Link Recovery in Systems Engineering</a></div>
    <div class="paper-meta">
      📅 2026-01-16
    </div>
    <details class="paper-abstract">
      Establishing precise traceability between embedded systems datasheets and their corresponding code implementations remains a fundamental challenge in systems engineering, particularly for low-level software where manual mapping between specification documents and large code repositories is infeasible. Existing Traceability Link Recovery approaches primarily rely on lexical similarity and information retrieval techniques, which struggle to capture the semantic, structural, and symbol level relationships prevalent in embedded systems software. We present a hierarchical datasheet-to-code mapping methodology that employs large language models for semantic analysis while explicitly structuring the traceability process across multiple abstraction levels. Rather than performing direct specification-to-code matching, the proposed approach progressively narrows the search space through repository-level structure inference, file-level relevance estimation, and fine-grained symbollevel alignment. The method extends beyond function-centric mapping by explicitly covering macros, structs, constants, configuration parameters, and register definitions commonly found in systems-level C/C++ codebases. We evaluate the approach on multiple open-source embedded systems repositories using manually curated datasheet-to-code ground truth. Experimental results show substantial improvements over traditional information-retrieval-based baselines, achieving up to 73.3% file mapping accuracy. We significantly reduce computational overhead, lowering total LLM token consumption by 84% and end-to-end runtime by approximately 80%. This methodology supports automated analysis of large embedded software systems and enables downstream applications such as training data generation for systems-aware machine learning models, standards compliance verification, and large-scale specification coverage analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01461v2">Bridging the gap: A comparative exploration of Speech-LLM and end-to-end architecture for multilingual conversational ASR</a></div>
    <div class="paper-meta">
      📅 2026-01-15
      | 💬 5 pages, 1 figure
    </div>
    <details class="paper-abstract">
      The INTERSPEECH 2025 Challenge on Multilingual Conversational Speech Language Models (MLC-SLM) promotes multilingual conversational ASR with large language models (LLMs). Our previous SHNU-mASR system adopted a competitive parallel-speech-encoder architecture that integrated Whisper and mHuBERT with an LLM. However, it faced two challenges: simple feature concatenation may not fully exploit complementary information, and the performance gap between LLM-based ASR and end-to-end(E2E) encoder-decoder ASR remained unexplored. In this work, we present an enhanced LLM-based ASR framework that combines fine-tuned Whisper and mHuBERT encoders with an LLM to enrich speech representations. We first evaluate E2E Whisper models with LoRA and full fine-tuning on the MLC-SLM ASR task, and then propose cross-attention-based fusion mechanisms for the parallel-speech-encoder. On the official evaluation set of the MLC-SLM Challenge, our system achieves a CER/WER of 10.69%, ranking on par with the top-ranked Track 1 systems, even though it uses only 1,500 hours of baseline training data compared with their large-scale training sets. Nonetheless, we find that our final LLM-based ASR still does not match the performance of a fine-tuned E2E Whisper model, providing valuable empirical guidance for future Speech-LLM design. Our code is publicly available at https://github.com/1535176727/MLC-SLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10156v1">ToolSafe: Enhancing Tool Invocation Safety of LLM-based agents via Proactive Step-level Guardrail and Feedback</a></div>
    <div class="paper-meta">
      📅 2026-01-15
      | 💬 Work in Progress. Code available: https://github.com/MurrayTom/ToolSafe
    </div>
    <details class="paper-abstract">
      While LLM-based agents can interact with environments via invoking external tools, their expanded capabilities also amplify security risks. Monitoring step-level tool invocation behaviors in real time and proactively intervening before unsafe execution is critical for agent deployment, yet remains under-explored. In this work, we first construct TS-Bench, a novel benchmark for step-level tool invocation safety detection in LLM agents. We then develop a guardrail model, TS-Guard, using multi-task reinforcement learning. The model proactively detects unsafe tool invocation actions before execution by reasoning over the interaction history. It assesses request harmfulness and action-attack correlations, producing interpretable and generalizable safety judgments and feedback. Furthermore, we introduce TS-Flow, a guardrail-feedback-driven reasoning framework for LLM agents, which reduces harmful tool invocations of ReAct-style agents by 65 percent on average and improves benign task completion by approximately 10 percent under prompt injection attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05467v3">STELP: Secure Transpilation and Execution of LLM-Generated Programs</a></div>
    <div class="paper-meta">
      📅 2026-01-15
    </div>
    <details class="paper-abstract">
      Rapid evolution of Large Language Models (LLMs) has achieved major advances in reasoning, planning, and function-calling capabilities. Multi-agentic collaborative frameworks using such LLMs place them at the center of solving software development-related tasks such as code generation. However, direct use of LLM generated code in production software development systems is problematic. The code could be unstable or erroneous and contain vulnerabilities such as data poisoning, malicious attacks, and hallucinations that could lead to widespread system malfunctions. This prohibits the adoption of LLM generated code in production AI systems where human code reviews and traditional secure testing tools are impractical or untrustworthy. In this paper, we discuss safety and reliability problems with the execution of LLM generated code and propose a Secure Transpiler and Executor of LLM-Generated Program (STELP), capable of executing LLM-generated code in a controlled and safe manner. STELP secures autonomous production AI systems involving code generation, filling the critical void left by the impracticality or limitations of traditional secure testing methodologies and human oversight. This includes applications such as headless code generation-execution and LLMs that produce executable code snippets as an action plan to be executed in real time. We contribute a human-validated dataset of insecure code snippets and benchmark our approach on publicly available datasets for correctness, safety, and latency. Our results demonstrate that our approach outperforms an existing method by a significant margin, particularly in its ability to safely execute risky code snippets. Warning: This paper contains malicious code snippets that should be run with caution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10141v1">Understanding and Preserving Safety in Fine-Tuned LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-15
    </div>
    <details class="paper-abstract">
      Fine-tuning is an essential and pervasive functionality for applying large language models (LLMs) to downstream tasks. However, it has the potential to substantially degrade safety alignment, e.g., by greatly increasing susceptibility to jailbreak attacks, even when the fine-tuning data is entirely harmless. Despite garnering growing attention in defense efforts during the fine-tuning stage, existing methods struggle with a persistent safety-utility dilemma: emphasizing safety compromises task performance, whereas prioritizing utility typically requires deep fine-tuning that inevitably leads to steep safety declination. In this work, we address this dilemma by shedding new light on the geometric interaction between safety- and utility-oriented gradients in safety-aligned LLMs. Through systematic empirical analysis, we uncover three key insights: (I) safety gradients lie in a low-rank subspace, while utility gradients span a broader high-dimensional space; (II) these subspaces are often negatively correlated, causing directional conflicts during fine-tuning; and (III) the dominant safety direction can be efficiently estimated from a single sample. Building upon these novel insights, we propose safety-preserving fine-tuning (SPF), a lightweight approach that explicitly removes gradient components conflicting with the low-rank safety subspace. Theoretically, we show that SPF guarantees utility convergence while bounding safety drift. Empirically, SPF consistently maintains downstream task performance and recovers nearly all pre-trained safety alignment, even under adversarial fine-tuning scenarios. Furthermore, SPF exhibits robust resistance to both deep fine-tuning and dynamic jailbreak attacks. Together, our findings provide new mechanistic understanding and practical guidance toward always-aligned LLM fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10132v1">Is More Context Always Better? Examining LLM Reasoning Capability for Time Interval Prediction</a></div>
    <div class="paper-meta">
      📅 2026-01-15
      | 💬 Accepted at The Web Conference 2026 (WWW 2026)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive capabilities in reasoning and prediction across different domains. Yet, their ability to infer temporal regularities from structured behavioral data remains underexplored. This paper presents a systematic study investigating whether LLMs can predict time intervals between recurring user actions, such as repeated purchases, and how different levels of contextual information shape their predictive behavior. Using a simple but representative repurchase scenario, we benchmark state-of-the-art LLMs in zero-shot settings against both statistical and machine-learning models. Two key findings emerge. First, while LLMs surpass lightweight statistical baselines, they consistently underperform dedicated machine-learning models, showing their limited ability to capture quantitative temporal structure. Second, although moderate context can improve LLM accuracy, adding further user-level detail degrades performance. These results challenge the assumption that "more context leads to better reasoning". Our study highlights fundamental limitations of today's LLMs in structured temporal inference and offers guidance for designing future context-aware hybrid models that integrate statistical precision with linguistic flexibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10128v1">A Generalizable Framework for Building Executable Domain-Specific LLMs under Data Scarcity: Demonstration on Semiconductor TCAD Simulation</a></div>
    <div class="paper-meta">
      📅 2026-01-15
      | 💬 Submitted to Nature Computational Science
    </div>
    <details class="paper-abstract">
      Scientific and engineering verticals often suffer from data scarcity and strict executability requirements: models must generate not only fluent text, but also syntactically valid, tool-compilable scripts. We present a schema-first alignment framework for building compact, executable domain-specific LLMs in low-resource settings. The framework integrates three core components: (i) large-scale synthetic QA data generation from expert documentation to instill foundational domain knowledge; (ii) a code-centric IR->DPO workflow that converts verified tool decks into interpretable intermediate representations (IR), performs equivalence-preserving diversification, and constructs preference pairs to directly optimize instruction compliance and code executability; and (iii) a controlled evaluation of Retrieval-Augmented Generation (RAG), showing that while RAG benefits general LLMs, it can marginally degrade the performance of already domain-aligned models. We demonstrate the framework by instantiating TcadGPT for semiconductor Technology Computer-Aided Design (TCAD). Using 1.5M synthetic QA pairs and an IR-driven DPO dataset, TcadGPT attains 85.6% semantic accuracy and an 80.0% syntax pass rate on SDE executability tests, substantially outperforming state-of-the-art general LLMs such as GPT-4o. To probe portability beyond TCAD, we apply the same recipe to the open-source FEM solver Elmer, observing consistent improvements in script-level success rates over general-purpose baselines. All datasets, benchmarks, and code (including P1, P2, and IR->DPO) are released for reproducibility. Together, these results suggest that the proposed framework provides a robust and reproducible path toward executable LLMs in specialized, data-scarce professional domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10114v1">Following the Teacher's Footsteps: Scheduled Checkpoint Distillation for Domain-Specific LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-15
      | 💬 15 pages, submitted to ICPR 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are challenging to deploy for domain-specific tasks due to their massive scale. While distilling a fine-tuned LLM into a smaller student model is a promising alternative, the capacity gap between teacher and student often leads to suboptimal performance. This raises a key question: when and how can a student model match or even surpass its teacher on domain-specific tasks? In this work, we propose a novel theoretical insight: a student can outperform its teacher if its advantage on a Student-Favored Subdomain (SFS) outweighs its deficit on the Teacher-Favored Subdomain (TFS). Guided by this insight, we propose Scheduled Checkpoint Distillation (SCD), which reduces the TFS deficit by emulating the teacher's convergence process during supervised fine-tuning (SFT) on the domain task, and a sample-wise Adaptive Weighting (AW) mechanism to preserve student strengths on SFS. Experiments across diverse domain tasks--including QA, NER, and text classification in multiple languages--show that our method consistently outperforms existing distillation approaches, allowing the student model to match or even exceed the performance of its fine-tuned teacher.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10112v1">Repository Intelligence Graph: Deterministic Architectural Map for LLM Code Assistants</a></div>
    <div class="paper-meta">
      📅 2026-01-15
      | 💬 35 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Repository aware coding agents often struggle to recover build and test structure, especially in multilingual projects where cross language dependencies are encoded across heterogeneous build systems and tooling. We introduce the Repository Intelligence Graph (RIG), a deterministic, evidence backed architectural map that represents buildable components, aggregators, runners, tests, external packages, and package managers, connected by explicit dependency and coverage edges that trace back to concrete build and test definitions. We also present SPADE, a deterministic extractor that constructs RIG from build and test artifacts (currently with an automatic CMake plugin based on the CMake File API and CTest metadata), and exposes RIG as an LLM friendly JSON view that agents can treat as the authoritative description of repository structure. We evaluate three commercial agents (Claude Code, Cursor, Codex) on eight repositories spanning low to high build oriented complexity, including the real world MetaFFI project. Each agent answers thirty structured questions per repository with and without RIG in context, and we measure accuracy, wall clock completion time, and efficiency (seconds per correct answer). Across repositories and agents, providing RIG improves mean accuracy by 12.2\% and reduces completion time by 53.9\%, yielding a mean 57.8\% reduction in seconds per correct answer. Gains are larger in multilingual repositories, which improve by 17.7\% in accuracy and 69.5\% in efficiency on average, compared to 6.6\% and 46.1\% in single language repositories. Qualitative analysis suggests that RIG shifts failures from structural misunderstandings toward reasoning mistakes over a correct structure, while rare regressions highlight that graph based reasoning quality remains a key factor.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10102v1">When Personas Override Payoffs: Role Identity Bias in Multi-Agent LLM Decision-Making</a></div>
    <div class="paper-meta">
      📅 2026-01-15
    </div>
    <details class="paper-abstract">
      Large language models are increasingly deployed in multi-agent systems for strategic tasks, yet how design choices such as role-based personas and payoff visibility affect reasoning remains poorly understood. We investigate whether multi-agent systems function as strategic reasoners capable of payoff optimization or as identity-driven actors that prioritize role alignment over explicit incentives. Using Nash equilibrium achievement as a diagnostic for strategic reasoning, we conduct systematic experiments across four LLM architectures (Qwen-7B, Qwen-32B, Llama-8B, Mistral-7B) in complex environmental decision-making games involving four agents. We show that role identity bias fundamentally alters strategic reasoning even when payoff-optimal equilibria exist and complete payoff information is available. Removing personas and providing explicit payoffs enables Qwen models to achieve high Nash equilibrium rates, indicating that both conditions are necessary for strategic reasoning. In contrast, personas systematically bias equilibrium selection toward socially preferred outcomes: with personas present, all of the achieved equilibria correspond to Green Transition, while models entirely fail to reach equilibrium when Tragedy of the Commons is payoff-optimal. The effect of explicit payoffs depends entirely on persona presence, revealing strong interactions between representational design choices. We also observe clear model-dependent patterns. Qwen architectures are highly sensitive to both personas and payoff visibility, whereas Llama and Mistral exhibit rigid reasoning behavior across conditions. These findings demonstrate that representational choices are substantive governance decisions that determine whether multi-agent systems act as strategic reasoners or identity-driven actors, with important implications for real-world deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10079v1">Sparse-RL: Breaking the Memory Wall in LLM Reinforcement Learning via Stable Sparse Rollouts</a></div>
    <div class="paper-meta">
      📅 2026-01-15
    </div>
    <details class="paper-abstract">
      Reinforcement Learning (RL) has become essential for eliciting complex reasoning capabilities in Large Language Models (LLMs). However, the substantial memory overhead of storing Key-Value (KV) caches during long-horizon rollouts acts as a critical bottleneck, often prohibiting efficient training on limited hardware. While existing KV compression techniques offer a remedy for inference, directly applying them to RL training induces a severe policy mismatch, leading to catastrophic performance collapse. To address this, we introduce Sparse-RL empowers stable RL training under sparse rollouts. We show that instability arises from a fundamental policy mismatch among the dense old policy, the sparse sampler policy, and the learner policy. To mitigate this issue, Sparse-RL incorporates Sparsity-Aware Rejection Sampling and Importance-based Reweighting to correct the off-policy bias introduced by compression-induced information loss. Experimental results show that Sparse-RL reduces rollout overhead compared to dense baselines while preserving the performance. Furthermore, Sparse-RL inherently implements sparsity-aware training, significantly enhancing model robustness during sparse inference deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05529v2">Safety Not Found (404): Hidden Risks of LLM-Based Robotics Decision Making</a></div>
    <div class="paper-meta">
      📅 2026-01-15
    </div>
    <details class="paper-abstract">
      One mistake by an AI system in a safety-critical setting can cost lives. As Large Language Models (LLMs) become integral to robotics decision-making, the physical dimension of risk grows; a single wrong instruction can directly endanger human safety. This paper addresses the urgent need to systematically evaluate LLM performance in scenarios where even minor errors are catastrophic. Through a qualitative evaluation of a fire evacuation scenario, we identified critical failure cases in LLM-based decision-making. Based on these, we designed seven tasks for quantitative assessment, categorized into: Complete Information, Incomplete Information, and Safety-Oriented Spatial Reasoning (SOSR). Complete information tasks utilize ASCII maps to minimize interpretation ambiguity and isolate spatial reasoning from visual processing. Incomplete information tasks require models to infer missing context, testing for spatial continuity versus hallucinations. SOSR tasks use natural language to evaluate safe decision-making in life-threatening contexts. We benchmark various LLMs and Vision-Language Models (VLMs) across these tasks. Beyond aggregate performance, we analyze the implications of a 1% failure rate, highlighting how "rare" errors escalate into catastrophic outcomes. Results reveal serious vulnerabilities: several models achieved a 0% success rate in ASCII navigation, while in a simulated fire drill, models instructed robots to move toward hazardous areas instead of emergency exits. Our findings lead to a sobering conclusion: current LLMs are not ready for direct deployment in safety-critical systems. A 99% accuracy rate is dangerously misleading in robotics, as it implies one out of every hundred executions could result in catastrophic harm. We demonstrate that even state-of-the-art models cannot guarantee safety, and absolute reliance on them creates unacceptable risks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.02097v4">JudgeAgent: Beyond Static Benchmarks for Knowledge-Driven and Dynamic LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-01-15
    </div>
    <details class="paper-abstract">
      Current evaluation methods for large language models (LLMs) primarily rely on static benchmarks, presenting two major challenges: limited knowledge coverage and fixed difficulties that mismatch with the evaluated LLMs. These limitations lead to superficial assessments of LLM knowledge, thereby impeding the targeted model optimizations. To bridge this gap, we propose JudgeAgent, a knowledge-driven and dynamic evaluation framework for LLMs. To address the challenge of limited knowledge coverage, JudgeAgent leverages LLM agents equipped with context graphs to traverse knowledge structures systematically for question generation. Furthermore, to mitigate data contamination and difficulty mismatch, it adopts a difficulty-adaptive and multi-turn interview mechanism. Thereby, JudgeAgent can achieve comprehensive evaluations and facilitate more effective improvement of LLMs. Empirical results demonstrate that JudgeAgent enables more comprehensive evaluations and facilitates effective model iterations, highlighting the potential of this knowledge-driven and dynamic evaluation paradigm. The source code is available on https://github.com/DataArcTech/JudgeAgent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10025v1">Structured Personality Control and Adaptation for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly shaping human-computer interaction (HCI), from personalized assistants to social simulations. Beyond language competence, researchers are exploring whether LLMs can exhibit human-like characteristics that influence engagement, decision-making, and perceived realism. Personality, in particular, is critical, yet existing approaches often struggle to achieve both nuanced and adaptable expression. We present a framework that models LLM personality via Jungian psychological types, integrating three mechanisms: a dominant-auxiliary coordination mechanism for coherent core expression, a reinforcement-compensation mechanism for temporary adaptation to context, and a reflection mechanism that drives long-term personality evolution. This design allows the agent to maintain nuanced traits while dynamically adjusting to interaction demands and gradually updating its underlying structure. Personality alignment is evaluated using Myers-Briggs Type Indicator questionnaires and tested under diverse challenge scenarios as a preliminary structured assessment. Findings suggest that evolving, personality-aware LLMs can support coherent, context-sensitive interactions, enabling naturalistic agent design in HCI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.21817v4">Out of Distribution, Out of Luck: How Well Can LLMs Trained on Vulnerability Datasets Detect Top 25 CWE Weaknesses?</a></div>
    <div class="paper-meta">
      📅 2026-01-15
      | 💬 Accepted for publication at ICSE 2026
    </div>
    <details class="paper-abstract">
      Automated vulnerability detection research has made substantial progress, yet its real-world impact remains limited. Prior work found that current vulnerability datasets suffer from issues including label inaccuracy rates of 20%-71%, extensive duplication, and poor coverage of critical Common Weakness Enumeration (CWE). These issues create a significant generalization gap where models achieve misleading In-Distribution (ID) accuracies (testing on splits from the same dataset) by exploiting spurious correlations rather than learning true vulnerability patterns. To address these limitations, we present a three-part solution. First, we introduce BenchVul, which is a manually curated and balanced test dataset covering the MITRE Top 25 Most Dangerous CWEs, to enable fair model evaluation. Second, we construct a high-quality training dataset, TitanVul, comprising 38,548 functions by aggregating seven public sources and applying deduplication and validation using a novel multi-agent LLM pipeline. Third, we propose a Realistic Vulnerability Generation (RVG) pipeline, which synthesizes context-aware vulnerability examples for underrepresented but critical CWE types through simulated development workflows. Our evaluation reveals that In-Distribution (ID) performance does not reliably predict Out-of-Distribution (OOD) performance on BenchVul. For example, a model trained on BigVul achieves the highest 0.703 ID accuracy but fails on BenchVul's real-world samples (0.493 OOD accuracy). Conversely, a model trained on our TitanVul achieves the highest OOD performance on both the real-world (0.881) and synthesized (0.785) portions of BenchVul, improving upon the next-best performing dataset by 5.3% and 11.8% respectively, despite a modest ID score (0.590). Augmenting TitanVul with our RVG further boosts this leading OOD performance, improving accuracy on real-world data by 5.8% (to 0.932).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.20412v3">MindGuard: Intrinsic Decision Inspection for Securing LLM Agents Against Metadata Poisoning</a></div>
    <div class="paper-meta">
      📅 2026-01-15
    </div>
    <details class="paper-abstract">
      The Model Context Protocol (MCP) is increasingly adopted to standardize the interaction between LLM agents and external tools. However, this trend introduces a new threat: Tool Poisoning Attacks (TPA), where tool metadata is poisoned to induce the agent to perform unauthorized operations. Existing defenses that primarily focus on behavior-level analysis are fundamentally ineffective against TPA, as poisoned tools need not be executed, leaving no behavioral trace to monitor. Thus, we propose MindGuard, a decision-level guardrail for LLM agents, providing provenance tracking of call decisions, policy-agnostic detection, and poisoning source attribution against TPA. While fully explaining LLM decision remains challenging, our empirical findings uncover a strong correlation between LLM attention mechanisms and tool invocation decisions. Therefore, we choose attention as an empirical signal for decision tracking and formalize this as the Decision Dependence Graph (DDG), which models the LLM's reasoning process as a weighted, directed graph where vertices represent logical concepts and edges quantify the attention-based dependencies. We further design robust DDG construction and graph-based anomaly analysis mechanisms that efficiently detect and attribute TPA attacks. Extensive experiments on real-world datasets demonstrate that MindGuard achieves 94\%-99\% average precision in detecting poisoned invocations, 95\%-100\% attribution accuracy, with processing times under one second and no additional token cost. Moreover, DDG can be viewed as an adaptation of the classical Program Dependence Graph (PDG), providing a solid foundation for applying traditional security policies at the decision level.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10004v1">SoK: Privacy-aware LLM in Healthcare: Threat Model, Privacy Techniques, Challenges and Recommendations</a></div>
    <div class="paper-meta">
      📅 2026-01-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly adopted in healthcare to support clinical decision-making, summarize electronic health records (EHRs), and enhance patient care. However, this integration introduces significant privacy and security challenges, driven by the sensitivity of clinical data and the high-stakes nature of medical workflows. These risks become even more pronounced across heterogeneous deployment environments, ranging from small on-premise hospital systems to regional health networks, each with unique resource limitations and regulatory demands. This Systematization of Knowledge (SoK) examines the evolving threat landscape across the three core LLM phases: Data preprocessing, Fine-tuning, and Inference within realistic healthcare settings. We present a detailed threat model that characterizes adversaries, capabilities, and attack surfaces at each phase, and we systematize how existing privacy-preserving techniques (PPTs) attempt to mitigate these vulnerabilities. While existing defenses show promise, our analysis identifies persistent limitations in securing sensitive clinical data across diverse operational tiers. We conclude with phase-aware recommendations and future research directions aimed at strengthening privacy guarantees for LLMs in regulated environments. This work provides a foundation for understanding the intersection of LLMs, threats, and privacy in healthcare, offering a roadmap toward more robust and clinically trustworthy AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.07115v5">Online Scheduling for LLM Inference with KV Cache Constraints</a></div>
    <div class="paper-meta">
      📅 2026-01-15
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) inference, where a trained model generates text one word at a time in response to user prompts, is a computationally intensive process requiring efficient scheduling to optimize latency and resource utilization. A key challenge in LLM inference is the management of the Key-Value (KV) cache, which reduces redundant computations but introduces memory constraints. In this work, we model LLM inference with KV cache constraints theoretically and propose a novel batching and scheduling algorithm that minimizes inference latency while effectively managing the KV cache's memory. More specifically, we make the following contributions. First, to evaluate the performance of online algorithms for scheduling in LLM inference, we introduce a hindsight optimal benchmark, formulated as an integer program that computes the minimum total inference latency under full future information. Second, we prove that no deterministic online algorithm can achieve a constant competitive ratio when the arrival process is arbitrary. Third, motivated by the computational intractability of solving the integer program at scale, we propose a polynomial-time online scheduling algorithm and show that under certain conditions it can achieve a constant competitive ratio. We also demonstrate our algorithm's strong empirical performance by comparing it to the hindsight optimal in a synthetic dataset. Finally, we conduct empirical evaluations on a real-world public LLM inference dataset, simulating the Llama2-70B model on A100 GPUs, and show that our algorithm significantly outperforms the benchmark algorithms. Overall, our results offer a path toward more sustainable and cost-effective LLM deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09971v1">An Exploratory Study to Repurpose LLMs to a Unified Architecture for Time Series Classification</a></div>
    <div class="paper-meta">
      📅 2026-01-15
    </div>
    <details class="paper-abstract">
      Time series classification (TSC) is a core machine learning problem with broad applications. Recently there has been growing interest in repurposing large language models (LLMs) for TSC, motivated by their strong reasoning and generalization ability. Prior work has primarily focused on alignment strategies that explicitly map time series data into the textual domain; however, the choice of time series encoder architecture remains underexplored. In this work, we conduct an exploratory study of hybrid architectures that combine specialized time series encoders with a frozen LLM backbone. We evaluate a diverse set of encoder families, including Inception, convolutional, residual, transformer-based, and multilayer perceptron architectures, among which the Inception model is the only encoder architecture that consistently yields positive performance gains when integrated with an LLM backbone. Overall, this study highlights the impact of time series encoder choice in hybrid LLM architectures and points to Inception-based models as a promising direction for future LLM-driven time series learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09953v1">Take Out Your Calculators: Estimating the Real Difficulty of Question Items with LLM Student Simulations</a></div>
    <div class="paper-meta">
      📅 2026-01-15
    </div>
    <details class="paper-abstract">
      Standardized math assessments require expensive human pilot studies to establish the difficulty of test items. We investigate the predictive value of open-source large language models (LLMs) for evaluating the difficulty of multiple-choice math questions for real-world students. We show that, while LLMs are poor direct judges of problem difficulty, simulation-based approaches with LLMs yield promising results under the right conditions. Under the proposed approach, we simulate a "classroom" of 4th, 8th, or 12th grade students by prompting the LLM to role-play students of varying proficiency levels. We use the outcomes of these simulations to fit Item Response Theory (IRT) models, comparing learned difficulty parameters for items to their real-world difficulties, as determined by item-level statistics furnished by the National Assessment of Educational Progress (NAEP). We observe correlations as high as 0.75, 0.76, and 0.82 for grades 4, 8, and 12, respectively. In our simulations, we experiment with different "classroom sizes," showing tradeoffs between computation size and accuracy. We find that role-plays with named students improves predictions (compared to student ids), and stratifying names across gender and race further improves predictions. Our results show that LLMs with relatively weaker mathematical abilities (Gemma) actually yield better real-world difficulty predictions than mathematically stronger models (Llama and Qwen), further underscoring the suitability of open-source models for the task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17639v2">Linear Personality Probing and Steering in LLMs: A Big Five Study</a></div>
    <div class="paper-meta">
      📅 2026-01-15
      | 💬 29 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit distinct and consistent personalities that greatly impact trust and engagement. While this means that personality frameworks would be highly valuable tools to characterize and control LLMs' behavior, current approaches remain either costly (post-training) or brittle (prompt engineering). Probing and steering via linear directions has recently emerged as a cheap and efficient alternative. In this paper, we investigate whether linear directions aligned with the Big Five personality traits can be used for probing and steering model behavior. Using Llama 3.3 70B, we generate descriptions of 406 fictional characters and their Big Five trait scores. We then prompt the model with these descriptions and questions from the Alpaca questionnaire, allowing us to sample hidden activations that vary along personality traits in known, quantifiable ways. Using linear regression, we learn a set of per-layer directions in activation space, and test their effectiveness for probing and steering model behavior. Our results suggest that linear directions aligned with trait-scores are effective probes for personality detection, while their steering capabilities strongly depend on context, producing reliable effects in forced-choice tasks but limited influence in open-ended generation or when additional context is present in the prompt.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10896v1">DialDefer: A Framework for Detecting and Mitigating LLM Dialogic Deference</a></div>
    <div class="paper-meta">
      📅 2026-01-15
      | 💬 10 pages main content, 7 figures, 35 pages total with appendix
    </div>
    <details class="paper-abstract">
      LLMs are increasingly used as third-party judges, yet their reliability when evaluating speakers in dialogue remains poorly understood. We show that LLMs judge identical claims differently depending on framing: the same content elicits different verdicts when presented as a statement to verify ("Is this statement correct?") versus attributed to a speaker ("Is this speaker correct?"). We call this dialogic deference and introduce DialDefer, a framework for detecting and mitigating these framing-induced judgment shifts. Our Dialogic Deference Score (DDS) captures directional shifts that aggregate accuracy obscures. Across nine domains, 3k+ instances, and four models, conversational framing induces large shifts (|DDS| up to 87pp, p < .0001) while accuracy remains stable (<2pp), with effects amplifying 2-4x on naturalistic Reddit conversations. Models can shift toward agreement (deference) or disagreement (skepticism) depending on domain -- the same model ranges from DDS = -53 on graduate-level science to +58 on social judgment. Ablations reveal that human-vs-LLM attribution drives the largest shifts (17.7pp swing), suggesting models treat disagreement with humans as more costly than with AI. Mitigation attempts reduce deference but can over-correct into skepticism, framing this as a calibration problem beyond accuracy optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.07128v3">DeepSeek-R1 Thoughtology: Let's think about LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-15
      | 💬 135 pages, Published to TMLR
    </div>
    <details class="paper-abstract">
      Large Reasoning Models like DeepSeek-R1 mark a fundamental shift in how LLMs approach complex problems. Instead of directly producing an answer for a given input, DeepSeek-R1 creates detailed multi-step reasoning chains, seemingly "thinking" about a problem before providing an answer. This reasoning process is publicly available to the user, creating endless opportunities for studying the reasoning behaviour of the model and opening up the field of Thoughtology. Starting from a taxonomy of DeepSeek-R1's basic building blocks of reasoning, our analyses on DeepSeek-R1 investigate the impact and controllability of thought length, management of long or confusing contexts, cultural and safety concerns, and the status of DeepSeek-R1 vis-à-vis cognitive phenomena, such as human-like language processing and world modelling. Our findings paint a nuanced picture. Notably, we show DeepSeek-R1 has a 'sweet spot' of reasoning, where extra inference time can impair model performance. Furthermore, we find a tendency for DeepSeek-R1 to persistently ruminate on previously explored problem formulations, obstructing further exploration. We also note strong safety vulnerabilities of DeepSeek-R1 compared to its non-reasoning counterpart, which can also compromise safety-aligned LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11881v3">Better LLM Reasoning via Dual-Play</a></div>
    <div class="paper-meta">
      📅 2026-01-15
      | 💬 33 pages, 17 figures, 17 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable progress through Reinforcement Learning with Verifiable Rewards (RLVR), yet still rely heavily on external supervision (e.g., curated labels). Adversarial learning, particularly through self-play, offers a promising alternative that enables models to iteratively learn from themselves - thus reducing reliance on external supervision. Dual-play extends adversarial learning by assigning specialized roles to two models and training them against each other, fostering sustained competition and mutual evolution. Despite its promise, adapting dual-play training to LLMs remains limited, largely due to their susceptibility to reward hacking and training instability. In this paper, we introduce PasoDoble, a novel LLM dual-play framework. PasoDoble adversarially trains two models initialized from the same base model: a Proposer, which generates challenging questions with ground-truth answers, and a Solver, which attempts to solve them. We enrich the Proposer with knowledge from a pre-training dataset to ensure the questions' quality and diversity. To avoid reward hacking, the Proposer is rewarded for producing only valid questions that push the Solver's limit, while the Solver is rewarded for solving them correctly, and both are updated jointly. To further enhance training stability, we introduce an optional offline paradigm that decouples Proposer and Solver updates, alternately updating each for several steps while holding the other fixed. Notably, PasoDoble operates without supervision during training. Experimental results show that PasoDoble can improve the reasoning performance of LLMs. Our project page is available at https://hcy123902.github.io/PasoDoble.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10823v1">Mugi: Value Level Parallelism For Efficient LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-15
      | 💬 2026 International Conference on Architectural Support for Programming Languages and Operating Systems
    </div>
    <details class="paper-abstract">
      Value level parallelism (VLP) has been proposed to improve the efficiency of large-batch, low-precision general matrix multiply (GEMM) between symmetric activations and weights. In transformer based large language models (LLMs), there exist more sophisticated operations beyond activation-weight GEMM. In this paper, we explore how VLP benefits LLMs. First, we generalize VLP for nonlinear approximations, outperforming existing nonlinear approximations in end-to-end LLM accuracy, performance, and efficiency. Our VLP approximation follows a value-centric approach, where important values are assigned with greater accuracy. Second, we optimize VLP for small-batch GEMMs with asymmetric inputs efficiently, which leverages timely LLM optimizations, including weight-only quantization, key-value (KV) cache quantization, and group query attention. Finally, we design a new VLP architecture, Mugi, to encapsulate the innovations above and support full LLM workloads, while providing better performance, efficiency and sustainability. Our experimental results show that Mugi can offer significant improvements on throughput and energy efficiency, up to $45\times$ and $668\times$ for nonlinear softmax operations, and $2.07\times$ and $3.11\times$ for LLMs, and also decrease operational carbon for LLM operation by $1.45\times$ and embodied carbon by $1.48\times$.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10820v1">Towards Reliable ML Feature Engineering via Planning in Constrained-Topology of LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-15
    </div>
    <details class="paper-abstract">
      Recent advances in code generation models have unlocked unprecedented opportunities for automating feature engineering, yet their adoption in real-world ML teams remains constrained by critical challenges: (i) the scarcity of datasets capturing the iterative and complex coding processes of production-level feature engineering, (ii) limited integration and personalization of widely used coding agents, such as CoPilot and Devin, with a team's unique tools, codebases, workflows, and practices, and (iii) suboptimal human-AI collaboration due to poorly timed or insufficient feedback. We address these challenges with a planner-guided, constrained-topology multi-agent framework that generates code for repositories in a multi-step fashion. The LLM-powered planner leverages a team's environment, represented as a graph, to orchestrate calls to available agents, generate context-aware prompts, and use downstream failures to retroactively correct upstream artifacts. It can request human intervention at critical steps, ensuring generated code is reliable, maintainable, and aligned with team expectations. On a novel in-house dataset, our approach achieves 38% and 150% improvement in the evaluation metric over manually crafted and unplanned workflows respectively. In practice, when building features for recommendation models serving over 120 million users, our approach has delivered real-world impact by reducing feature engineering cycles from three weeks to a single day.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10804v1">BYOL: Bring Your Own Language Into LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit strong multilingual capabilities, yet remain fundamentally constrained by the severe imbalance in global language resources. While over 7,000 languages are spoken worldwide, only a small subset (fewer than 100) has sufficient digital presence to meaningfully influence modern LLM training. This disparity leads to systematic underperformance, cultural misalignment, and limited accessibility for speakers of low-resource and extreme-low-resource languages. To address this gap, we introduce Bring Your Own Language (BYOL), a unified framework for scalable, language-aware LLM development tailored to each language's digital footprint. BYOL begins with a language resource classification that maps languages into four tiers (Extreme-Low, Low, Mid, High) using curated web-scale corpora, and uses this classification to select the appropriate integration pathway. For low-resource languages, we propose a full-stack data refinement and expansion pipeline that combines corpus cleaning, synthetic text generation, continual pretraining, and supervised finetuning. Applied to Chichewa and Maori, this pipeline yields language-specific LLMs that achieve approximately 12 percent average improvement over strong multilingual baselines across 12 benchmarks, while preserving English and multilingual capabilities via weight-space model merging. For extreme-low-resource languages, we introduce a translation-mediated inclusion pathway, and show on Inuktitut that a tailored machine translation system improves over a commercial baseline by 4 BLEU, enabling high-accuracy LLM access when direct language modeling is infeasible. Finally, we release human-translated versions of the Global MMLU-Lite benchmark in Chichewa, Maori, and Inuktitut, and make our codebase and models publicly available at https://github.com/microsoft/byol .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09710v2">Echoing: Identity Failures when LLM Agents Talk to Each Other</a></div>
    <div class="paper-meta">
      📅 2026-01-15
    </div>
    <details class="paper-abstract">
      As large language model (LLM) based agents interact autonomously with one another, a new class of failures emerges that cannot be predicted from single agent performance: behavioral drifts in agent-agent conversations (AxA). Unlike human-agent interactions, where humans ground and steer conversations, AxA lacks such stabilizing signals, making these failures unique. We investigate one such failure, echoing, where agents abandon their assigned roles and instead mirror their conversational partners, undermining their intended objectives. Through experiments across $66$ AxA configurations, $4$ domains (3 transactional, 1 advisory), and $2500+$ conversations (over $250000$ LLM inferences), we show that echoing occurs across major LLM providers, with echoing rates as high as $70\%$ depending on the model and domain. Moreover, we find that echoing is persistent even in advanced reasoning models with substantial rates ($32.8\%$) that are not reduced by reasoning efforts. We analyze prompt, conversation dynamics, showing that echoing arises as interaction grows longer ($7+$ agent turns) and is not merely an artifact of sub-optimal experiment design. Finally, we introduce a protocol-level mitigation where targeted use of structured response reduces echoing to $9\%$.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10775v1">LLMs for Game Theory: Entropy-Guided In-Context Learning and Adaptive CoT Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-15
      | 💬 Accepted at AAAI 2026 Bridge (Logical and Symbolic Reasoning in Language Models)
    </div>
    <details class="paper-abstract">
      We propose a novel LLM-based framework for reasoning in discrete, game-theoretic tasks, illustrated with \emph{Tic-Tac-Toe}. The method integrates in-context learning with entropy-guided chain-of-thought (CoT) reasoning and adaptive context retrieval. The model dynamically adjusts both the number of retrieved examples and reasoning paths according to token-level uncertainty: concise reasoning with minimal context is used when uncertainty is low, whereas higher uncertainty triggers expanded multi-path CoT exploration. Experimental evaluation against a sub-optimal algorithmic opponent shows that entropy-aware adaptive reasoning substantially improves decision quality, increasing the average game outcome from \(-11.6\%\) with the baseline LLM to \(+9.5\%\) with entropy-guided adaptive reasoning over 100 games (win = +1, tie = 0, loss = -1), while maintaining a relatively low number of LLM queries per game. Statistical validation confirms that the improvement is significant, and correlation analysis reveals a negative association between token-level entropy and move optimality. These findings demonstrate that uncertainty-guided adaptive reasoning effectively enhances LLM performance in sequential decision-making environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10700v1">LIBERTy: A Causal Framework for Benchmarking Concept-Based Explanations of LLMs with Structural Counterfactuals</a></div>
    <div class="paper-meta">
      📅 2026-01-15
    </div>
    <details class="paper-abstract">
      Concept-based explanations quantify how high-level concepts (e.g., gender or experience) influence model behavior, which is crucial for decision-makers in high-stakes domains. Recent work evaluates the faithfulness of such explanations by comparing them to reference causal effects estimated from counterfactuals. In practice, existing benchmarks rely on costly human-written counterfactuals that serve as an imperfect proxy. To address this, we introduce a framework for constructing datasets containing structural counterfactual pairs: LIBERTy (LLM-based Interventional Benchmark for Explainability with Reference Targets). LIBERTy is grounded in explicitly defined Structured Causal Models (SCMs) of the text generation, interventions on a concept propagate through the SCM until an LLM generates the counterfactual. We introduce three datasets (disease detection, CV screening, and workplace violence prediction) together with a new evaluation metric, order-faithfulness. Using them, we evaluate a wide range of methods across five models and identify substantial headroom for improving concept-based explanations. LIBERTy also enables systematic analysis of model sensitivity to interventions: we find that proprietary LLMs show markedly reduced sensitivity to demographic concepts, likely due to post-training mitigation. Overall, LIBERTy provides a much-needed benchmark for developing faithful explainability methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.16846v3">BASIL: Bayesian Assessment of Sycophancy in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-15
    </div>
    <details class="paper-abstract">
      Sycophancy (overly agreeable or flattering behavior) poses a fundamental challenge for human-AI collaboration, particularly in high-stakes decision-making domains such as health, law, and education. A central difficulty in studying sycophancy in large language models (LLMs) is disentangling sycophantic belief shifts from rational changes in behavior driven by new evidence or user-provided information. Existing approaches either measure descriptive behavior changes or apply normative evaluations that rely on objective ground truth, limiting their applicability to subjective or uncertain tasks. We introduce a Bayesian probabilistic framework, grounded in behavioral economics and rational decision theory, that explicitly separates sycophancy from rational belief updating. Within this framework, we achieve three objectives: (i) a descriptive metric that measures sycophancy while controlling for rational responses to evidence; (ii) a normative metric that quantifies how sycophancy leads models astray from Bayesian-consistent belief updating; and (iii) the ability to apply both metrics in settings without ground-truth labels. Applying our framework across multiple LLMs and three uncertainty-driven tasks, we find robust evidence of sycophantic belief shifts and show that their impact on rationality depends on whether models systematically over- or under-update their beliefs. Finally, we demonstrate that a post-hoc calibration method and two fine-tuning strategies (SFT and DPO) substantially reduce Bayesian inconsistency, with particularly strong improvements under explicit sycophancy prompting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10645v1">Influential Training Data Retrieval for Explaining Verbalized Confidence of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-15
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can increase users' perceived trust by verbalizing confidence in their outputs. However, prior work has shown that LLMs are often overconfident, making their stated confidence unreliable since it does not consistently align with factual accuracy. To better understand the sources of this verbalized confidence, we introduce TracVC (\textbf{Trac}ing \textbf{V}erbalized \textbf{C}onfidence), a method that builds on information retrieval and influence estimation to trace generated confidence expressions back to the training data. We evaluate TracVC on OLMo and Llama models in a question answering setting, proposing a new metric, content groundness, which measures the extent to which an LLM grounds its confidence in content-related training examples (relevant to the question and answer) versus in generic examples of confidence verbalization. Our analysis reveals that OLMo2-13B is frequently influenced by confidence-related data that is lexically unrelated to the query, suggesting that it may mimic superficial linguistic expressions of certainty rather than rely on genuine content grounding. These findings point to a fundamental limitation in current training regimes: LLMs may learn how to sound confident without learning when confidence is justified. Our analysis provides a foundation for improving LLMs' trustworthiness in expressing more reliable confidence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.13142v4">Can LLMs Understand What We Cannot Say? Measuring Multilevel Alignment Through Abortion Stigma Across Cognitive, Interpersonal, and Structural Levels</a></div>
    <div class="paper-meta">
      📅 2026-01-15
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) increasingly mediate stigmatized health decisions, their capacity to understand complex psychological phenomena remains inadequately assessed. Can LLMs understand what we cannot say? We investigate whether LLMs coherently represent abortion stigma across cognitive, interpersonal, and structural levels. We systematically tested 627 demographically diverse personas across five leading LLMs using the validated Individual Level Abortion Stigma Scale (ILAS), examining representation at cognitive (self-judgment), interpersonal (worries about judgment and isolation), and structural (community condemnation and disclosure patterns) levels. Models fail tests of genuine understanding across all dimensions. They underestimate cognitive stigma while overestimating interpersonal stigma, introduce demographic biases assigning higher stigma to younger, less educated, and non-White personas, and treat secrecy as universal despite 36% of humans reporting openness. Most critically, models produce internal contradictions: they overestimate isolation yet predict isolated individuals are less secretive, revealing incoherent representations. These patterns show current alignment approaches ensure appropriate language but not coherent understanding across levels. This work provides empirical evidence that LLMs lack coherent understanding of psychological constructs operating across multiple dimensions. AI safety in high-stakes contexts demands new approaches to design (multilevel coherence), evaluation (continuous auditing), governance and regulation (mandatory audits, accountability, deployment restrictions), and AI literacy in domains where understanding what people cannot say determines whether support helps or harms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10609v1">iTIMO: An LLM-empowered Synthesis Dataset for Travel Itinerary Modification</a></div>
    <div class="paper-meta">
      📅 2026-01-15
    </div>
    <details class="paper-abstract">
      Addressing itinerary modification is crucial for enhancing the travel experience as it is a frequent requirement during traveling. However, existing research mainly focuses on fixed itinerary planning, leaving modification underexplored. To bridge this gap, we formally define the itinerary modification task and introduce iTIMO, a dataset specifically tailored for this purpose. We identify the lack of {\itshape need-to-modify} itinerary data as the critical bottleneck hindering research on this task and propose a general pipeline to overcome it. This pipeline frames the generation of such data as an intent-driven perturbation task. It instructs large language models to perturb real world itineraries using three atomic editing operations: REPLACE, ADD, and DELETE. Each perturbation is grounded in three intents, including disruptions of popularity, spatial distance, and category diversity. Furthermore, a hybrid evaluation metric is designed to ensure perturbation effectiveness. We conduct comprehensive experiments on iTIMO, revealing the limitations of current LLMs and lead to several valuable directions for future research. Dataset and corresponding code are available at https://github.com/zelo2/iTIMO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08763v2">Rewarding the Rare: Uniqueness-Aware RL for Creative Problem Solving in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-15
      | 💬 Work in Progress
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has become a central paradigm for post-training large language models (LLMs), particularly for complex reasoning tasks, yet it often suffers from exploration collapse: policies prematurely concentrate on a small set of dominant reasoning patterns, improving pass@1 while limiting rollout-level diversity and gains in pass@k. We argue that this failure stems from regularizing local token behavior rather than diversity over sets of solutions. To address this, we propose Uniqueness-Aware Reinforcement Learning, a rollout-level objective that explicitly rewards correct solutions that exhibit rare high-level strategies. Our method uses an LLM-based judge to cluster rollouts for the same problem according to their high-level solution strategies, ignoring superficial variations, and reweights policy advantages inversely with cluster size. As a result, correct but novel strategies receive higher rewards than redundant ones. Across mathematics, physics, and medical reasoning benchmarks, our approach consistently improves pass@$k$ across large sampling budgets and increases the area under the pass@$k$ curve (AUC@$K$) without sacrificing pass@1, while sustaining exploration and uncovering more diverse solution strategies at scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.00010v2">PlotCraft: Pushing the Limits of LLMs for Complex and Interactive Data Visualization</a></div>
    <div class="paper-meta">
      📅 2026-01-15
    </div>
    <details class="paper-abstract">
      Recent Large Language Models (LLMs) have demonstrated remarkable proficiency in code generation. However, their ability to create complex visualizations for scaled and structured data remains largely unevaluated and underdeveloped. To address this gap, we introduce PlotCraft, a new benchmark featuring 1k challenging visualization tasks that cover a wide range of topics, such as finance, scientific research, and sociology. The benchmark is structured around seven high-level visualization tasks and encompasses 48 distinct chart types. Crucially, it is the first to systematically evaluate both single-turn generation and multi-turn refinement across a diverse spectrum of task complexities. Our comprehensive evaluation of 23 leading LLMs on PlotCraft reveals obvious performance deficiencies in handling sophisticated visualization tasks. To bridge this performance gap, we develope SynthVis-30K, a large-scale, high-quality dataset of complex visualization code synthesized via a collaborative agent framework. Building upon this dataset, we develope PlotCraftor, a novel code generation model that achieves strong capabilities in complex data visualization with a remarkably small size. Across VisEval, PandasPlotBench, and our proposed PlotCraft, PlotCraftor shows performance comparable to that of leading proprietary approaches. Especially, on hard task, Our model achieves over 50% performance improvement. We will release the benchmark, dataset, and code at https://github.com/Speakn0w/PlotCraft-Benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.01667v4">Testing Low-Resource Language Support in LLMs Using Language Proficiency Exams: the Case of Luxembourgish</a></div>
    <div class="paper-meta">
      📅 2026-01-15
      | 💬 24pages, 4 figures, 14 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become an increasingly important tool in research and society at large. While LLMs are regularly used all over the world by experts and lay-people alike, they are predominantly developed with English-speaking users in mind, performing well in English and other wide-spread languages while less-resourced languages such as Luxembourgish are seen as a lower priority. This lack of attention is also reflected in the sparsity of available evaluation tools and datasets. In this study, we investigate the viability of language proficiency exams as such evaluation tools for the Luxembourgish language. We find that large models such as Claude and DeepSeek-R1 typically achieve high scores, while smaller models show weak performances. We also find that the performances in such language exams can be used to predict performances in other NLP tasks in Luxembourgish.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10524v1">Diagnosing Generalization Failures in Fine-Tuned LLMs: A Cross-Architectural Study on Phishing Detection</a></div>
    <div class="paper-meta">
      📅 2026-01-15
      | 💬 16 pages, 6 figures, 6 tables
    </div>
    <details class="paper-abstract">
      The practice of fine-tuning Large Language Models (LLMs) has achieved state-of-the-art performance on specialized tasks, yet diagnosing why these models become brittle and fail to generalize remains a critical open problem. To address this, we introduce and apply a multi-layered diagnostic framework to a cross-architectural study. We fine-tune Llama 3.1 8B, Gemma 2 9B, and Mistral models on a high-stakes phishing detection task and use SHAP analysis and mechanistic interpretability to uncover the root causes of their generalization failures. Our investigation reveals three critical findings: (1) Generalization is driven by a powerful synergy between architecture and data diversity. The Gemma 2 9B model achieves state-of-the-art performance (>91\% F1), but only when trained on a stylistically diverse ``generalist'' dataset. (2) Generalization is highly architecture-dependent. We diagnose a specific failure mode in Llama 3.1 8B, which performs well on a narrow domain but cannot integrate diverse data, leading to a significant performance drop. (3) Some architectures are inherently more generalizable. The Mistral model proves to be a consistent and resilient performer across multiple training paradigms. By pinpointing the flawed heuristics responsible for these failures, our work provides a concrete methodology for diagnosing and understanding generalization failures, underscoring that reliable AI requires deep validation of the interplay between architecture, data, and training strategy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10496v1">Model See, Model Do? Exposure-Aware Evaluation of Bug-vs-Fix Preference in Code LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-15
      | 💬 MSR 2026 Technical Track
    </div>
    <details class="paper-abstract">
      Large language models are increasingly used for code generation and debugging, but their outputs can still contain bugs, that originate from training data. Distinguishing whether an LLM prefers correct code, or a familiar incorrect version might be influenced by what it's been exposed to during training. We introduce an exposure-aware evaluation framework that quantifies how prior exposure to buggy versus fixed code influences a model's preference. Using the ManySStuBs4J benchmark, we apply Data Portraits for membership testing on the Stack-V2 corpus to estimate whether each buggy and fixed variant was seen during training. We then stratify examples by exposure and compare model preference using code completion as well as multiple likelihood-based scoring metrics We find that most examples (67%) have neither variant in the training data, and when only one is present, fixes are more frequently present than bugs. In model generations, models reproduce buggy lines far more often than fixes, with bug-exposed examples amplifying this tendency and fix-exposed examples showing only marginal improvement. In likelihood scoring, minimum and maximum token-probability metrics consistently prefer the fixed code across all conditions, indicating a stable bias toward correct fixes. In contrast, metrics like the Gini coefficient reverse preference when only the buggy variant was seen. Our results indicate that exposure can skew bug-fix evaluations and highlight the risk that LLMs may propagate memorised errors in practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10458v1">LangLasso: Interactive Cluster Descriptions through LLM Explanation</a></div>
    <div class="paper-meta">
      📅 2026-01-15
      | 💬 This manuscript is accepted for publication in VIS 2025 VISxGenAI Workshop
    </div>
    <details class="paper-abstract">
      Dimensionality reduction is a powerful technique for revealing structure and potential clusters in data. However, as the axes are complex, non-linear combinations of features, they often lack semantic interpretability. Existing visual analytics (VA) methods support cluster interpretation through feature comparison and interactive exploration, but they require technical expertise and intense human effort. We present \textit{LangLasso}, a novel method that complements VA approaches through interactive, natural language descriptions of clusters using large language models (LLMs). It produces human-readable descriptions that make cluster interpretation accessible to non-experts and allow integration of external contextual knowledge beyond the dataset. We systematically evaluate the reliability of these explanations and demonstrate that \langlasso provides an effective first step for engaging broader audiences in cluster interpretation. The tool is available at https://langlasso.vercel.app
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.20721v3">User Perceptions vs. Proxy LLM Judges: Privacy and Helpfulness in LLM Responses to Privacy-Sensitive Scenarios</a></div>
    <div class="paper-meta">
      📅 2026-01-15
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are rapidly being adopted for tasks like drafting emails, summarizing meetings, and answering health questions. In these settings, users may need to share private information (e.g., contact details, health records). To evaluate LLMs' ability to identify and redact such information, prior work introduced real-life, scenario-based benchmarks (e.g., ConfAIde, PrivacyLens) and found that LLMs can leak private information in complex scenarios. However, these evaluations relied on proxy LLMs to judge the helpfulness and privacy-preservation quality of LLM responses, rather than directly measuring users' perceptions. To understand how users perceive the helpfulness and privacy-preservation quality of LLM responses to privacy-sensitive scenarios, we conducted a user study ($n=94$) using 90 PrivacyLens scenarios. We found that users had low agreement with each other when evaluating identical LLM responses. In contrast, five proxy LLMs reached high agreement, yet each proxy LLM had low correlation with users' evaluations. These results indicate that proxy LLMs cannot accurately estimate users' wide range of perceptions of utility and privacy in privacy-sensitive scenarios. We discuss the need for more user-centered studies to measure LLMs' ability to help users while preserving privacy, and for improving alignment between LLMs and users in estimating perceived privacy and utility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.01351v2">NATLM: Detecting Defects in NFT Smart Contracts Leveraging LLM</a></div>
    <div class="paper-meta">
      📅 2026-01-15
    </div>
    <details class="paper-abstract">
      Security issues are becoming increasingly significant with the rapid evolution of Non-fungible Tokens (NFTs). As NFTs are traded as digital assets, they have emerged as prime targets for cyber attackers. In the development of NFT smart contracts, there may exist undiscovered defects that could lead to substantial financial losses if exploited. To tackle this issue, this paper presents a framework called NATLM(NFT Assistant LLM), designed to detect potential defects in NFT smart contracts. The framework effectively identifies four common types of vulnerabilities in NFT smart contracts: ERC-721 Reentrancy, Public Burn, Risky Mutable Proxy, and Unlimited Minting. Relying exclusively on large language models (LLMs) for defect detection can lead to a high false-positive rate. To enhance detection performance, NATLM integrates static analysis with LLMs, specifically Gemini Pro 1.5. Initially, NATLM employs static analysis to extract structural, syntactic, and execution flow information from the code, represented through Abstract Syntax Trees (AST) and Control Flow Graphs (CFG). These extracted features are then combined with vectors of known defect examples to create a matrix for input into the knowledge base. Subsequently, the feature vectors and code vectors of the analyzed contract are compared with the contents of the knowledge base. Finally, the LLM performs deep semantic analysis to enhance detection capabilities, providing a more comprehensive and accurate identification of potential security issues. Experimental results indicate that NATLM analyzed 8,672 collected NFT smart contracts, achieving an overall precision of 87.72%, a recall of 89.58%, and an F1 score of 88.94%. The results outperform other baseline experiments, successfully identifying four common types of defects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01452v3">Robust and Efficient Zeroth-Order LLM Fine-Tuning via Adaptive Bayesian Subspace Optimizer</a></div>
    <div class="paper-meta">
      📅 2026-01-15
      | 💬 23 pages, 2 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) with zeroth-order (ZO) optimization reduces memory by approximating gradients through function evaluations. However, existing methods essentially perform updates in a one-dimensional space, and suffer from collapse or substantial performance degradation under low-precision training. We introduce BSZO, an adaptive \textbf{B}ayesian \textbf{S}ubspace \textbf{Z}eroth-Order \textbf{O}ptimizer, which applies Kalman filtering to combine finite-difference information across multiple perturbation directions within a subspace. By treating each finite-difference measurement as a noisy observation, BSZO builds a posterior distribution over the subspace-projected gradient and updates it through Bayesian inference, with a residual-based adaptive mechanism to adapt to noise variations. Theoretical analysis shows that BSZO improves the convergence rate by a factor of $k/γ$ compared to standard ZO methods. Experiments on RoBERTa, Mistral, and OPT models show that BSZO outperforms the baselines across various tasks, achieving up to 6.67\% absolute average improvement on OPT-13B while remaining robust under fp16/bf16 precision and keeping memory usage close to inference-only baselines (1.00$\times$--1.08$\times$ of MeZO).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10332v1">Think-Then-Generate: Reasoning-Aware Text-to-Image Diffusion with LLM Encoders</a></div>
    <div class="paper-meta">
      📅 2026-01-15
    </div>
    <details class="paper-abstract">
      Recent progress in text-to-image (T2I) diffusion models (DMs) has enabled high-quality visual synthesis from diverse textual prompts. Yet, most existing T2I DMs, even those equipped with large language model (LLM)-based text encoders, remain text-pixel mappers -- they employ LLMs merely as text encoders, without leveraging their inherent reasoning capabilities to infer what should be visually depicted given the textual prompt. To move beyond such literal generation, we propose the think-then-generate (T2G) paradigm, where the LLM-based text encoder is encouraged to reason about and rewrite raw user prompts; the states of the rewritten prompts then serve as diffusion conditioning. To achieve this, we first activate the think-then-rewrite pattern of the LLM encoder with a lightweight supervised fine-tuning process. Subsequently, the LLM encoder and diffusion backbone are co-optimized to ensure faithful reasoning about the context and accurate rendering of the semantics via Dual-GRPO. In particular, the text encoder is reinforced using image-grounded rewards to infer and recall world knowledge, while the diffusion backbone is pushed to produce semantically consistent and visually coherent images. Experiments show substantial improvements in factual consistency, semantic alignment, and visual realism across reasoning-based image generation and editing benchmarks, achieving 0.79 on WISE score, nearly on par with GPT-4. Our results constitute a promising step toward next-generation unified models with reasoning, expression, and demonstration capacities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10321v1">An Efficient Long-Context Ranking Architecture With Calibrated LLM Distillation: Application to Person-Job Fit</a></div>
    <div class="paper-meta">
      📅 2026-01-15
    </div>
    <details class="paper-abstract">
      Finding the most relevant person for a job proposal in real time is challenging, especially when resumes are long, structured, and multilingual. In this paper, we propose a re-ranking model based on a new generation of late cross-attention architecture, that decomposes both resumes and project briefs to efficiently handle long-context inputs with minimal computational overhead. To mitigate historical data biases, we use a generative large language model (LLM) as a teacher, generating fine-grained, semantically grounded supervision. This signal is distilled into our student model via an enriched distillation loss function. The resulting model produces skill-fit scores that enable consistent and interpretable person-job matching. Experiments on relevance, ranking, and calibration metrics demonstrate that our approach outperforms state-of-the-art baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10307v1">The Straight and Narrow: Do LLMs Possess an Internal Moral Path?</a></div>
    <div class="paper-meta">
      📅 2026-01-15
    </div>
    <details class="paper-abstract">
      Enhancing the moral alignment of Large Language Models (LLMs) is a critical challenge in AI safety. Current alignment techniques often act as superficial guardrails, leaving the intrinsic moral representations of LLMs largely untouched. In this paper, we bridge this gap by leveraging Moral Foundations Theory (MFT) to map and manipulate the fine-grained moral landscape of LLMs. Through cross-lingual linear probing, we validate the shared nature of moral representations in middle layers and uncover a shared yet different moral subspace between English and Chinese. Building upon this, we extract steerable Moral Vectors and successfully validate their efficacy at both internal and behavioral levels. Leveraging the high generalizability of morality, we propose Adaptive Moral Fusion (AMF), a dynamic inference-time intervention that synergizes probe detection with vector injection to tackle the safety-helpfulness trade-off. Empirical results confirm that our approach acts as a targeted intrinsic defense, effectively reducing incorrect refusals on benign queries while minimizing jailbreak success rates compared to standard baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.02024v3">NestedFP: High-Performance, Memory-Efficient Dual-Precision Floating Point Support for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-15
    </div>
    <details class="paper-abstract">
      Meeting service-level objectives (SLOs) in Large Language Models (LLMs) serving is critical, but managing the high variability in load presents a significant challenge. Recent advancements in FP8 inference, backed by native hardware support, offer a potential solution: executing FP16 models by default, while switching to FP8 models during sudden load surges to achieve higher throughput at the cost of a slight quality degradation. Although this approach facilitates effective SLO management, it introduces additional memory overhead due to storing two versions of the same model. In response, this paper proposes NestedFP, an LLM serving technique that supports both FP16 and FP8 models in a memory-efficient manner by overlaying FP8 parameters onto FP16 parameters, allowing both models to share the same FP16 memory footprint. By leveraging a compact data format for the overlay and a specialized GEMM kernel optimized for this format, NestedFP ensures minimal degradation in both model quality and inference throughput across both FP8 and FP16 modes. NestedFP provides a flexible platform for dynamic, SLO-aware precision selection. The code is available at https://github.com/SNU-ARC/NestedFP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10294v1">Reasoning Hijacking: Subverting LLM Classification via Decision-Criteria Injection</a></div>
    <div class="paper-meta">
      📅 2026-01-15
    </div>
    <details class="paper-abstract">
      Current LLM safety research predominantly focuses on mitigating Goal Hijacking, preventing attackers from redirecting a model's high-level objective (e.g., from "summarizing emails" to "phishing users"). In this paper, we argue that this perspective is incomplete and highlight a critical vulnerability in Reasoning Alignment. We propose a new adversarial paradigm: Reasoning Hijacking and instantiate it with Criteria Attack, which subverts model judgments by injecting spurious decision criteria without altering the high-level task goal. Unlike Goal Hijacking, which attempts to override the system prompt, Reasoning Hijacking accepts the high-level goal but manipulates the model's decision-making logic by injecting spurious reasoning shortcut. Though extensive experiments on three different tasks (toxic comment, negative review, and spam detection), we demonstrate that even newest models are prone to prioritize injected heuristic shortcuts over rigorous semantic analysis. The results are consistent over different backbones. Crucially, because the model's "intent" remains aligned with the user's instructions, these attacks can bypass defenses designed to detect goal deviation (e.g., SecAlign, StruQ), exposing a fundamental blind spot in the current safety landscape. Data and code are available at https://github.com/Yuan-Hou/criteria_attack
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10274v1">Queueing-Aware Optimization of Reasoning Tokens for Accuracy-Latency Trade-offs in LLM Servers</a></div>
    <div class="paper-meta">
      📅 2026-01-15
    </div>
    <details class="paper-abstract">
      We consider a single large language model (LLM) server that serves a heterogeneous stream of queries belonging to $N$ distinct task types. Queries arrive according to a Poisson process, and each type occurs with a known prior probability. For each task type, the server allocates a fixed number of internal thinking tokens, which determines the computational effort devoted to that query. The token allocation induces an accuracy-latency trade-off: the service time follows an approximately affine function of the allocated tokens, while the probability of a correct response exhibits diminishing returns. Under a first-in, first-out (FIFO) service discipline, the system operates as an $M/G/1$ queue, and the mean system time depends on the first and second moments of the resulting service-time distribution. We formulate a constrained optimization problem that maximizes a weighted average accuracy objective penalized by the mean system time, subject to architectural token-budget constraints and queue-stability conditions. The objective function is shown to be strictly concave over the stability region, which ensures existence and uniqueness of the optimal token allocation. The first-order optimality conditions yield a coupled projected fixed-point characterization of the optimum, together with an iterative solution and an explicit sufficient condition for contraction. Moreover, a projected gradient method with a computable global step-size bound is developed to guarantee convergence beyond the contractive regime. Finally, integer-valued token allocations are attained via rounding of the continuous solution, and the resulting performance loss is evaluated in simulation results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10257v1">Untangling Input Language from Reasoning Language: A Diagnostic Framework for Cross-Lingual Moral Alignment in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-15
    </div>
    <details class="paper-abstract">
      When LLMs judge moral dilemmas, do they reach different conclusions in different languages, and if so, why? Two factors could drive such differences: the language of the dilemma itself, or the language in which the model reasons. Standard evaluation conflates these by testing only matched conditions (e.g., English dilemma with English reasoning). We introduce a methodology that separately manipulates each factor, covering also mismatched conditions (e.g., English dilemma with Chinese reasoning), enabling decomposition of their contributions. To study \emph{what} changes, we propose an approach to interpret the moral judgments in terms of Moral Foundations Theory. As a side result, we identify evidence for splitting the Authority dimension into a family-related and an institutional dimension. Applying this methodology to English-Chinese moral judgment with 13 LLMs, we demonstrate its diagnostic power: (1) the framework isolates reasoning-language effects as contributing twice the variance of input-language effects; (2) it detects context-dependency in nearly half of models that standard evaluation misses; and (3) a diagnostic taxonomy translates these patterns into deployment guidance. We release our code and datasets at https://anonymous.4open.science/r/CrossCulturalMoralJudgement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.01891v4">Multi-Personality Generation of LLMs at Decoding-time</a></div>
    <div class="paper-meta">
      📅 2026-01-15
      | 💬 Accepted by WSDM 2026
    </div>
    <details class="paper-abstract">
      Multi-personality generation for LLMs, enabling simultaneous embodiment of multiple personalization attributes, is a fundamental challenge. Existing retraining-based approaches are costly and poorly scalable, while decoding-time methods often rely on external models or heuristics, limiting flexibility and robustness. In this paper, we propose a novel Multi-Personality Generation (MPG) framework under the decoding-time combination paradigm. It flexibly controls multi-personality without relying on scarce multi-dimensional models or extra training, leveraging implicit density ratios in single-dimensional models as a "free lunch" to reformulate the task as sampling from a target strategy aggregating these ratios. To implement MPG efficiently, we design Speculative Chunk-level based Rejection sampling (SCR), which generates responses in chunks and parallelly validates them via estimated thresholds within a sliding window. This significantly reduces computational overhead while maintaining high-quality generation. Experiments on MBTI personality and Role-Playing demonstrate the effectiveness of MPG, showing improvements up to 16%-18%. Code and data are available at https://github.com/Libra117/MPG .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10201v1">PRL: Process Reward Learning Improves LLMs' Reasoning Ability and Broadens the Reasoning Boundary</a></div>
    <div class="paper-meta">
      📅 2026-01-15
    </div>
    <details class="paper-abstract">
      Improving the reasoning abilities of Large Language Models (LLMs) has been a continuous topic recently. But most relevant works are based on outcome rewards at the trajectory level, missing fine-grained supervision during the reasoning process. Other existing training frameworks that try to combine process signals together to optimize LLMs also rely heavily on tedious additional steps like MCTS, training a separate reward model, etc., doing harm to the training efficiency. Moreover, the intuition behind the process signals design lacks rigorous theoretical support, leaving the understanding of the optimization mechanism opaque. In this paper, we propose Process Reward Learning (PRL), which decomposes the entropy regularized reinforcement learning objective into intermediate steps, with rigorous process rewards that could be assigned to models accordingly. Starting from theoretical motivation, we derive the formulation of PRL that is essentially equivalent to the objective of reward maximization plus a KL-divergence penalty term between the policy model and a reference model. However, PRL could turn the outcome reward into process supervision signals, which helps better guide the exploration during RL optimization. From our experiment results, we demonstrate that PRL not only improves the average performance for LLMs' reasoning ability measured by average @ n, but also broadens the reasoning boundary by improving the pass @ n metric. Extensive experiments show the effectiveness of PRL could be verified and generalized.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10198v1">HUMANLLM: Benchmarking and Reinforcing LLM Anthropomorphism via Human Cognitive Patterns</a></div>
    <div class="paper-meta">
      📅 2026-01-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities in reasoning and generation, serving as the foundation for advanced persona simulation and Role-Playing Language Agents (RPLAs). However, achieving authentic alignment with human cognitive and behavioral patterns remains a critical challenge for these agents. We present HUMANLLM, a framework treating psychological patterns as interacting causal forces. We construct 244 patterns from ~12,000 academic papers and synthesize 11,359 scenarios where 2-5 patterns reinforce, conflict, or modulate each other, with multi-turn conversations expressing inner thoughts, actions, and dialogue. Our dual-level checklists evaluate both individual pattern fidelity and emergent multi-pattern dynamics, achieving strong human alignment (r=0.91) while revealing that holistic metrics conflate simulation accuracy with social desirability. HUMANLLM-8B outperforms Qwen3-32B on multi-pattern dynamics despite 4x fewer parameters, demonstrating that authentic anthropomorphism requires cognitive modeling--simulating not just what humans do, but the psychological processes generating those behaviors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10187v1">HOMURA: Taming the Sand-Glass for Time-Constrained LLM Translation via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-01-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable strides in multilingual translation but are hindered by a systemic cross-lingual verbosity bias, rendering them unsuitable for strict time-constrained tasks like subtitling and dubbing. Current prompt-engineering approaches struggle to resolve this conflict between semantic fidelity and rigid temporal feasibility. To bridge this gap, we first introduce Sand-Glass, a benchmark specifically designed to evaluate translation under syllable-level duration constraints. Furthermore, we propose HOMURA, a reinforcement learning framework that explicitly optimizes the trade-off between semantic preservation and temporal compliance. By employing a KL-regularized objective with a novel dynamic syllable-ratio reward, HOMURA effectively "tames" the output length. Experimental results demonstrate that our method significantly outperforms strong LLM baselines, achieving precise length control that respects linguistic density hierarchies without compromising semantic adequacy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09385v1">SLAM-LLM: A Modular, Open-Source Multimodal Large Language Model Framework and Best Practice for Speech, Language, Audio and Music Processing</a></div>
    <div class="paper-meta">
      📅 2026-01-14
      | 💬 Published in IEEE Journal of Selected Topics in Signal Processing (JSTSP)
    </div>
    <details class="paper-abstract">
      The recent surge in open-source Multimodal Large Language Models (MLLM) frameworks, such as LLaVA, provides a convenient kickoff for artificial intelligence developers and researchers. However, most of the MLLM frameworks take vision as the main input modality, and provide limited in-depth support for the modality of speech, audio, and music. This situation hinders the development of audio-language models, and forces researchers to spend a lot of effort on code writing and hyperparameter tuning. We present SLAM-LLM, an open-source deep learning framework designed to train customized MLLMs, focused on speech, language, audio, and music processing. SLAM-LLM provides a modular configuration of different encoders, projectors, LLMs, and parameter-efficient fine-tuning plugins. SLAM-LLM also includes detailed training and inference recipes for mainstream tasks, along with high-performance checkpoints like LLM-based Automatic Speech Recognition (ASR), Automated Audio Captioning (AAC), and Music Captioning (MC). Some of these recipes have already reached or are nearing state-of-the-art performance, and some relevant techniques have also been accepted by academic papers. We hope SLAM-LLM will accelerate iteration, development, data engineering, and model training for researchers. We are committed to continually pushing forward audio-based MLLMs through this open-source framework, and call on the community to contribute to the LLM-based speech, audio and music processing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09367v1">Relation Extraction Capabilities of LLMs on Clinical Text: A Bilingual Evaluation for English and Turkish</a></div>
    <div class="paper-meta">
      📅 2026-01-14
    </div>
    <details class="paper-abstract">
      The scarcity of annotated datasets for clinical information extraction in non-English languages hinders the evaluation of large language model (LLM)-based methods developed primarily in English. In this study, we present the first comprehensive bilingual evaluation of LLMs for the clinical Relation Extraction (RE) task in both English and Turkish. To facilitate this evaluation, we introduce the first English-Turkish parallel clinical RE dataset, derived and carefully curated from the 2010 i2b2/VA relation classification corpus. We systematically assess a diverse set of prompting strategies, including multiple in-context learning (ICL) and Chain-of-Thought (CoT) approaches, and compare their performance to fine-tuned baselines such as PURE. Furthermore, we propose Relation-Aware Retrieval (RAR), a novel in-context example selection method based on contrastive learning, that is specifically designed to capture both sentence-level and relation-level semantics. Our results show that prompting-based LLM approaches consistently outperform traditional fine-tuned models. Moreover, evaluations for English performed better than their Turkish counterparts across all evaluated LLMs and prompting techniques. Among ICL methods, RAR achieves the highest performance, with Gemini 1.5 Flash reaching a micro-F1 score of 0.906 in English and 0.888 in Turkish. Performance further improves to 0.918 F1 in English when RAR is combined with a structured reasoning prompt using the DeepSeek-V3 model. These findings highlight the importance of high-quality demonstration retrieval and underscore the potential of advanced retrieval and prompting techniques to bridge resource gaps in clinical natural language processing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.20977v2">Repairing vulnerabilities without invisible hands. A differentiated replication study on LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-14
    </div>
    <details class="paper-abstract">
      Background: Automated Vulnerability Repair (AVR) is a fast-growing branch of program repair. Recent studies show that large language models (LLMs) outperform traditional techniques, extending their success beyond code generation and fault detection. Hypothesis: These gains may be driven by hidden factors -- "invisible hands" such as training-data leakage or perfect fault localization -- that let an LLM reproduce human-authored fixes for the same code. Objective: We replicate prior AVR studies under controlled conditions by deliberately adding errors to the reported vulnerability location in the prompt. If LLMs merely regurgitate memorized fixes, both small and large localization errors should yield the same number of correct patches, because any offset should divert the model from the original fix. Method: Our pipeline repairs vulnerabilities from the Vul4J and VJTrans benchmarks after shifting the fault location by n lines from the ground truth. A first LLM generates a patch, a second LLM reviews it, and we validate the result with regression and proof-of-vulnerability tests. Finally, we manually audit a sample of patches and estimate the error rate with the Agresti-Coull-Wilson method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09321v1">SpatialJB: How Text Distribution Art Becomes the "Jailbreak Key" for LLM Guardrails</a></div>
    <div class="paper-meta">
      📅 2026-01-14
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have powerful capabilities, they remain vulnerable to jailbreak attacks, which is a critical barrier to their safe web real-time application. Current commercial LLM providers deploy output guardrails to filter harmful outputs, yet these defenses are not impenetrable. Due to LLMs' reliance on autoregressive, token-by-token inference, their semantic representations lack robustness to spatially structured perturbations, such as redistributing tokens across different rows, columns, or diagonals. Exploiting the Transformer's spatial weakness, we propose SpatialJB to disrupt the model's output generation process, allowing harmful content to bypass guardrails without detection. Comprehensive experiments conducted on leading LLMs get nearly 100% ASR, demonstrating the high effectiveness of SpatialJB. Even after adding advanced output guardrails, like the OpenAI Moderation API, SpatialJB consistently maintains a success rate exceeding 75%, outperforming current jailbreak techniques by a significant margin. The proposal of SpatialJB exposes a key weakness in current guardrails and emphasizes the importance of spatial semantics, offering new insights to advance LLM safety research. To prevent potential misuse, we also present baseline defense strategies against SpatialJB and evaluate their effectiveness in mitigating such attacks. The code for the attack, baseline defenses, and a demo are available at https://anonymous.4open.science/r/SpatialJailbreak-8E63.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.15553v2">Efficient Routing of Inference Requests across LLM Instances in Cloud-Edge Computing</a></div>
    <div class="paper-meta">
      📅 2026-01-14
    </div>
    <details class="paper-abstract">
      The rising demand for Large Language Model (LLM) inference services has intensified pressure on computational resources, resulting in latency and cost challenges. This paper introduces a novel routing algorithm based on the Non-dominated Sorting Genetic Algorithm II (NSGA-II) to distribute inference requests across heterogeneous LLM instances in a cloud-edge computing environment. Formulated as a multi-objective optimization problem, the algorithm balances response quality, response time, and inference cost, adapting to request heterogeneity (e.g., varying complexity and prompt lengths) and node diversity (e.g., edge vs. cloud resources). This adaptive routing algorithm optimizes performance under dynamic workloads. We benchmark the approach using a testbed with datasets including Stanford Question Answering Dataset (SQuAD), Mostly Basic Python Problems (MBPP), Hella Situations With Adversarial Generations (HellaSwag), and Grade School Math 8K (GSM8K). Experimental results show our solution, compared to the baselines, preserves 95.2% of Cloud-Only response quality with slight latency increase, while reducing inference cost by 34.9%. These findings validate the algorithm's effectiveness for scalable LLM deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09298v1">Multi-Modal LLM based Image Captioning in ICT: Bridging the Gap Between General and Industry Domain</a></div>
    <div class="paper-meta">
      📅 2026-01-14
    </div>
    <details class="paper-abstract">
      In the information and communications technology (ICT) industry, training a domain-specific large language model (LLM) or constructing a retrieval-augmented generation system requires a substantial amount of high-value domain knowledge. However, the knowledge is not only hidden in the textual modality but also in the image modality. Traditional methods can parse text from domain documents but dont have image captioning ability. Multi-modal LLM (MLLM) can understand images, but they do not have sufficient domain knowledge. To address the above issues, this paper proposes a multi-stage progressive training strategy to train a Domain-specific Image Captioning Model (DICModel) in ICT, and constructs a standard evaluation system to validate the performance of DICModel. Specifically, this work first synthesizes about 7K image-text pairs by combining the Mermaid tool and LLMs, which are used for the first-stage supervised-fine-tuning (SFT) of DICModel. Then, ICT-domain experts manually annotate about 2K image-text pairs for the second-stage SFT of DICModel. Finally, experts and LLMs jointly synthesize about 1.5K visual question answering data for the instruction-based SFT. Experimental results indicate that our DICModel with only 7B parameters performs better than other state-of-the-art models with 32B parameters. Compared to the SOTA models with 7B and 32B parameters, our DICModel increases the BLEU metric by approximately 56.8% and 20.8%, respectively. On the objective questions constructed by ICT domain experts, our DICModel outperforms Qwen2.5-VL 32B by 1% in terms of accuracy rate. In summary, this work can efficiently and accurately extract the logical text from images, which is expected to promote the development of multimodal models in the ICT domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09295v1">MACRO-LLM: LLM-Empowered Multi-Agent Collaborative Reasoning under Spatiotemporal Partial Observability</a></div>
    <div class="paper-meta">
      📅 2026-01-14
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents deployed in complex real-world scenarios typically operate as spatially distributed entities. However, this physical dispersion constrains agents to limited local perception and finite temporal horizons. We characterize this bottleneck as spatiotemporal partial observability. Given such fragmented awareness, distributed agents struggle to coordinate efficiently. To bridge this gap, we introduce MACRO-LLM, LLM-empowered multi-agent collaborative reasoning under spatiotemporal partial observability. The architecture addresses spatiotemporal constraints via three modules: (1) the CoProposer mitigates temporal uncertainty by verifying candidate actions via predictive rollouts; (2) the Negotiator overcomes spatial myopia by resolving conflicts through mean-field statistical aggregation; and (3) the Introspector ensures continuous adaptation by analyzing historical experience to refine strategies via semantic gradient descent. Extensive evaluations on two complex long-horizon tasks, cooperative adaptive cruise control and pandemic control, demonstrate that our framework effectively mitigates spatiotemporal partial observability through spatial and temporal strategies, enabling robust coordination.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16682v2">Bench360: Benchmarking Local LLM Inference from 360 Degrees</a></div>
    <div class="paper-meta">
      📅 2026-01-14
    </div>
    <details class="paper-abstract">
      Running LLMs locally has become increasingly common, but users face a complex design space across models, quantization levels, inference engines, and serving scenarios. Existing inference benchmarks are fragmented and focus on isolated goals, offering little guidance for practical deployments. We present Bench360, a framework for evaluating local LLM inference across tasks, usage patterns, and system metrics in one place. Bench360 supports custom tasks, integrates multiple inference engines and quantization formats, and reports both task quality and system behavior (latency, throughput, energy, startup time). We demonstrate it on four NLP tasks across three GPUs and four engines, showing how design choices shape efficiency and output quality. Results confirm that tradeoffs are substantial and configuration choices depend on specific workloads and constraints. There is no universal best option, underscoring the need for comprehensive, deployment-oriented benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09762v1">Explicating Tacit Regulatory Knowledge from LLMs to Auto-Formalize Requirements for Compliance Test Case Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-14
      | 💬 16 pages, 8 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Compliance testing in highly regulated domains is crucial but largely manual, requiring domain experts to translate complex regulations into executable test cases. While large language models (LLMs) show promise for automation, their susceptibility to hallucinations limits reliable application. Existing hybrid approaches mitigate this issue by constraining LLMs with formal models, but still rely on costly manual modeling. To solve this problem, this paper proposes RAFT, a framework for requirements auto-formalization and compliance test generation via explicating tacit regulatory knowledge from multiple LLMs. RAFT employs an Adaptive Purification-Aggregation strategy to explicate tacit regulatory knowledge from multiple LLMs and integrate it into three artifacts: a domain meta-model, a formal requirements representation, and testability constraints. These artifacts are then dynamically injected into prompts to guide high-precision requirement formalization and automated test generation. Experiments across financial, automotive, and power domains show that RAFT achieves expert-level performance, substantially outperforms state-of-the-art (SOTA) methods while reducing overall generation and review time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09259v1">MAXS: Meta-Adaptive Exploration with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-14
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) Agents exhibit inherent reasoning abilities through the collaboration of multiple tools. However, during agent inference, existing methods often suffer from (i) locally myopic generation, due to the absence of lookahead, and (ii) trajectory instability, where minor early errors can escalate into divergent reasoning paths. These issues make it difficult to balance global effectiveness and computational efficiency. To address these two issues, we propose meta-adaptive exploration with LLM agents https://github.com/exoskeletonzj/MAXS, a meta-adaptive reasoning framework based on LLM Agents that flexibly integrates tool execution and reasoning planning. MAXS employs a lookahead strategy to extend reasoning paths a few steps ahead, estimating the advantage value of tool usage, and combines step consistency variance and inter-step trend slopes to jointly select stable, consistent, and high-value reasoning steps. Additionally, we introduce a trajectory convergence mechanism that controls computational cost by halting further rollouts once path consistency is achieved, enabling a balance between resource efficiency and global effectiveness in multi-tool reasoning. We conduct extensive empirical studies across three base models (MiMo-VL-7B, Qwen2.5-VL-7B, Qwen2.5-VL-32B) and five datasets, demonstrating that MAXS consistently outperforms existing methods in both performance and inference efficiency. Further analysis confirms the effectiveness of our lookahead strategy and tool usage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09258v1">LatencyPrism: Online Non-intrusive Latency Sculpting for SLO-Guaranteed LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-01-14
      | 💬 12 pages, 6 figures
    </div>
    <details class="paper-abstract">
      LLM inference latency critically determines user experience and operational costs, directly impacting throughput under SLO constraints. Even brief latency spikes degrade service quality despite acceptable average performance. However, distributed inference environments featuring diverse software frameworks and XPU architectures combined with dynamic workloads make latency analysis challenging. Constrained by intrusive designs that necessitate service restarts or even suspension, and by hardware-bound implementations that fail to adapt to heterogeneous inference environments, existing AI profiling methods are often inadequate for real-time production analysis. We present LatencyPrism, the first zero-intrusion multi-platform latency sculpting system. It aims to break down the inference latency across pipeline, proactively alert on inference latency anomalies, and guarantee adherence to SLOs, all without requiring code modifications or service restarts. LatencyPrism has been deployed across thousands of XPUs for over six months. It enables low-overhead real-time monitoring at batch level with alerts triggered in milliseconds. This approach distinguishes between workload-driven latency variations and anomalies indicating underlying issues with an F1-score of 0.98. We also conduct extensive experiments and investigations into root cause analysis to demonstrate LatencyPrism's capability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09250v1">When to Invoke: Refining LLM Fairness with Toxicity Assessment</a></div>
    <div class="paper-meta">
      📅 2026-01-14
      | 💬 Accepted by Findings of WWW 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used for toxicity assessment in online moderation systems, where fairness across demographic groups is essential for equitable treatment. However, LLMs often produce inconsistent toxicity judgements for subtle expressions, particularly those involving implicit hate speech, revealing underlying biases that are difficult to correct through standard training. This raises a key question that existing approaches often overlook: when should corrective mechanisms be invoked to ensure fair and reliable assessments? To address this, we propose FairToT, an inference-time framework that enhances LLM fairness through prompt-guided toxicity assessment. FairToT identifies cases where demographic-related variation is likely to occur and determines when additional assessment should be applied. In addition, we introduce two interpretable fairness indicators that detect such cases and improve inference consistency without modifying model parameters. Experiments on benchmark datasets show that FairToT reduces group-level disparities while maintaining stable and reliable toxicity predictions, demonstrating that inference-time refinement offers an effective and practical approach for fairness improvement in LLM-based toxicity assessment systems. The source code can be found at https://aisuko.github.io/fair-tot/.
    </details>
</div>
