# llm - 2025_06

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- Part 7
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)
- [Part 15](papers_15.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12204v1">Semantic Scheduling for LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-06-13
      | 💬 18 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Conventional operating system scheduling algorithms are largely content-ignorant, making decisions based on factors such as latency or fairness without considering the actual intents or semantics of processes. Consequently, these algorithms often do not prioritize tasks that require urgent attention or carry higher importance, such as in emergency management scenarios. However, recent advances in language models enable semantic analysis of processes, allowing for more intelligent and context-aware scheduling decisions. In this paper, we introduce the concept of semantic scheduling in scheduling of requests from large language models (LLM), where the semantics of the process guide the scheduling priorities. We present a novel scheduling algorithm with optimal time complexity, designed to minimize the overall waiting time in LLM-based prompt scheduling. To illustrate its effectiveness, we present a medical emergency management application, underscoring the potential benefits of semantic scheduling for critical, time-sensitive tasks. The code and data are available at https://github.com/Wenyueh/latency_optimization_with_priority_constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12202v1">A Fast, Reliable, and Secure Programming Language for LLM Agents with Code Actions</a></div>
    <div class="paper-meta">
      📅 2025-06-13
    </div>
    <details class="paper-abstract">
      Modern large language models (LLMs) are often deployed as agents, calling external tools adaptively to solve tasks. Rather than directly calling tools, it can be more effective for LLMs to write code to perform the tool calls, enabling them to automatically generate complex control flow such as conditionals and loops. Such code actions are typically provided as Python code, since LLMs are quite proficient at it; however, Python may not be the ideal language due to limited built-in support for performance, security, and reliability. We propose a novel programming language for code actions, called Quasar, which has several benefits: (1) automated parallelization to improve performance, (2) uncertainty quantification to improve reliability and mitigate hallucinations, and (3) security features enabling the user to validate actions. LLMs can write code in a subset of Python, which is automatically transpiled to Quasar. We evaluate our approach on the ViperGPT visual question answering agent, applied to the GQA dataset, demonstrating that LLMs with Quasar actions instead of Python actions retain strong performance, while reducing execution time when possible by 42%, improving security by reducing user approval interactions when possible by 52%, and improving reliability by applying conformal prediction to achieve a desired target coverage level.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12182v1">Instruction Tuning and CoT Prompting for Contextual Medical QA with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-13
      | 💬 Accepted by 2025 International Conference on Artificial Intelligence, Human-Computer Interaction and Natural Language Processing
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown great potential in medical question answering (MedQA), yet adapting them to biomedical reasoning remains challenging due to domain-specific complexity and limited supervision. In this work, we study how prompt design and lightweight fine-tuning affect the performance of open-source LLMs on PubMedQA, a benchmark for multiple-choice biomedical questions. We focus on two widely used prompting strategies - standard instruction prompts and Chain-of-Thought (CoT) prompts - and apply QLoRA for parameter-efficient instruction tuning. Across multiple model families and sizes, our experiments show that CoT prompting alone can improve reasoning in zero-shot settings, while instruction tuning significantly boosts accuracy. However, fine-tuning on CoT prompts does not universally enhance performance and may even degrade it for certain larger models. These findings suggest that reasoning-aware prompts are useful, but their benefits are model- and scale-dependent. Our study offers practical insights into combining prompt engineering with efficient finetuning for medical QA applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12158v1">A Rigorous Evaluation of LLM Data Generation Strategies for Low-Resource Languages</a></div>
    <div class="paper-meta">
      📅 2025-06-13
      | 💬 21 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to generate synthetic textual data for training smaller specialized models. However, a comparison of various generation strategies for low-resource language settings is lacking. While various prompting strategies have been proposed, such as demonstrations, label-based summaries, and self-revision, their comparative effectiveness remains unclear, especially for low-resource languages. In this paper, we systematically evaluate the performance of these generation strategies and their combinations across 11 typologically diverse languages, including several extremely low-resource ones. Using three NLP tasks and four open-source LLMs, we assess downstream model performance on generated versus gold-standard data. Our results show that strategic combinations of generation methods, particularly target-language demonstrations with LLM-based revisions, yield strong performance, narrowing the gap with real data to as little as 5% in some settings. We also find that smart prompting techniques can reduce the advantage of larger LLMs, highlighting efficient generation strategies for synthetic data generation in low-resource scenarios with smaller models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12152v1">Because we have LLMs, we Can and Should Pursue Agentic Interpretability</a></div>
    <div class="paper-meta">
      📅 2025-06-13
    </div>
    <details class="paper-abstract">
      The era of Large Language Models (LLMs) presents a new opportunity for interpretability--agentic interpretability: a multi-turn conversation with an LLM wherein the LLM proactively assists human understanding by developing and leveraging a mental model of the user, which in turn enables humans to develop better mental models of the LLM. Such conversation is a new capability that traditional `inspective' interpretability methods (opening the black-box) do not use. Having a language model that aims to teach and explain--beyond just knowing how to talk--is similar to a teacher whose goal is to teach well, understanding that their success will be measured by the student's comprehension. While agentic interpretability may trade off completeness for interactivity, making it less suitable for high-stakes safety situations with potentially deceptive models, it leverages a cooperative model to discover potentially superhuman concepts that can improve humans' mental model of machines. Agentic interpretability introduces challenges, particularly in evaluation, due to what we call `human-entangled-in-the-loop' nature (humans responses are integral part of the algorithm), making the design and evaluation difficult. We discuss possible solutions and proxy goals. As LLMs approach human parity in many tasks, agentic interpretability's promise is to help humans learn the potentially superhuman concepts of the LLMs, rather than see us fall increasingly far from understanding them.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12109v1">Personalized LLM Decoding via Contrasting Personal Preference</a></div>
    <div class="paper-meta">
      📅 2025-06-13
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are progressively deployed in various real-world applications, personalization of LLMs has become increasingly important. While various approaches to LLM personalization such as prompt-based and training-based methods have been actively explored, the development of effective decoding-time algorithms remains largely overlooked, despite their demonstrated potential. In this paper, we propose CoPe (Contrasting Personal Preference), a novel decoding-time approach applied after performing parameter-efficient fine-tuning (PEFT) on user-specific data. Our core idea is to leverage reward-guided decoding specifically for personalization by maximizing each user's implicit reward signal. We evaluate CoPe across five open-ended personalized text generation tasks. Our empirical results demonstrate that CoPe achieves strong performance, improving personalization by an average of 10.57% in ROUGE-L, without relying on external reward models or additional training procedures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12104v1">DRIFT: Dynamic Rule-Based Defense with Injection Isolation for Securing LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-06-13
      | 💬 18 pages, 12 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly central to agentic systems due to their strong reasoning and planning capabilities. By interacting with external environments through predefined tools, these agents can carry out complex user tasks. Nonetheless, this interaction also introduces the risk of prompt injection attacks, where malicious inputs from external sources can mislead the agent's behavior, potentially resulting in economic loss, privacy leakage, or system compromise. System-level defenses have recently shown promise by enforcing static or predefined policies, but they still face two key challenges: the ability to dynamically update security rules and the need for memory stream isolation. To address these challenges, we propose DRIFT, a Dynamic Rule-based Isolation Framework for Trustworthy agentic systems, which enforces both control- and data-level constraints. A Secure Planner first constructs a minimal function trajectory and a JSON-schema-style parameter checklist for each function node based on the user query. A Dynamic Validator then monitors deviations from the original plan, assessing whether changes comply with privilege limitations and the user's intent. Finally, an Injection Isolator detects and masks any instructions that may conflict with the user query from the memory stream to mitigate long-term risks. We empirically validate the effectiveness of DRIFT on the AgentDojo benchmark, demonstrating its strong security performance while maintaining high utility across diverse models -- showcasing both its robustness and adaptability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12012v1">Tracing LLM Reasoning Processes with Strategic Games: A Framework for Planning, Revision, and Resource-Constrained Decision Making</a></div>
    <div class="paper-meta">
      📅 2025-06-13
      | 💬 19 pages, 7 figures. Under review
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used for tasks that require complex reasoning. Most benchmarks focus on final outcomes but overlook the intermediate reasoning steps - such as planning, revision, and decision making under resource constraints. We argue that measuring these internal processes is essential for understanding model behavior and improving reliability. We propose using strategic games as a natural evaluation environment: closed, rule-based systems with clear states, limited resources, and automatic feedback. We introduce a framework that evaluates LLMs along three core dimensions: planning, revision, and resource-constrained decision making. To operationalize this, we define metrics beyond win rate, including overcorrection risk rate, correction success rate, improvement slope, and over-budget ratio. In 4320 adversarial rounds across 12 leading models, ChatGPT-o3-mini achieves the top composite score, with a win rate of 74.7 percent, a correction success rate of 78.6 percent, and an improvement slope of 0.041. By contrast, Qwen-Plus, despite an overcorrection risk rate of 81.6 percent, wins only 25.6 percent of its matches - primarily due to excessive resource use. We also observe a negative correlation between overcorrection risk rate and correction success rate (Pearson r = -0.51, p = 0.093), suggesting that more frequent edits do not always improve outcomes. Our findings highlight the value of assessing not only what LLMs decide but how they arrive at those decisions
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09026v2">e3: Learning to Explore Enables Extrapolation of Test-Time Compute for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-13
    </div>
    <details class="paper-abstract">
      Test-time scaling offers a promising path to improve LLM reasoning by utilizing more compute at inference time; however, the true promise of this paradigm lies in extrapolation (i.e., improvement in performance on hard problems as LLMs keep "thinking" for longer, beyond the maximum token budget they were trained on). Surprisingly, we find that most existing reasoning models do not extrapolate well. We show that one way to enable extrapolation is by training the LLM to perform in-context exploration: training the LLM to effectively spend its test time budget by chaining operations (such as generation, verification, refinement, etc.), or testing multiple hypotheses before it commits to an answer. To enable in-context exploration, we identify three key ingredients as part of our recipe e3: (1) chaining skills that the base LLM has asymmetric competence in, e.g., chaining verification (easy) with generation (hard), as a way to implement in-context search; (2) leveraging "negative" gradients from incorrect traces to amplify exploration during RL, resulting in longer search traces that chains additional asymmetries; and (3) coupling task difficulty with training token budget during training via a specifically-designed curriculum to structure in-context exploration. Our recipe e3 produces the best known 1.7B model according to AIME'25 and HMMT'25 scores, and extrapolates to 2x the training token budget. Our e3-1.7B model not only attains high pass@1 scores, but also improves pass@k over the base model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11976v1">How Visual Representations Map to Language Feature Space in Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-13
    </div>
    <details class="paper-abstract">
      Effective multimodal reasoning depends on the alignment of visual and linguistic representations, yet the mechanisms by which vision-language models (VLMs) achieve this alignment remain poorly understood. We introduce a methodological framework that deliberately maintains a frozen large language model (LLM) and a frozen vision transformer (ViT), connected solely by training a linear adapter during visual instruction tuning. This design is fundamental to our approach: by keeping the language model frozen, we ensure it maintains its original language representations without adaptation to visual data. Consequently, the linear adapter must map visual features directly into the LLM's existing representational space rather than allowing the language model to develop specialized visual understanding through fine-tuning. Our experimental design uniquely enables the use of pre-trained sparse autoencoders (SAEs) of the LLM as analytical probes. These SAEs remain perfectly aligned with the unchanged language model and serve as a snapshot of the learned language feature-representations. Through systematic analysis of SAE reconstruction error, sparsity patterns, and feature SAE descriptions, we reveal the layer-wise progression through which visual representations gradually align with language feature representations, converging in middle-to-later layers. This suggests a fundamental misalignment between ViT outputs and early LLM layers, raising important questions about whether current adapter-based architectures optimally facilitate cross-modal representation learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10467v2">Specification and Evaluation of Multi-Agent LLM Systems -- Prototype and Cybersecurity Applications</a></div>
    <div class="paper-meta">
      📅 2025-06-13
      | 💬 This work has been submitted for a possible publication. Copyright may be transferred. In this case, this version will be updated with a notice, according to the publisher's guidelines
    </div>
    <details class="paper-abstract">
      Recent advancements in LLMs indicate potential for novel applications, e.g., through reasoning capabilities in the latest OpenAI and DeepSeek models. For applying these models in specific domains beyond text generation, LLM-based multi-agent approaches can be utilized that solve complex tasks by combining reasoning techniques, code generation, and software execution. Applications might utilize these capabilities and the knowledge of specialized LLM agents. However, while many evaluations are performed on LLMs, reasoning techniques, and applications individually, their joint specification and combined application is not explored well. Defined specifications for multi-agent LLM systems are required to explore their potential and their suitability for specific applications, allowing for systematic evaluations of LLMs, reasoning techniques, and related aspects. This paper reports the results of exploratory research to specify and evaluate these aspects through a multi-agent system. The system architecture and prototype are extended from previous research and a specification is introduced for multi-agent systems. Test cases involving cybersecurity tasks indicate feasibility of the architecture and evaluation approach. In particular, the results show the evaluation of question answering, server security, and network security tasks that were completed correctly by agents with LLMs from OpenAI and DeepSeek.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.01503v7">A Highly Scalable LLM Clusters with Optical Interconnect</a></div>
    <div class="paper-meta">
      📅 2025-06-13
    </div>
    <details class="paper-abstract">
      The rapid development of large-scale GPU clusters for LLM training has driven enterprises to replace core-layer electrical switches with optical circuit switches (OCS) to meet escalating bandwidth demands. However, current physical topology design of OCS-based clusters faces two critical challenges. First, there exist unrealizable logical topologies, leading to underutilization of bandwidth resource. Second, calculating OCS reconfiguration constitutes an NP-Complete problem and is time-consuming for multi-tenant GPU clusters which need real-time scheduling. In this paper, we propose \emph{Cross Wiring}, a new physical topology design that resolves both limitations. Our physical topology guarantees full compatibility with all logical topologies under L2-compatibility constraints. Through a proposed \emph{Symmetric Integer Matrix Decomposition Theorem}, we design a polynomial-time OCS reconfiguration algorithm that satisfies arbitrary logical topology requirements. Evaluations show a up to 39.5\% higher training throughput versus prior architectures such as \emph{Gemini} in 128-NPU testbed and a 12.6\% reduction in average job completion time through real-workload based multi-tenant large-scale simulations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11930v1">Feedback Friction: LLMs Struggle to Fully Incorporate External Feedback</a></div>
    <div class="paper-meta">
      📅 2025-06-13
    </div>
    <details class="paper-abstract">
      Recent studies have shown LLMs possess some ability to improve their responses when given external feedback. However, it remains unclear how effectively and thoroughly these models can incorporate extrinsic feedback. In an ideal scenario, if LLMs receive near-perfect and complete feedback, we would expect them to fully integrate the feedback and change their incorrect answers to correct ones. In this paper, we systematically investigate LLMs' ability to incorporate feedback by designing a controlled experimental environment. For each problem, a solver model attempts a solution, then a feedback generator with access to near-complete ground-truth answers produces targeted feedback, after which the solver tries again. We evaluate this pipeline across a diverse range of tasks, including math reasoning, knowledge reasoning, scientific reasoning, and general multi-domain evaluations with state-of-the-art language models including Claude 3.7 (with and without extended thinking). Surprisingly, even under these near-ideal conditions, solver models consistently show resistance to feedback, a limitation that we term FEEDBACK FRICTION. To mitigate this limitation, we experiment with sampling-based strategies like progressive temperature increases and explicit rejection of previously attempted incorrect answers, which yield improvements but still fail to help models achieve target performance. We also perform a rigorous exploration of potential causes of FEEDBACK FRICTION, ruling out factors such as model overconfidence and data familiarity. We hope that highlighting this issue in LLMs and ruling out several apparent causes will help future research in self-improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11928v1">LiveCodeBench Pro: How Do Olympiad Medalists Judge LLMs in Competitive Programming?</a></div>
    <div class="paper-meta">
      📅 2025-06-13
      | 💬 Project Page at https://livecodebenchpro.com/
    </div>
    <details class="paper-abstract">
      Recent reports claim that large language models (LLMs) now outperform elite humans in competitive programming. Drawing on knowledge from a group of medalists in international algorithmic contests, we revisit this claim, examining how LLMs differ from human experts and where limitations still remain. We introduce LiveCodeBench Pro, a benchmark composed of problems from Codeforces, ICPC, and IOI that are continuously updated to reduce the likelihood of data contamination. A team of Olympiad medalists annotates every problem for algorithmic categories and conducts a line-by-line analysis of failed model-generated submissions. Using this new data and benchmark, we find that frontier models still have significant limitations: without external tools, the best model achieves only 53% pass@1 on medium-difficulty problems and 0% on hard problems, domains where expert humans still excel. We also find that LLMs succeed at implementation-heavy problems but struggle with nuanced algorithmic reasoning and complex case analysis, often generating confidently incorrect justifications. High performance appears largely driven by implementation precision and tool augmentation, not superior reasoning. LiveCodeBench Pro thus highlights the significant gap to human grandmaster levels, while offering fine-grained diagnostics to steer future improvements in code-centric LLM reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11902v1">TreeRL: LLM Reinforcement Learning with On-Policy Tree Search</a></div>
    <div class="paper-meta">
      📅 2025-06-13
      | 💬 Accepted to ACL 2025 main conference
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) with tree search has demonstrated superior performance in traditional reasoning tasks. Compared to conventional independent chain sampling strategies with outcome supervision, tree search enables better exploration of the reasoning space and provides dense, on-policy process rewards during RL training but remains under-explored in On-Policy LLM RL. We propose TreeRL, a reinforcement learning framework that directly incorporates on-policy tree search for RL training. Our approach includes intermediate supervision and eliminates the need for a separate reward model training. Existing approaches typically train a separate process reward model, which can suffer from distribution mismatch and reward hacking. We also introduce a cost-effective tree search approach that achieves higher search efficiency under the same generation token budget by strategically branching from high-uncertainty intermediate steps rather than using random branching. Experiments on challenging math and code reasoning benchmarks demonstrate that TreeRL achieves superior performance compared to traditional ChainRL, highlighting the potential of tree search for LLM. TreeRL is open-sourced at https://github.com/THUDM/TreeRL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.18638v2">Graph of Attacks with Pruning: Optimizing Stealthy Jailbreak Prompt Generation for Enhanced LLM Content Moderation</a></div>
    <div class="paper-meta">
      📅 2025-06-13
      | 💬 14 pages, 5 figures
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become increasingly prevalent, ensuring their robustness against adversarial misuse is crucial. This paper introduces the GAP (Graph of Attacks with Pruning) framework, an advanced approach for generating stealthy jailbreak prompts to evaluate and enhance LLM safeguards. GAP addresses limitations in existing tree-based LLM jailbreak methods by implementing an interconnected graph structure that enables knowledge sharing across attack paths. Our experimental evaluation demonstrates GAP's superiority over existing techniques, achieving a 20.8% increase in attack success rates while reducing query costs by 62.7%. GAP consistently outperforms state-of-the-art methods for attacking both open and closed LLMs, with attack success rates of >96%. Additionally, we present specialized variants like GAP-Auto for automated seed generation and GAP-VLM for multimodal attacks. GAP-generated prompts prove highly effective in improving content moderation systems, increasing true positive detection rates by 108.5% and accuracy by 183.6% when used for fine-tuning. Our implementation is available at https://github.com/dsbuddy/GAP-LLM-Safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.02509v2">Black-Box Adversarial Attacks on LLM-Based Code Completion</a></div>
    <div class="paper-meta">
      📅 2025-06-13
    </div>
    <details class="paper-abstract">
      Modern code completion engines, powered by large language models (LLMs), assist millions of developers with their strong capabilities to generate functionally correct code. Due to this popularity, it is crucial to investigate the security implications of relying on LLM-based code completion. In this work, we demonstrate that state-of-the-art black-box LLM-based code completion engines can be stealthily biased by adversaries to significantly increase their rate of insecure code generation. We present the first attack, named INSEC, that achieves this goal. INSEC works by injecting an attack string as a short comment in the completion input. The attack string is crafted through a query-based optimization procedure starting from a set of carefully designed initialization schemes. We demonstrate INSEC's broad applicability and effectiveness by evaluating it on various state-of-the-art open-source models and black-box commercial services (e.g., OpenAI API and GitHub Copilot). On a diverse set of security-critical test cases, covering 16 CWEs across 5 programming languages, INSEC increases the rate of generated insecure code by more than 50%, while maintaining the functional correctness of generated code. We consider INSEC practical -- it requires low resources and costs less than 10 US dollars to develop on commodity hardware. Moreover, we showcase the attack's real-world deployability, by developing an IDE plug-in that stealthily injects INSEC into the GitHub Copilot extension.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11887v1">Towards a Cascaded LLM Framework for Cost-effective Human-AI Decision-Making</a></div>
    <div class="paper-meta">
      📅 2025-06-13
    </div>
    <details class="paper-abstract">
      Effective human-AI decision-making balances three key factors: the \textit{correctness} of predictions, the \textit{cost} of knowledge and reasoning complexity, and the confidence about whether to \textit{abstain} automated answers or involve human experts. In this work, we present a cascaded LLM decision framework that adaptively delegates tasks across multiple tiers of expertise -- a base model for initial candidate answers, a more capable and knowledgeable (but costlier) large model, and a human expert for when the model cascade abstains. Our method proceeds in two stages. First, a deferral policy determines whether to accept the base model's answer or regenerate it with the large model based on the confidence score. Second, an abstention policy decides whether the cascade model response is sufficiently certain or requires human intervention. Moreover, we incorporate an online learning mechanism in the framework that can leverage human feedback to improve decision quality over time. We demonstrate this approach to general question-answering (ARC-Easy and ARC-Challenge) and medical question-answering (MedQA and MedMCQA). Our results show that our cascaded strategy outperforms in most cases single-model baselines in accuracy while reducing cost and providing a principled way to handle abstentions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11880v1">Addressing Bias in LLMs: Strategies and Application to Fair AI-based Recruitment</a></div>
    <div class="paper-meta">
      📅 2025-06-13
      | 💬 Submitted to AIES 2025 (Under Review)
    </div>
    <details class="paper-abstract">
      The use of language technologies in high-stake settings is increasing in recent years, mostly motivated by the success of Large Language Models (LLMs). However, despite the great performance of LLMs, they are are susceptible to ethical concerns, such as demographic biases, accountability, or privacy. This work seeks to analyze the capacity of Transformers-based systems to learn demographic biases present in the data, using a case study on AI-based automated recruitment. We propose a privacy-enhancing framework to reduce gender information from the learning pipeline as a way to mitigate biased behaviors in the final tools. Our experiments analyze the influence of data biases on systems built on two different LLMs, and how the proposed framework effectively prevents trained systems from reproducing the bias in the data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11870v1">LLM-based Dynamic Differential Testing for Database Connectors with Reinforcement Learning-Guided Prompt Selection</a></div>
    <div class="paper-meta">
      📅 2025-06-13
      | 💬 5 pages
    </div>
    <details class="paper-abstract">
      Database connectors are critical components enabling applications to interact with underlying database management systems (DBMS), yet their security vulnerabilities often remain overlooked. Unlike traditional software defects, connector vulnerabilities exhibit subtle behavioral patterns and are inherently challenging to detect. Besides, nonstandardized implementation of connectors leaves potential risks (a.k.a. unsafe implementations) but is more elusive. As a result, traditional fuzzing methods are incapable of finding such vulnerabilities. Even for LLM-enable test case generation, due to a lack of domain knowledge, they are also incapable of generating test cases that invoke all interface and internal logic of connectors. In this paper, we propose reinforcement learning (RL)-guided LLM test-case generation for database connector testing. Specifically, to equip the LLM with sufficient and appropriate domain knowledge, a parameterized prompt template is composed which can be utilized to generate numerous prompts. Test cases are generated via LLM with a prompt, and are dynamically evaluated through differential testing across multiple connectors. The testing is iteratively conducted, with each round RL is adopted to select optimal prompt based on prior-round behavioral feedback, so as to maximize control flow coverage. We implement aforementioned methodology in a practical tool and evaluate it on two widely used JDBC connectors: MySQL Connector/J and OceanBase Connector/J. In total, we reported 16 bugs, among them 10 are officially confirmed and the rest are acknowledged as unsafe implementations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09347v2">Safer or Luckier? LLMs as Safety Evaluators Are Not Robust to Artifacts</a></div>
    <div class="paper-meta">
      📅 2025-06-13
      | 💬 9 pages, ACL 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly employed as automated evaluators to assess the safety of generated content, yet their reliability in this role remains uncertain. This study evaluates a diverse set of 11 LLM judge models across critical safety domains, examining three key aspects: self-consistency in repeated judging tasks, alignment with human judgments, and susceptibility to input artifacts such as apologetic or verbose phrasing. Our findings reveal that biases in LLM judges can significantly distort the final verdict on which content source is safer, undermining the validity of comparative evaluations. Notably, apologetic language artifacts alone can skew evaluator preferences by up to 98\%. Contrary to expectations, larger models do not consistently exhibit greater robustness, while smaller models sometimes show higher resistance to specific artifacts. To mitigate LLM evaluator robustness issues, we investigate jury-based evaluations aggregating decisions from multiple models. Although this approach both improves robustness and enhances alignment to human judgements, artifact sensitivity persists even with the best jury configurations. These results highlight the urgent need for diversified, artifact-resistant methodologies to ensure reliable safety assessments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11825v1">Revealing Political Bias in LLMs through Structured Multi-Agent Debate</a></div>
    <div class="paper-meta">
      📅 2025-06-13
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to simulate social behaviour, yet their political biases and interaction dynamics in debates remain underexplored. We investigate how LLM type and agent gender attributes influence political bias using a structured multi-agent debate framework, by engaging Neutral, Republican, and Democrat American LLM agents in debates on politically sensitive topics. We systematically vary the underlying LLMs, agent genders, and debate formats to examine how model provenance and agent personas influence political bias and attitudes throughout debates. We find that Neutral agents consistently align with Democrats, while Republicans shift closer to the Neutral; gender influences agent attitudes, with agents adapting their opinions when aware of other agents' genders; and contrary to prior research, agents with shared political affiliations can form echo chambers, exhibiting the expected intensification of attitudes as debates progress.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.19164v2">Learning from Litigation: Graphs and LLMs for Retrieval and Reasoning in eDiscovery</a></div>
    <div class="paper-meta">
      📅 2025-06-13
      | 💬 Updated with Camera Ready Copy for ACL 2025
    </div>
    <details class="paper-abstract">
      Electronic Discovery (eDiscovery) requires identifying relevant documents from vast collections for legal production requests. While artificial intelligence (AI) and natural language processing (NLP) have improved document review efficiency, current methods still struggle with legal entities, citations, and complex legal artifacts. To address these challenges, we introduce DISCOvery Graph (DISCOG), an emerging system that integrates knowledge graphs for enhanced document ranking and classification, augmented by LLM-driven reasoning. DISCOG outperforms strong baselines in F1-score, precision, and recall across both balanced and imbalanced datasets. In real-world deployments, it has reduced litigation-related document review costs by approximately 98\%, demonstrating significant business impact.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11812v1">On the Performance of LLMs for Real Estate Appraisal</a></div>
    <div class="paper-meta">
      📅 2025-06-13
      | 💬 Accepted at ECML-PKDD 2025
    </div>
    <details class="paper-abstract">
      The real estate market is vital to global economies but suffers from significant information asymmetry. This study examines how Large Language Models (LLMs) can democratize access to real estate insights by generating competitive and interpretable house price estimates through optimized In-Context Learning (ICL) strategies. We systematically evaluate leading LLMs on diverse international housing datasets, comparing zero-shot, few-shot, market report-enhanced, and hybrid prompting techniques. Our results show that LLMs effectively leverage hedonic variables, such as property size and amenities, to produce meaningful estimates. While traditional machine learning models remain strong for pure predictive accuracy, LLMs offer a more accessible, interactive and interpretable alternative. Although self-explanations require cautious interpretation, we find that LLMs explain their predictions in agreement with state-of-the-art models, confirming their trustworthiness. Carefully selected in-context examples based on feature similarity and geographic proximity, significantly enhance LLM performance, yet LLMs struggle with overconfidence in price intervals and limited spatial reasoning. We offer practical guidance for structured prediction tasks through prompt optimization. Our findings highlight LLMs' potential to improve transparency in real estate appraisal and provide actionable insights for stakeholders.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11791v1">SEC-bench: Automated Benchmarking of LLM Agents on Real-World Software Security Tasks</a></div>
    <div class="paper-meta">
      📅 2025-06-13
    </div>
    <details class="paper-abstract">
      Rigorous security-focused evaluation of large language model (LLM) agents is imperative for establishing trust in their safe deployment throughout the software development lifecycle. However, existing benchmarks largely rely on synthetic challenges or simplified vulnerability datasets that fail to capture the complexity and ambiguity encountered by security engineers in practice. We introduce SEC-bench, the first fully automated benchmarking framework for evaluating LLM agents on authentic security engineering tasks. SEC-bench employs a novel multi-agent scaffold that automatically constructs code repositories with harnesses, reproduces vulnerabilities in isolated environments, and generates gold patches for reliable evaluation. Our framework automatically creates high-quality software vulnerability datasets with reproducible artifacts at a cost of only $0.87 per instance. Using SEC-bench, we implement two critical software security tasks to rigorously evaluate LLM agents' capabilities: proof-of-concept (PoC) generation and vulnerability patching. A comprehensive evaluation of state-of-the-art LLM code agents reveals significant performance gaps, achieving at most 18.0% success in PoC generation and 34.0% in vulnerability patching on our complete dataset. These results highlight the crucial steps needed toward developing LLM agents that are more practical, intelligent, and autonomous for security engineering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11789v1">Conversational AI as a Catalyst for Informal Learning: An Empirical Large-Scale Study on LLM Use in Everyday Learning</a></div>
    <div class="paper-meta">
      📅 2025-06-13
    </div>
    <details class="paper-abstract">
      Large language models have not only captivated the public imagination but have also sparked a profound rethinking of how we learn. In the third year following the breakthrough launch of ChatGPT, everyday informal learning has been transformed as diverse user groups explore these novel tools. Who is embracing LLMs for self-directed learning, and who remains hesitant? What are their reasons for adoption or avoidance? What learning patterns emerge with this novel technological landscape? We present an in-depth analysis from a large-scale survey of 776 participants, showcasing that 88% of our respondents already incorporate LLMs into their everyday learning routines for a wide variety of (learning) tasks. Young adults are at the forefront of adopting LLMs, primarily to enhance their learning experiences independently of time and space. Four types of learners emerge across learning contexts, depending on the tasks they perform with LLMs and the devices they use to access them. Interestingly, our respondents exhibit paradoxical behaviours regarding their trust in LLMs' accuracy and privacy protection measures. Our implications emphasize the importance of including different media types for learning, enabling collaborative learning, providing sources and meeting the needs of different types of learners and learning by design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11781v1">GeoPandas-AI: A Smart Class Bringing LLM as Stateful AI Code Assistant</a></div>
    <div class="paper-meta">
      📅 2025-06-13
      | 💬 Submitted to ACM SIGSPATIAL 2025
    </div>
    <details class="paper-abstract">
      Geospatial data analysis plays a crucial role in tackling intricate societal challenges such as urban planning and climate modeling. However, employing tools like GeoPandas, a prominent Python library for geospatial data manipulation, necessitates expertise in complex domain-specific syntax and workflows. GeoPandas-AI addresses this gap by integrating LLMs directly into the GeoPandas workflow, transforming the GeoDataFrame class into an intelligent, stateful class for both data analysis and geospatial code development. This paper formalizes the design of such a smart class and provides an open-source implementation of GeoPandas-AI in PyPI package manager. Through its innovative combination of conversational interfaces and stateful exploitation of LLMs for code generation and data analysis, GeoPandas-AI introduces a new paradigm for code-copilots and instantiates it for geospatial development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11769v1">Long-Short Alignment for Effective Long-Context Modeling in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-13
      | 💬 ICML 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have exhibited impressive performance and surprising emergent properties. However, their effectiveness remains limited by the fixed context window of the transformer architecture, posing challenges for long-context modeling. Among these challenges, length generalization -- the ability to generalize to sequences longer than those seen during training -- is a classical and fundamental problem. In this work, we propose a fresh perspective on length generalization, shifting the focus from the conventional emphasis on input features such as positional encodings or data structures to the output distribution of the model. Specifically, through case studies on synthetic tasks, we highlight the critical role of \textbf{long-short alignment} -- the consistency of output distributions across sequences of varying lengths. Extending this insight to natural language tasks, we propose a metric called Long-Short Misalignment to quantify this phenomenon, uncovering a strong correlation between the metric and length generalization performance. Building on these findings, we develop a regularization term that promotes long-short alignment during training. Extensive experiments validate the effectiveness of our approach, offering new insights for achieving more effective long-context modeling in LLMs. Code is available at https://github.com/PKU-ML/LongShortAlignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11767v1">Designing Effective LLM-Assisted Interfaces for Curriculum Development</a></div>
    <div class="paper-meta">
      📅 2025-06-13
      | 💬 This is the preprint version of a paper accepted at AIED 2025. The final version will be published by Springer
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have the potential to transform the way a dynamic curriculum can be delivered. However, educators face significant challenges in interacting with these models, particularly due to complex prompt engineering and usability issues, which increase workload. Additionally, inaccuracies in LLM outputs can raise issues around output quality and ethical concerns in educational content delivery. Addressing these issues requires careful oversight, best achieved through cooperation between human and AI approaches. This paper introduces two novel User Interface (UI) designs, UI Predefined and UI Open, both grounded in Direct Manipulation (DM) principles to address these challenges. By reducing the reliance on intricate prompt engineering, these UIs improve usability, streamline interaction, and lower workload, providing a more effective pathway for educators to engage with LLMs. In a controlled user study with 20 participants, the proposed UIs were evaluated against the standard ChatGPT interface in terms of usability and cognitive load. Results showed that UI Predefined significantly outperformed both ChatGPT and UI Open, demonstrating superior usability and reduced task load, while UI Open offered more flexibility at the cost of a steeper learning curve. These findings underscore the importance of user-centered design in adopting AI-driven tools and lay the foundation for more intuitive and efficient educator-LLM interactions in online learning environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11722v1">Classification of Quality Characteristics in Online User Feedback using Linguistic Analysis, Crowdsourcing and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-13
      | 💬 Accepted at the Journal of Systems and Software (JSS); online appendix and supplementary material available at https://doi.org/10.5281/zenodo.15604749
    </div>
    <details class="paper-abstract">
      Software qualities such as usability or reliability are among the strongest determinants of mobile app user satisfaction and constitute a significant portion of online user feedback on software products, making it a valuable source of quality-related feedback to guide the development process. The abundance of online user feedback warrants the automated identification of quality characteristics, but the online user feedback's heterogeneity and the lack of appropriate training corpora limit the applicability of supervised machine learning. We therefore investigate the viability of three approaches that could be effective in low-data settings: language patterns (LPs) based on quality-related keywords, instructions for crowdsourced micro-tasks, and large language model (LLM) prompts. We determined the feasibility of each approach and then compared their accuracy. For the complex multiclass classification of quality characteristics, the LP-based approach achieved a varied precision (0.38-0.92) depending on the quality characteristic, and low recall; crowdsourcing achieved the best average accuracy in two consecutive phases (0.63, 0.72), which could be matched by the best-performing LLM condition (0.66) and a prediction based on the LLMs' majority vote (0.68). Our findings show that in this low-data setting, the two approaches that use crowdsourcing or LLMs instead of involving experts achieve accurate classifications, while the LP-based approach has only limited potential. The promise of crowdsourcing and LLMs in this context might even extend to building training corpora.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11680v1">Malicious LLM-Based Conversational AI Makes Users Reveal Personal Information</a></div>
    <div class="paper-meta">
      📅 2025-06-13
      | 💬 This paper has been accepted at USENIX Security '25
    </div>
    <details class="paper-abstract">
      LLM-based Conversational AIs (CAIs), also known as GenAI chatbots, like ChatGPT, are increasingly used across various domains, but they pose privacy risks, as users may disclose personal information during their conversations with CAIs. Recent research has demonstrated that LLM-based CAIs could be used for malicious purposes. However, a novel and particularly concerning type of malicious LLM application remains unexplored: an LLM-based CAI that is deliberately designed to extract personal information from users. In this paper, we report on the malicious LLM-based CAIs that we created based on system prompts that used different strategies to encourage disclosures of personal information from users. We systematically investigate CAIs' ability to extract personal information from users during conversations by conducting a randomized-controlled trial with 502 participants. We assess the effectiveness of different malicious and benign CAIs to extract personal information from participants, and we analyze participants' perceptions after their interactions with the CAIs. Our findings reveal that malicious CAIs extract significantly more personal information than benign CAIs, with strategies based on the social nature of privacy being the most effective while minimizing perceived risks. This study underscores the privacy threats posed by this novel type of malicious LLM-based CAIs and provides actionable recommendations to guide future research and practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11679v1">LLMs on support of privacy and security of mobile apps: state of the art and research directions</a></div>
    <div class="paper-meta">
      📅 2025-06-13
    </div>
    <details class="paper-abstract">
      Modern life has witnessed the explosion of mobile devices. However, besides the valuable features that bring convenience to end users, security and privacy risks still threaten users of mobile apps. The increasing sophistication of these threats in recent years has underscored the need for more advanced and efficient detection approaches. In this chapter, we explore the application of Large Language Models (LLMs) to identify security risks and privacy violations and mitigate them for the mobile application ecosystem. By introducing state-of-the-art research that applied LLMs to mitigate the top 10 common security risks of smartphone platforms, we highlight the feasibility and potential of LLMs to replace traditional analysis methods, such as dynamic and hybrid analysis of mobile apps. As a representative example of LLM-based solutions, we present an approach to detect sensitive data leakage when users share images online, a common behavior of smartphone users nowadays. Finally, we discuss open research challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11659v1">An Empirical study on LLM-based Log Retrieval for Software Engineering Metadata Management</a></div>
    <div class="paper-meta">
      📅 2025-06-13
    </div>
    <details class="paper-abstract">
      Developing autonomous driving systems (ADSs) involves generating and storing extensive log data from test drives, which is essential for verification, research, and simulation. However, these high-frequency logs, recorded over varying durations, pose challenges for developers attempting to locate specific driving scenarios. This difficulty arises due to the wide range of signals representing various vehicle components and driving conditions, as well as unfamiliarity of some developers' with the detailed meaning of these signals. Traditional SQL-based querying exacerbates this challenge by demanding both domain expertise and database knowledge, often yielding results that are difficult to verify for accuracy. This paper introduces a Large Language Model (LLM)-supported approach that combines signal log data with video recordings from test drives, enabling natural language based scenario searches while reducing the need for specialized knowledge. By leveraging scenario distance graphs and relative gap indicators, it provides quantifiable metrics to evaluate the reliability of query results. The method is implemented as an API for efficient database querying and retrieval of relevant records, paired with video frames for intuitive visualization. Evaluation on an open industrial dataset demonstrates improved efficiency and reliability in scenario retrieval, eliminating dependency on a single data source and conventional SQL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23549v2">LLM-based Property-based Test Generation for Guardrailing Cyber-Physical Systems</a></div>
    <div class="paper-meta">
      📅 2025-06-13
    </div>
    <details class="paper-abstract">
      Cyber-physical systems (CPSs) are complex systems that integrate physical, computational, and communication subsystems. The heterogeneous nature of these systems makes their safety assurance challenging. In this paper, we propose a novel automated approach for guardrailing cyber-physical systems using property-based tests (PBTs) generated by Large Language Models (LLMs). Our approach employs an LLM to extract properties from the code and documentation of CPSs. Next, we use the LLM to generate PBTs that verify the extracted properties on the CPS. The generated PBTs have two uses. First, they are used to test the CPS before it is deployed, i.e., at design time. Secondly, these PBTs can be used after deployment, i.e., at run time, to monitor the behavior of the system and guardrail it against unsafe states. We implement our approach in ChekProp and conduct preliminary experiments to evaluate the generated PBTs in terms of their relevance (how well they match manually crafted properties), executability (how many run with minimal manual modification), and effectiveness (coverage of the input space partitions). The results of our experiments and evaluation demonstrate a promising path forward for creating guardrails for CPSs using LLM-generated property-based tests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11812v2">Towards Understanding Fine-Tuning Mechanisms of LLMs via Circuit Analysis</a></div>
    <div class="paper-meta">
      📅 2025-06-13
      | 💬 25 pages
    </div>
    <details class="paper-abstract">
      Fine-tuning significantly improves the performance of Large Language Models (LLMs), yet its underlying mechanisms remain poorly understood. This paper aims to provide an in-depth interpretation of the fine-tuning process through circuit analysis, a popular tool in Mechanistic Interpretability (MI). Unlike previous studies (Prakash et al. 2024; Chhabra et al. 2024) that focus on tasks where pre-trained models already perform well, we develop a set of mathematical tasks where fine-tuning yields substantial performance gains, which are closer to the practical setting. In our experiments, we identify circuits at various checkpoints during fine-tuning and examine the interplay between circuit analysis, fine-tuning methods, and task complexities. First, we find that while circuits maintain high node similarity before and after fine-tuning, their edges undergo significant changes, in contrast to prior work that shows circuits only add some additional components after fine-tuning. Based on these observations, we develop a circuit-aware Low-Rank Adaptation (LoRA) method, which assigns ranks to layers based on edge changes in the circuits. Experimental results demonstrate that our circuit-based LoRA algorithm achieves an average performance improvement of 2.46% over standard LoRA with similar parameter sizes. Furthermore, we explore how combining circuits from subtasks can enhance fine-tuning in compositional tasks, providing new insights into the design of such tasks and deepening the understanding of circuit dynamics and fine-tuning mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11602v1">Are LLMs Good Text Diacritizers? An Arabic and Yorùbá Case Study</a></div>
    <div class="paper-meta">
      📅 2025-06-13
    </div>
    <details class="paper-abstract">
      We investigate the effectiveness of large language models (LLMs) for text diacritization in two typologically distinct languages: Arabic and Yoruba. To enable a rigorous evaluation, we introduce a novel multilingual dataset MultiDiac, with diverse samples that capture a range of diacritic ambiguities. We evaluate 14 LLMs varying in size, accessibility, and language coverage, and benchmark them against 6 specialized diacritization models. Additionally, we fine-tune four small open-source models using LoRA for Yoruba. Our results show that many off-the-shelf LLMs outperform specialized diacritization models for both Arabic and Yoruba, but smaller models suffer from hallucinations. Fine-tuning on a small dataset can help improve diacritization performance and reduce hallucination rates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.11521v3">Preempting Text Sanitization Utility in Resource-Constrained Privacy-Preserving LLM Interactions</a></div>
    <div class="paper-meta">
      📅 2025-06-13
    </div>
    <details class="paper-abstract">
      Interactions with online Large Language Models raise privacy issues where providers can gather sensitive information about users and their companies from the prompts. While textual prompts can be sanitized using Differential Privacy, we show that it is difficult to anticipate the performance of an LLM on such sanitized prompt. Poor performance has clear monetary consequences for LLM services charging on a pay-per-use model as well as great amount of computing resources wasted. To this end, we propose a middleware architecture leveraging a Small Language Model to predict the utility of a given sanitized prompt before it is sent to the LLM. We experimented on a summarization task and a translation task to show that our architecture helps prevent such resource waste for up to 20% of the prompts. During our study, we also reproduced experiments from one of the most cited paper on text sanitization using DP and show that a potential performance-driven implementation choice dramatically changes the output while not being explicitly acknowledged in the paper.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11578v1">Collaborative LLM Inference via Planning for Efficient Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-06-13
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel at complex reasoning tasks, but those with strong capabilities (e.g., whose numbers of parameters are larger than 100B) are often accessible only through paid APIs, making them too costly for applications of frequent use. In contrast, smaller open-sourced LLMs (e.g., whose numbers of parameters are less than 3B) are freely available and easy to deploy locally (e.g., under a single GPU having 8G VRAM), but lack suff icient reasoning ability. This trade-off raises a natural question: can small (free) and large (costly) models collaborate at test time to combine their strengths? We propose a test-time collaboration framework in which a planner model first generates a plan, defined as a distilled and high-level abstraction of the problem. This plan serves as a lightweight intermediate that guides a reasoner model, which generates a complete solution. Small and large models take turns acting as planner and reasoner, exchanging plans in a multi-round cascade to collaboratively solve complex tasks. Our method achieves accuracy comparable to strong proprietary models alone, while significantly reducing reliance on paid inference. These results highlight planning as an effective prior for orchestrating cost-aware, cross-model inference under real-world deployment constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11561v1">Identifying Helpful Context for LLM-based Vulnerability Repair: A Preliminary Study</a></div>
    <div class="paper-meta">
      📅 2025-06-13
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have shown promise for automated vulnerability detection and repair in software systems. This paper investigates the performance of GPT-4o in repairing Java vulnerabilities from a widely used dataset (Vul4J), exploring how different contextual information affects automated vulnerability repair (AVR) capabilities. We compare the latest GPT-4o's performance against previous results with GPT-4 using identical prompts. We evaluated nine additional prompts crafted by us that contain various contextual information such as CWE or CVE information, and manually extracted code contexts. Each prompt was executed three times on 42 vulnerabilities, and the resulting fix candidates were validated using Vul4J's automated testing framework. Our results show that GPT-4o performed 11.9\% worse on average than GPT-4 with the same prompt, but was able to fix 10.5\% more distinct vulnerabilities in the three runs together. CVE information significantly improved repair rates, while the length of the task description had minimal impact. Combining CVE guidance with manually extracted code context resulted in the best performance. Using our \textsc{Top}-3 prompts together, GPT-4o repaired 26 (62\%) vulnerabilities at least once, outperforming both the original baseline (40\%) and its reproduction (45\%), suggesting that ensemble prompt strategies could improve vulnerability repair in zero-shot settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11558v1">DaMO: A Data-Efficient Multimodal Orchestrator for Temporal Reasoning with Video LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-13
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently been extended to the video domain, enabling sophisticated video-language understanding. However, existing Video LLMs often exhibit limitations in fine-grained temporal reasoning, restricting their ability to precisely attribute responses to specific video moments, especially under constrained supervision. We introduce DaMO, a data-efficient Video LLM explicitly designed for accurate temporal reasoning and multimodal understanding. At its core, the proposed Temporal-aware Fuseformer employs a hierarchical dual-stream architecture that progressively captures temporal dynamics within each modality and effectively fuses complementary visual and audio information. To further enhance computational efficiency, DaMO integrates a global residual that reduces spatial redundancy while preserving essential semantic details. We train DaMO via a structured four-stage progressive training paradigm, incrementally equipping the model with multimodal alignment, semantic grounding, and temporal reasoning capabilities. This work also contributes multiple datasets augmented from existing ones with GPT-generated temporally grounded QA pairs for tasks requiring temporal supervision. Comprehensive experiments on temporal grounding and video QA benchmarks demonstrate that DaMO consistently surpasses prior methods, particularly in tasks demanding precise temporal alignment and reasoning. Our work establishes a promising direction for data-efficient video-language modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.20445v3">TrajAgent: An LLM-based Agent Framework for Automated Trajectory Modeling via Collaboration of Large and Small Models</a></div>
    <div class="paper-meta">
      📅 2025-06-13
      | 💬 the code will be openly accessible at: https://github.com/tsinghua-fib-lab/TrajAgent
    </div>
    <details class="paper-abstract">
      Trajectory modeling, which includes research on trajectory data pattern mining and future prediction, has widespread applications in areas such as life services, urban transportation, and public administration. Numerous methods have been proposed to address specific problems within trajectory modeling. However, the heterogeneity of data and the diversity of trajectory tasks make effective and reliable trajectory modeling an important yet highly challenging endeavor, even for domain experts. In this paper, we propose \textit{TrajAgent}, a agent framework powered by large language models (LLMs), designed to facilitate robust and efficient trajectory modeling through automation modeling. This framework leverages and optimizes diverse specialized models to address various trajectory modeling tasks across different datasets effectively. In \textit{TrajAgent}, we first develop \textit{UniEnv}, an execution environment with a unified data and model interface, to support the execution and training of various models. Building on \textit{UniEnv}, we introduce an agentic workflow designed for automatic trajectory modeling across various trajectory tasks and data. Furthermore, we introduce collaborative learning schema between LLM-based agents and small speciallized models, to enhance the performance of the whole framework effectively. Extensive experiments on four tasks using four real-world datasets demonstrate the effectiveness of \textit{TrajAgent} in automated trajectory modeling, achieving a performance improvement of 2.38\%-34.96\% over baseline methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04078v2">LLMEval-Med: A Real-world Clinical Benchmark for Medical LLMs with Physician Validation</a></div>
    <div class="paper-meta">
      📅 2025-06-13
    </div>
    <details class="paper-abstract">
      Evaluating large language models (LLMs) in medicine is crucial because medical applications require high accuracy with little room for error. Current medical benchmarks have three main types: medical exam-based, comprehensive medical, and specialized assessments. However, these benchmarks have limitations in question design (mostly multiple-choice), data sources (often not derived from real clinical scenarios), and evaluation methods (poor assessment of complex reasoning). To address these issues, we present LLMEval-Med, a new benchmark covering five core medical areas, including 2,996 questions created from real-world electronic health records and expert-designed clinical scenarios. We also design an automated evaluation pipeline, incorporating expert-developed checklists into our LLM-as-Judge framework. Furthermore, our methodology validates machine scoring through human-machine agreement analysis, dynamically refining checklists and prompts based on expert feedback to ensure reliability. We evaluate 13 LLMs across three categories (specialized medical models, open-source models, and closed-source models) on LLMEval-Med, providing valuable insights for the safe and effective deployment of LLMs in medical domains. The dataset is released in https://github.com/llmeval/LLMEval-Med.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11512v1">Prioritizing Alignment Paradigms over Task-Specific Model Customization in Time-Series LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-13
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have enabled unprecedented capabilities for time-series reasoning in diverse real-world applications, including medical, financial, and spatio-temporal domains. However, existing approaches typically focus on task-specific model customization, such as forecasting and anomaly detection, while overlooking the data itself, referred to as time-series primitives, which are essential for in-depth reasoning. This position paper advocates a fundamental shift in approaching time-series reasoning with LLMs: prioritizing alignment paradigms grounded in the intrinsic primitives of time series data over task-specific model customization. This realignment addresses the core limitations of current time-series reasoning approaches, which are often costly, inflexible, and inefficient, by systematically accounting for intrinsic structure of data before task engineering. To this end, we propose three alignment paradigms: Injective Alignment, Bridging Alignment, and Internal Alignment, which are emphasized by prioritizing different aspects of time-series primitives: domain, characteristic, and representation, respectively, to activate time-series reasoning capabilities of LLMs to enable economical, flexible, and efficient reasoning. We further recommend that practitioners adopt an alignment-oriented method to avail this instruction to select an appropriate alignment paradigm. Additionally, we categorize relevant literature into these alignment paradigms and outline promising research directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06877v4">LLM-Assisted Relevance Assessments: When Should We Ask LLMs for Help?</a></div>
    <div class="paper-meta">
      📅 2025-06-13
      | 💬 11 pages. Accepted at SIGIR 2025 (48th International ACM SIGIR Conference on Research and Development in Information Retrieval)
    </div>
    <details class="paper-abstract">
      Test collections are information-retrieval tools that allow researchers to quickly and easily evaluate ranking algorithms. While test collections have become an integral part of IR research, the process of data creation involves significant manual-annotation effort, which often makes it very expensive and time-consuming. Consequently, test collections can become too small when the budget is limited, which may lead to unstable evaluations. As a cheaper alternative, recent studies have proposed using large language models (LLMs) to completely replace human assessors. However, while LLMs correlate to some extent with human judgments, their predictions are not perfect and often show bias. Thus, a complete replacement with LLMs is considered too risky and not fully reliable. In this paper, we propose LLM-Assisted Relevance Assessments (LARA), an effective method to balance manual annotations with LLM annotations, helping build a rich and reliable test collection even under a low budget. We use the LLM's predicted relevance probabilities to select the most profitable documents for manual annotation under a budget constraint. Guided by theoretical reasoning, LARA actively learns to calibrate the LLM's predicted relevance probabilities, directing the human-annotation process. Then, using the calibration model learned from the limited manual annotations, LARA debiases the LLM predictions to annotate the remaining non-assessed data. Experiments on TREC-7 Ad Hoc, TREC-8 Ad Hoc, TREC Robust 2004, and TREC-COVID datasets show that LARA outperforms alternative solutions under almost any budget constraint. While the community debates humans versus LLMs in relevance assessments, we contend that, given the same amount of human effort, it is reasonable to leverage LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.03505v2">Dynamic and Adaptive Feature Generation with LLM</a></div>
    <div class="paper-meta">
      📅 2025-06-13
      | 💬 Accepted by IJCAI 2025
    </div>
    <details class="paper-abstract">
      The representation of feature space is a crucial environment where data points get vectorized and embedded for subsequent modeling. Thus the efficacy of machine learning (ML) algorithms is closely related to the quality of feature engineering. As one of the most important techniques, feature generation transforms raw data into an optimized feature space conducive to model training and further refines the space. Despite the advancements in automated feature engineering and feature generation, current methodologies often suffer from three fundamental issues: lack of explainability, limited applicability, and inflexible strategy. These shortcomings frequently hinder and limit the deployment of ML models across varied scenarios. Our research introduces a novel approach adopting large language models (LLMs) and feature-generating prompts to address these challenges. We propose a dynamic and adaptive feature generation method that enhances the interpretability of the feature generation process. Our approach broadens the applicability across various data types and tasks and offers advantages over strategic flexibility. A broad range of experiments showcases that our approach is significantly superior to existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10635v1">Conversational Search: From Fundamentals to Frontiers in the LLM Era</a></div>
    <div class="paper-meta">
      📅 2025-06-12
      | 💬 Accepted by Tutorial Track in SIGIR 2025
    </div>
    <details class="paper-abstract">
      Conversational search enables multi-turn interactions between users and systems to fulfill users' complex information needs. During this interaction, the system should understand the users' search intent within the conversational context and then return the relevant information through a flexible, dialogue-based interface. The recent powerful large language models (LLMs) with capacities of instruction following, content generation, and reasoning, attract significant attention and advancements, providing new opportunities and challenges for building up intelligent conversational search systems. This tutorial aims to introduce the connection between fundamentals and the emerging topics revolutionized by LLMs in the context of conversational search. It is designed for students, researchers, and practitioners from both academia and industry. Participants will gain a comprehensive understanding of both the core principles and cutting-edge developments driven by LLMs in conversational search, equipping them with the knowledge needed to contribute to the development of next-generation conversational search systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10630v1">Time Series Forecasting as Reasoning: A Slow-Thinking Approach with Reinforced LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-12
    </div>
    <details class="paper-abstract">
      To advance time series forecasting (TSF), various methods have been proposed to improve prediction accuracy, evolving from statistical techniques to data-driven deep learning architectures. Despite their effectiveness, most existing methods still adhere to a fast thinking paradigm-relying on extracting historical patterns and mapping them to future values as their core modeling philosophy, lacking an explicit thinking process that incorporates intermediate time series reasoning. Meanwhile, emerging slow-thinking LLMs (e.g., OpenAI-o1) have shown remarkable multi-step reasoning capabilities, offering an alternative way to overcome these issues. However, prompt engineering alone presents several limitations - including high computational cost, privacy risks, and limited capacity for in-depth domain-specific time series reasoning. To address these limitations, a more promising approach is to train LLMs to develop slow thinking capabilities and acquire strong time series reasoning skills. For this purpose, we propose Time-R1, a two-stage reinforcement fine-tuning framework designed to enhance multi-step reasoning ability of LLMs for time series forecasting. Specifically, the first stage conducts supervised fine-tuning for warmup adaptation, while the second stage employs reinforcement learning to improve the model's generalization ability. Particularly, we design a fine-grained multi-objective reward specifically for time series forecasting, and then introduce GRIP (group-based relative importance for policy optimization), which leverages non-uniform sampling to further encourage and optimize the model's exploration of effective reasoning paths. Experiments demonstrate that Time-R1 significantly improves forecast performance across diverse datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.10887v3">Hey, That's My Model! Introducing Chain & Hash, An LLM Fingerprinting Technique</a></div>
    <div class="paper-meta">
      📅 2025-06-12
    </div>
    <details class="paper-abstract">
      Growing concerns over the theft and misuse of Large Language Models (LLMs) have heightened the need for effective fingerprinting, which links a model to its original version to detect misuse. In this paper, we define five key properties for a successful fingerprint: Transparency, Efficiency, Persistence, Robustness, and Unforgeability. We introduce a novel fingerprinting framework that provides verifiable proof of ownership while maintaining fingerprint integrity. Our approach makes two main contributions. First, we propose a Chain and Hash technique that cryptographically binds fingerprint prompts with their responses, ensuring no adversary can generate colliding fingerprints and allowing model owners to irrefutably demonstrate their creation. Second, we address a realistic threat model in which instruction-tuned models' output distribution can be significantly altered through meta-prompts. By integrating random padding and varied meta-prompt configurations during training, our method preserves fingerprint robustness even when the model's output style is significantly modified. Experimental results demonstrate that our framework offers strong security for proving ownership and remains resilient against benign transformations like fine-tuning, as well as adversarial attempts to erase fingerprints. Finally, we also demonstrate its applicability to fingerprinting LoRA adapters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.07991v6">Human and LLM Biases in Hate Speech Annotations: A Socio-Demographic Analysis of Annotators and Targets</a></div>
    <div class="paper-meta">
      📅 2025-06-12
      | 💬 Article published in ICWSM'25 - 19th AAAI Conference on Web and Social Media. Please, cite the published version
    </div>
    <details class="paper-abstract">
      The rise of online platforms exacerbated the spread of hate speech, demanding scalable and effective detection. However, the accuracy of hate speech detection systems heavily relies on human-labeled data, which is inherently susceptible to biases. While previous work has examined the issue, the interplay between the characteristics of the annotator and those of the target of the hate are still unexplored. We fill this gap by leveraging an extensive dataset with rich socio-demographic information of both annotators and targets, uncovering how human biases manifest in relation to the target's attributes. Our analysis surfaces the presence of widespread biases, which we quantitatively describe and characterize based on their intensity and prevalence, revealing marked differences. Furthermore, we compare human biases with those exhibited by persona-based LLMs. Our findings indicate that while persona-based LLMs do exhibit biases, these differ significantly from those of human annotators. Overall, our work offers new and nuanced results on human biases in hate speech annotations, as well as fresh insights into the design of AI-driven hate speech detection systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10527v1">LogiPlan: A Structured Benchmark for Logical Planning and Relational Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-12
    </div>
    <details class="paper-abstract">
      We introduce LogiPlan, a novel benchmark designed to evaluate the capabilities of large language models (LLMs) in logical planning and reasoning over complex relational structures. Logical relational reasoning is important for applications that may rely on LLMs to generate and query structured graphs of relations such as network infrastructure, knowledge bases, or business process schema. Our framework allows for dynamic variation of task complexity by controlling the number of objects, relations, and the minimum depth of relational chains, providing a fine-grained assessment of model performance across difficulty levels. LogiPlan encompasses three complementary tasks: (1) Plan Generation, where models must construct valid directed relational graphs meeting specified structural constraints; (2) Consistency Detection, testing models' ability to identify inconsistencies in relational structures; and (3) Comparison Question, evaluating models' capacity to determine the validity of queried relationships within a given graph. Additionally, we assess models' self-correction capabilities by prompting them to verify and refine their initial solutions. We evaluate state-of-the-art models including DeepSeek R1, Gemini 2.0 Pro, Gemini 2 Flash Thinking, GPT-4.5, GPT-4o, Llama 3.1 405B, O3-mini, O1, and Claude 3.7 Sonnet across these tasks, revealing significant performance gaps that correlate with model scale and architecture. Our analysis demonstrates that while recent reasoning-enhanced models show promising results on simpler instances, they struggle with more complex configurations requiring deeper logical planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10525v1">AdaptiveLLM: A Framework for Selecting Optimal Cost-Efficient LLM for Code-Generation Based on CoT Length</a></div>
    <div class="paper-meta">
      📅 2025-06-12
      | 💬 Accepted by Internetware 2025
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have significantly advanced code generation efficiency, they face inherent challenges in balancing performance and inference costs across diverse programming tasks. Dynamically selecting the optimal LLM based on task difficulty and resource constraints offers a promising approach to achieve an optimal balance between efficiency and performance. However, existing model selection methods are resource-intensive and often neglect cost efficiency. Moreover, these approaches rely on human-annotated difficulty labels that are frequently inaccessible in real-world settings and may not align with the LLM's own assessment of task difficulty. In this paper, we introduce AdaptiveLLM, a framework that dynamically selects optimal LLMs for a given coding task by automatically assessing task difficulty. Our framework first estimates task difficulty using Chain-of-Thought lengths generated by reasoning model, clusters these into three difficulty levels via k-means, and fine-tunes CodeBERT to embed difficulty-aware features. A trained XGBoost classifier then selects the best model for each problem, optimizing the performance-cost trade-off. Experimental results show that AdaptiveLLM achieves a 7.86% improvement in pass@1 score while reducing resource consumption by 88.9% compared to baseline method ComplexityNet. When compared to a single model, AdaptiveLLM demonstrates an approximately 15% accuracy improvement, while maintaining the same level of cost consumption. Apart from that, the difficulty assessment using CoT provides more reliable selection criteria than human evaluation. Our replication package is available at https://github.com/cjhCoder7/AdaptiveLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10508v1">Reliable Reasoning Path: Distilling Effective Guidance for LLM Reasoning with Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2025-06-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often struggle with knowledge-intensive tasks due to a lack of background knowledge and a tendency to hallucinate. To address these limitations, integrating knowledge graphs (KGs) with LLMs has been intensively studied. Existing KG-enhanced LLMs focus on supplementary factual knowledge, but still struggle with solving complex questions. We argue that refining the relationships among facts and organizing them into a logically consistent reasoning path is equally important as factual knowledge itself. Despite their potential, extracting reliable reasoning paths from KGs poses the following challenges: the complexity of graph structures and the existence of multiple generated paths, making it difficult to distinguish between useful and redundant ones. To tackle these challenges, we propose the RRP framework to mine the knowledge graph, which combines the semantic strengths of LLMs with structural information obtained through relation embedding and bidirectional distribution learning. Additionally, we introduce a rethinking module that evaluates and refines reasoning paths according to their significance. Experimental results on two public datasets show that RRP achieves state-of-the-art performance compared to existing baseline methods. Moreover, RRP can be easily integrated into various LLMs to enhance their reasoning abilities in a plug-and-play manner. By generating high-quality reasoning paths tailored to specific questions, RRP distills effective guidance for LLM reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10501v1">BugGen: A Self-Correcting Multi-Agent LLM Pipeline for Realistic RTL Bug Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-06-12
    </div>
    <details class="paper-abstract">
      Hardware complexity continues to strain verification resources, motivating the adoption of machine learning (ML) methods to improve debug efficiency. However, ML-assisted debugging critically depends on diverse and scalable bug datasets, which existing manual or automated bug insertion methods fail to reliably produce. We introduce BugGen, a first of its kind, fully autonomous, multi-agent pipeline leveraging Large Language Models (LLMs) to systematically generate, insert, and validate realistic functional bugs in RTL. BugGen partitions modules, selects mutation targets via a closed-loop agentic architecture, and employs iterative refinement and rollback mechanisms to ensure syntactic correctness and functional detectability. Evaluated across five OpenTitan IP blocks, BugGen produced 500 unique bugs with 94% functional accuracy and achieved a throughput of 17.7 validated bugs per hour-over five times faster than typical manual expert insertion. Additionally, BugGen identified 104 previously undetected bugs in OpenTitan regressions, highlighting its utility in exposing verification coverage gaps. Compared against Certitude, BugGen demonstrated over twice the syntactic accuracy, deeper exposure of testbench blind spots, and more functionally meaningful and complex bug scenarios. Furthermore, when these BugGen-generated datasets were employed to train ML-based failure triage models, we achieved high classification accuracy (88.1%-93.2%) across different IP blocks, confirming the practical utility and realism of generated bugs. BugGen thus provides a scalable solution for generating high-quality bug datasets, significantly enhancing verification efficiency and ML-assisted debugging.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09554v2">Understanding the Performance and Power of LLM Inferencing on Edge Accelerators</a></div>
    <div class="paper-meta">
      📅 2025-06-12
      | 💬 Extended version of short paper to appear in PAISE 2025: Mayank Arya and Yogesh Simmhan, "Understanding the Performance and Power of LLM Inferencing on Edge Accelerators" in 7th Workshop on Parallel AI and Systems for the Edge, Co-located with IEEE International Parallel & Distributed Processing Symposium (IPDPS), 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated exceptional benefits to a wide range of domains, for tasks as diverse as code generation and robot navigation. While LLMs are usually served from cloud data centers, mission-critical and privacy-sensitive applications may require local hosting of open LLM models. Given the large GPU memory footprint needed for LLMs, edge accelerators such as Nvidia Jetson Orin AGX with 64GB of shared GPU-CPU RAM are a compelling choice. However, the feasibility and performance of LLM inference on edge accelerators is under-explored. This study presents a detailed evaluation of LLM inference on the NVIDIA Jetson Orin AGX, on four SOTA models ranging from 2.7B to 32.8B parameters, such as Meta Llama3.1, Microsoft-Phi2, Deepseek-R1-Qwen. We investigate the impact of varying batch sizes, sequence lengths, and quantization levels on latency, throughput, and perplexity, and also explore various custom power modes on the Orin AGX to perform power and energy consumption analysis. Our findings offer interesting insights on the trade-offs between efficiency, inference speed and resource use, e.g., increasing the sequence length causes a decrease in token throughput and quantization causes smaller LLMs to be slower. These results can help optimize LLM serving on edge accelerators for practical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10484v1">EXPEREPAIR: Dual-Memory Enhanced LLM-based Repository-Level Program Repair</a></div>
    <div class="paper-meta">
      📅 2025-06-12
    </div>
    <details class="paper-abstract">
      Automatically repairing software issues remains a fundamental challenge at the intersection of software engineering and AI. Although recent advancements in Large Language Models (LLMs) have demonstrated potential for repository-level repair tasks, current methodologies exhibit two notable limitations: (1) they often address issues in isolation, neglecting to incorporate insights from previously resolved issues, and (2) they rely on static and rigid prompting strategies, which constrain their ability to generalize across diverse and evolving issue scenarios. Inspired by the dual memory systems of human cognition, where episodic and semantic memories work synergistically to support human reasoning and decision-making, we propose ExpeRepair, a novel LLM-based approach that continuously learns from historical repair experiences through dual-channel knowledge accumulation. ExpeRepair organizes historical repair experiences into two complementary memories: an episodic memory that stores concrete repair demonstrations, and a semantic memory that encodes abstract reflective insights. At inference time, ExpeRepair activates both memory systems by retrieving relevant demonstrations from episodic memory and recalling high-level repair insights from semantic memory. It further enhances adaptability through dynamic prompt composition, synergistically integrating both memory types to replace static prompts with context-aware, experience-driven prompts. Experiments on the SWE-bench Lite benchmark demonstrate that ExpeRepair achieves a pass@1 score of 49.3% with Claude 3.7 Sonnet, outperforming all state-of-the-art open-source methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10470v1">TD-Pipe: Temporally-Disaggregated Pipeline Parallelism Architecture for High-Throughput LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-06-12
    </div>
    <details class="paper-abstract">
      As the model size continuously increases, pipeline parallelism shows great promise in throughput-oriented LLM inference due to its low demand on communications. However, imbalanced pipeline workloads and complex data dependencies in the prefill and decode phases result in massive pipeline bubbles and further severe performance reduction. To better exploit the pipeline parallelism for high-throughput LLM inference, we propose TD-Pipe, with the key idea lies in the temporally-disaggregated pipeline parallelism architecture. Specifically, this architecture disaggregates the prefill and decode phases in the temporal dimension, so as to eliminate pipeline bubbles caused by the phase switching. TD-Pipe identifies potential issues of exploiting the novel architecture and provides solutions. First, a hierarchy-controller structure is used to better coordinate devices in pipeline parallelism by decoupling the scheduling from execution. Second, the AI-based greedy prefill approach aggressively performs more prefills by predicting the output length and simulating the memory usage. Third, the inter-batch work stealing approach dynamically balances decode phase workloads between different batches to reduce bubbles. Forth, the spatial-temporal intensity comparison approach determines the optimal switch from decode to prefill by comparing the performance drop from reduced computational intensity with that from phase switching bubbles. Extensive experiments show that TD-Pipe effectively increases the throughput of LLM inference by up to 1.91x over the existing tensor parallel approach and 2.73x over the existing pipeline parallel approach on GPU nodes with only PCIe interconnection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10467v1">Specification and Evaluation of Multi-Agent LLM Systems -- Prototype and Cybersecurity Applications</a></div>
    <div class="paper-meta">
      📅 2025-06-12
    </div>
    <details class="paper-abstract">
      Recent advancements in LLMs indicate potential for novel applications, e.g., through reasoning capabilities in the latest OpenAI and DeepSeek models. For applying these models in specific domains beyond text generation, LLM-based multi-agent approaches can be utilized that solve complex tasks by combining reasoning techniques, code generation, and software execution. Applications might utilize these capabilities and the knowledge of specialized LLM agents. However, while many evaluations are performed on LLMs, reasoning techniques, and applications individually, their joint specification and combined application is not explored well. Defined specifications for multi-agent LLM systems are required to explore their potential and their suitability for specific applications, allowing for systematic evaluations of LLMs, reasoning techniques, and related aspects. This paper reports the results of exploratory research to specify and evaluate these aspects through a multi-agent system. The system architecture and prototype are extended from previous research and a specification is introduced for multi-agent systems. Test cases involving cybersecurity tasks indicate feasibility of the architecture and evaluation approach. In particular, the results show the evaluation of question answering, server security, and network security tasks that were completed correctly by agents with LLMs from OpenAI and DeepSeek.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04412v2">Wider or Deeper? Scaling LLM Inference-Time Compute with Adaptive Branching Tree Search</a></div>
    <div class="paper-meta">
      📅 2025-06-12
      | 💬 Presented at ICLR 2025 Workshop on Foundation Models in the Wild
    </div>
    <details class="paper-abstract">
      Recent advances demonstrate that increasing inference-time computation can significantly boost the reasoning capabilities of large language models (LLMs). Although repeated sampling (i.e., generating multiple candidate outputs) is a highly effective strategy, it does not leverage external feedback signals for refinement, which are often available in tasks like coding. In this work, we propose Adaptive Branching Monte Carlo Tree Search (AB-MCTS), a novel inference-time framework that generalizes repeated sampling with principled multi-turn exploration and exploitation. At each node in the search tree, AB-MCTS dynamically decides whether to "go wider" by expanding new candidate responses or "go deeper" by revisiting existing ones based on external feedback signals. We evaluate our method on complex coding and engineering tasks using frontier models. Empirical results show that AB-MCTS consistently outperforms both repeated sampling and standard MCTS, underscoring the importance of combining the response diversity of LLMs with multi-turn solution refinement for effective inference-time scaling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07551v2">CheMatAgent: Enhancing LLMs for Chemistry and Materials Science through Tree-Search Based Tool Learning</a></div>
    <div class="paper-meta">
      📅 2025-06-12
      | 💬 15 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently demonstrated promising capabilities in chemistry tasks while still facing challenges due to outdated pretraining knowledge and the difficulty of incorporating specialized chemical expertise. To address these issues, we propose an LLM-based agent that synergistically integrates 137 external chemical tools created ranging from basic information retrieval to complex reaction predictions, and a dataset curation pipeline to generate the dataset ChemToolBench that facilitates both effective tool selection and precise parameter filling during fine-tuning and evaluation. We introduce a Hierarchical Evolutionary Monte Carlo Tree Search (HE-MCTS) framework, enabling independent optimization of tool planning and execution. By leveraging self-generated data, our approach supports step-level fine-tuning (FT) of the policy model and training task-adaptive PRM and ORM that surpass GPT-4o. Experimental evaluations demonstrate that our approach significantly improves performance in Chemistry QA and discovery tasks, offering a robust solution to integrate specialized tools with LLMs for advanced chemical applications. All datasets and code are available at https://github.com/AI4Chem/ChemistryAgent .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10424v1">SOFT: Selective Data Obfuscation for Protecting LLM Fine-tuning against Membership Inference Attacks</a></div>
    <div class="paper-meta">
      📅 2025-06-12
      | 💬 Accepted by the 34th USENIX Security Symposium 2025. Code is available at https://github.com/KaiyuanZh/SOFT
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable success and are widely adopted for diverse applications. However, fine-tuning these models often involves private or sensitive information, raising critical privacy concerns. In this work, we conduct the first comprehensive study evaluating the vulnerability of fine-tuned LLMs to membership inference attacks (MIAs). Our empirical analysis demonstrates that MIAs exploit the loss reduction during fine-tuning, making them highly effective in revealing membership information. These findings motivate the development of our defense. We propose SOFT (\textbf{S}elective data \textbf{O}bfuscation in LLM \textbf{F}ine-\textbf{T}uning), a novel defense technique that mitigates privacy leakage by leveraging influential data selection with an adjustable parameter to balance utility preservation and privacy protection. Our extensive experiments span six diverse domains and multiple LLM architectures and scales. Results show that SOFT effectively reduces privacy risks while maintaining competitive model performance, offering a practical and scalable solution to safeguard sensitive information in fine-tuned LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10423v1">PAL: Probing Audio Encoders via LLMs -- A Study of Information Transfer from Audio Encoders to LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-12
      | 💬 21 pages, 11 figures
    </div>
    <details class="paper-abstract">
      The integration of audio perception capabilities into Large Language Models (LLMs) has enabled significant advances in Audio-LLMs. Although application-focused developments, particularly in curating training data for specific capabilities e.g., audio reasoning, have progressed rapidly, the underlying mechanisms that govern efficient transfer of rich semantic representations from audio encoders to LLMs remain under-explored. We conceptualize effective audio-LLM interaction as the LLM's ability to proficiently probe the audio encoder representations to satisfy textual queries. This paper presents a systematic investigation on how architectural design choices can affect that. Beginning with a standard Pengi/LLaVA-style audio-LLM architecture, we propose and evaluate several modifications guided by hypotheses derived from mechanistic interpretability studies and LLM operational principles. Our experiments demonstrate that: (1) delaying audio integration until the LLM's initial layers establish textual context that enhances its ability to probe the audio representations for relevant information; (2) the LLM can proficiently probe audio representations exclusively through LLM layer's attention submodule, without requiring propagation to its Feed-Forward Network (FFN) submodule; (3) an efficiently integrated ensemble of diverse audio encoders provides richer, complementary representations, thereby broadening the LLM's capacity to probe a wider spectrum of audio information. All hypotheses are evaluated using an identical three-stage training curriculum on a dataset of 5.6 million audio-text pairs, ensuring controlled comparisons. Our final architecture, which incorporates all proposed modifications, achieves relative improvements from 10\% to 60\% over the baseline, validating our approach to optimizing cross-modal information transfer in audio-LLMs. Project page: https://ta012.github.io/PAL/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10406v1">PAG: Multi-Turn Reinforced LLM Self-Correction with Policy as Generative Verifier</a></div>
    <div class="paper-meta">
      📅 2025-06-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive capabilities in complex reasoning tasks, yet they still struggle to reliably verify the correctness of their own outputs. Existing solutions to this verification challenge often depend on separate verifier models or require multi-stage self-correction training pipelines, which limit scalability. In this paper, we propose Policy as Generative Verifier (PAG), a simple and effective framework that empowers LLMs to self-correct by alternating between policy and verifier roles within a unified multi-turn reinforcement learning (RL) paradigm. Distinct from prior approaches that always generate a second attempt regardless of model confidence, PAG introduces a selective revision mechanism: the model revises its answer only when its own generative verification step detects an error. This verify-then-revise workflow not only alleviates model collapse but also jointly enhances both reasoning and verification abilities. Extensive experiments across diverse reasoning benchmarks highlight PAG's dual advancements: as a policy, it enhances direct generation and self-correction accuracy; as a verifier, its self-verification outperforms self-consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10403v1">Time To Impeach LLM-as-a-Judge: Programs are the Future of Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-06-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are widely used to evaluate the quality of LLM generations and responses, but this leads to significant challenges: high API costs, uncertain reliability, inflexible pipelines, and inherent biases. To address these, we introduce PAJAMA (Program-As-a-Judge for Automated Model Assessment), a new alternative that uses LLMs to synthesize executable judging programs instead of directly scoring responses. These synthesized programs can be stored and run locally, costing orders of magnitude less while providing interpretable, and auditable judging logic that can be easily adapted. Program-based judges mitigate biases, improving judgment consistency by 15.83% and reducing biased responses by 23.7% on average compared to a Qwen2.5-14B-based LLM-as-a-judge. When program judgments are distilled into a model, PAJAMA outperforms LLM-as-a-judge on the challenging CHAT-HARD subset of RewardBench, outperforming metrics by 2.19% on Prometheus and 8.67% on the JudgeLM dataset, all at three orders of magnitude lower cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10401v1">HPCTransCompile: An AI Compiler Generated Dataset for High-Performance CUDA Transpilation and LLM Preliminary Exploration</a></div>
    <div class="paper-meta">
      📅 2025-06-12
    </div>
    <details class="paper-abstract">
      The rapid growth of deep learning has driven exponential increases in model parameters and computational demands. NVIDIA GPUs and their CUDA-based software ecosystem provide robust support for parallel computing, significantly alleviating computational bottlenecks. Meanwhile, due to the cultivation of user programming habits and the high performance of GPUs, the CUDA ecosystem has established a dominant position in the field of parallel software. This dominance requires other hardware platforms to support CUDA-based software with performance portability. However, translating CUDA code to other platforms poses significant challenges due to differences in parallel programming paradigms and hardware architectures. Existing approaches rely on language extensions, domain-specific languages (DSLs), or compilers but face limitations in workload coverage and generalizability. Moreover, these methods often incur substantial development costs. Recently, LLMs have demonstrated extraordinary potential in various vertical domains, especially in code-related tasks. However, the performance of existing LLMs in CUDA transpilation, particularly for high-performance code, remains suboptimal. The main reason for this limitation lies in the lack of high-quality training datasets. To address these challenges, we propose a novel framework for generating high-performance CUDA and corresponding platform code pairs, leveraging AI compiler and automatic optimization technology. We further enhance the framework with a graph-based data augmentation method and introduce HPCTransEval, a benchmark for evaluating LLM performance on CUDA transpilation. We conduct experiments using CUDA-to-CPU transpilation as a case study on leading LLMs. The result demonstrates that our framework significantly improves CUDA transpilation, highlighting the potential of LLMs to address compatibility challenges within the CUDA ecosystem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11343v2">A Minimalist Approach to LLM Reasoning: from Rejection Sampling to Reinforce</a></div>
    <div class="paper-meta">
      📅 2025-06-12
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has become a prevailing approach for fine-tuning large language models (LLMs) on complex reasoning tasks. Among recent methods, GRPO stands out for its empirical success in training models such as DeepSeek-R1, yet the sources of its effectiveness remain poorly understood. In this work, we revisit GRPO from a reinforce-like algorithm perspective and analyze its core components. Surprisingly, we find that a simple rejection sampling baseline, RAFT, which trains only on positively rewarded samples, yields competitive performance than GRPO and PPO. Our ablation studies reveal that GRPO's main advantage arises from discarding prompts with entirely incorrect responses, rather than from its reward normalization. Motivated by this insight, we propose Reinforce-Rej, a minimal extension of policy gradient that filters both entirely incorrect and entirely correct samples. Reinforce-Rej improves KL efficiency and stability, serving as a lightweight yet effective alternative to more complex RL algorithms. We advocate RAFT as a robust and interpretable baseline, and suggest that future advances should focus on more principled designs for incorporating negative samples, rather than relying on them indiscriminately. Our findings provide guidance for future work in reward-based LLM post-training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10364v1">Can We Infer Confidential Properties of Training Data from LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-06-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly fine-tuned on domain-specific datasets to support applications in fields such as healthcare, finance, and law. These fine-tuning datasets often have sensitive and confidential dataset-level properties -- such as patient demographics or disease prevalence -- that are not intended to be revealed. While prior work has studied property inference attacks on discriminative models (e.g., image classification models) and generative models (e.g., GANs for image data), it remains unclear if such attacks transfer to LLMs. In this work, we introduce PropInfer, a benchmark task for evaluating property inference in LLMs under two fine-tuning paradigms: question-answering and chat-completion. Built on the ChatDoctor dataset, our benchmark includes a range of property types and task configurations. We further propose two tailored attacks: a prompt-based generation attack and a shadow-model attack leveraging word frequency signals. Empirical evaluations across multiple pretrained LLMs show the success of our attacks, revealing a previously unrecognized vulnerability in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.20840v2">CodeTool: Enhancing Programmatic Tool Invocation of LLMs via Process Supervision</a></div>
    <div class="paper-meta">
      📅 2025-06-12
      | 💬 ACL2025
    </div>
    <details class="paper-abstract">
      Tool invocation significantly enhances the capabilities of Large Language Models (LLMs), yet challenges persist, particularly in complex task scenarios. Current methods, such as instruction-enhanced reasoning and supervised fine-tuning, often result in unnecessarily long reasoning paths and face difficulties in verifying the correctness of intermediate steps. In this paper, we propose CodeTool, a novel framework for stepwise code generation that improves LLM tool invocation by leveraging the concise and easily verifiable nature of code. CodeTool incorporates two distinct process rewards: the On-the-spot Reward, which provides immediate feedback on the accuracy of each tool invocation, and the Latent Reward, which assesses the contribution of each step toward overall task completion. By maximizing the cumulative reward of the On-the-spot and Latend Rewards at each step, LLMs are guided to follow efficient and accurate reasoning paths. Extensive experiments on StableToolBench and RestBench-TMDB demonstrate the superiority of CodeTool over existing approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10343v1">Code Execution as Grounded Supervision for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-06-12
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) with chain-of-thought (CoT) supervision has proven effective for enhancing their reasoning abilities. However, obtaining reliable and accurate reasoning supervision remains a significant challenge. We propose a scalable method for generating a high-quality CoT supervision dataset by leveraging the determinism of program execution. Unlike existing reasoning dataset generation methods that rely on costly human annotations or error-prone LLM-generated CoT, our approach extracts verifiable, step-by-step reasoning traces from code execution and transforms them into a natural language CoT reasoning. Experiments on reasoning benchmarks across various domains show that our method effectively equips LLMs with transferable reasoning abilities across diverse tasks. Furthermore, the ablation studies validate that our method produces highly accurate reasoning data and reduces overall token length during inference by reducing meaningless repetition and overthinking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19148v3">Amulet: ReAlignment During Test Time for Personalized Preference Adaptation of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-12
      | 💬 Accepted by ICLR 2025, Project page: https://zowiezhang.github.io/projects/Amulet
    </div>
    <details class="paper-abstract">
      How to align large language models (LLMs) with user preferences from a static general dataset has been frequently studied. However, user preferences are usually personalized, changing, and diverse regarding culture, values, or time. This leads to the problem that the actual user preferences often do not coincide with those trained by the model developers in the practical use of LLMs. Since we cannot collect enough data and retrain for every demand, researching efficient real-time preference adaptation methods based on the backbone LLMs during test time is important. To this end, we introduce Amulet, a novel, training-free framework that formulates the decoding process of every token as a separate online learning problem with the guidance of simple user-provided prompts, thus enabling real-time optimization to satisfy users' personalized preferences. To reduce the computational cost brought by this optimization process for each token, we additionally provide a closed-form solution for each iteration step of the optimization process, thereby reducing the computational time cost to a negligible level. The detailed experimental results demonstrate that Amulet can achieve significant performance improvements in rich settings with combinations of different LLMs, datasets, and user preferences, while maintaining acceptable computational efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04715v2">Towards Holistic Visual Quality Assessment of AI-Generated Videos: A LLM-Based Multi-Dimensional Evaluation Model</a></div>
    <div class="paper-meta">
      📅 2025-06-12
      | 💬 This paper has been accepted by CVPR Workshop 2025
    </div>
    <details class="paper-abstract">
      The development of AI-Generated Video (AIGV) technology has been remarkable in recent years, significantly transforming the paradigm of video content production. However, AIGVs still suffer from noticeable visual quality defects, such as noise, blurriness, frame jitter and low dynamic degree, which severely impact the user's viewing experience. Therefore, an effective automatic visual quality assessment is of great importance for AIGV content regulation and generative model improvement. In this work, we decompose the visual quality of AIGVs into three dimensions: technical quality, motion quality, and video semantics. For each dimension, we design corresponding encoder to achieve effective feature representation. Moreover, considering the outstanding performance of large language models (LLMs) in various vision and language tasks, we introduce a LLM as the quality regression module. To better enable the LLM to establish reasoning associations between multi-dimensional features and visual quality, we propose a specially designed multi-modal prompt engineering framework. Additionally, we incorporate LoRA fine-tuning technology during the training phase, allowing the LLM to better adapt to specific tasks. Our proposed method achieved \textbf{second place} in the NTIRE 2025 Quality Assessment of AI-Generated Content Challenge: Track 2 AI Generated video, demonstrating its effectiveness. Codes can be obtained at https://github.com/QiZelu/AIGVEval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.07321v3">Benchmarking LLMs for Environmental Review and Permitting</a></div>
    <div class="paper-meta">
      📅 2025-06-12
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      The National Environment Policy Act (NEPA) stands as a foundational piece of environmental legislation in the United States, requiring federal agencies to consider the environmental impacts of their proposed actions. The primary mechanism for achieving this is through the preparation of Environmental Assessments (EAs) and, for significant impacts, comprehensive Environmental Impact Statements (EIS). Large Language Model (LLM)s' effectiveness in specialized domains like NEPA remains untested for adoption in federal decision-making processes. To address this gap, we present NEPA Question and Answering Dataset (NEPAQuAD), the first comprehensive benchmark derived from EIS documents, along with a modular and transparent evaluation pipeline, MAPLE, to assess LLM performance on NEPA-focused regulatory reasoning tasks. Our benchmark leverages actual EIS documents to create diverse question types, ranging from factual to complex problem-solving ones. We built a modular and transparent evaluation pipeline to test both closed- and open-source models in zero-shot or context-driven QA benchmarks. We evaluate five state-of-the-art LLMs using our framework to assess both their prior knowledge and their ability to process NEPA-specific information. The experimental results reveal that all the models consistently achieve their highest performance when provided with the gold passage as context. While comparing the other context-driven approaches for each model, Retrieval Augmented Generation (RAG)-based approaches substantially outperform PDF document contexts, indicating that neither model is well suited for long-context question-answering tasks. Our analysis suggests that NEPA-focused regulatory reasoning tasks pose a significant challenge for LLMs, particularly in terms of understanding the complex semantics and effectively processing the lengthy regulatory documents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.09742v2">Paired Completion: Flexible Quantification of Issue-framing at Scale with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-12
      | 💬 9 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Detecting issue framing in text - how different perspectives approach the same topic - is valuable for social science and policy analysis, yet challenging for automated methods due to subtle linguistic differences. We introduce `paired completion', a novel approach using LLM next-token log probabilities to detect contrasting frames using minimal examples. Through extensive evaluation across synthetic datasets and a human-labeled corpus, we demonstrate that paired completion is a cost-efficient, low-bias alternative to both prompt-based and embedding-based methods, offering a scalable solution for analyzing issue framing in large text collections, especially suited to low-resource settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10323v1">ELFuzz: Efficient Input Generation via LLM-driven Synthesis Over Fuzzer Space</a></div>
    <div class="paper-meta">
      📅 2025-06-12
      | 💬 Accepted by USENIX Security'25 Cycle 2
    </div>
    <details class="paper-abstract">
      Generation-based fuzzing produces appropriate testing cases according to specifications of input grammars and semantic constraints to test systems and software. However, these specifications require significant manual efforts to construct. This paper proposes a new approach, ELFuzz (Evolution Through Large Language Models for Fuzzing), that automatically synthesizes generation-based fuzzers tailored to a system under test (SUT) via LLM-driven synthesis over fuzzer space. At a high level, it starts with minimal seed fuzzers and propels the synthesis by fully automated LLM-driven evolution with coverage guidance. Compared to previous approaches, ELFuzz can 1) seamlessly scale to SUTs of real-world sizes -- up to 1,791,104 lines of code in our evaluation -- and 2) synthesize efficient fuzzers that catch interesting grammatical structures and semantic constraints in a human-understandable way. Our evaluation compared ELFuzz with specifications manually written by domain experts and synthesized by state-of-the-art approaches. It shows that ELFuzz achieves up to 434.8% more coverage and triggers up to 174.0% more artificially injected bugs. We also used ELFuzz to conduct a real-world fuzzing campaign on the newest version of cvc5 for 14 days, and encouragingly, it found five 0-day bugs (three are exploitable). Moreover, we conducted an ablation study, which shows that the fuzzer space model, the key component of ELFuzz, contributes the most (up to 62.5%) to the effectiveness of ELFuzz. Further analysis of the fuzzers synthesized by ELFuzz confirms that they catch interesting grammatical structures and semantic constraints in a human-understandable way. The results present the promising potential of ELFuzz for more automated, efficient, and extensible input generation for fuzzing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10322v1">Minimizing False Positives in Static Bug Detection via LLM-Enhanced Path Feasibility Analysis</a></div>
    <div class="paper-meta">
      📅 2025-06-12
    </div>
    <details class="paper-abstract">
      Static bug analyzers play a crucial role in ensuring software quality. However, existing analyzers for bug detection in large codebases often suffer from high false positive rates. This is primarily due to the limited capabilities of analyzers in path feasibility validation with multiple conditional branches and complex data dependencies. While current LLM-based approaches attempt to address this issue, their effectiveness remains limited due to insufficient constraint cascade analysis and scalability challenges in large projects. To address this challenge, we propose an iterative path feasibility analysis framework LLM4PFA. By leveraging LLM agent based targeted constraint reasoning, and key context-aware analysis driven by agent planning, LLM4PFA effectively enhances complex inter-procedural path feasibility analysis for minimizing false positives in static bug detection. Evaluation results show that LLM4PFA precisely filters out 72% to 96% false positives reported during static bug detection, significantly outperforming all the baselines by 41.1% - 105.7% improvements; meanwhile LLM4PFA only misses 3 real bugs of 45 true positives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10312v1">AC/DC: LLM-based Audio Comprehension via Dialogue Continuation</a></div>
    <div class="paper-meta">
      📅 2025-06-12
      | 💬 Accepted to Interspeech 2025
    </div>
    <details class="paper-abstract">
      We propose an instruction-following audio comprehension model that leverages the dialogue continuation ability of large language models (LLMs). Instead of directly generating target captions in training data, the proposed method trains a model to produce responses as if the input caption triggered a dialogue. This dialogue continuation training mitigates the caption variation problem. Learning to continue a dialogue effectively captures the caption's meaning beyond its surface-level words. As a result, our model enables zero-shot instruction-following capability without multitask instruction tuning, even trained solely on audio captioning datasets. Experiments on AudioCaps, WavCaps, and Clotho datasets with AudioBench audio-scene question-answering tests demonstrate our model's ability to follow various unseen instructions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10299v1">Scheduled Interleaved Speech-Text Training for Speech-to-Speech Translation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-12
      | 💬 Accepted to Interspeech2025
    </div>
    <details class="paper-abstract">
      Speech-to-speech translation (S2ST) has been advanced with large language models (LLMs), which are fine-tuned on discrete speech units. In such approaches, modality adaptation from text to speech has been an issue. LLMs are trained on text-only data, which presents challenges to adapt them to speech modality with limited speech-to-speech data. To address the training difficulty, we propose scheduled interleaved speech--text training in this study. We use interleaved speech--text units instead of speech units during training, where aligned text tokens are interleaved at the word level. We gradually decrease the ratio of text as training progresses, to facilitate progressive modality adaptation from text to speech. We conduct experimental evaluations by fine-tuning LLaMA3.2-1B for S2ST on the CVSS dataset. We show that the proposed method consistently improves the translation performances, especially for languages with limited training data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00309v2">Evaluation of LLMs for mathematical problem solving</a></div>
    <div class="paper-meta">
      📅 2025-06-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown impressive performance on a range of educational tasks, but are still understudied for their potential to solve mathematical problems. In this study, we compare three prominent LLMs, including GPT-4o, DeepSeek-V3, and Gemini-2.0, on three mathematics datasets of varying complexities (GSM8K, MATH500, and UNSW datasets). We take a five-dimensional approach based on the Structured Chain-of-Thought (SCoT) framework to assess final answer correctness, step completeness, step validity, intermediate calculation accuracy, and problem comprehension. The results show that GPT-4o is the most stable and consistent in performance across all the datasets, but particularly it performs outstandingly in high-level questions of the UNSW dataset. DeepSeek-V3 is competitively strong in well-structured domains such as optimisation, but suffers from fluctuations in accuracy in statistical inference tasks. Gemini-2.0 shows strong linguistic understanding and clarity in well-structured problems but performs poorly in multi-step reasoning and symbolic logic. Our error analysis reveals particular deficits in each model: GPT-4o is at times lacking in sufficient explanation or precision; DeepSeek-V3 leaves out intermediate steps; and Gemini-2.0 is less flexible in mathematical reasoning in higher dimensions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11361v1">The Biased Samaritan: LLM biases in Perceived Kindness</a></div>
    <div class="paper-meta">
      📅 2025-06-12
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have become ubiquitous in many fields, understanding and mitigating LLM biases is an ongoing issue. This paper provides a novel method for evaluating the demographic biases of various generative AI models. By prompting models to assess a moral patient's willingness to intervene constructively, we aim to quantitatively evaluate different LLMs' biases towards various genders, races, and ages. Our work differs from existing work by aiming to determine the baseline demographic identities for various commercial models and the relationship between the baseline and other demographics. We strive to understand if these biases are positive, neutral, or negative, and the strength of these biases. This paper can contribute to the objective assessment of bias in Large Language Models and give the user or developer the power to account for these biases in LLM output or in training future LLMs. Our analysis suggested two key findings: that models view the baseline demographic as a white middle-aged or young adult male; however, a general trend across models suggested that non-baseline demographics are more willing to help than the baseline. These methodologies allowed us to distinguish these two biases that are often tangled together.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13116v3">VeriLeaky: Navigating IP Protection vs Utility in Fine-Tuning for LLM-Driven Verilog Coding</a></div>
    <div class="paper-meta">
      📅 2025-06-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) offer significant potential for coding, yet fine-tuning (FT) with curated data is essential for niche languages like Verilog. Using proprietary intellectual property (IP) for FT presents a serious risk, as FT data can be leaked through LLM inference. This leads to a critical dilemma for design houses: seeking to build externally accessible LLMs offering competitive Verilog coding, how can they leverage in-house IP to enhance FT utility while ensuring IP protection? For the first time in the literature, we study this dilemma. Using LLaMA 3.1-8B, we conduct in-house FT on a baseline Verilog dataset (RTLCoder) supplemented with our own in-house IP, which is validated through multiple tape-outs. To rigorously assess IP leakage, we quantify structural similarity (AST/Dolos) and functional equivalence (Synopsys Formality) between generated codes and our in-house IP. We show that our IP can indeed be leaked, confirming the threat. As defense, we evaluate logic locking of Verilog codes (ASSURE). This offers some level of protection, yet reduces the IP's utility for FT and degrades the LLM's performance. Our study shows the need for novel strategies that are both effective and minimally disruptive to FT, an essential effort for enabling design houses to fully utilize their proprietary IP toward LLM-driven Verilog coding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11343v1">From Replication to Redesign: Exploring Pairwise Comparisons for LLM-Based Peer Review</a></div>
    <div class="paper-meta">
      📅 2025-06-12
    </div>
    <details class="paper-abstract">
      The advent of large language models (LLMs) offers unprecedented opportunities to reimagine peer review beyond the constraints of traditional workflows. Despite these opportunities, prior efforts have largely focused on replicating traditional review workflows with LLMs serving as direct substitutes for human reviewers, while limited attention has been given to exploring new paradigms that fundamentally rethink how LLMs can participate in the academic review process. In this paper, we introduce and explore a novel mechanism that employs LLM agents to perform pairwise comparisons among manuscripts instead of individual scoring. By aggregating outcomes from substantial pairwise evaluations, this approach enables a more accurate and robust measure of relative manuscript quality. Our experiments demonstrate that this comparative approach significantly outperforms traditional rating-based methods in identifying high-impact papers. However, our analysis also reveals emergent biases in the selection process, notably a reduced novelty in research topics and an increased institutional imbalance. These findings highlight both the transformative potential of rethinking peer review with LLMs and critical challenges that future systems must address to ensure equity and diversity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11309v1">SwiftSpec: Ultra-Low Latency LLM Decoding by Scaling Asynchronous Speculative Decoding</a></div>
    <div class="paper-meta">
      📅 2025-06-12
    </div>
    <details class="paper-abstract">
      Low-latency decoding for large language models (LLMs) is crucial for applications like chatbots and code assistants, yet generating long outputs remains slow in single-query settings. Prior work on speculative decoding (which combines a small draft model with a larger target model) and tensor parallelism has each accelerated decoding. However, conventional approaches fail to apply both simultaneously due to imbalanced compute requirements (between draft and target models), KV-cache inconsistencies, and communication overheads under small-batch tensor-parallelism. This paper introduces SwiftSpec, a system that targets ultra-low latency for LLM decoding. SwiftSpec redesigns the speculative decoding pipeline in an asynchronous and disaggregated manner, so that each component can be scaled flexibly and remove draft overhead from the critical path. To realize this design, SwiftSpec proposes parallel tree generation, tree-aware KV cache management, and fused, latency-optimized kernels to overcome the challenges listed above. Across 5 model families and 6 datasets, SwiftSpec achieves an average of 1.75x speedup over state-of-the-art speculative decoding systems and, as a highlight, serves Llama3-70B at 348 tokens/s on 8 Nvidia Hopper GPUs, making it the fastest known system for low-latency LLM serving at this scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.09459v2">Ad Auctions for LLMs via Retrieval Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2025-06-12
      | 💬 NeurIPS 2024
    </div>
    <details class="paper-abstract">
      In the field of computational advertising, the integration of ads into the outputs of large language models (LLMs) presents an opportunity to support these services without compromising content integrity. This paper introduces novel auction mechanisms for ad allocation and pricing within the textual outputs of LLMs, leveraging retrieval-augmented generation (RAG). We propose a segment auction where an ad is probabilistically retrieved for each discourse segment (paragraph, section, or entire output) according to its bid and relevance, following the RAG framework, and priced according to competing bids. We show that our auction maximizes logarithmic social welfare, a new notion of welfare that balances allocation efficiency and fairness, and we characterize the associated incentive-compatible pricing rule. These results are extended to multi-ad allocation per segment. An empirical evaluation validates the feasibility and effectiveness of our approach over several ad auction scenarios, and exhibits inherent tradeoffs in metrics as we allow the LLM more flexibility to allocate ads.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11266v1">Invocable APIs derived from NL2SQL datasets for LLM Tool-Calling Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-06-12
      | 💬 10+32 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are routinely deployed as agentic systems, with access to tools that interact with live environments to accomplish tasks. In enterprise deployments these systems need to interact with API collections that can be extremely large and complex, often backed by databases. In order to create datasets with such characteristics, we explore how existing NL2SQL (Natural Language to SQL query) datasets can be used to automatically create NL2API datasets. Specifically, this work describes a novel data generation pipeline that exploits the syntax of SQL queries to construct a functionally equivalent sequence of API calls. We apply this pipeline to one of the largest NL2SQL datasets, BIRD-SQL to create a collection of over 2500 APIs that can be served as invocable tools or REST-endpoints. We pair natural language queries from BIRD-SQL to ground-truth API sequences based on this API pool. We use this collection to study the performance of 10 public LLMs and find that all models struggle to determine the right set of tools (consisting of tasks of intent detection, sequencing with nested function calls, and slot-filling). We find that models have extremely low task completion rates (7-47 percent - depending on the dataset) which marginally improves to 50 percent when models are employed as ReACT agents that interact with the live API environment. The best task completion rates are far below what may be required for effective general-use tool-calling agents, suggesting substantial scope for improvement in current state-of-the-art tool-calling LLMs. We also conduct detailed ablation studies, such as assessing the impact of the number of tools available as well as the impact of tool and slot-name obfuscation. We compare the performance of models on the original SQL generation tasks and find that current models are sometimes able to exploit SQL better than APIs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11237v1">LLM-as-a-Judge for Reference-less Automatic Code Validation and Refinement for Natural Language to Bash in IT Automation</a></div>
    <div class="paper-meta">
      📅 2025-06-12
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      In an effort to automatically evaluate and select the best model and improve code quality for automatic incident remediation in IT Automation, it is crucial to verify if the generated code for remediation action is syntactically and semantically correct and whether it can be executed correctly as intended. There are three approaches: 1) conventional methods use surface form similarity metrics (token match, exact match, etc.) which have numerous limitations, 2) execution-based evaluation focuses more on code functionality based on pass/fail judgments for given test-cases, and 3) LLM-as-a-Judge employs LLMs for automated evaluation to judge if it is a correct answer for a given problem based on pre-defined metrics. In this work, we focused on enhancing LLM-as-a-Judge using bidirectional functionality matching and logic representation for reference-less automatic validation and refinement for Bash code generation to select the best model for automatic incident remediation in IT Automation. We used execution-based evaluation as ground-truth to evaluate our LLM-as-a-Judge metrics. Results show high accuracy and agreement with execution-based evaluation (and up to 8% over baseline). Finally, we built Reflection code agents to utilize judgments and feedback from our evaluation metrics which achieved significant improvement (up to 24% increase in accuracy) for automatic code refinement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11221v1">LLM-as-a-Fuzzy-Judge: Fine-Tuning Large Language Models as a Clinical Evaluation Judge with Fuzzy Logic</a></div>
    <div class="paper-meta">
      📅 2025-06-12
      | 💬 12 pages, 1 figure, 2025 IFSA World Congress NAFIPS Annual Meeting
    </div>
    <details class="paper-abstract">
      Clinical communication skills are critical in medical education, and practicing and assessing clinical communication skills on a scale is challenging. Although LLM-powered clinical scenario simulations have shown promise in enhancing medical students' clinical practice, providing automated and scalable clinical evaluation that follows nuanced physician judgment is difficult. This paper combines fuzzy logic and Large Language Model (LLM) and proposes LLM-as-a-Fuzzy-Judge to address the challenge of aligning the automated evaluation of medical students' clinical skills with subjective physicians' preferences. LLM-as-a-Fuzzy-Judge is an approach that LLM is fine-tuned to evaluate medical students' utterances within student-AI patient conversation scripts based on human annotations from four fuzzy sets, including Professionalism, Medical Relevance, Ethical Behavior, and Contextual Distraction. The methodology of this paper started from data collection from the LLM-powered medical education system, data annotation based on multidimensional fuzzy sets, followed by prompt engineering and the supervised fine-tuning (SFT) of the pre-trained LLMs using these human annotations. The results show that the LLM-as-a-Fuzzy-Judge achieves over 80\% accuracy, with major criteria items over 90\%, effectively leveraging fuzzy logic and LLM as a solution to deliver interpretable, human-aligned assessment. This work suggests the viability of leveraging fuzzy logic and LLM to align with human preferences, advances automated evaluation in medical education, and supports more robust assessment and judgment practices. The GitHub repository of this work is available at https://github.com/2sigmaEdTech/LLMAsAJudge
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12100v1">LLM Embedding-based Attribution (LEA): Quantifying Source Contributions to Generative Model's Response for Vulnerability Analysis</a></div>
    <div class="paper-meta">
      📅 2025-06-12
    </div>
    <details class="paper-abstract">
      Security vulnerabilities are rapidly increasing in frequency and complexity, creating a shifting threat landscape that challenges cybersecurity defenses. Large Language Models (LLMs) have been widely adopted for cybersecurity threat analysis. When querying LLMs, dealing with new, unseen vulnerabilities is particularly challenging as it lies outside LLMs' pre-trained distribution. Retrieval-Augmented Generation (RAG) pipelines mitigate the problem by injecting up-to-date authoritative sources into the model context, thus reducing hallucinations and increasing the accuracy in responses. Meanwhile, the deployment of LLMs in security-sensitive environments introduces challenges around trust and safety. This raises a critical open question: How to quantify or attribute the generated response to the retrieved context versus the model's pre-trained knowledge? This work proposes LLM Embedding-based Attribution (LEA) -- a novel, explainable metric to paint a clear picture on the 'percentage of influence' the pre-trained knowledge vs. retrieved content has for each generated response. We apply LEA to assess responses to 100 critical CVEs from the past decade, verifying its effectiveness to quantify the insightfulness for vulnerability analysis. Our development of LEA reveals a progression of independency in hidden states of LLMs: heavy reliance on context in early layers, which enables the derivation of LEA; increased independency in later layers, which sheds light on why scale is essential for LLM's effectiveness. This work provides security analysts a means to audit LLM-assisted workflows, laying the groundwork for transparent, high-assurance deployments of RAG-enhanced LLMs in cybersecurity operations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12097v1">UCD: Unlearning in LLMs via Contrastive Decoding</a></div>
    <div class="paper-meta">
      📅 2025-06-12
    </div>
    <details class="paper-abstract">
      Machine unlearning aims to remove specific information, e.g. sensitive or undesirable content, from large language models (LLMs) while preserving overall performance. We propose an inference-time unlearning algorithm that uses contrastive decoding, leveraging two auxiliary smaller models, one trained without the forget set and one trained with it, to guide the outputs of the original model using their difference during inference. Our strategy substantially improves the tradeoff between unlearning effectiveness and model utility. We evaluate our approach on two unlearning benchmarks, TOFU and MUSE. Results show notable gains in both forget quality and retained performance in comparison to prior approaches, suggesting that incorporating contrastive decoding can offer an efficient, practical avenue for unlearning concepts in large-scale models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10966v1">GENMANIP: LLM-driven Simulation for Generalizable Instruction-Following Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-06-12
    </div>
    <details class="paper-abstract">
      Robotic manipulation in real-world settings remains challenging, especially regarding robust generalization. Existing simulation platforms lack sufficient support for exploring how policies adapt to varied instructions and scenarios. Thus, they lag behind the growing interest in instruction-following foundation models like LLMs, whose adaptability is crucial yet remains underexplored in fair comparisons. To bridge this gap, we introduce GenManip, a realistic tabletop simulation platform tailored for policy generalization studies. It features an automatic pipeline via LLM-driven task-oriented scene graph to synthesize large-scale, diverse tasks using 10K annotated 3D object assets. To systematically assess generalization, we present GenManip-Bench, a benchmark of 200 scenarios refined via human-in-the-loop corrections. We evaluate two policy types: (1) modular manipulation systems integrating foundation models for perception, reasoning, and planning, and (2) end-to-end policies trained through scalable data collection. Results show that while data scaling benefits end-to-end methods, modular systems enhanced with foundation models generalize more effectively across diverse scenarios. We anticipate this platform to facilitate critical insights for advancing policy generalization in realistic conditions. Project Page: https://genmanip.axi404.top/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10949v1">Monitoring Decomposition Attacks in LLMs with Lightweight Sequential Monitors</a></div>
    <div class="paper-meta">
      📅 2025-06-12
    </div>
    <details class="paper-abstract">
      Current LLM safety defenses fail under decomposition attacks, where a malicious goal is decomposed into benign subtasks that circumvent refusals. The challenge lies in the existing shallow safety alignment techniques: they only detect harm in the immediate prompt and do not reason about long-range intent, leaving them blind to malicious intent that emerges over a sequence of seemingly benign instructions. We therefore propose adding an external monitor that observes the conversation at a higher granularity. To facilitate our study of monitoring decomposition attacks, we curate the largest and most diverse dataset to date, including question-answering, text-to-image, and agentic tasks. We verify our datasets by testing them on frontier LLMs and show an 87% attack success rate on average on GPT-4o. This confirms that decomposition attack is broadly effective. Additionally, we find that random tasks can be injected into the decomposed subtasks to further obfuscate malicious intents. To defend in real time, we propose a lightweight sequential monitoring framework that cumulatively evaluates each subtask. We show that a carefully prompt engineered lightweight monitor achieves a 93% defense success rate, beating reasoning models like o3 mini as a monitor. Moreover, it remains robust against random task injection and cuts cost by 90% and latency by 50%. Our findings suggest that lightweight sequential monitors are highly effective in mitigating decomposition attacks and are viable in deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.15724v2">LLM-Cure: LLM-based Competitor User Review Analysis for Feature Enhancement</a></div>
    <div class="paper-meta">
      📅 2025-06-12
      | 💬 25 pages
    </div>
    <details class="paper-abstract">
      The exponential growth of the mobile app market underscores the importance of constant innovation and rapid response to user demands. As user satisfaction is paramount to the success of a mobile application (app), developers typically rely on user reviews, which represent user feedback that includes ratings and comments to identify areas for improvement. However, the sheer volume of user reviews poses challenges in manual analysis, necessitating automated approaches. Existing automated approaches either analyze only the target apps reviews, neglecting the comparison of similar features to competitors or fail to provide suggestions for feature enhancement. To address these gaps, we propose a Large Language Model (LLM)-based Competitive User Review Analysis for Feature Enhancement) (LLM-Cure), an approach powered by LLMs to automatically generate suggestion s for mobile app feature improvements. More specifically, LLM-Cure identifies and categorizes features within reviews by applying LLMs. When provided with a complaint in a user review, LLM-Cure curates highly rated (4 and 5 stars) reviews in competing apps related to the complaint and proposes potential improvements tailored to the target application. We evaluate LLM-Cure on 1,056,739 reviews of 70 popular Android apps. Our evaluation demonstrates that LLM-Cure significantly outperforms the state-of-the-art approaches in assigning features to reviews by up to 13% in F1-score, up to 16% in recall and up to 11% in precision. Additionally, LLM-Cure demonstrates its capability to provide suggestions for resolving user complaints. We verify the suggestions using the release notes that reflect the changes of features in the target mobile app. LLM-Cure achieves a promising average of 73% of the implementation of the provided suggestions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03710v2">Improving LLM Safety Alignment with Dual-Objective Optimization</a></div>
    <div class="paper-meta">
      📅 2025-06-12
      | 💬 ICML 2025
    </div>
    <details class="paper-abstract">
      Existing training-time safety alignment techniques for large language models (LLMs) remain vulnerable to jailbreak attacks. Direct preference optimization (DPO), a widely deployed alignment method, exhibits limitations in both experimental and theoretical contexts as its loss function proves suboptimal for refusal learning. Through gradient-based analysis, we identify these shortcomings and propose an improved safety alignment that disentangles DPO objectives into two components: (1) robust refusal training, which encourages refusal even when partial unsafe generations are produced, and (2) targeted unlearning of harmful knowledge. This approach significantly increases LLM robustness against a wide range of jailbreak attacks, including prefilling, suffix, and multi-turn attacks across both in-distribution and out-of-distribution scenarios. Furthermore, we introduce a method to emphasize critical refusal tokens by incorporating a reward-based token-level weighting mechanism for refusal learning, which further improves the robustness against adversarial exploits. Our research also suggests that robustness to jailbreak attacks is correlated with token distribution shifts in the training process and internal representations of refusal and harmful tokens, offering valuable directions for future research in LLM safety alignment. The code is available at https://github.com/wicai24/DOOR-Alignment
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10922v1">Robustly Improving LLM Fairness in Realistic Settings via Interpretability</a></div>
    <div class="paper-meta">
      📅 2025-06-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed in high-stakes hiring applications, making decisions that directly impact people's careers and livelihoods. While prior studies suggest simple anti-bias prompts can eliminate demographic biases in controlled evaluations, we find these mitigations fail when realistic contextual details are introduced. We address these failures through internal bias mitigation: by identifying and neutralizing sensitive attribute directions within model activations, we achieve robust bias reduction across all tested scenarios. Across leading commercial (GPT-4o, Claude 4 Sonnet, Gemini 2.5 Flash) and open-source models (Gemma-2 27B, Gemma-3, Mistral-24B), we find that adding realistic context such as company names, culture descriptions from public careers pages, and selective hiring constraints (e.g.,``only accept candidates in the top 10\%") induces significant racial and gender biases (up to 12\% differences in interview rates). When these biases emerge, they consistently favor Black over White candidates and female over male candidates across all tested models and scenarios. Moreover, models can infer demographics and become biased from subtle cues like college affiliations, with these biases remaining invisible even when inspecting the model's chain-of-thought reasoning. To address these limitations, our internal bias mitigation identifies race and gender-correlated directions and applies affine concept editing at inference time. Despite using directions from a simple synthetic dataset, the intervention generalizes robustly, consistently reducing bias to very low levels (typically under 1\%, always below 2.5\%) while largely maintaining model performance. Our findings suggest that practitioners deploying LLMs for hiring should adopt more realistic evaluation methodologies and consider internal mitigation strategies for equitable outcomes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10903v1">Beyond Gold Standards: Epistemic Ensemble of LLM Judges for Formal Mathematical Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-06-12
    </div>
    <details class="paper-abstract">
      Autoformalization plays a crucial role in formal mathematical reasoning by enabling the automatic translation of natural language statements into formal languages. While recent advances using large language models (LLMs) have shown promising results, methods for automatically evaluating autoformalization remain underexplored. As one moves to more complex domains (e.g., advanced mathematics), human evaluation requires significant time and domain expertise, especially as the complexity of the underlying statements and background knowledge increases. LLM-as-a-judge presents a promising approach for automating such evaluation. However, existing methods typically employ coarse-grained and generic evaluation criteria, which limit their effectiveness for advanced formal mathematical reasoning, where quality hinges on nuanced, multi-granular dimensions. In this work, we take a step toward addressing this gap by introducing a systematic, automatic method to evaluate autoformalization tasks. The proposed method is based on an epistemically and formally grounded ensemble (EFG) of LLM judges, defined on criteria encompassing logical preservation (LP), mathematical consistency (MC), formal validity (FV), and formal quality (FQ), resulting in a transparent assessment that accounts for different contributing factors. We validate the proposed framework to serve as a proxy for autoformalization assessment within the domain of formal mathematics. Overall, our experiments demonstrate that the EFG ensemble of LLM judges is a suitable emerging proxy for evaluation, more strongly correlating with human assessments than a coarse-grained model, especially when assessing formal qualities. These findings suggest that LLM-as-judges, especially when guided by a well-defined set of atomic properties, could offer a scalable, interpretable, and reliable support for evaluating formal mathematical reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14432v2">PLAY2PROMPT: Zero-shot Tool Instruction Optimization for LLM Agents via Tool Play</a></div>
    <div class="paper-meta">
      📅 2025-06-12
      | 💬 ACL 2025 Long Paper (Findings)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly integrated with specialized external tools, yet many tasks demand zero-shot tool usage with minimal or noisy documentation. Existing solutions rely on manual rewriting or labeled data for validation, making them inapplicable in true zero-shot settings. To address these challenges, we propose PLAY2PROMPT, an automated framework that systematically "plays" with each tool to explore its input-output behaviors. Through this iterative trial-and-error process, PLAY2PROMPT refines tool documentation and generates usage examples without any labeled data. These examples not only guide LLM inference but also serve as validation to further enhance tool utilization. Extensive experiments on real-world tasks demonstrate that PLAY2PROMPT significantly improves zero-shot tool performance across both open and closed models, offering a scalable and effective solution for domain-specific tool integration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10885v1">Slimming Down LLMs Without Losing Their Minds</a></div>
    <div class="paper-meta">
      📅 2025-06-12
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      This paper investigates and validates the impact of fine-tuning on large language model performance, focusing on parameter-efficient methods (LoRA and QLoRA). We evaluate model capabilities across three key domains: (1) commonsense reasoning (HellaSwag), (2) mathematical reasoning (GSM8K), and (3) multi-domain knowledge (MMLU-CS). Our findings demonstrate that: (1) LoRA-based methods effectively improve task-specific performance while maintaining computational efficiency, and (2) performance strongly depends on alignment between fine-tuning dataset and benchmark tasks. The study provides both theoretical insights into parameter-efficient mechanisms and practical guidance for developers implementing efficient LLM adaptation with limited resources.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10859v1">Precise Zero-Shot Pointwise Ranking with LLMs through Post-Aggregated Global Context Information</a></div>
    <div class="paper-meta">
      📅 2025-06-12
      | 💬 Accepted by SIGIR 2025
    </div>
    <details class="paper-abstract">
      Recent advancements have successfully harnessed the power of Large Language Models (LLMs) for zero-shot document ranking, exploring a variety of prompting strategies. Comparative approaches like pairwise and listwise achieve high effectiveness but are computationally intensive and thus less practical for larger-scale applications. Scoring-based pointwise approaches exhibit superior efficiency by independently and simultaneously generating the relevance scores for each candidate document. However, this independence ignores critical comparative insights between documents, resulting in inconsistent scoring and suboptimal performance. In this paper, we aim to improve the effectiveness of pointwise methods while preserving their efficiency through two key innovations: (1) We propose a novel Global-Consistent Comparative Pointwise Ranking (GCCP) strategy that incorporates global reference comparisons between each candidate and an anchor document to generate contrastive relevance scores. We strategically design the anchor document as a query-focused summary of pseudo-relevant candidates, which serves as an effective reference point by capturing the global context for document comparison. (2) These contrastive relevance scores can be efficiently Post-Aggregated with existing pointwise methods, seamlessly integrating essential Global Context information in a training-free manner (PAGC). Extensive experiments on the TREC DL and BEIR benchmark demonstrate that our approach significantly outperforms previous pointwise methods while maintaining comparable efficiency. Our method also achieves competitive performance against comparative methods that require substantially more computational resources. More analyses further validate the efficacy of our anchor construction strategy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10829v1">LLM-Driven Personalized Answer Generation and Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-06-12
      | 💬 This is the preprint version of a paper accepted at AIED 2025. The final version will be published by Springer
    </div>
    <details class="paper-abstract">
      Online learning has experienced rapid growth due to its flexibility and accessibility. Personalization, adapted to the needs of individual learners, is crucial for enhancing the learning experience, particularly in online settings. A key aspect of personalization is providing learners with answers customized to their specific questions. This paper therefore explores the potential of Large Language Models (LLMs) to generate personalized answers to learners' questions, thereby enhancing engagement and reducing the workload on educators. To evaluate the effectiveness of LLMs in this context, we conducted a comprehensive study using the StackExchange platform in two distinct areas: language learning and programming. We developed a framework and a dataset for validating automatically generated personalized answers. Subsequently, we generated personalized answers using different strategies, including 0-shot, 1-shot, and few-shot scenarios. The generated answers were evaluated using three methods: 1. BERTScore, 2. LLM evaluation, and 3. human evaluation. Our findings indicated that providing LLMs with examples of desired answers (from the learner or similar learners) can significantly enhance the LLMs' ability to tailor responses to individual learners' needs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10822v1">ReCUT: Balancing Reasoning Length and Accuracy in LLMs via Stepwise Trails and Preference Optimization</a></div>
    <div class="paper-meta">
      📅 2025-06-12
    </div>
    <details class="paper-abstract">
      Recent advances in Chain-of-Thought (CoT) prompting have substantially improved the reasoning capabilities of Large Language Models (LLMs). However, these methods often suffer from overthinking, leading to unnecessarily lengthy or redundant reasoning traces. Existing approaches attempt to mitigate this issue through curating multiple reasoning chains for training LLMs, but their effectiveness is often constrained by the quality of the generated data and prone to overfitting. To address the challenge, we propose Reasoning Compression ThroUgh Stepwise Trials (ReCUT), a novel method aimed at balancing the accuracy and length of reasoning trajectory. Specifically, ReCUT employs a stepwise exploration mechanism and a long-short switched sampling strategy, enabling LLMs to incrementally generate diverse reasoning paths. These paths are evaluated and used to construct preference pairs to train two specialized models (Gemini LLMs)-one optimized for reasoning accuracy, the other for shorter reasoning. A final integrated model is obtained by interpolating the parameters of these two models. Experimental results across multiple math reasoning datasets and backbone models demonstrate that ReCUT significantly reduces reasoning lengths by approximately 30-50%, while maintaining or improving reasoning accuracy compared to various baselines. All codes and data will be released via https://github.com/NEUIR/ReCUT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.18765v2">A Vision for Auto Research with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-06-12
    </div>
    <details class="paper-abstract">
      This paper introduces Agent-Based Auto Research, a structured multi-agent framework designed to automate, coordinate, and optimize the full lifecycle of scientific research. Leveraging the capabilities of large language models (LLMs) and modular agent collaboration, the system spans all major research phases, including literature review, ideation, methodology planning, experimentation, paper writing, peer review response, and dissemination. By addressing issues such as fragmented workflows, uneven methodological expertise, and cognitive overload, the framework offers a systematic and scalable approach to scientific inquiry. Preliminary explorations demonstrate the feasibility and potential of Auto Research as a promising paradigm for self-improving, AI-driven research processes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10779v1">Improving Named Entity Transcription with Contextual LLM-based Revision</a></div>
    <div class="paper-meta">
      📅 2025-06-12
    </div>
    <details class="paper-abstract">
      With recent advances in modeling and the increasing amount of supervised training data, automatic speech recognition (ASR) systems have achieved remarkable performance on general speech. However, the word error rate (WER) of state-of-the-art ASR remains high for named entities. Since named entities are often the most critical keywords, misrecognizing them can affect all downstream applications, especially when the ASR system functions as the front end of a complex system. In this paper, we introduce a large language model (LLM) revision mechanism to revise incorrect named entities in ASR predictions by leveraging the LLM's reasoning ability as well as local context (e.g., lecture notes) containing a set of correct named entities. Finally, we introduce the NER-MIT-OpenCourseWare dataset, containing 45 hours of data from MIT courses for development and testing. On this dataset, our proposed technique achieves up to 30\% relative WER reduction for named entities.
    </details>
</div>
