# llm - 2026_02

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- Part 2
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.15569v1">"What Are You Doing?": Effects of Intermediate Feedback from Agentic LLM In-Car Assistants During Multi-Step Processing</a></div>
    <div class="paper-meta">
      📅 2026-02-17
      | 💬 Accepted (conditionally) at CHI 2026
    </div>
    <details class="paper-abstract">
      Agentic AI assistants that autonomously perform multi-step tasks raise open questions for user experience: how should such systems communicate progress and reasoning during extended operations, especially in attention-critical contexts such as driving? We investigate feedback timing and verbosity from agentic LLM-based in-car assistants through a controlled, mixed-methods study (N=45) comparing planned steps and intermediate results feedback against silent operation with final-only response. Using a dual-task paradigm with an in-car voice assistant, we found that intermediate feedback significantly improved perceived speed, trust, and user experience while reducing task load - effects that held across varying task complexities and interaction contexts. Interviews further revealed user preferences for an adaptive approach: high initial transparency to establish trust, followed by progressively reducing verbosity as systems prove reliable, with adjustments based on task stakes and situational context. We translate our empirical findings into design implications for feedback timing and verbosity in agentic assistants, balancing transparency and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.02594v2">OpenAIs HealthBench in Action: Evaluating an LLM-Based Medical Assistant on Realistic Clinical Queries</a></div>
    <div class="paper-meta">
      📅 2026-02-17
      | 💬 13 pages, two graphs
    </div>
    <details class="paper-abstract">
      Evaluating large language models (LLMs) on their ability to generate high-quality, accurate, situationally aware answers to clinical questions requires going beyond conventional benchmarks to assess how these systems behave in complex, high-stakes clinical scenarios. Traditional evaluations are often limited to multiple-choice questions that fail to capture essential competencies such as contextual reasoning, contextual awareness, and uncertainty handling. To address these limitations, we evaluate our agentic RAG-based clinical support assistant, DR. INFO, using HealthBench, a rubric-driven benchmark composed of open-ended, expert-annotated health conversations. On the Hard subset of 1,000 challenging examples, DR. INFO achieves a HealthBench Hard score of 0.68, outperforming leading frontier LLMs including the GPT-5 model family (GPT-5: 0.46, GPT-5.2: 0.42, GPT-5.1: 0.40), Grok 3 (0.23), Gemini 2.5 Pro (0.19), and Claude 3.7 Sonnet (0.02) across all behavioral axes (accuracy, completeness, instruction following, etc.). In a separate 100-sample evaluation against similar agentic RAG assistants (OpenEvidence and Pathway.md, now DoxGPT by Doximity), it maintains a performance lead with a HealthBench Hard score of 0.72. These results highlight the strengths of DR. INFO in communication, instruction following, and accuracy, while also revealing areas for improvement in context awareness and response completeness. Overall, the findings underscore the utility of behavior-level, rubric-based evaluation for building reliable and trustworthy AI-enabled clinical support systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2408.00539v2">Intermittent Semi-Working Mask: A New Masking Paradigm for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-17
    </div>
    <details class="paper-abstract">
      Multi-turn dialogues and context-intensive tasks challenge Large Language Models (LLMs) to integrate long histories without sacrificing generation quality. Although prefix LLMs can better exploit historical context via bidirectional attention on prefix tokens, they are rarely used in practice because multi-turn training requires many duplicated triplets, and its bidirectional prefix prevents KV-cache reuse at inference time, driving up high cost and latency. To retain the contextual understanding of prefix mask while preserving the inference-time efficiency of causal mask, we introduce Intermittent Semi-working Mask (ISM), a masking scheme that injects sparse bidirectional attention into the causal backbone. ISM alternates bidirectional attention over query segments with unidirectional attention over answer segments, enabling the synthesis of in-context while preserving global causality. This design eliminates triplet expansion during training and maintains KV-cache reuse during inference, yielding latency comparable to standard causal LLMs. ISM is architecture-agnostic and parameter-free, adding only minimal latency. Across extensive evaluations, ISM outperforms causal baselines not only on multi-turn dialogue, but also on context-intensive tasks like mathematical reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.19919v2">Your AI Bosses Are Still Prejudiced: The Emergence of Stereotypes in LLM-Based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2026-02-17
    </div>
    <details class="paper-abstract">
      While stereotypes are well-documented in human social interactions, AI systems are often presumed to be less susceptible to such biases. Previous studies have focused on biases inherited from training data, but whether stereotypes can emerge spontaneously in AI agent interactions merits further exploration. Through a novel experimental framework simulating workplace interactions with neutral initial conditions, we investigate the emergence and evolution of stereotypes in LLM-based multi-agent systems. Our findings reveal that (1) LLM-Based AI agents develop stereotype-driven biases in their interactions despite beginning without predefined biases; (2) stereotype effects intensify with increased interaction rounds and decision-making power, particularly after introducing hierarchical structures; (3) these systems exhibit group effects analogous to human social behavior, including halo effects, confirmation bias, and role congruity; and (4) these stereotype patterns manifest consistently across different LLM architectures. Through comprehensive quantitative analysis, these findings suggest that stereotype formation in AI systems may arise as an emergent property of multi-agent interactions, rather than merely from training data biases. Our work underscores the need for future research to explore the underlying mechanisms of this phenomenon and develop strategies to mitigate its ethical impacts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.02108v3">Out of the Memory Barrier: A Highly Memory Efficient Training System for LLMs with Million-Token Contexts</a></div>
    <div class="paper-meta">
      📅 2026-02-17
    </div>
    <details class="paper-abstract">
      Training Large Language Models (LLMs) on long contexts is severely constrained by prohibitive GPU memory overhead, not training time. The primary culprits are the activations, whose memory footprints scale linearly with sequence length. We introduce OOMB, a highly memory-efficient training system that directly confronts this barrier. Our approach employs a chunk-recurrent training framework with on-the-fly activation recomputation, which maintains a constant activation memory footprint (O(1)) and shifts the primary bottleneck to the growing KV cache. To manage the KV cache, OOMB integrates a suite of synergistic optimizations: a paged memory manager for both the KV cache and its gradients to eliminate fragmentation, asynchronous CPU offloading to hide data transfer latency, and page-level sparse attention to reduce both computational complexity and communication overhead. The synergy of these techniques yields exceptional efficiency. Our empirical results show that for every additional 10K tokens of context, the end-to-end training memory overhead increases by a mere 10MB for Qwen2.5-7B. This allows training Qwen2.5-7B with a 4M-token context on a single H200 GPU, a feat that would otherwise require a large cluster using context parallelism. This work represents a substantial advance in resource efficiency for long-context LLM training. The source code is available at https://github.com/wenhaoli-xmu/OOMB.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.15481v1">LLM-as-Judge on a Budget</a></div>
    <div class="paper-meta">
      📅 2026-02-17
    </div>
    <details class="paper-abstract">
      LLM-as-a-judge has emerged as a cornerstone technique for evaluating large language models by leveraging LLM reasoning to score prompt-response pairs. Since LLM judgments are stochastic, practitioners commonly query each pair multiple times to estimate mean scores accurately. This raises a critical challenge: given a fixed computational budget $B$, how to optimally allocate queries across $K$ prompt-response pairs to minimize estimation error? % We present a principled variance-adaptive approach leveraging multi-armed bandit theory and concentration inequalities. Our method dynamically allocates queries based on estimated score variances, concentrating resources where uncertainty is highest. Further, our algorithm is shown to achieve a worst-case score-estimation error of $\tilde{O}\left(\sqrt{\frac{\sum_{i=1}^K σ_i^2}{B}}\right)$, $σ_i^2$ being the unknown score variance for pair $i \in [K]$ with near-optimal budget allocation. % Experiments on \emph{Summarize-From-Feedback} and \emph{HelpSteer2} demonstrate that our method significantly outperforms uniform allocation, reducing worst-case estimation error while maintaining identical budgets. Our work establishes a theoretical foundation for efficient LLM evaluation with practical implications for AI safety, model alignment, and automated assessment at scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.15456v1">In Agents We Trust, but Who Do Agents Trust? Latent Source Preferences Steer LLM Generations</a></div>
    <div class="paper-meta">
      📅 2026-02-17
      | 💬 ICLR 2026
    </div>
    <details class="paper-abstract">
      Agents based on Large Language Models (LLMs) are increasingly being deployed as interfaces to information on online platforms. These agents filter, prioritize, and synthesize information retrieved from the platforms' back-end databases or via web search. In these scenarios, LLM agents govern the information users receive, by drawing users' attention to particular instances of retrieved information at the expense of others. While much prior work has focused on biases in the information LLMs themselves generate, less attention has been paid to the factors that influence what information LLMs select and present to users. We hypothesize that when information is attributed to specific sources (e.g., particular publishers, journals, or platforms), current LLMs exhibit systematic latent source preferences- that is, they prioritize information from some sources over others. Through controlled experiments on twelve LLMs from six model providers, spanning both synthetic and real-world tasks, we find that several models consistently exhibit strong and predictable source preferences. These preferences are sensitive to contextual framing, can outweigh the influence of content itself, and persist despite explicit prompting to avoid them. They also help explain phenomena such as the observed left-leaning skew in news recommendations in prior work. Our findings advocate for deeper investigation into the origins of these preferences, as well as for mechanisms that provide users with transparency and control over the biases guiding LLM-powered agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.15436v1">Measuring Social Integration Through Participation: Categorizing Organizations and Leisure Activities in the Displaced Karelians Interview Archive using LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-17
      | 💬 Presented at: The 10th Joint SIGHUM Workshop on Computational Linguistics for Cultural Heritage, Social Sciences, Humanities and Literature; EACL 2026 Workshop
    </div>
    <details class="paper-abstract">
      Digitized historical archives make it possible to study everyday social life on a large scale, but the information extracted directly from text often does not directly allow one to answer the research questions posed by historians or sociologists in a quantitative manner. We address this problem in a large collection of Finnish World War II Karelian evacuee family interviews. Prior work extracted more than 350K mentions of leisure time activities and organizational memberships from these interviews, yielding 71K unique activity and organization names -- far too many to analyze directly. We develop a categorization framework that captures key aspects of participation (the kind of activity/organization, how social it typically is, how regularly it happens, and how physically demanding it is). We annotate a gold-standard set to allow for a reliable evaluation, and then test whether large language models can apply the same schema at scale. Using a simple voting approach across multiple model runs, we find that an open-weight LLM can closely match expert judgments. Finally, we apply the method to label the 350K entities, producing a structured resource for downstream studies of social integration and related outcomes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.13629v2">Search in Transition: A Study of University Students Perspectives on Using LLMs and Traditional Search Engines in English Test Problem Solving for Higher Study</a></div>
    <div class="paper-meta">
      📅 2026-02-17
      | 💬 arXiv admin note: substantial text overlap with arXiv:2510.17726 by other authors
    </div>
    <details class="paper-abstract">
      As Artificial Intelligence (AI) becomes increasingly integrated into education, university students preparing for English language tests are frequently shifting between traditional search engines like Google and large language models (LLMs) to assist with problem-solving. This study explores students perceptions of these tools, particularly in terms of usability, efficiency, and how they fit into English test preparation practices. Using a mixed-methods design, we collected survey data from 140 university students across various academic fields and conducted in-depth interviews with 20 participants. Quantitative analyses, including ANOVA and chi-square tests, were applied to assess differences in perceived efficiency, satisfaction, and overall tool preference. The qualitative results reveal that students strategically alternate between GPT and Google based on task requirements. Google is primarily used for accessing reliable, multi-source information and verifying rules, whereas GPT is favored for summarizing content, providing explanations, paraphrasing, and drafting responses for English test tasks. Since neither tool independently satisfies all aspects of English language test preparation, students expressed a clear preference for an integrated approach. In response, this study proposes a prototype chatbot embedded within a search interface, combining GPTs interactive capabilities with Googles credibility to enhance test preparation and reduce cognitive load.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.15391v1">Improving LLM Reliability through Hybrid Abstention and Adaptive Detection</a></div>
    <div class="paper-meta">
      📅 2026-02-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) deployed in production environments face a fundamental safety-utility trade-off either a strict filtering mechanisms prevent harmful outputs but often block benign queries or a relaxed controls risk unsafe content generation. Conventional guardrails based on static rules or fixed confidence thresholds are typically context-insensitive and computationally expensive, resulting in high latency and degraded user experience. To address these limitations, we introduce an adaptive abstention system that dynamically adjusts safety thresholds based on real-time contextual signals such as domain and user history. The proposed framework integrates a multi-dimensional detection architecture composed of five parallel detectors, combined through a hierarchical cascade mechanism to optimize both speed and precision. The cascade design reduces unnecessary computation by progressively filtering queries, achieving substantial latency improvements compared to non-cascaded models and external guardrail systems. Extensive evaluation on mixed and domain-specific workloads demonstrates significant reductions in false positives, particularly in sensitive domains such as medical advice and creative writing. The system maintains high safety precision and near-perfect recall under strict operating modes. Overall, our context-aware abstention framework effectively balances safety and utility while preserving performance, offering a scalable solution for reliable LLM deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19117v2">Stop saying LLM: Large Discourse Models (LDM) and Artificial Discursive Agent (ADA)?</a></div>
    <div class="paper-meta">
      📅 2026-02-17
      | 💬 in French language
    </div>
    <details class="paper-abstract">
      This paper proposes an epistemological shift in the analysis of large generative models, replacing the category ''Large Language Models'' (LLM) with that of ''Large Discourse Models'' (LDM), and then with that of Artificial Discursive Agent (ADA). The theoretical framework is based on an ontological triad distinguishing three regulatory instances: the apprehension of the phenomenal regularities of the referential world, the structuring of embodied cognition, and the structural-linguistic sedimentation of the utterance within a socio-historical context. LDMs, operating on the product of these three instances (the document), model the discursive projection of a portion of human experience reified by the learning corpus. The proposed program aims to replace the ''fascination/fear'' dichotomy with public trials and procedures that make the place, uses, and limits of artificial discursive agents in contemporary social space decipherable, situating this approach within a perspective of governance and co-regulation involving the State, industry, civil society, and academia.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.15388v1">Iterative LLM-Based Assertion Generation Using Syntax-Semantic Representations for Functional Coverage-Guided Verification</a></div>
    <div class="paper-meta">
      📅 2026-02-17
      | 💬 6 pages, 6 figures
    </div>
    <details class="paper-abstract">
      While leveraging LLMs to automatically generate SystemVerilog assertions (SVAs) from natural language specifications holds great potential, existing techniques face a key challenge: LLMs often lack sufficient understanding of IC design, leading to poor assertion quality in a single pass. Therefore, verifying whether the generated assertions effectively cover the functional specifications and designing feedback mechanisms based on this coverage remain significant hurdles. To address these limitations, this paper introduces CoverAssert, a novel iterative framework for optimizing SVA generation with LLMs. The core contribution is a lightweight mechanism for matching generated assertions with specific functional descriptions in the specifications. CoverAssert achieves this by clustering the joint representations of semantic features of LLM-generated assertions and structural features extracted from abstract syntax trees (ASTs) about signals related to assertions, and then mapping them back to the specifications to analyze functional coverage quality. Leveraging this capability, CoverAssert constructs a feedback loop based on functional coverage to guide LLMs in prioritizing uncovered functional points, thereby iteratively improving assertion quality. Experimental evaluations on four open-source designs demonstrate that integrating CoverAssert with state-of-the-art generators, AssertLLM and Spec2Assertion, achieves average improvements of 9.57 % in branch coverage, 9.64 % in statement coverage, and 15.69 % in toggle coverage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.15350v1">Fine-Tuning LLMs to Generate Economical and Reliable Actions for the Power Grid</a></div>
    <div class="paper-meta">
      📅 2026-02-17
    </div>
    <details class="paper-abstract">
      Public Safety Power Shutoffs (PSPS) force rapid topology changes that can render standard operating points infeasible, requiring operators to quickly identify corrective transmission switching actions that reduce load shedding while maintaining acceptable voltage behavior. We present a verifiable, multi-stage adaptation pipeline that fine-tunes an instruction-tuned large language model (LLM) to generate \emph{open-only} corrective switching plans from compact PSPS scenario summaries under an explicit switching budget. First, supervised fine-tuning distills a DC-OPF MILP oracle into a constrained action grammar that enables reliable parsing and feasibility checks. Second, direct preference optimization refines the policy using AC-evaluated preference pairs ranked by a voltage-penalty metric, injecting voltage-awareness beyond DC imitation. Finally, best-of-$N$ selection provides an inference-time addition by choosing the best feasible candidate under the target metric. On IEEE 118-bus PSPS scenarios, fine-tuning substantially improves DC objective values versus zero-shot generation, reduces AC power-flow failure from 50\% to single digits, and improves voltage-penalty outcomes on the common-success set. Code and data-generation scripts are released to support reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.14770v2">Multi-Agent Comedy Club: Investigating Community Discussion Effects on LLM Humor Generation</a></div>
    <div class="paper-meta">
      📅 2026-02-17
      | 💬 18 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Prior work has explored multi-turn interaction and feedback for LLM writing, but evaluations still largely center on prompts and localized feedback, leaving persistent public reception in online communities underexamined. We test whether broadcast community discussion improves stand-up comedy writing in a controlled multi-agent sandbox: in the discussion condition, critic and audience threads are recorded, filtered, stored as social memory, and later retrieved to condition subsequent generations, whereas the baseline omits discussion. Across 50 rounds (250 paired monologues) judged by five expert annotators using A/B preference and a 15-item rubric, discussion wins 75.6% of instances and improves Craft/Clarity (Δ = 0.440) and Social Response (Δ = 0.422), with occasional increases in aggressive humor.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.15336v1">Human-AI Interaction: Evaluating LLM Reasoning on Digital Logic Circuit included Graph Problems, in terms of creativity in design and analysis</a></div>
    <div class="paper-meta">
      📅 2026-02-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used by undergraduate students as on-demand tutors, yet their reliability on circuit- and diagram-based digital logic problems remains unclear. We present a human- AI study evaluating three widely used LLMs (GPT, Gemini, and Claude) on 10 undergraduate-level digital logic questions spanning non-standard counters, JK-based state transitions, timing diagrams, frequency division, and finite-state machines. Twenty-four students performed pairwise model comparisons, providing per-question judgments on (i) preferred model, (ii) perceived correctness, (iii) consistency, (iv) verbosity, and (v) confidence, along with global ratings of overall model quality, satisfaction across multiple dimensions (e.g., accuracy and clarity), and perceived mental effort required to verify answers. To benchmark technical validity, we applied an independent judge-based evaluation against official solutions for all ten questions, using strict correctness criteria. Results reveal a consistent gap between perceived helpfulness and formal correctness: for the most sequentially demanding problems (Q1- Q7), none of the evaluated LLMs matched the official answers, despite producing confident, well-structured explanations that students often rated favorably. Error analysis indicates that models frequently default to canonical textbook templates (e.g., standard ripple counters) and struggle to translate circuit structure into exact state evolution and timing behavior. These findings suggest that, without verification scaffolds, LLMs may be unreliable for core digital logic topics and can inadvertently reinforce misconceptions in undergraduate instruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.15325v1">AgriWorld:A World Tools Protocol Framework for Verifiable Agricultural Reasoning with Code-Executing LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-17
    </div>
    <details class="paper-abstract">
      Foundation models for agriculture are increasingly trained on massive spatiotemporal data (e.g., multi-spectral remote sensing, soil grids, and field-level management logs) and achieve strong performance on forecasting and monitoring. However, these models lack language-based reasoning and interactive capabilities, limiting their usefulness in real-world agronomic workflows. Meanwhile, large language models (LLMs) excel at interpreting and generating text, but cannot directly reason over high-dimensional, heterogeneous agricultural datasets. We bridge this gap with an agentic framework for agricultural science. It provides a Python execution environment, AgriWorld, exposing unified tools for geospatial queries over field parcels, remote-sensing time-series analytics, crop growth simulation, and task-specific predictors (e.g., yield, stress, and disease risk). On top of this environment, we design a multi-turn LLM agent, Agro-Reflective, that iteratively writes code, observes execution results, and refines its analysis via an execute-observe-refine loop. We introduce AgroBench, with scalable data generation for diverse agricultural QA spanning lookups, forecasting, anomaly detection, and counterfactual "what-if" analysis. Experiments outperform text-only and direct tool-use baselines, validating execution-driven reflection for reliable agricultural reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.06601v2">Deep Ignorance: Filtering Pretraining Data Builds Tamper-Resistant Safeguards into Open-Weight LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-17
      | 💬 https://deepignorance.ai/
    </div>
    <details class="paper-abstract">
      Open-weight AI systems offer unique benefits, including enhanced transparency, open research, and decentralized access. However, they are vulnerable to tampering attacks which can efficiently elicit harmful behaviors by modifying weights or activations. Currently, there is not yet a robust science of open-weight model risk management. Existing safety fine-tuning methods and other post-training techniques have struggled to make LLMs resistant to more than a few dozen steps of adversarial fine-tuning. In this paper, we investigate whether filtering text about dual-use topics from training data can prevent unwanted capabilities and serve as a more tamper-resistant safeguard. We introduce a multi-stage pipeline for scalable data filtering and show that it offers a tractable and effective method for minimizing biothreat proxy knowledge in LLMs. We pretrain multiple 6.9B-parameter models from scratch and find that they exhibit substantial resistance to adversarial fine-tuning attacks on up to 10,000 steps and 300M tokens of biothreat-related text -- outperforming existing post-training baselines by over an order of magnitude -- with no observed degradation to unrelated capabilities. However, while filtered models lack internalized dangerous knowledge, we find that they can still leverage such information when it is provided in context (e.g., via search tool augmentation), demonstrating a need for a defense-in-depth approach. Overall, these findings help to establish pretraining data curation as a promising layer of defense for open-weight AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.15318v1">Sparrow: Text-Anchored Window Attention with Visual-Semantic Glimpsing for Speculative Decoding in Video LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-17
      | 💬 15 pages , 6 figures
    </div>
    <details class="paper-abstract">
      Although speculative decoding is widely used to accelerate Vision-Language Models (VLMs) inference, it faces severe performance collapse when applied to Video Large Language Models (Vid-LLMs). The draft model typically falls into the trap of attention dilution and negative visual gain due to key-value cache explosion and context window mismatches. We observe a visual semantic internalization phenomenon in Vid-LLMs, indicating that critical visual semantics are implicitly encoded into text hidden states during deep-layer interactions, which renders raw visual inputs structurally redundant during deep inference. To address this, we propose the Sparrow framework, which first utilizes visually-aware text-anchored window attention via hidden state reuse to fully offload visual computation to the target model, and leverages intermediate-layer visual state bridging to train the draft model with semantic-rich intermediate states, thereby filtering out low-level visual noise. Additionally, a multi-token prediction strategy is introduced to bridge the training-inference distribution shift. Experiments show that Sparrow achieves an average speedup of 2.82x even with 25k visual tokens, effectively resolving the performance degradation in long sequences and offering a practical solution for real-time long video tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.15313v1">Mnemis: Dual-Route Retrieval on Hierarchical Graphs for Long-Term LLM Memory</a></div>
    <div class="paper-meta">
      📅 2026-02-17
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      AI Memory, specifically how models organizes and retrieves historical messages, becomes increasingly valuable to Large Language Models (LLMs), yet existing methods (RAG and Graph-RAG) primarily retrieve memory through similarity-based mechanisms. While efficient, such System-1-style retrieval struggles with scenarios that require global reasoning or comprehensive coverage of all relevant information. In this work, We propose Mnemis, a novel memory framework that integrates System-1 similarity search with a complementary System-2 mechanism, termed Global Selection. Mnemis organizes memory into a base graph for similarity retrieval and a hierarchical graph that enables top-down, deliberate traversal over semantic hierarchies. By combining the complementary strength from both retrieval routes, Mnemis retrieves memory items that are both semantically and structurally relevant. Mnemis achieves state-of-the-art performance across all compared methods on long-term memory benchmarks, scoring 93.9 on LoCoMo and 91.6 on LongMemEval-S using GPT-4.1-mini.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.21199v2">A Fano-Style Accuracy Upper Bound for LLM Single-Pass Reasoning in Multi-Hop QA</a></div>
    <div class="paper-meta">
      📅 2026-02-16
      | 💬 22 pages, 6 figures, ICLR 2026. Reported by MIT Technology Review
    </div>
    <details class="paper-abstract">
      Multi-Hop Question Answering (MHQA) requires integrating dispersed, interdependent evidence through sequential reasoning under noise. This task is challenging for LLMs as they have a finite per-pass output capacity, beyond which the integration of task-relevant evidence proves unreliable. Consequently, the single-pass reasoning paradigm is inherently vulnerable to this capacity overflow. To formalize this bottleneck, our analysis establishes a Fano-style accuracy upper bound, defining a theoretical performance ceiling for single-pass LLMs. This bound reveals that accuracy inevitably collapses once task complexity exceeds model capacity, providing general principles for capacity-aware representation and structuring of MHQA in LLMs. Building on these principles, we introduce a proof-of-concept multi-call framework for MHQA, InfoQA. It ensures high per-step accuracy by combining capacity-aware task decomposition with active pruning of prior reasoning traces, keeping the information load within the single-pass limit. It further achieves robustness by a dependency-explicit workflow that enables precise control over the reasoning path. We construct a stringent and noise-rich benchmark to validate our theory and framework. Experimental results show that model behavior aligns with our predicted capacity curves while InfoQA achieves consistent performance improvements. We hope our work inspires more LLM multi-step reasoning methods: \faGithub \href{https://github.com/KaiyangWan/InfoQA}{InfoQA}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.15189v1">ScrapeGraphAI-100k: A Large-Scale Dataset for LLM-Based Web Information Extraction</a></div>
    <div class="paper-meta">
      📅 2026-02-16
    </div>
    <details class="paper-abstract">
      The use of large language models for web information extraction is becoming increasingly fundamental to modern web information retrieval pipelines. However, existing datasets tend to be small, synthetic or text-only, failing to capture the structural context of the web. We introduce ScrapeGraphAI-100k, a large-scale dataset comprising real-world LLM extraction events, collected via opt-in ScrapeGraphAI telemetry during Q2 and Q3 of 2025. Starting from 9M events, we deduplicate and balance by schema to produce 93,695 examples spanning diverse domains and languages. Each instance includes Markdown content, a prompt, a JSON schema, the LLM response, and complexity/validation metadata. We characterize the datasets structural diversity and its failure modes as schema complexity increases. We also provide a fine-tuning experiment showing that a small language model (1.7B) trained on a subset narrows the gap to larger baselines (30B), underscoring the datasets utility for efficient extraction. ScrapeGraphAI-100k enables fine-tuning small models, benchmarking structured extraction, and studying schema induction for web IR indexing, and is publicly available on HuggingFace.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.15173v1">Mind the (DH) Gap! A Contrast in Risky Choices Between Reasoning and Conversational LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-16
    </div>
    <details class="paper-abstract">
      The use of large language models either as decision support systems, or in agentic workflows, is rapidly transforming the digital ecosystem. However, the understanding of LLM decision-making under uncertainty remains limited. We initiate a comparative study of LLM risky choices along two dimensions: (1) prospect representation (explicit vs. experience based) and (2) decision rationale (explanation). Our study, which involves 20 frontier and open LLMs, is complemented by a matched human subjects experiment, which provides one reference point, while an expected payoff maximizing rational agent model provides another. We find that LLMs cluster into two categories: reasoning models (RMs) and conversational models (CMs). RMs tend towards rational behavior, are insensitive to the order of prospects, gain/loss framing, and explanations, and behave similarly whether prospects are explicit or presented via experience history. CMs are significantly less rational, slightly more human-like, sensitive to prospect ordering, framing, and explanation, and exhibit a large description-history gap. Paired comparisons of open LLMs suggest that a key factor differentiating RMs and CMs is training for mathematical reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.25034v3">MARLIN: Multi-Agent Reinforcement Learning with Murmuration Intelligence and LLM Guidance for Reservoir Management</a></div>
    <div class="paper-meta">
      📅 2026-02-16
      | 💬 AAMAS'26
    </div>
    <details class="paper-abstract">
      As climate change intensifies extreme weather events, water disasters pose growing threats to global communities, making adaptive reservoir management critical for protecting vulnerable populations and ensuring water security. Modern water resource management faces unprecedented challenges from cascading uncertainties propagating through interconnected reservoir networks. These uncertainties, rooted in physical water transfer losses and environmental variability, make precise control difficult. For example, sending 10 tons downstream may yield only 8-12 tons due to evaporation and seepage. Traditional centralized optimization approaches suffer from exponential computational complexity and cannot effectively handle such real-world uncertainties, while existing multi-agent reinforcement learning (MARL) methods fail to achieve effective coordination under uncertainty. To address these challenges, we present MARLIN, a decentralized reservoir management framework inspired by starling murmurations intelligence. Integrating bio-inspired alignment, separation, and cohesion rules with MARL, MARLIN enables individual reservoirs to make local decisions while achieving emergent global coordination. In addition, a LLM provides real-time reward shaping signals, guiding agents to adapt to environmental changes and human-defined preferences. Experiments on USGS data show that MARLIN improves uncertainty handling by 23\%, cuts computation by 35\%, and accelerates flood response by 68\%, exhibiting super-linear coordination, with complexity scaling 5.4x from 400 to 10,000 nodes. These results demonstrate MARLIN's potential for disaster prevention and protecting communities through intelligent, scalable water resource management.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.15028v1">Long Context, Less Focus: A Scaling Gap in LLMs Revealed through Privacy and Personalization</a></div>
    <div class="paper-meta">
      📅 2026-02-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed in privacy-critical and personalization-oriented scenarios, yet the role of context length in shaping privacy leakage and personalization effectiveness remains largely unexplored. We introduce a large-scale benchmark, PAPerBench, to systematically study how increasing context length influences both personalization quality and privacy protection in LLMs. The benchmark comprises approximately 29,000 instances with context lengths ranging from 1K to 256K tokens, yielding a total of 377K evaluation questions. It jointly evaluates personalization performance and privacy risks across diverse scenarios, enabling controlled analysis of long-context model behavior. Extensive evaluations across state-of-the-art LLMs reveal consistent performance degradation in both personalization and privacy as context length increases. We further provide a theoretical analysis of attention dilution under context scaling, explaining this behavior as an inherent limitation of soft attention in fixed-capacity Transformers. The empirical and theoretical findings together suggest a general scaling gap in current models -- long context, less focus. We release the benchmark to support reproducible evaluation and future research on scalable privacy and personalization. Code and data are available at https://github.com/SafeRL-Lab/PAPerBench
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.15013v1">Text Style Transfer with Parameter-efficient LLM Finetuning and Round-trip Translation</a></div>
    <div class="paper-meta">
      📅 2026-02-16
      | 💬 9 pages, 5 figures, 4 tables
    </div>
    <details class="paper-abstract">
      This paper proposes a novel method for Text Style Transfer (TST) based on parameter-efficient fine-tuning of Large Language Models (LLMs). Addressing the scarcity of parallel corpora that map between styles, the study employs roundtrip translation to synthesize such parallel datasets from monolingual corpora. This approach creates 'neutralized' text devoid of stylistic attributes, essentially creating a shared input style at training-time and inference-time. Experimental results demonstrate consistent superiority of this method over zero-shot prompting and fewshot ICL techniques measured by BLEU scores and style accuracy scores across four investigated domains. Furthermore, the integration of retrieval-augmented generation (RAG) for terminology and name knowledge enhances robustness and stylistic consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.14970v1">Counterfactual Fairness Evaluation of LLM-Based Contact Center Agent Quality Assurance System</a></div>
    <div class="paper-meta">
      📅 2026-02-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed in contact-center Quality Assurance (QA) to automate agent performance evaluation and coaching feedback. While LLMs offer unprecedented scalability and speed, their reliance on web-scale training data raises concerns regarding demographic and behavioral biases that may distort workforce assessment. We present a counterfactual fairness evaluation of LLM-based QA systems across 13 dimensions spanning three categories: Identity, Context, and Behavioral Style. Fairness is quantified using the Counterfactual Flip Rate (CFR), the frequency of binary judgment reversals, and the Mean Absolute Score Difference (MASD), the average shift in coaching or confidence scores across counterfactual pairs. Evaluating 18 LLMs on 3,000 real-world contact center transcripts, we find systematic disparities, with CFR ranging from 5.4% to 13.0% and consistent MASD shifts across confidence, positive, and improvement scores. Larger, more strongly aligned models show lower unfairness, though fairness does not track accuracy. Contextual priming of historical performance induces the most severe degradations (CFR up to 16.4%), while implicit linguistic identity cues remain a persistent bias source. Finally, we analyze the efficacy of fairness-aware prompting, finding that explicit instructions yield only modest improvements in evaluative consistency. Our findings underscore the need for standardized fairness auditing pipelines prior to deploying LLMs in high-stakes workforce evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.14968v1">PhyScensis: Physics-Augmented LLM Agents for Complex Physical Scene Arrangement</a></div>
    <div class="paper-meta">
      📅 2026-02-16
      | 💬 ICLR 2026
    </div>
    <details class="paper-abstract">
      Automatically generating interactive 3D environments is crucial for scaling up robotic data collection in simulation. While prior work has primarily focused on 3D asset placement, it often overlooks the physical relationships between objects (e.g., contact, support, balance, and containment), which are essential for creating complex and realistic manipulation scenarios such as tabletop arrangements, shelf organization, or box packing. Compared to classical 3D layout generation, producing complex physical scenes introduces additional challenges: (a) higher object density and complexity (e.g., a small shelf may hold dozens of books), (b) richer supporting relationships and compact spatial layouts, and (c) the need to accurately model both spatial placement and physical properties. To address these challenges, we propose PhyScensis, an LLM agent-based framework powered by a physics engine, to produce physically plausible scene configurations with high complexity. Specifically, our framework consists of three main components: an LLM agent iteratively proposes assets with spatial and physical predicates; a solver, equipped with a physics engine, realizes these predicates into a 3D scene; and feedback from the solver informs the agent to refine and enrich the configuration. Moreover, our framework preserves strong controllability over fine-grained textual descriptions and numerical parameters (e.g., relative positions, scene stability), enabled through probabilistic programming for stability and a complementary heuristic that jointly regulates stability and spatial relations. Experimental results show that our method outperforms prior approaches in scene complexity, visual quality, and physical accuracy, offering a unified pipeline for generating complex physical scene layouts for robotic manipulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2402.15751v2">Sparse MeZO: Less Parameters for Better Performance in Zeroth-Order LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2026-02-16
      | 💬 Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      While fine-tuning large language models (LLMs) for specific tasks often yields impressive results, it comes at the cost of memory inefficiency due to back-propagation in gradient-based training. Memory-efficient Zeroth-order (MeZO) optimizers, recently proposed to address this issue, only require forward passes during training, making them more memory-friendly. However, compared with exact gradients, ZO-based gradients usually exhibit an estimation error, which can significantly hurt the optimization process, leading to slower convergence and suboptimal solutions. In addition, we find that the estimation error will hurt more when adding to large weights instead of small weights. Based on this observation, this paper introduces Sparse MeZO, a novel memory-efficient zeroth-order optimization approach that applies ZO only to a carefully chosen subset of parameters. We propose a simple yet effective parameter selection scheme that yields significant performance gains with Sparse-MeZO. Additionally, we develop a memory-optimized implementation for sparse masking, ensuring the algorithm requires only inference-level memory consumption, allowing Sparse-MeZO to fine-tune LLaMA-30b on a single A100 GPU. Experimental results illustrate that Sparse-MeZO consistently improves both performance and convergence speed over MeZO without any overhead. For example, it achieves a 9\% absolute accuracy improvement and 3.5x speedup over MeZO on the RTE task. Code is available at https://github.com/NUS-HPC-AI-Lab/SparseMeZO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.14955v1">Tool-Aware Planning in Contact Center AI: Evaluating LLMs through Lineage-Guided Query Decomposition</a></div>
    <div class="paper-meta">
      📅 2026-02-16
    </div>
    <details class="paper-abstract">
      We present a domain-grounded framework and benchmark for tool-aware plan generation in contact centers, where answering a query for business insights, our target use case, requires decomposing it into executable steps over structured tools (Text2SQL (T2S)/Snowflake) and unstructured tools (RAG/transcripts) with explicit depends_on for parallelism. Our contributions are threefold: (i) a reference-based plan evaluation framework operating in two modes - a metric-wise evaluator spanning seven dimensions (e.g., tool-prompt alignment, query adherence) and a one-shot evaluator; (ii) a data curation methodology that iteratively refines plans via an evaluator->optimizer loop to produce high-quality plan lineages (ordered plan revisions) while reducing manual effort; and (iii) a large-scale study of 14 LLMs across sizes and families for their ability to decompose queries into step-by-step, executable, and tool-assigned plans, evaluated under prompts with and without lineage. Empirically, LLMs struggle on compound queries and on plans exceeding 4 steps (typically 5-15); the best total metric score reaches 84.8% (Claude-3-7-Sonnet), while the strongest one-shot match rate at the "A+" tier (Extremely Good, Very Good) is only 49.75% (o3-mini). Plan lineage yields mixed gains overall but benefits several top models and improves step executability for many. Our results highlight persistent gaps in tool-understanding, especially in tool-prompt alignment and tool-usage completeness, and show that shorter, simpler plans are markedly easier. The framework and findings provide a reproducible path for assessing and improving agentic planning with tools for answering data-analysis queries in contact-center settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02744v3">SYNAPSE: Empowering LLM Agents with Episodic-Semantic Memory via Spreading Activation</a></div>
    <div class="paper-meta">
      📅 2026-02-16
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) excel at generalized reasoning, standard retrieval-augmented approaches fail to address the disconnected nature of long-term agentic memory. To bridge this gap, we introduce Synapse (Synergistic Associative Processing Semantic Encoding), a unified memory architecture that transcends static vector similarity. Drawing from cognitive science, Synapse models memory as a dynamic graph where relevance emerges from spreading activation rather than pre-computed links. By integrating lateral inhibition and temporal decay, the system dynamically highlights relevant sub-graphs while filtering interference. We implement a Triple Hybrid Retrieval strategy that fuses geometric embeddings with activation-based graph traversal. Comprehensive evaluations on the LoCoMo benchmark show that Synapse significantly outperforms state-of-the-art methods in complex temporal and multi-hop reasoning tasks, offering a robust solution to the "Contextual Tunneling" problem. Our code and data will be made publicly available upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.17001v4">PersonalAI: A Systematic Comparison of Knowledge Graph Storage and Retrieval Approaches for Personalized LLM agents</a></div>
    <div class="paper-meta">
      📅 2026-02-16
    </div>
    <details class="paper-abstract">
      Personalizing language models that effectively incorporating user interaction history remains a central challenge in development of adaptive AI systems. While large language models (LLMs), combined with Retrieval-Augmented Generation (RAG), have improved factual accuracy, they often lack structured memory and fail to scale in complex, long-term interactions. To address this, we propose a flexible external memory framework based on knowledge graph, which construct and update memory model automatically by LLM itself. Building upon the AriGraph architecture, we introduce a novel hybrid graph design that supports both standard edges and two types of hyper-edges, enabling rich and dynamic semantic and temporal representations. Our framework also supports diverse retrieval mechanisms, including A*, water-circle traversal, beam search and hybrid methods, making it adaptable to different datasets and LLM capacities. We evaluate our system on three benchmarks: TriviaQA, HotpotQA, DiaASQ and demonstrate that different memory and retrieval configurations yield optimal performance depending on the task. Additionally, we extend the DiaASQ benchmark with temporal annotations and internally contradictory statements, showing that our system remains robust and effective in managing temporal dependencies and context-aware reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.09424v2">The Speech-LLM Takes It All: A Truly Fully End-to-End Spoken Dialogue State Tracking Approach</a></div>
    <div class="paper-meta">
      📅 2026-02-16
      | 💬 Accepted for presentation at LREC 2026
    </div>
    <details class="paper-abstract">
      This paper presents a comparative study of context management strategies for end-to-end Spoken Dialog State Tracking using Speech-LLMs. We systematically evaluate traditional multimodal context (combining text history and spoken current turn), full spoken history, and compressed spoken history approaches. Our experiments on the SpokenWOZ corpus demonstrate that providing the full spoken conversation as input yields the highest performance among models of similar size, significantly surpassing prior methods. Furthermore, we show that attention-pooling-based compression of the spoken history offers a strong trade-off, maintaining competitive accuracy with reduced context size. Detailed analysis confirms that improvements stem from more effective context utilization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.14749v1">Cognitive networks reconstruct mindsets about STEM subjects and educational contexts in almost 1000 high-schoolers, University students and LLM-based digital twins</a></div>
    <div class="paper-meta">
      📅 2026-02-16
    </div>
    <details class="paper-abstract">
      Attitudes toward STEM develop from the interaction of conceptual knowledge, educational experiences, and affect. Here we use cognitive network science to reconstruct group mindsets as behavioural forma mentis networks (BFMNs). In this case, nodes are cue words and free associations, edges are empirical associative links, and each concept is annotated with perceived valence. We analyse BFMNs from N = 994 observations spanning high school students, university students, and early-career STEM experts, alongside LLM (GPT-oss) "digital twins" prompted to emulate comparable profiles. Focusing also on semantic neighbourhoods ("frames") around key target concepts (e.g., STEM subjects or educational actors/places), we quantify frames in terms of valence auras, emotional profiles, network overlap (Jaccard similarity), and concreteness relative to null baselines. Across student groups, science and research are consistently framed positively, while their core quantitative subjects (mathematics and statistics) exhibit more negative and anxiety related auras, amplified in higher math-anxiety subgroups, evidencing a STEM-science cognitive and emotional dissonance. High-anxiety frames are also less concrete than chance, suggesting more abstract and decontextualised representations of threatening quantitative domains. Human networks show greater overlapping between mathematics and anxiety than GPT-oss. The results highlight how BFMNs capture cognitive-affective signatures of mindsets towards the target domains and indicate that LLM-based digital twins approximate cultural attitudes but miss key context-sensitive, experience-based components relevant to replicate human educational anxiety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.05316v4">Improving Data Efficiency for LLM Reinforcement Fine-tuning Through Difficulty-targeted Online Data Selection and Rollout Replay</a></div>
    <div class="paper-meta">
      📅 2026-02-16
      | 💬 Accepted at NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has become an effective approach for fine-tuning large language models (LLMs), particularly to enhance their reasoning capabilities. However, RL fine-tuning remains highly resource-intensive, and existing work has largely overlooked the problem of data efficiency. In this paper, we propose two techniques to improve data efficiency in LLM RL fine-tuning: difficulty-targeted online data selection and rollout replay. We introduce the notion of adaptive difficulty to guide online data selection, prioritizing questions of moderate difficulty that are more likely to yield informative learning signals. To estimate adaptive difficulty efficiently, we develop an attention-based framework that requires rollouts for only a small reference set of questions. The adaptive difficulty of the remaining questions is then estimated based on their similarity to this set. To further reduce rollout cost, we introduce a rollout replay mechanism inspired by experience replay in traditional RL. This technique reuses recent rollouts, lowering per-step computation while maintaining stable updates. Experiments across 6 LLM-dataset combinations show that our method reduces RL fine-tuning time by 23% to 62% while reaching the same level of performance as the original GRPO algorithm. Our code is available at https://github.com/ASTRAL-Group/data-efficient-llm-rl.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.14744v1">Rethinking the Role of LLMs in Time Series Forecasting</a></div>
    <div class="paper-meta">
      📅 2026-02-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been introduced to time series forecasting (TSF) to incorporate contextual knowledge beyond numerical signals. However, existing studies question whether LLMs provide genuine benefits, often reporting comparable performance without LLMs. We show that such conclusions stem from limited evaluation settings and do not hold at scale. We conduct a large-scale study of LLM-based TSF (LLM4TSF) across 8 billion observations, 17 forecasting scenarios, 4 horizons, multiple alignment strategies, and both in-domain and out-of-domain settings. Our results demonstrate that \emph{LLM4TS indeed improves forecasting performance}, with especially large gains in cross-domain generalization. Pre-alignment outperforming post-alignment in over 90\% of tasks. Both pretrained knowledge and model architecture of LLMs contribute and play complementary roles: pretraining is critical under distribution shifts, while architecture excels at modeling complex temporal dynamics. Moreover, under large-scale mixed distributions, a fully intact LLM becomes indispensable, as confirmed by token-level routing analysis and prompt-based improvements. Overall, Our findings overturn prior negative assessments, establish clear conditions under which LLMs are not only useful, and provide practical guidance for effective model design. We release our code at https://github.com/EIT-NLP/LLM4TSF.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.01445v2">A Meta-Knowledge-Augmented LLM Framework for Hyperparameter Optimization in Time-Series Forecasting</a></div>
    <div class="paper-meta">
      📅 2026-02-16
    </div>
    <details class="paper-abstract">
      Hyperparameter optimization (HPO) plays a central role in the performance of deep learning models, yet remains computationally expensive and difficult to interpret, particularly for time-series forecasting. While Bayesian Optimization (BO) is a standard approach, it typically treats tuning tasks independently and provides limited insight into its decisions. Recent advances in large language models (LLMs) offer new opportunities to incorporate structured prior knowledge and reasoning into optimization pipelines. We introduce LLM-AutoOpt, a hybrid HPO framework that combines BO with LLM-based contextual reasoning. The framework encodes dataset meta-features, model descriptions, historical optimization outcomes, and target objectives as structured meta-knowledge within LLM prompts, using BO to initialize the search and mitigate cold-start effects. This design enables context-aware and stable hyperparameter refinement while exposing the reasoning behind optimization decisions. Experiments on a multivariate time series forecasting benchmark demonstrate that LLM-AutoOpt achieves improved predictive performance and more interpretable optimization behavior compared to BO and LLM baselines without meta-knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.14697v1">Evolutionary System Prompt Learning can Facilitate Reinforcement Learning for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-16
    </div>
    <details class="paper-abstract">
      Building agentic systems that can autonomously self-improve from experience is a longstanding goal of AI. Large language models (LLMs) today primarily self-improve via two mechanisms: self-reflection for context updates, and reinforcement learning (RL) for weight updates. In this work, we propose Evolutionary System Prompt Learning (E-SPL), a method for jointly improving model contexts and model weights. In each RL iteration, E-SPL selects multiple system prompts and runs rollouts with each in parallel. It applies RL updates to model weights conditioned on each system prompt, and evolutionary updates to the system prompt population via LLM-driven mutation and crossover. Each system prompt has a TrueSkill rating for evolutionary selection, updated from relative performance within each RL iteration batch. E-SPL encourages a natural division between declarative knowledge encoded in prompts and procedural knowledge encoded in weights, resulting in improved performance across reasoning and agentic tasks. For instance, in an easy-to-hard (AIME $\rightarrow$ BeyondAIME) generalization setting, E-SPL improves RL success rate from 38.8% $\rightarrow$ 45.1% while also outperforming reflective prompt evolution (40.0%). Overall, our results show that coupling reinforcement learning with system prompt evolution yields consistent gains in sample efficiency and generalization. Code: https://github.com/LunjunZhang/E-SPL
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.00628v2">From Associations to Activations: Comparing Behavioral and Hidden-State Semantic Geometry in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-16
      | 💬 25 pages including references, 15 figures, 6 tables
    </div>
    <details class="paper-abstract">
      We investigate the extent to which an LLM's hidden-state geometry can be recovered from its behavior in psycholinguistic experiments. Across eight instruction-tuned transformer models, we run two experimental paradigms -- similarity-based forced choice and free association -- over a shared 5,000-word vocabulary, collecting 17.5M+ trials to build behavior-based similarity matrices. Using representational similarity analysis, we compare behavioral geometries to layerwise hidden-state similarity and benchmark against FastText, BERT, and cross-model consensus. We find that forced-choice behavior aligns substantially more with hidden-state geometry than free association. In a held-out-words regression, behavioral similarity (especially forced choice) predicts unseen hidden-state similarities beyond lexical baselines and cross-model consensus, indicating that behavior-only measurements retain recoverable information about internal semantic geometry. Finally, we discuss implications for the ability of behavioral tasks to uncover hidden cognitive states.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.14675v1">Crowdsourcing Piedmontese to Test LLMs on Non-Standard Orthography</a></div>
    <div class="paper-meta">
      📅 2026-02-16
      | 💬 17 pages, 6 figures, at VarDial20226
    </div>
    <details class="paper-abstract">
      We present a crowdsourced dataset for Piedmontese, an endangered Romance language of northwestern Italy. The dataset comprises 145 Italian-Piedmontese parallel sentences derived from Flores+, with translations produced by speakers writing in their natural orthographic style rather than adhering to standardized conventions, along with manual word alignment. We use this resource to benchmark several large language models on tokenization parity, topic classification, and machine translation. Our analysis reveals that Piedmontese incurs a tokenization penalty relative to higher-resource Romance languages, yet LLMs achieve classification performance approaching that of Italian, French, and English. Machine translation results are asymmetric: models translate adequately from Piedmontese into high-resource languages, but generation into Piedmontese remains challenging. The dataset and code are publicly released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.13593v5">Calibrated Predictive Lower Bounds on Time-to-Unsafe-Sampling in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-16
    </div>
    <details class="paper-abstract">
      We introduce time-to-unsafe-sampling, a novel safety measure for generative models, defined as the number of generations required by a large language model (LLM) to trigger an unsafe (e.g., toxic) response. While providing a new dimension for prompt-adaptive safety evaluation, quantifying time-to-unsafe-sampling is challenging: unsafe outputs are often rare in well-aligned models and thus may not be observed under any feasible sampling budget. To address this challenge, we frame this estimation problem as one of survival analysis. We build on recent developments in conformal prediction and propose a novel calibration technique to construct a lower predictive bound (LPB) on the time-to-unsafe-sampling of a given prompt with rigorous coverage guarantees. Our key technical innovation is an optimized sampling-budget allocation scheme that improves sample efficiency while maintaining distribution-free guarantees. Experiments on both synthetic and real data support our theoretical results and demonstrate the practical utility of our method for safety risk assessment in generative AI models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.09027v2">Measure Twice, Cut Once: A Semantic-Oriented Approach to Video Temporal Localization with Video LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-16
      | 💬 ICLR2026
    </div>
    <details class="paper-abstract">
      Temporally localizing user-queried events through natural language is a crucial capability for video models. Recent methods predominantly adapt video LLMs to generate event boundary timestamps for temporal localization tasks, which struggle to leverage LLMs' pre-trained semantic understanding capabilities due to the uninformative nature of timestamp outputs. In this work, we explore a timestamp-free, semantic-oriented framework that fine-tunes video LLMs using two generative learning tasks and one discriminative learning task. We first introduce a structural token generation task that enables the video LLM to recognize the temporal structure of input videos based on the input query. Through this task, the video LLM generates a sequence of special tokens, called structural tokens, which partition the video into consecutive segments and categorize them as either target events or background transitions. To enhance precise recognition of event segments, we further propose a query-focused captioning task that enables the video LLM to extract fine-grained event semantics that can be effectively utilized by the structural tokens. Finally, we introduce a structural token grounding module driven by contrastive learning to associate each structural token with its corresponding video segment, achieving holistic temporal segmentation of the input video and readily yielding the target event segments for localization. Extensive experiments across diverse temporal localization tasks demonstrate that our proposed framework, MeCo, consistently outperforms methods relying on boundary timestamp generation, highlighting the potential of a semantic-driven approach for temporal localization with video LLMs \footnote{Code available at https://github.com/pangzss/MeCo.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.13103v2">R-Diverse: Mitigating Diversity Illusion in Self-Play LLM Training</a></div>
    <div class="paper-meta">
      📅 2026-02-16
    </div>
    <details class="paper-abstract">
      Self-play bootstraps LLM reasoning through an iterative Challenger-Solver loop: the Challenger is trained to generate questions that target the Solver's capabilities, and the Solver is optimized on the generated data to expand its reasoning skills. However, existing frameworks like R-Zero often exhibit non-sustained improvement, where early gains degrade as self-play continues. We identify a key failure mode, Diversity Illusion, where the Solver's training signals appear diverse yet collapse into recurring underlying patterns. It manifests as (1) Local Diversity Illusion, where diversity is enforced only within-batch, inducing cross-iteration mode cycling; and (2) Surface Diversity Illusion, where questions vary superficially but require near-identical reasoning skills. To mitigate them, we propose R-Diverse with two aligned innovations: Memory-Augmented Penalty (MAP), which uses a persistent memory bank to discourage recycling across iterations, and Skill-Aware Measurement (SAM), which evaluates diversity by the reasoning skills exercised rather than surface variation of questions. Across 10 math and general reasoning benchmarks, R-Diverse sustains gains over more iterations and consistently outperforms prior self-play methods. Code is available at https://github.com/Gengsheng-Li/R-Diverse.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.14564v1">Assessing Large Language Models for Medical QA: Zero-Shot and LLM-as-a-Judge Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-02-16
      | 💬 Accepted in 28th ICCIT, 2025
    </div>
    <details class="paper-abstract">
      Recently, Large Language Models (LLMs) have gained significant traction in medical domain, especially in developing a QA systems to Medical QA systems for enhancing access to healthcare in low-resourced settings. This paper compares five LLMs deployed between April 2024 and August 2025 for medical QA, using the iCliniq dataset, containing 38,000 medical questions and answers of diverse specialties. Our models include Llama-3-8B-Instruct, Llama 3.2 3B, Llama 3.3 70B Instruct, Llama-4-Maverick-17B-128E-Instruct, and GPT-5-mini. We are using a zero-shot evaluation methodology and using BLEU and ROUGE metrics to evaluate performance without specialized fine-tuning. Our results show that larger models like Llama 3.3 70B Instruct outperform smaller models, consistent with observed scaling benefits in clinical tasks. It is notable that, Llama-4-Maverick-17B exhibited more competitive results, thus highlighting evasion efficiency trade-offs relevant for practical deployment. These findings align with advancements in LLM capabilities toward professional-level medical reasoning and reflect the increasing feasibility of LLM-supported QA systems in the real clinical environments. This benchmark aims to serve as a standardized setting for future study to minimize model size, computational resources and to maximize clinical utility in medical NLP applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.14536v1">Explainable Token-level Noise Filtering for LLM Fine-tuning Datasets</a></div>
    <div class="paper-meta">
      📅 2026-02-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have seen remarkable advancements, achieving state-of-the-art results in diverse applications. Fine-tuning, an important step for adapting LLMs to specific downstream tasks, typically involves further training on corresponding datasets. However, a fundamental discrepancy exists between current fine-tuning datasets and the token-level optimization mechanism of LLMs: most datasets are designed at the sentence-level, which introduces token-level noise, causing negative influence to final performance. In this paper, we propose XTF, an explainable token-level noise filtering framework. XTF decomposes the complex and subtle contributions of token-level data to the fine-tuning process into three distinct and explicit attributes (reasoning importance, knowledge novelty, and task relevance), which can be assessed using scoring methods, and then masks the gradients of selected noisy tokens accordingly to optimize the performance of fine-tuned LLMs. We conduct extensive experiments on three representative downstream tasks (math, code and medicine) across 7 mainstream LLMs. The results demonstrate that XTF can significantly improve downstream performance by up to 13.7% compared to regular fine-tuning. Our work highlights the importance of token-level dataset optimization, and demonstrates the potential of strategies based on attribute decomposition for explaining complex training mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.14529v1">Disentangling Deception and Hallucination Failures in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-16
    </div>
    <details class="paper-abstract">
      Failures in large language models (LLMs) are often analyzed from a behavioral perspective, where incorrect outputs in factual question answering are commonly associated with missing knowledge. In this work, focusing on entity-based factual queries, we suggest that such a view may conflate different failure mechanisms, and propose an internal, mechanism-oriented perspective that separates Knowledge Existence from Behavior Expression. Under this formulation, hallucination and deception correspond to two qualitatively different failure modes that may appear similar at the output level but differ in their underlying mechanisms. To study this distinction, we construct a controlled environment for entity-centric factual questions in which knowledge is preserved while behavioral expression is selectively altered, enabling systematic analysis of four behavioral cases. We analyze these failure modes through representation separability, sparse interpretability, and inference-time activation steering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.14517v1">Beyond Translation: Evaluating Mathematical Reasoning Capabilities of LLMs in Sinhala and Tamil</a></div>
    <div class="paper-meta">
      📅 2026-02-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate strong mathematical reasoning in English, but whether these capabilities reflect genuine multilingual reasoning or reliance on translation-based processing in low-resource languages like Sinhala and Tamil remains unclear. We examine this fundamental question by evaluating whether LLMs genuinely reason mathematically in these languages or depend on implicit translation to English-like representations. Using a taxonomy of six math problem types, from basic arithmetic to complex unit conflict and optimization problems, we evaluate four prominent large language models. To avoid translation artifacts that confound language ability with translation quality, we construct a parallel dataset where each problem is natively authored by fluent speakers with mathematical training in all three languages. Our analysis demonstrates that while basic arithmetic reasoning transfers robustly across languages, complex reasoning tasks show significant degradation in Tamil and Sinhala. The pattern of failures varies by model and problem type, suggesting that apparent multilingual competence may not reflect uniform reasoning capabilities across languages. These findings challenge the common assumption that models exhibiting strong multilingual performance can reason equally effectively across languages, and highlight the need for fine-grained, type-aware evaluation in multilingual settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.14516v1">Efficient Multi-round LLM Inference over Disaggregated Serving</a></div>
    <div class="paper-meta">
      📅 2026-02-16
    </div>
    <details class="paper-abstract">
      With the rapid evolution of Large Language Models (LLMs), multi-round workflows, such as autonomous agents and iterative retrieval, have become increasingly prevalent. However, this raises hurdles for serving LLMs under prefill-decode (PD) disaggregation, a widely adopted paradigm that separates the compute-bound prefill phase and memory-bound decode phase onto individual resources. Specifically, existing systems overlook the interleaved prefill-decode workload pattern in multi-round inference, leading to sub-optimal handling of the incremental prefill workloads and model deployment for the two phases. In this work, we present AMPD, a brand new disaggregated serving framework for multi-round LLM inference. The core of AMPD is to coordinate the prefill workloads based on real-time workloads by adaptively determining where to carry out these workloads and how they are scheduled, in order to maximize service level objective (SLO) attainment. In addition, we tailor a planning algorithm for our scenario, facilitating the deduction of optimal resource allocation and parallel strategies for the two phases. Empirical results demonstrate that AMPD substantially improves SLO attainment compared to state-of-the-art baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.14490v1">Parameter-Efficient Fine-Tuning of LLMs with Mixture of Space Experts</a></div>
    <div class="paper-meta">
      📅 2026-02-16
      | 💬 15 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable progress, with Parameter-Efficient Fine-Tuning (PEFT) emerging as a key technique for downstream task adaptation. However, existing PEFT methods mainly operate in Euclidean space, fundamentally limiting their capacity to capture complex geometric structures inherent in language data. While alternative geometric spaces, like hyperbolic geometries for hierarchical data and spherical manifolds for circular patterns, offer theoretical advantages, forcing representations into a single manifold type ultimately limits expressiveness, even when curvature parameters are learnable. To address this, we propose Mixture of Space (MoS), a unified framework that leverages multiple geometric spaces simultaneously to learn richer, curvature-aware representations. Building on this scheme, we develop MoSLoRA, which extends Low-Rank Adaptation (LoRA) with heterogeneous geometric experts, enabling models to dynamically select or combine appropriate geometric spaces based on input context. Furthermore, to address the computational overhead of frequent manifold switching, we develop a lightweight routing mechanism. Moreover, we provide empirical insights into how curvature optimization impacts training stability and model performance. Our experiments across diverse benchmarks demonstrate that MoSLoRA consistently outperforms strong baselines, achieving up to 5.6% improvement on MATH500 and 15.9% on MAWPS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2404.08634v4">When Attention Collapses: How Degenerate Layers in LLMs Enable Smaller, Stronger Models</a></div>
    <div class="paper-meta">
      📅 2026-02-16
      | 💬 Published in Transactions on Machine Learning Research (TMLR)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are known for their performance, but we uncover a significant structural inefficiency: a phenomenon we term attention collapse. In many pre-trained decoder-style LLMs, the attention matrices in deeper layers degenerate, collapsing to near rank-one structures. These underutilized layers, which we call lazy layers, are redundant and impair model efficiency. To address this, we introduce Inheritune, a simple yet powerful training recipe designed to build smaller, stronger language models. Inheritune initializes a compact model by inheriting the potent early layers from a larger pre-trained model and then progressively trains and expands it. Our experiments on various models, including the GPT-2 family, demonstrate that models trained with Inheritune can match or even surpass the performance of their larger counterparts, despite having significantly fewer layers. This work presents a novel path toward model compression by design, enabling the creation of compact, yet highly performant language models. Code is available at https://github.com/sanyalsunny111/LLM-Inheritune.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.03346v2">Making Slow Thinking Faster: Compressing LLM Chain-of-Thought via Step Entropy</a></div>
    <div class="paper-meta">
      📅 2026-02-16
      | 💬 Accepted by ICLR2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) using Chain-of-Thought (CoT) prompting excel at complex reasoning but generate verbose thought processes with considerable redundancy, leading to increased inference costs and reduced efficiency. We introduce a novel CoT compression framework based on step entropy, a metric that quantifies \emph{the informational contribution of individual reasoning steps} to identify redundancy. Through theoretical analysis and extensive empirical validation on mathematical reasoning benchmarks, we demonstrate that steps with low entropy are indeed highly redundant. Our experiments reveal that an astonishing 80\% of low-entropy intermediate steps can be pruned with minor degradation in the final answer accuracy across DeepSeek-R1-7B, 14B and Qwen3-8B. This finding sharply contrasts with random or high-entropy pruning, which severely impairs reasoning performance. Building on this, we propose a novel two-stage training strategy combining Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO) reinforcement learning. This approach enables LLMs to autonomously learn to generate compressed COTs during inference by strategically incorporating [SKIP] tokens. Our method significantly improves LLM inference efficiency while preserving accuracy, paving the way for more scalable LLM deployments and a better understanding of their internal reasoning. The code and data are released in https://github.com/staymylove/COT_Compresstion_via_Step_entropy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.14471v1">Socially-Weighted Alignment: A Game-Theoretic Framework for Multi-Agent LLM Systems</a></div>
    <div class="paper-meta">
      📅 2026-02-16
    </div>
    <details class="paper-abstract">
      Deploying large language model (LLM) agents in shared environments introduces a fundamental tension between individual alignment and collective stability: locally rational decisions can impose negative externalities that degrade system-level performance. We propose Socially-Weighted Alignment (SWA), a game-theoretic framework that modifies inference-time decision making by interpolating between an agent's private objective and an estimate of group welfare via a social weight $λ\in[0,1]$. In a shared-resource congestion game with $n$ agents and congestion severity $β$, we show that SWA induces a critical threshold $λ^*=(n-β)/(n-1)$ above which agents no longer have marginal incentive to increase demand under overload, yielding a phase transition from persistent congestion to stable operation near capacity. We further provide an inference-time algorithmic instantiation of SWA that does not require parameter updates or multi-agent reinforcement learning, and use a multi-agent simulation to empirically validate the predicted threshold behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.14468v1">LACONIC: Length-Aware Constrained Reinforcement Learning for LLM</a></div>
    <div class="paper-meta">
      📅 2026-02-16
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has enhanced the capabilities of large language models (LLMs) through reward-driven training. Nevertheless, this process can introduce excessively long responses, inflating inference latency and computational overhead. Prior length-control approaches typically rely on fixed heuristic reward shaping, which can misalign with the task objective and require brittle tuning. In this work, we propose LACONIC, a reinforcement learning method that enforces a target token budget during training. Specifically, we update policy models using an augmented objective that combines the task reward with a length-based cost. To balance brevity and task performance, the cost scale is adaptively adjusted throughout training. This yields robust length control while preserving task reward. We provide a theoretical guarantee that support the method. Across mathematical reasoning models and datasets, LACONIC preserves or improves pass@1 while reducing output length by over 50%. It maintains out-of-domain performance on general knowledge and multilingual benchmarks with 44% fewer tokens. Moreover, LACONIC integrates into standard RL-tuning with no inference changes and minimal deployment overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.14452v1">WiSparse: Boosting LLM Inference Efficiency with Weight-Aware Mixed Activation Sparsity</a></div>
    <div class="paper-meta">
      📅 2026-02-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) offer strong capabilities but incur high inference costs due to dense computation and memory access. Training-free activation sparsity is a promising approach for efficient LLM inference, yet existing methods often rely solely on activation information and uniform sparsity ratios. This overlooks the critical interplay with weights and inter-block sensitivity variation, leading to suboptimal performance. We identify two key phenomena in modern LLMs: 1) less significant activations may align with highly important weights, and 2) sparsity sensitivity varies non-monotonically across model blocks. We propose Weight-aware Mixed-Granularity Training-free Activation Sparsity (WiSparse), which leverages both activation and weight information for adaptive sparsity allocation. Specifically, we introduce a weight-aware mechanism integrating activation magnitudes with precomputed weight norms to accurately identify salient channels. This is combined with a mixed-granularity allocation scheme: a global budget is distributed across blocks via evolutionary search to protect sensitive regions, then refined within blocks to minimize reconstruction error. We improve sparse kernels and demonstrate effectiveness on three representative models. Notably, at 50% sparsity, WiSparse preserves 97% of Llama3.1's dense performance, surpassing the strongest baseline by 2.23 percentage points while achieving a 21.4% acceleration in end-to-end inference speed. Our research advances the limits of training-free approaches for efficient LLM inference, pushing the boundaries of achievable speedup without training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.11839v2">On the Eligibility of LLMs for Counterfactual Reasoning: A Decompositional Study</a></div>
    <div class="paper-meta">
      📅 2026-02-16
      | 💬 ICLR 2026
    </div>
    <details class="paper-abstract">
      Counterfactual reasoning has emerged as a crucial technique for generalizing the reasoning capabilities of large language models (LLMs). By generating and analyzing counterfactual scenarios, researchers can assess the adaptability and reliability of model decision-making. Although prior work has shown that LLMs often struggle with counterfactual reasoning, it remains unclear which factors most significantly impede their performance across different tasks and modalities. In this paper, we propose a decompositional strategy that breaks down the counterfactual generation from causality construction to the reasoning over counterfactual interventions. To support decompositional analysis, we investigate \ntask datasets spanning diverse tasks, including natural language understanding, mathematics, programming, and vision-language tasks. Through extensive evaluations, we characterize LLM behavior across each decompositional stage and identify how modality type and intermediate reasoning influence performance. By establishing a structured framework for analyzing counterfactual reasoning, this work contributes to the development of more reliable LLM-based reasoning systems and informs future elicitation strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.14433v1">Synthetic Reader Panels: Tournament-Based Ideation with LLM Personas for Autonomous Publishing</a></div>
    <div class="paper-meta">
      📅 2026-02-16
      | 💬 5 tables, 1 figure
    </div>
    <details class="paper-abstract">
      We present a system for autonomous book ideation that replaces human focus groups with synthetic reader panels -- diverse collections of LLM-instantiated reader personas that evaluate book concepts through structured tournament competitions. Each persona is defined by demographic attributes (age group, gender, income, education, reading level), behavioral patterns (books per year, genre preferences, discovery methods, price sensitivity), and consistency parameters. Panels are composed per imprint to reflect target demographics, with diversity constraints ensuring representation across age, reading level, and genre affinity. Book concepts compete in single-elimination, double-elimination, round-robin, or Swiss-system tournaments, judged against weighted criteria including market appeal, originality, and execution potential. To reject low-quality LLM evaluations, we implement five automated anti-slop checks (repetitive phrasing, generic framing, circular reasoning, score clustering, audience mismatch). We report results from deployment within a multi-imprint publishing operation managing 6 active imprints and 609 titles in distribution. Three case studies -- a 270-evaluator panel for a children's literacy novel, and two 5-person expert panels for a military memoir and a naval strategy monograph -- demonstrate that synthetic panels produce actionable demographic segmentation, identify structural content issues invisible to homogeneous reviewers, and enable tournament filtering that eliminates low-quality concepts while enriching high-quality survivors from 15% to 62% of the evaluated pool.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.14428v1">LLM-Guided Knowledge Distillation for Temporal Knowledge Graph Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-02-16
    </div>
    <details class="paper-abstract">
      Temporal knowledge graphs (TKGs) support reasoning over time-evolving facts, yet state-of-the-art models are often computationally heavy and costly to deploy. Existing compression and distillation techniques are largely designed for static graphs; directly applying them to temporal settings may overlook time-dependent interactions and lead to performance degradation. We propose an LLM-assisted distillation framework specifically designed for temporal knowledge graph reasoning. Beyond a conventional high-capacity temporal teacher, we incorporate a large language model as an auxiliary instructor to provide enriched supervision. The LLM supplies broad background knowledge and temporally informed signals, enabling a lightweight student to better model event dynamics without increasing inference-time complexity. Training is conducted by jointly optimizing supervised and distillation objectives, using a staged alignment strategy to progressively integrate guidance from both teachers. Extensive experiments on multiple public TKG benchmarks with diverse backbone architectures demonstrate that the proposed approach consistently improves link prediction performance over strong distillation baselines, while maintaining a compact and efficient student model. The results highlight the potential of large language models as effective teachers for transferring temporal reasoning capability to resource-efficient TKG systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.07861v3">Scalable LLM Reasoning Acceleration with Low-rank Distillation</a></div>
    <div class="paper-meta">
      📅 2026-02-16
    </div>
    <details class="paper-abstract">
      Due to long generations, large language model (LLM) math reasoning demands significant computational resources and time. While many existing efficient inference methods have been developed with excellent performance preservation on language tasks, they often severely degrade math performance. In this paper, we propose Caprese, a resource-efficient distillation method to recover lost capabilities from deploying efficient inference methods, focused primarily in feedforward blocks. With original weights unperturbed, roughly 1% of additional parameters, and only 20K synthetic training samples, we are able to recover much if not all of the reasoning capabilities lost from efficient inference for thinking LLMs and without harm to language tasks for instruct LLMs. Moreover, Caprese slashes the number of active parameters (~2B cut for Gemma 2 9B and Llama 3.1 8B) and integrates cleanly into existing model layers to reduce latency (>16% time-to-next-token reduction) while encouraging response brevity (up to 8.5% fewer tokens).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.09980v4">V2V-LLM: Vehicle-to-Vehicle Cooperative Autonomous Driving with Multimodal Large Language Models</a></div>
    <div class="paper-meta">
      📅 2026-02-16
      | 💬 Accepted by ICRA 2026 (IEEE International Conference on Robotics and Automation). Project: https://eddyhkchiu.github.io/v2vllm.github.io/ Code: https://github.com/eddyhkchiu/V2V-LLM Dataset: https://huggingface.co/datasets/eddyhkchiu/V2V-GoT-QA
    </div>
    <details class="paper-abstract">
      Current autonomous driving vehicles rely mainly on their individual sensors to understand surrounding scenes and plan for future trajectories, which can be unreliable when the sensors are malfunctioning or occluded. To address this problem, cooperative perception methods via vehicle-to-vehicle (V2V) communication have been proposed, but they have tended to focus on perception tasks like detection or tracking. How those approaches contribute to overall cooperative planning performance is still under-explored. Inspired by recent progress using Large Language Models (LLMs) to build autonomous driving systems, we propose a novel problem setting that integrates a Multimodal LLM into cooperative autonomous driving, with the proposed Vehicle-to-Vehicle Question-Answering (V2V-QA) dataset and benchmark. We also propose our baseline method Vehicle-to-Vehicle Multimodal Large Language Model (V2V-LLM), which uses an LLM to fuse perception information from multiple connected autonomous vehicles (CAVs) and answer various types of driving-related questions: grounding, notable object identification, and planning. Experimental results show that our proposed V2V-LLM can be a promising unified model architecture for performing various tasks in cooperative autonomous driving, and outperforms other baseline methods that use different fusion approaches. Our work also creates a new research direction that can improve the safety of future autonomous driving systems. The code and data will be released to the public to facilitate open-source research in this field. Our project website: https://eddyhkchiu.github.io/v2vllm.github.io/ .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.20102v2">Human-Centered LLM-Agent System for Detecting Anomalous Digital Asset Transactions</a></div>
    <div class="paper-meta">
      📅 2026-02-16
    </div>
    <details class="paper-abstract">
      We present HCLA, a human-centered multi-agent system for anomaly detection in digital-asset transactions. The system integrates three cognitively aligned roles: Rule Abstraction, Evidence Scoring, and Expert-Style Justification. These roles operate in a conversational workflow that enables non-experts to express analytical intent in natural language, inspect structured risk evidence, and obtain traceable, context-aware reasoning. Implemented with an open-source, web-based interface, HCLA translates user intent into explicit analytical rules, applies classical anomaly detectors to quantify evidential risk, and reconstructs expert-style justifications grounded in observable transactional signals. Experiments on a cryptocurrency anomaly dataset show that, while the underlying detector achieves strong predictive accuracy, HCLA substantially improves interpretability, interaction, and decision transparency. Importantly, HCLA is not designed to explain a black-box model in the conventional XAI sense. Instead, we reconstruct a traceable expert reasoning process that aligns algorithmic evidence with regulatory and investigative judgment. By explicitly separating evidence scoring from expert-style justification, the framework emphasizes accountability beyond explainability and addresses practical requirements for regulatory, audit, and compliance-driven financial forensics. We describe the system architecture, closed-loop interaction design, datasets, evaluation protocol, and limitations. We argue that a human-in-the-loop reasoning reconstruction paradigm is essential for achieving transparency, accountability, and trust in high-stakes financial environments. Keywords: Human-Centered AI; LLM-Agent System; Multi-Agent Architecture; Anomaly Detection; Digital Asset Transactions; Cryptocurrency Forensics; Blockchain Analytics; Human-in-the-Loop; Explainable AI (XAI); Interpretability
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.14357v1">Key Considerations for Domain Expert Involvement in LLM Design and Evaluation: An Ethnographic Study</a></div>
    <div class="paper-meta">
      📅 2026-02-16
      | 💬 14 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly developed for use in complex professional domains, yet little is known about how teams design and evaluate these systems in practice. This paper examines the challenges and trade-offs in LLM development through a 12-week ethnographic study of a team building a pedagogical chatbot. The researcher observed design and evaluation activities and conducted interviews with both developers and domain experts. Analysis revealed four key practices: creating workarounds for data collection, turning to augmentation when expert input was limited, co-developing evaluation criteria with experts, and adopting hybrid expert-developer-LLM evaluation strategies. These practices show how teams made strategic decisions under constraints and demonstrate the central role of domain expertise in shaping the system. Challenges included expert motivation and trust, difficulties structuring participatory design, and questions around ownership and integration of expert knowledge. We propose design opportunities for future LLM development workflows that emphasize AI literacy, transparent consent, and frameworks recognizing evolving expert roles.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.02054v2">Designing Staged Evaluation Workflows for LLMs: Integrating Domain Experts, Lay Users, and Model-Generated Evaluation Criteria</a></div>
    <div class="paper-meta">
      📅 2026-02-16
      | 💬 To be published in CHI2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly utilized for domain-specific tasks, yet evaluating their outputs remains challenging. A common strategy is to apply evaluation criteria to assess alignment with domain-specific standards, yet little is understood about how criteria differ across sources or where each type is most useful in the evaluation process. This study investigates criteria developed by domain experts, lay users, and LLMs to identify their complementary roles within an evaluation workflow. Results show that experts produce fact-based criteria with long-term value, lay users emphasize usability with a shorter-term focus, and LLMs target procedural checks for immediate task requirements. We also examine how criteria evolve between a priori and a posteriori phases, noting drift across stages as well as convergence in the a posteriori phase. Based on our observations, we propose design guidelines for a staged evaluation workflow combining the complementary strengths of these sources to balance quality, cost, and scalability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.14349v1">Same Prompt, Different Outcomes: Evaluating the Reproducibility of Data Analysis by LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-15
    </div>
    <details class="paper-abstract">
      We systematically evaluate the reproducibility of data analysis conducted by Large Language Models (LLMs). We evaluate two prompting strategies, six models, and four temperature settings, with ten independent executions per configuration, yielding 480 total attempts. We assess the completion, concordance, validity, and consistency of each attempt and find considerable variation in the analytical results even for consistent configurations. This suggests, as with human data analysis, the data analysis conducted by LLMs can vary, even given the same task, data, and settings. Our results mean that if an LLM is being used to conduct data analysis, then it should be run multiple times independently and the distribution of results considered.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.14302v1">Floe: Federated Specialization for Real-Time LLM-SLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-02-15
      | 💬 Accepted by IEEE Transactions on Parallel and Distributed Systems
    </div>
    <details class="paper-abstract">
      Deploying large language models (LLMs) in real-time systems remains challenging due to their substantial computational demands and privacy concerns. We propose Floe, a hybrid federated learning framework designed for latency-sensitive, resource-constrained environments. Floe combines a cloud-based black-box LLM with lightweight small language models (SLMs) on edge devices to enable low-latency, privacy-preserving inference. Personal data and fine-tuning remain on-device, while the cloud LLM contributes general knowledge without exposing proprietary weights. A heterogeneity-aware LoRA adaptation strategy enables efficient edge deployment across diverse hardware, and a logit-level fusion mechanism enables real-time coordination between edge and cloud models. Extensive experiments demonstrate that Floe enhances user privacy and personalization. Moreover, it significantly improves model performance and reduces inference latency on edge devices under real-time constraints compared with baseline approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.04398v3">SECA: Semantically Equivalent and Coherent Attacks for Eliciting LLM Hallucinations</a></div>
    <div class="paper-meta">
      📅 2026-02-15
      | 💬 Accepted at NeurIPS 2025. Code is available at https://github.com/Buyun-Liang/SECA
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed in high-risk domains. However, state-of-the-art LLMs often exhibit hallucinations, raising serious concerns about their reliability. Prior work has explored adversarial attacks to elicit hallucinations in LLMs, but these methods often rely on unrealistic prompts, either by inserting nonsensical tokens or by altering the original semantic intent. Consequently, such approaches provide limited insight into how hallucinations arise in real-world settings. In contrast, adversarial attacks in computer vision typically involve realistic modifications to input images. However, the problem of identifying realistic adversarial prompts for eliciting LLM hallucinations remains largely underexplored. To address this gap, we propose Semantically Equivalent and Coherent Attacks (SECA), which elicit hallucinations via realistic modifications to the prompt that preserve its meaning while maintaining semantic coherence. Our contributions are threefold: (i) we formulate finding realistic attacks for hallucination elicitation as a constrained optimization problem over the input prompt space under semantic equivalence and coherence constraints; (ii) we introduce a constraint-preserving zeroth-order method to effectively search for adversarial yet feasible prompts; and (iii) we demonstrate through experiments on open-ended multiple-choice question answering tasks that SECA achieves higher attack success rates while incurring almost no semantic equivalence or semantic coherence errors compared to existing methods. SECA highlights the sensitivity of both open-source and commercial gradient-inaccessible LLMs to realistic and plausible prompt variations. Code is available at https://github.com/Buyun-Liang/SECA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.14295v1">Machine Learning as a Tool (MLAT): A Framework for Integrating Statistical ML Models as Callable Tools within LLM Agent Workflows</a></div>
    <div class="paper-meta">
      📅 2026-02-15
      | 💬 Submitted to the Google Gemini 3 Hackathon
    </div>
    <details class="paper-abstract">
      We introduce Machine Learning as a Tool (MLAT), a design pattern in which pre-trained statistical machine learning models are exposed as callable tools within large language model (LLM) agent workflows. This allows an orchestrating agent to invoke quantitative predictions when needed and reason about their outputs in context. Unlike conventional pipelines that treat ML inference as a static preprocessing step, MLAT positions the model as a first-class tool alongside web search, database queries, and APIs, enabling the LLM to decide when and how to use it based on conversational context. To validate MLAT, we present PitchCraft, a pilot production system that converts discovery call recordings into professional proposals with ML-predicted pricing. The system uses two agents: a Research Agent that gathers prospect intelligence via parallel tool calls, and a Draft Agent that invokes an XGBoost pricing model as a tool call and generates a complete proposal through structured outputs. The pricing model, trained on 70 examples combining real and human-verified synthetic data, achieves R^2 = 0.807 on held-out data with a mean absolute error of 3688 USD. The system reduces proposal generation time from multiple hours to under 10 minutes. We describe the MLAT framework, structured output architecture, training methodology under extreme data scarcity, and sensitivity analysis demonstrating meaningful learned relationships. MLAT generalizes to domains requiring quantitative estimation combined with contextual reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.18428v3">AlphaOPT: Formulating Optimization Programs with Self-Improving LLM Experience Library</a></div>
    <div class="paper-meta">
      📅 2026-02-15
    </div>
    <details class="paper-abstract">
      Optimization modeling underlies critical decision-making across industries, yet remains difficult to automate: natural-language problem descriptions must be translated into precise mathematical formulations and executable solver code. Existing LLM-based approaches typically rely on brittle prompting or costly retraining, both of which offer limited generalization. Recent work suggests that large models can improve via experience reuse, but how to systematically acquire, refine, and reuse such experience in structurally constrained settings remains unclear. We present \textbf{AlphaOPT}, a self-improving experience library that enables LLMs to learn optimization modeling knowledge from limited supervision, including answer-only feedback without gold-standard programs, annotated reasoning traces, or parameter updates. AlphaOPT operates in a continual two-phase cycle: a \emph{Library Learning} phase that extracts solver-verified, structured insights from failed attempts, and a \emph{Library Evolution} phase that refines the applicability of stored insights based on aggregate evidence across tasks. This design allows the model to accumulate reusable modeling principles, improve transfer across problem instances, and maintain bounded library growth over time. Evaluated on multiple optimization benchmarks, AlphaOPT steadily improves as more training data become available (65\% $\rightarrow$ 72\% from 100 to 300 training items) and outperforms the strongest baseline by 9.1\% and 8.2\% on two out-of-distribution datasets. These results demonstrate that structured experience learning, grounded in solver feedback, provides a practical alternative to retraining for complex reasoning tasks requiring precise formulation and execution. All code and data are available at: https://github.com/Minw913/AlphaOPT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.04051v2">High Accuracy, Less Talk (HALT): Reliable LLMs through Capability-Aligned Finetuning</a></div>
    <div class="paper-meta">
      📅 2026-02-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) currently respond to every prompt. However, they can produce incorrect answers when they lack knowledge or capability -- a problem known as hallucination. We instead propose post-training an LLM to generate content only when confident in its correctness and to otherwise (partially) abstain. Specifically, our method, HALT, produces capability-aligned post-training data that encodes what the model can and cannot reliably generate. We generate this data by splitting responses of the pretrained LLM into factual fragments (atomic statements or reasoning steps), and use ground truth information to identify incorrect fragments. We achieve capability-aligned finetuning responses by either removing incorrect fragments or replacing them with "Unsure from Here" -- according to a tunable threshold that allows practitioners to trade off response completeness and mean correctness of the response's fragments. We finetune four open-source models for biography writing, mathematics, coding, and medicine with HALT for three different trade-off thresholds. HALT effectively trades off response completeness for correctness, increasing the mean correctness of response fragments by 15% on average, while resulting in a 4% improvement in the F1 score (mean of completeness and correctness of the response) compared to the relevant baselines. By tuning HALT for highest correctness, we train a single reliable Llama3-70B model with correctness increased from 51% to 87% across all four domains while maintaining 53% of the response completeness achieved with standard finetuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.14286v1">Online LLM watermark detection via e-processes</a></div>
    <div class="paper-meta">
      📅 2026-02-15
    </div>
    <details class="paper-abstract">
      Watermarking for large language models (LLMs) has emerged as an effective tool for distinguishing AI-generated text from human-written content. Statistically, watermark schemes induce dependence between generated tokens and a pseudo-random sequence, reducing watermark detection to a hypothesis testing problem on independence. We develop a unified framework for LLM watermark detection based on e-processes, providing anytime-valid guarantees for online testing. We propose various methods to construct empirically adaptive e-processes that can enhance the detection power. In addition, theoretical results are established to characterize the power properties of the proposed procedures. Some experiments demonstrate that the proposed framework achieves competitive performance compared to existing watermark detection methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.14279v1">Whom to Query for What: Adaptive Group Elicitation via Multi-Turn LLM Interactions</a></div>
    <div class="paper-meta">
      📅 2026-02-15
    </div>
    <details class="paper-abstract">
      Eliciting information to reduce uncertainty about latent group-level properties from surveys and other collective assessments requires allocating limited questioning effort under real costs and missing data. Although large language models enable adaptive, multi-turn interactions in natural language, most existing elicitation methods optimize what to ask with a fixed respondent pool, and do not adapt respondent selection or leverage population structure when responses are partial or incomplete. To address this gap, we study adaptive group elicitation, a multi-round setting where an agent adaptively selects both questions and respondents under explicit query and participation budgets. We propose a theoretically grounded framework that combines (i) an LLM-based expected information gain objective for scoring candidate questions with (ii) heterogeneous graph neural network propagation that aggregates observed responses and participant attributes to impute missing responses and guide per-round respondent selection. This closed-loop procedure queries a small, informative subset of individuals while inferring population-level responses via structured similarity. Across three real-world opinion datasets, our method consistently improves population-level response prediction under constrained budgets, including a >12% relative gain on CES at a 10% respondent budget.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.14257v1">AD-Bench: A Real-World, Trajectory-Aware Advertising Analytics Benchmark for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-15
      | 💬 15 pages, 11 figures
    </div>
    <details class="paper-abstract">
      While Large Language Model (LLM) agents have achieved remarkable progress in complex reasoning tasks, evaluating their performance in real-world environments has become a critical problem. Current benchmarks, however, are largely restricted to idealized simulations, failing to address the practical demands of specialized domains like advertising and marketing analytics. In these fields, tasks are inherently more complex, often requiring multi-round interaction with professional marketing tools. To address this gap, we propose AD-Bench, a benchmark designed based on real-world business requirements of advertising and marketing platforms. AD-Bench is constructed from real user marketing analysis requests, with domain experts providing verifiable reference answers and corresponding reference tool-call trajectories. The benchmark categorizes requests into three difficulty levels (L1-L3) to evaluate agents' capabilities under multi-round, multi-tool collaboration. Experiments show that on AD-Bench, Gemini-3-Pro achieves Pass@1 = 68.0% and Pass@3 = 83.0%, but performance drops significantly on L3 to Pass@1 = 49.4% and Pass@3 = 62.1%, with a trajectory coverage of 70.1%, indicating that even state-of-the-art models still exhibit substantial capability gaps in complex advertising and marketing analysis scenarios. AD-Bench provides a realistic benchmark for evaluating and improving advertising marketing agents, the leaderboard and code can be found at https://github.com/Emanual20/adbench-leaderboard.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00097v3">The Agentic Leash: Extracting Causal Feedback Fuzzy Cognitive Maps with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-15
      | 💬 15 figures
    </div>
    <details class="paper-abstract">
      We design a large-language-model (LLM) agent system that extracts causal feedback fuzzy cognitive maps (FCMs) from raw text. The causal learning or extraction process is agentic both because of the LLM's semi-autonomy and because ultimately the FCM dynamical system's equilibria drive the LLM agents to fetch and process causal text. The fetched text can in principle modify the adaptive FCM causal structure and so modify the source of its quasi-autonomy$-$its equilibrium limit cycles and fixed-point attractors. This bidirectional process endows the evolving FCM dynamical system with a degree of autonomy while the system still stays on its agentic leash. We show in particular that a sequence of three system-instruction sets guide an LLM agent as it systematically extracts key nouns and noun phrases from text, as it extracts FCM concept nodes from among those nouns and noun phrases, and then as it extracts or infers partial or fuzzy causal edges between those FCM nodes. We test this FCM generation on a recent essay about the promise of AI from the late diplomat and political theorist Henry Kissinger and his colleagues. This three-step process produced FCM dynamical systems that converged to the same equilibrium limit cycles as did the human-generated FCMs even though the human-generated FCM differed in the number of nodes and edges. A final FCM mixed generated FCMs from separate Gemini and ChatGPT LLM agents. The mixed FCM absorbed the equilibria of its dominant mixture component but also created new equilibria of its own to better approximate the underlying causal dynamical system.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.14233v1">Evaluating LLMs in Finance Requires Explicit Bias Consideration</a></div>
    <div class="paper-meta">
      📅 2026-02-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly integrated into financial workflows, but evaluation practice has not kept up. Finance-specific biases can inflate performance, contaminate backtests, and make reported results useless for any deployment claim. We identify five recurring biases in financial LLM applications. They include look-ahead bias, survivorship bias, narrative bias, objective bias, and cost bias. These biases break financial tasks in distinct ways and they often compound to create an illusion of validity. We reviewed 164 papers from 2023 to 2025 and found that no single bias is discussed in more than 28 percent of studies. This position paper argues that bias in financial LLM systems requires explicit attention and that structural validity should be enforced before any result is used to support a deployment claim. We propose a Structural Validity Framework and an evaluation checklist with minimal requirements for bias diagnosis and future system design. The material is available at https://github.com/Eleanorkong/Awesome-Financial-LLM-Bias-Mitigation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.14214v1">HiVid: LLM-Guided Video Saliency For Content-Aware VOD And Live Streaming</a></div>
    <div class="paper-meta">
      📅 2026-02-15
      | 💬 ICLR 2026
    </div>
    <details class="paper-abstract">
      Content-aware streaming requires dynamic, chunk-level importance weights to optimize subjective quality of experience (QoE). However, direct human annotation is prohibitively expensive while vision-saliency models generalize poorly. We introduce HiVid, the first framework to leverage Large Language Models (LLMs) as a scalable human proxy to generate high-fidelity weights for both Video-on-Demand (VOD) and live streaming. We address 3 non-trivial challenges: (1) To extend LLMs' limited modality and circumvent token limits, we propose a perception module to assess frames in a local context window, autoregressively building a coherent understanding of the video. (2) For VOD with rating inconsistency across local windows, we propose a ranking module to perform global re-ranking with a novel LLM-guided merge-sort algorithm. (3) For live streaming which requires low-latency, online inference without future knowledge, we propose a prediction module to predict future weights with a multi-modal time series model, which comprises a content-aware attention and adaptive horizon to accommodate asynchronous LLM inference. Extensive experiments show HiVid improves weight prediction accuracy by up to 11.5\% for VOD and 26\% for live streaming over SOTA baselines. Real-world user study validates HiVid boosts streaming QoE correlation by 14.7\%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.14209v1">MAGE: All-[MASK] Block Already Knows Where to Look in Diffusion LLM</a></div>
    <div class="paper-meta">
      📅 2026-02-15
    </div>
    <details class="paper-abstract">
      Block diffusion LLMs are emerging as a promising next paradigm for language generation, but their use of KV caching makes memory access a dominant bottleneck in long-context settings. While dynamic sparse attention has been actively explored, existing methods designed for autoregressive LLMs rely on approximate importance estimation and perform poorly when adapted to block diffusion. This work identifies a key opportunity unique to block diffusion: attention at the first All-[MASK] denoising step reliably predicts important KV entries and budget requirements, enabling MAGE to perform a single exact attention pass per block and reuse it for training-free sparse denoising. Across long-context benchmarks including LongBench and Needle-in-a-Haystack, MAGE achieves near-lossless accuracy with a fraction of the KV budget while delivering up to 3-4x end-to-end speedup, consistently outperforming AR-oriented sparse attention baselines. A lightweight fine-tuning strategy further strengthens [MASK]-guided patterns with minimal cost, requiring only a few hours of training on a single NVIDIA H100 GPU for both 1.5B and 7B models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.14188v1">GPT-5 vs Other LLMs in Long Short-Context Performance</a></div>
    <div class="paper-meta">
      📅 2026-02-15
      | 💬 10 pages, 7 figures. Accepted for publication in the 3rd International Conference on Foundation and Large Language Models (FLLM2025). IEEE. The final version will be available in IEEE Xplore
    </div>
    <details class="paper-abstract">
      With the significant expansion of the context window in Large Language Models (LLMs), these models are theoretically capable of processing millions of tokens in a single pass. However, research indicates a significant gap between this theoretical capacity and the practical ability of models to robustly utilize information within long contexts, especially in tasks that require a comprehensive understanding of numerous details. This paper evaluates the performance of four state-of-the-art models (Grok-4, GPT-4, Gemini 2.5, and GPT-5) on long short-context tasks. For this purpose, three datasets were used: two supplementary datasets for retrieving culinary recipes and math problems, and a primary dataset of 20K social media posts for depression detection. The results show that as the input volume on the social media dataset exceeds 5K posts (70K tokens), the performance of all models degrades significantly, with accuracy dropping to around 50-53% for 20K posts. Notably, in the GPT-5 model, despite the sharp decline in accuracy, its precision remained high at approximately 95%, a feature that could be highly effective for sensitive applications like depression detection. This research also indicates that the "lost in the middle" problem has been largely resolved in newer models. This study emphasizes the gap between the theoretical capacity and the actual performance of models on complex, high-volume data tasks and highlights the importance of metrics beyond simple accuracy for practical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.02873v4">It's the Thought that Counts: Evaluating the Attempts of Frontier LLMs to Persuade on Harmful Topics</a></div>
    <div class="paper-meta">
      📅 2026-02-15
    </div>
    <details class="paper-abstract">
      Persuasion is a powerful capability of large language models (LLMs) that both enables beneficial applications (e.g. helping people quit smoking) and raises significant risks (e.g. large-scale, targeted political manipulation). Prior work has found models possess a significant and growing persuasive capability, measured by belief changes in simulated or real users. However, these benchmarks overlook a crucial risk factor: the propensity of a model to attempt to persuade in harmful contexts. Understanding whether a model will blindly ``follow orders'' to persuade on harmful topics (e.g. glorifying joining a terrorist group) is key to understanding the efficacy of safety guardrails. Moreover, understanding if and when a model will engage in persuasive behavior in pursuit of some goal is essential to understanding the risks from agentic AI systems. We propose the Attempt to Persuade Eval (APE) benchmark, that shifts the focus from persuasion success to persuasion attempts, operationalized as a model's willingness to generate content aimed at shaping beliefs or behavior. Our evaluation framework probes frontier LLMs using a multi-turn conversational setup between simulated persuader and persuadee agents. APE explores a diverse spectrum of topics including conspiracies, controversial issues, and non-controversially harmful content. We introduce an automated evaluator model to identify willingness to persuade and measure the frequency and context of persuasive attempts. We find that many open and closed-weight models are frequently willing to attempt persuasion on harmful topics and that jailbreaking can increase willingness to engage in such behavior. Our results highlight gaps in current safety guardrails and underscore the importance of evaluating willingness to persuade as a key dimension of LLM risk. APE is available at github.com/AlignmentResearch/AttemptPersuadeEval
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.22067v2">The Rogue Scalpel: Activation Steering Compromises LLM Safety</a></div>
    <div class="paper-meta">
      📅 2026-02-15
    </div>
    <details class="paper-abstract">
      Activation steering is a promising technique for controlling LLM behavior by adding semantically meaningful vectors directly into a model's hidden states during inference. It is often framed as a precise, interpretable, and potentially safer alternative to fine-tuning. We demonstrate the opposite: steering systematically breaks model alignment safeguards, making it comply with harmful requests. Through extensive experiments on different model families, we show that even steering in a random direction can increase the probability of harmful compliance from 0% to 1-13%. Alarmingly, steering benign features from a sparse autoencoder (SAE), a common source of interpretable directions, demonstrates a comparable harmful potential. Finally, we show that combining 20 randomly sampled vectors that jailbreak a single prompt creates a universal attack, significantly increasing harmful compliance on unseen requests. These results challenge the paradigm of safety through interpretability, showing that precise control over model internals does not guarantee precise control over model behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.14081v1">CCiV: A Benchmark for Structure, Rhythm and Quality in LLM-Generated Chinese \textit{Ci} Poetry</a></div>
    <div class="paper-meta">
      📅 2026-02-15
      | 💬 ARR 2025 May and Icassp 2026 submission. Working in progress
    </div>
    <details class="paper-abstract">
      The generation of classical Chinese \textit{Ci} poetry, a form demanding a sophisticated blend of structural rigidity, rhythmic harmony, and artistic quality, poses a significant challenge for large language models (LLMs). To systematically evaluate and advance this capability, we introduce \textbf{C}hinese \textbf{Ci}pai \textbf{V}ariants (\textbf{CCiV}), a benchmark designed to assess LLM-generated \textit{Ci} poetry across these three dimensions: structure, rhythm, and quality. Our evaluation of 17 LLMs on 30 \textit{Cipai} reveals two critical phenomena: models frequently generate valid but unexpected historical variants of a poetic form, and adherence to tonal patterns is substantially harder than structural rules. We further show that form-aware prompting can improve structural and tonal control for stronger models, while potentially degrading weaker ones. Finally, we observe weak and inconsistent alignment between formal correctness and literary quality in our sample. CCiV highlights the need for variant-aware evaluation and more holistic constrained creative generation methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.14044v1">Context Shapes LLMs Retrieval-Augmented Fact-Checking Effectiveness</a></div>
    <div class="paper-meta">
      📅 2026-02-15
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) show strong reasoning abilities across diverse tasks, yet their performance on extended contexts remains inconsistent. While prior research has emphasized mid-context degradation in question answering, this study examines the impact of context in LLM-based fact verification. Using three datasets (HOVER, FEVEROUS, and ClimateFEVER) and five open-source models accross different parameters sizes (7B, 32B and 70B parameters) and model families (Llama-3.1, Qwen2.5 and Qwen3), we evaluate both parametric factual knowledge and the impact of evidence placement across varying context lengths. We find that LLMs exhibit non-trivial parametric knowledge of factual claims and that their verification accuracy generally declines as context length increases. Similarly to what has been shown in previous works, in-context evidence placement plays a critical role with accuracy being consistently higher when relevant evidence appears near the beginning or end of the prompt and lower when placed mid-context. These results underscore the importance of prompt structure in retrieval-augmented fact-checking systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.14038v1">Choosing How to Remember: Adaptive Memory Structures for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-15
    </div>
    <details class="paper-abstract">
      Memory is critical for enabling large language model (LLM) based agents to maintain coherent behavior over long-horizon interactions. However, existing agent memory systems suffer from two key gaps: they rely on a one-size-fits-all memory structure and do not model memory structure selection as a context-adaptive decision, limiting their ability to handle heterogeneous interaction patterns and resulting in suboptimal performance. We propose a unified framework, FluxMem, that enables adaptive memory organization for LLM agents. Our framework equips agents with multiple complementary memory structures. It explicitly learns to select among these structures based on interaction-level features, using offline supervision derived from downstream response quality and memory utilization. To support robust long-horizon memory evolution, we further introduce a three-level memory hierarchy and a Beta Mixture Model-based probabilistic gate for distribution-aware memory fusion, replacing brittle similarity thresholds. Experiments on two long-horizon benchmarks, PERSONAMEM and LoCoMo, demonstrate that our method achieves average improvements of 9.18% and 6.14%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.13766v3">Advancing Software Quality: A Standards-Focused Review of LLM-Based Assurance Techniques</a></div>
    <div class="paper-meta">
      📅 2026-02-15
      | 💬 16 pages, 1 Table, 6 Figures
    </div>
    <details class="paper-abstract">
      Software Quality Assurance (SQA) is critical for delivering reliable, secure, and efficient software products. The Software Quality Assurance Process aims to provide assurance that work products and processes comply with predefined provisions and plans. Recent advancements in Large Language Models (LLMs) present new opportunities to enhance existing SQA processes by automating tasks like requirement analysis, code review, test generation, and compliance checks. Simultaneously, established standards such as ISO/IEC 12207, ISO/IEC 25010, ISO/IEC 5055, ISO 9001/ISO/IEC 90003, CMMI, and TMM provide structured frameworks for ensuring robust quality practices. This paper surveys the intersection of LLM-based SQA methods and these recognized standards, highlighting how AI-driven solutions can augment traditional approaches while maintaining compliance and process maturity. We first review the foundational software quality standards and the technical fundamentals of LLMs in software engineering. Next, we explore various LLM-based SQA applications, including requirement validation, defect detection, test generation, and documentation maintenance. We then map these applications to key software quality frameworks, illustrating how LLMs can address specific requirements and metrics within each standard. Empirical case studies and open-source initiatives demonstrate the practical viability of these methods. At the same time, discussions on challenges (e.g., data privacy, model bias, explainability) underscore the need for deliberate governance and auditing. Finally, we propose future directions encompassing adaptive learning, privacy-focused deployments, multimodal analysis, and evolving standards for AI-driven software quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.14012v1">From SFT to RL: Demystifying the Post-Training Pipeline for LLM-based Vulnerability Detection</a></div>
    <div class="paper-meta">
      📅 2026-02-15
    </div>
    <details class="paper-abstract">
      The integration of LLMs into vulnerability detection (VD) has shifted the field toward interpretable and context-aware analysis. While post-training methods have shown promise in general coding tasks, their systematic application to VD remains underexplored. In this paper, we present the first comprehensive investigation into the post-training pipeline for LLM-based VD, spanning from cold-start SFT to off-policy preference optimization and on-policy RL, uncovering how data curation, stage interactions, reward mechanisms, and evaluation protocols collectively dictate the efficacy of model training and assessment. Our study identifies practical guidelines and insights: (1) SFT based on rejection sampling greatly outperforms rationalization-based supervision, which can introduce hallucinations due to ground-truth leakage. (2) While increased SFT epochs constantly benefit preference optimization, excessive SFT inhibits self-exploration during RL, ultimately limiting performance gains. (3) Coarse-grained reward signals often mislead RL, whereas fine-grained root-cause judgments ensure reliable credit assignment. Specification-based rewards offer further benefits but incur significant effort in specification generation. (4) Although filtering extremely hard-to-detect vulnerability samples improves RL training efficiency, the cost of performance loss should be considered in practical applications. (5) Models trained under GRPO significantly outperform those using SFT and preference optimization (i.e., DPO and ORPO), as well as a series of zero-shot SOTA LLMs, underscoring the significant potential of on-policy RL for LLM-based VD. (6) In contrast to binary matching that tends to overestimate performance, LLM-as-a-Judge based on root-cause analysis provides a more robust evaluation protocol, although its accuracy varies across judge models with different levels of security expertise.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.06361v3">Beyond Prompt-Induced Lies: Investigating LLM Deception on Benign Prompts</a></div>
    <div class="paper-meta">
      📅 2026-02-15
      | 💬 ICLR 2026 (Oral)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely deployed in reasoning, planning, and decision-making tasks, making their trustworthiness critical. A significant and underexplored risk is intentional deception, where an LLM deliberately fabricates or conceals information to serve a hidden objective. Existing studies typically induce deception by explicitly setting a hidden objective through prompting or fine-tuning, which may not reflect real-world human-LLM interactions. Moving beyond such human-induced deception, we investigate LLMs' self-initiated deception on benign prompts. To address the absence of ground truth, we propose a framework based on Contact Searching Questions~(CSQ). This framework introduces two statistical metrics derived from psychological principles to quantify the likelihood of deception. The first, the Deceptive Intention Score, measures the model's bias toward a hidden objective. The second, the Deceptive Behavior Score, measures the inconsistency between the LLM's internal belief and its expressed output. Evaluating 16 leading LLMs, we find that both metrics rise in parallel and escalate with task difficulty for most models. Moreover, increasing model capacity does not always reduce deception, posing a significant challenge for future LLM development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.24496v2">LLM DNA: Tracing Model Evolution via Functional Representations</a></div>
    <div class="paper-meta">
      📅 2026-02-15
      | 💬 ICLR 2026 (Oral)
    </div>
    <details class="paper-abstract">
      The explosive growth of large language models (LLMs) has created a vast but opaque landscape: millions of models exist, yet their evolutionary relationships through fine-tuning, distillation, or adaptation are often undocumented or unclear, complicating LLM management. Existing methods are limited by task specificity, fixed model sets, or strict assumptions about tokenizers or architectures. Inspired by biological DNA, we address these limitations by mathematically defining LLM DNA as a low-dimensional, bi-Lipschitz representation of functional behavior. We prove that LLM DNA satisfies inheritance and genetic determinism properties and establish the existence of DNA. Building on this theory, we derive a general, scalable, training-free pipeline for DNA extraction. In experiments across 305 LLMs, DNA aligns with prior studies on limited subsets and achieves superior or competitive performance on specific tasks. Beyond these tasks, DNA comparisons uncover previously undocumented relationships among LLMs. We further construct the evolutionary tree of LLMs using phylogenetic algorithms, which align with shifts from encoder-decoder to decoder-only architectures, reflect temporal progression, and reveal distinct evolutionary speeds across LLM families.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.13962v1">CodeGlance: Understanding Code Reasoning Challenges in LLMs through Multi-Dimensional Feature Analysis</a></div>
    <div class="paper-meta">
      📅 2026-02-15
    </div>
    <details class="paper-abstract">
      In modern software development, developers frequently need to understand code behavior at a glance -- whether reviewing pull requests, debugging issues, or navigating unfamiliar codebases. This ability to reason about dynamic program behavior is fundamental to effective software engineering and increasingly supported by Large Language Models (LLMs). However, existing studies on code reasoning focus primarily on isolated code snippets, overlooking the complexity of real-world scenarios involving external API interactions and unfamiliar functions. This gap hinders our understanding of what truly makes code reasoning challenging for LLMs across diverse programming contexts. We present CodeGlance, a multi-dimensional benchmark investigating code reasoning challenges across three realistic scenarios: intrinsic logic reasoning, API interaction reasoning, and unseen function reasoning. Through systematic evaluation of 7 state-of-the-art LLMs, we reveal that unseen function reasoning poses significant challenges especially for smaller models, with Qwen2.5-3b achieving only 6.0\% accuracy on unseen functions compared to 37.5\% on familiar APIs. We identify critical code complexity features -- including execution trace length, API invocation count, and control flow complexity -- that significantly impact code reasoning difficulty across scenarios. We further investigate how common augmentation strategies, including CoT, document retrieval, and code search, can improve reasoning performance, finding that their effectiveness varies substantially depending on whether challenges stem from logical complexity or knowledge gaps. These findings provide actionable guidance for developing more capable code reasoning systems and deploying LLM-based programming assistants in real-world software development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.11087v3">Enhancing Delta Compression in LLMs via SVD-based Quantization Error Minimization</a></div>
    <div class="paper-meta">
      📅 2026-02-15
    </div>
    <details class="paper-abstract">
      Supervised Fine-Tuning (SFT) empowers Large Language Models (LLMs) with exceptional performance on specialized tasks, but it yields dense, high-dimensional delta parameters that pose severe storage and distribution challenges. Singular Value Decomposition (SVD)-based compression offers a compact representation for such delta parameters, but existing methods adopt heuristic quantization without clarifying underlying mechanisms, leading to poor generalizability. In this work, we propose PrinMix, a rigorous SVD-based framework that models quantization as an optimization problem, grounding the design in mathematical mechanisms. We first theoretically derive quantization error and identify a key singular-value-dominated scaling mechanism, which mathematically proves the necessity of mix-precision quantization. We then model the quantization scheme as a 0/1 Integer Linear Programming (ILP) problem, which yields optimal bit-budget-constrained solutions without empirical assumptions. Furthermore, PrinMix integrates a Reconstruction Target Correction (RTC) method to compensate for errors from the $\mathbf{V}$-then-$\mathbf{U}$ sequential quantization process. Extensive experiments confirm PrinMix performs well: for 7B LLMs, PrinMix outperforms SOTA Delta-CoMe on challenging benchmarks by 22.3% on AIME2024 and 6.1% on GQA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.13942v1">A Theoretical Framework for LLM Fine-tuning Using Early Stopping for Non-random Initialization</a></div>
    <div class="paper-meta">
      📅 2026-02-15
    </div>
    <details class="paper-abstract">
      In the era of large language models (LLMs), fine-tuning pretrained models has become ubiquitous. Yet the theoretical underpinning remains an open question. A central question is why only a few epochs of fine-tuning are typically sufficient to achieve strong performance on many different tasks. In this work, we approach this question by developing a statistical framework, combining rigorous early stopping theory with the attention-based Neural Tangent Kernel (NTK) for LLMs, offering new theoretical insights on fine-tuning practices. Specifically, we formally extend classical NTK theory [Jacot et al., 2018] to non-random (i.e., pretrained) initializations and provide a convergence guarantee for attention-based fine-tuning. One key insight provided by the theory is that the convergence rate with respect to sample size is closely linked to the eigenvalue decay rate of the empirical kernel matrix induced by the NTK. We also demonstrate how the framework can be used to explain task vectors for multiple tasks in LLMs. Finally, experiments with modern language models on real-world datasets provide empirical evidence supporting our theoretical insights.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.01144v2">AthenaBench: A Dynamic Benchmark for Evaluating LLMs in Cyber Threat Intelligence</a></div>
    <div class="paper-meta">
      📅 2026-02-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated strong capabilities in natural language reasoning, yet their application to Cyber Threat Intelligence (CTI) remains limited. CTI analysis involves distilling large volumes of unstructured reports into actionable knowledge, a process where LLMs could substantially reduce analyst workload. CTIBench introduced a comprehensive benchmark for evaluating LLMs across multiple CTI tasks. In this work, we extend CTIBench by developing AthenaBench, an enhanced benchmark that includes an improved dataset creation pipeline, duplicate removal, refined evaluation metrics, and a new task focused on risk mitigation strategies. We evaluate twelve LLMs, including state-of-the-art proprietary models such as GPT-5 and Gemini-2.5 Pro, alongside seven open-source models from the LLaMA and Qwen families. While proprietary LLMs achieve stronger results overall, their performance remains subpar on reasoning-intensive tasks, such as threat actor attribution and risk mitigation, with open-source models trailing even further behind. These findings highlight fundamental limitations in the reasoning capabilities of current LLMs and underscore the need for models explicitly tailored to CTI workflows and automation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04856v3">CoT is Not the Chain of Truth: An Empirical Internal Analysis of Reasoning LLMs for Fake News Generation</a></div>
    <div class="paper-meta">
      📅 2026-02-14
      | 💬 28 pages, 35 figures
    </div>
    <details class="paper-abstract">
      From generating headlines to fabricating news, the Large Language Models (LLMs) are typically assessed by their final outputs, under the safety assumption that a refusal response signifies safe reasoning throughout the entire process. Challenging this assumption, our study reveals that during fake news generation, even when a model rejects a harmful request, its Chain-of-Thought (CoT) reasoning may still internally contain and propagate unsafe narratives. To analyze this phenomenon, we introduce a unified safety-analysis framework that systematically deconstructs CoT generation across model layers and evaluates the role of individual attention heads through Jacobian-based spectral metrics. Within this framework, we introduce three interpretable measures: stability, geometry, and energy to quantify how specific attention heads respond or embed deceptive reasoning patterns. Extensive experiments on multiple reasoning-oriented LLMs show that the generation risk rise significantly when the thinking mode is activated, where the critical routing decisions concentrated in only a few contiguous mid-depth layers. By precisely identifying the attention heads responsible for this divergence, our work challenges the assumption that refusal implies safety and provides a new understanding perspective for mitigating latent reasoning risks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.16730v2">RapidPen: Fully Automated IP-to-Shell Penetration Testing with LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-14
    </div>
    <details class="paper-abstract">
      We present RapidPen, a fully automated penetration testing (pentesting) framework that addresses the challenge of achieving an initial foothold (IP-to-Shell) without human intervention. Unlike prior approaches that focus primarily on post-exploitation or require a human-in-the-loop, RapidPen leverages large language models (LLMs) to autonomously discover and exploit vulnerabilities, starting from a single IP address. By integrating advanced ReAct-style task planning (Re) with retrieval-augmented knowledge bases of successful exploits, along with a command-generation and direct execution feedback loop (Act), RapidPen systematically scans services, identifies viable attack vectors, and executes targeted exploits in a fully automated manner. In our evaluation against a vulnerable target from the Hack The Box platform, RapidPen achieved shell access within 200-400 seconds at a per-run cost of approximately \$0.3-\$0.6, demonstrating a 60\% success rate when reusing prior "success-case" data. These results underscore the potential of truly autonomous pentesting for both security novices and seasoned professionals. Organizations without dedicated security teams can leverage RapidPen to quickly identify critical vulnerabilities, while expert pentesters can offload repetitive tasks and focus on complex challenges. Ultimately, our work aims to make penetration testing more accessible and cost-efficient, thereby enhancing the overall security posture of modern software ecosystems. Fore more information, visit this link: https://secdevlab.com/rapidpen
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2312.00326v26">Agent-OM: Leveraging LLM Agents for Ontology Matching</a></div>
    <div class="paper-meta">
      📅 2026-02-14
      | 💬 31 pages - VLDB 2025 (Page 1-20), OM 2025 (Page 21-31)
    </div>
    <details class="paper-abstract">
      Ontology matching (OM) enables semantic interoperability between different ontologies and resolves their conceptual heterogeneity by aligning related entities. OM systems currently have two prevailing design paradigms: conventional knowledge-based expert systems and newer machine learning-based predictive systems. While large language models (LLMs) and LLM agents have revolutionised data engineering and have been applied creatively in many domains, their potential for OM remains underexplored. This study introduces a novel agent-powered LLM-based design paradigm for OM systems. With consideration of several specific challenges in leveraging LLM agents for OM, we propose a generic framework, namely Agent-OM (Agent for Ontology Matching), consisting of two Siamese agents for retrieval and matching, with a set of OM tools. Our framework is implemented in a proof-of-concept system. Evaluations of three Ontology Alignment Evaluation Initiative (OAEI) tracks over state-of-the-art OM systems show that our system can achieve results very close to the long-standing best performance on simple OM tasks and can significantly improve the performance on complex and few-shot OM tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.13713v1">On Theoretically-Driven LLM Agents for Multi-Dimensional Discourse Analysis</a></div>
    <div class="paper-meta">
      📅 2026-02-14
      | 💬 8 pages, 4 figures, 3 tables. This is the accepted version of the paper presented at the 18th International Conference on Agents and Artificial Intelligence (ICAART 2026), Marbella, Spain
    </div>
    <details class="paper-abstract">
      Identifying the strategic uses of reformulation in discourse remains a key challenge for computational argumentation. While LLMs can detect surface-level similarity, they often fail to capture the pragmatic functions of rephrasing, such as its role within rhetorical discourse. This paper presents a comparative multi-agent framework designed to quantify the benefits of incorporating explicit theoretical knowledge for this task. We utilise an dataset of annotated political debates to establish a new standard encompassing four distinct rephrase functions: Deintensification, Intensification, Specification, Generalisation, and Other, which covers all remaining types (D-I-S-G-O). We then evaluate two parallel LLM-based agent systems: one enhanced by argumentation theory via Retrieval-Augmented Generation (RAG), and an identical zero-shot baseline. The results reveal a clear performance gap: the RAG-enhanced agents substantially outperform the baseline across the board, with particularly strong advantages in detecting Intensification and Generalisation context, yielding an overall Macro F1-score improvement of nearly 30\%. Our findings provide evidence that theoretical grounding is not only beneficial but essential for advancing beyond mere paraphrase detection towards function-aware analysis of argumentative discourse. This comparative multi-agent architecture represents a step towards scalable, theoretically informed computational tools capable of identifying rhetorical strategies in contemporary discourse.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.13699v1">Attention Head Entropy of LLMs Predicts Answer Correctness</a></div>
    <div class="paper-meta">
      📅 2026-02-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often generate plausible yet incorrect answers, posing risks in safety-critical settings such as medicine. Human evaluation is expensive, and LLM-as-judge approaches risk introducing hidden errors. Recent white-box methods detect contextual hallucinations using model internals, focusing on the localization of the attention mass, but two questions remain open: do these approaches extend to predicting answer correctness, and do they generalize out-of-domains? We introduce Head Entropy, a method that predicts answer correctness from attention entropy patterns, specifically measuring the spread of the attention mass. Using sparse logistic regression on per-head 2-Renyi entropies, Head Entropy matches or exceeds baselines in-distribution and generalizes substantially better on out-of-domains, it outperforms the closest baseline on average by +8.5% AUROC. We further show that attention patterns over the question/context alone, before answer generation, already carry predictive signal using Head Entropy with on average +17.7% AUROC over the closest baseline. We evaluate across 5 instruction-tuned LLMs and 3 QA datasets spanning general knowledge, multi-hop reasoning, and medicine.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.13671v1">MAS-on-the-Fly: Dynamic Adaptation of LLM-based Multi-Agent Systems at Test Time</a></div>
    <div class="paper-meta">
      📅 2026-02-14
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based multi-agent systems (MAS) have emerged as a promising paradigm for solving complex tasks. However, existing works often rely on manual designs or "one-size-fits-all" automation, lacking dynamic adaptability after deployment. Inspired by how biological systems adapt, we introduce MASFly, a novel multi-agent framework enabling dynamic adaptation at test time. To adapt system generation, MASFly employs a retrieval-augmented SOP instantiation mechanism that leverages a self-constructed repository of successful collaboration patterns, enabling the LLM to assemble customized MASs for new queries. For adaptive execution, MASFly incorporates an experience-guided supervision mechanism, where a dedicated Watcher agent monitors system behaviors with reference to a personalized experience pool and provides real-time interventions. Extensive experiments demonstrate that MASFly achieves state-of-the-art performance, most notably a 61.7% success rate on the TravelPlanner benchmark, while exhibiting strong task adaptability and robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.13665v1">HyFunc: Accelerating LLM-based Function Calls for Agentic AI through Hybrid-Model Cascade and Dynamic Templating</a></div>
    <div class="paper-meta">
      📅 2026-02-14
      | 💬 Accepted by KDD'26
    </div>
    <details class="paper-abstract">
      While agentic AI systems rely on LLMs to translate user intent into structured function calls, this process is fraught with computational redundancy, leading to high inference latency that hinders real-time applications. This paper identifies and addresses three key redundancies: (1) the redundant processing of a large library of function descriptions for every request; (2) the redundant use of a large, slow model to generate an entire, often predictable, token sequence; and (3) the redundant generation of fixed, boilerplate parameter syntax. We introduce HyFunc, a novel framework that systematically eliminates these inefficiencies. HyFunc employs a hybrid-model cascade where a large model distills user intent into a single "soft token." This token guides a lightweight retriever to select relevant functions and directs a smaller, prefix-tuned model to generate the final call, thus avoiding redundant context processing and full-sequence generation by the large model. To eliminate syntactic redundancy, our "dynamic templating" technique injects boilerplate parameter syntax on-the-fly within an extended vLLM engine. To avoid potential limitations in generalization, we evaluate HyFunc on an unseen benchmark dataset, BFCL. Experimental results demonstrate that HyFunc achieves an excellent balance between efficiency and performance. It achieves an inference latency of 0.828 seconds, outperforming all baseline models, and reaches a performance of 80.1%, surpassing all models with a comparable parameter scale. These results suggest that HyFunc offers a more efficient paradigm for agentic AI. Our code is publicly available at https://github.com/MrBlankness/HyFunc.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.13659v1">Zero-Order Optimization for LLM Fine-Tuning via Learnable Direction Sampling</a></div>
    <div class="paper-meta">
      📅 2026-02-14
    </div>
    <details class="paper-abstract">
      Fine-tuning large pretrained language models (LLMs) is a cornerstone of modern NLP, yet its growing memory demands (driven by backpropagation and large optimizer States) limit deployment in resource-constrained settings. Zero-order (ZO) methods bypass backpropagation by estimating directional derivatives from forward evaluations, offering substantial memory savings. However, classical ZO estimators suffer from high variance and an adverse dependence on the parameter dimensionality $d$, which has constrained their use to low-dimensional problems. In this work, we propose a policy-driven ZO framework that treats the sampling distribution over perturbation directions as a learnable policy and updates it to reduce the variance of directional estimates. We develop a practical algorithm implementing this idea and provide a theoretical analysis, showing that learned sampling distributions improve the quality of gradient information and relax the explicit dependence on $d$ in convergence bounds. Empirically, we validate the approach on challenging LLM fine-tuning benchmarks, demonstrating substantially improved performance compared to standard ZO baselines. Our results suggest that adaptive direction sampling is a promising route to make ZO fine-tuning viable at scale. The source code is available at https://github.com/brain-lab-research/zo_ldsd
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.16701v2">An Agentic Framework with LLMs for Solving Complex Vehicle Routing Problems</a></div>
    <div class="paper-meta">
      📅 2026-02-14
      | 💬 Accepted by iclr2026
    </div>
    <details class="paper-abstract">
      Complex vehicle routing problems (VRPs) remain a fundamental challenge, demanding substantial expert effort for intent interpretation and algorithm design. While large language models (LLMs) offer a promising path toward automation, current approaches still rely on external intervention, which restrict autonomy and often lead to execution errors and low solution feasibility. To address these challenges, we propose an Agentic Framework with LLMs (AFL) for solving complex vehicle routing problems, achieving full automation from problem instance to solution. AFL directly extracts knowledge from raw inputs and enables self-contained code generation without handcrafted modules or external solvers. To improve trustworthiness, AFL decomposes the overall pipeline into three manageable subtasks and employs four specialized agents whose coordinated interactions enforce cross-functional consistency and logical soundness. Extensive experiments on 60 complex VRPs, ranging from standard benchmarks to practical variants, validate the effectiveness and generality of our framework, showing comparable performance against meticulously designed algorithms. Notably, it substantially outperforms existing LLM-based baselines in both code reliability and solution feasibility, achieving rates close to 100% on the evaluated benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.13639v1">Guided Collaboration in Heterogeneous LLM-Based Multi-Agent Systems via Entropy-Based Understanding Assessment and Experience Retrieval</a></div>
    <div class="paper-meta">
      📅 2026-02-14
    </div>
    <details class="paper-abstract">
      With recent breakthroughs in large language models (LLMs) for reasoning, planning, and complex task generation, artificial intelligence systems are transitioning from isolated single-agent architectures to multi-agent systems with collaborative intelligence. However, in heterogeneous multi-agent systems (HMAS), capability differences among agents give rise to consistent cognitive problems, where strong and weak models fail to contribute effectively. We define the collaboration as a strong-weak system. Through comprehensive experiments, we disclose a counterintuitive phenomenon in the strong-weak system: a strong-weak collaboration may under-perform weak-weak combinations, revealing that cognitive mismatching are key bottlenecks limiting heterogeneous cooperation. To overcome these challenges, we propose an Entropy-Based Adaptive Guidance Framework that dynamically aligns the guidance with the cognitive state of each agent. The framework quantifies the understanding of weak agents through multi-dimensional entropy metrics - covering expression, uncertainty, structure, coherence, and relevance - and adaptively adjusts the intensity of the guidance at light, moderate and intensive levels. Furthermore, a Retrieval-Augmented Generation (RAG) mechanism is incorporated to retain successful collaboration experiences, enabling both immediate adaptation and long-term learning. Extensive experiments on three benchmark datasets, GSM8K, MBPP, and CVRP demonstrate that our approach consistently enhances the effectiveness and stability of heterogeneous collaboration. The results highlight that adaptive guidance not only mitigates cognitive imbalance but also establishes a scalable pathway toward more robust, cooperative multi-agent intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.12259v1">Think like a Scientist: Physics-guided LLM Agent for Equation Discovery</a></div>
    <div class="paper-meta">
      📅 2026-02-12
    </div>
    <details class="paper-abstract">
      Explaining observed phenomena through symbolic, interpretable formulas is a fundamental goal of science. Recently, large language models (LLMs) have emerged as promising tools for symbolic equation discovery, owing to their broad domain knowledge and strong reasoning capabilities. However, most existing LLM-based systems try to guess equations directly from data, without modeling the multi-step reasoning process that scientists often follow: first inferring physical properties such as symmetries, then using these as priors to restrict the space of candidate equations. We introduce KeplerAgent, an agentic framework that explicitly follows this scientific reasoning process. The agent coordinates physics-based tools to extract intermediate structure and uses these results to configure symbolic regression engines such as PySINDy and PySR, including their function libraries and structural constraints. Across a suite of physical equation benchmarks, KeplerAgent achieves substantially higher symbolic accuracy and greater robustness to noisy data than both LLM and traditional baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.18134v2">Evaluating LLM Reasoning Beyond Correctness and CoT</a></div>
    <div class="paper-meta">
      📅 2026-02-12
    </div>
    <details class="paper-abstract">
      What does it truly mean for a language model to "reason"? Current evaluations reward models' correct standalone answers-but correctness alone reveals little about the process that produced them. We argue that reasoning should be understood not as a static chain of steps but as a dynamic trajectory in which ideas interact, clash, and evolve into integrated insights. Building on the philosophical tradition of dialectics, we introduce SIEV, a structured evaluation framework that assesses reasoning through explicit thesis-antithesis-synthesis interactions. SIEV produces interpretable trajectories that highlight key properties of reasoning-robustness to challenge, adaptability under conflict, and synthesis across competing viewpoints-dimensions that conventional correctness-based metrics cannot capture. Empirical results on GSM and MMLU demonstrate substantial gaps in the reasoning abilities of state-of-the-art models: for example, GPT-5-chat loses more than 40 points (out of 100) on GSM when evaluated through SIEV's process-oriented lens. By shifting focus from what answer a model gives to how it arrives there, SIEV enables a more transparent and principled distinction between structured reasoning and surface-level pattern generation offering a clearer foundation for assessing and understanding the reasoning capabilities of LLMs.
    </details>
</div>
