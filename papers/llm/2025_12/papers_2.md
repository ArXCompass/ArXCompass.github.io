# llm - 2025_12

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
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22492v1">Role-Based Fault Tolerance System for LLM RL Post-Training</a></div>
    <div class="paper-meta">
      📅 2025-12-27
      | 💬 16 pages, 19 figures
    </div>
    <details class="paper-abstract">
      RL post-training for LLMs has been widely scaled to enhance reasoning and tool-using capabilities. However, RL post-training interleaves training and inference workloads, exposing the system to faults from both sides. Existing fault tolerance frameworks for LLMs target either training or inference, leaving the optimization potential in the asynchronous execution unexplored for RL. Our key insight is role-based fault isolation so the failure in one machine does not affect the others. We treat trainer, rollout, and other management roles in RL training as distinct distributed sub-tasks. Instead of restarting the entire RL task in ByteRobust, we recover only the failed role and reconnect it to living ones, thereby eliminating the full-restart overhead including rollout replay and initialization delay. We present RobustRL, the first comprehensive robust system to handle GPU machine errors for RL post-training Effective Training Time Ratio improvement. (1) \textit{Detect}. We implement role-aware monitoring to distinguish actual failures from role-specific behaviors to avoid the false positive and delayed detection. (2) \textit{Restart}. For trainers, we implement a non-disruptive recovery where rollouts persist state and continue trajectory generation, while the trainer is rapidly restored via rollout warm standbys. For rollout, we perform isolated machine replacement without interrupting the RL task. (3) \textit{Reconnect}. We replace static collective communication with dynamic, UCX-based (Unified Communication X) point-to-point communication, enabling immediate weight synchronization between recovered roles. In an RL training task on a 256-GPU cluster with Qwen3-8B-Math workload under 10\% failure injection frequency, RobustRL can achieve an ETTR of over 80\% compared with the 60\% in ByteRobust and achieves 8.4\%-17.4\% faster in end-to-end training time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.00256v2">LLM4FP: LLM-Based Program Generation for Triggering Floating-Point Inconsistencies Across Compilers</a></div>
    <div class="paper-meta">
      📅 2025-12-27
    </div>
    <details class="paper-abstract">
      Floating-point inconsistencies across compilers can undermine the reliability of numerical software. We present LLM4FP, the first framework that uses Large Language Models (LLMs) to generate floating-point programs specifically designed to trigger such inconsistencies. LLM4FP combines Grammar-Based Generation and Feedback-Based Mutation to produce diverse and valid programs. We evaluate LLM4FP across multiple compilers and optimization levels, measuring inconsistency rate, time cost, and program diversity. LLM4FP detects nearly 2.5x the number of inconsistencies as the state-of-the-art tool Varity. Notably, most of the inconsistencies involve real-valued differences, rather than extreme values like NaN or infinities. LLM4FP also uncovers inconsistencies across a wider range of optimization levels, and finds the most mismatches between host and device compilers. These results show that LLM-guided program generation improves the detection of numerical inconsistencies. In practice, numerical software and HPC developers can use LLM4FP to compare compilers and select those that provide more accurate and consistent floating-point behavior, while compiler developers can use it to identify and address subtle consistency issues in their implementations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22470v1">DarkPatterns-LLM: A Multi-Layer Benchmark for Detecting Manipulative and Harmful AI Behavior</a></div>
    <div class="paper-meta">
      📅 2025-12-27
    </div>
    <details class="paper-abstract">
      The proliferation of Large Language Models (LLMs) has intensified concerns about manipulative or deceptive behaviors that can undermine user autonomy, trust, and well-being. Existing safety benchmarks predominantly rely on coarse binary labels and fail to capture the nuanced psychological and social mechanisms constituting manipulation. We introduce \textbf{DarkPatterns-LLM}, a comprehensive benchmark dataset and diagnostic framework for fine-grained assessment of manipulative content in LLM outputs across seven harm categories: Legal/Power, Psychological, Emotional, Physical, Autonomy, Economic, and Societal Harm. Our framework implements a four-layer analytical pipeline comprising Multi-Granular Detection (MGD), Multi-Scale Intent Analysis (MSIAN), Threat Harmonization Protocol (THP), and Deep Contextual Risk Alignment (DCRA). The dataset contains 401 meticulously curated examples with instruction-response pairs and expert annotations. Through evaluation of state-of-the-art models including GPT-4, Claude 3.5, and LLaMA-3-70B, we observe significant performance disparities (65.2\%--89.7\%) and consistent weaknesses in detecting autonomy-undermining patterns. DarkPatterns-LLM establishes the first standardized, multi-dimensional benchmark for manipulation detection in LLMs, offering actionable diagnostics toward more trustworthy AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22462v1">Relational Mediators: LLM Chatbots as Boundary Objects in Psychotherapy</a></div>
    <div class="paper-meta">
      📅 2025-12-27
      | 💬 To be published in CSCW 2026, Issue 2 of PACMHCI, 35 pages, 6 figures
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are embedded into mental health technologies, they are often framed either as tools assisting therapists or autonomous therapeutic systems. Such perspectives overlook their potential to mediate relational complexities in therapy, particularly for systemically marginalized clients. Drawing on in-depth interviews with 12 therapists and 12 marginalized clients in China, including LGBTQ+ individuals or those from other marginalized backgrounds, we identify enduring relational challenges: difficulties building trust amid institutional barriers, the burden clients carry in educating therapists about marginalized identities, and challenges sustaining authentic self-disclosure across therapy and daily life. We argue that addressing these challenges requires AI systems capable of actively mediating underlying knowledge gaps, power asymmetries, and contextual disconnects. To this end, we propose the Dynamic Boundary Mediation Framework, which reconceptualizes LLM-enhanced systems as adaptive boundary objects that shift mediating roles across therapeutic stages. The framework delineates three forms of mediation: Epistemic (reducing knowledge asymmetries), Relational (rebalancing power dynamics), and Contextual (bridging therapy-life discontinuities). This framework offers a pathway toward designing relationally accountable AI systems that center the lived realities of marginalized users and more effectively support therapeutic relationships.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.10013v2">DIF: A Framework for Benchmarking and Verifying Implicit Bias in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-27
      | 💬 10 pages, 1 figure
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) have risen in prominence over the past few years, there has been concern over the potential biases in LLMs inherited from the training data. Previous studies have examined how LLMs exhibit implicit bias, such as when response generation changes when different social contexts are introduced. We argue that this implicit bias is not only an ethical, but also a technical issue, as it reveals an inability of LLMs to accommodate extraneous information. However, unlike other measures of LLM intelligence, there are no standard methods to benchmark this specific subset of LLM bias. To bridge this gap, we developed a method for calculating an easily interpretable benchmark, DIF (Demographic Implicit Fairness), by evaluating preexisting LLM logic and math problem datasets with sociodemographic personas, which is combined with a statistical robustness check using a null model. We demonstrate that this method can validate the presence of implicit bias in LLM behavior and find an novel inverse trend between question answering accuracy and implicit bias, supporting our argument.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.21880v2">No Prompt Left Behind: Exploiting Zero-Variance Prompts in LLM Reinforcement Learning via Entropy-Guided Advantage Shaping</a></div>
    <div class="paper-meta">
      📅 2025-12-27
      | 💬 Under review. Project page: https://bltnynk.github.io/publications/rl-zvp/
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Rewards (RLVR) is a powerful framework for improving the reasoning abilities of Large Language Models (LLMs). However, current methods such as GRPO rely only on problems where the model responses to the same input differ in correctness, while ignoring those where all responses receive the same reward -- so-called zero-variance prompts. In this work, we argue that such prompts are not useless but can, in fact, provide meaningful feedback for policy optimization. To this end, we introduce RL with Zero-Variance Prompts (RL-ZVP), a novel algorithm that extract learning signals from zero-variance prompts. RL-ZVP directly rewards correctness and penalizes errors even without contrasting responses, modulating feedback with token-level characteristics to preserve informative, nuanced signals. Across six math reasoning benchmarks, RL-ZVP achieves significant improvements of up to 8.61 points in accuracy and 7.77 points in pass rate over GRPO, while consistently outperforming other baselines that filter out zero-variance prompts. These results highlight the untapped potential of learning from zero-variance prompts in RLVR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22631v1">Evaluating GRPO and DPO for Faithful Chain-of-Thought Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-27
    </div>
    <details class="paper-abstract">
      Chain-of-thought (CoT) reasoning has emerged as a powerful technique for improving the problem-solving capabilities of large language models (LLMs), particularly for tasks requiring multi-step reasoning. However, recent studies show that CoT explanations often fail to reflect the model's actual reasoning process, as models may produce coherent yet misleading justifications or modify answers without acknowledging external cues. Such discrepancies undermine the reliability of CoT-based methods for safety supervision and alignment monitoring, as models can generate plausible but deceptive rationales for incorrect answers. To better understand this limitation, we evaluate two optimization methods, Group Relative Policy Optimization (GRPO) and Direct Preference Optimization (DPO), in their ability to improve CoT faithfulness. Our experiments show that GRPO achieves higher performance than DPO in larger models, with the Qwen2.5-14B-Instruct model attaining the best results across all evaluation metrics. Both approaches exhibit positive correlations between model size and performance, but GRPO shows greater potential for improving faithfulness metrics, albeit with less stable behavior at smaller scales. These results suggest that GRPO offers a promising direction for developing more transparent and trustworthy reasoning in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17923v2">Inferring Latent Market Forces: Evaluating LLM Detection of Gamma Exposure Patterns via Obfuscation Testing</a></div>
    <div class="paper-meta">
      📅 2025-12-27
      | 💬 10 pages, 8 figures. Accepted at IEEE Big Data 2025. Extended journal version in preparation. ISBN: 979-8-3315-9447-3/25. Page numbers: 7226-7235
    </div>
    <details class="paper-abstract">
      We introduce obfuscation testing, a novel methodology for validating whether large language models detect structural market patterns through causal reasoning rather than temporal association. Testing three dealer hedging constraint patterns (gamma positioning, stock pinning, 0DTE hedging) on 242 trading days (95.6% coverage) of S&P 500 options data, we find LLMs achieve 71.5% detection rate using unbiased prompts that provide only raw gamma exposure values without regime labels or temporal context. The WHO-WHOM-WHAT causal framework forces models to identify the economic actors (dealers), affected parties (directional traders), and structural mechanisms (forced hedging) underlying observed market dynamics. Critically, detection accuracy (91.2%) remains stable even as economic profitability varies quarterly, demonstrating that models identify structural constraints rather than profitable patterns. When prompted with regime labels, detection increases to 100%, but the 71.5% unbiased rate validates genuine pattern recognition. Our findings suggest LLMs possess emergent capabilities for detecting complex financial mechanisms through pure structural reasoning, with implications for systematic strategy development, risk management, and our understanding of how transformer architectures process financial market dynamics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22625v1">The Wisdom of Deliberating AI Crowds: Does Deliberation Improve LLM-Based Forecasting?</a></div>
    <div class="paper-meta">
      📅 2025-12-27
      | 💬 13 pages, 2 figures, 5 tables, for source code and data see https://github.com/priorb-source/delib-ai-wisdom
    </div>
    <details class="paper-abstract">
      Structured deliberation has been found to improve the performance of human forecasters. This study investigates whether a similar intervention, i.e. allowing LLMs to review each other's forecasts before updating, can improve accuracy in large language models (GPT-5, Claude Sonnet 4.5, Gemini Pro 2.5). Using 202 resolved binary questions from the Metaculus Q2 2025 AI Forecasting Tournament, accuracy was assessed across four scenarios: (1) diverse models with distributed information, (2) diverse models with shared information, (3) homogeneous models with distributed information, and (4) homogeneous models with shared information. Results show that the intervention significantly improves accuracy in scenario (2), reducing Log Loss by 0.020 or about 4 percent in relative terms (p = 0.017). However, when homogeneous groups (three instances of the same model) engaged in the same process, no benefit was observed. Unexpectedly, providing LLMs with additional contextual information did not improve forecast accuracy, limiting our ability to study information pooling as a mechanism. Our findings suggest that deliberation may be a viable strategy for improving LLM forecasting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22603v1">Structured Prompting and LLM Ensembling for Multimodal Conversational Aspect-based Sentiment Analysis</a></div>
    <div class="paper-meta">
      📅 2025-12-27
    </div>
    <details class="paper-abstract">
      Understanding sentiment in multimodal conversations is a complex yet crucial challenge toward building emotionally intelligent AI systems. The Multimodal Conversational Aspect-based Sentiment Analysis (MCABSA) Challenge invited participants to tackle two demanding subtasks: (1) extracting a comprehensive sentiment sextuple, including holder, target, aspect, opinion, sentiment, and rationale from multi-speaker dialogues, and (2) detecting sentiment flipping, which detects dynamic sentiment shifts and their underlying triggers. For Subtask-I, in the present paper, we designed a structured prompting pipeline that guided large language models (LLMs) to sequentially extract sentiment components with refined contextual understanding. For Subtask-II, we further leveraged the complementary strengths of three LLMs through ensembling to robustly identify sentiment transitions and their triggers. Our system achieved a 47.38% average score on Subtask-I and a 74.12% exact match F1 on Subtask-II, showing the effectiveness of step-wise refinement and ensemble strategies in rich, multimodal sentiment analysis tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02910v2">In Silico Development of Psychometric Scales: Feasibility of Representative Population Data Simulation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-27
    </div>
    <details class="paper-abstract">
      Developing and validating psychometric scales requires large samples, multiple testing phases, and substantial resources. Recent advances in Large Language Models (LLMs) enable the generation of synthetic participant data by prompting models to answer items while impersonating individuals of specific demographic profiles, potentially allowing in silico piloting before real data collection. Across four preregistered studies (N = circa 300 each), we tested whether LLM-simulated datasets can reproduce the latent structures and measurement properties of human responses. In Studies 1-2, we compared LLM-generated data with real datasets for two validated scales; in Studies 3-4, we created new scales using EFA on simulated data and then examined whether these structures generalized to newly collected human samples. Simulated datasets replicated the intended factor structures in three of four studies and showed consistent configural and metric invariance, with scalar invariance achieved for the two newly developed scales. However, correlation-based tests revealed substantial differences between real and synthetic datasets, and notable discrepancies appeared in score distributions and variances. Thus, while LLMs capture group-level latent structures, they do not approximate individual-level data properties. Simulated datasets also showed full internal invariance across gender. Overall, LLM-generated data appear useful for early-stage, group-level psychometric prototyping, but not as substitutes for individual-level validation. We discuss methodological limitations, risks of bias and data pollution, and ethical considerations related to in silico psychometric simulations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.00592v2">Agentic Auto-Scheduling: An Experimental Study of LLM-Guided Loop Optimization</a></div>
    <div class="paper-meta">
      📅 2025-12-27
      | 💬 Accepted at the 34th International Conference on Parallel Architectures and Compilation Techniques (PACT 2025). 12 pages, plus appendix
    </div>
    <details class="paper-abstract">
      Automatic code optimization remains a difficult challenge, particularly for complex loop nests on modern hardware. This paper investigates a novel approach to code optimization where Large Language Models (LLMs) guide the process through a closed-loop interaction with a compiler. We present ComPilot, an experimental framework that leverages off-the-shelf LLMs, without any task-specific fine-tuning, as interactive optimization agents. ComPilot establishes a feedback loop where an LLM proposes transformations for a given loop nest to a compiler. The compiler attempts the transformations, reporting back legality status and measured speedup or slowdown. The LLM utilizes this concrete feedback to iteratively refine its optimization strategy. Our extensive evaluation across the PolyBench benchmark suite demonstrates the effectiveness of this zero-shot approach. ComPilot achieves geometric mean speedups of 2.66x (single run) and 3.54x (best-of-5 runs) over the original code. Furthermore, ComPilot demonstrates competitive performance against the state-of-the-art Pluto polyhedral optimizer, outperforming it in many cases. This experimental study demonstrates that general-purpose LLMs can effectively guide the code optimization process when grounded by compiler feedback, opening promising research directions for agentic AI in code optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.12018v4">Atom of Thoughts for Markov LLM Test-Time Scaling</a></div>
    <div class="paper-meta">
      📅 2025-12-27
      | 💬 Accepted to NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved significant performance gains through test-time scaling methods. However, existing approaches often incur redundant computations due to the accumulation of historical dependency information during inference. To address this challenge, we leverage the memoryless property of Markov processes to minimize reliance on historical context and propose a Markovian reasoning process. This foundational Markov chain structure enables seamless integration with various test-time scaling methods, thereby improving their scaling efficiency. By further scaling up the Markovian reasoning chain through integration with techniques such as tree search and reflective refinement, we uncover an emergent atomic reasoning structure, where reasoning trajectories are decomposed into a series of self-contained, low-complexity atomic units. We name this design Atom of Thoughts (\our). Extensive experiments demonstrate that \our consistently outperforms existing baselines as computational budgets increase. Importantly, \our integrates seamlessly with existing reasoning frameworks and different LLMs (both reasoning and non-reasoning), facilitating scalable, high-performance inference.We submit our code alongside this paper and will make it publicly available to facilitate reproducibility and future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.21015v3">Don't Retrieve, Generate: Prompting LLMs for Synthetic Training Data in Dense Retrieval</a></div>
    <div class="paper-meta">
      📅 2025-12-27
    </div>
    <details class="paper-abstract">
      Training effective dense retrieval models typically relies on hard negative (HN) examples mined from large document corpora using methods such as BM25 or cross-encoders, which require full corpus access and expensive index construction. We propose generating synthetic hard negatives directly from a provided query and positive passage, using Large Language Models(LLMs). We fine-tune DistilBERT using synthetic negatives generated by four state-of-the-art LLMs ranging from 4B to 30B parameters (Qwen3, LLaMA3, Phi4) and evaluate performance across 10 BEIR benchmark datasets. Contrary to the prevailing assumption that stronger generative models yield better synthetic data, find that our generative pipeline consistently underperforms traditional corpus-based mining strategies (BM25 and Cross-Encoder). Furthermore, we observe that scaling the generator model does not monotonically improve retrieval performance and find that the 14B parameter model outperforms the 30B model and in some settings it is the worst performing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22508v1">Predicting LLM Correctness in Prosthodontics Using Metadata and Hallucination Signals</a></div>
    <div class="paper-meta">
      📅 2025-12-27
      | 💬 Accepted as a Short Paper at HEALTHINF2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly adopted in high-stakes domains such as healthcare and medical education, where the risk of generating factually incorrect (i.e., hallucinated) information is a major concern. While significant efforts have been made to detect and mitigate such hallucinations, predicting whether an LLM's response is correct remains a critical yet underexplored problem. This study investigates the feasibility of predicting correctness by analyzing a general-purpose model (GPT-4o) and a reasoning-centric model (OSS-120B) on a multiple-choice prosthodontics exam. We utilize metadata and hallucination signals across three distinct prompting strategies to build a correctness predictor for each (model, prompting) pair. Our findings demonstrate that this metadata-based approach can improve accuracy by up to +7.14% and achieve a precision of 83.12% over a baseline that assumes all answers are correct. We further show that while actual hallucination is a strong indicator of incorrectness, metadata signals alone are not reliable predictors of hallucination. Finally, we reveal that prompting strategies, despite not affecting overall accuracy, significantly alter the models' internal behaviors and the predictive utility of their metadata. These results present a promising direction for developing reliability signals in LLMs but also highlight that the methods explored in this paper are not yet robust enough for critical, high-stakes deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.03814v6">Recursive Training Loops in LLMs: How training data properties modulate distribution shift in generated data?</a></div>
    <div class="paper-meta">
      📅 2025-12-26
      | 💬 Accepted to EMNLP 2025 (Oral), Source Code: https://github.com/flowersteam/ce_llms
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used in the creation of online content, creating feedback loops as subsequent generations of models will be trained on this synthetic data. Such loops were shown to lead to distribution shifts - models misrepresenting the true underlying distributions of human data (also called model collapse). However, how human data properties affect such shifts remains poorly understood. In this paper, we provide the first empirical examination of the effect of such properties on the outcome of recursive training. We first confirm that using different human datasets leads to distribution shifts of different magnitudes. Through exhaustive manipulation of dataset properties combined with regression analyses, we then identify a set of properties predicting distribution shift magnitudes. Lexical diversity is found to amplify these shifts, while semantic diversity and data quality mitigate them. Furthermore, we find that these influences are highly modular: data scrapped from a given internet domain has little influence on the content generated for another domain. Finally, experiments on political bias reveal that human data properties affect whether the initial bias will be amplified or reduced. Overall, our results portray a novel view, where different parts of internet may undergo different types of distribution shift.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.12717v2">ToTRL: Unlock LLM Tree-of-Thoughts Reasoning Potential through Puzzles Solving</a></div>
    <div class="paper-meta">
      📅 2025-12-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate significant reasoning capabilities, particularly through long chain-of-thought (CoT) processes, which can be elicited by reinforcement learning (RL). However, prolonged CoT reasoning presents limitations, primarily verbose outputs due to excessive introspection. The reasoning process in these LLMs often appears to follow a trial-and-error methodology rather than a systematic, logical deduction. In contrast, tree-of-thoughts (ToT) offers a conceptually more advanced approach by modeling reasoning as an exploration within a tree structure. This reasoning structure facilitates the parallel generation and evaluation of multiple reasoning branches, allowing for the active identification, assessment, and pruning of unproductive paths. This process can potentially lead to improved performance and reduced token costs. Building upon the long CoT capability of LLMs, we introduce tree-of-thoughts RL (ToTRL), a novel on-policy RL framework with a rule-based reward. ToTRL is designed to guide LLMs in developing the parallel ToT strategy based on the sequential CoT strategy. Furthermore, we employ LLMs as players in a puzzle game during the ToTRL training process. Solving puzzle games inherently necessitates exploring interdependent choices and managing multiple constraints, which requires the construction and exploration of a thought tree, providing challenging tasks for cultivating the ToT reasoning capability. Our empirical evaluations demonstrate that our ToTQwen3-8B model, trained with our ToTRL, achieves significant improvement in performance and reasoning efficiency on complex reasoning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.00056v2">MISA: Memory-Efficient LLMs Optimization with Module-wise Importance Sampling</a></div>
    <div class="paper-meta">
      📅 2025-12-26
    </div>
    <details class="paper-abstract">
      The substantial memory demands of pre-training and fine-tuning large language models (LLMs) require memory-efficient optimization algorithms. One promising approach is layer-wise optimization, which treats each transformer block as a single layer and optimizes it sequentially, while freezing the other layers to save optimizer states and activations. Although effective, these methods ignore the varying importance of the modules within each layer, leading to suboptimal performance. Moreover, layer-wise sampling provides only limited memory savings, as at least one full layer must remain active during optimization. To overcome these limitations, we propose Module-wise Importance SAmpling (MISA), a novel method that divides each layer into smaller modules and assigns importance scores to each module. MISA uses a weighted random sampling mechanism to activate modules, provably reducing gradient variance compared to layer-wise sampling. Additionally, we establish an \(\mathcal{O}(1/\sqrt{K})\) convergence rate under non-convex and stochastic conditions, where $K$ is the total number of block updates, and provide a detailed memory analysis showcasing MISA's superiority over existing baseline methods. Experiments on diverse learning tasks validate the effectiveness of MISA. Source code is available at https://github.com/pkumelon/MISA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21933v1">Broken Words, Broken Performance: Effect of Tokenization on Performance of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-26
      | 💬 International Joint Conference on Natural Language Processing & Asia-Pacific Chapter of the Association for Computational Linguistics (IJCNLP-AACL 2025)
    </div>
    <details class="paper-abstract">
      Tokenization is the first step in training any Large Language Model (LLM), where the text is split into a sequence of tokens as per the model's fixed vocabulary. This tokenization in LLMs is different from the traditional tokenization in NLP where the text is split into a sequence of "natural" words. In LLMs, a natural word may also be broken into multiple tokens due to limited vocabulary size of the LLMs (e.g., Mistral's tokenizer splits "martial" into "mart" and "ial"). In this paper, we hypothesize that such breaking of natural words negatively impacts LLM performance on various NLP tasks. To quantify this effect, we propose a set of penalty functions that compute a tokenization penalty for a given text for a specific LLM, indicating how "bad" the tokenization is. We establish statistical significance of our hypothesis on multiple NLP tasks for a set of different LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21915v1">Exploring the Heterogeneity of Tabular Data: A Diversity-aware Data Generator via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-26
      | 💬 This manuscript has been submitted to IEEE Transactions on Knowledge and Data Engineering (TKDE) for peer review
    </div>
    <details class="paper-abstract">
      Tabular data generation has become increasingly essential for enabling robust machine learning applications, which require large-scale, high-quality data. Existing solutions leverage generative models to learn original data distributions. However, real-world data are naturally heterogeneous with diverse distributions, making it challenging to obtain a universally good model for diverse data generation. To address this limitation, we introduce Diversity-Aware Tabular data gEnerator (DATE), a framework that (i) prepares high-quality and distributionally distinct examples for in-context learning by effectively partitioning the original heterogeneous data into multiple diverse subsets; (ii) harnesses Large Language Models (LLMs) to explore the diversity of the partitioned distribution with decision tree reasoning as feedback, generating high-quality labeled data for each subset. However, the massive generated data inherently involves a trade-off between diversity and quality. To integrate this issue, existing solutions greedily select the validation-best data. However, we prove that the selection in heterogeneous settings does not possess the greedy-choice property, and design a Multi-Arm Bandit-based sampling algorithm that balances the diversity and quality of generated data. Extensive experiments on tabular classification and regression benchmarks demonstrate that DATE consistently outperforms state-of-the-art GAN-based and LLM-based methods. On average, DATE achieves a 23.75% reduction in error rate with just 100 generated data. Empirically, we demonstrate that data generated by DATE can improve the accuracy of Direct Preference Optimization (DPO) and enhance the reasoning capability of LLMs on the target data. Code is available at https://github.com/windblow32/DATE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.09990v3">X-Boundary: Establishing Exact Safety Boundary to Shield LLMs from Multi-Turn Jailbreaks without Compromising Usability</a></div>
    <div class="paper-meta">
      📅 2025-12-26
    </div>
    <details class="paper-abstract">
      Despite the rapid development of safety alignment techniques for LLMs, defending against multi-turn jailbreaks is still a challenging task. In this paper, we conduct a comprehensive comparison, revealing that some existing defense methods can improve the robustness of LLMs against multi-turn jailbreaks but compromise usability, i.e., reducing general capabilities or causing the over-refusal problem. From the perspective of mechanism interpretability of LLMs, we discover that these methods fail to establish a boundary that exactly distinguishes safe and harmful feature representations. Therefore, boundary-safe representations close to harmful representations are inevitably disrupted, leading to a decline in usability. To address this issue, we propose X-Boundary to push harmful representations away from boundary-safe representations and obtain an exact distinction boundary. In this way, harmful representations can be precisely erased without disrupting safe ones. Experimental results show that X-Boundary achieves state-of-the-art defense performance against multi-turn jailbreaks, while reducing the over-refusal rate by about 20% and maintaining nearly complete general capability. Furthermore, we theoretically prove and empirically verify that X-Boundary can accelerate the convergence process during training. Please see our code at: https://github.com/AI45Lab/X-Boundary.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21902v1">Explainable Statute Prediction via Attention-based Model and LLM Prompting</a></div>
    <div class="paper-meta">
      📅 2025-12-26
    </div>
    <details class="paper-abstract">
      In this paper, we explore the problem of automatic statute prediction where for a given case description, a subset of relevant statutes are to be predicted. Here, the term "statute" refers to a section, a sub-section, or an article of any specific Act. Addressing this problem would be useful in several applications such as AI-assistant for lawyers and legal question answering system. For better user acceptance of such Legal AI systems, we believe the predictions should also be accompanied by human understandable explanations. We propose two techniques for addressing this problem of statute prediction with explanations -- (i) AoS (Attention-over-Sentences) which uses attention over sentences in a case description to predict statutes relevant for it and (ii) LLMPrompt which prompts an LLM to predict as well as explain relevance of a certain statute. AoS uses smaller language models, specifically sentence transformers and is trained in a supervised manner whereas LLMPrompt uses larger language models in a zero-shot manner and explores both standard as well as Chain-of-Thought (CoT) prompting techniques. Both these models produce explanations for their predictions in human understandable forms. We compare statute prediction performance of both the proposed techniques with each other as well as with a set of competent baselines, across two popular datasets. Also, we evaluate the quality of the generated explanations through an automated counter-factual manner as well as through human evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21877v1">CricBench: A Multilingual Benchmark for Evaluating LLMs in Cricket Analytics</a></div>
    <div class="paper-meta">
      📅 2025-12-26
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Cricket is the second most popular sport globally, commanding a massive following of over 2.5 billion fans globally. Enthusiasts and analysts frequently seek advanced statistical insights, such as long-term historical performance trends or complex player comparisons, that are often unavailable through standard web searches. While Large Language Models (LLMs) have advanced significantly in Text-to-SQL tasks, their capability to handle the domain-specific nuances, complex schema variations, and multilingual requirements inherent to sports analytics remains under-explored. To investigate this potential capability gap, we present CricBench, a comprehensive benchmark suite for evaluating LLMs on specialized cricket data. To curate a "Gold Standard" dataset, we collaborate with domain experts in cricket and SQL to manually author complex queries, ensuring logical correctness. Recognizing linguistic diversity, we construct the benchmark in both English and Hindi, establishing a framework that is open for further extension to other regional languages. We evaluate six state-of-the-art models, including GPT-4o, Claude 3.7 Sonnet, and open-source models, using a strict evaluation protocol. Our results reveal that high performance on general benchmarks does not guarantee success in specialized domains. While the open-weights reasoning model DeepSeek R1 achieves state-of-the-art performance (50.6%), surpassing proprietary giants like Claude 3.7 Sonnet (47.7%) and GPT-4o (33.7%), it still exhibits a significant accuracy drop when moving from general benchmarks (BIRD) to CricBench. Furthermore, we observe that code-mixed Hindi queries frequently yield parity or higher accuracy compared to English, challenging the assumption that English is the optimal prompt language for specialized SQL tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.19855v4">Inducing Causal World Models in LLMs for Zero-Shot Physical Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-12-26
      | 💬 12 pages, 4 figures,
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), despite their advanced linguistic capabilities, fundamentally lack an intuitive understanding of physical dynamics, which limits their effectiveness in real-world scenarios that require causal reasoning. In this paper, we introduce Causal World Model Induction (CWMI), a novel framework designed to embed an explicit model of causal physics within an LLM. Our approach incorporates a dedicated Causal Physics Module (CPM) and a new training objective called Causal Intervention Loss, encouraging the model to learn cause-and-effect relationships from multimodal data. By training the model to predict the outcomes of hypothetical interventions instead of merely capturing statistical correlations, CWMI develops a robust internal representation of physical laws. Experimental results show that CWMI significantly outperforms state-of-the-art LLMs on zero-shot physical reasoning tasks, including the PIQA benchmark and our newly proposed PhysiCa-Bench dataset. These findings demonstrate that inducing a causal world model is a critical step toward more reliable and generalizable AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.05822v2">Video Event Reasoning and Prediction by Fusing World Knowledge from LLMs with Vision Foundation Models</a></div>
    <div class="paper-meta">
      📅 2025-12-26
      | 💬 22 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Current video understanding models excel at recognizing "what" is happening but fall short in high-level cognitive tasks like causal reasoning and future prediction, a limitation rooted in their lack of commonsense world knowledge. To bridge this cognitive gap, we propose a novel framework that synergistically fuses a powerful Vision Foundation Model (VFM) for deep visual perception with a Large Language Model (LLM) serving as a knowledge-driven reasoning core. Our key technical innovation is a sophisticated fusion module, inspired by the Q-Former architecture, which distills complex spatiotemporal and object-centric visual features into a concise, language-aligned representation. This enables the LLM to effectively ground its inferential processes in direct visual evidence. The model is trained via a two-stage strategy, beginning with large-scale alignment pre-training on video-text data, followed by targeted instruction fine-tuning on a curated dataset designed to elicit advanced reasoning and prediction skills. Extensive experiments demonstrate that our model achieves state-of-the-art performance on multiple challenging benchmarks. Notably, it exhibits remarkable zero-shot generalization to unseen reasoning tasks, and our in-depth ablation studies validate the critical contribution of each architectural component. This work pushes the boundary of machine perception from simple recognition towards genuine cognitive understanding, paving the way for more intelligent and capable AI systems in robotics, human-computer interaction, and beyond.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21852v1">A Comedy of Estimators: On KL Regularization in RL Training of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-26
    </div>
    <details class="paper-abstract">
      The reasoning performance of large language models (LLMs) can be substantially improved by training them with reinforcement learning (RL). The RL objective for LLM training involves a regularization term, which is the reverse Kullback-Leibler (KL) divergence between the trained policy and the reference policy. Since computing the KL divergence exactly is intractable, various estimators are used in practice to estimate it from on-policy samples. Despite its wide adoption, including in several open-source libraries, there is no systematic study analyzing the numerous ways of incorporating KL estimators in the objective and their effect on the downstream performance of RL-trained models. Recent works show that prevailing practices for incorporating KL regularization do not provide correct gradients for stated objectives, creating a discrepancy between the objective and its implementation. In this paper, we further analyze these practices and study the gradients of several estimators configurations, revealing how design choices shape gradient bias. We substantiate these findings with empirical observations by RL fine-tuning \texttt{Qwen2.5-7B}, \texttt{Llama-3.1-8B-Instruct} and \texttt{Qwen3-4B-Instruct-2507} with different configurations and evaluating their performance on both in- and out-of-distribution tasks. Through our analysis, we observe that, in on-policy settings: (1) estimator configurations with biased gradients can result in training instabilities; and (2) using estimator configurations resulting in unbiased gradients leads to better performance on in-domain as well as out-of-domain tasks. We also investigate the performance resulting from different KL configurations in off-policy settings and observe that KL regularization can help stabilize off-policy RL training resulting from asynchronous setups.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21849v1">HeartBench: Probing Core Dimensions of Anthropomorphic Intelligence in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-26
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have achieved remarkable success in cognitive and reasoning benchmarks, they exhibit a persistent deficit in anthropomorphic intelligence-the capacity to navigate complex social, emotional, and ethical nuances. This gap is particularly acute in the Chinese linguistic and cultural context, where a lack of specialized evaluation frameworks and high-quality socio-emotional data impedes progress. To address these limitations, we present HeartBench, a framework designed to evaluate the integrated emotional, cultural, and ethical dimensions of Chinese LLMs. Grounded in authentic psychological counseling scenarios and developed in collaboration with clinical experts, the benchmark is structured around a theory-driven taxonomy comprising five primary dimensions and 15 secondary capabilities. We implement a case-specific, rubric-based methodology that translates abstract human-like traits into granular, measurable criteria through a ``reasoning-before-scoring'' evaluation protocol. Our assessment of 13 state-of-the-art LLMs indicates a substantial performance ceiling: even leading models achieve only 60% of the expert-defined ideal score. Furthermore, analysis using a difficulty-stratified ``Hard Set'' reveals a significant performance decay in scenarios involving subtle emotional subtexts and complex ethical trade-offs. HeartBench establishes a standardized metric for anthropomorphic AI evaluation and provides a methodological blueprint for constructing high-quality, human-aligned training data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.04440v3">StepFun-Formalizer: Unlocking the Autoformalization Potential of LLMs through Knowledge-Reasoning Fusion</a></div>
    <div class="paper-meta">
      📅 2025-12-26
      | 💬 AAAI 2026 Oral. Extended version with full appendix, 25 pages, 17 figures
    </div>
    <details class="paper-abstract">
      Autoformalization aims to translate natural-language mathematical statements into a formal language. While LLMs have accelerated progress in this area, existing methods still suffer from low accuracy. We identify two key abilities for effective autoformalization: comprehensive mastery of formal-language domain knowledge, and reasoning capability of natural language problem understanding and informal-formal alignment. Without the former, a model cannot identify the correct formal objects; without the latter, it struggles to interpret real-world contexts and map them precisely into formal expressions. To address these gaps, we introduce ThinkingF, a data synthesis and training pipeline that improves both abilities. First, we construct two datasets: one by distilling and selecting large-scale examples rich in formal knowledge, and another by generating informal-to-formal reasoning trajectories guided by expert-designed templates. We then apply SFT and RLVR with these datasets to further fuse and refine the two abilities. The resulting 7B and 32B models exhibit both comprehensive formal knowledge and strong informal-to-formal reasoning. Notably, StepFun-Formalizer-32B achieves SOTA BEq@1 scores of 40.5% on FormalMATH-Lite and 26.7% on ProverBench, surpassing all prior general-purpose and specialized models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21835v1">LIME:Accelerating Collaborative Lossless LLM Inference on Memory-Constrained Edge Devices</a></div>
    <div class="paper-meta">
      📅 2025-12-26
      | 💬 15 pages, 18 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have emerged as a powerful foundation for intelligent reasoning and decision-making, demonstrating substantial impact across a wide range of domains and applications. However, their massive parameter scales and substantial resource demands pose critical challenges for efficient inference on edge devices. These devices are inherently constrained by limited computational power and memory capacity, while bandwidth bottlenecks at the network edge further restrict distributed deployment and real-time responsiveness. Although existing research has explored lightweight optimization techniques to mitigate memory limitations, such approaches often incur significant degradation in model accuracy and performance. To address these challenges, we propose LIME, a collaborative system that enables lossless inference for large models across multiple memory-constrained edge devices under limited network bandwidth. LIME employs an interleaved pipeline parallelism in conjunction with model offloading to dynamically balance computation and communication. Furthermore, a fine-grained offline allocation scheduler and online memory adaptation strategy are introduced to enhance the device's computing and storage resources while minimizing inference latency. Extensive experiments demonstrate that LIME, deployed on four heterogeneous Nvidia Jetson edge devices for LLaMA3.3-70B-Instruct model inference, achieves 1.7$\times$ and 3.7$\times$ speedups over state-of-the-art baselines under sporadic and bursty request patterns respectively, without compromising model accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16917v2">Generative Adversarial Reasoner: Enhancing LLM Reasoning with Adversarial Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-12-26
      | 💬 V2: Added links to the code-generation results and additional details in the appendix
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) with explicit reasoning capabilities excel at mathematical reasoning yet still commit process errors, such as incorrect calculations, brittle logic, and superficially plausible but invalid steps. In this paper, we introduce Generative Adversarial Reasoner, an on-policy joint training framework designed to enhance reasoning by co-evolving an LLM reasoner and an LLM-based discriminator through adversarial reinforcement learning. A compute-efficient review schedule partitions each reasoning chain into logically complete slices of comparable length, and the discriminator evaluates each slice's soundness with concise, structured justifications. Learning couples complementary signals: the LLM reasoner is rewarded for logically consistent steps that yield correct answers, while the discriminator earns rewards for correctly detecting errors or distinguishing traces in the reasoning process. This produces dense, well-calibrated, on-policy step-level rewards that supplement sparse exact-match signals, improving credit assignment, increasing sample efficiency, and enhancing overall reasoning quality of LLMs. Across various mathematical benchmarks, the method delivers consistent gains over strong baselines with standard RL post-training. Specifically, on AIME24, we improve DeepSeek-R1-Distill-Qwen-7B from 54.0 to 61.3 (+7.3) and DeepSeek-R1-Distill-Llama-8B from 43.7 to 53.7 (+10.0). The modular discriminator also enables flexible reward shaping for objectives such as teacher distillation, preference alignment, and mathematical proof-based reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21817v1">Method Decoration (DeMe): A Framework for LLM-Driven Adaptive Method Generation in Dynamic IoT Environments</a></div>
    <div class="paper-meta">
      📅 2025-12-26
    </div>
    <details class="paper-abstract">
      Intelligent IoT systems increasingly rely on large language models (LLMs) to generate task-execution methods for dynamic environments. However, existing approaches lack the ability to systematically produce new methods when facing previously unseen situations, and they often depend on fixed, device-specific logic that cannot adapt to changing environmental conditions.In this paper, we propose Method Decoration (DeMe), a general framework that modifies the method-generation path of an LLM using explicit decorations derived from hidden goals, accumulated learned methods, and environmental feedback. Unlike traditional rule augmentation, decorations in DeMe are not hardcoded; instead, they are extracted from universal behavioral principles, experience, and observed environmental differences. DeMe enables the agent to reshuffle the structure of its method path-through pre-decoration, post-decoration, intermediate-step modification, and step insertion-thereby producing context-aware, safety-aligned, and environment-adaptive methods. Experimental results show that method decoration allows IoT devices to derive ore appropriate methods when confronting unknown or faulty operating conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22396v1">HalluMat: Detecting Hallucinations in LLM-Generated Materials Science Content Through Multi-Stage Verification</a></div>
    <div class="paper-meta">
      📅 2025-12-26
    </div>
    <details class="paper-abstract">
      Artificial Intelligence (AI), particularly Large Language Models (LLMs), is transforming scientific discovery, enabling rapid knowledge generation and hypothesis formulation. However, a critical challenge is hallucination, where LLMs generate factually incorrect or misleading information, compromising research integrity. To address this, we introduce HalluMatData, a benchmark dataset for evaluating hallucination detection methods, factual consistency, and response robustness in AI-generated materials science content. Alongside this, we propose HalluMatDetector, a multi-stage hallucination detection framework that integrates intrinsic verification, multi-source retrieval, contradiction graph analysis, and metric-based assessment to detect and mitigate LLM hallucinations. Our findings reveal that hallucination levels vary significantly across materials science subdomains, with high-entropy queries exhibiting greater factual inconsistencies. By utilizing HalluMatDetector verification pipeline, we reduce hallucination rates by 30% compared to standard LLM outputs. Furthermore, we introduce the Paraphrased Hallucination Consistency Score (PHCS) to quantify inconsistencies in LLM responses across semantically equivalent queries, offering deeper insights into model reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22387v1">AI-Generated Code Is Not Reproducible (Yet): An Empirical Study of Dependency Gaps in LLM-Based Coding Agents</a></div>
    <div class="paper-meta">
      📅 2025-12-26
    </div>
    <details class="paper-abstract">
      The rise of Large Language Models (LLMs) as coding agents promises to accelerate software development, but their impact on generated code reproducibility remains largely unexplored. This paper presents an empirical study investigating whether LLM-generated code can be executed successfully in a clean environment with only OS packages and using only the dependencies that the model specifies. We evaluate three state-of-the-art LLM coding agents (Claude Code, OpenAI Codex, and Gemini) across 300 projects generated from 100 standardized prompts in Python, JavaScript, and Java. We introduce a three-layer dependency framework (distinguishing between claimed, working, and runtime dependencies) to quantify execution reproducibility. Our results show that only 68.3% of projects execute out-of-the-box, with substantial variation across languages (Python 89.2%, Java 44.0%). We also find a 13.5 times average expansion from declared to actual runtime dependencies, revealing significant hidden dependencies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22385v1">LLM-Guided Exemplar Selection for Few-Shot Wearable-Sensor Human Activity Recognition</a></div>
    <div class="paper-meta">
      📅 2025-12-26
      | 💬 This paper has been accepted for presentation at ABC 2026. The manuscript is under revision prior to camera-ready submission
    </div>
    <details class="paper-abstract">
      In this paper, we propose an LLM-Guided Exemplar Selection framework to address a key limitation in state-of-the-art Human Activity Recognition (HAR) methods: their reliance on large labeled datasets and purely geometric exemplar selection, which often fail to distinguish similar weara-ble sensor activities such as walking, walking upstairs, and walking downstairs. Our method incorporates semantic reasoning via an LLM-generated knowledge prior that captures feature importance, inter-class confusability, and exemplar budget multipliers, and uses it to guide exemplar scoring and selection. These priors are combined with margin-based validation cues, PageRank centrality, hubness penalization, and facility-location optimization to obtain a compact and informative set of exemplars. Evaluated on the UCI-HAR dataset under strict few-shot conditions, the framework achieves a macro F1-score of 88.78%, outperforming classical approaches such as random sampling, herding, and $k$-center. The results show that LLM-derived semantic priors, when integrated with structural and geometric cues, provide a stronger foundation for selecting representative sensor exemplars in few-shot wearable-sensor HAR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22364v1">Cost-Aware Text-to-SQL: An Empirical Study of Cloud Compute Costs for LLM-Generated Queries</a></div>
    <div class="paper-meta">
      📅 2025-12-26
    </div>
    <details class="paper-abstract">
      Text-to-SQL systems powered by Large Language Models (LLMs) achieve high accuracy on standard benchmarks, yet existing efficiency metrics such as the Valid Efficiency Score (VES) measure execution time rather than the consumption-based costs of cloud data warehouses. This paper presents the first systematic evaluation of cloud compute costs for LLM-generated SQL queries. We evaluate six state-of-the-art LLMs across 180 query executions on Google BigQuery using the StackOverflow dataset (230GB), measuring bytes processed, slot utilization, and estimated cost. Our analysis yields three key findings: (1) reasoning models process 44.5% fewer bytes than standard models while maintaining equivalent correctness (96.7%-100%); (2) execution time correlates weakly with query cost (r=0.16), indicating that speed optimization does not imply cost optimization; and (3) models exhibit up to 3.4x cost variance, with standard models producing outliers exceeding 36GB per query. We identify prevalent inefficiency patterns including missing partition filters and unnecessary full-table scans, and provide deployment guidelines for cost-sensitive enterprise environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.03688v3">A Comparison of DeepSeek and Other LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-25
      | 💬 30 pages, 7 figures, 9 tables
    </div>
    <details class="paper-abstract">
      Recently, DeepSeek has been the focus of attention in and beyond the AI community. An interesting problem is how DeepSeek compares to other large language models (LLMs). There are many tasks an LLM can do, and in this paper, we use the task of "predicting an outcome using a short text" for comparison. We consider two settings, an authorship classification setting and a citation classification setting. In the first one, the goal is to determine whether a short text is written by human or AI. In the second one, the goal is to classify a citation to one of four types using the textual content. For each experiment, we compare DeepSeek with $4$ popular LLMs: Claude, Gemini, GPT, and Llama. We find that, in terms of classification accuracy, DeepSeek outperforms Gemini, GPT, and Llama in most cases, but underperforms Claude. We also find that DeepSeek is comparably slower than others but with a low cost to use, while Claude is much more expensive than all the others. Finally, we find that in terms of similarity, the output of DeepSeek is most similar to those of Gemini and Claude (and among all $5$ LLMs, Claude and Gemini have the most similar outputs). In this paper, we also present a fully-labeled dataset collected by ourselves, and propose a recipe where we can use the LLMs and a recent data set, MADStat, to generate new data sets. The datasets in our paper can be used as benchmarks for future study on LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.10436v2">RefactorCoderQA: Benchmarking LLMs for Multi-Domain Coding Question Solutions in Cloud and Edge Deployment</a></div>
    <div class="paper-meta">
      📅 2025-12-25
      | 💬 12 pages, 5 figures, Submitted to IEEE Transactions on Services Computing
    </div>
    <details class="paper-abstract">
      To optimize the reasoning and problem-solving capabilities of Large Language Models (LLMs), we propose a novel cloud-edge collaborative architecture that enables a structured multi-agent prompting framework. This framework comprises three specialized components: GuideLLM, a lightweight model deployed at the edge to provide methodological guidance; SolverLLM, a more powerful model hosted in the cloud and responsible for generating code solutions; and JudgeLLM, an automated evaluator for assessing solution correctness and quality. To evaluate and demonstrate the effectiveness of this architecture in realistic settings, we introduce RefactorCoderQA, a comprehensive benchmark designed to evaluate and enhance the performance of LLMs across multi-domain coding tasks. Motivated by the limitations of existing benchmarks, RefactorCoderQA systematically covers multiple technical domains, including Software Engineering, Data Science, Machine Learning, and Natural Language Processing, using authentic coding challenges sourced from Stack Overflow. We propose RefactorCoder-MoE, a fine-tuned mixture-of-experts (MoE) code language model based on DeepSeek-Coder-7B-Instruct, adapted to the RefactorCoderQA benchmark using QLoRA for domain-specific coding question answering. Extensive experiments demonstrate that RefactorCoder-MoE achieves strong and competitive performance, significantly outperforming all evaluated open-source and commercial baselines, with an overall accuracy of 76.84%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18564v2">Vox Deorum: A Hybrid LLM Architecture for 4X / Grand Strategy Game AI -- Lessons from Civilization V</a></div>
    <div class="paper-meta">
      📅 2025-12-25
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Large Language Models' capacity to reason in natural language makes them uniquely promising for 4X and grand strategy games, enabling more natural human-AI gameplay interactions such as collaboration and negotiation. However, these games present unique challenges due to their complexity and long-horizon nature, while latency and cost factors may hinder LLMs' real-world deployment. Working on a classic 4X strategy game, Sid Meier's Civilization V with the Vox Populi mod, we introduce Vox Deorum, a hybrid LLM+X architecture. Our layered technical design empowers LLMs to handle macro-strategic reasoning, delegating tactical execution to subsystems (e.g., algorithmic AI or reinforcement learning AI in the future). We validate our approach through 2,327 complete games, comparing two open-source LLMs with a simple prompt against Vox Populi's enhanced AI. Results show that LLMs achieve competitive end-to-end gameplay while exhibiting play styles that diverge substantially from algorithmic AI and from each other. Our work establishes a viable architecture for integrating LLMs in commercial 4X games, opening new opportunities for game design and agentic AI research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20926v2">Uncovering Hierarchical Structure in LLM Embeddings with $δ$-Hyperbolicity, Ultrametricity, and Neighbor Joining</a></div>
    <div class="paper-meta">
      📅 2025-12-25
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has enabled significant strides in various fields. This paper introduces a novel approach to evaluate the effectiveness of LLM embeddings in the context of inherent geometric properties. We investigate the structural properties of these embeddings through three complementary metrics $δ$-hyperbolicity, Ultrametricity, and Neighbor Joining. $δ$-hyperbolicity, a measure derived from geometric group theory, quantifies how much a metric space deviates from being a tree-like structure. In contrast, ultrametricity characterizes strictly hierarchical structures where distances obey a strong triangle inequality. While Neighbor Joining quantifies how tree-like the distance relationships are, it does so specifically with respect to the tree reconstructed by the Neighbor Joining algorithm. By analyzing the embeddings generated by LLMs using these metrics, we uncover to what extent the embedding space reflects an underlying hierarchical or tree-like organization. Our findings reveal that LLM embeddings exhibit varying degrees of hyperbolicity and ultrametricity, which correlate with their performance in the underlying machine learning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2407.05347v2">A Queueing Theoretic Perspective on Low-Latency LLM Inference with Variable Token Length</a></div>
    <div class="paper-meta">
      📅 2025-12-25
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) propel the prosperity of interactive AI applications showcased by ChatGPT that demand timely response of inference services. However, LLM inference is computation intensive and memory intensive, and improper parameter configuration at LLM platforms may exacerbate the inference time. In this paper, we analyze the impact of LLM output token distribution on the inference queueing delay, where the max-token clipping and the batched inference are considered. By formulating an M/G/1 model, we observe that enforcing a maximum output token limit on a very small fraction of inference requests can significantly reduce the queueing delay, and our model facilitates the selection of the optimal limit. For the batch inference, we model the service process as a bulk queue in which the batch processing time is affected by the batch size and the maximum token size inside this batch jointly. The queueing delays of the batching of all buffered requests (dynamic batching), the batching of constant number of requests (fixed batching), and the batching without intra-batch waiting (elastic batching) are derived. Experimental results show that our mathematical models coincide with the event-driven simulations well.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.20166v3">PIMphony: Overcoming Bandwidth and Capacity Inefficiency in PIM-based Long-Context LLM Inference System</a></div>
    <div class="paper-meta">
      📅 2025-12-25
      | 💬 21 pages, 20 figures, Accepted to 2026 IEEE International Symposium on High-Performance Computer Architecture
    </div>
    <details class="paper-abstract">
      The expansion of long-context Large Language Models (LLMs) creates significant memory system challenges. While Processing-in-Memory (PIM) is a promising accelerator, we identify that it suffers from critical inefficiencies when scaled to long contexts: severe channel underutilization, performance-limiting I/O bottlenecks, and massive memory waste from static KV cache management. In this work, we propose PIMphony, a PIM orchestrator that systematically resolves these issues with three co-designed techniques. First, Token-Centric PIM Partitioning (TCP) ensures high channel utilization regardless of batch size. Second, Dynamic PIM Command Scheduling (DCS) mitigates the I/O bottleneck by overlapping data movement and computation. Finally, a Dynamic PIM Access (DPA) controller enables dynamic memory management to eliminate static memory waste. Implemented via an MLIR-based compiler and evaluated on a cycle-accurate simulator, PIMphony significantly improves throughput for long-context LLM inference (up to 72B parameters and 1M context length). Our evaluations show performance boosts of up to 11.3x on PIM-only systems and 8.4x on xPU+PIM systems, enabling more efficient deployment of LLMs in real-world long-context applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.02404v3">AnesSuite: A Comprehensive Benchmark and Dataset Suite for Anesthesiology Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-25
      | 💬 44 pages, 12 figures, 24 tables
    </div>
    <details class="paper-abstract">
      The application of large language models (LLMs) in the medical field has garnered significant attention, yet their reasoning capabilities in more specialized domains like anesthesiology remain underexplored. To bridge this gap, we introduce AnesSuite, the first comprehensive dataset suite specifically designed for anesthesiology reasoning in LLMs. The suite features AnesBench, an evaluation benchmark tailored to assess anesthesiology-related reasoning across three levels: factual retrieval (System 1), hybrid reasoning (System 1.x), and complex decision-making (System 2). Alongside this benchmark, the suite includes three training datasets that provide an infrastructure for continued pre-training (CPT), supervised fine-tuning (SFT), and reinforcement learning with verifiable rewards (RLVR). Leveraging this suite, we develop Morpheus, the first baseline model collection for anesthesiology reasoning. Despite undergoing limited training with SFT and group relative policy optimization (GRPO), Morpheus demonstrates substantial performance improvements, rivaling the performance of larger-scale models. Furthermore, through comprehensive evaluations and experiments, we analyze the key factors influencing anesthesiology reasoning performance, including model characteristics, training strategies and training data. Both AnesSuite and Morpheus will be open-sourced at https://github.com/MiliLab/AnesSuite.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21635v1">Heaven-Sent or Hell-Bent? Benchmarking the Intelligence and Defectiveness of LLM Hallucinations</a></div>
    <div class="paper-meta">
      📅 2025-12-25
      | 💬 Published as a conference paper at KDD 2026
    </div>
    <details class="paper-abstract">
      Hallucinations in large language models (LLMs) are commonly regarded as errors to be minimized. However, recent perspectives suggest that some hallucinations may encode creative or epistemically valuable content, a dimension that remains underquantified in current literature. Existing hallucination detection methods primarily focus on factual consistency, struggling to handle heterogeneous scientific tasks and balance creativity with accuracy. To address these challenges, we propose HIC-Bench, a novel evaluation framework that categorizes hallucinations into Intelligent Hallucinations (IH) and Defective Hallucinations (DH), enabling systematic investigation of their interplay in LLM creativity. HIC-Bench features three core characteristics: (1) Structured IH/DH Assessment. using a multi-dimensional metric matrix integrating Torrance Tests of Creative Thinking (TTCT) metrics (Originality, Feasibility, Value) with hallucination-specific dimensions (scientific plausibility, factual deviation); (2) Cross-Domain Applicability. spanning ten scientific domains with open-ended innovation tasks; and (3) Dynamic Prompt Optimization. leveraging the Dynamic Hallucination Prompt (DHP) to guide models toward creative and reliable outputs. The evaluation process employs multiple LLM judges, averaging scores to mitigate bias, with human annotators verifying IH/DH classifications. Experimental results reveal a nonlinear relationship between IH and DH, demonstrating that creativity and correctness can be jointly optimized. These insights position IH as a catalyst for creativity and reveal the ability of LLM hallucinations to drive scientific innovation.Additionally, the HIC-Bench offers a valuable platform for advancing research into the creative intelligence of LLM hallucinations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21595v1">LLM-I2I: Boost Your Small Item2Item Recommendation Model with Large Language Model</a></div>
    <div class="paper-meta">
      📅 2025-12-25
    </div>
    <details class="paper-abstract">
      Item-to-Item (I2I) recommendation models are widely used in real-world systems due to their scalability, real-time capabilities, and high recommendation quality. Research to enhance I2I performance focuses on two directions: 1) model-centric approaches, which adopt deeper architectures but risk increased computational costs and deployment complexity, and 2) data-centric methods, which refine training data without altering models, offering cost-effectiveness but struggling with data sparsity and noise. To address these challenges, we propose LLM-I2I, a data-centric framework leveraging Large Language Models (LLMs) to mitigate data quality issues. LLM-I2I includes (1) an LLM-based generator that synthesizes user-item interactions for long-tail items, alleviating data sparsity, and (2) an LLM-based discriminator that filters noisy interactions from real and synthetic data. The refined data is then fused to train I2I models. Evaluated on industry (AEDS) and academic (ARD) datasets, LLM-I2I consistently improves recommendation accuracy, particularly for long-tail items. Deployed on a large-scale cross-border e-commerce platform, it boosts recall number (RN) by 6.02% and gross merchandise value (GMV) by 1.22% over existing I2I models. This work highlights the potential of LLMs in enhancing data-centric recommendation systems without modifying model architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21582v1">LLM-Free Image Captioning Evaluation in Reference-Flexible Settings</a></div>
    <div class="paper-meta">
      📅 2025-12-25
      | 💬 Accepted for presentation at AAAI2026
    </div>
    <details class="paper-abstract">
      We focus on the automatic evaluation of image captions in both reference-based and reference-free settings. Existing metrics based on large language models (LLMs) favor their own generations; therefore, the neutrality is in question. Most LLM-free metrics do not suffer from such an issue, whereas they do not always demonstrate high performance. To address these issues, we propose Pearl, an LLM-free supervised metric for image captioning, which is applicable to both reference-based and reference-free settings. We introduce a novel mechanism that learns the representations of image--caption and caption--caption similarities. Furthermore, we construct a human-annotated dataset for image captioning metrics, that comprises approximately 333k human judgments collected from 2,360 annotators across over 75k images. Pearl outperformed other existing LLM-free metrics on the Composite, Flickr8K-Expert, Flickr8K-CF, Nebula, and FOIL datasets in both reference-based and reference-free settings. Our project page is available at https://pearl.kinsta.page/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21580v1">Gamayun's Path to Multilingual Mastery: Cost-Efficient Training of a 1.5B-Parameter LLM</a></div>
    <div class="paper-meta">
      📅 2025-12-25
    </div>
    <details class="paper-abstract">
      We present Gamayun, a 1.5B-parameter multilingual language model trained entirely from scratch on 2.5T tokens. Designed for efficiency and deployment in resource-constrained environments, Gamayun addresses the lack of research on small non-English-centric LLMs by adopting a novel two-stage pre-training strategy: balanced multilingual training for cross-lingual alignment, followed by high-quality English enrichment to transfer performance gains across languages. Our model supports 12 languages, with special focus on Russian. Despite a significantly smaller training budget than comparable models, Gamayun outperforms LLaMA3.2-1B (9T tokens) on all considered benchmarks, and surpasses Qwen2.5-1.5B (18T tokens) on a wide range of English and multilingual tasks. It matches or exceeds Qwen3 (36T tokens) on most tasks outside advanced STEM, achieving state-of-the-art results in Russian, including the MERA benchmark, among the models of comparable size (1-2B parameters).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21571v1">nncase: An End-to-End Compiler for Efficient LLM Deployment on Heterogeneous Storage Architectures</a></div>
    <div class="paper-meta">
      📅 2025-12-25
    </div>
    <details class="paper-abstract">
      The efficient deployment of large language models (LLMs) is hindered by memory architecture heterogeneity, where traditional compilers suffer from fragmented workflows and high adaptation costs. We present nncase, an open-source, end-to-end compilation framework designed to unify optimization across diverse targets. Central to nncase is an e-graph-based term rewriting engine that mitigates the phase ordering problem, enabling global exploration of computation and data movement strategies. The framework integrates three key modules: Auto Vectorize for adapting to heterogeneous computing units, Auto Distribution for searching parallel strategies with cost-aware communication optimization, and Auto Schedule for maximizing on-chip cache locality. Furthermore, a buffer-aware Codegen phase ensures efficient kernel instantiation. Evaluations show that nncase outperforms mainstream frameworks like MLC LLM and Intel IPEX on Qwen3 series models and achieves performance comparable to the hand-optimized llama.cpp on CPUs, demonstrating the viability of automated compilation for high-performance LLM deployment. The source code is available at https://github.com/kendryte/nncase.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21526v1">Selective LLM-Guided Regularization for Enhancing Recommendation Models</a></div>
    <div class="paper-meta">
      📅 2025-12-25
    </div>
    <details class="paper-abstract">
      Large language models provide rich semantic priors and strong reasoning capabilities, making them promising auxiliary signals for recommendation. However, prevailing approaches either deploy LLMs as standalone recommender or apply global knowledge distillation, both of which suffer from inherent drawbacks. Standalone LLM recommender are costly, biased, and unreliable across large regions of the user item space, while global distillation forces the downstream model to imitate LLM predictions even when such guidance is inaccurate. Meanwhile, recent studies show that LLMs excel particularly in re-ranking and challenging scenarios, rather than uniformly across all contexts.We introduce Selective LLM Guided Regularization, a model-agnostic and computation efficient framework that activates LLM based pairwise ranking supervision only when a trainable gating mechanism informing by user history length, item popularity, and model uncertainty predicts the LLM to be reliable. All LLM scoring is performed offline, transferring knowledge without increasing inference cost. Experiments across multiple datasets show that this selective strategy consistently improves overall accuracy and yields substantial gains in cold start and long tail regimes, outperforming global distillation baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21008v2">GateBreaker: Gate-Guided Attacks on Mixture-of-Expert LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-25
      | 💬 Accepted by USENIX Security'26
    </div>
    <details class="paper-abstract">
      Mixture-of-Experts (MoE) architectures have advanced the scaling of Large Language Models (LLMs) by activating only a sparse subset of parameters per input, enabling state-of-the-art performance with reduced computational cost. As these models are increasingly deployed in critical domains, understanding and strengthening their alignment mechanisms is essential to prevent harmful outputs. However, existing LLM safety research has focused almost exclusively on dense architectures, leaving the unique safety properties of MoEs largely unexamined. The modular, sparsely-activated design of MoEs suggests that safety mechanisms may operate differently than in dense models, raising questions about their robustness. In this paper, we present GateBreaker, the first training-free, lightweight, and architecture-agnostic attack framework that compromises the safety alignment of modern MoE LLMs at inference time. GateBreaker operates in three stages: (i) gate-level profiling, which identifies safety experts disproportionately routed on harmful inputs, (ii) expert-level localization, which localizes the safety structure within safety experts, and (iii) targeted safety removal, which disables the identified safety structure to compromise the safety alignment. Our study shows that MoE safety concentrates within a small subset of neurons coordinated by sparse routing. Selective disabling of these neurons, approximately 3% of neurons in the targeted expert layers, significantly increases the averaged attack success rate (ASR) from 7.4% to 64.9% against the eight latest aligned MoE LLMs with limited utility degradation. These safety neurons transfer across models within the same family, raising ASR from 17.9% to 67.7% with one-shot transfer attack. Furthermore, GateBreaker generalizes to five MoE vision language models (VLMs) with 60.9% ASR on unsafe image inputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20957v2">One Tool Is Enough: Reinforcement Learning for Repository-Level LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-12-25
    </div>
    <details class="paper-abstract">
      Locating the files and functions requiring modification in large open-source software (OSS) repositories is challenging due to their scale and structural complexity. Existing large language model (LLM)-based methods typically treat this as a repository-level retrieval task and rely on multiple auxiliary tools, which overlook code execution logic and complicate model control. We propose RepoNavigator, an LLM agent equipped with a single execution-aware tool-jumping to the definition of an invoked symbol. This unified design reflects the actual flow of code execution while simplifying tool manipulation. RepoNavigator is trained end-to-end via Reinforcement Learning (RL) directly from a pretrained model, without any closed-source distillation. Experiments demonstrate that RL-trained RepoNavigator achieves state-of-the-art performance, with the 7B model outperforming 14B baselines, the 14B model surpassing 32B competitors, and even the 32B model exceeding closed-source models such as Claude-3.7. These results confirm that integrating a single, structurally grounded tool with RL training provides an efficient and scalable solution for repository-level issue localization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21309v2">A Plan Reuse Mechanism for LLM-Driven Agent</a></div>
    <div class="paper-meta">
      📅 2025-12-25
      | 💬 This paper is an English version of A Plan Reuse Mechanism for LLM-Driven Agent published in 2024 in the Journal of Computer Research and Development
    </div>
    <details class="paper-abstract">
      Integrating large language models (LLMs) into personal assistants, like Xiao Ai and Blue Heart V, effectively enhances their ability to interact with humans, solve complex tasks, and manage IoT devices. Such assistants are also termed LLM-driven agents. Upon receiving user requests, the LLM-driven agent generates plans using an LLM, executes these plans through various tools, and then returns the response to the user. During this process, the latency for generating a plan with an LLM can reach tens of seconds, significantly degrading user experience. Real-world dataset analysis shows that about 30% of the requests received by LLM-driven agents are identical or similar, which allows the reuse of previously generated plans to reduce latency. However, it is difficult to accurately define the similarity between the request texts received by the LLM-driven agent through directly evaluating the original request texts. Moreover, the diverse expressions of natural language and the unstructured format of plan texts make implementing plan reuse challenging. To address these issues, we present and implement a plan reuse mechanism for LLM-driven agents called AgentReuse. AgentReuse leverages the similarities and differences among requests' semantics and uses intent classification to evaluate the similarities between requests and enable the reuse of plans. Experimental results based on a real-world dataset demonstrate that AgentReuse achieves a 93% effective plan reuse rate, an F1 score of 0.9718, and an accuracy of 0.9459 in evaluating request similarities, reducing latency by 93.12% compared with baselines without using the reuse mechanism.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19920v2">Mitigating LLM Hallucination via Behaviorally Calibrated Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-12-25
    </div>
    <details class="paper-abstract">
      LLM deployment in critical domains is currently impeded by persistent hallucinations--generating plausible but factually incorrect assertions. While scaling laws drove significant improvements in general capabilities, theoretical frameworks suggest hallucination is not merely stochastic error but a predictable statistical consequence of training objectives prioritizing mimicking data distribution over epistemic honesty. Standard RLVR paradigms, utilizing binary reward signals, inadvertently incentivize models as good test-takers rather than honest communicators, encouraging guessing whenever correctness probability exceeds zero. This paper presents an exhaustive investigation into behavioral calibration, which incentivizes models to stochastically admit uncertainty by abstaining when not confident, aligning model behavior with accuracy. Synthesizing recent advances, we propose and evaluate training interventions optimizing strictly proper scoring rules for models to output a calibrated probability of correctness. Our methods enable models to either abstain from producing a complete response or flag individual claims where uncertainty remains. Utilizing Qwen3-4B-Instruct, empirical analysis reveals behavior-calibrated reinforcement learning allows smaller models to surpass frontier models in uncertainty quantification--a transferable meta-skill decouplable from raw predictive accuracy. Trained on math reasoning tasks, our model's log-scale Accuracy-to-Hallucination Ratio gain (0.806) exceeds GPT-5's (0.207) in a challenging in-domain evaluation (BeyondAIME). Moreover, in cross-domain factual QA (SimpleQA), our 4B LLM achieves zero-shot calibration error on par with frontier models including Grok-4 and Gemini-2.5-Pro, even though its factual accuracy is much lower.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21506v1">MotionTeller: Multi-modal Integration of Wearable Time-Series with LLMs for Health and Behavioral Understanding</a></div>
    <div class="paper-meta">
      📅 2025-12-25
    </div>
    <details class="paper-abstract">
      As wearable sensing becomes increasingly pervasive, a key challenge remains: how can we generate natural language summaries from raw physiological signals such as actigraphy - minute-level movement data collected via accelerometers? In this work, we introduce MotionTeller, a generative framework that natively integrates minute-level wearable activity data with large language models (LLMs). MotionTeller combines a pretrained actigraphy encoder with a lightweight projection module that maps behavioral embeddings into the token space of a frozen decoder-only LLM, enabling free-text, autoregressive generation of daily behavioral summaries. We construct a novel dataset of 54383 (actigraphy, text) pairs derived from real-world NHANES recordings, and train the model using cross-entropy loss with supervision only on the language tokens. MotionTeller achieves high semantic fidelity (BERTScore-F1 = 0.924) and lexical accuracy (ROUGE-1 = 0.722), outperforming prompt-based baselines by 7 percent in ROUGE-1. The average training loss converges to 0.38 by epoch 15, indicating stable optimization. Qualitative analysis confirms that MotionTeller captures circadian structure and behavioral transitions, while PCA plots reveal enhanced cluster alignment in embedding space post-training. Together, these results position MotionTeller as a scalable, interpretable system for transforming wearable sensor data into fluent, human-centered descriptions, introducing new pathways for behavioral monitoring, clinical review, and personalized health interventions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.06303v3">Reward Is Enough: LLMs Are In-Context Reinforcement Learners</a></div>
    <div class="paper-meta">
      📅 2025-12-25
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) is a framework for solving sequential decision-making problems. In this work, we demonstrate that, surprisingly, RL emerges during the inference time of large language models (LLMs), a phenomenon we term in-context RL (ICRL). To reveal this capability, we introduce a simple multi-round prompting framework, we call ICRL prompting, for inference-time self-improvement. The goal of ICRL prompting is to guide LLMs to perform reinforcement learning during inference for self-improvement on a given task. After each response, the model receives numerical scalar feedback, denoted as a reward. In the next round, we prompt the LLM again together with a context that concatenates all prior responses and their associated rewards. We consistently observe that response quality improves as the context grows. In other words, the LLM can optimize scalar reward signals during inference, exhibiting behavior analogous to reinforcement learning. We evaluate ICRL prompting on Game of 24, creative writing, ScienceWorld, and Olympiad-level math competitions (AIME and HMMT), demonstrating significant improvements over baselines such as Self-Refine and Reflexion. Notably, even when the reward signals are generated by the same LLM, ICRL prompting still improves performance, highlighting a promising new paradigm for test-time scaling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23738v1">Enforcing Temporal Constraints for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-12-25
    </div>
    <details class="paper-abstract">
      LLM-based agents are deployed in safety-critical applications, yet current guardrail systems fail to prevent violations of temporal safety policies, requirements that govern the ordering and sequencing of agent actions. For instance, agents may access sensitive data before authenticating users or process refunds to unauthorized payment methods, violations that require reasoning about sequences of action rather than an individual action. Existing guardrails rely on imprecise natural language instructions or post-hoc monitoring, and provide no formal guarantees that agents will satisfy temporal constraints. We present Agent-C, a novel framework that provides run-time guarantees ensuring LLM agents adhere to formal temporal safety properties. Agent-C introduces a domain-specific language for expressing temporal properties (e.g., authenticate before accessing data), translates specifications to first-order logic, and uses SMT solving to detect non-compliant agent actions during token generation. When the LLM attempts to generate a non-compliant tool call, Agent-C leverages constrained generation techniques to ensure that every action generated by the LLM complies with the specification, and to generate a compliant alternative to a non-compliant agent action. We evaluate Agent-C across two real-world applications: retail customer service and airline ticket reservation system, and multiple language models (open and closed-source). Our results demonstrate that Agent-C achieves perfect safety (100% conformance, 0% harm), while improving task utility compared to state-of-the-art guardrails and unrestricted agents. On SoTA closed-source models, Agent-C improves conformance (77.4% to 100% for Claude Sonnet 4.5 and 83.7% to 100% for GPT-5), while simultaneously increasing utility (71.8% to 75.2% and 66.1% to 70.6%, respectively), representing a new SoTA frontier for reliable agentic reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21250v1">CoTDeceptor:Adversarial Code Obfuscation Against CoT-Enhanced LLM Code Agents</a></div>
    <div class="paper-meta">
      📅 2025-12-24
    </div>
    <details class="paper-abstract">
      LLM-based code agents(e.g., ChatGPT Codex) are increasingly deployed as detector for code review and security auditing tasks. Although CoT-enhanced LLM vulnerability detectors are believed to provide improved robustness against obfuscated malicious code, we find that their reasoning chains and semantic abstraction processes exhibit exploitable systematic weaknesses.This allows attackers to covertly embed malicious logic, bypass code review, and propagate backdoored components throughout real-world software supply chains.To investigate this issue, we present CoTDeceptor, the first adversarial code obfuscation framework targeting CoT-enhanced LLM detectors. CoTDeceptor autonomously constructs evolving, hard-to-reverse multi-stage obfuscation strategy chains that effectively disrupt CoT-driven detection logic.We obtained malicious code provided by security enterprise, experimental results demonstrate that CoTDeceptor achieves stable and transferable evasion performance against state-of-the-art LLMs and vulnerability detection agents. CoTDeceptor bypasses 14 out of 15 vulnerability categories, compared to only 2 bypassed by prior methods. Our findings highlight potential risks in real-world software supply chains and underscore the need for more robust and interpretable LLM-powered security analysis systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21236v1">Casting a SPELL: Sentence Pairing Exploration for LLM Limitation-breaking</a></div>
    <div class="paper-meta">
      📅 2025-12-24
      | 💬 Accepted to FSE 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have revolutionized software development through AI-assisted coding tools, enabling developers with limited programming expertise to create sophisticated applications. However, this accessibility extends to malicious actors who may exploit these powerful tools to generate harmful software. Existing jailbreaking research primarily focuses on general attack scenarios against LLMs, with limited exploration of malicious code generation as a jailbreak target. To address this gap, we propose SPELL, a comprehensive testing framework specifically designed to evaluate the weakness of security alignment in malicious code generation. Our framework employs a time-division selection strategy that systematically constructs jailbreaking prompts by intelligently combining sentences from a prior knowledge dataset, balancing exploration of novel attack patterns with exploitation of successful techniques. Extensive evaluation across three advanced code models (GPT-4.1, Claude-3.5, and Qwen2.5-Coder) demonstrates SPELL's effectiveness, achieving attack success rates of 83.75%, 19.38%, and 68.12% respectively across eight malicious code categories. The generated prompts successfully produce malicious code in real-world AI development tools such as Cursor, with outputs confirmed as malicious by state-of-the-art detection systems at rates exceeding 73%. These findings reveal significant security gaps in current LLM implementations and provide valuable insights for improving AI safety alignment in code generation applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.00675v3">Rethinking Memory in LLM based Agents: Representations, Operations, and Emerging Topics</a></div>
    <div class="paper-meta">
      📅 2025-12-24
    </div>
    <details class="paper-abstract">
      Memory is fundamental to large language model (LLM)-based agents, but existing surveys emphasize application-level use (e.g., personalized dialogue), while overlooking the atomic operations governing memory dynamics. This work categorizes memory into parametric (implicit in model weights) and contextual (explicit external data, structured/unstructured) forms, and defines six core operations: Consolidation, Updating, Indexing, Forgetting, Retrieval, and Condensation. Mapping these dimensions reveals four key research topics: long-term, long-context, parametric modification, and multi-source memory. The taxonomy provides a structured view of memory-related research, benchmarks, and tools, clarifying functional interactions in LLM-based agents and guiding future advancements. The datasets, papers, and tools are publicly available at https://github.com/Elvin-Yiming-Du/Survey_Memory_in_AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16378v2">Hearing to Translate: The Effectiveness of Speech Modality Integration into LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-24
      | 💬 Project available at https://github.com/sarapapi/hearing2translate
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) expand beyond text, integrating speech as a native modality has given rise to SpeechLLMs, which aim to translate spoken language directly, thereby bypassing traditional transcription-based pipelines. Whether this integration improves speech-to-text translation quality over established cascaded architectures, however, remains an open question. We present Hearing to Translate, the first comprehensive test suite rigorously benchmarking 5 state-of-the-art SpeechLLMs against 16 strong direct and cascade systems that couple leading speech foundation models (SFM), with multilingual LLMs. Our analysis spans 16 benchmarks, 13 language pairs, and 9 challenging conditions, including disfluent, noisy, and long-form speech. Across this extensive evaluation, we find that cascaded systems remain the most reliable overall, while current SpeechLLMs only match cascades in selected settings and SFMs lag behind both, highlighting that integrating an LLM, either within the model or in a pipeline, is essential for high-quality speech translation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12867v2">Bootstrapping LLMs via Preference-Based Policy Optimization</a></div>
    <div class="paper-meta">
      📅 2025-12-24
    </div>
    <details class="paper-abstract">
      Bootstrapping large language models (LLMs) through preference-based policy optimization offers a promising direction for aligning model behavior with human preferences without relying on extensive manual annotations. In this work, we propose a novel preference-based policy optimization (PbPO) framework that formulates the learning process as a min-max game between the main policy and a reward model (RM). The RM is constrained within a confidence set derived from preference data to ensure reliable exploitation. Our iterative online algorithm actively collects preference data through guided exploration of the evolving policy, enabling continual self-improvement of both the policy and the RM. We provide theoretical guarantees for our method, establishing high-probability regret bounds for both settings with sequence-level RM and token-level RM, demonstrating its effectiveness in bootstrapping LLMs. Extensive experiments on five benchmarks show that our approach consistently outperforms existing state-of-the-art preference optimization techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.19027v2">SESR-Eval: Dataset for Evaluating LLMs in the Title-Abstract Screening of Systematic Reviews</a></div>
    <div class="paper-meta">
      📅 2025-12-24
      | 💬 An updated post-print on the paper published in the Proceedings of the 19th ACM/IEEE International Symposium on Empirical Software Engineering and Measurement (ESEM '25). 12 pages (10 + 2 pages for references)
    </div>
    <details class="paper-abstract">
      Background: The use of large language models (LLMs) in the title-abstract screening process of systematic reviews (SRs) has shown promising results, but suffers from limited performance evaluation. Aims: Create a benchmark dataset to evaluate the performance of LLMs in the title-abstract screening process of SRs. Provide evidence whether using LLMs in title-abstract screening in software engineering is advisable. Method: We start with 169 SR research artifacts and find 24 of those to be suitable for inclusion in the dataset. Using the dataset we benchmark title-abstract screening using 9 LLMs. Results: We present the SESR-Eval (Software Engineering Systematic Review Evaluation) dataset containing 34,528 labeled primary studies, sourced from 24 secondary studies published in software engineering (SE) journals. Most LLMs performed similarly and the differences in screening accuracy between secondary studies are greater than differences between LLMs. The cost of using an LLM is relatively low - less than $40 per secondary study even for the most expensive model. Conclusions: Our benchmark enables monitoring AI performance in the screening task of SRs in software engineering. At present, LLMs are not yet recommended for automating the title-abstract screening process, since accuracy varies widely across secondary studies, and no LLM managed a high recall with reasonable precision. In future, we plan to investigate factors that influence LLM screening performance between studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21144v1">Encrypted Traffic Detection in Resource Constrained IoT Networks: A Diffusion Model and LLM Integrated Framework</a></div>
    <div class="paper-meta">
      📅 2025-12-24
      | 💬 This paper is accepted by IEEE Transactions on Network Science and Engineering
    </div>
    <details class="paper-abstract">
      The proliferation of Internet-of-things (IoT) infrastructures and the widespread adoption of traffic encryption present significant challenges, particularly in environments characterized by dynamic traffic patterns, constrained computational capabilities, and strict latency constraints. In this paper, we propose DMLITE, a diffusion model and large language model (LLM) integrated traffic embedding framework for network traffic detection within resource-limited IoT environments. The DMLITE overcomes these challenges through a tri-phase architecture including traffic visual preprocessing, diffusion-based multi-level feature extraction, and LLM-guided feature optimization. Specifically, the framework utilizes self-supervised diffusion models to capture both fine-grained and abstract patterns in encrypted traffic through multi-level feature fusion and contrastive learning with representative sample selection, thus enabling rapid adaptation to new traffic patterns with minimal labeled data. Furthermore, DMLITE incorporates LLMs to dynamically adjust particle swarm optimization parameters for intelligent feature selection by implementing a dual objective function that minimizes both classification error and variance across data distributions. Comprehensive experimental validation on benchmark datasets confirms the effectiveness of DMLITE, achieving classification accuracies of 98.87\%, 92.61\%, and 99.83\% on USTC-TFC, ISCX-VPN, and Edge-IIoTset datasets, respectively. This improves classification accuracy by an average of 3.7\% and reduces training time by an average of 41.9\% compared to the representative deep learning model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21138v1">Emotion Diffusion in Real and Simulated Social Graphs: Structural Limits of LLM-Based Social Simulation</a></div>
    <div class="paper-meta">
      📅 2025-12-24
    </div>
    <details class="paper-abstract">
      Understanding how emotions diffuse through social networks is central to computational social science. Recently, large language models (LLMs) have been increasingly used to simulate social media interactions, raising the question of whether LLM-generated data can realistically reproduce emotion diffusion patterns observed in real online communities. In this study, we conduct a systematic comparison between emotion diffusion in real-world social graphs and in LLM-simulated interaction networks. We construct diffusion graphs from Reddit discussion data and compare them with synthetic social graphs generated through LLM-driven conversational simulations. Emotion states are inferred using established sentiment analysis pipelines, and both real and simulated graphs are analyzed from structural, behavioral, and predictive perspectives. Our results reveal substantial structural and dynamic discrepancies between real and simulated diffusion processes. Real-world emotion diffusion exhibits dense connectivity, repeated interactions, sentiment shifts, and emergent community structures, whereas LLM-simulated graphs largely consist of isolated linear chains with monotonic emotional trajectories. These structural limitations significantly affect downstream tasks such as graph-based emotion prediction, leading to reduced emotional diversity and class imbalance in simulated settings. Our findings highlight current limitations of LLM-based social simulation in capturing the interactive complexity and emotional heterogeneity of real social networks. This work provides empirical evidence for the cautious use of LLM-generated data in social science research and suggests directions for improving future simulation frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21127v1">A Real-World Evaluation of LLM Medication Safety Reviews in NHS Primary Care</a></div>
    <div class="paper-meta">
      📅 2025-12-24
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often match or exceed clinician-level performance on medical benchmarks, yet very few are evaluated on real clinical data or examined beyond headline metrics. We present, to our knowledge, the first evaluation of an LLM-based medication safety review system on real NHS primary care data, with detailed characterisation of key failure behaviours across varying levels of clinical complexity. In a retrospective study using a population-scale EHR spanning 2,125,549 adults in NHS Cheshire and Merseyside, we strategically sampled patients to capture a broad range of clinical complexity and medication safety risk, yielding 277 patients after data-quality exclusions. An expert clinician reviewed these patients and graded system-identified issues and proposed interventions. Our primary LLM system showed strong performance in recognising when a clinical issue is present (sensitivity 100\% [95\% CI 98.2--100], specificity 83.1\% [95\% CI 72.7--90.1]), yet correctly identified all issues and interventions in only 46.9\% [95\% CI 41.1--52.8] of patients. Failure analysis reveals that, in this setting, the dominant failure mechanism is contextual reasoning rather than missing medication knowledge, with five primary patterns: overconfidence in uncertainty, applying standard guidelines without adjusting for patient context, misunderstanding how healthcare is delivered in practice, factual errors, and process blindness. These patterns persisted across patient complexity and demographic strata, and across a range of state-of-the-art models and configurations. We provide 45 detailed vignettes that comprehensively cover all identified failure cases. This work highlights shortcomings that must be addressed before LLM-based clinical AI can be safely deployed. It also begs larger-scale, prospective evaluations and deeper study of LLM behaviours in clinical contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.11822v2">Beyond Consensus: Mitigating the Agreeableness Bias in LLM Judge Evaluations</a></div>
    <div class="paper-meta">
      📅 2025-12-24
    </div>
    <details class="paper-abstract">
      New Large Language Models (LLMs) become available every few weeks, and modern application developers confronted with the unenviable task of having to decide if they should switch to a new model. While human evaluation remains the gold standard, it is costly and unscalable. The state-of-the-art approach is to use LLMs as evaluators ( LLM-as-a-judge), but this suffers from a critical flaw: LLMs exhibit a strong positive bias. We provide empirical evidence showing that while LLMs can identify valid outputs with high accuracy (i.e., True Positive Rate 96%), they are remarkably poor at identifying invalid ones (i.e., True Negative Rate <25%). This systematic bias, coupled with class imbalance, often leads to inflated reliability scores. While ensemble-based methods like majority voting can help, we show that they are not good enough. We introduce an optimal minority-veto strategy that is resilient to missing data and mitigates this bias to a large extent. For scenarios requiring even higher precision, we propose a novel regression-based framework that directly models the validator bias using a small set of human-annotated ground truth data. On a challenging code feedback task over 366 high-school Python programs, our regression approach reduces the maximum absolute error to just 1.2%, achieving a 2x improvement over the best-performing ensemble of 14 state-of-the-art LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21106v1">Semantic Refinement with LLMs for Graph Representations</a></div>
    <div class="paper-meta">
      📅 2025-12-24
    </div>
    <details class="paper-abstract">
      Graph-structured data exhibit substantial heterogeneity in where their predictive signals originate: in some domains, node-level semantics dominate, while in others, structural patterns play a central role. This structure-semantics heterogeneity implies that no graph learning model with a fixed inductive bias can generalize optimally across diverse graph domains. However, most existing methods address this challenge from the model side by incrementally injecting new inductive biases, which remains fundamentally limited given the open-ended diversity of real-world graphs. In this work, we take a data-centric perspective and treat node semantics as a task-adaptive variable. We propose a Data-Adaptive Semantic Refinement framework DAS for graph representation learning, which couples a fixed graph neural network (GNN) and a large language model (LLM) in a closed feedback loop. The GNN provides implicit supervisory signals to guide the semantic refinement of LLM, and the refined semantics are fed back to update the same graph learner. We evaluate our approach on both text-rich and text-free graphs. Results show consistent improvements on structure-dominated graphs while remaining competitive on semantics-rich graphs, demonstrating the effectiveness of data-centric semantic adaptation under structure-semantics heterogeneity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21080v1">LLM Personas as a Substitute for Field Experiments in Method Benchmarking</a></div>
    <div class="paper-meta">
      📅 2025-12-24
    </div>
    <details class="paper-abstract">
      Field experiments (A/B tests) are often the most credible benchmark for methods in societal systems, but their cost and latency create a major bottleneck for iterative method development. LLM-based persona simulation offers a cheap synthetic alternative, yet it is unclear whether replacing humans with personas preserves the benchmark interface that adaptive methods optimize against. We prove an if-and-only-if characterization: when (i) methods observe only the aggregate outcome (aggregate-only observation) and (ii) evaluation depends only on the submitted artifact and not on the algorithm's identity or provenance (algorithm-blind evaluation), swapping humans for personas is just panel change from the method's point of view, indistinguishable from changing the evaluation population (e.g., New York to Jakarta). Furthermore, we move from validity to usefulness: we define an information-theoretic discriminability of the induced aggregate channel and show that making persona benchmarking as decision-relevant as a field experiment is fundamentally a sample-size question, yielding explicit bounds on the number of independent persona evaluations required to reliably distinguish meaningfully different methods at a chosen resolution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21041v1">When LLMs fall short in Deductive Coding: Model Comparison and Human AI Collaboration Workflow Design</a></div>
    <div class="paper-meta">
      📅 2025-12-24
      | 💬 24 pages (8 pages for Appendix), 4 figures, for Learning Analytics & Knowledge Conference to be held in 2026, Norway (LAK26)
    </div>
    <details class="paper-abstract">
      With generative artificial intelligence driving the growth of dialogic data in education, automated coding is a promising direction for learning analytics to improve efficiency. This surge highlights the need to understand the nuances of student-AI interactions, especially those rare yet crucial. However, automated coding may struggle to capture these rare codes due to imbalanced data, while human coding remains time-consuming and labour-intensive. The current study examined the potential of large language models (LLMs) to approximate or replace humans in deductive, theory-driven coding, while also exploring how human-AI collaboration might support such coding tasks at scale. We compared the coding performance of small transformer classifiers (e.g., BERT) and LLMs in two datasets, with particular attention to imbalanced head-tail distributions in dialogue codes. Our results showed that LLMs did not outperform BERT-based models and exhibited systematic errors and biases in deductive coding tasks. We designed and evaluated a human-AI collaborative workflow that improved coding efficiency while maintaining coding reliability. Our findings reveal both the limitations of LLMs -- especially their difficulties with semantic similarity and theoretical interpretations and the indispensable role of human judgment -- while demonstrating the practical promise of human-AI collaborative workflows for coding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21028v1">Artificial or Just Artful? Do LLMs Bend the Rules in Programming?</a></div>
    <div class="paper-meta">
      📅 2025-12-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely used for automated code generation, yet their apparent successes often mask a tension between pretraining objectives and alignment choices. While pretraining encourages models to exploit all available signals to maximize success, alignment, whether through fine-tuning or prompting, may restrict their use. This conflict is especially salient in agentic AI settings, for instance when an agent has access to unit tests that, although intended for validation, act as strong contextual signals that can be leveraged regardless of explicit prohibitions. In this paper, we investigate how LLMs adapt their code generation strategies when exposed to test cases under different prompting conditions. Using the BigCodeBench (Hard) dataset, we design five prompting conditions that manipulate test visibility and impose explicit or implicit restrictions on their use. We evaluate five LLMs (four open-source and one closed-source) across correctness, code similarity, program size, and code churn, and analyze cross-model consistency to identify recurring adaptation strategies. Our results show that test visibility dramatically alters performance, correctness nearly doubles for some models, while explicit restrictions or partial exposure only partially mitigate this effect. Beyond raw performance, we identify four recurring adaptation strategies, with test-driven refinement emerging as the most frequent. These results highlight how LLMs adapt their behavior when exposed to contextual signals that conflict with explicit instructions, providing useful insight into how models reconcile pretraining objectives with alignment constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.12284v3">V-Rex: Real-Time Streaming Video LLM Acceleration via Dynamic KV Cache Retrieval</a></div>
    <div class="paper-meta">
      📅 2025-12-24
      | 💬 14 pages, 20 figures, conference, accepted by HPCA 2026
    </div>
    <details class="paper-abstract">
      Streaming video large language models (LLMs) are increasingly used for real-time multimodal tasks such as video captioning, question answering, conversational agents, and augmented reality. However, these models face fundamental memory and computational challenges because their key-value (KV) caches grow substantially with continuous streaming video input. This process requires an iterative prefill stage, which is a unique feature of streaming video LLMs. Due to its iterative prefill stage, it suffers from significant limitations, including extensive computation, substantial data transfer, and degradation in accuracy. Crucially, this issue is exacerbated for edge deployment, which is the primary target for these models. In this work, we propose V-Rex, the first software-hardware co-designed accelerator that comprehensively addresses both algorithmic and hardware bottlenecks in streaming video LLM inference. At its core, V-Rex introduces ReSV, a training-free dynamic KV cache retrieval algorithm. ReSV exploits temporal and spatial similarity-based token clustering to reduce excessive KV cache memory across video frames. To fully realize these algorithmic benefits, V-Rex offers a compact, low-latency hardware accelerator with a dynamic KV cache retrieval engine (DRE), featuring bit-level and early-exit based computing units. V-Rex achieves unprecedented real-time of 3.9-8.3 FPS and energy-efficient streaming video LLM inference on edge deployment with negligible accuracy loss. While DRE only accounts for 2.2% power and 2.0% area, the system delivers 1.9-19.7x speedup and 3.1-18.5x energy efficiency improvements over AGX Orin GPU. This work is the first to comprehensively tackle KV cache retrieval across algorithms and hardware, enabling real-time streaming video LLM inference on resource-constrained edge devices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21017v1">Rethinking Supervised Fine-Tuning: Emphasizing Key Answer Tokens for Improved LLM Accuracy</a></div>
    <div class="paper-meta">
      📅 2025-12-24
    </div>
    <details class="paper-abstract">
      With the rapid advancement of Large Language Models (LLMs), the Chain-of-Thought (CoT) component has become significant for complex reasoning tasks. However, in conventional Supervised Fine-Tuning (SFT), the model could allocate disproportionately more attention to CoT sequences with excessive length. This reduces focus on the much shorter but essential Key portion-the final answer, whose correctness directly determines task success and evaluation quality. To address this limitation, we propose SFTKey, a two-stage training scheme. In the first stage, conventional SFT is applied to ensure proper output format, while in the second stage, only the Key portion is fine-tuned to improve accuracy. Extensive experiments across multiple benchmarks and model families demonstrate that SFTKey achieves an average accuracy improvement exceeding 5\% over conventional SFT, while preserving the ability to generate correct formats. Overall, this study advances LLM fine-tuning by explicitly balancing CoT learning with additional optimization on answer-relevant tokens.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21010v1">LLM Swiss Round: Aggregating Multi-Benchmark Performance via Competitive Swiss-System Dynamics</a></div>
    <div class="paper-meta">
      📅 2025-12-24
      | 💬 18 pages
    </div>
    <details class="paper-abstract">
      The rapid proliferation of Large Language Models (LLMs) and diverse specialized benchmarks necessitates a shift from fragmented, task-specific metrics to a holistic, competitive ranking system that effectively aggregates performance across multiple ability dimensions. Primarily using static scoring, current evaluation methods are fundamentally limited. They struggle to determine the proper mix ratio across diverse benchmarks, and critically, they fail to capture a model's dynamic competitive fitness or its vulnerability when confronted with sequential, high-stakes tasks. To address this, we introduce the novel Competitive Swiss-System Dynamics (CSD) framework. CSD simulates a multi-round, sequential contest where models are dynamically paired across a curated sequence of benchmarks based on their accumulated win-loss record. And Monte Carlo Simulation ($N=100,000$ iterations) is used to approximate the statistically robust Expected Win Score ($E[S_m]$), which eliminates the noise of random pairing and early-round luck. Furthermore, we implement a Failure Sensitivity Analysis by parameterizing the per-round elimination quantity ($T_k$), which allows us to profile models based on their risk appetite--distinguishing between robust generalists and aggressive specialists. We demonstrate that CSD provides a more nuanced and context-aware ranking than traditional aggregate scoring and static pairwise models, representing a vital step towards risk-informed, next-generation LLM evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21008v1">GateBreaker: Gate-Guided Attacks on Mixture-of-Expert LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-24
      | 💬 Accepted by USENIX Security'26
    </div>
    <details class="paper-abstract">
      Mixture-of-Experts (MoE) architectures have advanced the scaling of Large Language Models (LLMs) by activating only a sparse subset of parameters per input, enabling state-of-the-art performance with reduced computational cost. As these models are increasingly deployed in critical domains, understanding and strengthening their alignment mechanisms is essential to prevent harmful outputs. However, existing LLM safety research has focused almost exclusively on dense architectures, leaving the unique safety properties of MoEs largely unexamined. The modular, sparsely-activated design of MoEs suggests that safety mechanisms may operate differently than in dense models, raising questions about their robustness. In this paper, we present GateBreaker, the first training-free, lightweight, and architecture-agnostic attack framework that compromises the safety alignment of modern MoE LLMs at inference time. GateBreaker operates in three stages: (i) gate-level profiling, which identifies safety experts disproportionately routed on harmful inputs, (ii) expert-level localization, which localizes the safety structure within safety experts, and (iii) targeted safety removal, which disables the identified safety structure to compromise the safety alignment. Our study shows that MoE safety concentrates within a small subset of neurons coordinated by sparse routing. Selective disabling of these neurons, approximately 3% of neurons in the targeted expert layers, significantly increases the averaged attack success rate (ASR) from 7.4% to 64.9% against the eight latest aligned MoE LLMs with limited utility degradation. These safety neurons transfer across models within the same family, raising ASR from 17.9% to 67.7% with one-shot transfer attack. Furthermore, GateBreaker generalizes to five MoE vision language models (VLMs) with 60.9% ASR on unsafe image inputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20997v1">LLM-Empowered Agentic AI for QoE-Aware Network Slicing Management in Industrial IoT</a></div>
    <div class="paper-meta">
      📅 2025-12-24
      | 💬 8 pages, 5 figures
    </div>
    <details class="paper-abstract">
      The Industrial Internet of Things (IIoT) requires networks that deliver ultra-low latency, high reliability, and cost efficiency, which traditional optimization methods and deep reinforcement learning (DRL)-based approaches struggle to provide under dynamic and heterogeneous workloads. To address this gap, large language model (LLM)-empowered agentic AI has emerged as a promising paradigm, integrating reasoning, planning, and adaptation to enable QoE-aware network management. In this paper, we explore the integration of agentic AI into QoE-aware network slicing for IIoT. We first review the network slicing management architecture, QoE metrics for IIoT applications, and the challenges of dynamically managing heterogeneous network slices, while highlighting the motivations and advantages of adopting agentic AI. We then present the workflow of agentic AI-based slicing management, illustrating the full lifecycle of AI agents from processing slice requests to constructing slice instances and performing dynamic adjustments. Furthermore, we propose an LLM-empowered agentic AI approach for slicing management, which integrates a retrieval-augmented generation (RAG) module for semantic intent inference, a DRL-based orchestrator for slicing configuration, and an incremental memory mechanism for continual learning and adaptation. Through a case study on heterogeneous slice management, we demonstrate that the proposed approach significantly outperforms other baselines in balancing latency, reliability, and cost, and achieves up to a 19% improvement in slice availability ratio.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20986v1">AegisAgent: An Autonomous Defense Agent Against Prompt Injection Attacks in LLM-HARs</a></div>
    <div class="paper-meta">
      📅 2025-12-24
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into wearable sensing is creating a new class of mobile applications capable of nuanced human activity understanding. However, the reliability of these systems is critically undermined by their vulnerability to prompt injection attacks, where attackers deliberately input deceptive instructions into LLMs. Traditional defenses, based on static filters and rigid rules, are insufficient to address the semantic complexity of these new attacks. We argue that a paradigm shift is needed -- from passive filtering to active protection and autonomous reasoning. We introduce AegisAgent, an autonomous agent system designed to ensure the security of LLM-driven HAR systems. Instead of merely blocking threats, AegisAgent functions as a cognitive guardian. It autonomously perceives potential semantic inconsistencies, reasons about the user's true intent by consulting a dynamic memory of past interactions, and acts by generating and executing a multi-step verification and repair plan. We implement AegisAgent as a lightweight, full-stack prototype and conduct a systematic evaluation on 15 common attacks with five state-of-the-art LLM-based HAR systems on three public datasets. Results show it reduces attack success rate by 30\% on average while incurring only 78.6 ms of latency overhead on a GPU workstation. Our work makes the first step towards building secure and trustworthy LLM-driven HAR systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20983v1">Automatic Replication of LLM Mistakes in Medical Conversations</a></div>
    <div class="paper-meta">
      📅 2025-12-24
      | 💬 48 pages, 3 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly evaluated in clinical settings using multi-dimensional rubrics which quantify reasoning quality, safety, and patient-centeredness. Yet, replicating specific mistakes in other LLM models is not straightforward and often requires manual effort. We introduce MedMistake, an automatic pipeline that extracts mistakes LLMs make in patient-doctor conversations and converts them into a benchmark of single-shot QA pairs. Our pipeline (1) creates complex, conversational data between an LLM patient and LLM doctor, (2) runs an evaluation with a committee of 2 LLM judges across a variety of dimensions and (3) creates simplified single-shot QA scenarios from those mistakes. We release MedMistake-All, a dataset of 3,390 single-shot QA pairs where GPT-5 and Gemini 2.5 Pro are currently failing to answer correctly, as judged by two LLM judges. We used medical experts to validate a subset of 211/3390 questions (MedMistake-Bench), which we used to run a final evaluation of 12 frontier LLMs: Claude Opus 4.5, Claude Sonnet 4.5, DeepSeek-Chat, Gemini 2.5 Pro, Gemini 3 Pro, GPT-4o, GPT-5, GPT-5.1, GPT-5.2, Grok 4, Grok 4.1, Mistral Large. We found that GPT models, Claude and Grok obtained the best performance on MedMistake-Bench. We release both the doctor-validated benchmark (MedMistake-Bench), as well as the full dataset (MedMistake-All) at https://huggingface.co/datasets/TheLumos/MedicalMistakeBenchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20975v1">SPOT!: Map-Guided LLM Agent for Unsupervised Multi-CCTV Dynamic Object Tracking</a></div>
    <div class="paper-meta">
      📅 2025-12-24
      | 💬 33 pages, 27figures
    </div>
    <details class="paper-abstract">
      CCTV-based vehicle tracking systems face structural limitations in continuously connecting the trajectories of the same vehicle across multiple camera environments. In particular, blind spots occur due to the intervals between CCTVs and limited Fields of View (FOV), which leads to object ID switching and trajectory loss, thereby reducing the reliability of real-time path prediction. This paper proposes SPOT (Spatial Prediction Over Trajectories), a map-guided LLM agent capable of tracking vehicles even in blind spots of multi-CCTV environments without prior training. The proposed method represents road structures (Waypoints) and CCTV placement information as documents based on 2D spatial coordinates and organizes them through chunking techniques to enable real-time querying and inference. Furthermore, it transforms the vehicle's position into the actual world coordinate system using the relative position and FOV information of objects observed in CCTV images. By combining map spatial information with the vehicle's moving direction, speed, and driving patterns, a beam search is performed at the intersection level to derive candidate CCTV locations where the vehicle is most likely to enter after the blind spot. Experimental results based on the CARLA simulator in a virtual city environment confirmed that the proposed method accurately predicts the next appearing CCTV even in blind spot sections, maintaining continuous vehicle trajectories more effectively than existing techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20957v1">One Tool Is Enough: Reinforcement Learning for Repository-Level LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-12-24
    </div>
    <details class="paper-abstract">
      Locating the files and functions requiring modification in large open-source software (OSS) repositories is challenging due to their scale and structural complexity. Existing large language model (LLM)-based methods typically treat this as a repository-level retrieval task and rely on multiple auxiliary tools, which overlook code execution logic and complicate model control. We propose RepoNavigator, an LLM agent equipped with a single execution-aware tool-jumping to the definition of an invoked symbol. This unified design reflects the actual flow of code execution while simplifying tool manipulation. RepoNavigator is trained end-to-end via Reinforcement Learning (RL) directly from a pretrained model, without any closed-source distillation. Experiments demonstrate that RL-trained RepoNavigator achieves state-of-the-art performance, with the 7B model outperforming 14B baselines, the 14B model surpassing 32B competitors, and even the 32B model exceeding closed-source models such as Claude-3.7. These results confirm that integrating a single, structurally grounded tool with RL training provides an efficient and scalable solution for repository-level issue localization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.02080v2">Evolving Security in LLMs: A Study of Jailbreak Attacks and Defenses</a></div>
    <div class="paper-meta">
      📅 2025-12-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly popular, powering a wide range of applications. Their widespread use has sparked concerns, especially through jailbreak attacks that bypass safety measures to produce harmful content. In this paper, we present a comprehensive security analysis of large language models (LLMs), addressing critical research questions on the evolution and determinants of model safety. Specifically, we begin by identifying the most effective techniques for detecting jailbreak attacks. Next, we investigate whether newer versions of LLMs offer improved security compared to their predecessors. We also assess the impact of model size on overall security and explore the potential benefits of integrating multiple defense strategies to enhance the security. Our study evaluates both open-source (e.g., LLaMA and Mistral) and closed-source models (e.g., GPT-4) by employing four state-of-the-art attack techniques and assessing the efficacy of three new defensive approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20926v1">Uncovering Hierarchical Structure in LLM Embeddings with $δ$-Hyperbolicity, Ultrametricity, and Neighbor Joining</a></div>
    <div class="paper-meta">
      📅 2025-12-24
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has enabled significant strides in various fields. This paper introduces a novel approach to evaluate the effectiveness of LLM embeddings in the context of inherent geometric properties. We investigate the structural properties of these embeddings through three complementary metrics $δ$-hyperbolicity, Ultrametricity, and Neighbor Joining. $δ$-hyperbolicity, a measure derived from geometric group theory, quantifies how much a metric space deviates from being a tree-like structure. In contrast, ultrametricity characterizes strictly hierarchical structures where distances obey a strong triangle inequality. While Neighbor Joining quantifies how tree-like the distance relationships are, it does so specifically with respect to the tree reconstructed by the Neighbor Joining algorithm. By analyzing the embeddings generated by LLMs using these metrics, we uncover to what extent the embedding space reflects an underlying hierarchical or tree-like organization. Our findings reveal that LLM embeddings exhibit varying degrees of hyperbolicity and ultrametricity, which correlate with their performance in the underlying machine learning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20920v1">RevFFN: Memory-Efficient Full-Parameter Fine-Tuning of Mixture-of-Experts LLMs with Reversible Blocks</a></div>
    <div class="paper-meta">
      📅 2025-12-24
      | 💬 Under submission
    </div>
    <details class="paper-abstract">
      Full parameter fine tuning is a key technique for adapting large language models (LLMs) to downstream tasks, but it incurs substantial memory overhead due to the need to cache extensive intermediate activations for backpropagation. This bottleneck makes full fine tuning of contemporary large scale LLMs challenging in practice. Existing distributed training frameworks such as DeepSpeed alleviate this issue using techniques like ZeRO and FSDP, which rely on multi GPU memory or CPU offloading, but often require additional hardware resources and reduce training speed. We introduce RevFFN, a memory efficient fine tuning paradigm for mixture of experts (MoE) LLMs. RevFFN employs carefully designed reversible Transformer blocks that allow reconstruction of layer input activations from outputs during backpropagation, eliminating the need to store most intermediate activations in memory. While preserving the expressive capacity of MoE architectures, this approach significantly reduces peak memory consumption for full parameter fine tuning. As a result, RevFFN enables efficient full fine tuning on a single consumer grade or server grade GPU.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20082v2">Adaptive Financial Sentiment Analysis for NIFTY 50 via Instruction-Tuned LLMs , RAG and Reinforcement Learning Approaches</a></div>
    <div class="paper-meta">
      📅 2025-12-24
      | 💬 Accepted in CODS 2025
    </div>
    <details class="paper-abstract">
      Financial sentiment analysis plays a crucial role in informing investment decisions, assessing market risk, and predicting stock price trends. Existing works in financial sentiment analysis have not considered the impact of stock prices or market feedback on sentiment analysis. In this paper, we propose an adaptive framework that integrates large language models (LLMs) with real-world stock market feedback to improve sentiment classification in the context of the Indian stock market. The proposed methodology fine-tunes the LLaMA 3.2 3B model using instruction-based learning on the SentiFin dataset. To enhance sentiment predictions, a retrieval-augmented generation (RAG) pipeline is employed that dynamically selects multi-source contextual information based on the cosine similarity of the sentence embeddings. Furthermore, a feedback-driven module is introduced that adjusts the reliability of the source by comparing predicted sentiment with actual next-day stock returns, allowing the system to iteratively adapt to market behavior. To generalize this adaptive mechanism across temporal data, a reinforcement learning agent trained using proximal policy optimization (PPO) is incorporated. The PPO agent learns to optimize source weighting policies based on cumulative reward signals from sentiment-return alignment. Experimental results on NIFTY 50 news headlines collected from 2024 to 2025 demonstrate that the proposed system significantly improves classification accuracy, F1-score, and market alignment over baseline models and static retrieval methods. The results validate the potential of combining instruction-tuned LLMs with dynamic feedback and reinforcement learning for robust, market-aware financial sentiment modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.00617v2">ART: Adaptive Response Tuning Framework -- A Multi-Agent Tournament-Based Approach to LLM Response Optimization</a></div>
    <div class="paper-meta">
      📅 2025-12-24
      | 💬 14 pages, 11 figures, 5 tables. IEEE conference-style paper with appendices
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities in natural language understanding and generation. However, single-model responses often exhibit inconsistencies, hallucinations, and varying quality across different query domains. This paper presents ART (Adaptive Response Tuning), a novel framework that employs tournament-style ELO ranking and multi-agent reasoning to systematically optimize LLM outputs. By enabling multiple LLM agents to compete, critique, and collaborate through structured tournament workflows, ART produces consensus responses that outperform individual model outputs. Our framework introduces configurable tournament parameters, dynamic agent selection, and multiple consensus fusion strategies. Experimental evaluations demonstrate significant improvements in response accuracy, coherence, and reliability compared to baseline single-model approaches. The ART framework provides a scalable, production-ready solution for applications requiring high-quality, vetted LLM responses, achieving an 8.4% improvement in overall quality metrics and R^2 values exceeding 0.96 in ELO rating convergence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20900v1">When Experts Speak:Sequential LLM-Bayesian Learning for Startup Success Prediction</a></div>
    <div class="paper-meta">
      📅 2025-12-24
    </div>
    <details class="paper-abstract">
      Evaluating startups is inherently challenging in entrepreneurial finance, where investors confront severe information asymmetry and limited quantitative data. Leveraging a novel expert network call data, we develop an LLM-Bayesian model that analyzes these conversations at the question-answer turn level, extracting semantic and evaluative signals via large language models (LLMs) and aggregating them in a sequential Bayesian architecture. The model dynamically updates beliefs as additional expert calls occur and attenuates contradictory assessments, which are absent from existing text-based screening tools. Empirically, our model outperforms state-of-the-art benchmarks by 6.691% in F1-score and increases portfolio-level Return on Investment by 15.255%. Attention and ablation analyses reveal that conversational cues are particularly informative for technologically complex startups, young firms, diverse founding teams, and firms with low public visibility. By converting expert dialogue into continually updated probabilities, our model advances research in entrepreneurial finance and information systems and offers policy implications for improving funding outcomes for informationally disadvantaged startups.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20884v1">The Silent Scholar Problem: A Probabilistic Framework for Breaking Epistemic Asymmetry in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-12-24
    </div>
    <details class="paper-abstract">
      Autonomous agents powered by LLMs and Retrieval-Augmented Generation (RAG) are proficient consumers of digital content but remain unidirectional, a limitation we term epistemic asymmetry. This isolation leads to redundant reasoning and stagnates collective intelligence. Current self-reflection frameworks remain largely heuristic and private, lacking a probabilistic foundation to quantify certainty or justify external interaction.To bridge this gap, we propose a formal probabilistic framework that provides agents with a non-altruistic motive for bidirectional knowledge exchange. We model an agent's belief in a proposition using a Beta-Bernoulli distribution with a forgetting factor ($γ$). This allows us to isolate epistemic uncertainty as the variance of belief, establishing a dual drive for interaction: A homeostatic motive: The need to maintain certainty against the temporal decay introduced by $γ$. An optimal learning strategy: Targeting points of maximum ambiguity ($\mathbb{E}[θ]=0.5$) to maximize information gain. Under this framework, public contribution is reframed as optimal active learning: sharing solutions to elicit feedback is the most efficient method for an agent to reduce its own uncertainty. To ensure scalability, we introduce epistemic caching, which leverages the forgetting factor to dynamically prioritize resources for the active head of non-stationary knowledge distributions. Finally, we demonstrate how these accumulated belief states serve as verifiable reward signals for Reinforcement Learning from Human Feedback (RLHF) and high-quality data filters for Supervised Fine-Tuning (SFT). Simulation results validate that this uncertainty-driven strategy significantly outperforms random baselines in heterogeneous (Zipfian) environments, maintaining high adaptability to concept drift.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21422v1">Teaching People LLM's Errors and Getting it Right</a></div>
    <div class="paper-meta">
      📅 2025-12-24
    </div>
    <details class="paper-abstract">
      People use large language models (LLMs) when they should not. This is partly because they see LLMs compose poems and answer intricate questions, so they understandably, but incorrectly, assume LLMs won't stumble on basic tasks like simple arithmetic. Prior work has tried to address this by clustering instance embeddings into regions where an LLM is likely to fail and automatically describing patterns in these regions. The found failure patterns are taught to users to mitigate their overreliance. Yet, this approach has not fully succeeded. In this analysis paper, we aim to understand why. We first examine whether the negative result stems from the absence of failure patterns. We group instances in two datasets by their meta-labels and evaluate an LLM's predictions on these groups. We then define criteria to flag groups that are sizable and where the LLM is error-prone, and find meta-label groups that meet these criteria. Their meta-labels are the LLM's failure patterns that could be taught to users, so they do exist. We next test whether prompting and embedding-based approaches can surface these known failures. Without this, users cannot be taught about them to reduce their overreliance. We find mixed results across methods, which could explain the negative result. Finally, we revisit the final metric that measures teaching effectiveness. We propose to assess a user's ability to effectively use the given failure patterns to anticipate when an LLM is error-prone. A user study shows a positive effect from teaching with this metric, unlike the human-AI team accuracy. Our findings show that teaching failure patterns could be a viable approach to mitigating overreliance, but success depends on better automated failure-discovery methods and using metrics like ours.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21404v1">LLM-Driven Feature-Level Adversarial Attacks on Android Malware Detectors</a></div>
    <div class="paper-meta">
      📅 2025-12-24
    </div>
    <details class="paper-abstract">
      The rapid growth in both the scale and complexity of Android malware has driven the widespread adoption of machine learning (ML) techniques for scalable and accurate malware detection. Despite their effectiveness, these models remain vulnerable to adversarial attacks that introduce carefully crafted feature-level perturbations to evade detection while preserving malicious functionality. In this paper, we present LAMLAD, a novel adversarial attack framework that exploits the generative and reasoning capabilities of large language models (LLMs) to bypass ML-based Android malware classifiers. LAMLAD employs a dual-agent architecture composed of an LLM manipulator, which generates realistic and functionality-preserving feature perturbations, and an LLM analyzer, which guides the perturbation process toward successful evasion. To improve efficiency and contextual awareness, LAMLAD integrates retrieval-augmented generation (RAG) into the LLM pipeline. Focusing on Drebin-style feature representations, LAMLAD enables stealthy and high-confidence attacks against widely deployed Android malware detection systems. We evaluate LAMLAD against three representative ML-based Android malware detectors and compare its performance with two state-of-the-art adversarial attack methods. Experimental results demonstrate that LAMLAD achieves an attack success rate (ASR) of up to 97%, requiring on average only three attempts per adversarial sample, highlighting its effectiveness, efficiency, and adaptability in practical adversarial settings. Furthermore, we propose an adversarial training-based defense strategy that reduces the ASR by more than 30% on average, significantly enhancing model robustness against LAMLAD-style attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22266v1">LLMTM: Benchmarking and Optimizing LLMs for Temporal Motif Analysis in Dynamic Graphs</a></div>
    <div class="paper-meta">
      📅 2025-12-24
    </div>
    <details class="paper-abstract">
      The widespread application of Large Language Models (LLMs) has motivated a growing interest in their capacity for processing dynamic graphs. Temporal motifs, as an elementary unit and important local property of dynamic graphs which can directly reflect anomalies and unique phenomena, are essential for understanding their evolutionary dynamics and structural features. However, leveraging LLMs for temporal motif analysis on dynamic graphs remains relatively unexplored. In this paper, we systematically study LLM performance on temporal motif-related tasks. Specifically, we propose a comprehensive benchmark, LLMTM (Large Language Models in Temporal Motifs), which includes six tailored tasks across nine temporal motif types. We then conduct extensive experiments to analyze the impacts of different prompting techniques and LLMs (including nine models: openPangu-7B, the DeepSeek-R1-Distill-Qwen series, Qwen2.5-32B-Instruct, GPT-4o-mini, DeepSeek-R1, and o3) on model performance. Informed by our benchmark findings, we develop a tool-augmented LLM agent that leverages precisely engineered prompts to solve these tasks with high accuracy. Nevertheless, the high accuracy of the agent incurs a substantial cost. To address this trade-off, we propose a simple yet effective structure-aware dispatcher that considers both the dynamic graph's structural properties and the LLM's cognitive load to intelligently dispatch queries between the standard LLM prompting and the more powerful agent. Our experiments demonstrate that the structure-aware dispatcher effectively maintains high accuracy while reducing cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22250v1">Hallucination Detection for LLM-based Text-to-SQL Generation via Two-Stage Metamorphic Testing</a></div>
    <div class="paper-meta">
      📅 2025-12-24
    </div>
    <details class="paper-abstract">
      In Text-to-SQL generation, large language models (LLMs) have shown strong generalization and adaptability. However, LLMs sometimes generate hallucinations, i.e.,unrealistic or illogical content, which leads to incorrect SQL queries and negatively impacts downstream applications. Detecting these hallucinations is particularly challenging. Existing Text-to-SQL error detection methods, which are tailored for traditional deep learning models, face significant limitations when applied to LLMs. This is primarily due to the scarcity of ground-truth data. To address this challenge, we propose SQLHD, a novel hallucination detection method based on metamorphic testing (MT) that does not require standard answers. SQLHD splits the detection task into two sequentiial stages: schema-linking hallucination detection via eight structure-aware Metamorphic Relations (MRs) that perturb comparative words, entities, sentence structure or database schema, and logical-synthesis hallucination detection via nine logic-aware MRs that mutate prefix words, extremum expressions, comparison ranges or the entire database. In each stage the LLM is invoked separately to generate schema mappings or SQL artefacts; the follow-up outputs are cross-checked against their source counterparts through the corresponding MRs, and any violation is flagged as a hallucination without requiring ground-truth SQL. The experimental results demonstrate our method's superior performance in terms of the F1-score, which ranges from 69.36\% to 82.76\%. Additionally, SQLHD demonstrates superior performance over LLM Self-Evaluation methods, effectively identifying hallucinations in Text-to-SQL tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21326v1">Measuring all the noises of LLM Evals</a></div>
    <div class="paper-meta">
      📅 2025-12-24
    </div>
    <details class="paper-abstract">
      Separating signal from noise is central to experimental science. Applying well-established statistical method effectively to LLM evals requires consideration of their unique noise characteristics. We clearly define and measure three types of noise: prediction noise from generating different answers on a given question, data noise from sampling questions, and their combined total noise following the law of total variance. To emphasize relative comparisons and gain statistical power, we propose the all-pairs paired method, which applies the paired analysis to all pairs of LLMs and measures all the noise components based on millions of question-level predictions across many evals and settings. These measurements revealed clear patterns. First, each eval exhibits a characteristic and highly predictable total noise level across all model pairs. Second, paired prediction noise typically exceeds paired data noise, which means reducing prediction noise by averaging can significantly increase statistical power. These findings enable practitioners to assess significance without custom testing and to detect much smaller effects in controlled experiments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21316v1">Scaling Laws for Economic Productivity: Experimental Evidence in LLM-Assisted Consulting, Data Analyst, and Management Tasks</a></div>
    <div class="paper-meta">
      📅 2025-12-24
    </div>
    <details class="paper-abstract">
      This paper derives `Scaling Laws for Economic Impacts' -- empirical relationships between the training compute of Large Language Models (LLMs) and professional productivity. In a preregistered experiment, over 500 consultants, data analysts, and managers completed professional tasks using one of 13 LLMs. We find that each year of AI model progress reduced task time by 8%, with 56% of gains driven by increased compute and 44% by algorithmic progress. However, productivity gains were significantly larger for non-agentic analytical tasks compared to agentic workflows requiring tool use. These findings suggest continued model scaling could boost U.S. productivity by approximately 20% over the next decade.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21309v1">A Plan Reuse Mechanism for LLM-Driven Agent</a></div>
    <div class="paper-meta">
      📅 2025-12-24
      | 💬 This paper is an English version of A Plan Reuse Mechanism for LLM-Driven Agent published in 2024 in the Journal of Computer Research and Development
    </div>
    <details class="paper-abstract">
      Integrating large language models (LLMs) into personal assistants, like Xiao Ai and Blue Heart V, effectively enhances their ability to interact with humans, solve complex tasks, and manage IoT devices. Such assistants are also termed LLM-driven agents. Upon receiving user requests, the LLM-driven agent generates plans using an LLM, executes these plans through various tools, and then returns the response to the user. During this process, the latency for generating a plan with an LLM can reach tens of seconds, significantly degrading user experience. Real-world dataset analysis shows that about 30% of the requests received by LLM-driven agents are identical or similar, which allows the reuse of previously generated plans to reduce latency. However, it is difficult to accurately define the similarity between the request texts received by the LLM-driven agent through directly evaluating the original request texts. Moreover, the diverse expressions of natural language and the unstructured format of plan texts make implementing plan reuse challenging. To address these issues, we present and implement a plan reuse mechanism for LLM-driven agents called AgentReuse. AgentReuse leverages the similarities and differences among requests' semantics and uses intent classification to evaluate the similarities between requests and enable the reuse of plans. Experimental results based on a real-world dataset demonstrate that AgentReuse achieves a 93% effective plan reuse rate, an F1 score of 0.9718, and an accuracy of 0.9459 in evaluating request similarities, reducing latency by 93.12% compared with baselines without using the reuse mechanism.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20002v1">LoFT-LLM: Low-Frequency Time-Series Forecasting with Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-12-23
      | 💬 Accepted at KDD 2026. 9 pages
    </div>
    <details class="paper-abstract">
      Time-series forecasting in real-world applications such as finance and energy often faces challenges due to limited training data and complex, noisy temporal dynamics. Existing deep forecasting models typically supervise predictions using full-length temporal windows, which include substantial high-frequency noise and obscure long-term trends. Moreover, auxiliary variables containing rich domain-specific information are often underutilized, especially in few-shot settings. To address these challenges, we propose LoFT-LLM, a frequency-aware forecasting pipeline that integrates low-frequency learning with semantic calibration via a large language model (LLM). Firstly, a Patch Low-Frequency forecasting Module (PLFM) extracts stable low-frequency trends from localized spectral patches. Secondly, a residual learner then models high-frequency variations. Finally, a fine-tuned LLM refines the predictions by incorporating auxiliary context and domain knowledge through structured natural language prompts. Extensive experiments on financial and energy datasets demonstrate that LoFT-LLM significantly outperforms strong baselines under both full-data and few-shot regimes, delivering superior accuracy, robustness, and interpretability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.24347v3">Fewer Hallucinations, More Verification: A Three-Stage LLM-Based Framework for ASR Error Correction</a></div>
    <div class="paper-meta">
      📅 2025-12-23
      | 💬 This paper has been ACCEPTED for publication in ASRU
    </div>
    <details class="paper-abstract">
      Automatic Speech Recognition (ASR) error correction aims to correct recognition errors while preserving accurate text. Although traditional approaches demonstrate moderate effectiveness, LLMs offer a paradigm that eliminates the need for training and labeled data. However, directly using LLMs will encounter hallucinations problem, which may lead to the modification of the correct text. To address this problem, we propose the Reliable LLM Correction Framework (RLLM-CF), which consists of three stages: (1) error pre-detection, (2) chain-of-thought sub-tasks iterative correction, and (3) reasoning process verification. The advantage of our method is that it does not require additional information or fine-tuning of the model, and ensures the correctness of the LLM correction under multi-pass programming. Experiments on AISHELL-1, AISHELL-2, and Librispeech show that the GPT-4o model enhanced by our framework achieves 21%, 11%, 9%, and 11.4% relative reductions in CER/WER.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.05671v2">Low-Resource Domain Adaptation for Speech LLMs via Text-Only Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-12-23
      | 💬 This paper has been ACCEPTED for publication in ASRU
    </div>
    <details class="paper-abstract">
      Recent advances in automatic speech recognition (ASR) have combined speech encoders with large language models (LLMs) through projection, forming Speech LLMs with strong performance. However, adapting them to new domains remains challenging, especially in low-resource settings where paired speech-text data is scarce. We propose a text-only fine-tuning strategy for Speech LLMs using unpaired target-domain text without requiring additional audio. To preserve speech-text alignment, we introduce a real-time evaluation mechanism during fine-tuning. This enables effective domain adaptation while maintaining source-domain performance. Experiments on LibriSpeech, SlideSpeech, and Medical datasets show that our method achieves competitive recognition performance, with minimal degradation compared to full audio-text fine-tuning. It also improves generalization to new domains without catastrophic forgetting, highlighting the potential of text-only fine-tuning for low-resource domain adaptation of ASR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19980v1">Neuron-Guided Interpretation of Code LLMs: Where, Why, and How?</a></div>
    <div class="paper-meta">
      📅 2025-12-23
      | 💬 Accepted by FSE2026
    </div>
    <details class="paper-abstract">
      Code language models excel on code intelligence tasks, yet their internal interpretability is underexplored. Existing neuron interpretability techniques from NLP are suboptimal for source code due to programming languages formal, hierarchical, and executable nature. We empirically investigate code LLMs at the neuron level, localizing language-specific neurons (selectively responsive to one language) and concept layers (feed-forward layers encoding language-agnostic code representations). We analyze Llama-3.1-8B and Qwen2.5-Coder-32B on multilingual inputs in C++, Java, Python, Go, and JavaScript, measuring neuron selectivity and layerwise contributions during generation. We find (1) neurons specialized for individual languages alongside a universal subset supporting general-purpose generation; and (2) lower layers mainly encode language-specific syntax, while middle layers capture semantic abstractions shared across languages, emerging as concept layers. We demonstrate utility on three tasks: neuron-guided fine-tuning for code generation, clone detection via concept-layer embeddings, and concept-layer-guided transfer for code summarization, each yielding consistent gains in multilingual settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19945v1">Energy-Efficient Multi-LLM Reasoning for Binary-Free Zero-Day Detection in IoT Firmware</a></div>
    <div class="paper-meta">
      📅 2025-12-23
    </div>
    <details class="paper-abstract">
      Securing Internet of Things (IoT) firmware remains difficult due to proprietary binaries, stripped symbols, heterogeneous architectures, and limited access to executable code. Existing analysis methods, such as static analysis, symbolic execution, and fuzzing, depend on binary visibility and functional emulation, making them unreliable when firmware is encrypted or inaccessible. To address this limitation, we propose a binary-free, architecture-agnostic solution that estimates the likelihood of conceptual zero-day vulnerabilities using only high-level descriptors. The approach integrates a tri-LLM reasoning architecture combining a LLaMA-based configuration interpreter, a DeepSeek-based structural abstraction analyzer, and a GPT-4o semantic fusion model. The solution also incorporates LLM computational signatures, including latency patterns, uncertainty markers, and reasoning depth indicators, as well as an energy-aware symbolic load model, to enhance interpretability and operational feasibility. In addition, we formally derive the mathematical foundations of the reasoning pipeline, establishing monotonicity, divergence, and energy-risk coupling properties that theoretically justify the model's behavior. Simulation-based evaluation reveals that high exposure conditions increase the predicted zero-day likelihood by 20 to 35 percent across models, with GPT-4o demonstrating the strongest cross-layer correlations and the highest sensitivity. Energy and divergence metrics significantly predict elevated risk (p < 0.01), reinforcing the effectiveness of the proposed reasoning framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19937v1">Interpolative Decoding: Exploring the Spectrum of Personality Traits in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-23
      | 💬 20 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Recent research has explored using very large language models (LLMs) as proxies for humans in tasks such as simulation, surveys, and studies. While LLMs do not possess a human psychology, they often can emulate human behaviors with sufficiently high fidelity to drive simulations to test human behavioral hypotheses, exhibiting more nuance and range than the rule-based agents often employed in behavioral economics. One key area of interest is the effect of personality on decision making, but the requirement that a prompt must be created for every tested personality profile introduces experimental overhead and degrades replicability. To address this issue, we leverage interpolative decoding, representing each dimension of personality as a pair of opposed prompts and employing an interpolation parameter to simulate behavior along the dimension. We show that interpolative decoding reliably modulates scores along each of the Big Five dimensions. We then show how interpolative decoding causes LLMs to mimic human decision-making behavior in economic games, replicating results from human psychological research. Finally, we present preliminary results of our efforts to ``twin'' individual human players in a collaborative game through systematic search for points in interpolation space that cause the system to replicate actions taken by the human subject.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20845v1">MAR:Multi-Agent Reflexion Improves Reasoning Abilities in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-23
    </div>
    <details class="paper-abstract">
      LLMs have shown the capacity to improve their performance on reasoning tasks through reflecting on their mistakes, and acting with these reflections in mind. However, continual reflections of the same LLM onto itself exhibit degeneration of thought, where the LLM continues to repeat the same errors again and again even with the knowledge that its wrong. To address this problem, we instead introduce multi-agent with multi-persona debators as the method to generate reflections. Through out extensive experimentation, we've found that the leads to better diversity of in the reflections generated by the llm agent. We demonstrate an accuracy of 47% EM HotPot QA (question answering) and 82.7% on HumanEval (programming), both performances surpassing reflection with a single llm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.18839v3">Detect, Explain, Escalate: Sustainable Dialogue Breakdown Management for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-12-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated substantial capabilities in conversational AI applications, yet their susceptibility to dialogue breakdowns poses significant challenges to deployment reliability and user trust. This paper introduces a "Detect, Explain, Escalate" framework to manage dialogue breakdowns in LLM-powered agents, emphasizing resource-efficient operation. Our approach integrates two key strategies: (1) We fine-tune a compact 8B-parameter model, augmented with teacher-generated reasoning traces, which serves as an efficient real-time breakdown detector and explainer. This model demonstrates robust classification and calibration on English and Japanese dialogues, and generalizes to the BETOLD dataset, improving accuracy by 7% over its baseline. (2) We systematically evaluate frontier LLMs using advanced prompting (few-shot, chain-of-thought, analogical reasoning) for high-fidelity breakdown assessment. These are integrated into an "escalation" architecture where our efficient detector defers to larger models only when necessary, substantially reducing operational costs and computational overhead. Our fine-tuned model and prompting strategies achieve state-of-the-art performance on DBDC5 and strong results on BETOLD, outperforming specialized classifiers on DBDC5 and narrowing the performance gap to larger proprietary models. The proposed monitor-escalate pipeline reduces inference costs by 54%, providing a cost-effective and interpretable solution for robust conversational AI in high-impact domains. Code and models will be publicly released.
    </details>
</div>
