# llm - 2025_10

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- Part 8
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20293v2">When Judgment Becomes Noise: How Design Failures in LLM Judge Benchmarks Silently Undermine Validity</a></div>
    <div class="paper-meta">
      📅 2025-10-04
    </div>
    <details class="paper-abstract">
      LLM-judged benchmarks are increasingly used to evaluate complex model behaviors, yet their design introduces failure modes absent in conventional ground-truth based benchmarks. We argue that without tight objectives and verifiable constructions, benchmark rankings can produce high-confidence rankings that are in fact largely noise. We introduce two mechanisms to diagnose these issues. Schematic adherence quantifies how much of a judge's overall verdict is explained by the explicit evaluation schema, revealing unexplained variance when judges deviate from their own rubric. Psychometric validity aggregates internal consistency and discriminant validity signals to quantify irreducible uncertainty in any benchmarking run. Applying these tools to Arena-Hard Auto, we find severe schema incoherence and factor collapse across popular judges: for example, unexplained variance exceeding 90 percent for DeepSeek-R1-32B and factor correlations above 0.93 for most criteria. We also show that the ELO-style aggregation used by Arena-Hard Auto collapses and masks genuine ranking uncertainty. Our results highlight design failures that undermine validity and offer actionable principles for building better-scoped, reliability-aware LLM-judged benchmarks. We released our code and dataset at https://github.com/penfever/judgment-to-noise
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03611v1">Can an LLM Induce a Graph? Investigating Memory Drift and Context Length</a></div>
    <div class="paper-meta">
      📅 2025-10-04
      | 💬 2025 IEEE International Conference on Knowledge Graph (ICKG)
    </div>
    <details class="paper-abstract">
      Recently proposed evaluation benchmarks aim to characterize the effective context length and the forgetting tendencies of large language models (LLMs). However, these benchmarks often rely on simplistic 'needle in a haystack' retrieval or continuation tasks that may not accurately reflect the performance of these models in information-dense scenarios. Thus, rather than simple next token prediction, we argue for evaluating these models on more complex reasoning tasks that requires them to induce structured relational knowledge from the text - such as graphs from potentially noisy natural language content. While the input text can be viewed as generated in terms of a graph, its structure is not made explicit and connections must be induced from distributed textual cues, separated by long contexts and interspersed with irrelevant information. Our findings reveal that LLMs begin to exhibit memory drift and contextual forgetting at much shorter effective lengths when tasked with this form of relational reasoning, compared to what existing benchmarks suggest. With these findings, we offer recommendations for the optimal use of popular LLMs for complex reasoning tasks. We further show that even models specialized for reasoning, such as OpenAI o1, remain vulnerable to early memory drift in these settings. These results point to significant limitations in the models' ability to abstract structured knowledge from unstructured input and highlight the need for architectural adaptations to improve long-range reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18350v2">How Many Parameters Does Your Task Really Need? Task Specific Pruning with LLM-Sieve</a></div>
    <div class="paper-meta">
      📅 2025-10-04
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) are increasingly deployed for narrow tasks in resource-constrained settings, a central question arises: how much of an LLM is truly necessary for a given task? We present LLM-Sieve, a framework that prunes LLMs down to the minimal parameter subset needed to preserve task performance. Our approach introduces two innovations: (i) output-aligned non-orthogonal projections, which yield more faithful low-rank approximations than traditional PCA/SVD by aligning directly with layer outputs; and (ii) adaptive pruning via a Genetic Algorithm, which automatically discovers matrix-specific pruning levels and exposes the uneven distribution of task-relevant knowledge. Across models from 3.8B to 70B parameters, LLM-Sieve removes 20-75% of weights with only 1-5% accuracy loss-substantially ahead of prior pruning methods. Beyond efficiency, our framework reveals bottleneck matrices that concentrate critical knowledge, suggesting architectural implications for future LLM design. LLM-Sieve integrates seamlessly with LoRA fine-tuning and quantization, enabling both efficient deployment and deeper understanding of knowledge organization in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01315v2">Wired for Reuse: Automating Context-Aware Code Adaptation in IDEs via LLM-Based Agent</a></div>
    <div class="paper-meta">
      📅 2025-10-04
      | 💬 Accepted by ASE2025
    </div>
    <details class="paper-abstract">
      Copy-paste-modify is a widespread and pragmatic practice in software development, where developers adapt reused code snippets, sourced from platforms such as Stack Overflow, GitHub, or LLM outputs, into their local codebase. A critical yet underexplored aspect of this adaptation is code wiring: the context-aware process of substituting unresolved variables in pasted code with suitable variables or expressions from the surrounding context. Existing solutions either rely on heuristic rules or historical templates, often failing to effectively utilize contextual information, despite studies showing that over half of adaptation cases are context-dependent. In this paper, we introduce WIRL, an LLM-based agent for code wiring framed as a Retrieval-Augmented Generation (RAG) infilling task. WIRL combines an LLM, a customized toolkit, and an orchestration module to identify unresolved variables, retrieve context, and perform context-aware substitutions. To balance efficiency and autonomy, the agent adopts a mixed strategy: deterministic rule-based steps for common patterns, and a state-machine-guided decision process for intelligent exploration. We evaluate WIRL on a carefully curated, high-quality dataset consisting of real-world code adaptation scenarios. Our approach achieves an exact match precision of 91.7% and a recall of 90.0%, outperforming advanced LLMs by 22.6 and 13.7 percentage points in precision and recall, respectively, and surpassing IntelliJ IDEA by 54.3 and 49.9 percentage points. These results underscore its practical utility, particularly in contexts with complex variable dependencies or multiple unresolved variables. We believe WIRL paves the way for more intelligent and context-aware developer assistance in modern IDEs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23252v2">NanoFlux: Adversarial Dual-LLM Evaluation and Distillation For Multi-Domain Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-04
      | 💬 preprint version
    </div>
    <details class="paper-abstract">
      We present NanoFlux, a novel adversarial framework for generating targeted training data to improve LLM reasoning, where adversarially-generated datasets containing fewer than 200 examples outperform conventional fine-tuning approaches. The framework employs a competitive dynamic between models alternating as Attacker and Defender, supervised by a tool-augmented Judge, synthesizing multi-step questions with explanatory annotations that target specific reasoning capabilities. Fine-tuning a 4B-parameter model on NanoFlux-generated data yields performance gains across diverse domains compared to full-benchmark fine-tuning: +5.9% on mathematical reasoning (GSMHard), +3.6% on scientific reasoning (GenomeBench), and +16.6% on medical reasoning (MultiMedQA), while reducing computational requirements by 3-14x. Ablation studies reveal a non-monotonic relationship between dataset characteristics and model performance, uncovering domain-specific optimal points for question complexity and reasoning quality. NanoFlux automates training data generation through embedding-based novelty filtering, tool-augmented evaluation, and multi-hop reasoning, suggesting that future model improvements may lie in the intelligent synthesis of small, precisely targeted training datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03595v1">Decoupling Task-Solving and Output Formatting in LLM Generation</a></div>
    <div class="paper-meta">
      📅 2025-10-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly adept at following instructions containing task descriptions to solve complex problems, such as mathematical reasoning and automatic evaluation (LLM-as-a-Judge). However, as prompts grow more complex, models often struggle to adhere to all instructions. This difficulty is especially common when instructive prompts intertwine reasoning directives -- specifying what the model should solve -- with rigid formatting requirements that dictate how the solution must be presented. The entanglement creates competing goals for the model, suggesting that more explicit separation of these two aspects could lead to improved performance. To this front, we introduce Deco-G, a decoding framework that explicitly decouples format adherence from task solving. Deco-G handles format compliance with a separate tractable probabilistic model (TPM), while prompts LLMs with only task instructions. At each decoding step, Deco-G combines next token probabilities from the LLM with the TPM calculated format compliance likelihood to form the output probability. To make this approach both practical and scalable for modern instruction-tuned LLMs, we introduce three key innovations: instruction-aware distillation, a flexible trie-building algorithm, and HMM state pruning for computational efficiency. We demonstrate the effectiveness of Deco-G across a wide range of tasks with diverse format requirements, including mathematical reasoning, LLM-as-a-judge, and event argument extraction. Overall, our approach yields 1.0% to 6.0% relative gain over regular prompting practice with guaranteed format compliance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04439v3">ArcMemo: Abstract Reasoning Composition with Lifelong LLM Memory</a></div>
    <div class="paper-meta">
      📅 2025-10-04
    </div>
    <details class="paper-abstract">
      While inference-time scaling enables LLMs to carry out increasingly long and capable reasoning traces, the patterns and insights uncovered during these traces are immediately discarded once the context window is reset for a new query. External memory is a natural way to persist these discoveries, and recent work has shown clear benefits for reasoning-intensive tasks. We see an opportunity to make such memories more broadly reusable and scalable by moving beyond instance-based memory entries (e.g. exact query/response pairs, or summaries tightly coupled with the original problem context) toward concept-level memory: reusable, modular abstractions distilled from solution traces and stored in natural language. For future queries, relevant concepts are selectively retrieved and integrated into the prompt, enabling test-time continual learning without weight updates. Our design introduces new strategies for abstracting takeaways from rollouts and retrieving entries for new queries, promoting reuse and allowing memory to expand with additional experiences. We evaluate on ARC-AGI, a benchmark that stresses compositional generalization and abstract reasoning, making it a natural fit for concept memory. Our method yields a 7.5% relative gain over a strong no-memory baseline with performance continuing to scale with inference compute. We find abstract concepts to be the most consistent memory design, outscoring the baseline at all tested inference compute scales. Moreover, dynamically updating memory during test-time outperforms fixed settings, supporting the hypothesis that accumulating and abstracting patterns enables further solutions in a form of self-improvement. Code is available at https://github.com/matt-seb-ho/arc_memo.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03955v1">Harnessing Synthetic Preference Data for Enhancing Temporal Understanding of Video-LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-04
      | 💬 17 pages, 9 figures, 6 tables. Presents TimeWarp, a synthetic preference data framework to improve temporal understanding in Video-LLMs, showing consistent gains across seven benchmarks. Includes supplementary material in the Appendix
    </div>
    <details class="paper-abstract">
      While Video Large Language Models (Video-LLMs) have demonstrated remarkable performance across general video understanding benchmarks-particularly in video captioning and descriptive tasks-they consistently underperform on tasks that require fine-grained temporal understanding. This limitation arises due to the lack of visual complexity and temporal nuance in current fine-tuning datasets, leading these models to rely heavily on language-based reasoning rather than truly understanding video dynamics. In this work, we propose TimeWarp, a systematic method to create a targeted synthetic temporal dataset to fine-tune the model's responses to encourage it to focus on the given input video. We introduce a large-scale preference dataset, created using TimeWarp, that captures intricate temporal dynamics often overlooked, grounding the model's responses to visual and temporal information. We demonstrate that when our method is applied to existing models, it significantly improves performance on temporal understanding benchmarks, highlighting the effectiveness of our proposed datasets in advancing temporal understanding in Video-LLMs, resulting in an absolute improvement in performance across seven benchmarks. Code is available at https://github.com/sameepv21/timewarp.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.21700v2">XBreaking: Explainable Artificial Intelligence for Jailbreaking LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      Large Language Models are fundamental actors in the modern IT landscape dominated by AI solutions. However, security threats associated with them might prevent their reliable adoption in critical application scenarios such as government organizations and medical institutions. For this reason, commercial LLMs typically undergo a sophisticated censoring mechanism to eliminate any harmful output they could possibly produce. In response to this, LLM Jailbreaking is a significant threat to such protections, and many previous approaches have already demonstrated its effectiveness across diverse domains. Existing jailbreak proposals mostly adopt a generate-and-test strategy to craft malicious input. To improve the comprehension of censoring mechanisms and design a targeted jailbreak attack, we propose an Explainable-AI solution that comparatively analyzes the behavior of censored and uncensored models to derive unique exploitable alignment patterns. Then, we propose XBreaking, a novel jailbreak attack that exploits these unique patterns to break the security constraints of LLMs by targeted noise injection. Our thorough experimental campaign returns important insights about the censoring mechanisms and demonstrates the effectiveness and performance of our attack.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02816v1">NCV: A Node-Wise Consistency Verification Approach for Low-Cost Structured Error Localization in LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      Verifying multi-step reasoning in large language models is difficult due to imprecise error localization and high token costs. Existing methods either assess entire reasoning chains, suffering attention dilution, or rely on expensive multi-sampling. We introduce Node-wise Consistency Verification (NCV), a training-free framework that recasts verification as lightweight binary consistency checks at the node level. By decomposing the chain of thought into interconnected verification nodes, NCV precisely localizes errors and avoids unnecessary long-form generation. Experiments demonstrate that our approach enhances interpretability and efficiency, presenting a scalable solution for reliable LLM reasoning verification. On public datasets, NCV achieves a 10\% to 25\% improvement in F1 scores over baselines while utilizing $6\times$~$58\times$ fewer tokens than traditional methods like CoT-based verifiers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11191v3">Primus: A Pioneering Collection of Open-Source Datasets for Cybersecurity LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-10-03
      | 💬 Accepted to EMNLP 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable advancements in specialized fields such as finance, law, and medicine. However, in cybersecurity, we have noticed a lack of open-source datasets, with a particular lack of high-quality cybersecurity pretraining corpora, even though much research indicates that LLMs acquire their knowledge during pretraining. To address this, we present a comprehensive suite of datasets covering all major training stages, including pretraining, instruction fine-tuning, and reasoning distillation with cybersecurity-specific self-reflection data. Extensive ablation studies demonstrate their effectiveness on public cybersecurity benchmarks. In particular, continual pre-training on our dataset yields a 15.9% improvement in the aggregate score, while reasoning distillation leads to a 15.8% gain in security certification (CISSP). We will release all datasets and trained cybersecurity LLMs under the ODC-BY and MIT licenses to encourage further research in the community. For access to all datasets and model weights, please refer to https://huggingface.co/collections/trendmicro-ailab/primus-67b1fd27052b802b4af9d243.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02758v1">TokenFlow: Responsive LLM Text Streaming Serving under Request Burst via Preemptive Scheduling</a></div>
    <div class="paper-meta">
      📅 2025-10-03
      | 💬 Accepted by EuroSys 2026
    </div>
    <details class="paper-abstract">
      Real-time LLM interactions demand streamed token generations, where text tokens are progressively generated and delivered to users while balancing two objectives: responsiveness (i.e., low time-to-first-token) and steady generation (i.e.,required time-between-tokens). Standard LLM serving systems suffer from the inflexibility caused by non-preemptive request scheduling and reactive memory management, leading to poor resource utilization and low request processing parallelism under request bursts. Therefore, we present TokenFlow, a novel LLM serving system with enhanced text streaming performance via preemptive request scheduling and proactive key-value (KV) cache management. TokenFlow dynamically prioritizes requests based on real-time token buffer occupancy and token consumption rate, while actively transferring KV cache between GPU and CPU memory in the background and overlapping I/O with computation to minimize request preemption overhead. Extensive experiments on Llama3-8B and Qwen2.5-32B across multiple GPUs (RTX 4090, A6000, H200) demonstrate that TokenFlow achieves up to 82.5% higher effective throughput (accounting for actual user consumption) while reducing P99 TTFT by up to 80.2%, without degrading overall token throughput.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00415v2">MarketSenseAI 2.0: Enhancing Stock Analysis through LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-03
      | 💬 25 pages, 7 figures, Under review at Financial Innovation (FIN)
    </div>
    <details class="paper-abstract">
      MarketSenseAI is a novel framework for holistic stock analysis which leverages Large Language Models (LLMs) to process financial news, historical prices, company fundamentals and the macroeconomic environment to support decision making in stock analysis and selection. In this paper, we present the latest advancements on MarketSenseAI, driven by rapid technological expansion in LLMs. Through a novel architecture combining Retrieval-Augmented Generation and LLM agents, the framework processes SEC filings and earnings calls, while enriching macroeconomic analysis through systematic processing of diverse institutional reports. We demonstrate a significant improvement in fundamental analysis accuracy over the previous version. Empirical evaluation on S\&P 100 stocks over two years (2023-2024) shows MarketSenseAI achieving cumulative returns of 125.9% compared to the index return of 73.5%, while maintaining comparable risk profiles. Further validation on S\&P 500 stocks during 2024 demonstrates the framework's scalability, delivering a 33.8% higher Sortino ratio than the market. This work marks a significant advancement in applying LLM technology to financial analysis, offering insights into the robustness of LLM-driven investment strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02742v1">IndiCASA: A Dataset and Bias Evaluation Framework in LLMs Using Contrastive Embedding Similarity in the Indian Context</a></div>
    <div class="paper-meta">
      📅 2025-10-03
      | 💬 Accepted at 8th AAAI/ACM Conference on AI, Ethics, and Society (AIES) 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have gained significant traction across critical domains owing to their impressive contextual understanding and generative capabilities. However, their increasing deployment in high stakes applications necessitates rigorous evaluation of embedded biases, particularly in culturally diverse contexts like India where existing embedding-based bias assessment methods often fall short in capturing nuanced stereotypes. We propose an evaluation framework based on a encoder trained using contrastive learning that captures fine-grained bias through embedding similarity. We also introduce a novel dataset - IndiCASA (IndiBias-based Contextually Aligned Stereotypes and Anti-stereotypes) comprising 2,575 human-validated sentences spanning five demographic axes: caste, gender, religion, disability, and socioeconomic status. Our evaluation of multiple open-weight LLMs reveals that all models exhibit some degree of stereotypical bias, with disability related biases being notably persistent, and religion bias generally lower likely due to global debiasing efforts demonstrating the need for fairer model development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02719v1">TravelBench : Exploring LLM Performance in Low-Resource Domains</a></div>
    <div class="paper-meta">
      📅 2025-10-03
      | 💬 10 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Results on existing LLM benchmarks capture little information over the model capabilities in low-resource tasks, making it difficult to develop effective solutions in these domains. To address these challenges, we curated 14 travel-domain datasets spanning 7 common NLP tasks using anonymised data from real-world scenarios, and analysed the performance across LLMs. We report on the accuracy, scaling behaviour, and reasoning capabilities of LLMs in a variety of tasks. Our results confirm that general benchmarking results are insufficient for understanding model performance in low-resource tasks. Despite the amount of training FLOPs, out-of-the-box LLMs hit performance bottlenecks in complex, domain-specific scenarios. Furthermore, reasoning provides a more significant boost for smaller LLMs by making the model a better judge on certain tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02716v1">A $1000\times$ Faster LLM-enhanced Algorithm For Path Planning in Large-scale Grid Maps</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      Path planning in grid maps, arising from various applications, has garnered significant attention. Existing methods, such as A*, Dijkstra, and their variants, work well for small-scale maps but fail to address large-scale ones due to high search time and memory consumption. Recently, Large Language Models (LLMs) have shown remarkable performance in path planning but still suffer from spatial illusion and poor planning performance. Among all the works, LLM-A* \cite{meng2024llm} leverages LLM to generate a series of waypoints and then uses A* to plan the paths between the neighboring waypoints. In this way, the complete path is constructed. However, LLM-A* still suffers from high computational time for large-scale maps. To fill this gap, we conducted a deep investigation into LLM-A* and found its bottleneck, resulting in limited performance. Accordingly, we design an innovative LLM-enhanced algorithm, abbr. as iLLM-A*. iLLM-A* includes 3 carefully designed mechanisms, including the optimization of A*, an incremental learning method for LLM to generate high-quality waypoints, and the selection of the appropriate waypoints for A* for path planning. Finally, a comprehensive evaluation on various grid maps shows that, compared with LLM-A*, iLLM-A* \textbf{1) achieves more than $1000\times$ speedup on average, and up to $2349.5\times$ speedup in the extreme case, 2) saves up to $58.6\%$ of the memory cost, 3) achieves both obviously shorter path length and lower path length standard deviation.}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02694v1">MALF: A Multi-Agent LLM Framework for Intelligent Fuzzing of Industrial Control Protocols</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      Industrial control systems (ICS) are vital to modern infrastructure but increasingly vulnerable to cybersecurity threats, particularly through weaknesses in their communication protocols. This paper presents MALF (Multi-Agent LLM Fuzzing Framework), an advanced fuzzing solution that integrates large language models (LLMs) with multi-agent coordination to identify vulnerabilities in industrial control protocols (ICPs). By leveraging Retrieval-Augmented Generation (RAG) for domain-specific knowledge and QLoRA fine-tuning for protocol-aware input generation, MALF enhances fuzz testing precision and adaptability. The multi-agent framework optimizes seed generation, mutation strategies, and feedback-driven refinement, leading to improved vulnerability discovery. Experiments on protocols like Modbus/TCP, S7Comm, and Ethernet/IP demonstrate that MALF surpasses traditional methods, achieving a test case pass rate (TCPR) of 88-92% and generating more exception triggers (ETN). MALF also maintains over 90% seed coverage and Shannon entropy values between 4.2 and 4.6 bits, ensuring diverse, protocol-compliant mutations. Deployed in a real-world Industrial Attack-Defense Range for power plants, MALF identified critical vulnerabilities, including three zero-day flaws, one confirmed and registered by CNVD. These results validate MALF's effectiveness in real-world fuzzing applications. This research highlights the transformative potential of multi-agent LLMs in ICS cybersecurity, offering a scalable, automated framework that sets a new standard for vulnerability discovery and strengthens critical infrastructure security against emerging threats.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02675v1">HALO: Memory-Centric Heterogeneous Accelerator with 2.5D Integration for Low-Batch LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      The rapid adoption of Large Language Models (LLMs) has driven a growing demand for efficient inference, particularly in latency-sensitive applications such as chatbots and personalized assistants. Unlike traditional deep neural networks, LLM inference proceeds in two distinct phases: the prefill phase, which processes the full input sequence in parallel, and the decode phase, which generates tokens sequentially. These phases exhibit highly diverse compute and memory requirements, which makes accelerator design particularly challenging. Prior works have primarily been optimized for high-batch inference or evaluated only short input context lengths, leaving the low-batch and long context regime, which is critical for interactive applications, largely underexplored. We propose HALO, a heterogeneous memory centric accelerator designed for these unique challenges of prefill and decode phases in low-batch LLM inference. HALO integrates HBM based Compute-in-DRAM (CiD) with an on-chip analog Compute-in-Memory (CiM), co-packaged using 2.5D integration. To further improve the hardware utilization, we introduce a phase-aware mapping strategy that adapts to the distinct demands of the prefill and decode phases. Compute bound operations in the prefill phase are mapped to CiM to exploit its high throughput matrix multiplication capability, while memory-bound operations in the decode phase are executed on CiD to benefit from reduced data movement within DRAM. Additionally, we present an analysis of the performance tradeoffs of LLMs under two architectural extremes: a fully CiD and a fully on-chip analog CiM design to highlight the need for a heterogeneous design. We evaluate HALO on LLaMA-2 7B and Qwen3 8B models. Our experimental results show that LLMs mapped to HALO achieve up to 18x geometric mean speedup over AttAcc, an attention-optimized mapping and 2.5x over CENT, a fully CiD based mapping.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00907v5">Grounding Multimodal LLMs to Embodied Agents that Ask for Help with Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      Embodied agents operating in household environments must interpret ambiguous and under-specified human instructions. A capable household robot should recognize ambiguity and ask relevant clarification questions to infer the user intent accurately, leading to more effective task execution. To study this problem, we introduce the Ask-to-Act task, where an embodied agent is tasked with a single or multi-object rearrangement task using an under-specified instruction in a home environment. The agent must strategically ask minimal, yet relevant, clarification questions to resolve ambiguity while navigating under partial observability. To address this challenge, we propose a novel approach that fine-tunes multi-modal large language models (MLLMs) as vision-language-action (VLA) policies using online reinforcement learning (RL) with LLM-generated rewards. Our method eliminates the need for large-scale human demonstrations or manually engineered rewards for training such agents. We benchmark against strong zero-shot baselines including GPT-4o as well as supervised fine-tuned MLLMs on our task. Our results show that our RL-finetuned MLLM outperforms all baselines by a significant margin (10.4-16.5%), generalizing well to novel scenes and tasks. To the best of our knowledge, this is the first demonstration of adapting MLLMs as VLA agents that can act and ask for help using LLM-generated rewards with online RL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02657v1">Less LLM, More Documents: Searching for Improved RAG</a></div>
    <div class="paper-meta">
      📅 2025-10-03
      | 💬 16 pages. Submitted to ECIR 2026
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) couples document retrieval with large language models (LLMs). While scaling generators improves accuracy, it also raises cost and limits deployability. We explore an orthogonal axis: enlarging the retriever's corpus to reduce reliance on large LLMs. Experimental results show that corpus scaling consistently strengthens RAG and can often serve as a substitute for increasing model size, though with diminishing returns at larger scales. Small- and mid-sized generators paired with larger corpora often rival much larger models with smaller corpora; mid-sized models tend to gain the most, while tiny and large models benefit less. Our analysis shows that improvements arise primarily from increased coverage of answer-bearing passages, while utilization efficiency remains largely unchanged. These findings establish a principled corpus-generator trade-off: investing in larger corpora offers an effective path to stronger RAG, often comparable to enlarging the LLM itself.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02645v1">Mind the Gap: Linguistic Divergence and Adaptation Strategies in Human-LLM Assistant vs. Human-Human Interactions</a></div>
    <div class="paper-meta">
      📅 2025-10-03
      | 💬 Accepted to The Second Workshop on Generative AI for E-commerce (GenAIECommerce '25), held September 22, 2025, in Prague, Czech Republic
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) are increasingly deployed in customer-facing applications, a critical yet underexplored question is how users communicate differently with LLM chatbots compared to human agent. In this study, we present empirical evidence that users adopt distinct communication styles when users interact with chatbots versus human agents. Our analysis reveals significant differences in grammatical fluency, politeness, and lexical diversity in user language between the two settings. These findings suggest that models trained exclusively on human-human interaction data may not adequately accommodate the communication style shift that occurs once an LLM chatbot is deployed. To enhance LLM robustness to post-launch communication style changes, we experimented with two strategies: (1) data augmentation during the post-training phase and (2) inference-time user message reformulation. Our results indicate that models trained on stylistically diverse datasets significantly outperform those trained exclusively on original or stylistically uniform datasets, while inference-time reformulation proved less effective. These insights help us to better adapt our models for improved LLM-user interaction experiences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02637v1">Homophily-induced Emergence of Biased Structures in LLM-based Multi-Agent AI Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-03
      | 💬 Accepted for publication in Social Network Analysis and Mining
    </div>
    <details class="paper-abstract">
      This study examines how interactions among artificially intelligent (AI) agents, guided by large language models (LLMs), drive the evolution of collective network structures. We ask LLM-driven agents to grow a network by informing them about current link constellations. Our observations confirm that agents consistently apply a preferential attachment mechanism, favoring connections to nodes with higher degrees. We systematically solicited more than a million decisions from four different LLMs, including Gemini, ChatGPT, Llama, and Claude. When social attributes such as age, gender, religion, and political orientation are incorporated, the resulting networks exhibit heightened assortativity, leading to the formation of distinct homophilic communities. This significantly alters the network topology from what would be expected under a pure preferential attachment model alone. Political and religious attributes most significantly fragment the collective, fostering polarized subgroups, while age and gender yield more gradual structural shifts. Strikingly, LLMs also reveal asymmetric patterns in heterophilous ties, suggesting embedded directional biases reflective of societal norms. As autonomous AI agents increasingly shape the architecture of online systems, these findings contribute to how algorithmic choices of generative AI collectives not only reshape network topology, but offer critical insights into how AI-driven systems co-evolve and self-organize.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01471v2">Fine-tuning LLMs with variational Bayesian last layer for high-dimensional Bayesian optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      A plethora of applications entail solving black-box optimization problems with high evaluation costs, including drug discovery, material design, as well as hyperparameter tuning. Toward finding the global optimum of such black-box optimization problems with sample efficiency, Bayesian optimization (BO) is a theoretically elegant framework that relies on a probabilistic surrogate model so as to iteratively select the query point with well-balanced exploration-exploitation tradeoffs. The Gaussian process (GP), as the de-facto choice for surrogate modeling, has achieved compelling performances for vanilla BO with low-dimensional continuous variables. However, GPs fall short in coping with high-dimensional counterparts with {\it irregular} variables (e.g., categorical, ordinal, etc.). To alleviate this, neural network-based surrogates have been explored. Inspired by the powerful capabilities of LLMs, we adopt the LLM as the surrogate to model the mapping from the high-dimensional input variables to the objective function. To adapt to the current problem, we leverage the low-rank adaptation (LoRA) to fine-tune the LLM parameters together with the posterior of a linear regression head via the variational Bayesian last layer (VBLL) framework. The resulting LoRA-VBLL is not only computationally light compared to existing alternatives, but also admits recursive updates. To automate the critical selection of the LoRA rank as well as other hyperparameters, a weighted ensemble (ENS) of LoRA-VBLL surrogates has been devised, which further accommodates continual update of the per-model weight and individual LoRA-VBLL parameters via recursive Bayes. Extensive experimental results demonstrate the compelling performance of the proposed (ENS-)LoRA-VBLL approaches on various high-dimensional benchmarks and the real-world molecular optimization tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03577v1">LLM, Reporting In! Medical Information Extraction Across Prompting, Fine-tuning and Post-correction</a></div>
    <div class="paper-meta">
      📅 2025-10-03
      | 💬 in French language
    </div>
    <details class="paper-abstract">
      This work presents our participation in the EvalLLM 2025 challenge on biomedical Named Entity Recognition (NER) and health event extraction in French (few-shot setting). For NER, we propose three approaches combining large language models (LLMs), annotation guidelines, synthetic data, and post-processing: (1) in-context learning (ICL) with GPT-4.1, incorporating automatic selection of 10 examples and a summary of the annotation guidelines into the prompt, (2) the universal NER system GLiNER, fine-tuned on a synthetic corpus and then verified by an LLM in post-processing, and (3) the open LLM LLaMA-3.1-8B-Instruct, fine-tuned on the same synthetic corpus. Event extraction uses the same ICL strategy with GPT-4.1, reusing the guideline summary in the prompt. Results show GPT-4.1 leads with a macro-F1 of 61.53% for NER and 15.02% for event extraction, highlighting the importance of well-crafted prompting to maximize performance in very low-resource scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03567v1">Machine Unlearning Meets Adversarial Robustness via Constrained Interventions on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      With the increasing adoption of Large Language Models (LLMs), more customization is needed to ensure privacy-preserving and safe generation. We address this objective from two critical aspects: unlearning of sensitive information and robustness to jail-breaking attacks. We investigate various constrained optimization formulations that address both aspects in a \emph{unified manner}, by finding the smallest possible interventions on LLM weights that either make a given vocabulary set unreachable or embed the LLM with robustness to tailored attacks by shifting part of the weights to a \emph{safer} region. Beyond unifying two key properties, this approach contrasts with previous work in that it doesn't require an oracle classifier that is typically not available or represents a computational overhead. Surprisingly, we find that the simplest point-wise constraint-based intervention we propose leads to better performance than max-min interventions, while having a lower computational cost. Comparison against state-of-the-art defense methods demonstrates superior performance of the proposed approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21016v2">RL Grokking Recipe: How Does RL Unlock and Transfer New Algorithms in LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      It remains an open question whether LLMs can acquire or generalize genuinely new reasoning strategies, beyond the sharpened skills encoded in their parameters during pre-training or post-training. To attempt to answer this debate, we introduce DELTA-Code -- Distributional Evaluation of Learnability and Transferrability in Algorithmic Coding -- a controlled benchmark of synthetic coding problem families designed to probe two fundamental aspects: learnability -- can LLMs, through reinforcement learning (RL), solve problem families where pretrained models exhibit failure with large enough attempts (pass@K=0)? -- and transferrability -- if learnability happens, can such skills transfer systematically to out-of-distribution (OOD) test sets? Unlike prior public coding datasets, DELTA isolates reasoning skills through templated problem generators and introduces fully OOD problem families that demand novel strategies rather than tool invocation or memorized patterns. Our experiments reveal a striking grokking phase transition: after an extended period with near-zero reward, RL-trained models abruptly climb to near-perfect accuracy. To enable learnability on previously unsolvable problem families, we explore key training ingredients such as staged warm-up with dense rewards, experience replay, curriculum training, and verification-in-the-loop. Beyond learnability, we use DELTA to evaluate transferability or generalization along exploratory, compositional, and transformative axes, as well as cross-family transfer. Results show solid gains within families and for recomposed skills, but persistent weaknesses in transformative cases. DELTA thus offers a clean testbed for probing the limits of RL-driven reasoning and for understanding how models can move beyond existing priors to acquire new algorithmic skills.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03541v1">What is a protest anyway? Codebook conceptualization is still a first-order concern in LLM-era classification</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      Generative large language models (LLMs) are now used extensively for text classification in computational social science (CSS). In this work, focus on the steps before and after LLM prompting -- conceptualization of concepts to be classified and using LLM predictions in downstream statistical inference -- which we argue have been overlooked in much of LLM-era CSS. We claim LLMs can tempt analysts to skip the conceptualization step, creating conceptualization errors that bias downstream estimates. Using simulations, we show that this conceptualization-induced bias cannot be corrected for solely by increasing LLM accuracy or post-hoc bias correction methods. We conclude by reminding CSS analysts that conceptualization is still a first-order concern in the LLM-era and provide concrete advice on how to pursue low-cost, unbiased, low-variance downstream estimates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03519v1">TS-Reasoner: Aligning Time Series Foundation Models with LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      Time series reasoning is crucial to decision-making in diverse domains, including finance, energy usage, traffic, weather, and scientific discovery. While existing time series foundation models (TSFMs) can capture low-level dynamic patterns and provide accurate forecasting, further analysis usually requires additional background knowledge and sophisticated reasoning, which are lacking in most TSFMs but can be achieved through large language models (LLMs). On the other hand, without expensive post-training, LLMs often struggle with the numerical understanding of time series data. Although it is intuitive to integrate the two types of models, developing effective training recipes that align the two modalities for reasoning tasks is still an open challenge. To this end, we propose TS-Reasoner that aligns the latent representations of TSFMs with the textual inputs of LLMs for downstream understanding/reasoning tasks. Specifically, we propose a simple yet effective method to curate diverse, synthetic pairs of time series and textual captions for alignment training. We then develop a two-stage training recipe that applies instruction finetuning after the alignment pretraining. Unlike existing works that train an LLM to take time series as inputs, we leverage a pretrained TSFM and freeze it during training. Extensive experiments on several benchmarks demonstrate that TS-Reasoner not only outperforms a wide range of prevailing LLMs, Vision Language Models (VLMs), and Time Series LLMs, but also achieves this with remarkable data efficiency, e.g., using less than half the training data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21551v3">Grokking in LLM Pretraining? Monitor Memorization-to-Generalization without Test</a></div>
    <div class="paper-meta">
      📅 2025-10-03
      | 💬 10 pages, 8 figures
    </div>
    <details class="paper-abstract">
      This paper presents the first study of grokking in practical LLM pretraining. Specifically, we investigate when an LLM memorizes the training data, when its generalization on downstream tasks starts to improve, and what happens if there is a lag between the two. Unlike existing works studying when a small model generalizes to limited and specified tasks during thousands epochs' training on algorithmic data, we focus on a practical setting for LLMs, i.e., one-epoch pretraining of next-token prediction on a cross-domain, large-scale corpus, and generalization on diverse benchmark tasks covering math/commonsense reasoning, code generation, and domain-specific retrieval. Our study, for the first time, verifies that grokking still emerges in pretraining mixture-of-experts (MoE) LLMs, though different local data groups may enter their grokking stages asynchronously due to the heterogeneity of their distributions and attributions to others. To find a mechanistic interpretation of this local grokking, we investigate the dynamics of training data's pathways (i.e., expert choices across layers in MoE). Our primary discovery is that the pathways evolve from random, non-smooth across layers, instance-specific to more structured and transferable across samples, despite the converged pretraining loss. This depicts a transition from memorization to generalization. Two novel metrics are developed to quantify these patterns: one computes the pathway similarity between samples, while the other measures the consistency of aggregated experts between subsequent layers for each sample. These training data based metrics induce zero cost but can faithfully track and monitor the generalization of LLMs on downstream tasks, which, in conventional settings, requires costly instruction tuning and benchmark evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05278v2">Micro-Act: Mitigating Knowledge Conflict in LLM-based RAG via Actionable Self-Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-03
      | 💬 Accepted by ACL 2025 Main
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) systems commonly suffer from Knowledge Conflicts, where retrieved external knowledge contradicts the inherent, parametric knowledge of large language models (LLMs). It adversely affects performance on downstream tasks such as question answering (QA). Existing approaches often attempt to mitigate conflicts by directly comparing two knowledge sources in a side-by-side manner, but this can overwhelm LLMs with extraneous or lengthy contexts, ultimately hindering their ability to identify and mitigate inconsistencies. To address this issue, we propose Micro-Act a framework with a hierarchical action space that automatically perceives context complexity and adaptively decomposes each knowledge source into a sequence of fine-grained comparisons. These comparisons are represented as actionable steps, enabling reasoning beyond the superficial context. Through extensive experiments on five benchmark datasets, Micro-Act consistently achieves significant increase in QA accuracy over state-of-the-art baselines across all 5 datasets and 3 conflict types, especially in temporal and semantic types where all baselines fail significantly. More importantly, Micro-Act exhibits robust performance on non-conflict questions simultaneously, highlighting its practical value in real-world RAG applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03502v1">ALHD: A Large-Scale and Multigenre Benchmark Dataset for Arabic LLM-Generated Text Detection</a></div>
    <div class="paper-meta">
      📅 2025-10-03
      | 💬 47 pages, 15 figures. Dataset available at Zenodo: https://doi.org/10.5281/zenodo.17249602 Codebase available at GitHub: https://github.com/alikhairallah/ALHD-Benchmarking
    </div>
    <details class="paper-abstract">
      We introduce ALHD, the first large-scale comprehensive Arabic dataset explicitly designed to distinguish between human- and LLM-generated texts. ALHD spans three genres (news, social media, reviews), covering both MSA and dialectal Arabic, and contains over 400K balanced samples generated by three leading LLMs and originated from multiple human sources, which enables studying generalizability in Arabic LLM-genearted text detection. We provide rigorous preprocessing, rich annotations, and standardized balanced splits to support reproducibility. In addition, we present, analyze and discuss benchmark experiments using our new dataset, in turn identifying gaps and proposing future research directions. Benchmarking across traditional classifiers, BERT-based models, and LLMs (zero-shot and few-shot) demonstrates that fine-tuned BERT models achieve competitive performance, outperforming LLM-based models. Results are however not always consistent, as we observe challenges when generalizing across genres; indeed, models struggle to generalize when they need to deal with unseen patterns in cross-genre settings, and these challenges are particularly prominent when dealing with news articles, where LLM-generated texts resemble human texts in style, which opens up avenues for future research. ALHD establishes a foundation for research related to Arabic LLM-detection and mitigating risks of misinformation, academic dishonesty, and cyber threats.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03480v1">LLM Agents for Automated Dependency Upgrades</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      As a codebase expands over time, its library dependencies can become outdated and require updates to maintain innovation and security. However, updating a library can introduce breaking changes in the code, necessitating significant developer time for maintenance. To address this, we introduce a framework of LLM agents to be used in combination with migration documentation to automatically recommend and apply code updates and ensure compatibility with new versions. Our solution can automatically localize updated library usages in live Java codebases and implement recommended fixes in a user-friendly manner. The system architecture consists of multiple key components: a Summary Agent, Control Agent, and Code Agent. To validate our approach, we apply the framework on an industrial use case by which we create three synthetic code repositories with major Upgrade changes and benchmark our approach against state-of-the-art methods. Results show that our approach not only performs upgrades using fewer tokens across all cases but also achieves a precision of 71.4%, highlighting its efficiency and effectiveness compared to state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04557v2">SSA-COMET: Do LLMs Outperform Learned Metrics in Evaluating MT for Under-Resourced African Languages?</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      Evaluating machine translation (MT) quality for under-resourced African languages remains a significant challenge, as existing metrics often suffer from limited language coverage and poor performance in low-resource settings. While recent efforts, such as AfriCOMET, have addressed some of the issues, they are still constrained by small evaluation sets, a lack of publicly available training data tailored to African languages, and inconsistent performance in extremely low-resource scenarios. In this work, we introduce SSA-MTE, a large-scale human-annotated MT evaluation (MTE) dataset covering 14 African language pairs from the News domain, with over 73,000 sentence-level annotations from a diverse set of MT systems. Based on this data, we develop SSA-COMET and SSA-COMET-QE, improved reference-based and reference-free evaluation metrics. We also benchmark prompting-based approaches using state-of-the-art LLMs like GPT-4o, Claude-3.7 and Gemini 2.5 Pro. Our experimental results show that SSA-COMET models significantly outperform AfriCOMET and are competitive with the strongest LLM Gemini 2.5 Pro evaluated in our study, particularly on low-resource languages such as Twi, Luo, and Yoruba. All resources are released under open licenses to support future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03469v1">Bridging LLM Planning Agents and Formal Methods: A Case Study in Plan Verification</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      We introduce a novel framework for evaluating the alignment between natural language plans and their expected behavior by converting them into Kripke structures and Linear Temporal Logic (LTL) using Large Language Models (LLMs) and performing model checking. We systematically evaluate this framework on a simplified version of the PlanBench plan verification dataset and report on metrics like Accuracy, Precision, Recall and F1 scores. Our experiments demonstrate that GPT-5 achieves excellent classification performance (F1 score of 96.3%) while almost always producing syntactically perfect formal representations that can act as guarantees. However, the synthesis of semantically perfect formal models remains an area for future exploration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03463v1">ALMAS: an Autonomous LLM-based Multi-Agent Software Engineering Framework</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      Multi-agent Large Language Model (LLM) systems have been leading the way in applied LLM research across a number of fields. One notable area is software development, where researchers have advanced the automation of code implementation, code testing, code maintenance, inter alia, using LLM agents. However, software development is a multifaceted environment that extends beyond just code. As such, a successful LLM system must factor in multiple stages of the software development life-cycle (SDLC). In this paper, we propose a vision for ALMAS, an Autonomous LLM-based Multi-Agent Software Engineering framework, which follows the above SDLC philosophy such that it may work within an agile software development team to perform several tasks end-to-end. ALMAS aligns its agents with agile roles, and can be used in a modular fashion to seamlessly integrate with human developers and their development environment. We showcase the progress towards ALMAS through our published works and a use case demonstrating the framework, where ALMAS is able to seamlessly generate an application and add a new feature.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03425v1">Memory-Efficient Backpropagation for Fine-Tuning LLMs on Resource-Constrained Mobile Devices</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) with backpropagation\textemdash even for a subset of parameters such as LoRA\textemdash can be much more memory-consuming than inference and is often deemed impractical for resource-constrained mobile devices. Alternative methods, such as zeroth-order optimization (ZO), can greatly reduce the memory footprint but come at the cost of significantly slower model convergence (10$\times$ to 100$\times$ more steps than backpropagation). We propose a memory-efficient implementation of backpropagation (MeBP) on mobile devices that provides better trade-off between memory usage and compute time, while converging faster and achieving better performance than the ZO baseline. We verify the effectiveness of MeBP on an iPhone 15 Pro Max and show that various LLMs, ranging from 0.5B to 4B parameters, can be fine-tuned using less than 1GB of memory. We release an example of the MeBP implementation at https://github.com/apple/ml-mebp.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03417v1">NEXUS: Network Exploration for eXploiting Unsafe Sequences in Multi-Turn LLM Jailbreaks</a></div>
    <div class="paper-meta">
      📅 2025-10-03
      | 💬 Javad Rafiei Asl and Sidhant Narula are co-first authors
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized natural language processing but remain vulnerable to jailbreak attacks, especially multi-turn jailbreaks that distribute malicious intent across benign exchanges and bypass alignment mechanisms. Existing approaches often explore the adversarial space poorly, rely on hand-crafted heuristics, or lack systematic query refinement. We present NEXUS (Network Exploration for eXploiting Unsafe Sequences), a modular framework for constructing, refining, and executing optimized multi-turn attacks. NEXUS comprises: (1) ThoughtNet, which hierarchically expands a harmful intent into a structured semantic network of topics, entities, and query chains; (2) a feedback-driven Simulator that iteratively refines and prunes these chains through attacker-victim-judge LLM collaboration using harmfulness and semantic-similarity benchmarks; and (3) a Network Traverser that adaptively navigates the refined query space for real-time attacks. This pipeline uncovers stealthy, high-success adversarial paths across LLMs. On several closed-source and open-source LLMs, NEXUS increases attack success rate by 2.1% to 19.4% over prior methods. Code: https://github.com/inspire-lab/NEXUS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.09071v2">Elastic On-Device LLM Service</a></div>
    <div class="paper-meta">
      📅 2025-10-03
      | 💬 MobiCom'25
    </div>
    <details class="paper-abstract">
      On-device Large Language Models (LLMs) are transforming mobile AI, catalyzing applications like UI automation without privacy concerns. Nowadays the common practice is to deploy a single yet powerful LLM as a general task solver for multiple requests. We identify a key system challenge in this paradigm: current LLMs lack the elasticity to serve requests that have diversified Service-Level Objectives (SLOs) on inference latency. To tackle this, we present \sys, an on-device LLM service that elasticizes both the model and the prompt dimension of a full LLM. It incorporates (1) a one-shot neuron-reordering method, which leverages the intrinsic permutation consistency in transformer models to generate high-quality elasticized sub-models with minimal runtime switching overhead; (2) a dual-head tiny language model, which efficiently and effectively refines the prompt and orchestrates the elastification between model and prompt. We implement such an elastic on-device LLM service on multiple COTS smartphones, and evaluate \sys on both standalone NLP/mobile-agent datasets and end-to-end synthesized traces. On diverse SLOs, \sys outperforms 7 strong baselines in (absolute) accuracy by up to 14.83\% and 10.45\% on average, with <1\% TTFT switching overhead, on-par memory consumption and <100 offline GPU hours.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03384v1">Implicit Values Embedded in How Humans and LLMs Complete Subjective Everyday Tasks</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can underpin AI assistants that help users with everyday tasks, such as by making recommendations or performing basic computation. Despite AI assistants' promise, little is known about the implicit values these assistants display while completing subjective everyday tasks. Humans may consider values like environmentalism, charity, and diversity. To what extent do LLMs exhibit these values in completing everyday tasks? How do they compare with humans? We answer these questions by auditing how six popular LLMs complete 30 everyday tasks, comparing LLMs to each other and to 100 human crowdworkers from the US. We find LLMs often do not align with humans, nor with other LLMs, in the implicit values exhibited.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03217v1">Abstain and Validate: A Dual-LLM Policy for Reducing Noise in Agentic Program Repair</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      Agentic Automated Program Repair (APR) is increasingly tackling complex, repository-level bugs in industry, but ultimately agent-generated patches still need to be reviewed by a human before committing them to ensure they address the bug. Showing unlikely patches to developers can lead to substantial noise, wasting valuable developer time and eroding trust in automated code changes. We introduce two complementary LLM-based policies to reduce such noise: bug abstention and patch validation policies. Bug abstention excludes bugs that the agentic APR system is unlikely to fix. Patch validation rejects patches that are unlikely to be a good fix for the given bug. We evaluate both policies on three sets of bugs from Google's codebase, and their candidate patches generated by an internal agentic APR system. On a set of 174 human-reported bugs, removing bugs and patch trajectories rejected by our policies can raise success rates by up to 13 percentage points and 15 percentage points, respectively, and by up to 39 percentage points in combination. On null pointer exceptions and sanitizer-reported bugs with machine-generated bug reports, patch validation also improves average single-sample success rates. This two-policy approach provides a practical path to the reliable, industrial-scale deployment of agentic APR systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03195v1">Can LLMs Hit Moving Targets? Tracking Evolving Signals in Corporate Disclosures</a></div>
    <div class="paper-meta">
      📅 2025-10-03
      | 💬 8 pages, 5 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Moving targets -- managers' strategic shifting of key performance metrics when the original targets become difficult to achieve -- have been shown to predict subsequent stock underperformance. However, our work reveals that the method employed in that study exhibits two key limitations that hinder the accuracy -- noise in the extracted targets and loss of contextual information -- both of which stem primarily from the use of a named entity recognition (NER). To address these two limitations, we propose an LLM-based target extraction} method with a newly defined metric that better captures semantic context. This approach preserves semantic context beyond simple entity recognition and yields consistently higher predictive power than the original approach. Overall, our approach enhances the granularity and accuracy of financial text-based performance prediction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.24015v3">Hierarchical Knowledge Injection for Improving LLM-based Program Repair</a></div>
    <div class="paper-meta">
      📅 2025-10-03
      | 💬 Accepted at IEEE/ACM Automated Software Engineering (ASE) 2025 Conference
    </div>
    <details class="paper-abstract">
      Prompting LLMs with bug-related context (e.g., error messages, stack traces) improves automated program repair, but many bugs still remain unresolved. In real-world projects, developers often rely on broader repository and project-level context beyond the local code to resolve such bugs. In this paper, we investigate how automatically extracting and providing such knowledge can improve LLM-based program repair. We propose a layered knowledge injection framework that incrementally augments LLMs with structured context. It starts with the Bug Knowledge Layer, which includes information such as the buggy function and failing tests; expands to the Repository Knowledge Layer, which adds structural dependencies, related files, and commit history; and finally injects the Project Knowledge Layer, which incorporates relevant details from documentation and previously fixed bugs. We evaluate this framework on a dataset of 314 bugs from BugsInPy using two LLMs (Llama 3.3 and GPT-4o-mini), and analyze fix rates across six bug types. By progressively injecting knowledge across layers, our approach achieves a fix rate of 79% (250/314) using Llama 3.3, a significant improvement of 23% over previous work. All bug types show improvement with the addition of repository-level context, while only a subset benefit further from project-level knowledge, highlighting that different bug types require different levels of contextual information for effective repair. We also analyze the remaining unresolved bugs and find that more complex and structurally isolated bugs, such as Program Anomaly and GUI bugs, remain difficult even after injecting all available information. Our results show that layered context injection improves program repair and suggest the need for interactive and adaptive APR systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.01443v2">Tuning LLM-based Code Optimization via Meta-Prompting: An Industrial Perspective</a></div>
    <div class="paper-meta">
      📅 2025-10-03
      | 💬 Accepted by ASE'25 Industry Showcase
    </div>
    <details class="paper-abstract">
      There is a growing interest in leveraging multiple large language models (LLMs) for automated code optimization. However, industrial platforms deploying multiple LLMs face a critical challenge: prompts optimized for one LLM often fail with others, requiring expensive model-specific prompt engineering. This cross-model prompt engineering bottleneck severely limits the practical deployment of multi-LLM systems in production environments. We introduce Meta-Prompted Code Optimization (MPCO), a framework that automatically generates high-quality, task-specific prompts across diverse LLMs while maintaining industrial efficiency requirements. MPCO leverages metaprompting to dynamically synthesize context-aware optimization prompts by integrating project metadata, task requirements, and LLM-specific contexts. It is an essential part of the ARTEMIS code optimization platform for automated validation and scaling. Our comprehensive evaluation on five real-world codebases with 366 hours of runtime benchmarking demonstrates MPCO's effectiveness: it achieves overall performance improvements up to 19.06% with the best statistical rank across all systems compared to baseline methods. Analysis shows that 96% of the top-performing optimizations stem from meaningful edits. Through systematic ablation studies and meta-prompter sensitivity analysis, we identify that comprehensive context integration is essential for effective meta-prompting and that major LLMs can serve effectively as meta-prompters, providing actionable insights for industrial practitioners.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03178v1">When Names Disappear: Revealing What LLMs Actually Understand About Code</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) achieve strong results on code tasks, but how they derive program meaning remains unclear. We argue that code communicates through two channels: structural semantics, which define formal behavior, and human-interpretable naming, which conveys intent. Removing the naming channel severely degrades intent-level tasks such as summarization, where models regress to line-by-line descriptions. Surprisingly, we also observe consistent reductions on execution tasks that should depend only on structure, revealing that current benchmarks reward memorization of naming patterns rather than genuine semantic reasoning. To disentangle these effects, we introduce a suite of semantics-preserving obfuscations and show that they expose identifier leakage across both summarization and execution. Building on these insights, we release ClassEval-Obf, an obfuscation-enhanced benchmark that systematically suppresses naming cues while preserving behavior. Our results demonstrate that ClassEval-Obf reduces inflated performance gaps, weakens memorization shortcuts, and provides a more reliable basis for assessing LLMs' code understanding and generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03174v1">Topic Modeling as Long-Form Generation: Can Long-Context LLMs revolutionize NTM via Zero-Shot Prompting?</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      Traditional topic models such as neural topic models rely on inference and generation networks to learn latent topic distributions. This paper explores a new paradigm for topic modeling in the era of large language models, framing TM as a long-form generation task whose definition is updated in this paradigm. We propose a simple but practical approach to implement LLM-based topic model tasks out of the box (sample a data subset, generate topics and representative text with our prompt, text assignment with keyword match). We then investigate whether the long-form generation paradigm can beat NTMs via zero-shot prompting. We conduct a systematic comparison between NTMs and LLMs in terms of topic quality and empirically examine the claim that "a majority of NTMs are outdated."
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24827v2">Putnam-like dataset summary: LLMs as mathematical competition contestants</a></div>
    <div class="paper-meta">
      📅 2025-10-03
      | 💬 11 pages, 11 figures
    </div>
    <details class="paper-abstract">
      In this paper we summarize the results of the Putnam-like benchmark published by Google DeepMind. This dataset consists of 96 original problems in the spirit of the Putnam Competition and 576 solutions of LLMs. We analyse the performance of models on this set of problems to verify their ability to solve problems from mathematical contests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09901v2">Comparing Exploration-Exploitation Strategies of LLMs and Humans: Insights from Standard Multi-armed Bandit Experiments</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to simulate or automate human behavior in complex sequential decision-making settings. A natural question is then whether LLMs exhibit similar decision-making behavior to humans, and can achieve comparable (or superior) performance. In this work, we focus on the exploration-exploitation (E&E) tradeoff, a fundamental aspect of dynamic decision-making under uncertainty. We employ canonical multi-armed bandit (MAB) experiments introduced in the cognitive science and psychiatry literature to conduct a comparative study of the E&E strategies of LLMs, humans, and MAB algorithms. We use interpretable choice models to capture the E&E strategies of the agents and investigate how enabling thinking traces, through both prompting strategies and thinking models, shapes LLM decision-making. We find that enabling thinking in LLMs shifts their behavior toward more human-like behavior, characterized by a mix of random and directed exploration. In a simple stationary setting, thinking-enabled LLMs exhibit similar levels of random and directed exploration compared to humans. However, in more complex, non-stationary environments, LLMs struggle to match human adaptability, particularly in effective directed exploration, despite achieving similar regret in certain scenarios. Our findings highlight both the promise and limits of LLMs as simulators of human behavior and tools for automated decision-making and point to potential areas for improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22811v2">Highly Efficient and Effective LLMs with Multi-Boolean Architectures</a></div>
    <div class="paper-meta">
      📅 2025-10-03
      | 💬 Preprint. Under Review
    </div>
    <details class="paper-abstract">
      Weight binarization has emerged as a promising strategy to reduce the complexity of large language models (LLMs). Existing approaches fall into post-training binarization, which is simple but causes severe performance loss, and training-aware methods, which depend on full-precision latent weights, adding complexity and limiting efficiency. We propose a novel framework that represents LLMs with multi-kernel Boolean parameters and, for the first time, enables direct finetuning LMMs in the Boolean domain, eliminating the need for latent weights. This enhances representational capacity and dramatically reduces complexity during both finetuning and inference. Extensive experiments across diverse LLMs show our method outperforms recent ultra low-bit quantization and binarization techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14837v2">Towards Economical Inference: Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-03
      | 💬 16 pages, 8 figures; Accepted to ACL 2025
    </div>
    <details class="paper-abstract">
      Multi-head Latent Attention (MLA) is an innovative architecture proposed by DeepSeek, designed to ensure efficient and economical inference by significantly compressing the Key-Value (KV) cache into a latent vector. Compared to MLA, standard LLMs employing Multi-Head Attention (MHA) and its variants such as Grouped-Query Attention (GQA) exhibit significant cost disadvantages. Enabling well-trained LLMs (e.g., Llama) to rapidly adapt to MLA without pre-training from scratch is both meaningful and challenging. This paper proposes the first data-efficient fine-tuning method for transitioning from MHA to MLA (MHA2MLA), which includes two key components: for partial-RoPE, we remove RoPE from dimensions of queries and keys that contribute less to the attention scores, for low-rank approximation, we introduce joint SVD approximations based on the pre-trained parameters of keys and values. These carefully designed strategies enable MHA2MLA to recover performance using only a small fraction (0.3% to 0.6%) of the data, significantly reducing inference costs while seamlessly integrating with compression techniques such as KV cache quantization. For example, the KV cache size of Llama2-7B is reduced by 92.19%, with only a 0.5% drop in LongBench performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03102v1">Semantic Similarity in Radiology Reports via LLMs and NER</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      Radiology report evaluation is a crucial part of radiologists' training and plays a key role in ensuring diagnostic accuracy. As part of the standard reporting workflow, a junior radiologist typically prepares a preliminary report, which is then reviewed and edited by a senior radiologist to produce the final report. Identifying semantic differences between preliminary and final reports is essential for junior doctors, both as a training tool and to help uncover gaps in clinical knowledge. While AI in radiology is a rapidly growing field, the application of large language models (LLMs) remains challenging due to the need for specialised domain knowledge. In this paper, we explore the ability of LLMs to provide explainable and accurate comparisons of reports in the radiology domain. We begin by comparing the performance of several LLMs in comparing radiology reports. We then assess a more traditional approach based on Named-Entity-Recognition (NER). However, both approaches exhibit limitations in delivering accurate feedback on semantic similarity. To address this, we propose Llama-EntScore, a semantic similarity scoring method using a combination of Llama 3.1 and NER with tunable weights to emphasise or de-emphasise specific types of differences. Our approach generates a quantitative similarity score for tracking progress and also gives an interpretation of the score that aims to offer valuable guidance in reviewing and refining their reporting. We find our method achieves 67% exact-match accuracy and 93% accuracy within +/- 1 when compared to radiologist-provided ground truth scores - outperforming both LLMs and NER used independently. Code is available at: \href{https://github.com/otmive/llama_reports}{github.com/otmive/llama\_reports}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03093v1">Revisiting Direct Speech-to-Text Translation with Speech LLMs: Better Scaling than CoT Prompting?</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      Recent work on Speech-to-Text Translation (S2TT) has focused on LLM-based models, introducing the increasingly adopted Chain-of-Thought (CoT) prompting, where the model is guided to first transcribe the speech and then translate it. CoT typically outperforms direct prompting primarily because it can exploit abundant Automatic Speech Recognition (ASR) and Text-to-Text Translation (T2TT) datasets to explicitly model its steps. In this paper, we systematically compare CoT and Direct prompting under increasing amounts of S2TT data. To this end, we pseudo-label an ASR corpus by translating its transcriptions into six European languages, and train LLM-based S2TT systems with both prompting strategies at different data scales. Our results show that Direct improves more consistently as the amount of data increases, suggesting that it may become a more effective approach as larger S2TT resources are created.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22860v2">Permissioned LLMs: Enforcing Access Control in Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      In enterprise settings, organizational data is segregated, siloed and carefully protected by elaborate access control frameworks. These access control structures can completely break down if an LLM fine-tuned on the siloed data serves requests, for downstream tasks, from individuals with disparate access privileges. We propose Permissioned LLMs (PermLLM), a new class of LLMs that superimpose the organizational data access control structures on query responses they generate. We formalize abstractions underpinning the means to determine whether access control enforcement happens correctly over LLM query responses. Our formalism introduces the notion of a relevant response that can be used to prove whether a PermLLM mechanism has been implemented correctly. We also introduce a novel metric, called access advantage, to empirically evaluate the efficacy of a PermLLM mechanism. We introduce three novel PermLLM mechanisms that build on Parameter Efficient Fine-Tuning to achieve the desired access control. We furthermore present two instantiations of access advantage--(i) Domain Distinguishability Index (DDI) based on Membership Inference Attacks, and (ii) Utility Gap Index (UGI) based on LLM utility evaluation. We demonstrate the efficacy of our PermLLM mechanisms through extensive experiments on five public datasets (GPQA, RCV1, SimpleQA, WMDP, and PubMedQA), in addition to evaluating the validity of DDI and UGI metrics themselves for quantifying access control in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03029v1">Investigating The Smells of LLM Generated Code</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      Context: Large Language Models (LLMs) are increasingly being used to generate program code. Much research has been reported on the functional correctness of generated code, but there is far less on code quality. Objectives: In this study, we propose a scenario-based method of evaluating the quality of LLM-generated code to identify the weakest scenarios in which the quality of LLM generated code should be improved. Methods: The method measures code smells, an important indicator of code quality, and compares them with a baseline formed from reference solutions of professionally written code. The test dataset is divided into various subsets according to the topics of the code and complexity of the coding tasks to represent different scenarios of using LLMs for code generation. We will also present an automated test system for this purpose and report experiments with the Java programs generated in response to prompts given to four state-of-the-art LLMs: Gemini Pro, ChatGPT, Codex, and Falcon. Results: We find that LLM-generated code has a higher incidence of code smells compared to reference solutions. Falcon performed the least badly, with a smell increase of 42.28%, followed by Gemini Pro (62.07%), ChatGPT (65.05%) and finally Codex (84.97%). The average smell increase across all LLMs was 63.34%, comprising 73.35% for implementation smells and 21.42% for design smells. We also found that the increase in code smells is greater for more complex coding tasks and for more advanced topics, such as those involving object-orientated concepts. Conclusion: In terms of code smells, LLM's performances on various coding task complexities and topics are highly correlated to the quality of human written code in the corresponding scenarios. However, the quality of LLM generated code is noticeably poorer than human written code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07044v2">DatawiseAgent: A Notebook-Centric LLM Agent Framework for Adaptive and Robust Data Science Automation</a></div>
    <div class="paper-meta">
      📅 2025-10-03
      | 💬 The camera-ready version for EMNLP 2025 Main Conference
    </div>
    <details class="paper-abstract">
      Existing large language model (LLM) agents for automating data science show promise, but they remain constrained by narrow task scopes, limited generalization across tasks and models, and over-reliance on state-of-the-art (SOTA) LLMs. We introduce DatawiseAgent, a notebook-centric LLM agent framework for adaptive and robust data science automation. Inspired by how human data scientists work in computational notebooks, DatawiseAgent introduces a unified interaction representation and a multi-stage architecture based on finite-state transducers (FSTs). This design enables flexible long-horizon planning, progressive solution development, and robust recovery from execution failures. Extensive experiments across diverse data science scenarios and models show that DatawiseAgent consistently achieves SOTA performance by surpassing strong baselines such as AutoGen and TaskWeaver, demonstrating superior effectiveness and adaptability. Further evaluations reveal graceful performance degradation under weaker or smaller models, underscoring the robustness and scalability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02934v1">Model-Agnostic Correctness Assessment for LLM-Generated Code via Dynamic Internal Representation Selection</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive capabilities in code generation and are increasingly integrated into the software development process. However, ensuring the correctness of LLM-generated code remains a critical concern. Prior work has shown that the internal representations of LLMs encode meaningful signals for assessing code correctness. Nevertheless, the existing methods rely on representations from pre-selected/fixed layers and token positions, which could limit its generalizability across diverse model architectures and tasks. In this work, we introduce AUTOPROBE, a novel model-agnostic approach that dynamically selects the most informative internal representations for code correctness assessment. AUTOPROBE employs an attention-based mechanism to learn importance scores for hidden states, enabling it to focus on the most relevant features. These weighted representations are then aggregated and passed to a probing classifier to predict code correctness across multiple dimensions, including compilability, functionality, and security. To evaluate the performance of AUTOPROBE, we conduct extensive experiments across multiple benchmarks and code LLMs. Our experimental results show that AUTOPROBE consistently outperforms the baselines. For security assessment, AUTOPROBE surpasses the state-of-the-art white-box approach by 18%. For compilability and functionality assessment, AUTOPROBE demonstrates its highest robustness to code complexity, with the performance higher than the other approaches by up to 19% and 111%, respectively. These findings highlight that dynamically selecting important internal signals enables AUTOPROBE to serve as a robust and generalizable solution for assessing the correctness of code generated by various LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02917v1">Mechanistic Interpretability of Code Correctness in LLMs via Sparse Autoencoders</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      As Large Language Models become integral to software development, with substantial portions of AI-suggested code entering production, understanding their internal correctness mechanisms becomes critical for safe deployment. We apply sparse autoencoders to decompose LLM representations, identifying directions that correspond to code correctness. We select predictor directions using t-statistics and steering directions through separation scores from base model representations, then analyze their mechanistic properties through steering, attention analysis, and weight orthogonalization. We find that code correctness directions in LLMs reliably predict incorrect code, while correction capabilities, though statistically significant, involve tradeoffs between fixing errors and preserving correct code. Mechanistically, successful code generation depends on attending to test cases rather than problem descriptions. Moreover, directions identified in base models retain their effectiveness after instruction-tuning, suggesting code correctness mechanisms learned during pre-training are repurposed during fine-tuning. Our mechanistic insights suggest three practical applications: prompting strategies should prioritize test examples over elaborate problem descriptions, predictor directions can serve as error alarms for developer review, and these same predictors can guide selective steering, intervening only when errors are anticipated to prevent the code corruption from constant steering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23799v2">Enhancing LLM Steering through Sparse Autoencoder-Based Vector Refinement</a></div>
    <div class="paper-meta">
      📅 2025-10-03
      | 💬 19 pages, 11 figures, 7 tables
    </div>
    <details class="paper-abstract">
      Steering has emerged as a promising approach in controlling large language models (LLMs) without modifying model parameters. However, most existing steering methods rely on large-scale datasets to learn clear behavioral information, which limits their applicability in many real-world scenarios. The steering vectors extracted from small dataset often contain task-irrelevant noising features, which degrades their effectiveness. To refine the steering vectors learned from limited data, we introduce Refinement of Steering Vector via Sparse Autoencoder (SAE-RSV) that leverages SAEs to semantically denoise and augment the steering vectors. In our framework, we first remove task-irrelevant features according to their semantics provided by SAEs, and then enrich task-relevant features missing from the small dataset through their semantic similarity to the identified relevant features. Extensive experiments demonstrate that the proposed SAE-RSV substantially outperforms all the baseline methods including supervised fine-tuning. Our findings show that effective steering vector can be constructed from limited training data by refining the original steering vector through SAEs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02833v1">Attack via Overfitting: 10-shot Benign Fine-tuning to Jailbreak LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      Despite substantial efforts in safety alignment, recent research indicates that Large Language Models (LLMs) remain highly susceptible to jailbreak attacks. Among these attacks, finetuning-based ones that compromise LLMs' safety alignment via fine-tuning stand out due to its stable jailbreak performance. In particular, a recent study indicates that fine-tuning with as few as 10 harmful question-answer (QA) pairs can lead to successful jailbreaking across various harmful questions. However, such malicious fine-tuning attacks are readily detectable and hence thwarted by moderation models. In this paper, we demonstrate that LLMs can be jailbroken by fine-tuning with only 10 benign QA pairs; our attack exploits the increased sensitivity of LLMs to fine-tuning data after being overfitted. Specifically, our fine-tuning process starts with overfitting an LLM via fine-tuning with benign QA pairs involving identical refusal answers. Further fine-tuning is then performed with standard benign answers, causing the overfitted LLM to forget the refusal attitude and thus provide compliant answers regardless of the harmfulness of a question. We implement our attack on the ten LLMs and compare it with five existing baselines. Experiments demonstrate that our method achieves significant advantages in both attack effectiveness and attack stealth. Our findings expose previously unreported security vulnerabilities in current LLMs and provide a new perspective on understanding how LLMs' security is compromised, even with benign fine-tuning. Our code is available at https://github.com/ZHIXINXIE/tenBenign.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02306v1">Drawing Conclusions from Draws: Rethinking Preference Semantics in Arena-Style LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 6 pages, 4 figures
    </div>
    <details class="paper-abstract">
      In arena-style evaluation of large language models (LLMs), two LLMs respond to a user query, and the user chooses the winning response or deems the "battle" a draw, resulting in an adjustment to the ratings of both models. The prevailing approach for modeling these rating dynamics is to view battles as two-player game matches, as in chess, and apply the Elo rating system and its derivatives. In this paper, we critically examine this paradigm. Specifically, we question whether a draw genuinely means that the two models are equal and hence whether their ratings should be equalized. Instead, we conjecture that draws are more indicative of query difficulty: if the query is too easy, then both models are more likely to succeed equally. On three real-world arena datasets, we show that ignoring rating updates for draws yields a 1-3% relative increase in battle outcome prediction accuracy (which includes draws) for all four rating systems studied. Further analyses suggest that draws occur more for queries rated as very easy and those as highly objective, with risk ratios of 1.37 and 1.35, respectively. We recommend future rating systems to reconsider existing draw semantics and to account for query properties in rating updates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12021v2">LitterBox+: An Extensible Framework for LLM-enhanced Scratch Static Code Analysis</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 ASE 2025 Tool Demonstration Track
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have become an essential tool to support developers using traditional text-based programming languages, but the graphical notation of the block-based Scratch programming environment inhibits the use of LLMs. To overcome this limitation, we propose the LitterBox+ framework that extends the Scratch static code analysis tool LitterBox with the generative abilities of LLMs. By converting block-based code to a textual representation suitable for LLMs, LitterBox+ allows users to query LLMs about their programs, about quality issues reported by LitterBox, and it allows generating code fixes. Besides offering a programmatic API for these functionalities, LitterBox+ also extends the Scratch user interface to make these functionalities available directly in the environment familiar to learners. The framework is designed to be easily extensible with other prompts, LLM providers, and new features combining the program analysis capabilities of LitterBox with the generative features of LLMs. We provide a screencast demonstrating the tool at https://youtu.be/RZ6E0xgrIgQ.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18436v4">Can Code-Switched Texts Activate a Knowledge Switch in LLMs? A Case Study on English-Korean Code-Switching</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 Accepted to EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Recent large language models (LLMs) demonstrate multilingual abilities, yet they are English-centric due to dominance of English in training corpora. The limited resource for low-resource languages remains a crucial challenge. Code-switching (CS), a phenomenon where multilingual speakers alternate between languages in a discourse, can convey subtle cultural and linguistic nuances that can be otherwise lost in translation and elicits language-specific knowledge in human communications. In light of this, we investigate whether code-switching can activate, or identify and leverage knowledge for reasoning when LLMs solve low-resource language tasks. To facilitate the research, we first present EnKoQA, a synthetic English-Korean CS question-answering dataset. We provide comprehensive analysis on a variety of multilingual LLMs by subdividing activation process into knowledge identification and knowledge leveraging. Our results demonstrate that compared to English text, CS can faithfully activate knowledge inside LLMs especially on language-specific domains, suggesting the potential of code-switching on low-resource language tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02249v1">Explore Briefly, Then Decide: Mitigating LLM Overthinking via Cumulative Entropy Regulation</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable reasoning abilities on complex problems using long Chain-of-Thought (CoT) reasoning. However, they often suffer from overthinking, meaning generating unnecessarily lengthy reasoning steps for simpler problems. This issue may degrade the efficiency of the models and make them difficult to adapt the reasoning depth to the complexity of problems. To address this, we introduce a novel metric Token Entropy Cumulative Average (TECA), which measures the extent of exploration throughout the reasoning process. We further propose a novel reasoning paradigm -- Explore Briefly, Then Decide -- with an associated Cumulative Entropy Regulation (CER) mechanism. This paradigm leverages TECA to help the model dynamically determine the optimal point to conclude its thought process and provide a final answer, thus achieving efficient reasoning. Experimental results across diverse mathematical benchmarks show that our approach substantially mitigates overthinking without sacrificing problem-solving ability. With our thinking paradigm, the average response length decreases by up to 71% on simpler datasets, demonstrating the effectiveness of our method in creating a more efficient and adaptive reasoning process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09674v4">Probabilistic Reasoning with LLMs for k-anonymity Estimation</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 10 pages, Accepted to NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Probabilistic reasoning is a key aspect of both human and artificial intelligence that allows for handling uncertainty and ambiguity in decision-making. In this paper, we introduce a new numerical reasoning task under uncertainty for large language models, focusing on estimating the privacy risk of user-generated documents containing privacy-sensitive information. We propose BRANCH, a new LLM methodology that estimates the k-privacy value of a text-the size of the population matching the given information. BRANCH factorizes a joint probability distribution of personal information as random variables. The probability of each factor in a population is estimated separately using a Bayesian network and combined to compute the final k-value. Our experiments show that this method successfully estimates the k-value 73% of the time, a 13% increase compared to o3-mini with chain-of-thought reasoning. We also find that LLM uncertainty is a good indicator for accuracy, as high-variance predictions are 37.47% less accurate on average.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02241v1">Study on LLMs for Promptagator-Style Dense Retriever Training</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 CIKM 2025 short research paper
    </div>
    <details class="paper-abstract">
      Promptagator demonstrated that Large Language Models (LLMs) with few-shot prompts can be used as task-specific query generators for fine-tuning domain-specialized dense retrieval models. However, the original Promptagator approach relied on proprietary and large-scale LLMs which users may not have access to or may be prohibited from using with sensitive data. In this work, we study the impact of open-source LLMs at accessible scales ($\leq$14B parameters) as an alternative. Our results demonstrate that open-source LLMs as small as 3B parameters can serve as effective Promptagator-style query generators. We hope our work will inform practitioners with reliable alternatives for synthetic data generation and give insights to maximize fine-tuning results for domain-specific applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00708v2">DrKGC: Dynamic Subgraph Retrieval-Augmented LLMs for Knowledge Graph Completion across General and Biomedical Domains</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 Accepted at EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Knowledge graph completion (KGC) aims to predict missing triples in knowledge graphs (KGs) by leveraging existing triples and textual information. Recently, generative large language models (LLMs) have been increasingly employed for graph tasks. However, current approaches typically encode graph context in textual form, which fails to fully exploit the potential of LLMs for perceiving and reasoning about graph structures. To address this limitation, we propose DrKGC (Dynamic Subgraph Retrieval-Augmented LLMs for Knowledge Graph Completion). DrKGC employs a flexible lightweight model training strategy to learn structural embeddings and logical rules within the KG. It then leverages a novel bottom-up graph retrieval method to extract a subgraph for each query guided by the learned rules. Finally, a graph convolutional network (GCN) adapter uses the retrieved subgraph to enhance the structural embeddings, which are then integrated into the prompt for effective LLM fine-tuning. Experimental results on two general domain benchmark datasets and two biomedical datasets demonstrate the superior performance of DrKGC. Furthermore, a realistic case study in the biomedical domain highlights its interpretability and practical utility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02209v1">StockBench: Can LLM Agents Trade Stocks Profitably In Real-world Markets?</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently demonstrated strong capabilities as autonomous agents, showing promise in reasoning, tool use, and sequential decision-making. While prior benchmarks have evaluated LLM agents in domains such as software engineering and scientific discovery, the finance domain remains underexplored, despite its direct relevance to economic value and high-stakes decision-making. Existing financial benchmarks primarily test static knowledge through question answering, but they fall short of capturing the dynamic and iterative nature of trading. To address this gap, we introduce StockBench, a contamination-free benchmark designed to evaluate LLM agents in realistic, multi-month stock trading environments. Agents receive daily market signals -- including prices, fundamentals, and news -- and must make sequential buy, sell, or hold decisions. Performance is assessed using financial metrics such as cumulative return, maximum drawdown, and the Sortino ratio. Our evaluation of state-of-the-art proprietary (e.g., GPT-5, Claude-4) and open-weight (e.g., Qwen3, Kimi-K2, GLM-4.5) models shows that while most LLM agents struggle to outperform the simple buy-and-hold baseline, several models demonstrate the potential to deliver higher returns and manage risk more effectively. These findings highlight both the challenges and opportunities in developing LLM-powered financial agents, showing that excelling at static financial knowledge tasks does not necessarily translate into successful trading strategies. We release StockBench as an open-source resource to support reproducibility and advance future research in this domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02200v1">ARUQULA -- An LLM based Text2SPARQL Approach using ReAct and Knowledge Graph Exploration Utilities</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 peer reviewed publication at Text2SPARQL Workshop @ ESWC 2025
    </div>
    <details class="paper-abstract">
      Interacting with knowledge graphs can be a daunting task for people without a background in computer science since the query language that is used (SPARQL) has a high barrier of entry. Large language models (LLMs) can lower that barrier by providing support in the form of Text2SPARQL translation. In this paper we introduce a generalized method based on SPINACH, an LLM backed agent that translates natural language questions to SPARQL queries not in a single shot, but as an iterative process of exploration and execution. We describe the overall architecture and reasoning behind our design decisions, and also conduct a thorough analysis of the agent behavior to gain insights into future areas for targeted improvements. This work was motivated by the Text2SPARQL challenge, a challenge that was held to facilitate improvements in the Text2SPARQL domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21499v2">On Code-Induced Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      Code data has been shown to enhance the reasoning capabilities of large language models (LLMs), but it remains unclear which aspects of code are most responsible. We investigate this question with a systematic, data-centric framework. We construct parallel instruction datasets in ten programming languages and apply controlled perturbations that selectively disrupt structural or semantic properties of code. We then finetune LLMs from five model families and eight scales on each variant and evaluate their performance on natural language, math, and code tasks. Across 3,331 experiments, our results show that LLMs are more vulnerable to structural perturbations than semantic ones, particularly on math and code tasks. Appropriate abstractions like pseudocode and flowcharts can be as effective as code, while encoding the same information with fewer tokens without adhering to original syntax can often retain or even improve performance. Remarkably, even corrupted code with misleading signals remains competitive when surface-level regularities persist. Finally, syntactic styles also shape task-specific gains with Python favoring natural language reasoning and lower-level languages such as Java and Rust favoring math. Through our systematic framework, we aim to provide insight into how different properties of code influence reasoning and inform the design of training data for enhancing LLM reasoning capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15153v2">When Disagreements Elicit Robustness: Investigating Self-Repair Capabilities under LLM Multi-Agent Disagreements</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 Working in progress
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have upgraded them from sophisticated text generators to autonomous agents capable of cooperation and tool use in multi-agent systems (MAS). However, it remains unclear how disagreements shape collective decision-making. In this paper, we revisit the role of disagreement and argue that general, partially overlapping disagreements prevent premature consensus and expand the explored solution space, while disagreements on task-critical steps can derail collaboration depending on the topology of solution paths. We investigate two collaborative settings with distinct path structures: collaborative reasoning (CounterFact, MQuAKE-cf), which typically follows a single evidential chain, whereas collaborative programming (HumanEval, GAIA) often adopts multiple valid implementations. Disagreements are instantiated as general heterogeneity among agents and as task-critical counterfactual knowledge edits injected into context or parameters. Experiments reveal that general disagreements consistently improve success by encouraging complementary exploration. By contrast, task-critical disagreements substantially reduce success on single-path reasoning, yet have a limited impact on programming, where agents can choose alternative solutions. Trace analyses show that MAS frequently bypasses the edited facts in programming but rarely does so in reasoning, revealing an emergent self-repair capability that depends on solution-path rather than scale alone. Our code is available at https://github.com/wbw625/MultiAgentRobustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18379v2">REALM: Recursive Relevance Modeling for LLM-based Document Re-Ranking</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 EMNLP 2025 (Main Conference, Oral). 15 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown strong capabilities in document re-ranking, a key component in modern Information Retrieval (IR) systems. However, existing LLM-based approaches face notable limitations, including ranking uncertainty, unstable top-k recovery, and high token cost due to token-intensive prompting. To effectively address these limitations, we propose REALM, an uncertainty-aware re-ranking framework that models LLM-derived relevance as Gaussian distributions and refines them through recursive Bayesian updates. By explicitly capturing uncertainty and minimizing redundant queries, REALM achieves better rankings more efficiently. Experimental results demonstrate that our REALM surpasses state-of-the-art re-rankers while significantly reducing token usage and latency, improving NDCG@10 by 0.7-11.9 and simultaneously reducing the number of LLM inferences by 23.4-84.4%, promoting it as the next-generation re-ranker for modern IR systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02104v1">LangGrasp: Leveraging Fine-Tuned LLMs for Language Interactive Robot Grasping with Ambiguous Instructions</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 8 pages, 6 figures
    </div>
    <details class="paper-abstract">
      The existing language-driven grasping methods struggle to fully handle ambiguous instructions containing implicit intents. To tackle this challenge, we propose LangGrasp, a novel language-interactive robotic grasping framework. The framework integrates fine-tuned large language models (LLMs) to leverage their robust commonsense understanding and environmental perception capabilities, thereby deducing implicit intents from linguistic instructions and clarifying task requirements along with target manipulation objects. Furthermore, our designed point cloud localization module, guided by 2D part segmentation, enables partial point cloud localization in scenes, thereby extending grasping operations from coarse-grained object-level to fine-grained part-level manipulation. Experimental results show that the LangGrasp framework accurately resolves implicit intents in ambiguous instructions, identifying critical operations and target information that are unstated yet essential for task completion. Additionally, it dynamically selects optimal grasping poses by integrating environmental information. This enables high-precision grasping from object-level to part-level manipulation, significantly enhancing the adaptability and task execution efficiency of robots in unstructured environments. More information and code are available here: https://github.com/wu467/LangGrasp.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02091v1">Demystifying the Roles of LLM Layers in Retrieval, Knowledge, and Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 ICASSP 2025
    </div>
    <details class="paper-abstract">
      Recent studies suggest that the deeper layers of Large Language Models (LLMs) contribute little to representation learning and can often be removed without significant performance loss. However, such claims are typically drawn from narrow evaluations and may overlook important aspects of model behavior. In this work, we present a systematic study of depth utilization across diverse dimensions, including evaluation protocols, task categories, and model architectures. Our analysis confirms that very deep layers are generally less effective than earlier ones, but their contributions vary substantially with the evaluation setting. Under likelihood-based metrics without generation, pruning most layers preserves performance, with only the initial few being critical. By contrast, generation-based evaluation uncovers indispensable roles for middle and deeper layers in enabling reasoning and maintaining long-range coherence. We further find that knowledge and retrieval are concentrated in shallow components, whereas reasoning accuracy relies heavily on deeper layers -- yet can be reshaped through distillation. These results highlight that depth usage in LLMs is highly heterogeneous and context-dependent, underscoring the need for task-, metric-, and model-aware perspectives in both interpreting and compressing large models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.18262v3">Break the ID-Language Barrier: An Adaption Framework for LLM-based Sequential Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      The recent breakthrough of large language models (LLMs) in natural language processing has sparked exploration in recommendation systems, however, their limited domain-specific knowledge remains a critical bottleneck. Specifically, LLMs lack key pieces of information crucial for sequential recommendations, such as user behavior patterns. To address this critical gap, we propose IDLE-Adapter, a novel framework that integrates pre-trained ID embeddings, rich in domain-specific knowledge, into LLMs to improve recommendation accuracy. IDLE-Adapter acts as a bridge, transforming sparse user-item interaction data into dense, LLM-compatible representations through a Pre-trained ID Sequential Model, Dimensionality Alignment, Layer-wise Embedding Refinement, and Layer-wise Distribution Alignment. Furthermore, IDLE-Adapter demonstrates remarkable flexibility by seamlessly integrating ID embeddings from diverse ID-based sequential models and LLM architectures. Extensive experiments across various datasets demonstrate the superiority of IDLE-Adapter, achieving over 10\% and 20\% improvements in HitRate@5 and NDCG@5 metrics, respectively, compared to state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02047v1">LLM-Enhanced, Data-Driven Personalized and Equitable Clinician Scheduling: A Predict-then-Optimize Approach</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 10 pages, 5 figures, Accepted to IEEE ICDM 2025 Workshops Proceedings; IEEE Computer Society Press
    </div>
    <details class="paper-abstract">
      Clinician scheduling remains a persistent challenge due to limited clinical resources and fluctuating demands. This complexity is especially acute in large academic anesthesiology departments as physicians balance responsibilities across multiple clinical sites with conflicting priorities. Further, scheduling must account for individual clinical and lifestyle preferences to ensure job satisfaction and well-being. Traditional approaches, often based on statistical or rule-based optimization models, rely on structured data and explicit domain knowledge. However, these methods often overlook unstructured information, e.g., free-text notes from routinely administered clinician well-being surveys and scheduling platforms. These notes may reveal implicit and underutilized clinical resources. Neglecting such information can lead to misaligned schedules, increased burnout, overlooked staffing flexibility, and suboptimal utilization of available resources. To address this gap, we propose a predict-then-optimize framework that integrates classification-based clinician availability predictions with a mixed-integer programming schedule optimization model. Large language models (LLMs) are employed to extract actionable preferences and implicit constraints from unstructured schedule notes, enhancing the reliability of availability predictions. These predictions then inform the schedule optimization considering four objectives: first, ensuring clinical full-time equivalent compliance, second, reducing workload imbalances by enforcing equitable proportions of shift types, third, maximizing clinician availability for assigned shifts, and fourth, schedule consistency. By combining the interpretive power of LLMs with the rigor of mathematical optimization, our framework provides a robust, data-driven solution that enhances operational efficiency while supporting equity and clinician well-being.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05735v3">Do LLMs Really Forget? Evaluating Unlearning with Knowledge Correlation and Confidence Awareness</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      Machine unlearning techniques aim to mitigate unintended memorization in large language models (LLMs). However, existing approaches predominantly focus on the explicit removal of isolated facts, often overlooking latent inferential dependencies and the non-deterministic nature of knowledge within LLMs. Consequently, facts presumed forgotten may persist implicitly through correlated information. To address these challenges, we propose a knowledge unlearning evaluation framework that more accurately captures the implicit structure of real-world knowledge by representing relevant factual contexts as knowledge graphs with associated confidence scores. We further develop an inference-based evaluation protocol leveraging powerful LLMs as judges; these judges reason over the extracted knowledge subgraph to determine unlearning success. Our LLM judges utilize carefully designed prompts and are calibrated against human evaluations to ensure their trustworthiness and stability. Extensive experiments on our newly constructed benchmark demonstrate that our framework provides a more realistic and rigorous assessment of unlearning performance. Moreover, our findings reveal that current evaluation strategies tend to overestimate unlearning effectiveness. Our code is publicly available at https://github.com/Graph-COM/Knowledge_Unlearning.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17674v2">Push the Limit of Multi-modal Emotion Recognition by Prompting LLMs with Receptive-Field-Aware Attention Weighting</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      Understanding the emotions in a dialogue usually requires external knowledge to accurately understand the contents. As the LLMs become more and more powerful, we do not want to settle on the limited ability of the pre-trained language model. However, the LLMs either can only process text modality or are too expensive to process the multimedia information. We aim to utilize both the power of LLMs and the supplementary features from the multimedia modalities. In this paper, we present a framework, Lantern, that can improve the performance of a certain vanilla model by prompting large language models with receptive-field-aware attention weighting. This framework trained a multi-task vanilla model to produce probabilities of emotion classes and dimension scores. These predictions are fed into the LLMs as references to adjust the predicted probabilities of each emotion class with its external knowledge and contextual understanding. We slice the dialogue into different receptive fields, and each sample is included in exactly t receptive fields. Finally, the predictions of LLMs are merged with a receptive-field-aware attention-driven weighting module. In the experiments, vanilla models CORECT and SDT are deployed in Lantern with GPT-4 or Llama-3.1-405B. The experiments in IEMOCAP with 4-way and 6-way settings demonstrated that the Lantern can significantly improve the performance of current vanilla models by up to 1.23% and 1.80%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25264v2">GeoSQL-Eval: First Evaluation of LLMs on PostGIS-Based NL2GeoSQL Queries</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown strong performance in natural language to SQL (NL2SQL) tasks within general databases. However, extending to GeoSQL introduces additional complexity from spatial data types, function invocation, and coordinate systems, which greatly increases generation and execution difficulty. Existing benchmarks mainly target general SQL, and a systematic evaluation framework for GeoSQL is still lacking. To fill this gap, we present GeoSQL-Eval, the first end-to-end automated evaluation framework for PostGIS query generation, together with GeoSQL-Bench, a benchmark for assessing LLM performance in NL2GeoSQL tasks. GeoSQL-Bench defines three task categories-conceptual understanding, syntax-level SQL generation, and schema retrieval-comprising 14,178 instances, 340 PostGIS functions, and 82 thematic databases. GeoSQL-Eval is grounded in Webb's Depth of Knowledge (DOK) model, covering four cognitive dimensions, five capability levels, and twenty task types to establish a comprehensive process from knowledge acquisition and syntax generation to semantic alignment, execution accuracy, and robustness. We evaluate 24 representative models across six categories and apply the entropy weight method with statistical analyses to uncover performance differences, common error patterns, and resource usage. Finally, we release a public GeoSQL-Eval leaderboard platform for continuous testing and global comparison. This work extends the NL2GeoSQL paradigm and provides a standardized, interpretable, and extensible framework for evaluating LLMs in spatial database contexts, offering valuable references for geospatial information science and related applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01995v1">LLM-Based Multi-Task Bangla Hate Speech Detection: Type, Severity, and Target</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      Online social media platforms are central to everyday communication and information seeking. While these platforms serve positive purposes, they also provide fertile ground for the spread of hate speech, offensive language, and bullying content targeting individuals, organizations, and communities. Such content undermines safety, participation, and equity online. Reliable detection systems are therefore needed, especially for low-resource languages where moderation tools are limited. In Bangla, prior work has contributed resources and models, but most are single-task (e.g., binary hate/offense) with limited coverage of multi-facet signals (type, severity, target). We address these gaps by introducing the first multi-task Bangla hate-speech dataset, BanglaMultiHate, one of the largest manually annotated corpus to date. Building on this resource, we conduct a comprehensive, controlled comparison spanning classical baselines, monolingual pretrained models, and LLMs under zero-shot prompting and LoRA fine-tuning. Our experiments assess LLM adaptability in a low-resource setting and reveal a consistent trend: although LoRA-tuned LLMs are competitive with BanglaBERT, culturally and linguistically grounded pretraining remains critical for robust performance. Together, our dataset and findings establish a stronger benchmark for developing culturally aligned moderation tools in low-resource contexts. For reproducibility, we will release the dataset and all related scripts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11997v2">Can LLMs Find Fraudsters? Multi-level LLM Enhanced Graph Fraud Detection</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 Accepted by ACM MM 2025
    </div>
    <details class="paper-abstract">
      Graph fraud detection has garnered significant attention as Graph Neural Networks (GNNs) have proven effective in modeling complex relationships within multimodal data. However, existing graph fraud detection methods typically use preprocessed node embeddings and predefined graph structures to reveal fraudsters, which ignore the rich semantic cues contained in raw textual information. Although Large Language Models (LLMs) exhibit powerful capabilities in processing textual information, it remains a significant challenge to perform multimodal fusion of processed textual embeddings with graph structures. In this paper, we propose a \textbf{M}ulti-level \textbf{L}LM \textbf{E}nhanced Graph Fraud \textbf{D}etection framework called MLED. In MLED, we utilize LLMs to extract external knowledge from textual information to enhance graph fraud detection methods. To integrate LLMs with graph structure information and enhance the ability to distinguish fraudsters, we design a multi-level LLM enhanced framework including type-level enhancer and relation-level enhancer. One is to enhance the difference between the fraudsters and the benign entities, the other is to enhance the importance of the fraudsters in different relations. The experiments on four real-world datasets show that MLED achieves state-of-the-art performance in graph fraud detection as a generalized framework that can be applied to existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01929v1">Inverse Language Modeling towards Robust and Grounded LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      The current landscape of defensive mechanisms for LLMs is fragmented and underdeveloped, unlike prior work on classifiers. To further promote adversarial robustness in LLMs, we propose Inverse Language Modeling (ILM), a unified framework that simultaneously 1) improves the robustness of LLMs to input perturbations, and, at the same time, 2) enables native grounding by inverting model outputs to identify potentially toxic or unsafe input triggers. ILM transforms LLMs from static generators into analyzable and robust systems, potentially helping RED teaming. ILM can lay the foundation for next-generation LLMs that are not only robust and grounded but also fundamentally more controllable and trustworthy. The code is publicly available at github.com/davegabe/pag-llm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.22255v3">Evaluating LLMs for Combinatorial Optimization: One-Phase and Two-Phase Heuristics for 2D Bin-Packing</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 1 table, 6 figures. 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Accepted for the Workshop: Evaluating the Evolving LLM Lifecycle Benchmarks, Emergent Abilities, and Scaling
    </div>
    <details class="paper-abstract">
      This paper presents an evaluation framework for assessing Large Language Models' (LLMs) capabilities in combinatorial optimization, specifically addressing the 2D bin-packing problem. We introduce a systematic methodology that combines LLMs with evolutionary algorithms to generate and refine heuristic solutions iteratively. Through comprehensive experiments comparing LLM generated heuristics against traditional approaches (Finite First-Fit and Hybrid First-Fit), we demonstrate that LLMs can produce more efficient solutions while requiring fewer computational resources. Our evaluation reveals that GPT-4o achieves optimal solutions within two iterations, reducing average bin usage from 16 to 15 bins while improving space utilization from 0.76-0.78 to 0.83. This work contributes to understanding LLM evaluation in specialized domains and establishes benchmarks for assessing LLM performance in combinatorial optimization tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14245v2">Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 Update with more experiments
    </div>
    <details class="paper-abstract">
      Recent advancements in long chain-of-thought (CoT) reasoning, particularly through the Group Relative Policy Optimization algorithm used by DeepSeek-R1, have led to significant interest in the potential of Reinforcement Learning with Verifiable Rewards (RLVR) for Large Language Models (LLMs). While RLVR promises to improve reasoning by allowing models to learn from free exploration, there remains debate over whether it truly enhances reasoning abilities or simply boosts sampling efficiency. This paper systematically investigates the impact of RLVR on LLM reasoning. We revisit Pass@K experiments and demonstrate that RLVR can extend the reasoning boundary for both mathematical and coding tasks. This is supported by our introduction of a novel evaluation metric, CoT-Pass@K, which captures reasoning success by accounting for both the final answer and intermediate reasoning steps. Furthermore, we present a theoretical framework explaining RLVR's incentive mechanism, demonstrating how it can encourage correct reasoning even when rewards are based solely on answer correctness. Our analysis of RLVR's training dynamics reveals that it incentivizes correct reasoning early in the process, with substantial improvements in reasoning quality confirmed through extensive evaluations. These findings provide strong evidence of RLVR's potential to enhance LLM reasoning, offering valuable insights into its mechanisms and performance improvements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01910v1">Are LLMs Better GNN Helpers? Rethinking Robust Graph Learning under Deficiencies with Iterative Refinement</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 14 pages
    </div>
    <details class="paper-abstract">
      Graph Neural Networks (GNNs) are widely adopted in Web-related applications, serving as a core technique for learning from graph-structured data, such as text-attributed graphs. Yet in real-world scenarios, such graphs exhibit deficiencies that substantially undermine GNN performance. While prior GNN-based augmentation studies have explored robustness against individual imperfections, a systematic understanding of how graph-native and Large Language Models (LLMs) enhanced methods behave under compound deficiencies is still missing. Specifically, there has been no comprehensive investigation comparing conventional approaches and recent LLM-on-graph frameworks, leaving their merits unclear. To fill this gap, we conduct the first empirical study that benchmarks these two lines of methods across diverse graph deficiencies, revealing overlooked vulnerabilities and challenging the assumption that LLM augmentation is consistently superior. Building on empirical findings, we propose Robust Graph Learning via Retrieval-Augmented Contrastive Refinement (RoGRAD) framework. Unlike prior one-shot LLM-as-Enhancer designs, RoGRAD is the first iterative paradigm that leverages Retrieval-Augmented Generation (RAG) to inject retrieval-grounded augmentations by supplying class-consistent, diverse augmentations and enforcing discriminative representations through iterative graph contrastive learning. It transforms LLM augmentation for graphs from static signal injection into dynamic refinement. Extensive experiments demonstrate RoGRAD's superiority over both conventional GNN- and LLM-enhanced baselines, achieving up to 82.43% average improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23281v2">MathArena: Evaluating LLMs on Uncontaminated Math Competitions</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      The rapid advancement of reasoning capabilities in large language models (LLMs) has led to notable improvements on mathematical benchmarks. However, many of the most commonly used evaluation datasets (e.g., AIME 2024) are widely available online, making it difficult to disentangle genuine reasoning from potential memorization. Furthermore, these benchmarks do not evaluate proof-writing capabilities, which are crucial for many mathematical tasks. To address this, we introduce MathArena, a new benchmark based on the following key insight: recurring math competitions provide a stream of high-quality, challenging problems that can be used for real-time evaluation of LLMs. By evaluating models as soon as new problems are released, we effectively eliminate the risk of contamination. Using this framework, we find strong signs of contamination in AIME 2024. Nonetheless, evaluations on harder competitions, such as CMIMC 2025, demonstrate impressive reasoning capabilities in top-performing models. MathArena is also the first benchmark for proof-writing capabilities. On IMO 2025, top models achieve slightly less than 40%, demonstrating both notable progress and significant room for improvement. So far, we have evaluated over $50$ models across seven competitions, totaling $162$ problems. As an evolving benchmark, MathArena will continue to track the progress of LLMs on newly released competitions, ensuring rigorous and up-to-date evaluation of mathematical reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01842v1">Pre-Hoc Predictions in AutoML: Leveraging LLMs to Enhance Model Selection and Benchmarking for Tabular datasets</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 Oral Presentations ADAPT Annual Scientific Conference 2025
    </div>
    <details class="paper-abstract">
      The field of AutoML has made remarkable progress in post-hoc model selection, with libraries capable of automatically identifying the most performing models for a given dataset. Nevertheless, these methods often rely on exhaustive hyperparameter searches, where methods automatically train and test different types of models on the target dataset. Contrastingly, pre-hoc prediction emerges as a promising alternative, capable of bypassing exhaustive search through intelligent pre-selection of models. Despite its potential, pre-hoc prediction remains under-explored in the literature. This paper explores the intersection of AutoML and pre-hoc model selection by leveraging traditional models and Large Language Model (LLM) agents to reduce the search space of AutoML libraries. By relying on dataset descriptions and statistical information, we reduce the AutoML search space. Our methodology is applied to the AWS AutoGluon portfolio dataset, a state-of-the-art AutoML benchmark containing 175 tabular classification datasets available on OpenML. The proposed approach offers a shift in AutoML workflows, significantly reducing computational overhead, while still selecting the best model for the given dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01833v1">Plan Then Action:High-Level Planning Guidance Reinforcement Learning for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 19 pages and 5 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable reasoning abilities in complex tasks, often relying on Chain-of-Thought (CoT) reasoning. However, due to their autoregressive token-level generation, the reasoning process is largely constrained to local decision-making and lacks global planning. This limitation frequently results in redundant, incoherent, or inaccurate reasoning, which significantly degrades overall performance. Existing approaches, such as tree-based algorithms and reinforcement learning (RL), attempt to address this issue but suffer from high computational costs and often fail to produce optimal reasoning trajectories. To tackle this challenge, we propose Plan-Then-Action Enhanced Reasoning with Group Relative Policy Optimization PTA-GRPO, a two-stage framework designed to improve both high-level planning and fine-grained CoT reasoning. In the first stage, we leverage advanced LLMs to distill CoT into compact high-level guidance, which is then used for supervised fine-tuning (SFT). In the second stage, we introduce a guidance-aware RL method that jointly optimizes the final output and the quality of high-level guidance, thereby enhancing reasoning effectiveness. We conduct extensive experiments on multiple mathematical reasoning benchmarks, including MATH, AIME2024, AIME2025, and AMC, across diverse base models such as Qwen2.5-7B-Instruct, Qwen3-8B, Qwen3-14B, and LLaMA3.2-3B. Experimental results demonstrate that PTA-GRPO consistently achieves stable and significant improvements across different models and tasks, validating its effectiveness and generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01831v1">Syntactic Blind Spots: How Misalignment Leads to LLMs Mathematical Errors</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 14 pages, 5 Tables, 9 Figures; Accepted to MathNLP 2025: The 3rd Workshop on Mathematical Natural Language Processing (co-located with EMNLP 2025)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate strong mathematical problem-solving abilities but frequently fail on problems that deviate syntactically from their training distribution. We identify a systematic failure mode, syntactic blind spots, in which models misapply familiar reasoning strategies to problems that are semantically straightforward but phrased in unfamiliar ways. These errors are not due to gaps in mathematical competence, but rather reflect a brittle coupling between surface form and internal representation. To test this, we rephrase incorrectly answered questions using syntactic templates drawn from correct examples. These rephrasings, which preserve semantics while reducing structural complexity, often lead to correct answers. We quantify syntactic complexity using a metric based on Dependency Locality Theory (DLT), and show that higher DLT scores are associated with increased failure rates across multiple datasets. Our findings suggest that many reasoning errors stem from structural misalignment rather than conceptual difficulty, and that syntax-aware interventions can reveal and mitigate these inductive failures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01801v1">Detecting LLM-Generated Spam Reviews by Integrating Language Model Embeddings and Graph Neural Network</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      The rise of large language models (LLMs) has enabled the generation of highly persuasive spam reviews that closely mimic human writing. These reviews pose significant challenges for existing detection systems and threaten the credibility of online platforms. In this work, we first create three realistic LLM-generated spam review datasets using three distinct LLMs, each guided by product metadata and genuine reference reviews. Evaluations by GPT-4.1 confirm the high persuasion and deceptive potential of these reviews. To address this threat, we propose FraudSquad, a hybrid detection model that integrates text embeddings from a pre-trained language model with a gated graph transformer for spam node classification. FraudSquad captures both semantic and behavioral signals without relying on manual feature engineering or massive training resources. Experiments show that FraudSquad outperforms state-of-the-art baselines by up to 44.22% in precision and 43.01% in recall on three LLM-generated datasets, while also achieving promising results on two human-written spam datasets. Furthermore, FraudSquad maintains a modest model size and requires minimal labeled training data, making it a practical solution for real-world applications. Our contributions include new synthetic datasets, a practical detection framework, and empirical evidence highlighting the urgency of adapting spam detection to the LLM era. Our code and datasets are available at: https://anonymous.4open.science/r/FraudSquad-5389/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01782v1">Can LLMs Refuse Questions They Do Not Know? Measuring Knowledge-Aware Refusal in Factual Tasks</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) should refuse to answer questions beyond their knowledge. This capability, which we term knowledge-aware refusal, is crucial for factual reliability. However, existing metrics fail to faithfully measure this ability. On the one hand, simple refusal-based metrics are biased by refusal rates and yield inconsistent scores when models exhibit different refusal tendencies. On the other hand, existing calibration metrics are proxy-based, capturing the performance of auxiliary calibration processes rather than the model's actual refusal behavior. In this work, we propose the Refusal Index (RI), a principled metric that measures how accurately LLMs refuse questions they do not know. We define RI as Spearman's rank correlation between refusal probability and error probability. To make RI practically measurable, we design a lightweight two-pass evaluation method that efficiently estimates RI from observed refusal rates across two standard evaluation runs. Extensive experiments across 16 models and 5 datasets demonstrate that RI accurately quantifies a model's intrinsic knowledge-aware refusal capability in factual tasks. Notably, RI remains stable across different refusal rates and provides consistent model rankings independent of a model's overall accuracy and refusal rates. More importantly, RI provides insight into an important but previously overlooked aspect of LLM factuality: while LLMs achieve high accuracy on factual tasks, their refusal behavior can be unreliable and fragile. This finding highlights the need to complement traditional accuracy metrics with the Refusal Index for comprehensive factuality evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00761v2">Downgrade to Upgrade: Optimizer Simplification Enhances Robustness in LLM Unlearning</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      Large language model (LLM) unlearning aims to surgically remove the influence of undesired data or knowledge from an existing model while preserving its utility on unrelated tasks. This paradigm has shown promise in addressing privacy and safety concerns. However, recent findings reveal that unlearning effects are often fragile: post-unlearning manipulations such as weight quantization or fine-tuning can quickly neutralize the intended forgetting. Prior efforts to improve robustness primarily reformulate unlearning objectives by explicitly assuming the role of vulnerability sources. In this work, we take a different perspective by investigating the role of the optimizer, independent of unlearning objectives and formulations, in shaping unlearning robustness. We show that the 'grade' of the optimizer, defined by the level of information it exploits, ranging from zeroth-order (gradient-free) to first-order (gradient-based) to second-order (Hessian-based), is tightly linked to the resilience of unlearning. Surprisingly, we find that downgrading the optimizer, such as using zeroth-order methods or compressed-gradient variants (e.g., gradient sign-based optimizers), often leads to stronger robustness. While these optimizers produce noisier and less precise updates, they encourage convergence to harder-to-disturb basins in the loss landscape, thereby resisting post-training perturbations. By connecting zeroth-order methods with randomized smoothing, we further highlight their natural advantage for robust unlearning. Motivated by these insights, we propose a hybrid optimizer that combines first-order and zeroth-order updates, preserving unlearning efficacy while enhancing robustness. Extensive experiments on the MUSE and WMDP benchmarks, across multiple LLM unlearning algorithms, validate that our approach achieves more resilient forgetting without sacrificing unlearning quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.14459v2">Reasoning over User Preferences: Knowledge Graph-Augmented LLMs for Explainable Conversational Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 Accepted by ICDM 2025
    </div>
    <details class="paper-abstract">
      Conversational Recommender Systems (CRSs) aim to provide personalized recommendations by capturing user preferences through interactive dialogues. Explainability in CRSs is crucial as it enables users to understand the reasoning behind recommendations, increasing system transparency and trustworthiness. However, current CRSs often leverage knowledge graphs (KGs) or language models to extract and represent user preferences as latent vectors, which limits their explainability. Large language models (LLMs) offer powerful reasoning capabilities that can bridge this gap by generating human-understandable preference summaries. However, effectively reasoning over user preferences in CRSs remains challenging as LLMs pre-trained on large-scale corpora may not be well-suited for analyzing user preferences. While KGs provide rich domain knowledge, integrating them with LLMs encounters a significant modality gap between structured KG information and unstructured conversations. In this paper, we propose COMPASS, a plug-and-play framework that synergizes LLMs and KGs to reason over user preferences, enhancing the performance and explainability of existing CRSs. COMPASS employs a two-stage training approach: first, it bridges the gap between the structured KG and natural language through novel graph entity captioning pre-training. Next, COMPASS optimizes user preference reasoning via knowledge-aware instruction fine-tuning, where the LLM learns to reason and summarize user preferences from dialogue histories and KG-augmented context. This enables COMPASS to perform knowledge-aware reasoning and generate interpretable user preferences that can seamlessly integrate with existing CRS models for improving recommendation performance and explainability. Our experiments on benchmark datasets demonstrate the effectiveness of COMPASS in improving various CRS models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01698v1">TalkPlay-Tools: Conversational Music Recommendation with LLM Tool Calling</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 Accepted for publication at The Workshop on AI for Music, Neural Information Processing Systems (NeurIPS-AI4Music)
    </div>
    <details class="paper-abstract">
      While the recent developments in large language models (LLMs) have successfully enabled generative recommenders with natural language interactions, their recommendation behavior is limited, leaving other simpler yet crucial components such as metadata or attribute filtering underutilized in the system. We propose an LLM-based music recommendation system with tool calling to serve as a unified retrieval-reranking pipeline. Our system positions an LLM as an end-to-end recommendation system that interprets user intent, plans tool invocations, and orchestrates specialized components: boolean filters (SQL), sparse retrieval (BM25), dense retrieval (embedding similarity), and generative retrieval (semantic IDs). Through tool planning, the system predicts which types of tools to use, their execution order, and the arguments needed to find music matching user preferences, supporting diverse modalities while seamlessly integrating multiple database filtering methods. We demonstrate that this unified tool-calling framework achieves competitive performance across diverse recommendation scenarios by selectively employing appropriate retrieval methods based on user queries, envisioning a new paradigm for conversational music recommendation systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24858v2">MetaFaith: Faithful Natural Language Uncertainty Expression in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 EMNLP 2025
    </div>
    <details class="paper-abstract">
      A critical component in the trustworthiness of LLMs is reliable uncertainty communication, yet LLMs often use assertive language when conveying false claims, leading to over-reliance and eroded trust. We present the first systematic study of $\textit{faithful confidence calibration}$ of LLMs, benchmarking models' ability to use linguistic expressions of uncertainty that $\textit{faithfully reflect}$ their intrinsic uncertainty, across a comprehensive array of models, datasets, and prompting strategies. Our results demonstrate that LLMs largely fail at this task, and that existing interventions are insufficient: standard prompt approaches provide only marginal gains, and existing, factuality-based calibration techniques can even harm faithful calibration. To address this critical gap, we introduce MetaFaith, a novel prompt-based calibration approach inspired by human metacognition. We show that MetaFaith robustly improves faithful calibration across diverse models and task domains, enabling up to 61% improvement in faithfulness and achieving an 83% win rate over original generations as judged by humans.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01688v1">Format Inertia: A Failure Mechanism of LLMs in Medical Pre-Consultation</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 EMNLP 2025 Industry Track
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have brought significant improvements to various service domains, including chatbots and medical pre-consultation applications. In the healthcare domain, the most common approach for adapting LLMs to multi-turn dialogue generation is Supervised Fine-Tuning (SFT). However, datasets for SFT in tasks like medical pre-consultation typically exhibit a skewed turn-count distribution. Training on such data induces a novel failure mechanism we term **Format Inertia**, where models tend to generate repetitive, format-correct, but diagnostically uninformative questions in long medical dialogues. To mitigate this observed failure mechanism, we adopt a simple, data-centric method that rebalances the turn-count distribution of the training dataset. Experimental results show that our approach substantially alleviates Format Inertia in medical pre-consultation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.12885v2">VAR-MATH: Probing True Mathematical Reasoning in LLMS via Symbolic Multi-Instance Benchmarks</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      Recent advances in reinforcement learning (RL) have led to substantial improvements in the mathematical reasoning abilities of LLMs, as measured by standard benchmarks. Yet these gains often persist even when models are trained with flawed signals, such as random or inverted rewards. This raises a fundamental question: do such improvements reflect genuine reasoning, or are they merely artifacts of overfitting to benchmark-specific patterns? To answer this question, we adopt an evaluation-centric perspective and highlight two critical shortcomings in existing protocols. First, benchmark contamination arises because test problems are publicly available, thereby increasing the risk of data leakage. Second, evaluation fragility results from reliance on single-instance assessments, which are sensitive to stochastic outputs and fail to capture reasoning consistency. These limitations suggest the need for a new evaluation paradigm that can probe reasoning ability beyond memorization and one-off success. As response, we propose VAR-MATH, a symbolic evaluation framework that converts fixed numerical problems into parameterized templates and requires models to solve multiple instantiations of each. This design enforces consistency across structurally equivalent variants, mitigates contamination, and enhances robustness through bootstrapped metrics. We apply VAR-MATH to transform three popular benchmarks, AMC23, AIME24, and AIME25, into their symbolic counterparts, VAR-AMC23, VAR-AIME24, and VAR-AIME25. Experimental results show substantial performance drops for RL-trained models on these variabilized benchmarks, especially for smaller models, with average declines of 47.9\% on AMC23, 58.8\% on AIME24, and 72.9\% on AIME25. These findings indicate that some existing RL methods rely on superficial heuristics and fail to generalize beyond specific numerical forms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01664v1">GuruAgents: Emulating Wise Investors with Prompt-Guided LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 7 Pages, 2 figures
    </div>
    <details class="paper-abstract">
      This study demonstrates that GuruAgents, prompt-guided AI agents, can systematically operationalize the strategies of legendary investment gurus. We develop five distinct GuruAgents, each designed to emulate an iconic investor, by encoding their distinct philosophies into LLM prompts that integrate financial tools and a deterministic reasoning pipeline. In a backtest on NASDAQ-100 constituents from Q4 2023 to Q2 2025, the GuruAgents exhibit unique behaviors driven by their prompted personas. The Buffett GuruAgent achieves the highest performance, delivering a 42.2\% CAGR that significantly outperforms benchmarks, while other agents show varied results. These findings confirm that prompt engineering can successfully translate the qualitative philosophies of investment gurus into reproducible, quantitative strategies, highlighting a novel direction for automated systematic investing. The source code and data are available at https://github.com/yejining99/GuruAgents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01656v1">Asymmetric Proximal Policy Optimization: mini-critics boost LLM reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      Most recent RL for LLMs (RL4LLM) methods avoid explicit critics, replacing them with average advantage baselines. This shift is largely pragmatic: conventional value functions are computationally expensive to train at LLM scale and often fail under sparse rewards and long reasoning horizons. We revisit this bottleneck from an architectural perspective and introduce Asymmetric Proximal Policy Optimization (AsyPPO), a simple and scalable framework that restores the critics role while remaining efficient in large-model settings. AsyPPO employs a set of lightweight mini-critics, each trained on disjoint prompt shards. This design encourages diversity while preserving calibration, reducing value-estimation bias. Beyond robust estimation, AsyPPO leverages inter-critic uncertainty to refine the policy update: (i) masking advantages in states where critics agree and gradients add little learning signal, and (ii) filtering high-divergence states from entropy regularization, suppressing spurious exploration. After training on open-source data with only 5,000 samples, AsyPPO consistently improves learning stability and performance across multiple benchmarks over strong baselines, such as GRPO, achieving performance gains of more than six percent on Qwen3-4b-Base and about three percent on Qwen3-8b-Base and Qwen3-14b-Base over classic PPO, without additional tricks. These results highlight the importance of architectural innovations for scalable, efficient algorithms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01652v1">Learning to Look at the Other Side: A Semantic Probing Study of Word Embeddings in LLMs with Enabled Bidirectional Attention</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      Autoregressive Large Language Models (LLMs) demonstrate exceptional performance in language understanding and generation. However, their application in text embedding tasks has been relatively slow, along with the analysis of their semantic representation in probing tasks, due to the constraints of the unidirectional attention mechanism. This paper aims to explore whether such constraints can be overcome by enabling bidirectional attention in LLMs. We tested different variants of the Llama architecture through additional training steps, progressively enabling bidirectional attention and unsupervised/supervised contrastive learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.26306v3">Interactive Learning for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 The code is available at https://github.com/linhh29/Interactive-Learning-for-LLM-Reasoning
    </div>
    <details class="paper-abstract">
      Existing multi-agent learning approaches have developed interactive training environments to explicitly promote collaboration among multiple Large Language Models (LLMs), thereby constructing stronger multi-agent systems (MAS). However, during inference, they require re-executing the MAS to obtain final solutions, which diverges from human cognition that individuals can enhance their reasoning capabilities through interactions with others and resolve questions independently in the future. To investigate whether multi-agent interaction can enhance LLMs' independent problem-solving ability, we introduce ILR, a novel co-learning framework for MAS that integrates two key components: Dynamic Interaction and Perception Calibration. Specifically, Dynamic Interaction first adaptively selects either cooperative or competitive strategies depending on question difficulty and model ability. LLMs then exchange information through Idea3 (Idea Sharing, Idea Analysis, and Idea Fusion), an innovative interaction paradigm designed to mimic human discussion, before deriving their respective final answers. In Perception Calibration, ILR employs Group Relative Policy Optimization (GRPO) to train LLMs while integrating one LLM's reward distribution characteristics into another's reward function, thereby enhancing the cohesion of multi-agent interactions. We validate ILR on three LLMs across two model families of varying scales, evaluating performance on five mathematical benchmarks and one coding benchmark. Experimental results show that ILR consistently outperforms single-agent learning, yielding an improvement of up to 5% over the strongest baseline. We further discover that Idea3 can enhance the robustness of stronger LLMs during multi-agent inference, and dynamic interaction types can boost multi-agent learning compared to pure cooperative or competitive strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01650v1">The Unseen Frontier: Pushing the Limits of LLM Sparsity with Surrogate-Free ADMM</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Neural network pruning is a promising technique to mitigate the excessive computational and memory requirements of large language models (LLMs). Despite its promise, however, progress in this area has diminished, as conventional methods are seemingly unable to surpass moderate sparsity levels (50-60%) without severely degrading model accuracy. This work breaks through the current impasse, presenting a principled and effective method called $\texttt{Elsa}$, which achieves extreme sparsity levels of up to 90% while retaining high model fidelity. This is done by identifying several limitations in current practice, all of which can be traced back to their reliance on a surrogate objective formulation. $\texttt{Elsa}$ tackles this issue directly and effectively via standard and well-established constrained optimization techniques based on ADMM. Our extensive experiments across a wide range of models and scales show that $\texttt{Elsa}$ achieves substantial improvements over existing methods; e.g., it achieves 7.8$\times$ less perplexity than the best existing method on LLaMA-2-7B at 90% sparsity. Furthermore, we present $\texttt{Elsa}_{\text{-L}}$, a quantized variant that scales to extremely large models (27B), and establish its theoretical convergence guarantees. These results highlight meaningful progress in advancing the frontier of LLM sparsity, while promising that significant opportunities for further advancement may remain in directions that have so far attracted limited exploration.
    </details>
</div>
