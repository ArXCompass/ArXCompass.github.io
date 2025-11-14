# llm - 2025_05

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- Part 11
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)
- [Part 15](papers_15.md)
- [Part 16](papers_16.md)
- [Part 17](papers_17.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14918v1">Reliable Decision Support with LLMs: A Framework for Evaluating Consistency in Binary Text Classification Applications</a></div>
    <div class="paper-meta">
      📅 2025-05-20
      | 💬 25 pages
    </div>
    <details class="paper-abstract">
      This study introduces a framework for evaluating consistency in large language model (LLM) binary text classification, addressing the lack of established reliability assessment methods. Adapting psychometric principles, we determine sample size requirements, develop metrics for invalid responses, and evaluate intra- and inter-rater reliability. Our case study examines financial news sentiment classification across 14 LLMs (including claude-3-7-sonnet, gpt-4o, deepseek-r1, gemma3, llama3.2, phi4, and command-r-plus), with five replicates per model on 1,350 articles. Models demonstrated high intra-rater consistency, achieving perfect agreement on 90-98% of examples, with minimal differences between expensive and economical models from the same families. When validated against StockNewsAPI labels, models achieved strong performance (accuracy 0.76-0.88), with smaller models like gemma3:1B, llama3.2:3B, and claude-3-5-haiku outperforming larger counterparts. All models performed at chance when predicting actual market movements, indicating task constraints rather than model limitations. Our framework provides systematic guidance for LLM selection, sample size planning, and reliability assessment, enabling organizations to optimize resources for classification tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14899v1">Think, Reflect, Create: Metacognitive Learning for Zero-Shot Robotic Planning with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-20
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have shown great potential across various domains, their applications in robotics remain largely limited to static, prompt-based behaviors and still face challenges in handling complex tasks under zero-shot or few-shot settings. Inspired by human metacognitive learning and creative problem-solving, we address this limitation by exploring a fundamental research question: Can LLMs be empowered with metacognitive capabilities to reason, reflect, and create, thereby enhancing their ability to perform robotic tasks with minimal demonstrations? In this paper, we present an early-stage framework that integrates metacognitive learning into LLM-powered multi-robot collaboration. The proposed framework equips the LLM-powered robotic agents with a skill decomposition and self-reflection mechanism that identifies modular skills from prior tasks, reflects on failures in unseen task scenarios, and synthesizes effective new solutions. Experimental results show that our metacognitive-learning-empowered LLM framework significantly outperforms existing baselines. Moreover, we observe that the framework is capable of generating solutions that differ from the ground truth yet still successfully complete the tasks. These exciting findings support our hypothesis that metacognitive learning can foster creativity in robotic planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19721v2">Sensing and Steering Stereotypes: Extracting and Applying Gender Representation Vectors in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are known to perpetuate stereotypes and exhibit biases. Various strategies have been proposed to mitigate these biases, but most work studies biases in LLMs as a black-box problem without considering how concepts are represented within the model. We adapt techniques from representation engineering to study how the concept of "gender" is represented within LLMs. We introduce a new method that extracts concept representations via probability weighting without labeled data and efficiently selects a steering vector for measuring and manipulating the model's representation. We also present a projection-based method that enables precise steering of model predictions and demonstrate its effectiveness in mitigating gender bias in LLMs. Our code is available at: https://github.com/hannahxchen/gender-bias-steering
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14884v1">Polar Sparsity: High Throughput Batched LLM Inferencing with Scalable Contextual Sparsity</a></div>
    <div class="paper-meta">
      📅 2025-05-20
    </div>
    <details class="paper-abstract">
      Accelerating large language model (LLM) inference is critical for real-world deployments requiring high throughput and low latency. Contextual sparsity, where each token dynamically activates only a small subset of the model parameters, shows promise but does not scale to large batch sizes due to union of active neurons quickly approaching dense computation. We introduce Polar Sparsity, highlighting a key shift in sparsity importance from MLP to Attention layers as we scale batch size and sequence length. While MLP layers become more compute-efficient under batching, their sparsity vanishes. In contrast, attention becomes increasingly more expensive at scale, while their head sparsity remains stable and batch-invariant. We develop hardware-efficient, sparsity-aware GPU kernels for selective MLP and Attention computations, delivering up to \(2.2\times\) end-to-end speedups for models like OPT, LLaMA-2 \& 3, across various batch sizes and sequence lengths without compromising accuracy. To our knowledge, this is the first work to demonstrate that contextual sparsity can scale effectively to large batch sizes, delivering substantial inference acceleration with minimal changes, making Polar Sparsity practical for large-scale, high-throughput LLM deployment systems. Our code is available at: https://github.com/susavlsh10/Polar-Sparsity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.12264v2">Uncertainty quantification in fine-tuned LLMs using LoRA ensembles</a></div>
    <div class="paper-meta">
      📅 2025-05-20
      | 💬 Accepted for ICLR2025 Workshop "Quantify Uncertainty and Hallucination in Foundation Models: The Next Frontier in Reliable AI"
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models can improve task specific performance, although a general understanding of what the fine-tuned model has learned, forgotten and how to trust its predictions is still missing. We derive principled uncertainty quantification for fine-tuned LLMs with posterior approximations using computationally efficient low-rank adaptation ensembles. We analyze three common multiple-choice datasets using low-rank adaptation ensembles based on Mistral-7b, and draw quantitative and qualitative conclusions on their perceived complexity and balance between retained prior knowledge and domain specific adaptation during and after fine-tuning. We identify unexpected retention of acquired knowledge during fine-tuning in the overfitting regime.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14864v1">Balanced and Elastic End-to-end Training of Dynamic LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-20
    </div>
    <details class="paper-abstract">
      To reduce computational and memory costs in Large Language Models (LLMs), dynamic workload reduction schemes like Mixture of Experts (MoEs), parameter pruning, layer freezing, sparse attention, early token exit, and Mixture of Depths (MoDs) have emerged. However, these methods introduce severe workload imbalances, limiting their practicality for large-scale distributed training. We propose DynMo, an autonomous dynamic load balancing solution that ensures optimal compute distribution when using pipeline parallelism in training dynamic models. DynMo adaptively balances workloads, dynamically packs tasks into fewer workers to free idle resources, and supports both multi-GPU single-node and multi-node systems. Compared to static training methods (Megatron-LM, DeepSpeed), DynMo accelerates training by up to 1.23x (MoEs), 3.18x (pruning), 2.23x (layer freezing), 4.02x (sparse attention), 4.52x (early exit), and 1.17x (MoDs). DynMo is available at https://anonymous.4open.science/r/DynMo-4D04/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14832v1">SEPS: A Separability Measure for Robust Unlearning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-20
      | 💬 32 pages
    </div>
    <details class="paper-abstract">
      Machine unlearning aims to selectively remove targeted knowledge from Large Language Models (LLMs), ensuring they forget specified content while retaining essential information. Existing unlearning metrics assess whether a model correctly answers retain queries and rejects forget queries, but they fail to capture real-world scenarios where forget queries rarely appear in isolation. In fact, forget and retain queries often coexist within the same prompt, making mixed-query evaluation crucial. We introduce SEPS, an evaluation framework that explicitly measures a model's ability to both forget and retain information within a single prompt. Through extensive experiments across three benchmarks, we identify two key failure modes in existing unlearning methods: (1) untargeted unlearning indiscriminately erases both forget and retain content once a forget query appears, and (2) targeted unlearning overfits to single-query scenarios, leading to catastrophic failures when handling multiple queries. To address these issues, we propose Mixed Prompt (MP) unlearning, a strategy that integrates both forget and retain queries into a unified training objective. Our approach significantly improves unlearning effectiveness, demonstrating robustness even in complex settings with up to eight mixed forget and retain queries in a single prompt.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14818v1">WebNovelBench: Placing LLM Novelists on the Web Novel Distribution</a></div>
    <div class="paper-meta">
      📅 2025-05-20
    </div>
    <details class="paper-abstract">
      Robustly evaluating the long-form storytelling capabilities of Large Language Models (LLMs) remains a significant challenge, as existing benchmarks often lack the necessary scale, diversity, or objective measures. To address this, we introduce WebNovelBench, a novel benchmark specifically designed for evaluating long-form novel generation. WebNovelBench leverages a large-scale dataset of over 4,000 Chinese web novels, framing evaluation as a synopsis-to-story generation task. We propose a multi-faceted framework encompassing eight narrative quality dimensions, assessed automatically via an LLM-as-Judge approach. Scores are aggregated using Principal Component Analysis and mapped to a percentile rank against human-authored works. Our experiments demonstrate that WebNovelBench effectively differentiates between human-written masterpieces, popular web novels, and LLM-generated content. We provide a comprehensive analysis of 24 state-of-the-art LLMs, ranking their storytelling abilities and offering insights for future development. This benchmark provides a scalable, replicable, and data-driven methodology for assessing and advancing LLM-driven narrative generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13358v2">FineEdit: Unlock Instruction-Based Text Editing for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have significantly advanced natural language processing, demonstrating strong capabilities in tasks such as text generation, summarization, and reasoning. Recently, their potential for automating precise text editing tasks across specialized domains, such as programming code, LaTeX, and structured database languages, has gained attention. However, current state-of-the-art LLMs still struggle with executing precise, instruction-driven edits, particularly when structural accuracy and strict adherence to domain conventions are required. To address these challenges, we introduce InstrEditBench, an automated benchmark dataset comprising over 30,000 structured editing tasks spanning diverse domains, including Wikipedia articles, LaTeX documents, source code, and database languages. Using this benchmark, we develop FineEdit, a specialized editing model explicitly trained for accurate, context-aware text modifications. Experimental evaluations demonstrate that FineEdit outperforms state-of-the-art models, achieving improvements of approximately 10% over Gemini models on single-turn edits, up to 30% over Llama-3.2-3B, and exceeding Mistral-7B-OpenOrca performance by over 40% on direct editing tasks. FineEdit also effectively generalizes to realistic multi-turn editing scenarios, highlighting its practical applicability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14668v1">ContextAgent: Context-Aware Proactive LLM Agents with Open-World Sensory Perceptions</a></div>
    <div class="paper-meta">
      📅 2025-05-20
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have propelled intelligent agents from reactive responses to proactive support. While promising, existing proactive agents either rely exclusively on observations from enclosed environments (e.g., desktop UIs) with direct LLM inference or employ rule-based proactive notifications, leading to suboptimal user intent understanding and limited functionality for proactive service. In this paper, we introduce ContextAgent, the first context-aware proactive agent that incorporates extensive sensory contexts to enhance the proactive capabilities of LLM agents. ContextAgent first extracts multi-dimensional contexts from massive sensory perceptions on wearables (e.g., video and audio) to understand user intentions. ContextAgent then leverages the sensory contexts and the persona contexts from historical data to predict the necessity for proactive services. When proactive assistance is needed, ContextAgent further automatically calls the necessary tools to assist users unobtrusively. To evaluate this new task, we curate ContextAgentBench, the first benchmark for evaluating context-aware proactive LLM agents, covering 1,000 samples across nine daily scenarios and twenty tools. Experiments on ContextAgentBench show that ContextAgent outperforms baselines by achieving up to 8.5% and 6.0% higher accuracy in proactive predictions and tool calling, respectively. We hope our research can inspire the development of more advanced, human-centric, proactive AI assistants.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15364v3">KeyDiff: Key Similarity-Based KV Cache Eviction for Long-Context LLM Inference in Resource-Constrained Environments</a></div>
    <div class="paper-meta">
      📅 2025-05-20
      | 💬 9 pages, 7 figures
    </div>
    <details class="paper-abstract">
      We demonstrate that geometrically distinctive keys during LLM inference tend to have high attention scores. Based on the phenomenon we propose KeyDiff, a training-free KV cache eviction method based solely on key similarity. Unlike other KV cache eviction methods, KeyDiff can process arbitrarily long prompts within strict resource constraints and efficiently generate responses. We provide a theoretical basis for KeyDiff by relating key diversity with attention scores. These results imply KeyDiff can efficiently identify the most important tokens to retain. Notably KeyDiff does not rely on attention scores, allowing the use of optimized attention mechanisms like FlashAttention. Under a strict memory allowance, we demonstrate the effectiveness of KeyDiff for the Llama and Qwen model families by observing a performance gap of less than 0.04% with 8K cache budget ($\sim$23% KV cache reduction) from the non-evicting baseline on LongBench for Llama 3.1-8B and Llama 3.2-3B. We also observe near baseline performance for Deepseek-R1-Distill-Llama-8B on the Math500 reasoning benchmark and decrease end-to-end inference latency by up to 30% compared to the other token-eviction methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14656v1">Cost-Augmented Monte Carlo Tree Search for LLM-Assisted Planning</a></div>
    <div class="paper-meta">
      📅 2025-05-20
    </div>
    <details class="paper-abstract">
      While LLMs excel at open-ended reasoning, they often struggle with cost-sensitive planning, either treating all actions as having equal cost or failing to stay within strict budgets. In this paper, we introduce Cost-Augmented Monte Carlo Tree Search (CATS), a novel approach that brings explicit cost-awareness into LLM-guided planning. Tight cost constraints push the planner to quickly identify infeasible solutions, while looser constraints encourage optimization for minimal cost. We benchmark top LLMs such as GPT-4.1, Claude-3.7-Sonnet, and DeepSeek-R1, against our CATS planner to evaluate their performance in cost-sensitive scenarios. Our experiments suggest that raw LLMs such as GPT-4.1 often falter under tight budgets, whereas CATS consistently delivers strong performance, achieving higher task success rates and better cost efficiency. CATS provides an effective solution for budget-aware decision-making by combining the reasoning power of LLMs with structured search.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14654v1">Beyond Words: Multimodal LLM Knows When to Speak</a></div>
    <div class="paper-meta">
      📅 2025-05-20
      | 💬 Project page: https://github.com/lzk901372/MM-When2Speak
    </div>
    <details class="paper-abstract">
      While large language model (LLM)-based chatbots have demonstrated strong capabilities in generating coherent and contextually relevant responses, they often struggle with understanding when to speak, particularly in delivering brief, timely reactions during ongoing conversations. This limitation arises largely from their reliance on text input, lacking the rich contextual cues in real-world human dialogue. In this work, we focus on real-time prediction of response types, with an emphasis on short, reactive utterances that depend on subtle, multimodal signals across vision, audio, and text. To support this, we introduce a new multimodal dataset constructed from real-world conversational videos, containing temporally aligned visual, auditory, and textual streams. This dataset enables fine-grained modeling of response timing in dyadic interactions. Building on this dataset, we propose MM-When2Speak, a multimodal LLM-based model that adaptively integrates visual, auditory, and textual context to predict when a response should occur, and what type of response is appropriate. Experiments show that MM-When2Speak significantly outperforms state-of-the-art unimodal and LLM-based baselines, achieving up to a 4x improvement in response timing accuracy over leading commercial LLMs. These results underscore the importance of multimodal inputs for producing timely, natural, and engaging conversational AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10624v3">SensorLLM: Human-Intuitive Alignment of Multivariate Sensor Data with LLMs for Activity Recognition</a></div>
    <div class="paper-meta">
      📅 2025-05-20
    </div>
    <details class="paper-abstract">
      We introduce SensorLLM, a two-stage framework that enables Large Language Models (LLMs) to perform human activity recognition (HAR) from wearable sensor data. While LLMs excel at reasoning and generalization, they struggle with time-series inputs due to limited semantic context, numerical complexity, and sequence variability. To address these challenges, we construct SensorQA, a question-answering dataset of human-intuitive sensor-text pairs spanning diverse HAR scenarios. It supervises the Sensor-Language Alignment stage, where the model aligns sensor inputs with trend descriptions. Special tokens are introduced to mark channel boundaries. This alignment enables LLMs to interpret numerical patterns, channel-specific signals, and variable-length inputs--without requiring human annotation. In the subsequent Task-Aware Tuning stage, we adapt the model for multivariate HAR classification, achieving performance that matches or exceeds state-of-the-art methods. Our results show that, guided by human-intuitive alignment, SensorLLM becomes an effective sensor learner, reasoner, and classifier--generalizing across varied HAR settings and paving the way for foundation model research in time-series analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14625v1">TinyV: Reducing False Negatives in Verification Improves RL for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-20
    </div>
    <details class="paper-abstract">
      Reinforcement Learning (RL) has become a powerful tool for enhancing the reasoning abilities of large language models (LLMs) by optimizing their policies with reward signals. Yet, RL's success relies on the reliability of rewards, which are provided by verifiers. In this paper, we expose and analyze a widespread problem--false negatives--where verifiers wrongly reject correct model outputs. Our in-depth study of the Big-Math-RL-Verified dataset reveals that over 38% of model-generated responses suffer from false negatives, where the verifier fails to recognize correct answers. We show, both empirically and theoretically, that these false negatives severely impair RL training by depriving the model of informative gradient signals and slowing convergence. To mitigate this, we propose tinyV, a lightweight LLM-based verifier that augments existing rule-based methods, which dynamically identifies potential false negatives and recovers valid responses to produce more accurate reward estimates. Across multiple math-reasoning benchmarks, integrating TinyV boosts pass rates by up to 10% and accelerates convergence relative to the baseline. Our findings highlight the critical importance of addressing verifier false negatives and offer a practical approach to improve RL-based fine-tuning of LLMs. Our code is available at https://github.com/uw-nsl/TinyV.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.11726v2">Revealing and Mitigating the Challenge of Detecting Character Knowledge Errors in LLM Role-Playing</a></div>
    <div class="paper-meta">
      📅 2025-05-20
      | 💬 25 pages, 6 figures, 20 tables
    </div>
    <details class="paper-abstract">
      Large language model (LLM) role-playing has gained widespread attention. Authentic character knowledge is crucial for constructing realistic LLM role-playing agents. However, existing works usually overlook the exploration of LLMs' ability to detect characters' known knowledge errors (KKE) and unknown knowledge errors (UKE) while playing roles, which would lead to low-quality automatic construction of character trainable corpus. In this paper, we propose RoleKE-Bench to evaluate LLMs' ability to detect errors in KKE and UKE. The results indicate that even the latest LLMs struggle to detect these two types of errors effectively, especially when it comes to familiar knowledge. We experimented with various reasoning strategies and propose an agent-based reasoning method, Self-Recollection and Self-Doubt (S$^2$RD), to explore further the potential for improving error detection capabilities. Experiments show that our method effectively improves the LLMs' ability to detect error character knowledge, but it remains an issue that requires ongoing attention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14615v1">SATBench: Benchmarking LLMs' Logical Reasoning via Automated Puzzle Generation from SAT Formulas</a></div>
    <div class="paper-meta">
      📅 2025-05-20
    </div>
    <details class="paper-abstract">
      We introduce SATBench, a benchmark for evaluating the logical reasoning capabilities of large language models (LLMs) through logical puzzles derived from Boolean satisfiability (SAT) problems. Unlike prior work that focuses on inference rule-based reasoning, which often involves deducing conclusions from a set of premises, our approach leverages the search-based nature of SAT problems, where the objective is to find a solution that fulfills a specified set of logical constraints. Each instance in SATBench is generated from a SAT formula, then translated into a story context and conditions using LLMs. The generation process is fully automated and allows for adjustable difficulty by varying the number of clauses. All 2100 puzzles are validated through both LLM-assisted and solver-based consistency checks, with human validation on a subset. Experimental results show that even the strongest model, o4-mini, achieves only 65.0% accuracy on hard UNSAT problems, close to the random baseline of 50%. SATBench exposes fundamental limitations in the search-based logical reasoning abilities of current LLMs and provides a scalable testbed for future research in logical reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15241v2">MrGuard: A Multilingual Reasoning Guardrail for Universal LLM Safety</a></div>
    <div class="paper-meta">
      📅 2025-05-20
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are susceptible to adversarial attacks such as jailbreaking, which can elicit harmful or unsafe behaviors. This vulnerability is exacerbated in multilingual settings, where multilingual safety-aligned data is often limited. Thus, developing a guardrail capable of detecting and filtering unsafe content across diverse languages is critical for deploying LLMs in real-world applications. In this work, we introduce a multilingual guardrail with reasoning for prompt classification. Our method consists of: (1) synthetic multilingual data generation incorporating culturally and linguistically nuanced variants, (2) supervised fine-tuning, and (3) a curriculum-based Group Relative Policy Optimization (GRPO) framework that further improves performance. Experimental results demonstrate that our multilingual guardrail, MrGuard, consistently outperforms recent baselines across both in-domain and out-of-domain languages by more than 15%. We also evaluate MrGuard's robustness to multilingual variations, such as code-switching and low-resource language distractors in the prompt, and demonstrate that it preserves safety judgments under these challenging conditions. The multilingual reasoning capability of our guardrail enables it to generate explanations, which are particularly useful for understanding language-specific risks and ambiguities in multilingual content moderation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14597v1">Success is in the Details: Evaluate and Enhance Details Sensitivity of Code LLMs through Counterfactuals</a></div>
    <div class="paper-meta">
      📅 2025-05-20
      | 💬 Code & Model is https://github.com/Luowaterbi/CTF-Instruct
    </div>
    <details class="paper-abstract">
      Code Sensitivity refers to the ability of Code LLMs to recognize and respond to details changes in problem descriptions. While current code benchmarks and instruction data focus on difficulty and diversity, sensitivity is overlooked. We first introduce the CTF-Code benchmark, constructed using counterfactual perturbations, minimizing input changes while maximizing output changes. The evaluation shows that many LLMs have a more than 10\% performance drop compared to the original problems. To fully utilize sensitivity, CTF-Instruct, an incremental instruction fine-tuning framework, extends on existing data and uses a selection mechanism to meet the three dimensions of difficulty, diversity, and sensitivity. Experiments show that LLMs fine-tuned with CTF-Instruct data achieve over a 2\% improvement on CTF-Code, and more than a 10\% performance boost on LiveCodeBench, validating the feasibility of enhancing LLMs' sensitivity to improve performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07115v4">Online Scheduling for LLM Inference with KV Cache Constraints</a></div>
    <div class="paper-meta">
      📅 2025-05-20
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) inference, where a trained model generates text one word at a time in response to user prompts, is a computationally intensive process requiring efficient scheduling to optimize latency and resource utilization. A key challenge in LLM inference is the management of the Key-Value (KV) cache, which reduces redundant computations but introduces memory constraints. In this work, we model LLM inference with KV cache constraints theoretically and propose a novel batching and scheduling algorithm that minimizes inference latency while effectively managing the KV cache's memory. More specifically, we make the following contributions. First, to evaluate the performance of online algorithms for scheduling in LLM inference, we introduce a hindsight optimal benchmark, formulated as an integer program that computes the minimum total inference latency under full future information. Second, we prove that no deterministic online algorithm can achieve a constant competitive ratio when the arrival process is arbitrary. Third, motivated by the computational intractability of solving the integer program at scale, we propose a polynomial-time online scheduling algorithm and show that under certain conditions it can achieve a constant competitive ratio. We also demonstrate our algorithm's strong empirical performance by comparing it to the hindsight optimal in a synthetic dataset. Finally, we conduct empirical evaluations on a real-world public LLM inference dataset, simulating the Llama2-70B model on A100 GPUs, and show that our algorithm significantly outperforms the benchmark algorithms. Overall, our results offer a path toward more sustainable and cost-effective LLM deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10940v2">CoLA: Compute-Efficient Pre-Training of LLMs via Low-Rank Activation</a></div>
    <div class="paper-meta">
      📅 2025-05-20
      | 💬 v2
    </div>
    <details class="paper-abstract">
      The full-size MLPs and the projection layers in attention introduce tremendous model sizes of large language models (LLMs), imposing extremely demanding needs of computational resources in the pre-training stage. However, we empirically observe that the activations of pre-trained LLMs exhibit low-rank property. Motivated by such observations, we propose CoLA and its memory-efficient implementation, CoLA-M, to replace these full-size layers with compute-efficient auto-encoders that naturally enforce low-rank activations throughout training. This fundamental architectural change eliminates the activation redundancy and significantly boosts model capacity and training efficiency. Experiments on LLaMA models with 60 million to 7 billion parameters show that CoLA reduces the computing cost by $\bf 2\pmb{\times}$ and improves training throughput by $\bf 1.86\pmb{\times}$ while maintaining full-rank level performance. CoLA-M further squeezes memory cost without sacrificing throughput, offering a pre-training approach with collectively superior parameter, computing, and memory efficiency. The LLMs produced are also $\bf 2\pmb{\times}$ smaller, enabling faster inference with lower memory cost on resource-constrained platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17388v3">Can LLMs be Good Graph Judge for Knowledge Graph Construction?</a></div>
    <div class="paper-meta">
      📅 2025-05-20
    </div>
    <details class="paper-abstract">
      In real-world scenarios, most of the data obtained from the information retrieval (IR) system is unstructured. Converting natural language sentences into structured Knowledge Graphs (KGs) remains a critical challenge. We identified three limitations with respect to existing KG construction methods: (1) There could be a large amount of noise in real-world documents, which could result in extracting messy information. (2) Naive LLMs usually extract inaccurate knowledge from some domain-specific documents. (3) Hallucination phenomenon cannot be overlooked when directly using LLMs to construct KGs. In this paper, we propose \textbf{GraphJudge}, a KG construction framework to address the aforementioned challenges. In this framework, we designed an entity-centric strategy to eliminate the noise information in the documents. And we fine-tuned a LLM as a graph judge to finally enhance the quality of generated KGs. Experiments conducted on two general and one domain-specific text-graph pair datasets demonstrate state-of-the-art performance against various baseline methods with strong generalization abilities. Our code is available at \href{https://github.com/hhy-huang/GraphJudge}{https://github.com/hhy-huang/GraphJudge}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18169v4">KunServe: Efficient Parameter-centric Memory Management for LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-05-20
    </div>
    <details class="paper-abstract">
      Serving LLMs with a cluster of GPUs is common nowadays, where the serving system must meet strict latency SLOs required by applications. However, the stateful nature of LLM serving requires maintaining huge states (i.e., KVCache) in limited GPU memory. Under spikes in real-world workloads, GPU memory can be easily throttled, leading to orders of magnitude higher response latency due to queuing introduced by waiting for KVCache to be reclaimed. Prior KVCache-centric approaches handle load throttling by dropping, migrating, or swapping KVCache. These methods fail to release sufficient memory quickly with requests still queued. This paper proposes the first parameter-centric approach to handling throttling by selectively dropping replicated parameters to instantly free memory for requests, based on an unnoticed observation that model parameters are commonly replicated across GPUs for serving LLMs. With additional memory, all requests can be served with a larger batch without queuing. To make the parameter-centric approach correct and efficient, we cooperatively execute requests on GPUs with a complete copy of parameters using pipeline parallelism, and derive an appropriate drop plan without unnecessary cooperation. We also design techniques to minimize the performance overhead due to pipeline parallelism with the execution patterns of requests under drop. Evaluations show that {\sys} reduces the tail TTFT of requests under throttling by up to 72.2 times compared to the state-of-the-art systems including Llumnix, vLLM and InferCept.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14536v1">Breaking Bad Tokens: Detoxification of LLMs Using Sparse Autoencoders</a></div>
    <div class="paper-meta">
      📅 2025-05-20
      | 💬 Preprint: 19 pages, 7 figures, 1 table
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are now ubiquitous in user-facing applications, yet they still generate undesirable toxic outputs, including profanity, vulgarity, and derogatory remarks. Although numerous detoxification methods exist, most apply broad, surface-level fixes and can therefore easily be circumvented by jailbreak attacks. In this paper we leverage sparse autoencoders (SAEs) to identify toxicity-related directions in the residual stream of models and perform targeted activation steering using the corresponding decoder vectors. We introduce three tiers of steering aggressiveness and evaluate them on GPT-2 Small and Gemma-2-2B, revealing trade-offs between toxicity reduction and language fluency. At stronger steering strengths, these causal interventions surpass competitive baselines in reducing toxicity by up to 20%, though fluency can degrade noticeably on GPT-2 Small depending on the aggressiveness. Crucially, standard NLP benchmark scores upon steering remain stable, indicating that the model's knowledge and general abilities are preserved. We further show that feature-splitting in wider SAEs hampers safety interventions, underscoring the importance of disentangled feature learning. Our findings highlight both the promise and the current limitations of SAE-based causal interventions for LLM detoxification, further suggesting practical guidelines for safer language-model deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14756v1">$\texttt{LLINBO}$: Trustworthy LLM-in-the-Loop Bayesian Optimization</a></div>
    <div class="paper-meta">
      📅 2025-05-20
    </div>
    <details class="paper-abstract">
      Bayesian optimization (BO) is a sequential decision-making tool widely used for optimizing expensive black-box functions. Recently, Large Language Models (LLMs) have shown remarkable adaptability in low-data regimes, making them promising tools for black-box optimization by leveraging contextual knowledge to propose high-quality query points. However, relying solely on LLMs as optimization agents introduces risks due to their lack of explicit surrogate modeling and calibrated uncertainty, as well as their inherently opaque internal mechanisms. This structural opacity makes it difficult to characterize or control the exploration-exploitation trade-off, ultimately undermining theoretical tractability and reliability. To address this, we propose LLINBO: LLM-in-the-Loop BO, a hybrid framework for BO that combines LLMs with statistical surrogate experts (e.g., Gaussian Processes (GP)). The core philosophy is to leverage contextual reasoning strengths of LLMs for early exploration, while relying on principled statistical models to guide efficient exploitation. Specifically, we introduce three mechanisms that enable this collaboration and establish their theoretical guarantees. We end the paper with a real-life proof-of-concept in the context of 3D printing. The code to reproduce the results can be found at https://github.com/UMDataScienceLab/LLM-in-the-Loop-BO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14530v1">Internal Chain-of-Thought: Empirical Evidence for Layer-wise Subtask Scheduling in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-20
      | 💬 27 pages, 17 figures
    </div>
    <details class="paper-abstract">
      We show that large language models (LLMs) exhibit an $\textit{internal chain-of-thought}$: they sequentially decompose and execute composite tasks layer-by-layer. Two claims ground our study: (i) distinct subtasks are learned at different network depths, and (ii) these subtasks are executed sequentially across layers. On a benchmark of 15 two-step composite tasks, we employ layer-from context-masking and propose a novel cross-task patching method, confirming (i). To examine claim (ii), we apply LogitLens to decode hidden states, revealing a consistent layerwise execution pattern. We further replicate our analysis on the real-world $\text{TRACE}$ benchmark, observing the same stepwise dynamics. Together, our results enhance LLMs transparency by showing their capacity to internally plan and execute subtasks (or instructions), opening avenues for fine-grained, instruction-level activation steering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09083v2">Evaluating the Correctness of Inference Patterns Used by LLMs for Judgment</a></div>
    <div class="paper-meta">
      📅 2025-05-20
    </div>
    <details class="paper-abstract">
      This paper presents a method to analyze the inference patterns used by Large Language Models (LLMs) for judgment in a case study on legal LLMs, so as to identify potential incorrect representations of the LLM, according to human domain knowledge. Unlike traditional evaluations on language generation results, we propose to evaluate the correctness of the detailed inference patterns of an LLM behind its seemingly correct outputs. To this end, we quantify the interactions between input phrases used by the LLM as primitive inference patterns, because recent theoretical achievements have proven several mathematical guarantees of the faithfulness of the interaction-based explanation. We design a set of metrics to evaluate the detailed inference patterns of LLMs. Experiments show that even when the language generation results appear correct, a significant portion of the inference patterns used by the LLM for the legal judgment may represent misleading or irrelevant logic.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14499v1">Enhanced Multimodal Aspect-Based Sentiment Analysis by LLM-Generated Rationales</a></div>
    <div class="paper-meta">
      📅 2025-05-20
    </div>
    <details class="paper-abstract">
      There has been growing interest in Multimodal Aspect-Based Sentiment Analysis (MABSA) in recent years. Existing methods predominantly rely on pre-trained small language models (SLMs) to collect information related to aspects and sentiments from both image and text, with an aim to align these two modalities. However, small SLMs possess limited capacity and knowledge, often resulting in inaccurate identification of meaning, aspects, sentiments, and their interconnections in textual and visual data. On the other hand, Large language models (LLMs) have shown exceptional capabilities in various tasks by effectively exploring fine-grained information in multimodal data. However, some studies indicate that LLMs still fall short compared to fine-tuned small models in the field of ABSA. Based on these findings, we propose a novel framework, termed LRSA, which combines the decision-making capabilities of SLMs with additional information provided by LLMs for MABSA. Specifically, we inject explanations generated by LLMs as rationales into SLMs and employ a dual cross-attention mechanism for enhancing feature interaction and fusion, thereby augmenting the SLMs' ability to identify aspects and sentiments. We evaluated our method using two baseline models, numerous experiments highlight the superiority of our approach on three widely-used benchmarks, indicating its generalizability and applicability to most pre-trained models for MABSA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14479v1">Towards Reliable Proof Generation with LLMs: A Neuro-Symbolic Approach</a></div>
    <div class="paper-meta">
      📅 2025-05-20
      | 💬 long paper
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) struggle with formal domains that require rigorous logical deduction and symbolic reasoning, such as mathematical proof generation. We propose a neuro-symbolic approach that combines LLMs' generative strengths with structured components to overcome this challenge. As a proof-of-concept, we focus on geometry problems. Our approach is two-fold: (1) we retrieve analogous problems and use their proofs to guide the LLM, and (2) a formal verifier evaluates the generated proofs and provides feedback, helping the model fix incorrect proofs. We demonstrate that our method significantly improves proof accuracy for OpenAI's o1 model (58%-70% improvement); both analogous problems and the verifier's feedback contribute to these gains. More broadly, shifting to LLMs that generate provably correct conclusions could dramatically improve their reliability, accuracy and consistency, unlocking complex tasks and critical real-world applications that require trustworthiness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14468v1">ServerlessLoRA: Minimizing Latency and Cost in Serverless Inference for LoRA-Based LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-20
    </div>
    <details class="paper-abstract">
      Serverless computing has grown rapidly for serving Large Language Model (LLM) inference due to its pay-as-you-go pricing, fine-grained GPU usage, and rapid scaling. However, our analysis reveals that current serverless can effectively serve general LLM but fail with Low-Rank Adaptation (LoRA) inference due to three key limitations: 1) massive parameter redundancy among functions where 99% of weights are unnecessarily duplicated, 2) costly artifact loading latency beyond LLM loading, and 3) magnified resource contention when serving multiple LoRA LLMs. These inefficiencies lead to massive GPU wastage, increased Time-To-First-Token (TTFT), and high monetary costs. We propose ServerlessLoRA, a novel serverless inference system designed for faster and cheaper LoRA LLM serving. ServerlessLoRA enables secure backbone LLM sharing across isolated LoRA functions to reduce redundancy. We design a pre-loading method that pre-loads comprehensive LoRA artifacts to minimize cold-start latency. Furthermore, ServerlessLoRA employs contention aware batching and offloading to mitigate GPU resource conflicts during bursty workloads. Experiment on industrial workloads demonstrates that ServerlessLoRA reduces TTFT by up to 86% and cuts monetary costs by up to 89% compared to state-of-the-art LLM inference solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.18062v2">LLM-hRIC: LLM-empowered Hierarchical RAN Intelligent Control for O-RAN</a></div>
    <div class="paper-meta">
      📅 2025-05-20
    </div>
    <details class="paper-abstract">
      Despite recent advances in applying large language models (LLMs) and machine learning (ML) techniques to open radio access network (O-RAN), critical challenges remain, such as insufficient cooperation between radio access network (RAN) intelligent controllers (RICs), high computational demands hindering real-time decisions, and the lack of domain-specific finetuning. Therefore, this article introduces the LLM-empowered hierarchical RIC (LLM-hRIC) framework to improve the collaboration between RICs in O-RAN. The LLM-empowered non-real-time RIC (non-RT RIC) acts as a guider, offering a strategic guidance to the near-real-time RIC (near-RT RIC) using global network information. The RL-empowered near-RT RIC acts as an implementer, combining this guidance with local real-time data to make near-RT decisions. We evaluate the feasibility and performance of the LLM-hRIC framework in an integrated access and backhaul (IAB) network setting, and finally, discuss the open challenges of the LLM-hRIC framework for O-RAN.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.10657v2">RouterEval: A Comprehensive Benchmark for Routing LLMs to Explore Model-level Scaling Up in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-20
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Routing large language models (LLMs) is a new paradigm that uses a router to recommend the best LLM from a pool of candidates for a given input. In this paper, our comprehensive analysis with more than 8,500 LLMs reveals a novel model-level scaling up phenomenon in Routing LLMs, i.e., a capable router can significantly enhance the performance of this paradigm as the number of candidates increases. This improvement can even surpass the performance of the best single model in the pool and many existing strong LLMs, confirming it a highly promising paradigm. However, the lack of comprehensive and open-source benchmarks for Routing LLMs has hindered the development of routers. In this paper, we introduce RouterEval, a benchmark tailored for router research, which includes over 200,000,000 performance records for 12 popular LLM evaluations across various areas such as commonsense reasoning, semantic understanding, etc., based on over 8,500 various LLMs. Using RouterEval, extensive evaluations of existing Routing LLM methods reveal that most still have significant room for improvement. See https://github.com/MilkThink-Lab/RouterEval for all data, code and tutorial.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.12188v2">LLM-DSE: Searching Accelerator Parameters with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-20
    </div>
    <details class="paper-abstract">
      Even though high-level synthesis (HLS) tools mitigate the challenges of programming domain-specific accelerators (DSAs) by raising the abstraction level, optimizing hardware directive parameters remains a significant hurdle. Existing heuristic and learning-based methods struggle with adaptability and sample efficiency. We present LLM-DSE, a multi-agent framework designed specifically for optimizing HLS directives. Combining LLM with design space exploration (DSE), our explorer coordinates four agents: Router, Specialists, Arbitrator, and Critic. These multi-agent components interact with various tools to accelerate the optimization process. LLM-DSE leverages essential domain knowledge to identify efficient parameter combinations while maintaining adaptability through verbal learning from online interactions. Evaluations on the HLSyn dataset demonstrate that LLM-DSE achieves substantial $2.55\times$ performance gains over state-of-the-art methods, uncovering novel designs while reducing runtime. Ablation studies validate the effectiveness and necessity of the proposed agent interactions. Our code is open-sourced here: https://github.com/Nozidoali/LLM-DSE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15585v2">A Comprehensive Survey in LLM(-Agent) Full Stack Safety: Data, Training and Deployment</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      The remarkable success of Large Language Models (LLMs) has illuminated a promising pathway toward achieving Artificial General Intelligence for both academic and industrial communities, owing to their unprecedented performance across various applications. As LLMs continue to gain prominence in both research and commercial domains, their security and safety implications have become a growing concern, not only for researchers and corporations but also for every nation. Currently, existing surveys on LLM safety primarily focus on specific stages of the LLM lifecycle, e.g., deployment phase or fine-tuning phase, lacking a comprehensive understanding of the entire "lifechain" of LLMs. To address this gap, this paper introduces, for the first time, the concept of "full-stack" safety to systematically consider safety issues throughout the entire process of LLM training, deployment, and eventual commercialization. Compared to the off-the-shelf LLM safety surveys, our work demonstrates several distinctive advantages: (I) Comprehensive Perspective. We define the complete LLM lifecycle as encompassing data preparation, pre-training, post-training, deployment and final commercialization. To our knowledge, this represents the first safety survey to encompass the entire lifecycle of LLMs. (II) Extensive Literature Support. Our research is grounded in an exhaustive review of over 800+ papers, ensuring comprehensive coverage and systematic organization of security issues within a more holistic understanding. (III) Unique Insights. Through systematic literature analysis, we have developed reliable roadmaps and perspectives for each chapter. Our work identifies promising research directions, including safety in data generation, alignment techniques, model editing, and LLM-based agent systems. These insights provide valuable guidance for researchers pursuing future work in this field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12891v1">TIME: A Multi-level Benchmark for Temporal Reasoning of LLMs in Real-World Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-05-19
      | 💬 First version. There are still some examples to be added into the appendix
    </div>
    <details class="paper-abstract">
      Temporal reasoning is pivotal for Large Language Models (LLMs) to comprehend the real world. However, existing works neglect the real-world challenges for temporal reasoning: (1) intensive temporal information, (2) fast-changing event dynamics, and (3) complex temporal dependencies in social interactions. To bridge this gap, we propose a multi-level benchmark TIME, designed for temporal reasoning in real-world scenarios. TIME consists of 38,522 QA pairs, covering 3 levels with 11 fine-grained sub-tasks. This benchmark encompasses 3 sub-datasets reflecting different real-world challenges: TIME-Wiki, TIME-News, and TIME-Dial. We conduct extensive experiments on reasoning models and non-reasoning models. And we conducted an in-depth analysis of temporal reasoning performance across diverse real-world scenarios and tasks, and summarized the impact of test-time scaling on temporal reasoning capabilities. Additionally, we release TIME-Lite, a human-annotated subset to foster future research and standardized evaluation in temporal reasoning. The code is available at https://github.com/sylvain-wei/TIME , and the dataset is available at https://huggingface.co/datasets/SylvainWei/TIME .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.05999v3">LLMPot: Dynamically Configured LLM-based Honeypot for Industrial Protocol and Physical Process Emulation</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      Industrial Control Systems (ICS) are extensively used in critical infrastructures ensuring efficient, reliable, and continuous operations. However, their increasing connectivity and addition of advanced features make them vulnerable to cyber threats, potentially leading to severe disruptions in essential services. In this context, honeypots play a vital role by acting as decoy targets within ICS networks, or on the Internet, helping to detect, log, analyze, and develop mitigations for ICS-specific cyber threats. Deploying ICS honeypots, however, is challenging due to the necessity of accurately replicating industrial protocols and device characteristics, a crucial requirement for effectively mimicking the unique operational behavior of different industrial systems. Moreover, this challenge is compounded by the significant manual effort required in also mimicking the control logic the PLC would execute, in order to capture attacker traffic aiming to disrupt critical infrastructure operations. In this paper, we propose LLMPot, a novel approach for designing honeypots in ICS networks harnessing the potency of Large Language Models (LLMs). LLMPot aims to automate and optimize the creation of realistic honeypots with vendor-agnostic configurations, and for any control logic, aiming to eliminate the manual effort and specialized knowledge traditionally required in this domain. We conducted extensive experiments focusing on a wide array of parameters, demonstrating that our LLM-based approach can effectively create honeypot devices implementing different industrial protocols and diverse control logic.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12833v1">Reasoning BO: Enhancing Bayesian Optimization with Long-Context Reasoning Power of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      Many real-world scientific and industrial applications require the optimization of expensive black-box functions. Bayesian Optimization (BO) provides an effective framework for such problems. However, traditional BO methods are prone to get trapped in local optima and often lack interpretable insights. To address this issue, this paper designs Reasoning BO, a novel framework that leverages reasoning models to guide the sampling process in BO while incorporating multi-agent systems and knowledge graphs for online knowledge accumulation. By integrating the reasoning and contextual understanding capabilities of Large Language Models (LLMs), we can provide strong guidance to enhance the BO process. As the optimization progresses, Reasoning BO provides real-time sampling recommendations along with critical insights grounded in plausible scientific theories, aiding in the discovery of superior solutions within the search space. We systematically evaluate our approach across 10 diverse tasks encompassing synthetic mathematical functions and complex real-world applications. The framework demonstrates its capability to progressively refine sampling strategies through real-time insights and hypothesis evolution, effectively identifying higher-performing regions of the search space for focused exploration. This process highlights the powerful reasoning and context-learning abilities of LLMs in optimization scenarios. For example, in the Direct Arylation task, our method increased the yield to 60.7%, whereas traditional BO achieved only a 25.2% yield. Furthermore, our investigation reveals that smaller LLMs, when fine-tuned through reinforcement learning, can attain comparable performance to their larger counterparts. This enhanced reasoning capability paves the way for more efficient automated scientific experimentation while maintaining computational feasibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12831v1">Contrastive Prompting Enhances Sentence Embeddings in LLMs through Inference-Time Steering</a></div>
    <div class="paper-meta">
      📅 2025-05-19
      | 💬 ACL 2025
    </div>
    <details class="paper-abstract">
      Extracting sentence embeddings from large language models (LLMs) is a practical direction, as it requires neither additional data nor fine-tuning. Previous studies usually focus on prompt engineering to guide LLMs to encode the core semantic information of the sentence into the embedding of the last token. However, the last token in these methods still encodes an excess of non-essential information, such as stop words, limiting its encoding capacity. To this end, we propose a Contrastive Prompting (CP) method that introduces an extra auxiliary prompt to elicit better sentence embedding. By contrasting with the auxiliary prompt, CP can steer existing prompts to encode the core semantics of the sentence, rather than non-essential information. CP is a plug-and-play inference-time intervention method that can be combined with various prompt-based methods. Extensive experiments on Semantic Textual Similarity (STS) tasks and downstream classification tasks demonstrate that our method can improve the performance of existing prompt-based methods across different LLMs. Our code will be released at https://github.com/zifengcheng/CP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12814v1">PsyMem: Fine-grained psychological alignment and Explicit Memory Control for Advanced Role-Playing LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      Existing LLM-based role-playing methods often rely on superficial textual descriptions or simplistic metrics, inadequately modeling both intrinsic and extrinsic character dimensions. Additionally, they typically simulate character memory with implicit model knowledge or basic retrieval augment generation without explicit memory alignment, compromising memory consistency. The two issues weaken reliability of role-playing LLMs in several applications, such as trustworthy social simulation. To address these limitations, we propose PsyMem, a novel framework integrating fine-grained psychological attributes and explicit memory control for role-playing. PsyMem supplements textual descriptions with 26 psychological indicators to detailed model character. Additionally, PsyMem implements memory alignment training, explicitly trains the model to align character's response with memory, thereby enabling dynamic memory-controlled responding during inference. By training Qwen2.5-7B-Instruct on our specially designed dataset (including 5,414 characters and 38,962 dialogues extracted from novels), the resulting model, termed as PsyMem-Qwen, outperforms baseline models in role-playing, achieving the best performance in human-likeness and character fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12792v1">EAVIT: Efficient and Accurate Human Value Identification from Text data via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      The rapid evolution of large language models (LLMs) has revolutionized various fields, including the identification and discovery of human values within text data. While traditional NLP models, such as BERT, have been employed for this task, their ability to represent textual data is significantly outperformed by emerging LLMs like GPTs. However, the performance of online LLMs often degrades when handling long contexts required for value identification, which also incurs substantial computational costs. To address these challenges, we propose EAVIT, an efficient and accurate framework for human value identification that combines the strengths of both locally fine-tunable and online black-box LLMs. Our framework employs a value detector - a small, local language model - to generate initial value estimations. These estimations are then used to construct concise input prompts for online LLMs, enabling accurate final value identification. To train the value detector, we introduce explanation-based training and data generation techniques specifically tailored for value identification, alongside sampling strategies to optimize the brevity of LLM input prompts. Our approach effectively reduces the number of input tokens by up to 1/6 compared to directly querying online LLMs, while consistently outperforming traditional NLP methods and other LLM-based strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18169v3">KunServe: Parameter-centric Memory Management for Efficient Memory Throttling Handling in LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      Serving LLMs with a cluster of GPUs is common nowadays, where the serving system must meet strict latency SLOs required by applications. However, the stateful nature of LLM serving requires maintaining huge states (i.e., KVCache) in limited GPU memory. Under spikes in real-world workloads, GPU memory can be easily throttled, leading to orders of magnitude higher response latency due to queuing introduced by waiting for KVCache to be reclaimed. Prior KVCache-centric approaches handle load throttling by dropping, migrating, or swapping KVCache. These methods fail to release sufficient memory quickly with requests still queued. This paper proposes the first parameter-centric approach to handling throttling by selectively dropping replicated parameters to instantly free memory for requests, based on an unnoticed observation that model parameters are commonly replicated across GPUs for serving LLMs. With additional memory, all requests can be served with a larger batch without queuing. To make the parameter-centric approach correct and efficient, we cooperatively execute requests on GPUs with a complete copy of parameters using pipeline parallelism, and derive an appropriate drop plan without unnecessary cooperation. We also design techniques to minimize the performance overhead due to pipeline parallelism with the execution patterns of requests under drop. Evaluations show that {\sys} reduces the tail TTFT of requests under throttling by up to 72.2 times compared to the state-of-the-art systems including Llumnix, vLLM and InferCept.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12717v1">ToTRL: Unlock LLM Tree-of-Thoughts Reasoning Potential through Puzzles Solving</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate significant reasoning capabilities, particularly through long chain-of-thought (CoT) processes, which can be elicited by reinforcement learning (RL). However, prolonged CoT reasoning presents limitations, primarily verbose outputs due to excessive introspection. The reasoning process in these LLMs often appears to follow a trial-and-error methodology rather than a systematic, logical deduction. In contrast, tree-of-thoughts (ToT) offers a conceptually more advanced approach by modeling reasoning as an exploration within a tree structure. This reasoning structure facilitates the parallel generation and evaluation of multiple reasoning branches, allowing for the active identification, assessment, and pruning of unproductive paths. This process can potentially lead to improved performance and reduced token costs. Building upon the long CoT capability of LLMs, we introduce tree-of-thoughts RL (ToTRL), a novel on-policy RL framework with a rule-based reward. ToTRL is designed to guide LLMs in developing the parallel ToT strategy based on the sequential CoT strategy. Furthermore, we employ LLMs as players in a puzzle game during the ToTRL training process. Solving puzzle games inherently necessitates exploring interdependent choices and managing multiple constraints, which requires the construction and exploration of a thought tree, providing challenging tasks for cultivating the ToT reasoning capability. Our empirical evaluations demonstrate that our ToTQwen3-8B model, trained with our ToTRL, achieves significant improvement in performance and reasoning efficiency on complex reasoning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12694v1">LLM-based Query Expansion Fails for Unfamiliar and Ambiguous Queries</a></div>
    <div class="paper-meta">
      📅 2025-05-19
      | 💬 Accepted at SIGIR 2025 short paper track
    </div>
    <details class="paper-abstract">
      Query expansion (QE) enhances retrieval by incorporating relevant terms, with large language models (LLMs) offering an effective alternative to traditional rule-based and statistical methods. However, LLM-based QE suffers from a fundamental limitation: it often fails to generate relevant knowledge, degrading search performance. Prior studies have focused on hallucination, yet its underlying cause--LLM knowledge deficiencies--remains underexplored. This paper systematically examines two failure cases in LLM-based QE: (1) when the LLM lacks query knowledge, leading to incorrect expansions, and (2) when the query is ambiguous, causing biased refinements that narrow search coverage. We conduct controlled experiments across multiple datasets, evaluating the effects of knowledge and query ambiguity on retrieval performance using sparse and dense retrieval models. Our results reveal that LLM-based QE can significantly degrade the retrieval effectiveness when knowledge in the LLM is insufficient or query ambiguity is high. We introduce a framework for evaluating QE under these conditions, providing insights into the limitations of LLM-based retrieval augmentation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12692v1">Bullying the Machine: How Personas Increase LLM Vulnerability</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed in interactions where they are prompted to adopt personas. This paper investigates whether such persona conditioning affects model safety under bullying, an adversarial manipulation that applies psychological pressures in order to force the victim to comply to the attacker. We introduce a simulation framework in which an attacker LLM engages a victim LLM using psychologically grounded bullying tactics, while the victim adopts personas aligned with the Big Five personality traits. Experiments using multiple open-source LLMs and a wide range of adversarial goals reveal that certain persona configurations -- such as weakened agreeableness or conscientiousness -- significantly increase victim's susceptibility to unsafe outputs. Bullying tactics involving emotional or sarcastic manipulation, such as gaslighting and ridicule, are particularly effective. These findings suggest that persona-driven interaction introduces a novel vector for safety risks in LLMs and highlight the need for persona-aware safety evaluation and alignment strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.15087v2">Multi-Grained Patch Training for Efficient LLM-based Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have emerged as a new paradigm for recommendation by converting interacted item history into language modeling. However, constrained by the limited context length of LLMs, existing approaches have to truncate item history in the prompt, focusing only on recent interactions and sacrificing the ability to model long-term history. To enable LLMs to model long histories, we pursue a concise embedding representation for items and sessions. In the LLM embedding space, we construct an item's embedding by aggregating its textual token embeddings; similarly, we construct a session's embedding by aggregating its item embeddings. While efficient, this way poses two challenges since it ignores the temporal significance of user interactions and LLMs do not natively interpret our custom embeddings. To overcome these, we propose PatchRec, a multi-grained patch training method consisting of two stages: (1) Patch Pre-training, which familiarizes LLMs with aggregated embeddings -- patches, and (2) Patch Fine-tuning, which enables LLMs to capture time-aware significance in interaction history. Extensive experiments show that PatchRec effectively models longer behavior histories with improved efficiency. This work facilitates the practical use of LLMs for modeling long behavior histories. Codes are available at https://github.com/ljy0ustc/PatchRec.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12666v1">Adapting to LLMs: How Insiders and Outsiders Reshape Scientific Knowledge Production</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      CSCW has long examined how emerging technologies reshape the ways researchers collaborate and produce knowledge, with scientific knowledge production as a central area of focus. As AI becomes increasingly integrated into scientific research, understanding how researchers adapt to it reveals timely opportunities for CSCW research -- particularly in supporting new forms of collaboration, knowledge practices, and infrastructure in AI-driven science. This study quantifies LLM impacts on scientific knowledge production based on an evaluation workflow that combines an insider-outsider perspective with a knowledge production framework. Our findings reveal how LLMs catalyze both innovation and reorganization in scientific communities, offering insights into the broader transformation of knowledge production in the age of generative AI and sheds light on new research opportunities in CSCW.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10597v2">Two Minds Better Than One: Collaborative Reward Modeling for LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      Reward models (RMs) play a pivotal role in aligning large language models (LLMs) with human values. However, noisy preferences in human feedback can lead to reward misgeneralization - a phenomenon where reward models learn spurious correlations or overfit to noisy preferences, which poses important challenges to the generalization of RMs. This paper systematically analyzes the characteristics of preference pairs and aims to identify how noisy preferences differ from human-aligned preferences in reward modeling. Our analysis reveals that noisy preferences are difficult for RMs to fit, as they cause sharp training fluctuations and irregular gradient updates. These distinctive dynamics suggest the feasibility of identifying and excluding such noisy preferences. Empirical studies demonstrate that policy LLM optimized with a reward model trained on the full preference dataset, which includes substantial noise, performs worse than the one trained on a subset of exclusively high quality preferences. To address this challenge, we propose an online Collaborative Reward Modeling (CRM) framework to achieve robust preference learning through peer review and curriculum learning. In particular, CRM maintains two RMs that collaboratively filter potential noisy preferences by peer-reviewing each other's data selections. Curriculum learning synchronizes the capabilities of two models, mitigating excessive disparities to promote the utility of peer review. Extensive experiments demonstrate that CRM significantly enhances RM generalization, with up to 9.94 points improvement on RewardBench under an extreme 40\% noise. Moreover, CRM can seamlessly extend to implicit-reward alignment methods, offering a robust and versatile alignment strategy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09970v2">Pre-Act: Multi-Step Planning and Reasoning Improves Acting in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      The ReAct (Reasoning + Action) capability in large language models (LLMs) has become the foundation of modern agentic systems. Recent LLMs, such as DeepSeek-R1 and OpenAI o1/o3, exemplify this by emphasizing reasoning through the generation of ample intermediate tokens, which help build a strong premise before producing the final output tokens. In this paper, we introduce Pre-Act, a novel approach that enhances the agent's performance by creating a multi-step execution plan along with the detailed reasoning for the given user input. This plan incrementally incorporates previous steps and tool outputs, refining itself after each step execution until the final response is obtained. Our approach is applicable to both conversational and non-conversational agents. To measure the performance of task-oriented agents comprehensively, we propose a two-level evaluation framework: (1) turn level and (2) end-to-end. Our turn-level evaluation, averaged across five models, shows that our approach, Pre-Act, outperforms ReAct by 70% in Action Recall on the Almita dataset. While this approach is effective for larger models, smaller models crucial for practical applications, where latency and cost are key constraints, often struggle with complex reasoning tasks required for agentic systems. To address this limitation, we fine-tune relatively small models such as Llama 3.1 (8B & 70B) using the proposed Pre-Act approach. Our experiments show that the fine-tuned 70B model outperforms GPT-4, achieving a 69.5% improvement in action accuracy (turn-level) and a 28% improvement in goal completion rate (end-to-end) on the Almita (out-of-domain) dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11441v2">D3: Diversity, Difficulty, and Dependability-Aware Data Selection for Sample-Efficient LLM Instruction Tuning</a></div>
    <div class="paper-meta">
      📅 2025-05-19
      | 💬 IJCAI'25
    </div>
    <details class="paper-abstract">
      Recent advancements in instruction tuning for large language models (LLMs) suggest that a small, high-quality dataset can significantly equip LLMs with instruction-following capabilities, outperforming large datasets often burdened by quality and redundancy issues. However, the challenge lies in automatically identifying valuable subsets from large datasets to boost both the effectiveness and efficiency of instruction tuning. In this paper, we first establish data selection criteria based on three distinct aspects of data value: diversity, difficulty, and dependability, and then propose the D3 method comprising two key steps of scoring and selection. Specifically, in the scoring step, we define the diversity function to measure sample distinctiveness and introduce the uncertainty-based prediction difficulty to evaluate sample difficulty by mitigating the interference of context-oriented generation diversity. Additionally, we integrate an external LLM for dependability assessment. In the selection step, we formulate the D3 weighted coreset objective, which jointly optimizes three aspects of data value to solve for the most valuable subset. The two steps of D3 can iterate multiple rounds, incorporating feedback to refine the selection focus adaptively. Experiments on both public datasets and the real-world Taobao Live application demonstrate the effectiveness of D3 in endowing LLMs with competitive or even superior instruction-following capabilities using less than 10\% of the entire dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12621v1">Think Before You Attribute: Improving the Performance of LLMs Attribution Systems</a></div>
    <div class="paper-meta">
      📅 2025-05-19
      | 💬 22 pages (9 pages of content, 4 pages of references, 9 pages of supplementary material), 7 figures, 10 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly applied in various science domains, yet their broader adoption remains constrained by a critical challenge: the lack of trustworthy, verifiable outputs. Current LLMs often generate answers without reliable source attribution, or worse, with incorrect attributions, posing a barrier to their use in scientific and high-stakes settings, where traceability and accountability are non-negotiable. To be reliable, attribution systems need high accuracy and retrieve data with short lengths, i.e., attribute to a sentence within a document rather than a whole document. We propose a sentence-level pre-attribution step for Retrieve-Augmented Generation (RAG) systems that classify sentences into three categories: not attributable, attributable to a single quote, and attributable to multiple quotes. By separating sentences before attribution, a proper attribution method can be selected for the type of sentence, or the attribution can be skipped altogether. Our results indicate that classifiers are well-suited for this task. In this work, we propose a pre-attribution step to reduce the computational complexity of attribution, provide a clean version of the HAGRID dataset, and provide an end-to-end attribution system that works out of the box.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15772v2">Detecting LLM-Generated Peer Reviews</a></div>
    <div class="paper-meta">
      📅 2025-05-19
      | 💬 27 pages, 2 figures
    </div>
    <details class="paper-abstract">
      The integrity of peer review is fundamental to scientific progress, but the rise of large language models (LLMs) has introduced concerns that some reviewers may rely on these tools to generate reviews rather than writing them independently. Although some venues have banned LLM-assisted reviewing, enforcement remains difficult as existing detection tools cannot reliably distinguish between fully generated reviews and those merely polished with AI assistance. In this work, we address the challenge of detecting LLM-generated reviews. We consider the approach of performing indirect prompt injection via the paper's PDF, prompting the LLM to embed a covert watermark in the generated review, and subsequently testing for presence of the watermark in the review. We identify and address several pitfalls in na\"ive implementations of this approach. Our primary contribution is a rigorous watermarking and detection framework that offers strong statistical guarantees. Specifically, we introduce watermarking schemes and hypothesis tests that control the family-wise error rate across multiple reviews, achieving higher statistical power than standard corrections such as Bonferroni, while making no assumptions about the nature of human-written reviews. We explore multiple indirect prompt injection strategies--including font-based embedding and obfuscated prompts--and evaluate their effectiveness under various reviewer defense scenarios. Our experiments find high success rates in watermark embedding across various LLMs. We also empirically find that our approach is resilient to common reviewer defenses, and that the bounds on error rates in our statistical tests hold in practice. In contrast, we find that Bonferroni-style corrections are too conservative to be useful in this setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12601v1">Rethinking Predictive Modeling for LLM Routing: When Simple kNN Beats Complex Learned Routers</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) grow in scale and specialization, routing--selecting the best model for a given input--has become essential for efficient and effective deployment. While recent methods rely on complex learned routing strategies, their dependence on disparate training data and evaluation setups makes comparison and generalization difficult. In this work, we revisit LLM routing through the lens of simplicity. We show that a well-tuned k-Nearest Neighbors (kNN) approach not only matches but often outperforms state-of-the-art learned routers across diverse tasks. To support systematic evaluation, we introduce a suite of standardized routing benchmarks spanning instruction-following, question-answering, and reasoning tasks, as well as the first multi-modal routing dataset involving visual inputs. Our findings reveal that the locality properties of model performance in embedding space enable simple non-parametric methods to achieve strong routing decisions with lower sample complexity than parametric approaches. This challenges the prevailing trend toward sophisticated architectures and highlights the importance of thoroughly evaluating simple baselines before investing in complex solutions. To support reproducibility and further exploration, we will release all benchmarks and code upon publication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13778v1">CoIn: Counting the Invisible Reasoning Tokens in Commercial Opaque LLM APIs</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      As post-training techniques evolve, large language models (LLMs) are increasingly augmented with structured multi-step reasoning abilities, often optimized through reinforcement learning. These reasoning-enhanced models outperform standard LLMs on complex tasks and now underpin many commercial LLM APIs. However, to protect proprietary behavior and reduce verbosity, providers typically conceal the reasoning traces while returning only the final answer. This opacity introduces a critical transparency gap: users are billed for invisible reasoning tokens, which often account for the majority of the cost, yet have no means to verify their authenticity. This opens the door to token count inflation, where providers may overreport token usage or inject synthetic, low-effort tokens to inflate charges. To address this issue, we propose CoIn, a verification framework that audits both the quantity and semantic validity of hidden tokens. CoIn constructs a verifiable hash tree from token embedding fingerprints to check token counts, and uses embedding-based relevance matching to detect fabricated reasoning content. Experiments demonstrate that CoIn, when deployed as a trusted third-party auditor, can effectively detect token count inflation with a success rate reaching up to 94.7%, showing the strong ability to restore billing transparency in opaque LLM services. The dataset and code are available at https://github.com/CASE-Lab-UMD/LLM-Auditing-CoIn.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.18416v4">PersonaGym: Evaluating Persona Agents and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-19
      | 💬 21 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Persona agents, which are LLM agents conditioned to act according to an assigned persona, enable contextually rich and user aligned interactions across domains like education and healthcare. However, evaluating how faithfully these agents adhere to their personas remains a significant challenge, particularly in free-form settings that demand consistency across diverse, persona-relevant environments. We introduce PersonaGym, the first dynamic evaluation framework for persona agents, and PersonaScore, a human-aligned automatic metric grounded in decision theory that enables comprehensive large-scale evaluation. Our evaluation of 10 leading LLMs across 200 personas and 10,000 questions reveals significant advancement opportunities. For example, GPT-4.1 had the exact same PersonaScore as LLaMA-3-8b despite being a more recent and advanced closed source model. Importantly, increased model size and complexity do not necessarily enhance persona agent capabilities, underscoring the need for algorithmic and architectural innovation toward faithful, performant persona agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13770v1">Ice Cream Doesn't Cause Drowning: Benchmarking LLMs Against Statistical Pitfalls in Causal Inference</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      Reliable causal inference is essential for making decisions in high-stakes areas like medicine, economics, and public policy. However, it remains unclear whether large language models (LLMs) can handle rigorous and trustworthy statistical causal inference. Current benchmarks usually involve simplified tasks. For example, these tasks might only ask LLMs to identify semantic causal relationships or draw conclusions directly from raw data. As a result, models may overlook important statistical pitfalls, such as Simpson's paradox or selection bias. This oversight limits the applicability of LLMs in the real world. To address these limitations, we propose CausalPitfalls, a comprehensive benchmark designed to rigorously evaluate the capability of LLMs in overcoming common causal inference pitfalls. Our benchmark features structured challenges across multiple difficulty levels, each paired with grading rubrics. This approach allows us to quantitatively measure both causal reasoning capabilities and the reliability of LLMs' responses. We evaluate models using two protocols: (1) direct prompting, which assesses intrinsic causal reasoning, and (2) code-assisted prompting, where models generate executable code for explicit statistical analysis. Additionally, we validate the effectiveness of this judge by comparing its scoring with assessments from human experts. Our results reveal significant limitations in current LLMs when performing statistical causal inference. The CausalPitfalls benchmark provides essential guidance and quantitative metrics to advance the development of trustworthy causal reasoning systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13766v1">Advancing Software Quality: A Standards-Focused Review of LLM-Based Assurance Techniques</a></div>
    <div class="paper-meta">
      📅 2025-05-19
      | 💬 16 pages, 1 Table, 6 Figures
    </div>
    <details class="paper-abstract">
      Software Quality Assurance (SQA) is critical for delivering reliable, secure, and efficient software products. The Software Quality Assurance Process aims to provide assurance that work products and processes comply with predefined provisions and plans. Recent advancements in Large Language Models (LLMs) present new opportunities to enhance existing SQA processes by automating tasks like requirement analysis, code review, test generation, and compliance checks. Simultaneously, established standards such as ISO/IEC 12207, ISO/IEC 25010, ISO/IEC 5055, ISO 9001/ISO/IEC 90003, CMMI, and TMM provide structured frameworks for ensuring robust quality practices. This paper surveys the intersection of LLM-based SQA methods and these recognized standards, highlighting how AI-driven solutions can augment traditional approaches while maintaining compliance and process maturity. We first review the foundational software quality standards and the technical fundamentals of LLMs in software engineering. Next, we explore various LLM-based SQA applications, including requirement validation, defect detection, test generation, and documentation maintenance. We then map these applications to key software quality frameworks, illustrating how LLMs can address specific requirements and metrics within each standard. Empirical case studies and open-source initiatives demonstrate the practical viability of these methods. At the same time, discussions on challenges (e.g., data privacy, model bias, explainability) underscore the need for deliberate governance and auditing. Finally, we propose future directions encompassing adaptive learning, privacy-focused deployments, multimodal analysis, and evolving standards for AI-driven software quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12924v3">Interpreting token compositionality in LLMs: A robustness analysis</a></div>
    <div class="paper-meta">
      📅 2025-05-19
      | 💬 23 pages, 3 Figures, 14 tables
    </div>
    <details class="paper-abstract">
      Understanding the internal mechanisms of large language models (LLMs) is integral to enhancing their reliability, interpretability, and inference processes. We present Constituent-Aware Pooling (CAP), a methodology designed to analyse how LLMs process compositional linguistic structures. Grounded in principles of compositionality, mechanistic interpretability, and information theory, CAP systematically intervenes in model activations through constituent-based pooling at various model levels. Our experiments on inverse definition modelling, hypernym and synonym prediction reveal critical insights into transformers' limitations in handling compositional abstractions. No specific layer integrates tokens into unified semantic representations based on their constituent parts. We observe fragmented information processing, which intensifies with model size, suggesting that larger models struggle more with these interventions and exhibit greater information dispersion. This fragmentation likely stems from transformers' training objectives and architectural design, preventing systematic and cohesive representations. Our findings highlight fundamental limitations in current transformer architectures regarding compositional semantics processing and model interpretability, underscoring the critical need for novel approaches in LLM design to address these challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13757v1">LLM-Based Compact Reranking with Document Features for Scientific Retrieval</a></div>
    <div class="paper-meta">
      📅 2025-05-19
      | 💬 17 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Scientific retrieval is essential for advancing academic discovery. Within this process, document reranking plays a critical role by refining first-stage retrieval results. However, large language model (LLM) listwise reranking faces unique challenges in the scientific domain. First-stage retrieval is often suboptimal in the scientific domain, so relevant documents are ranked lower. Moreover, conventional listwise reranking uses the full text of candidate documents in the context window, limiting the number of candidates that can be considered. As a result, many relevant documents are excluded before reranking, which constrains overall retrieval performance. To address these challenges, we explore compact document representations based on semantic features such as categories, sections, and keywords, and propose a training-free, model-agnostic reranking framework for scientific retrieval called CoRank. The framework involves three stages: (i) offline extraction of document-level features, (ii) coarse reranking using these compact representations, and (iii) fine-grained reranking on full texts of the top candidates from stage (ii). This hybrid design provides a high-level abstraction of document semantics, expands candidate coverage, and retains critical details required for precise ranking. Experiments on LitSearch and CSFCube show that CoRank significantly improves reranking performance across different LLM backbones, increasing nDCG@10 from 32.0 to 39.7. Overall, these results highlight the value of information extraction for reranking in scientific retrieval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02362v4">Premise-Augmented Reasoning Chains Improve Error Identification in Math reasoning with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-19
      | 💬 Accepted at ICML 2025
    </div>
    <details class="paper-abstract">
      Chain-of-Thought (CoT) prompting enhances mathematical reasoning in large language models (LLMs) by enabling detailed step-by-step solutions. However, due to the verbosity of LLMs, the resulting reasoning chains can be long, making it harder to verify the reasoning steps and trace issues resulting from dependencies between the steps that may be farther away in the sequence of steps. Importantly, mathematical reasoning allows each step to be derived from a small set of premises, which are a subset of the preceding steps in the reasoning chain. In this paper, we present a framework that identifies the premises for each step, to improve the evaluation of reasoning. We restructure conventional linear reasoning chains into Premise Augmented Reasoning Chains (PARC) by introducing premise links, resulting in a directed acyclic graph where the nodes are the steps and the edges are the premise links. Through experiments with a PARC-based dataset that we built, namely PERL (Premises and ERrors identification in LLMs), we demonstrate that LLMs can reliably identify premises within complex reasoning chains. In particular, even open-source LLMs achieve 90% recall in premise identification. We also show that PARC helps to identify errors in reasoning chains more reliably. The accuracy of error identification improves by 6% to 16% absolute when step-by-step verification is carried out in PARC under the premises. Our findings highlight the utility of premise-centric representations in addressing complex problem-solving tasks and open new avenues for improving the reliability of LLM-based reasoning evaluations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.01082v5">Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs</a></div>
    <div class="paper-meta">
      📅 2025-05-19
      | 💬 In line with ICLR/Openreview changes + better overall reading flow. https://iclr.cc/virtual/2025/poster/30358
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) generate text by sampling the next token from a probability distribution over the vocabulary at each decoding step. Popular sampling methods like top-p (nucleus sampling) often struggle to balance quality and diversity, especially at higher temperatures which lead to incoherent or repetitive outputs. We propose min-p sampling, a dynamic truncation method that adjusts the sampling threshold based on the model's confidence by using the top token's probability as a scaling factor. Our experiments on benchmarks including GPQA, GSM8K, and AlpacaEval Creative Writing show that min-p sampling improves both the quality and diversity of generated text across different model families (Mistral and Llama 3) and model sizes (1B to 123B parameters), especially at higher temperatures. Human evaluations further show a clear preference for min-p sampling, in both text quality and creativity. Min-p sampling has been adopted by popular open-source LLM frameworks, including Hugging Face Transformers, VLLM, and many others, highlighting its considerable impact on improving text generation quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13738v1">Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      Efficient LLM pre-training requires well-tuned hyperparameters (HPs), including learning rate {\eta} and weight decay {\lambda}. We study scaling laws for HPs: formulas for how to scale HPs as we scale model size N, dataset size D, and batch size B. Recent work suggests the AdamW timescale, B/({\eta}{\lambda}D), should remain constant across training settings, and we verify the implication that optimal {\lambda} scales linearly with B, for a fixed N,D. However, as N,D scale, we show the optimal timescale obeys a precise power law in the tokens-per-parameter ratio, D/N. This law thus provides a method to accurately predict {\lambda}opt in advance of large-scale training. We also study scaling laws for optimal batch size Bopt (the B enabling lowest loss at a given N,D) and critical batch size Bcrit (the B beyond which further data parallelism becomes ineffective). In contrast with prior work, we find both Bopt and Bcrit scale as power laws in D, independent of model size, N. Finally, we analyze how these findings inform the real-world selection of Pareto-optimal N and D under dual training time and compute objectives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14161v2">TheAgentCompany: Benchmarking LLM Agents on Consequential Real World Tasks</a></div>
    <div class="paper-meta">
      📅 2025-05-19
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      We interact with computers on an everyday basis, be it in everyday life or work, and many aspects of work can be done entirely with access to a computer and the Internet. At the same time, thanks to improvements in large language models (LLMs), there has also been a rapid development in AI agents that interact with and affect change in their surrounding environments. But how performant are AI agents at accelerating or even autonomously performing work-related tasks? The answer to this question has important implications both for industry looking to adopt AI into their workflows and for economic policy to understand the effects that adoption of AI may have on the labor market. To measure the progress of these LLM agents' performance on performing real-world professional tasks, in this paper we introduce TheAgentCompany, an extensible benchmark for evaluating AI agents that interact with the world in similar ways to those of a digital worker: by browsing the Web, writing code, running programs, and communicating with other coworkers. We build a self-contained environment with internal web sites and data that mimics a small software company environment, and create a variety of tasks that may be performed by workers in such a company. We test baseline agents powered by both closed API-based and open-weights language models (LMs), and find that the most competitive agent can complete 30% of tasks autonomously. This paints a nuanced picture on task automation with LM agents--in a setting simulating a real workplace, a good portion of simpler tasks could be solved autonomously, but more difficult long-horizon tasks are still beyond the reach of current systems. We release code, data, environment, and experiments on https://the-agent-company.com.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13725v1">SQLForge: Synthesizing Reliable and Diverse Data to Enhance Text-to-SQL Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-19
      | 💬 12 pages, 7 figures, accepted to ACL Findings 2025
    </div>
    <details class="paper-abstract">
      Large Language models (LLMs) have demonstrated significant potential in text-to-SQL reasoning tasks, yet a substantial performance gap persists between existing open-source models and their closed-source counterparts. In this paper, we introduce SQLForge, a novel approach for synthesizing reliable and diverse data to enhance text-to-SQL reasoning in LLMs. We improve data reliability through SQL syntax constraints and SQL-to-question reverse translation, ensuring data logic at both structural and semantic levels. We also propose an SQL template enrichment and iterative data domain exploration mechanism to boost data diversity. Building on the augmented data, we fine-tune a variety of open-source models with different architectures and parameter sizes, resulting in a family of models termed SQLForge-LM. SQLForge-LM achieves the state-of-the-art performance on the widely recognized Spider and BIRD benchmarks among the open-source models. Specifically, SQLForge-LM achieves EX accuracy of 85.7% on Spider Dev and 59.8% on BIRD Dev, significantly narrowing the performance gap with closed-source methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.19202v3">Improving LLM Unlearning Robustness via Random Perturbations</a></div>
    <div class="paper-meta">
      📅 2025-05-19
      | 💬 23 pages, 10 figures, 5 tables
    </div>
    <details class="paper-abstract">
      In this paper, we show that current state-of-the-art LLM unlearning methods inherently reduce models' robustness, causing them to misbehave even when a single non-adversarial forget-token is in the retain-query. Toward understanding underlying causes, we reframe the unlearning process as backdoor attacks and defenses: forget-tokens act as backdoor triggers that, when activated in retain-queries, cause disruptions in unlearned models' behaviors, similar to successful backdoor attacks. To mitigate this vulnerability, we propose Random Noise Augmentation (RNA) -- a plug-and-play, model and method agnostic approach with theoretical guarantees for improving the robustness of unlearned models. Extensive experiments demonstrate that RNA significantly improves the robustness of unlearned models, maintains unlearning performances while introducing no additional computational overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13697v1">RL in Name Only? Analyzing the Structural Assumptions in RL post-training for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      Reinforcement learning-based post-training of large language models (LLMs) has recently gained attention, particularly following the release of DeepSeek R1, which applied GRPO for fine-tuning. Amid the growing hype around improved reasoning abilities attributed to RL post-training, we critically examine the formulation and assumptions underlying these methods. We start by highlighting the popular structural assumptions made in modeling LLM training as a Markov Decision Process (MDP), and show how they lead to a degenerate MDP that doesn't quite need the RL/GRPO apparatus. The two critical structural assumptions include (1) making the MDP states be just a concatenation of the actions-with states becoming the context window and the actions becoming the tokens in LLMs and (2) splitting the reward of a state-action trajectory uniformly across the trajectory. Through a comprehensive analysis, we demonstrate that these simplifying assumptions make the approach effectively equivalent to an outcome-driven supervised learning. Our experiments on benchmarks including GSM8K and Countdown using Qwen-2.5 base models show that iterative supervised fine-tuning, incorporating both positive and negative samples, achieves performance comparable to GRPO-based training. We will also argue that the structural assumptions indirectly incentivize the RL to generate longer sequences of intermediate tokens-which in turn feeds into the narrative of "RL generating longer thinking traces." While RL may well be a very useful technique for improving the reasoning abilities of LLMs, our analysis shows that the simplistic structural assumptions made in modeling the underlying MDP render the popular LLM RL frameworks and their interpretations questionable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00038v2">HyPerAlign: Interpretable Personalized LLM Alignment via Hypothesis Generation</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      Alignment algorithms are widely used to align large language models (LLMs) to human users based on preference annotations. Typically these (often divergent) preferences are aggregated over a diverse set of users, resulting in fine-tuned models that are aligned to the ``average-user'' preference. Nevertheless, current models are used by individual users in very specific contexts and situations, emphasizing the need for user-dependent preference control. In this work we address the problem of personalizing LLM outputs to their users. We aim to generate customized responses tailored to specific individuals instead of generic outputs that emulate the collective voices of diverse populations. We propose HyPerAlign, an interpretable and sample-efficient hypothesis-driven personalization approach for LLM models. Given few-shot examples written by a particular user, we first infer hypotheses about their communication strategies, personality, and writing style, then prompt LLM models with these hypotheses and user-specific attributes to generate customized outputs. We conduct experiments on two different personalization tasks, namely authorship attribution and deliberative alignment, with datasets from diverse domains (news articles, blog posts, emails, jailbreaking benchmarks). Results demonstrate the superiority of hypothesis-driven LLM personalization compared to preference-based fine-tuning methods. For authorship attribution, HyPerAlign generations have consistently high win-rates (commonly $> 90\%$) against state-of-the-art preference fine-tuning approaches across diverse user profiles and LLM models. For deliberative alignment, the helpfulness of LLM models is improved by up to $70\%$ on average. Overall, HyPerAlign represents an interpretable and sample-efficient strategy for the personalization of LLM models to individual users.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.00948v4">Large Linguistic Models: Investigating LLMs' metalinguistic abilities</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      The performance of large language models (LLMs) has recently improved to the point where models can perform well on many language tasks. We show here that--for the first time--the models can also generate valid metalinguistic analyses of language data. We outline a research program where the behavioral interpretability of LLMs on these tasks is tested via prompting. LLMs are trained primarily on text--as such, evaluating their metalinguistic abilities improves our understanding of their general capabilities and sheds new light on theoretical models in linguistics. We show that OpenAI's (2024) o1 vastly outperforms other models on tasks involving drawing syntactic trees and phonological generalization. We speculate that OpenAI o1's unique advantage over other models may result from the model's chain-of-thought mechanism, which mimics the structure of human reasoning used in complex cognitive tasks, such as linguistic analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13418v1">Dementia Through Different Eyes: Explainable Modeling of Human and LLM Perceptions for Early Awareness</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      Cognitive decline often surfaces in language years before diagnosis. It is frequently non-experts, such as those closest to the patient, who first sense a change and raise concern. As LLMs become integrated into daily communication and used over prolonged periods, it may even be an LLM that notices something is off. But what exactly do they notice--and should be noticing--when making that judgment? This paper investigates how dementia is perceived through language by non-experts. We presented transcribed picture descriptions to non-expert humans and LLMs, asking them to intuitively judge whether each text was produced by someone healthy or with dementia. We introduce an explainable method that uses LLMs to extract high-level, expert-guided features representing these picture descriptions, and use logistic regression to model human and LLM perceptions and compare with clinical diagnoses. Our analysis reveals that human perception of dementia is inconsistent and relies on a narrow, and sometimes misleading, set of cues. LLMs, by contrast, draw on a richer, more nuanced feature set that aligns more closely with clinical patterns. Still, both groups show a tendency toward false negatives, frequently overlooking dementia cases. Through our interpretable framework and the insights it provides, we hope to help non-experts better recognize the linguistic signs that matter.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13416v1">Gluon: Making Muon & Scion Great Again! (Bridging Theory and Practice of LMO-based Optimizers for LLMs)</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      Recent developments in deep learning optimization have brought about radically new algorithms based on the Linear Minimization Oracle (LMO) framework, such as $\sf Muon$ and $\sf Scion$. After over a decade of $\sf Adam$'s dominance, these LMO-based methods are emerging as viable replacements, offering several practical advantages such as improved memory efficiency, better hyperparameter transferability, and most importantly, superior empirical performance on large-scale tasks, including LLM training. However, a significant gap remains between their practical use and our current theoretical understanding: prior analyses (1) overlook the layer-wise LMO application of these optimizers in practice, and (2) rely on an unrealistic smoothness assumption, leading to impractically small stepsizes. To address both, we propose a new LMO-based method called $\sf Gluon$, capturing prior theoretically analyzed methods as special cases, and introduce a new refined generalized smoothness model that captures the layer-wise geometry of neural networks, matches the layer-wise practical implementation of $\sf Muon$ and $\sf Scion$, and leads to convergence guarantees with strong practical predictive power. Unlike prior results, our theoretical stepsizes closely match the fine-tuned values reported by Pethick et al. (2025). Our experiments with NanoGPT and CNN confirm that our assumption holds along the optimization trajectory, ultimately closing the gap between theory and practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13406v1">AutoMathKG: The automated mathematical knowledge graph based on LLM and vector database</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      A mathematical knowledge graph (KG) presents knowledge within the field of mathematics in a structured manner. Constructing a math KG using natural language is an essential but challenging task. There are two major limitations of existing works: first, they are constrained by corpus completeness, often discarding or manually supplementing incomplete knowledge; second, they typically fail to fully automate the integration of diverse knowledge sources. This paper proposes AutoMathKG, a high-quality, wide-coverage, and multi-dimensional math KG capable of automatic updates. AutoMathKG regards mathematics as a vast directed graph composed of Definition, Theorem, and Problem entities, with their reference relationships as edges. It integrates knowledge from ProofWiki, textbooks, arXiv papers, and TheoremQA, enhancing entities and relationships with large language models (LLMs) via in-context learning for data augmentation. To search for similar entities, MathVD, a vector database, is built through two designed embedding strategies using SBERT. To automatically update, two mechanisms are proposed. For knowledge completion mechanism, Math LLM is developed to interact with AutoMathKG, providing missing proofs or solutions. For knowledge fusion mechanism, MathVD is used to retrieve similar entities, and LLM is used to determine whether to merge with a candidate or add as a new entity. A wide range of experiments demonstrate the advanced performance and broad applicability of the AutoMathKG system, including superior reachability query results in MathVD compared to five baselines and robust mathematical reasoning capability in Math LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13379v1">Thinkless: LLM Learns When to Think</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      Reasoning Language Models, capable of extended chain-of-thought reasoning, have demonstrated remarkable performance on tasks requiring complex logical inference. However, applying elaborate reasoning for all queries often results in substantial computational inefficiencies, particularly when many problems admit straightforward solutions. This motivates an open question: Can LLMs learn when to think? To answer this, we propose Thinkless, a learnable framework that empowers an LLM to adaptively select between short-form and long-form reasoning, based on both task complexity and the model's ability. Thinkless is trained under a reinforcement learning paradigm and employs two control tokens, <short> for concise responses and <think> for detailed reasoning. At the core of our method is a Decoupled Group Relative Policy Optimization (DeGRPO) algorithm, which decomposes the learning objective of hybrid reasoning into two components: (1) a control token loss that governs the selection of the reasoning mode, and (2) a response loss that improves the accuracy of the generated answers. This decoupled formulation enables fine-grained control over the contributions of each objective, stabilizing training and effectively preventing collapse observed in vanilla GRPO. Empirically, on several benchmarks such as Minerva Algebra, MATH-500, and GSM8K, Thinkless is able to reduce the usage of long-chain thinking by 50% - 90%, significantly improving the efficiency of Reasoning Language Models. The code is available at https://github.com/VainF/Thinkless
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13376v1">Seeing, Saying, Solving: An LLM-to-TL Framework for Cooperative Robots</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      Increased robot deployment, such as in warehousing, has revealed a need for seamless collaboration among heterogeneous robot teams to resolve unforeseen conflicts. To address this challenge, we propose a novel, decentralized framework for robots to request and provide help. The framework begins with robots detecting conflicts using a Vision Language Model (VLM), then reasoning over whether help is needed. If so, it crafts and broadcasts a natural language (NL) help request using a Large Language Model (LLM). Potential helper robots reason over the request and offer help (if able), along with information about impact to their current tasks. Helper reasoning is implemented via an LLM grounded in Signal Temporal Logic (STL) using a Backus-Naur Form (BNF) grammar to guarantee syntactically valid NL-to-STL translations, which are then solved as a Mixed Integer Linear Program (MILP). Finally, the requester robot chooses a helper by reasoning over impact on the overall system. We evaluate our system via experiments considering different strategies for choosing a helper, and find that a requester robot can minimize overall time impact on the system by considering multiple help offers versus simple heuristics (e.g., selecting the nearest robot to help).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11380v2">From the New World of Word Embeddings: A Comparative Study of Small-World Lexico-Semantic Networks in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-19
      | 💬 Paper under review
    </div>
    <details class="paper-abstract">
      Lexico-semantic networks represent words as nodes and their semantic relatedness as edges. While such networks are traditionally constructed using embeddings from encoder-based models or static vectors, embeddings from decoder-only large language models (LLMs) remain underexplored. Unlike encoder models, LLMs are trained with a next-token prediction objective, which does not directly encode the meaning of the current token. In this paper, we construct lexico-semantic networks from the input embeddings of LLMs with varying parameter scales and conduct a comparative analysis of their global and local structures. Our results show that these networks exhibit small-world properties, characterized by high clustering and short path lengths. Moreover, larger LLMs yield more intricate networks with less small-world effects and longer paths, reflecting richer semantic structures and relations. We further validate our approach through analyses of common conceptual pairs, structured lexical relations derived from WordNet, and a cross-lingual semantic network for qualitative words.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13360v1">What Prompts Don't Say: Understanding and Managing Underspecification in LLM Prompts</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      Building LLM-powered software requires developers to communicate their requirements through natural language, but developer prompts are frequently underspecified, failing to fully capture many user-important requirements. In this paper, we present an in-depth analysis of prompt underspecification, showing that while LLMs can often (41.1%) guess unspecified requirements by default, such behavior is less robust: Underspecified prompts are 2x more likely to regress over model or prompt changes, sometimes with accuracy drops by more than 20%. We then demonstrate that simply adding more requirements to a prompt does not reliably improve performance, due to LLMs' limited instruction-following capabilities and competing constraints, and standard prompt optimizers do not offer much help. To address this, we introduce novel requirements-aware prompt optimization mechanisms that can improve performance by 4.8% on average over baselines that naively specify everything in the prompt. Beyond prompt optimization, we envision that effectively managing prompt underspecification requires a broader process, including proactive requirements discovery, evaluation, and monitoring.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13348v1">Investigating the Vulnerability of LLM-as-a-Judge Architectures to Prompt-Injection Attacks</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly employed as evaluators (LLM-as-a-Judge) for assessing the quality of machine-generated text. This paradigm offers scalability and cost-effectiveness compared to human annotation. However, the reliability and security of such systems, particularly their robustness against adversarial manipulations, remain critical concerns. This paper investigates the vulnerability of LLM-as-a-Judge architectures to prompt-injection attacks, where malicious inputs are designed to compromise the judge's decision-making process. We formalize two primary attack strategies: Comparative Undermining Attack (CUA), which directly targets the final decision output, and Justification Manipulation Attack (JMA), which aims to alter the model's generated reasoning. Using the Greedy Coordinate Gradient (GCG) optimization method, we craft adversarial suffixes appended to one of the responses being compared. Experiments conducted on the MT-Bench Human Judgments dataset with open-source instruction-tuned LLMs (Qwen2.5-3B-Instruct and Falcon3-3B-Instruct) demonstrate significant susceptibility. The CUA achieves an Attack Success Rate (ASR) exceeding 30\%, while JMA also shows notable effectiveness. These findings highlight substantial vulnerabilities in current LLM-as-a-Judge systems, underscoring the need for robust defense mechanisms and further research into adversarial evaluation and trustworthiness in LLM-based assessment frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08140v2">Lost in Transmission: When and Why LLMs Fail to Reason Globally</a></div>
    <div class="paper-meta">
      📅 2025-05-19
      | 💬 28 pages
    </div>
    <details class="paper-abstract">
      Despite their many successes, transformer-based large language models (LLMs) continue to struggle with tasks that require complex reasoning over large parts of their input. We argue that these failures arise due to capacity limits on the accurate flow of information within LLMs. To formalize this issue, we introduce the bounded attention prefix oracle (BAPO) model, a new computational framework that models bandwidth constraints on attention heads, the mechanism for internal communication in LLMs. We show that several important reasoning problems like graph reachability require high communication bandwidth for BAPOs to solve; we call these problems BAPO-hard. Our experiments corroborate our theoretical predictions: GPT-4o, Claude, and Gemini succeed on BAPO-easy tasks and fail even on relatively small BAPO-hard tasks. BAPOs also reveal another benefit of chain of thought (CoT): we prove that breaking down a task using CoT can turn any BAPO-hard problem into a BAPO-easy one. Our results offer principled explanations for key LLM failures and suggest directions for architectures and inference methods that mitigate bandwidth limits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13326v1">Thinking Short and Right Over Thinking Long: Serving LLM Reasoning Efficiently and Accurately</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      Recent advances in test-time scaling suggest that Large Language Models (LLMs) can gain better capabilities by generating Chain-of-Thought reasoning (analogous to human thinking) to respond a given request, and meanwhile exploring more reasoning branches (i.e., generating multiple responses and ensembling them) can improve the final output quality. However, when incorporating the two scaling dimensions, we find that the system efficiency is dampened significantly for two reasons. Firstly, the time cost to generate the final output increases substantially as many reasoning branches would be trapped in the over-thinking dilemma, producing excessively long responses. Secondly, generating multiple reasoning branches for each request increases memory consumption, which is unsuitable for LLM serving since we can only batch a limited number of requests to process simultaneously. To address this, we present SART, a serving framework for efficient and accurate LLM reasoning. The essential idea is to manage the thinking to be short and right, rather than long. For one thing, we devise a redundant sampling with early stopping approach based on empirical observations and theoretic analysis, which increases the likelihood of obtaining short-thinking responses when sampling reasoning branches. For another, we propose to dynamically prune low-quality branches so that only right-thinking branches are maintained, reducing the memory consumption and allowing us to batch more requests. Experimental results demonstrate that SART not only improves the accuracy of LLM reasoning but also enhances the serving efficiency, outperforming existing methods by up to 28.2 times and on average 15.7 times in terms of efficiency when achieving the same level of accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13312v1">GUARD: Generation-time LLM Unlearning via Adaptive Restriction and Detection</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated strong capabilities in memorizing vast amounts of knowledge across diverse domains. However, the ability to selectively forget specific knowledge is critical for ensuring the safety and compliance of deployed models. Existing unlearning efforts typically fine-tune the model with resources such as forget data, retain data, and a calibration model. These additional gradient steps blur the decision boundary between forget and retain knowledge, making unlearning often at the expense of overall performance. To avoid the negative impact of fine-tuning, it would be better to unlearn solely at inference time by safely guarding the model against generating responses related to the forget target, without destroying the fluency of text generation. In this work, we propose Generation-time Unlearning via Adaptive Restriction and Detection (GUARD), a framework that enables dynamic unlearning during LLM generation. Specifically, we first employ a prompt classifier to detect unlearning targets and extract the corresponding forbidden token. We then dynamically penalize and filter candidate tokens during generation using a combination of token matching and semantic matching, effectively preventing the model from leaking the forgotten content. Experimental results on copyright content unlearning tasks over the Harry Potter dataset and the MUSE benchmark, as well as entity unlearning tasks on the TOFU dataset, demonstrate that GUARD achieves strong forget quality across various tasks while causing almost no degradation to the LLM's general capabilities, striking an excellent trade-off between forgetting and utility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.17140v2">AXIS: Efficient Human-Agent-Computer Interaction with API-First LLM-Based Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      Multimodal large language models (MLLMs) have enabled LLM-based agents to directly interact with application user interfaces (UIs), enhancing agents' performance in complex tasks. However, these agents often suffer from high latency and low reliability due to the extensive sequential UI interactions. To address this issue, we propose AXIS, a novel LLM-based agents framework that prioritize actions through application programming interfaces (APIs) over UI actions. This framework also facilitates the creation and expansion of APIs through automated exploration of applications. Our experiments on Microsoft Word demonstrate that AXIS reduces task completion time by 65%-70% and cognitive workload by 38%-53%, while maintaining accuracy of 97%-98% compared to humans. Our work contributes to a new human-agent-computer interaction (HACI) framework and explores a fresh UI design principle for application providers to turn applications into agents in the era of LLMs, paving the way towards an agent-centric operating system (Agent OS).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.09493v2">Hacking, The Lazy Way: LLM Augmented Pentesting</a></div>
    <div class="paper-meta">
      📅 2025-05-19
      | 💬 This work has been submitted to the IEEE for possible publication. Nisha P. Shetty has been added as an author as the new version includes work under her supervision, enhancing the research. Significant changes have been made in the methodology, survey, and introduction sections
    </div>
    <details class="paper-abstract">
      In our research, we introduce a new concept called "LLM Augmented Pentesting" demonstrated with a tool named "Pentest Copilot," that revolutionizes the field of ethical hacking by integrating Large Language Models (LLMs) into penetration testing workflows, leveraging the advanced GPT-4-turbo model. Our approach focuses on overcoming the traditional resistance to automation in penetration testing by employing LLMs to automate specific sub-tasks while ensuring a comprehensive understanding of the overall testing process. Pentest Copilot showcases remarkable proficiency in tasks such as utilizing testing tools, interpreting outputs, and suggesting follow-up actions, efficiently bridging the gap between automated systems and human expertise. By integrating a "chain of thought" mechanism, Pentest Copilot optimizes token usage and enhances decision-making processes, leading to more accurate and context-aware outputs. Additionally, our implementation of Retrieval-Augmented Generation (RAG) minimizes hallucinations and ensures the tool remains aligned with the latest cybersecurity techniques and knowledge. We also highlight a unique infrastructure system that supports in-browser penetration testing, providing a robust platform for cybersecurity professionals. Our findings demonstrate that LLM Augmented Pentesting can not only significantly enhance task completion rates in penetration testing but also effectively addresses real-world challenges, marking a substantial advancement in the cybersecurity domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.16707v3">Enhancing LLMs for Power System Simulations: A Feedback-driven Multi-agent Framework</a></div>
    <div class="paper-meta">
      📅 2025-05-19
      | 💬 16 pages
    </div>
    <details class="paper-abstract">
      The integration of experimental technologies with large language models (LLMs) is transforming scientific research. It positions AI as a versatile research assistant rather than a mere problem-solving tool. In the field of power systems, however, managing simulations -- one of the essential experimental technologies -- remains a challenge for LLMs due to their limited domain-specific knowledge, restricted reasoning capabilities, and imprecise handling of simulation parameters. To address these limitations, this paper proposes a feedback-driven, multi-agent framework. It incorporates three proposed modules: an enhanced retrieval-augmented generation (RAG) module, an improved reasoning module, and a dynamic environmental acting module with an error-feedback mechanism. Validated on 69 diverse tasks from Daline and MATPOWER, this framework achieves success rates of 93.13% and 96.85%, respectively. It significantly outperforms ChatGPT 4o, o1-preview, and the fine-tuned GPT-4o, which all achieved a success rate lower than 30% on complex tasks. Additionally, the proposed framework also supports rapid, cost-effective task execution, completing each simulation in approximately 30 seconds at an average cost of 0.014 USD for tokens. Overall, this adaptable framework lays a foundation for developing intelligent LLM-based assistants for human researchers, facilitating power system research and beyond.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13263v1">Are requirements really all you need? A case study of LLM-driven configuration code generation for automotive simulations</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are taking many industries by storm. They possess impressive reasoning capabilities and are capable of handling complex problems, as shown by their steadily improving scores on coding and mathematical benchmarks. However, are the models currently available truly capable of addressing real-world challenges, such as those found in the automotive industry? How well can they understand high-level, abstract instructions? Can they translate these instructions directly into functional code, or do they still need help and supervision? In this work, we put one of the current state-of-the-art models to the test. We evaluate its performance in the task of translating abstract requirements, extracted from automotive standards and documents, into configuration code for CARLA simulations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13246v1">Agentic Publications: An LLM-Driven Framework for Interactive Scientific Publishing, Supplementing Traditional Papers with AI-Powered Knowledge Systems</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      The exponential growth of scientific literature presents significant challenges for researchers navigating the complex knowledge landscape. We propose "Agentic Publications", a novel LLM-driven framework complementing traditional publishing by transforming papers into interactive knowledge systems. Our architecture integrates structured data with unstructured content through retrieval-augmented generation and multi-agent verification. The framework offers interfaces for both humans and machines, combining narrative explanations with machine-readable outputs while addressing ethical considerations through automated validation and transparent governance. Key features include continuous knowledge updates, automatic integration of new findings, and customizable detail levels. Our proof-of-concept demonstrates multilingual interaction, API accessibility, and structured knowledge representation through vector databases, knowledge graphs, and verification agents. This approach enhances scientific communication across disciplines, improving efficiency and collaboration while preserving traditional publishing pathways, particularly valuable for interdisciplinary fields where knowledge integration remains challenging.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.11266v5">VersaTune: An Efficient Data Composition Framework for Training Multi-Capability LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      As demonstrated by the proprietary Large Language Models (LLMs) such as GPT and Claude series, LLMs have the potential to achieve remarkable proficiency across a wide range of domains, including law, medicine, finance, science, code, etc., all within a single model. These capabilities are further augmented during the Supervised Fine-Tuning (SFT) phase. Despite their potential, existing work mainly focuses on domain-specific enhancements during fine-tuning, the challenge of which lies in catastrophic forgetting of knowledge across other domains. In this study, we introduce **VersaTune**, a novel data composition framework designed for enhancing LLMs' overall multi-domain capabilities during training. We begin with detecting the distribution of domain-specific knowledge within the base model, followed by the training data composition that aligns with the model's existing knowledge distribution. During the subsequent training process, domain weights are dynamically adjusted based on their learnable potential and forgetting degree. Experimental results indicate that VersaTune is effective in multi-domain fostering, with an improvement of 35.21\% in the overall multi-ability performances compared to uniform domain weights. Furthermore, we find that Qwen-2.5-32B + VersaTune even surpasses frontier models, including GPT-4o, Claude3.5-Sonnet and DeepSeek-V3 by 0.86\%, 4.76\% and 4.60\%. Additionally, in scenarios where flexible expansion of a specific domain is required, VersaTune reduces the performance degradation in other domains by 38.77\%, while preserving the training efficacy of the target domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13195v1">Adversarial Testing in LLMs: Insights into Decision-Making Vulnerabilities</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) become increasingly integrated into real-world decision-making systems, understanding their behavioural vulnerabilities remains a critical challenge for AI safety and alignment. While existing evaluation metrics focus primarily on reasoning accuracy or factual correctness, they often overlook whether LLMs are robust to adversarial manipulation or capable of using adaptive strategy in dynamic environments. This paper introduces an adversarial evaluation framework designed to systematically stress-test the decision-making processes of LLMs under interactive and adversarial conditions. Drawing on methodologies from cognitive psychology and game theory, our framework probes how models respond in two canonical tasks: the two-armed bandit task and the Multi-Round Trust Task. These tasks capture key aspects of exploration-exploitation trade-offs, social cooperation, and strategic flexibility. We apply this framework to several state-of-the-art LLMs, including GPT-3.5, GPT-4, Gemini-1.5, and DeepSeek-V3, revealing model-specific susceptibilities to manipulation and rigidity in strategy adaptation. Our findings highlight distinct behavioral patterns across models and emphasize the importance of adaptability and fairness recognition for trustworthy AI deployment. Rather than offering a performance benchmark, this work proposes a methodology for diagnosing decision-making weaknesses in LLM-based agents, providing actionable insights for alignment and safety research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13175v1">Enhancing LLMs for Time Series Forecasting via Structure-Guided Cross-Modal Alignment</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      The emerging paradigm of leveraging pretrained large language models (LLMs) for time series forecasting has predominantly employed linguistic-temporal modality alignment strategies through token-level or layer-wise feature mapping. However, these approaches fundamentally neglect a critical insight: the core competency of LLMs resides not merely in processing localized token features but in their inherent capacity to model holistic sequence structures. This paper posits that effective cross-modal alignment necessitates structural consistency at the sequence level. We propose the Structure-Guided Cross-Modal Alignment (SGCMA), a framework that fully exploits and aligns the state-transition graph structures shared by time-series and linguistic data as sequential modalities, thereby endowing time series with language-like properties and delivering stronger generalization after modality alignment. SGCMA consists of two key components, namely Structure Alignment and Semantic Alignment. In Structure Alignment, a state transition matrix is learned from text data through Hidden Markov Models (HMMs), and a shallow transformer-based Maximum Entropy Markov Model (MEMM) receives the hot-start transition matrix and annotates each temporal patch into state probability, ensuring that the temporal representation sequence inherits language-like sequential dynamics. In Semantic Alignment, cross-attention is applied between temporal patches and the top-k tokens within each state, and the ultimate temporal embeddings are derived by the expected value of these embeddings using a weighted average based on state probabilities. Experiments on multiple benchmarks demonstrate that SGCMA achieves state-of-the-art performance, offering a novel approach to cross-modal alignment in time series forecasting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13173v1">A Case Study of Cross-Lingual Zero-Shot Generalization for Classical Languages in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-19
      | 💬 Accepted to ACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable generalization capabilities across diverse tasks and languages. In this study, we focus on natural language understanding in three classical languages -- Sanskrit, Ancient Greek and Latin -- to investigate the factors affecting cross-lingual zero-shot generalization. First, we explore named entity recognition and machine translation into English. While LLMs perform equal to or better than fine-tuned baselines on out-of-domain data, smaller models often struggle, especially with niche or abstract entity types. In addition, we concentrate on Sanskrit by presenting a factoid question-answering (QA) dataset and show that incorporating context via retrieval-augmented generation approach significantly boosts performance. In contrast, we observe pronounced performance drops for smaller LLMs across these QA tasks. These results suggest model scale as an important factor influencing cross-lingual generalization. Assuming that models used such as GPT-4o and Llama-3.1 are not instruction fine-tuned on classical languages, our findings provide insights into how LLMs may generalize on these languages and their consequent utility in classical studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.02613v2">ACCO: Accumulate While You Communicate for Communication-Overlapped Sharded LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      Training LLMs relies on distributed implementations using multiple GPUs to compute gradients in parallel with sharded optimizers. However, synchronizing gradients in data parallel setups introduces communication overhead that grows with the number of workers, limiting parallelization efficiency. Local optimization algorithms reduce communications but incur high memory costs as they prevent optimizer state sharding, hindering scalability. To address this, we propose \textbf{AC}cumulate while \textbf{CO}mmunicate (\acco), a memory-efficient optimization algorithm for distributed LLM training. By synchronizing delayed gradients while computing new ones, \acco~reduces GPU idle time and supports heterogeneous hardware. To mitigate the convergence issues caused by delayed updates, we introduce a novel technique ensuring training dynamics align with standard distributed optimization. Compared to ZeRO-1, our approach is significantly faster and scales effectively across heterogeneous hardware.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13691v2">Is This Collection Worth My LLM's Time? Automatically Measuring Information Potential in Text Corpora</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) converge towards similar capabilities, the key to advancing their performance lies in identifying and incorporating valuable new information sources. However, evaluating which text collections are worth the substantial investment required for digitization, preprocessing, and integration into LLM systems remains a significant challenge. We present a novel approach to this challenge: an automated pipeline that evaluates the potential information gain from text collections without requiring model training or fine-tuning. Our method generates multiple choice questions (MCQs) from texts and measures an LLM's performance both with and without access to the source material. The performance gap between these conditions serves as a proxy for the collection's information potential. We validate our approach using five strategically selected datasets: EPFL PhD manuscripts, a private collection of Venetian historical records, two sets of Wikipedia articles on related topics, and a synthetic baseline dataset. Our results demonstrate that this method effectively identifies collections containing valuable novel information, providing a practical tool for prioritizing data acquisition and integration efforts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07610v2">Concept-Level Explainability for Auditing & Steering LLM Responses</a></div>
    <div class="paper-meta">
      📅 2025-05-19
      | 💬 9 pages, 7 figures, Submission to Neurips 2025
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become widely deployed, concerns about their safety and alignment grow. An approach to steer LLM behavior, such as mitigating biases or defending against jailbreaks, is to identify which parts of a prompt influence specific aspects of the model's output. Token-level attribution methods offer a promising solution, but still struggle in text generation, explaining the presence of each token in the output separately, rather than the underlying semantics of the entire LLM response. We introduce ConceptX, a model-agnostic, concept-level explainability method that identifies the concepts, i.e., semantically rich tokens in the prompt, and assigns them importance based on the outputs' semantic similarity. Unlike current token-level methods, ConceptX also offers to preserve context integrity through in-place token replacements and supports flexible explanation goals, e.g., gender bias. ConceptX enables both auditing, by uncovering sources of bias, and steering, by modifying prompts to shift the sentiment or reduce the harmfulness of LLM responses, without requiring retraining. Across three LLMs, ConceptX outperforms token-level methods like TokenSHAP in both faithfulness and human alignment. Steering tasks boost sentiment shift by 0.252 versus 0.131 for random edits and lower attack success rates from 0.463 to 0.242, outperforming attribution and paraphrasing baselines. While prompt engineering and self-explaining methods sometimes yield safer responses, ConceptX offers a transparent and faithful alternative for improving LLM safety and alignment, demonstrating the practical value of attribution-based explainability in guiding LLM behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11525v2">Beyond Single-Task: Robust Multi-Task Length Generalization for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      Length generalization, the ability to solve problems longer than those seen during training, remains a critical challenge for large language models (LLMs). Previous work modifies positional encodings (PEs) and data formats to improve length generalization on specific symbolic tasks such as addition and sorting. However, these approaches are fundamentally limited to special tasks, often degrading general language performance. Furthermore, they are typically evaluated on small transformers trained from scratch on single tasks and can cause performance drop when applied during post-training stage of practical LLMs with general capabilities. Hu et al., (2024) proposed Rule-Following Fine-Tuning (RFFT) to improve length generalization in the post-training stage of LLMs. Despite its compatibility with practical models and strong performance, RFFT is proposed for single tasks too, requiring re-training for each individual task with extensive examples. In this paper, we study length generalization in multi-task settings and propose Meta Rule-Following Fine-Tuning (Meta-RFFT), the first framework enabling robust cross-task length generalization. As our first contribution, we construct a large length generalization dataset containing 86 tasks spanning code execution, number processing, symbolic and logical reasoning tasks, beyond the common addition or multiplication tasks. Secondly, we show that cross-task length generalization is possible with Meta-RFFT. After training on a large number of tasks and instances, the models achieve remarkable length generalization ability on unseen tasks with minimal fine-tuning or one-shot prompting. For example, after fine-tuning on 1 to 5 digit addition, our 32B model achieves 95% accuracy on 30 digit addition, significantly outperforming the state-of-the-art reasoning models (DeepSeek-R1-671B: 72%), despite never seeing this task during RF-pretraining.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13109v1">FreeKV: Boosting KV Cache Retrieval for Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been widely deployed with rapidly expanding context windows to support increasingly demanding applications. However, long contexts pose significant deployment challenges, primarily due to the KV cache whose size grows proportionally with context length. While KV cache compression methods are proposed to address this issue, KV dropping methods incur considerable accuracy loss, and KV retrieval methods suffer from significant efficiency bottlenecks. We propose FreeKV, an algorithm-system co-optimization framework to enhance KV retrieval efficiency while preserving accuracy. On the algorithm side, FreeKV introduces speculative retrieval to shift the KV selection and recall processes out of the critical path, combined with fine-grained correction to ensure accuracy. On the system side, FreeKV employs hybrid KV layouts across CPU and GPU memory to eliminate fragmented data transfers, and leverages double-buffered streamed recall to further improve efficiency. Experiments demonstrate that FreeKV achieves near-lossless accuracy across various scenarios and models, delivering up to 13$\times$ speedup compared to SOTA KV retrieval methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13098v1">LLM-KG-Bench 3.0: A Compass for SemanticTechnology Capabilities in the Ocean of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-19
      | 💬 Peer reviewed publication at ESWC 2025 Resources Track
    </div>
    <details class="paper-abstract">
      Current Large Language Models (LLMs) can assist developing program code beside many other things, but can they support working with Knowledge Graphs (KGs) as well? Which LLM is offering the best capabilities in the field of Semantic Web and Knowledge Graph Engineering (KGE)? Is this possible to determine without checking many answers manually? The LLM-KG-Bench framework in Version 3.0 is designed to answer these questions. It consists of an extensible set of tasks for automated evaluation of LLM answers and covers different aspects of working with semantic technologies. In this paper the LLM-KG-Bench framework is presented in Version 3 along with a dataset of prompts, answers and evaluations generated with it and several state-of-the-art LLMs. Significant enhancements have been made to the framework since its initial release, including an updated task API that offers greater flexibility in handling evaluation tasks, revised tasks, and extended support for various open models through the vllm library, among other improvements. A comprehensive dataset has been generated using more than 30 contemporary open and proprietary LLMs, enabling the creation of exemplary model cards that demonstrate the models' capabilities in working with RDF and SPARQL, as well as comparing their performance on Turtle and JSON-LD RDF serialization tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13028v1">Evaluatiing the efficacy of LLM Safety Solutions : The Palit Benchmark Dataset</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly integrated into critical systems in industries like healthcare and finance. Users can often submit queries to LLM-enabled chatbots, some of which can enrich responses with information retrieved from internal databases storing sensitive data. This gives rise to a range of attacks in which a user submits a malicious query and the LLM-system outputs a response that creates harm to the owner, such as leaking internal data or creating legal liability by harming a third-party. While security tools are being developed to counter these threats, there is little formal evaluation of their effectiveness and usability. This study addresses this gap by conducting a thorough comparative analysis of LLM security tools. We identified 13 solutions (9 closed-source, 4 open-source), but only 7 were evaluated due to a lack of participation by proprietary model owners.To evaluate, we built a benchmark dataset of malicious prompts, and evaluate these tools performance against a baseline LLM model (ChatGPT-3.5-Turbo). Our results show that the baseline model has too many false positives to be used for this task. Lakera Guard and ProtectAI LLM Guard emerged as the best overall tools showcasing the tradeoff between usability and performance. The study concluded with recommendations for greater transparency among closed source providers, improved context-aware detections, enhanced open-source engagement, increased user awareness, and the adoption of more representative performance metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13026v1">Step-wise Adaptive Integration of Supervised Fine-tuning and Reinforcement Learning for Task-Specific LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel at mathematical reasoning and logical problem-solving. The current popular training paradigms primarily use supervised fine-tuning (SFT) and reinforcement learning (RL) to enhance the models' reasoning abilities. However, when using SFT or RL alone, there are respective challenges: SFT may suffer from overfitting, while RL is prone to mode collapse. The state-of-the-art methods have proposed hybrid training schemes. However, static switching faces challenges such as poor generalization across different tasks and high dependence on data quality. In response to these challenges, inspired by the curriculum learning-quiz mechanism in human reasoning cultivation, We propose SASR, a step-wise adaptive hybrid training framework that theoretically unifies SFT and RL and dynamically balances the two throughout optimization. SASR uses SFT for initial warm-up to establish basic reasoning skills, and then uses an adaptive dynamic adjustment algorithm based on gradient norm and divergence relative to the original distribution to seamlessly integrate SFT with the online RL method GRPO. By monitoring the training status of LLMs and adjusting the training process in sequence, SASR ensures a smooth transition between training schemes, maintaining core reasoning abilities while exploring different paths. Experimental results demonstrate that SASR outperforms SFT, RL, and static hybrid training methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13004v1">EffiBench-X: A Multi-Language Benchmark for Measuring Efficiency of LLM-Generated Code</a></div>
    <div class="paper-meta">
      📅 2025-05-19
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Existing code generation benchmarks primarily evaluate functional correctness, with limited focus on code efficiency and often restricted to a single language like Python. To address this gap, we introduce EffiBench-X, the first multi-language benchmark designed to measure the efficiency of LLM-generated code. EffiBench-X supports Python, C++, Java, JavaScript, Ruby, and Golang. It comprises competitive programming tasks with human-expert solutions as efficiency baselines. Evaluating state-of-the-art LLMs on EffiBench-X reveals that while models generate functionally correct code, they consistently underperform human experts in efficiency. Even the most efficient LLM-generated solutions (Qwen3-32B) achieve only around \textbf{62\%} of human efficiency on average, with significant language-specific variations. LLMs show better efficiency in Python, Ruby, and JavaScript than in Java, C++, and Golang. For instance, DeepSeek-R1's Python code is significantly more efficient than its Java code. These results highlight the critical need for research into LLM optimization techniques to improve code efficiency across diverse languages. The dataset and evaluation infrastructure are submitted and available at https://github.com/EffiBench/EffiBench-X.git and https://huggingface.co/datasets/EffiBench/effibench-x.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12991v1">Personalized Fine-Tuning with Controllable Synthetic Speech from LLM-Generated Transcripts for Dysarthric Speech Recognition</a></div>
    <div class="paper-meta">
      📅 2025-05-19
      | 💬 Accepted at Interspeech 2025
    </div>
    <details class="paper-abstract">
      In this work, we present our submission to the Speech Accessibility Project challenge for dysarthric speech recognition. We integrate parameter-efficient fine-tuning with latent audio representations to improve an encoder-decoder ASR system. Synthetic training data is generated by fine-tuning Parler-TTS to mimic dysarthric speech, using LLM-generated prompts for corpus-consistent target transcripts. Personalization with x-vectors consistently reduces word error rates (WERs) over non-personalized fine-tuning. AdaLoRA adapters outperform full fine-tuning and standard low-rank adaptation, achieving relative WER reductions of ~23% and ~22%, respectively. Further improvements (~5% WER reduction) come from incorporating wav2vec 2.0-based audio representations. Training with synthetic dysarthric speech yields up to ~7% relative WER improvement over personalized fine-tuning alone.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12981v1">From Assistants to Adversaries: Exploring the Security Risks of Mobile LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      The growing adoption of large language models (LLMs) has led to a new paradigm in mobile computing--LLM-powered mobile AI agents--capable of decomposing and automating complex tasks directly on smartphones. However, the security implications of these agents remain largely unexplored. In this paper, we present the first comprehensive security analysis of mobile LLM agents, encompassing three representative categories: System-level AI Agents developed by original equipment manufacturers (e.g., YOYO Assistant), Third-party Universal Agents (e.g., Zhipu AI AutoGLM), and Emerging Agent Frameworks (e.g., Alibaba Mobile Agent). We begin by analyzing the general workflow of mobile agents and identifying security threats across three core capability dimensions: language-based reasoning, GUI-based interaction, and system-level execution. Our analysis reveals 11 distinct attack surfaces, all rooted in the unique capabilities and interaction patterns of mobile LLM agents, and spanning their entire operational lifecycle. To investigate these threats in practice, we introduce AgentScan, a semi-automated security analysis framework that systematically evaluates mobile LLM agents across all 11 attack scenarios. Applying AgentScan to nine widely deployed agents, we uncover a concerning trend: every agent is vulnerable to targeted attacks. In the most severe cases, agents exhibit vulnerabilities across eight distinct attack vectors. These attacks can cause behavioral deviations, privacy leakage, or even full execution hijacking. Based on these findings, we propose a set of defensive design principles and practical recommendations for building secure mobile LLM agents. Our disclosures have received positive feedback from two major device vendors. Overall, this work highlights the urgent need for standardized security practices in the fast-evolving landscape of LLM-driven mobile automation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12951v1">DGRO: Enhancing LLM Reasoning via Exploration-Exploitation Control and Reward Variance Management</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      Inference scaling further accelerates Large Language Models (LLMs) toward Artificial General Intelligence (AGI), with large-scale Reinforcement Learning (RL) to unleash long Chain-of-Thought reasoning. Most contemporary reasoning approaches usually rely on handcrafted rule-based reward functions. However, the tarde-offs of exploration and exploitation in RL algorithms involves multiple complex considerations, and the theoretical and empirical impacts of manually designed reward functions remain insufficiently explored. In this paper, we propose Decoupled Group Reward Optimization (DGRO), a general RL algorithm for LLM reasoning. On the one hand, DGRO decouples the traditional regularization coefficient into two independent hyperparameters: one scales the policy gradient term, and the other regulates the distance from the sampling policy. This decoupling not only enables precise control over balancing exploration and exploitation, but also can be seamlessly extended to Online Policy Mirror Descent (OPMD) algorithms in Kimi k1.5 and Direct Reward Optimization. On the other hand, we observe that reward variance significantly affects both convergence speed and final model performance. We conduct both theoretical analysis and extensive empirical validation to assess DGRO, including a detailed ablation study that investigates its performance and optimization dynamics. Experimental results show that DGRO achieves state-of-the-art performance on the Logic dataset with an average accuracy of 96.9\%, and demonstrates strong generalization across mathematical benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12938v1">Leveraging LLM Inconsistency to Boost Pass@k Performance</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) achieve impressive abilities in numerous domains, but exhibit inconsistent performance in response to minor input changes. Rather than view this as a drawback, in this paper we introduce a novel method for leveraging models' inconsistency to boost Pass@k performance. Specifically, we present a "Variator" agent that generates k variants of a given task and submits one candidate solution for each one. Our variant generation approach is applicable to a wide range of domains as it is task agnostic and compatible with free-form inputs. We demonstrate the efficacy of our agent theoretically using a probabilistic model of the inconsistency effect, and show empirically that it outperforms the baseline on the APPS dataset. Furthermore, we establish that inconsistency persists even in frontier reasoning models across coding and cybersecurity domains, suggesting our method is likely to remain relevant for future model generations.
    </details>
</div>
