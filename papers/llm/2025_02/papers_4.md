# llm - 2025_02

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
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15601v1">WorldCraft: Photo-Realistic 3D World Creation and Customization via LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-02-21
    </div>
    <details class="paper-abstract">
      Constructing photorealistic virtual worlds has applications across various fields, but it often requires the extensive labor of highly trained professionals to operate conventional 3D modeling software. To democratize this process, we introduce WorldCraft, a system where large language model (LLM) agents leverage procedural generation to create indoor and outdoor scenes populated with objects, allowing users to control individual object attributes and the scene layout using intuitive natural language commands. In our framework, a coordinator agent manages the overall process and works with two specialized LLM agents to complete the scene creation: ForgeIt, which integrates an ever-growing manual through auto-verification to enable precise customization of individual objects, and ArrangeIt, which formulates hierarchical optimization problems to achieve a layout that balances ergonomic and aesthetic considerations. Additionally, our pipeline incorporates a trajectory control agent, allowing users to animate the scene and operate the camera through natural language interactions. Our system is also compatible with off-the-shelf deep 3D generators to enrich scene assets. Through evaluations and comparisons with state-of-the-art methods, we demonstrate the versatility of WorldCraft, ranging from single-object customization to intricate, large-scale interior and exterior scene designs. This system empowers non-professionals to bring their creative visions to life.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.18328v2">Unveiling Scoring Processes: Dissecting the Differences between LLMs and Human Graders in Automatic Scoring</a></div>
    <div class="paper-meta">
      📅 2025-02-21
      | 💬 Accepted by Technology, Knowledge, and Learning (TKNL)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated strong potential in performing automatic scoring for constructed response assessments. While constructed responses graded by humans are usually based on given grading rubrics, the methods by which LLMs assign scores remain largely unclear. It is also uncertain how closely AI's scoring process mirrors that of humans or if it adheres to the same grading criteria. To address this gap, this paper uncovers the grading rubrics that LLMs used to score students' written responses to science tasks and their alignment with human scores. We also examine whether enhancing the alignments can improve scoring accuracy. Specifically, we prompt LLMs to generate analytic rubrics that they use to assign scores and study the alignment gap with human grading rubrics. Based on a series of experiments with various configurations of LLM settings, we reveal a notable alignment gap between human and LLM graders. While LLMs can adapt quickly to scoring tasks, they often resort to shortcuts, bypassing deeper logical reasoning expected in human grading. We found that incorporating high-quality analytical rubrics designed to reflect human grading logic can mitigate this gap and enhance LLMs' scoring accuracy. These results underscore the need for a nuanced approach when applying LLMs in science education and highlight the importance of aligning LLM outputs with human expectations to ensure efficient and accurate automatic scoring.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15576v1">Interpreting and Steering LLMs with Mutual Information-based Explanations on Sparse Autoencoders</a></div>
    <div class="paper-meta">
      📅 2025-02-21
      | 💬 Pre-print. 20 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel at handling human queries, but they can occasionally generate flawed or unexpected responses. Understanding their internal states is crucial for understanding their successes, diagnosing their failures, and refining their capabilities. Although sparse autoencoders (SAEs) have shown promise for interpreting LLM internal representations, limited research has explored how to better explain SAE features, i.e., understanding the semantic meaning of features learned by SAE. Our theoretical analysis reveals that existing explanation methods suffer from the frequency bias issue, where they emphasize linguistic patterns over semantic concepts, while the latter is more critical to steer LLM behaviors. To address this, we propose using a fixed vocabulary set for feature interpretations and designing a mutual information-based objective, aiming to better capture the semantic meaning behind these features. We further propose two runtime steering strategies that adjust the learned feature activations based on their corresponding explanations. Empirical results show that, compared to baselines, our method provides more discourse-level explanations and effectively steers LLM behaviors to defend against jailbreak attacks. These findings highlight the value of explanations for steering LLM behaviors in downstream applications. We will release our code and data once accepted.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.05673v4">Flow of Reasoning:Training LLMs for Divergent Problem Solving with Minimal Examples</a></div>
    <div class="paper-meta">
      📅 2025-02-21
    </div>
    <details class="paper-abstract">
      The ability to generate diverse solutions to a given problem is a hallmark of human creativity. This divergent reasoning is also crucial for machines, enhancing their robustness and enabling them to assist humans in many applications such as scientific discovery. However, existing approaches to multi-step reasoning with large language models (LLMs) have mostly focused only on reasoning accuracy, without further discovering more diverse valid solutions. For example, supervised fine-tuning can improve LLM reasoning quality, but requires extensive supervised data to capture the full range of possible solutions. Reward-maximization reinforcement learning aims to find limited highest-reward solutions while neglecting the solution diversity. To fill this gap, we propose Flow of Reasoning (FoR), an efficient diversity-seeking LLM finetuning method aimed at improving reasoning quality and diversity with minimal data. FoR formulates multi-step LLM reasoning as a Markovian flow on a DAG-structured reasoning graph. This formulation allows us to incorporate and adapt principled GFlowNet approaches, for finetuning LLMs to sample divergent paths with probabilities proportional to the (unnormalized) reward of target problems. Extensive experiments show that, with limited training examples (e.g., 15 examples), FoR enables the discovery of diverse, creative, high-quality solutions, greatly outperforming a wide range of existing inference and training methods across six challenging reasoning tasks, including BlocksWorld (embodied reasoning), Game24 (math puzzle solving), Rubik's Cube (spatial reasoning), 1D-ARC (abstraction reasoning), GSM8k (math reasoning), and ProntoQA (logical reasoning). Code is available at https://github.com/Yu-Fangxu/FoR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.21102v2">Exploring and Controlling Diversity in LLM-Agent Conversation</a></div>
    <div class="paper-meta">
      📅 2025-02-21
      | 💬 Accepted for the AAAI 2025 Workshop on Advancing LLM-Based Multi-Agent Collaboration (v1); updated version (v2)
    </div>
    <details class="paper-abstract">
      Controlling diversity in LLM-agent world simulations is essential for maintaining stability in structured tasks while enabling variation where creativity is needed. However, we observe that dialogue diversity declines significantly over long-term simulation. To investigate the role of prompt design in conversational diversity, we modularized the utterance generation prompt and found that reducing the given information leads to more diverse outputs. Based on this insight, we propose Adaptive Prompt Pruning (APP), a novel method that allows users to control diversity through a single parameter, lambda. APP dynamically prunes the utterance generation prompt based on their attention weights and is compatible with traditional diversity control techniques. We demonstrate that APP effectively controls output diversity through extensive experiments, and propose a method to balance the control trade-offs. Additionally, we provide an in-depth analysis to offer insights into optimizing diversity control in multi-agent simulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15526v1">Scaling Sparse and Dense Retrieval in Decoder-Only LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-21
    </div>
    <details class="paper-abstract">
      Scaling large language models (LLMs) has shown great potential for improving retrieval model performance; however, previous studies have mainly focused on dense retrieval trained with contrastive loss (CL), neglecting the scaling behavior of other retrieval paradigms and optimization techniques, such as sparse retrieval and knowledge distillation (KD). In this work, we conduct a systematic comparative study on how different retrieval paradigms (sparse vs. dense) and fine-tuning objectives (CL vs. KD vs. their combination) affect retrieval performance across different model scales. Using MSMARCO passages as the training dataset, decoder-only LLMs (Llama-3 series: 1B, 3B, 8B), and a fixed compute budget, we evaluate various training configurations on both in-domain (MSMARCO, TREC DL) and out-of-domain (BEIR) benchmarks. Our key findings reveal that: (1) Scaling behaviors emerge clearly only with CL, where larger models achieve significant performance gains, whereas KD-trained models show minimal improvement, performing similarly across the 1B, 3B, and 8B scales. (2) Sparse retrieval models consistently outperform dense retrieval across both in-domain (MSMARCO, TREC DL) and out-of-domain (BEIR) benchmarks, and they demonstrate greater robustness to imperfect supervised signals. (3) We successfully scale sparse retrieval models with the combination of CL and KD losses at 8B scale, achieving state-of-the-art (SOTA) results in all evaluation sets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15524v1">Towards Swift Serverless LLM Cold Starts with ParaServe</a></div>
    <div class="paper-meta">
      📅 2025-02-21
    </div>
    <details class="paper-abstract">
      With the surge in number of large language models (LLMs), the industry turns to serverless computing for LLM inference serving. However, serverless LLM serving suffers from significant cold start latency and service level objective (SLO) violations due to the substantial model size, which leads to prolonged model fetching time from remote storage. We present ParaServe, a serverless LLM serving system that minimizes cold start latency through the novel use of pipeline parallelism. Our insight is that by distributing model parameters across multiple GPU servers, we can utilize their aggregated network bandwidth to concurrently fetch different parts of the model. ParaServe adopts a two-level hierarchical design. At the cluster level, ParaServe determines the optimal degree of parallelism based on user SLOs and carefully places GPU workers across servers to reduce network interference. At the worker level, ParaServe overlaps model fetching, loading, and runtime initialization to further accelerate cold starts. Additionally, ParaServe introduces pipeline consolidation, which merges parallel groups back to individual workers to maintain optimal performance for warm requests. Our comprehensive evaluations under diverse settings demonstrate that ParaServe reduces the cold start latency by up to 4.7x and improves SLO attainment by up to 1.74x compared to baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15506v1">Construction and Evaluation of LLM-based agents for Semi-Autonomous penetration testing</a></div>
    <div class="paper-meta">
      📅 2025-02-21
      | 💬 7 pages, 4 tables and 1 figure
    </div>
    <details class="paper-abstract">
      With the emergence of high-performance large language models (LLMs) such as GPT, Claude, and Gemini, the autonomous and semi-autonomous execution of tasks has significantly advanced across various domains. However, in highly specialized fields such as cybersecurity, full autonomy remains a challenge. This difficulty primarily stems from the limitations of LLMs in reasoning capabilities and domain-specific knowledge. We propose a system that semi-autonomously executes complex cybersecurity workflows by employing multiple LLMs modules to formulate attack strategies, generate commands, and analyze results, thereby addressing the aforementioned challenges. In our experiments using Hack The Box virtual machines, we confirmed that our system can autonomously construct attack strategies, issue appropriate commands, and automate certain processes, thereby reducing the need for manual intervention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11242v2">LLMs and Childhood Safety: Identifying Risks and Proposing a Protection Framework for Safe Child-LLM Interaction</a></div>
    <div class="paper-meta">
      📅 2025-02-21
    </div>
    <details class="paper-abstract">
      This study examines the growing use of Large Language Models (LLMs) in child-centered applications, highlighting safety and ethical concerns such as bias, harmful content, and cultural insensitivity. Despite their potential to enhance learning, there is a lack of standardized frameworks to mitigate these risks. Through a systematic literature review, we identify key parental and empirical concerns, including toxicity and ethical breaches in AI outputs. Moreover, to address these issues, this paper proposes a protection framework for safe Child-LLM interaction, incorporating metrics for content safety, behavioral ethics, and cultural sensitivity. The framework provides practical tools for evaluating LLM safety, offering guidance for developers, policymakers, and educators to ensure responsible AI deployment for children.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11910v2">Adversarial Alignment for LLMs Requires Simpler, Reproducible, and More Measurable Objectives</a></div>
    <div class="paper-meta">
      📅 2025-02-21
    </div>
    <details class="paper-abstract">
      Misaligned research objectives have considerably hindered progress in adversarial robustness research over the past decade. For instance, an extensive focus on optimizing target metrics, while neglecting rigorous standardized evaluation, has led researchers to pursue ad-hoc heuristic defenses that were seemingly effective. Yet, most of these were exposed as flawed by subsequent evaluations, ultimately contributing little measurable progress to the field. In this position paper, we illustrate that current research on the robustness of large language models (LLMs) risks repeating past patterns with potentially worsened real-world implications. To address this, we argue that realigned objectives are necessary for meaningful progress in adversarial alignment. To this end, we build on established cybersecurity taxonomy to formally define differences between past and emerging threat models that apply to LLMs. Using this framework, we illustrate that progress requires disentangling adversarial alignment into addressable sub-problems and returning to core academic principles, such as measureability, reproducibility, and comparability. Although the field presents significant challenges, the fresh start on adversarial robustness offers the unique opportunity to build on past experience while avoiding previous mistakes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15427v1">Adversarial Prompt Evaluation: Systematic Benchmarking of Guardrails Against Prompt Input Attacks on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-21
      | 💬 NeurIPS 2024, Safe Generative AI Workshop
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become integrated into everyday applications, ensuring their robustness and security is increasingly critical. In particular, LLMs can be manipulated into unsafe behaviour by prompts known as jailbreaks. The variety of jailbreak styles is growing, necessitating the use of external defences known as guardrails. While many jailbreak defences have been proposed, not all defences are able to handle new out-of-distribution attacks due to the narrow segment of jailbreaks used to align them. Moreover, the lack of systematisation around defences has created significant gaps in their practical application. In this work, we perform systematic benchmarking across 15 different defences, considering a broad swathe of malicious and benign datasets. We find that there is significant performance variation depending on the style of jailbreak a defence is subject to. Additionally, we show that based on current datasets available for evaluation, simple baselines can display competitive out-of-distribution performance compared to many state-of-the-art defences. Code is available at https://github.com/IBM/Adversarial-Prompt-Evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15419v1">Beyond Translation: LLM-Based Data Generation for Multilingual Fact-Checking</a></div>
    <div class="paper-meta">
      📅 2025-02-21
      | 💬 15 pages, 1 figure, 18 tables
    </div>
    <details class="paper-abstract">
      Robust automatic fact-checking systems have the potential to combat online misinformation at scale. However, most existing research primarily focuses on English. In this paper, we introduce MultiSynFact, the first large-scale multilingual fact-checking dataset containing 2.2M claim-source pairs designed to support Spanish, German, English, and other low-resource languages. Our dataset generation pipeline leverages Large Language Models (LLMs), integrating external knowledge from Wikipedia and incorporating rigorous claim validation steps to ensure data quality. We evaluate the effectiveness of MultiSynFact across multiple models and experimental settings. Additionally, we open-source a user-friendly framework to facilitate further research in multilingual fact-checking and dataset generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15401v1">Problem-Solving Logic Guided Curriculum In-Context Learning for LLMs Complex Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-21
    </div>
    <details class="paper-abstract">
      In-context learning (ICL) can significantly enhance the complex reasoning capabilities of large language models (LLMs), with the key lying in the selection and ordering of demonstration examples. Previous methods typically relied on simple features to measure the relevance between examples. We argue that these features are not sufficient to reflect the intrinsic connections between examples. In this study, we propose a curriculum ICL strategy guided by problem-solving logic. We select demonstration examples by analyzing the problem-solving logic and order them based on curriculum learning. Specifically, we constructed a problem-solving logic instruction set based on the BREAK dataset and fine-tuned a language model to analyze the problem-solving logic of examples. Subsequently, we selected appropriate demonstration examples based on problem-solving logic and assessed their difficulty according to the number of problem-solving steps. In accordance with the principles of curriculum learning, we ordered the examples from easy to hard to serve as contextual prompts. Experimental results on multiple benchmarks indicate that our method outperforms previous ICL approaches in terms of performance and efficiency, effectively enhancing the complex reasoning capabilities of LLMs. Our project will be publicly available subsequently.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15395v1">Beyond Tools: Understanding How Heavy Users Integrate LLMs into Everyday Tasks and Decision-Making</a></div>
    <div class="paper-meta">
      📅 2025-02-21
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used for both everyday and specialized tasks. While HCI research focuses on domain-specific applications, little is known about how heavy users integrate LLMs into everyday decision-making. Through qualitative interviews with heavy LLM users (n=7) who employ these systems for both intuitive and analytical thinking tasks, our findings show that participants use LLMs for social validation, self-regulation, and interpersonal guidance, seeking to build self-confidence and optimize cognitive resources. These users viewed LLMs either as rational, consistent entities or average human decision-makers. Our findings suggest that heavy LLM users develop nuanced interaction patterns beyond simple delegation, highlighting the need to reconsider how we study LLM integration in decision-making processes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11187v2">TituLLMs: A Family of Bangla LLMs with Comprehensive Benchmarking</a></div>
    <div class="paper-meta">
      📅 2025-02-21
      | 💬 LLMs, Benchmarking, Large Language Models, Bangla
    </div>
    <details class="paper-abstract">
      In this paper, we present TituLLMs, the first large pretrained Bangla LLMs, available in 1b and 3b parameter sizes. Due to computational constraints during both training and inference, we focused on smaller models. To train TituLLMs, we collected a pretraining dataset of approximately ~37 billion tokens. We extended the Llama-3.2 tokenizer to incorporate language- and culture-specific knowledge, which also enables faster training and inference. There was a lack of benchmarking datasets to benchmark LLMs for Bangla. To address this gap, we developed five benchmarking datasets. We benchmarked various LLMs, including TituLLMs, and demonstrated that TituLLMs outperforms its initial multilingual versions. However, this is not always the case, highlighting the complexities of language adaptation. Our work lays the groundwork for adapting existing multilingual open models to other low-resource languages. To facilitate broader adoption and further research, we have made the TituLLMs models and benchmarking datasets publicly available (https://huggingface.co/collections/hishab/titulm-llama-family-6718d31fc1b83529276f490a).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15361v1">Evaluating Social Biases in LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-21
    </div>
    <details class="paper-abstract">
      In the recent development of AI reasoning, large language models (LLMs) are trained to automatically generate chain-of-thought reasoning steps, which have demonstrated compelling performance on math and coding tasks. However, when bias is mixed within the reasoning process to form strong logical arguments, it could cause even more harmful results and further induce hallucinations. In this paper, we have evaluated the 8B and 32B variants of DeepSeek-R1 against their instruction tuned counterparts on the BBQ dataset, and investigated the bias that is elicited out and being amplified through reasoning steps. To the best of our knowledge, this empirical study is the first to assess bias issues in LLM reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15335v1">Stepwise Informativeness Search for Improving LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-21
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Advances in Large Language Models (LLMs) have significantly improved multi-step reasoning through generating free-text rationales. However, recent studies show that LLMs tend to lose focus over the middle of long contexts. This raises concerns that as reasoning progresses, LLMs may overlook information in earlier steps when decoding subsequent steps, leading to generate unreliable and redundant rationales. To address this, we propose guiding LLMs to generate more accurate and concise step-by-step rationales by (1) proactively referencing information from underutilized prior steps, and (2) minimizing redundant information between new and existing steps. We introduce stepwise informativeness search, an inference-time tree search framework incorporating two selection heuristics: grounding-guided selection which prioritizes steps paying higher attention over underutilized steps; and novelty-guided selection which encourages steps with novel conclusions. During rationale generation, we use a self-grounding strategy that prompts LLMs to explicitly reference relevant prior steps to provide premises before deduction at each step. Experimental results on four reasoning datasets demonstrate that our approach improves reasoning accuracy by generating higher-quality rationales with reduced errors and redundancy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15304v1">SVDq: 1.25-bit and 410x Key Cache Compression for LLM Attention</a></div>
    <div class="paper-meta">
      📅 2025-02-21
    </div>
    <details class="paper-abstract">
      For the efficient inference of Large Language Models (LLMs), the effective compression of key-value (KV) cache is essential. Three main types of KV cache compression techniques, namely sparsity, channel compression, and quantization, have been identified. This study presents SVDq, a Singular Value Decomposition (SVD) - based mixed precision quantization method for K cache. Initially, K cache is transformed into latent channels using SVD basis representations. Since the values in latent channels decay rapidly and become negligible after only a few latent channels, our method then incorporates importance-aware quantization and compression for latent channels. This enables the effective allocation of higher precision to more significant channels. Theoretically, we prove that SVDq results in quantization errors (x0.1 or even lower) that are much lower than those of per-channel key quantization in the original space. Our findings based on RULER and LongBench benchmarks demonstrate that SVDq can achieve an equivalent key cache precision as low as 1.25-bit. When combined with key sparsity, it can reach a key compression ratio of up to 410x for attention computation, all while maintaining comparable model performance. Notably, our method is nearly lossless for LongBench datasets. This indicates that SVDq enables high-precision low-bit quantization, providing a more efficient solution for KV cache compression in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15294v1">Round Attention: A Novel Round-Level Attention Mechanism to Accelerate LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-02-21
    </div>
    <details class="paper-abstract">
      The increasing context window size in large language models (LLMs) has improved their ability to handle complex, long-text tasks. However, as the conversation rounds continue, it is required to store a large amount of KV cache in GPU memory, which significantly affects the efficiency and even availability of the model serving systems. This paper analyzes dialogue data from real users and discovers that the LLM inference manifests a watershed layer, after which the distribution of round-level attention shows notable similarity. We propose Round Attention, a novel round-level attention mechanism that only recalls and computes the KV cache of the most relevant rounds. The experiments show that our method saves 55\% memory usage without compromising model performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.00750v2">Mitigating Tail Narrowing in LLM Self-Improvement via Socratic-Guided Sampling</a></div>
    <div class="paper-meta">
      📅 2025-02-21
      | 💬 Accepted to NAACL 2025 Main Conference. Codes are publicly available at https://github.com/Yiwen-Ding/Guided-Self-Improvement
    </div>
    <details class="paper-abstract">
      Self-improvement methods enable large language models (LLMs) to generate solutions themselves and iteratively train on filtered, high-quality rationales. This process proves effective and reduces the reliance on human supervision in LLMs' reasoning, but the performance soon plateaus. We delve into the process and find that models tend to over-sample on easy queries and under-sample on queries they have yet to master. As iterations proceed, this imbalance in sampling is exacerbated, leading to a long-tail distribution where solutions to difficult queries almost diminish. This phenomenon limits the performance gain of self-improving models. A straightforward solution is brute-force sampling to balance the distribution, which significantly raises computational costs. In this paper, we introduce Guided Self-Improvement (GSI), a strategy aimed at improving the efficiency of sampling challenging heavy-tailed data. It leverages Socratic-style guidance signals to help LLM reasoning with complex queries, reducing the exploration effort and minimizing computational overhead. Experiments on four models across diverse mathematical tasks show that GSI strikes a balance between performance and efficiency, while also being effective on held-out tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15266v1">A Training-free LLM-based Approach to General Chinese Character Error Correction</a></div>
    <div class="paper-meta">
      📅 2025-02-21
      | 💬 25 pages, 12 figures
    </div>
    <details class="paper-abstract">
      Chinese spelling correction (CSC) is a crucial task that aims to correct character errors in Chinese text. While conventional CSC focuses on character substitution errors caused by mistyping, two other common types of character errors, missing and redundant characters, have received less attention. These errors are often excluded from CSC datasets during the annotation process or ignored during evaluation, even when they have been annotated. This issue limits the practicality of the CSC task. To address this issue, we introduce the task of General Chinese Character Error Correction (C2EC), which focuses on all three types of character errors. We construct a high-quality C2EC benchmark by combining and manually verifying data from CCTC and Lemon datasets. We extend the training-free prompt-free CSC method to C2EC by using Levenshtein distance for handling length changes and leveraging an additional prompt-based large language model (LLM) to improve performance. Experiments show that our method enables a 14B-parameter LLM to be on par with models nearly 50 times larger on both conventional CSC and C2EC tasks, without any fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.01830v3">PiCO: Peer Review in LLMs based on the Consistency Optimization</a></div>
    <div class="paper-meta">
      📅 2025-02-21
    </div>
    <details class="paper-abstract">
      Existing large language models (LLMs) evaluation methods typically focus on testing the performance on some closed-environment and domain-specific benchmarks with human annotations. In this paper, we explore a novel unsupervised evaluation direction, utilizing peer-review mechanisms to measure LLMs automatically. In this setting, both open-source and closed-source LLMs lie in the same environment, capable of answering unlabeled questions and evaluating each other, where each LLM's response score is jointly determined by other anonymous ones. To obtain the ability hierarchy among these models, we assign each LLM a learnable capability parameter to adjust the final ranking. We formalize it as a constrained optimization problem, intending to maximize the consistency of each LLM's capabilities and scores. The key assumption behind is that high-level LLM can evaluate others' answers more accurately than low-level ones, while higher-level LLM can also achieve higher response scores. Moreover, we propose three metrics called PEN, CIN, and LIS to evaluate the gap in aligning human rankings. We perform experiments on multiple datasets with these metrics, validating the effectiveness of the proposed approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15233v1">A General Pseudonymization Framework for Cloud-Based LLMs: Replacing Privacy Information in Controlled Text Generation</a></div>
    <div class="paper-meta">
      📅 2025-02-21
      | 💬 under review
    </div>
    <details class="paper-abstract">
      An increasing number of companies have begun providing services that leverage cloud-based large language models (LLMs), such as ChatGPT. However, this development raises substantial privacy concerns, as users' prompts are transmitted to and processed by the model providers. Among the various privacy protection methods for LLMs, those implemented during the pre-training and fine-tuning phrases fail to mitigate the privacy risks associated with the remote use of cloud-based LLMs by users. On the other hand, methods applied during the inference phrase are primarily effective in scenarios where the LLM's inference does not rely on privacy-sensitive information. In this paper, we outline the process of remote user interaction with LLMs and, for the first time, propose a detailed definition of a general pseudonymization framework applicable to cloud-based LLMs. The experimental results demonstrate that the proposed framework strikes an optimal balance between privacy protection and utility. The code for our method is available to the public at https://github.com/Mebymeby/Pseudonymization-Framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15229v1">User Experience with LLM-powered Conversational Recommendation Systems: A Case of Music Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-02-21
    </div>
    <details class="paper-abstract">
      The advancement of large language models (LLMs) now allows users to actively interact with conversational recommendation systems (CRS) and build their own personalized recommendation services tailored to their unique needs and goals. This experience offers users a significantly higher level of controllability compared to traditional RS, enabling an entirely new dimension of recommendation experiences. Building on this context, this study explored the unique experiences that LLM-powered CRS can provide compared to traditional RS. Through a three-week diary study with 12 participants using custom GPTs for music recommendations, we found that LLM-powered CRS can (1) help users clarify implicit needs, (2) support unique exploration, and (3) facilitate a deeper understanding of musical preferences. Based on these findings, we discuss the new design space enabled by LLM-powered CRS and highlight its potential to support more personalized, user-driven recommendation experiences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12566v2">Exploring the Impact of Personality Traits on LLM Bias and Toxicity</a></div>
    <div class="paper-meta">
      📅 2025-02-21
    </div>
    <details class="paper-abstract">
      With the different roles that AI is expected to play in human life, imbuing large language models (LLMs) with different personalities has attracted increasing research interests. While the "personification" enhances human experiences of interactivity and adaptability of LLMs, it gives rise to critical concerns about content safety, particularly regarding bias, sentiment and toxicity of LLM generation. This study explores how assigning different personality traits to LLMs affects the toxicity and biases of their outputs. Leveraging the widely accepted HEXACO personality framework developed in social psychology, we design experimentally sound prompts to test three LLMs' performance on three toxic and bias benchmarks. The findings demonstrate the sensitivity of all three models to HEXACO personality traits and, more importantly, a consistent variation in the biases, negative sentiment and toxicity of their output. In particular, adjusting the levels of several personality traits can effectively reduce bias and toxicity in model performance, similar to humans' correlations between personality traits and toxic behaviors. The findings highlight the additional need to examine content safety besides the efficiency of training or fine-tuning methods for LLM personification. They also suggest a potential for the adjustment of personalities to be a simple and low-cost method to conduct controlled text generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15226v1">Understand User Opinions of Large Language Models via LLM-Powered In-the-Moment User Experience Interviews</a></div>
    <div class="paper-meta">
      📅 2025-02-21
    </div>
    <details class="paper-abstract">
      Which large language model (LLM) is better? Every evaluation tells a story, but what do users really think about current LLMs? This paper presents CLUE, an LLM-powered interviewer that conducts in-the-moment user experience interviews, right after users interacted with LLMs, and automatically gathers insights about user opinions from massive interview logs. We conduct a study with thousands of users to understand user opinions on mainstream LLMs, recruiting users to first chat with a target LLM and then interviewed by CLUE. Our experiments demonstrate that CLUE captures interesting user opinions, for example, the bipolar views on the displayed reasoning process of DeepSeek-R1 and demands for information freshness and multi-modality. Our collected chat-and-interview logs will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15224v1">Auto-Bench: An Automated Benchmark for Scientific Discovery in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-21
      | 💬 13 pages
    </div>
    <details class="paper-abstract">
      Given the remarkable performance of Large Language Models (LLMs), an important question arises: Can LLMs conduct human-like scientific research and discover new knowledge, and act as an AI scientist? Scientific discovery is an iterative process that demands efficient knowledge updating and encoding. It involves understanding the environment, identifying new hypotheses, and reasoning about actions; however, no standardized benchmark specifically designed for scientific discovery exists for LLM agents. In response to these limitations, we introduce a novel benchmark, \textit{Auto-Bench}, that encompasses necessary aspects to evaluate LLMs for scientific discovery in both natural and social sciences. Our benchmark is based on the principles of causal graph discovery. It challenges models to uncover hidden structures and make optimal decisions, which includes generating valid justifications. By engaging interactively with an oracle, the models iteratively refine their understanding of underlying interactions, the chemistry and social interactions, through strategic interventions. We evaluate state-of-the-art LLMs, including GPT-4, Gemini, Qwen, Claude, and Llama, and observe a significant performance drop as the problem complexity increases, which suggests an important gap between machine and human intelligence that future development of LLMs need to take into consideration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15217v1">FormalSpecCpp: A Dataset of C++ Formal Specifications created using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-21
      | 💬 Accepted at the 2025 IEEE/ACM 22nd International Conference on Mining Software Repositories (MSR)
    </div>
    <details class="paper-abstract">
      FormalSpecCpp is a dataset designed to fill the gap in standardized benchmarks for verifying formal specifications in C++ programs. To the best of our knowledge, this is the first comprehensive collection of C++ programs with well-defined preconditions and postconditions. It provides a structured benchmark for evaluating specification inference tools and testing theaccuracy of generated specifications. Researchers and developers can use this dataset to benchmark specification inference tools,fine-tune Large Language Models (LLMs) for automated specification generation, and analyze the role of formal specifications in improving program verification and automated testing. By making this dataset publicly available, we aim to advance research in program verification, specification inference, and AI-assisted software development. The dataset and the code are available at https://github.com/MadhuNimmo/FormalSpecCpp.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.21255v3">AQUA: Network-Accelerated Memory Offloading for LLMs in Scale-Up GPU Domains</a></div>
    <div class="paper-meta">
      📅 2025-02-21
    </div>
    <details class="paper-abstract">
      Inference on large-language models (LLMs) is constrained by GPU memory capacity. A sudden increase in the number of inference requests to a cloud-hosted LLM can deplete GPU memory, leading to contention between multiple prompts for limited resources. Modern LLM serving engines deal with the challenge of limited GPU memory using admission control, which causes them to be unresponsive during request bursts. We propose that preemptive scheduling of prompts in time slices is essential for ensuring responsive LLM inference, especially under conditions of high load and limited GPU memory. However, preempting prompt inference incurs a high paging overhead, which reduces inference throughput. We present Aqua, a GPU memory management framework that significantly reduces the overhead of paging inference state, achieving both responsive and high-throughput inference even under bursty request patterns. We evaluate Aqua by hosting several state-of-the-art large generative ML models of different modalities on servers with 8 Nvidia H100 80G GPUs. Aqua improves the responsiveness of LLM inference by 20X compared to the state-of-the-art and improves LLM inference throughput over a single long prompt by 4X.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15172v1">BP-GPT: Auditory Neural Decoding Using fMRI-prompted LLM</a></div>
    <div class="paper-meta">
      📅 2025-02-21
      | 💬 arXiv admin note: substantial text overlap with arXiv:2405.07840
    </div>
    <details class="paper-abstract">
      Decoding language information from brain signals represents a vital research area within brain-computer interfaces, particularly in the context of deciphering the semantic information from the fMRI signal. Although existing work uses LLM to achieve this goal, their method does not use an end-to-end approach and avoids the LLM in the mapping of fMRI-to-text, leaving space for the exploration of the LLM in auditory decoding. In this paper, we introduce a novel method, the Brain Prompt GPT (BP-GPT). By using the brain representation that is extracted from the fMRI as a prompt, our method can utilize GPT-2 to decode fMRI signals into stimulus text. Further, we introduce the text prompt and align the fMRI prompt to it. By introducing the text prompt, our BP-GPT can extract a more robust brain prompt and promote the decoding of pre-trained LLM. We evaluate our BP-GPT on the open-source auditory semantic decoding dataset and achieve a significant improvement up to 4.61 on METEOR and 2.43 on BERTScore across all the subjects compared to the state-of-the-art method. The experimental results demonstrate that using brain representation as a prompt to further drive LLM for auditory neural decoding is feasible and effective. The code is available at https://github.com/1994cxy/BP-GPT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.07355v2">Think Together and Work Better: Combining Humans' and LLMs' Think-Aloud Outcomes for Effective Text Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      This study introduces \textbf{InteractEval}, a framework that integrates human expertise and Large Language Models (LLMs) using the Think-Aloud (TA) method to generate attributes for checklist-based text evaluation. By combining human flexibility and reasoning with LLM consistency, InteractEval outperforms traditional non-LLM-based and LLM-based baselines across four distinct dimensions, consisting of Coherence, Fluency, Consistency, and Relevance. The experiment also investigates the effectiveness of the TA method, showing that it promotes divergent thinking in both humans and LLMs, leading to the generation of a wider range of relevant attributes and enhance text evaluation performance. Comparative analysis reveals that humans excel at identifying attributes related to internal quality (Coherence and Fluency), but LLMs perform better at those attributes related to external alignment (Consistency and Relevance). Consequently, leveraging both humans and LLMs together produces the best evaluation outcomes. In other words, this study emphasizes the necessity of effectively combining humans and LLMs in an automated checklist-based text evaluation framework. The code is available at \textbf{\url{https://github.com/BBeeChu/InteractEval.git}}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14202v1">Do LLMs Consider Security? An Empirical Study on Responses to Programming Questions</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      The widespread adoption of conversational LLMs for software development has raised new security concerns regarding the safety of LLM-generated content. Our motivational study outlines ChatGPT's potential in volunteering context-specific information to the developers, promoting safe coding practices. Motivated by this finding, we conduct a study to evaluate the degree of security awareness exhibited by three prominent LLMs: Claude 3, GPT-4, and Llama 3. We prompt these LLMs with Stack Overflow questions that contain vulnerable code to evaluate whether they merely provide answers to the questions or if they also warn users about the insecure code, thereby demonstrating a degree of security awareness. Further, we assess whether LLM responses provide information about the causes, exploits, and the potential fixes of the vulnerability, to help raise users' awareness. Our findings show that all three models struggle to accurately detect and warn users about vulnerabilities, achieving a detection rate of only 12.6% to 40% across our datasets. We also observe that the LLMs tend to identify certain types of vulnerabilities related to sensitive information exposure and improper input neutralization much more frequently than other types, such as those involving external control of file names or paths. Furthermore, when LLMs do issue security warnings, they often provide more information on the causes, exploits, and fixes of vulnerabilities compared to Stack Overflow responses. Finally, we provide an in-depth discussion on the implications of our findings and present a CLI-based prompting tool that can be used to generate significantly more secure LLM responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14192v1">NLP-AKG: Few-Shot Construction of NLP Academic Knowledge Graph Based on LLM</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been widely applied in question answering over scientific research papers. To enhance the professionalism and accuracy of responses, many studies employ external knowledge augmentation. However, existing structures of external knowledge in scientific literature often focus solely on either paper entities or domain concepts, neglecting the intrinsic connections between papers through shared domain concepts. This results in less comprehensive and specific answers when addressing questions that combine papers and concepts. To address this, we propose a novel knowledge graph framework that captures deep conceptual relations between academic papers, constructing a relational network via intra-paper semantic elements and inter-paper citation relations. Using a few-shot knowledge graph construction method based on LLM, we develop NLP-AKG, an academic knowledge graph for the NLP domain, by extracting 620,353 entities and 2,271,584 relations from 60,826 papers in ACL Anthology. Based on this, we propose a 'sub-graph community summary' method and validate its effectiveness on three NLP scientific literature question answering datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14182v1">Multi-Faceted Studies on Data Poisoning can Advance LLM Development</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      The lifecycle of large language models (LLMs) is far more complex than that of traditional machine learning models, involving multiple training stages, diverse data sources, and varied inference methods. While prior research on data poisoning attacks has primarily focused on the safety vulnerabilities of LLMs, these attacks face significant challenges in practice. Secure data collection, rigorous data cleaning, and the multistage nature of LLM training make it difficult to inject poisoned data or reliably influence LLM behavior as intended. Given these challenges, this position paper proposes rethinking the role of data poisoning and argue that multi-faceted studies on data poisoning can advance LLM development. From a threat perspective, practical strategies for data poisoning attacks can help evaluate and address real safety risks to LLMs. From a trustworthiness perspective, data poisoning can be leveraged to build more robust LLMs by uncovering and mitigating hidden biases, harmful outputs, and hallucinations. Moreover, from a mechanism perspective, data poisoning can provide valuable insights into LLMs, particularly the interplay between data and model behavior, driving a deeper understanding of their underlying mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.10490v3">Learning Dynamics of LLM Finetuning</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Learning dynamics, which describes how the learning of specific training examples influences the model's predictions on other examples, gives us a powerful tool for understanding the behavior of deep learning systems. We study the learning dynamics of large language models during different types of finetuning, by analyzing the step-wise decomposition of how influence accumulates among different potential responses. Our framework allows a uniform interpretation of many interesting observations about the training of popular algorithms for both instruction tuning and preference tuning. In particular, we propose a hypothetical explanation of why specific types of hallucination are strengthened after finetuning, e.g., the model might use phrases or facts in the response for question B to answer question A, or the model might keep repeating similar simple phrases when generating responses. We also extend our framework and highlight a unique "squeezing effect" to explain a previously observed phenomenon in off-policy direct preference optimization (DPO), where running DPO for too long makes even the desired outputs less likely. This framework also provides insights into where the benefits of on-policy DPO and other variants come from. The analysis not only provides a novel perspective of understanding LLM's finetuning but also inspires a simple, effective method to improve alignment performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15097v1">LUME: LLM Unlearning with Multitask Evaluations</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Unlearning aims to remove copyrighted, sensitive, or private content from large language models (LLMs) without a full retraining. In this work, we develop a multi-task unlearning benchmark (LUME) which features three tasks: (1) unlearn synthetically generated creative short novels, (2) unlearn synthetic biographies with sensitive information, and (3) unlearn a collection of public biographies. We further release two fine-tuned LLMs of 1B and 7B parameter sizes as the target models. We conduct detailed evaluations of several recently proposed unlearning algorithms and present results on carefully crafted metrics to understand their behavior and limitations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.05315v2">Aligned at the Start: Conceptual Groupings in LLM Embeddings</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      This paper shifts focus to the often-overlooked input embeddings - the initial representations fed into transformer blocks. Using fuzzy graph, k-nearest neighbor (k-NN), and community detection, we analyze embeddings from diverse LLMs, finding significant categorical community structure aligned with predefined concepts and categories aligned with humans. We observe these groupings exhibit within-cluster organization (such as hierarchies, topological ordering, etc.), hypothesizing a fundamental structure that precedes contextual processing. To further investigate the conceptual nature of these groupings, we explore cross-model alignments across different LLM categories within their input embeddings, observing a medium to high degree of alignment. Furthermore, provide evidence that manipulating these groupings can play a functional role in mitigating ethnicity bias in LLM tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15090v1">Analyze the Neurons, not the Embeddings: Understanding When and Where LLM Representations Align with Humans</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Modern large language models (LLMs) achieve impressive performance on some tasks, while exhibiting distinctly non-human-like behaviors on others. This raises the question of how well the LLM's learned representations align with human representations. In this work, we introduce a novel approach to the study of representation alignment: we adopt a method from research on activation steering to identify neurons responsible for specific concepts (e.g., 'cat') and then analyze the corresponding activation patterns. Our findings reveal that LLM representations closely align with human representations inferred from behavioral data. Notably, this alignment surpasses that of word embeddings, which have been center stage in prior work on human and model alignment. Additionally, our approach enables a more granular view of how LLMs represent concepts. Specifically, we show that LLMs organize concepts in a way that reflects hierarchical relationships interpretable to humans (e.g., 'animal'-'dog').
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.04616v2">Can LLMs Improve Multimodal Fact-Checking by Asking Relevant Questions?</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Traditional fact-checking relies on humans to formulate relevant and targeted fact-checking questions (FCQs), search for evidence, and verify the factuality of claims. While Large Language Models (LLMs) have been commonly used to automate evidence retrieval and factuality verification at scale, their effectiveness for fact-checking is hindered by the absence of FCQ formulation. To bridge this gap, we seek to answer two research questions: (1) Can LLMs generate relevant FCQs? (2) Can LLM-generated FCQs improve multimodal fact-checking? We therefore introduce a framework LRQ-FACT for using LLMs to generate relevant FCQs to facilitate evidence retrieval and enhance fact-checking by probing information across multiple modalities. Through extensive experiments, we verify if LRQ-FACT can generate relevant FCQs of different types and if LRQ-FACT can consistently outperform baseline methods in multimodal fact-checking. Further analysis illustrates how each component in LRQ-FACT works toward improving the fact-checking performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09933v2">MIR-Bench: Benchmarking LLM's Long-Context Intelligence via Many-Shot In-Context Inductive Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-20
      | 💬 32 pages, 11 figures. v2 updated the project page and dataset link
    </div>
    <details class="paper-abstract">
      Inductive Reasoning (IR), the ability to summarize rules from examples and apply on new ones, has long been viewed as a primal ability for general intelligence and widely studied by cognitive science and AI researchers. Many benchmarks have been proposed to measure such ability for Large Language Models (LLMs); however, they focus on few-shot (usually $<$10) setting and lack evaluation for aggregating many pieces of information from long contexts. On the other hand, the ever-growing context length of LLMs have brought forth the novel paradigm of many-shot In-Context Learning (ICL), which addresses new tasks with hundreds to thousands of examples without expensive and inefficient fine-tuning. However, many-shot evaluations are mostly focused on classification (a very limited aspect of IR), and popular long-context LLM tasks such as Needle-In-A-Haystack (NIAH) seldom require complicated intelligence for integrating many pieces of information. To fix the issues from both worlds, we propose MIR-Bench, the first many-shot in-context inductive reasoning benchmark that asks LLM to induce output via input-output examples from underlying functions with diverse data format. Based on MIR-Bench, we study many novel problems for inductive reasoning and many-shot ICL, including robustness against erroneous shots and the effect of Chain-of-Thought (CoT), and acquired insightful findings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15018v1">Using tournaments to calculate AUROC for zero-shot classification with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Large language models perform surprisingly well on many zero-shot classification tasks, but are difficult to fairly compare to supervised classifiers due to the lack of a modifiable decision boundary. In this work, we propose and evaluate a method that converts binary classification tasks into pairwise comparison tasks, obtaining relative rankings from LLMs. Repeated pairwise comparisons can be used to score instances using the Elo rating system (used in chess and other competitions), inducing a confidence ordering over instances in a dataset. We evaluate scheduling algorithms for their ability to minimize comparisons, and show that our proposed algorithm leads to improved classification performance, while also providing more information than traditional zero-shot classification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15009v1">Contextualizing Search Queries In-Context Learning for Conversational Rewriting with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Conversational query rewriting is crucial for effective conversational search, yet traditional supervised methods require substantial labeled data, which is scarce in low-resource settings. This paper introduces Prompt-Guided In-Context Learning, a novel approach that leverages the in-context learning capabilities of Large Language Models (LLMs) for few-shot conversational query rewriting. Our method employs carefully designed prompts, incorporating task descriptions, input/output format specifications, and a small set of illustrative examples, to guide pre-trained LLMs to generate context-independent queries without explicit fine-tuning. Extensive experiments on benchmark datasets, TREC and Taskmaster-1, demonstrate that our approach significantly outperforms strong baselines, including supervised models and contrastive co-training methods, across various evaluation metrics such as BLEU, ROUGE-L, Success Rate, and MRR. Ablation studies confirm the importance of in-context examples, and human evaluations further validate the superior fluency, relevance, and context utilization of our generated rewrites. The results highlight the potential of prompt-guided in-context learning as an efficient and effective paradigm for low-resource conversational query rewriting, reducing the reliance on extensive labeled data and complex training procedures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14461v2">From Human Annotation to LLMs: SILICON Annotation Workflow for Management Research</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Unstructured text data annotation and analysis are fundamental to management research, often relying on human annotators through crowdsourcing platforms. While Large Language Models (LLMs) promise to provide a cost-effective and efficient alternative to human annotation, there lacks a systematic workflow that evaluate when LLMs are suitable or how to proceed with LLM-based text annotation in a reproducible manner. This paper addresses this methodological gap by introducing the ``SILICON" (Systematic Inference with LLMs for Information Classification and Notation) workflow. The workflow integrates established principles of human annotation with systematic prompt optimization and model selection, addressing challenges such as developing robust annotation guidelines, establishing high-quality human baselines, optimizing prompts, and ensuring reproducibility across LLMs. We validate the SILICON workflow through seven case studies covering common management research tasks. Our findings highlight the importance of validating annotation guideline agreement, the superiority of expert-developed human baselines over crowdsourced ones, the iterative nature of prompt optimization, and the necessity of testing multiple LLMs. We also find that LLMs agree well with expert annotations in most cases but show low agreement in more complex multi-label classification tasks. Notably, we propose a regression-based methodology to empirically compare LLM outputs across prompts and models. Our workflow advances management research by establishing rigorous, transparent, and reproducible processes for LLM-based annotation. We provide practical guidance for researchers to effectively navigate the evolving landscape of generative AI tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.03884v3">AlphaPO -- Reward shape matters for LLM alignment</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Human Feedback (RLHF) and its variants have made huge strides toward the effective alignment of large language models (LLMs) to follow instructions and reflect human values. More recently, Direct Alignment Algorithms (DAAs) have emerged in which the reward modeling stage of RLHF is skipped by characterizing the reward directly as a function of the policy being learned. Some popular examples of DAAs include Direct Preference Optimization (DPO) and Simple Preference Optimization (SimPO). These methods often suffer from likelihood displacement, a phenomenon by which the probabilities of preferred responses are often reduced undesirably. In this paper, we argue that, for DAAs the reward (function) shape matters. We introduce \textbf{AlphaPO}, a new DAA method that leverages an $\alpha$-parameter to help change the shape of the reward function beyond the standard log reward. AlphaPO helps maintain fine-grained control over likelihood displacement and over-optimization. Compared to SimPO, one of the best performing DAAs, AlphaPO leads to about 7\% to 10\% relative improvement in alignment performance for the instruct versions of Mistral-7B and Llama3-8B while achieving 15\% to 50\% relative improvement over DPO on the same models. The analysis and results presented highlight the importance of the reward shape, and how one can systematically change it to affect training dynamics, as well as improve alignment performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15835v1">Pragmatic Reasoning improves LLM Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive potential in translating natural language (NL) instructions into program code. However, user instructions often contain inherent ambiguities, making it challenging for LLMs to generate code that accurately reflects the user's true intent. To address this challenge, researchers have proposed to produce multiple candidates of the program code and then rerank them to identify the best solution. In this paper, we propose CodeRSA, a novel code candidate reranking mechanism built upon the Rational Speech Act (RSA) framework, designed to guide LLMs toward more comprehensive pragmatic reasoning about user intent. We evaluate CodeRSA using one of the latest LLMs on a popular code generation dataset. Our experiment results show that CodeRSA consistently outperforms common baselines, surpasses the state-of-the-art approach in most cases, and demonstrates robust overall performance. These findings underscore the effectiveness of integrating pragmatic reasoning into code candidate reranking, offering a promising direction for enhancing code generation quality in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14866v1">LServe: Efficient Long-sequence LLM Serving with Unified Sparse Attention</a></div>
    <div class="paper-meta">
      📅 2025-02-20
      | 💬 Accepted by MLSys 2025. Code available at: https://github.com/mit-han-lab/omniserve
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable potential in processing long sequences, yet efficiently serving these long-context models remains challenging due to the quadratic computational complexity of attention in the prefilling stage and the large memory footprint of the KV cache in the decoding stage. To address these issues, we introduce LServe, an efficient system that accelerates long-sequence LLM serving via hybrid sparse attention. This method unifies different hardware-friendly, structured sparsity patterns for both prefilling and decoding attention into a single framework, where computations on less important tokens are skipped block-wise. LServe demonstrates the compatibility of static and dynamic sparsity in long-context LLM attention. This design enables multiplicative speedups by combining these optimizations. Specifically, we convert half of the attention heads to nearly free streaming heads in both the prefilling and decoding stages. Additionally, we find that only a constant number of KV pages is required to preserve long-context capabilities, irrespective of context length. We then design a hierarchical KV page selection policy that dynamically prunes KV pages based on query-centric similarity. On average, LServe accelerates LLM prefilling by up to 2.9x and decoding by 1.3-2.1x over vLLM, maintaining long-context accuracy. Code is released at https://github.com/mit-han-lab/omniserve.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14860v1">Aligning LLMs to Ask Good Questions A Case Study in Clinical Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-20
      | 💬 22 pages, 8 figures, 8 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often fail to ask effective questions under uncertainty, making them unreliable in domains where proactive information-gathering is essential for decisionmaking. We present ALFA, a framework that improves LLM question-asking by (i) decomposing the notion of a "good" question into a set of theory-grounded attributes (e.g., clarity, relevance), (ii) controllably synthesizing attribute-specific question variations, and (iii) aligning models via preference-based optimization to explicitly learn to ask better questions along these fine-grained attributes. Focusing on clinical reasoning as a case study, we introduce the MediQ-AskDocs dataset, composed of 17k real-world clinical interactions augmented with 80k attribute-specific preference pairs of follow-up questions, as well as a novel expert-annotated interactive healthcare QA task to evaluate question-asking abilities. Models aligned with ALFA reduce diagnostic errors by 56.6% on MediQ-AskDocs compared to SOTA instruction-tuned LLMs, with a question-level win-rate of 64.4% and strong generalizability. Our findings suggest that explicitly guiding question-asking with structured, fine-grained attributes offers a scalable path to improve LLMs, especially in expert application domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14847v1">Red-Teaming LLM Multi-Agent Systems via Communication Attacks</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Large Language Model-based Multi-Agent Systems (LLM-MAS) have revolutionized complex problem-solving capability by enabling sophisticated agent collaboration through message-based communications. While the communication framework is crucial for agent coordination, it also introduces a critical yet unexplored security vulnerability. In this work, we introduce Agent-in-the-Middle (AiTM), a novel attack that exploits the fundamental communication mechanisms in LLM-MAS by intercepting and manipulating inter-agent messages. Unlike existing attacks that compromise individual agents, AiTM demonstrates how an adversary can compromise entire multi-agent systems by only manipulating the messages passing between agents. To enable the attack under the challenges of limited control and role-restricted communication format, we develop an LLM-powered adversarial agent with a reflection mechanism that generates contextually-aware malicious instructions. Our comprehensive evaluation across various frameworks, communication structures, and real-world applications demonstrates that LLM-MAS is vulnerable to communication-based attacks, highlighting the need for robust security measures in multi-agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14837v1">Towards Economical Inference: Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-20
      | 💬 16 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Multi-head Latent Attention (MLA) is an innovative architecture proposed by DeepSeek, designed to ensure efficient and economical inference by significantly compressing the Key-Value (KV) cache into a latent vector. Compared to MLA, standard LLMs employing Multi-Head Attention (MHA) and its variants such as Grouped-Query Attention (GQA) exhibit significant cost disadvantages. Enabling well-trained LLMs (e.g., Llama) to rapidly adapt to MLA without pre-training from scratch is both meaningful and challenging. This paper proposes the first data-efficient fine-tuning method for transitioning from MHA to MLA (MHA2MLA), which includes two key components: for partial-RoPE, we remove RoPE from dimensions of queries and keys that contribute less to the attention scores, for low-rank approximation, we introduce joint SVD approximations based on the pre-trained parameters of keys and values. These carefully designed strategies enable MHA2MLA to recover performance using only a small fraction (0.3% to 0.6%) of the data, significantly reducing inference costs while seamlessly integrating with compression techniques such as KV cache quantization. For example, the KV cache size of Llama2-7B is reduced by 92.19%, with only a 0.5% drop in LongBench performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07752v2">Towards Efficient Optimizer Design for LLM via Structured Fisher Approximation with a Low-Rank Extension</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Designing efficient optimizers for large language models (LLMs) with low-memory requirements and fast convergence is an important and challenging problem. This paper makes a step towards the systematic design of such optimizers through the lens of structured Fisher information matrix (FIM) approximation. We show that many state-of-the-art efficient optimizers can be viewed as solutions to FIM approximation (under the Frobenius norm) with specific structural assumptions. Building on these insights, we propose two design recommendations of practical efficient optimizers for LLMs, involving the careful selection of structural assumptions to balance generality and efficiency, and enhancing memory efficiency of optimizers with general structures through a novel low-rank extension framework. We demonstrate how to use each design approach by deriving new memory-efficient optimizers: Row and Column Scaled SGD (RACS) and Adaptive low-dimensional subspace estimation (Alice). Experiments on LLaMA pre-training (up to 1B parameters) validate the effectiveness, showing faster and better convergence than existing memory-efficient baselines and Adam with little memory overhead. Notably, Alice achieves better than 2x faster convergence over Adam, while RACS delivers strong performance on the 1B model with SGD-like memory.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14830v1">Middle-Layer Representation Alignment for Cross-Lingual Transfer in Fine-Tuned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      While large language models demonstrate remarkable capabilities at task-specific applications through fine-tuning, extending these benefits across diverse languages is essential for broad accessibility. However, effective cross-lingual transfer is hindered by LLM performance gaps across languages and the scarcity of fine-tuning data in many languages. Through analysis of LLM internal representations from over 1,000+ language pairs, we discover that middle layers exhibit the strongest potential for cross-lingual alignment. Building on this finding, we propose a middle-layer alignment objective integrated into task-specific training. Our experiments on slot filling, machine translation, and structured text generation show consistent improvements in cross-lingual transfer, especially to lower-resource languages. The method is robust to the choice of alignment languages and generalizes to languages unseen during alignment. Furthermore, we show that separately trained alignment modules can be merged with existing task-specific modules, improving cross-lingual capabilities without full re-training. Our code is publicly available (https://github.com/dannigt/mid-align).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14828v1">Fundamental Limitations in Defending LLM Finetuning APIs</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      LLM developers have imposed technical interventions to prevent fine-tuning misuse attacks, attacks where adversaries evade safeguards by fine-tuning the model using a public API. Previous work has established several successful attacks against specific fine-tuning API defences. In this work, we show that defences of fine-tuning APIs that seek to detect individual harmful training or inference samples ('pointwise' detection) are fundamentally limited in their ability to prevent fine-tuning attacks. We construct 'pointwise-undetectable' attacks that repurpose entropy in benign model outputs (e.g. semantic or syntactic variations) to covertly transmit dangerous knowledge. Our attacks are composed solely of unsuspicious benign samples that can be collected from the model before fine-tuning, meaning training and inference samples are all individually benign and low-perplexity. We test our attacks against the OpenAI fine-tuning API, finding they succeed in eliciting answers to harmful multiple-choice questions, and that they evade an enhanced monitoring system we design that successfully detects other fine-tuning attacks. We encourage the community to develop defences that tackle the fundamental limitations we uncover in pointwise fine-tuning API defences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14748v1">Large Language Models Struggle to Describe the Haystack without Human Help: Human-in-the-loop Evaluation of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-20
      | 💬 21 Pages. LLM for Data Exploration and content analysis
    </div>
    <details class="paper-abstract">
      A common use of NLP is to facilitate the understanding of large document collections, with a shift from using traditional topic models to Large Language Models. Yet the effectiveness of using LLM for large corpus understanding in real-world applications remains under-explored. This study measures the knowledge users acquire with unsupervised, supervised LLM-based exploratory approaches or traditional topic models on two datasets. While LLM-based methods generate more human-readable topics and show higher average win probabilities than traditional models for data exploration, they produce overly generic topics for domain-specific datasets that do not easily allow users to learn much about the documents. Adding human supervision to the LLM generation process improves data exploration by mitigating hallucination and over-genericity but requires greater human effort. In contrast, traditional. models like Latent Dirichlet Allocation (LDA) remain effective for exploration but are less user-friendly. We show that LLMs struggle to describe the haystack of large corpora without human help, particularly domain-specific data, and face scaling and hallucination limitations due to context length constraints. Dataset available at https://huggingface. co/datasets/zli12321/Bills.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14739v1">SuperGPQA: Scaling LLM Evaluation across 285 Graduate Disciplines</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable proficiency in mainstream academic disciplines such as mathematics, physics, and computer science. However, human knowledge encompasses over 200 specialized disciplines, far exceeding the scope of existing benchmarks. The capabilities of LLMs in many of these specialized fields-particularly in light industry, agriculture, and service-oriented disciplines-remain inadequately evaluated. To address this gap, we present SuperGPQA, a comprehensive benchmark that evaluates graduate-level knowledge and reasoning capabilities across 285 disciplines. Our benchmark employs a novel Human-LLM collaborative filtering mechanism to eliminate trivial or ambiguous questions through iterative refinement based on both LLM responses and expert feedback. Our experimental results reveal significant room for improvement in the performance of current state-of-the-art LLMs across diverse knowledge domains (e.g., the reasoning-focused model DeepSeek-R1 achieved the highest accuracy of 61.82% on SuperGPQA), highlighting the considerable gap between current model capabilities and artificial general intelligence. Additionally, we present comprehensive insights from our management of a large-scale annotation process, involving over 80 expert annotators and an interactive Human-LLM collaborative system, offering valuable methodological guidance for future research initiatives of comparable scope.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.08469v6">LLM4TS: Aligning Pre-Trained LLMs as Data-Efficient Time-Series Forecasters</a></div>
    <div class="paper-meta">
      📅 2025-02-20
      | 💬 Accepted for publication in ACM Transactions on Intelligent Systems and Technology (TIST) 2025. The final published version will be available at https://doi.org/10.1145/3719207
    </div>
    <details class="paper-abstract">
      Multivariate time-series forecasting is vital in various domains, e.g., economic planning and weather prediction. Deep train-from-scratch models have exhibited effective performance yet require large amounts of data, which limits real-world applicability. Recently, researchers have leveraged the representation learning transferability of pre-trained Large Language Models (LLMs) to handle limited non-linguistic datasets effectively. However, incorporating LLMs with time-series data presents challenges of limited adaptation due to different compositions between time-series and linguistic data, and the inability to process multi-scale temporal information. To tackle these challenges, we propose LLM4TS, a framework for time-series forecasting with pre-trained LLMs. LLM4TS consists of a two-stage fine-tuning strategy: the time-series alignment stage to align LLMs with the nuances of time-series data, and the forecasting fine-tuning stage for downstream time-series forecasting tasks. Furthermore, our framework features a novel two-level aggregation method that integrates multi-scale temporal data within pre-trained LLMs, enhancing their ability to interpret time-specific information. In experiments across 7 time-series forecasting datasets, LLM4TS is superior to existing state-of-the-art methods compared with trained-from-scratch models in full-shot scenarios, and also achieves the highest rank in few-shot scenarios. In addition, evaluations compared with different unsupervised representation learning approaches highlight LLM4TS's effectiveness with representation learning in forecasting tasks. Ablation studies further validate each component's contribution to LLM4TS and underscore the essential role of utilizing LLM's pre-trained weights for optimal performance. The code is available at https://github.com/blacksnail789521/LLM4TS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14678v1">How to Get Your LLM to Generate Challenging Problems for Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      The pace of evolution of Large Language Models (LLMs) necessitates new approaches for rigorous and comprehensive evaluation. Traditional human annotation is increasingly impracticable due to the complexities and costs involved in generating high-quality, challenging problems. In this work, we introduce CHASE, a unified framework to synthetically generate challenging problems using LLMs without human involvement. For a given task, our approach builds a hard problem in a bottom-up manner from simpler components. Moreover, our framework decomposes the generation process into independently verifiable sub-tasks, thereby ensuring a high level of quality and correctness. We implement CHASE to create evaluation benchmarks across three diverse domains: (1) document-based question answering, (2) repository-level code completion, and (3) math reasoning. The performance of state-of-the-art LLMs on these synthetic benchmarks lies in the range of 40-60% accuracy, thereby demonstrating the effectiveness of our framework at generating challenging problems. We publicly release our benchmarks and code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14662v1">InstructAgent: Building User Controllable Recommender via LLM Agent</a></div>
    <div class="paper-meta">
      📅 2025-02-20
      | 💬 WWW2025@HCRS
    </div>
    <details class="paper-abstract">
      Traditional recommender systems usually take the user-platform paradigm, where users are directly exposed under the control of the platform's recommendation algorithms. However, the defect of recommendation algorithms may put users in very vulnerable positions under this paradigm. First, many sophisticated models are often designed with commercial objectives in mind, focusing on the platform's benefits, which may hinder their ability to protect and capture users' true interests. Second, these models are typically optimized using data from all users, which may overlook individual user's preferences. Due to these shortcomings, users may experience several disadvantages under the traditional user-platform direct exposure paradigm, such as lack of control over the recommender system, potential manipulation by the platform, echo chamber effects, or lack of personalization for less active users due to the dominance of active users during collaborative learning. Therefore, there is an urgent need to develop a new paradigm to protect user interests and alleviate these issues. Recently, some researchers have introduced LLM agents to simulate user behaviors, these approaches primarily aim to optimize platform-side performance, leaving core issues in recommender systems unresolved. To address these limitations, we propose a new user-agent-platform paradigm, where agent serves as the protective shield between user and recommender system that enables indirect exposure. To this end, we first construct four recommendation datasets, denoted as $\dataset$, along with user instructions for each record.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14660v1">Beyond the Surface: Uncovering Implicit Locations with LLMs for Personalized Local News</a></div>
    <div class="paper-meta">
      📅 2025-02-20
      | 💬 10 pages, 2 figures, submitted to kdd
    </div>
    <details class="paper-abstract">
      News recommendation systems personalize homepage content to boost engagement, but factors like content type, editorial stance, and geographic focus impact recommendations. Local newspapers balance coverage across regions, yet identifying local articles is challenging due to implicit location cues like slang or landmarks. Traditional methods, such as Named Entity Recognition (NER) and Knowledge Graphs, infer locations, but Large Language Models (LLMs) offer new possibilities while raising concerns about accuracy and explainability. This paper explores LLMs for local article classification in Taboola's "Homepage For You" system, comparing them to traditional techniques. Key findings: (1) Knowledge Graphs enhance NER models' ability to detect implicit locations, (2) LLMs outperform traditional methods, and (3) LLMs can effectively identify local content without requiring Knowledge Graph integration. Offline evaluations showed LLMs excel at implicit location classification, while online A/B tests showed a significant increased in local views. A scalable pipeline integrating LLM-based location classification boosted local article distribution by 27%, preserving newspapers' brand identity and enhancing homepage personalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14645v1">Edit Once, Update Everywhere: A Simple Framework for Cross-Lingual Knowledge Synchronization in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Knowledge editing allows for efficient adaptation of large language models (LLMs) to new information or corrections without requiring full retraining. However, prior methods typically focus on either single-language editing or basic multilingual editing, failing to achieve true cross-linguistic knowledge synchronization. To address this, we present a simple and practical state-of-the-art (SOTA) recipe Cross-Lingual Knowledge Democracy Edit (X-KDE), designed to propagate knowledge from a dominant language to other languages effectively. Our X-KDE comprises two stages: (i) Cross-lingual Edition Instruction Tuning (XE-IT), which fine-tunes the model on a curated parallel dataset to modify in-scope knowledge while preserving unrelated information, and (ii) Target-language Preference Optimization (TL-PO), which applies advanced optimization techniques to ensure consistency across languages, fostering the transfer of updates. Additionally, we contribute a high-quality, cross-lingual dataset, specifically designed to enhance knowledge transfer across languages. Extensive experiments on the Bi-ZsRE and MzsRE benchmarks show that X-KDE significantly enhances cross-lingual performance, achieving an average improvement of +8.19%, while maintaining high accuracy in monolingual settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14642v1">How Far are LLMs from Being Our Digital Twins? A Benchmark for Persona-Based Behavior Chain Simulation</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Recently, LLMs have garnered increasing attention across academic disciplines for their potential as human digital twins, virtual proxies designed to replicate individuals and autonomously perform tasks such as decision-making, problem-solving, and reasoning on their behalf. However, current evaluations of LLMs primarily emphasize dialogue simulation while overlooking human behavior simulation, which is crucial for digital twins. To address this gap, we introduce BehaviorChain, the first benchmark for evaluating LLMs' ability to simulate continuous human behavior. BehaviorChain comprises diverse, high-quality, persona-based behavior chains, totaling 15,846 distinct behaviors across 1,001 unique personas, each with detailed history and profile metadata. For evaluation, we integrate persona metadata into LLMs and employ them to iteratively infer contextually appropriate behaviors within dynamic scenarios provided by BehaviorChain. Comprehensive evaluation results demonstrated that even state-of-the-art models struggle with accurately simulating continuous human behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14634v1">CER: Confidence Enhanced Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Ensuring the reliability of Large Language Models (LLMs) in complex reasoning tasks remains a formidable challenge, particularly in scenarios that demand precise mathematical calculations and knowledge-intensive open-domain generation. In this work, we introduce an uncertainty-aware framework designed to enhance the accuracy of LLM responses by systematically incorporating model confidence at critical decision points. We propose an approach that encourages multi-step reasoning in LLMs and quantify the confidence of intermediate answers such as numerical results in mathematical reasoning and proper nouns in open-domain generation. Then, the overall confidence of each reasoning chain is evaluated based on confidence of these critical intermediate steps. Finally, we aggregate the answer of generated response paths in a way that reflects the reliability of each generated content (as opposed to self-consistency in which each generated chain contributes equally to majority voting). We conducted extensive experiments in five datasets, three mathematical datasets and two open-domain datasets, using four LLMs. The results consistently validate the effectiveness of our novel confidence aggregation method, leading to an accuracy improvement of up to 7.4% and 5.8% over baseline approaches in math and open-domain generation tasks, respectively. Code is publicly available at https://github.com/ Aquasar11/CER.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14628v1">PEARL: Towards Permutation-Resilient LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-20
      | 💬 ICLR 2025
    </div>
    <details class="paper-abstract">
      The in-context learning (ICL) capability of large language models (LLMs) enables them to perform challenging tasks using provided demonstrations. However, ICL is highly sensitive to the ordering of demonstrations, leading to instability in predictions. This paper shows that this vulnerability can be exploited to design a natural attack - difficult for model providers to detect - that achieves nearly 80% success rate on LLaMA-3 by simply permuting the demonstrations. Existing mitigation methods primarily rely on post-processing and fail to enhance the model's inherent robustness to input permutations, raising concerns about safety and reliability of LLMs. To address this issue, we propose Permutation-resilient learning (PEARL), a novel framework based on distributionally robust optimization (DRO), which optimizes model performance against the worst-case input permutation. Specifically, PEARL consists of a permutation-proposal network (P-Net) and the LLM. The P-Net generates the most challenging permutations by treating it as an optimal transport problem, which is solved using an entropy-constrained Sinkhorn algorithm. Through minimax optimization, the P-Net and the LLM iteratively optimize against each other, progressively improving the LLM's robustness. Experiments on synthetic pre-training and real-world instruction tuning tasks demonstrate that PEARL effectively mitigates permutation attacks and enhances performance. Notably, despite being trained on fewer shots and shorter contexts, PEARL achieves performance gains of up to 40% when scaled to many-shot and long-context scenarios, highlighting its efficiency and generalization capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.05806v2">CKnowEdit: A New Chinese Knowledge Editing Dataset for Linguistics, Facts, and Logic Error Correction in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-20
      | 💬 Ongoing work; project website is available at https://zjunlp.github.io/project/CKnowEdit code and dataset are available at https://github.com/zjunlp/EasyEdit
    </div>
    <details class="paper-abstract">
      Chinese, as a linguistic system rich in depth and complexity, is characterized by distinctive elements such as ancient poetry, proverbs, idioms, and other cultural constructs. However, current Large Language Models (LLMs) face limitations in these specialized domains, highlighting the need for the development of comprehensive datasets that can assess, continuously update, and progressively improve these culturally-grounded linguistic competencies through targeted training optimizations. To address this gap, we introduce CKnowEdit, the first-ever Chinese knowledge editing dataset designed to correct linguistic, factual, and logical errors in LLMs. We collect seven types of knowledge from a wide range of sources, including classical texts, idioms, and content from Baidu Tieba Ruozhiba, taking into account the unique polyphony, antithesis, and logical structures inherent in the Chinese language. By analyzing this dataset, we highlight the challenges current LLMs face in mastering Chinese. Furthermore, our evaluation of state-of-the-art knowledge editing techniques reveals opportunities to advance the correction of Chinese knowledge. Code and dataset are available at https://github.com/zjunlp/EasyEdit.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14617v1">Serving Models, Fast and Slow:Optimizing Heterogeneous LLM Inferencing Workloads at Scale</a></div>
    <div class="paper-meta">
      📅 2025-02-20
      | 💬 15 pages, 17 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) inference workloads handled by global cloud providers can include both latency-sensitive and insensitive tasks, creating a diverse range of Service Level Agreement (SLA) requirements. Managing these mixed workloads is challenging due to the complexity of the inference stack, which includes multiple LLMs, hardware configurations, and geographic distributions. Current optimization strategies often silo these tasks to ensure that SLAs are met for latency-sensitive tasks, but this leads to significant under-utilization of expensive GPU resources despite the availability of spot and on-demand Virtual Machine (VM) provisioning. We propose SAGESERVE, a comprehensive LLM serving framework that employs adaptive control knobs at varying time scales, ensuring SLA compliance while maximizing the utilization of valuable GPU resources. Short-term optimizations include efficient request routing to data center regions, while long-term strategies involve scaling GPU VMs out/in and redeploying models to existing VMs to align with traffic patterns. These strategies are formulated as an optimization problem for resource allocation and solved using Integer Linear Programming (ILP). We perform empirical and simulation studies based on production workload traces with over 8M requests using four open-source models deployed across three regions. SAGESERVE achieves up to 25% savings in GPU-hours while maintaining tail latency and satisfying all SLOs, and it reduces the scaling overhead compared to baselines by up to 80%, confirming the effectiveness of our proposal. In terms of dollar cost, this can save cloud providers up to $2M over the course of a month.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11844v2">BaxBench: Can LLMs Generate Correct and Secure Backends?</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      The automatic generation of programs has long been a fundamental challenge in computer science. Recent benchmarks have shown that large language models (LLMs) can effectively generate code at the function level, make code edits, and solve algorithmic coding tasks. However, to achieve full automation, LLMs should be able to generate production-quality, self-contained application modules. To evaluate the capabilities of LLMs in solving this challenge, we introduce BaxBench, a novel evaluation benchmark consisting of 392 tasks for the generation of backend applications. We focus on backends for three critical reasons: (i) they are practically relevant, building the core components of most modern web and cloud software, (ii) they are difficult to get right, requiring multiple functions and files to achieve the desired functionality, and (iii) they are security-critical, as they are exposed to untrusted third-parties, making secure solutions that prevent deployment-time attacks an imperative. BaxBench validates the functionality of the generated applications with comprehensive test cases, and assesses their security exposure by executing end-to-end exploits. Our experiments reveal key limitations of current LLMs in both functionality and security: (i) even the best model, OpenAI o1, achieves a mere 60% on code correctness; (ii) on average, we could successfully execute security exploits on more than half of the correct programs generated by each LLM; and (iii) in less popular backend frameworks, models further struggle to generate correct and secure applications. Progress on BaxBench signifies important steps towards autonomous and secure software development with LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01387v3">TeLL-Drive: Enhancing Autonomous Driving with Teacher LLM-Guided Deep Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Although Deep Reinforcement Learning (DRL) and Large Language Models (LLMs) each show promise in addressing decision-making challenges in autonomous driving, DRL often suffers from high sample complexity, while LLMs have difficulty ensuring real-time decision making. To address these limitations, we propose TeLL-Drive, a hybrid framework that integrates a Teacher LLM to guide an attention-based Student DRL policy. By incorporating risk metrics, historical scenario retrieval, and domain heuristics into context-rich prompts, the LLM produces high-level driving strategies through chain-of-thought reasoning. A self-attention mechanism then fuses these strategies with the DRL agent's exploration, accelerating policy convergence and boosting robustness across diverse driving conditions. The experimental results, evaluated across multiple traffic scenarios, show that TeLL-Drive outperforms existing baseline methods, including other LLM-based approaches, in terms of success rates, average returns, and real-time feasibility. Ablation studies underscore the importance of each model component, especially the synergy between the attention mechanism and LLM-driven guidance. Finally, we build a virtual-real fusion experimental platform to verify the real-time performance, robustness, and reliability of the algorithm running on real vehicles through vehicle-in-loop experiments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14563v1">Plan-over-Graph: Towards Parallelable LLM Agent Schedule</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated exceptional abilities in reasoning for task planning. However, challenges remain under-explored for parallel schedules. This paper introduces a novel paradigm, plan-over-graph, in which the model first decomposes a real-life textual task into executable subtasks and constructs an abstract task graph. The model then understands this task graph as input and generates a plan for parallel execution. To enhance the planning capability of complex, scalable graphs, we design an automated and controllable pipeline to generate synthetic graphs and propose a two-stage training scheme. Experimental results show that our plan-over-graph method significantly improves task performance on both API-based LLMs and trainable open-sourced LLMs. By normalizing complex tasks as graphs, our method naturally supports parallel execution, demonstrating global efficiency. The code and data are available at https://github.com/zsq259/Plan-over-Graph.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14561v1">Can LLMs Predict Citation Intent? An Experimental Analysis of In-context Learning and Fine-tuning on Open LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      This work investigates the ability of open Large Language Models (LLMs) to predict citation intent through in-context learning and fine-tuning. Unlike traditional approaches that rely on pre-trained models like SciBERT, which require extensive domain-specific pretraining and specialized architectures, we demonstrate that general-purpose LLMs can be adapted to this task with minimal task-specific data. We evaluate twelve model variations across five prominent open LLM families using zero, one, few, and many-shot prompting to assess performance across scenarios. Our experimental study identifies the top-performing model through extensive experimentation of in-context learning-related parameters, which we fine-tune to further enhance task performance. The results highlight the strengths and limitations of LLMs in recognizing citation intents, providing valuable insights for model selection and prompt engineering. Additionally, we make our end-to-end evaluation framework and models openly available for future use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14541v1">LLM-based User Profile Management for Recommender System</a></div>
    <div class="paper-meta">
      📅 2025-02-20
      | 💬 Submitted to ACL 2025
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) has opened new opportunities in recommender systems by enabling zero-shot recommendation without conventional training. Despite their potential, most existing works rely solely on users' purchase histories, leaving significant room for improvement by incorporating user-generated textual data, such as reviews and product descriptions. Addressing this gap, we propose PURE, a novel LLM-based recommendation framework that builds and maintains evolving user profiles by systematically extracting and summarizing key information from user reviews. PURE consists of three core components: a Review Extractor for identifying user preferences and key product features, a Profile Updater for refining and updating user profiles, and a Recommender for generating personalized recommendations using the most current profile. To evaluate PURE, we introduce a continuous sequential recommendation task that reflects real-world scenarios by adding reviews over time and updating predictions incrementally. Our experimental results on Amazon datasets demonstrate that PURE outperforms existing LLM-based methods, effectively leveraging long-term user information while managing token limitations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14507v1">Can LLMs Simulate L2-English Dialogue? An Information-Theoretic Analysis of L1-Dependent Biases</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      This study evaluates Large Language Models' (LLMs) ability to simulate non-native-like English use observed in human second language (L2) learners interfered with by their native first language (L1). In dialogue-based interviews, we prompt LLMs to mimic L2 English learners with specific L1s (e.g., Japanese, Thai, Urdu) across seven languages, comparing their outputs to real L2 learner data. Our analysis examines L1-driven linguistic biases, such as reference word usage and avoidance behaviors, using information-theoretic and distributional density measures. Results show that modern LLMs (e.g., Qwen2.5, LLAMA3.3, DeepseekV3, GPT-4o) replicate L1-dependent patterns observed in human L2 data, with distinct influences from various languages (e.g., Japanese, Korean, and Mandarin significantly affect tense agreement, and Urdu influences noun-verb collocations). Our results reveal the potential of LLMs for L2 dialogue generation and evaluation for future educational applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14502v1">How Much Knowledge Can You Pack into a LoRA Adapter without Harming LLM?</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      The performance of Large Language Models (LLMs) on many tasks is greatly limited by the knowledge learned during pre-training and stored in the model's parameters. Low-rank adaptation (LoRA) is a popular and efficient training technique for updating or domain-specific adaptation of LLMs. In this study, we investigate how new facts can be incorporated into the LLM using LoRA without compromising the previously learned knowledge. We fine-tuned Llama-3.1-8B-instruct using LoRA with varying amounts of new knowledge. Our experiments have shown that the best results are obtained when the training data contains a mixture of known and new facts. However, this approach is still potentially harmful because the model's performance on external question-answering benchmarks declines after such fine-tuning. When the training data is biased towards certain entities, the model tends to regress to few overrepresented answers. In addition, we found that the model becomes more confident and refuses to provide an answer in only few cases. These findings highlight the potential pitfalls of LoRA-based LLM updates and underscore the importance of training data composition and tuning parameters to balance new knowledge integration and general model capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14450v1">LLM4FaaS: No-Code Application Development using LLMs and FaaS</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are powerful tools that can generate code from natural language descriptions. While this theoretically enables non-technical users to develop their own applications, they typically lack the expertise to execute, deploy, and operate generated code. This poses a barrier for such users to leverage the power of LLMs for application development. In this paper, we propose leveraging the high levels of abstraction of the Function-as-a-Service (FaaS) paradigm to handle code execution and operation for non-technical users. FaaS offers function deployment without handling the underlying infrastructure, enabling users to execute LLM-generated code without concern for its operation and without requiring any technical expertise. We propose LLM4FaaS, a novel no-code application development approach that combines LLMs and FaaS platforms to enable non-technical users to build and run their own applications using only natural language descriptions. Specifically, LLM4FaaS takes user prompts, uses LLMs to generate function code based on those prompts, and deploys these functions through a FaaS platform that handles the application's operation. LLM4FaaS also leverages the FaaS infrastructure abstractions to reduce the task complexity for the LLM, improving result accuracy. We evaluate LLM4FaaS with a proof-of-concept implementation based on GPT-4o and an open-source FaaS platform, using real prompts from non-technical users. Our evaluation based on these real user prompts demonstrates the feasibility of our approach and shows that LLM4FaaS can reliably build and deploy code in 71.47% of cases, up from 43.48% in a baseline without FaaS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14445v1">PredictaBoard: Benchmarking LLM Score Predictability</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Despite possessing impressive skills, Large Language Models (LLMs) often fail unpredictably, demonstrating inconsistent success in even basic common sense reasoning tasks. This unpredictability poses a significant challenge to ensuring their safe deployment, as identifying and operating within a reliable "safe zone" is essential for mitigating risks. To address this, we present PredictaBoard, a novel collaborative benchmarking framework designed to evaluate the ability of score predictors (referred to as assessors) to anticipate LLM errors on specific task instances (i.e., prompts) from existing datasets. PredictaBoard evaluates pairs of LLMs and assessors by considering the rejection rate at different tolerance errors. As such, PredictaBoard stimulates research into developing better assessors and making LLMs more predictable, not only with a higher average performance. We conduct illustrative experiments using baseline assessors and state-of-the-art LLMs. PredictaBoard highlights the critical need to evaluate predictability alongside performance, paving the way for safer AI systems where errors are not only minimised but also anticipated and effectively mitigated. Code for our benchmark can be found at https://github.com/Kinds-of-Intelligence-CFI/PredictaBoard
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12769v2">How Much Do LLMs Hallucinate across Languages? On Multilingual Estimation of LLM Hallucination in the Wild</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      In the age of misinformation, hallucination -- the tendency of Large Language Models (LLMs) to generate non-factual or unfaithful responses -- represents the main risk for their global utility. Despite LLMs becoming increasingly multilingual, the vast majority of research on detecting and quantifying LLM hallucination are (a) English-centric and (b) focus on machine translation (MT) and summarization, tasks that are less common ``in the wild'' than open information seeking. In contrast, we aim to quantify the extent of LLM hallucination across languages in knowledge-intensive long-form question answering. To this end, we train a multilingual hallucination detection model and conduct a large-scale study across 30 languages and 6 open-source LLM families. We start from an English hallucination detection dataset and rely on MT to generate (noisy) training data in other languages. We also manually annotate gold data for five high-resource languages; we then demonstrate, for these languages, that the estimates of hallucination rates are similar between silver (LLM-generated) and gold test sets, validating the use of silver data for estimating hallucination rates for other languages. For the final rates estimation, we build a knowledge-intensive QA dataset for 30 languages with LLM-generated prompts and Wikipedia articles as references. We find that, while LLMs generate longer responses with more hallucinated tokens for higher-resource languages, there is no correlation between length-normalized hallucination rates of languages and their digital representation. Further, we find that smaller LLMs exhibit larger hallucination rates than larger models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14389v1">Leveraging Small LLMs for Argument Mining in Education: Argument Component Identification, Classification, and Assessment</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Argument mining algorithms analyze the argumentative structure of essays, making them a valuable tool for enhancing education by providing targeted feedback on the students' argumentation skills. While current methods often use encoder or encoder-decoder deep learning architectures, decoder-only models remain largely unexplored, offering a promising research direction. This paper proposes leveraging open-source, small Large Language Models (LLMs) for argument mining through few-shot prompting and fine-tuning. These models' small size and open-source nature ensure accessibility, privacy, and computational efficiency, enabling schools and educators to adopt and deploy them locally. Specifically, we perform three tasks: segmentation of student essays into arguments, classification of the arguments by type, and assessment of their quality. We empirically evaluate the models on the Feedback Prize - Predicting Effective Arguments dataset of grade 6-12 students essays and demonstrate how fine-tuned small LLMs outperform baseline methods in segmenting the essays and determining the argument types while few-shot prompting yields comparable performance to that of the baselines in assessing quality. This work highlights the educational potential of small, open-source LLMs to provide real-time, personalized feedback, enhancing independent learning and writing skills while ensuring low computational cost and privacy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14359v1">Triangulating LLM Progress through Benchmarks, Games, and Cognitive Tests</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      We examine three evaluation paradigms: large question-answering benchmarks (e.g., MMLU and BBH), interactive games (e.g., Signalling Games or Taboo), and cognitive tests (e.g., for working memory or theory of mind). First, we investigate which of the former two-benchmarks or games-is most effective at discriminating LLMs of varying quality. Then, inspired by human cognitive assessments, we compile a suite of targeted tests that measure cognitive abilities deemed essential for effective language use, and we investigate their correlation with model performance in benchmarks and games. Our analyses reveal that interactive games are superior to standard benchmarks in discriminating models. Causal and logical reasoning correlate with both static and interactive tests, while differences emerge regarding core executive functions and social/emotional skills, which correlate more with games. We advocate the development of new interactive benchmarks and targeted cognitive tasks inspired by assessing human abilities but designed specifically for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14328v1">SolSearch: An LLM-Driven Framework for Efficient SAT-Solving Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      The Satisfiability (SAT) problem is a core challenge with significant applications in software engineering, including automated testing, configuration management, and program verification. This paper presents SolSearch, a novel framework that harnesses large language models (LLMs) to discover and optimize SAT-solving strategies automatically. Leveraging a curriculum-based, trial-and-error process, SolSearch enables the LLM to iteratively modify and generate SAT solver code, thereby improving solving efficiency and performance. This automated SAT-solving paradigm has the advantage of being plug-and-play, allowing integration with any SAT solver and accelerating the development or design process of new SAT solvers (new methods). Our preliminary experimental results are encouraging by demonstrating that the LLM-powered paradigm improves state-of-the-art SAT solvers on general SAT benchmarks and significantly enhances the performance of the widely used Z3 solver (11\% on PAR-2 score). These results highlight the potential for using LLM-driven methods to advance solver adaptability and effectiveness in real-world software engineering challenges. Future research directions are discussed to further refine and validate this approach, offering a promising avenue for integrating AI with traditional software engineering tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14321v1">Beyond Self-Talk: A Communication-Centric Survey of LLM-Based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently demonstrated remarkable capabilities in reasoning, planning, and decision-making. Building upon these strengths, researchers have begun incorporating LLMs into multi-agent systems (MAS), where agents collaborate or compete through natural language interactions to tackle tasks beyond the scope of single-agent setups. In this survey, we present a communication-centric perspective on LLM-based multi-agent systems, examining key system-level features such as architecture design and communication goals, as well as internal mechanisms like communication strategies, paradigms, objects and content. We illustrate how these communication elements interplay to enable collective intelligence and flexible collaboration. Furthermore, we discuss prominent challenges, including scalability, security, and multimodal integration, and propose directions for future work to advance research in this emerging domain. Ultimately, this survey serves as a catalyst for further innovation, fostering more robust, scalable, and intelligent multi-agent systems across diverse application domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14305v1">Efficient AI in Practice: Training and Deployment of Efficient LLMs for Industry Applications</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable performance across a wide range of industrial applications, from search and recommendations to generative tasks. Although scaling laws indicate that larger models generally yield better generalization and performance, their substantial computational requirements often render them impractical for many real-world scenarios at scale. In this paper, we present methods and insights for training small language models (SLMs) that deliver high performance and efficiency in deployment. We focus on two key techniques: (1) knowledge distillation and (2) model compression via quantization and pruning. These approaches enable SLMs to retain much of the quality of their larger counterparts while significantly reducing training, serving costs, and latency. We detail the impact of these techniques on a variety of use cases at a large professional social network platform and share deployment lessons - including hardware optimization strategies that enhance speed and throughput for both predictive and reasoning-based applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13497v2">Towards Geo-Culturally Grounded LLM Generations</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Generative large language models (LLMs) have been demonstrated to have gaps in diverse, cultural knowledge across the globe. We investigate the effect of retrieval augmented generation and search-grounding techniques on the ability of LLMs to display familiarity with a diverse range of national cultures. Specifically, we compare the performance of standard LLMs, LLMs augmented with retrievals from a bespoke knowledge base (i.e., KB grounding), and LLMs augmented with retrievals from a web search (i.e., search grounding) on a series of cultural familiarity benchmarks. We find that search grounding significantly improves the LLM performance on multiple-choice benchmarks that test propositional knowledge (e.g., the norms, artifacts, and institutions of national cultures), while KB grounding's effectiveness is limited by inadequate knowledge base coverage and a suboptimal retriever. However, search grounding also increases the risk of stereotypical judgments by language models, while failing to improve evaluators' judgments of cultural familiarity in a human evaluation with adequate statistical power. These results highlight the distinction between propositional knowledge about a culture and open-ended cultural fluency when it comes to evaluating the cultural familiarity of generative LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.05040v2">SWE-Fixer: Training Open-Source LLMs for Effective and Efficient GitHub Issue Resolution</a></div>
    <div class="paper-meta">
      📅 2025-02-20
      | 💬 Our code, data, and model will be released at https://github.com/InternLM/SWE-Fixer
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable proficiency across a variety of complex tasks. One significant application of LLMs is in tackling software engineering challenges, particularly in resolving real-world tasks on GitHub by fixing code based on the issues reported by the users. However, many current approaches rely on proprietary LLMs, which limits reproducibility, accessibility, and transparency. The critical components of LLMs for addressing software engineering issues and how their capabilities can be effectively enhanced remain unclear. To address these challenges, we introduce SWE-Fixer, a novel open-source framework designed to effectively and efficiently resolve GitHub issues. SWE-Fixer comprises two essential modules: a code file retrieval module and a code editing module. The retrieval module employs BM25 along with a lightweight model to achieve coarse-to-fine file retrieval. Subsequently, the code editing module utilizes the other model to generate patches for the identified files. To mitigate the lack of publicly available datasets, we compile an extensive dataset that includes 110K GitHub issues along with their corresponding patches and train the two models of SWE-Fixer separately. We assess our approach on the SWE-Bench Lite and Verified benchmarks, achieving state-of-the-art performance among open-source models with scores of 24.7% and 32.8%, respectively. Additionally, our approach requires only two model calls per instance, making it significantly more efficient than existing methods. These results highlight the effectiveness of SWE-Fixer in real-world code-fixing scenarios. We will make our model, dataset, and code publicly available at https://github.com/InternLM/SWE-Fixer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.16406v4">SpinQuant: LLM quantization with learned rotations</a></div>
    <div class="paper-meta">
      📅 2025-02-20
      | 💬 ICLR 2025
    </div>
    <details class="paper-abstract">
      Post-training quantization (PTQ) techniques applied to weights, activations, and the KV cache greatly reduce memory usage, latency, and power consumption of Large Language Models (LLMs), but may lead to large quantization errors when outliers are present. Rotating activation or weight matrices helps remove outliers and benefits quantization. In this work, we identify a collection of applicable rotation parameterizations that lead to identical outputs in full-precision Transformer architectures while enhancing quantization accuracy. In addition, we find that some random rotations lead to much better quantization than others, with an up to 13 points difference in downstream zero-shot reasoning performance. As a result, we propose SpinQuant, a novel approach that incorporates learned rotation matrices for optimal quantized network accuracy. With 4-bit quantization of weight, activation, and KV-cache, SpinQuant narrows the accuracy gap on zero-shot reasoning tasks with full precision to merely 2.9 points on the LLaMA-2 7B model, surpassing LLM-QAT by 19.1 points and SmoothQuant by 25.0 points. Furthermore, SpinQuant also outperforms concurrent work QuaRot, which applies random rotations to remove outliers. In particular, for LLaMA-3 8B models that are hard to quantize, SpinQuant reduces the gap to full precision by up to 45.1% relative to QuaRot. Code is available at https://github.com/facebookresearch/SpinQuant.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14276v1">STeCa: Step-level Trajectory Calibration for LLM Agent Learning</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based agents have shown promise in tackling complex tasks by interacting dynamically with the environment. Existing work primarily focuses on behavior cloning from expert demonstrations and preference learning through exploratory trajectory sampling. However, these methods often struggle in long-horizon tasks, where suboptimal actions accumulate step by step, causing agents to deviate from correct task trajectories. To address this, we highlight the importance of timely calibration and the need to automatically construct calibration trajectories for training agents. We propose Step-Level Trajectory Calibration (STeCa), a novel framework for LLM agent learning. Specifically, STeCa identifies suboptimal actions through a step-level reward comparison during exploration. It constructs calibrated trajectories using LLM-driven reflection, enabling agents to learn from improved decision-making processes. These calibrated trajectories, together with successful trajectory data, are utilized for reinforced training. Extensive experiments demonstrate that STeCa significantly outperforms existing methods. Further analysis highlights that step-level calibration enables agents to complete tasks with greater robustness. Our code and data are available at https://github.com/WangHanLinHenry/STeCa.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14273v1">LLM-EvRep: Learning an LLM-Compatible Event Representation Using a Self-Supervised Framework</a></div>
    <div class="paper-meta">
      📅 2025-02-20
      | 💬 6 pages, 2 figures,Companion Proceedings of the ACM Web Conference 2025 (WWW Companion '25)
    </div>
    <details class="paper-abstract">
      Recent advancements in event-based recognition have demonstrated significant promise, yet most existing approaches rely on extensive training, limiting their adaptability for efficient processing of event-driven visual content. Meanwhile, large language models (LLMs) have exhibited remarkable zero-shot capabilities across diverse domains, but their application to event-based visual recognition remains largely unexplored. To bridge this gap, we propose \textbf{LLM-EvGen}, an event representation generator that produces LLM-compatible event representations \textbf{LLM-EvRep}, thereby enhancing the performance of LLMs on event recognition tasks. The generator is trained using a self-supervised framework, aligning the generated representations with semantic consistency and structural fidelity. Comprehensive experiments were conducted on three datasets: N-ImageNet, N-Caltech101, and N-MNIST. The results demonstrate that our method, \textbf{LLM-EvRep}, outperforms the event-to-video method, E2VID, by 15.93\%, 0.82\%, and 50.21\%, respectively, in recognition tasks when evaluated using GPT-4o.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14271v1">PaperHelper: Knowledge-Based LLM QA Paper Reading Assistant</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      In the paper, we introduce a paper reading assistant, PaperHelper, a potent tool designed to enhance the capabilities of researchers in efficiently browsing and understanding scientific literature. Utilizing the Retrieval-Augmented Generation (RAG) framework, PaperHelper effectively minimizes hallucinations commonly encountered in large language models (LLMs), optimizing the extraction of accurate, high-quality knowledge. The implementation of advanced technologies such as RAFT and RAG Fusion significantly boosts the performance, accuracy, and reliability of the LLMs-based literature review process. Additionally, PaperHelper features a user-friendly interface that facilitates the batch downloading of documents and uses the Mermaid format to illustrate structural relationships between documents. Experimental results demonstrate that PaperHelper, based on a fine-tuned GPT-4 API, achieves an F1 Score of 60.04, with a latency of only 5.8 seconds, outperforming the basic RAG model by 7\% in F1 Score.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04323v2">SCOOT: SLO-Oriented Performance Tuning for LLM Inference Engines</a></div>
    <div class="paper-meta">
      📅 2025-02-20
      | 💬 Accepted by The ACM Web Conference 2025 (WWW'25)
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are gaining increasing popularity across a wide range of web applications, it is of great importance to optimize service-level objectives (SLOs) for LLM inference services to enhance user satisfaction and improve the competitiveness of cloud vendors. In this paper, we observe that adjusting the parameters of LLM inference engines can improve service performance, and the optimal parameter configurations of different services are different. Therefore, we propose SCOOT, an automatic performance tuning system to optimize SLOs for each LLM inference service by tuning the parameters of the inference engine. SCOOT jointly exploits single-objective and multiple-objective Bayesian optimization (BO) techniques to handle various optimization objectives via exploration and exploitation. Moreover, SCOOT prunes the search space with known constraints and adopts a random forest to learn hidden constraints during the tuning process to mitigate invalid exploration. To improve the tuning efficiency, SCOOT utilizes the parallel suggestion to accelerate the tuning process. Extensive experiments demonstrate that SCOOT considerably outperforms existing tuning techniques in SLO optimization while greatly improving the tuning efficiency. Moreover, SCOOT is universally applicable to various LLM inference engines including vLLM and TensorRT-LLM. Currently, SCOOT has already been implemented in the production environment at Ant Group.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14251v2">Synthesizing Post-Training Data for LLMs through Multi-Agent Simulation</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Post-training is essential for enabling large language models (LLMs) to follow human instructions. However, its effectiveness depends on high-quality instruction data, which is challenging to obtain in the real world due to privacy concerns, data scarcity, and high annotation costs. To fill this gap, inspired by the recent success of using LLMs to simulate human society, we propose MATRIX, a multi-agent simulator that automatically generates diverse text-based scenarios, capturing a wide range of real-world human needs in a realistic and scalable manner. Leveraging these outputs, we introduce a novel scenario-driven instruction generator MATRIX-Gen for controllable and highly realistic data synthesis. Extensive experiments demonstrate that our framework effectively generates both general and domain-specific data. On AlpacaEval 2 and Arena-Hard benchmarks, Llama-3-8B-Base, post-trained on datasets synthesized by MATRIX-Gen with just 20K instruction-response pairs, outperforms Meta's Llama-3-8B-Instruct model, which was trained on over 10M pairs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14219v1">Investigating the Impact of LLM Personality on Cognitive Bias Manifestation in Automated Decision-Making Tasks</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used in decision-making, yet their susceptibility to cognitive biases remains a pressing challenge. This study explores how personality traits influence these biases and evaluates the effectiveness of mitigation strategies across various model architectures. Our findings identify six prevalent cognitive biases, while the sunk cost and group attribution biases exhibit minimal impact. Personality traits play a crucial role in either amplifying or reducing biases, significantly affecting how LLMs respond to debiasing techniques. Notably, Conscientiousness and Agreeableness may generally enhance the efficacy of bias mitigation strategies, suggesting that LLMs exhibiting these traits are more receptive to corrective measures. These findings address the importance of personality-driven bias dynamics and highlight the need for targeted mitigation approaches to improve fairness and reliability in AI-assisted decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14215v1">Towards Secure Program Partitioning for Smart Contracts with LLM's In-Context Learning</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Smart contracts are highly susceptible to manipulation attacks due to the leakage of sensitive information. Addressing manipulation vulnerabilities is particularly challenging because they stem from inherent data confidentiality issues rather than straightforward implementation bugs. To tackle this by preventing sensitive information leakage, we present PartitionGPT, the first LLM-driven approach that combines static analysis with the in-context learning capabilities of large language models (LLMs) to partition smart contracts into privileged and normal codebases, guided by a few annotated sensitive data variables. We evaluated PartitionGPT on 18 annotated smart contracts containing 99 sensitive functions. The results demonstrate that PartitionGPT successfully generates compilable, and verified partitions for 78% of the sensitive functions while reducing approximately 30% code compared to function-level partitioning approach. Furthermore, we evaluated PartitionGPT on nine real-world manipulation attacks that lead to a total loss of 25 million dollars, PartitionGPT effectively prevents eight cases, highlighting its potential for broad applicability and the necessity for secure program partitioning during smart contract development to diminish manipulation vulnerabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14924v1">A Tale of Two Structures: Do LLMs Capture the Fractal Complexity of Language?</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Language exhibits a fractal structure in its information-theoretic complexity (i.e. bits per token), with self-similarity across scales and long-range dependence (LRD). In this work, we investigate whether large language models (LLMs) can replicate such fractal characteristics and identify conditions-such as temperature setting and prompting method-under which they may fail. Moreover, we find that the fractal parameters observed in natural language are contained within a narrow range, whereas those of LLMs' output vary widely, suggesting that fractal parameters might prove helpful in detecting a non-trivial portion of LLM-generated texts. Notably, these findings, and many others reported in this work, are robust to the choice of the architecture; e.g. Gemini 1.0 Pro, Mistral-7B and Gemma-2B. We also release a dataset comprising of over 240,000 articles generated by various LLMs (both pretrained and instruction-tuned) with different decoding temperatures and prompting methods, along with their corresponding human-generated texts. We hope that this work highlights the complex interplay between fractal properties, prompting, and statistical mimicry in LLMs, offering insights for generating, evaluating and detecting synthetic texts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14922v1">SIFT: Grounding LLM Reasoning in Contexts via Stickers</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      This paper identifies the misinterpretation of the context can be a significant issue during the reasoning process of large language models, spanning from smaller models like Llama3.2-3B-Instruct to cutting-edge ones like DeepSeek-R1. For example, in the phrase "10 dollars per kilo," LLMs might not recognize that "per" means "for each," leading to calculation errors. We introduce a novel, post-training approach called **Stick to the Facts (SIFT)** to tackle this. SIFT leverages increasing inference-time compute to ground LLM reasoning in contexts. At the core of SIFT lies the *Sticker*, which is generated by the model itself to explicitly emphasize the key information within the context. Given the curated Sticker, SIFT generates two predictions -- one from the original query and one from the query augmented with the Sticker. If they differ, the Sticker is sequentially refined via *forward* optimization (to better align the extracted facts with the query) and *inverse* generation (to conform with the model's inherent tendencies) for more faithful reasoning outcomes. Studies across diverse models (from 3B to 100B+) and benchmarks (e.g., GSM8K, MATH-500) reveal consistent performance improvements. Notably, SIFT improves the pass@1 accuracy of DeepSeek-R1 on AIME2024 from 78.33% to **85.67**%, establishing a new state-of-the-art in the open-source community. The code is available at https://github.com/zhijie-group/SIFT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14921v1">The Canary's Echo: Auditing Privacy Risks of LLM-Generated Synthetic Text</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      How much information about training samples can be gleaned from synthetic data generated by Large Language Models (LLMs)? Overlooking the subtleties of information flow in synthetic data generation pipelines can lead to a false sense of privacy. In this paper, we design membership inference attacks (MIAs) that target data used to fine-tune pre-trained LLMs that are then used to synthesize data, particularly when the adversary does not have access to the fine-tuned model but only to the synthetic data. We show that such data-based MIAs do significantly better than a random guess, meaning that synthetic data leaks information about the training data. Further, we find that canaries crafted to maximize vulnerability to model-based MIAs are sub-optimal for privacy auditing when only synthetic data is released. Such out-of-distribution canaries have limited influence on the model's output when prompted to generate useful, in-distribution synthetic data, which drastically reduces their vulnerability. To tackle this problem, we leverage the mechanics of auto-regressive models to design canaries with an in-distribution prefix and a high-perplexity suffix that leave detectable traces in synthetic data. This enhances the power of data-based MIAs and provides a better assessment of the privacy risks of releasing synthetic data generated by LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14910v1">EvoP: Robust LLM Inference via Evolutionary Pruning</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable success in natural language processing tasks, but their massive size and computational demands hinder their deployment in resource-constrained environments. Existing structured pruning methods address this issue by removing redundant structures (e.g., elements, channels, layers) from the model. However, these methods employ a heuristic pruning strategy, which leads to suboptimal performance. Besides, they also ignore the data characteristics when pruning the model. To overcome these limitations, we propose EvoP, an evolutionary pruning framework for robust LLM inference. EvoP first presents a cluster-based calibration dataset sampling (CCDS) strategy for creating a more diverse calibration dataset. EvoP then introduces an evolutionary pruning pattern searching (EPPS) method to find the optimal pruning pattern. Compared to existing structured pruning techniques, EvoP achieves the best performance while maintaining the best efficiency. Experiments across different LLMs and different downstream tasks validate the effectiveness of the proposed EvoP, making it a practical and scalable solution for deploying LLMs in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14907v1">GneissWeb: Preparing High Quality Data for LLMs at Scale</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Data quantity and quality play a vital role in determining the performance of Large Language Models (LLMs). High-quality data, in particular, can significantly boost the LLM's ability to generalize on a wide range of downstream tasks. Large pre-training datasets for leading LLMs remain inaccessible to the public, whereas many open datasets are small in size (less than 5 trillion tokens), limiting their suitability for training large models. In this paper, we introduce GneissWeb, a large dataset yielding around 10 trillion tokens that caters to the data quality and quantity requirements of training LLMs. Our GneissWeb recipe that produced the dataset consists of sharded exact sub-string deduplication and a judiciously constructed ensemble of quality filters. GneissWeb achieves a favorable trade-off between data quality and quantity, producing models that outperform models trained on state-of-the-art open large datasets (5+ trillion tokens). We show that models trained using GneissWeb dataset outperform those trained on FineWeb-V1.1.0 by 2.73 percentage points in terms of average score computed on a set of 11 commonly used benchmarks (both zero-shot and few-shot) for pre-training dataset evaluation. When the evaluation set is extended to 20 benchmarks (both zero-shot and few-shot), models trained using GneissWeb still achieve a 1.75 percentage points advantage over those trained on FineWeb-V1.1.0.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.14043v3">Taxonomy-Guided Zero-Shot Recommendations with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      With the emergence of large language models (LLMs) and their ability to perform a variety of tasks, their application in recommender systems (RecSys) has shown promise. However, we are facing significant challenges when deploying LLMs into RecSys, such as limited prompt length, unstructured item information, and un-constrained generation of recommendations, leading to sub-optimal performance. To address these issues, we propose a novel method using a taxonomy dictionary. This method provides a systematic framework for categorizing and organizing items, improving the clarity and structure of item information. By incorporating the taxonomy dictionary into LLM prompts, we achieve efficient token utilization and controlled feature generation, leading to more accurate and contextually relevant recommendations. Our Taxonomy-guided Recommendation (TaxRec) approach features a two-step process: one-time taxonomy categorization and LLM-based recommendation, enabling zero-shot recommendations without the need for domain-specific fine-tuning. Experimental results demonstrate TaxRec significantly enhances recommendation quality compared to traditional zero-shot approaches, showcasing its efficacy as personal recommender with LLMs. Code is available at https://github.com/yueqingliang1/TaxRec.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14122v1">Benchmarking LLMs for Political Science: A United Nations Perspective</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved significant advances in natural language processing, yet their potential for high-stake political decision-making remains largely unexplored. This paper addresses the gap by focusing on the application of LLMs to the United Nations (UN) decision-making process, where the stakes are particularly high and political decisions can have far-reaching consequences. We introduce a novel dataset comprising publicly available UN Security Council (UNSC) records from 1994 to 2024, including draft resolutions, voting records, and diplomatic speeches. Using this dataset, we propose the United Nations Benchmark (UNBench), the first comprehensive benchmark designed to evaluate LLMs across four interconnected political science tasks: co-penholder judgment, representative voting simulation, draft adoption prediction, and representative statement generation. These tasks span the three stages of the UN decision-making process--drafting, voting, and discussing--and aim to assess LLMs' ability to understand and simulate political dynamics. Our experimental analysis demonstrates the potential and challenges of applying LLMs in this domain, providing insights into their strengths and limitations in political science. This work contributes to the growing intersection of AI and political science, opening new avenues for research and practical applications in global governance. The UNBench Repository can be accessed at: https://github.com/yueqingliang1/UNBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.10735v3">Assessing the Reasoning Capabilities of LLMs in the context of Evidence-based Claim Verification</a></div>
    <div class="paper-meta">
      📅 2025-02-19
      | 💬 {\dag} These authors contributed equally to this work. 23 pages, 3 figure
    </div>
    <details class="paper-abstract">
      Although LLMs have shown great performance on Mathematics and Coding related reasoning tasks, the reasoning capabilities of LLMs regarding other forms of reasoning are still an open problem. Here, we examine the issue of reasoning from the perspective of claim verification. We propose a framework designed to break down any claim paired with evidence into atomic reasoning types that are necessary for verification. We use this framework to create Reasoning in Evidence-based Claim Verification (RECV), the first claim verification benchmark, incorporating real-world claims, to assess the deductive and abductive reasoning capabilities of LLMs. The benchmark comprises of three datasets, covering reasoning problems of increasing complexity. We evaluate three state-of-the-art proprietary LLMs under multiple prompt settings. Our results show that while LLMs can address deductive reasoning problems, they consistently fail in cases of abductive reasoning. Moreover, we observe that enhancing LLMs with rationale generation is not always beneficial. Nonetheless, we find that generated rationales are semantically similar to those provided by humans, especially in deductive reasoning cases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14100v1">Towards Context-Robust LLMs: A Gated Representation Fine-tuning Approach</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) enhanced with external contexts, such as through retrieval-augmented generation (RAG), often face challenges in handling imperfect evidence. They tend to over-rely on external knowledge, making them vulnerable to misleading and unhelpful contexts. To address this, we propose the concept of context-robust LLMs, which can effectively balance internal knowledge with external context, similar to human cognitive processes. Specifically, context-robust LLMs should rely on external context only when lacking internal knowledge, identify contradictions between internal and external knowledge, and disregard unhelpful contexts. To achieve this goal, we introduce Grft, a lightweight and plug-and-play gated representation fine-tuning approach. Grft consists of two key components: a gating mechanism to detect and filter problematic inputs, and low-rank representation adapters to adjust hidden representations. By training a lightweight intervention function with only 0.0004\% of model size on fewer than 200 examples, Grft can effectively adapt LLMs towards context-robust behaviors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14074v1">Investigating Non-Transitivity in LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-02-19
      | 💬 8 pages, 6 figures, 2 tables (30 pages, 11 figures, 8 tables including references and appendices)
    </div>
    <details class="paper-abstract">
      Automatic evaluation methods based on large language models (LLMs) are emerging as the standard tool for assessing the instruction-following abilities of LLM-based agents. The most common method in this paradigm, pairwise comparisons with a baseline model, critically depends on the assumption of transitive preferences. However, the validity of this assumption remains largely unexplored. In this study, we investigate the presence of non-transitivity within the AlpacaEval framework and analyze its effects on model rankings. We find that LLM judges exhibit non-transitive preferences, leading to rankings that are sensitive to the choice of the baseline model. To mitigate this issue, we show that round-robin tournaments combined with Bradley-Terry models of preference can produce more reliable rankings. Notably, our method increases both the Spearman correlation and the Kendall correlation with Chatbot Arena (95.0% -> 96.4% and 82.1% -> 86.3% respectively). To address the computational cost of round-robin tournaments, we propose Swiss-Wise Iterative Matchmaking (Swim) tournaments, using a dynamic matching strategy to capture the benefits of round-robin tournaments while maintaining computational efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14052v1">A Matter of Perspective(s): Contrasting Human and LLM Argumentation in Subjective Decision-Making on Subtle Sexism</a></div>
    <div class="paper-meta">
      📅 2025-02-19
      | 💬 Accepted at CHI Conference on Human Factors in Computing Systems (CHI '25), April 26-May 1, 2025, Yokohama, Japan
    </div>
    <details class="paper-abstract">
      In subjective decision-making, where decisions are based on contextual interpretation, Large Language Models (LLMs) can be integrated to present users with additional rationales to consider. The diversity of these rationales is mediated by the ability to consider the perspectives of different social actors. However, it remains unclear whether and how models differ in the distribution of perspectives they provide. We compare the perspectives taken by humans and different LLMs when assessing subtle sexism scenarios. We show that these perspectives can be classified within a finite set (perpetrator, victim, decision-maker), consistently present in argumentations produced by humans and LLMs, but in different distributions and combinations, demonstrating differences and similarities with human responses, and between models. We argue for the need to systematically evaluate LLMs' perspective-taking to identify the most suitable models for a given decision-making task. We discuss the implications for model evaluation.
    </details>
</div>
