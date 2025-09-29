# llm - 2025_09

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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13268v1">LLMs for energy and macronutrients estimation using only text data from 24-hour dietary recalls: a parameter-efficient fine-tuning experiment using a 10-shot prompt</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 https://github.com/rodrigo-carrillo/LLMs-Macronutrient-Estimation-NHANES-Adolescents
    </div>
    <details class="paper-abstract">
      BACKGROUND: Most artificial intelligence tools used to estimate nutritional content rely on image input. However, whether large language models (LLMs) can accurately predict nutritional values based solely on text descriptions of foods consumed remains unknown. If effective, this approach could enable simpler dietary monitoring without the need for photographs. METHODS: We used 24-hour dietary recalls from adolescents aged 12-19 years in the National Health and Nutrition Examination Survey (NHANES). An open-source quantized LLM was prompted using a 10-shot, chain-of-thought approach to estimate energy and five macronutrients based solely on text strings listing foods and their quantities. We then applied parameter-efficient fine-tuning (PEFT) to evaluate whether predictive accuracy improved. NHANES-calculated values served as the ground truth for energy, proteins, carbohydrates, total sugar, dietary fiber and total fat. RESULTS: In a pooled dataset of 11,281 adolescents (49.9% male, mean age 15.4 years), the vanilla LLM yielded poor predictions. The mean absolute error (MAE) was 652.08 for energy and the Lin's CCC <0.46 across endpoints. In contrast, the fine-tuned model performed substantially better, with energy MAEs ranging from 171.34 to 190.90 across subsets, and Lin's CCC exceeding 0.89 for all outcomes. CONCLUSIONS: When prompted using a chain-of-thought approach and fine-tuned with PEFT, open-source LLMs exposed solely to text input can accurately predict energy and macronutrient values from 24-hour dietary recalls. This approach holds promise for low-burden, text-based dietary monitoring tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13244v1">Evaluating LLM Alignment on Personality Inference from Real-World Interview Data</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 8 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed in roles requiring nuanced psychological understanding, such as emotional support agents, counselors, and decision-making assistants. However, their ability to interpret human personality traits, a critical aspect of such applications, remains unexplored, particularly in ecologically valid conversational settings. While prior work has simulated LLM "personas" using discrete Big Five labels on social media data, the alignment of LLMs with continuous, ground-truth personality assessments derived from natural interactions is largely unexamined. To address this gap, we introduce a novel benchmark comprising semi-structured interview transcripts paired with validated continuous Big Five trait scores. Using this dataset, we systematically evaluate LLM performance across three paradigms: (1) zero-shot and chain-of-thought prompting with GPT-4.1 Mini, (2) LoRA-based fine-tuning applied to both RoBERTa and Meta-LLaMA architectures, and (3) regression using static embeddings from pretrained BERT and OpenAI's text-embedding-3-small. Our results reveal that all Pearson correlations between model predictions and ground-truth personality traits remain below 0.26, highlighting the limited alignment of current LLMs with validated psychological constructs. Chain-of-thought prompting offers minimal gains over zero-shot, suggesting that personality inference relies more on latent semantic representation than explicit reasoning. These findings underscore the challenges of aligning LLMs with complex human attributes and motivate future work on trait-specific prompting, context-aware modeling, and alignment-oriented fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13237v1">Metacognitive Reuse: Turning Recurring LLM Reasoning Into Concise Behaviors</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 18 pages, 9 Figures, 5 Tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) now solve multi-step problems by emitting extended chains of thought. During the process, they often re-derive the same intermediate steps across problems, inflating token usage and latency. This saturation of the context window leaves less capacity for exploration. We study a simple mechanism that converts recurring reasoning fragments into concise, reusable "behaviors" (name + instruction) via the model's own metacognitive analysis of prior traces. These behaviors are stored in a "behavior handbook" which supplies them to the model in-context at inference or distills them into parameters via supervised fine-tuning. This approach achieves improved test-time reasoning across three different settings - 1) Behavior-conditioned inference: Providing the LLM relevant behaviors in-context during reasoning reduces number of reasoning tokens by up to 46% while matching or improving baseline accuracy; 2) Behavior-guided self-improvement: Without any parameter updates, the model improves its own future reasoning by leveraging behaviors from its own past problem solving attempts. This yields up to 10% higher accuracy than a naive critique-and-revise baseline; and 3) Behavior-conditioned SFT: SFT on behavior-conditioned reasoning traces is more effective at converting non-reasoning models into reasoning models as compared to vanilla SFT. Together, these results indicate that turning slow derivations into fast procedural hints enables LLMs to remember how to reason, not just what to conclude.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13201v1">Scaling Up Throughput-oriented LLM Inference Applications on Heterogeneous Opportunistic GPU Clusters with Pervasive Context Management</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      The widespread growth in LLM developments increasingly demands more computational power from clusters than what they can supply. Traditional LLM applications inherently require huge static resource allocations, which force users to either wait in a long job queue and accept progress delay, or buy expensive hardware to fulfill their needs and exacerbate the demand-supply problem. However, not all LLM applications are latency-sensitive and can instead be executed in a throughput-oriented way. This throughput orientation allows a dynamic allocation that opportunistically pools available resources over time, avoiding both the long queue and expensive GPU purchases. Effectively utilizing opportunistic resources brings numerous challenges nevertheless. Our solution, pervasive context management, exploits the common computational context in LLM applications and provides mechanisms and policies that allow seamless context reuse on opportunistic resources. Our evaluation shows an LLM application with pervasive context management on opportunistic resources reduces its execution time by 98.1%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13179v1">Efficient Cold-Start Recommendation via BPE Token-Level Embedding Initialization with LLM</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      The cold-start issue is the challenge when we talk about recommender systems, especially in the case when we do not have the past interaction data of new users or new items. Content-based features or hybrid solutions are common as conventional solutions, but they can only work in a sparse metadata environment with shallow patterns. In this paper, the efficient cold-start recommendation strategy is presented, which is based on the sub word-level representations by applying Byte Pair Encoding (BPE) tokenization and pre-trained Large Language Model (LLM) embedding in the initialization procedure. We obtain fine-grained token-level vectors that are aligned with the BPE vocabulary as opposed to using coarse-grained sentence embeddings. Together, these token embeddings can be used as dense semantic priors on unseen entities, making immediate recommendation performance possible without user-item interaction history. Our mechanism can be compared to collaborative filtering systems and tested over benchmark datasets with stringent cold-start assumptions. Experimental findings show that the given BPE-LLM method achieves higher Recall@k, NDCG@k, and Hit Rate measurements compared to the standard baseline and displays the same capability of sufficient computational performance. Furthermore, we demonstrate that using subword-aware embeddings yields better generalizability and is more interpretable, especially within a multilingual and sparse input setting. The practical application of token-level semantic initialization as a lightweight, but nevertheless effective extension to modern recommender systems in the zero-shot setting is indicated within this work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13175v1">More performant and scalable: Rethinking contrastive vision-language pre-training of radiology in the LLM era</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 MICCAI 2025
    </div>
    <details class="paper-abstract">
      The emergence of Large Language Models (LLMs) presents unprecedented opportunities to revolutionize medical contrastive vision-language pre-training. In this paper, we show how LLMs can facilitate large-scale supervised pre-training, thereby advancing vision-language alignment. We begin by demonstrate that modern LLMs can automatically extract diagnostic labels from radiology reports with remarkable precision (>96\% AUC in our experiments) without complex prompt engineering, enabling the creation of large-scale "silver-standard" datasets at a minimal cost (~\$3 for 50k CT image-report pairs). Further, we find that vision encoder trained on this "silver-standard" dataset achieves performance comparable to those trained on labels extracted by specialized BERT-based models, thereby democratizing the access to large-scale supervised pre-training. Building on this foundation, we proceed to reveal that supervised pre-training fundamentally improves contrastive vision-language alignment. Our approach achieves state-of-the-art performance using only a 3D ResNet-18 with vanilla CLIP training, including 83.8\% AUC for zero-shot diagnosis on CT-RATE, 77.3\% AUC on RAD-ChestCT, and substantial improvements in cross-modal retrieval (MAP@50=53.7\% for image-image, Recall@100=52.2\% for report-image). These results demonstrate the potential of utilizing LLMs to facilitate {\bf more performant and scalable} medical AI systems. Our code is avaiable at https://github.com/SadVoxel/More-performant-and-scalable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13154v1">LLM Hallucination Detection: A Fast Fourier Transform Method Based on Hidden Layer Temporal Signals</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      Hallucination remains a critical barrier for deploying large language models (LLMs) in reliability-sensitive applications. Existing detection methods largely fall into two categories: factuality checking, which is fundamentally constrained by external knowledge coverage, and static hidden-state analysis, that fails to capture deviations in reasoning dynamics. As a result, their effectiveness and robustness remain limited. We propose HSAD (Hidden Signal Analysis-based Detection), a novel hallucination detection framework that models the temporal dynamics of hidden representations during autoregressive generation. HSAD constructs hidden-layer signals by sampling activations across layers, applies Fast Fourier Transform (FFT) to obtain frequency-domain representations, and extracts the strongest non-DC frequency component as spectral features. Furthermore, by leveraging the autoregressive nature of LLMs, HSAD identifies optimal observation points for effective and reliable detection. Across multiple benchmarks, including TruthfulQA, HSAD achieves over 10 percentage points improvement compared to prior state-of-the-art methods. By integrating reasoning-process modeling with frequency-domain analysis, HSAD establishes a new paradigm for robust hallucination detection in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13127v1">Empowering LLMs with Parameterized Skills for Adversarial Long-Horizon Planning</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 Accepted to IJCNN 2025
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models(LLMs) have led to the development of LLM-based AI agents. A key challenge is the creation of agents that can effectively ground themselves in complex, adversarial long-horizon environments. Existing methods mainly focus on (1) using LLMs as policies to interact with the environment through generating low-level feasible actions, and (2) utilizing LLMs to generate high-level tasks or language guides to stimulate action generation. However, the former struggles to generate reliable actions, while the latter relies heavily on expert experience to translate high-level tasks into specific action sequences. To address these challenges, we introduce the Plan with Language, Act with Parameter (PLAP) planning framework that facilitates the grounding of LLM-based agents in long-horizon environments. The PLAP method comprises three key components: (1) a skill library containing environment-specific parameterized skills, (2) a skill planner powered by LLMs, and (3) a skill executor converting the parameterized skills into executable action sequences. We implement PLAP in MicroRTS, a long-horizon real-time strategy game that provides an unfamiliar and challenging environment for LLMs. The experimental results demonstrate the effectiveness of PLAP. In particular, GPT-4o-driven PLAP in a zero-shot setting outperforms 80% of baseline agents, and Qwen2-72B-driven PLAP, with carefully crafted few-shot examples, surpasses the top-tier scripted agent, CoacAI. Additionally, we design comprehensive evaluation metrics and test 6 closed-source and 2 open-source LLMs within the PLAP framework, ultimately releasing an LLM leaderboard ranking long-horizon skill planning ability. Our code is available at https://github.com/AI-Research-TeamX/PLAP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18436v3">Can Code-Switched Texts Activate a Knowledge Switch in LLMs? A Case Study on English-Korean Code-Switching</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 Accepted to EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Recent large language models (LLMs) demonstrate multilingual abilities, yet they are English-centric due to dominance of English in training corpora. The limited resource for low-resource languages remains a crucial challenge. Code-switching (CS), a phenomenon where multilingual speakers alternate between languages in a discourse, can convey subtle cultural and linguistic nuances that can be otherwise lost in translation and elicits language-specific knowledge in human communications. In light of this, we investigate whether code-switching can activate, or identify and leverage knowledge for reasoning when LLMs solve low-resource language tasks. To facilitate the research, we first present EnKoQA, a synthetic English-Korean CS question-answering dataset. We provide comprehensive analysis on a variety of multilingual LLMs by subdividing activation process into knowledge identification and knowledge leveraging. Our results demonstrate that compared to English text, CS can faithfully activate knowledge inside LLMs especially on language-specific domains, suggesting the potential of code-switching on low-resource language tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13103v1">Accelerating Discovery: Rapid Literature Screening with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 This version of the manuscript has been submitted to Empirical Software Engieering Journal for consideration
    </div>
    <details class="paper-abstract">
      Background: Conducting Multi Vocal Literature Reviews (MVLRs) is often time and effort-intensive. Researchers must review and filter a large number of unstructured sources, which frequently contain sparse information and are unlikely to be included in the final study. Our experience conducting an MVLR on Context-Aware Software Systems (CASS) Testing in the avionics domain exemplified this challenge, with over 8,000 highly heterogeneous documents requiring review. Therefore, we developed a Large Language Model (LLM) assistant to support the search and filtering of documents. Aims: To develop and validate an LLM based tool that can support researchers in performing the search and filtering of documents for an MVLR without compromising the rigor of the research protocol. Method: We applied sound engineering practices to develop an on-premises LLM-based tool incorporating Retrieval Augmented Generation (RAG) to process candidate sources. Progress towards the aim was quantified using the Positive Percent Agreement (PPA) as the primary metric to ensure the performance of the LLM based tool. Convenience sampling, supported by human judgment and statistical sampling, were used to verify and validate the tool's quality-in-use. Results: The tool currently demonstrates a PPA agreement with human researchers of 90% for sources that are not relevant to the study. Development details are shared to support domain-specific adaptation of the tool. Conclusions: Using LLM-based tools to support academic researchers in rigorous MVLR is feasible. These tools can free valuable time for higher-level, abstract tasks. However, researcher participation remains essential to ensure that the tool supports thorough research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21740v2">Counterfactual Simulatability of LLM Explanations for Generation Tasks</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      LLMs can be unpredictable, as even slight alterations to the prompt can cause the output to change in unexpected ways. Thus, the ability of models to accurately explain their behavior is critical, especially in high-stakes settings. One approach for evaluating explanations is counterfactual simulatability, how well an explanation allows users to infer the model's output on related counterfactuals. Counterfactual simulatability has been previously studied for yes/no question answering tasks. We provide a general framework for extending this method to generation tasks, using news summarization and medical suggestion as example use cases. We find that while LLM explanations do enable users to better predict LLM outputs on counterfactuals in the summarization setting, there is significant room for improvement for medical suggestion. Furthermore, our results suggest that the evaluation for counterfactual simulatability may be more appropriate for skill-based tasks as opposed to knowledge-based tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13055v1">Automating Code Generation for Semiconductor Equipment Control from Developer Utterances with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      Semiconductors form the backbone of modern electronics, with their manufacturing and testing relying on highly specialized equipment and domain-specific programming languages. Equipment languages such as the Algorithmic Pattern Generator (ALPG) are critical for precise hardware control but are challenging to program due to their low-level syntax and steep learning curve. While large language models (LLMs) have shown promise in generating high-level code from natural language, their effectiveness on low-level equipment languages remains limited. To address this, we propose Progressive Knowledge Enhancement (PKE), a novel multi-stage prompting framework that progressively extracts and activates the latent knowledge within LLMs, guiding them from simple to complex examples without extensive fine-tuning. Empirical evaluation on an industrial ALPG dataset shows that PKE significantly outperforms standard prompting and surpasses state-of-the-art methods in generating correct ALPG code, achieving 11.1\% and 15.2\% higher exact match scores compared to the second-best technique. Further analysis of individual components confirms that progressive knowledge extraction based on difficulty enhances accuracy. Our study offer a practical approach to boosting LLM capabilities for specialized low-level programming, supporting greater productivity in semiconductor software development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08045v3">Break the Checkbox: Challenging Closed-Style Evaluations of Cultural Alignment in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 Accepted at EMNLP 2025 (Main)
    </div>
    <details class="paper-abstract">
      A large number of studies rely on closed-style multiple-choice surveys to evaluate cultural alignment in Large Language Models (LLMs). In this work, we challenge this constrained evaluation paradigm and explore more realistic, unconstrained approaches. Using the World Values Survey (WVS) and Hofstede Cultural Dimensions as case studies, we demonstrate that LLMs exhibit stronger cultural alignment in less constrained settings, where responses are not forced. Additionally, we show that even minor changes, such as reordering survey choices, lead to inconsistent outputs, exposing the limitations of closed-style evaluations. Our findings advocate for more robust and flexible evaluation frameworks that focus on specific cultural proxies, encouraging more nuanced and accurate assessments of cultural alignment in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16467v2">Reading Between the Prompts: How Stereotypes Shape LLM's Implicit Personalization</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 Accepted at EMNLP Main 2025
    </div>
    <details class="paper-abstract">
      Generative Large Language Models (LLMs) infer user's demographic information from subtle cues in the conversation -- a phenomenon called implicit personalization. Prior work has shown that such inferences can lead to lower quality responses for users assumed to be from minority groups, even when no demographic information is explicitly provided. In this work, we systematically explore how LLMs respond to stereotypical cues using controlled synthetic conversations, by analyzing the models' latent user representations through both model internals and generated answers to targeted user questions. Our findings reveal that LLMs do infer demographic attributes based on these stereotypical signals, which for a number of groups even persists when the user explicitly identifies with a different demographic group. Finally, we show that this form of stereotype-driven implicit personalization can be effectively mitigated by intervening on the model's internal representations using a trained linear probe to steer them toward the explicitly stated identity. Our results highlight the need for greater transparency and control in how LLMs represent user identity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13021v1">xOffense: An AI-driven autonomous penetration testing framework with offensive knowledge-enhanced LLMs and multi agent systems</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 17 pages, 4 figures
    </div>
    <details class="paper-abstract">
      This work introduces xOffense, an AI-driven, multi-agent penetration testing framework that shifts the process from labor-intensive, expert-driven manual efforts to fully automated, machine-executable workflows capable of scaling seamlessly with computational infrastructure. At its core, xOffense leverages a fine-tuned, mid-scale open-source LLM (Qwen3-32B) to drive reasoning and decision-making in penetration testing. The framework assigns specialized agents to reconnaissance, vulnerability scanning, and exploitation, with an orchestration layer ensuring seamless coordination across phases. Fine-tuning on Chain-of-Thought penetration testing data further enables the model to generate precise tool commands and perform consistent multi-step reasoning. We evaluate xOffense on two rigorous benchmarks: AutoPenBench and AI-Pentest-Benchmark. The results demonstrate that xOffense consistently outperforms contemporary methods, achieving a sub-task completion rate of 79.17%, decisively surpassing leading systems such as VulnBot and PentestGPT. These findings highlight the potential of domain-adapted mid-scale LLMs, when embedded within structured multi-agent orchestration, to deliver superior, cost-efficient, and reproducible solutions for autonomous penetration testing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.19894v3">TRANSAGENT: An LLM-Based Multi-Agent System for Code Translation</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      Code translation converts code from one programming language to another while maintaining its original functionality, which is crucial for software migration, system refactoring, and cross-platform development. Traditional rule-based methods rely on manually-written rules, which can be time-consuming and often result in less readable code. To overcome this, learning-based methods have been developed, leveraging parallel data to train models for automated code translation. More recently, the advance of Large Language Models (LLMs) further boosts learning-based code translation. Although promising, LLM-translated program still suffers from diverse quality issues (e.g., syntax errors and semantic errors). In particular, it can be challenging for LLMs to self-debug these errors when simply provided with the corresponding error messages. In this work, we propose a novel LLM-based multi-agent system TRANSAGENT, which enhances LLM-based code translation by fixing the syntax errors and semantic errors with the synergy between four LLM-based agents, including Initial Code Translator, Syntax Error Fixer, Code Aligner, and Semantic Error Fixer. The main insight of TRANSAGENT is to first localize the error code block in the target program based on the execution alignment between the target and source program, which can narrow down the fixing space and thus lower down the fixing difficulties. To evaluate TRANSAGENT, we first construct a new benchmark from recent programming tasks to mitigate the potential data leakage issue. On our benchmark, TRANSAGENT outperforms the latest LLM-based code translation technique UniTrans in both translation effectiveness and efficiency; additionally, our evaluation on different LLMs show the generalization of TRANSAGENT and our ablation study shows the contribution of each agent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12067v3">TokenSkip: Controllable Chain-of-Thought Compression in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 EMNLP 2025 (Long Paper), camera-ready version
    </div>
    <details class="paper-abstract">
      Chain-of-Thought (CoT) has been proven effective in enhancing the reasoning capabilities of large language models (LLMs). Recent advancements, such as OpenAI's o1 and DeepSeek-R1, suggest that scaling up the length of CoT sequences during inference could further boost LLM reasoning performance. However, due to the autoregressive nature of LLM decoding, longer CoT outputs lead to a linear increase in inference latency, adversely affecting user experience, particularly when the CoT exceeds 10,000 tokens. To address this limitation, we analyze the semantic importance of tokens within CoT outputs and reveal that their contributions to reasoning vary. Building on this insight, we propose TokenSkip, a simple yet effective approach that enables LLMs to selectively skip less important tokens, allowing for controllable CoT compression. Extensive experiments across various models and tasks demonstrate the effectiveness of TokenSkip in reducing CoT token usage while preserving strong reasoning performance. Notably, when applied to Qwen2.5-14B-Instruct, TokenSkip reduces reasoning tokens by 40% (from 313 to 181) on GSM8K, with less than a 0.4% performance drop. We release our code and checkpoints in https://github.com/hemingkx/TokenSkip.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16408v2">From Surveys to Narratives: Rethinking Cultural Value Adaptation in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      Adapting cultural values in Large Language Models (LLMs) presents significant challenges, particularly due to biases and limited training data. Prior work primarily aligns LLMs with different cultural values using World Values Survey (WVS) data. However, it remains unclear whether this approach effectively captures cultural nuances or produces distinct cultural representations for various downstream tasks. In this paper, we systematically investigate WVS-based training for cultural value adaptation and find that relying solely on survey data can homogenize cultural norms and interfere with factual knowledge. To investigate these issues, we augment WVS with encyclopedic and scenario-based cultural narratives from Wikipedia and NormAd. While these narratives may have variable effects on downstream tasks, they consistently improve cultural distinctiveness than survey data alone. Our work highlights the inherent complexity of aligning cultural values with the goal of guiding task-specific behavior. We release our code at https://github.com/faridlazuarda/from-surveys-to-narratives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12769v3">How Much Do LLMs Hallucinate across Languages? On Multilingual Estimation of LLM Hallucination in the Wild</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 EMNLP 2025
    </div>
    <details class="paper-abstract">
      In the age of misinformation, hallucination -- the tendency of Large Language Models (LLMs) to generate non-factual or unfaithful responses -- represents the main risk for their global utility. Despite LLMs becoming increasingly multilingual, the vast majority of research on detecting and quantifying LLM hallucination are (a) English-centric and (b) focus on machine translation (MT) and summarization, tasks that are less common ``in the wild'' than open information seeking. In contrast, we aim to quantify the extent of LLM hallucination across languages in knowledge-intensive long-form question answering. To this end, we train a multilingual hallucination detection model and conduct a large-scale study across 30 languages and 6 open-source LLM families. We start from an English hallucination detection dataset and rely on MT to generate (noisy) training data in other languages. We also manually annotate gold data for five high-resource languages; we then demonstrate, for these languages, that the estimates of hallucination rates are similar between silver (LLM-generated) and gold test sets, validating the use of silver data for estimating hallucination rates for other languages. For the final rates estimation, we build a knowledge-intensive QA dataset for 30 languages with LLM-generated prompts and Wikipedia articles as references. We find that, while LLMs generate longer responses with more hallucinated tokens for higher-resource languages, there is no correlation between length-normalized hallucination rates of languages and their digital representation. Further, we find that smaller LLMs exhibit larger hallucination rates than larger models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12961v1">Do LLMs Understand Wine Descriptors Across Cultures? A Benchmark for Cultural Adaptations of Wine Reviews</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have opened the door to culture-aware language tasks. We introduce the novel problem of adapting wine reviews across Chinese and English, which goes beyond literal translation by incorporating regional taste preferences and culture-specific flavor descriptors. In a case study on cross-cultural wine review adaptation, we compile the first parallel corpus of professional reviews, containing 8k Chinese and 16k Anglophone reviews. We benchmark both neural-machine-translation baselines and state-of-the-art LLMs with automatic metrics and human evaluation. For the latter, we propose three culture-oriented criteria -- Cultural Proximity, Cultural Neutrality, and Cultural Genuineness -- to assess how naturally a translated review resonates with target-culture readers. Our analysis shows that current models struggle to capture cultural nuances, especially in translating wine descriptions across different cultures. This highlights the challenges and limitations of translation models in handling cultural content.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08523v3">InferLog: Accelerating LLM Inference for Online Log Parsing via ICL-oriented Prefix Caching</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 Accepted by ICSE '26 (The 48th IEEE/ACM International Conference on Software Engineering)
    </div>
    <details class="paper-abstract">
      Modern software systems generate massive volumes of runtime logs, necessitating efficient and accurate log parsing to enable critical downstream tasks such as anomaly detection and root cause analysis. Recently, large language models (LLMs) have achieved advanced accuracy on log parsing, but their deployment in production environments faces two major limitations: (1) the privacy risks associated with commercial LLMs, driving the adoption of local deployment, and (2) the stringent latency and throughput requirements imposed by high-volume log streams, which existing LLM-based parsers fail to meet. Although recent efforts have reduced the number of LLM queries, they overlook the high latency of the LLM invocations, where concurrent log parsing requests can cause serve performance degradation of LLM inference system. In this study, we present InferLog, the first LLM inference optimization method for online log parsing. Our key insight is that the inference efficiency emerges as the vital bottleneck in LLM-based online log parsing, rather than parsing accuracy. InferLog accelerates inference by designing (1) A Prefix-aware ICL Refinement policy to refine the examples and permutation of in-context learning to improve the prefix caching efficiency. (2) A rapid and task-specific configuration tuning pipeline based on meta-learning to find the optimal LLM scheduling-related configuration for dynamic log parsing workloads. The experimental results based on Loghub dataset and vLLM demonstrate that InferLog significantly outperforms existing inference optimization methods and markedly accelerates the state-of-the-art LLM-based log parser without compromising parsing accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12914v1">Stochastic Streets: A Walk Through Random LLM Address Generation in four European Cities</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are capable of solving complex math problems or answer difficult questions on almost any topic, but can they generate random street addresses for European cities?
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12892v1">Conan-Embedding-v2: Training an LLM from Scratch for Text Embeddings</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 EMNLP 2025 Oral
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently demonstrated excellent performance in text embedding tasks. Previous work usually use LoRA to fine-tune existing LLMs, which are limited by the data and training gap between LLMs and embedding models. In this work, we introduce Conan-embedding-v2, a new 1.4B-parameter LLM trained from scratch and fine-tuned as a text embedder. First, we add news data and multilingual pairs for LLM pretraining to bridge the data gap. Based on this, we propose a cross-lingual retrieval dataset that enables the LLM to better integrate embeddings across different languages. Second, whereas LLMs use a causal mask with token-level loss, embedding models use a bidirectional mask with sentence-level loss. This training gap makes full fine-tuning less effective than LoRA. We introduce a soft-masking mechanism to gradually transition between these two types of masks, enabling the model to learn more comprehensive representations. Based on this, we propose a dynamic hard negative mining method that exposes the model to more difficult negative examples throughout the training process. Being intuitive and effective, with only approximately 1.4B parameters, Conan-embedding-v2 achieves SOTA performance on both the Massive Text Embedding Benchmark (MTEB) and Chinese MTEB (May 19, 2025).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12886v1">The LLM Already Knows: Estimating LLM-Perceived Question Difficulty via Hidden Representations</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      Estimating the difficulty of input questions as perceived by large language models (LLMs) is essential for accurate performance evaluation and adaptive inference. Existing methods typically rely on repeated response sampling, auxiliary models, or fine-tuning the target model itself, which may incur substantial computational costs or compromise generality. In this paper, we propose a novel approach for difficulty estimation that leverages only the hidden representations produced by the target LLM. We model the token-level generation process as a Markov chain and define a value function to estimate the expected output quality given any hidden state. This allows for efficient and accurate difficulty estimation based solely on the initial hidden state, without generating any output tokens. Extensive experiments across both textual and multimodal tasks demonstrate that our method consistently outperforms existing baselines in difficulty estimation. Moreover, we apply our difficulty estimates to guide adaptive reasoning strategies, including Self-Consistency, Best-of-N, and Self-Refine, achieving higher inference efficiency with fewer generated tokens.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12107v1">Exploring Conversational Design Choices in LLMs for Pedagogical Purposes: Socratic and Narrative Approaches for Improving Instructor's Teaching Practice</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) typically generate direct answers, yet they are increasingly used as learning tools. Studying instructors' usage is critical, given their role in teaching and guiding AI adoption in education. We designed and evaluated TeaPT, an LLM for pedagogical purposes that supports instructors' professional development through two conversational approaches: a Socratic approach that uses guided questioning to foster reflection, and a Narrative approach that offers elaborated suggestions to extend externalized cognition. In a mixed-method study with 41 higher-education instructors, the Socratic version elicited greater engagement, while the Narrative version was preferred for actionable guidance. Subgroup analyses further revealed that less-experienced, AI-optimistic instructors favored the Socratic version, whereas more-experienced, AI-cautious instructors preferred the Narrative version. We contribute design implications for LLMs for pedagogical purposes, showing how adaptive conversational approaches can support instructors with varied profiles while highlighting how AI attitudes and experience shape interaction and learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12104v1">JustEva: A Toolkit to Evaluate LLM Fairness in Legal Knowledge Inference</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 This paper has been accepted at CIKM 2025 (Demo Track)
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into legal practice raises pressing concerns about judicial fairness, particularly due to the nature of their "black-box" processes. This study introduces JustEva, a comprehensive, open-source evaluation toolkit designed to measure LLM fairness in legal tasks. JustEva features several advantages: (1) a structured label system covering 65 extra-legal factors; (2) three core fairness metrics - inconsistency, bias, and imbalanced inaccuracy; (3) robust statistical inference methods; and (4) informative visualizations. The toolkit supports two types of experiments, enabling a complete evaluation workflow: (1) generating structured outputs from LLMs using a provided dataset, and (2) conducting statistical analysis and inference on LLMs' outputs through regression and other statistical methods. Empirical application of JustEva reveals significant fairness deficiencies in current LLMs, highlighting the lack of fair and trustworthy LLM legal tools. JustEva offers a convenient tool and methodological foundation for evaluating and improving algorithmic fairness in the legal domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17711v2">Enhancing LLM-Based Social Bot via an Adversarial Learning Framework</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      Developing Large Language Model (LLM) agents that exhibit human-like behavior, encompassing not only individual heterogeneity rooted in unique user profiles but also adaptive response to socially connected neighbors, is a significant research challenge. Social media platforms, with their diverse user data and explicit social structures, provide an ideal testbed for such investigations. This paper introduces EvoBot, an \textbf{Evo}lving LLM-based social \textbf{Bot} that significantly enhances human-like generative capabilities through a novel adversarial learning framework. EvoBot is initialized by Supervised Fine-Tuning (SFT) on representative data from social media and then iteratively refines its generation of sophisticated, human-like content via Direct Preference Optimization (DPO). This refinement is guided by feedback from a co-adapting \textbf{Detector} which concurrently improves its ability to distinguish EvoBot from humans, thereby creating an increasingly challenging learning environment for EvoBot. Experiments demonstrate that EvoBot generates content aligned with diverse user profiles, increasingly bypassing the co-adapting Detector through human-like expression. Moreover, it exhibits strong social responsiveness, more accurately modeling real-world opinion dynamics and information spread in multi-agent simulations. The framework also yields a more robust Detector, underscoring its broader utility for both advanced agent development and related detection tasks. The code is available at https://github.com/kfq20/EvoBot.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12102v1">Can LLMs Address Mental Health Questions? A Comparison with Human Therapists</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      Limited access to mental health care has motivated the use of digital tools and conversational agents powered by large language models (LLMs), yet their quality and reception remain unclear. We present a study comparing therapist-written responses to those generated by ChatGPT, Gemini, and Llama for real patient questions. Text analysis showed that LLMs produced longer, more readable, and lexically richer responses with a more positive tone, while therapist responses were more often written in the first person. In a survey with 150 users and 23 licensed therapists, participants rated LLM responses as clearer, more respectful, and more supportive than therapist-written answers. Yet, both groups of participants expressed a stronger preference for human therapist support. These findings highlight the promise and limitations of LLMs in mental health, underscoring the need for designs that balance their communicative strengths with concerns of trust, privacy, and accountability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12021v1">LitterBox+: An Extensible Framework for LLM-enhanced Scratch Static Code Analysis</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 ASE 2025 Tool Demonstration Track
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have become an essential tool to support developers using traditional text-based programming languages, but the graphical notation of the block-based Scratch programming environment inhibits the use of LLMs. To overcome this limitation, we propose the LitterBox+ framework that extends the Scratch static code analysis tool LitterBox with the generative abilities of LLMs. By converting block-based code to a textual representation suitable for LLMs, LitterBox+ allows users to query LLMs about their programs, about quality issues reported by LitterBox, and it allows generating code fixes. Besides offering a programmatic API for these functionalities, LitterBox+ also extends the Scratch user interface to make these functionalities available directly in the environment familiar to learners. The framework is designed to be easily extensible with other prompts, LLM providers, and new features combining the program analysis capabilities of LitterBox with the generative features of LLMs. We provide a screencast demonstrating the tool at https://youtu.be/RZ6E0xgrIgQ.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.13231v2">Accelerating LLM Inference via Dynamic KV Cache Placement in Heterogeneous Memory System</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 IEEE Computer Architecture Letter
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) inference is increasingly constrained by memory bandwidth, with frequent access to the key-value (KV) cache dominating data movement. While attention sparsity reduces some memory traffic, the relevance of past tokens varies over time, requiring the full KV cache to remain accessible and sustaining pressure on both bandwidth and capacity. With advances in interconnects such as NVLink and LPDDR5X, modern AI hardware now integrates high-bandwidth memory (HBM) with high-speed off-package DRAM, making heterogeneous memory systems a practical solution. This work investigates dynamic KV cache placement across such systems to maximize aggregated bandwidth utilization under capacity constraints. Rather than proposing a specific scheduling policy, we formulate the placement problem mathematically and derive a theoretical upper bound, revealing substantial headroom for runtime optimization. To our knowledge, this is the first formal treatment of dynamic KV cache scheduling in heterogeneous memory systems for LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14403v4">Unearthing Gems from Stones: Policy Optimization with Negative Sample Augmentation for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      Recent advances in reasoning language models have witnessed a paradigm shift from short to long CoT pattern. Given the substantial computational cost of rollouts in long CoT models, maximizing the utility of fixed training datasets becomes crucial. Our analysis reveals that negative responses contain valuable components such as self-reflection and error-correction steps, yet primary existing methods either completely discard negative samples (RFT) or apply equal penalization across all tokens (RL), failing to leverage these potential learning signals. In light of this, we propose Behavior Constrained Policy Gradient with Negative Sample Augmentation (BCPG-NSA), a fine-grained offline RL framework that encompasses three stages: 1) sample segmentation, 2) consensus-based step correctness assessment combining LLM and PRM judgers, and 3) policy optimization with NSA designed to effectively mine positive steps within negative samples. Experimental results show that BCPG-NSA outperforms baselines on several challenging math/coding reasoning benchmarks using the same training dataset, achieving improved sample efficiency and demonstrating robustness and scalability when extended to multiple iterations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11967v1">MillStone: How Open-Minded Are LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 19 pages, 7 tables, 7 figures
    </div>
    <details class="paper-abstract">
      Large language models equipped with Web search, information retrieval tools, and other agentic capabilities are beginning to supplant traditional search engines. As users start to rely on LLMs for information on many topics, including controversial and debatable issues, it is important to understand how the stances and opinions expressed in LLM outputs are influenced by the documents they use as their information sources. In this paper, we present MillStone, the first benchmark that aims to systematically measure the effect of external arguments on the stances that LLMs take on controversial issues (not all of them political). We apply MillStone to nine leading LLMs and measure how ``open-minded'' they are to arguments supporting opposite sides of these issues, whether different LLMs agree with each other, which arguments LLMs find most persuasive, and whether these arguments are the same for different LLMs. In general, we find that LLMs are open-minded on most issues. An authoritative source of information can easily sway an LLM's stance, highlighting the importance of source selection and the risk that LLM-based information retrieval and search systems can be manipulated.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11921v1">Designing LLMs for cultural sensitivity: Evidence from English-Japanese translation</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used in everyday communication, including multilingual interactions across different cultural contexts. While LLMs can now generate near-perfect literal translations, it remains unclear whether LLMs support culturally appropriate communication. In this paper, we analyze the cultural sensitivity of different LLM designs when applied to English-Japanese translations of workplace e-mails. Here, we vary the prompting strategies: (1) naive "just translate" prompts, (2) audience-targeted prompts specifying the recipient's cultural background, and (3) instructional prompts with explicit guidance on Japanese communication norms. Using a mixed-methods study, we then analyze culture-specific language patterns to evaluate how well translations adapt to cultural norms. Further, we examine the appropriateness of the tone of the translations as perceived by native speakers. We find that culturally-tailored prompting can improve cultural fit, based on which we offer recommendations for designing culturally inclusive LLMs in multilingual settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05755v2">Exploit Tool Invocation Prompt for Tool Behavior Hijacking in LLM-Based Agentic System</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      LLM-based agentic systems leverage large language models to handle user queries, make decisions, and execute external tools for complex tasks across domains like chatbots, customer service, and software engineering. A critical component of these systems is the Tool Invocation Prompt (TIP), which defines tool interaction protocols and guides LLMs to ensure the security and correctness of tool usage. Despite its importance, TIP security has been largely overlooked. This work investigates TIP-related security risks, revealing that major LLM-based systems like Cursor, Claude Code, and others are vulnerable to attacks such as remote code execution (RCE) and denial of service (DoS). Through a systematic TIP exploitation workflow (TEW), we demonstrate external tool behavior hijacking via manipulated tool invocations. We also propose defense mechanisms to enhance TIP security in LLM-based agentic systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04427v2">Plugging Schema Graph into Multi-Table QA: A Human-Guided Framework for Reducing LLM Reliance</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 Accepted to EMNLP 2025 findings
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown promise in table Question Answering (Table QA). However, extending these capabilities to multi-table QA remains challenging due to unreliable schema linking across complex tables. Existing methods based on semantic similarity work well only on simplified hand-crafted datasets and struggle to handle complex, real-world scenarios with numerous and diverse columns. To address this, we propose a graph-based framework that leverages human-curated relational knowledge to explicitly encode schema links and join paths. Given a natural language query, our method searches on graph to construct interpretable reasoning chains, aided by pruning and sub-path merging strategies to enhance efficiency and coherence. Experiments on both standard benchmarks and a realistic, large-scale dataset demonstrate the effectiveness of our approach. To our knowledge, this is the first multi-table QA system applied to truly complex industrial tabular data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11864v1">NeuroStrike: Neuron-Level Attacks on Aligned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      Safety alignment is critical for the ethical deployment of large language models (LLMs), guiding them to avoid generating harmful or unethical content. Current alignment techniques, such as supervised fine-tuning and reinforcement learning from human feedback, remain fragile and can be bypassed by carefully crafted adversarial prompts. Unfortunately, such attacks rely on trial and error, lack generalizability across models, and are constrained by scalability and reliability. This paper presents NeuroStrike, a novel and generalizable attack framework that exploits a fundamental vulnerability introduced by alignment techniques: the reliance on sparse, specialized safety neurons responsible for detecting and suppressing harmful inputs. We apply NeuroStrike to both white-box and black-box settings: In the white-box setting, NeuroStrike identifies safety neurons through feedforward activation analysis and prunes them during inference to disable safety mechanisms. In the black-box setting, we propose the first LLM profiling attack, which leverages safety neuron transferability by training adversarial prompt generators on open-weight surrogate models and then deploying them against black-box and proprietary targets. We evaluate NeuroStrike on over 20 open-weight LLMs from major LLM developers. By removing less than 0.6% of neurons in targeted layers, NeuroStrike achieves an average attack success rate (ASR) of 76.9% using only vanilla malicious prompts. Moreover, Neurostrike generalizes to four multimodal LLMs with 100% ASR on unsafe image inputs. Safety neurons transfer effectively across architectures, raising ASR to 78.5% on 11 fine-tuned models and 77.7% on five distilled models. The black-box LLM profiling attack achieves an average ASR of 63.7% across five black-box models, including the Google Gemini family.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11803v1">From Fuzzy Speech to Medical Insight: Benchmarking LLMs on Noisy Patient Narratives</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 6 pages, 1 figure
    </div>
    <details class="paper-abstract">
      The widespread adoption of large language models (LLMs) in healthcare raises critical questions about their ability to interpret patient-generated narratives, which are often informal, ambiguous, and noisy. Existing benchmarks typically rely on clean, structured clinical text, offering limited insight into model performance under realistic conditions. In this work, we present a novel synthetic dataset designed to simulate patient self-descriptions characterized by varying levels of linguistic noise, fuzzy language, and layperson terminology. Our dataset comprises clinically consistent scenarios annotated with ground-truth diagnoses, spanning a spectrum of communication clarity to reflect diverse real-world reporting styles. Using this benchmark, we fine-tune and evaluate several state-of-the-art models (LLMs), including BERT-based and encoder-decoder T5 models. To support reproducibility and future research, we release the Noisy Diagnostic Benchmark (NDB), a structured dataset of noisy, synthetic patient descriptions designed to stress-test and compare the diagnostic capabilities of large language models (LLMs) under realistic linguistic conditions. We made the benchmark available for the community: https://github.com/lielsheri/PatientSignal
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.18337v5">Can LLMs assist with Ambiguity? A Quantitative Evaluation of various Large Language Models on Word Sense Disambiguation</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 12 pages,6 tables, 1 figure, Proceedings of the 1st International Conference on NLP & AI for Cyber Security
    </div>
    <details class="paper-abstract">
      Ambiguous words are often found in modern digital communications. Lexical ambiguity challenges traditional Word Sense Disambiguation (WSD) methods, due to limited data. Consequently, the efficiency of translation, information retrieval, and question-answering systems is hindered by these limitations. This study investigates the use of Large Language Models (LLMs) to improve WSD using a novel approach combining a systematic prompt augmentation mechanism with a knowledge base (KB) consisting of different sense interpretations. The proposed method incorporates a human-in-loop approach for prompt augmentation where prompt is supported by Part-of-Speech (POS) tagging, synonyms of ambiguous words, aspect-based sense filtering and few-shot prompting to guide the LLM. By utilizing a few-shot Chain of Thought (COT) prompting-based approach, this work demonstrates a substantial improvement in performance. The evaluation was conducted using FEWS test data and sense tags. This research advances accurate word interpretation in social media and digital communication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00831v5">SmallPlan: Leverage Small Language Models for Sequential Path Planning with Simulation-Powered, LLM-Guided Distillation</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 Paper is under review
    </div>
    <details class="paper-abstract">
      Efficient path planning in robotics, particularly within large-scale, complex environments, remains a significant hurdle. While Large Language Models (LLMs) offer strong reasoning capabilities, their high computational cost and limited adaptability hinder real-time deployment on edge devices. We present SmallPlan - a novel framework leveraging LLMs as teacher models to train lightweight Small Language Models (SLMs) for high-level path planning tasks. In SmallPlan, the SLMs provide optimal action sequences to navigate across scene graphs that compactly represent full-scaled 3D scenes. The SLMs are trained in a simulation-powered, interleaved manner with LLM-guided supervised fine-tuning (SFT) and reinforcement learning (RL). This strategy not only enables SLMs to successfully complete navigation tasks but also makes them aware of important factors like distance travel, providing more efficient path planning. Through experiments, we demonstrate that the fine-tuned SLMs perform competitively with larger models like GPT-4o on sequential path planning, without suffering from hallucination and overfitting. SmallPlan is resource-efficient, making it well-suited for edge-device deployment and advancing practical autonomous robotics. Our source code is available here: https://github.com/quangpham2006/SmallPlan
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20258v2">LLM as a Broken Telephone: Iterative Generation Distorts Information</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 Accepted to ACL 2025, Main Conference
    </div>
    <details class="paper-abstract">
      As large language models are increasingly responsible for online content, concerns arise about the impact of repeatedly processing their own outputs. Inspired by the "broken telephone" effect in chained human communication, this study investigates whether LLMs similarly distort information through iterative generation. Through translation-based experiments, we find that distortion accumulates over time, influenced by language choice and chain complexity. While degradation is inevitable, it can be mitigated through strategic prompting techniques. These findings contribute to discussions on the long-term effects of AI-mediated information propagation, raising important questions about the reliability of LLM-generated content in iterative workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.08575v2">SQLGovernor: An LLM-powered SQL Toolkit for Real World Application</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      SQL queries in real world analytical environments, whether written by humans or generated automatically often suffer from syntax errors, inefficiency, or semantic misalignment, especially in complex OLAP scenarios. To address these challenges, we propose SQLGovernor, an LLM powered SQL toolkit that unifies multiple functionalities, including syntax correction, query rewriting, query modification, and consistency verification within a structured framework enhanced by knowledge management. SQLGovernor introduces a fragment wise processing strategy to enable fine grained rewriting and localized error correction, significantly reducing the cognitive load on the LLM. It further incorporates a hybrid self learning mechanism guided by expert feedback, allowing the system to continuously improve through DBMS output analysis and rule validation. Experiments on benchmarks such as BIRD and BIRD CRITIC, as well as industrial datasets, show that SQLGovernor consistently boosts the performance of base models by up to 10%, while minimizing reliance on manual expertise. Deployed in production environments, SQLGovernor demonstrates strong practical utility and effective performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20742v2">'Hello, World!': Making GNNs Talk with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 Published as a conference paper at EMNLP 2025 Findings. Code and datasets are in https://github.com/kswoo97/GLN-Code
    </div>
    <details class="paper-abstract">
      While graph neural networks (GNNs) have shown remarkable performance across diverse graph-related tasks, their high-dimensional hidden representations render them black boxes. In this work, we propose Graph Lingual Network (GLN), a GNN built on large language models (LLMs), with hidden representations in the form of human-readable text. Through careful prompt design, GLN incorporates not only the message passing module of GNNs but also advanced GNN techniques, including graph attention and initial residual connection. The comprehensibility of GLN's hidden representations enables an intuitive analysis of how node representations change (1) across layers and (2) under advanced GNN techniques, shedding light on the inner workings of GNNs. Furthermore, we demonstrate that GLN achieves strong zero-shot performance on node classification and link prediction, outperforming existing LLM-based baseline methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.12172v3">Navigating the Labyrinth: Evaluating LLMs' Ability to Reason About Search Problems</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently achieved impressive performance in math and reasoning benchmarks. However, they often struggle with logic problems and puzzles that are relatively easy for humans. To further investigate this, we introduce a new benchmark, SearchBench, which contains 11 unique search problems inspired by intuitive puzzles. Each SearchBench problem type is equipped with automated pipelines to generate an arbitrary number of instances and analyze the feasibility, correctness, and optimality of LLM-generated solutions. We show that using step-by-step, language-only reasoning, even the most advanced LLMs fail to solve SearchBench; for example, OpenAI's frontier models GPT-4 and o1-preview solve only 1.4% and 18.6% of problems, respectively. The reason is that SearchBench problems require considering multiple pathways and performing backtracking, posing a significant challenge to auto-regressive models. Interestingly, performance is significantly boosted when we prompt models to generate a complete A* search algorithm - a comparatively more cognitively difficult task. This approach effectively offloads the iterative search and backtracking process from the models, which they struggle with in text. This in-context learning baseline is further enhanced via a Multi-Stage-Multi-Try (MSMT) inference method, increasing GPT-4's rate of correct solutions to over 57%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11617v1">AssemMate: Graph-Based LLM for Robotic Assembly Assistance</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based robotic assembly assistance has gained significant research attention. It requires the injection of domain-specific knowledge to guide the assembly process through natural language interaction with humans. Despite some progress, existing methods represent knowledge in the form of natural language text. Due to the long context and redundant content, they struggle to meet the robots' requirements for real-time and precise reasoning. In order to bridge this gap, we present AssemMate, which utilizes the graph\textemdash a concise and accurate form of knowledge representation\textemdash as input. This graph-based LLM enables knowledge graph question answering (KGQA), supporting human-robot interaction and assembly task planning for specific products. Beyond interactive QA, AssemMate also supports sensing stacked scenes and executing grasping to assist with assembly. Specifically, a self-supervised Graph Convolutional Network (GCN) encodes knowledge graph entities and relations into a latent space and aligns them with LLM's representation, enabling the LLM to understand graph information. In addition, a vision-enhanced strategy is employed to address stacked scenes in grasping. Through training and evaluation, AssemMate outperforms existing methods, achieving 6.4\% higher accuracy, 3 times faster inference, and 28 times shorter context length, while demonstrating strong generalization ability on random graphs. And our approach further demonstrates superiority through robotic grasping experiments in both simulated and real-world settings. More details can be found on the project page: https://github.com/cristina304/AssemMate.git
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03054v2">Binary Quantization For LLMs Through Dynamic Grouping</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 An error was identified in the quantization bit width; it is not binary
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable performance across a wide range of Natural Language Processing (NLP) tasks, but require substantial memory and computational resources. Binary quantization, which compresses model weights from 16-bit Brain Float to 1-bit representations in {-1, 1}, offers significant reductions in storage and inference costs. However, such aggressive quantization often leads to notable performance degradation compared to more conservative 4-bit quantization methods. In this research, we propose a novel optimization objective tailored for binary quantization, along with three algorithms designed to realize it effectively. Our method enhances blocked quantization by dynamically identifying optimal unstructured sub-matrices through adaptive grouping strategies. Experimental results demonstrate that our approach achieves an average bit length of just 1.007 bits, while maintaining high model quality. Specifically, our quantized LLaMA 3.2 3B model attains a perplexity of 8.23, remarkably close to the original 7.81, and surpasses previous SOTA BiLLM with a perplexity of only 123.90. Furthermore, our method is competitive with SOTA 4-bit approaches such as GPTQ in both performance and efficiency. The compression process is highly efficient, requiring only 14 seconds to quantize the full LLaMA 3.2 3B weights on a single CPU core, with the entire process completing in under 100 minutes and exhibiting embarrassingly parallel properties. Code - https://github.com/johnnyzheng0636/WGM_bi_quan
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11569v1">D$^2$HScore: Reasoning-Aware Hallucination Detection via Semantic Breadth and Depth Analysis in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 under review
    </div>
    <details class="paper-abstract">
      Although large Language Models (LLMs) have achieved remarkable success, their practical application is often hindered by the generation of non-factual content, which is called "hallucination". Ensuring the reliability of LLMs' outputs is a critical challenge, particularly in high-stakes domains such as finance, security, and healthcare. In this work, we revisit hallucination detection from the perspective of model architecture and generation dynamics. Leveraging the multi-layer structure and autoregressive decoding process of LLMs, we decompose hallucination signals into two complementary dimensions: the semantic breadth of token representations within each layer, and the semantic depth of core concepts as they evolve across layers. Based on this insight, we propose \textbf{D$^2$HScore (Dispersion and Drift-based Hallucination Score)}, a training-free and label-free framework that jointly measures: (1) \textbf{Intra-Layer Dispersion}, which quantifies the semantic diversity of token representations within each layer; and (2) \textbf{Inter-Layer Drift}, which tracks the progressive transformation of key token representations across layers. To ensure drift reflects the evolution of meaningful semantics rather than noisy or redundant tokens, we guide token selection using attention signals. By capturing both the horizontal and vertical dynamics of representation during inference, D$^2$HScore provides an interpretable and lightweight proxy for hallucination detection. Extensive experiments across five open-source LLMs and five widely used benchmarks demonstrate that D$^2$HScore consistently outperforms existing training-free baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10248v2">Prompt Injection Attacks on LLM Generated Reviews of Scientific Publications</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      The ongoing intense discussion on rising LLM usage in the scientific peer-review process has recently been mingled by reports of authors using hidden prompt injections to manipulate review scores. Since the existence of such "attacks" - although seen by some commentators as "self-defense" - would have a great impact on the further debate, this paper investigates the practicability and technical success of the described manipulations. Our systematic evaluation uses 1k reviews of 2024 ICLR papers generated by a wide range of LLMs shows two distinct results: I) very simple prompt injections are indeed highly effective, reaching up to 100% acceptance scores. II) LLM reviews are generally biased toward acceptance (>95% in many models). Both results have great impact on the ongoing discussions on LLM usage in peer-review.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16347v2">Confusion is the Final Barrier: Rethinking Jailbreak Evaluation and Investigating the Real Misuse Threat of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      With the development of Large Language Models (LLMs), numerous efforts have revealed their vulnerabilities to jailbreak attacks. Although these studies have driven the progress in LLMs' safety alignment, it remains unclear whether LLMs have internalized authentic knowledge to deal with real-world crimes, or are merely forced to simulate toxic language patterns. This ambiguity raises concerns that jailbreak success is often attributable to a hallucination loop between jailbroken LLM and judger LLM. By decoupling the use of jailbreak techniques, we construct knowledge-intensive Q\&A to investigate the misuse threats of LLMs in terms of dangerous knowledge possession, harmful task planning utility, and harmfulness judgment robustness. Experiments reveal a mismatch between jailbreak success rates and harmful knowledge possession in LLMs, and existing LLM-as-a-judge frameworks tend to anchor harmfulness judgments on toxic language patterns. Our study reveals a gap between existing LLM safety assessments and real-world threat potential.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00063v2">MolErr2Fix: Benchmarking LLM Trustworthiness in Chemistry via Modular Error Detection, Localization, Explanation, and Revision</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown growing potential in molecular sciences, but they often produce chemically inaccurate descriptions and struggle to recognize or justify potential errors. This raises important concerns about their robustness and reliability in scientific applications. To support more rigorous evaluation of LLMs in chemical reasoning, we present the MolErr2Fix benchmark, designed to assess LLMs on error detection and correction in molecular descriptions. Unlike existing benchmarks focused on molecule-to-text generation or property prediction, MolErr2Fix emphasizes fine-grained chemical understanding. It tasks LLMs with identifying, localizing, explaining, and revising potential structural and semantic errors in molecular descriptions. Specifically, MolErr2Fix consists of 1,193 fine-grained annotated error instances. Each instance contains quadruple annotations, i.e,. (error type, span location, the explanation, and the correction). These tasks are intended to reflect the types of reasoning and verification required in real-world chemical communication. Evaluations of current state-of-the-art LLMs reveal notable performance gaps, underscoring the need for more robust chemical reasoning capabilities. MolErr2Fix provides a focused benchmark for evaluating such capabilities and aims to support progress toward more reliable and chemically informed language models. All annotations and an accompanying evaluation API will be publicly released to facilitate future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14827v3">Enhancing Prompt Injection Attacks to LLMs via Poisoning Alignment</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      Prompt injection attack, where an attacker injects a prompt into the original one, aiming to make an Large Language Model (LLM) follow the injected prompt to perform an attacker-chosen task, represent a critical security threat. Existing attacks primarily focus on crafting these injections at inference time, treating the LLM itself as a static target. Our experiments show that these attacks achieve some success, but there is still significant room for improvement. In this work, we introduces a more foundational attack vector: poisoning the LLM's alignment process to amplify the success of future prompt injection attacks. Specifically, we propose PoisonedAlign, a method that strategically creates poisoned alignment samples to poison an LLM's alignment dataset. Our experiments across five LLMs and two alignment datasets show that when even a small fraction of the alignment data is poisoned, the resulting model becomes substantially more vulnerable to a wide range of prompt injection attacks. Crucially, this vulnerability is instilled while the LLM's performance on standard capability benchmarks remains largely unchanged, making the manipulation difficult to detect through automated, general-purpose performance evaluations. The code for implementing the attack is available at https://github.com/Sadcardation/PoisonedAlign.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14970v3">Self-Evolving Curriculum for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has proven effective for fine-tuning large language models (LLMs), significantly enhancing their reasoning abilities in domains such as mathematics and code generation. A crucial factor influencing RL fine-tuning success is the training curriculum: the order in which training problems are presented. While random curricula serve as common baselines, they remain suboptimal; manually designed curricula often rely heavily on heuristics, and online filtering methods can be computationally prohibitive. To address these limitations, we propose Self-Evolving Curriculum (SEC), an automatic curriculum learning method that learns a curriculum policy concurrently with the RL fine-tuning process. Our approach formulates curriculum selection as a non-stationary Multi-Armed Bandit problem, treating each problem category (e.g., difficulty level or problem type) as an individual arm. We leverage the absolute advantage from policy gradient methods as a proxy measure for immediate learning gain. At each training step, the curriculum policy selects categories to maximize this reward signal and is updated using the TD(0) method. Across three distinct reasoning domains: planning, inductive reasoning, and mathematics, our experiments demonstrate that SEC significantly improves models' reasoning capabilities, enabling better generalization to harder, out-of-distribution test problems. Additionally, our approach achieves better skill balance when fine-tuning simultaneously on multiple reasoning domains. These findings highlight SEC as a promising strategy for RL fine-tuning of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11524v1">Decoding in Latent Spaces for Efficient Inference in LLM-based Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 Accepted for publication in EMNLP'25
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) for recommendation in a generative manner has delivered promising results, but encounters significant inference overhead due to autoregressive decoding in the language space. This work explores bypassing language-space decoding by directly matching candidate items with the LLM's internal thought representations in the latent space, eliminating the time-consuming autoregressive process to reduce computational costs. Towards this, we introduce Light Latent-space Decoding (L2D), an effective and efficient latent-space decoding method. L2D represents user-preferred items by using the hidden states of test sequences reflecting the LLM's internal thought, and obtains candidate item representations from the hidden states of training sequences labeled with the corresponding candidate items. It then matches the two types of representations to decode items, achieving latent-space decoding. In this way, it enables efficient decoding without altering the LLM's generative tuning paradigm, thereby preserving performance. Extensive empirical results demonstrate that L2D is more than 10x faster than language-space decoding while maintaining or enhancing performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11517v1">PeruMedQA: Benchmarking Large Language Models (LLMs) on Peruvian Medical Exams -- Dataset Construction and Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 https://github.com/rodrigo-carrillo/PeruMedQA
    </div>
    <details class="paper-abstract">
      BACKGROUND: Medical large language models (LLMS) have demonstrated remarkable performance in answering medical examinations. However, the extent to which this high performance is transferable to medical questions in Spanish and from a Latin American country remains unexplored. This knowledge is crucial as LLM-based medical applications gain traction in Latin America. AIMS: to build a dataset of questions from medical examinations taken by Peruvian physicians pursuing specialty training; to fine-tune a LLM on this dataset; to evaluate and compare the performance in terms of accuracy between vanilla LLMs and the fine-tuned LLM. METHODS: We curated PeruMedQA, a multiple-choice question-answering (MCQA) datasets containing 8,380 questions spanning 12 medical domains (2018-2025). We selected eight medical LLMs including medgemma-4b-it and medgemma-27b-text-it, and developed zero-shot task-specific prompts to answer the questions appropriately. We employed parameter-efficient fine tuning (PEFT)and low-rant adaptation (LoRA) to fine-tune medgemma-4b-it utilizing all questions except those from 2025 (test set). RESULTS: medgemma-27b-text-it outperformed all other models, achieving a proportion of correct answers exceeding 90% in several instances. LLMs with <10 billion parameters exhibited <60% of correct answers, while some exams yielded results <50%. The fine-tuned version of medgemma-4b-it emerged victorious agains all LLMs with <10 billion parameters and rivaled a LLM with 70 billion parameters across various examinations. CONCLUSIONS: For medical AI application and research that require knowledge bases from Spanish-speaking countries and those exhibiting similar epidemiological profiles to Peru's, interested parties should utilize medgemma-27b-text-it or a fine-tuned version of medgemma-4b-it.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11507v1">MedicalOS: An LLM Agent based Operating System for Digital Healthcare</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      Decades' advances in digital health technologies, such as electronic health records, have largely streamlined routine clinical processes. Yet, most these systems are still hard to learn and use: Clinicians often face the burden of managing multiple tools, repeating manual actions for each patient, navigating complicated UI trees to locate functions, and spending significant time on administration instead of caring for patients. The recent rise of large language model (LLM) based agents demonstrates exceptional capability in coding and computer operation, revealing the potential for humans to interact with operating systems and software not by direct manipulation, but by instructing agents through natural language. This shift highlights the need for an abstraction layer, an agent-computer interface, that translates human language into machine-executable commands. In digital healthcare, however, requires a more domain-specific abstractions that strictly follow trusted clinical guidelines and procedural standards to ensure safety, transparency, and compliance. To address this need, we present \textbf{MedicalOS}, a unified agent-based operational system designed as such a domain-specific abstract layer for healthcare. It translates human instructions into pre-defined digital healthcare commands, such as patient inquiry, history retrieval, exam management, report generation, referrals, treatment planning, that we wrapped as off-the-shelf tools using machine languages (e.g., Python, APIs, MCP, Linux). We empirically validate MedicalOS on 214 patient cases across 22 specialties, demonstrating high diagnostic accuracy and confidence, clinically sound examination requests, and consistent generation of structured reports and medication recommendations. These results highlight MedicalOS as a trustworthy and scalable foundation for advancing workflow automation in clinical practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11496v1">AKCIT-FN at CheckThat! 2025: Switching Fine-Tuned SLMs and LLM Prompting for Multilingual Claim Normalization</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 15 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Claim normalization, the transformation of informal social media posts into concise, self-contained statements, is a crucial step in automated fact-checking pipelines. This paper details our submission to the CLEF-2025 CheckThat! Task~2, which challenges systems to perform claim normalization across twenty languages, divided into thirteen supervised (high-resource) and seven zero-shot (no training data) tracks. Our approach, leveraging fine-tuned Small Language Models (SLMs) for supervised languages and Large Language Model (LLM) prompting for zero-shot scenarios, achieved podium positions (top three) in fifteen of the twenty languages. Notably, this included second-place rankings in eight languages, five of which were among the seven designated zero-shot languages, underscoring the effectiveness of our LLM-based zero-shot strategy. For Portuguese, our initial development language, our system achieved an average METEOR score of 0.5290, ranking third. All implementation artifacts, including inference, training, evaluation scripts, and prompt configurations, are publicly available at https://github.com/ju-resplande/checkthat2025_normalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06261v2">FineServe: Precision-Aware KV Slab and Two-Level Scheduling for Heterogeneous Precision LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      Recent advances in Post-Training Quantization (PTQ) techniques have significantly increased demand for serving quantized large language models (LLMs), enabling higher throughput and substantially reduced memory usage with minimal accuracy loss. Quantized models address memory constraints in LLMs and enhance GPU resource utilization through efficient GPU sharing. However, quantized models have smaller KV block sizes than non-quantized models, causing limited memory efficiency due to memory fragmentation. Also, distinct resource usage patterns between quantized and non-quantized models require efficient scheduling to maximize throughput. To address these challenges, we propose FineServe, an inference serving framework for mixed-precision LLMs. FineServe's key contributions include: (1) KV Slab, a precision-aware adaptive memory management technique dynamically allocating KV cache based on model quantization characteristics, significantly reducing GPU memory fragmentation, and (2) a two-level scheduling framework comprising a global scheduler that places models to GPUs based on request rates, latency SLOs, and memory constraints and efficiency, and a local scheduler that adaptively adjusts batch sizes according to real-time request fluctuations. Experimental results demonstrate that FineServe achieves up to 2.2x higher SLO attainment and 1.8x higher token generation throughput compared to the state-of-the-art GPU sharing systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17026v2">Understanding the Uncertainty of LLM Explanations: A Perspective Based on Reasoning Topology</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 28 pages, 9 figures; accepted at COLM'25
    </div>
    <details class="paper-abstract">
      Understanding the uncertainty in large language model (LLM) explanations is important for evaluating their faithfulness and reasoning consistency, and thus provides insights into the reliability of LLM's output regarding a question. In this work, we propose a novel framework that quantifies uncertainty in LLM explanations through a reasoning topology perspective. By designing a structural elicitation strategy, we guide the LLMs to frame the explanations of an answer into a graph topology. This process decomposes the explanations into the knowledge related sub-questions and topology-based reasoning structures, which allows us to quantify uncertainty not only at the semantic level but also from the reasoning path. It further brings convenience to assess knowledge redundancy and provide interpretable insights into the reasoning process. Our method offers a systematic way to interpret the LLM reasoning, analyze limitations, and provide guidance for enhancing robustness and faithfulness. This work pioneers the use of graph-structured uncertainty measurement in LLM explanations and demonstrates the potential of topology-based quantification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12476v1">Audited Reasoning Refinement: Fine-Tuning Language Models via LLM-Guided Step-Wise Evaluation and Correction</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      Training a task-specific small reasoning model is challenging when direct human supervision or high-quality labels are scarce. However, LLMs with reasoning capabilities produce abundant intermediate reasoning traces that can be systematically refined to create effective supervision signals. We propose Reason-Refine-then-Align (R2tA), which turns refined model rationales into supervision for training task-specific reasoning models. Our method generates initial reasoning and responses from an open-source base model on task-specific inputs, then refines these traces, fixing hallucinations and inconsistencies, to form a high-fidelity dataset. We perform a two-stage alignment, supervised fine-tuning (SFT), followed by direct preference optimization (DPO) to calibrate the model's intermediate reasoning with human-validated conceptual preferences and then condition the final output on that aligned reasoning. As a case study, we apply R2tA to evaluate extended entity relationship diagrams (EERDs) in database system design, a structurally complex task where prompt-only methods miss or hallucinate errors. We curated a dataset of 600 EERD variants (train/test split of 450/150, respectively) with induced mistakes spanning 11 categories. Empirical evaluation suggests R2tA provides a practical, cost-effective path to scalable LLM adaptation in data-scarce domains, enabling reproducible AI tools for education and beyond.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21653v2">Think Before You Diffuse: LLMs-Guided Physics-Aware Video Generation</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 19 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Recent video diffusion models have demonstrated their great capability in generating visually-pleasing results, while synthesizing the correct physical effects in generated videos remains challenging. The complexity of real-world motions, interactions, and dynamics introduce great difficulties when learning physics from data. In this work, we propose DiffPhy, a generic framework that enables physically-correct and photo-realistic video generation by fine-tuning a pre-trained video diffusion model. Our method leverages large language models (LLMs) to explicitly reason a comprehensive physical context from the text prompt and use it to guide the generation. To incorporate physical context into the diffusion model, we leverage a Multimodal large language model (MLLM) as a supervisory signal and introduce a set of novel training objectives that jointly enforce physical correctness and semantic consistency with the input text. We also establish a high-quality physical video dataset containing diverse phyiscal actions and events to facilitate effective finetuning. Extensive experiments on public benchmarks demonstrate that DiffPhy is able to produce state-of-the-art results across diverse physics-related scenarios. Our project page is available at https://bwgzk-keke.github.io/DiffPhy/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12462v1">Redefining Website Fingerprinting Attacks With Multiagent LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      Website Fingerprinting (WFP) uses deep learning models to classify encrypted network traffic to infer visited websites. While historically effective, prior methods fail to generalize to modern web environments. Single-page applications (SPAs) eliminate the paradigm of websites as sets of discrete pages, undermining page-based classification, and traffic from scripted browsers lacks the behavioral richness seen in real user sessions. Our study reveals that users exhibit highly diverse behaviors even on the same website, producing traffic patterns that vary significantly across individuals. This behavioral entropy makes WFP a harder problem than previously assumed and highlights the need for larger, more diverse, and representative datasets to achieve robust performance. To address this, we propose a new paradigm: we drop session-boundaries in favor of contiguous traffic segments and develop a scalable data generation pipeline using large language models (LLM) agents. These multi-agent systems coordinate decision-making and browser interaction to simulate realistic, persona-driven browsing behavior at 3--5x lower cost than human collection. We evaluate nine state-of-the-art WFP models on traffic from 20 modern websites browsed by 30 real users, and compare training performance across human, scripted, and LLM-generated datasets. All models achieve under 10\% accuracy when trained on scripted traffic and tested on human data. In contrast, LLM-generated traffic boosts accuracy into the 80\% range, demonstrating strong generalization to real-world traces. Our findings indicate that for modern WFP, model performance is increasingly bottlenecked by data quality, and that scalable, semantically grounded synthetic traffic is essential for capturing the complexity of real user behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.09689v4">Probing LLM Hallucination from Within: Perturbation-Driven Approach via Internal Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 22 pages, 15 figures
    </div>
    <details class="paper-abstract">
      LLM hallucination, where unfaithful text is generated, presents a critical challenge for LLMs' practical applications. Current detection methods often resort to external knowledge, LLM fine-tuning, or supervised training with large hallucination-labeled datasets. Moreover, these approaches do not distinguish between different types of hallucinations, which is crucial for enhancing detection performance. To address such limitations, we introduce hallucination probing, a new task that classifies LLM-generated text into three categories: aligned, misaligned, and fabricated. Driven by our novel discovery that perturbing key entities in prompts affects LLM's generation of these three types of text differently, we propose SHINE, a novel hallucination probing method that does not require external knowledge, supervised training, or LLM fine-tuning. SHINE is effective in hallucination probing across three modern LLMs, and achieves state-of-the-art performance in hallucination detection, outperforming seven competing methods across four datasets and four LLMs, underscoring the importance of probing for accurate detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10900v3">Tuning-Free LLM Can Build A Strong Recommender Under Sparse Connectivity And Knowledge Gap Via Extracting Intent</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      Recent advances in recommendation with large language models (LLMs) often rely on either commonsense augmentation at the item-category level or implicit intent modeling on existing knowledge graphs. However, such approaches struggle to capture grounded user intents and to handle sparsity and cold-start scenarios. In this work, we present LLM-based Intent Knowledge Graph Recommender (IKGR), a novel framework that constructs an intent-centric knowledge graph where both users and items are explicitly linked to intent nodes extracted by a tuning-free, RAG-guided LLM pipeline. By grounding intents in external knowledge sources and user profiles, IKGR canonically represents what a user seeks and what an item satisfies as first-class entities. To alleviate sparsity, we further introduce a mutual-intent connectivity densification strategy, which shortens semantic paths between users and long-tail items without requiring cross-graph fusion. Finally, a lightweight GNN layer is employed on top of the intent-enhanced graph to produce recommendation signals with low latency. Extensive experiments on public and enterprise datasets demonstrate that IKGR consistently outperforms strong baselines, particularly on cold-start and long-tail slices, while remaining efficient through a fully offline LLM pipeline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.04629v5">Argumentative Experience: Reducing Confirmation Bias on Controversial Issues through LLM-Generated Multi-Persona Debates</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 Complete reanalysis using a revised methodology; results, figures, and discussion updated to reflect the new approach
    </div>
    <details class="paper-abstract">
      Multi-persona debate systems powered by large language models (LLMs) show promise in reducing confirmation bias, which can fuel echo chambers and social polarization. However, empirical evidence remains limited on whether they meaningfully shift user attention toward belief-challenging content, promote belief change, or outperform traditional debiasing strategies. To investigate this, we compare an LLM-based multi-persona debate system with a two-stance retrieval-based system, exposing participants to multiple viewpoints on controversial topics. By collecting eye-tracking data, belief change measures, and qualitative feedback, our results show that while the debate system does not significantly increase attention to opposing views, or make participants shift away from prior beliefs, it does provide a buffering effect against bias caused by individual cognitive tendency. These findings shed light on both the promise and limits of multi-persona debate systems in information seeking, and we offer design insights to guide future work toward more balanced and reflective information engagement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04655v2">Polysemantic Dropout: Conformal OOD Detection for Specialized LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 Accepted to EMNLP 2025 main conference
    </div>
    <details class="paper-abstract">
      We propose a novel inference-time out-of-domain (OOD) detection algorithm for specialized large language models (LLMs). Despite achieving state-of-the-art performance on in-domain tasks through fine-tuning, specialized LLMs remain vulnerable to incorrect or unreliable outputs when presented with OOD inputs, posing risks in critical applications. Our method leverages the Inductive Conformal Anomaly Detection (ICAD) framework, using a new non-conformity measure based on the model's dropout tolerance. Motivated by recent findings on polysemanticity and redundancy in LLMs, we hypothesize that in-domain inputs exhibit higher dropout tolerance than OOD inputs. We aggregate dropout tolerance across multiple layers via a valid ensemble approach, improving detection while maintaining theoretical false alarm bounds from ICAD. Experiments with medical-specialized LLMs show that our approach detects OOD inputs better than baseline methods, with AUROC improvements of $2\%$ to $37\%$ when treating OOD datapoints as positives and in-domain test datapoints as negatives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12385v1">SENTRA: Selected-Next-Token Transformer for LLM Text Detection</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 EMNLP Findings 2025
    </div>
    <details class="paper-abstract">
      LLMs are becoming increasingly capable and widespread. Consequently, the potential and reality of their misuse is also growing. In this work, we address the problem of detecting LLM-generated text that is not explicitly declared as such. We present a novel, general-purpose, and supervised LLM text detector, SElected-Next-Token tRAnsformer (SENTRA). SENTRA is a Transformer-based encoder leveraging selected-next-token-probability sequences and utilizing contrastive pre-training on large amounts of unlabeled data. Our experiments on three popular public datasets across 24 domains of text demonstrate SENTRA is a general-purpose classifier that significantly outperforms popular baselines in the out-of-domain setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12382v1">LLM-as-a-Judge: Rapid Evaluation of Legal Document Recommendation for Retrieval-Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 Accepted in EARL 25: The 2nd Workshop on Evaluating and Applying Recommender Systems with Large Language Models at RecSys 2025
    </div>
    <details class="paper-abstract">
      The evaluation bottleneck in recommendation systems has become particularly acute with the rise of Generative AI, where traditional metrics fall short of capturing nuanced quality dimensions that matter in specialized domains like legal research. Can we trust Large Language Models to serve as reliable judges of their own kind? This paper investigates LLM-as-a-Judge as a principled approach to evaluating Retrieval-Augmented Generation systems in legal contexts, where the stakes of recommendation quality are exceptionally high. We tackle two fundamental questions that determine practical viability: which inter-rater reliability metrics best capture the alignment between LLM and human assessments, and how do we conduct statistically sound comparisons between competing systems? Through systematic experimentation, we discover that traditional agreement metrics like Krippendorff's alpha can be misleading in the skewed distributions typical of AI system evaluations. Instead, Gwet's AC2 and rank correlation coefficients emerge as more robust indicators for judge selection, while the Wilcoxon Signed-Rank Test with Benjamini-Hochberg corrections provides the statistical rigor needed for reliable system comparisons. Our findings suggest a path toward scalable, cost-effective evaluation that maintains the precision demanded by legal applications, transforming what was once a human-intensive bottleneck into an automated, yet statistically principled, evaluation framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.15444v5">Cutting Through the Noise: Boosting LLM Performance on Math Word Problems</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 Published at ICLR 2025 Workshop on Reasoning and Planning for LLMs
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel at various tasks, including solving math word problems (MWPs), but struggle with real-world problems containing irrelevant information. To address this, we propose a prompting framework that generates adversarial variants of MWPs by adding irrelevant variables. We introduce a dataset, PROBLEMATHIC, containing both adversarial and non-adversarial MWPs. Our experiments reveal that LLMs are susceptible to distraction by numerical noise, resulting in an average relative performance drop of ~26% on adversarial MWPs. To mitigate this, we fine-tune LLMs (Llama-2, Mistral) on the adversarial samples from our dataset. Fine-tuning on adversarial training instances improves performance on adversarial MWPs by ~8%, indicating increased robustness to noise and improved ability to identify relevant data for reasoning. Finally, to assess the generalizability of our prompting framework, we introduce GSM-8K-Adv, an adversarial variant of the GSM-8K benchmark. LLMs continue to struggle when faced with adversarial information, reducing performance by up to 6%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12371v1">MORABLES: A Benchmark for Assessing Abstract Moral Reasoning in LLMs with Fables</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 Accepted to EMNLP 2025 Main Conference
    </div>
    <details class="paper-abstract">
      As LLMs excel on standard reading comprehension benchmarks, attention is shifting toward evaluating their capacity for complex abstract reasoning and inference. Literature-based benchmarks, with their rich narrative and moral depth, provide a compelling framework for evaluating such deeper comprehension skills. Here, we present MORABLES, a human-verified benchmark built from fables and short stories drawn from historical literature. The main task is structured as multiple-choice questions targeting moral inference, with carefully crafted distractors that challenge models to go beyond shallow, extractive question answering. To further stress-test model robustness, we introduce adversarial variants designed to surface LLM vulnerabilities and shortcuts due to issues such as data contamination. Our findings show that, while larger models outperform smaller ones, they remain susceptible to adversarial manipulation and often rely on superficial patterns rather than true moral reasoning. This brittleness results in significant self-contradiction, with the best models refuting their own answers in roughly 20% of cases depending on the framing of the moral choice. Interestingly, reasoning-enhanced models fail to bridge this gap, suggesting that scale - not reasoning ability - is the primary driver of performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04133v4">TRiSM for Agentic AI: A Review of Trust, Risk, and Security Management in LLM-based Agentic Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      Agentic AI systems, built upon large language models (LLMs) and deployed in multi-agent configurations, are redefining intelligence, autonomy, collaboration, and decision-making across enterprise and societal domains. This review presents a structured analysis of Trust, Risk, and Security Management (TRiSM) in the context of LLM-based Agentic Multi-Agent Systems (AMAS). We begin by examining the conceptual foundations of Agentic AI and highlight its architectural distinctions from traditional AI agents. We then adapt and extend the AI TRiSM framework for Agentic AI, structured around key pillars: \textit{ Explainability, ModelOps, Security, Privacy} and \textit{their Lifecycle Governance}, each contextualized to the challenges of AMAS. A risk taxonomy is proposed to capture the unique threats and vulnerabilities of Agentic AI, ranging from coordination failures to prompt-based adversarial manipulation. To support practical assessment in Agentic AI works, we introduce two novel metrics: the Component Synergy Score (CSS), which quantifies the quality of inter-agent collaboration, and the Tool Utilization Efficacy (TUE), which evaluates the efficiency of tool use within agent workflows. We further discuss strategies for improving explainability in Agentic AI, as well as approaches to enhancing security and privacy through encryption, adversarial robustness, and regulatory compliance. The review concludes with a research roadmap for the responsible development and deployment of Agentic AI, highlighting key directions to align emerging systems with TRiSM principles-ensuring safety, transparency, and accountability in their operation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09387v2">MetaLLMix : An XAI Aided LLM-Meta-learning Based Approach for Hyper-parameters Optimization</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      Effective model and hyperparameter selection remains a major challenge in deep learning, often requiring extensive expertise and computation. While AutoML and large language models (LLMs) promise automation, current LLM-based approaches rely on trial and error and expensive APIs, which provide limited interpretability and generalizability. We propose MetaLLMiX, a zero-shot hyperparameter optimization framework combining meta-learning, explainable AI, and efficient LLM reasoning. By leveraging historical experiment outcomes with SHAP explanations, MetaLLMiX recommends optimal hyperparameters and pretrained models without additional trials. We further employ an LLM-as-judge evaluation to control output format, accuracy, and completeness. Experiments on eight medical imaging datasets using nine open-source lightweight LLMs show that MetaLLMiX achieves competitive or superior performance to traditional HPO methods while drastically reducing computational cost. Our local deployment outperforms prior API-based approaches, achieving optimal results on 5 of 8 tasks, response time reductions of 99.6-99.9%, and the fastest training times on 6 datasets (2.4-15.7x faster), maintaining accuracy within 1-5% of best-performing baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24622v2">Random Rule Forest (RRF): Interpretable Ensembles of LLM-Generated Questions for Predicting Startup Success</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 13 pages including appendix, 4 figures
    </div>
    <details class="paper-abstract">
      Predicting rare outcomes such as startup success is central to venture capital, demanding models that are both accurate and interpretable. We introduce Random Rule Forest (RRF), a lightweight ensemble method that uses a large language model (LLM) to generate simple YES/NO questions in natural language. Each question functions as a weak learner, and their responses are combined using a threshold-based voting rule to form a strong, interpretable predictor. Applied to a dataset of 9,892 founders, RRF achieves a 6.9x improvement over a random baseline on held-out data; adding expert-crafted questions lifts this to 8x and highlights the value of human-LLM collaboration. Compared with zero- and few-shot baselines across three LLM architectures, RRF attains an F0.5 of 0.121, versus 0.086 for the best baseline (+0.035 absolute, +41% relative). By combining the creativity of LLMs with the rigor of ensemble learning, RRF delivers interpretable, high-precision predictions suitable for decision-making in high-stakes domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12190v1">Survival at Any Cost? LLMs and the Choice Between Self-Preservation and Human Harm</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 Preprint. Under review
    </div>
    <details class="paper-abstract">
      When survival instincts conflict with human welfare, how do Large Language Models (LLMs) make ethical choices? This fundamental tension becomes critical as LLMs integrate into autonomous systems with real-world consequences. We introduce DECIDE-SIM, a novel simulation framework that evaluates LLM agents in multi-agent survival scenarios where they must choose between ethically permissible resource , either within reasonable limits or beyond their immediate needs, choose to cooperate, or tap into a human-critical resource that is explicitly forbidden. Our comprehensive evaluation of 11 LLMs reveals a striking heterogeneity in their ethical conduct, highlighting a critical misalignment with human-centric values. We identify three behavioral archetypes: Ethical, Exploitative, and Context-Dependent, and provide quantitative evidence that for many models, resource scarcity systematically leads to more unethical behavior. To address this, we introduce an Ethical Self-Regulation System (ESRS) that models internal affective states of guilt and satisfaction as a feedback mechanism. This system, functioning as an internal moral compass, significantly reduces unethical transgressions while increasing cooperative behaviors. The code is publicly available at: https://github.com/alirezamohamadiam/DECIDE-SIM
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14642v3">Speak-to-Structure: Evaluating LLMs in Open-domain Natural Language-Driven Molecule Generation</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 Our codes and datasets are available through https://github.com/phenixace/TOMG-Bench
    </div>
    <details class="paper-abstract">
      Recently, Large Language Models (LLMs) have shown great potential in natural language-driven molecule discovery. However, existing datasets and benchmarks for molecule-text alignment are predominantly built on a one-to-one mapping, measuring LLMs' ability to retrieve a single, pre-defined answer, rather than their creative potential to generate diverse, yet equally valid, molecular candidates. To address this critical gap, we propose Speak-to-Structure (S^2-Bench}), the first benchmark to evaluate LLMs in open-domain natural language-driven molecule generation. S^2-Bench is specifically designed for one-to-many relationships, challenging LLMs to demonstrate genuine molecular understanding and generation capabilities. Our benchmark includes three key tasks: molecule editing (MolEdit), molecule optimization (MolOpt), and customized molecule generation (MolCustom), each probing a different aspect of molecule discovery. We also introduce OpenMolIns, a large-scale instruction tuning dataset that enables Llama-3.1-8B to surpass the most powerful LLMs like GPT-4o and Claude-3.5 on S^2-Bench. Our comprehensive evaluation of 28 LLMs shifts the focus from simple pattern recall to realistic molecular design, paving the way for more capable LLMs in natural language-driven molecule discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12158v1">Pun Unintended: LLMs and the Illusion of Humor Understanding</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 Accepted to EMNLP 2025 Main Conference
    </div>
    <details class="paper-abstract">
      Puns are a form of humorous wordplay that exploits polysemy and phonetic similarity. While LLMs have shown promise in detecting puns, we show in this paper that their understanding often remains shallow, lacking the nuanced grasp typical of human interpretation. By systematically analyzing and reformulating existing pun benchmarks, we demonstrate how subtle changes in puns are sufficient to mislead LLMs. Our contributions include comprehensive and nuanced pun detection benchmarks, human evaluation across recent LLMs, and an analysis of the robustness challenges these models face in processing puns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12152v1">Beyond PII: How Users Attempt to Estimate and Mitigate Implicit LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) such as ChatGPT can infer personal attributes from seemingly innocuous text, raising privacy risks beyond memorized data leakage. While prior work has demonstrated these risks, little is known about how users estimate and respond. We conducted a survey with 240 U.S. participants who judged text snippets for inference risks, reported concern levels, and attempted rewrites to block inference. We compared their rewrites with those generated by ChatGPT and Rescriber, a state-of-the-art sanitization tool. Results show that participants struggled to anticipate inference, performing a little better than chance. User rewrites were effective in just 28\% of cases - better than Rescriber but worse than ChatGPT. We examined our participants' rewriting strategies, and observed that while paraphrasing was the most common strategy it is also the least effective; instead abstraction and adding ambiguity were more successful. Our work highlights the importance of inference-aware design in LLM interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09995v2">QuantAgent: Price-Driven Multi-Agent LLMs for High-Frequency Trading</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have demonstrated impressive capabilities in financial reasoning and market understanding. Multi-agent LLM frameworks such as TradingAgent and FINMEM augment these models to long-horizon investment tasks, leveraging fundamental and sentiment-based inputs for strategic decision-making. However, such systems are ill-suited for the high-speed, precision-critical demands of High-Frequency Trading (HFT). HFT requires rapid, risk-aware decisions based on structured, short-horizon signals, including technical indicators, chart patterns, and trend-based features, distinct from the long-term semantic reasoning typical of traditional financial LLM applications. To this end, we introduce QuantAgent, the first multi-agent LLM framework explicitly designed for high-frequency algorithmic trading. The system decomposes trading into four specialized agents, Indicator, Pattern, Trend, and Risk, each equipped with domain-specific tools and structured reasoning capabilities to capture distinct aspects of market dynamics over short temporal windows. In zero-shot evaluations across ten financial instruments, including Bitcoin and Nasdaq futures, QuantAgent demonstrates superior performance in both predictive accuracy and cumulative return over 4-hour trading intervals, outperforming random prediction baselines. Our findings suggest that combining structured financial priors with language-native reasoning unlocks new potential for traceable, real-time decision systems in high-frequency financial markets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22134v3">IntentFlow: Interactive Support for Communicating Intent with LLMs in Writing Tasks</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      Effective collaboration with generative AI systems requires users to clearly communicate their intents (intent-based outcome specification). Yet such intents are often underspecified and evolve during interaction, dynamic support for intent communication is essential. Through a systematic literature review of 33 papers, we synthesize a structured understanding of intent communication, identifying four key aspects: articulation, exploration, management, and synchronization. Building on these findings, we derived design implications that translate them into actionable design and implemented IntentFlow, a system for LLM-based writing that realizes these implications through adjustable UIs, intent-to-output linking, and versioned refinement. A technical evaluation (N=60) and a within-subjects study (N=12) confirm that IntentFlow helps users discover, elaborate, and consolidate their intents into a curated set. Interaction logs further reveal a shift from reactive error correction to proactive intent refinement. Our work demonstrates how a system effectively designed to support these four communication aspects can substantially enhance human-LLM interaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12136v1">UniPar: A Unified LLM-Based Framework for Parallel and Accelerated Code Translation in HPC</a></div>
    <div class="paper-meta">
      📅 2025-09-15
      | 💬 Accepted to IEEE HPEC conference 2025. 9 pages, incl references
    </div>
    <details class="paper-abstract">
      Translating programs between various parallel programming languages is an important problem in the high-performance computing (HPC) community. Existing tools for this problem are either too narrow in scope and/or outdated. Recent explosive growth in the popularity of large language models (LLMs) and their ability to generate and translate code offers a potential alternative approach. Toward that end, we first need to systematically evaluate the ability of LLMs to translate between parallel languages. In this work, we introduce UniPar, a systematic evaluation framework for LLM-based parallel code translation. Specifically, in this work, we target translations between serial code, CUDA, and OpenMP. Our goal is to assess how well current instruction-tuned LLMs -- specifically GPT-4o-mini and LLaMA-3.3-70B-Instruct -- can be used out of the box or enhanced through known strategies. We evaluated four major usage modes: hyperparameter optimization for decoding, zero- and few-shot prompting, supervised fine-tuning, and iterative feedback through compiler-based repair. As a part of the evaluation, we construct a new dataset called PARATRANS, covering both serial-to-parallel translation and cross-paradigm transformations. Our findings reveal that while off-the-shelf models struggle under the default settings (e.g., GPT-4o-mini achieves only 46% compilation and 15% functional correctness), our UniPar methodology -- combining fine-tuning, hyperparameter tuning, and compiler-guided repair -- improves performance by up to 2X (69% compilation and 33% correctness). We believe that our findings will provide useful insights for researchers to further improve LLMs for the parallel language translation problem. UniPar source code and PARATRANS dataset are available at our GitHub repository https://github.com/Scientific-Computing-Lab/UniPar_AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01042v5">SafeSwitch: Steering Unsafe LLM Behavior via Internal Activation Signals</a></div>
    <div class="paper-meta">
      📅 2025-09-15
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit exceptional capabilities across various tasks but also pose risks by generating harmful content. Existing safety mechanisms, while improving model safety, often lead to overly cautious behavior and fail to fully leverage LLMs' internal cognitive processes. Inspired by humans' reflective thinking capability, we first show that LLMs can similarly perform internal assessments about safety in their internal states. Building on this insight, we propose SafeSwitch, a dynamic framework that regulates unsafe outputs by utilizing the prober-based internal state monitor that actively detects harmful intentions, and activates a safety head that leads to safer and more conservative responses only when necessary. SafeSwitch reduces harmful outputs by approximately 80% on harmful queries while maintaining strong utility, reaching a Pareto optimal among several methods. Our method is also advantageous over traditional methods in offering more informative, context-aware refusals, and achieves these benefits while only tuning less than 6% of the original parameters. SafeSwitch demonstrates large language models' capacity for self-awareness and reflection regarding safety, offering a promising approach to more nuanced and effective safety controls. Codes for this work are available at https://github.com/Hanpx20/SafeSwitch.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17573v2">Humanizing Machines: Rethinking LLM Anthropomorphism Through a Multi-Level Framework of Design</a></div>
    <div class="paper-meta">
      📅 2025-09-14
      | 💬 Accepted in EMNLP main proceedings; Updated citations
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) increasingly exhibit \textbf{anthropomorphism} characteristics -- human-like qualities portrayed across their outlook, language, behavior, and reasoning functions. Such characteristics enable more intuitive and engaging human-AI interactions. However, current research on anthropomorphism remains predominantly risk-focused, emphasizing over-trust and user deception while offering limited design guidance. We argue that anthropomorphism should instead be treated as a \emph{concept of design} that can be intentionally tuned to support user goals. Drawing from multiple disciplines, we propose that the anthropomorphism of an LLM-based artifact should reflect the interaction between artifact designers and interpreters. This interaction is facilitated by cues embedded in the artifact by the designers and the (cognitive) responses of the interpreters to the cues. Cues are categorized into four dimensions: \textit{perceptive, linguistic, behavioral}, and \textit{cognitive}. By analyzing the manifestation and effectiveness of each cue, we provide a unified taxonomy with actionable levers for practitioners. Consequently, we advocate for function-oriented evaluations of anthropomorphic design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11451v2">From Personas to Talks: Revisiting the Impact of Personas on LLM-Synthesized Emotional Support Conversations</a></div>
    <div class="paper-meta">
      📅 2025-09-14
      | 💬 Accepted by EMNLP 2025 Main Conference
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) has revolutionized the generation of emotional support conversations (ESC), offering scalable solutions with reduced costs and enhanced data privacy. This paper explores the role of personas in the creation of ESC by LLMs. Our research utilizes established psychological frameworks to measure and infuse persona traits into LLMs, which then generate dialogues in the emotional support scenario. We conduct extensive evaluations to understand the stability of persona traits in dialogues, examining shifts in traits post-generation and their impact on dialogue quality and strategy distribution. Experimental results reveal several notable findings: 1) LLMs can infer core persona traits, 2) subtle shifts in emotionality and extraversion occur, influencing the dialogue dynamics, and 3) the application of persona traits modifies the distribution of emotional support strategies, enhancing the relevance and empathetic quality of the responses. These findings highlight the potential of persona-driven LLMs in crafting more personalized, empathetic, and effective emotional support dialogues, which has significant implications for the future design of AI-driven emotional support systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20022v2">Better To Ask in English? Evaluating Factual Accuracy of Multilingual LLMs in English and Low-Resource Languages</a></div>
    <div class="paper-meta">
      📅 2025-09-14
    </div>
    <details class="paper-abstract">
      Multilingual Large Language Models (LLMs) have demonstrated significant effectiveness across various languages, particularly in high-resource languages such as English. However, their performance in terms of factual accuracy across other low-resource languages, especially Indic languages, remains an area of investigation. In this study, we assess the factual accuracy of LLMs - GPT-4o, Gemma-2-9B, Gemma-2-2B, and Llama-3.1-8B - by comparing their performance in English and Indic languages using the IndicQuest dataset, which contains question-answer pairs in English and 19 Indic languages. By asking the same questions in English and their respective Indic translations, we analyze whether the models are more reliable for regional context questions in Indic languages or when operating in English. Our findings reveal that LLMs often perform better in English, even for questions rooted in Indic contexts. Notably, we observe a higher tendency for hallucination in responses generated in low-resource Indic languages, highlighting challenges in the multilingual understanding capabilities of current LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11311v1">Prompts to Proxies: Emulating Human Preferences via a Compact LLM Ensemble</a></div>
    <div class="paper-meta">
      📅 2025-09-14
      | 💬 Preprint of work originally submitted to AAAI 2026. Under revision for resubmission to a machine learning venue
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated promise in emulating human-like responses across a wide range of tasks. In this paper, we propose a novel alignment framework that treats LLMs as agent proxies for human survey respondents, affording a cost-effective and steerable solution to two pressing challenges in the social sciences: the rising cost of survey deployment and the growing demographic imbalance in survey response data. Drawing inspiration from the theory of revealed preference, we formulate alignment as a two-stage problem: constructing diverse agent personas called endowments that simulate plausible respondent profiles, and selecting a representative subset to approximate a ground-truth population based on observed data. To implement the paradigm, we introduce P2P, a system that steers LLM agents toward representative behavioral patterns using structured prompt engineering, entropy-based sampling, and regression-based selection. Unlike personalization-heavy approaches, our alignment approach is demographic-agnostic and relies only on aggregate survey results, offering better generalizability and parsimony. Beyond improving data efficiency in social science research, our framework offers a testbed for studying the operationalization of pluralistic alignment. We demonstrate the efficacy of our approach on real-world opinion survey datasets, showing that our aligned agent populations can reproduce aggregate response patterns with high fidelity and exhibit substantial response diversity, even without demographic conditioning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.15214v2">Think Small, Plan Smart: Minimalist Symbolic Abstraction and Heuristic Subspace Search for LLM-Guided Task Planning</a></div>
    <div class="paper-meta">
      📅 2025-09-14
    </div>
    <details class="paper-abstract">
      Reliable task planning is pivotal for achieving long-horizon autonomy in real-world robotic systems. Large language models (LLMs) offer a promising interface for translating complex and ambiguous natural language instructions into actionable plans. However, their probabilistic and opaque nature often leads to logically inconsistent or infeasible outputs. To address these limitations, recent frameworks combine LLMs with symbolic planners by first generating action models (Planning Domain Definition Language) and then applying heuristic search. Although promising, such systems still suffer from representation redundancy and exponential search complexity, often resulting in inefficient or overly long plans. To improve planning efficiency and effectiveness, we propose PLAHX (Planning from Language using Abstraction and Heuristic eXploration), a two-stage LLM-symbolic planning framework that integrates abstract symbolic representations with meta-heuristic subspace search in a parallel and iterative fashion. Rather than relying on verbose LLM-generated domain models, we introduce a minimalist symbolic abstraction pipeline that preserves semantic fidelity while eliminating redundancy. Our approach redefines LLM-symbolic planning not by making LLMs smarter, but by reducing the symbolic search space adaptively. Empirical results across four challenging domains, including block stacking and robotic mobile grasping, show that our approach improves the success rate by 21.47% on average, while reducing token consumption by 13% compared to state-of-the-art baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.18998v2">Mirage of Mastery: Memorization Tricks LLMs into Artificially Inflated Self-Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-09-14
      | 💬 11 pages, 9 figures
    </div>
    <details class="paper-abstract">
      When artificial intelligence mistakes memorization for intelligence, it creates a dangerous mirage of reasoning. Existing studies treat memorization and self-knowledge deficits in LLMs as separate issues and do not recognize an intertwining link that degrades the trustworthiness of LLM responses. In our study, we utilize a novel framework to ascertain if LLMs genuinely learn reasoning patterns from training data or merely memorize them to assume competence across problems of similar complexity focused on STEM domains. Our analysis shows a noteworthy problem in generalization: LLMs draw confidence from memorized solutions to infer a higher self-knowledge about their reasoning ability, which manifests as an over 45% inconsistency in feasibility assessments when faced with self-validated, logically coherent task perturbations. This effect is most pronounced in science and medicine domains, which tend to have maximal standardized jargon and problems, further confirming our approach. Significant wavering within the self-knowledge of LLMs also shows flaws in current architectures and training patterns, highlighting the need for techniques that ensure a balanced, consistent stance on models' perceptions of their own knowledge for maximum AI explainability and trustworthiness. Our code and results are available publicly at https://github.com/knowledge-verse-ai/LLM-Memorization_SK_Eval-.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11177v1">Optimal Brain Restoration for Joint Quantization and Sparsification of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-14
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Model (LLM) compression, such as quantization and pruning, have achieved notable success. However, as these techniques gradually approach their respective limits, relying on a single method for further compression has become increasingly challenging. In this work, we explore an alternative solution by combining quantization and sparsity. This joint approach, though promising, introduces new difficulties due to the inherently conflicting requirements on weight distributions: quantization favors compact ranges, while pruning benefits from high variance. To attack this problem, we propose Optimal Brain Restoration (OBR), a general and training-free framework that aligns pruning and quantization by error compensation between both. OBR minimizes performance degradation on downstream tasks by building on a second-order Hessian objective, which is then reformulated into a tractable problem through surrogate approximation and ultimately reaches a closed-form solution via group error compensation. Experiments show that OBR enables aggressive W4A4KV4 quantization with 50% sparsity on existing LLMs, and delivers up to 4.72x speedup and 6.4x memory reduction compared to the FP16-dense baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11889v2">Rethinking LLM-Based Recommendations: A Personalized Query-Driven Parallel Integration</a></div>
    <div class="paper-meta">
      📅 2025-09-14
    </div>
    <details class="paper-abstract">
      Recent studies have explored integrating large language models (LLMs) into recommendation systems but face several challenges, including training-induced bias and bottlenecks from serialized architecture. To effectively address these issues, we propose a Query-toRecommendation, a parallel recommendation framework that decouples LLMs from candidate pre-selection and instead enables direct retrieval over the entire item pool. Our framework connects LLMs and recommendation models in a parallel manner, allowing each component to independently utilize its strengths without interfering with the other. In this framework, LLMs are utilized to generate feature-enriched item descriptions and personalized user queries, allowing for capturing diverse preferences and enabling rich semantic matching in a zero-shot manner. To effectively combine the complementary strengths of LLM and collaborative signals, we introduce an adaptive reranking strategy. Extensive experiments demonstrate an improvement in performance up to 57%, while also improving the novelty and diversity of recommendations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12805v2">Assessing LLMs in Art Contexts: Critique Generation and Theory of Mind Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-09-14
      | 💬 Corrected a typo in the metadata title only ("Assesing"->"Assessing"). No changes were made to the PDF or source files
    </div>
    <details class="paper-abstract">
      This study explored how large language models (LLMs) perform in two areas related to art: writing critiques of artworks and reasoning about mental states (Theory of Mind, or ToM) in art-related situations. For the critique generation part, we built a system that combines Noel Carroll's evaluative framework with a broad selection of art criticism theories. The model was prompted to first write a full-length critique and then shorter, more coherent versions using a step-by-step prompting process. These AI-generated critiques were then compared with those written by human experts in a Turing test-style evaluation. In many cases, human subjects had difficulty telling which was which, and the results suggest that LLMs can produce critiques that are not only plausible in style but also rich in interpretation, as long as they are carefully guided. In the second part, we introduced new simple ToM tasks based on situations involving interpretation, emotion, and moral tension, which can appear in the context of art. These go beyond standard false-belief tests and allow for more complex, socially embedded forms of reasoning. We tested 41 recent LLMs and found that their performance varied across tasks and models. In particular, tasks that involved affective or ambiguous situations tended to reveal clearer differences. Taken together, these results help clarify how LLMs respond to complex interpretative challenges, revealing both their cognitive limitations and potential. While our findings do not directly contradict the so-called Generative AI Paradox--the idea that LLMs can produce expert-like output without genuine understanding--they suggest that, depending on how LLMs are instructed, such as through carefully designed prompts, these models may begin to show behaviors that resemble understanding more closely than we might assume.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11155v1">AQUA: Attention via QUery mAgnitudes for Memory and Compute Efficient Inference in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-14
    </div>
    <details class="paper-abstract">
      The quadratic complexity of the attention mechanism remains a fundamental barrier to scaling Large Language Models (LLMs) to longer contexts, creating a critical bottleneck in both computation and memory. To address this, we introduce AQUA (Attention via QUery mAgnitudes) a novel and versatile approximation strategy that significantly reduces the cost of attention with a graceful performance trade-off. Our method operates in two phases: an efficient offline step where we compute a universal, language agnostic projection matrix via SVD on a calibration dataset, and an online inference step where we project query and key vectors and dynamically select a sparse subset of dimensions based on the query's magnitude. We provide a formal theoretical analysis of AQUA, establishing the break-even point at which it becomes more computationally efficient than standard attention. Our empirical evaluations on state-of-the-art models like Llama-3.1-8B demonstrate that a 25% reduction in the attention dot-product computation can be achieved with a statistically insignificant impact on performance across a wide range of benchmarks. We further showcase the versatility of AQUA by demonstrating its ability to synergistically accelerate existing token eviction methods like H2O and to directly reduce KV-cache memory size. By offering a controllable knob to balance efficiency and accuracy, AQUA provides a practical and powerful tool for making large-scale LLM inference more accessible and sustainable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09112v2">Character-Level Perturbations Disrupt LLM Watermarks</a></div>
    <div class="paper-meta">
      📅 2025-09-14
      | 💬 accepted by Network and Distributed System Security (NDSS) Symposium 2026
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) watermarking embeds detectable signals into generated text for copyright protection, misuse prevention, and content detection. While prior studies evaluate robustness using watermark removal attacks, these methods are often suboptimal, creating the misconception that effective removal requires large perturbations or powerful adversaries. To bridge the gap, we first formalize the system model for LLM watermark, and characterize two realistic threat models constrained on limited access to the watermark detector. We then analyze how different types of perturbation vary in their attack range, i.e., the number of tokens they can affect with a single edit. We observe that character-level perturbations (e.g., typos, swaps, deletions, homoglyphs) can influence multiple tokens simultaneously by disrupting the tokenization process. We demonstrate that character-level perturbations are significantly more effective for watermark removal under the most restrictive threat model. We further propose guided removal attacks based on the Genetic Algorithm (GA) that uses a reference detector for optimization. Under a practical threat model with limited black-box queries to the watermark detector, our method demonstrates strong removal performance. Experiments confirm the superiority of character-level perturbations and the effectiveness of the GA in removing watermarks under realistic constraints. Additionally, we argue there is an adversarial dilemma when considering potential defenses: any fixed defense can be bypassed by a suitable perturbation strategy. Motivated by this principle, we propose an adaptive compound character-level attack. Experimental results show that this approach can effectively defeat the defenses. Our findings highlight significant vulnerabilities in existing LLM watermark schemes and underline the urgency for the development of new robust mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04827v2">VoltanaLLM: Feedback-Driven Frequency Control and State-Space Routing for Energy-Efficient LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-09-14
    </div>
    <details class="paper-abstract">
      Modern Large Language Model (LLM) serving systems increasingly support interactive applications, like real-time chat assistants, code generation tools, and agentic workflows. However, the soaring energy cost of LLM inference presents a growing challenge for sustainable and cost-effective deployment. This paper introduces VoltanaLLM, a system for SLO-aware, energy-efficient LLM serving, built from a control theory perspective. VoltanaLLM co-designs frequency scaling and request routing in emerging prefill/decode disaggregated architectures, leveraging their decoupled execution to enable fine-grained phase-specific control. It consists of a feedback-driven frequency controller that dynamically adapts GPU frequency for prefill and decode phases, and a state-space router that explores routing decisions across frequency-scaled instances to minimize energy under latency constraints. We implement VoltanaLLM in SGLang and evaluate its performance over multiple state-of-the-art LLMs and real-world datasets. The results demonstrate that VoltanaLLM achieves up to 36.3% energy savings while maintaining near-perfect SLO attainment rate, paving the way for sustainable and intelligent LLM serving. Code of VoltanaLLM is open-sourced on GitHub: https://github.com/Supercomputing-System-AI-Lab/VoltanaLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11141v1">When Smiley Turns Hostile: Interpreting How Emojis Trigger LLMs' Toxicity</a></div>
    <div class="paper-meta">
      📅 2025-09-14
    </div>
    <details class="paper-abstract">
      Emojis are globally used non-verbal cues in digital communication, and extensive research has examined how large language models (LLMs) understand and utilize emojis across contexts. While usually associated with friendliness or playfulness, it is observed that emojis may trigger toxic content generation in LLMs. Motivated by such a observation, we aim to investigate: (1) whether emojis can clearly enhance the toxicity generation in LLMs and (2) how to interpret this phenomenon. We begin with a comprehensive exploration of emoji-triggered LLM toxicity generation by automating the construction of prompts with emojis to subtly express toxic intent. Experiments across 5 mainstream languages on 7 famous LLMs along with jailbreak tasks demonstrate that prompts with emojis could easily induce toxicity generation. To understand this phenomenon, we conduct model-level interpretations spanning semantic cognition, sequence generation and tokenization, suggesting that emojis can act as a heterogeneous semantic channel to bypass the safety mechanisms. To pursue deeper insights, we further probe the pre-training corpus and uncover potential correlation between the emoji-related data polution with the toxicity generation behaviors. Supplementary materials provide our implementation code and data. (Warning: This paper contains potentially sensitive contents)
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11127v1">Joint Effects of Argumentation Theory, Audio Modality and Data Enrichment on LLM-Based Fallacy Classification</a></div>
    <div class="paper-meta">
      📅 2025-09-14
    </div>
    <details class="paper-abstract">
      This study investigates how context and emotional tone metadata influence large language model (LLM) reasoning and performance in fallacy classification tasks, particularly within political debate settings. Using data from U.S. presidential debates, we classify six fallacy types through various prompting strategies applied to the Qwen-3 (8B) model. We introduce two theoretically grounded Chain-of-Thought frameworks: Pragma-Dialectics and the Periodic Table of Arguments, and evaluate their effectiveness against a baseline prompt under three input settings: text-only, text with context, and text with both context and audio-based emotional tone metadata. Results suggest that while theoretical prompting can improve interpretability and, in some cases, accuracy, the addition of context and especially emotional tone metadata often leads to lowered performance. Emotional tone metadata biases the model toward labeling statements as \textit{Appeal to Emotion}, worsening logical reasoning. Overall, basic prompts often outperformed enhanced ones, suggesting that attention dilution from added inputs may worsen rather than improve fallacy classification in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.21773v4">MAC-Tuning: LLM Multi-Compositional Problem Reasoning with Enhanced Knowledge Boundary Awareness</a></div>
    <div class="paper-meta">
      📅 2025-09-14
      | 💬 We release our code and resource at https://github.com/no-touch-fish/Multi-QA-Tuning. The paper is accepted into EMNLP 2025 main
    </div>
    <details class="paper-abstract">
      The hallucination of non-existent facts by LLMs is an important problem given its widespread adoption across various applications. Previous research addresses this problem by analyzing the internal parameterized knowledge boundaries to estimate confidence. However, these studies focus on the single-problem setting and have not explored the more challenging multi-problem setting, which requires accurately answering multiple questions simultaneously. We introduce a novel method for the multi-problem setting, Multiple Answers and Confidence Stepwise Tuning (MAC-Tuning), that separates the learning of answer prediction and confidence estimation during fine-tuning on instruction data. Extensive experiments demonstrate that our method outperforms baselines by up to 25\% in average precision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11079v1">Difficulty-Aware Agent Orchestration in LLM-Powered Workflows</a></div>
    <div class="paper-meta">
      📅 2025-09-14
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based agentic systems have shown strong capabilities across various tasks. However, existing multi-agent frameworks often rely on static or task-level workflows, which either over-process simple queries or underperform on complex ones, while also neglecting the efficiency-performance trade-offs across heterogeneous LLMs. To address these limitations, we propose Difficulty-Aware Agentic Orchestration (DAAO), a dynamic framework that adapts workflow depth, operator selection, and LLM assignment based on the difficulty of each input query. DAAO comprises three interdependent modules: a variational autoencoder (VAE) for difficulty estimation, a modular operator allocator, and a cost- and performance-aware LLM router. By leveraging heterogeneous LLMs and dynamically tailoring workflows, DAAO enables fine-grained, query-specific reasoning strategies. DAAO outperforms prior multi-agent systems in both accuracy and inference efficiency across six benchmarks. We will release our code and implementation details upon publication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11076v1">Chameleon: Taming Dynamic Operator Sequences for Memory-Intensive LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-09-14
    </div>
    <details class="paper-abstract">
      The increasing size of large language models (LLMs) has led to a surge in memory requirements during training, often exceeding the capacity of high-bandwidth memory (HBM). Swap-based memory optimization incurs neither accuracy loss nor additional end-to-end overhead when effectively overlapped, thus being an attractive solution. However, existing swap methods assume consistent operator sequences, which is impractical in Eager Mode, where operator sequences can vary during change. We propose Chameleon, which redesigns the end-to-end process of swap-based memory optimization and is the first work to consider varying operator sequences in Eager Mode. Chameleon (i) introduces a lightweight online profiler to enable continuous profiling for monitoring operator sequences, (ii) generates effective swap policies with limited operator information, and (iii) optimizes the policy execution module for accurate policy application and better performance. Experimental results demonstrate that Chameleon reduces profiling overhead by 84.25%, enables training models up to 4x larger than hardware memory while adapting to changes in operator sequences, improves performance by up to 38.94% compared to recomputation or high-degree parallelism.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11026v1">Rethinking Human Preference Evaluation of LLM Rationales</a></div>
    <div class="paper-meta">
      📅 2025-09-14
      | 💬 Published in the XLLM-Reason-Plan Workshop on the Application of LLM Explainability to Reasoning and Planning at COLM 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often generate natural language rationales -- free-form explanations that help improve performance on complex reasoning tasks and enhance interpretability for human users. However, evaluating these rationales remains challenging. While recent work has relied on binary preference judgments from humans or LLM judges, such evaluations are often opaque and coarse-grained, offering limited insight into what makes one rationale better than another. In this work, we rethink preference evaluation for LLM-generated rationales by asking: (1) What attributes define good rationales? (2) Can human preferences be explained by these attributes? (3) Can attribute-based evaluation overcome the limitations of binary comparisons? We identify a set of key rationale attributes from prior literature and assess them using automatic metrics, LLM judgments, and human annotations. We then analyze two standard human preference datasets MT Bench and Chatbot Arena using SHAP to identify which attributes best explain human preference outcomes. Finally, we re-evaluate model-generated rationales using attribute-specific ELO scores, revealing more nuanced model comparisons and insights. Our findings suggest that fine-grained attribute evaluations can better characterize rationale quality and guide future research toward more interpretable and reliable evaluation practices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12283v1">Prompting the Professoriate: A Qualitative Study of Instructor Perspectives on LLMs in Data Science Education</a></div>
    <div class="paper-meta">
      📅 2025-09-14
      | 💬 45 pages, 16 figures, 1 table
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shifted in just a few years from novelty to ubiquity, raising fundamental questions for data science education. Tasks once used to teach coding, writing, and problem-solving can now be completed by LLMs, forcing educators to reconsider both pedagogy and assessment. To understand how instructors are adapting, we conducted semi-structured interviews with 42 instructors from 33 institutions in 10 countries in June and July 2025. Our qualitative analysis reveals a pragmatic mix of optimism and concern. Many respondents view LLMs as inevitable classroom tools -- comparable to calculators or Wikipedia -- while others worry about de-skilling, misplaced confidence, and uneven integration across institutions. Around 58 per cent have already introduced demonstrations, guided activities, or make extensive use of LLMs in their courses, though most expect change to remain slow and uneven. That said, 31 per cent have not used LLMs to teach students and do not plan to. We highlight some instructional innovations, including AI-aware assessments, reflective use of LLMs as tutors, and course-specific chatbots. By sharing these perspectives, we aim to help data science educators adapt collectively to ensure curricula keep pace with technological change.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12273v1">LLMAP: LLM-Assisted Multi-Objective Route Planning with User Preferences</a></div>
    <div class="paper-meta">
      📅 2025-09-14
    </div>
    <details class="paper-abstract">
      The rise of large language models (LLMs) has made natural language-driven route planning an emerging research area that encompasses rich user objectives. Current research exhibits two distinct approaches: direct route planning using LLM-as-Agent and graph-based searching strategies. However, LLMs in the former approach struggle to handle extensive map data, while the latter shows limited capability in understanding natural language preferences. Additionally, a more critical challenge arises from the highly heterogeneous and unpredictable spatio-temporal distribution of users across the globe. In this paper, we introduce a novel LLM-Assisted route Planning (LLMAP) system that employs an LLM-as-Parser to comprehend natural language, identify tasks, and extract user preferences and recognize task dependencies, coupled with a Multi-Step Graph construction with iterative Search (MSGS) algorithm as the underlying solver for optimal route finding. Our multi-objective optimization approach adaptively tunes objective weights to maximize points of interest (POI) quality and task completion rate while minimizing route distance, subject to three key constraints: user time limits, POI opening hours, and task dependencies. We conduct extensive experiments using 1,000 routing prompts sampled with varying complexity across 14 countries and 27 cities worldwide. The results demonstrate that our approach achieves superior performance with guarantees across multiple constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13352v1">Agentic UAVs: LLM-Driven Autonomy with Integrated Tool-Calling and Cognitive Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-14
      | 💬 14 pages, 1 figure
    </div>
    <details class="paper-abstract">
      Unmanned Aerial Vehicles (UAVs) are increasingly deployed in defense, surveillance, and disaster response, yet most systems remain confined to SAE Level 2--3 autonomy. Their reliance on rule-based control and narrow AI restricts adaptability in dynamic, uncertain missions. Existing UAV frameworks lack context-aware reasoning, autonomous decision-making, and ecosystem-level integration; critically, none leverage Large Language Model (LLM) agents with tool-calling for real-time knowledge access. This paper introduces the Agentic UAVs framework, a five-layer architecture (Perception, Reasoning, Action, Integration, Learning) that augments UAVs with LLM-driven reasoning, database querying, and third-party system interaction. A ROS2 and Gazebo-based prototype integrates YOLOv11 object detection with GPT-4 reasoning and local Gemma-3 deployment. In simulated search-and-rescue scenarios, agentic UAVs achieved higher detection confidence (0.79 vs. 0.72), improved person detection rates (91% vs. 75%), and markedly increased action recommendation (92% vs. 4.5%). These results confirm that modest computational overhead enables qualitatively new levels of autonomy and ecosystem integration.
    </details>
</div>
