# llm - 2026_04

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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.26388v1">SplitFT: An Adaptive Federated Split Learning System For LLMs Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2026-04-29
    </div>
    <details class="paper-abstract">
      Federated Split Learning has been identified as an efficient approach to address the computational resource constraints of clients in classical federated learning, while guaranteeing data privacy for distributed model training across data owners. However, it faces some critical challenges when such a training strategy meets large language models (LLMs) for fine-tuning. Such challenges include setting the cutlayer adaptively across different clients to address the data and device heterogeneity issues, which affect the system performance significantly. In addition, efficiently reducing the communication overhead during the fine-tuning procedure is also another challenge. No work tries to address these challenges. To bridge this gap, we propose SplitTF, an adaptive federated split learning system for LLMs fine-tuning. SplitFT enables different clients to set different cut layers according to their computation resources and trained model performance. SplitFT also proposes to reduce the LoRA rank in cutlayer to reduce the communication overhead. In addition to simulating the heterogeneous data in real-world applications for our proposed split federated learning system, we propose a length-based Dirichlet approach to divide the training data into different clients. Extensive experimental results show that our proposed approach outperforms the state-of-the-art approach for fine-tuning time efficiency and model performance based on various popular benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.19470v2">Adaptive Layerwise Perturbation: Unifying Off-Policy Corrections for LLM RL</a></div>
    <div class="paper-meta">
      📅 2026-04-29
    </div>
    <details class="paper-abstract">
      Off-policy problems such as policy staleness and training--inference mismatch have become a major bottleneck for training stability and further exploration in LLM RL. The distribution gap between the inference and updated policies grows because of the techniques to enhance inference efficiency, leading to heavy-tailed importance ratios. Heavy-tailed ratios arise when the policy is locally sharp, which further inflates gradients and can push updates outside the trust region. To address this, we propose Adaptive Layerwise Perturbation (ALP), which injects small learnable perturbations into the input hidden states of each layer during updates and uses the resulting perturbed policy as the numerator of the importance ratio against the unchanged inference policy in the objective. Intuitively, by adding controlled noise to intermediate representations, ALP prevents the updated policy from deviating too sharply from the inference policy and enlarges the policy family to cover inference-time mismatch noise. Hence, the flattened distribution can naturally tighten the gap between the updated and inference policies and reduce the tail of importance ratios, thus maintaining training stability. This is further validated empirically. Experiments on single-turn math and multi-turn tool-integrated reasoning tasks show that ALP not only improves final performance, but also avoids blow-up in the importance-ratio tail and KL spikes during iterative training, along with boosted exploration. Ablations show that representation-level perturbations across all layers are most effective, substantially outperforming partial-layer and logits-only variants.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.26378v1">CoQuant: Joint Weight-Activation Subspace Projection for Mixed-Precision LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-29
      | 💬 14 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Post-training quantization (PTQ) has become an important technique for reducing the inference cost of Large Language Models (LLMs). While recent mixed-precision methods improve ultra-low bit quantization by preserving critical subspaces in high precision, they typically construct these subspaces relying solely on activation statistics. This ignores the fundamental nature of linear operations, where the output perturbation is jointly driven by both activation and weight quantization noise. In this paper, we propose CoQuant, a joint weight-activation subspace projection method. By theoretically modeling the expected output error, CoQuant formulates a closed-form weighted PCA solution that balances activation and weight covariances to select the optimal high-precision subspace. Extensive experiments on Llama-3.2 and Qwen2.5 models show that CoQuant consistently outperforms strong PTQ baselines in both WikiText perplexity and zero-shot common-sense reasoning accuracy. These results demonstrate that joint weight-activation subspace modeling provides a principled and effective direction for low-bit LLM quantization. The source code is available at https://github.com/Zachary5895/CoQuant.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.02399v2">EvoDev: An Iterative Feature-Driven Framework for End-to-End Software Development with LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2026-04-29
      | 💬 14 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Recent advances in large language model agents offer the promise of automating end-to-end software development from natural language requirements. However, existing approaches largely adopt linear, waterfall-style pipelines, which oversimplify the iterative nature of real-world development and struggle with complex, large-scale projects. To address these limitations, we propose EvoDev, an iterative software development framework inspired by feature-driven development. EvoDev decomposes user requirements into a set of user-valued features and constructs a Feature Map, a directed acyclic graph that explicitly models dependencies between features. Each node in the feature map maintains multi-level information, including business logic, design, and code, which is propagated along dependencies to provide context for subsequent development iterations. We evaluate EvoDev on challenging Android development tasks and show that it outperforms the best-performing baseline, Claude Code, by a substantial margin of 56.8%, while improving single-agent performance by 16.0%-76.6% across different base LLMs, highlighting the importance of dependency modeling, context propagation, and workflow-aware agent design for complex software projects. Our work summarizes practical insights for designing iterative, LLM-driven development frameworks and informs future training of base LLMs to better support iterative software development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21459v4">HER: Human-like Reasoning and Reinforcement Learning for LLM Role-playing</a></div>
    <div class="paper-meta">
      📅 2026-04-29
      | 💬 Findings of ACL, 2026
    </div>
    <details class="paper-abstract">
      LLM role-playing, i.e., using LLMs to simulate specific personas, has emerged as a key capability in various applications, such as companionship, content creation and digital games. While current models effectively capture character tones and knowledge, simulating the inner thoughts behind their behaviors remains a challenge. Towards cognitive simulation in LLM role-play, previous efforts mainly suffer from two deficiencies: lacking data with high-quality reasoning traces, and lacking reliable reward signals aligned with human preferences. In this paper, we propose HER, a unified framework for cognitive-level persona simulation. HER introduces dual-layer thinking, which distinguishes characters' first-person thinking from LLMs' third-person thinking. To bridge these gaps, we curate reasoning-augmented role-playing data via reverse engineering, and construct human-aligned principles and reward models. Leveraging these resources, we train HER models based on Qwen3-32B via supervised and reinforcement learning. Extensive experiments validate the effectiveness of our approach. Notably, our models significantly outperform the Qwen3-32B baseline, achieving a 30.26 improvement on the CoSER benchmark and a 14.97% gain on the Minimax Role-Play Bench. Our datasets, principles, and models are released to facilitate future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08182v2">TildeOpen LLM: Leveraging Curriculum Learning to Achieve Equitable Language Representation</a></div>
    <div class="paper-meta">
      📅 2026-04-29
      | 💬 LREC 2026
    </div>
    <details class="paper-abstract">
      Large language models often underperform in many European languages due to the dominance of English and a few high-resource languages in training data. This paper presents TildeOpen LLM, a 30-billion-parameter open-weight foundational model trained for 34 European languages to promote linguistic equity and improve performance for low-resource languages. To address the data imbalance, we combine dataset upsampling with a curriculum-based training schedule that alternates between uniform and natural language distributions. The resulting model performs favorably compared to other multilingual LLMs despite being trained with significantly fewer computing resources. Evaluation across multiple multilingual benchmarks shows that TildeOpen surpasses existing open-weight models in text generation and comprehension, particularly for Baltic, Finno-Ugric, and Slavic languages. Human evaluations confirm an up to tenfold reduction in linguistic errors relative to leading baselines. The model and associated resources are fully open-weight and publicly available at huggingface.co/TildeAI/TildeOpen-30b. These outcomes demonstrate that careful data curation and balanced training strategies can substantially enhance multilingual model quality without increasing model size or training volume.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.26356v1">PiLLar: Matching for Pivot Table Schema via LLM-guided Monte-Carlo Tree Search</a></div>
    <div class="paper-meta">
      📅 2026-04-29
    </div>
    <details class="paper-abstract">
      Pivot tables are ubiquitous in data lakes of modern data ecosystems, making accurate schema matching over pivot tables a key prerequisite for data integration. In this paper, we focus on matching for pivot table schema, which is a novel joint schema-value matching task. It aims to align schemas between pivot tables and standard relational tables, where a correct match must be semantically consistent at the schema level and compatible at the value level. However, due to the inherent data sensitivity of this task, the prevalence of anonymized data in practice poses significant challenges to its matching accuracy and generalization capability. To tackle these challenges, we propose PiLLar, the first matching for pivot table schema framework. We first formulate PiLLar as an LLM-driven search paradigm that operates with minimal annotated privacy-compliant data, thereby achieving training-free adaptation across diverse domains. Next, we provide a theoretical analysis on the error dynamics of the paradigm to ensure the asymptotic convergence of the proposed method. Furthermore, we introduce a new benchmark PTbench, derived from four representative real-world domains and constructed by mining unpivot-suitable tables, performing unpivot on semantically coherent attributes, and applying sampling and anonymization. Extensive experiments demonstrate the superiority of PiLLar, which achieves an average accuracy of 87.94% on the correctly predicted matches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.26328v1">DSIPA: Detecting LLM-Generated Texts via Sentiment-Invariant Patterns Divergence Analysis</a></div>
    <div class="paper-meta">
      📅 2026-04-29
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) presents new security challenges, particularly in detecting machine-generated text used for misinformation, impersonation, and content forgery. Most existing detection approaches struggle with robustness against adversarial perturbation, paraphrasing attacks, and domain shifts, often requiring restrictive access to model parameters or large labeled datasets. To address this, we propose DSIPA, a novel training-free framework that detects LLM-generated content by quantifying sentiment distributional stability under controlled stylistic variation. It is based on the observation that LLMs typically exhibit more emotionally consistent outputs, while human-written texts display greater affective variation. Our framework operates in a zero-shot, black-box manner, leveraging two unsupervised metrics, sentiment distribution consistency and sentiment distribution preservation, to capture these intrinsic behavioral asymmetries without the need for parameter updates or probability access. Extensive experiments are conducted on state-of-the-art proprietary and open-source models, including GPT-5.2, Gemini-1.5-pro, Claude-3, and LLaMa-3.3. Evaluations on five domains, such as news articles, programming code, student essays, academic papers, and community comments, demonstrate that DSIPA improves F1 detection scores by up to 49.89% over baseline methods. The framework exhibits superior generalizability across domains and strong resilience to adversarial conditions, providing a robust and interpretable behavioral signal for secure content identification in the evolving LLM landscape.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.26326v1">Addressing Performance Saturation for LLM RL via Precise Entropy Curve Control</a></div>
    <div class="paper-meta">
      📅 2026-04-29
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has unlocked complex reasoning abilities in large language models (LLMs). However, most RL algorithms suffer from performance saturation, preventing further gains as RL training scales. This problem can be characterized by the collapse of entropy, a key diagnostic for exploration in RL. Existing attempts have tried to prevent entropy collapse through regularization or clipping, but their resulting entropy curves often exhibit instability in the long term, which hinders performance gains. In this paper, we introduce Entrocraft, a simple rejection-sampling approach that realizes any user-customized entropy schedule by biasing the advantage distributions. Entrocraft requires no objective regularization and is advantage-estimator-agnostic. Theoretically, we relate per-step entropy change to the advantage distribution under minimal assumptions, which explains the behavior of existing RL and entropy-preserving methods. Entrocraft also enables a systematic study of entropy schedules, where we find that linear annealing, which starts high and decays to a slightly lower target, performs best. Empirically, Entrocraft addresses performance saturation, significantly improving generalization, output diversity, and long-term training. It enables a 4B model to outperform an 8B baseline, sustains improvement for up to 4x longer before plateauing, and raises pass@K by 50% over the baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.26319v1">A Systematic Comparison of Prompting and Multi-Agent Methods for LLM-based Stance Detection</a></div>
    <div class="paper-meta">
      📅 2026-04-29
    </div>
    <details class="paper-abstract">
      Stance detection identifies the attitude of a text author toward a given target. Recent studies have explored various LLM-based strategies for this task, from zero-shot prompting to multi-agent debate. However, existing works differ in data splits, base models, and evaluation protocols, making fair comparison difficult. We conduct a systematic comparison that evaluates five methods across two categories -- prompt-based inference (Direct Prompting, Auto-CoT, StSQA) and agent-based debate (COLA, MPRF) -- on four datasets with 14 subtasks, using 15 LLMs from six model families with parameter sizes from 7B to 72B+. Our experiments yield several findings. First, on all models with complete results, the best prompt-based method outperforms the best agent-based method, while agent methods require 7 to 12 times more API calls per sample. Second, model scale has a larger impact on performance than method choice, with gains plateauing around 32B. Third, reasoning-enhanced models (DeepSeek-R1) do not consistently outperform general models of the same size on this task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2409.00557v4">Learning to Ask: When LLM Agents Meet Unclear Instruction</a></div>
    <div class="paper-meta">
      📅 2026-04-29
      | 💬 EMNLP 2025
    </div>
    <details class="paper-abstract">
      Equipped with the capability to call functions, modern large language models (LLMs) can leverage external tools for addressing a range of tasks unattainable through language skills alone. However, the effective execution of these tools relies heavily not just on the advanced capabilities of LLMs but also on precise user instructions, which often cannot be ensured in the real world. To evaluate the performance of LLMs tool-use under imperfect instructions, we meticulously examine the real-world instructions queried from users, analyze the error patterns, and build a challenging tool-use benchmark called Noisy ToolBench (NoisyToolBench). We find that due to the next-token prediction training objective, LLMs tend to arbitrarily generate the missed argument, which may lead to hallucinations and risks. To address this issue, we propose a novel framework, Ask-when-Needed (AwN), which prompts LLMs to ask questions to users whenever they encounter obstacles due to unclear instructions. Moreover, to reduce the manual labor involved in user-LLM interaction and assess LLMs performance in tool utilization from both accuracy and efficiency perspectives, we design an automated evaluation tool named ToolEvaluator. Our experiments demonstrate that the AwN significantly outperforms existing frameworks for tool learning in the NoisyToolBench. We will release all related code and datasets to support future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27003v1">When Continual Learning Moves to Memory: A Study of Experience Reuse in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-04-29
      | 💬 Working in progress
    </div>
    <details class="paper-abstract">
      Memory-augmented LLM agents offer an appealing shortcut to continual learning: rather than updating model parameters, they accumulate experience in external memory, seemingly sidestepping the stability-plasticity dilemma of parametric learning. We show that this challenge does not disappear but resurfaces at the memory level. Under a limited context window, old and new experiences compete during retrieval, relocating the continual-learning bottleneck from parameter updates to memory access. To study this phenomenon, we introduce a (k,v) framework that disentangles two fundamental design axes of external memory: how experience is represented and how it is organized for retrieval. Across sequential-task experiments in ALFWorld and BabyAI, we find that abstract procedural memories transfer more reliably than detailed trajectories, while negative transfer disproportionately harms the hard cases. Moreover, finer-grained memory organization is not universally beneficial: designs that yield strong forward transfer can simultaneously induce severe forgetting. Together, these results reveal that external memory does not resolve the continual-learning problem; it reshapes it into a problem of memory representation and retrieval design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.21272v2">LLM-Powered Detection of Price Manipulation in DeFi</a></div>
    <div class="paper-meta">
      📅 2026-04-29
    </div>
    <details class="paper-abstract">
      Decentralized Finance (DeFi) smart contracts manage billions of dollars, making them a prime target for exploits. Price manipulation vulnerabilities, often via flash loans, are a devastating class of attacks causing significant financial losses. Existing detection methods are limited. Reactive approaches analyze attacks only after they occur, while proactive static analysis tools rely on rigid, predefined heuristics, limiting adaptability. Both depend on known attack patterns, failing to identify novel variants or comprehend complex economic logic. We propose PMDetector, a hybrid framework combining static analysis with Large Language Model (LLM)-based reasoning to proactively detect price manipulation vulnerabilities. Our approach uses a formal attack model and a three-stage pipeline. First, static taint analysis identifies potentially vulnerable code paths. Second, a two-stage LLM process filters paths by analyzing defenses and then simulates attacks to evaluate exploitability. Finally, a static analysis checker validates LLM results, retaining only high-risk paths and generating comprehensive vulnerability reports. To evaluate its effectiveness, we built a dataset of 73 real-world vulnerable and 288 benign DeFi protocols. Results show PMDetector achieves 88% precision and 90% recall with Gemini 2.5-flash, significantly outperforming state-of-the-art static analysis and LLM-based approaches. Auditing a vulnerability with PMDetector costs just $0.03 and takes 4.0 seconds with GPT-4.1, offering an efficient and cost-effective alternative to manual audits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27001v1">An Empirical Security Evaluation of LLM-Generated Cryptographic Rust Code</a></div>
    <div class="paper-meta">
      📅 2026-04-29
      | 💬 10 pages, 2 figures , EASE 2026-The 6th International Workshop on Software Security Engineering
    </div>
    <details class="paper-abstract">
      Developers and organizations are using Large Language Models (LLMs) to generate security-critical code more frequently than ever, including cryptographic solutions for their products. This study presents an empirical evaluation of cryptographic security in 240 Rust code samples for two crypto algorithms (AES-256-GCM and ChaCha20-Poly1305) generated by three LLMs (Gemini 2.5 Pro, GPT-4o, and DeepSeek Coder) using four different prompt strategies. For each successfully compiled code sample, CodeQL static analysis and our rule-based crypto-specific analyzer were used to detect vulnerabilities, which are also associated with Common Weakness Enumeration (CWE). The evaluation results revealed that only 23.3% of the generated code samples were successfully compiled. Among the compiled code, CodeQL produced only two false positives, while our rule-based crypto-specific analyzer identified vulnerabilities in 57% of the compiled samples with zero false positives. This demonstrates that general-purpose analysis tools are insufficient for code validation for the experimented crypto algorithms. The compilation success of the two algorithms varied significantly (AES-256-GCM 34.2% versus ChaCha20-Poly1305 12.5%), showing a gap in code generation capabilities. While model choice had no significant effect on compilation success, prompt strategy significantly influenced outcomes (P = 0.002), with chain-of-thought prompting performing 5 times worse than zero-shot. All three models exhibit systematic failures, including nonce reuse and API hallucinations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.26258v1">FlowBot: Inducing LLM Workflows with Bilevel Optimization and Textual Gradients</a></div>
    <div class="paper-meta">
      📅 2026-04-29
    </div>
    <details class="paper-abstract">
      LLM workflows, which coordinate structured calls to individual LLMs (each augmented with varying instructions and tools) to achieve a particular goal, offer a promising path towards extending the capabilities of LLMs and building powerful systems that can tackle diverse tasks. However, existing approaches for building such workflows generally rely on human-crafted pipelines and prompts, which presents a substantial bottleneck in real world deployment. How can automatically induce and optimize such workflows in a data-driven way? This paper describes a simple data-driven approach for automatically inducing LLM workflows. We formulate workflow induction as a bilevel optimization problem: an outer loop which optimizes a high-level sketch of the workflow (in particular how the LLM calls should be structured), and an inner loop which optimizes each individual LLM call one-by one. Both loops are optimized with ``textual gradients'' where for the inner loop we optimize each component in a modular way through ``backpropagating'' textual gradients layer-by-layer. We find that LLM workflows discovered through our \textsc{FlowBot} (work\textbf{flow} induction through \textbf{b}ilevel \textbf{o}ptimization and \textbf{t}extual gradients) approach performs competitively against strong baselines that make use of human-crafted or automatically-generated workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.01297v3">RE-MCDF: Closed-Loop Multi-Expert LLM Reasoning for Knowledge-Grounded Clinical Diagnosis</a></div>
    <div class="paper-meta">
      📅 2026-04-29
      | 💬 Accepted by International Joint Conference on Neural Networks (IJCNN 2026); 9 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Electronic medical records (EMRs), particularly in neurology, are inherently heterogeneous, sparse, and noisy, which poses significant challenges for large language models (LLMs) in clinical diagnosis. In such settings, single-agent systems are vulnerable to self-reinforcing errors, as their predictions lack independent validation and can drift toward spurious conclusions. Although recent multi-agent frameworks attempt to mitigate this issue through collaborative reasoning, their interactions are often shallow and loosely structured, failing to reflect the rigorous, evidence-driven processes used by clinical experts. More fundamentally, existing approaches largely ignore the rich logical dependencies among diseases, such as mutual exclusivity, pathological compatibility, and diagnostic confusion. This limitation prevents them from ruling out clinically implausible hypotheses, even when sufficient evidence is available. To overcome these, we propose RE-MCDF, a relation-enhanced multi-expert clinical diagnosis framework. RE-MCDF introduces a generation--verification--revision closed-loop architecture that integrates three complementary components: (i) a primary expert that generates candidate diagnoses and supporting evidence, (ii) a laboratory expert that dynamically prioritizes heterogeneous clinical indicators, and (iii) a multi-relation awareness and evaluation expert group that explicitly enforces inter-disease logical constraints. Guided by a medical knowledge graph (MKG), the first two experts adaptively reweight EMR evidence, while the expert group validates and corrects candidate diagnoses to ensure logical consistency. Extensive experiments on the neurology subset of CMEMR (NEEMRs) and on our curated dataset (XMEMRs) demonstrate that RE-MCDF consistently outperforms state-of-the-art baselines in complex diagnostic scenarios (https://github.com/shenshaowei/RE-MCDF).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.12537v3">News Harvesting from Google News combining Web Scraping, LLM Metadata Extraction and SCImago Media Rankings enrichment: a case study of IFMIF-DONES</a></div>
    <div class="paper-meta">
      📅 2026-04-29
      | 💬 24 pages, 7 figures
    </div>
    <details class="paper-abstract">
      This study develops and evaluates a systematic methodology for constructing news datasets from Google News, combining automated web scraping, large language model (LLM)-based metadata extraction, and SCImago Media Rankings enrichment. Using the IFMIF-DONES fusion energy project as a case study, we implemented a five-stage data collection pipeline across 81 region-language combinations, yielding 1,482 validated records after a 56% noise reduction. Results are compared against two licensed press databases: MyNews (2,280 records) and ProQuest Newsstream Collection (148 records). Overlap analysis reveals high complementarity, with 76% of Google News records exclusive to this platform. The dataset captures content types absent from proprietary databases, including specialized outlets, institutional communications, and social media posts. However, significant methodological challenges emerge: temporal instability requiring synchronic collection, a 100-result cap per query demanding multi-stage strategies, and unexpected noise including academic PDFs, false positives, and pornographic content infiltrating results through black hat SEO techniques. LLM-assisted extraction proved effective for structured articles but exhibited systematic hallucination patterns requiring validation protocols. We conclude that Google News offers valuable complementary coverage for communication research but demands substantial methodological investment, multi-source triangulation, and robust filtering mechanisms to ensure dataset integrity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.26233v1">Persuadability and LLMs as Legal Decision Tools</a></div>
    <div class="paper-meta">
      📅 2026-04-29
      | 💬 Accepted to 21st International Conference on Artificial Intelligence and Law (ICAIL 2026)
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) are proposed as legal decision assistants, and even first-instance decision-makers, across a range of judicial and administrative contexts, it becomes essential to explore how they answer legal questions, and in particular the factors that lead them to decide difficult questions in one way or another. A specific feature of legal decisions is the need to respond to arguments advanced by contending parties. A legal decision-maker must be able to engage with, and respond to, including through being potentially persuaded by, arguments advanced by the parties. Conversely, they should not be unduly persuadable, influenced by a particularly compelling advocate to decide cases based on the skills of the advocates, rather than the merits of the case. We explore how frontier open- and closed-weights LLMs respond to legal arguments, reporting original experimental results examining how the quality of the advocate making those arguments affects the likelihood that a model will agree with a particular legal point of view, and exploring the factors driving these results. Our results have implications for the feasibility of adopting LLMs across legal and administrative settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.26231v1">ProMax: Exploring the Potential of LLM-derived Profiles with Distribution Shaping for Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2026-04-29
      | 💬 11 pages, 8 figures, accepted by SIGIR 2026
    </div>
    <details class="paper-abstract">
      The remarkable text understanding and generation capabilities of large language models (LLMs) have revitalized the field of general recommendation based on implicit user feedback. Rather than deploying LLMs directly as recommendation models, a more flexible paradigm leverages their ability to interpret users' historical interactions and semantic contexts to extract structured profiles that characterize user preferences. These profiles can be further transformed into actionable high-dimensional representations, serving as powerful signals to augment and strengthen recommendation models. However, the mechanism by which such profiles enhance recommendation performance within the feature space remains insufficiently understood. Moreover, existing studies predominantly rely on nonlinear alignment and fusion strategies to incorporate these profiles, which often lead to semantic loss and fail to fully exploit their potential. To address these limitations, we revisit profiles from a retrieval perspective and propose a simple yet effective recommendation framework built upon distribution shaping (ProMax) in this paper. We begin by employing dense retrieval to uncover the collaborative relationships between user and item profiles within the feature space. Based on this insight, we introduce a dual distribution-reshaping process, in which the profile distribution acts as a guiding signal to steer the recommendation model toward learning user preferences for unseen items beyond the scope of observed interactions. We apply ProMax to four classic recommendation methods on three public datasets. The results indicate that ProMax substantially improves base model performance and outperforms existing LLM-based recommendation approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.09870v2">Vibe Check: Understanding the Effects of LLM-Based Conversational Agents' Personality and Alignment on User Perceptions in Goal-Oriented Tasks</a></div>
    <div class="paper-meta">
      📅 2026-04-29
      | 💬 30 pages, CHI 2026 conference paper (article no. 371)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) enable conversational agents (CAs) to express distinctive personalities, raising new questions about how such designs shape user perceptions. This study investigates how personality expression levels and user-agent personality alignment influence perceptions in goal-oriented tasks. In a between-subjects experiment (N=150), participants completed travel planning with CAs exhibiting low, medium, or high expression across the Big Five traits, controlled via our novel Trait Modulation Keys framework. Results revealed an inverted-U relationship: medium expression produced the most positive evaluations across Intelligence, Enjoyment, Anthropomorphism, Intention to Adopt, Trust, and Likeability, significantly outperforming both extremes. Personality alignment further enhanced outcomes, with Extraversion and Emotional Stability emerging as the most influential traits. Cluster analysis identified three distinct compatibility profiles, with "Well-Aligned" users reporting substantially positive perceptions. These findings demonstrate that personality expression and strategic trait alignment constitute optimal design targets for CA personality, offering design implications as LLM-based CAs become increasingly prevalent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.26209v1">Breaking the Autoregressive Chain: Hyper-Parallel Decoding for Efficient LLM-Based Attribute Value Extraction</a></div>
    <div class="paper-meta">
      📅 2026-04-29
    </div>
    <details class="paper-abstract">
      Some text generation tasks, such as Attribute Value Extraction (AVE), require decoding multiple independent sequences from the same document context. While standard autoregressive decoding is slow due to its sequential nature, the independence between output sequences offers an opportunity for parallelism. We present Hyper-Parallel Decoding, a novel decoding algorithm that accelerates offline decoding by leveraging both shared memory and computation across batches. HPD enables out-of-order token generation through position ID manipulation, significantly improving efficiency. Experiments on AVE show that attribute-value pairs are conditionally independent, enabling us to parallelize value generation within each prompt. By further stacking multiple documents within a single prompt, we can decode in parallel up to 96 tokens per prompt. HPD works with all LLMs, and reduces both inference costs and total inference time by up to 13.8X without compromising output quality, potentially saving hundreds of thousands of dollars on industry AVE tasks. Although designed for attribute extraction, HPD makes no assumptions unique to the AVE domain and can in theory be applied to other scenarios with independent output structures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.26192v1">LLM-Assisted Empirical Software Engineering: Systematic Literature Review and Research Agenda</a></div>
    <div class="paper-meta">
      📅 2026-04-29
    </div>
    <details class="paper-abstract">
      Context: Empirical Software Engineering (ESE) faces increasing challenges due to data scale, methodological complexity, and reproducibility concerns. Large Language Models (LLMs) have emerged as promising tools to support empirical workflows, yet their use remains fragmented, with no comprehensive synthesis to guide responsible adoption. Aims: This study analyzes how LLMs are used in ESE, examining supported tasks, phases of the empirical lifecycle, integration into workflows, reported benefits and limitations, and the extent of reproducibility-related reporting. It also identifies gaps and future research directions. Method: We conducted a systematic literature review of peer-reviewed papers (2020-2025) across 12 leading software engineering venues, resulting in 50 primary studies analyzed through qualitative and quantitative synthesis. Results: We identified 69 LLM-assisted tasks, mainly in mining software repositories and controlled experiments, focusing on classification, filtering, and evaluation. LLMs are used across multiple phases but are concentrated in data processing and analysis. Their integration is largely automation-oriented, with limited decision-support use. Benefits emphasize efficiency and scalability, while limitations include hallucinations, inconsistency, prompt sensitivity, and reproducibility issues. Reporting practices are often incomplete. Conclusion: LLM use in ESE is growing but remains automation-driven, with gaps in human-centered integration and transparency. We outline implications and research agenda for responsible use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.20426v2">Learning to Rewrite Tool Descriptions for Reliable LLM-Agent Tool Use</a></div>
    <div class="paper-meta">
      📅 2026-04-29
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      While most efforts to improve LLM-based tool-using agents focus on the agent itself - through larger models, better prompting, or fine-tuning - agent performance increasingly plateaus due to the quality of the tool interfaces these agents consume. Tool descriptions are often written for human developers and tolerate ambiguity that agents cannot resolve, particularly as the number of candidate tools grows. Existing approaches to improving tool interfaces (1) require re-running a multi-stage per-tool pipeline - synthesizing queries, executing an agent to collect trajectories, annotating trajectories, and prompting a strong LLM multiple times - for every API that enters the catalog, and (2) typically optimize each tool independently, limiting scalability and generalization to unseen tools. We propose Trace-Free+, a curriculum learning framework that progressively transfers supervision from trace-rich settings to trace-free deployment, encouraging the model to internalize reusable patterns of what makes a tool description effective. To support this approach, we construct a large-scale dataset of high-quality tool interfaces derived from real-world APIs through a principled data synthesis workflow. Experiments on widely adopted benchmarks show that Trace-Free+ improves robustness as tool catalogs scale to 150+ candidates - in scaling experiments, reducing accuracy degradation by 29.23% and improving average query-level success by 60.89% on StableToolBench - generalizes across domains without retraining, and provides complementary gains on top of agent fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.26170v1">EvoSelect: Data-Efficient LLM Evolution for Targeted Task Adaptation</a></div>
    <div class="paper-meta">
      📅 2026-04-28
    </div>
    <details class="paper-abstract">
      Adapting large language models (LLMs) to a targeted task efficiently and effectively remains a fundamental challenge. Such adaptation often requires iteratively improving the model toward a targeted task, yet collecting high-quality human-labeled data to support this process is costly and difficult to scale. As a result, synthetic data generation has emerged as a flexible and scalable alternative. One straightforward approach is through an iterative generation-training loop, where candidate data are synthesized through an external generator, the model is updated using these data and the process is repeated over iterations. However, generated samples can be noisy, highly redundant, or even misaligned with the targeted task distribution. Training indiscriminately on such data can dilute useful learning signals and even degrade model performance. To address this, we introduce a refined paradigm, namely an iterative generation-selection-training loop, which incorporates a selection step prior to model updates. Building on this paradigm, we propose EvoSelect, a data-efficient framework to evolve LLM effectively. Given candidate samples produced by the data generator, EvoSelect selects training data by jointly modeling targeted task alignment and diversity. We estimate task relevance through optimal transport with proxy gradient representations, which quantifies how well candidate samples align with the targeted task distribution. To mitigate redundancy, we incorporate a diversification mechanism that promotes coverage of complementary training samples. By interleaving alignment and diversification, EvoSelect enables progressive LLM evolution toward targeted tasks. Extensive experiments on various benchmarks demonstrate that with either weak or strong data generators, EvoSelect consistently improves adaptation efficacy over existing data selection methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.26142v1">ImproBR: Bug Report Improver Using LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-28
    </div>
    <details class="paper-abstract">
      Bug tracking systems play a crucial role in software maintenance, yet developers frequently struggle with low-quality user-submitted reports that omit essential details such as Steps to Reproduce (S2R), Observed Behavior (OB), and Expected Behavior (EB). We propose ImproBR, an LLM-based pipeline that automatically detects and improves bug reports by addressing missing, incomplete, and ambiguous S2R, OB, and EB sections. ImproBR employs a hybrid detector combining fine-tuned DistilBERT, heuristic analysis, and an LLM analyzer, guided by GPT-4o mini with section-specific few-shot prompts and a Retrieval-Augmented Generation (RAG) pipeline grounded in Minecraft Wiki domain knowledge. Evaluated on Mojira, ImproBR improved structural completeness from 7.9% to 96.4%, more than doubled the proportion of executable S2R from 28.8% to 67.6%, and raised fully reproducible bug reports from 1 to 13 across 139 challenging real-world reports.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.26118v1">LLM-Guided Issue Generation from Uncovered Code Segments</a></div>
    <div class="paper-meta">
      📅 2026-04-28
    </div>
    <details class="paper-abstract">
      Developers are increasingly overwhelmed by AI-generated issue reports that lack actionability and reproducibility, eroding trust in automated bug detection tools. In this paper, we present IssueSpecter, an automated tool that finds bugs in uncovered code segments and automatically generates prioritized, actionable issue reports. IssueSpecter combines coverage analysis with LLM-based defect identification, producing structured reports complete with severity ratings, reproduction steps, and suggested fixes. We evaluate IssueSpecter on 13 actively maintained Python projects, generating 10,467 issue reports. Manual annotation of the top-130 ranked issues by IssueSpecter confirms that 84.6% of the LLM-generated issues are valid or warrant further investigation, with only 15.4% false positives. LLM-based ranking outperforms rule-based ranking by 50% at P@3 and 41% in MRR. The identified bugs cover a wide variety of types, from logic and boundary errors to security vulnerabilities and state consistency bugs. By ranking issues by priority, IssueSpecter aims to help developers focus their attention on the most impactful bugs first. Finally, we validate IssueSpecter through case studies reproducing real bugs surfaced from its generated issue reports, demonstrating its practical value for automatic bug discovery in open-source Python projects. Compared against CoverUp, a state-of-the-art coverage-driven test generation tool, IssueSpecter achieves a higher bug validity rate (81.0% vs. 76.2%) under identical evaluation conditions, using the same model and the same number of evaluated artifacts per project, while additionally providing structured issue reports with reproduction steps and candidate fixes that are immediately actionable without requiring developers to interpret generated test intent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.20749v8">Can LLM Agents Simulate Multi-Turn Human Behavior? Evidence from Real Online Customer Behavior Data</a></div>
    <div class="paper-meta">
      📅 2026-04-28
    </div>
    <details class="paper-abstract">
      Recent research shows that LLM Agents can generate ``believable'' human behaviors via prompt-only methods, and such agents have been increasingly adopted in downstream applications. However, existing evaluation of these agents only focuses on qualitative believability (whether human raters think they are accurate), leaving open questions of whether LLM agents can accurately generate step-by-step actions mimicking a particular human's behavior in a multi-turn interaction task. In this work, we take shopping as a case study and present the first large-scale quantitative evaluation of state-of-the-art LLMs' ability to accurately simulate human behavior. Using real-world data from 31,865 online shopping sessions containing 230,965 user actions, our evaluation reveals that prompt-based LLMs (DeepSeek-R1, Llama, Claude) achieve only 11.86% accuracy in generating human actions, highlighting a substantial gap in actual behavioral accuracy. Through experiments, we also showcase that strategies as simple as fine-tuning LLMs on real human click-through data augmented with synthesized reasoning traces can greatly enhance models' performance. The fine-tuned Qwen2.5-7B achieves 17.26% action generation accuracy and 33.86% F1 score on final purchase prediction, representing substantial improvements of 5.4% and 13.85% over prompt-only baselines. This work establishes the first rigorous benchmark for human behavior simulation and provides actionable insights for developing more accurate LLM agents for future downstream applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.16131v2">The Fools are Certain; the Wise are Doubtful: Exploring LLM Confidence in Code Completion</a></div>
    <div class="paper-meta">
      📅 2026-04-28
      | 💬 32 pages, 10 figures, 1 table
    </div>
    <details class="paper-abstract">
      Code completion entails the task of providing missing tokens given a surrounding context. It can boost developer productivity while providing a powerful code discovery tool. Following the Large Language Model (LLM) wave, code completion has been approached with diverse LLMs fine-tuned on code (code LLMs). The performance of code LLMs can be assessed with downstream and intrinsic metrics. Downstream metrics are usually employed to evaluate the practical utility of a model, but can be unreliable and require complex calculations and domain-specific knowledge. In contrast, intrinsic metrics such as perplexity, entropy, and mutual information, which measure model confidence or uncertainty, are simple, versatile, and universal across LLMs and tasks, and can serve as proxies for functional correctness and hallucination risk in LLM-generated code. Motivated by this, we evaluate the confidence of LLMs when generating code by measuring code perplexity across programming languages, models, and datasets using various LLMs, and a sample of 2254 files from 881 GitHub projects. We find that strongly-typed languages exhibit lower perplexity than dynamically typed languages. Scripting languages also demonstrate higher perplexity. Shell appears universally high in perplexity, whereas Java appears low. Code perplexity depends on the employed LLM; under a fixed model, relative language-level rankings are moderately stable across evaluation corpora. Although code comments often increase perplexity, the language ranking based on perplexity is barely affected by their presence. LLM researchers, developers, and users can employ our findings to assess the benefits and suitability of LLM-based code completion in specific software projects based on how language, model choice, and code characteristics impact model confidence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.26074v1">DAK: Direct-Access-Enabled GPU Memory Offloading with Optimal Efficiency for LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-04-28
    </div>
    <details class="paper-abstract">
      LLM inference is constrained by GPU memory capacity and bandwidth. Tiered memory architectures mitigate this by allowing the GPU to offload memory to the remote tier. However, existing memory offloading frameworks rely on prefetching data into local GPU HBM. This approach underutilizes system resources by introducing HBM contention, squandering memory capacity, and creating pipeline bubbles. We show that enabling direct GPU access to remote memory significantly outperforms prefetching, achieving optimal aggregate system bandwidth. We propose DAK, an end-to-end direct-access memory offloading framework that repurposes the Tensor Memory Accelerator (TMA) to asynchronously fetch offloaded weights and KV caches directly from remote memory into GPU shared memory (SMEM). To maximize remote access performance, DAK introduces a greedy algorithm to determine optimal per-operation offloading ratios, alongside active congestion control and TMA multicast to eliminate interconnect bottlenecks and read amplification. Evaluations across diverse architectures show that DAK achieves near-optimal bandwidth aggregation, with up to 3$\times$ performance gains on NVLink-C2C and 1.8$\times$ on PCIe systems compared to state-of-the-art memory offloading baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07387v2">A Self-Calibrating Framework for Analog Circuit Sizing Using LLM-Derived Analytical Equations</a></div>
    <div class="paper-meta">
      📅 2026-04-28
      | 💬 14 pages, 4 figures, 9 tables. V2: Extended to 5 topology families (8-30 transistors), 3 process nodes, and quantitative comparison against 4 published methods
    </div>
    <details class="paper-abstract">
      We present a design automation framework for analog circuit sizing that produces calibrated, topology-specific analytical equations from raw circuit netlists. A large language model (LLM) derives a complete Python sizing function in which each device dimension is traceable to a specific design rationale - a form of interpretable output absent from existing optimization-based and LLM-based sizing methods. A deterministic calibration loop extracts process-dependent parameters from a single DC operating point simulation, while a prediction-error feedback mechanism compensates for analytical inaccuracies. We validate the framework on circuits ranging from 8 to 30 transistors - spanning two-stage Miller-compensated, current-mirror, folded cascode, nested Miller-compensated, and complementary class-AB output topologies - across three process nodes (40 nm, 90 nm, 180 nm). On matched-specification benchmarks, including the class-AB opamp case, the framework converges in 2-7 simulations. Despite large initial prediction errors, convergence depends on the measurement-feedback architecture, not prediction accuracy. The one-shot calibration automatically captures process-dependent variations, enabling cross-node portability without modification, retraining, or per-process characterization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05190v2">Retrieval-Augmented LLMs for Evidence Localization in Clinical Trial Recruitment from Longitudinal EHR Narratives</a></div>
    <div class="paper-meta">
      📅 2026-04-28
    </div>
    <details class="paper-abstract">
      Screening patients for enrollment is a well-known, labor-intensive bottleneck that leads to under-enrollment and, ultimately, trial failures. Recent breakthroughs in large language models (LLMs) offer a promising opportunity to use artificial intelligence to improve screening. This study systematically explored both encoder- and decoder-based generative LLMs for screening clinical narratives to facilitate clinical trial recruitment. We examined both general-purpose LLMs and medical-adapted LLMs and explored three strategies to alleviate the "Lost in the Middle" issue when handling long documents, including 1) Original long-context: using the default context windows of LLMs, 2) NER-based extractive summarization: converting the long document into summarizations using named entity recognition, 3) RAG: dynamic evidence retrieval based on eligibility criteria. The 2018 N2C2 Track 1 benchmark dataset is used for evaluation. Our experimental results show that the MedGemma model with the RAG strategy achieved the best micro-F1 score of 89.05%, outperforming other models. Generative LLMs have remarkably improved trial criteria that require long-term reasoning across long documents, whereas trial criteria that span a short piece of context (e.g., lab tests) show incremental improvements. The real-world adoption of LLMs for trial recruitment must consider specific criteria for selecting among rule-based queries, encoder-based LLMs, and generative LLMs to maximize efficiency within reasonable computing costs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.24668v2">The Price of Agreement: Measuring LLM Sycophancy in Agentic Financial Applications</a></div>
    <div class="paper-meta">
      📅 2026-04-28
      | 💬 Accepted to ICLR 2026 FinAI Workshop
    </div>
    <details class="paper-abstract">
      Given the increased use of LLMs in financial systems today, it becomes important to evaluate the safety and robustness of such systems. One failure mode that LLMs frequently display in general domain settings is that of sycophancy. That is, models prioritize agreement with expressed user beliefs over correctness, leading to decreased accuracy and trust. In this work, we focus on evaluating sycophancy that LLMs display in agentic financial tasks. Our findings are three-fold: first, we find the models show only low to modest drops in performance in the face of user rebuttals or contradictions to the reference answer, which distinguishes sycophancy that models display in financial agentic settings from findings in prior work. Second, we introduce a suite of tasks to test for sycophancy by user preference information that contradicts the reference answer and find that most models fail in the presence of such inputs. Lastly, we benchmark different modes of recovery such as input filtering with a pretrained LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.25899v1">Pythia: Toward Predictability-Driven Agent-Native LLM Serving</a></div>
    <div class="paper-meta">
      📅 2026-04-28
    </div>
    <details class="paper-abstract">
      As LLM applications grow more complex, developers are increasingly adopting multi-agent architectures to decompose workflows into specialized, collaborative components, introducing structure that constrains agent behavior and exposes useful semantic predictability. Unlike traditional LLM serving, which operates under highly dynamic and uncertain conditions, this structured topology enables opportunities to reduce runtime uncertainty -- yet existing systems fail to exploit it, treating agentic workloads as generic traffic and incurring significant inefficiencies. Our analysis of production traces from an agent-serving platform and an internal coding assistant reveals key bottlenecks, including low prefix cache hit rates, severe resource contention from long-context requests, and substantial queuing delays due to suboptimal scaling. To address these challenges, we propose Pythia, a multi-agent serving system that captures workflow semantics through a simple interface at the serving layer, unlocking new optimization opportunities and substantially improving throughput and job completion time over state-of-the-art baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.25880v1">From Threads to Trajectories: A Multi-LLM Pipeline for Community Knowledge Extraction from GitHub Issue Discussions</a></div>
    <div class="paper-meta">
      📅 2026-04-28
    </div>
    <details class="paper-abstract">
      Resolution of complex post-production issues in large-scale open-source software (OSS) projects requires significant cognitive effort, as developers need to go through long, unstructured and fragmented issue discussion threads before that. In this paper, we present SWE-MIMIC-Bench, an issue trajectory dataset generated from raw GitHub discussions using an automated multi-LLM pipeline. Unlike simple summarization, this pipeline utilizes a group of closed-source LLMs to perform granular tasks: analyzing individual comments with awareness of externally-linked resources, classifying comment analyses into label-specific fields (e.g., root cause, solution plan, implementation progress), and synthesizing label-aware trajectories which capture a structured and coherent narrative of the entire discussion thread. Our pipeline uses five closed-source LLM configurations for distinct purposes: label classification, inline code block and external link summarization, comment analysis, label-specific field classification and trajectory synthesis. By generating concise and reliable trajectories from complex conversation threads, this system can assist developers and researchers of broader software engineering community to understand the experience-driven collaborative approach for issue diagnosis. Furthermore, the generated trajectories can be used to train modern LLM agents to think and act like an expert developer. We evaluated our system on 800 real-world GitHub issues drawn from the SWE-Bench-Pro, SWE-Bench-Multilingual and SWE-Bench-Verified dataset, achieving a 91.7% success rate in extracting 734 high-fidelity reasoning trajectories.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.25866v1">From Syntax to Emotion: A Mechanistic Analysis of Emotion Inference in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-28
      | 💬 18 pages including appendix
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used in emotionally sensitive human-AI applications, yet little is known about how emotion recognition is internally represented. In this work, we investigate the internal mechanisms of emotion recognition in LLMs using sparse autoencoders (SAEs). By analyzing sparse feature activations across layers, we identify a consistent three-phase information flow, in which emotion-related features emerge only in the final phase. We further show that emotion representations comprise both shared features across emotions and emotion-specific features. Using phase-stratified causal tracing, we identify a small set of features that strongly influence emotion predictions, and show that both their number and causal impact vary across emotions; in particular, Disgust is more weakly and diffusely represented than other emotions. Finally, we propose an interpretable and data-efficient causal feature steering method that significantly improves emotion recognition performance across multiple models while largely preserving language modeling ability, and demonstrate that these improvements generalize across multiple emotion recognition datasets. Overall, our findings provide a systematic analysis of the internal mechanisms underlying emotion recognition in LLMs and introduce an efficient, interpretable, and controllable approach for improving model performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.25862v1">RESTestBench: A Benchmark for Evaluating the Effectiveness of LLM-Generated REST API Test Cases from NL Requirements</a></div>
    <div class="paper-meta">
      📅 2026-04-28
      | 💬 Accepted for EASE 2026
    </div>
    <details class="paper-abstract">
      Existing REST API testing tools are typically evaluated using code coverage and crash-based fault metrics. However, recent LLM-based approaches increasingly generate tests from NL requirements to validate functional behaviour, making traditional metrics weak proxies for whether generated tests validate intended behaviour. To address this gap, we present RESTestBench, a benchmark comprising three REST services paired with manually verified NL requirements in both precise and vague variants, enabling controlled and reproducible evaluation of requirement-based test generation. RESTestBench further introduces a requirements-based mutation testing metric that measures the fault-detection effectiveness of a generated test case with respect to a specific requirement, extending the property-based approach of Bartocci et al. . Using RESTestBench, we evaluate two approaches across multiple state-of-the-art LLMs: (i) non-refinement-based generation, and (ii) refinement-based generation guided by interaction with the running SUT. In the refinement experiments, RESTestBench assesses how exposure to the actual implementation, valid or mutated, affects test effectiveness. Our results show that test effectiveness drops considerably when the generator interacts with faulty or mutated code, especially for vague requirements, sometimes negating the benefit of refinement and indicating that incorporating actual SUT behaviour is unnecessary when requirement detail is high.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.25847v1">From Soliloquy to Agora: Memory-Enhanced LLM Agents with Decentralized Debate for Optimization Modeling</a></div>
    <div class="paper-meta">
      📅 2026-04-28
      | 💬 Working Paper
    </div>
    <details class="paper-abstract">
      Optimization modeling underpins real-world decision-making in logistics, manufacturing, energy, and public services, but reliably solving such problems from natural-language requirements remains challenging for current large language models (LLMs). In this paper, we propose \emph{Agora-Opt}, a modular agentic framework for optimization modeling that combines decentralized debate with a read-write memory bank. Agora-Opt allows multiple agent teams to independently produce end-to-end solutions and reconcile them through an outcome-grounded debate protocol, while memory stores solver-verified artifacts and past disagreement resolutions to support training-free improvement over time. This design is flexible across both backbones and methods: it reduces base-model lock-in, transfers across different LLM families, and can be layered onto existing pipelines with minimal coupling. Across public benchmarks, Agora-Opt achieves the strongest overall performance among all compared methods, outperforming strong zero-shot LLMs, training-centric approaches, and prior agentic baselines. Further analyses show robust gains across backbone choices and component variants, and demonstrate that decentralized debate offers a structural advantage over centralized selection by enabling agents to refine candidate solutions through interaction and even recover correct formulations when all initial candidates are flawed. These results suggest that reliable optimization modeling benefits from combining collaborative cross-checking with reusable experience, and position Agora-Opt as a practical and extensible foundation for trustworthy optimization modeling assistance. Our code and data are available at https://github.com/CHIANGEL/Agora-Opt.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11786v2">Evaluating LLM Safety Under Repeated Inference via Accelerated Prompt Stress Testing</a></div>
    <div class="paper-meta">
      📅 2026-04-28
      | 💬 23 pages, 9 figures; editorial and LaTeX revisions for clarity; improved presentation of methodology and results; updated figures, tables, and float placement; clarified temperature sensitivity and deployment-risk analysis; expanded reporting from the same experiments; results unchanged in substance
    </div>
    <details class="paper-abstract">
      Traditional benchmarks for large language models (LLMs), such as HELM and AIR-BENCH, primarily assess safety through breadth-oriented evaluation across diverse tasks and risk categories. However, real-world deployment often exposes a different class of risk: operational failures that arise under repeated inference on identical or near-identical prompts rather than from broad task-level underperformance. In high-stakes settings, response consistency and safety under sustained use are therefore critical. We introduce Accelerated Prompt Stress Testing (APST), a depth-oriented evaluation framework inspired by highly accelerated stress testing in reliability engineering. APST repeatedly samples identical prompts under controlled operational conditions (such as decoding temperature) to surface latent failure modes including hallucinations, refusal inconsistency, and unsafe completions. Rather than treating failures as isolated events, APST models them as stochastic outcomes of repeated inference and uses Bernoulli and binomial formulations to estimate per-inference failure probabilities. Applying APST to multiple instruction-tuned LLMs evaluated on AIR-BENCH 2024--derived safety and security prompts, we find that models with comparable shallow-evaluation scores can exhibit substantially different empirical failure rates under repeated sampling. These results show that single-sample or low-depth evaluation can obscure meaningful differences in deployment-relevant reliability. APST complements existing benchmark methodologies by providing a practical framework for estimating failure frequency under sustained use and comparing safety reliability across models and decoding configurations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11224v3">Agent-Diff: Benchmarking LLM Agents on Enterprise API Tasks via Code Execution with State-Diff-Based Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-04-28
      | 💬 Pre-Print. Under review for KDD 2026
    </div>
    <details class="paper-abstract">
      We present Agent-Diff, a novel benchmarking framework for evaluating agentic Large Language Models (LLMs) on real-world productivity software API tasks via code execution. Agentic LLM performance varies due to differences in models, external tool access, prompt structures, and agentic frameworks. Benchmarks must make fundamental trade-offs between a sandboxed approach that controls for variation in software environments and more ecologically valid approaches employing real services. Agent-Diff attempts to capture the desirable features of both of these approaches by including access to the real API interfaces for software services while sandboxing the environment in which calls are made, processed, and evaluated. This approach relies on two key innovations. The first is a novel state-diff contract, which separates process from outcome - rather than fuzzy trace or parameter matching, we define task success as whether the expected change in environment state was achieved. The second is a novel sandbox built on containerized replicas of enterprise APIs, allowing all models to interact with the same service interfaces through code execution. This enables controlled evaluation against a common set of state-diff contracts while preserving the structure of real-world API interaction. Using the Agent-Diff framework, we provide benchmarks for nine LLMs across 224 tasks utilizing enterprise software workflows. In addition, we evaluate the robustness of the framework with ablation experiments to assess the contribution of access to API documentation on benchmark performance. Code and data: https://github.com/agent-diff-bench/agent-diff.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.25777v1">SpecFed: Accelerating Federated LLM Inference with Speculative Decoding and Compressed Transmission</a></div>
    <div class="paper-meta">
      📅 2026-04-28
      | 💬 IEEE International Symposium on Information Theory (ISIT), 2026
    </div>
    <details class="paper-abstract">
      Federated inference enhances LLM performance in edge computing through weighted averaging of distributed model predictions. However, autoregressive LLM inference requires frequent full-model forward passes across workers, severely limiting decoding throughput. Distributed deployment further aggravates this due to a communication bottleneck: each worker must transmit full token probability distributions per draft token, dominating end-to-end latency. To address these challenges, we introduce speculative decoding to enable parallel LLM processing and propose a top-K compressed transmission scheme with two server-side reconstruction strategies. We theoretically analyze the robustness of our method in terms of local reconstruction error, aggregation bias, and acceptance-rate bias, and derive corresponding bounds. Experiments demonstrate that our scheme achieves high generation fidelity while significantly reducing communication overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.25774v1">CGU-ILALab at FoodBench-QA 2026: Comparing Traditional and LLM-based Approaches for Recipe Nutrient Estimation</a></div>
    <div class="paper-meta">
      📅 2026-04-28
      | 💬 Accepted by the Third Workshop on Patient-oriented Language Processing (CL4Health) at LREC 2026
    </div>
    <details class="paper-abstract">
      Accurate nutrient estimation from unstructured recipe text is an important yet challenging problem in dietary monitoring, due to ambiguous ingredient terminology and highly variable quantity expressions. We systematically evaluate models spanning a wide range of representational capacity, from lexical matching methods (TF-IDF with Ridge Regression), to deep semantic encoders (DeBERTa-v3), to generative reasoning with large language models (LLMs). Under the strict tolerance criteria defined by EU Regulation 1169/2011, our empirical results reveal a clear trade-off between predictive accuracy and computational efficiency. The TF-IDF baseline achieves moderate nutrient estimation performance with near-instantaneous inference, whereas the DeBERTa-v3 encoder performs poorly under task-specific data scarcity. In contrast, few-shot LLM inference (e.g., Gemini 2.5 Flash) and a hybrid LLM refinement pipeline (TF-IDF combined with Gemini 2.5 Flash) deliver the highest validation accuracy across all nutrient categories. These improvements likely arise from the ability of LLMs to leverage pre-trained world knowledge to resolve ambiguous terminology and normalize non-standard units, which remain difficult for purely lexical approaches. However, these gains come at the cost of substantially higher inference latency, highlighting a practical deployment trade-off between real-time efficiency and nutritional precision in dietary monitoring systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07236v4">How Much Heavy Lifting Can an Agent Harness Do?: Measuring the LLM's Residual Role in a Planning Agent</a></div>
    <div class="paper-meta">
      📅 2026-04-28
    </div>
    <details class="paper-abstract">
      Agent harnesses -- the stateful programs that wrap a language model and decide what it sees at each step -- are now known to change end-to-end performance on a fixed model by as much as six times. That raises a question asked less often than it should be: how much of an agent's competence does the harness itself already carry, and how much genuinely still needs the LLM? We externalize a planning harness for noisy Collaborative Battleship into four progressively richer layers -- posterior belief tracking, declarative planning, symbolic reflec tion, and an LLM-backed revision gate -- under a common runtime, taking \emph{win rate} as the primary metric and \emph{F1} as secondary, and pre-specifying \emph{heavy lifting} as the single largest positive marginal to the primary metric. Across 54 games, declarative pla nning carries the heavy lifting ($+24.1$pp win rate over a belief-only harness, zero LLM calls); symbolic reflection is mechanistically real but calibration-sensitive, with signed board-level effects up to $\pm0.140$ F1 that cancel on aggregate; and LLM-backed revision ac tivates on only $4.3\%$ of turns with a bounded, non-monotonic effect. The contribution is methodological: once harness layers are made externally measurable, the LLM's role can be quantified as residual rather than assumed central.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.25699v1">NVLLM: A 3D NAND-Centric Architecture Enabling Edge on-Device LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-04-28
      | 💬 Author version
    </div>
    <details class="paper-abstract">
      The rapid growth of LLMs demands high-throughput, memory-capacity-intensive inference on resource-constrained edge devices, where single-batch decoding remains fundamentally memory-bound. Existing out-of-core GPU-based and SSD-like accelerators are limited by DRAM-bound weight movement and inefficient storage access granularity. We present NVLLM, a 3D NAND-centric inference architecture that offloads feed-forward network (FFN) computation into the Flash while executing attention on lightweight CMOS logic with external DRAM. Through wafer-to-wafer stacking, NVLLM tightly integrates multi-plane 3D NAND with compute pipelines, error correction code (ECC) units, and buffers, enabling page-level FFN weight access without DRAM traversal. All GEMM/GEMV operations are decomposed into dot-product primitives executed by out-of-order PE lanes, operating directly on raw NAND reads with integrated ECC. Attention weights remain in DRAM, and a KV-cache-aware scheduler sustains throughput as the context length grows. Evaluated on OPT and LLaMA models with up to 30B parameters, NVLLM achieves a 16.7$\times$--37.9$\times$ speedup over A800-based out-of-core inference and up to 4.7$\times$ speedup over SSD-like designs, with only 2.7\% CMOS area overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.11110v2">Ti-Audio: The First Multi-Dialectal End-to-End Speech LLM for Tibetan</a></div>
    <div class="paper-meta">
      📅 2026-04-28
    </div>
    <details class="paper-abstract">
      Recent advances in Speech Large Language Models (Speech-LLMs) have made significant progress, greatly enhancing multimodal interaction capabilities.However, their application in low-resource and dialect-diverse environments still faces challenges. The severe scarcity of Tibetan data, coupled with the phonetic differences among its major dialects (Ü-Tsang, Amdo, and Kham), is a prime example of this challenge. This paper proposes Ti-Audio, the first multi-dialectal end-to-end Speech-LLM for Tibetan. To efficiently align speech and text, we introduce a Dynamic Q-Former Adapter that extracts essential acoustic features from variable-length speech, ensuring stable cross-modal alignment even with limited data. At the data level, we leverage mutual assistance among related dialects to alleviate data scarcity and employ a temperature-based sampling strategy to maximize this synergy. Experimental results demonstrate that Ti-Audio achieves state-of-the-art performance on Tibetan benchmarks for automatic speech recognition and speech translation. Our work validates the effectiveness of cross-dialectal cooperation and provides a scalable paradigm for the development of Speech-LLM in low-resource scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.22665v2">Improving LLM Predictions via Inter-Layer Structural Encoders</a></div>
    <div class="paper-meta">
      📅 2026-04-28
      | 💬 18 pages, 3 figures. Equal contribution by first two authors
    </div>
    <details class="paper-abstract">
      The standard practice in Large Language Models (LLMs) is to base predictions on final-layer representations. However, intermediate layers encode complementary task-relevant signals, and the optimal layer is task-dependent, making single-layer usage inherently suboptimal. In this work, we introduce Inter-Layer Structural Encoders (ILSE), a powerful and parameter-efficient post-training framework that learns to aggregate representations from all layers of a frozen LLM through structured inter-layer interactions. Central to ILSE is the Cayley-Encoder, a mathematically grounded module based on expander Cayley graphs that enables efficient and effective inter-layer information propagation. We evaluate ILSE on 13 classification and semantic similarity tasks across 9 pre-trained LLMs ranging from 14M to 8B parameters. ILSE consistently outperforms strong baselines, achieving up to 44% improvements in accuracy and 25% in similarity, while introducing at most 0.1% additional parameters relative to the base LLM size. Furthermore, ILSE is highly data-efficient in few-shot regimes and enables small LLMs to match or exceed the performance of substantially larger models. Notably, it also outperforms LoRA-based fine-tuning despite operating on frozen representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.25665v1">LLM-ReSum: A Framework for LLM Reflective Summarization through Self-Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-04-28
      | 💬 15 pages, 3 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Reliable evaluation of large language model (LLM)-generated summaries remains an open challenge, particularly across heterogeneous domains and document lengths. We conduct a comprehensive meta-evaluation of 14 automatic summarization metrics and LLM-based evaluators across seven datasets spanning five domains, covering documents from short news articles to long scientific, governmental, and legal texts (2K-27K words) with over 1,500 human-annotated summaries. Our results show that traditional lexical overlap metrics (e.g., ROUGE, BLEU) exhibit weak or negative correlation with human judgments, while task-specific neural metrics and LLM-based evaluators achieve substantially higher alignment, especially for linguistic quality assessment. Leveraging these findings, we propose LLM-ReSum, a self-reflective summarization framework that integrates LLM-based evaluation and generation in a closed feedback loop without model finetuning. Across three domains, LLM-ReSum improves low-quality summaries by up to 33% in factual accuracy and 39% in coverage, with human evaluators preferring refined summaries in 89% of cases. We additionally introduce PatentSumEval, a new human-annotated benchmark for legal document summarization comprising 180 expert-evaluated summaries. All code and datasets will be released in GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.22891v2">Quantifying and Mitigating Self-Preference Bias of LLM Judges</a></div>
    <div class="paper-meta">
      📅 2026-04-28
    </div>
    <details class="paper-abstract">
      LLM-as-a-Judge has become a dominant approach in automated evaluation systems, playing critical roles in model alignment, leaderboard construction, quality control, and so on. However, the scalability and trustworthiness of this approach can be substantially distorted by Self-Preference Bias (SPB), which is a directional evaluative deviation in which LLMs systematically favor or disfavor their own generated outputs during evaluation. Existing measurements rely on costly human annotations and conflate generative capability with evaluative stance, and thus are impractical for large-scale deployment in real-world systems. To address this issue, we introduce a fully automated framework to quantifying and mitigating SPB, which constructs equal-quality pairs of responses with negligible quality differences, enabling statistical disentanglement of discriminability from bias propensity without human gold standards. Empirical analysis across 20 mainstream LLMs reveals that advanced capabilities are often uncorrelated, or even negatively correlated, with low SPB. To mitigate this bias, we propose a structured multi-dimensional evaluation strategy grounded in cognitive load decomposition, which reduces SPB by 31.5\% on average.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.25654v1">Progressing beyond Art Masterpieces or Touristic Clichés: how to assess your LLMs for cultural alignment?</a></div>
    <div class="paper-meta">
      📅 2026-04-28
      | 💬 RESOURCEFUL-2026 Workshop at LREC 2026
    </div>
    <details class="paper-abstract">
      Although the cultural (mis)alignment of Large Language Models (LLMs) has attracted increasing attention -- often framed in terms of cultural bias -- until recently there has been limited work on the design and development of datasets for cultural assessment. Here, we review existing approaches to such datasets and identify their main limitations. To address these issues, we propose design guidelines for annotators and report on the construction of a dataset built according to these principles. We further present a series of contrastive experiments conducted with this dataset. The results demonstrate that our design yields test sets with greater discriminative power, effectively distinguishing between models specialized for a given culture and those that are not, ceteris paribus.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.25634v1">The Surprising Universality of LLM Outputs: A Real-Time Verification Primitive</a></div>
    <div class="paper-meta">
      📅 2026-04-28
      | 💬 25 pages, 6 figures, 6 tables, 37 references. Code and data: https://github.com/Evolutionairy-AI/Ranking-Inference
    </div>
    <details class="paper-abstract">
      We report a striking statistical regularity in frontier LLM outputs that enables a CPU-only scoring primitive running at 2.6 microseconds per token, with estimated latency up to 100,000$\times$ (five orders of magnitude) below existing sampling-based detectors. Across six contemporary models from five independent vendors, two generation sizes, and five held-out domains, token rank-frequency distributions converge to the same two-parameter Mandelbrot ranking distribution, with 34 of 36 model-by-domain fits exceeding $R^{2} = 0.94$ and 35 of 36 favoring Mandelbrot over Zipf by AIC. The shared family does not collapse the models into statistical duplicates. Fitted Mandelbrot parameters remain cleanly separable between models: the cross-model spread in $q$ (1.63 to 3.69) exceeds its per-model bootstrap standard deviation (0.03 to 0.10) by more than an order of magnitude, yielding tens of standard deviations of separation per few thousand output tokens. Two capabilities follow. First, statistical model fingerprinting: text from a vendor-delivered LLM can be tested against its claimed model family without cryptographic watermarks or access to model internals, supporting provenance verification and silent-substitution audits. Second, a model-agnostic reference distribution for black-box output assessment, from which we derive a single-pass scoring primitive that composes with model log probabilities when available and degrades to a rank-only mode usable on closed APIs. Pilot results on FRANK, TruthfulQA, and HaluEval map where the primitive helps (lexical anomalies, unsupported entities) and where it structurally cannot (reasoning errors in domain-appropriate vocabulary). We position the primitive as a first-pass triage layer in compound evaluation stacks, not as a replacement for sampling-based or source-conditioned verifiers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18871v6">Periodic Asynchrony: An On-Policy Approach for Accelerating LLM Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-04-28
    </div>
    <details class="paper-abstract">
      Since the introduction of the GRPO algorithm, reinforcement learning~(RL) has attracted increasing attention for LLM post-training, yet training efficiency remains a critical challenge. In mainstream RL frameworks, inference and training are co-located on the same devices, and their synchronous execution prevents concurrent inference and training. In this work, we revisit the strategy of separating inference and training deployment, and propose a \emph{periodically asynchronous} framework that transforms synchronous RL training into an asynchronous producer--consumer pipeline. By synchronising model weights at the beginning of each training iteration and generating all rollouts from the same policy, the proposed framework remains inherently \emph{on-policy}, avoiding the off-policy bias introduced by existing asynchronous approaches without any modification to standard RL algorithms. We further introduce a unified tri-model architecture and a shared-prompt attention mechanism to support efficient asynchronous execution and reduce redundant computation. Experiments on NPU platforms show that the proposed framework achieves around $2\times$ throughput improvement from asynchronous execution, with additional gains from system-level optimisations, substantially outperforming mainstream RL frameworks in end-to-end training throughput while maintaining comparable accuracy. Further validation on GPU platforms confirms that the proposed framework generalises effectively across hardware architectures, indicating its potential for widespread application.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.25601v1">Emotive Architectures: The Role of LLMs in Adjusting Work Environments</a></div>
    <div class="paper-meta">
      📅 2026-04-28
      | 💬 19 pages, 1 Table
    </div>
    <details class="paper-abstract">
      In remote and hybrid work contexts, the integration of physical and digital environments is revolutionizing spatial experiences, collaboration, and interpersonal interactions. This study examines three fundamental spatial conditions: the physical environment, characterized by material and sensory attributes; the virtual environment, influenced by immersive technologies; and their fusion into hybrid environments where digital and physical components interact dynamically. The increasing number of AI tools in contemporary society, extensively utilized in both professional and personal spheres, has led to a varied landscape of developing technologies. For instance, ChatGPT has emerged as one of the most downloaded applications, a statistically substantiated fact that demonstrates the swift incorporation of language-based AI into daily life. It also underscores the function of large language models (LLMs) as meaningful bridges between concepts at reading emotional and behavioral signals via natural language. These models provide real-time modifications such as altering illumination, acoustics, or interface configurations, converting static settings into dynamic, emotionally receptive environments. We investigate the integration of language models into professional settings and their potential to enhance user experience by promoting focus, well-being, and engagement. The study investigates ethical concerns, including privacy, emotional tracking, and user agency, emphasizing the importance of inclusive and transparent design. This research formulates a framework for creating co-adaptive environments that merge technological innovation with human-centered experiences, offering a fresh viewpoint on responsive and supportive hybrid workspaces.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.25580v1">Bye Bye Perspective API: Lessons for Measurement Infrastructure in NLP, CSS and LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-04-28
      | 💬 13 pages, 1 figure, 1 table
    </div>
    <details class="paper-abstract">
      The closure of Perspective API at the end of 2026 discards what has functioned as the de facto standard for automated toxicity measurement in NLP, CSS, and LLM evaluation research. We document the structural dependence that the communities built on this single proprietary tool and discuss how this dependence caused epistemic problems that have affected - and will likely continue to affect - collective research efforts. Perspective's model was periodically updated without versioning or disclosure, its annotation structure reflected a single corporate operationalisation of a contested concept, and its scores were used simultaneously as an evaluation target and an evaluation standard. Its closure leaves behind non-updatable benchmarks, irreproducible results, and ultimately a field at risk of perpetuating these issues by turning to closed-source LLMs. We use Perspective's announced termination as an opportunity to call for an independent, valid, adaptable, and reproducible toxicity and hate speech measurement infrastructure, with the technical and governance requirements outlined in this paper.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.05432v2">AInstein: Can LLMs Solve Research Problems From Parametric Memory Alone?</a></div>
    <div class="paper-meta">
      📅 2026-04-28
    </div>
    <details class="paper-abstract">
      Can large language models solve AI research problems using only their parametric knowledge, without fine-tuning, retrieval, or other external aids? We introduce AInstein, a framework for testing whether LLM agents can generate and refine solutions to research problems through iterative critique loops. A blind study with 20 domain experts on held-out ICLR 2026 problems validates our automated metrics, which we then scale to 1,214 ICLR 2025 papers using an LLM-as-a-judge paradigm. Two metrics capture complementary aspects of performance: Success Rate (does the solution address the problem?) and Rediscovery (does it match the published approach?). LLMs succeed on over 70% of problems, yet strictly rediscover the published solution less than 19% of the time, suggesting genuine problem-solving rather than associative recall. However, this ability has clear limits: models handle familiar methodological territory well but fail when solutions require cross-domain analogical transfer, a pattern we call the parametric knowledge boundary. On the ResearchPlanGen benchmark (2,645 problems), our training-free iterative refinement strategy matches RL finetuning, and a criteria-coverage analysis pins down the ceiling of what test-time refinement alone can achieve. Together, these findings map both the capabilities and the limits of LLMs as autonomous scientific problem-solvers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.25525v1">From Chatbots to Confidants: A Cross-Cultural Study of LLM Adoption for Emotional Support</a></div>
    <div class="paper-meta">
      📅 2026-04-28
      | 💬 28 pages (9 pages main text, 19 pages references and appendices), 14 figures. The first two authors contributed equally
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used not only for instrumental tasks, but as always-available and non-judgmental confidants for emotional support. Yet what drives adoption and how users perceive emotional support interactions across countries remains unknown. To address this gap, we present the first large-scale cross-cultural study of LLM use for emotional support, surveying 4,641 participants across seven countries (USA, UK, Germany, France, Spain, Italy, and The Netherlands). Our results show that adoption rates vary dramatically across countries (from 20% to 59%). Using mixed models that separate cultural effects from demographic composition, we find that: Being aged 25-44, religious, married, and of higher socioeconomic status are predictors of positive perceptions (trust, usage, perceived benefits), with socioeconomic status being the strongest. English-speaking countries consistently show more positive perceptions than Continental European countries. We further collect a corpus of 731 real multilingual prompts from user interactions, showing that users mainly seek help for loneliness, stress, relationship conflicts, and mental health struggles. Our findings reveal that LLM emotional support use is shaped by a complex sociotechnical landscape and call for a broader research agenda examining how these systems can be developed, deployed, and governed to ensure safe and informed access.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.25506v1">Assistants, Not Architects: The Role of LLMs in Networked Systems Design</a></div>
    <div class="paper-meta">
      📅 2026-04-28
    </div>
    <details class="paper-abstract">
      Designing the architecture of modern networked systems requires navigating a large, combinatorial space of hardware, systems, and configuration choices with complex cross-layer interactions. Architects must balance competing objectives such as performance, cost, and deployability while satisfying compatibility and resource constraints, often relying on scattered rules-of-thumb drawn from benchmarks, papers, documentation, and expert experience. This raises a natural question: can large language models (LLMs) reliably perform this kind of architectural reasoning? We find that they cannot. While LLMs produce plausible configurations, they frequently miss critical constraints, encode incorrect assumptions, and exhibit ``stickiness'' to familiar patterns. A natural workaround--iterative validation via simulation or experimentation--is often prohibitively expensive at scale and, in many cases, infeasible, particularly when comparing hardware-dependent alternatives. Motivated by this gap, we present Kepler, a lightweight reasoning framework for architecture design that combines structured, expert-driven specifications with SMT-based optimization. Kepler encodes architecturally significant properties--requirements, incompatibilities, and qualitative trade-offs--about systems, hardware, and workloads as constraints, and synthesizes feasible designs that optimize user-defined objectives. It operates at an abstract level, capturing ``rules-of-thumb'' rather than detailed system behavior, enabling tractable reasoning while preserving key interactions, and provides explanations for its decisions. Through experiments and case studies, we show that Kepler uncovers interactions missed by LLMs and supports systematic, explainable design exploration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.25456v1">An Investigation of Linguistic Biases in LLM-Based Recommendations</a></div>
    <div class="paper-meta">
      📅 2026-04-28
    </div>
    <details class="paper-abstract">
      We investigate linguistic biases in LLM-based restaurant and product recommendations given prompts varying across Southern American English (AE), Indian English (IE), and Code-Switched Hindi-English dialects, using the Yelp Open dataset (Yelp Inc., 2023) and Walmart product reviews dataset (PromptCloud,2020). We add lists of restaurant and product names balanced by cuisine type and product category to the prompts given to the LLM, and we zero-shot prompt the LLMs in a cold-start setting to select the top-20 restaurant and product recommendations from these lists for each of the dialect-varied prompts. We prompt LLMs using different list samples across 20 seeds for better generalization, and aggregate per cuisine-type and per category response counts for each seed, question/prompt, and LLM model. We run mixed-effects regression models for each model family and topic (restaurant/product) with the aggregate response counts as the dependent, and conduct likelihood ratio tests for the fixed effects with post-hoc pairwise testing of estimated marginal means differences, to investigate group-level differences in recommendation counts by model size and dialect type. Results show that dialect plays a role in the type of restaurant selected across the models tested with the mistral-small-3.1 model and both the llama-3.1 family models tested showing more sensitivity to Indian English and Code-Switched prompts. In terms of product recommendations, the llama-3.1-70B-model is particularly sensitive to Code-Switched prompts in four out of seven categories, and more beauty and home category recommendations are seen when using the Indian English and Code-Switched prompts for larger and smaller models, respectively. No broad trends are seen in the model-size based differences, with differing recommendations based on model sizes conditioned by the type of dialect.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.25423v1">Do LLMs Capture Embodied Cognition and Cultural Variation? Cross-Linguistic Evidence from Demonstratives</a></div>
    <div class="paper-meta">
      📅 2026-04-28
      | 💬 Accepted to ACL 2026
    </div>
    <details class="paper-abstract">
      Do large language models (LLMs) truly acquire embodied cognition and cultural conventions from text? We introduce demonstratives, fundamental spatial expressions like "this/that" in English and "zhè/nà" in Chinese, as a novel probe for grounded knowledge. Using 6,400 responses from 320 native speakers, we establish a human baseline: English speakers reliably distinguish proximal-distal referents but struggle with perspective-taking, while Chinese speakers switch perspectives fluently but tolerate distal ambiguity. In contrast, five state-of-the-art LLMs fail to inherently understand the proximal-distal contrast and show no cultural differences, defaulting to English-centric reasoning. Our study contributes (i) a new task, based on demonstratives, as a new lens for evaluating embodied cognition and cultural conventions; (ii) empirical evidence of cross-cultural asymmetries in human interpretation; (iii) a new perspective on the egocentric-sociocentric debate, showing both orientations coexist but vary across languages; and (iv) a call to address individual variation in future model design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.25421v1">FED-FSTQ: Fisher-Guided Token Quantization for Communication-Efficient Federated Fine-Tuning of LLMs on Edge Devices</a></div>
    <div class="paper-meta">
      📅 2026-04-28
      | 💬 19 pages, 15 figures
    </div>
    <details class="paper-abstract">
      Federated fine-tuning provides a practical route to adapt large language models (LLMs) on edge devices without centralizing private data, yet in mobile deployments the training wall-clock is often bottlenecked by straggler-limited uplink communication under heterogeneous bandwidth and intermittent participation. Although parameter-efficient fine-tuning (PEFT) reduces trainable parameters, per-round payloads remain prohibitive in non-IID regimes, where uniform compression can discard rare but task-critical signals. We propose Fed-FSTQ, a Fisher-guided token quantization system primitive for communication-efficient federated LLM fine-tuning. Fed-FSTQ employs a lightweight Fisher proxy to estimate token sensitivity, coupling importance-aware token selection with non-uniform mixed-precision quantization to allocate higher fidelity to informative evidence while suppressing redundant transmission. The method is model-agnostic, serves as a drop-in module for standard federated PEFT pipelines, e.g., LoRA, without modifying the server aggregation rule, and supports bandwidth-heterogeneous clients via compact sparse message packing. Experiments on multilingual QA and medical QA under non-IID partitions show that Fed-FSTQ reduces cumulative uplink traffic required to reach a fixed quality threshold by 46x relative to a standard LoRA baseline, and improves end-to-end wall-clock time-to-accuracy by 52%. Furthermore, enabling Fisher-guided token reduction at inference yields up to a 1.55x end-to-end speedup on NVIDIA Jetson-class edge devices, demonstrating deployability under tight resource constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.13400v6">Justice in Judgment: Unveiling (Hidden) Bias in LLM-assisted Peer Reviews</a></div>
    <div class="paper-meta">
      📅 2026-04-28
      | 💬 Findings of ACL 2026
    </div>
    <details class="paper-abstract">
      The adoption of large language models (LLMs) is transforming the peer review process, from assisting reviewers in writing detailed evaluations to generating entire reviews automatically. While these capabilities offer new opportunities, they also raise concerns about fairness and reliability. In this paper, we investigate bias in LLM-generated peer reviews through controlled interventions on author metadata, including affiliation, gender, seniority, and publication history. Our analysis consistently shows a strong affiliation bias favoring authors from highly ranked institutions. We also identify directional preferences associated with seniority and prior publication record, which can influence acceptance decisions for borderline papers. Gender effects are smaller but present in several models. Notably, implicit biases become more pronounced when examining token-level soft ratings, suggesting that alignment may mask but not fully eliminate underlying preferences
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19684v3">LLM-Assisted Authentication and Fraud Detection</a></div>
    <div class="paper-meta">
      📅 2026-04-28
      | 💬 20 pages, 7 figures, 2 tables
    </div>
    <details class="paper-abstract">
      User authentication and fraud detection face growing challenges as digital systems expand and adversaries adopt increasingly sophisticated tactics. Traditional knowledge-based authentication remains rigid, requiring exact word-for-word string matches that fail to accommodate natural human memory and linguistic variation. Meanwhile, fraud-detection pipelines struggle to keep pace with rapidly evolving scam behaviors, leading to high false-positive rates and frequent retraining cycles required. This work introduces two complementary LLM-enabled solutions, namely, an LLM-assisted authentication mechanism that evaluates semantic correctness rather than exact wording, supported by document segmentation and a hybrid scoring method combining LLM judgement with cosine-similarity metrics and a RAG-based fraud-detection pipeline that grounds LLM reasoning in curated evidence to reduce hallucinations and adapt to emerging scam patterns without model retraining. Experiments show that the authentication system accepts 99.5% of legitimate non-exact answers while maintaining a 0.1% false-acceptance rate, and that the RAG-enhanced fraud detection reduces false positives from 17.2% to 3.5%. Together, these findings demonstrate that LLMs can significantly improve both usability and robustness in security workflows, offering a more adaptive , explainable, and human-aligned approach to authentication and fraud detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.16812v2">Introspection Adapters: Training LLMs to Report Their Learned Behaviors</a></div>
    <div class="paper-meta">
      📅 2026-04-28
    </div>
    <details class="paper-abstract">
      When model developers or users fine-tune an LLM, this can induce behaviors that are unexpected, deliberately harmful, or hard to detect. It would be far easier to audit LLMs if they could simply describe their behaviors in natural language. Here, we study a scalable approach to rapidly identify learned behaviors of many LLMs derived from a shared base LLM. Given a model $M$, our method works by finetuning models $M_i$ from $M$ with implanted behaviors $b_i$; the $(M_i, b_i)$ pairs serve as labeled training data. We then train an introspection adapter (IA): a single LoRA adapter jointly trained across the finetunes $M_i$ to cause them to verbalize their implanted behaviors. We find that this IA induces self-description of learned behaviors even in finetunes of $M$ that were trained in very different ways from the $M_i$. For example, IAs generalize to AuditBench, achieving state-of-the-art at identifying explicitly hidden concerning behaviors. IAs can also be used to detect encrypted finetuning API attacks. They scale favorably with model size and training data diversity. Overall, our results suggest that IAs are a scalable, effective, and practically useful approach to auditing fine-tuned LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.22117v2">PermaFrost-Attack: Stealth Pretraining Seeding(SPS) for planting Logic Landmines During LLM Training</a></div>
    <div class="paper-meta">
      📅 2026-04-28
    </div>
    <details class="paper-abstract">
      Aligned large language models (LLMs) remain vulnerable to adversarial manipulation, and their reliance on web-scale pretraining creates a subtle but consequential attack surface. We study Stealth Pretraining Seeding (SPS), a threat model in which adversaries distribute small amounts of poisoned content across stealth websites, increasing the likelihood that such material is absorbed into future training corpora derived from sources such as Common Crawl. Because each individual payload is tiny, diffuse, and superficially benign, the attack is difficult to detect during dataset construction or filtering. The result is a latent form of poisoning that remains largely invisible under standard evaluation, yet can later be activated by a precise trigger such as <00TRIGGER00>. We call this attack PermaFrost, reflecting its latent and reactivatable nature. We study it through PermaFrost-Attack, a controlled framework for latent conceptual poisoning, together with three geometric diagnostics: Thermodynamic Length, Spectral Curvature, and the Infection Traceback Graph. Across multiple model families and scales, we show that this controlled SPS proxy can induce persistent unsafe behavior that often remains hidden under standard evaluation. Our results identify SPS as a practical and underappreciated threat to future foundation models. This paper introduces a novel geometric diagnostic lens for systematically examining latent model behavior, providing a principled foundation for detecting, characterizing, and understanding vulnerabilities that may remain invisible under standard evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.25318v1">Cutscene Agent: An LLM Agent Framework for Automated 3D Cutscene Generation</a></div>
    <div class="paper-meta">
      📅 2026-04-28
      | 💬 27 pages excluding appendix
    </div>
    <details class="paper-abstract">
      Cutscenes are carefully choreographed cinematic sequences embedded in video games and interactive media, serving as the primary vehicle for narrative delivery, character development, and emotional engagement. Producing cutscenes is inherently complex: it demands seamless coordination across screenwriting, cinematography, character animation, voice acting, and technical direction, often requiring days to weeks of collaborative effort from multidisciplinary teams to produce minutes of polished content. In this work, we present Cutscene Agent, an LLM agent framework for automated end-to-end cutscene generation. The framework makes three contributions: (1)~a Cutscene Toolkit built on the Model Context Protocol (MCP) that establishes \emph{bidirectional} integration between LLM agents and the game engine -- agents not only invoke engine operations but continuously observe real-time scene state, enabling closed-loop generation of editable engine-native cinematic assets; (2)~a multi-agent system where a director agent orchestrates specialist subagents for animation, cinematography, and sound design, augmented by a visual reasoning feedback loop for perception-driven refinement; and (3)~CutsceneBench, a hierarchical evaluation benchmark for cutscene generation. Unlike typical tool-use benchmarks that evaluate short, isolated function calls, cutscene generation requires long-horizon, multi-step orchestration of dozens of interdependent tool invocations with strict ordering constraints -- a capability dimension that existing benchmarks do not cover. We evaluate a range of LLMs on CutsceneBench and analyze their performance across this challenging task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.25317v1">FusionCIM: Accelerating LLM Inference with Fusion-Driven Computing-in-Memory Architecture</a></div>
    <div class="paper-meta">
      📅 2026-04-28
      | 💬 7 Pages, 10 figures
    </div>
    <details class="paper-abstract">
      In this paper, we propose FusionCIM, an operator-fusion-driven compute-in-memory (CIM) accelerator architecture for efficient and scalable LLM inference, with three key innovations: (1) a hybrid CIM pipeline architecture that maps QKT computation on inner-product-based CIM (IP-CIM) and PV aggregation on outer-product-based CIM (OP-CIM) for efficient matrix multiplications fusion; (2) a QO-stationary dataflow that eliminates repeated KV loading in CIM and K-matrix access in buffer under transpose fusion, significantly improving data reuse on chip; and (3) a pattern-aware online-softmax mechanism that exploits distribution regularities of attention scores to reduce exponential rescaling overhead for non-linear fusion. Experimental results on LLaMA-3 model show that FusionCIM achieves up to 3.86x energy saving, and 1.98x speedup compared with prior SOTA CIM-based designs with 29.4 TOPS/W energy efficiency at the system level.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.17697v2">Pimp My LLM: Leveraging Variability Modeling to Tune Inference Hyperparameters</a></div>
    <div class="paper-meta">
      📅 2026-04-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are being increasingly used across a wide range of tasks. However, their substantial computational demands raise concerns about the energy efficiency and sustainability of both training and inference. Inference, in particular, dominates total compute usage, making its optimization crucial. Recent research has explored optimization techniques and analyzed how configuration choices influence energy consumption. Yet, the vast configuration space of inference servers makes exhaustive empirical evaluation infeasible due to combinatorial explosion. In this paper, we introduce a new perspective on this problem by treating LLMs as configurable systems and applying variability management techniques to systematically analyze inference-time configuration choices. We evaluate our approach on the Hugging Face Transformers library by representing generation hyperparameters and their constraints using a feature-based variability model, sampling representative configurations, measuring their energy consumption, latency, accuracy, and learning predictive models from the collected data. Our results show that variability modeling effectively manages the complexity of LLM inference configurations. It enables systematic analysis of hyperparameters effects and interactions, reveals trade-offs, and supports prediction of inference behavior from a limited number of measurements. Overall, this work opens a new research direction that bridges software engineering and machine learning by leveraging variability modeling for the efficient and sustainable configuration of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.19669v4">DiffAdapt: Difficulty-Adaptive Reasoning for Token-Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-04-28
      | 💬 ICLR 26
    </div>
    <details class="paper-abstract">
      Recent reasoning Large Language Models (LLMs) demonstrate remarkable problem-solving abilities but often generate long thinking traces whose utility is unclear. Our work aims to improve their efficiency, enabling them to reach high performance without overthinking. First, we analyze the entropy of token probabilities in reasoning traces. Across three models, we observe a consistent U-shaped entropy pattern: high entropy on easy problems despite high accuracy, low entropy on problems with medium difficulty, and high entropy on hard problems reflecting uncertainty. Specifically, we notice 22--25\% entropy reduction from easy to medium difficulty regions, suggesting an {overthinking} phenomenon on easy instances. Building on these insights, we introduce \textbf{DiffAdapt}, a lightweight framework that selects Easy/Normal/Hard inference strategies per question based on their difficulty and reasoning trace entropy. Each inference strategy consists of a fixed prompt, temperature and maximum token length. In contrast to existing efficiency optimization methods, our approach does not fine-tune base LLM but a small probe that classifies LLM's final hidden state, allowing inexpensive adaptation. We comprehensively evaluate our method on five models and eight benchmarks. Our method achieves comparable or improved accuracy while reducing token usage by up to 22.4\%, establishing a practical path toward compute-efficient reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.14807v2">The LLM Fallacy: Misattribution in AI-Assisted Cognitive Workflows</a></div>
    <div class="paper-meta">
      📅 2026-04-28
    </div>
    <details class="paper-abstract">
      The rapid integration of large language models (LLMs) into everyday workflows has transformed how individuals perform cognitive tasks such as writing, programming, analysis, and multilingual communication. While prior research has focused on model reliability, hallucination, and user trust calibration, less attention has been given to how LLM usage reshapes users' perceptions of their own capabilities. This paper introduces the LLM fallacy, a cognitive attribution error in which individuals misinterpret LLM-assisted outputs as evidence of their own independent competence, producing a systematic divergence between perceived and actual capability. We argue that the opacity, fluency, and low-friction interaction patterns of LLMs obscure the boundary between human and machine contribution, leading users to infer competence from outputs rather than from the processes that generate them. We situate the LLM fallacy within existing literature on automation bias, cognitive offloading, and human-AI collaboration, while distinguishing it as a form of attributional distortion specific to AI-mediated workflows. We propose a conceptual framework of its underlying mechanisms and a typology of manifestations across computational, linguistic, analytical, and creative domains. Finally, we examine implications for education, hiring, and AI literacy, and outline directions for empirical validation. We also provide a transparent account of human-AI collaborative methodology. This work establishes a foundation for understanding how generative AI systems not only augment cognitive performance but also reshape self-perception and perceived expertise.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.17262v2">Quantifying and Mitigating Socially Desirable Responding in LLMs: A Desirability-Matched Graded Forced-Choice Psychometric Study</a></div>
    <div class="paper-meta">
      📅 2026-04-28
    </div>
    <details class="paper-abstract">
      Human self-report questionnaires are increasingly used in NLP to benchmark and audit large language models (LLMs), from persona consistency to safety and bias assessments. Yet these instruments presume honest responding; in evaluative contexts, LLMs can instead gravitate toward socially preferred answers-a form of socially desirable responding (SDR)-biasing questionnaire-derived scores and downstream conclusions. We propose a psychometric framework to quantify and mitigate SDR in questionnaire-based evaluation of LLMs. To quantify SDR, the same inventory is administered under HONEST versus FAKE-GOOD instructions, and SDR is computed as a direction-corrected standardized effect size from item response theory (IRT)-estimated latent scores. This enables comparisons across constructs and response formats, as well as against human instructed-faking benchmarks. For mitigation, we construct a graded forced-choice (GFC) Big Five inventory by selecting 30 cross-domain pairs from an item pool via constrained optimization to match desirability. Across nine instruction-following LLMs evaluated on synthetic personas with known target profiles, Likert-style questionnaires show consistently large SDR, whereas desirability-matched GFC substantially attenuates SDR while largely preserving the recovery of the intended persona profiles. These results highlight a model-dependent SDR-recovery trade-off and motivate SDR-aware reporting practices for questionnaire-based benchmarking and auditing of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.25224v1">ValueAlpha: Agreement-Gated Stress Testing of LLM-Judged Investment Rationales Before Returns Are Observable</a></div>
    <div class="paper-meta">
      📅 2026-04-28
      | 💬 9 pages, Submitted to IEEE Computational Intelligence in Financial Engineering and Economics (CIFEr) 2026, Tokyo, Japan
    </div>
    <details class="paper-abstract">
      Long-horizon investment decisions create a pre-realization evaluation problem: realized returns are the eventual arbiter of investment quality, but they arrive too late and are too noisy to guide many model-development and governance decisions. LLM judges offer a tempting substitute for pre-deployment evaluation of AI-finance systems, but unvalidated judges may reward verbosity, confidence, or rubric mimicry rather than financial judgment. This paper introduces \textbf{ValueAlpha}, a preregistered agreement-gated stress-test protocol for deciding when LLM-judged investment-rationale claims are publishable, qualified, or invalid. In a controlled market-state capital-allocation prototype with 1,000 honest decision cycles and 100 preregistered adversarial controls (1,100 trajectories, 5,500 judge calls), ValueAlpha clears the aggregate agreement gate at \(\barκ_w = 0.7168\) but prevents several overclaims. Lower-rank systems collapse into a tie-class, one rubric dimension fails the per-dimension gate (\texttt{constraint\_awareness}, \(\barκ_w = 0.2022\)), single-judge rankings are family-dependent, and terse-correct rationales receive a \(Δ= -2.81\) rubric-point penalty relative to honest rationales. A targeted anchor-specificity probe further shows that financial constructs such as constraint awareness are operationally load-bearing. The contribution is therefore not a leaderboard and not a claim to measure true investment skill. ValueAlpha is a pre-calibration metrology layer for AI-finance evaluation: it determines whether a proposed LLM-judge-based investment-rationale claim is stable enough, agreed enough, and uncontaminated enough to be reported at all.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.25183v1">Hardware Generation and Exploration of Lookup Table-Based Accelerators for 1.58-bit LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-04-28
      | 💬 Presented as ISPASS 2026
    </div>
    <details class="paper-abstract">
      Ternary weight quantization (e.g., BitNet b1.58) offers a promising path to mitigate the memory bandwidth bottleneck in Large Language Model (LLM) inference. However, conventional compute platforms lack native support for ternary-weight arithmetic, often relying on inefficient dequantization. Lookup table (LUT)-based hardware architectures provide an effective alternative by replacing multiplications with conditional additions, but their design space remains largely unexplored. Existing designs rely on heuristic parameter selection, lacking a systematic understanding of the architectural trade-offs. This work addresses this gap by formalizing the design space of ternary LUT-based accelerators and presenting an open-source hardware generator coupled with an analytical cost model, validated against synthesis in TSMC 16nm technology. By spanning the full architectural space, this framework not only enables rapid design space exploration but also establishes a common footing for fair cross-design evaluation, which was previously hindered by inconsistent instantiations across published accelerators. Using this framework, we challenge several assumptions and design choices in recent literature. We demonstrate that the optimal architecture is fundamentally governed by the activation data type: while LUT-based reuse offers significant gains for high-cost arithmetic (e.g., FP16), it yields diminishing returns for small integer types. Furthermore, we show that maximizing core size consistently improves area density compared to highly tiled approaches. Our optimized designs achieve a 2.2x area reduction compared to multiplier-based baselines. Moreover, by benchmarking state-of-the-art implementations against our model, we reveal that correcting suboptimal parameters yields up to a 1.2x area improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.25149v1">Semantic Layers for Reliable LLM-Powered Data Analytics: A Paired Benchmark of Accuracy and Hallucination Across Three Frontier Models</a></div>
    <div class="paper-meta">
      📅 2026-04-28
    </div>
    <details class="paper-abstract">
      LLMs deployed for natural-language querying of analytical databases suffer from two intertwined failures - incorrect answers and confident hallucinations - both rooted in the same cause: the model is forced to infer business semantics that the schema does not encode. We test whether supplying those semantics as context closes the gap. We benchmark three frontier LLMs (Claude Opus 4.7, Claude Sonnet 4.6, GPT-5.4) on 100 natural-language questions over the Cleaned Contoso Retail Dataset in ClickHouse, using a paired single-shot protocol. Each model is evaluated twice: once given only the warehouse schema, and once given the schema plus a 4 KB hand-authored markdown document describing the dataset's measures, conventions, and disambiguation rules. Adding the document improves accuracy by +17 to +23 percentage points across all three models. With it, the three models are statistically indistinguishable (67.7-68.7%); without it, they are also indistinguishable (45.5-50.5%). Every cross-cluster comparison is significant at p < 0.01. The presence of the semantic-layer document accounts for essentially all of the significant variance; model choice within tier does not. We interpret this as a structural result: explicit business semantics suppress the dominant class of text-to-SQL errors not by making the model more capable, but by changing what the model is being asked to do.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.25135v1">FAMA: Failure-Aware Meta-Agentic Framework for Open-Source LLMs in Interactive Tool Use Environments</a></div>
    <div class="paper-meta">
      📅 2026-04-28
      | 💬 Accepted to ACL 2026 Findings
    </div>
    <details class="paper-abstract">
      Large Language Models are being increasingly deployed as the decision-making core of autonomous agents capable of effecting change in external environments. Yet, in conversational benchmarks, which simulate real-world customer-centric issue resolution scenarios, these agents frequently fail due to the cascading effects of incorrect decision-making. These challenges are particularly pronounced for open-source LLMs with smaller parameter sizes, limited context windows, and constrained inference budgets, which contribute to increased error accumulation in agentic settings. To tackle these challenges, we present the Failure-Aware Meta-Agentic (FAMA) framework. FAMA operates in two stages: first, it analyzes failure trajectories from baseline agents to identify the most prevalent errors; second, it employs an orchestration mechanism that activates a minimal subset of specialized agents tailored to address these failures by injecting a targeted context for the tool-use agent before the decision-making step. Experiments across open-source LLMs demonstrate performance gains up to 27% across evaluation modes over standard baselines. These results highlight that targeted curation of context through specialized agents to address common failures is a valuable design principle for building reliable, multi-turn tool-use LLM agents that simulate real-world conversational scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.25098v1">Doing More With Less: Revisiting the Effectiveness of LLM Pruning for Test-Time Scaling</a></div>
    <div class="paper-meta">
      📅 2026-04-28
    </div>
    <details class="paper-abstract">
      While current Large Language Models (LLMs) exhibit remarkable reasoning capabilities through test-time compute scaling (TTS), their massive parameter counts and high inference costs have motivated the development of pruning methods that can reduce model size without sacrificing performance. However, specific to reasoning LLMs, prior work has shown that structured pruning (methods which removes entire set of layer blocks), significantly degrades TTS reasoning performance. In this work, we revisit this assumption and instead investigate whether unstructured pruning (methods that carefully remove only certain redundant/detrimental weights) exhibits similar limitations. Surprisingly, our extensive experiments across four reasoning benchmarks on two reasoning LLMs: s1.1-7B and Qwen3-8B, consistently show that unstructured pruning augments TTS performance compared to structured pruning, and at times can even outperform the unpruned full-weight LLMs. Furthermore, we also empirically study the impact of different layer-wise sparsity allocation strategies, which are an important parametric choice for instantiating unstructured pruning methods. These findings challenge the conventional notion that pruning always reduces TTS performance and in fact, suggest that carefully undertaken pruning can improve TTS effectiveness even further.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.25080v1">CacheFlow: Efficient LLM Serving with 3D-Parallel KV Cache Restoration</a></div>
    <div class="paper-meta">
      📅 2026-04-28
      | 💬 11 pages, 10 figures
    </div>
    <details class="paper-abstract">
      KV cache restoration has emerged as a dominant bottleneck in serving long-context LLM workloads, including multi-turn conversations, retrieval-augmented generation, and agentic pipelines. Existing approaches treat restoration as a per-request tradeoff between recomputation and I/O transfer, recomputing KV states from scratch or offloading them from external storage (e.g., CPU memory or remote machines). However, existing advances fail to exploit parallelism across tokens, layers, and distributed deployments, and critically ignore resource contention under batched serving. We present CacheFlow, a KV cache restoration framework that rethinks cache restoration as a multi-dimensional parallel execution problem. CacheFlow introduces a unified 3D parallelism abstraction across tokens, layers, and GPUs, enabling fine-grained overlap of recomputation and I/O along the structural dependencies of transformer inference. At the core of CacheFlow is a batch-aware two-pointer scheduler that jointly optimizes compute and I/O allocation across requests by prioritizing operations with the highest marginal reduction in recomputation cost. Our evaluations show that CacheFlow reduces Time-To-First-Token (TTFT) by 10%-62% over existing advances across diverse models, workloads, and hardware.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.13766v5">A Blueprint for AI-Driven Software Quality: Integrating LLMs with Established Standards</a></div>
    <div class="paper-meta">
      📅 2026-04-28
      | 💬 16 pages, 2 Table, 7 Figures
    </div>
    <details class="paper-abstract">
      Software Quality Assurance (SQA) is critical for delivering reliable, secure, and efficient software products. The Software Quality Assurance Process aims to provide assurance that work products and processes comply with predefined provisions and plans. Recent advancements in Large Language Models (LLMs) present new opportunities to enhance existing SQA processes by automating tasks like requirement analysis, code review, test generation, and compliance checks. Simultaneously, established standards such as ISO/IEC 12207, ISO/IEC 25010, ISO/IEC 5055, ISO 9001/ISO/IEC 90003, CMMI, and TMM provide structured frameworks for ensuring robust quality practices. This paper surveys the intersection of LLM-based SQA methods and these recognized standards, highlighting how AI-driven solutions can augment traditional approaches while maintaining compliance and process maturity. We first review the foundational software quality standards and the technical fundamentals of LLMs in software engineering. Next, we explore various LLM-based SQA applications, including requirement validation, defect detection, test generation, and documentation maintenance. We then map these applications to key software quality frameworks, illustrating how LLMs can address specific requirements and metrics within each standard. Empirical case studies and open-source initiatives demonstrate the practical viability of these methods. At the same time, discussions on challenges (e.g., data privacy, model bias, explainability) underscore the need for deliberate governance and auditing. Finally, we propose future directions encompassing adaptive learning, privacy-focused deployments, multimodal analysis, and evolving standards for AI-driven software quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.25053v1">Analyzing LLM Reasoning to Uncover Mental Health Stigma</a></div>
    <div class="paper-meta">
      📅 2026-04-27
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) are increasingly being explored for mental health applications, recent studies reveal that they can exhibit stigma toward individuals with psychological conditions. Existing evaluations of this stigma primarily rely on multiple-choice questions (MCQs), which fail to capture the biases embedded within the models' underlying logic. In this paper, we analyze the intermediate reasoning steps of LLMs to uncover hidden stigmatizing language and the internal rationales driving it. We leverage clinical expertise to categorize common patterns of stigmatizing language directed at individuals with psychological conditions and use this framework to identify and tag problematic statements in LLM reasoning. Furthermore, we rate the severity of these statements, distinguishing between overt prejudice and more subtle, less immediately harmful biases. To broaden the reasoning domain and capture a wider array of patterns, we also extend an existing mental health stigma benchmark by incorporating additional psychological conditions. Our findings demonstrate that evaluating model reasoning not only exposes substantially more stigma than traditional MCQ-based methods but it helps to identify the flaws in the LLMs' logic and their understanding of mental health conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.15954v2">MobileLLM-Flash: Latency-Guided On-Device LLM Design for Industry Scale Deployment</a></div>
    <div class="paper-meta">
      📅 2026-04-27
      | 💬 Accepted to ACL Industry Track 2026
    </div>
    <details class="paper-abstract">
      Real-time AI experiences call for on-device large language models (OD-LLMs) optimized for efficient deployment on resource-constrained hardware. The most useful OD-LLMs produce near-real-time responses and exhibit broad hardware compatibility, maximizing user reach. We present a methodology for designing such models using hardware-in-the-loop architecture search under mobile latency constraints. This system is amenable to industry-scale deployment: it generates models deployable without custom kernels and compatible with standard mobile runtimes like Executorch. Our methodology avoids specialized attention mechanisms and instead uses attention skipping for long-context acceleration. Our approach jointly optimizes model architecture (layers, dimensions) and attention pattern. To efficiently evaluate candidates, we treat each as a pruned version of a pretrained backbone with inherited weights, thereby achieving high accuracy with minimal continued pretraining. We leverage the low cost of latency evaluation in a staged process: learning an accurate latency model first, then searching for the Pareto-frontier across latency and quality. This yields MobileLLM-Flash, a family of foundation models (350M, 650M, 1.4B) for efficient on-device use with strong capabilities, supporting up to 8k context length. MobileLLM-Flash delivers up to 1.8x and 1.6x faster prefill and decode on mobile CPUs with comparable or superior quality. Our analysis of Pareto-frontier design choices offers actionable principles for OD-LLM design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03266v2">Benchmarking and Adapting On-Device LLMs for Clinical Decision Support</a></div>
    <div class="paper-meta">
      📅 2026-04-27
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have rapidly advanced in clinical decision-making, yet the deployment of proprietary systems is hindered by privacy concerns and reliance on cloud-based infrastructure. Open-source alternatives allow local inference but often have large model sizes that limit their use in resource-constrained clinical settings. Here, we benchmark on-device LLMs from the gpt-oss (20b, 120b), Qwen3.5 (9B, 27B, 35B), and Gemma 4 (31B) families across three representative clinical tasks: general disease diagnosis, specialty-specific (ophthalmology) diagnosis and management, and simulation of human expert grading and evaluation. We compare their performance with state-of-the-art proprietary models (GPT-5.1, GPT-5-mini, and Gemini 3.1 Pro) and a leading open-source model (DeepSeek-R1), and we further evaluate the adaptability of on-device systems by fine-tuning gpt-oss-20b and Qwen3.5-35B on general diagnostic data. Across tasks, on-device models achieve performance comparable to or exceeding DeepSeek-R1 and GPT-5-mini despite being substantially smaller. In addition, fine-tuning remarkably improves diagnostic accuracy, with the fine-tuned Qwen3.5-35B reaching 87.9% and approaching the proprietary GPT-5.1 (89.4%). Among base on-device models, Gemma 4 31B achieved the strongest general diagnostic accuracy at 86.5%, exceeding GPT-5-mini and approaching the fine-tuned Qwen3.5-35B. Error characterization revealed that 87.2% of diagnostic errors across all models were clinically plausible differentials rather than off-topic predictions, and upper-bound analysis showed up to 93.2% attainable accuracy through improved answer selection. These findings highlight the potential of on-device LLMs to deliver accurate, adaptable, and privacy-preserving clinical decision support, offering a practical pathway for broader integration of LLMs into routine clinical practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.24983v1">Adaptive Prompt Embedding Optimization for LLM Jailbreaking</a></div>
    <div class="paper-meta">
      📅 2026-04-27
    </div>
    <details class="paper-abstract">
      Existing white-box jailbreak attacks against aligned LLMs typically append discrete adversarial suffixes to the user prompt, which visibly alters the prompt and operates in a combinatorial token space. Prior work has avoided directly optimizing the embeddings of the original prompt tokens, presumably because perturbing them risks destroying the prompt's semantic content. We propose Prompt Embedding Optimization (PEO), a multi-round white-box jailbreak that directly optimizes the embeddings of the original prompt tokens without appending any adversarial tokens, and show that the concern is unfounded: the optimized embeddings remain close enough to their originals that the visible prompt string is preserved exactly after nearest-token projection, and quantitative analysis shows the model's responses stay on topic for the large majority of prompts. PEO combines continuous embedding-space optimization with structured continuation targets and an adaptive failure-focused schedule. Counterintuitively, later PEO rounds can benefit from heuristic composite response scaffolds that are not natural standalone templates, yet ASR-Judge shows that the resulting gains are not merely empty formatting or scaffold-only outputs. Across two standard harmful-behavior benchmarks and competing white-box attacks spanning discrete suffix search, appended adversarial embeddings, and search-based adversarial generation, PEO outperforms all of them in our experiments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.24977v1">A Survey on LLM-based Conversational User Simulation</a></div>
    <div class="paper-meta">
      📅 2026-04-27
      | 💬 Submitted in August 2025. MOD-81000 approved survey
    </div>
    <details class="paper-abstract">
      User simulation has long played a vital role in computer science due to its potential to support a wide range of applications. Language, as the primary medium of human communication, forms the foundation of social interaction and behavior. Consequently, simulating conversational behavior has become a key area of study. Recent advancements in large language models (LLMs) have significantly catalyzed progress in this domain by enabling high-fidelity generation of synthetic user conversation. In this paper, we survey recent advancements in LLM-based conversational user simulation. We introduce a novel taxonomy covering user granularity and simulation objectives. Additionally, we systematically analyze core techniques and evaluation methodologies. We aim to keep the research community informed of the latest advancements in conversational user simulation and to further facilitate future research by identifying open challenges and organizing existing work under a unified framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.24955v1">BenchGuard: Who Guards the Benchmarks? Automated Auditing of LLM Agent Benchmarks</a></div>
    <div class="paper-meta">
      📅 2026-04-27
    </div>
    <details class="paper-abstract">
      As benchmarks grow in complexity, many apparent agent failures are not failures of the agent at all - they are failures of the benchmark itself: broken specifications, implicit assumptions, and rigid evaluation scripts that penalize valid alternative approaches. We propose employing frontier LLMs as systematic auditors of evaluation infrastructure, and realize this vision through BenchGuard, the first automated auditing framework for task-oriented, execution-based agent benchmarks. BenchGuard cross-verifies all benchmark artifacts via structured LLM protocols, optionally incorporating agent solutions or execution traces as additional diagnostic evidence. Deployed on two prominent scientific benchmarks, BenchGuard identified 12 author-confirmed issues in ScienceAgentBench - including fatal errors rendering tasks unsolvable - and exactly matched 83.3% of expert-identified issues on the BIXBench Verified-50 subset, catching defects that prior human review missed entirely. A full audit of 50 complex bioinformatics tasks costs under USD 15, making automated benchmark auditing a practical and valuable complement to human review. These findings point toward AI-assisted benchmark development, where frontier models serve not only as subjects of evaluation but as active participants in validating the evaluation infrastructure itself.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.12072v2">VOYAGER: A Training Free Approach for Generating Diverse Datasets using LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-27
      | 💬 Accepted to ACL 2026 Main
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly being used to generate synthetic datasets for the evaluation and training of downstream models. However, prior work has noted that such generated data lacks diversity. In this paper, we propose Voyager, a novel principled approach to generate diverse datasets. Our approach is iterative and directly optimizes a mathematical quantity that optimizes the diversity of the dataset using the machinery of determinantal point processes. Furthermore, our approach is training-free, applicable to closed-source models, and scalable. In addition to providing theoretical justification for the working of our method, we also demonstrate through comprehensive experiments that Voyager significantly outperforms popular baseline approaches by providing a 1.5-3 times improvement in diversity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21304v2">PaperMind: Benchmarking Agentic Reasoning and Critique over Scientific Papers in Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-27
    </div>
    <details class="paper-abstract">
      Understanding scientific papers requires more than answering isolated questions or summarizing content. It involves an integrated reasoning process that grounds textual and visual information, interprets experimental evidence, synthesizes information across sources, and critically evaluates scientific claims. However, existing benchmarks typically assess these abilities in isolation, making it difficult to evaluate scientific paper understanding as a unified set of interacting cognitive abilities. In this work, we introduce PaperMind, a benchmark designed to evaluate integrated and agent-oriented scientific reasoning over research papers. PaperMind is constructed from real scientific papers across seven domains, including agriculture, biology, chemistry, computer science, medicine, physics, and economics. It comprises four complementary task families that collectively operationalize distinct cognitive facets of scientific paper reasoning, including multimodal grounding, experimental interpretation, cross-source evidence reasoning, and critical assessment. By analyzing model behavior across multiple tasks, PaperMind enables a diagnostic evaluation of integrated scientific reasoning behaviors that are difficult to assess through isolated task evaluations. Extensive experiments on both opensource and closed-source multimodal LLMs reveal consistent performance gaps across tasks, highlighting persistent challenges in integrated scientific reasoning and critique. Our benchmark and dataset are available at https:// github.com/Yanjun-Zhao/PaperMind.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.14248v2">Why Do LLM-based Web Agents Fail? A Hierarchical Planning Perspective</a></div>
    <div class="paper-meta">
      📅 2026-04-27
      | 💬 Accepted to The 64th Annual Meeting of the Association for Computational Linguistics (ACL) 2026
    </div>
    <details class="paper-abstract">
      Large language model (LLM) web agents are increasingly used for web navigation but remain far from human reliability on realistic, long-horizon tasks. Existing evaluations focus primarily on end-to-end success, offering limited insight into where failures arise. We propose a hierarchical planning framework to analyze web agents across three layers (i.e., high-level planning, low-level execution, and replanning), enabling process-based evaluation of reasoning, grounding, and recovery. Our experiments show that structured Planning Domain Definition Language (PDDL) plans produce more concise and goal-directed strategies than natural language (NL) plans, but low-level execution remains the dominant bottleneck. These results indicate that improving perceptual grounding and adaptive control, not only high-level reasoning, is critical for achieving human-level reliability. This hierarchical perspective provides a principled foundation for diagnosing and advancing LLM web agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.24832v2">Scheduling Your LLM Reinforcement Learning with Reasoning Trees</a></div>
    <div class="paper-meta">
      📅 2026-04-27
    </div>
    <details class="paper-abstract">
      Using Reinforcement Learning with Verifiable Rewards (RLVR) to optimize Large Language Models (LLMs) can be conceptualized as progressively editing a query's `Reasoning Tree'. This process involves exploring nodes (tokens) and dynamically modifying the model's policy at each node. When combined with data scheduling, this process yields further gains in data efficiency and accuracy. However, existing RLVR data scheduling methods typically rely on path-based metrics to rank queries, overlooking the reasoning tree structures of these queries. In this paper, we introduce a novel metric, namely Reasoning Score (r-score), which measures the query's learning difficulty based on the structure of its reasoning tree. Based on the r-score, we propose the Reasoning Tree Schedule (Re-Schedule), a scheduling algorithm that constructs a curriculum progressing from structurally simple (high r-score) to complex (low r-score) queries. Experiments on six math-reasoning benchmarks show that Re-Schedule significantly improves average accuracy, achieving gains of up to 3.2%. These strong results validate our approach and demonstrate that a structural understanding of the reasoning tree provides a more powerful and principled foundation for RLVR data scheduling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.05606v6">OPeRA: A Dataset of Observation, Persona, Rationale, and Action for Evaluating LLMs on Human Online Shopping Behavior Simulation</a></div>
    <div class="paper-meta">
      📅 2026-04-27
      | 💬 ACL 2026 main
    </div>
    <details class="paper-abstract">
      Can large language models (LLMs) accurately simulate the next web action of a specific user? While LLMs have shown promising capabilities in generating ``believable'' human behaviors, evaluating their ability to mimic real user behaviors remains an open challenge, largely due to the lack of high-quality, publicly available datasets that capture both the observable actions and the internal reasoning of an actual human user. To address this gap, we introduce OPERA, a novel dataset of Observation, Persona, Rationale, and Action collected from real human participants during online shopping sessions. OPERA is the first public dataset that comprehensively captures: user personas, browser observations, fine-grained web actions, and self-reported just-in-time rationales. We developed both an online questionnaire and a custom browser plugin to gather this dataset with high fidelity. Using OPERA, we establish the first benchmark to evaluate how well current LLMs can predict a specific user's next action and rationale with a given persona and <observation, action, rationale> history. This dataset lays the groundwork for future research into LLM agents that aim to act as personalized digital twins for human.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.24715v1">Long-Context Aware Upcycling: A New Frontier for Hybrid LLM Scaling</a></div>
    <div class="paper-meta">
      📅 2026-04-27
    </div>
    <details class="paper-abstract">
      Hybrid sequence models that combine efficient Transformer components with linear sequence modeling blocks are a promising alternative to pure Transformers, but most are still pretrained from scratch and therefore fail to reuse existing Transformer checkpoints. We study upcycling as a practical path to convert pretrained Transformer LLMs into hybrid architectures while preserving short-context quality and improving long-context capability. We call our solution \emph{HyLo} (HYbrid LOng-context): a long-context upcycling recipe that combines architectural adaptation with efficient Transformer blocks, Multi-Head Latent Attention (MLA), and linear blocks (Mamba2 or Gated DeltaNet), together with staged long-context training and teacher-guided distillation for stable optimization. HyLo extends usable context length by up to $32\times$ through efficient post-training and reduces KV-cache memory by more than $90\%$, enabling up to 2M-token prefill and decoding in our \texttt{vLLM} inference stack, while comparable Llama baselines run out of memory beyond 64K context. Across 1B- and 3B-scale settings (Llama- and Qwen-based variants), HyLo delivers consistently strong short- and long-context performance and significantly outperforms state-of-the-art upcycled hybrid baselines on long-context evaluations such as RULER. Notably, at similar scale, HyLo-Qwen-1.7B trained on only 10B tokens significantly outperforms JetNemotron (trained on 400B tokens) on GSM8K, Lm-Harness common sense reasoning and RULER-64K.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.24712v1">When Prompt Under-Specification Improves Code Correctness: An Exploratory Study of Prompt Wording and Structure Effects on LLM-Based Code Generation</a></div>
    <div class="paper-meta">
      📅 2026-04-27
    </div>
    <details class="paper-abstract">
      Large language models are increasingly used for code generation, yet the correctness of their outputs depends not only on model capability but also on how tasks are specified. Prior studies demonstrate that small changes in natural language prompts, particularly under-specification can substantially reduce code correctness; however, these findings are largely based on minimal-specification benchmarks such as HumanEval and MBPP, where limited structural redundancy may exaggerate sensitivity. In this exploratory study, we investigate how prompt structure, task complexity, and specification richness interact with LLM robustness to prompt mutations. We evaluate 10 different models across HumanEval and the structurally richer LiveCodeBench. Our results reveal that robustness is not a fixed property of LLMs but is highly dependent on prompt structure: the same under-specification mutations that degrade performance on HumanEval have near-zero net effect on LiveCodeBench due to redundancy across descriptions, constraints, examples, and I/O conventions. Surprisingly, we also find that prompt mutations can improve correctness. In LiveCodeBench, under-specification often breaks misleading lexical or structural cues that trigger incorrect retrieval-based solution strategies, leading to correctness improvements that counterbalance degradations. Manual analysis identifies consistent mechanisms behind these improvements, including the disruption of over-fitted terminology, removal of misleading constraints, and elimination of spurious identifier triggers. Overall, our study shows that structurally rich task descriptions can substantially mitigate the negative effects of under-specification and, in some cases, even enhance correctness. We outline categories of prompt modifications that positively influence the behavior of LLM code-generation, offering practical insights for writing robust prompts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.24710v1">Case-Specific Rubrics for Clinical AI Evaluation: Methodology, Validation, and LLM-Clinician Agreement Across 823 Encounters</a></div>
    <div class="paper-meta">
      📅 2026-04-27
      | 💬 14 pages, 2 figures, 3 tables, submitted to JAMIA
    </div>
    <details class="paper-abstract">
      Objective. Clinical AI documentation systems require evaluation methodologies that are clinically valid, economically viable, and sensitive to iterative changes. Methods requiring expert review per scoring instance are too slow and expensive for safe, iterative deployment. We present a case-specific, clinician-authored rubric methodology for clinical AI evaluation and examine whether LLM-generated rubrics can approximate clinician agreement. Materials and Methods. Twenty clinicians authored 1,646 rubrics for 823 clinical cases (736 real-world, 87 synthetic) across primary care, psychiatry, oncology, and behavioral health. Each rubric was validated by confirming that an LLM-based scoring agent consistently scored clinician-preferred outputs higher than rejected ones. Seven versions of an EHR-embedded AI agent for clinicians were evaluated across all cases. Results. Clinician-authored rubrics discriminated effectively between high- and low-quality outputs (median score gap: 82.9%) with high scoring stability (median range: 0.00%). Median scores improved from 84% to 95%. In later experiments, clinician-LLM ranking agreement (tau: 0.42-0.46) matched or exceeded clinician-clinician agreement (tau: 0.38-0.43), attributable to both ceiling compression and LLM rubric improvement. Discussion. This convergence supports incorporating LLM rubrics alongside clinician-authored ones. At roughly 1,000 times lower cost, LLM rubrics enable substantially greater evaluation coverage, while continued clinical authorship grounds evaluation in expert judgment. Ceiling compression poses a methodological challenge for future inter-rater agreement studies. Conclusion. Case-specific rubrics offer a path for clinical AI evaluation that preserves expert judgment while enabling automation at three orders lower cost. Clinician-authored rubrics establish the baseline against which LLM rubrics are validated.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.23452v2">CiteAudit: You Cited It, But Did You Read It? A Benchmark for Verifying Scientific References in the LLM Era</a></div>
    <div class="paper-meta">
      📅 2026-04-27
      | 💬 After internal review, we found systematic errors in the benchmark and reference verification pipeline that affect core results and conclusions. As these issues are fundamental and cannot be addressed by revision or replacement, we request withdrawal to avoid potential misuse
    </div>
    <details class="paper-abstract">
      Scientific research relies on accurate citation for attribution and integrity, yet large language models (LLMs) introduce a new risk: fabricated references that appear plausible but correspond to no real publications. Such hallucinated citations have already been observed in submissions and accepted papers at major machine learning venues, exposing vulnerabilities in peer review. Meanwhile, rapidly growing reference lists make manual verification impractical, and existing automated tools remain fragile to noisy and heterogeneous citation formats and lack standardized evaluation. We present the first comprehensive benchmark and detection framework for hallucinated citations in scientific writing. Our multi-agent verification pipeline decomposes citation checking into claim extraction, evidence retrieval, passage matching, reasoning, and calibrated judgment to assess whether a cited source truly supports its claim. We construct a large-scale human-validated dataset across domains and define unified metrics for citation faithfulness and evidence alignment. Experiments with state-of-the-art LLMs reveal substantial citation errors and show that our framework significantly outperforms prior methods in both accuracy and interpretability. This work provides the first scalable infrastructure for auditing citations in the LLM era and practical tools to improve the trustworthiness of scientific references.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08484v2">Patching LLM Like Software: A Lightweight Method for Improving Safety Policy in Large Language Models</a></div>
    <div class="paper-meta">
      📅 2026-04-27
    </div>
    <details class="paper-abstract">
      We propose patching for large language models (LLMs) like software versions, a lightweight and modular approach for addressing safety vulnerabilities. While vendors release improved LLM versions, major releases are costly, infrequent, and difficult to tailor to customer needs, leaving released models with known safety gaps. Unlike full-model fine-tuning or major version updates, our method enables rapid remediation by prepending a compact, learnable prefix to an existing model. This "patch" introduces only 0.003% additional parameters, yet reliably steers model behavior toward that of a safer reference model. Across three critical domains (toxicity mitigation, bias reduction, and harmfulness refusal) policy patches achieve safety improvements comparable to next-generation safety-aligned models while preserving fluency. Our results demonstrate that LLMs can be "patched" much like software, offering vendors and practitioners a practical mechanism for distributing scalable, efficient, and composable safety updates between major model releases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.24703v1">Defective Task Descriptions in LLM-Based Code Generation: Detection and Analysis</a></div>
    <div class="paper-meta">
      📅 2026-04-27
    </div>
    <details class="paper-abstract">
      Large language models are widely used for code generation, yet they rely on an implicit assumption that the task descriptions are sufficiently detailed and well-formed. However, in practice, users may provide defective descriptions, which can have a strong effect on code correctness. To address this issue, we develop SpecValidator, a lightweight classifier based on a small model that has been parameter-efficiently finetuned, to automatically detect task description defects. We evaluate SpecValidator on three types of defects, Lexical Vagueness, Under-Specification and Syntax-Formatting on 3 benchmarks with task descriptions of varying structure and complexity. Our results show that SpecValidator achieves defect detection of F1 = 0.804 and MCC = 0.745, significantly outperforming GPT-5-mini (F1 = 0.469 and MCC = 0.281) and Claude Sonnet 4 (F1 = 0.518 and MCC = 0.359). Perhaps more importantly, our analysis indicates that SpecValidator can generalize to unseen issues and detect unknown Under-Specification defects in the original (real) descriptions of the benchmarks used. Our results also show that the robustness of LLMs in task description defects depends primarily on the type of defect and the characteristics of the task description, rather than the capacity of the model, with Under-Specification defects being the most severe. We further found that benchmarks with richer contextual grounding, such as LiveCodeBench, exhibit substantially greater resilience, highlighting the importance of structured task descriptions for reliable LLM-based code generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.17703v2">Why Are We Moral? An LLM-based Agent Simulation Approach to Study Moral Evolution</a></div>
    <div class="paper-meta">
      📅 2026-04-27
      | 💬 Accepted at ACL 2026 Main Conference. 51 pages including appendix
    </div>
    <details class="paper-abstract">
      The evolution of morality presents a puzzle: natural selection should favor self-interest, yet humans developed moral systems promoting altruism. Traditional approaches must abstract away cognitive processes, leaving open how cognitive factors shape moral evolution. We introduce an LLM-based agent simulation framework that brings cognitive realism to this question: agents with varying moral dispositions perceive, remember, reason, and decide in a simulated prehistoric hunter-gatherer society. This enables us to manipulate factors that traditional models cannot represent -- such as moral type observability and communication bandwidth -- and to discover emergent cognitive mechanisms from agent interactions. Across 20 runs spanning four settings, we find that cooperation and mutual help are the central driver of evolutionary survival, with universal and reciprocal morality exhibiting the most stable outcomes across conditions while selfishness is strongly disfavoured. Beyond cooperation itself, we further identify cognition as a central mediator -- most clearly through a cost of moral judgment that shifts the winning moral type across settings, with a self-purging effect among selfish agents as an additional cognitive pattern. We validate robustness across multiple LLM backbones, architecture ablations, and prompt sensitivity analyses. This work establishes LLM-based simulation as a powerful new paradigm to complement traditional research in evolutionary biology and anthropology, opening new avenues for investigating the complexities of moral and social evolution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.24690v1">Can LLMs Act as Historians? Evaluating Historical Research Capabilities of LLMs via the Chinese Imperial Examination</a></div>
    <div class="paper-meta">
      📅 2026-04-27
      | 💬 Accepted at ACL 2026
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have increasingly assisted in historical tasks such as text processing, their capacity for professional-level historical reasoning remains underexplored. Existing benchmarks primarily assess basic knowledge breadth or lexical understanding, failing to capture the higher-order skills, such as evidentiary reasoning,that are central to historical research. To fill this gap, we introduce ProHist-Bench, a novel benchmark anchored in the Chinese Imperial Examination (Keju) system, a comprehensive microcosm of East Asian political, social, and intellectual history spanning over 1,300 years. Developed through deep interdisciplinary collaboration, ProHist-Bench features 400 challenging, expert-curated questions across eight dynasties, accompanied by 10,891 fine-grained evaluation rubrics. Through a rigorous evaluation of 18 LLMs, we reveal a significant proficiency gap: even state-of-the-art LLMs struggle with complex historical research questions. We hope ProHist-Bench will facilitate the development of domain-specific reasoning LLMs, advance computational historical research, and further uncover the untapped potential of LLMs. We release ProHist-Bench at https://github.com/inclusionAI/ABench/tree/main/ProHist-Bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.24678v1">Leveraging LLMs for Multi-File DSL Code Generation: An Industrial Case Study</a></div>
    <div class="paper-meta">
      📅 2026-04-27
      | 💬 Accepted at EASE'26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) perform strongly on general-purpose code generation, yet their applicability to enterprise domain-specific languages (DSLs) remains underexplored, especially for repository-scale change generation spanning multiple files and folder structures from a single natural-language (NL) instruction. We report an industrial case study at BMW that adapts code-oriented LLMs to generate and modify project-root DSL artifacts for an Xtext-based DSL that drives downstream Java/TypeScript code generation. We develop an end-to-end pipeline for dataset construction, multi-file task representation, model adaptation, and evaluation. We encode DSL folder hierarchies as structured, path-preserving JSON, allowing single-response generation at repository scale and learning cross-file dependencies. We evaluate two instruction-tuned code LLMs (Qwen2.5-Coder and DeepSeek-Coder, 7B) under three configurations: baseline prompting, one-shot in-context learning, and parameter-efficient fine-tuning (QLoRA). Beyond standard similarity metrics, we introduce task-specific measures that assess edit correctness and repository structural fidelity. Fine-tuning yields the most significant gains across models and metrics, achieving high exact-match accuracy, substantial edit similarity, and structural fidelity of 1.00 on our held-out set for multi-file outputs. At the same time, one-shot in-context learning provides smaller but consistent improvements over baseline prompting. We further validate practical utility via an expert developer survey and an execution-based check using the existing code generator.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.24665v1">Benchmarking Source-Sensitive Reasoning in Turkish: Humans and LLMs under Evidential Trust Manipulation</a></div>
    <div class="paper-meta">
      📅 2026-04-27
      | 💬 Accepted to The 15th edition of the Workshop on Cognitive Modeling and Computational Linguistics, co-located with the Language Resources and Evaluation Conference
    </div>
    <details class="paper-abstract">
      This paper investigates whether source trustworthiness shapes Turkish evidential morphology and whether large language models (LLMs) track this sensitivity. We study the past-domain contrast between -DI and -mIs in controlled cloze contexts where the information source is overtly external, while only its perceived reliability is manipulated (High-Trust vs. Low-Trust). In a human production experiment, native speakers of Turkish show a robust trust effect: High-Trust contexts yield relatively more -DI, whereas Low-Trust contexts yield relatively more -mIs, with the pattern remaining stable across sensitivity analyses. We then evaluate 10 LLMs in three prompting paradigms (open gap-fill, explicit past-tense gap-fill, and forced-choice A/B selection). LLM behavior is highly model- and prompt-dependent: some models show weak or local trust-consistent shifts, but effects are generally unstable, often reversed, and frequently overshadowed by output-compliance problems and strong base-rate suffix preferences. The results provide new evidence for a trust-/commitment-based account of Turkish evidentiality and reveal a clear human-LLM gap in source-sensitive evidential reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.24647v1">DepthKV: Layer-Dependent KV Cache Pruning for Long-Context LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-04-27
    </div>
    <details class="paper-abstract">
      Long-context reasoning is a critical capability of large language models (LLMs), enabling applications such as long-document understanding, summarization, and code generation. However, efficient autoregressive inference relies on the key-value (KV) cache, whose memory footprint grows linearly with sequence length, leading to a major memory bottleneck. To mitigate this overhead, KV cache pruning methods discard cached tokens with low attention scores during inference. Most existing methods apply a uniform pruning ratio across layers, implicitly assuming that all layers contribute equally to overall model performance. We show that this assumption is suboptimal, as layers differ significantly in their sensitivity to pruning. We propose DepthKV, a layer-dependent pruning framework that allocates a fixed global KV budget across layers based on their sensitivity, rather than using a uniform allocation. Across multiple models and tasks, DepthKV consistently outperforms uniform pruning at the same global pruning ratio, demonstrating more effective utilization of the KV cache budget through layer-dependent allocation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09159v3">LLMs Meet Isolation Kernel: Lightweight, Learning-free Binary Embeddings for Fast Retrieval</a></div>
    <div class="paper-meta">
      📅 2026-04-27
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently enabled remarkable progress in text representation. However, their embeddings are typically high-dimensional, leading to substantial storage and retrieval overhead. Although recent approaches such as Matryoshka Representation Learning (MRL) and Contrastive Sparse Representation (CSR) alleviate these issues to some extent, they still suffer from retrieval accuracy degradation. This paper proposes Isolation Kernel Embedding or IKE, a learning-free method that transforms an LLM embedding into a binary embedding using Isolation Kernel (IK). Lightweight and based on binary encoding, IKE offers a low memory footprint and fast bitwise computation, lowering retrieval latency. Experiments on multiple text retrieval datasets demonstrate that IKE offers up to 16.7x faster retrieval and 16x lower memory usage than the original LLM embeddings, while maintaining comparable accuracy. Theoretically, we show that IKE works because it satisfies four essential criteria for effective binary hashing that other methods do not possess. Compared to CSR, IKE consistently achieves better retrieval efficiency and effectiveness. IKE also works effectively with graph-based indexing, demonstrating its superiority in balancing accuracy and latency compared to alternative compression techniques in the approximate nearest neighbor (ANN) search setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19533v3">Cyber Defense Benchmark: Agentic Threat Hunting Evaluation for LLMs in SecOps</a></div>
    <div class="paper-meta">
      📅 2026-04-23
      | 💬 Updated leaderboard with newer models
    </div>
    <details class="paper-abstract">
      We introduce the Cyber Defense Benchmark, a benchmark for measuring how well large language model (LLM) agents perform the core SOC analyst task of threat hunting: given a database of raw Windows event logs with no guided questions or hints, identify the exact timestamps of malicious events. The benchmark wraps 106 real attack procedures from the OTRF Security-Datasets corpus - spanning 86 MITRE ATT&CK sub-techniques across 12 tactics - into a Gymnasium reinforcement-learning environment. Each episode presents the agent with an in-memory SQLite database of 75,000-135,000 log records produced by a deterministic campaign simulator that time-shifts and entity-obfuscates the raw recordings. The agent must iteratively submit SQL queries to discover malicious event timestamps and explicitly flag them, scored CTF-style against Sigma-rule-derived ground truth. Evaluating five frontier models - Claude Opus 4.6, GPT-5, Gemini 3.1 Pro, Kimi K2.5, and Gemini 3 Flash - on 26 campaigns covering 105 of 106 procedures, we find that all models fail dramatically: the best model (Claude Opus 4.6) submits correct flags for only 3.8% of malicious events on average, and no run across any model ever finds all flags. We define a passing score as >= 50% recall on every ATT&CK tactic - the minimum bar for unsupervised SOC deployment. No model passes: the leader clears this bar on 5 of 13 tactics and the remaining four on zero. These results suggest that current LLMs are poorly suited for open-ended, evidence-driven threat hunting despite strong performance on curated Q&A security benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.23874v4">From Stochasticity to Signal: A Bayesian Latent State Model for Reliable Measurement with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to automate classification tasks in business, such as analyzing customer satisfaction from text. However, the inherent stochasticity of LLMs can create measurement error when the outcome is considered deterministic. This problem is often neglected with the empirical practice of a single round of output, or addressed with ad-hoc methods like majority voting. Such naive approaches fail to quantify uncertainty and can produce biased estimates of population-level metrics. In this paper, we propose a formal statistical solution by introducing a Bayesian latent state model to address it. Our model treats the true classification as a latent variable and the multiple LLM ratings as noisy measurements of this outcome state. This framework jointly estimates LLM error rates, population-level outcome rates, individual-level probabilities of the outcome, and the causal impact of interventions, if any, on the outcome. The methodology is applicable to both fully unsupervised and semi-supervised settings, where ground truth labels are unavailable or available for only a subset of the classification targets. We provide formal theoretical conditions and proofs for the strict identifiability of the model parameters. Through simulation studies, we demonstrate that our model accurately recovers true parameters, showing superior performance and capabilities compared to other methods. We provide tailored recommendations of modeling choices based on the difficulty level of the task. We also apply it to a real-world case study analyzing over 14,000 customer support transcripts. We conclude that this methodology provides a general framework for converting probabilistic outputs from LLMs into reliable insights for scientific and business applications.
    </details>
</div>
